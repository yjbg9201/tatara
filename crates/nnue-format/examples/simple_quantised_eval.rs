//! Simple ネットの量子化 CPU forward を推論エンジンと突き合わせるための道具。
//!
//! ```text
//! # 乱数の weight で .bin を書く (入玉スカラーつきなら --nyugyoku-rule を付ける)
//! cargo run -p nnue-format --release --example simple_quantised_eval -- random-net \
//!     --out /path/to/nn.bin [--nyugyoku-rule csa27] [--nyugyoku-encoding thermometer] \
//!     [--seed 1] [--fv-scale 16] [--scalar-weight 127]
//!
//! # L1 weight の列ブロックごとの L2 ノルム (i8 / 64 の実数値) を出す。入玉スカラーの列が
//! # 学習で 0 から動いたかを、学習前後の .bin で見比べるのに使う
//! cargo run -p nnue-format --release --example simple_quantised_eval -- l1-norms \
//!     --net /path/to/nn.bin [--nyugyoku-rule csa27]
//!
//! # SFEN 1 行 1 局面のファイルを評価し、「出力層の生の値<TAB>評価値」を 1 行ずつ出す。
//! # --stats を付けると、各層で 0 / 127 に張り付いた値の割合を標準エラーに出す
//! cargo run -p nnue-format --release --example simple_quantised_eval -- eval \
//!     --net /path/to/nn.bin --sfens /path/to/positions.sfen [--nyugyoku-rule csa27]
//! ```
//!
//! ネットは HalfKP・CReLU・256x2-32-32 (L1 入力は入玉スカラーつきで 608、なしで 512)。
//! `random-net` の weight は量子化格子上の値で、FT の加算が i16 を超えない範囲に抑える。
//! 入玉スカラーの列の weight は `--scalar-weight` (既定 127 = i8 の上限) まで大きくし、
//! L1 の ClippedReLU が飽和・クリップする経路も通るようにする。

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use nnue_format::{SimpleActivation, SimpleId, SimpleQuantised, SimpleWeights};
use shogi_features::{
    DeclarationRule, FeatureSet, NYUGYOKU_SCALAR_DIMS, NyugyokuScalarConfig, ScalarEncoding,
};
use shogi_format::ShogiBoard;

struct Args {
    command: String,
    out: PathBuf,
    net: PathBuf,
    sfens: PathBuf,
    seed: u64,
    fv_scale: i32,
    scalar_weight: i64,
    nyugyoku: Option<NyugyokuScalarConfig>,
    stats: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it
        .next()
        .ok_or("expected a subcommand: random-net, eval or l1-norms")?;
    let mut args = Args {
        command,
        out: PathBuf::new(),
        net: PathBuf::new(),
        sfens: PathBuf::new(),
        seed: 1,
        fv_scale: 16,
        scalar_weight: 127,
        nyugyoku: None,
        stats: false,
    };
    let mut rule: Option<DeclarationRule> = None;
    let mut encoding = ScalarEncoding::Thermometer;
    while let Some(flag) = it.next() {
        if flag == "--stats" {
            args.stats = true;
            continue;
        }
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--out" => args.out = PathBuf::from(value),
            "--net" => args.net = PathBuf::from(value),
            "--sfens" => args.sfens = PathBuf::from(value),
            "--seed" => args.seed = value.parse().map_err(|e| format!("{flag}: {e}"))?,
            "--fv-scale" => args.fv_scale = value.parse().map_err(|e| format!("{flag}: {e}"))?,
            "--scalar-weight" => {
                args.scalar_weight = value.parse().map_err(|e| format!("{flag}: {e}"))?;
                if !(0..=127).contains(&args.scalar_weight) {
                    return Err("--scalar-weight must be in 0..=127".to_string());
                }
            }
            "--nyugyoku-rule" => {
                rule = Some(
                    DeclarationRule::from_canonical_name(&value)
                        .ok_or(format!("unknown rule {value}"))?,
                )
            }
            "--nyugyoku-encoding" => {
                encoding = ScalarEncoding::from_canonical_name(&value)
                    .ok_or(format!("unknown encoding {value}"))?
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    args.nyugyoku = rule.map(|rule| NyugyokuScalarConfig { rule, encoding });
    Ok(args)
}

fn net_id(nyugyoku: Option<NyugyokuScalarConfig>) -> SimpleId {
    SimpleId {
        feature_set: FeatureSet::HalfKp.spec(),
        activation: SimpleActivation::CReLU,
        ft_out: 256,
        l1_out: 32,
        l2_out: 32,
        nyugyoku_scalar: nyugyoku,
    }
}

struct XorShift64(u64);

impl XorShift64 {
    /// `[-range, range]` の一様整数。
    fn int(&mut self, range: i64) -> i64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % (2 * range as u64 + 1)) as i64 - range
    }
}

fn random_net(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let id = net_id(args.nyugyoku);
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15 ^ args.seed.wrapping_mul(0x2545_F491_4F6C_DD1D));
    let mut w = SimpleWeights::zeroed(id, args.fv_scale);
    // FT: |bias| <= 64、|weight| <= 40 なので、active feature 38 個でも |acc| <= 1584 で
    // i16 に収まる。CReLU (0..127) の両端で clip する値が多く出る。
    let qa = 127.0;
    for v in &mut w.ft_w {
        *v = rng.int(40) as f32 / qa;
    }
    for v in &mut w.ft_b {
        *v = rng.int(64) as f32 / qa;
    }
    let qb = 64.0;
    let bias_scale = (127 * 64) as f32;
    let l1_in = id.combined_dim();
    let ft_cols = id.ft_output_dim();
    for o in 0..id.l1_out {
        for i in 0..l1_in {
            let range = if i < ft_cols { 12 } else { args.scalar_weight };
            w.l1_w[o * l1_in + i] = rng.int(range) as f32 / qb;
        }
    }
    for v in &mut w.l1_b {
        *v = rng.int(60_000) as f32 / bias_scale;
    }
    for v in &mut w.l2_w {
        *v = rng.int(64) as f32 / qb;
    }
    for v in &mut w.l2_b {
        *v = rng.int(60_000) as f32 / bias_scale;
    }
    for v in &mut w.l3_w {
        *v = rng.int(127) as f32 / qb;
    }
    w.l3_b[0] = rng.int(60_000) as f32 / bias_scale;

    let mut out = BufWriter::new(std::fs::File::create(&args.out)?);
    w.save_quantised(&mut out)?;
    out.flush()?;
    println!(
        "wrote {} (L1 input {}, entering-king scalars: {})",
        args.out.display(),
        l1_in,
        match args.nyugyoku {
            Some(c) => format!(
                "{} {} x{}",
                c.rule.canonical_name(),
                c.encoding.canonical_name(),
                NYUGYOKU_SCALAR_DIMS
            ),
            None => "none".to_string(),
        }
    );
    Ok(())
}

fn eval(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let id = net_id(args.nyugyoku);
    let net = SimpleQuantised::read(&mut BufReader::new(std::fs::File::open(&args.net)?), id)?;
    let input = BufReader::new(std::fs::File::open(&args.sfens)?);
    let stdout = std::io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut stats = Stats::default();
    for line in input.lines() {
        let line = line?;
        let sfen = line.trim();
        if sfen.is_empty() {
            continue;
        }
        let sfen = sfen.strip_prefix("sfen ").unwrap_or(sfen);
        let board = ShogiBoard::from_sfen(sfen)?;
        let trace = net.forward_trace(&board)?;
        let raw = trace.output;
        writeln!(out, "{raw}\t{}", raw / net.fv_scale)?;
        if args.stats {
            let ft_cols = id.ft_output_dim();
            stats.add("FT output", &trace.input[..ft_cols]);
            stats.add("scalars", &trace.input[ft_cols..]);
            stats.add("hidden1", &trace.hidden1);
            stats.add("hidden2", &trace.hidden2);
            stats.positions += 1;
            stats.declarable += usize::from(
                args.nyugyoku.is_some()
                    && (trace.input[ft_cols + 40] != 0 || trace.input[ft_cols + 44 + 40] != 0),
            );
        }
    }
    out.flush()?;
    if args.stats {
        stats.print(args.nyugyoku);
    }
    Ok(())
}

/// 層ごとの (値の数, 0 の数, 127 の数)。
#[derive(Default)]
struct Stats {
    layers: Vec<(&'static str, usize, usize, usize)>,
    positions: usize,
    declarable: usize,
}

impl Stats {
    fn add(&mut self, name: &'static str, values: &[u8]) {
        let entry = match self.layers.iter_mut().find(|l| l.0 == name) {
            Some(e) => e,
            None => {
                self.layers.push((name, 0, 0, 0));
                self.layers.last_mut().unwrap()
            }
        };
        entry.1 += values.len();
        entry.2 += values.iter().filter(|&&v| v == 0).count();
        entry.3 += values.iter().filter(|&&v| v == 127).count();
    }

    fn print(&self, nyugyoku: Option<NyugyokuScalarConfig>) {
        for (name, n, zero, max) in &self.layers {
            let pct = |k: &usize| 100.0 * *k as f64 / (*n).max(1) as f64;
            eprintln!(
                "{name:>10}: {n} values, 0: {:.1}%, 127: {:.1}%, in between: {:.1}%",
                pct(zero),
                pct(max),
                100.0 - pct(zero) - pct(max)
            );
        }
        if nyugyoku.is_some_and(|c| c.encoding == ScalarEncoding::Thermometer) {
            eprintln!(
                "positions with a declarable king (either side): {}/{}",
                self.declarable, self.positions
            );
        }
    }
}

/// L1 weight を列ブロック (FT の手番側・相手側、入玉スカラーの手番側・相手側の玉と 0 埋めの
/// 列) に分け、ブロックごとの L2 ノルムを出す。
fn l1_norms(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let id = net_id(args.nyugyoku);
    let mut reader = BufReader::new(std::fs::File::open(&args.net)?);
    let q = SimpleQuantised::read(&mut reader, id)?;
    let padded = nnue_format::simple_weights::pad32(id.combined_dim());
    let ft = id.ft_out;
    let block = shogi_features::nyugyoku_scalar::NYUGYOKU_SCALAR_BLOCK_DIMS;
    let mut blocks = vec![("ft_stm", 0, ft), ("ft_nstm", ft, 2 * ft)];
    if id.nyugyoku_scalar.is_some() {
        // 入玉スカラーは手番側の玉・相手側の玉のブロック (各 44 列) と、残りの 0 埋め 8 列。
        blocks.push(("scalar_stm", 2 * ft, 2 * ft + block));
        blocks.push(("scalar_nstm", 2 * ft + block, 2 * ft + 2 * block));
        blocks.push((
            "scalar_pad",
            2 * ft + 2 * block,
            2 * ft + NYUGYOKU_SCALAR_DIMS,
        ));
    }
    for (name, begin, end) in blocks {
        let sum: f64 = (0..id.l1_out)
            .flat_map(|o| q.l1_w[o * padded + begin..o * padded + end].iter())
            .map(|&w| {
                let x = f64::from(w) / 64.0;
                x * x
            })
            .sum();
        println!("{name}\t{:.6}", sum.sqrt());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    match args.command.as_str() {
        "random-net" => random_net(&args),
        "eval" => eval(&args),
        "l1-norms" => l1_norms(&args),
        other => Err(format!("unknown subcommand {other}").into()),
    }
}
