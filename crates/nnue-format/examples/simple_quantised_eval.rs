//! Simple ネットの量子化 CPU forward を推論エンジンと突き合わせるための道具。
//!
//! ```text
//! # 乱数の weight で .bin を書く (入玉スカラーつきなら --nyugyoku-rule を付ける)
//! cargo run -p nnue-format --release --example simple_quantised_eval -- random-net \
//!     --out /path/to/nn.bin [--nyugyoku-rule csa27] [--nyugyoku-encoding thermometer] \
//!     [--seed 1] [--fv-scale 16] [--scalar-weight 127]
//!
//! # SFEN 1 行 1 局面のファイルを評価し、「出力層の生の値<TAB>評価値」を 1 行ずつ出す
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
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it
        .next()
        .ok_or("expected a subcommand: random-net or eval")?;
    let mut args = Args {
        command,
        out: PathBuf::new(),
        net: PathBuf::new(),
        sfens: PathBuf::new(),
        seed: 1,
        fv_scale: 16,
        scalar_weight: 127,
        nyugyoku: None,
    };
    let mut rule: Option<DeclarationRule> = None;
    let mut encoding = ScalarEncoding::Thermometer;
    while let Some(flag) = it.next() {
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
    for line in input.lines() {
        let line = line?;
        let sfen = line.trim();
        if sfen.is_empty() {
            continue;
        }
        let sfen = sfen.strip_prefix("sfen ").unwrap_or(sfen);
        let board = ShogiBoard::from_sfen(sfen)?;
        let raw = net.forward_raw(&board)?;
        writeln!(out, "{raw}\t{}", raw / net.fv_scale)?;
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    match args.command.as_str() {
        "random-net" => random_net(&args),
        "eval" => eval(&args),
        other => Err(format!("unknown subcommand {other}").into()),
    }
}
