//! dataloader のスループット (局面/秒) を、入玉スカラーあり・なしで測る。
//!
//! ```text
//! cargo run -p nnue-train --release --example dataloader_throughput -- \
//!     --data /path/to/train.psv [--feature-set halfkp] [--batch-size 16384] \
//!     [--threads 4] [--batches 200] [--warmup 20] [--repeat 3] [--mode loader|push]
//! ```
//!
//! - `loader`: `BucketedPrefetchedLoader` (bucket 計算なし = Simple アーキと同じ経路) から
//!   `--batches` 個の batch を受け取るまでの時間。I/O・decode・特徴抽出を含む。
//! - `push`: 先頭 `--batch-size` 局面を一度 decode しておき、`Batch::push_decoded` だけを
//!   `--batches` 回繰り返す (1 thread)。スカラー計算そのものの追加コストを見る。
//!
//! 各モードで「スカラーなし」と「27 点法・thermometer」を交互に `--repeat` 回測り、
//! 中央値と低下率を表示する。

use std::path::PathBuf;
use std::time::Instant;

use nnue_train::dataloader::{Batch, BucketMode, BucketedPrefetchedLoader};
use shogi_features::{FeatureSet, FeatureSetSpec, NyugyokuScalarConfig};
use shogi_format::{PackedSfenValue, ShogiBoard};

struct Args {
    data: PathBuf,
    feature_set: FeatureSetSpec,
    batch_size: usize,
    threads: usize,
    batches: usize,
    warmup: usize,
    repeat: usize,
    mode: String,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        data: PathBuf::new(),
        feature_set: FeatureSet::HalfKp.spec(),
        batch_size: 16384,
        threads: 4,
        batches: 200,
        warmup: 20,
        repeat: 3,
        mode: "loader".to_string(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        let parse = |v: String| v.parse::<usize>().map_err(|e| format!("{flag}: {e}"));
        match flag.as_str() {
            "--data" => args.data = PathBuf::from(value()?),
            "--feature-set" => {
                let name = value()?;
                args.feature_set = FeatureSet::from_canonical_name(&name)
                    .ok_or(format!("unknown feature set {name}"))?
                    .spec();
            }
            "--batch-size" => args.batch_size = parse(value()?)?,
            "--threads" => args.threads = parse(value()?)?,
            "--batches" => args.batches = parse(value()?)?,
            "--warmup" => args.warmup = parse(value()?)?,
            "--repeat" => args.repeat = parse(value()?)?,
            "--mode" => args.mode = value()?,
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    if args.data.as_os_str().is_empty() {
        return Err("--data is required".to_string());
    }
    Ok(args)
}

fn loader_rate(args: &Args, scalar: Option<NyugyokuScalarConfig>) -> std::io::Result<f64> {
    let size = std::fs::metadata(&args.data)?.len();
    let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
        &args.data,
        args.batch_size,
        None,
        None,
        args.threads,
        BucketMode::KingRank9,
        args.feature_set,
        false,
        1,
        size,
        false,
        None,
        0,
        false,
        0,
        scalar,
    )?;
    let next = |loader: &mut BucketedPrefetchedLoader| -> std::io::Result<()> {
        let slot = loader
            .next_batch()?
            .ok_or_else(|| std::io::Error::other("loader ended"))?;
        loader.recycle(slot);
        Ok(())
    };
    for _ in 0..args.warmup {
        next(&mut loader)?;
    }
    let start = Instant::now();
    for _ in 0..args.batches {
        next(&mut loader)?;
    }
    let secs = start.elapsed().as_secs_f64();
    Ok((args.batches * args.batch_size) as f64 / secs)
}

fn read_boards(args: &Args) -> std::io::Result<Vec<ShogiBoard>> {
    let bytes = std::fs::read(&args.data)?;
    Ok(bytes
        .chunks_exact(40)
        .take(args.batch_size)
        .map(|chunk| {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(chunk);
            psv.decode()
        })
        .collect())
}

fn push_rate(
    args: &Args,
    boards: &[ShogiBoard],
    scalar: Option<NyugyokuScalarConfig>,
) -> std::io::Result<f64> {
    let mut batch =
        Batch::with_capacity_and_nyugyoku_scalars(boards.len(), args.feature_set, scalar);
    let run = |batch: &mut Batch| -> std::io::Result<()> {
        batch.reset();
        for board in boards {
            batch.push_decoded(board)?;
        }
        std::hint::black_box(&batch.nyugyoku_scalars);
        Ok(())
    };
    for _ in 0..args.warmup {
        run(&mut batch)?;
    }
    let start = Instant::now();
    for _ in 0..args.batches {
        run(&mut batch)?;
    }
    let secs = start.elapsed().as_secs_f64();
    Ok((args.batches * boards.len()) as f64 / secs)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let boards = if args.mode == "push" {
        read_boards(&args)?
    } else {
        Vec::new()
    };
    let measure = |scalar: Option<NyugyokuScalarConfig>| -> std::io::Result<f64> {
        match args.mode.as_str() {
            "loader" => loader_rate(&args, scalar),
            "push" => push_rate(&args, &boards, scalar),
            other => Err(std::io::Error::other(format!("unknown mode {other}"))),
        }
    };
    let mut off = Vec::new();
    let mut on = Vec::new();
    for _ in 0..args.repeat {
        off.push(measure(None)?);
        on.push(measure(Some(NyugyokuScalarConfig::default()))?);
    }
    let (off_m, on_m) = (median(off.clone()), median(on.clone()));
    println!(
        "mode={} feature_set={} batch_size={} threads={} batches={}",
        args.mode,
        args.feature_set.canonical_name(),
        args.batch_size,
        if args.mode == "push" { 1 } else { args.threads },
        args.batches
    );
    println!("scalars off: {off:.0?} pos/s, median {off_m:.0}");
    println!("scalars on : {on:.0?} pos/s, median {on_m:.0}");
    println!("change: {:+.2}%", (on_m / off_m - 1.0) * 100.0);
    Ok(())
}
