//! `Batch` の入玉スカラーのテスト。
//!
//! golden 局面 (`shogi-features/tests/data/nyugyoku_scalar_golden.tsv`) を PackedSfen に
//! 符号化して各経路 (直接 `push` / 学習 loader / held-out 検証集合) で `Batch` に詰め、
//! 取り出したスカラーが golden の期待値と一致することと、スカラーの有無で sparse
//! 特徴などの既存 buffer が変わらないことを検査する。

use std::io::Write;
use std::path::{Path, PathBuf};

use shogi_features::{
    DeclarationRule, FeatureSet, FeatureSetSpec, NYUGYOKU_SCALAR_DIMS, NyugyokuScalarConfig,
    ScalarEncoding,
};
use shogi_format::{PackedSfenValue, ShogiBoard};

use super::{Batch, BucketMode, BucketedPrefetchedLoader};
use crate::validation::HeldoutSet;

struct GoldenRow {
    name: String,
    config: NyugyokuScalarConfig,
    board: ShogiBoard,
    expected: Vec<u8>,
}

fn golden_rows() -> Vec<GoldenRow> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shogi-features/tests/data/nyugyoku_scalar_golden.tsv");
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 5);
            let mut board = ShogiBoard::from_sfen(cols[3]).unwrap();
            // score / result は Batch の score / wdl の経路確認用に局面ごとに変える。
            board.score = i16::try_from(cols[3].len()).unwrap();
            board.result = 1;
            GoldenRow {
                name: cols[0].to_string(),
                config: NyugyokuScalarConfig {
                    rule: DeclarationRule::from_canonical_name(cols[1]).unwrap(),
                    encoding: ScalarEncoding::from_canonical_name(cols[2]).unwrap(),
                },
                board,
                expected: cols[4]
                    .split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect(),
            }
        })
        .collect()
}

fn configs() -> Vec<NyugyokuScalarConfig> {
    let mut out = Vec::new();
    for rule in DeclarationRule::ALL {
        for encoding in ScalarEncoding::ALL {
            out.push(NyugyokuScalarConfig { rule, encoding });
        }
    }
    out
}

fn spec() -> FeatureSetSpec {
    FeatureSet::HalfKp.spec()
}

/// 指定設定の golden 行 (局面の並び順はファイル順)。
fn rows_for(rows: &[GoldenRow], config: NyugyokuScalarConfig) -> Vec<&GoldenRow> {
    rows.iter().filter(|r| r.config == config).collect()
}

fn write_psv(path: &Path, boards: &[&ShogiBoard]) {
    let mut file = std::fs::File::create(path).unwrap();
    for board in boards {
        let psv = PackedSfenValue::from_board(board).unwrap();
        file.write_all(psv.as_bytes()).unwrap();
    }
}

fn assert_rows_match(batch: &Batch, rows: &[&GoldenRow], label: &str) {
    assert_eq!(batch.n_positions, rows.len(), "{label}");
    for (bi, row) in rows.iter().enumerate() {
        assert_eq!(
            batch.nyugyoku_scalar_row(bi).unwrap(),
            row.expected.as_slice(),
            "{label}: {} {:?} {:?}",
            row.name,
            row.config.rule,
            row.config.encoding
        );
        assert_eq!(batch.score[bi], f32::from(row.board.score), "{label}");
    }
}

/// golden 局面を PackedSfen に符号化して `Batch::push` で詰めると、golden と一致する。
#[test]
fn push_through_packed_sfen_matches_golden() {
    let rows = golden_rows();
    for config in configs() {
        let selected = rows_for(&rows, config);
        assert_eq!(selected.len(), 8);
        let mut batch =
            Batch::with_capacity_and_nyugyoku_scalars(selected.len(), spec(), Some(config));
        assert_eq!(
            batch.nyugyoku_scalars.len(),
            selected.len() * NYUGYOKU_SCALAR_DIMS
        );
        for row in &selected {
            let psv = PackedSfenValue::from_board(&row.board).unwrap();
            assert!(batch.push(&psv).unwrap());
        }
        assert_rows_match(&batch, &selected, "push");
    }
}

/// 学習 loader (複数 worker) と held-out 検証集合でも golden と一致する。
#[test]
fn loader_and_heldout_match_golden() {
    let rows = golden_rows();
    let dir = tempfile::tempdir().unwrap();
    for config in configs() {
        let selected = rows_for(&rows, config);
        let path = dir.path().join(format!(
            "golden-{}-{}.psv",
            config.rule.canonical_name(),
            config.encoding.canonical_name()
        ));
        let boards: Vec<&ShogiBoard> = selected.iter().map(|r| &r.board).collect();
        write_psv(&path, &boards);
        let size = std::fs::metadata(&path).unwrap().len();

        // batch_size = 局面数なので、各 batch は 1 epoch 分をファイル順に持つ
        // (worker が複数でも 1 batch は 1 worker が連続して読む)。
        let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
            &path,
            selected.len(),
            None,
            None,
            3,
            BucketMode::KingRank9,
            spec(),
            false,
            1,
            size,
            false,
            None,
            0,
            false,
            0,
            Some(config),
        )
        .unwrap();
        for _ in 0..4 {
            let (batch, buckets) = loader.next_batch().unwrap().unwrap();
            assert_rows_match(&batch, &selected, "loader");
            loader.recycle((batch, buckets));
        }
        drop(loader);

        let set = HeldoutSet::load_from_range_with_score_sources(
            &path,
            0,
            size,
            4,
            None,
            None,
            8,
            &BucketMode::KingRank9,
            spec(),
            1,
            None,
            Some(config),
        )
        .unwrap();
        assert_eq!(set.batches().len(), 2);
        for (k, (batch, _)) in set.batches().iter().enumerate() {
            assert_rows_match(batch, &selected[4 * k..4 * k + 4], "heldout");
        }
    }
}

/// スカラーを有効にしても sparse 特徴・score・wdl などの既存 buffer は変わらず、
/// スカラー無しの batch はスカラーの buffer を確保しない。
#[test]
fn scalars_do_not_change_existing_buffers() {
    let rows = golden_rows();
    let boards: Vec<&ShogiBoard> = rows_for(&rows, NyugyokuScalarConfig::default())
        .iter()
        .map(|r| &r.board)
        .collect();
    for fs in FeatureSet::ALL {
        let spec = fs.spec();
        let mut plain = Batch::with_capacity(boards.len(), spec);
        let mut with = Batch::with_capacity_and_nyugyoku_scalars(
            boards.len(),
            spec,
            Some(NyugyokuScalarConfig::default()),
        );
        assert!(plain.nyugyoku_scalar.is_none());
        assert_eq!(plain.nyugyoku_scalars.capacity(), 0);
        for board in &boards {
            assert!(plain.push_decoded(board).unwrap());
            assert!(with.push_decoded(board).unwrap());
        }
        assert_eq!(plain.nyugyoku_scalars.capacity(), 0);
        assert_eq!(plain.nyugyoku_scalar_row(0), None);
        assert_eq!(plain.stm_indices, with.stm_indices, "{fs:?}");
        assert_eq!(plain.nstm_indices, with.nstm_indices, "{fs:?}");
        assert_eq!(plain.nnz, with.nnz, "{fs:?}");
        assert_eq!(plain.score, with.score, "{fs:?}");
        assert_eq!(plain.wdl, with.wdl, "{fs:?}");
        assert_eq!(plain.per_pos_norm, with.per_pos_norm, "{fs:?}");
        assert_eq!(plain.n_positions, with.n_positions, "{fs:?}");
    }
}

/// スカラー有効時、玉が欠けた局面は buffer を書き換える前に error にする。
#[test]
fn missing_king_is_an_error_with_scalars() {
    let board = ShogiBoard::from_sfen("9/9/9/9/9/9/9/9/4K4 b P 1").unwrap();
    let mut with =
        Batch::with_capacity_and_nyugyoku_scalars(1, spec(), Some(NyugyokuScalarConfig::default()));
    let before = with.clone();
    let err = with.push_decoded(&board).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(with.n_positions, 0);
    assert_eq!(with.stm_indices, before.stm_indices);
    assert_eq!(with.nyugyoku_scalars, before.nyugyoku_scalars);
}

/// HCPE 入力でも、loader が詰めたスカラーは同じ record を復号した盤面から直接計算した
/// スカラーと一致する (HCPE のエンコーダは無いので、golden ではなく直接計算と比べる)。
#[test]
fn hcpe_loader_matches_direct_computation() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shogi-format/tests/data/sample.hcpe");
    let size = std::fs::metadata(&path).unwrap().len();
    let mut direct = super::HcpeFileLoader::new(&path).unwrap();
    let mut boards = Vec::new();
    while let Some(board) = direct.next_board().unwrap() {
        boards.push(board);
    }
    assert_eq!(boards.len(), 100);

    for config in configs() {
        // batch_size = 局面数なので、各 batch は 1 epoch 分をファイル順に持つ。
        let mut loader = BucketedPrefetchedLoader::spawn_with_score_sources(
            &path,
            boards.len(),
            None,
            None,
            2,
            BucketMode::KingRank9,
            spec(),
            false,
            1,
            size,
            false,
            None,
            0,
            false,
            0,
            Some(config),
        )
        .unwrap();
        for _ in 0..3 {
            let (batch, buckets) = loader.next_batch().unwrap().unwrap();
            assert_eq!(batch.n_positions, boards.len());
            for (bi, board) in boards.iter().enumerate() {
                assert_eq!(
                    batch.nyugyoku_scalar_row(bi).unwrap(),
                    shogi_features::nyugyoku_scalars(board, config).as_slice(),
                    "record {bi} {config:?}"
                );
            }
            loader.recycle((batch, buckets));
        }
    }
}
