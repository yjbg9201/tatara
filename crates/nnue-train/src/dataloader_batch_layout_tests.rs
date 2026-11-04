//! 既存 feature set の `Batch` の中身をバイト単位で固定するテスト。
//!
//! 各経路 (直接 `push` / 学習 loader の PSV・HCPE / held-out 検証集合) で作った
//! `Batch` の全 buffer (padding 領域・`[n_positions, batch_size)` の行を含む) を
//! FNV-1a 64bit に畳み、期待値と比較する。期待値は入玉スカラーを `Batch` に足す前の
//! dataloader で計算したもので、スカラーを使わない構成の出力が変わっていないことを
//! 検査する。

use std::path::PathBuf;

use shogi_features::{FeatureSet, FeatureSetSpec, ThreatProfile};
use shogi_format::PackedSfenValue;

use super::{Batch, BucketMode, BucketedPrefetchedLoader};
use crate::validation::HeldoutSet;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("shogi-format/tests/data")
        .join(name)
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xCBF2_9CE4_8422_2325)
    }
    fn bytes(&mut self, data: &[u8]) {
        for &b in data {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    fn usize(&mut self, v: usize) {
        self.bytes(&(v as u64).to_le_bytes());
    }
    fn i32s(&mut self, v: &[i32]) {
        self.usize(v.len());
        for x in v {
            self.bytes(&x.to_le_bytes());
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        self.usize(v.len());
        for x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
}

/// スカラー導入前から `Batch` にある全 field を畳む。
fn digest_batch(h: &mut Fnv64, batch: &Batch) {
    h.usize(batch.batch_size);
    h.bytes(batch.feature_set.canonical_name().as_bytes());
    h.usize(batch.feature_set.ft_in());
    h.usize(batch.max_active);
    h.i32s(&batch.stm_indices);
    h.i32s(&batch.nstm_indices);
    h.i32s(&batch.nnz);
    h.f32s(&batch.score);
    h.f32s(&batch.wdl);
    h.f32s(&batch.per_pos_norm);
    h.usize(batch.n_positions);
}

fn specs() -> Vec<FeatureSetSpec> {
    let mut specs: Vec<FeatureSetSpec> = FeatureSet::ALL.iter().map(|fs| fs.spec()).collect();
    specs.push(
        FeatureSet::HalfKaHmMerged
            .spec()
            .with_threat_profile(ThreatProfile::Full),
    );
    specs
}

fn sample_records() -> Vec<PackedSfenValue> {
    let bytes = std::fs::read(fixture("sample.psv")).unwrap();
    bytes
        .chunks_exact(40)
        .map(|chunk| {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(chunk);
            psv
        })
        .collect()
}

/// 直接 `push` で 64 局面ずつ詰め、2 batch 目は `reset` 後に 36 局面だけ詰める
/// (前 batch の残骸が残る行・slot も含めて固定する)。
fn direct_push_digest(spec: FeatureSetSpec) -> u64 {
    let mut h = Fnv64::new();
    let records = sample_records();
    let mut batch = Batch::with_capacity(64, spec);
    for chunk in records.chunks(64) {
        batch.reset();
        for psv in chunk {
            assert!(batch.push(psv).unwrap());
        }
        digest_batch(&mut h, &batch);
    }
    h.0
}

/// 学習 loader (1 worker で順序を決定的にする) から 5 batch 読む。100 局面の
/// sample を batch 32 で読むので epoch の折り返しも通る。
fn loader_digest(name: &str, spec: FeatureSetSpec) -> u64 {
    let path = fixture(name);
    let size = std::fs::metadata(&path).unwrap().len();
    let mut loader = BucketedPrefetchedLoader::spawn(
        &path,
        32,
        None,
        None,
        1,
        BucketMode::KingRank9,
        spec,
        true,
        9,
        size,
        false,
    )
    .unwrap();
    let mut h = Fnv64::new();
    for _ in 0..5 {
        let (batch, buckets) = loader.next_batch().unwrap().unwrap();
        digest_batch(&mut h, &batch);
        h.i32s(&buckets);
        loader.recycle((batch, buckets));
    }
    h.0
}

fn heldout_digest(spec: FeatureSetSpec) -> u64 {
    let set = HeldoutSet::load(
        &fixture("sample.psv"),
        16,
        None,
        None,
        80,
        &BucketMode::KingRank9,
        spec,
        9,
    )
    .unwrap();
    let mut h = Fnv64::new();
    for (batch, buckets) in set.batches() {
        digest_batch(&mut h, batch);
        h.i32s(buckets);
    }
    h.0
}

/// (feature set の canonical 名, threat 付きか, direct push, PSV loader, HCPE loader, held-out)
const EXPECTED: [(&str, bool, u64, u64, u64, u64); 6] = [
    (
        "halfkp",
        false,
        0x8aa9_65fe_5b7f_79d1,
        0x3cda_e75a_e30e_953d,
        0x3cda_e75a_e30e_953d,
        0xbeea_6e86_99a7_f9de,
    ),
    (
        "halfka-split",
        false,
        0x662f_c526_7778_8be0,
        0x4388_d76e_a8e8_e9d2,
        0x4388_d76e_a8e8_e9d2,
        0xae7d_77ce_896c_7d62,
    ),
    (
        "halfka-merged",
        false,
        0x6d77_f3ad_c3e1_92d0,
        0x4949_ace1_5a9c_b8da,
        0x4949_ace1_5a9c_b8da,
        0x7990_bf14_2390_0962,
    ),
    (
        "halfka-hm-split",
        false,
        0xb281_c83f_5ca0_756d,
        0x3c77_ef30_4b1c_de32,
        0x3c77_ef30_4b1c_de32,
        0xdb61_f412_d425_1e7b,
    ),
    (
        "halfka-hm-merged",
        false,
        0x6f02_5a43_adc2_fd4c,
        0xaec2_5ffc_4c1d_cac7,
        0xaec2_5ffc_4c1d_cac7,
        0x10a2_660a_c1dc_9a52,
    ),
    (
        "halfka-hm-merged",
        true,
        0x3715_ced9_2a05_d30d,
        0x43de_e33c_9438_35e3,
        0x43de_e33c_9438_35e3,
        0xfe7d_445f_665d_477e,
    ),
];

#[test]
fn existing_feature_sets_produce_identical_batches() {
    let specs = specs();
    assert_eq!(specs.len(), EXPECTED.len());
    let mut mismatches = Vec::new();
    for (spec, &(name, threat, direct, psv, hcpe, heldout)) in specs.into_iter().zip(&EXPECTED) {
        assert_eq!(spec.canonical_name(), name);
        assert_eq!(spec.threat_profile().is_some(), threat);
        let got = (
            direct_push_digest(spec),
            loader_digest("sample.psv", spec),
            loader_digest("sample.hcpe", spec),
            heldout_digest(spec),
        );
        if got != (direct, psv, hcpe, heldout) {
            mismatches.push(format!(
                "(\"{name}\", {threat}, {:#018x}, {:#018x}, {:#018x}, {:#018x}),",
                got.0, got.1, got.2, got.3
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "Batch contents changed:\n{}",
        mismatches.join("\n")
    );
}
