//! 既存の Simple 構成の量子化 binary (ハッシュ・arch 文字列・バイト列) を固定するテスト。
//!
//! 期待値は入玉スカラーの密入力を Simple に足す前の実装で計算したもので、密入力を
//! 使わない構成の出力が変わっていないことを検査する。

use nnue_format::simple_weights::{build_arch_str, compute_fc_hash, ft_hash, network_hash};
use nnue_format::{SimpleActivation, SimpleId, SimpleWeights};
use shogi_features::FeatureSet;

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// 量子化格子上の値で埋めた決定的な weight (save で丸めが起きない)。
fn deterministic_weights(id: SimpleId) -> SimpleWeights {
    let mut w = SimpleWeights::zeroed(id, 16);
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |range: i64| -> i64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % (2 * range as u64 + 1)) as i64 - range
    };
    let qa = id.activation.qa() as f32;
    for v in w.ft_w.iter_mut().step_by(97) {
        *v = next(60) as f32 / qa;
    }
    for v in &mut w.ft_b {
        *v = next(60) as f32 / qa;
    }
    let bias_scale = (127 * 64) as f32;
    for v in w.l1_w.iter_mut().chain(&mut w.l2_w).chain(&mut w.l3_w) {
        *v = next(127) as f32 / 64.0;
    }
    for v in w.l1_b.iter_mut().chain(&mut w.l2_b).chain(&mut w.l3_b) {
        *v = next(100_000) as f32 / bias_scale;
    }
    w
}

fn cases() -> Vec<(&'static str, SimpleId)> {
    let id = |fs: FeatureSet, activation, ft_out, l1_out, l2_out| SimpleId {
        feature_set: fs.spec(),
        activation,
        ft_out,
        l1_out,
        l2_out,
        nyugyoku_scalar: None,
    };
    vec![
        (
            "halfkp-crelu-256-32-32",
            id(FeatureSet::HalfKp, SimpleActivation::CReLU, 256, 32, 32),
        ),
        (
            "halfkp-crelu-512-8-64",
            id(FeatureSet::HalfKp, SimpleActivation::CReLU, 512, 8, 64),
        ),
        (
            "hm-merged-crelu-256-32-32",
            id(
                FeatureSet::HalfKaHmMerged,
                SimpleActivation::CReLU,
                256,
                32,
                32,
            ),
        ),
        (
            "hm-merged-screlu-256-32-32",
            id(
                FeatureSet::HalfKaHmMerged,
                SimpleActivation::SCReLU,
                256,
                32,
                32,
            ),
        ),
        (
            "hm-merged-pairwise-256-32-32",
            id(
                FeatureSet::HalfKaHmMerged,
                SimpleActivation::Pairwise,
                256,
                32,
                32,
            ),
        ),
    ]
}

/// (名前, network_hash, fc_hash, ft_hash, arch 文字列の FNV-1a 64, 保存バイト列の FNV-1a 64)
const EXPECTED: [(&str, u32, u32, u32, u64, u64); 5] = [
    (
        "halfkp-crelu-256-32-32",
        0x3e5a_a6ee,
        0x6333_7156,
        0x5d69_d7b8,
        0xbd30_a244_c792_c55c,
        0x3e21_a778_dcd9_c1bb,
    ),
    (
        "halfkp-crelu-512-8-64",
        0x3e5a_a71c,
        0x6333_76a4,
        0x5d69_d1b8,
        0x4142_a9b4_fb11_af0a,
        0x4a65_5c70_07fe_a5b3,
    ),
    (
        "hm-merged-crelu-256-32-32",
        0x1c20_3fee,
        0x6333_7156,
        0x7f13_4eb8,
        0x40e8_62aa_e150_52b7,
        0x6c01_717c_04dc_e06a,
    ),
    (
        "hm-merged-screlu-256-32-32",
        0x1c20_3fee,
        0x6333_7156,
        0x7f13_4eb8,
        0xb0e6_f2f5_1ffa_ce23,
        0x946c_d26e_0e82_d024,
    ),
    (
        "hm-merged-pairwise-256-32-32",
        0x1c20_3f8e,
        0x6333_7136,
        0x7f13_4eb8,
        0x84b2_7359_fa06_b112,
        0x0cb2_0e4a_fd7c_847d,
    ),
];

#[test]
fn existing_simple_formats_are_unchanged() {
    let mut mismatches = Vec::new();
    for ((name, id), &(exp_name, net, fc, ft, arch, bytes)) in cases().into_iter().zip(&EXPECTED) {
        assert_eq!(name, exp_name);
        let mut buf = Vec::new();
        deterministic_weights(id).save_quantised(&mut buf).unwrap();
        let got = (
            network_hash(&id),
            compute_fc_hash(
                2 * id.ft_out / id.activation.ft_output_divisor(),
                id.l1_out,
                id.l2_out,
            ),
            ft_hash(id.feature_set.feature_hash(), id.ft_out),
            fnv64(build_arch_str(&id, 16).as_bytes()),
            fnv64(&buf),
        );
        if got != (net, fc, ft, arch, bytes) {
            mismatches.push(format!(
                "(\"{name}\", {:#010x}, {:#010x}, {:#010x}, {:#018x}, {:#018x}),",
                got.0, got.1, got.2, got.3, got.4
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "Simple format changed:\n{}",
        mismatches.join("\n")
    );
}
