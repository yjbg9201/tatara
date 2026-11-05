//! 入玉スカラーの密入力を持つ Simple ネットの format と量子化 CPU forward のテスト。

use nnue_format::simple_weights::{
    build_arch_str, network_hash, nyugyoku_scalar_hash, nyugyoku_scalar_token,
};
use nnue_format::{SimpleActivation, SimpleId, SimpleQuantised, SimpleWeights};
use shogi_features::{
    DeclarationRule, FeatureSet, NYUGYOKU_SCALAR_DIMS, NyugyokuScalarConfig, ScalarEncoding,
    nyugyoku_scalars,
};
use shogi_format::ShogiBoard;

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn id(ft_out: usize, scalar: Option<NyugyokuScalarConfig>) -> SimpleId {
    SimpleId {
        feature_set: FeatureSet::HalfKp.spec(),
        activation: SimpleActivation::CReLU,
        ft_out,
        l1_out: 32,
        l2_out: 32,
        nyugyoku_scalar: scalar,
    }
}

/// 量子化格子上の値で埋めた決定的な weight (save で丸めが起きない)。
fn deterministic_weights(id: SimpleId) -> SimpleWeights {
    let mut w = SimpleWeights::zeroed(id, 16);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = |range: i64| -> i64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % (2 * range as u64 + 1)) as i64 - range
    };
    for v in w.ft_w.iter_mut().step_by(89) {
        *v = next(60) as f32 / 127.0;
    }
    for v in &mut w.ft_b {
        *v = next(60) as f32 / 127.0;
    }
    for v in w.l1_w.iter_mut().chain(&mut w.l2_w).chain(&mut w.l3_w) {
        *v = next(127) as f32 / 64.0;
    }
    for v in w.l1_b.iter_mut().chain(&mut w.l2_b).chain(&mut w.l3_b) {
        *v = next(100_000) as f32 / (127.0 * 64.0);
    }
    w
}

#[test]
fn combined_dim_and_arch_string() {
    let config = NyugyokuScalarConfig::default();
    let with = id(256, Some(config));
    assert_eq!(with.ft_output_dim(), 512);
    assert_eq!(with.combined_dim(), 512 + NYUGYOKU_SCALAR_DIMS);
    assert_eq!(
        nyugyoku_scalar_token(config),
        "NyugyokuScalar(csa27,thermometer)[96]"
    );
    assert_eq!(
        build_arch_str(&with, 16),
        "Features=HalfKP(Friend)[125388->256x2],Network=AffineTransform[1<-32](\
         ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](\
         AffineTransformSparseInput[32<-608](InputSlice[608(0:512)+\
         NyugyokuScalar(csa27,thermometer)[96](512:608)]))))),fv_scale=16"
    );
}

/// ルール・符号化ごとに hash と arch 文字列が異なり、密入力なしとも異なる。値は推論エンジン
/// 側 (YaneuraOu の入玉スカラー入力層) と一致させる必要があるので固定する。
#[test]
fn hashes_are_pinned_and_distinct() {
    let expected: [(DeclarationRule, ScalarEncoding, u32, u32); 4] = [
        (
            DeclarationRule::Csa27,
            ScalarEncoding::Thermometer,
            0x4102_27c0,
            0x763e_6a3a,
        ),
        (
            DeclarationRule::Csa27,
            ScalarEncoding::OneHot,
            0xc2e8_c2a1,
            0x4615_ae56,
        ),
        (
            DeclarationRule::Csa24,
            ScalarEncoding::Thermometer,
            0x9997_614b,
            0x8f09_02c4,
        ),
        (
            DeclarationRule::Csa24,
            ScalarEncoding::OneHot,
            0x9e24_874c,
            0xac9e_f105,
        ),
    ];
    let mut seen = vec![network_hash(&id(256, None))];
    let mut mismatches = Vec::new();
    for &(rule, encoding, scalar_hash, net_hash) in &expected {
        let config = NyugyokuScalarConfig { rule, encoding };
        let with = id(256, Some(config));
        let got = (nyugyoku_scalar_hash(config), network_hash(&with));
        if got != (scalar_hash, net_hash) {
            mismatches.push(format!(
                "({rule:?}, {encoding:?}, {:#010x}, {:#010x}),",
                got.0, got.1
            ));
        }
        assert!(
            !seen.contains(&got.1),
            "network hash collides for {config:?}"
        );
        seen.push(got.1);
    }
    assert!(
        mismatches.is_empty(),
        "hash changed:\n{}",
        mismatches.join("\n")
    );
}

/// 入玉スカラーつき HalfKP 256x2-32-32 の保存バイト列を固定する (byte-golden)。
#[test]
fn saved_bytes_are_pinned() {
    let expected: [(DeclarationRule, ScalarEncoding, u64); 2] = [
        (
            DeclarationRule::Csa27,
            ScalarEncoding::Thermometer,
            0xcd5d_e529_be81_821d,
        ),
        (
            DeclarationRule::Csa24,
            ScalarEncoding::OneHot,
            0x2ebc_9e3a_7393_d36a,
        ),
    ];
    let mut mismatches = Vec::new();
    for &(rule, encoding, digest) in &expected {
        let with = id(256, Some(NyugyokuScalarConfig { rule, encoding }));
        let mut buf = Vec::new();
        deterministic_weights(with)
            .save_quantised(&mut buf)
            .unwrap();
        // header + FT + fc hash + L1 (bias 32 + 32 × 608) + L2 + L3。
        let arch_len = build_arch_str(&with, 16).len();
        let expected_len = 12
            + arch_len
            + 4
            + 2 * 256
            + 2 * 125_388 * 256
            + 4
            + (4 * 32 + 32 * 608)
            + (4 * 32 + 32 * 32)
            + (4 + 32);
        assert_eq!(buf.len(), expected_len);
        if fnv64(&buf) != digest {
            mismatches.push(format!("({rule:?}, {encoding:?}, {:#018x}),", fnv64(&buf)));
        }
    }
    assert!(
        mismatches.is_empty(),
        "saved bytes changed:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn save_load_round_trip_and_rejections() {
    let config = NyugyokuScalarConfig {
        rule: DeclarationRule::Csa24,
        encoding: ScalarEncoding::OneHot,
    };
    let with = id(32, Some(config));
    let weights = deterministic_weights(with);
    let mut buf = Vec::new();
    weights.save_quantised(&mut buf).unwrap();

    let loaded = SimpleWeights::load(&mut std::io::Cursor::new(&buf), with).unwrap();
    assert_eq!(loaded.l1_w, weights.l1_w);
    assert_eq!(loaded.ft_w, weights.ft_w);
    assert_eq!(loaded.l3_b, weights.l3_b);

    // 密入力の有無・ルール・符号化が違う id では読めない。
    for other in [
        None,
        Some(NyugyokuScalarConfig::default()),
        Some(NyugyokuScalarConfig {
            rule: DeclarationRule::Csa24,
            encoding: ScalarEncoding::Thermometer,
        }),
    ] {
        let expected = id(32, other);
        assert!(SimpleWeights::load(&mut std::io::Cursor::new(&buf), expected).is_err());
        assert!(SimpleQuantised::read(&mut std::io::Cursor::new(&buf), expected).is_err());
    }

    // 密入力は CReLU のみ。
    let screlu = SimpleId {
        activation: SimpleActivation::SCReLU,
        ..with
    };
    assert!(screlu.validate().is_err());
    let mut out = Vec::new();
    assert!(
        SimpleWeights::zeroed(screlu, 16)
            .save_quantised(&mut out)
            .is_err()
    );
}

/// FT の寄与を 0 にし、L1 はスカラー列の 1 本だけを使うネットで、出力が入玉スカラーから
/// 手計算した値と一致する (スカラーが L1 入力の `[ft_output_dim, combined_dim)` に並ぶ)。
#[test]
fn forward_reads_scalars_from_the_tail_columns() {
    let config = NyugyokuScalarConfig::default();
    let with = id(32, Some(config));
    let l1_in = with.combined_dim();
    let mut w = SimpleWeights::zeroed(with, 1);
    // hidden1[o] = clamp((Σ_k scalar[k] * 64 * [k == o+offset]) >> 6, 0, 127) = scalar[o+offset]
    let offset = 19; // 手番側ブロックの「宣言点数の残り」thermometer の先頭
    for o in 0..32 {
        let k = offset + o;
        if k < NYUGYOKU_SCALAR_DIMS {
            w.l1_w[o * l1_in + with.ft_output_dim() + k] = 63.0 / 64.0;
        }
    }
    // hidden2[0] = hidden1[0]、出力 = hidden2[0] * 1。
    w.l2_w[0] = 1.0;
    w.l3_w[0] = 1.0 / 64.0;
    let mut buf = Vec::new();
    w.save_quantised(&mut buf).unwrap();
    let q = SimpleQuantised::read(&mut std::io::Cursor::new(&buf), with).unwrap();

    for sfen in [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L4P2s2n2l2p 1",
        "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L3P2s2n2l3p 1",
    ] {
        let board = ShogiBoard::from_sfen(sfen).unwrap();
        let s = nyugyoku_scalars(&board, config);
        // L1: (s * 63) >> 6、L2: (h1 * 64) >> 6 = h1、出力: h2 * 1。
        let h1 = (i32::from(s[offset]) * 63) >> 6;
        assert_eq!(q.forward_raw(&board).unwrap(), h1, "{sfen}");
        assert_eq!(q.evaluate(&board).unwrap(), h1, "{sfen}");
    }
}
