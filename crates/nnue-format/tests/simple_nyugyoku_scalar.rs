//! 入玉スカラーの密入力を持つ Simple ネットの format と量子化 CPU forward のテスト。

use nnue_format::simple_weights::{
    build_arch_str, network_hash, nyugyoku_scalar_hash, nyugyoku_scalar_token, pad32,
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

struct XorShift64(u64);

impl XorShift64 {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// 平手と同じ駒の組 (玉以外 38 枚) を盤上・持ち駒にランダムに配った局面。合法性は問わない。
/// 玉は盤上のどこにでも置くので、入玉した局面 (敵陣の玉・敵陣の駒) も多く含む。
fn random_full_set_board(rng: &mut XorShift64) -> shogi_format::ShogiBoard {
    use shogi_format::types::{Color, Hand, Piece, PieceType, Square};
    const PIECES: [(PieceType, Option<PieceType>, u64); 7] = [
        (PieceType::Pawn, Some(PieceType::ProPawn), 18),
        (PieceType::Lance, Some(PieceType::ProLance), 4),
        (PieceType::Knight, Some(PieceType::ProKnight), 4),
        (PieceType::Silver, Some(PieceType::ProSilver), 4),
        (PieceType::Gold, None, 4),
        (PieceType::Bishop, Some(PieceType::Horse), 2),
        (PieceType::Rook, Some(PieceType::Dragon), 2),
    ];
    let color_of = |bit: u64| if bit == 0 { Color::Black } else { Color::White };
    let mut board = ShogiBoard {
        side_to_move: color_of(rng.below(2)),
        ..Default::default()
    };
    let place = |board: &mut ShogiBoard, rng: &mut XorShift64, piece: Piece| loop {
        let sq = Square::from_index(rng.below(81) as usize);
        if board.board[sq.index()].is_none() {
            board.board[sq.index()] = piece;
            return sq;
        }
    };
    board.black_king_sq = place(&mut board, rng, Piece::new(Color::Black, PieceType::King));
    board.white_king_sq = place(&mut board, rng, Piece::new(Color::White, PieceType::King));
    let (mut black_hand, mut white_hand) = (Hand::EMPTY, Hand::EMPTY);
    for (pt, promoted, count) in PIECES {
        for _ in 0..count {
            let color = color_of(rng.below(2));
            if rng.below(4) == 0 {
                let hand = if color == Color::Black {
                    &mut black_hand
                } else {
                    &mut white_hand
                };
                hand.set(pt, hand.count(pt) + 1);
            } else {
                let on_board = match promoted {
                    Some(p) if rng.below(5) == 0 => p,
                    _ => pt,
                };
                place(&mut board, rng, Piece::new(color, on_board));
            }
        }
    }
    board.black_hand = black_hand;
    board.white_hand = white_hand;
    board
}

/// HalfKP 256x2-32-32 の入玉スカラーなしのネットを、入玉スカラーの列の L1 weight 0 で広げて
/// 量子化 CPU forward で比べると、10,000 局面で出力層の生の値が元のネットと完全に一致する
/// (ワームスタート直後の関数保存)。量子化した整数も FT と L1 の FT 列で元と同じになる。
#[test]
fn widening_a_net_without_scalars_keeps_the_function() {
    let config = NyugyokuScalarConfig::default();
    let (without, with) = (id(256, None), id(256, Some(config)));
    let original = deterministic_weights(without);
    let mut original_bytes = Vec::new();
    original.save_quantised(&mut original_bytes).unwrap();

    let zero_scalar_cols = || vec![0.0_f32; with.l1_out * NYUGYOKU_SCALAR_DIMS];
    let (widened, was_widened) = SimpleWeights::load_for_init(
        &mut std::io::Cursor::new(&original_bytes),
        with,
        zero_scalar_cols,
    )
    .unwrap();
    assert!(was_widened);
    assert_eq!(widened.id, with);
    let mut widened_bytes = Vec::new();
    widened.save_quantised(&mut widened_bytes).unwrap();

    let q0 = SimpleQuantised::read(&mut std::io::Cursor::new(&original_bytes), without).unwrap();
    let q1 = SimpleQuantised::read(&mut std::io::Cursor::new(&widened_bytes), with).unwrap();
    assert_eq!(q0.fv_scale, q1.fv_scale);
    assert_eq!(q0.ft_w, q1.ft_w);
    assert_eq!(q0.ft_b, q1.ft_b);
    assert_eq!(
        (&q0.l1_b, &q0.l2_w, &q0.l2_b),
        (&q1.l1_b, &q1.l2_w, &q1.l2_b)
    );
    assert_eq!((&q0.l3_w, &q0.l3_b), (&q1.l3_w, &q1.l3_b));
    let (in0, in1) = (pad32(without.combined_dim()), pad32(with.combined_dim()));
    let ft_dim = with.ft_output_dim();
    for o in 0..with.l1_out {
        assert_eq!(
            q0.l1_w[o * in0..o * in0 + ft_dim],
            q1.l1_w[o * in1..o * in1 + ft_dim]
        );
        assert!(
            q1.l1_w[o * in1 + ft_dim..(o + 1) * in1]
                .iter()
                .all(|&w| w == 0)
        );
    }

    let mut rng = XorShift64(0x2026_1105_0000_0004);
    let boards: Vec<ShogiBoard> = (0..10_000)
        .map(|_| random_full_set_board(&mut rng))
        .collect();
    let king_in_camp = boards
        .iter()
        .filter(|b| nyugyoku_scalars(b, config)[8] != 0)
        .count();
    assert!(
        king_in_camp > 1_000,
        "too few entering-king positions: {king_in_camp}"
    );
    let mut distinct = std::collections::BTreeSet::new();
    for board in &boards {
        let raw = q0.forward_raw(board).unwrap();
        assert_eq!(q1.forward_raw(board).unwrap(), raw);
        distinct.insert(raw);
    }
    assert!(
        distinct.len() > 100,
        "outputs barely vary: {}",
        distinct.len()
    );

    // 入玉スカラーの列に 0 でない weight を足すと出力が変わる (比較が素通りでない)。
    let (noisy, _) =
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&original_bytes), with, || {
            vec![1.0_f32; with.l1_out * NYUGYOKU_SCALAR_DIMS]
        })
        .unwrap();
    let mut noisy_bytes = Vec::new();
    noisy.save_quantised(&mut noisy_bytes).unwrap();
    let q2 = SimpleQuantised::read(&mut std::io::Cursor::new(&noisy_bytes), with).unwrap();
    let changed = boards
        .iter()
        .filter(|b| q2.forward_raw(b).unwrap() != q0.forward_raw(b).unwrap())
        .count();
    assert!(changed > 1_000, "scalar columns had no effect: {changed}");
}

/// `load_for_init` は入玉スカラーつきのネットをそのまま読み (広げない)、入玉スカラーなしの
/// 構成では従来の `load` と同じ。別の構成 (層の幅違い、別ルール) のネットは拒否する。
#[test]
fn load_for_init_reads_matching_nets_and_rejects_others() {
    let config = NyugyokuScalarConfig::default();
    let with = id(32, Some(config));
    let mut with_bytes = Vec::new();
    deterministic_weights(with)
        .save_quantised(&mut with_bytes)
        .unwrap();
    let unused = || -> Vec<f32> { panic!("must not widen a net that already has scalars") };
    let (w, widened) =
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&with_bytes), with, unused).unwrap();
    assert!(!widened);
    assert_eq!(w.id, with);

    let without = id(32, None);
    let mut without_bytes = Vec::new();
    deterministic_weights(without)
        .save_quantised(&mut without_bytes)
        .unwrap();
    let (w, widened) =
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&without_bytes), without, Vec::new)
            .unwrap();
    assert!(!widened);
    assert_eq!(w.id, without);

    let zeros = || vec![0.0_f32; 32 * NYUGYOKU_SCALAR_DIMS];
    // 層の幅が違う入玉スカラーなしのネット
    let mut other_width = Vec::new();
    deterministic_weights(id(64, None))
        .save_quantised(&mut other_width)
        .unwrap();
    assert!(
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&other_width), with, zeros).is_err()
    );
    // 別ルールの入玉スカラーつきのネット
    let csa24 = id(
        32,
        Some(NyugyokuScalarConfig {
            rule: DeclarationRule::Csa24,
            ..config
        }),
    );
    let mut other_rule = Vec::new();
    deterministic_weights(csa24)
        .save_quantised(&mut other_rule)
        .unwrap();
    assert!(
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&other_rule), with, zeros).is_err()
    );
    // 広げる列の数が合わない
    assert!(
        SimpleWeights::load_for_init(&mut std::io::Cursor::new(&without_bytes), with, || vec![
            0.0_f32;
            3
        ])
        .is_err()
    );
}
