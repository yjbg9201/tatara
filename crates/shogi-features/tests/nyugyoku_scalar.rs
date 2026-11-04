//! 入玉スカラー特徴の golden テストと視点反転の property テスト。
//!
//! golden ファイル `tests/data/nyugyoku_scalar_golden.tsv` は YaneuraOu 側
//! (`source/eval/nnue/features/testdata/nyugyoku_scalar_golden.tsv`) にも同一内容の
//! コピーがあり、両リポのテストが同じ期待値で実装を検査する。期待値は実装とコードを
//! 共有しない参照実装 `scripts/gen-nyugyoku-scalar-golden.py` で生成する。再生成したら
//! 両リポのコピーと [`GOLDEN_FNV1A32`] (YaneuraOu 側のテストにも同じ値がある) を
//! スクリプトが表示する値にそろえて更新する。

use shogi_features::nyugyoku_scalar::{
    DeclarationRule, NYUGYOKU_SCALAR_BLOCK_DIMS, NYUGYOKU_SCALAR_DIMS, NyugyokuScalarConfig,
    ScalarEncoding, king_declaration_state, nyugyoku_scalars,
};
use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Hand, Piece, PieceType, Square};

const GOLDEN: &str = include_str!("data/nyugyoku_scalar_golden.tsv");

/// golden ファイルの FNV-1a 32bit (CR を除いたバイト列)。改行コード変換の影響を受けない。
const GOLDEN_FNV1A32: u32 = 0xE063_FF67;

struct GoldenRow {
    name: String,
    config: NyugyokuScalarConfig,
    sfen: String,
    expected: Vec<u8>,
}

fn fnv1a32_without_cr(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for &b in bytes.iter().filter(|&&b| b != b'\r') {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

fn golden_rows() -> Vec<GoldenRow> {
    GOLDEN
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 5, "golden row must have 5 columns: {line}");
            let rule = DeclarationRule::from_canonical_name(cols[1])
                .unwrap_or_else(|| panic!("unknown rule {}", cols[1]));
            let encoding = ScalarEncoding::from_canonical_name(cols[2])
                .unwrap_or_else(|| panic!("unknown encoding {}", cols[2]));
            let expected: Vec<u8> = cols[4]
                .split_whitespace()
                .map(|v| v.parse().expect("u8 value"))
                .collect();
            assert_eq!(expected.len(), NYUGYOKU_SCALAR_DIMS, "row {}", cols[0]);
            GoldenRow {
                name: cols[0].to_string(),
                config: NyugyokuScalarConfig { rule, encoding },
                sfen: cols[3].to_string(),
                expected,
            }
        })
        .collect()
}

#[test]
fn golden_file_checksum_is_pinned() {
    assert_eq!(
        fnv1a32_without_cr(GOLDEN.as_bytes()),
        GOLDEN_FNV1A32,
        "golden file changed: update the YaneuraOu copy and both checksum constants"
    );
}

#[test]
fn golden_covers_required_positions_and_configs() {
    let rows = golden_rows();
    for name in [
        "startpos",
        "midgame",
        "black_entered",
        "both_entered",
        "black28_white26",
        "black27_white27",
    ] {
        for rule in DeclarationRule::ALL {
            for encoding in ScalarEncoding::ALL {
                assert!(
                    rows.iter()
                        .any(|r| r.name == name
                            && r.config == NyugyokuScalarConfig { rule, encoding }),
                    "missing golden row {name} {rule:?} {encoding:?}"
                );
            }
        }
    }
}

#[test]
fn matches_golden() {
    let rows = golden_rows();
    assert!(!rows.is_empty());
    for row in &rows {
        let board = ShogiBoard::from_sfen(&row.sfen).unwrap();
        let got = nyugyoku_scalars(&board, row.config);
        assert_eq!(
            got.as_slice(),
            row.expected.as_slice(),
            "{} {:?} {:?}\n got: {}",
            row.name,
            row.config.rule,
            row.config.encoding,
            got.iter().map(u8::to_string).collect::<Vec<_>>().join(" ")
        );
    }
}

/// 宣言条件の境界 (27 点法: 先手 28 点・後手 27 点) を状態で直接確認する。
#[test]
fn declaration_boundary_states() {
    let declarable = |sfen: &str, owner: Color| {
        let board = ShogiBoard::from_sfen(sfen).unwrap();
        king_declaration_state(&board, owner, DeclarationRule::Csa27)
    };
    // 先手 28 点ちょうど (成立) / 後手 26 点 (1 点手前)。
    let s = "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L4P2s2n2l2p 1";
    let (b, w) = (declarable(s, Color::Black), declarable(s, Color::White));
    assert_eq!((b.declaration_points, b.pieces_in_camp), (28, 10));
    assert!(b.declarable());
    assert_eq!(w.declaration_points, 26);
    assert_eq!(w.remaining_points(), 1);
    assert!(!w.declarable());
    // 先手 27 点 (1 点手前) / 後手 27 点ちょうど (成立)。
    let s = "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L3P2s2n2l3p 1";
    let (b, w) = (declarable(s, Color::Black), declarable(s, Color::White));
    assert_eq!(b.remaining_points(), 1);
    assert!(!b.declarable());
    assert_eq!(w.declaration_points, 27);
    assert!(w.declarable());
    // 点数は足りるが敵陣内の駒が 9 枚。
    let s = "R7B/3G1G3/PPPPKP3/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L5P2s2n2l2p 1";
    let b = declarable(s, Color::Black);
    assert_eq!((b.declaration_points, b.pieces_in_camp), (28, 9));
    assert!(!b.declarable());
}

// ---------------------------------------------------------------------------
// 視点反転 property テスト
// ---------------------------------------------------------------------------

/// 盤面を 180° 回転して先後を入れ替える (駒の色・持ち駒・玉の位置)。手番の色は
/// 変えないので、手番側の玉は反転前の相手側の玉に対応する。
fn flip(board: &ShogiBoard) -> ShogiBoard {
    let mut out = ShogiBoard {
        black_hand: board.white_hand,
        white_hand: board.black_hand,
        black_king_sq: board.white_king_sq.inverse(),
        white_king_sq: board.black_king_sq.inverse(),
        ..board.clone()
    };
    for (index, piece) in board.board.iter().enumerate() {
        let dst = Square::from_index(index).inverse().index();
        out.board[dst] = if piece.is_none() {
            Piece::NONE
        } else {
            Piece::new(piece.color.opponent(), piece.piece_type)
        };
    }
    out
}

struct XorShift64(u64);

impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// 合法性は問わないランダム局面 (両玉＋最大 38 枚の駒＋少数の持ち駒)。
fn random_board(rng: &mut XorShift64) -> ShogiBoard {
    const BOARD_TYPES: [PieceType; 13] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::ProPawn,
        PieceType::ProLance,
        PieceType::ProKnight,
        PieceType::ProSilver,
        PieceType::Horse,
        PieceType::Dragon,
    ];
    const HAND_TYPES: [PieceType; 7] = [
        PieceType::Pawn,
        PieceType::Lance,
        PieceType::Knight,
        PieceType::Silver,
        PieceType::Gold,
        PieceType::Bishop,
        PieceType::Rook,
    ];
    let color_of = |bit: u64| if bit == 0 { Color::Black } else { Color::White };
    let mut board = ShogiBoard {
        side_to_move: color_of(rng.below(2)),
        ..Default::default()
    };
    let mut place = |rng: &mut XorShift64, piece: Piece| loop {
        let sq = Square::from_index(rng.below(81) as usize);
        if board.board[sq.index()].is_none() {
            board.board[sq.index()] = piece;
            return sq;
        }
    };
    let bk = place(rng, Piece::new(Color::Black, PieceType::King));
    let wk = place(rng, Piece::new(Color::White, PieceType::King));
    let n = rng.below(39);
    for _ in 0..n {
        let pt = BOARD_TYPES[rng.below(BOARD_TYPES.len() as u64) as usize];
        let color = color_of(rng.below(2));
        place(rng, Piece::new(color, pt));
    }
    board.black_king_sq = bk;
    board.white_king_sq = wk;
    let random_hand = |rng: &mut XorShift64| {
        let mut hand = Hand::EMPTY;
        for pt in HAND_TYPES {
            hand.set(pt, rng.below(4) as u8);
        }
        hand
    };
    board.black_hand = random_hand(rng);
    board.white_hand = random_hand(rng);
    board
}

fn property_boards() -> Vec<ShogiBoard> {
    let mut rng = XorShift64(0x9E37_79B9_7F4A_7C15);
    let mut boards: Vec<ShogiBoard> = (0..2000).map(|_| random_board(&mut rng)).collect();
    boards.extend(
        golden_rows()
            .iter()
            .map(|r| ShogiBoard::from_sfen(&r.sfen).unwrap()),
    );
    boards
}

/// 先手フラグの offset (符号化ごと)。
const fn black_flag_offset(encoding: ScalarEncoding) -> usize {
    match encoding {
        ScalarEncoding::Thermometer => 41,
        ScalarEncoding::OneHot => 15,
    }
}

/// 先後で必要点が同じルール (24 点法) では、反転後のベクトルは反転前のブロックを
/// 入れ替えたものに等しい (先手フラグだけ反転する)。
#[test]
fn flip_swaps_blocks_under_symmetric_rule() {
    let b = NYUGYOKU_SCALAR_BLOCK_DIMS;
    for board in property_boards() {
        let flipped = flip(&board);
        for encoding in ScalarEncoding::ALL {
            let config = NyugyokuScalarConfig {
                rule: DeclarationRule::Csa24,
                encoding,
            };
            let orig = nyugyoku_scalars(&board, config);
            let got = nyugyoku_scalars(&flipped, config);
            let mut want = [0u8; NYUGYOKU_SCALAR_DIMS];
            want[..b].copy_from_slice(&orig[b..2 * b]);
            want[b..2 * b].copy_from_slice(&orig[..b]);
            let f = black_flag_offset(encoding);
            for base in [0, b] {
                want[base + f] = 127 - want[base + f];
            }
            assert_eq!(got, want, "{encoding:?}");
        }
    }
}

/// 先後で必要点が違うルール (27 点法) でも、必要点に依存しない量は反転で玉の持ち主
/// ごと入れ替わる。
#[test]
fn flip_preserves_rule_independent_state() {
    for board in property_boards() {
        let flipped = flip(&board);
        for owner in [Color::Black, Color::White] {
            let a = king_declaration_state(&board, owner, DeclarationRule::Csa27);
            let z = king_declaration_state(&flipped, owner.opponent(), DeclarationRule::Csa27);
            assert_eq!(a.advance, z.advance);
            assert_eq!(a.king_in_enemy_camp, z.king_in_enemy_camp);
            assert_eq!(a.pieces_in_camp, z.pieces_in_camp);
            assert_eq!(a.declaration_points, z.declaration_points);
            assert_eq!(a.potential_points, z.potential_points);
            assert_eq!(a.hand_points, z.hand_points);
        }
    }
}

#[test]
fn double_flip_is_identity() {
    for board in property_boards() {
        let twice = flip(&flip(&board));
        for rule in DeclarationRule::ALL {
            for encoding in ScalarEncoding::ALL {
                let config = NyugyokuScalarConfig { rule, encoding };
                assert_eq!(
                    nyugyoku_scalars(&board, config),
                    nyugyoku_scalars(&twice, config)
                );
            }
        }
    }
}
