//! 入玉スカラー特徴: 入玉宣言の点数勘定・宣言条件までの距離を、L1 に直接連結する
//! 少数の密な値として表す。
//!
//! 出力は [`NYUGYOKU_SCALAR_DIMS`] (= 96) 個の u8 (0〜127)。並びは
//! `[手番側の玉のブロック 44 | 相手側の玉のブロック 44 | 0 × 8]` で、各ブロックは
//! その玉の持ち主から見た量。値の尺度は CReLU 後の FT 出力 (0〜127) と同じで、
//! 学習では `u8 / 127` を f32 入力に使う。
//!
//! ルール表・区切り表・ブロック配置は YaneuraOu 側
//! (`source/eval/nnue/features/nyugyoku_scalar.h`) と同じ定義を持つ。両者の一致は
//! `tests/data/nyugyoku_scalar_golden.tsv` (YaneuraOu 側にも同一内容のコピーがある) で
//! 検査する。
//!
//! 宣言条件の判定は YaneuraOu の `Position::DeclarationWin()` の条件 (b)(c)(d) に
//! 合わせる: 玉が敵陣三段目以内、敵陣内の玉以外の自駒が 10 枚以上、宣言点数
//! (敵陣内の玉以外の自駒＋持ち駒、大駒 5 点・小駒 1 点) が必要点以上。手番・王手・
//! 持ち時間の条件は含めない (両玉について計算するため)。

use shogi_format::ShogiBoard;
use shogi_format::types::{Color, Hand, PieceType, Square};

/// 出力ベクトルの次元 (2 玉分のブロック＋32 の倍数へのパディング)。
pub const NYUGYOKU_SCALAR_DIMS: usize = 96;

/// 玉 1 つ分のブロックの次元。
pub const NYUGYOKU_SCALAR_BLOCK_DIMS: usize = 44;

/// thermometer 符号化で 1 を表す値 (CReLU 後の FT 出力の上限と同じ尺度)。
pub const NYUGYOKU_SCALAR_ONE: u8 = 127;

/// 宣言点数の残り `r` に対する thermometer の区切り (`r <= t` を 1 にする)。
pub const REMAINING_POINT_THRESHOLDS: [u8; 10] = [0, 1, 2, 3, 5, 8, 12, 16, 20, 28];

/// 入玉宣言の点数ルール。ネット側に固定し、エンジンの入玉ルール設定とは独立に使う。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeclarationRule {
    /// CSA ルール (27 点法、平手)。先手 28 点・後手 27 点以上。
    Csa27,
    /// 24 点法。先後とも 31 点以上 (24〜30 点の引き分け帯は特徴化しない)。
    Csa24,
}

impl DeclarationRule {
    /// 全ルール。
    pub const ALL: [DeclarationRule; 2] = [DeclarationRule::Csa27, DeclarationRule::Csa24];

    /// `owner` が宣言に必要な点数。
    pub const fn required_points(self, owner: Color) -> u8 {
        match (self, owner) {
            (DeclarationRule::Csa27, Color::Black) => 28,
            (DeclarationRule::Csa27, Color::White) => 27,
            (DeclarationRule::Csa24, _) => 31,
        }
    }

    /// CLI / golden ファイルで使う名前。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            DeclarationRule::Csa27 => "csa27",
            DeclarationRule::Csa24 => "csa24",
        }
    }

    /// [`canonical_name`](Self::canonical_name) の逆引き。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.canonical_name() == name)
    }
}

/// ブロック内の値の符号化。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarEncoding {
    /// 段数・枚数・残り点数を thermometer (`x >= k` / `r <= t` の 0/1 列) で表す。
    Thermometer,
    /// 玉の段を one-hot、枚数・点数を正規化した 1 値で表す。ブロックの未使用部分は 0。
    OneHot,
}

impl ScalarEncoding {
    /// 全符号化。
    pub const ALL: [ScalarEncoding; 2] = [ScalarEncoding::Thermometer, ScalarEncoding::OneHot];

    /// CLI / golden ファイルで使う名前。
    pub const fn canonical_name(self) -> &'static str {
        match self {
            ScalarEncoding::Thermometer => "thermometer",
            ScalarEncoding::OneHot => "onehot",
        }
    }

    /// [`canonical_name`](Self::canonical_name) の逆引き。未知の名前は `None`。
    pub fn from_canonical_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|e| e.canonical_name() == name)
    }
}

/// 入玉スカラーの設定 (ルール × 符号化)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NyugyokuScalarConfig {
    pub rule: DeclarationRule,
    pub encoding: ScalarEncoding,
}

impl Default for NyugyokuScalarConfig {
    fn default() -> Self {
        Self {
            rule: DeclarationRule::Csa27,
            encoding: ScalarEncoding::Thermometer,
        }
    }
}

/// 玉 1 つについての入玉宣言の状態 (すべて玉の持ち主から見た量)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KingDeclarationState {
    /// 自陣最下段からの前進段数 (0〜8)。
    pub advance: u8,
    /// 玉が敵陣三段目以内にいる。
    pub king_in_enemy_camp: bool,
    /// 敵陣内の玉以外の自駒の枚数。
    pub pieces_in_camp: u8,
    /// 宣言点数 (敵陣内の玉以外の自駒＋持ち駒)。
    pub declaration_points: u8,
    /// 潜在宣言点数 (盤上の玉以外の全自駒＋持ち駒)。
    pub potential_points: u8,
    /// 持ち駒の点数。
    pub hand_points: u8,
    /// 宣言に必要な点数 (ルールと持ち主で決まる)。
    pub required_points: u8,
}

impl KingDeclarationState {
    /// 宣言点数の残り `max(0, 必要点 - 宣言点数)`。
    pub const fn remaining_points(&self) -> u8 {
        self.required_points.saturating_sub(self.declaration_points)
    }

    /// 潜在宣言点数の残り `max(0, 必要点 - 潜在宣言点数)`。
    pub const fn remaining_potential_points(&self) -> u8 {
        self.required_points.saturating_sub(self.potential_points)
    }

    /// 手番・王手・持ち時間を除いた宣言条件をすべて満たす。
    pub const fn declarable(&self) -> bool {
        self.king_in_enemy_camp
            && self.pieces_in_camp >= 10
            && self.declaration_points >= self.required_points
    }
}

/// 駒の宣言点数。角・飛と馬・竜は 5 点、それ以外は 1 点 (玉は呼び出し側で除外する)。
const fn piece_points(pt: PieceType) -> u8 {
    match pt {
        PieceType::Bishop | PieceType::Rook | PieceType::Horse | PieceType::Dragon => 5,
        _ => 1,
    }
}

fn hand_points(hand: &Hand) -> u8 {
    hand.pawn()
        + hand.lance()
        + hand.knight()
        + hand.silver()
        + hand.gold()
        + 5 * (hand.bishop() + hand.rook())
}

/// `owner` から見て自陣最下段から何段前進しているか (0〜8)。
const fn advance_of(sq: Square, owner: Color) -> u8 {
    match owner {
        Color::Black => 8 - sq.rank(),
        Color::White => sq.rank(),
    }
}

/// `owner` から見て敵陣 (三段目以内) か。
const fn in_enemy_camp(sq: Square, owner: Color) -> bool {
    advance_of(sq, owner) >= 6
}

/// `board` の `owner` の玉について入玉宣言の状態を求める。
///
/// # Panics
/// `owner` の玉が盤上にない場合。
pub fn king_declaration_state(
    board: &ShogiBoard,
    owner: Color,
    rule: DeclarationRule,
) -> KingDeclarationState {
    let king_sq = match owner {
        Color::Black => board.black_king_sq,
        Color::White => board.white_king_sq,
    };
    assert!(
        king_sq.is_valid(),
        "entering-king scalars need both kings on the board"
    );
    let hand = match owner {
        Color::Black => &board.black_hand,
        Color::White => &board.white_hand,
    };

    let mut pieces_in_camp = 0u8;
    let mut camp_points = 0u8;
    let mut board_points = 0u8;
    for (index, piece) in board.board.iter().enumerate() {
        if piece.is_none() || piece.color != owner || piece.piece_type == PieceType::King {
            continue;
        }
        let points = piece_points(piece.piece_type);
        board_points += points;
        if in_enemy_camp(Square::from_index(index), owner) {
            pieces_in_camp += 1;
            camp_points += points;
        }
    }
    let hand_points = hand_points(hand);

    KingDeclarationState {
        advance: advance_of(king_sq, owner),
        king_in_enemy_camp: in_enemy_camp(king_sq, owner),
        pieces_in_camp,
        declaration_points: camp_points + hand_points,
        potential_points: board_points + hand_points,
        hand_points,
        required_points: rule.required_points(owner),
    }
}

/// `round(127 * num / den)` (四捨五入は 0.5 を切り上げ)。`num <= den` かつ `den > 0` を
/// caller が保証する。YaneuraOu 側と同じ整数式で計算し、浮動小数の丸め差を避ける。
const fn scale_to_u8(num: u8, den: u8) -> u8 {
    let num = num as u32;
    let den = den as u32;
    ((254 * num + den) / (2 * den)) as u8
}

const fn flag(b: bool) -> u8 {
    if b { NYUGYOKU_SCALAR_ONE } else { 0 }
}

/// 玉 1 つ分のブロックを書く。`out.len() == NYUGYOKU_SCALAR_BLOCK_DIMS`。
///
/// thermometer のブロック配置:
///
/// | offset | 次元 | 内容 |
/// |---|---|---|
/// | 0 | 8 | 前進段数 `d >= 1` … `d >= 8` |
/// | 8 | 1 | 玉が敵陣 |
/// | 9 | 10 | 敵陣内の駒数 `n >= 1` … `n >= 10` |
/// | 19 | 10 | 宣言点数の残り `r <= t` ([`REMAINING_POINT_THRESHOLDS`]) |
/// | 29 | 10 | 潜在宣言点数の残り (区切りは同じ) |
/// | 39 | 1 | 持ち駒点数 / 必要点 (1 で頭打ち) |
/// | 40 | 1 | 宣言条件の成立 |
/// | 41 | 1 | 玉の持ち主が先手 |
/// | 42 | 2 | 予約 (0) |
///
/// one-hot のブロック配置:
///
/// | offset | 次元 | 内容 |
/// |---|---|---|
/// | 0 | 9 | 前進段数 `d == 0` … `d == 8` |
/// | 9 | 1 | 玉が敵陣 |
/// | 10 | 1 | 敵陣内の駒数 / 10 (1 で頭打ち) |
/// | 11 | 1 | 宣言点数 / 必要点 (1 で頭打ち) |
/// | 12 | 1 | 宣言点数の残り / 必要点 |
/// | 13 | 1 | 持ち駒点数 / 必要点 (1 で頭打ち) |
/// | 14 | 1 | 宣言条件の成立 |
/// | 15 | 1 | 玉の持ち主が先手 |
/// | 16 | 28 | 未使用 (0) |
fn write_block(
    state: &KingDeclarationState,
    owner: Color,
    encoding: ScalarEncoding,
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), NYUGYOKU_SCALAR_BLOCK_DIMS);
    out.fill(0);
    let need = state.required_points;
    let hand_ratio = scale_to_u8(state.hand_points.min(need), need);
    match encoding {
        ScalarEncoding::Thermometer => {
            for k in 1..=8u8 {
                out[usize::from(k) - 1] = flag(state.advance >= k);
            }
            out[8] = flag(state.king_in_enemy_camp);
            for k in 1..=10u8 {
                out[8 + usize::from(k)] = flag(state.pieces_in_camp >= k);
            }
            let remaining = state.remaining_points();
            let remaining_potential = state.remaining_potential_points();
            for (i, &t) in REMAINING_POINT_THRESHOLDS.iter().enumerate() {
                out[19 + i] = flag(remaining <= t);
                out[29 + i] = flag(remaining_potential <= t);
            }
            out[39] = hand_ratio;
            out[40] = flag(state.declarable());
            out[41] = flag(owner == Color::Black);
        }
        ScalarEncoding::OneHot => {
            out[usize::from(state.advance)] = NYUGYOKU_SCALAR_ONE;
            out[9] = flag(state.king_in_enemy_camp);
            out[10] = scale_to_u8(state.pieces_in_camp.min(10), 10);
            out[11] = scale_to_u8(state.declaration_points.min(need), need);
            out[12] = scale_to_u8(state.remaining_points(), need);
            out[13] = hand_ratio;
            out[14] = flag(state.declarable());
            out[15] = flag(owner == Color::Black);
        }
    }
}

/// `board` の入玉スカラーを u8 (0〜127) で求める。
///
/// # Panics
/// 両玉のどちらかが盤上にない場合。
pub fn nyugyoku_scalars(
    board: &ShogiBoard,
    config: NyugyokuScalarConfig,
) -> [u8; NYUGYOKU_SCALAR_DIMS] {
    let mut out = [0u8; NYUGYOKU_SCALAR_DIMS];
    let us = board.side_to_move;
    let (ours, rest) = out.split_at_mut(NYUGYOKU_SCALAR_BLOCK_DIMS);
    let theirs = &mut rest[..NYUGYOKU_SCALAR_BLOCK_DIMS];
    for (owner, block) in [(us, ours), (us.opponent(), theirs)] {
        let state = king_declaration_state(board, owner, config.rule);
        write_block(&state, owner, config.encoding, block);
    }
    out
}

/// [`nyugyoku_scalars`] を学習用の f32 (`u8 / 127`) で `out` に書く。
///
/// # Panics
/// `out.len() != NYUGYOKU_SCALAR_DIMS`、または両玉のどちらかが盤上にない場合。
pub fn nyugyoku_scalars_f32(board: &ShogiBoard, config: NyugyokuScalarConfig, out: &mut [f32]) {
    assert_eq!(out.len(), NYUGYOKU_SCALAR_DIMS);
    let scalars = nyugyoku_scalars(board, config);
    for (dst, &q) in out.iter_mut().zip(scalars.iter()) {
        *dst = f32::from(q) / f32::from(NYUGYOKU_SCALAR_ONE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_points_match_yaneuraou_entering_king_rule() {
        // YaneuraOu `Position::update_entering_point()`: EKR_27_POINT は先手 28 / 後手 27、
        // EKR_24_POINT は先後とも 31。
        assert_eq!(DeclarationRule::Csa27.required_points(Color::Black), 28);
        assert_eq!(DeclarationRule::Csa27.required_points(Color::White), 27);
        assert_eq!(DeclarationRule::Csa24.required_points(Color::Black), 31);
        assert_eq!(DeclarationRule::Csa24.required_points(Color::White), 31);
    }

    #[test]
    fn canonical_names_round_trip() {
        for rule in DeclarationRule::ALL {
            assert_eq!(
                DeclarationRule::from_canonical_name(rule.canonical_name()),
                Some(rule)
            );
        }
        for enc in ScalarEncoding::ALL {
            assert_eq!(
                ScalarEncoding::from_canonical_name(enc.canonical_name()),
                Some(enc)
            );
        }
        assert_eq!(DeclarationRule::from_canonical_name("csa27h"), None);
        assert_eq!(ScalarEncoding::from_canonical_name("binary"), None);
    }

    #[test]
    fn scale_to_u8_rounds_half_up_and_saturates_at_127() {
        assert_eq!(scale_to_u8(0, 28), 0);
        assert_eq!(scale_to_u8(28, 28), 127);
        assert_eq!(scale_to_u8(1, 2), 64); // 63.5 → 64
        assert_eq!(scale_to_u8(1, 28), 5); // 4.53 → 5
        assert_eq!(scale_to_u8(3, 10), 38); // 38.1 → 38
    }

    #[test]
    fn startpos_state() {
        let board = ShogiBoard::from_sfen(
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        )
        .unwrap();
        for owner in [Color::Black, Color::White] {
            let s = king_declaration_state(&board, owner, DeclarationRule::Csa27);
            assert_eq!(s.advance, 0);
            assert!(!s.king_in_enemy_camp);
            assert_eq!(s.pieces_in_camp, 0);
            assert_eq!(s.declaration_points, 0);
            // 大駒 2 枚 (10 点) ＋小駒 17 枚。
            assert_eq!(s.potential_points, 27);
            assert!(!s.declarable());
        }
    }

    #[test]
    fn f32_output_is_u8_over_127() {
        let board = ShogiBoard::from_sfen(
            "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L4P2s2n2l2p 1",
        )
        .unwrap();
        let config = NyugyokuScalarConfig::default();
        let q = nyugyoku_scalars(&board, config);
        let mut f = [0f32; NYUGYOKU_SCALAR_DIMS];
        nyugyoku_scalars_f32(&board, config, &mut f);
        for (&a, &b) in q.iter().zip(f.iter()) {
            assert_eq!(f32::from(a) / 127.0, b);
        }
        // パディング 8 次元は常に 0。
        assert!(q[2 * NYUGYOKU_SCALAR_BLOCK_DIMS..].iter().all(|&v| v == 0));
    }
}
