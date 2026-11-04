//! SFEN 文字列から [`ShogiBoard`] を組み立てる。
//!
//! 受け付けるのは `<盤面> <手番> <持ち駒> <手数>` の 4 フィールド (USI の
//! `position sfen` の引数部分)。合法性 (二歩・行き所のない駒・王手放置など) は
//! 検査しない。

use crate::packed_sfen::ShogiBoard;
use crate::types::{Color, Hand, Piece, PieceType, Square};

impl ShogiBoard {
    /// SFEN 文字列を parse する。`score` / `result` は 0 のまま。
    ///
    /// 両玉が盤上にない局面も受け付ける (玉の位置は [`Square::NONE`] のまま)。
    pub fn from_sfen(sfen: &str) -> Result<Self, String> {
        let mut parts = sfen.split_whitespace();
        let board_part = parts.next().ok_or("missing board")?;
        let side_part = parts.next().ok_or("missing side-to-move")?;
        let hand_part = parts.next().ok_or("missing hand")?;
        let ply_part = parts.next().ok_or("missing ply")?;
        if let Some(extra) = parts.next() {
            return Err(format!("unexpected trailing field: {extra}"));
        }

        let mut board = ShogiBoard {
            side_to_move: parse_side(side_part)?,
            ply: ply_part
                .parse::<u16>()
                .map_err(|_| format!("invalid ply: {ply_part}"))?,
            ..Default::default()
        };
        parse_board(board_part, &mut board)?;
        parse_hand(hand_part, &mut board)?;
        Ok(board)
    }
}

fn parse_side(s: &str) -> Result<Color, String> {
    match s {
        "b" => Ok(Color::Black),
        "w" => Ok(Color::White),
        _ => Err(format!("invalid side-to-move: {s}")),
    }
}

fn parse_board(board_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    let ranks: Vec<&str> = board_part.split('/').collect();
    if ranks.len() != 9 {
        return Err(format!("board must have 9 ranks: {board_part}"));
    }
    for (rank, row) in ranks.iter().enumerate() {
        // SFEN の各段は 9 筋 → 1 筋の順に並ぶ。
        let mut file_from_left = 0usize;
        let mut promoted = false;
        for ch in row.chars() {
            if ch == '+' {
                if promoted {
                    return Err(format!("duplicate promotion marker in rank: {row}"));
                }
                promoted = true;
                continue;
            }
            if let Some(skip) = ch.to_digit(10) {
                if promoted {
                    return Err(format!(
                        "promotion marker before empty squares in rank: {row}"
                    ));
                }
                file_from_left += skip as usize;
                continue;
            }
            if file_from_left >= 9 {
                return Err(format!("too many files in rank: {row}"));
            }
            let color = color_of(ch);
            let mut pt = piece_type_from_sfen(ch)?;
            if promoted {
                if !can_promote(pt) {
                    return Err(format!("piece cannot be promoted: +{ch}"));
                }
                pt = pt.promote();
                promoted = false;
            }
            let sq = Square::new((8 - file_from_left) as u8, rank as u8);
            board.board[sq.index()] = Piece::new(color, pt);
            if pt == PieceType::King {
                let king_sq = match color {
                    Color::Black => &mut board.black_king_sq,
                    Color::White => &mut board.white_king_sq,
                };
                if *king_sq != Square::NONE {
                    return Err(format!("multiple kings of the same color: {board_part}"));
                }
                *king_sq = sq;
            }
            file_from_left += 1;
        }
        if promoted {
            return Err(format!("dangling promotion marker in rank: {row}"));
        }
        if file_from_left != 9 {
            return Err(format!("rank does not contain 9 files: {row}"));
        }
    }
    Ok(())
}

fn parse_hand(hand_part: &str, board: &mut ShogiBoard) -> Result<(), String> {
    if hand_part == "-" {
        return Ok(());
    }
    let mut count = 0u8;
    for ch in hand_part.chars() {
        if let Some(digit) = ch.to_digit(10) {
            count = count
                .saturating_mul(10)
                .saturating_add(u8::try_from(digit).expect("single digit fits u8"));
            continue;
        }
        let n = if count == 0 { 1 } else { count };
        count = 0;
        let pt = piece_type_from_sfen(ch)?;
        if !pt.can_be_in_hand() {
            return Err(format!("piece cannot be in hand: {ch}"));
        }
        hand_mut(board, color_of(ch)).add(pt, n);
    }
    if count != 0 {
        return Err(format!("dangling hand count in: {hand_part}"));
    }
    Ok(())
}

fn color_of(ch: char) -> Color {
    if ch.is_ascii_uppercase() {
        Color::Black
    } else {
        Color::White
    }
}

const fn can_promote(pt: PieceType) -> bool {
    !matches!(pt, PieceType::Gold | PieceType::King)
}

fn piece_type_from_sfen(ch: char) -> Result<PieceType, String> {
    match ch.to_ascii_uppercase() {
        'P' => Ok(PieceType::Pawn),
        'L' => Ok(PieceType::Lance),
        'N' => Ok(PieceType::Knight),
        'S' => Ok(PieceType::Silver),
        'G' => Ok(PieceType::Gold),
        'B' => Ok(PieceType::Bishop),
        'R' => Ok(PieceType::Rook),
        'K' => Ok(PieceType::King),
        _ => Err(format!("invalid piece char: {ch}")),
    }
}

fn hand_mut(board: &mut ShogiBoard, color: Color) -> &mut Hand {
    match color {
        Color::Black => &mut board.black_hand,
        Color::White => &mut board.white_hand,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTPOS: &str = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

    #[test]
    fn startpos_places_pieces_and_kings() {
        let board = ShogiBoard::from_sfen(STARTPOS).unwrap();
        assert_eq!(board.side_to_move, Color::Black);
        assert_eq!(board.ply, 1);
        // 5九 (file index 4, rank index 8) に先手玉、5一に後手玉。
        assert_eq!(board.black_king_sq, Square::new(4, 8));
        assert_eq!(board.white_king_sq, Square::new(4, 0));
        // 8八角 / 2八飛 (先手)、8二飛 / 2二角 (後手)。
        assert_eq!(
            board.board[Square::new(7, 7).index()],
            Piece::new(Color::Black, PieceType::Bishop)
        );
        assert_eq!(
            board.board[Square::new(1, 7).index()],
            Piece::new(Color::Black, PieceType::Rook)
        );
        assert_eq!(
            board.board[Square::new(7, 1).index()],
            Piece::new(Color::White, PieceType::Rook)
        );
        let occupied = board.board.iter().filter(|p| p.is_some()).count();
        assert_eq!(occupied, 40);
    }

    #[test]
    fn promoted_pieces_and_multi_digit_hands() {
        let board = ShogiBoard::from_sfen("4k4/9/9/9/4+B4/9/9/9/4K4 w R10Pb2s 12").unwrap();
        assert_eq!(board.side_to_move, Color::White);
        assert_eq!(board.ply, 12);
        assert_eq!(
            board.board[Square::new(4, 4).index()],
            Piece::new(Color::Black, PieceType::Horse)
        );
        assert_eq!(board.black_hand.rook(), 1);
        assert_eq!(board.black_hand.pawn(), 10);
        assert_eq!(board.white_hand.bishop(), 1);
        assert_eq!(board.white_hand.silver(), 2);
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1 b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL x - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b K 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSG+KGSNL b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSN b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1 extra",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSKKGSNL b - 1",
        ] {
            assert!(ShogiBoard::from_sfen(bad).is_err(), "accepted: {bad}");
        }
    }
}
