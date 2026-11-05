//! PackedSfen エンコーダのテスト: 既存の教師データの再符号化がバイト単位で一致すること、
//! SFEN → PackedSfen → 復号の往復で局面が保たれること。

use std::path::PathBuf;

use shogi_format::{PackedSfen, PackedSfenValue, ShogiBoard};

fn sample_psv() -> Vec<PackedSfenValue> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample.psv");
    std::fs::read(path)
        .unwrap()
        .chunks_exact(40)
        .map(|chunk| {
            let mut psv = PackedSfenValue::default();
            psv.as_bytes_mut().copy_from_slice(chunk);
            psv
        })
        .collect()
}

fn assert_same_position(a: &ShogiBoard, b: &ShogiBoard, label: &str) {
    assert_eq!(a.board, b.board, "{label}: board");
    assert_eq!(
        a.black_hand.counts, b.black_hand.counts,
        "{label}: black hand"
    );
    assert_eq!(
        a.white_hand.counts, b.white_hand.counts,
        "{label}: white hand"
    );
    assert_eq!(a.black_king_sq, b.black_king_sq, "{label}: black king");
    assert_eq!(a.white_king_sq, b.white_king_sq, "{label}: white king");
    assert_eq!(a.side_to_move, b.side_to_move, "{label}: side to move");
}

/// 同じ手駒でも、金と角・飛を両方持つ手番があると、手駒の書き出し順によってバイト列が
/// 変わる (本エンコーダと現行の YaneuraOu `SfenPacker::pack` は 歩香桂銀金角飛、
/// `sample.psv` を作った packer は 歩香桂銀角飛金)。復号はどちらの順でも同じ局面になる。
fn hand_order_matters(board: &ShogiBoard) -> bool {
    [&board.black_hand, &board.white_hand]
        .iter()
        .any(|h| h.gold() > 0 && (h.bishop() > 0 || h.rook() > 0))
}

/// 教師データを復号して符号化し直すと、手駒の書き出し順が影響しない局面では
/// PackedSfen 部分が元のバイト列と一致し、全局面で復号結果が元の局面と一致する。
#[test]
fn reencoding_sample_psv_matches_original() {
    let records = sample_psv();
    assert_eq!(records.len(), 100);
    let mut order_dependent = 0;
    for (i, psv) in records.iter().enumerate() {
        let board = psv.decode();
        let rebuilt = PackedSfenValue::from_board(&board).unwrap();
        assert_same_position(&board, &rebuilt.decode(), &format!("record {i}"));
        assert_eq!(rebuilt.score(), psv.score(), "record {i}");
        assert_eq!(rebuilt.game_ply(), psv.game_ply(), "record {i}");
        assert_eq!(rebuilt.game_result(), psv.game_result(), "record {i}");
        if hand_order_matters(&board) {
            order_dependent += 1;
            continue;
        }
        let packed = PackedSfen::from_board(&board).unwrap();
        assert_eq!(&packed.data, &psv.as_bytes()[..32], "record {i}");
    }
    // バイト比較を省いた局面の数を固定し、判定条件の緩みで検査が空洞化しないようにする。
    assert_eq!(order_dependent, 11);
}

/// 駒が欠けた局面 (駒箱あり)・成駒・両者の持ち駒を含む局面が往復で保たれる。
#[test]
fn sfen_round_trip_through_packed_sfen() {
    for sfen in [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
        "ln2k1+R2/1r4GK1/p1pp1ppS+B/4p4/9/2P6/PP1PPPP1P/9/LNSG1GSNL b 2Pg 1",
        "4+P4/2+R2K3/1+P1G1+P3/9/9/9/3+p1g1+p1/3k2+r2/4+p4 b B6Pb6p 1",
        "R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b w 2S2N2L4P2s2n2l2p 1",
        // 玉以外の駒が 1 枚も無い局面 (全駒が駒箱)。
        "4k4/9/9/9/9/9/9/9/4K4 b - 1",
    ] {
        let mut board = ShogiBoard::from_sfen(sfen).unwrap();
        board.score = -321;
        board.result = -1;
        let psv = PackedSfenValue::from_board(&board).unwrap();
        let decoded = psv.decode();
        assert_same_position(&board, &decoded, sfen);
        assert_eq!(decoded.score, -321);
        assert_eq!(decoded.result, -1);
        assert_eq!(decoded.ply, board.ply);
    }
}

#[test]
fn rejects_positions_that_do_not_fit_a_piece_set() {
    // 歩が 19 枚。
    let too_many =
        ShogiBoard::from_sfen("4k4/9/9/9/9/9/PPPPPPPPP/9/4K4 b 10P 1").expect("valid sfen");
    assert!(PackedSfen::from_board(&too_many).is_err());
    // 後手玉がいない。
    let no_king = ShogiBoard::from_sfen("9/9/9/9/9/9/9/9/4K4 b - 1").expect("valid sfen");
    assert!(PackedSfen::from_board(&no_king).is_err());
}
