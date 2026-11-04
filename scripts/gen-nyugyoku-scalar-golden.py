#!/usr/bin/env python3
"""入玉スカラー特徴の golden ファイルを生成する。

tatara (crates/shogi-features/src/nyugyoku_scalar.rs) と YaneuraOu
(source/eval/nnue/features/nyugyoku_scalar.{h,cpp}) の実装とはコードを共有しない
参照実装で期待値を計算し、両リポのテストが読む TSV を書き出す。ルール表・区切り値・
golden 局面を変えるときは、このファイルの定数を直してから再生成する。

使い方 (tatara のリポジトリルートで):

    python3 scripts/gen-nyugyoku-scalar-golden.py
    python3 scripts/gen-nyugyoku-scalar-golden.py --yaneuraou /path/to/YaneuraOu

出力の最後に表示する FNV-1a 32bit を、tatara の
crates/shogi-features/tests/nyugyoku_scalar.rs の GOLDEN_FNV1A32 と、YaneuraOu の
source/eval/nnue/features/nyugyoku_scalar.cpp の kGoldenFnv1a32 に設定する。
"""

import argparse
import pathlib
import sys

# 宣言に必要な点数 (先手, 後手)。YaneuraOu の Position::update_entering_point() の
# EKR_27_POINT / EKR_24_POINT と同じ値。
REQUIRED_POINTS = {
    'csa27': (28, 27),
    'csa24': (31, 31),
}

# 宣言点数の残り r に対する thermometer の区切り (r <= t を 1 にする)。
REMAINING_POINT_THRESHOLDS = [0, 1, 2, 3, 5, 8, 12, 16, 20, 28]

ENCODINGS = ['thermometer', 'onehot']

BLOCK_DIMS = 44
DIMS = 96
ONE = 127

# 角・飛 (成っても同じ) は 5 点、それ以外の玉以外の駒は 1 点。
BIG_PIECES = {'B', 'R'}

POSITIONS = [
    ('startpos', 'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'),
    ('midgame', 'l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1'),
    ('black_entered', 'ln2k1+R2/1r4GK1/p1pp1ppS+B/4p4/9/2P6/PP1PPPP1P/9/LNSG1GSNL b 2Pg 1'),
    ('both_entered', '4+P4/2+R2K3/1+P1G1+P3/9/9/9/3+p1g1+p1/3k2+r2/4+p4 b B6Pb6p 1'),
    # 先手 28 点ちょうど (宣言可) / 後手 26 点 (27 点の 1 点手前)。
    ('black28_white26', 'R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L4P2s2n2l2p 1'),
    ('black28_white26_wtm', 'R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b w 2S2N2L4P2s2n2l2p 1'),
    # 先手 27 点 (28 点の 1 点手前) / 後手 27 点ちょうど (宣言可)。
    ('black27_white27', 'R7B/3G1G3/PPPPKPP2/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L3P2s2n2l3p 1'),
    # 先手は点数が足りるが敵陣内の駒が 9 枚。
    ('black28_nine_in_camp', 'R7B/3G1G3/PPPPKP3/9/9/9/ppppkpp2/3g1g3/r7b b 2S2N2L5P2s2n2l2p 1'),
]

HEADER = [
    '# 入玉スカラー特徴の golden データ。tatara と YaneuraOu の両方のテストが読む。',
    '# 列: name<TAB>rule<TAB>encoding<TAB>sfen<TAB>scalars (u8 96 個、空白区切り、'
    '[手番側の玉 44 | 相手側の玉 44 | 0 × 8])',
]

DEFAULT_OUTPUT = 'crates/shogi-features/tests/data/nyugyoku_scalar_golden.tsv'
YANEURAOU_COPY = 'source/eval/nnue/features/testdata/nyugyoku_scalar_golden.tsv'


def scale_to_u8(num, den):
    """round(127 * num / den)、0.5 は切り上げ。0 <= num <= den。"""
    return (254 * num + den) // (2 * den)


def parse_sfen(sfen):
    """(手番 0/1, 玉以外の駒 [(色, 文字, 段 1..9)], 玉の段 {色: 段}, 持ち駒点 [先手, 後手])"""
    board, side, hand, _ply = sfen.split()
    pieces = []
    king_rank = {}
    for rank, row in enumerate(board.split('/'), start=1):
        for ch in row:
            if ch == '+' or ch.isdigit():
                continue
            color = 0 if ch.isupper() else 1
            letter = ch.upper()
            if letter == 'K':
                king_rank[color] = rank
            else:
                pieces.append((color, letter, rank))
    hand_points = [0, 0]
    count = 0
    for ch in hand:
        if ch == '-':
            break
        if ch.isdigit():
            count = count * 10 + int(ch)
            continue
        color = 0 if ch.isupper() else 1
        hand_points[color] += (count or 1) * (5 if ch.upper() in BIG_PIECES else 1)
        count = 0
    return (0 if side == 'b' else 1), pieces, king_rank, hand_points


def in_enemy_camp(color, rank):
    return rank <= 3 if color == 0 else rank >= 7


def king_state(color, pieces, king_rank, hand_points, rule):
    points = lambda letter: 5 if letter in BIG_PIECES else 1
    own = [p for p in pieces if p[0] == color]
    camp = [p for p in own if in_enemy_camp(color, p[2])]
    rank = king_rank[color]
    need = REQUIRED_POINTS[rule][color]
    declaration = sum(points(p[1]) for p in camp) + hand_points[color]
    potential = sum(points(p[1]) for p in own) + hand_points[color]
    king_in_camp = in_enemy_camp(color, rank)
    return {
        'advance': 9 - rank if color == 0 else rank - 1,
        'king_in_camp': king_in_camp,
        'pieces_in_camp': len(camp),
        'declaration': declaration,
        'hand': hand_points[color],
        'need': need,
        'remaining': max(0, need - declaration),
        'remaining_potential': max(0, need - potential),
        'declarable': king_in_camp and len(camp) >= 10 and declaration >= need,
        'black': color == 0,
    }


def block(s, encoding):
    flag = lambda b: ONE if b else 0
    out = [0] * BLOCK_DIMS
    need = s['need']
    if encoding == 'thermometer':
        for k in range(1, 9):
            out[k - 1] = flag(s['advance'] >= k)
        out[8] = flag(s['king_in_camp'])
        for k in range(1, 11):
            out[8 + k] = flag(s['pieces_in_camp'] >= k)
        for i, t in enumerate(REMAINING_POINT_THRESHOLDS):
            out[19 + i] = flag(s['remaining'] <= t)
            out[29 + i] = flag(s['remaining_potential'] <= t)
        out[39] = scale_to_u8(min(s['hand'], need), need)
        out[40] = flag(s['declarable'])
        out[41] = flag(s['black'])
    elif encoding == 'onehot':
        out[s['advance']] = ONE
        out[9] = flag(s['king_in_camp'])
        out[10] = scale_to_u8(min(s['pieces_in_camp'], 10), 10)
        out[11] = scale_to_u8(min(s['declaration'], need), need)
        out[12] = scale_to_u8(s['remaining'], need)
        out[13] = scale_to_u8(min(s['hand'], need), need)
        out[14] = flag(s['declarable'])
        out[15] = flag(s['black'])
    else:
        raise ValueError(encoding)
    return out


def scalars(sfen, rule, encoding):
    stm, pieces, king_rank, hand_points = parse_sfen(sfen)
    out = []
    for color in (stm, 1 - stm):
        out += block(king_state(color, pieces, king_rank, hand_points, rule), encoding)
    out += [0] * (DIMS - len(out))
    return out


def render():
    lines = list(HEADER)
    for name, sfen in POSITIONS:
        for rule in REQUIRED_POINTS:
            for encoding in ENCODINGS:
                values = ' '.join(map(str, scalars(sfen, rule, encoding)))
                lines.append('\t'.join([name, rule, encoding, sfen, values]))
    return '\n'.join(lines) + '\n'


def fnv1a32_without_cr(data):
    h = 0x811C9DC5
    for b in data:
        if b == 0x0D:
            continue
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def main():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', type=pathlib.Path, default=repo_root / DEFAULT_OUTPUT,
                        help='golden ファイルの出力先 (既定: tatara の正本の場所)')
    parser.add_argument('--yaneuraou', type=pathlib.Path,
                        help='YaneuraOu のチェックアウト。指定するとコピーも同じ内容で書く')
    args = parser.parse_args()

    data = render().encode('utf-8')
    targets = [args.output]
    if args.yaneuraou is not None:
        targets.append(args.yaneuraou / YANEURAOU_COPY)
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f'wrote {path}')
    print(f'FNV-1a 32bit (CR を除く): 0x{fnv1a32_without_cr(data):08X}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
