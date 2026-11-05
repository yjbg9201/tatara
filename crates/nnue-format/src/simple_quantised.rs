//! Simple 4 層 NNUE の量子化 binary を整数のまま保持し、CPU で量子化 forward を計算する。
//!
//! 整数演算は YaneuraOu の従来 NNUE (`source/eval/nnue/`、`halfkp_256x2-32-32.h` 系の
//! ClippedReLU 構成) の推論と同じ手順に揃える:
//!
//! 1. FT: 視点ごとに i16 の bias へ active feature の i16 weight を加算する
//!    (i16 の加算は wrap する。YaneuraOu は 16bit SIMD 加算を使う)
//! 2. FT 出力: `clamp(acc, 0, 127)` を u8 にして `[手番側 ft_out | 相手側 ft_out]` に並べる。
//!    入玉スカラーを使う構成では、その後ろに入玉スカラー 96 個 (u8) を連結する
//! 3. L1 / L2: `bias (i32) + Σ w (i8) · x (u8)` を `clamp(sum >> 6, 0, 127)` で u8 にする
//!    (算術右シフト)
//! 4. 出力層: `bias + Σ w · x` (i32)。評価値は `raw / fv_scale` (0 方向への切り捨て)
//!
//! GPU を使わずに、export した `.bin` の出力を推論エンジンの評価値と突き合わせるのに使う。

use std::io::{self, Read};

use shogi_features::write_nyugyoku_scalars;
use shogi_format::ShogiBoard;

use crate::simple_weights::{
    SimpleActivation, SimpleId, invalid, pad32, read_fc_hash, read_ft_hash, read_header,
};

/// ClippedReLU で dense 層の出力を 127-scale に戻す右シフト量 (`log2(QB)`)。
const WEIGHT_SCALE_BITS: u32 = 6;

/// Simple 量子化 binary の中身 (整数のまま)。
///
/// dense 層の weight は file と同じ `[out][pad32(in)]` 配置で持つ (padding 列は 0)。
#[derive(Debug, Clone)]
pub struct SimpleQuantised {
    pub id: SimpleId,
    pub fv_scale: i32,
    /// FT bias (`ft_out`)。
    pub ft_b: Vec<i16>,
    /// FT weight (`ft_in * ft_out`、`ft_w[feat * ft_out + out]`)。
    pub ft_w: Vec<i16>,
    pub l1_b: Vec<i32>,
    /// `[l1_out][pad32(combined_dim)]`。
    pub l1_w: Vec<i8>,
    pub l2_b: Vec<i32>,
    /// `[l2_out][pad32(l1_out)]`。
    pub l2_w: Vec<i8>,
    pub l3_b: Vec<i32>,
    /// `[1][pad32(l2_out)]`。
    pub l3_w: Vec<i8>,
}

impl SimpleQuantised {
    /// Simple quantised binary を読む。version・hash・arch 文字列が `expected` から求めた値と
    /// 一致しなければ `InvalidData`。
    pub fn read<R: Read>(reader: &mut R, expected: SimpleId) -> io::Result<Self> {
        expected.validate().map_err(invalid)?;
        let fv_scale = read_header(reader, &expected)?;
        read_ft_hash(reader, &expected)?;
        let ft_b = read_i16s(reader, expected.ft_out)?;
        let ft_w = read_i16s(reader, expected.ft_in() * expected.ft_out)?;
        read_fc_hash(reader, &expected)?;
        let l1_b = read_i32s(reader, expected.l1_out)?;
        let l1_w = read_i8s(reader, expected.l1_out * pad32(expected.combined_dim()))?;
        let l2_b = read_i32s(reader, expected.l2_out)?;
        let l2_w = read_i8s(reader, expected.l2_out * pad32(expected.l1_out))?;
        let l3_b = read_i32s(reader, 1)?;
        let l3_w = read_i8s(reader, pad32(expected.l2_out))?;
        Ok(Self {
            id: expected,
            fv_scale,
            ft_b,
            ft_w,
            l1_b,
            l1_w,
            l2_b,
            l2_w,
            l3_b,
            l3_w,
        })
    }

    /// `board` の手番側から見た出力層の生の値 (i32)。CReLU の構成のみ対応。
    pub fn forward_raw(&self, board: &ShogiBoard) -> Result<i32, String> {
        Ok(self.forward_trace(board)?.output)
    }

    /// [`forward_raw`](Self::forward_raw) と同じ計算で、各層の入力・出力も返す。
    pub fn forward_trace(&self, board: &ShogiBoard) -> Result<ForwardTrace, String> {
        let id = &self.id;
        if id.activation != SimpleActivation::CReLU {
            return Err(format!(
                "quantised CPU forward supports only the crelu activation (got {})",
                id.activation.canonical_name()
            ));
        }
        let ft_out = id.ft_out;
        let max_active = id.feature_set.max_active();
        let mut stm = vec![-1i32; max_active];
        let mut nstm = vec![-1i32; max_active];
        let n = id
            .feature_set
            .extract_active_features(board, &mut stm, &mut nstm);
        if n > max_active {
            return Err(format!(
                "active feature count {n} exceeds max_active {max_active}"
            ));
        }

        let l1_in = id.combined_dim();
        let mut input = vec![0u8; pad32(l1_in)];
        for (p, indices) in [&stm, &nstm].into_iter().enumerate() {
            let mut acc = self.ft_b.clone();
            for &feat in &indices[..n] {
                let row = &self.ft_w[feat as usize * ft_out..(feat as usize + 1) * ft_out];
                for (a, &w) in acc.iter_mut().zip(row) {
                    *a = a.wrapping_add(w);
                }
            }
            for (dst, &a) in input[p * ft_out..(p + 1) * ft_out].iter_mut().zip(&acc) {
                *dst = a.clamp(0, 127) as u8;
            }
        }
        if let Some(config) = id.nyugyoku_scalar {
            let start = id.ft_output_dim();
            write_nyugyoku_scalars(board, config, &mut input[start..l1_in]);
        }

        let hidden1 = affine_clipped(&input, &self.l1_w, &self.l1_b, id.l1_out);
        let hidden2 = affine_clipped(&hidden1, &self.l2_w, &self.l2_b, id.l2_out);
        let output = affine_raw(&hidden2, &self.l3_w, self.l3_b[0]);
        input.truncate(l1_in);
        let (mut hidden1, mut hidden2) = (hidden1, hidden2);
        hidden1.truncate(id.l1_out);
        hidden2.truncate(id.l2_out);
        Ok(ForwardTrace {
            input,
            hidden1,
            hidden2,
            output,
        })
    }

    /// `board` の手番側から見た評価値 (`forward_raw / fv_scale`、0 方向へ切り捨て)。
    pub fn evaluate(&self, board: &ShogiBoard) -> Result<i32, String> {
        Ok(self.forward_raw(board)? / self.fv_scale)
    }
}

/// 量子化 forward の各層の値。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardTrace {
    /// L1 の入力 (`combined_dim` 個、FT 出力と入玉スカラー)。
    pub input: Vec<u8>,
    /// L1 の ClippedReLU 後 (`l1_out` 個)。
    pub hidden1: Vec<u8>,
    /// L2 の ClippedReLU 後 (`l2_out` 個)。
    pub hidden2: Vec<u8>,
    /// 出力層の生の値。
    pub output: i32,
}

/// `input` は `pad32(in_dim)` 長 (padding は 0)。出力も次の層の入力として `pad32(out_dim)` 長。
fn affine_clipped(input: &[u8], weights: &[i8], biases: &[i32], out_dim: usize) -> Vec<u8> {
    let padded_in = input.len();
    let mut out = vec![0u8; pad32(out_dim)];
    for (o, dst) in out.iter_mut().take(out_dim).enumerate() {
        let row = &weights[o * padded_in..(o + 1) * padded_in];
        let sum = affine_raw(input, row, biases[o]);
        *dst = (sum >> WEIGHT_SCALE_BITS).clamp(0, 127) as u8;
    }
    out
}

fn affine_raw(input: &[u8], row: &[i8], bias: i32) -> i32 {
    row.iter()
        .zip(input)
        .fold(bias, |acc, (&w, &x)| acc + i32::from(w) * i32::from(x))
}

fn read_i16s<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<i16>> {
    let mut bytes = vec![0u8; n * 2];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn read_i32s<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<i32>> {
    let mut bytes = vec![0u8; n * 4];
    reader.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i8s<R: Read>(reader: &mut R, n: usize) -> io::Result<Vec<i8>> {
    let mut bytes = vec![0u8; n];
    reader.read_exact(&mut bytes)?;
    Ok(bytes.into_iter().map(|b| b as i8).collect())
}
