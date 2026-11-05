//! Simple の L1 入力に連結する密入力 (入玉スカラー) を `combined` に書く kernel の reference。
//!
//! ## アルゴリズム
//!
//! ```text
//! combined[bi * combined_stride + col_offset + k] = scalars[bi * scalar_dim + k] / 127
//! ```
//!
//! - 入力は量子化済みの u8 (`0..=127`、127 が 1.0)。エンジンは同じ u8 を L1 に入れるので、
//!   学習は `u8 / 127` の f32 を使う。割り算は IEEE の正しい丸め (kernel も同じ式)
//! - 入力は学習データの定数で、勾配を受け取る parameter ではない。backward には対応する
//!   kernel が無く、`dcombined` の密入力の列は誰も読まない

/// 量子化した密入力の 1.0 に当たる値 (`shogi_features::nyugyoku_scalar::NYUGYOKU_SCALAR_ONE`
/// と同じ値)。
pub const DENSE_INPUT_ONE: f32 = 127.0;

/// 密入力 (u8、`batch × scalar_dim`) を `combined` の
/// `[bi * combined_stride + col_offset, + scalar_dim)` に `u8 / 127` で書く。
///
/// `scalars.len() >= batch * scalar_dim`、`col_offset + scalar_dim <= combined_stride`、
/// `combined.len() >= batch * combined_stride` が前提 (違反は index panic)。
pub fn simple_nyugyoku_scalar_to_combined_cpu(
    scalars: &[u8],
    combined: &mut [f32],
    batch: usize,
    scalar_dim: usize,
    combined_stride: usize,
    col_offset: usize,
) {
    for bi in 0..batch {
        for k in 0..scalar_dim {
            combined[bi * combined_stride + col_offset + k] =
                f32::from(scalars[bi * scalar_dim + k]) / DENSE_INPUT_ONE;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_scaled_values_into_dense_columns_only() {
        let (batch, ft, dim) = (3, 4, 6);
        let stride = 2 * ft + dim;
        let scalars: Vec<u8> = (0..batch * dim).map(|i| (i * 23 % 128) as u8).collect();
        let sentinel = -3.0_f32;
        let mut combined = vec![sentinel; batch * stride];
        simple_nyugyoku_scalar_to_combined_cpu(&scalars, &mut combined, batch, dim, stride, 2 * ft);
        for bi in 0..batch {
            let row = &combined[bi * stride..(bi + 1) * stride];
            assert!(row[..2 * ft].iter().all(|&v| v == sentinel));
            for k in 0..dim {
                assert_eq!(row[2 * ft + k], f32::from(scalars[bi * dim + k]) / 127.0);
            }
        }
        // 0 と 127 は 0.0 と 1.0 ちょうどになる。
        let mut ends = vec![0.0_f32; 2];
        simple_nyugyoku_scalar_to_combined_cpu(&[0, 127], &mut ends, 1, 2, 2, 0);
        assert_eq!(ends, vec![0.0, 1.0]);
    }
}
