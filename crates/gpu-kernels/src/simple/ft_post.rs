//! Simple の FT post (bias 加算 + 活性化 + `combined` への書き込み) と、その逆
//! (`dcombined` からの読み出し + 活性化の勾配) を融合した FP32 kernel の reference。
//!
//! ## アルゴリズム
//!
//! ```text
//! forward  (視点ごとに 1 回、dst_offset = 0 (stm) / ft_out (nstm)):
//!   x                = ft_out[bi][oi] + bias[oi]
//!   ft_out[bi][oi]   = x                                 (backward の指示関数用に保持)
//!   combined[bi * combined_stride + dst_offset + oi] = act(x)
//! backward (視点ごとに 1 回、src_offset = 0 / ft_out):
//!   g                = dcombined[bi * combined_stride + src_offset + oi]
//!   dft_out[bi][oi]  = g * act'(ft_pre_act[bi][oi])
//! ```
//!
//! - CReLU: `act(x) = clip(x, 0, 1)`。forward は kernel と同じく `x <= 0 → 0`、
//!   `x >= 1 → 1`。勾配は strict (`0 < x < 1` のときだけ `g`)
//! - SCReLU: `act(x) = clip(x, 0, 1)²`、`act'(x) = 2a` (`a = clip(x, 0, 1)`、`0 < a < 1`
//!   の外は 0)。backward は kernel と同じく `g * act'` を掛ける (`g` が NaN なら
//!   `act' = 0` でも NaN)
//! - `combined_stride >= 2 * ft_out` が前提。各行の `[0, 2 * ft_out)` の外 (密入力の列) は
//!   forward が書かず、backward が読まない

/// CReLU の fused FT post forward reference。`ft_out` に bias を in-place 加算し、
/// `combined` の `[bi * combined_stride + dst_offset, + ft_out_dim)` に活性化値を書く。
///
/// `ft_out.len() >= batch * ft_out_dim`、`bias.len() >= ft_out_dim`、
/// `dst_offset + ft_out_dim <= combined_stride`、`combined.len() >= batch * combined_stride`
/// が前提 (違反は index panic)。
pub fn simple_ft_post_fused_crelu_cpu(
    ft_out: &mut [f32],
    bias: &[f32],
    combined: &mut [f32],
    batch: usize,
    ft_out_dim: usize,
    combined_stride: usize,
    dst_offset: usize,
) {
    ft_post(
        ft_out,
        bias,
        combined,
        (batch, ft_out_dim, combined_stride, dst_offset),
        crelu_forward,
    );
}

/// SCReLU の fused FT post forward reference。前提は
/// [`simple_ft_post_fused_crelu_cpu`] と同じ。
pub fn simple_ft_post_fused_screlu_cpu(
    ft_out: &mut [f32],
    bias: &[f32],
    combined: &mut [f32],
    batch: usize,
    ft_out_dim: usize,
    combined_stride: usize,
    dst_offset: usize,
) {
    ft_post(
        ft_out,
        bias,
        combined,
        (batch, ft_out_dim, combined_stride, dst_offset),
        screlu_forward,
    );
}

/// CReLU の fused FT activation backward reference。`dcombined` の
/// `[bi * combined_stride + src_offset, + ft_out_dim)` を読み、`ft_pre_act` の指示関数を
/// 掛けて `dft_out` (`batch × ft_out_dim`) に書く。
///
/// `src_offset + ft_out_dim <= combined_stride`、`dcombined.len() >= batch * combined_stride`
/// が前提。
pub fn simple_bwd_ft_act_crelu_fused_cpu(
    dcombined: &[f32],
    ft_pre_act: &[f32],
    dft_out: &mut [f32],
    batch: usize,
    ft_out_dim: usize,
    combined_stride: usize,
    src_offset: usize,
) {
    for bi in 0..batch {
        for oi in 0..ft_out_dim {
            let i = bi * ft_out_dim + oi;
            let x = ft_pre_act[i];
            dft_out[i] = if x > 0.0_f32 && x < 1.0_f32 {
                dcombined[bi * combined_stride + src_offset + oi]
            } else {
                0.0_f32
            };
        }
    }
}

/// SCReLU の fused FT activation backward reference。前提は
/// [`simple_bwd_ft_act_crelu_fused_cpu`] と同じ。
pub fn simple_bwd_ft_act_screlu_fused_cpu(
    dcombined: &[f32],
    ft_pre_act: &[f32],
    dft_out: &mut [f32],
    batch: usize,
    ft_out_dim: usize,
    combined_stride: usize,
    src_offset: usize,
) {
    for bi in 0..batch {
        for oi in 0..ft_out_dim {
            let i = bi * ft_out_dim + oi;
            let a = clip_unit(ft_pre_act[i]);
            let derivative = if a > 0.0_f32 && a < 1.0_f32 {
                2.0_f32 * a
            } else {
                0.0_f32
            };
            dft_out[i] = dcombined[bi * combined_stride + src_offset + oi] * derivative;
        }
    }
}

/// (batch, ft_out_dim, combined_stride, dst_offset)
type FtPostShape = (usize, usize, usize, usize);

fn ft_post(
    ft_out: &mut [f32],
    bias: &[f32],
    combined: &mut [f32],
    (batch, ft_out_dim, combined_stride, dst_offset): FtPostShape,
    act: fn(f32) -> f32,
) {
    for bi in 0..batch {
        for oi in 0..ft_out_dim {
            let i = bi * ft_out_dim + oi;
            let x = ft_out[i] + bias[oi];
            ft_out[i] = x;
            combined[bi * combined_stride + dst_offset + oi] = act(x);
        }
    }
}

// kernel の三項演算子展開と同じ比較で書く (f32::clamp は cuda-oxide で lower できず、
// kernel 側は if-else。NaN はどの比較も false なのでそのまま通る)。
#[allow(clippy::manual_clamp)]
fn crelu_forward(x: f32) -> f32 {
    if x <= 0.0_f32 {
        0.0_f32
    } else if x >= 1.0_f32 {
        1.0_f32
    } else {
        x
    }
}

#[allow(clippy::manual_clamp)]
fn clip_unit(x: f32) -> f32 {
    if x < 0.0_f32 {
        0.0_f32
    } else if x > 1.0_f32 {
        1.0_f32
    } else {
        x
    }
}

fn screlu_forward(x: f32) -> f32 {
    let a = clip_unit(x);
    a * a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layerstack::crelu::{crelu_fwd_cpu, crelu_grad_cpu};
    use crate::layerstack::slice2d::{slice_extract_2d_cpu, slice_scatter_2d_cpu};
    use crate::pointwise::screlu_fwd::screlu_fwd_cpu;
    use crate::pointwise::screlu_grad::screlu_grad_cpu;

    const BATCH: usize = 5;
    const FT: usize = 24;
    /// 入玉スカラー 96 列を足した行幅に相当する、`2 * FT` より広い行幅。
    const WIDE: usize = 2 * FT + 96;

    /// 境界値 (ちょうど 0 / 1、負、1 超、NaN) を含む FT 出力と bias。
    fn inputs() -> (Vec<f32>, Vec<f32>) {
        let mut ft: Vec<f32> = (0..BATCH * FT)
            .map(|i| -1.25_f32 + 2.5_f32 * ((i * 37) % 41) as f32 / 40.0)
            .collect();
        let bias: Vec<f32> = (0..FT).map(|i| -0.3_f32 + 0.05 * (i % 13) as f32).collect();
        // bias 加算後にちょうど 0 / 1 になる cell と NaN。
        ft[3] = -bias[3];
        ft[FT + 4] = 1.0 - bias[4];
        ft[2 * FT + 5] = f32::NAN;
        (ft, bias)
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()))
    }

    /// 行幅を `2 * ft_out` に固定していた kernel の書き込み位置 (`bi * (2 * ft_out) +
    /// dst_offset + oi`) をそのまま書き写した forward。
    fn fixed_stride_forward(
        ft_out: &mut [f32],
        bias: &[f32],
        combined: &mut [f32],
        dst_offset: usize,
        act: fn(f32) -> f32,
    ) {
        for (i, cell) in ft_out.iter_mut().enumerate().take(BATCH * FT) {
            let (bi, oi) = (i / FT, i % FT);
            let x = *cell + bias[oi];
            *cell = x;
            combined[bi * (2 * FT) + dst_offset + oi] = act(x);
        }
    }

    type FusedForward = fn(&mut [f32], &[f32], &mut [f32], usize, usize, usize, usize);
    type FusedBackward = fn(&[f32], &[f32], &mut [f32], usize, usize, usize, usize);

    fn run_forward(f: FusedForward, stride: usize, fill: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let (mut stm, bias) = inputs();
        let mut nstm: Vec<f32> = stm.iter().rev().copied().collect();
        let mut combined = vec![fill; BATCH * stride];
        f(&mut stm, &bias, &mut combined, BATCH, FT, stride, 0);
        f(&mut nstm, &bias, &mut combined, BATCH, FT, stride, FT);
        (stm, nstm, combined)
    }

    #[test]
    fn forward_with_stride_two_ft_out_matches_fixed_stride_kernel() {
        for (fused, act) in [
            (
                simple_ft_post_fused_crelu_cpu as FusedForward,
                crelu_forward as fn(f32) -> f32,
            ),
            (simple_ft_post_fused_screlu_cpu, screlu_forward),
        ] {
            let (stm, nstm, combined) = run_forward(fused, 2 * FT, 0.0);
            let (mut stm_ref, bias) = inputs();
            let mut nstm_ref: Vec<f32> = stm_ref.iter().rev().copied().collect();
            let mut combined_ref = vec![0.0_f32; BATCH * 2 * FT];
            fixed_stride_forward(&mut stm_ref, &bias, &mut combined_ref, 0, act);
            fixed_stride_forward(&mut nstm_ref, &bias, &mut combined_ref, FT, act);
            assert!(same(&stm, &stm_ref) && same(&nstm, &nstm_ref));
            assert!(same(&combined, &combined_ref));
        }
    }

    #[test]
    fn forward_with_stride_two_ft_out_matches_unfused_kernels() {
        // 融合前の kernel 列 (bias 加算 → 活性化 → slice_scatter_2d) と同じ値になる。
        type Act = fn(&[f32], &mut [f32], usize);
        for (fused, act) in [
            (
                simple_ft_post_fused_crelu_cpu as FusedForward,
                crelu_fwd_cpu as Act,
            ),
            (simple_ft_post_fused_screlu_cpu, screlu_fwd_cpu),
        ] {
            let (_, _, combined) = run_forward(fused, 2 * FT, 0.0);
            let (stm, bias) = inputs();
            let nstm: Vec<f32> = stm.iter().rev().copied().collect();
            let mut combined_ref = vec![0.0_f32; BATCH * 2 * FT];
            for (src, offset) in [(&stm, 0), (&nstm, FT)] {
                let pre: Vec<f32> = src
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| x + bias[i % FT])
                    .collect();
                let mut acted = vec![0.0_f32; BATCH * FT];
                act(&pre, &mut acted, BATCH * FT);
                slice_scatter_2d_cpu(&acted, &mut combined_ref, BATCH, FT, 2 * FT, offset);
            }
            // 符号付きゼロの違い (CReLU の `x <= 0` と `x < 0`) は値として等しい。
            assert!(
                combined
                    .iter()
                    .zip(&combined_ref)
                    .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
            );
        }
    }

    #[test]
    fn forward_with_wider_stride_writes_ft_columns_only() {
        for fused in [
            simple_ft_post_fused_crelu_cpu as FusedForward,
            simple_ft_post_fused_screlu_cpu,
        ] {
            let (stm_n, nstm_n, narrow) = run_forward(fused, 2 * FT, 0.0);
            let sentinel = -7.5_f32;
            let (stm_w, nstm_w, wide) = run_forward(fused, WIDE, sentinel);
            assert!(same(&stm_n, &stm_w) && same(&nstm_n, &nstm_w));
            for bi in 0..BATCH {
                assert!(same(
                    &wide[bi * WIDE..bi * WIDE + 2 * FT],
                    &narrow[bi * 2 * FT..(bi + 1) * 2 * FT]
                ));
                assert!(
                    wide[bi * WIDE + 2 * FT..(bi + 1) * WIDE]
                        .iter()
                        .all(|&v| v == sentinel)
                );
            }
        }
    }

    fn backward_inputs(stride: usize) -> (Vec<f32>, Vec<f32>) {
        let (ft, bias) = inputs();
        let pre: Vec<f32> = ft
            .iter()
            .enumerate()
            .map(|(i, &x)| x + bias[i % FT])
            .collect();
        let mut dcombined: Vec<f32> = (0..BATCH * stride)
            .map(|i| -2.0_f32 + 0.0173 * i as f32)
            .collect();
        // 密入力の列の勾配は FT の backward に入らない (NaN を置いても漏れない)。
        for bi in 0..BATCH {
            for v in &mut dcombined[bi * stride + 2 * FT..(bi + 1) * stride] {
                *v = f32::NAN;
            }
        }
        (pre, dcombined)
    }

    #[test]
    fn backward_with_stride_two_ft_out_matches_unfused_kernels() {
        // 融合前の kernel 列 (slice_extract_2d → 活性化の勾配) と bit 単位で一致する。
        type Grad = fn(&[f32], &[f32], &mut [f32], usize);
        for (fused, grad) in [
            (
                simple_bwd_ft_act_crelu_fused_cpu as FusedBackward,
                crelu_grad_cpu as Grad,
            ),
            (simple_bwd_ft_act_screlu_fused_cpu, screlu_grad_cpu),
        ] {
            let (pre, dcombined) = backward_inputs(2 * FT);
            for offset in [0, FT] {
                let mut dft = vec![0.0_f32; BATCH * FT];
                fused(&dcombined, &pre, &mut dft, BATCH, FT, 2 * FT, offset);
                let mut extracted = vec![0.0_f32; BATCH * FT];
                slice_extract_2d_cpu(&dcombined, &mut extracted, BATCH, 2 * FT, offset, FT);
                let mut dft_ref = vec![0.0_f32; BATCH * FT];
                grad(&pre, &extracted, &mut dft_ref, BATCH * FT);
                assert!(same(&dft, &dft_ref));
            }
        }
    }

    #[test]
    fn backward_with_wider_stride_ignores_dense_columns() {
        for fused in [
            simple_bwd_ft_act_crelu_fused_cpu as FusedBackward,
            simple_bwd_ft_act_screlu_fused_cpu,
        ] {
            let (pre, narrow) = backward_inputs(2 * FT);
            let mut wide = vec![0.0_f32; BATCH * WIDE];
            for bi in 0..BATCH {
                wide[bi * WIDE..bi * WIDE + 2 * FT]
                    .copy_from_slice(&narrow[bi * 2 * FT..(bi + 1) * 2 * FT]);
                for v in &mut wide[bi * WIDE + 2 * FT..(bi + 1) * WIDE] {
                    *v = f32::NAN;
                }
            }
            for offset in [0, FT] {
                let mut from_narrow = vec![0.0_f32; BATCH * FT];
                fused(&narrow, &pre, &mut from_narrow, BATCH, FT, 2 * FT, offset);
                let mut from_wide = vec![0.0_f32; BATCH * FT];
                fused(&wide, &pre, &mut from_wide, BATCH, FT, WIDE, offset);
                assert!(same(&from_narrow, &from_wide));
                assert!(from_wide.iter().all(|v| !v.is_nan()));
            }
        }
    }
}
