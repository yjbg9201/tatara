//! Simple アーキ (bucket 無し 4 層) 専用 kernel の reference CPU 実装。
//!
//! GPU 側 `#[kernel]` は `bins/nnue_train/src/kernels/simple.rs` (cuda-oxide) と
//! `crates/cuda-native-runtime/kernels/native_kernels.cu` (CUDA C++) にある。
//!
//! ## `combined` の配置
//!
//! L1 の入力 `combined` は `batch × combined_stride` の row-major で、1 行は
//! `[FT_stm (ft_out) | FT_nstm (ft_out) | 密入力 (入玉スカラー 96、使う構成のみ)]`。
//! `combined_stride` は `SimpleId::combined_dim()` (密入力なしでは `2 * ft_out`)。
//!
//! ## kernel ↔ CPU reference 対応表
//!
//! | `#[kernel]` | CPU reference |
//! |---|---|
//! | `simple_ft_post_fused_crelu`      | [`ft_post::simple_ft_post_fused_crelu_cpu`] |
//! | `simple_ft_post_fused_screlu`     | [`ft_post::simple_ft_post_fused_screlu_cpu`] |
//! | `simple_bwd_ft_act_crelu_fused`   | [`ft_post::simple_bwd_ft_act_crelu_fused_cpu`] |
//! | `simple_bwd_ft_act_screlu_fused`  | [`ft_post::simple_bwd_ft_act_screlu_fused_cpu`] |
//! | `simple_nyugyoku_scalar_to_combined` | [`dense_input::simple_nyugyoku_scalar_to_combined_cpu`] |

pub mod dense_input;
pub mod ft_post;
