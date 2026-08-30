// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! GEMV kernels (Matrix-Vector Multiplication) — AVX2 and AVX-512.
//!
//! Includes `_small` variants specialized for Standard WaveNet (CH=16),
//! batch versions, and the fused operation `fused_add_gemv`.
//!
//! # Parallelism Strategy
//! - AVX2: 8 YMM accumulators (8×8 = 64 lanes), inner loop with step 8.
//! - AVX-512: 8 ZMM accumulators (8×16 = 128 lanes), inner loop with step 8.
//! - FMA dependency chain breaking via multiple accumulators.
//! - Software prefetch on in_frame to reduce cache miss latency.

#[doc = "GEMV kernels using AVX2 (fused_add_gemv_avx2, gemv_overwrite_avx2)."]
pub mod f16_avx2;
#[doc = "fused_add_gemv AVX2 specialized kernels (1×4, 4×4, 4×6, 8×4, 8×6, 8×8)."]
pub mod f16_avx2_fused;
#[doc = "gemv_overwrite AVX2 specialized kernels (1×4, 4×4, 4×6, 8×4, 8×6, 8×8)."]
pub mod f16_avx2_overwrite;
#[doc = "GEMV AVX2 specialized kernels for fixed dimensions (1×4, 4×4, 4×6, 8×4, 8×6, 8×8)."]
pub mod f16_avx2_specialized;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[doc = "GEMV kernels using AVX-512 (small, general, batch, fused variants)."]
pub mod f16_avx512;
#[doc = "f32 batched GEMV kernels using AVX2 (with_bias, no_bias)."]
pub mod f32_avx2;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[doc = "f32 batched GEMV kernels using AVX-512 (with_bias, no_bias)."]
pub mod f32_avx512;
#[doc = "Internal: the `gemv_f32_inner_loop` macro for batched f32 GEMV."]
mod f32_kernel_macro;
#[doc = "Internal: the `gemv_kernel!` macro used by f16 AVX2 and AVX-512 kernels."]
pub mod kernel_macro;

pub use f16_avx2::*;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use f16_avx512::*;
pub use f32_avx2::*;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use f32_avx512::*;

#[cfg(test)]
mod gemv_test;
