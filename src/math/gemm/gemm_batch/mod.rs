// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Batch GEMM and Fused Residual GEMM kernels — AVX2 and AVX-512.
//!
//! Process multiple audio frames simultaneously for efficient weight reuse.

#[cfg(feature = "avx512")]
mod avx512;
mod fused_add_gemm_batch;
mod fused_residual_batch;
mod kernel_macro;

#[cfg(feature = "avx512")]
pub use avx512::fused_add_gemm_batch_avx512;
pub use fused_add_gemm_batch::fused_add_gemm_batch_avx2;
pub use fused_residual_batch::*;
