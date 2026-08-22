// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD linear algebra kernels (GEMM, GEMV, Dot Product).
//!
//! This module is the high-throughput engine of NAM-rs, responsible for the
//! massive multiplication of weights by neural network states.
//!
//! # Performance Strategies
//! - **ILP (Instruction Level Parallelism)**: Multiple accumulators to saturate the FMA ports.
//! - **Interleaved Layout**: Weights organized to maximize data reuse in registers.
//! - **Tiling**: Block processing to optimize data locality.
//!
//! Extracted from `simd/avx2.rs` and `simd/avx512.rs`.
//! Contains AVX2 and AVX-512 implementations side by side, organized by operation.

/// 16-wide dot-product kernels: AVX-512 f32 and scalar reference oracle.
pub mod dot_16x;
/// 4-wide dot-product kernels: AVX2/AVX-512 f32 with ILP accumulation.
pub mod dot_4x;
/// 8-wide dot-product kernels: AVX2 f32 and scalar reference oracle.
pub mod dot_8x;
/// Scalar dot-product implementations used as test/bench oracles.
pub mod dot_basic;
/// Batched GEMM kernels: fused residual-add GEMV batch with AVX2/AVX-512.
pub mod gemm_batch;
/// General matrix-vector multiplication: f16/f32, overwrite and fused variants.
pub mod gemv;
/// 4-gate GEMV: specialized kernels for LSTM gate-block weight multiplication.
pub mod gemv_4gate;
/// BF16 matrix-vector multiplication with VNNI acceleration.
pub mod gemv_bf16;

pub use dot_4x::avx2::*;
pub use dot_4x::avx2_dual::*;
#[cfg(feature = "avx512")]
pub use dot_4x::avx512::*;
#[cfg(feature = "avx512")]
pub use dot_4x::avx512_dual::*;
pub use dot_4x::dot_f32_avx2::*;
#[cfg(feature = "avx512")]
pub use dot_4x::dot_f32_avx512::*;
pub use dot_4x::scalar::*;
pub use dot_8x::dot_f32_avx2::*;
pub use dot_8x::scalar::dot_product_8x_f32_scalar;
#[cfg(feature = "avx512")]
pub use dot_16x::dot_f32_avx512::*;
/// Test/bench oracle only — not for production dispatch.
pub use dot_16x::scalar::dot_product_16x_f32_scalar;
pub use dot_basic::*;
pub use gemm_batch::*;
pub use gemv::*;
pub use gemv_4gate::*;
