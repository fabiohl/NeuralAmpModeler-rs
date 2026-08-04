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

pub mod dot_16x;
pub mod dot_4x;
pub mod dot_8x;
pub mod dot_basic;
pub mod gemm_batch;
pub mod gemv;
pub mod gemv_4gate;
pub mod gemv_bf16;

pub use dot_4x::avx2::*;
pub use dot_4x::avx2_dual::*;
pub use dot_4x::avx512::*;
pub use dot_4x::avx512_dual::*;
pub use dot_4x::dot_f32_avx2::*;
pub use dot_4x::dot_f32_avx512::*;
pub use dot_4x::scalar::*;
pub use dot_8x::dot_f32_avx2::*;
pub use dot_8x::scalar::dot_product_8x_f32_scalar;
pub use dot_16x::dot_f32_avx512::*;
pub use dot_16x::scalar::dot_product_16x_f32_scalar; // Test/bench oracle only — not for production dispatch
pub use dot_basic::*;
pub use gemm_batch::*;
pub use gemv::*;
pub use gemv_4gate::*;
