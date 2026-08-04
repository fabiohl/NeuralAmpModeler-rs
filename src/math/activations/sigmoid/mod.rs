// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Sigmoid (Logistic) activation kernels.
//!
//! - **Production path:** Direct degree-17 minimax polynomial (9 odd terms)
//!   optimised for the interval [-8, 8], computed via Lawson's weighted minimax
//!   algorithm (`simd_sigmoid_avx2`, `simd_sigmoid_dual_avx2`, `simd_sigmoid_avx512`).
//! - **High-fidelity path:** Polynomial exp-based sigmoid (degree-6 Taylor with
//!   integer range reduction), max absolute error ≤ 2.1e-7
//!   (`simd_sigmoid_poly_avx2`, `simd_sigmoid_poly_avx512`).

/// Polynomial exp-based sigmoid kernels (degree-6 Taylor, ≤ 2.1e-7 error).
pub mod high_fidelity;
/// AVX-512 polynomial exp/sigmoid kernels (re-exported by `high_fidelity`).
pub mod high_fidelity_avx512;
pub mod production;

pub use high_fidelity::*;
pub use production::*;
