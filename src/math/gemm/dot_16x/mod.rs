// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 16x Dot Product Kernels (16 simultaneous output channels) — AVX2 and AVX-512.
//!
//! Processes 16 output channels per invocation. On the x86-64-v3 baseline (AVX2/FMA),
//! two `__m256` loads (lo/hi) cover the 16 weights per row. In AVX-512F,
//! a single `__m512` register covers all 16 weights.

pub mod dot_f32_avx2;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub mod dot_f32_avx512;
pub mod scalar;

pub use dot_f32_avx2::*;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use dot_f32_avx512::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_16x_test.rs"]
mod dot_16x_test;
