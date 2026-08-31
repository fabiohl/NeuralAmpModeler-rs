// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 8x Dot Product Kernels (8 simultaneous output channels) — AVX2/FMA.
//!
//! Processes 8 output channels per invocation, reducing outer loop iteration
//! count and maximizing register reuse.

pub mod dot_f32_avx2;
pub mod scalar;

pub use dot_f32_avx2::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_8x_test.rs"]
mod dot_8x_test;
