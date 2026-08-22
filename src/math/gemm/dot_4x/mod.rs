// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Kernels de Dot Product 4x (ILP interleaved, dual frame, batch) — AVX2 e AVX-512.
//!
//! Organized into submodules by ISA and variant:
//! - `avx2` / `avx2_dual`: AVX2 kernels with F16C.
//! - `avx512` / `avx512_dual`: AVX-512 kernels with permutexvar.
//! - `scalar`: reference scalar implementations.

pub mod avx2;
pub mod avx2_dual;
#[cfg(feature = "avx512")]
pub mod avx512;
#[cfg(feature = "avx512")]
pub mod avx512_dual;
pub mod dot_f32_avx2;
#[cfg(feature = "avx512")]
pub mod dot_f32_avx512;
#[cfg(feature = "avx512")]
pub mod dot_f32_avx512vl;
mod kernel_macro;
pub mod scalar;

pub use avx2::*;
pub use avx2_dual::*;
#[cfg(feature = "avx512")]
pub use avx512::*;
#[cfg(feature = "avx512")]
pub use avx512_dual::*;
pub use dot_f32_avx2::*;
#[cfg(feature = "avx512")]
pub use dot_f32_avx512::*;
#[cfg(feature = "avx512")]
pub use dot_f32_avx512vl::*;
pub use scalar::*;

#[cfg(test)]
#[path = "dot_4x_test.rs"]
mod dot_4x_test;
