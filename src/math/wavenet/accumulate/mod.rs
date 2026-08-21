// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Accumulation and activation kernels for WaveNet — AVX2, AVX-512 and scalar fallback.

#[cfg(test)]
mod accumulate_test;
mod avx2;
mod avx512;
mod avx512vl;
mod kernel_macro;
#[cfg(test)]
pub mod scalar;

pub use avx2::*;
pub use avx512::*;
pub use avx512vl::*;
#[cfg(test)]
pub use scalar::*;
