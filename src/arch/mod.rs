// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Architecture-neutral entry point for SIMD intrinsics.
//!
//! Kernels import their intrinsics from [`simd`] instead of naming
//! `core::arch::x86_64` directly. On x86_64 this is a plain re-export of the
//! native AVX2/FMA/F16C intrinsics. On aarch64 it resolves to a NEON
//! implementation of the same `__m256`/`__m128` surface, so the AVX2-shaped
//! kernels compile unchanged and keep identical per-lane operation order.
//!
//! AVX-512 kernels (behind the `avx512` feature) are x86_64-only: the feature
//! is rejected on other targets, and their `#[target_feature]` attributes stay
//! unconditional. Files that hold only AVX-512 code import `core::arch::x86_64`
//! directly; files that mix AVX2 and AVX-512 code import through [`simd`],
//! which on x86_64 is the same re-export.

#[cfg(target_arch = "aarch64")]
mod aarch64;

/// SIMD intrinsics for the current target architecture.
#[cfg(target_arch = "x86_64")]
pub mod simd {
    pub use core::arch::x86_64::*;
}

/// SIMD intrinsics for the current target architecture.
#[cfg(target_arch = "aarch64")]
pub mod simd {
    pub use super::aarch64::*;
}
