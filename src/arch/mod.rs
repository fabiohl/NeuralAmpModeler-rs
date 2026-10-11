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
//! AVX-512 kernels (behind the `avx512` feature) remain x86_64-only and keep
//! importing `core::arch::x86_64` directly.

/// SIMD intrinsics for the current target architecture.
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd {
    pub(crate) use core::arch::x86_64::*;
}
