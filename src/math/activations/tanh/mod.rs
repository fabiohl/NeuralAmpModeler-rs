// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized Tanh (Hyperbolic Tangent) activation kernels.
//!
//! - **Production path (default):** Padé \[5,4\] rational approximant with
//!   hardware division on the AVX2 backend (`simd_tanh_avx2`,
//!   `simd_tanh_dual_avx2`).
//! - **Research opt-in path (`avx512` feature):** AVX-512 equivalents
//!   (`simd_tanh_avx512`) compiled only under `--features avx512`; not a
//!   production/default backend.
//! - **Reference path:** Padé NR2 variants retained for benchmarking and
//!   documentation (`simd_tanh_pade_nr2_avx2`, `simd_tanh_pade_nr2_avx512`).
//! - **High-fidelity path:** Polynomial exp-based tanh kernels (degree-6 Taylor,
//!   ≤ 2.4e-7 error). Sigmoid polynomial kernels live in
//!   `crate::math::activations::sigmoid::high_fidelity`.

/// Polynomial exp-based tanh kernels (degree-6 Taylor, ≤ 2.4e-7 error).
/// Also exports the `simd_tanh_sigmoid_dual_poly_*` gated fusion.
pub mod high_fidelity;
/// AVX-512 polynomial exp/tanh kernels (re-exported by `high_fidelity`).
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub mod high_fidelity_avx512;
pub mod production;

/// Experimental / reference Padé NR2 variants retained for benchmarking.
pub mod reference;

pub use high_fidelity::*;
pub use production::*;
pub use reference::*;
