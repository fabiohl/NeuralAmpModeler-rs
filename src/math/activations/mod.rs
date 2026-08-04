// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(ambiguous_glob_reexports)]

//! Central activations module with automatic SIMD dispatch.
//!
//! Provides ultra-fast and branchless implementations of `tanh` and `sigmoid`.
//! - `tanh(x)` uses piecewise minimax odd polynomials (degree 5) with branchless
//!   blending for |x| < 4, saturated to [-1, 1] (7 symmetric sub-intervals).
//!   The production Padé path lives in `tanh/production.rs`; the high-fidelity
//!   polynomial path in `tanh/high_fidelity.rs`.
//! - `sigmoid(x)` uses a direct degree-17 minimax polynomial (9 odd terms) for |x| < 8,
//!   saturated to [0, 1].  Max error ~4.09e-4 vs `f32::exp` reference.
//!   The production minimax path lives in `sigmoid/production.rs`; the high-fidelity
//!   polynomial path in `sigmoid/high_fidelity.rs`.
//!
//! Reference for tanh: VDT library (CERN), Mineiro & Vorlicek (2016).
//! Sigmoid coefficients: Lawson-weighted minimax on [-8, 8].
//!
//! # Features
//! - **FastMath**: Approximations that prioritize speed while keeping error < 5e-3 (FP16-compatible).
//! - **Branchless**: Zero branches — only SIMD masks (max/min/blend).
//! - **Dispatch**: Automatic kernel selection (AVX2, AVX-512) via CPUID.
//! - **Zero-Alloc**: Operations strictly safe for use in real-time threads.
//! - **AMX/AVX10.2-ready**: Branchless approach compatible with future extensions.

pub mod fast_tanh;
pub mod fused;
pub mod hard_swish;
pub mod hard_tanh;
pub mod kernel_macro;
pub mod leaky_hard_tanh;
pub mod prelu;
pub mod relu;
pub mod sigmoid;
pub mod silu;
pub mod softsign;
pub mod tanh;

#[cfg(test)]
mod activations_test;

pub use fast_tanh::*;
pub use fused::*;
pub use hard_swish::*;
pub use hard_tanh::*;
pub use leaky_hard_tanh::*;
pub use prelu::*;
pub use relu::*;
use serde::{Deserialize, Serialize};
pub use sigmoid::*;
pub use silu::*;
pub use softsign::*;
pub use tanh::*;

/// Activation precision mode for tanh/sigmoid approximation.
///
/// Controls whether the high-fidelity polynomial exp-based kernels or the
/// cheaper production Padé/minimax kernels are used.
///
/// # Naming History
/// Prior to this rename, `Standard` denoted the Padé/minimax approximation
/// (now `Fast`) and `HighFidelity` denoted the exact-grade polynomial path
/// (now `Standard`). `Standard` is the universal production default for
/// every model in nam-rs — `Fast` is an explicit, opt-in trade-off of
/// fidelity for CPU headroom. Numeric discriminants are unchanged from the
/// previous naming, so existing host automation/state (raw `0`/`1`
/// values) keeps selecting the same underlying kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(usize)]
pub enum ActivationPrecision {
    /// Polynomial exp-based tanh/sigmoid (~2.4e-7 / ~2.1e-7).
    /// Universal production default — exact-grade, matches the NAMCore
    /// C++ reference (`using_fast_tanh = false`) to bit-exact precision.
    #[default]
    Standard = 1,
    /// Padé \[5,4\] tanh (~2.32e-3) + minimax degree-17 sigmoid (~4.09e-4).
    /// Cheaper/faster approximation; opt-in via `--activation fast` (CLI)
    /// or the host "Activation Precision" parameter.
    Fast = 0,
}

impl ActivationPrecision {
    /// Creates an `ActivationPrecision` from an f32 host parameter value.
    #[inline]
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            0 => Self::Fast,
            _ => Self::Standard,
        }
    }

    /// Converts to an f32 host parameter value (0.0 or 1.0).
    #[inline]
    pub fn to_f32(self) -> f32 {
        self as u32 as f32
    }
}

thread_local! {
    static ACTIVE_MODEL_PRECISION: std::cell::Cell<Option<ActivationPrecision>> = const { std::cell::Cell::new(None) };
}

/// RAII Guard that resets the thread-local activation precision override to `None` when dropped.
pub struct ActivationPrecisionGuard {
    _private: (),
}

impl Drop for ActivationPrecisionGuard {
    #[inline(always)]
    fn drop(&mut self) {
        ACTIVE_MODEL_PRECISION.with(|p| p.set(None));
    }
}

/// Sets the thread-local activation precision for the current thread.
///
/// Callers MUST ensure TLS is cleared via [`clear_activation_tls`] or by
/// holding the returned [`ActivationPrecisionGuard`] until processing completes.
///
/// This is the **only** API for setting activation precision — the
/// process-wide global has been removed in favor of per-thread TLS.
#[inline]
pub fn set_activation_tls(mode: ActivationPrecision) {
    ACTIVE_MODEL_PRECISION.with(|p| p.set(Some(mode)));
}

/// Clears the thread-local activation precision override, falling back to
/// `Standard` on subsequent [`activation_precision`] calls.
#[inline]
pub fn clear_activation_tls() {
    ACTIVE_MODEL_PRECISION.with(|p| p.set(None));
}

/// Temporarily overrides thread-local activation precision and returns a guard
/// that clears it on drop.
#[inline]
pub fn set_thread_local_activation_precision(
    mode: Option<ActivationPrecision>,
) -> ActivationPrecisionGuard {
    ACTIVE_MODEL_PRECISION.with(|p| p.set(mode));
    ActivationPrecisionGuard { _private: () }
}

/// Returns the current thread-local activation precision override (if any).
#[inline]
pub fn thread_local_activation_precision() -> Option<ActivationPrecision> {
    ACTIVE_MODEL_PRECISION.with(|p| p.get())
}

/// Returns the current activation precision from thread-local storage.
///
/// Falls back to `Standard` (exact-grade polynomial path) when TLS is not set.
/// This is safe for standalone/CLI use and for audio-thread windows between
/// `process()` calls where DSP is inactive.
#[inline]
pub fn activation_precision() -> ActivationPrecision {
    ACTIVE_MODEL_PRECISION
        .with(|p| p.get())
        .unwrap_or(ActivationPrecision::Standard)
}

/// Applies Tanh activation to a slice of f32 with automatic dispatch to the best SIMD implementation.
#[inline(always)]
pub fn tanh_slice(data: &mut [f32]) {
    if activation_precision() == ActivationPrecision::Standard {
        crate::math::common::dispatch_simd!(tanh_slice_hf(data));
    } else {
        crate::math::common::dispatch_simd!(tanh_slice(data));
    }
}

/// Applies Sigmoid activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn sigmoid_slice(data: &mut [f32]) {
    if activation_precision() == ActivationPrecision::Standard {
        crate::math::common::dispatch_simd!(sigmoid_slice_hf(data));
    } else {
        crate::math::common::dispatch_simd!(sigmoid_slice(data));
    }
}

/// Applies ReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn relu_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(relu_slice(data));
}

/// Applies PReLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn prelu_slice(data: &mut [f32], slopes: &[f32]) {
    crate::math::common::dispatch_simd!(prelu_slice(data, slopes));
}

/// Applies Softsign activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn softsign_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(softsign_slice(data));
}

/// Applies SiLU activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn silu_slice(data: &mut [f32]) {
    if activation_precision() == ActivationPrecision::Standard {
        crate::math::common::dispatch_simd!(silu_slice_hf(data));
    } else {
        crate::math::common::dispatch_simd!(silu_slice(data));
    }
}

/// Applies HardTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn hard_tanh_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(hard_tanh_slice(data));
}

/// Applies HardSwish activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn hard_swish_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(hard_swish_slice(data));
}

/// Applies FastTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn fast_tanh_slice(data: &mut [f32]) {
    crate::math::common::dispatch_simd!(fast_tanh_slice(data));
}

/// Applies LeakyHardTanh activation to a slice of f32 with automatic dispatch.
#[inline(always)]
pub fn leaky_hard_tanh_slice(
    data: &mut [f32],
    min_val: f32,
    max_val: f32,
    min_slope: f32,
    max_slope: f32,
) {
    crate::math::common::dispatch_simd!(leaky_hard_tanh_slice(
        data, min_val, max_val, min_slope, max_slope
    ));
}
