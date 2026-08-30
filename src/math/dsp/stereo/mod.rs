// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

//! DSP operations for stereo processing and signal metering.

mod conv_kernels;
mod convolution_avx2;
#[cfg(feature = "avx512")]
mod convolution_avx512;
mod energy;
mod max_diff;
mod peak;

pub use convolution_avx2::{
    convolve_mono_avx2, convolve_mono_dual_avx2, convolve_stereo_avx2, convolve_stereo_dual_avx2,
};
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use convolution_avx512::{
    convolve_mono_avx512, convolve_mono_dual_avx512, convolve_stereo_avx512,
    convolve_stereo_dual_avx512,
};
pub use energy::{compute_energy_avx2, compute_energy_stereo_avx2};
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use energy::{compute_energy_avx512, compute_energy_stereo_avx512};
pub use max_diff::compute_max_diff_avx2;
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use max_diff::compute_max_diff_avx512;
pub use peak::{compute_peak_abs_mono_avx2, compute_peak_abs_stereo_avx2};
#[cfg(feature = "avx512")]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub use peak::{compute_peak_abs_mono_avx512, compute_peak_abs_stereo_avx512};

/// Computes the maximum energy between two audio channels via SIMD dispatch.
///
/// # Safety
/// The slices must have the same length. The SIMD backends use unaligned loads,
/// so no alignment contract is imposed on the caller.
pub unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_energy_stereo(l, r))
}

/// Computes the maximum absolute difference between two blocks via SIMD dispatch.
///
/// # Safety
/// The slices must have the same length. The SIMD backends use unaligned loads,
/// so no alignment contract is imposed on the caller.
pub unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_max_diff(a, b))
}

/// Computes the peak absolute value of both stereo channels via SIMD dispatch.
///
/// # Safety
/// The slices must have the same length. The SIMD backends use unaligned loads,
/// so no alignment contract is imposed on the caller.
pub unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
    crate::math::common::dispatch_simd!(compute_peak_abs_stereo(left, right))
}

/// Computes the peak absolute value of a single channel via SIMD dispatch.
///
/// # Safety
/// The SIMD backends use unaligned loads; no alignment contract is imposed
/// on the caller.
pub unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
    crate::math::common::dispatch_simd!(compute_peak_abs_mono(data))
}

/// Stereo convolution (used in the resampler) via SIMD dispatch.
/// Computes the dot product between a filter bank and two input buffers (L/R).
///
/// # Safety
/// `coeffs`, `input_l`, and `input_r` must be valid pointers to at least `taps` elements.
/// `coeffs` must be aligned according to the SIMD register.
pub unsafe fn convolve_stereo(
    coeffs: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> (f32, f32) {
    crate::math::common::dispatch_simd!(convolve_stereo(coeffs, input_l, input_r, taps))
}

/// Mono convolution (used in the resampler) via SIMD dispatch.
/// Computes the dot product between a filter bank and an input buffer.
///
/// # Safety
/// `coeffs` and `input` must be valid pointers to at least `taps` elements.
/// `coeffs` must be aligned according to the SIMD register.
pub unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    crate::math::common::dispatch_simd!(convolve_mono(coeffs, input, taps))
}

/// Dual mono convolution (used in the resampler) via SIMD dispatch.
/// Computes two mono convolutions on the same input buffer, reusing the loaded input samples.
///
/// # Safety
/// `coeffs0`, `coeffs1`, and `input` must be valid pointers to at least `taps` elements.
/// `coeffs0` and `coeffs1` must be aligned according to the SIMD register.
pub unsafe fn convolve_mono_dual(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input: *const f32,
    taps: usize,
) -> (f32, f32) {
    crate::math::common::dispatch_simd!(convolve_mono_dual(coeffs0, coeffs1, input, taps))
}
