// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Stereo Convolution (used in the Resampler).
/// Convolution is like applying a filter (like an equalizer).
/// Here we do it for the Left (L) and Right (R) channels simultaneously.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn convolve_stereo_fallback(
    coeffs: *const f32,  // Filter coefficients.
    input_l: *const f32, // Left channel input.
    input_r: *const f32, // Right channel input.
    taps: usize,         // Filter "length".
) -> (f32, f32) {
    let mut sum_l = 0.0f32;
    let mut sum_r = 0.0f32;
    for i in 0..taps {
        let h = *coeffs.add(i);
        sum_l += h * *input_l.add(i);
        sum_r += h * *input_r.add(i);
    }
    (sum_l, sum_r)
}

/// Dual Stereo Convolution (used in the Resampler).
/// Performs two consecutive stereo convolutions reusing input loads.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn convolve_stereo_dual_fallback(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input_l: *const f32,
    input_r: *const f32,
    taps: usize,
) -> ((f32, f32), (f32, f32)) {
    let mut sum0_l = 0.0f32;
    let mut sum0_r = 0.0f32;
    let mut sum1_l = 0.0f32;
    let mut sum1_r = 0.0f32;
    for i in 0..taps {
        let h0 = *coeffs0.add(i);
        let h1 = *coeffs1.add(i);
        let xl = *input_l.add(i);
        let xr = *input_r.add(i);
        sum0_l += h0 * xl;
        sum0_r += h0 * xr;
        sum1_l += h1 * xl;
        sum1_r += h1 * xr;
    }
    ((sum0_l, sum0_r), (sum1_l, sum1_r))
}

/// Mono Dual Convolution (used in the Resampler).
/// Performs two mono convolutions on the same input buffer, reusing the loaded input samples.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn convolve_mono_dual_fallback(
    coeffs0: *const f32,
    coeffs1: *const f32,
    input: *const f32,
    taps: usize,
) -> (f32, f32) {
    let mut sum0 = 0.0f32;
    let mut sum1 = 0.0f32;
    for i in 0..taps {
        let x = *input.add(i);
        sum0 += *coeffs0.add(i) * x;
        sum1 += *coeffs1.add(i) * x;
    }
    (sum0, sum1)
}

/// Mono Convolution (used in the Resampler).
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn convolve_mono_fallback(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..taps {
        sum += *coeffs.add(i) * *input.add(i);
    }
    sum
}
