// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! DSP gain operations, clipping detection, and stereo ramp.
//!
//! Dynamically dispatches to the configured SIMD backend.

#[doc = "Internal: the `gain_kernel!` macro used by AVX2 gain kernels."]
mod kernel_macro;

mod avx2;

pub use avx2::*;

/// Applies constant gain to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain(data: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain(data, gain))
}

/// Applies constant gain in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
    crate::math::common::dispatch_simd!(apply_gain_stereo(left, right, gain))
}

/// Applies gain and detects clipping in mono in a single pass.
/// Returns `true` if any resulting sample has `|x| > 1.0`.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_mono(data, gain))
}

/// Applies gain and detects clipping in stereo in a single pass.
/// Returns `true` if any resulting sample has `|x| > 1.0`.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_gain_and_detect_clipping_stereo(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    crate::math::common::dispatch_simd!(apply_gain_and_detect_clipping_stereo(left, right, gain))
}

/// Applies linear gain ramp in stereo via SIMD dispatch.
///
/// # Safety
/// The buffers must be valid and have the same size.
pub unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp_stereo(left, right, start, step))
}

/// Applies linear gain ramp to a mono buffer via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
    crate::math::common::dispatch_simd!(apply_ramp(data, start, step))
}

/// Applies the linear gain multiplier to the buffer (safe wrapper).
///
/// Fast-returns if `gain_linear` ~= 1.0 (fast-path bypass).
/// Routes to the SIMD backend via monomorphized `dispatch_simd!`.
pub fn apply_gain_simd(buffer: &mut [f32], gain_linear: f32) {
    if (gain_linear - 1.0).abs() < 1e-6 {
        return;
    }
    unsafe { apply_gain(buffer, gain_linear) };
}

/// Applies a linear gain ramp to the buffer (safe wrapper).
///
/// If the increment is negligible, applies constant gain instead.
/// Routes to the SIMD backend via monomorphized `dispatch_simd!`.
pub fn apply_ramp_simd(buffer: &mut [f32], start: f32, step: f32) {
    if step.abs() < 1e-9 {
        apply_gain_simd(buffer, start);
        return;
    }
    unsafe { apply_ramp(buffer, start, step) };
}

/// Fused gain + dither: `data[i] = data[i] * gain + offset` in a single pass via SIMD dispatch.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32) {
    crate::math::common::dispatch_simd!(apply_gain_then_dither(data, gain, offset))
}

/// Adds a broadcast constant (dither offset) to every element of a mono buffer.
///
/// # Safety
/// The buffer must be valid.
pub unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
    crate::math::common::dispatch_simd!(apply_dither_add(data, offset))
}

/// Crossfade blend: `out[i] = fma(pending[i] - out[i], t, out[i])`.
///
/// # Safety
/// Buffers must be valid and have at least `min(out.len(), pending.len())` elements.
pub unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
    crate::math::common::dispatch_simd!(crossfade_blend_mono(out, pending, t))
}

/// Safe wrapper for crossfade blend: `out[i] = fma(pending[i] - out[i], t, out[i])`.
///
/// Handles mismatched buffer lengths by blending only the overlapping prefix.
/// Uses `[t]` as the blend coefficient (0.0 = keep current, 1.0 = replace with pending).
#[inline]
pub fn crossfade_blend_mono_simd(out: &mut [f32], pending: &[f32], t: f32) {
    if out.len() != pending.len() {
        let n = out.len().min(pending.len());
        // SAFETY: n is safe intersection of both buffers.
        unsafe { crossfade_blend_mono(&mut out[..n], &pending[..n], t) };
        return;
    }
    // SAFETY: out and pending have the same length by the check above.
    unsafe { crossfade_blend_mono(out, pending, t) };
}
/// Safe wrapper for dither offset addition via SIMD broadcast + vector add.
///
/// Injects a subnormal-safe DC offset (typically `DENORMAL_DITHER_OFFSET` or its
/// negative) to prevent subnormal floats from propagating through neural network
/// activations during fade-out/silence.
pub fn apply_dither_add_simd(buffer: &mut [f32], offset: f32) {
    // SAFETY: `buffer` is a valid mutable reference provided by the caller;
    // `offset` is a finite f32 constant (DENORMAL_DITHER_OFFSET or its negative).
    unsafe { apply_dither_add(buffer, offset) };
}

#[cfg(test)]
#[path = "../gain_test.rs"]
mod gain_test;
