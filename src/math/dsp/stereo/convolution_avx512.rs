// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 stereo convolution kernels generated via `impl_convolve_mono!`.
//!
//! Provides `convolve_stereo_avx512` and `convolve_stereo_avx512_pair` entry
//! points using 512-bit SIMD intrinsics for double the throughput per iteration
//! compared to the AVX2 path.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::impl_convolve_mono;
use crate::impl_convolve_mono_dual;
use crate::impl_convolve_stereo;
use crate::impl_convolve_stereo_dual;
use core::arch::x86_64::*;

const AVX512_STEP_DBL: usize = 32;
const AVX512_HALF_STEP: usize = 16;
const AVX512_ALIGN: usize = 64;

macro_rules! avx512_zero {
    () => {
        || _mm512_setzero_ps()
    };
}
macro_rules! avx512_loadu {
    () => {
        |p| _mm512_loadu_ps(p)
    };
}
macro_rules! avx512_fmadd {
    () => {
        |a, b, c| _mm512_fmadd_ps(a, b, c)
    };
}
macro_rules! avx512_add {
    () => {
        |a, b| _mm512_add_ps(a, b)
    };
}

impl_convolve_stereo!(
    #[doc = "Stereo Convolution AVX-512.\n\nApplies a filter (coefficients) to two audio channels at the same time.\nIt's like passing sound through an equalizer or simulating a room (reverb)."]
    #[target_feature(enable = "avx512f")]
    convolve_stereo_avx512,
    AVX512_STEP_DBL,
    AVX512_HALF_STEP,
    AVX512_ALIGN,
    avx512_zero!(),
    avx512_loadu!(),
    avx512_fmadd!(),
    avx512_add!(),
    _mm512_reduce_add_ps
);

impl_convolve_stereo_dual!(
    #[doc = "Stereo Dual Convolution AVX-512.\n\nPerforms two stereo convolutions (for two coefficient sets coeffs0 and coeffs1)\nover the same input buffers input_l and input_r.\nLoads input samples once and applies them to both coefficient sets."]
    #[target_feature(enable = "avx512f")]
    convolve_stereo_dual_avx512,
    AVX512_STEP_DBL,
    AVX512_HALF_STEP,
    AVX512_ALIGN,
    avx512_zero!(),
    avx512_loadu!(),
    avx512_fmadd!(),
    avx512_add!(),
    _mm512_reduce_add_ps
);

impl_convolve_mono_dual!(
    #[doc = "Mono Dual Convolution AVX-512.\n\nPerforms two mono convolutions on the same input buffer, reusing the loaded input samples."]
    #[target_feature(enable = "avx512f")]
    convolve_mono_dual_avx512,
    AVX512_STEP_DBL,
    AVX512_HALF_STEP,
    AVX512_ALIGN,
    avx512_zero!(),
    avx512_loadu!(),
    avx512_fmadd!(),
    avx512_add!(),
    _mm512_reduce_add_ps
);

impl_convolve_mono!(
    #[doc = "Mono Convolution AVX-512.\n\nLoads coefficients and applies them to a single channel."]
    #[target_feature(enable = "avx512f")]
    convolve_mono_avx512,
    AVX512_STEP_DBL,
    AVX512_HALF_STEP,
    AVX512_ALIGN,
    avx512_zero!(),
    avx512_loadu!(),
    avx512_fmadd!(),
    avx512_add!(),
    _mm512_reduce_add_ps
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_stereo_avx512_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 32]);
        let coeffs = Aligned([0.5; 32]);
        let input_l = [1.0f32; 32];
        let input_r = [2.0f32; 32];

        // SAFETY: `coeffs.0`, `input_l`, and `input_r` are 32-element f32 buffers and
        // `taps=32`; the guarded 16-lane AVX-512 loop (`i + 16 <= taps`) and scalar tail
        // keep every `_mm512_loadu_ps` load in bounds; `loadu` needs no 64-byte alignment.
        let (l, r) = unsafe {
            convolve_stereo_avx512(coeffs.0.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 32)
        };
        assert!((l - 16.0).abs() < 1e-5);
        assert!((r - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_stereo_avx512_odd_taps() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 17]);
        let coeffs = Aligned([0.5; 17]);
        let input_l = [1.0f32; 17];
        let input_r = [2.0f32; 17];

        // SAFETY: `coeffs.0`, `input_l`, and `input_r` are 17-element f32 buffers and
        // `taps=17`; the guarded 16-lane AVX-512 loop plus scalar tail only touch indices
        // in `[0, 17)`, and `_mm512_loadu_ps` needs no alignment.
        let (l, r) = unsafe {
            convolve_stereo_avx512(coeffs.0.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 17)
        };
        assert!((l - 8.5).abs() < 1e-5);
        assert!((r - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_stereo_dual_avx512_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 32]);
        let coeffs0 = Aligned([0.25; 32]);
        let coeffs1 = Aligned([0.5; 32]);
        let input_l = [1.0f32; 32];
        let input_r = [2.0f32; 32];

        // SAFETY: `coeffs0.0`, `coeffs1.0`, `input_l`, and `input_r` are 32-element f32
        // buffers with `taps=32`; the guarded 16-lane AVX-512 loops and scalar tail keep
        // every `_mm512_loadu_ps` load in bounds, and `loadu` needs no alignment.
        let ((l0, r0), (l1, r1)) = unsafe {
            convolve_stereo_dual_avx512(
                coeffs0.0.as_ptr(),
                coeffs1.0.as_ptr(),
                input_l.as_ptr(),
                input_r.as_ptr(),
                32,
            )
        };
        assert!((l0 - 8.0).abs() < 1e-5);
        assert!((r0 - 16.0).abs() < 1e-5);
        assert!((l1 - 16.0).abs() < 1e-5);
        assert!((r1 - 32.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_mono_avx512_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 32]);
        let coeffs = Aligned([0.5; 32]);
        let input = [1.0f32; 32];

        // SAFETY: `coeffs.0` and `input` are 32-element f32 buffers with `taps=32`; the
        // guarded 16-lane AVX-512 loop and scalar tail only touch indices in `[0, 32)`, and
        // `_mm512_loadu_ps` needs no alignment.
        let result = unsafe { convolve_mono_avx512(coeffs.0.as_ptr(), input.as_ptr(), 32) };
        assert!((result - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_mono_dual_avx512_basic() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        #[repr(align(64))]
        struct Aligned([f32; 32]);
        let coeffs0 = Aligned([0.25; 32]);
        let coeffs1 = Aligned([0.5; 32]);
        let input = [1.0f32; 32];

        // SAFETY: `coeffs0.0`, `coeffs1.0`, and `input` are 32-element f32 buffers with
        // `taps=32`; the guarded 16-lane AVX-512 loops and scalar tail keep every
        // `_mm512_loadu_ps` load in bounds, and `loadu` needs no alignment.
        let (out0, out1) = unsafe {
            convolve_mono_dual_avx512(coeffs0.0.as_ptr(), coeffs1.0.as_ptr(), input.as_ptr(), 32)
        };
        assert!((out0 - 8.0).abs() < 1e-5);
        assert!((out1 - 16.0).abs() < 1e-5);
    }
}
