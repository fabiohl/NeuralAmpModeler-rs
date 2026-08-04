// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 (x86-64-v3) stereo convolution kernels generated via `impl_convolve_mono!`.
//!
//! Provides `convolve_stereo_avx2` and `convolve_stereo_avx2_pair` entry points
//! that decompose stereo convolution into single-channel 256-bit SIMD passes
//! using AVX2+FMA intrinsics.

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

const AVX2_STEP_DBL: usize = 16;
const AVX2_HALF_STEP: usize = 8;
const AVX2_ALIGN: usize = 32;

macro_rules! avx2_zero {
    () => {
        || _mm256_setzero_ps()
    };
}
macro_rules! avx2_loadu {
    () => {
        |p| _mm256_loadu_ps(p)
    };
}
macro_rules! avx2_fmadd {
    () => {
        |a, b, c| _mm256_fmadd_ps(a, b, c)
    };
}
macro_rules! avx2_add {
    () => {
        |a, b| _mm256_add_ps(a, b)
    };
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn avx2_hsum(r: __m256) -> f32 {
    let hi128 = _mm256_extractf128_ps(r, 1);
    let lo128 = _mm256_castps256_ps128(r);
    let s128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = 0.0f32;
    _mm_store_ss(&mut out, r);
    out
}

impl_convolve_stereo!(
    #[doc = "Stereo Interleaved Convolution AVX2.\n\nLoads coefficients once and applies them to both channels."]
    #[target_feature(enable = "avx2,fma")]
    convolve_stereo_avx2,
    AVX2_STEP_DBL,
    AVX2_HALF_STEP,
    AVX2_ALIGN,
    avx2_zero!(),
    avx2_loadu!(),
    avx2_fmadd!(),
    avx2_add!(),
    avx2_hsum
);

impl_convolve_stereo_dual!(
    #[doc = "Stereo Dual Convolution AVX2.\n\nPerforms two stereo convolutions (for two coefficient sets coeffs0 and coeffs1)\nover the same input buffers input_l and input_r.\nLoads input samples once and applies them to both coefficient sets."]
    #[target_feature(enable = "avx2,fma")]
    convolve_stereo_dual_avx2,
    AVX2_STEP_DBL,
    AVX2_HALF_STEP,
    AVX2_ALIGN,
    avx2_zero!(),
    avx2_loadu!(),
    avx2_fmadd!(),
    avx2_add!(),
    avx2_hsum
);

impl_convolve_mono_dual!(
    #[doc = "Mono Dual Convolution AVX2.\n\nPerforms two mono convolutions on the same input buffer, reusing the loaded input samples."]
    #[target_feature(enable = "avx2,fma")]
    convolve_mono_dual_avx2,
    AVX2_STEP_DBL,
    AVX2_HALF_STEP,
    AVX2_ALIGN,
    avx2_zero!(),
    avx2_loadu!(),
    avx2_fmadd!(),
    avx2_add!(),
    avx2_hsum
);

impl_convolve_mono!(
    #[doc = "Mono Convolution AVX2.\n\nLoads coefficients and applies them to a single channel."]
    #[target_feature(enable = "avx2,fma")]
    convolve_mono_avx2,
    AVX2_STEP_DBL,
    AVX2_HALF_STEP,
    AVX2_ALIGN,
    avx2_zero!(),
    avx2_loadu!(),
    avx2_fmadd!(),
    avx2_add!(),
    avx2_hsum
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolve_stereo_avx2_basic() {
        #[repr(align(32))]
        struct Aligned([f32; 32]);
        let coeffs = Aligned([0.5; 32]);
        let input_l = [1.0f32; 32];
        let input_r = [2.0f32; 32];

        let (l, r) = unsafe {
            convolve_stereo_avx2(coeffs.0.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 32)
        };
        assert!((l - 16.0).abs() < 1e-5, "l={l}");
        assert!((r - 32.0).abs() < 1e-5, "r={r}");
    }

    #[test]
    fn test_convolve_stereo_avx2_odd_taps() {
        #[repr(align(32))]
        struct Aligned([f32; 17]);
        let coeffs = Aligned([0.5; 17]);
        let input_l = [1.0f32; 17];
        let input_r = [2.0f32; 17];

        let (l, r) = unsafe {
            convolve_stereo_avx2(coeffs.0.as_ptr(), input_l.as_ptr(), input_r.as_ptr(), 17)
        };
        assert!((l - 8.5).abs() < 1e-5);
        assert!((r - 17.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_stereo_dual_avx2_basic() {
        #[repr(align(32))]
        struct Aligned([f32; 32]);
        let coeffs0 = Aligned([0.25; 32]);
        let coeffs1 = Aligned([0.5; 32]);
        let input_l = [1.0f32; 32];
        let input_r = [2.0f32; 32];

        let ((l0, r0), (l1, r1)) = unsafe {
            convolve_stereo_dual_avx2(
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
    fn test_convolve_mono_avx2_basic() {
        #[repr(align(32))]
        struct Aligned([f32; 32]);
        let coeffs = Aligned([0.5; 32]);
        let input = [1.0f32; 32];

        let result = unsafe { convolve_mono_avx2(coeffs.0.as_ptr(), input.as_ptr(), 32) };
        assert!((result - 16.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve_mono_dual_avx2_basic() {
        #[repr(align(32))]
        struct Aligned([f32; 32]);
        let coeffs0 = Aligned([0.25; 32]);
        let coeffs1 = Aligned([0.5; 32]);
        let input = [1.0f32; 32];

        let (out0, out1) = unsafe {
            convolve_mono_dual_avx2(coeffs0.0.as_ptr(), coeffs1.0.as_ptr(), input.as_ptr(), 32)
        };
        assert!((out0 - 8.0).abs() < 1e-5);
        assert!((out1 - 16.0).abs() < 1e-5);
    }
}
