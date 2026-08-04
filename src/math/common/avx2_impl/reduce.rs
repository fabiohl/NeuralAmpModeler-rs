// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 (x86-64-v3) dispatch macro for reduction kernels
//! (`horizontal_sum`, `compute_energy_stereo`, `compute_energy_mono`).
//!
//! Delegates to AVX2 horizontal-sum and vector-lane energy accumulation
//! routines in [`crate::math::common::utility`] and
//! [`crate::math::dsp::stereo`]. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx2_reduce {
    () => {
        #[inline(always)]
        // SAFETY: ptr is a valid raw pointer to N valid f32 elements;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            // SAFETY: ptr and N satisfy the function's documented invariants.
            unsafe { super::utility::horizontal_sum_avx2(ptr, N) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::compute_energy_stereo_avx2(l, r) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::compute_energy_avx2(data) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::compute_max_diff_avx2(a, b) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::compute_peak_abs_stereo_avx2(left, right) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::compute_peak_abs_mono_avx2(data) }
        }
    };
}
pub(crate) use impl_avx2_reduce;
