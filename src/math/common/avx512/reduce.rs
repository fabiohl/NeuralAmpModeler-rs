// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 dispatch macro for reduction kernels
//! (`horizontal_sum`, `compute_energy_stereo`).
//!
//! Delegates to 512-bit reduction intrinsics in [`crate::math::common::utility`]
//! and [`crate::math::dsp::stereo`]. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx512_reduce {
    () => {
        #[inline(always)]
        // SAFETY: ptr is a valid raw pointer to N valid f32 elements;
        // CPU supports AVX-512F (verified by dispatch). Kernel uses unaligned 512-bit loads.
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            // SAFETY: ptr and N satisfy function invariants.
            unsafe { crate::math::common::utility::horizontal_sum_avx512(ptr, N) }
        }

        #[inline(always)]
        // SAFETY: l and r are valid f32 slices of equal length;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: the kernel is safe, but calling it requires AVX-512F,
            // which this dispatch impl does not enable in its own codegen: the
            // caller must guarantee the CPU supports it (verified by dispatch).
            unsafe { crate::math::dsp::stereo::compute_energy_stereo_avx512(l, r) }
        }

        #[inline(always)]
        // SAFETY: data is a valid f32 slice; CPU supports AVX-512F (verified by dispatch).
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: the kernel is safe, but calling it requires AVX-512F,
            // which this dispatch impl does not enable in its own codegen: the
            // caller must guarantee the CPU supports it (verified by dispatch).
            unsafe { crate::math::dsp::stereo::compute_energy_avx512(data) }
        }

        #[inline(always)]
        // SAFETY: a and b are valid f32 slices of equal length;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: a and b satisfy function invariants.
            unsafe { crate::math::dsp::stereo::compute_max_diff_avx512(a, b) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid f32 slices of equal length;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: the kernel is safe, but calling it requires AVX-512F,
            // which this dispatch impl does not enable in its own codegen: the
            // caller must guarantee the CPU supports it (verified by dispatch).
            unsafe { crate::math::dsp::stereo::compute_peak_abs_stereo_avx512(left, right) }
        }

        #[inline(always)]
        // SAFETY: data is a valid f32 slice; CPU supports AVX-512F (verified by dispatch).
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: the kernel is safe, but calling it requires AVX-512F,
            // which this dispatch impl does not enable in its own codegen: the
            // caller must guarantee the CPU supports it (verified by dispatch).
            unsafe { crate::math::dsp::stereo::compute_peak_abs_mono_avx512(data) }
        }
    };
}

macro_rules! impl_avx512vnni_bf16_reduce {
    () => {
        const IS_BF16: bool = true;

        #[inline(always)]
        // SAFETY: ptr is a valid raw pointer to N valid f32 elements;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn horizontal_sum<const N: usize>(ptr: *const f32) -> f32 {
            crate::math::common::utility::horizontal_sum_avx512(ptr, N)
        }

        #[inline(always)]
        // SAFETY: l and r are valid f32 slices of equal length;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn compute_energy_stereo(l: &[f32], r: &[f32]) -> f32 {
            // SAFETY: l and r satisfy function invariants.
            unsafe { Avx512Math::compute_energy_stereo(l, r) }
        }

        #[inline(always)]
        // SAFETY: data is a valid f32 slice; CPU supports AVX-512 VNNI+BF16.
        unsafe fn compute_energy(data: &[f32]) -> f32 {
            // SAFETY: data satisfies function invariants.
            unsafe { Avx512Math::compute_energy(data) }
        }

        #[inline(always)]
        // SAFETY: a and b are valid f32 slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn compute_max_diff(a: &[f32], b: &[f32]) -> f32 {
            // SAFETY: a and b satisfy function invariants.
            unsafe { Avx512Math::compute_max_diff(a, b) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid f32 slices of equal length;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn compute_peak_abs_stereo(left: &[f32], right: &[f32]) -> (f32, f32) {
            // SAFETY: left and right satisfy function invariants.
            unsafe { Avx512Math::compute_peak_abs_stereo(left, right) }
        }

        #[inline(always)]
        // SAFETY: data is a valid f32 slice; CPU supports AVX-512 VNNI+BF16.
        unsafe fn compute_peak_abs_mono(data: &[f32]) -> f32 {
            // SAFETY: data satisfies function invariants.
            unsafe { Avx512Math::compute_peak_abs_mono(data) }
        }
    };
}

pub(super) use impl_avx512_reduce;
pub(super) use impl_avx512vnni_bf16_reduce;
