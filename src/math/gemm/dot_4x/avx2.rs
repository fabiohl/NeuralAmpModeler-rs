// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Dot Product 4x kernels — AVX2 (basic + interleaved).

use crate::dot4x_simd8_avx2;
use crate::dot4x_simd8_avx2_tail2;
use crate::dot4x_simd16_avx2;
use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

/// Computes 4 Dot Products simultaneously with maximum parallelism (ILP) via AVX2.
///
/// This function is optimized for situations where we need to multiply a single "state"
/// by 4 different sets of "weights". By doing all 4 computations at once, we keep
/// the processor busy and take advantage of the fact that the "state" is already loaded
/// to save memory time.
///
/// # Safety
/// Each of `w0`, `w1`, `w2`, and `w3` must have a length greater than or equal to `state.len()`.
/// All slices must be valid and accessible for reading.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_avx2(
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    state: &[f32],
) -> [f32; 4] {
    let len = state
        .len()
        .min(w0.len())
        .min(w1.len())
        .min(w2.len())
        .min(w3.len());
    debug_assert!(w0.len() >= len && w1.len() >= len && w2.len() >= len && w3.len() >= len);
    let mut i = 0;

    // SAFETY: `len` is the minimum of `state.len()` and `w0..w3.len()`
    // (enforced by the `debug_assert!` above); the macro loops guarantee
    // `i + 16`/`i + 8`/`i < len`, so the 8-lane loads and 16-u16 loads at
    // offset `i` stay within all five slices.
    unsafe {
        let mut sum0_0 = _mm256_setzero_ps();
        let mut sum0_1 = _mm256_setzero_ps();
        let mut sum1_0 = _mm256_setzero_ps();
        let mut sum1_1 = _mm256_setzero_ps();
        let mut sum2_0 = _mm256_setzero_ps();
        let mut sum2_1 = _mm256_setzero_ps();
        let mut sum3_0 = _mm256_setzero_ps();
        let mut sum3_1 = _mm256_setzero_ps();

        dot4x_simd16_avx2!(i, len, {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w0.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w1.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w2.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(w3.as_ptr().wrapping_add(i + 32) as *const i8);

            let vs_0 = _mm256_loadu_ps(state.as_ptr().add(i));
            let vs_1 = _mm256_loadu_ps(state.as_ptr().add(i + 8));

            let vw0_0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i) as *const __m128i));
            let vw0_1 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i + 8) as *const __m128i));
            sum0_0 = _mm256_fmadd_ps(vw0_0, vs_0, sum0_0);
            sum0_1 = _mm256_fmadd_ps(vw0_1, vs_1, sum0_1);

            let vw1_0 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i) as *const __m128i));
            let vw1_1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i + 8) as *const __m128i));
            sum1_0 = _mm256_fmadd_ps(vw1_0, vs_0, sum1_0);
            sum1_1 = _mm256_fmadd_ps(vw1_1, vs_1, sum1_1);

            let vw2_0 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i) as *const __m128i));
            let vw2_1 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i + 8) as *const __m128i));
            sum2_0 = _mm256_fmadd_ps(vw2_0, vs_0, sum2_0);
            sum2_1 = _mm256_fmadd_ps(vw2_1, vs_1, sum2_1);

            let vw3_0 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i) as *const __m128i));
            let vw3_1 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i + 8) as *const __m128i));
            sum3_0 = _mm256_fmadd_ps(vw3_0, vs_0, sum3_0);
            sum3_1 = _mm256_fmadd_ps(vw3_1, vs_1, sum3_1);
        });

        dot4x_simd8_avx2!(i, len, {
            let vs = _mm256_loadu_ps(state.as_ptr().add(i));

            let vw0 = _mm256_cvtph_ps(_mm_loadu_si128(w0.as_ptr().add(i) as *const __m128i));
            sum0_0 = _mm256_fmadd_ps(vw0, vs, sum0_0);

            let vw1 = _mm256_cvtph_ps(_mm_loadu_si128(w1.as_ptr().add(i) as *const __m128i));
            sum1_0 = _mm256_fmadd_ps(vw1, vs, sum1_0);

            let vw2 = _mm256_cvtph_ps(_mm_loadu_si128(w2.as_ptr().add(i) as *const __m128i));
            sum2_0 = _mm256_fmadd_ps(vw2, vs, sum2_0);

            let vw3 = _mm256_cvtph_ps(_mm_loadu_si128(w3.as_ptr().add(i) as *const __m128i));
            sum3_0 = _mm256_fmadd_ps(vw3, vs, sum3_0);
        });

        let sum0 = _mm256_add_ps(sum0_0, sum0_1);
        let sum1 = _mm256_add_ps(sum1_0, sum1_1);
        let sum2 = _mm256_add_ps(sum2_0, sum2_1);
        let sum3 = _mm256_add_ps(sum3_0, sum3_1);

        let mut s0: f32 = crate::math::common::utility::hsum_avx2(sum0);
        let mut s1: f32 = crate::math::common::utility::hsum_avx2(sum1);
        let mut s2: f32 = crate::math::common::utility::hsum_avx2(sum2);
        let mut s3: f32 = crate::math::common::utility::hsum_avx2(sum3);
        let mut c0 = 0.0f32;
        let mut c1 = 0.0f32;
        let mut c2 = 0.0f32;
        let mut c3 = 0.0f32;

        while i < len {
            let sw = state[i];
            (s0, c0) = crate::math::common::kahan_add(s0, c0, f16_bits_to_f32_f16c(w0[i]) * sw);
            (s1, c1) = crate::math::common::kahan_add(s1, c1, f16_bits_to_f32_f16c(w1[i]) * sw);
            (s2, c2) = crate::math::common::kahan_add(s2, c2, f16_bits_to_f32_f16c(w2[i]) * sw);
            (s3, c3) = crate::math::common::kahan_add(s3, c3, f16_bits_to_f32_f16c(w3[i]) * sw);
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}

/// Computes 4 Dot Products simultaneously using the "interleaved weights" layout via AVX2.
///
/// In this format, the weights for the 4 computations are laid out together in memory
/// (in groups of 4). This function uses "shuffling" tricks (broadcast and blend) to align
/// the state data with these weights, enabling extremely fast and memory-access efficient
/// (cache friendly) processing.
///
/// # Safety
/// `weights` must have a length greater than or equal to `state.len()`.
/// Both slices must be valid and accessible for reading.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_interleaved_avx2(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut i = 0;

    // SAFETY: `len = state.len().min(weights.len())` and
    // `debug_assert!(weights.len() >= len)`; the `dot4x_simd8_avx2_tail2!`
    // loops keep `i + 8`/`i + 2 <= len`, so accesses to `state[i..i+8]` and
    // `weights[i..i+8]` stay in bounds.
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        dot4x_simd8_avx2_tail2!(
            i,
            len,
            {
                _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().wrapping_add(i + 16) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 8) as *const i8);
                _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 16) as *const i8);

                let s0 = _mm256_broadcast_ss(&state[i]);
                let s1 = _mm256_broadcast_ss(&state[i + 1]);
                let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
                let w01 =
                    _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
                sum0 = _mm256_fmadd_ps(w01, s01, sum0);

                let s2 = _mm256_broadcast_ss(&state[i + 2]);
                let s3 = _mm256_broadcast_ss(&state[i + 3]);
                let s23 = _mm256_blend_ps(s2, s3, 0b11110000);
                let w23 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 2) as *const __m128i
                ));
                sum1 = _mm256_fmadd_ps(w23, s23, sum1);

                let s4 = _mm256_broadcast_ss(&state[i + 4]);
                let s5 = _mm256_broadcast_ss(&state[i + 5]);
                let s45 = _mm256_blend_ps(s4, s5, 0b11110000);
                let w45 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 4) as *const __m128i
                ));
                sum2 = _mm256_fmadd_ps(w45, s45, sum2);

                let s6 = _mm256_broadcast_ss(&state[i + 6]);
                let s7 = _mm256_broadcast_ss(&state[i + 7]);
                let s67 = _mm256_blend_ps(s6, s7, 0b11110000);
                let w67 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 6) as *const __m128i
                ));
                sum3 = _mm256_fmadd_ps(w67, s67, sum3);
            },
            {
                let s0 = _mm256_broadcast_ss(&state[i]);
                let s1 = _mm256_broadcast_ss(&state[i + 1]);
                let s01 = _mm256_blend_ps(s0, s1, 0b11110000);
                let w01 =
                    _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
                sum0 = _mm256_fmadd_ps(w01, s01, sum0);
            }
        );

        let sum01 = _mm256_add_ps(sum0, sum1);
        let sum23 = _mm256_add_ps(sum2, sum3);
        let sum = _mm256_add_ps(sum01, sum23);

        let lower = _mm256_castps256_ps128(sum);
        let upper = _mm256_extractf128_ps(sum, 1);
        let mut sum128 = _mm_add_ps(lower, upper);

        while i < len {
            let s0 = _mm_load1_ps(state.as_ptr().add(i));
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(
                weights.as_ptr().add(i) as *const u16 as *const u8
            ));
            sum128 = _mm_fmadd_ps(w0, s0, sum128);
            i += 1;
        }

        let mut out = [0.0; 4];
        _mm_storeu_ps(out.as_mut_ptr(), sum128);
        out
    }
}
