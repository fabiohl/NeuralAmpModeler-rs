// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Dot Product 4x kernels — AVX2 (dual-frame + batch).

use crate::dot4x_simd8_avx2;
use crate::dot4x_simd8_avx2_tail2;
use crate::dot4x_simd16_avx2;
use crate::math::common::half::f16_bits_to_f32_f16c;
use core::arch::x86_64::*;

/// Computes 4 Dot Products for two simultaneous audio frames (Dual Frame) via AVX2.
///
/// This is one of the most efficient functions in the system. It takes advantage of the fact
/// that the weights have already been loaded into memory to apply them to two different audio
/// blocks (f0 and f1) at the same time. This doubles processor throughput, since each loaded
/// weight is "reused" immediately for two distinct computations.
///
/// # Safety
/// `weights` must have a length greater than or equal to both `state_f0.len()` and `state_f1.len()`.
/// All slices must be valid and accessible for reading.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx2(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = state_f0.len().min(state_f1.len()).min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut i = 0;

    // SAFETY: `len` is the minimum of `state_f0.len()`, `state_f1.len()`, and
    // `weights.len()` (with `debug_assert!(weights.len() >= len)`); the macro
    // loops keep `i + 8`/`i + 2 <= len`, so all accesses stay in bounds.
    unsafe {
        let mut sum0_f0 = _mm256_setzero_ps();
        let mut sum1_f0 = _mm256_setzero_ps();
        let mut sum2_f0 = _mm256_setzero_ps();
        let mut sum3_f0 = _mm256_setzero_ps();

        let mut sum0_f1 = _mm256_setzero_ps();
        let mut sum1_f1 = _mm256_setzero_ps();
        let mut sum2_f1 = _mm256_setzero_ps();
        let mut sum3_f1 = _mm256_setzero_ps();

        dot4x_simd8_avx2_tail2!(
            i,
            len,
            {
                let s0_f0 = _mm256_broadcast_ss(&state_f0[i]);
                let s1_f0 = _mm256_broadcast_ss(&state_f0[i + 1]);
                let s01_f0 = _mm256_blend_ps(s0_f0, s1_f0, 0b11110000);

                let s0_f1 = _mm256_broadcast_ss(&state_f1[i]);
                let s1_f1 = _mm256_broadcast_ss(&state_f1[i + 1]);
                let s01_f1 = _mm256_blend_ps(s0_f1, s1_f1, 0b11110000);

                let w01 =
                    _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
                sum0_f0 = _mm256_fmadd_ps(w01, s01_f0, sum0_f0);
                sum0_f1 = _mm256_fmadd_ps(w01, s01_f1, sum0_f1);

                let s2_f0 = _mm256_broadcast_ss(&state_f0[i + 2]);
                let s3_f0 = _mm256_broadcast_ss(&state_f0[i + 3]);
                let s23_f0 = _mm256_blend_ps(s2_f0, s3_f0, 0b11110000);

                let s2_f1 = _mm256_broadcast_ss(&state_f1[i + 2]);
                let s3_f1 = _mm256_broadcast_ss(&state_f1[i + 3]);
                let s23_f1 = _mm256_blend_ps(s2_f1, s3_f1, 0b11110000);

                let w23 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 2) as *const __m128i
                ));
                sum1_f0 = _mm256_fmadd_ps(w23, s23_f0, sum1_f0);
                sum1_f1 = _mm256_fmadd_ps(w23, s23_f1, sum1_f1);

                let s4_f0 = _mm256_broadcast_ss(&state_f0[i + 4]);
                let s5_f0 = _mm256_broadcast_ss(&state_f0[i + 5]);
                let s45_f0 = _mm256_blend_ps(s4_f0, s5_f0, 0b11110000);

                let s4_f1 = _mm256_broadcast_ss(&state_f1[i + 4]);
                let s5_f1 = _mm256_broadcast_ss(&state_f1[i + 5]);
                let s45_f1 = _mm256_blend_ps(s4_f1, s5_f1, 0b11110000);

                let w45 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 4) as *const __m128i
                ));
                sum2_f0 = _mm256_fmadd_ps(w45, s45_f0, sum2_f0);
                sum2_f1 = _mm256_fmadd_ps(w45, s45_f1, sum2_f1);

                let s6_f0 = _mm256_broadcast_ss(&state_f0[i + 6]);
                let s7_f0 = _mm256_broadcast_ss(&state_f0[i + 7]);
                let s67_f0 = _mm256_blend_ps(s6_f0, s7_f0, 0b11110000);

                let s6_f1 = _mm256_broadcast_ss(&state_f1[i + 6]);
                let s7_f1 = _mm256_broadcast_ss(&state_f1[i + 7]);
                let s67_f1 = _mm256_blend_ps(s6_f1, s7_f1, 0b11110000);

                let w67 = _mm256_cvtph_ps(_mm_loadu_si128(
                    weights.as_ptr().add(i + 6) as *const __m128i
                ));
                sum3_f0 = _mm256_fmadd_ps(w67, s67_f0, sum3_f0);
                sum3_f1 = _mm256_fmadd_ps(w67, s67_f1, sum3_f1);
            },
            {
                let s0_f0 = _mm256_broadcast_ss(&state_f0[i]);
                let s1_f0 = _mm256_broadcast_ss(&state_f0[i + 1]);
                let s01_f0 = _mm256_blend_ps(s0_f0, s1_f0, 0b11110000);

                let s0_f1 = _mm256_broadcast_ss(&state_f1[i]);
                let s1_f1 = _mm256_broadcast_ss(&state_f1[i + 1]);
                let s01_f1 = _mm256_blend_ps(s0_f1, s1_f1, 0b11110000);

                let w01 =
                    _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
                sum0_f0 = _mm256_fmadd_ps(w01, s01_f0, sum0_f0);
                sum0_f1 = _mm256_fmadd_ps(w01, s01_f1, sum0_f1);
            }
        );

        let sum01_f0 = _mm256_add_ps(sum0_f0, sum1_f0);
        let sum23_f0 = _mm256_add_ps(sum2_f0, sum3_f0);
        let sum_f0 = _mm256_add_ps(sum01_f0, sum23_f0);

        let sum01_f1 = _mm256_add_ps(sum0_f1, sum1_f1);
        let sum23_f1 = _mm256_add_ps(sum2_f1, sum3_f1);
        let sum_f1 = _mm256_add_ps(sum01_f1, sum23_f1);

        let lower_f0 = _mm256_castps256_ps128(sum_f0);
        let upper_f0 = _mm256_extractf128_ps(sum_f0, 1);
        let mut sum128_f0 = _mm_add_ps(lower_f0, upper_f0);

        let lower_f1 = _mm256_castps256_ps128(sum_f1);
        let upper_f1 = _mm256_extractf128_ps(sum_f1, 1);
        let mut sum128_f1 = _mm_add_ps(lower_f1, upper_f1);

        while i < len {
            let s0_f0 = _mm_load1_ps(state_f0.as_ptr().add(i));
            let s0_f1 = _mm_load1_ps(state_f1.as_ptr().add(i));
            let w0 = _mm_cvtph_ps(_mm_loadu_si64(
                weights.as_ptr().add(i) as *const u16 as *const u8
            ));
            sum128_f0 = _mm_fmadd_ps(w0, s0_f0, sum128_f0);
            sum128_f1 = _mm_fmadd_ps(w0, s0_f1, sum128_f1);
            i += 1;
        }

        let mut out_f0 = [0.0; 4];
        let mut out_f1 = [0.0; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), sum128_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), sum128_f1);
        (out_f0, out_f1)
    }
}

/// Specialized kernel for multiplying weights by 4 different audio channels at the same time.
/// This function is the "do-it-all" of WaveNet and LSTM neural networks when we process
/// audio in batch. It saves energy and time by not having to read
/// the same weights from memory repeatedly.
///
/// # Safety
/// `weights` must have a length greater than or equal to `h0.len()`, `h1.len()`, `h2.len()`, and `h3.len()`.
/// All slices must be valid and accessible for reading.
#[target_feature(enable = "f16c")]
pub unsafe fn dot_product_batch_4x_avx2(
    h0: &[f32],
    h1: &[f32],
    h2: &[f32],
    h3: &[f32],
    weights: &[u16],
) -> [f32; 4] {
    let len = weights
        .len()
        .min(h0.len())
        .min(h1.len())
        .min(h2.len())
        .min(h3.len());
    debug_assert!(h0.len() >= len && h1.len() >= len && h2.len() >= len && h3.len() >= len);
    let mut i = 0;

    // SAFETY: `len` is the minimum of `weights.len()` and `h0..h3.len()`
    // (enforced by the `debug_assert!` above); the macro loops guarantee
    // `i + 16`/`i + 8`/`i < len`, so the 8-lane loads and 16-u16 loads at
    // offset `i` stay within all five slices.
    unsafe {
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        dot4x_simd16_avx2!(i, len, {
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h0.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h1.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h2.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(h3.as_ptr().wrapping_add(i + 32) as *const i8);

            let vw_0 = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));

            let vh0_0 = _mm256_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(vw_0, vh0_0, sum0);

            let vh1_0 = _mm256_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm256_fmadd_ps(vw_0, vh1_0, sum1);

            let vh2_0 = _mm256_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm256_fmadd_ps(vw_0, vh2_0, sum2);

            let vh3_0 = _mm256_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm256_fmadd_ps(vw_0, vh3_0, sum3);

            let vw_1 = _mm256_cvtph_ps(_mm_loadu_si128(
                weights.as_ptr().add(i + 8) as *const __m128i
            ));
            let vh0_1 = _mm256_loadu_ps(h0.as_ptr().add(i + 8));
            sum0 = _mm256_fmadd_ps(vw_1, vh0_1, sum0);
            let vh1_1 = _mm256_loadu_ps(h1.as_ptr().add(i + 8));
            sum1 = _mm256_fmadd_ps(vw_1, vh1_1, sum1);
            let vh2_1 = _mm256_loadu_ps(h2.as_ptr().add(i + 8));
            sum2 = _mm256_fmadd_ps(vw_1, vh2_1, sum2);
            let vh3_1 = _mm256_loadu_ps(h3.as_ptr().add(i + 8));
            sum3 = _mm256_fmadd_ps(vw_1, vh3_1, sum3);
        });

        dot4x_simd8_avx2!(i, len, {
            let vw = _mm256_cvtph_ps(_mm_loadu_si128(weights.as_ptr().add(i) as *const __m128i));
            let vh0 = _mm256_loadu_ps(h0.as_ptr().add(i));
            sum0 = _mm256_fmadd_ps(vw, vh0, sum0);
            let vh1 = _mm256_loadu_ps(h1.as_ptr().add(i));
            sum1 = _mm256_fmadd_ps(vw, vh1, sum1);
            let vh2 = _mm256_loadu_ps(h2.as_ptr().add(i));
            sum2 = _mm256_fmadd_ps(vw, vh2, sum2);
            let vh3 = _mm256_loadu_ps(h3.as_ptr().add(i));
            sum3 = _mm256_fmadd_ps(vw, vh3, sum3);
        });

        let mut s0 = crate::math::common::utility::hsum_avx2(sum0);
        let mut s1 = crate::math::common::utility::hsum_avx2(sum1);
        let mut s2 = crate::math::common::utility::hsum_avx2(sum2);
        let mut s3 = crate::math::common::utility::hsum_avx2(sum3);
        let mut c0 = 0.0f32;
        let mut c1 = 0.0f32;
        let mut c2 = 0.0f32;
        let mut c3 = 0.0f32;

        while i < len {
            let w = f16_bits_to_f32_f16c(weights[i]);
            (s0, c0) = crate::math::common::kahan_add(s0, c0, w * h0[i]);
            (s1, c1) = crate::math::common::kahan_add(s1, c1, w * h1[i]);
            (s2, c2) = crate::math::common::kahan_add(s2, c2, w * h2[i]);
            (s3, c3) = crate::math::common::kahan_add(s3, c3, w * h3[i]);
            i += 1;
        }

        [s0, s1, s2, s3]
    }
}
