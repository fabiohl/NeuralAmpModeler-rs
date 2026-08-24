// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Dot Product 4x kernels — AVX-512 (interleaved).

use core::arch::x86_64::*;

/// Dot product interleaved 4x with 8 ZMM accumulators (2 alternating sets of 4)
/// to break dependency chains in the FMA pipeline.
///
/// Each ZMM processes 4 state values (16 f32 lanes: 4 output channels × 4 replications).
/// The 2 sets alternate every 16 state elements to keep the FMA pipeline saturated.
///
/// # Safety
/// `weights` must have a length greater than or equal to `state.len()`.
/// Both slices must be valid and accessible for reading.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_avx512(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut i = 0;

    // SAFETY: `len = state.len().min(weights.len())` with
    // `debug_assert!(weights.len() >= len)`; the loop guards `i + 32`,
    // `i + 16`, `i + 4`, and `i < len` bound every 4-f32/16-u16 load.
    unsafe {
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum_a0 = _mm512_setzero_ps();
        let mut sum_a1 = _mm512_setzero_ps();
        let mut sum_a2 = _mm512_setzero_ps();
        let mut sum_a3 = _mm512_setzero_ps();
        let mut sum_b0 = _mm512_setzero_ps();
        let mut sum_b1 = _mm512_setzero_ps();
        let mut sum_b2 = _mm512_setzero_ps();
        let mut sum_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state.as_ptr().wrapping_add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 32) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 48) as *const i8);

            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 4))),
            );
            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 8))),
            );
            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 12))),
            );
            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            let s_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 16))),
            );
            let w_e = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 16) as *const __m256i
            ));
            sum_b0 = _mm512_fmadd_ps(w_e, s_e, sum_b0);

            let s_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 20))),
            );
            let w_f = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 20) as *const __m256i
            ));
            sum_b1 = _mm512_fmadd_ps(w_f, s_f, sum_b1);

            let s_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 24))),
            );
            let w_g = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 24) as *const __m256i
            ));
            sum_b2 = _mm512_fmadd_ps(w_g, s_g, sum_b2);

            let s_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 28))),
            );
            let w_h = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 28) as *const __m256i
            ));
            sum_b3 = _mm512_fmadd_ps(w_h, s_h, sum_b3);

            i += 32;
        }

        while i + 16 <= len {
            let s_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w_a, s_a, sum_a0);

            let s_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 4))),
            );
            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            sum_a1 = _mm512_fmadd_ps(w_b, s_b, sum_a1);

            let s_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 8))),
            );
            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            sum_a2 = _mm512_fmadd_ps(w_c, s_c, sum_a2);

            let s_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i + 12))),
            );
            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            sum_a3 = _mm512_fmadd_ps(w_d, s_d, sum_a3);

            i += 16;
        }

        while i + 4 <= len {
            let s = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state.as_ptr().add(i))),
            );
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            sum_a0 = _mm512_fmadd_ps(w, s, sum_a0);
            i += 4;
        }

        let s_ab = _mm512_add_ps(_mm512_add_ps(sum_a0, sum_a1), _mm512_add_ps(sum_a2, sum_a3));
        let s_all = _mm512_add_ps(
            s_ab,
            _mm512_add_ps(_mm512_add_ps(sum_b0, sum_b1), _mm512_add_ps(sum_b2, sum_b3)),
        );

        let lo = _mm512_extractf32x4_ps(s_all, 0);
        let hi0 = _mm512_extractf32x4_ps(s_all, 1);
        let hi1 = _mm512_extractf32x4_ps(s_all, 2);
        let hi2 = _mm512_extractf32x4_ps(s_all, 3);
        let mut sum128 = _mm_add_ps(_mm_add_ps(lo, hi0), _mm_add_ps(hi1, hi2));

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
