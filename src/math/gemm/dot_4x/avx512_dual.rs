// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! Kernels Dot Product 4x — AVX-512 (dual-frame).

use core::arch::x86_64::*;

/// Processes two frames simultaneously with 16 ZMM accumulators (8 per frame)
/// in 2 alternating sets to break FMA dependencies.
///
/// Reuses the same weight load for both frames (f0 and f1),
/// doubling memory access efficiency.
///
/// # Safety
/// `weights` must have a length greater than or equal to both `state_f0.len()` and `state_f1.len()`.
/// All slices must be valid and accessible for reading.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_interleaved_dual_frame_avx512(
    weights: &[[u16; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = weights.len().min(state_f0.len()).min(state_f1.len());
    debug_assert!(weights.len() >= len);
    let mut i = 0;

    unsafe {
        // Accumulator alternation: 16 ZMM accumulators (8 per frame).
        // Set A (sum{a}{0..3}) and Set B (sum{b}{0..3}) alternate every
        // 4 samples to hide FMA latency. The 32-sample loop processes
        // 8 blocks of 4 (A,B,A,B,A,B,A,B), the 16-sample loop processes
        // 4 blocks (A only), and the tail handles < 4 elements.
        let perm_idx = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);

        let mut sum0_a0 = _mm512_setzero_ps();
        let mut sum0_a1 = _mm512_setzero_ps();
        let mut sum0_a2 = _mm512_setzero_ps();
        let mut sum0_a3 = _mm512_setzero_ps();
        let mut sum0_b0 = _mm512_setzero_ps();
        let mut sum0_b1 = _mm512_setzero_ps();
        let mut sum0_b2 = _mm512_setzero_ps();
        let mut sum0_b3 = _mm512_setzero_ps();

        let mut sum1_a0 = _mm512_setzero_ps();
        let mut sum1_a1 = _mm512_setzero_ps();
        let mut sum1_a2 = _mm512_setzero_ps();
        let mut sum1_a3 = _mm512_setzero_ps();
        let mut sum1_b0 = _mm512_setzero_ps();
        let mut sum1_b1 = _mm512_setzero_ps();
        let mut sum1_b2 = _mm512_setzero_ps();
        let mut sum1_b3 = _mm512_setzero_ps();

        while i + 32 <= len {
            _mm_prefetch::<_MM_HINT_T0>(state_f0.as_ptr().wrapping_add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(state_f1.as_ptr().wrapping_add(i + 64) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 16) as *const i8);
            _mm_prefetch::<_MM_HINT_T0>(weights.as_ptr().wrapping_add(i + 32) as *const i8);

            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            let w_e = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 16) as *const __m256i
            ));
            let s0_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 16))),
            );
            let s1_e = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 16))),
            );
            sum0_b0 = _mm512_fmadd_ps(w_e, s0_e, sum0_b0);
            sum1_b0 = _mm512_fmadd_ps(w_e, s1_e, sum1_b0);

            let w_f = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 20) as *const __m256i
            ));
            let s0_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 20))),
            );
            let s1_f = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 20))),
            );
            sum0_b1 = _mm512_fmadd_ps(w_f, s0_f, sum0_b1);
            sum1_b1 = _mm512_fmadd_ps(w_f, s1_f, sum1_b1);

            let w_g = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 24) as *const __m256i
            ));
            let s0_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 24))),
            );
            let s1_g = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 24))),
            );
            sum0_b2 = _mm512_fmadd_ps(w_g, s0_g, sum0_b2);
            sum1_b2 = _mm512_fmadd_ps(w_g, s1_g, sum1_b2);

            let w_h = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 28) as *const __m256i
            ));
            let s0_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 28))),
            );
            let s1_h = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 28))),
            );
            sum0_b3 = _mm512_fmadd_ps(w_h, s0_h, sum0_b3);
            sum1_b3 = _mm512_fmadd_ps(w_h, s1_h, sum1_b3);

            i += 32;
        }

        while i + 16 <= len {
            let w_a =
                _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1_a = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w_a, s0_a, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w_a, s1_a, sum1_a0);

            let w_b = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 4) as *const __m256i
            ));
            let s0_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 4))),
            );
            let s1_b = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 4))),
            );
            sum0_a1 = _mm512_fmadd_ps(w_b, s0_b, sum0_a1);
            sum1_a1 = _mm512_fmadd_ps(w_b, s1_b, sum1_a1);

            let w_c = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 8) as *const __m256i
            ));
            let s0_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 8))),
            );
            let s1_c = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 8))),
            );
            sum0_a2 = _mm512_fmadd_ps(w_c, s0_c, sum0_a2);
            sum1_a2 = _mm512_fmadd_ps(w_c, s1_c, sum1_a2);

            let w_d = _mm512_cvtph_ps(_mm256_loadu_si256(
                weights.as_ptr().add(i + 12) as *const __m256i
            ));
            let s0_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i + 12))),
            );
            let s1_d = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i + 12))),
            );
            sum0_a3 = _mm512_fmadd_ps(w_d, s0_d, sum0_a3);
            sum1_a3 = _mm512_fmadd_ps(w_d, s1_d, sum1_a3);

            i += 16;
        }

        while i + 4 <= len {
            let w = _mm512_cvtph_ps(_mm256_loadu_si256(weights.as_ptr().add(i) as *const __m256i));
            let s0 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f0.as_ptr().add(i))),
            );
            let s1 = _mm512_permutexvar_ps(
                perm_idx,
                _mm512_castps128_ps512(_mm_loadu_ps(state_f1.as_ptr().add(i))),
            );
            sum0_a0 = _mm512_fmadd_ps(w, s0, sum0_a0);
            sum1_a0 = _mm512_fmadd_ps(w, s1, sum1_a0);
            i += 4;
        }

        let s0_ab = _mm512_add_ps(
            _mm512_add_ps(sum0_a0, sum0_a1),
            _mm512_add_ps(sum0_a2, sum0_a3),
        );
        let s0_all = _mm512_add_ps(
            s0_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum0_b0, sum0_b1),
                _mm512_add_ps(sum0_b2, sum0_b3),
            ),
        );
        let s1_ab = _mm512_add_ps(
            _mm512_add_ps(sum1_a0, sum1_a1),
            _mm512_add_ps(sum1_a2, sum1_a3),
        );
        let s1_all = _mm512_add_ps(
            s1_ab,
            _mm512_add_ps(
                _mm512_add_ps(sum1_b0, sum1_b1),
                _mm512_add_ps(sum1_b2, sum1_b3),
            ),
        );

        let lo0 = _mm512_extractf32x4_ps(s0_all, 0);
        let hi00 = _mm512_extractf32x4_ps(s0_all, 1);
        let hi01 = _mm512_extractf32x4_ps(s0_all, 2);
        let hi02 = _mm512_extractf32x4_ps(s0_all, 3);
        let mut sum128_f0 = _mm_add_ps(_mm_add_ps(lo0, hi00), _mm_add_ps(hi01, hi02));

        let lo1 = _mm512_extractf32x4_ps(s1_all, 0);
        let hi10 = _mm512_extractf32x4_ps(s1_all, 1);
        let hi11 = _mm512_extractf32x4_ps(s1_all, 2);
        let hi12 = _mm512_extractf32x4_ps(s1_all, 3);
        let mut sum128_f1 = _mm_add_ps(_mm_add_ps(lo1, hi10), _mm_add_ps(hi11, hi12));

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
