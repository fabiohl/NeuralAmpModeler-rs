// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! GEMV BF16 kernel — matrix-vector multiplication with data in BF16 format.
//!
//! Uses `_mm512_dpbf16_ps` to process 32 BF16 pairs per instruction,
//! with 8 independent accumulators to break the FMA dependency chain.

use core::arch::x86_64::*;

/// GEMV BF16 AVX-512: Y = Bias + W * X, where X and W are in BF16 format.
///
/// Processes the linear projection in blocks of 16 output channels, with 8 independent
/// ZMM accumulators and step 16 in the inner loop (16 BF16 pairs = 32 elements).
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
pub unsafe fn gemv_overwrite_bf16_avx512(
    in_frame: &[u16],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        let mut in_c = 0;
        while in_c + 16 <= in_len {
            macro_rules! bf16_gemv_step {
                ($acc:ident, $k:expr) => {
                    let idx = in_c + ($k) * 2;
                    let v_in_raw = _mm256_set1_epi32(*(in_frame.as_ptr().add(idx) as *const i32));
                    let v_in = _mm512_broadcast_i32x8(v_in_raw);
                    let w_ptr = weights.as_ptr().add(idx * out_len + out_c);
                    let lo = _mm256_loadu_si256(w_ptr as *const __m256i);
                    let hi = _mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i);
                    let vw = _mm512_or_si512(
                        _mm512_cvtepu16_epi32(lo),
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16),
                    );
                    $acc = _mm512_dpbf16_ps(
                        $acc,
                        // SAFETY: __m512 → __m512bh is a no-op transmute of 512-bit
                        // vectors with identical ABI (same size, alignment, register class).
                        core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                        // SAFETY: __m512i → __m512bh is a no-op transmute of 512-bit
                        // vectors with identical ABI.
                        core::mem::transmute::<__m512i, __m512bh>(vw),
                    );
                };
            }
            bf16_gemv_step!(acc0, 0);
            bf16_gemv_step!(acc1, 1);
            bf16_gemv_step!(acc2, 2);
            bf16_gemv_step!(acc3, 3);
            bf16_gemv_step!(acc4, 4);
            bf16_gemv_step!(acc5, 5);
            bf16_gemv_step!(acc6, 6);
            bf16_gemv_step!(acc7, 7);
            in_c += 16;
        }

        acc0 = _mm512_add_ps(acc0, acc1);
        acc2 = _mm512_add_ps(acc2, acc3);
        acc4 = _mm512_add_ps(acc4, acc5);
        acc6 = _mm512_add_ps(acc6, acc7);
        acc0 = _mm512_add_ps(acc0, acc2);
        acc4 = _mm512_add_ps(acc4, acc6);
        acc0 = _mm512_add_ps(acc0, acc4);

        // Process remaining pairs with _mm512_dpbf16_ps.
        while in_c + 2 <= in_len {
            let v_in_raw = _mm256_set1_epi32(*(in_frame.as_ptr().add(in_c) as *const i32));
            let v_in = _mm512_broadcast_i32x8(v_in_raw);
            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let lo = _mm256_loadu_si256(w_ptr as *const __m256i);
            let hi = _mm256_loadu_si256(w_ptr.add(out_len) as *const __m256i);
            let vw = _mm512_or_si512(
                _mm512_cvtepu16_epi32(lo),
                _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16),
            );
            acc0 = _mm512_dpbf16_ps(
                acc0,
                // SAFETY: __m512 → __m512bh: no-op 512-bit transmute, identical ABI.
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                // SAFETY: __m512i → __m512bh: no-op 512-bit transmute, identical ABI.
                core::mem::transmute::<__m512i, __m512bh>(vw),
            );
            in_c += 2;
        }

        // Remaining odd element: simple broadcast + f32 FMA.
        if in_c < in_len {
            let si = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            let v_in = _mm512_set1_ps(si);
            let w_ptr = weights.as_ptr().add(in_c * out_len + out_c);
            let vw_raw = _mm512_cvtepu16_epi32(_mm256_loadu_si256(w_ptr as *const __m256i));
            let vw = _mm512_castsi512_ps(_mm512_slli_epi32(vw_raw, 16));
            acc0 = _mm512_fmadd_ps(v_in, vw, acc0);
        }

        _mm512_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        out_c += 16;
    }

    // Remaining output channels via scalar fallback (already corrected).
    while out_c < out_len {
        let mut sum = if do_bias {
            *bias.get_unchecked(out_c)
        } else {
            0.0f32
        };
        for in_c in 0..in_len {
            let s = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            sum +=
                s * f32::from_bits((*weights.get_unchecked(in_c * out_len + out_c) as u32) << 16);
        }
        *out_frame.get_unchecked_mut(out_c) = sum;
        out_c += 1;
    }
}
