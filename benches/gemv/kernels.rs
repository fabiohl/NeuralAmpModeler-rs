// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// Specialized GEMV kernels (AVX2, fully-unrolled by dimension).
// These are bench prototypes, not yet canonized in the library.

use core::arch::x86_64::*;
use neural_amp_modeler_rs::math::common::half::f16_bits_to_f32_f16c;

/// Specialized fused GEMV for: 1 input × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_1x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let v_in = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w_ptr = weights.as_ptr();
        let vw = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        let mut acc = _mm256_mul_ps(v_in, vw);
        if do_bias {
            let vb = _mm256_loadu_ps(bias.as_ptr());
            acc = _mm256_add_ps(acc, vb);
        }
        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        for (i, &val) in tmp.iter().enumerate().take(4) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 4 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_4x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(4) as *const __m128i));
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(12) as *const __m128i));
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        for (i, &val) in tmp.iter().enumerate().take(4) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 4 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_4x6(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm_loadu_si128(w_ptr as *const __m128i);
        let w0_ps = _mm256_cvtph_ps(w0);
        acc = _mm256_fmadd_ps(v0, w0_ps, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm_loadu_si128(w_ptr.add(6) as *const __m128i);
        let w1_ps = _mm256_cvtph_ps(w1);
        acc = _mm256_fmadd_ps(v1, w1_ps, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm_loadu_si128(w_ptr.add(12) as *const __m128i);
        let w2_ps = _mm256_cvtph_ps(w2);
        acc = _mm256_fmadd_ps(v2, w2_ps, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm_loadu_si128(w_ptr.add(18) as *const __m128i);
        let w3_ps = _mm256_cvtph_ps(w3);
        acc = _mm256_fmadd_ps(v3, w3_ps, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);

        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);

        // Lanes 4..5: scalar accumulation for the 2 outputs beyond the SIMD width.
        for oc in 4..6 {
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(oc));
            let mut sum = *in_frame.get_unchecked(0) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(6 + oc));
            sum += *in_frame.get_unchecked(1) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(12 + oc));
            sum += *in_frame.get_unchecked(2) * w;
            let w = f16_bits_to_f32_f16c(*weights.get_unchecked(18 + oc));
            sum += *in_frame.get_unchecked(3) * w;
            if do_bias {
                sum += bias[oc];
            }
            tmp[oc] = *out_frame.get_unchecked(oc) + sum;
        }

        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 8 inputs × 4 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_8x4(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();
        let mut acc = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };

        let v0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc = _mm256_fmadd_ps(v0, w0, acc);

        let v1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(4) as *const __m128i));
        acc = _mm256_fmadd_ps(v1, w1, acc);

        let v2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc = _mm256_fmadd_ps(v2, w2, acc);

        let v3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(12) as *const __m128i));
        acc = _mm256_fmadd_ps(v3, w3, acc);

        let v4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let w4 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(16) as *const __m128i));
        acc = _mm256_fmadd_ps(v4, w4, acc);

        let v5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let w5 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(20) as *const __m128i));
        acc = _mm256_fmadd_ps(v5, w5, acc);

        let v6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let w6 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(24) as *const __m128i));
        acc = _mm256_fmadd_ps(v6, w6, acc);

        let v7 = _mm256_set1_ps(*in_frame.get_unchecked(7));
        let w7 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(28) as *const __m128i));
        acc = _mm256_fmadd_ps(v7, w7, acc);

        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc = _mm256_add_ps(acc, out_val);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        for (i, &val) in tmp.iter().enumerate().take(4) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 8 inputs × 6 outputs.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_8x6(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let out_len = 6usize;
        let w_ptr = weights.as_ptr();

        let mut tmp = [0.0f32; 8];

        let mut acc_simd = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let out_simd = _mm256_loadu_ps(out_frame.as_ptr());
        acc_simd = _mm256_add_ps(acc_simd, out_simd);

        // Scalar accumulators for output lanes 4 and 5 (beyond the SIMD tail).
        let mut sum4 = if do_bias { bias[4] } else { 0.0 };
        let mut sum5 = if do_bias { bias[5] } else { 0.0 };

        for in_c in 0..8 {
            let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
            let w_ptr_row = w_ptr.add(in_c * out_len);
            let w_simd = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr_row as *const __m128i));
            acc_simd = _mm256_fmadd_ps(vs, w_simd, acc_simd);
            sum4 += *in_frame.get_unchecked(in_c)
                * f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + 4));
            sum5 += *in_frame.get_unchecked(in_c)
                * f16_bits_to_f32_f16c(*weights.get_unchecked(in_c * out_len + 5));
        }

        _mm256_storeu_ps(tmp.as_mut_ptr(), acc_simd);
        tmp[4] = *out_frame.get_unchecked(4) + sum4;
        tmp[5] = *out_frame.get_unchecked(5) + sum5;

        for (i, &val) in tmp.iter().enumerate().take(6) {
            *out_frame.get_unchecked_mut(i) = val;
        }
    }
}

/// Specialized fused GEMV for: 8 inputs × 8 outputs.
///
/// This matches the inner block size of the generic kernel (8 SIMD accumulators × step 8),
/// but without the outer loop overhead.
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn gemv_specialized_8x8(
    in_frame: &[f32],
    weights: &[u16],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    unsafe {
        let w_ptr = weights.as_ptr();

        let mut acc0 = if do_bias {
            _mm256_loadu_ps(bias.as_ptr())
        } else {
            _mm256_setzero_ps()
        };
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();
        let mut acc4 = _mm256_setzero_ps();
        let mut acc5 = _mm256_setzero_ps();
        let mut acc6 = _mm256_setzero_ps();
        let mut acc7 = _mm256_setzero_ps();

        // Fully unrolled 8-input accumulation (one YMM accumulator per input).
        let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(0));
        let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(1));
        let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(2));
        let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(3));
        let vs4 = _mm256_set1_ps(*in_frame.get_unchecked(4));
        let vs5 = _mm256_set1_ps(*in_frame.get_unchecked(5));
        let vs6 = _mm256_set1_ps(*in_frame.get_unchecked(6));
        let vs7 = _mm256_set1_ps(*in_frame.get_unchecked(7));

        let w0 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr as *const __m128i));
        acc0 = _mm256_fmadd_ps(vs0, w0, acc0);
        let w1 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(8) as *const __m128i));
        acc1 = _mm256_fmadd_ps(vs1, w1, acc1);
        let w2 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(16) as *const __m128i));
        acc2 = _mm256_fmadd_ps(vs2, w2, acc2);
        let w3 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(24) as *const __m128i));
        acc3 = _mm256_fmadd_ps(vs3, w3, acc3);
        let w4 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(32) as *const __m128i));
        acc4 = _mm256_fmadd_ps(vs4, w4, acc4);
        let w5 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(40) as *const __m128i));
        acc5 = _mm256_fmadd_ps(vs5, w5, acc5);
        let w6 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(48) as *const __m128i));
        acc6 = _mm256_fmadd_ps(vs6, w6, acc6);
        let w7 = _mm256_cvtph_ps(_mm_loadu_si128(w_ptr.add(56) as *const __m128i));
        acc7 = _mm256_fmadd_ps(vs7, w7, acc7);

        // Reduction tree: collapse the 8 accumulators into acc0.
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);
        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);
        acc0 = _mm256_add_ps(acc0, acc4);

        // Fused add: accumulate with the existing out_frame, then store back.
        let out_val = _mm256_loadu_ps(out_frame.as_ptr());
        acc0 = _mm256_add_ps(acc0, out_val);
        _mm256_storeu_ps(out_frame.as_mut_ptr(), acc0);
    }
}
