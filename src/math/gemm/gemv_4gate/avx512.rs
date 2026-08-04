// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

/// GEMV 4-gate kernel AVX-512 for LSTM.
/// Gates in an LSTM network control what should be remembered and what should be forgotten.
/// This function processes the 4 main gates at once for a major speed boost.
// AVX-512F+VL is the baseline AVX-512 target for this kernel.
// f16c kept for x86-64-v3 baseline compatibility with caller target-feature requirements.
#[target_feature(enable = "avx512f,avx512vl,f16c")]
#[expect(
    clippy::too_many_arguments,
    reason = "Performance-critical AVX-512 LSTM 4-gate kernel requiring many matrix strides/dimensions for maximum SIMD throughput"
)]
pub unsafe fn gemv_4gate_avx512(
    in_frame: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
        // Buckets for the 4 gates.
        let mut acc0 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc1 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc2 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 2 * out_len))
        } else {
            _mm512_setzero_ps()
        };
        let mut acc3 = if do_bias {
            _mm512_loadu_ps(bias.as_ptr().add(out_c + 3 * out_len))
        } else {
            _mm512_setzero_ps()
        };

        for in_c in 0..in_len {
            let vs = _mm512_set1_ps(*in_frame.get_unchecked(in_c));

            let vw0 = _mm512_loadu_ps(w0.as_ptr().add(in_c * out_len + out_c));
            let vw1 = _mm512_loadu_ps(w1.as_ptr().add(in_c * out_len + out_c));
            let vw2 = _mm512_loadu_ps(w2.as_ptr().add(in_c * out_len + out_c));
            let vw3 = _mm512_loadu_ps(w3.as_ptr().add(in_c * out_len + out_c));

            // Multiply and accumulate into all 4 buckets at the same time.
            acc0 = _mm512_fmadd_ps(vs, vw0, acc0);
            acc1 = _mm512_fmadd_ps(vs, vw1, acc1);
            acc2 = _mm512_fmadd_ps(vs, vw2, acc2);
            acc3 = _mm512_fmadd_ps(vs, vw3, acc3);
        }

        // Save the results.
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c), acc0);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + out_len), acc1);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 2 * out_len), acc2);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 3 * out_len), acc3);
        out_c += 16;
    }

    while out_c < out_len {
        let mut s0 = if do_bias { bias[out_c] } else { 0.0 };
        let mut s1 = if do_bias { bias[out_c + out_len] } else { 0.0 };
        let mut s2 = if do_bias {
            bias[out_c + 2 * out_len]
        } else {
            0.0
        };
        let mut s3 = if do_bias {
            bias[out_c + 3 * out_len]
        } else {
            0.0
        };
        for in_c in 0..in_len {
            let si = *in_frame.get_unchecked(in_c);
            s0 += si * w0.get_unchecked(in_c * out_len + out_c);
            s1 += si * w1.get_unchecked(in_c * out_len + out_c);
            s2 += si * w2.get_unchecked(in_c * out_len + out_c);
            s3 += si * w3.get_unchecked(in_c * out_len + out_c);
        }
        out[out_c] = s0;
        out[out_c + out_len] = s1;
        out[out_c + 2 * out_len] = s2;
        out[out_c + 3 * out_len] = s3;
        out_c += 1;
    }
}
