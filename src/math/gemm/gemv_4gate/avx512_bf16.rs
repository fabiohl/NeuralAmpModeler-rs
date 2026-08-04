// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use core::arch::x86_64::*;

/// GEMV 4-gate BF16 kernel AVX-512 for LSTM.
/// This version uses the BF16 format from the start to be even faster.
#[target_feature(enable = "avx512f,avx512vl,avx512bf16")]
#[expect(
    clippy::too_many_arguments,
    reason = "Performance-critical AVX-512 BF16 LSTM 4-gate kernel requiring many matrix strides/dimensions for maximum SIMD throughput"
)]
pub unsafe fn gemv_4gate_bf16_avx512(
    in_frame: &[u16],
    w0: &[u16],
    w1: &[u16],
    w2: &[u16],
    w3: &[u16],
    bias: &[f32],
    out: &mut [f32],
    do_bias: bool,
) {
    let out_len = out.len() / 4;
    let in_len = in_frame.len();

    let mut out_c = 0;
    while out_c + 16 <= out_len {
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

        let mut in_c = 0;
        while in_c + 2 <= in_len {
            // Load and broadcast the pair (in[i], in[i+1]) to all 16 slots.
            let v_in_raw = _mm256_set1_epi32(*(in_frame.as_ptr().add(in_c) as *const i32));
            let v_in = _mm512_broadcast_i32x8(v_in_raw);

            // Build BF16 pairs (w[i][j], w[i+1][j]) per channel j in each
            // 32-bit lane via zero-extend → shift → OR, without depending
            // on per-lane unpacking (which would swap channels 4-7 with 8-11).
            macro_rules! bf16_pair {
                ($w:ident) => {{
                    let lo = _mm256_loadu_si256(
                        $w.as_ptr().add(in_c * out_len + out_c) as *const __m256i
                    );
                    let hi = _mm256_loadu_si256(
                        $w.as_ptr().add((in_c + 1) * out_len + out_c) as *const __m256i
                    );
                    _mm512_or_si512(
                        _mm512_cvtepu16_epi32(lo),
                        _mm512_slli_epi32(_mm512_cvtepu16_epi32(hi), 16),
                    )
                }};
            }

            let vw0 = bf16_pair!(w0);
            let vw1 = bf16_pair!(w1);
            let vw2 = bf16_pair!(w2);
            let vw3 = bf16_pair!(w3);

            // Multiply and accumulate using the magical BF16 instruction.
            acc0 = _mm512_dpbf16_ps(
                acc0,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512i, __m512bh>(vw0),
            );
            acc1 = _mm512_dpbf16_ps(
                acc1,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512i, __m512bh>(vw1),
            );
            acc2 = _mm512_dpbf16_ps(
                acc2,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512i, __m512bh>(vw2),
            );
            acc3 = _mm512_dpbf16_ps(
                acc3,
                core::mem::transmute::<__m512, __m512bh>(_mm512_castsi512_ps(v_in)),
                core::mem::transmute::<__m512i, __m512bh>(vw3),
            );

            in_c += 2;
        }

        // Manual remainder if needed.
        if in_c < in_len {
            let si = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            let v_in = _mm512_set1_ps(si);

            let vw0 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w0.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw1 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w1.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw2 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w2.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));
            let vw3 = _mm512_castsi512_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                w3.as_ptr().add(in_c * out_len + out_c) as *const __m256i,
            )));

            let vw0_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw0), 16));
            let vw1_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw1), 16));
            let vw2_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw2), 16));
            let vw3_f32 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(vw3), 16));

            acc0 = _mm512_fmadd_ps(v_in, vw0_f32, acc0);
            acc1 = _mm512_fmadd_ps(v_in, vw1_f32, acc1);
            acc2 = _mm512_fmadd_ps(v_in, vw2_f32, acc2);
            acc3 = _mm512_fmadd_ps(v_in, vw3_f32, acc3);
        }

        _mm512_storeu_ps(out.as_mut_ptr().add(out_c), acc0);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + out_len), acc1);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 2 * out_len), acc2);
        _mm512_storeu_ps(out.as_mut_ptr().add(out_c + 3 * out_len), acc3);
        out_c += 16;
    }

    // Tail Loop Processing for AVX-512 BF16:
    // Processes the remaining channels in scalar mode, handling the BF16 in-memory representation.
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
            // Since the input and weights are in BF16 format (16 bits), to process in
            // scalar fashion, we shift the bits 16 positions to the left (<< 16) to convert
            // them back to the normal f32 representation (IEEE 754) before multiplying.
            let si = f32::from_bits((*in_frame.get_unchecked(in_c) as u32) << 16);
            s0 += si * f32::from_bits((*w0.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s1 += si * f32::from_bits((*w1.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s2 += si * f32::from_bits((*w2.get_unchecked(in_c * out_len + out_c) as u32) << 16);
            s3 += si * f32::from_bits((*w3.get_unchecked(in_c * out_len + out_c) as u32) << 16);
        }
        out[out_c] = s0;
        out[out_c + out_len] = s1;
        out[out_c + 2 * out_len] = s2;
        out[out_c + 3 * out_len] = s3;
        out_c += 1;
    }
}
