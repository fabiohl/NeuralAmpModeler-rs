// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::gemv_4gate_inner_dual_avx2;
use crate::gemv_4gate_simd_outer_avx2;
use core::arch::x86_64::*;

/// Performs the linear projection for the 4 "gates" of an LSTM cell simultaneously via AVX2.
///
/// In an LSTM neural network, each step requires computing 4 sub-results (gates). This
/// function executes all these computations at once, ensuring that the network's "memory"
/// update is done with maximum performance and minimum latency.
///
/// # Optimization
/// Processes 2 input columns per iteration with 8 independent FMA accumulators
/// (2 per gate: `acc_lo`/`acc_hi`), breaking the serial FMA dependency chain and
/// doubling port utilization on x86-64-v3 (Haswell+) where FMA has ~4–5 cycle
/// latency and 2 execution ports.
#[target_feature(enable = "avx2,fma,f16c")]
#[expect(
    clippy::too_many_arguments,
    reason = "Performance-critical AVX2 LSTM 4-gate kernel requiring many matrix strides/dimensions for maximum SIMD throughput"
)]
pub unsafe fn gemv_4gate_avx2(
    in_frame: &[f32],
    w0: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len() / 4;
    let in_len = in_frame.len();

    // SAFETY: the macro loops keep `out_c + 8 <= out_len` and `in_c + 2 <=
    // in_len`; with `out_frame.len() == 4 * out_len`, `bias.len() >= 4 *
    // out_len` (when `do_bias`), and each `w*.len() >= in_len * out_len`, all
    // 8-lane loads/stores stay in bounds.
    unsafe {
        let mut out_c = 0;
        gemv_4gate_simd_outer_avx2!(
            out_c,
            out_len,
            {
                let bias_g0 = if do_bias {
                    _mm256_loadu_ps(bias.as_ptr().add(out_c))
                } else {
                    _mm256_setzero_ps()
                };
                let bias_g1 = if do_bias {
                    _mm256_loadu_ps(bias.as_ptr().add(out_len + out_c))
                } else {
                    _mm256_setzero_ps()
                };
                let bias_g2 = if do_bias {
                    _mm256_loadu_ps(bias.as_ptr().add(2 * out_len + out_c))
                } else {
                    _mm256_setzero_ps()
                };
                let bias_g3 = if do_bias {
                    _mm256_loadu_ps(bias.as_ptr().add(3 * out_len + out_c))
                } else {
                    _mm256_setzero_ps()
                };

                let mut acc0_lo = bias_g0;
                let mut acc0_hi = _mm256_setzero_ps();
                let mut acc1_lo = bias_g1;
                let mut acc1_hi = _mm256_setzero_ps();
                let mut acc2_lo = bias_g2;
                let mut acc2_hi = _mm256_setzero_ps();
                let mut acc3_lo = bias_g3;
                let mut acc3_hi = _mm256_setzero_ps();

                let mut in_c = 0;
                gemv_4gate_inner_dual_avx2!(
                    in_c,
                    in_len,
                    {
                        let vs_lo = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                        let vs_hi = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));

                        let wp0_lo = w0.as_ptr().add(in_c * out_len + out_c);
                        let vw0_lo = _mm256_loadu_ps(wp0_lo);
                        acc0_lo = _mm256_fmadd_ps(vs_lo, vw0_lo, acc0_lo);
                        let wp0_hi = w0.as_ptr().add((in_c + 1) * out_len + out_c);
                        let vw0_hi = _mm256_loadu_ps(wp0_hi);
                        acc0_hi = _mm256_fmadd_ps(vs_hi, vw0_hi, acc0_hi);

                        let wp1_lo = w1.as_ptr().add(in_c * out_len + out_c);
                        let vw1_lo = _mm256_loadu_ps(wp1_lo);
                        acc1_lo = _mm256_fmadd_ps(vs_lo, vw1_lo, acc1_lo);
                        let wp1_hi = w1.as_ptr().add((in_c + 1) * out_len + out_c);
                        let vw1_hi = _mm256_loadu_ps(wp1_hi);
                        acc1_hi = _mm256_fmadd_ps(vs_hi, vw1_hi, acc1_hi);

                        let wp2_lo = w2.as_ptr().add(in_c * out_len + out_c);
                        let vw2_lo = _mm256_loadu_ps(wp2_lo);
                        acc2_lo = _mm256_fmadd_ps(vs_lo, vw2_lo, acc2_lo);
                        let wp2_hi = w2.as_ptr().add((in_c + 1) * out_len + out_c);
                        let vw2_hi = _mm256_loadu_ps(wp2_hi);
                        acc2_hi = _mm256_fmadd_ps(vs_hi, vw2_hi, acc2_hi);

                        let wp3_lo = w3.as_ptr().add(in_c * out_len + out_c);
                        let vw3_lo = _mm256_loadu_ps(wp3_lo);
                        acc3_lo = _mm256_fmadd_ps(vs_lo, vw3_lo, acc3_lo);
                        let wp3_hi = w3.as_ptr().add((in_c + 1) * out_len + out_c);
                        let vw3_hi = _mm256_loadu_ps(wp3_hi);
                        acc3_hi = _mm256_fmadd_ps(vs_hi, vw3_hi, acc3_hi);
                    },
                    {
                        let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
                        let wp0 = w0.as_ptr().add(in_c * out_len + out_c);
                        let vw0 = _mm256_loadu_ps(wp0);
                        acc0_lo = _mm256_fmadd_ps(vs, vw0, acc0_lo);
                        let wp1 = w1.as_ptr().add(in_c * out_len + out_c);
                        let vw1 = _mm256_loadu_ps(wp1);
                        acc1_lo = _mm256_fmadd_ps(vs, vw1, acc1_lo);
                        let wp2 = w2.as_ptr().add(in_c * out_len + out_c);
                        let vw2 = _mm256_loadu_ps(wp2);
                        acc2_lo = _mm256_fmadd_ps(vs, vw2, acc2_lo);
                        let wp3 = w3.as_ptr().add(in_c * out_len + out_c);
                        let vw3 = _mm256_loadu_ps(wp3);
                        acc3_lo = _mm256_fmadd_ps(vs, vw3, acc3_lo);
                    }
                );

                let acc0 = _mm256_add_ps(acc0_lo, acc0_hi);
                let acc1 = _mm256_add_ps(acc1_lo, acc1_hi);
                let acc2 = _mm256_add_ps(acc2_lo, acc2_hi);
                let acc3 = _mm256_add_ps(acc3_lo, acc3_hi);

                _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_len + out_c), acc1);
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(2 * out_len + out_c), acc2);
                _mm256_storeu_ps(out_frame.as_mut_ptr().add(3 * out_len + out_c), acc3);
            },
            {
                let mut sum0 = if do_bias { bias[out_c] } else { 0.0 };
                let mut sum1 = if do_bias { bias[out_len + out_c] } else { 0.0 };
                let mut sum2 = if do_bias {
                    bias[2 * out_len + out_c]
                } else {
                    0.0
                };
                let mut sum3 = if do_bias {
                    bias[3 * out_len + out_c]
                } else {
                    0.0
                };

                for in_c in 0..in_len {
                    let s = *in_frame.get_unchecked(in_c);
                    sum0 += s * w0.get_unchecked(in_c * out_len + out_c);
                    sum1 += s * w1.get_unchecked(in_c * out_len + out_c);
                    sum2 += s * w2.get_unchecked(in_c * out_len + out_c);
                    sum3 += s * w3.get_unchecked(in_c * out_len + out_c);
                }

                *out_frame.get_unchecked_mut(out_c) = sum0;
                *out_frame.get_unchecked_mut(out_len + out_c) = sum1;
                *out_frame.get_unchecked_mut(2 * out_len + out_c) = sum2;
                *out_frame.get_unchecked_mut(3 * out_len + out_c) = sum3;
            }
        );
    }
}
