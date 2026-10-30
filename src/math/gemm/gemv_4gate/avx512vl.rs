// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// Specialized GEMM micro-kernels utilize unsafe SIMD intrinsics and parameter lists.
#![allow(unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]

use core::arch::x86_64::*;

/// Performs the linear projection for the 4 gates of an LSTM cell simultaneously via AVX-512 VL256.
///
/// In an LSTM neural network, each step requires computing 4 sub-results (gates: input, forget,
/// candidate, and output). This kernel computes all 4 gates simultaneously using 256-bit EVEX
/// vector registers (`__m256`).
///
/// # Optimization (EVEX VL256 4-Column Unrolling)
/// With the 32 vector registers (`ymm0`..`ymm31`) available under `avx512vl`, this kernel
/// unrolls 4 input columns per iteration with 16 independent FMA accumulators (4 per gate:
/// `acc_0`, `acc_1`, `acc_2`, `acc_3`), fully breaking the serial FMA latency dependency chain
/// and saturating execution ports without spilling accumulators to the stack.
///
/// # Safety
/// With `out_len = out_frame.len() / 4` and `in_len = in_frame.len()`:
/// - `out_frame.len()` must be a multiple of 4 (gate-major layout: gate `g`
///   occupies `out_frame[g * out_len .. (g + 1) * out_len]`; a non-multiple
///   silently truncates the last lanes of the scalar tail).
/// - Each `w{i}` must hold a row-major `[in_len][out_len]` matrix, i.e.
///   `w{i}.len() >= in_len * out_len` — the SIMD loop reads 8 contiguous f32
///   per gate via unchecked `add(in_c * out_len + out_c)` (guarded by
///   `out_c + 8 <= out_len`).
/// - `bias.len() >= 4 * out_len` (one lane per output element).
/// - All slices must be valid for the full extents above for the duration of
///   the call (no `out_frame` element may overlap any input slice —
///   guaranteed by the borrow checker for `&[f32]`/`&mut [f32]` arguments).
/// - The CPU must support AVX-512F + AVX-512VL: the `#[target_feature]`
///   attribute does not check anything at runtime; callers must dispatch
///   through the runtime CPUID (`InstructionSet` detection) before calling.
#[target_feature(enable = "avx512f,avx512vl")]
#[expect(
    clippy::too_many_arguments,
    reason = "Performance-critical AVX-512 VL256 LSTM 4-gate kernel requiring many matrix strides/dimensions for maximum SIMD throughput"
)]
pub unsafe fn gemv_4gate_avx512vl(
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

    let mut out_c = 0;
    while out_c + 8 <= out_len {
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

        let mut acc0_0 = bias_g0;
        let mut acc0_1 = _mm256_setzero_ps();
        let mut acc0_2 = _mm256_setzero_ps();
        let mut acc0_3 = _mm256_setzero_ps();

        let mut acc1_0 = bias_g1;
        let mut acc1_1 = _mm256_setzero_ps();
        let mut acc1_2 = _mm256_setzero_ps();
        let mut acc1_3 = _mm256_setzero_ps();

        let mut acc2_0 = bias_g2;
        let mut acc2_1 = _mm256_setzero_ps();
        let mut acc2_2 = _mm256_setzero_ps();
        let mut acc2_3 = _mm256_setzero_ps();

        let mut acc3_0 = bias_g3;
        let mut acc3_1 = _mm256_setzero_ps();
        let mut acc3_2 = _mm256_setzero_ps();
        let mut acc3_3 = _mm256_setzero_ps();

        let mut in_c = 0;
        while in_c + 4 <= in_len {
            let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));
            let vs2 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 2));
            let vs3 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 3));

            // Gate 0
            let wp0_0 = w0.as_ptr().add(in_c * out_len + out_c);
            let vw0_0 = _mm256_loadu_ps(wp0_0);
            acc0_0 = _mm256_fmadd_ps(vs0, vw0_0, acc0_0);
            let wp0_1 = w0.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw0_1 = _mm256_loadu_ps(wp0_1);
            acc0_1 = _mm256_fmadd_ps(vs1, vw0_1, acc0_1);
            let wp0_2 = w0.as_ptr().add((in_c + 2) * out_len + out_c);
            let vw0_2 = _mm256_loadu_ps(wp0_2);
            acc0_2 = _mm256_fmadd_ps(vs2, vw0_2, acc0_2);
            let wp0_3 = w0.as_ptr().add((in_c + 3) * out_len + out_c);
            let vw0_3 = _mm256_loadu_ps(wp0_3);
            acc0_3 = _mm256_fmadd_ps(vs3, vw0_3, acc0_3);

            // Gate 1
            let wp1_0 = w1.as_ptr().add(in_c * out_len + out_c);
            let vw1_0 = _mm256_loadu_ps(wp1_0);
            acc1_0 = _mm256_fmadd_ps(vs0, vw1_0, acc1_0);
            let wp1_1 = w1.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw1_1 = _mm256_loadu_ps(wp1_1);
            acc1_1 = _mm256_fmadd_ps(vs1, vw1_1, acc1_1);
            let wp1_2 = w1.as_ptr().add((in_c + 2) * out_len + out_c);
            let vw1_2 = _mm256_loadu_ps(wp1_2);
            acc1_2 = _mm256_fmadd_ps(vs2, vw1_2, acc1_2);
            let wp1_3 = w1.as_ptr().add((in_c + 3) * out_len + out_c);
            let vw1_3 = _mm256_loadu_ps(wp1_3);
            acc1_3 = _mm256_fmadd_ps(vs3, vw1_3, acc1_3);

            // Gate 2
            let wp2_0 = w2.as_ptr().add(in_c * out_len + out_c);
            let vw2_0 = _mm256_loadu_ps(wp2_0);
            acc2_0 = _mm256_fmadd_ps(vs0, vw2_0, acc2_0);
            let wp2_1 = w2.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw2_1 = _mm256_loadu_ps(wp2_1);
            acc2_1 = _mm256_fmadd_ps(vs1, vw2_1, acc2_1);
            let wp2_2 = w2.as_ptr().add((in_c + 2) * out_len + out_c);
            let vw2_2 = _mm256_loadu_ps(wp2_2);
            acc2_2 = _mm256_fmadd_ps(vs2, vw2_2, acc2_2);
            let wp2_3 = w2.as_ptr().add((in_c + 3) * out_len + out_c);
            let vw2_3 = _mm256_loadu_ps(wp2_3);
            acc2_3 = _mm256_fmadd_ps(vs3, vw2_3, acc2_3);

            // Gate 3
            let wp3_0 = w3.as_ptr().add(in_c * out_len + out_c);
            let vw3_0 = _mm256_loadu_ps(wp3_0);
            acc3_0 = _mm256_fmadd_ps(vs0, vw3_0, acc3_0);
            let wp3_1 = w3.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw3_1 = _mm256_loadu_ps(wp3_1);
            acc3_1 = _mm256_fmadd_ps(vs1, vw3_1, acc3_1);
            let wp3_2 = w3.as_ptr().add((in_c + 2) * out_len + out_c);
            let vw3_2 = _mm256_loadu_ps(wp3_2);
            acc3_2 = _mm256_fmadd_ps(vs2, vw3_2, acc3_2);
            let wp3_3 = w3.as_ptr().add((in_c + 3) * out_len + out_c);
            let vw3_3 = _mm256_loadu_ps(wp3_3);
            acc3_3 = _mm256_fmadd_ps(vs3, vw3_3, acc3_3);

            in_c += 4;
        }

        if in_c + 2 <= in_len {
            let vs0 = _mm256_set1_ps(*in_frame.get_unchecked(in_c));
            let vs1 = _mm256_set1_ps(*in_frame.get_unchecked(in_c + 1));

            let wp0_0 = w0.as_ptr().add(in_c * out_len + out_c);
            let vw0_0 = _mm256_loadu_ps(wp0_0);
            acc0_0 = _mm256_fmadd_ps(vs0, vw0_0, acc0_0);
            let wp0_1 = w0.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw0_1 = _mm256_loadu_ps(wp0_1);
            acc0_1 = _mm256_fmadd_ps(vs1, vw0_1, acc0_1);

            let wp1_0 = w1.as_ptr().add(in_c * out_len + out_c);
            let vw1_0 = _mm256_loadu_ps(wp1_0);
            acc1_0 = _mm256_fmadd_ps(vs0, vw1_0, acc1_0);
            let wp1_1 = w1.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw1_1 = _mm256_loadu_ps(wp1_1);
            acc1_1 = _mm256_fmadd_ps(vs1, vw1_1, acc1_1);

            let wp2_0 = w2.as_ptr().add(in_c * out_len + out_c);
            let vw2_0 = _mm256_loadu_ps(wp2_0);
            acc2_0 = _mm256_fmadd_ps(vs0, vw2_0, acc2_0);
            let wp2_1 = w2.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw2_1 = _mm256_loadu_ps(wp2_1);
            acc2_1 = _mm256_fmadd_ps(vs1, vw2_1, acc2_1);

            let wp3_0 = w3.as_ptr().add(in_c * out_len + out_c);
            let vw3_0 = _mm256_loadu_ps(wp3_0);
            acc3_0 = _mm256_fmadd_ps(vs0, vw3_0, acc3_0);
            let wp3_1 = w3.as_ptr().add((in_c + 1) * out_len + out_c);
            let vw3_1 = _mm256_loadu_ps(wp3_1);
            acc3_1 = _mm256_fmadd_ps(vs1, vw3_1, acc3_1);

            in_c += 2;
        }

        if in_c < in_len {
            let vs = _mm256_set1_ps(*in_frame.get_unchecked(in_c));

            let wp0 = w0.as_ptr().add(in_c * out_len + out_c);
            let vw0 = _mm256_loadu_ps(wp0);
            acc0_0 = _mm256_fmadd_ps(vs, vw0, acc0_0);

            let wp1 = w1.as_ptr().add(in_c * out_len + out_c);
            let vw1 = _mm256_loadu_ps(wp1);
            acc1_0 = _mm256_fmadd_ps(vs, vw1, acc1_0);

            let wp2 = w2.as_ptr().add(in_c * out_len + out_c);
            let vw2 = _mm256_loadu_ps(wp2);
            acc2_0 = _mm256_fmadd_ps(vs, vw2, acc2_0);

            let wp3 = w3.as_ptr().add(in_c * out_len + out_c);
            let vw3 = _mm256_loadu_ps(wp3);
            acc3_0 = _mm256_fmadd_ps(vs, vw3, acc3_0);
        }

        let acc0_a = _mm256_add_ps(acc0_0, acc0_1);
        let acc0_b = _mm256_add_ps(acc0_2, acc0_3);
        let acc0 = _mm256_add_ps(acc0_a, acc0_b);

        let acc1_a = _mm256_add_ps(acc1_0, acc1_1);
        let acc1_b = _mm256_add_ps(acc1_2, acc1_3);
        let acc1 = _mm256_add_ps(acc1_a, acc1_b);

        let acc2_a = _mm256_add_ps(acc2_0, acc2_1);
        let acc2_b = _mm256_add_ps(acc2_2, acc2_3);
        let acc2 = _mm256_add_ps(acc2_a, acc2_b);

        let acc3_a = _mm256_add_ps(acc3_0, acc3_1);
        let acc3_b = _mm256_add_ps(acc3_2, acc3_3);
        let acc3 = _mm256_add_ps(acc3_a, acc3_b);

        _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_c), acc0);
        _mm256_storeu_ps(out_frame.as_mut_ptr().add(out_len + out_c), acc1);
        _mm256_storeu_ps(out_frame.as_mut_ptr().add(2 * out_len + out_c), acc2);
        _mm256_storeu_ps(out_frame.as_mut_ptr().add(3 * out_len + out_c), acc3);

        out_c += 8;
    }

    while out_c < out_len {
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
        out_c += 1;
    }
}

#[cfg(test)]
#[path = "avx512vl_test.rs"]
mod tests;
