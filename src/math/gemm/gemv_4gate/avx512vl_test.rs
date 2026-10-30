// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn scalar_gemv_4gate(
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

    for out_c in 0..out_len {
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
            let s = in_frame[in_c];
            sum0 += s * w0[in_c * out_len + out_c];
            sum1 += s * w1[in_c * out_len + out_c];
            sum2 += s * w2[in_c * out_len + out_c];
            sum3 += s * w3[in_c * out_len + out_c];
        }

        out_frame[out_c] = sum0;
        out_frame[out_len + out_c] = sum1;
        out_frame[2 * out_len + out_c] = sum2;
        out_frame[3 * out_len + out_c] = sum3;
    }
}

struct TestData {
    in_frame: Vec<f32>,
    w0: Vec<f32>,
    w1: Vec<f32>,
    w2: Vec<f32>,
    w3: Vec<f32>,
    bias: Vec<f32>,
}

fn make_test_data(in_len: usize, out_len: usize) -> TestData {
    let in_frame: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.17).sin() * 0.8).collect();
    let w0: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.07).sin() * 0.5)
        .collect();
    let w1: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.11).cos() * 0.5)
        .collect();
    let w2: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let w3: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.19).cos() * 0.5)
        .collect();
    let bias: Vec<f32> = (0..4 * out_len)
        .map(|i| (i as f32 * 0.23).sin() * 0.1)
        .collect();
    TestData {
        in_frame,
        w0,
        w1,
        w2,
        w3,
        bias,
    }
}

#[test]
fn test_gemv_4gate_avx512vl_parity() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let test_shapes = [
        // Standard LSTM hidden sizes
        (1, 16),
        (17, 16),
        (18, 16),
        (1, 24),
        (25, 24),
        (1, 32),
        (33, 32),
        (1, 48),
        (49, 48),
        (1, 64),
        (65, 64),
        // Odd shapes and tails
        (3, 8),
        (5, 7),
        (7, 11),
        (9, 15),
        (13, 1),
        (2, 4),
    ];

    for &(in_len, out_len) in &test_shapes {
        for &do_bias in &[true, false] {
            let data = make_test_data(in_len, out_len);
            let mut out_vl = vec![0.0f32; 4 * out_len];
            let mut out_ref = vec![0.0f32; 4 * out_len];
            let mut out_avx2 = vec![0.0f32; 4 * out_len];

            // SAFETY: `make_test_data` sizes every buffer to the kernel contract for the
            // current `(in_len, out_len)`: `in_frame` has `in_len` elements, `w0..w3`
            // have `in_len*out_len`, and `bias`/`out_vl`/`out_avx2` have `4*out_len`;
            // AVX-512F+VL availability is runtime-checked above.
            unsafe {
                gemv_4gate_avx512vl(
                    &data.in_frame,
                    &data.w0,
                    &data.w1,
                    &data.w2,
                    &data.w3,
                    &data.bias,
                    &mut out_vl,
                    do_bias,
                );
                crate::math::gemm::gemv_4gate_avx2(
                    &data.in_frame,
                    &data.w0,
                    &data.w1,
                    &data.w2,
                    &data.w3,
                    &data.bias,
                    &mut out_avx2,
                    do_bias,
                );
            }
            scalar_gemv_4gate(
                &data.in_frame,
                &data.w0,
                &data.w1,
                &data.w2,
                &data.w3,
                &data.bias,
                &mut out_ref,
                do_bias,
            );

            // Compute ESR and max absolute difference
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            let mut max_diff = 0.0f32;

            for i in 0..4 * out_len {
                let diff = (out_vl[i] - out_ref[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                num += (out_vl[i] as f64 - out_ref[i] as f64).powi(2);
                den += (out_ref[i] as f64).powi(2);
            }

            let esr = if den > 1e-12 { num / den } else { num };

            // Measured ESR is near machine precision (< 1e-10)
            assert!(
                max_diff < 1e-4,
                "in_len={} out_len={} do_bias={}: max_diff={} exceeds threshold",
                in_len,
                out_len,
                do_bias,
                max_diff
            );
            assert!(
                esr < 1e-8,
                "in_len={} out_len={} do_bias={}: ESR={} exceeds LSTM envelope",
                in_len,
                out_len,
                do_bias,
                esr
            );
        }
    }
}
