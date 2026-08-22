// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_gemv_data(in_len: usize, out_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.07).sin()).collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.25)
        .collect();
    let bias: Vec<f32> = (0..out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.1)
        .collect();
    (in_frames, weights, bias)
}

const GEMV_OUT_LENS: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const GEMV_IN_LENS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 48, 64, 96, 128];

// ── AVX2 with_bias ──────────────────────────────────────────────────────────

#[test]
fn test_gemv_with_bias_f32_avx2_vs_fallback() {
    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, bias) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_with_bias_f32_avx2(&in_frames, &weights, &bias, &mut out_simd, num_frames);
                scalar_ref::gemv_with_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &bias,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx2={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_with_bias_f32_avx2_batch_vs_fallback() {
    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, bias) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_with_bias_f32_avx2(&in_frames, &weights, &bias, &mut out_simd, num_frames);
            scalar_ref::gemv_with_bias_f32_fallback(
                &in_frames,
                &weights,
                &bias,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx2={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX2 no_bias ────────────────────────────────────────────────────────────

#[test]
fn test_gemv_no_bias_f32_avx2_vs_fallback() {
    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, _) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_simd, num_frames);
                scalar_ref::gemv_no_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx2={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
fn test_gemv_no_bias_f32_avx2_batch_vs_fallback() {
    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, _) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_no_bias_f32_avx2(&in_frames, &weights, &mut out_simd, num_frames);
            scalar_ref::gemv_no_bias_f32_fallback(
                &in_frames,
                &weights,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx2={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX-512 with_bias ───────────────────────────────────────────────────────

#[test]
#[cfg(feature = "avx512")]
fn test_gemv_with_bias_f32_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, bias) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_with_bias_f32_avx512(&in_frames, &weights, &bias, &mut out_simd, num_frames);
                scalar_ref::gemv_with_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &bias,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx512={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "avx512")]
fn test_gemv_with_bias_f32_avx512_batch_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, bias) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_with_bias_f32_avx512(&in_frames, &weights, &bias, &mut out_simd, num_frames);
            scalar_ref::gemv_with_bias_f32_fallback(
                &in_frames,
                &weights,
                &bias,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx512={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── AVX-512 no_bias ─────────────────────────────────────────────────────────

#[test]
#[cfg(feature = "avx512")]
fn test_gemv_no_bias_f32_avx512_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let num_frames = 1;
    for &out_len in GEMV_OUT_LENS {
        for &in_len in GEMV_IN_LENS {
            let (in_frames, weights, _) = make_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len * num_frames];
            let mut out_scalar = vec![0.0f32; out_len * num_frames];

            unsafe {
                gemv_no_bias_f32_avx512(&in_frames, &weights, &mut out_simd, num_frames);
                scalar_ref::gemv_no_bias_f32_fallback(
                    &in_frames,
                    &weights,
                    &mut out_scalar,
                    num_frames,
                );
            }
            for c in 0..out_len {
                assert!(
                    (out_simd[c] - out_scalar[c]).abs() < 5e-4,
                    "in_len={} out_len={} ch={}: avx512={}, scalar={}",
                    in_len,
                    out_len,
                    c,
                    out_simd[c],
                    out_scalar[c]
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "avx512")]
fn test_gemv_no_bias_f32_avx512_batch_vs_fallback() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    for &num_frames in &[1, 2, 3, 4, 8] {
        let out_len = 8;
        let in_len = 64;
        let (in_frames, weights, _) = make_gemv_data(in_len * num_frames, out_len);
        let mut out_simd = vec![0.0f32; out_len * num_frames];
        let mut out_scalar = vec![0.0f32; out_len * num_frames];

        unsafe {
            gemv_no_bias_f32_avx512(&in_frames, &weights, &mut out_simd, num_frames);
            scalar_ref::gemv_no_bias_f32_fallback(
                &in_frames,
                &weights,
                &mut out_scalar,
                num_frames,
            );
        }
        for f in 0..num_frames {
            for c in 0..out_len {
                assert!(
                    (out_simd[f * out_len + c] - out_scalar[f * out_len + c]).abs() < 5e-4,
                    "batch frames={} f={} ch={}: avx512={}, scalar={}",
                    num_frames,
                    f,
                    c,
                    out_simd[f * out_len + c],
                    out_scalar[f * out_len + c]
                );
            }
        }
    }
}

// ── f32 Specialized GEMV kernels ──────────────────────────────────────────

type F32Kernel = unsafe fn(&[f32], &[f32], &[f32], &mut [f32], bool);

const FUSED_ADD_SPECIALIZED: &[(usize, usize, F32Kernel)] = &[
    (1, 4, f16_avx2_fused::fused_add_gemv_avx2_1x4 as F32Kernel),
    (4, 4, f16_avx2_fused::fused_add_gemv_avx2_4x4 as F32Kernel),
    (4, 6, f16_avx2_fused::fused_add_gemv_avx2_4x6 as F32Kernel),
    (8, 4, f16_avx2_fused::fused_add_gemv_avx2_8x4 as F32Kernel),
    (8, 6, f16_avx2_fused::fused_add_gemv_avx2_8x6 as F32Kernel),
    (8, 8, f16_avx2_fused::fused_add_gemv_avx2_8x8 as F32Kernel),
];

const OVERWRITE_SPECIALIZED: &[(usize, usize, F32Kernel)] = &[
    (
        1,
        4,
        f16_avx2_overwrite::gemv_overwrite_avx2_1x4 as F32Kernel,
    ),
    (
        4,
        4,
        f16_avx2_overwrite::gemv_overwrite_avx2_4x4 as F32Kernel,
    ),
    (
        4,
        6,
        f16_avx2_overwrite::gemv_overwrite_avx2_4x6 as F32Kernel,
    ),
    (
        8,
        4,
        f16_avx2_overwrite::gemv_overwrite_avx2_8x4 as F32Kernel,
    ),
    (
        8,
        6,
        f16_avx2_overwrite::gemv_overwrite_avx2_8x6 as F32Kernel,
    ),
    (
        8,
        8,
        f16_avx2_overwrite::gemv_overwrite_avx2_8x8 as F32Kernel,
    ),
];

fn make_f32_gemv_data(in_len: usize, out_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.07).sin()).collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.1).sin() * 0.5 + 0.25)
        .collect();
    let bias: Vec<f32> = (0..out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.1)
        .collect();
    (in_frames, weights, bias)
}

unsafe fn fused_add_gemv_f32_ref(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for out_c in 0..out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            sum += in_frame[in_c] * weights[in_c * out_len + out_c];
        }
        out_frame[out_c] += sum;
    }
}

unsafe fn gemv_overwrite_f32_ref(
    in_frame: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_frame: &mut [f32],
    do_bias: bool,
) {
    let out_len = out_frame.len();
    let in_len = in_frame.len();
    for out_c in 0..out_len {
        let mut sum = if do_bias { bias[out_c] } else { 0.0 };
        for in_c in 0..in_len {
            sum += in_frame[in_c] * weights[in_c * out_len + out_c];
        }
        out_frame[out_c] = sum;
    }
}

// ── fused_add_gemv specialized vs fallback ────────────────────────────────

#[test]
fn test_fused_add_gemv_f32_specialized_vs_fallback() {
    for &(in_len, out_len, kernel) in FUSED_ADD_SPECIALIZED {
        for &do_bias in &[true, false] {
            let (in_frame, weights, bias) = make_f32_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len];
            let mut out_fb = vec![0.0f32; out_len];

            unsafe {
                kernel(&in_frame, &weights, &bias, &mut out_simd, do_bias);
                fused_add_gemv_f32_ref(&in_frame, &weights, &bias, &mut out_fb, do_bias);
            }
            for c in 0..out_len {
                let diff = (out_simd[c] - out_fb[c]).abs();
                assert!(
                    diff < 5e-4,
                    "fused_add {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}, diff={diff:e}",
                    out_simd[c],
                    out_fb[c],
                );
            }
        }
    }
}

// ── gemv_overwrite specialized vs fallback ────────────────────────────────

#[test]
fn test_gemv_overwrite_f32_specialized_vs_fallback() {
    for &(in_len, out_len, kernel) in OVERWRITE_SPECIALIZED {
        for &do_bias in &[true, false] {
            let (in_frame, weights, bias) = make_f32_gemv_data(in_len, out_len);
            let mut out_simd = vec![0.0f32; out_len];
            let mut out_fb = vec![0.0f32; out_len];

            unsafe {
                kernel(&in_frame, &weights, &bias, &mut out_simd, do_bias);
                gemv_overwrite_f32_ref(&in_frame, &weights, &bias, &mut out_fb, do_bias);
            }
            for c in 0..out_len {
                let diff = (out_simd[c] - out_fb[c]).abs();
                assert!(
                    diff < 5e-4,
                    "overwrite {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}, diff={diff:e}",
                    out_simd[c],
                    out_fb[c],
                );
            }
        }
    }
}

// ── Boundary conditions ───────────────────────────────────────────────────

#[test]
fn test_f32_specialized_denormal_f32_inputs() {
    let denormal_inputs: &[f32] = &[
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x000F_FFFF),
        f32::from_bits(0x007F_FFFF),
        -f32::from_bits(0x0000_0001),
        -f32::from_bits(0x007F_FFFF),
    ];

    for &(in_len, out_len, kernel) in FUSED_ADD_SPECIALIZED {
        for &d in denormal_inputs {
            let in_frames = vec![d; in_len];
            let weights: Vec<f32> = vec![0.5; in_len * out_len];
            let bias: Vec<f32> = vec![0.001; out_len];

            for &do_bias in &[true, false] {
                let mut out_simd = vec![0.0; out_len];
                let mut out_fb = vec![0.0; out_len];

                unsafe {
                    kernel(&in_frames, &weights, &bias, &mut out_simd, do_bias);
                    fused_add_gemv_f32_ref(&in_frames, &weights, &bias, &mut out_fb, do_bias);
                }
                for c in 0..out_len {
                    let diff = (out_simd[c] - out_fb[c]).abs();
                    assert!(
                        diff < 5e-4,
                        "denormal in={d:e} {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}, diff={diff:e}",
                        out_simd[c],
                        out_fb[c],
                    );
                }
            }
        }
    }
}

#[test]
fn test_f32_specialized_all_zeros() {
    for &(in_len, out_len, kernel) in FUSED_ADD_SPECIALIZED {
        let in_frames = vec![0.0f32; in_len];
        let weights: Vec<f32> = vec![0.0f32; in_len * out_len];
        let bias: Vec<f32> = vec![0.0; out_len];

        for &do_bias in &[true, false] {
            let mut out_simd = vec![1.0f32; out_len];
            let mut out_fb = vec![1.0f32; out_len];

            unsafe {
                kernel(&in_frames, &weights, &bias, &mut out_simd, do_bias);
                fused_add_gemv_f32_ref(&in_frames, &weights, &bias, &mut out_fb, do_bias);
            }
            for c in 0..out_len {
                let diff = (out_simd[c] - out_fb[c]).abs();
                assert!(
                    diff < 5e-4,
                    "zeros fused_add {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}",
                    out_simd[c],
                    out_fb[c],
                );
            }
        }
    }

    for &(in_len, out_len, kernel) in OVERWRITE_SPECIALIZED {
        let in_frames = vec![0.0f32; in_len];
        let weights: Vec<f32> = vec![0.0f32; in_len * out_len];
        let bias: Vec<f32> = vec![0.0; out_len];

        for &do_bias in &[true, false] {
            let mut out_simd = vec![1.0f32; out_len];
            let mut out_fb = vec![1.0f32; out_len];

            unsafe {
                kernel(&in_frames, &weights, &bias, &mut out_simd, do_bias);
                gemv_overwrite_f32_ref(&in_frames, &weights, &bias, &mut out_fb, do_bias);
            }
            for c in 0..out_len {
                let diff = (out_simd[c] - out_fb[c]).abs();
                assert!(
                    diff < 5e-4,
                    "zeros overwrite {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}",
                    out_simd[c],
                    out_fb[c],
                );
            }
        }
    }
}

#[test]
fn test_f32_specialized_large_values() {
    let large: f32 = 1e25;
    for &(in_len, out_len, kernel) in FUSED_ADD_SPECIALIZED {
        let in_frames = vec![large; in_len];
        let weights: Vec<f32> = vec![0.5; in_len * out_len];
        let bias: Vec<f32> = vec![large; out_len];

        for &do_bias in &[true, false] {
            let mut out_simd = vec![large; out_len];
            let mut out_fb = vec![large; out_len];

            unsafe {
                kernel(&in_frames, &weights, &bias, &mut out_simd, do_bias);
                fused_add_gemv_f32_ref(&in_frames, &weights, &bias, &mut out_fb, do_bias);
            }
            for c in 0..out_len {
                let diff = (out_simd[c] - out_fb[c]).abs();
                let max_val = out_fb[c].abs().max(1.0);
                assert!(
                    diff / max_val < 5e-4,
                    "large fused_add {in_len}x{out_len} bias={do_bias} ch={c}: simd={}, fb={}, rel_diff={:e}",
                    out_simd[c],
                    out_fb[c],
                    diff / max_val,
                );
            }
        }
    }
}
