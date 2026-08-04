// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_f32_16x_data(len: usize) -> (Vec<[f32; 16]>, Vec<f32>) {
    let weights: Vec<[f32; 16]> = (0..len)
        .map(|i| {
            let base = (i as f32 * 0.1).sin() * 0.5 + 0.5;
            [
                base,
                base * 1.1,
                base * 1.2,
                base * 1.3,
                base * 0.9,
                base * 0.8,
                base * 1.4,
                base * 1.5,
                base * 1.05,
                base * 0.95,
                base * 1.15,
                base * 0.85,
                base * 1.25,
                base * 0.75,
                base * 1.35,
                base * 0.65,
            ]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

#[test]
fn test_dot_16x_f32_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx2(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx2(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        let result = unsafe { dot_product_16x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 5e-4,
                "len={} f0 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 5e-4,
                "len={} f1 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        let result = unsafe { dot_product_16x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 2e-3,
                "stress len={} f0 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 2e-3,
                "stress len={} f1 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx2_single_vs_dual_invariance() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let single_f0 = unsafe { dot_product_16x_f32_avx2(&weights, &state_f0) };
        let single_f1 = unsafe { dot_product_16x_f32_avx2(&weights, &state_f1) };
        let dual = unsafe { dot_product_16x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (dual.0[j] - single_f0[j]).abs() < 5e-6,
                "len={} f0 ch={}: dual={}, single={}",
                len,
                j,
                dual.0[j],
                single_f0[j]
            );
            assert!(
                (dual.1[j] - single_f1[j]).abs() < 5e-6,
                "len={} f1 ch={}: dual={}, single={}",
                len,
                j,
                dual.1[j],
                single_f1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx2_decompose_vs_8x() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let result_16x = unsafe { dot_product_16x_f32_avx2(&weights, &state) };

        let w0: Vec<[f32; 8]> = weights
            .iter()
            .map(|w| [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]])
            .collect();
        let w1: Vec<[f32; 8]> = weights
            .iter()
            .map(|w| [w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]])
            .collect();
        let r0 = unsafe { crate::math::gemm::dot_8x::dot_product_8x_f32_avx2(&w0, &state) };
        let r1 = unsafe { crate::math::gemm::dot_8x::dot_product_8x_f32_avx2(&w1, &state) };
        for j in 0..8 {
            assert!(
                (result_16x[j] - r0[j]).abs() < 5e-4,
                "len={} ch={}: 16x={}, 8x_lo={}",
                len,
                j,
                result_16x[j],
                r0[j]
            );
        }
        for j in 0..8 {
            assert!(
                (result_16x[j + 8] - r1[j]).abs() < 5e-4,
                "len={} ch={}: 16x={}, 8x_hi={}",
                len,
                j + 8,
                result_16x[j + 8],
                r1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx512_vs_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx512(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_f32_16x_data(len);
        let expected = unsafe { scalar_ref::dot_product_16x_f32_scalar(&weights, &state) };
        let result = unsafe { dot_product_16x_f32_avx512(&weights, &state) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

fn make_f32_16x_dual_data(len: usize) -> (Vec<[f32; 16]>, Vec<f32>, Vec<f32>) {
    let (weights, state_f0) = make_f32_16x_data(len);
    let state_f1: Vec<f32> = state_f0.iter().map(|&s| s * 0.8 + 0.1).collect();
    (weights, state_f0, state_f1)
}

#[test]
fn test_dot_16x_f32_dual_avx512_vs_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        let result = unsafe { dot_product_16x_f32_dual_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 5e-4,
                "len={} f0 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 5e-4,
                "len={} f1 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx512_vs_avx2() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let avx2_result = unsafe { dot_product_16x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        let avx512_result =
            unsafe { dot_product_16x_f32_dual_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (avx512_result.0[j] - avx2_result.0[j]).abs() < 5e-4,
                "len={} f0 ch={}: avx512={}, avx2={}",
                len,
                j,
                avx512_result.0[j],
                avx2_result.0[j]
            );
            assert!(
                (avx512_result.1[j] - avx2_result.1[j]).abs() < 5e-4,
                "len={} f1 ch={}: avx512={}, avx2={}",
                len,
                j,
                avx512_result.1[j],
                avx2_result.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx512_single_vs_dual_invariance() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let single_f0 = unsafe { dot_product_16x_f32_avx512(&weights, &state_f0) };
        let single_f1 = unsafe { dot_product_16x_f32_avx512(&weights, &state_f1) };
        let dual = unsafe { dot_product_16x_f32_dual_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (dual.0[j] - single_f0[j]).abs() < 5e-6,
                "len={} f0 ch={}: dual={}, single={}",
                len,
                j,
                dual.0[j],
                single_f0[j]
            );
            assert!(
                (dual.1[j] - single_f1[j]).abs() < 5e-6,
                "len={} f1 ch={}: dual={}, single={}",
                len,
                j,
                dual.1[j],
                single_f1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        let result = unsafe { dot_product_16x_f32_dual_avx512(&weights, &state_f0, &state_f1) };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 2e-3,
                "stress len={} f0 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 2e-3,
                "stress len={} f1 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_avx512_vs_4x_decompose() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let result_16x = unsafe { dot_product_16x_f32_avx512(&weights, &state) };

        let w0: Vec<[f32; 4]> = weights.iter().map(|w| [w[0], w[1], w[2], w[3]]).collect();
        let w1: Vec<[f32; 4]> = weights.iter().map(|w| [w[4], w[5], w[6], w[7]]).collect();
        let w2: Vec<[f32; 4]> = weights.iter().map(|w| [w[8], w[9], w[10], w[11]]).collect();
        let w3: Vec<[f32; 4]> = weights
            .iter()
            .map(|w| [w[12], w[13], w[14], w[15]])
            .collect();
        let r0 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w0, &state) };
        let r1 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w1, &state) };
        let r2 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w2, &state) };
        let r3 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx512(&w3, &state) };
        assert!(
            (result_16x[0] - r0[0]).abs() < 5e-5,
            "len={}: 16x[0]={}, 4x[0]={}",
            len,
            result_16x[0],
            r0[0]
        );
        assert!(
            (result_16x[4] - r1[0]).abs() < 5e-5,
            "len={}: 16x[4]={}, 4x[4]={}",
            len,
            result_16x[4],
            r1[0]
        );
        assert!(
            (result_16x[8] - r2[0]).abs() < 5e-5,
            "len={}: 16x[8]={}, 4x[8]={}",
            len,
            result_16x[8],
            r2[0]
        );
        assert!(
            (result_16x[12] - r3[0]).abs() < 5e-5,
            "len={}: 16x[12]={}, 4x[12]={}",
            len,
            result_16x[12],
            r3[0]
        );
    }
}

// ── Accumulate kernel tests (f32 weights + init) ─────────────────────────────

fn make_f32_init_16() -> [f32; 16] {
    [
        0.1, -0.25, 0.05, 0.4, -0.15, 0.3, -0.05, 0.2, 0.35, -0.1, 0.15, -0.3, 0.25, -0.2, 0.1,
        -0.35,
    ]
}

#[test]
fn test_dot_16x_f32_accumulate_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let init = make_f32_init_16();
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_accumulate_scalar(&weights, &state, &init) };
        let result = unsafe { dot_product_16x_f32_accumulate_avx2(&weights, &state, &init) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_accumulate_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    let init = make_f32_init_16();
    for &len in &lengths {
        let (weights, state) = make_f32_16x_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_accumulate_scalar(&weights, &state, &init) };
        let result = unsafe { dot_product_16x_f32_accumulate_avx2(&weights, &state, &init) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx2={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

fn make_f32_init_dual_16() -> ([f32; 16], [f32; 16]) {
    (
        [
            0.1, -0.25, 0.05, 0.4, -0.15, 0.3, -0.05, 0.2, 0.35, -0.1, 0.15, -0.3, 0.25, -0.2, 0.1,
            -0.35,
        ],
        [
            -0.15, 0.3, -0.05, 0.2, 0.1, -0.25, 0.05, 0.4, -0.35, 0.1, -0.15, 0.3, -0.25, 0.2,
            -0.1, 0.35,
        ],
    )
}

#[test]
fn test_dot_16x_f32_dual_accumulate_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let (init_f0, init_f1) = make_f32_init_dual_16();
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected = unsafe {
            scalar_ref::dot_product_16x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        let result = unsafe {
            dot_product_16x_f32_dual_accumulate_avx2(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 5e-4,
                "len={} f0 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 5e-4,
                "len={} f1 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_accumulate_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    let (init_f0, init_f1) = make_f32_init_dual_16();
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected = unsafe {
            scalar_ref::dot_product_16x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        let result = unsafe {
            dot_product_16x_f32_dual_accumulate_avx2(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 2e-3,
                "stress len={} f0 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 2e-3,
                "stress len={} f1 ch={}: avx2={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

// ── AVX-512 accumulate kernel tests (f32) ─────────────────────────────────

#[test]
fn test_dot_16x_f32_accumulate_avx512_vs_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let init = make_f32_init_16();
    for &len in &sizes {
        let (weights, state) = make_f32_16x_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_accumulate_scalar(&weights, &state, &init) };
        let result = unsafe { dot_product_16x_f32_accumulate_avx512(&weights, &state, &init) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 5e-4,
                "len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_accumulate_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    let init = make_f32_init_16();
    for &len in &lengths {
        let (weights, state) = make_f32_16x_data(len);
        let expected =
            unsafe { scalar_ref::dot_product_16x_f32_accumulate_scalar(&weights, &state, &init) };
        let result = unsafe { dot_product_16x_f32_accumulate_avx512(&weights, &state, &init) };
        for j in 0..16 {
            assert!(
                (result[j] - expected[j]).abs() < 2e-3,
                "stress len={} channel={}: avx512={}, scalar={}",
                len,
                j,
                result[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_accumulate_avx512_vs_scalar() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let (init_f0, init_f1) = make_f32_init_dual_16();
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected = unsafe {
            scalar_ref::dot_product_16x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        let result = unsafe {
            dot_product_16x_f32_dual_accumulate_avx512(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 5e-4,
                "len={} f0 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 5e-4,
                "len={} f1 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}

#[test]
fn test_dot_16x_f32_dual_accumulate_avx512_stress() {
    if !std::is_x86_feature_detected!("avx512f") {
        return;
    }

    let lengths = [1024, 2048, 4096, 8192];
    let (init_f0, init_f1) = make_f32_init_dual_16();
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_16x_dual_data(len);
        let expected = unsafe {
            scalar_ref::dot_product_16x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        let result = unsafe {
            dot_product_16x_f32_dual_accumulate_avx512(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..16 {
            assert!(
                (result.0[j] - expected.0[j]).abs() < 2e-3,
                "stress len={} f0 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.0[j],
                expected.0[j]
            );
            assert!(
                (result.1[j] - expected.1[j]).abs() < 2e-3,
                "stress len={} f1 ch={}: avx512={}, scalar={}",
                len,
                j,
                result.1[j],
                expected.1[j]
            );
        }
    }
}
