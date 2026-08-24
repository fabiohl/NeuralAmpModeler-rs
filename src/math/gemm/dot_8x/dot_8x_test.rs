// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::scalar_ref;

fn make_f32_8x_data(len: usize) -> (Vec<[f32; 8]>, Vec<f32>) {
    let weights: Vec<[f32; 8]> = (0..len)
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
            ]
        })
        .collect();
    let state: Vec<f32> = (0..len).map(|i| (i as f32 * 0.07).sin()).collect();
    (weights, state)
}

#[test]
fn test_dot_8x_f32_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_8x_data(len);
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let expected = unsafe { scalar_ref::dot_product_8x_f32_scalar(&weights, &state) };
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_avx2(&weights, &state) };
        for j in 0..8 {
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
fn test_dot_8x_f32_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state) = make_f32_8x_data(len);
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let expected = unsafe { scalar_ref::dot_product_8x_f32_scalar(&weights, &state) };
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_avx2(&weights, &state) };
        for j in 0..8 {
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
fn test_dot_8x_f32_avx2_decompose_vs_4x() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state) = make_f32_8x_data(len);
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let result_8x = unsafe { dot_product_8x_f32_avx2(&weights, &state) };

        let w0: Vec<[f32; 4]> = weights.iter().map(|w| [w[0], w[1], w[2], w[3]]).collect();
        let w1: Vec<[f32; 4]> = weights.iter().map(|w| [w[4], w[5], w[6], w[7]]).collect();
        // SAFETY: `w0` was derived from `weights` and has the same length `len` as `state`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let r0 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx2(&w0, &state) };
        // SAFETY: `w1` was derived from `weights` and has the same length `len` as `state`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let r1 = unsafe { crate::math::gemm::dot_4x::dot_product_4x_f32_avx2(&w1, &state) };
        for j in 0..4 {
            assert!(
                (result_8x[j] - r0[j]).abs() < 5e-4,
                "len={} ch={}: 8x={}, 4x_lo={}",
                len,
                j,
                result_8x[j],
                r0[j]
            );
        }
        for j in 0..4 {
            assert!(
                (result_8x[j + 4] - r1[j]).abs() < 5e-4,
                "len={} ch={}: 8x={}, 4x_hi={}",
                len,
                j + 4,
                result_8x[j + 4],
                r1[j]
            );
        }
    }
}

fn make_f32_8x_dual_data(len: usize) -> (Vec<[f32; 8]>, Vec<f32>, Vec<f32>) {
    let (weights, state_f0) = make_f32_8x_data(len);
    let state_f1: Vec<f32> = state_f0.iter().map(|&s| s * 0.8 + 0.1).collect();
    (weights, state_f0, state_f1)
}

#[test]
fn test_dot_8x_f32_dual_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_8x_dual_data(len);
        let expected =
            // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with
            // equal length `len`, satisfying `weights.len() >= state_f0.len()` and
            // `weights.len() >= state_f1.len()`; all buffers outlive the call.
            unsafe { scalar_ref::dot_product_8x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with
        // equal length `len`, satisfying `weights.len() >= state_f0.len()` and
        // `weights.len() >= state_f1.len()`; all buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..8 {
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
fn test_dot_8x_f32_dual_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_8x_dual_data(len);
        let expected =
            // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with
            // equal length `len`, satisfying `weights.len() >= state_f0.len()` and
            // `weights.len() >= state_f1.len()`; all buffers outlive the call.
            unsafe { scalar_ref::dot_product_8x_f32_dual_scalar(&weights, &state_f0, &state_f1) };
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with
        // equal length `len`, satisfying `weights.len() >= state_f0.len()` and
        // `weights.len() >= state_f1.len()`; all buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..8 {
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
fn test_dot_8x_f32_dual_avx2_single_vs_dual_invariance() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_8x_dual_data(len);
        // SAFETY: `weights` and `state_f0` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let single_f0 = unsafe { dot_product_8x_f32_avx2(&weights, &state_f0) };
        // SAFETY: `weights` and `state_f1` were allocated in this test with equal length `len`,
        // satisfying the kernel contract `weights.len() >= state.len()`; both buffers outlive the call.
        let single_f1 = unsafe { dot_product_8x_f32_avx2(&weights, &state_f1) };
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with
        // equal length `len`, satisfying `weights.len() >= state_f0.len()` and
        // `weights.len() >= state_f1.len()`; all buffers outlive the call.
        let dual = unsafe { dot_product_8x_f32_dual_avx2(&weights, &state_f0, &state_f1) };
        for j in 0..8 {
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

// ── Accumulate kernel tests (f32 weights + init) ─────────────────────────────

fn make_f32_init_8() -> [f32; 8] {
    [0.1, -0.25, 0.05, 0.4, -0.15, 0.3, -0.05, 0.2]
}

#[test]
fn test_dot_8x_f32_accumulate_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let init = make_f32_init_8();
    for &len in &sizes {
        let (weights, state) = make_f32_8x_data(len);
        let expected =
            // SAFETY: `weights` and `state` were allocated in this test with equal length `len`
            // and `init` is a fixed `[f32; 8]`, satisfying `weights.len() >= state.len()`;
            // all buffers outlive the call.
            unsafe { scalar_ref::dot_product_8x_f32_accumulate_scalar(&weights, &state, &init) };
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`
        // and `init` is a fixed `[f32; 8]`, satisfying `weights.len() >= state.len()`;
        // all buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_accumulate_avx2(&weights, &state, &init) };
        for j in 0..8 {
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
fn test_dot_8x_f32_accumulate_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    let init = make_f32_init_8();
    for &len in &lengths {
        let (weights, state) = make_f32_8x_data(len);
        let expected =
            // SAFETY: `weights` and `state` were allocated in this test with equal length `len`
            // and `init` is a fixed `[f32; 8]`, satisfying `weights.len() >= state.len()`;
            // all buffers outlive the call.
            unsafe { scalar_ref::dot_product_8x_f32_accumulate_scalar(&weights, &state, &init) };
        // SAFETY: `weights` and `state` were allocated in this test with equal length `len`
        // and `init` is a fixed `[f32; 8]`, satisfying `weights.len() >= state.len()`;
        // all buffers outlive the call.
        let result = unsafe { dot_product_8x_f32_accumulate_avx2(&weights, &state, &init) };
        for j in 0..8 {
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

fn make_f32_init_dual_8() -> ([f32; 8], [f32; 8]) {
    (
        [0.1, -0.25, 0.05, 0.4, -0.15, 0.3, -0.05, 0.2],
        [-0.15, 0.3, -0.05, 0.2, 0.1, -0.25, 0.05, 0.4],
    )
}

#[test]
fn test_dot_8x_f32_dual_accumulate_avx2_vs_scalar() {
    let sizes = [
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 127, 128, 255, 256, 512,
    ];
    let (init_f0, init_f1) = make_f32_init_dual_8();
    for &len in &sizes {
        let (weights, state_f0, state_f1) = make_f32_8x_dual_data(len);
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with equal
        // length `len` and `init_f0`/`init_f1` are fixed `[f32; 8]`, satisfying
        // `weights.len() >= state_f0.len()` and `weights.len() >= state_f1.len()`;
        // all buffers outlive the call.
        let expected = unsafe {
            scalar_ref::dot_product_8x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with equal
        // length `len` and `init_f0`/`init_f1` are fixed `[f32; 8]`, satisfying
        // `weights.len() >= state_f0.len()` and `weights.len() >= state_f1.len()`;
        // all buffers outlive the call.
        let result = unsafe {
            dot_product_8x_f32_dual_accumulate_avx2(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..8 {
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
fn test_dot_8x_f32_dual_accumulate_avx2_stress() {
    let lengths = [1024, 2048, 4096, 8192];
    let (init_f0, init_f1) = make_f32_init_dual_8();
    for &len in &lengths {
        let (weights, state_f0, state_f1) = make_f32_8x_dual_data(len);
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with equal
        // length `len` and `init_f0`/`init_f1` are fixed `[f32; 8]`, satisfying
        // `weights.len() >= state_f0.len()` and `weights.len() >= state_f1.len()`;
        // all buffers outlive the call.
        let expected = unsafe {
            scalar_ref::dot_product_8x_f32_dual_accumulate_scalar(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        // SAFETY: `weights`, `state_f0`, and `state_f1` were allocated in this test with equal
        // length `len` and `init_f0`/`init_f1` are fixed `[f32; 8]`, satisfying
        // `weights.len() >= state_f0.len()` and `weights.len() >= state_f1.len()`;
        // all buffers outlive the call.
        let result = unsafe {
            dot_product_8x_f32_dual_accumulate_avx2(
                &weights, &state_f0, &state_f1, &init_f0, &init_f1,
            )
        };
        for j in 0..8 {
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
