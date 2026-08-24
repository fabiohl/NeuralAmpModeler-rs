// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for polynomial tanh SIMD kernels (T-HF1.1, T-HF1.3).
//!
//! Validates max absolute error ≤ 1e-6 vs `f32::tanh` references
//! on [-20, 20] dense sweep, edge cases, and saturation at extremes.
//!
//! Sigmoid polynomial tests live in `sigmoid/high_fidelity_test.rs`.
//! AVX-512 tests are gated on `is_x86_feature_detected!("avx512f")`.

use super::*;

const DENSE_SWEEP_POINTS: usize = 4001; // 0.01 step on [-20, 20]

// ══════════════════════════════════════════════════════════════════════════════
// Tanh sweep — AVX2
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_poly_avx2_sweep() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        // SAFETY: `chunk` has exactly 8 elements from `chunks_exact(8)`, and the store targets an 8-element array, so the AVX2 load/store stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_poly({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is an 8-element array, so `_mm256_loadu_ps`/`_mm256_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_poly({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.1] tanh_poly AVX2 sweep max error: {max_error:.4e} (limit 1e-6)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_poly_edge_cases() {
    let test_vals: [f32; 9] = [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0, f32::NAN];

    // SAFETY: `_mm256_set1_ps` broadcasts a scalar and `_mm256_storeu_ps` writes to the 8-element `result` array, so no out-of-bounds access occurs.
    unsafe {
        for &x in &test_vals {
            let vx = _mm256_set1_ps(x);
            let vy = simd_tanh_poly_avx2(vx);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), vy);
            let y = result[0];

            if x.is_nan() {
                assert!(y.is_nan(), "tanh_poly(NaN) should be NaN, got {y}");
            } else {
                assert!(
                    (-1.0..=1.0).contains(&y),
                    "tanh_poly({x}) = {y} out of [-1, 1]"
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Saturation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_poly_saturation() {
    // SAFETY: `_mm256_set1_ps` broadcasts scalars and `_mm256_storeu_ps` writes to the 8-element `result` array, staying in bounds.
    unsafe {
        let mut result = [0.0_f32; 8];

        // Large positive → 1
        let vx = _mm256_set1_ps(1000.0f32);
        let vy = simd_tanh_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 1.0).abs() < 1e-6, "tanh(+∞) should be 1");

        // Large negative → -1
        let vx = _mm256_set1_ps(-1000.0f32);
        let vy = simd_tanh_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] + 1.0).abs() < 1e-6, "tanh(-∞) should be -1");

        // Zero → 0
        let vx = _mm256_set1_ps(0.0f32);
        let vy = simd_tanh_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert_eq!(result[0], 0.0, "tanh(0) should be 0");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Dual gate
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_tanh_sigmoid_dual_poly_avx2() {
    let test_vals: [f32; 7] = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

    // SAFETY: `_mm256_set1_ps` broadcasts scalars and `_mm256_storeu_ps` writes to the 8-element `t_arr`/`s_arr` arrays, staying in bounds.
    unsafe {
        for &x1_val in &test_vals {
            for &x2_val in &test_vals {
                let x1 = _mm256_set1_ps(x1_val);
                let x2 = _mm256_set1_ps(x2_val);
                let (t, s) = simd_tanh_sigmoid_dual_poly_avx2(x1, x2);

                let mut t_arr = [0.0_f32; 8];
                let mut s_arr = [0.0_f32; 8];
                _mm256_storeu_ps(t_arr.as_mut_ptr(), t);
                _mm256_storeu_ps(s_arr.as_mut_ptr(), s);

                let expected_tanh = x1_val.tanh();
                let expected_sig = 1.0 / (1.0 + (-x2_val).exp());

                assert!(
                    (t_arr[0] - expected_tanh).abs() <= 1e-6,
                    "dual tanh({x1_val}) = {}, expected {expected_tanh}",
                    t_arr[0],
                );
                assert!(
                    (s_arr[0] - expected_sig).abs() <= 1e-6,
                    "dual sigmoid({x2_val}) = {}, expected {expected_sig}",
                    s_arr[0],
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 sweeps (T-HF1.3) — gated on hardware availability
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "avx512")]
#[test]
fn test_tanh_poly_avx512_sweep() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        // SAFETY: `chunk` has exactly 16 elements from `chunks_exact(16)`, and the store targets a 16-element array, so the AVX-512 load/store stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_poly_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is a 16-element array, so `_mm512_loadu_ps`/`_mm512_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "tanh_poly_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.3] tanh_poly AVX-512 sweep max error: {max_error:.4e} (limit 1e-6)");
}

#[cfg(feature = "avx512")]
#[test]
fn test_tanh_sigmoid_dual_poly_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let test_vals: [f32; 7] = [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0];

    // SAFETY: `_mm512_set1_ps` broadcasts scalars and `_mm512_storeu_ps` writes to the 16-element `t_arr`/`s_arr` arrays, staying in bounds.
    unsafe {
        for &x1_val in &test_vals {
            for &x2_val in &test_vals {
                let x1 = _mm512_set1_ps(x1_val);
                let x2 = _mm512_set1_ps(x2_val);
                let (t, s) = simd_tanh_sigmoid_dual_poly_avx512(x1, x2);

                let mut t_arr = [0.0_f32; 16];
                let mut s_arr = [0.0_f32; 16];
                _mm512_storeu_ps(t_arr.as_mut_ptr(), t);
                _mm512_storeu_ps(s_arr.as_mut_ptr(), s);

                let expected_tanh = x1_val.tanh();
                let expected_sig = 1.0 / (1.0 + (-x2_val).exp());

                assert!(
                    (t_arr[0] - expected_tanh).abs() <= 1e-6,
                    "dual_avx512 tanh({x1_val}) = {}, expected {expected_tanh}",
                    t_arr[0],
                );
                assert!(
                    (s_arr[0] - expected_sig).abs() <= 1e-6,
                    "dual_avx512 sigmoid({x2_val}) = {}, expected {expected_sig}",
                    s_arr[0],
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// TC3 — NR precision evaluation sweeps
// ══════════════════════════════════════════════════════════════════════════════
//
// Evaluates whether rcp_ps + Newton-Raphson can match hardware division
// within the 1e-6 error budget on [-20, 20].

#[test]
fn test_tanh_poly_nr1_vs_f32_tanh_avx2() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        // SAFETY: `chunk` has exactly 8 elements from `chunks_exact(8)`, and the store targets an 8-element array, so the AVX2 load/store stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_nr1_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is an 8-element array, so `_mm256_loadu_ps`/`_mm256_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_nr1_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    eprintln!(
        "[TC3] tanh_poly NR1 AVX2 sweep max error vs f32::tanh: {max_error:.4e} (limit 1e-6)"
    );
    assert!(
        max_error <= 1e-6_f32,
        "TC3 NR1 error {:.4e} > 1e-6",
        max_error,
    );
}

#[test]
fn test_tanh_poly_nr2_vs_f32_tanh_avx2() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        // SAFETY: `chunk` has exactly 8 elements from `chunks_exact(8)`, and the store targets an 8-element array, so the AVX2 load/store stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_nr2_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is an 8-element array, so `_mm256_loadu_ps`/`_mm256_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_nr2_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    eprintln!(
        "[TC3] tanh_poly NR2 AVX2 sweep max error vs f32::tanh: {max_error:.4e} (limit 1e-6)"
    );
    assert!(
        max_error <= 1e-6_f32,
        "TC3 NR2 error {:.4e} > 1e-6",
        max_error,
    );
}

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_tanh_poly_nr1_vs_div_avx2() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_delta: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        // SAFETY: `chunk` has exactly 8 elements from `chunks_exact(8)`, and the store targets an 8-element array, so the AVX2 load/store stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y_nr1 = simd_tanh_poly_nr1_avx2(x);
            let y_div = simd_tanh_poly_avx2(x);

            let mut nr1 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..8 {
                max_delta = max_delta.max((nr1[j] - div[j]).abs());
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is an 8-element array, so `_mm256_loadu_ps`/`_mm256_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y_nr1 = simd_tanh_poly_nr1_avx2(x);
            let y_div = simd_tanh_poly_avx2(x);
            let mut nr1 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);
            for j in 0..remainder.len() {
                max_delta = max_delta.max((nr1[j] - div[j]).abs());
            }
        }
    }

    eprintln!("[TC3] tanh_poly NR1 vs div_ps max delta: {max_delta:.4e}");
}

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_tanh_poly_nr2_vs_div_avx2() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_delta: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        // SAFETY: `chunk` has exactly 8 elements from `chunks_exact(8)`, and the store targets an 8-element array, so the AVX2 load/store stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y_nr2 = simd_tanh_poly_nr2_avx2(x);
            let y_div = simd_tanh_poly_avx2(x);

            let mut nr2 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr2.as_mut_ptr(), y_nr2);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..8 {
                max_delta = max_delta.max((nr2[j] - div[j]).abs());
            }
        }
    }

    let remainder = sweep.chunks_exact(8).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 8];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is an 8-element array, so `_mm256_loadu_ps`/`_mm256_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y_nr2 = simd_tanh_poly_nr2_avx2(x);
            let y_div = simd_tanh_poly_avx2(x);
            let mut nr2 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr2.as_mut_ptr(), y_nr2);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);
            for j in 0..remainder.len() {
                max_delta = max_delta.max((nr2[j] - div[j]).abs());
            }
        }
    }

    eprintln!("[TC3] tanh_poly NR2 vs div_ps max delta: {max_delta:.4e}");
}

// ══════════════════════════════════════════════════════════════════════════════
// TC3 — NR precision evaluation sweeps (AVX-512)
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "avx512")]
#[test]
fn test_tanh_poly_nr1_vs_f32_tanh_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        // SAFETY: `chunk` has exactly 16 elements from `chunks_exact(16)`, and the store targets a 16-element array, so the AVX-512 load/store stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_nr1_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is a 16-element array, so `_mm512_loadu_ps`/`_mm512_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_nr1_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    eprintln!(
        "[TC3] tanh_poly NR1 AVX-512 sweep max error vs f32::tanh: {max_error:.4e} (limit 1e-6)"
    );
    assert!(
        max_error <= 1e-6_f32,
        "TC3 AVX-512 NR1 error {:.4e} > 1e-6",
        max_error,
    );
}

#[cfg(feature = "avx512")]
#[test]
fn test_tanh_poly_nr2_vs_f32_tanh_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        // SAFETY: `chunk` has exactly 16 elements from `chunks_exact(16)`, and the store targets a 16-element array, so the AVX-512 load/store stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_tanh_poly_nr2_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is a 16-element array, so `_mm512_loadu_ps`/`_mm512_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_tanh_poly_nr2_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = input.tanh();
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
            }
        }
    }

    eprintln!(
        "[TC3] tanh_poly NR2 AVX-512 sweep max error vs f32::tanh: {max_error:.4e} (limit 1e-6)"
    );
    assert!(
        max_error <= 1e-6_f32,
        "TC3 AVX-512 NR2 error {:.4e} > 1e-6",
        max_error,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_tanh_poly_nr1_vs_div_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_delta: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        // SAFETY: `chunk` has exactly 16 elements from `chunks_exact(16)`, and the store targets a 16-element array, so the AVX-512 load/store stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y_nr1 = simd_tanh_poly_nr1_avx512(x);
            let y_div = simd_tanh_poly_avx512(x);

            let mut nr1 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..16 {
                max_delta = max_delta.max((nr1[j] - div[j]).abs());
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is a 16-element array, so `_mm512_loadu_ps`/`_mm512_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y_nr1 = simd_tanh_poly_nr1_avx512(x);
            let y_div = simd_tanh_poly_avx512(x);
            let mut nr1 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);
            for j in 0..remainder.len() {
                max_delta = max_delta.max((nr1[j] - div[j]).abs());
            }
        }
    }

    eprintln!("[TC3] tanh_poly NR1 vs div_ps AVX-512 max delta: {max_delta:.4e}");
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_tanh_poly_nr2_vs_div_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_delta: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        // SAFETY: `chunk` has exactly 16 elements from `chunks_exact(16)`, and the store targets a 16-element array, so the AVX-512 load/store stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y_nr2 = simd_tanh_poly_nr2_avx512(x);
            let y_div = simd_tanh_poly_avx512(x);

            let mut nr2 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr2.as_mut_ptr(), y_nr2);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..16 {
                max_delta = max_delta.max((nr2[j] - div[j]).abs());
            }
        }
    }

    let remainder = sweep.chunks_exact(16).remainder();
    if !remainder.is_empty() {
        let mut batch = [0.0_f32; 16];
        for (j, &input) in remainder.iter().enumerate() {
            batch[j] = input;
        }
        for item in batch.iter_mut().skip(remainder.len()) {
            *item = 0.0_f32;
        }
        // SAFETY: `batch` is a 16-element array, so `_mm512_loadu_ps`/`_mm512_storeu_ps` stay in bounds.
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y_nr2 = simd_tanh_poly_nr2_avx512(x);
            let y_div = simd_tanh_poly_avx512(x);
            let mut nr2 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr2.as_mut_ptr(), y_nr2);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);
            for j in 0..remainder.len() {
                max_delta = max_delta.max((nr2[j] - div[j]).abs());
            }
        }
    }

    eprintln!("[TC3] tanh_poly NR2 vs div_ps AVX-512 max delta: {max_delta:.4e}");
}
