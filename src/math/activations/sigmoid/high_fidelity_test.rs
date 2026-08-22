// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for polynomial sigmoid SIMD kernels.
//!
//! Validates max absolute error ≤ 1e-6 vs `f32::exp` references
//! on [-20, 20] dense sweep, edge cases, and saturation at extremes.
//!
//! AVX-512 tests are gated on `is_x86_feature_detected!("avx512f")`.

use super::*;

const DENSE_SWEEP_POINTS: usize = 4001; // 0.01 step on [-20, 20]

// ══════════════════════════════════════════════════════════════════════════════
// Sigmoid sweep — AVX2
// ══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_sigmoid_poly_avx2_sweep() {
    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(8) {
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y = simd_sigmoid_poly_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_poly({input}) = {}, expected {expected}, delta {error}",
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
        unsafe {
            let x = _mm256_loadu_ps(batch.as_ptr());
            let y = simd_sigmoid_poly_avx2(x);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_poly({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.1] sigmoid_poly AVX2 sweep max error: {max_error:.4e} (limit 1e-6)");
}

// ══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_poly_edge_cases() {
    let test_vals: [f32; 9] = [-100.0, -20.0, -1.0, -0.0, 0.0, 1.0, 20.0, 100.0, f32::NAN];

    unsafe {
        for &x in &test_vals {
            let vx = _mm256_set1_ps(x);
            let vy = simd_sigmoid_poly_avx2(vx);
            let mut result = [0.0_f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), vy);
            let y = result[0];

            if x.is_nan() {
                assert!(y.is_nan(), "sigmoid_poly(NaN) should be NaN, got {y}");
            } else {
                assert!(
                    (0.0..=1.0).contains(&y),
                    "sigmoid_poly({x}) = {y} out of [0, 1]"
                );
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Saturation
// ══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_sigmoid_poly_saturation() {
    unsafe {
        let mut result = [0.0_f32; 8];

        // Large positive → 1
        let vx = _mm256_set1_ps(1000.0f32);
        let vy = simd_sigmoid_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 1.0).abs() < 1e-6, "σ(+∞) should be 1");

        // Large negative → 0
        let vx = _mm256_set1_ps(-1000.0f32);
        let vy = simd_sigmoid_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0]).abs() < 1e-6, "σ(-∞) should be 0");

        // Zero → 0.5
        let vx = _mm256_set1_ps(0.0f32);
        let vy = simd_sigmoid_poly_avx2(vx);
        _mm256_storeu_ps(result.as_mut_ptr(), vy);
        assert!((result[0] - 0.5).abs() < 1e-6, "σ(0) should be 0.5");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 sweeps — gated on hardware availability
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "avx512")]
#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_sigmoid_poly_avx512_sweep() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_SWEEP_POINTS)
        .map(|i| -20.0_f32 + i as f32 * 0.01_f32)
        .collect();
    let mut max_error: f32 = 0.0_f32;

    for chunk in sweep.chunks_exact(16) {
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y = simd_sigmoid_poly_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for (j, &input) in chunk.iter().enumerate() {
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_poly_avx512({input}) = {}, expected {expected}, delta {error}",
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
        unsafe {
            let x = _mm512_loadu_ps(batch.as_ptr());
            let y = simd_sigmoid_poly_avx512(x);
            let mut result = [0.0_f32; 16];
            _mm512_storeu_ps(result.as_mut_ptr(), y);

            for j in 0..remainder.len() {
                let input = remainder[j];
                let expected = 1.0 / (1.0 + (-input).exp());
                let error = (expected - result[j]).abs();
                max_error = max_error.max(error);
                assert!(
                    error <= 1e-6_f32,
                    "sigmoid_poly_avx512({input}) = {}, expected {expected}, delta {error}",
                    result[j],
                );
            }
        }
    }

    eprintln!("[T-HF1.3] sigmoid_poly AVX-512 sweep max error: {max_error:.4e} (limit 1e-6)");
}
