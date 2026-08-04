// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::activations::{simd_tanh_avx2, simd_tanh_avx512, simd_tanh_dual_avx2};

const DENSE_POINTS: usize = 4001;

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_pade_nr1_vs_div_precision_avx2() {
    let sweep: Vec<f32> = (0..DENSE_POINTS)
        .map(|i| -4.0_f32 + i as f32 * 0.002_f32)
        .collect();

    let mut max_nr1_vs_div: f32 = 0.0;

    for chunk in sweep.chunks_exact(8) {
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y_nr1 = simd_tanh_pade_nr1_avx2(x);
            let y_div = simd_tanh_avx2(x);

            let mut nr1 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..8 {
                max_nr1_vs_div = max_nr1_vs_div.max((nr1[j] - div[j]).abs());
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
            let y_nr1 = simd_tanh_pade_nr1_avx2(x);
            let y_div = simd_tanh_avx2(x);
            let mut nr1 = [0.0_f32; 8];
            let mut div = [0.0_f32; 8];
            _mm256_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm256_storeu_ps(div.as_mut_ptr(), y_div);
            for j in 0..remainder.len() {
                max_nr1_vs_div = max_nr1_vs_div.max((nr1[j] - div[j]).abs());
            }
        }
    }

    assert!(
        max_nr1_vs_div <= 1e-4,
        "A8 NR1 error {:.4e} > -80 dB",
        max_nr1_vs_div
    );
}

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_pade_nr2_vs_nr1_precision_avx2() {
    let sweep: Vec<f32> = (0..DENSE_POINTS)
        .map(|i| -4.0_f32 + i as f32 * 0.002_f32)
        .collect();

    let mut max_nr1_vs_nr2: f32 = 0.0;

    for chunk in sweep.chunks_exact(8) {
        unsafe {
            let x = _mm256_loadu_ps(chunk.as_ptr());
            let y_nr1 = simd_tanh_pade_nr1_avx2(x);
            let y_nr2 = simd_tanh_pade_nr2_avx2(x);

            let mut nr1 = [0.0_f32; 8];
            let mut nr2 = [0.0_f32; 8];
            _mm256_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm256_storeu_ps(nr2.as_mut_ptr(), y_nr2);

            for j in 0..8 {
                max_nr1_vs_nr2 = max_nr1_vs_nr2.max((nr1[j] - nr2[j]).abs());
            }
        }
    }

    assert!(
        max_nr1_vs_nr2 <= 1e-4,
        "A8 NR1 vs NR2 {:.4e} > -80 dB",
        max_nr1_vs_nr2
    );
}

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_pade_nr1_vs_div_precision_avx512() {
    if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("avx512vl") {
        return;
    }

    let sweep: Vec<f32> = (0..DENSE_POINTS)
        .map(|i| -4.0_f32 + i as f32 * 0.002_f32)
        .collect();

    let mut max_nr1_vs_div: f32 = 0.0;

    for chunk in sweep.chunks_exact(16) {
        unsafe {
            let x = _mm512_loadu_ps(chunk.as_ptr());
            let y_nr1 = simd_tanh_pade_nr1_avx512(x);
            let y_div = simd_tanh_avx512(x);

            let mut nr1 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..16 {
                max_nr1_vs_div = max_nr1_vs_div.max((nr1[j] - div[j]).abs());
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
            let y_nr1 = simd_tanh_pade_nr1_avx512(x);
            let y_div = simd_tanh_avx512(x);

            let mut nr1 = [0.0_f32; 16];
            let mut div = [0.0_f32; 16];
            _mm512_storeu_ps(nr1.as_mut_ptr(), y_nr1);
            _mm512_storeu_ps(div.as_mut_ptr(), y_div);

            for j in 0..remainder.len() {
                max_nr1_vs_div = max_nr1_vs_div.max((nr1[j] - div[j]).abs());
            }
        }
    }

    assert!(
        max_nr1_vs_div <= 1e-4,
        "A8 AVX-512 NR1 error {:.4e} > -80 dB",
        max_nr1_vs_div
    );
}

#[test]
#[ignore = "consistency-only: oráculo f64 fornece correção absoluta; roda em long-suite"]
fn test_pade_nr1_dual_vs_production_avx2() {
    let sweep: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.03125) - 4.0).collect();
    let mut max_nr1_vs_div: f32 = 0.0;

    for pair in sweep.chunks_exact(16) {
        unsafe {
            let x1 = _mm256_loadu_ps(pair[0..8].as_ptr());
            let x2 = _mm256_loadu_ps(pair[8..16].as_ptr());
            let (y1_nr1, y2_nr1) = simd_tanh_pade_nr1_dual_avx2(x1, x2);
            let (y1_div, y2_div) = simd_tanh_dual_avx2(x1, x2);

            let mut nr1_1 = [0.0_f32; 8];
            let mut nr1_2 = [0.0_f32; 8];
            let mut div_1 = [0.0_f32; 8];
            let mut div_2 = [0.0_f32; 8];
            _mm256_storeu_ps(nr1_1.as_mut_ptr(), y1_nr1);
            _mm256_storeu_ps(nr1_2.as_mut_ptr(), y2_nr1);
            _mm256_storeu_ps(div_1.as_mut_ptr(), y1_div);
            _mm256_storeu_ps(div_2.as_mut_ptr(), y2_div);

            for j in 0..8 {
                max_nr1_vs_div = max_nr1_vs_div.max((nr1_1[j] - div_1[j]).abs());
                max_nr1_vs_div = max_nr1_vs_div.max((nr1_2[j] - div_2[j]).abs());
            }
        }
    }

    assert!(
        max_nr1_vs_div <= 1e-4,
        "A8 NR1 Dual error {:.4e} > -80 dB",
        max_nr1_vs_div
    );
}
