// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

/// Mean Squared Error between two sample buffers.
///
/// Panics if the vectors have different lengths.
pub fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors of different sizes for MSE");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (n as f64)
}

/// Maximum absolute sample-wise error between two buffers.
///
/// Panics if the vectors have different lengths.
pub fn compute_max_abs_error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "Vectors of different sizes for MaxAbsError"
    );
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0f64, f64::max)
}

/// Error-to-Signal Ratio (scale-invariant).
///
/// `ESR = Σ|ref[i] - test[i]|² / Σ|ref[i]|²`.
/// Returns 0.0 when signals are identical, ∞ when the reference is zero
/// but the test is not.
pub fn compute_esr(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "Vectors of different sizes for ESR"
    );
    let sig: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(r, t)| ((r - t) as f64).powi(2))
        .sum();
    if sig <= f64::EPSILON {
        if noise <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise / sig
    }
}
