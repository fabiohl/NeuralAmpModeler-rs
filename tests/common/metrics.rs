// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// Shared perceptual metric utilities compiled across integration test targets.
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
    if sig == 0.0 {
        if noise == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        noise / sig
    }
}

/// Result of an exact IEEE-754 bitwise sample representation comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitwiseComparison {
    /// Total samples compared.
    pub total_samples: usize,
    /// Number of samples whose IEEE-754 bit representations differ.
    pub mismatch_count: usize,
    /// First sample index where bits differed, if any.
    pub first_mismatch_idx: Option<usize>,
    /// Bit value of reference at first mismatch (`f32::to_bits()`), if any.
    pub first_ref_bits: Option<u32>,
    /// Bit value of test at first mismatch (`f32::to_bits()`), if any.
    pub first_test_bits: Option<u32>,
}

/// Bitwise comparison of two f32 sample buffers (R-07, R-11).
///
/// Compares the IEEE-754 bit representation (`f32::to_bits()`) of every sample.
/// Note that `+0.0` (`0x0000_0000`) and `-0.0` (`0x8000_0000`) have differing bit
/// representations and are correctly reported as mismatched bits.
pub fn compare_samples_bitwise(reference: &[f32], test: &[f32]) -> BitwiseComparison {
    assert_eq!(
        reference.len(),
        test.len(),
        "Vectors of different sizes for bitwise comparison"
    );
    let mut mismatch_count = 0;
    let mut first_mismatch_idx = None;
    let mut first_ref_bits = None;
    let mut first_test_bits = None;

    for (i, (&r, &t)) in reference.iter().zip(test.iter()).enumerate() {
        let r_bits = r.to_bits();
        let t_bits = t.to_bits();
        if r_bits != t_bits {
            if first_mismatch_idx.is_none() {
                first_mismatch_idx = Some(i);
                first_ref_bits = Some(r_bits);
                first_test_bits = Some(t_bits);
            }
            mismatch_count += 1;
        }
    }

    BitwiseComparison {
        total_samples: reference.len(),
        mismatch_count,
        first_mismatch_idx,
        first_ref_bits,
        first_test_bits,
    }
}
