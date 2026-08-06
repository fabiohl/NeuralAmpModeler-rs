// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LRA (EBU Tech 3342) and combined loudness measurement.

use super::lufs;
use super::lufs::apply_k_weighting;

/// LRA short-term block duration (3 s per EBU Tech 3342 §2.3).
const LRA_BLOCK_MS: u64 = 3000;

/// LRA relative gate threshold in LU (EBU Tech 3342 §2.3.4).
const LRA_REL_GATE: f64 = -20.0;

// =============================================================================
// LRA — EBU Tech 3342 Loudness Range
// =============================================================================

/// Computes the Loudness Range (LRA) per EBU Tech 3342.
///
/// Algorithm:
/// 1. Compute short-term loudness values (3-second blocks, non-overlapping)
/// 2. **Absolute gate:** discard blocks ≤ −70 LUFS
/// 3. Compute mean of surviving blocks → L_ASG
/// 4. **Relative gate at −20 LU:** discard blocks < (L_ASG − 20)
/// 5. Sort remaining blocks by loudness
/// 6. LRA = P95 − P10 (linear interpolation between samples)
///
/// Returns 0.0 if there are insufficient blocks after gating (need ≥ 2 blocks
/// for a meaningful range) or if `samples` is empty.
pub fn compute_lra(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let k_weighted = apply_k_weighting(samples);

    // Short-term blocks: 3 s, non-overlapping (hop = block_samples)
    let block_samples = (sample_rate as usize * LRA_BLOCK_MS as usize) / 1000;
    if block_samples > k_weighted.len() {
        return 0.0;
    }
    let num_blocks = (k_weighted.len() - block_samples) / block_samples + 1;
    let hop = block_samples; // non-overlapping for LRA per EBU Tech 3342 §2.3

    let mut short_term_lk: Vec<f64> = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * hop;
        let sum_sq: f64 = k_weighted[start..start + block_samples]
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        let power = sum_sq / block_samples as f64;
        short_term_lk.push(lufs::power_to_lkfs(power));
    }

    // Pass 1: absolute gate at -70 LUFS (EBU Tech 3342 §2.3.3)
    let abs_gated: Vec<f64> = short_term_lk
        .iter()
        .filter(|&&lk| lk > lufs::LUFS_ABS_GATE && lk.is_finite())
        .copied()
        .collect();
    if abs_gated.len() < 2 {
        return 0.0;
    }

    // Mean of absolute-gated blocks (L_ASG)
    let l_asg = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;

    // Pass 2: relative gate at -20 LU (EBU Tech 3342 §2.3.4)
    let rel_threshold = l_asg + LRA_REL_GATE;
    let mut gated: Vec<f64> = abs_gated
        .into_iter()
        .filter(|&lk| lk > rel_threshold)
        .collect();
    if gated.len() < 2 {
        return 0.0;
    }

    gated.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // P10 and P95 by linear interpolation (EBU Tech 3342 Annex)
    let p10 = percentile_by_interpolation(&gated, 10.0);
    let p95 = percentile_by_interpolation(&gated, 95.0);
    p95 - p10
}

/// Computes a percentile by linear interpolation between ordered samples.
///
/// Uses the C=1 method (inverse of ECDF with linear interpolation between
/// adjacent order statistics), matching EBU Tech 3342 Annex recommendation.
fn percentile_by_interpolation(sorted: &[f64], pct: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if hi >= n {
        return sorted[n - 1];
    }
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Short-term loudness values for detailed analysis.
///
/// Returns a `Vec` of LKFS values computed over 3-second non-overlapping blocks,
/// without any gating. Use `compute_lra` for the gated LRA metric.
pub fn short_term_loudness(samples: &[f32], sample_rate: u32) -> Vec<f64> {
    if samples.is_empty() {
        return Vec::new();
    }
    let k_weighted = apply_k_weighting(samples);
    let block_samples = (sample_rate as usize * LRA_BLOCK_MS as usize) / 1000;
    if block_samples > k_weighted.len() {
        return Vec::new();
    }
    let num_blocks = (k_weighted.len() - block_samples) / block_samples + 1;
    let mut values = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * block_samples;
        let sum_sq: f64 = k_weighted[start..start + block_samples]
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        values.push(lufs::power_to_lkfs(sum_sq / block_samples as f64));
    }
    values
}

// =============================================================================
// Combined loudness measurement result
// =============================================================================

/// Result of a complete BS.1770-4 + EBU Tech 3342 loudness measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct LoudnessResult {
    /// Integrated LUFS (ITU-R BS.1770-4 2-pass).
    pub integrated_lufs: f64,
    /// Loudness Range (EBU Tech 3342).
    pub lra: f64,
    /// True-peak level in dBTP (BS.1770-4 Annex 2).
    pub true_peak_db: f64,
    /// Raw short-term loudness values (ungated, 3 s blocks).
    pub short_term: Vec<f64>,
}

/// Runs a complete loudness measurement: integrated LUFS, LRA, and true-peak.
///
/// This is the recommended entry point for QA/reporting — computes all three
/// metrics in a single pass (K-weighting is shared between LUFS and LRA).
pub fn measure_loudness(samples: &[f32], sample_rate: u32) -> LoudnessResult {
    let integrated_lufs = super::compute_integrated_lufs(samples, sample_rate);
    let lra = compute_lra(samples, sample_rate);
    let true_peak_db = super::compute_true_peak_db(samples);
    let short_term = short_term_loudness(samples, sample_rate);
    LoudnessResult {
        integrated_lufs,
        lra,
        true_peak_db,
        short_term,
    }
}
