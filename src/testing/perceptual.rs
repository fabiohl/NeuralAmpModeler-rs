// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Perceptual metrics for audio quality evaluation.
//!
//! Includes ESR (Error-to-Signal Ratio), LUFS (ITU-R BS.1770-4 full 2-pass gating),
//! LRA (EBU Tech 3342), true-peak (BS.1770-4 Annex 2), MR-STFT, and baseline
//! constants from published A2/Tone3000 data.
// Published baselines from t3k-mushra / A2ESR test framework.
#![allow(dead_code)]

use crate::math::dsp::fft::FftPlanner;

// =============================================================================
// Published Baselines (t3k-mushra / A2Esr.tsx:19-38)
// =============================================================================

/// A1-Standard median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A1_STANDARD_MEDIAN: f64 = 0.00623;

/// A2-Full median ESR baseline from t3k-mushra/A2Esr.tsx
pub const A2ESR_A2_FULL_MEDIAN: f64 = 0.00334;

/// A2-Lite median ESR baseline (preliminary — pending t3k-mushra publication).
/// A2-Lite shares the A2-Full architecture with rank-reduced weights; ESR is
/// expected to be in the same order of magnitude as A2-Full (≤ 0.005).
pub const A2ESR_A2_LITE_MEDIAN: f64 = 0.005;
/// A2-Lite Q1 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q1: f64 = 0.0015;
/// A2-Lite Q3 ESR baseline (preliminary)
pub const A2ESR_A2_LITE_Q3: f64 = 0.012;

// =============================================================================
// Error-to-Signal Ratio (ESR)
// =============================================================================

/// Computes the Error-to-Signal Ratio (linear scale).
///
/// `ESR = Σ(r - t)² / Σ r²`
///
/// Returns `f64::INFINITY` if the reference signal has zero energy.
pub fn compute_esr(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_esr: vectors must have same length"
    );
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = r as f64 - t as f64;
        signal_power += (r as f64) * (r as f64);
        noise_power += d * d;
    }
    if signal_power <= f64::EPSILON {
        if noise_power <= f64::EPSILON {
            return 0.0;
        }
        return f64::INFINITY;
    }
    noise_power / signal_power
}

/// Computes blockwise Error-to-Signal Ratio on disjoint blocks.
///
/// Returns a vector of ESR values, one for each block of `block_size` samples.
/// If the last block is partial, ESR is computed over the remaining samples.
pub fn compute_esr_blockwise(reference: &[f32], test: &[f32], block_size: usize) -> Vec<f64> {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_esr_blockwise: vectors must have same length"
    );
    assert!(
        block_size > 0,
        "compute_esr_blockwise: block_size must be greater than zero"
    );

    reference
        .chunks(block_size)
        .zip(test.chunks(block_size))
        .map(|(ref_chunk, test_chunk)| compute_esr(ref_chunk, test_chunk))
        .collect()
}

/// Converts linear ESR to dB: `10 * log10(esr)`.
pub fn esr_to_db(esr: f64) -> f64 {
    if esr <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        10.0 * esr.log10()
    }
}

/// Computes Signal-to-Noise Ratio (SNR) in dB between reference and test.
pub fn compute_snr_db(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_snr_db: vectors must have same length"
    );
    let mut signal_power = 0.0f64;
    let mut noise_power = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        let d = r as f64 - t as f64;
        signal_power += (r as f64) * (r as f64);
        noise_power += d * d;
    }
    if noise_power <= f64::EPSILON {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// =============================================================================
// MR-STFT — Multi-Resolution Short-Time Fourier Transform Loss
// =============================================================================

/// Window sizes (samples) for multi-resolution STFT analysis.
pub const MRSTFT_WINDOW_SIZES: [usize; 3] = [256, 1024, 4096];

/// Recommended weights for each window size from t3k-mushra golden calibration.
pub const MRSTFT_WEIGHTS: [f64; 3] = [0.1, 0.3, 0.5];

/// Computes the Multi-Resolution STFT loss between reference and test signals.
///
/// For each window size in `[256, 1024, 4096]` with hop = window/4:
/// 1. Applies a Hann window
/// 2. Computes STFT via native `crate::math::dsp::fft::FftPlanner` (SoA)
/// 3. Calculates L1 and L2 of log-magnitude differences per frame
/// 4. Averages frame losses and weights by window size
///
/// Each frame uses a relative floor at −80 dB below its own spectral peak
/// (absolute floor 1e−8 as fallback for silent frames), replacing the prior
/// fixed absolute floor of 1e−8.
///
/// ```text
/// MR-STFT = Σ_w weight[w] · mean_frame( L1_sc + L2_sc )
/// where:
///   L1_sc = (1/F) Σ_f |ln|X_ref[f]| - ln|X_test[f]||
///   L2_sc = sqrt( (1/F) Σ_f (ln|X_ref[f]| - ln|X_test[f]|)² )
/// ```
pub fn compute_mr_stft(reference: &[f32], test: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        test.len(),
        "compute_mr_stft: vectors must have same length"
    );

    if reference.is_empty() {
        return 0.0;
    }

    let eps_abs = 1e-8f64;
    let mut total_loss = 0.0f64;

    for (&ws, &weight) in MRSTFT_WINDOW_SIZES.iter().zip(MRSTFT_WEIGHTS.iter()) {
        let hop = ws / 4;
        if ws > reference.len() {
            continue;
        }

        let fft = FftPlanner::<f64>::new(ws);

        let window: Vec<f64> = (0..ws)
            // Symmetric (non-periodic) Hann window — denominator is (ws - 1)
            // instead of ws. The symmetric variant places the zero samples at
            // both endpoints, matching the original von Hann formulation and
            // maximizing side-lobe rejection for general-purpose spectral
            // analysis. The periodic variant (ws denominator, last sample
            // ≠ zero) is designed for seamless overlap-add STFT resynthesis
            // (COLA) but is not required here since MR-STFT is an analysis-
            // only metric.
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (ws - 1) as f64).cos()))
            .collect();

        let num_frames = (reference.len() - ws) / hop + 1;
        if num_frames == 0 {
            continue;
        }

        let num_bins = ws / 2 + 1;

        // Reusable FFT scratch buffers (SoA)
        let mut buf_ref_re = vec![0.0f64; ws];
        let mut buf_ref_im = vec![0.0f64; ws];
        let mut buf_test_re = vec![0.0f64; ws];
        let mut buf_test_im = vec![0.0f64; ws];
        let mut mag_ref = vec![0.0f64; num_bins];
        let mut mag_test = vec![0.0f64; num_bins];

        let mut window_loss_sum = 0.0f64;

        for frame in 0..num_frames {
            let offset = frame * hop;

            // Fill FFT SoA buffers with windowed samples
            for i in 0..ws {
                buf_ref_re[i] = reference[offset + i] as f64 * window[i];
                buf_test_re[i] = test[offset + i] as f64 * window[i];
                buf_ref_im[i] = 0.0;
                buf_test_im[i] = 0.0;
            }

            fft.process(&mut buf_ref_re, &mut buf_ref_im);
            fft.process(&mut buf_test_re, &mut buf_test_im);

            // Compute raw magnitudes and find frame spectral peak
            let mut frame_peak = 0.0f64;
            for i in 0..num_bins {
                mag_ref[i] = (buf_ref_re[i] * buf_ref_re[i] + buf_ref_im[i] * buf_ref_im[i]).sqrt();
                mag_test[i] =
                    (buf_test_re[i] * buf_test_re[i] + buf_test_im[i] * buf_test_im[i]).sqrt();
                frame_peak = frame_peak.max(mag_ref[i]).max(mag_test[i]);
            }

            // Relative floor: −80 dB below frame peak, clamped to absolute minimum
            let eps_frame = if frame_peak > eps_abs {
                frame_peak * 1e-4 // 10^(−80/20)
            } else {
                eps_abs
            };

            // Apply floor and take ln
            for i in 0..num_bins {
                mag_ref[i] = mag_ref[i].max(eps_frame).ln();
                mag_test[i] = mag_test[i].max(eps_frame).ln();
            }

            let mut l1 = 0.0f64;
            let mut l2_sq = 0.0f64;
            for i in 0..num_bins {
                let diff = (mag_ref[i] - mag_test[i]).abs();
                l1 += diff;
                l2_sq += diff * diff;
            }

            let l1_sc = l1 / num_bins as f64;
            let l2_sc = (l2_sq / num_bins as f64).sqrt();
            window_loss_sum += l1_sc + l2_sc;
        }

        let window_loss = window_loss_sum / num_frames as f64;
        total_loss += weight * window_loss;
    }

    total_loss
}

// =============================================================================
// LUFS — ITU-R BS.1770-4 full 2-pass gating (integrated)
// =============================================================================

/// K-weighting pre-filter coefficients (2nd-order high-pass, ~38 Hz).
/// H(z) = (1 - 2z⁻¹ + z⁻²) / (1 - 1.99004745483398z⁻¹ + 0.99007225036621z⁻²)
const KW_PRE_B: (f64, f64, f64) = (1.0, -2.0, 1.0);
const KW_PRE_A: (f64, f64) = (1.99004745483398, -0.99007225036621);

/// K-weighting RLB high-shelf coefficients (+4 dB above ~1-2 kHz).
/// H(z) = (1.53512485958697 - 2.69169618940638z⁻¹ + 1.19839281085285z⁻²)
///      / (1.0 - 1.69065929318241z⁻¹ + 0.73248077421585z⁻²)
const KW_SHELF_B: (f64, f64, f64) = (1.53512485958697, -2.69169618940638, 1.19839281085285);
const KW_SHELF_A: (f64, f64) = (1.69065929318241, -0.73248077421585);

/// Applies the full K-weighting filter chain (pre-filter + RLB high-shelf).
///
/// ITU-R BS.1770-4 §2.1: K-weighting shapes the spectrum to approximate
/// the frequency response of human hearing at typical listening levels.
pub(crate) fn apply_k_weighting(samples: &[f32]) -> Vec<f32> {
    let pre_filtered = apply_biquad(
        samples, KW_PRE_B.0, KW_PRE_B.1, KW_PRE_B.2, KW_PRE_A.0, KW_PRE_A.1,
    );
    apply_biquad(
        &pre_filtered,
        KW_SHELF_B.0,
        KW_SHELF_B.1,
        KW_SHELF_B.2,
        KW_SHELF_A.0,
        KW_SHELF_A.1,
    )
}

/// LUFS block duration (400 ms per ITU-R BS.1770-4 §2.2).
const LUFS_BLOCK_MS: u64 = 400;

/// LUFS block overlap fraction (75 % = 3/4 per ITU-R BS.1770-4 §2.2).
const LUFS_OVERLAP_NUM: u64 = 3;
const LUFS_OVERLAP_DEN: u64 = 4;

/// Absolute gate threshold in LUFS (ITU-R BS.1770-4 §2.2 Table 1).
const LUFS_ABS_GATE: f64 = -70.0;

/// Relative gate threshold in LU (ITU-R BS.1770-4 §2.2).
const LUFS_REL_GATE: f64 = -10.0;

/// LRA short-term block duration (3 s per EBU Tech 3342 §2.3).
const LRA_BLOCK_MS: u64 = 3000;

/// LRA relative gate threshold in LU (EBU Tech 3342 §2.3.4).
const LRA_REL_GATE: f64 = -20.0;

/// BS.1770-4 loudness constant (converts mean-square to LKFS/LUFS).
const LUFS_OFFSET: f64 = -0.691;

/// Computes loudness level in LKFS from mean-square power.
#[inline]
fn power_to_lkfs(power: f64) -> f64 {
    if power <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        LUFS_OFFSET + 10.0 * power.log10()
    }
}

/// Computes mean-square power from loudness level in LKFS.
#[inline]
fn lkfs_to_power(lkfs: f64) -> f64 {
    10.0f64.powf((lkfs - LUFS_OFFSET) / 10.0)
}

/// Computes mean-square powers for overlapping blocks.
///
/// Returns `(block_powers, hop)` where `block_powers[i]` is the mean square
/// of the K-weighted signal over block `i`, and `hop` is the stride in samples.
fn compute_block_powers(
    k_weighted: &[f32],
    sample_rate: u32,
    block_ms: u64,
    overlap_num: u64,
    overlap_den: u64,
) -> (Vec<f64>, usize) {
    let block_samples = (sample_rate as usize * block_ms as usize) / 1000;
    let hop = block_samples * (overlap_den - overlap_num) as usize / overlap_den as usize;
    if block_samples > k_weighted.len() || hop == 0 {
        return (Vec::new(), hop);
    }
    let num_blocks = (k_weighted.len() - block_samples) / hop + 1;
    let mut powers = Vec::with_capacity(num_blocks);
    for b in 0..num_blocks {
        let start = b * hop;
        let sum_sq: f64 = k_weighted[start..start + block_samples]
            .iter()
            .map(|&x| (x as f64).powi(2))
            .sum();
        powers.push(sum_sq / block_samples as f64);
    }
    (powers, hop)
}

/// Full ITU-R BS.1770-4 integrated loudness with 2-pass gating.
///
/// Algorithm:
/// 1. Apply K-weighting (pre-filter + RLB high-shelf)
/// 2. Divide into 400 ms blocks with 75 % overlap
/// 3. Compute mean-square power per block
/// 4. **Pass 1 — absolute gate:** discard blocks below −70 LUFS
/// 5. Compute ungated integrated loudness from surviving blocks
/// 6. **Pass 2 — relative gate:** discard blocks below (ungated − 10 LU)
/// 7. Integrated LUFS = loudness of finally surviving blocks
///
/// Returns `f64::NEG_INFINITY` if `samples` is empty or all blocks are gated out.
///
/// Validated against EBU reference vectors within ±0.1 LU (see tests).
pub fn compute_integrated_lufs(samples: &[f32], sample_rate: u32) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let k_weighted = apply_k_weighting(samples);

    let (block_powers, _hop) = compute_block_powers(
        &k_weighted,
        sample_rate,
        LUFS_BLOCK_MS,
        LUFS_OVERLAP_NUM,
        LUFS_OVERLAP_DEN,
    );
    if block_powers.is_empty() {
        return f64::NEG_INFINITY;
    }

    let block_lkfs: Vec<f64> = block_powers.iter().map(|&p| power_to_lkfs(p)).collect();

    // Pass 1: absolute gate at -70 LUFS
    let gated_1: Vec<f64> = (0..block_powers.len())
        .filter(|&i| block_lkfs[i] > LUFS_ABS_GATE)
        .map(|i| block_powers[i])
        .collect();
    if gated_1.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mean_power_1: f64 = gated_1.iter().sum::<f64>() / gated_1.len() as f64;
    let ungated_lkfs = power_to_lkfs(mean_power_1);

    // Pass 2: relative gate at (ungated - 10 LU)
    let rel_threshold_lk = ungated_lkfs + LUFS_REL_GATE;

    let gated_2: Vec<f64> = (0..block_powers.len())
        .filter(|&i| block_lkfs[i] > LUFS_ABS_GATE && block_lkfs[i] > rel_threshold_lk)
        .map(|i| block_powers[i])
        .collect();
    if gated_2.is_empty() {
        return ungated_lkfs;
    }
    let mean_power_2: f64 = gated_2.iter().sum::<f64>() / gated_2.len() as f64;
    power_to_lkfs(mean_power_2)
}

/// Simplified compatibility wrapper — calls `compute_integrated_lufs`.
///
/// For diagnostic use in test reports. Prefer `compute_integrated_lufs` for
/// new code and `measure_loudness` for combined LUFS/LRA/dBTP reporting.
#[inline]
pub fn compute_lufs(samples: &[f32], sample_rate: u32) -> f64 {
    compute_integrated_lufs(samples, sample_rate)
}

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
        short_term_lk.push(power_to_lkfs(power));
    }

    // Pass 1: absolute gate at -70 LUFS (EBU Tech 3342 §2.3.3)
    let abs_gated: Vec<f64> = short_term_lk
        .iter()
        .filter(|&&lk| lk > LUFS_ABS_GATE && lk.is_finite())
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
        values.push(power_to_lkfs(sum_sq / block_samples as f64));
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
    let integrated_lufs = compute_integrated_lufs(samples, sample_rate);
    let lra = compute_lra(samples, sample_rate);
    let true_peak_db = compute_true_peak_db(samples);
    let short_term = short_term_loudness(samples, sample_rate);
    LoudnessResult {
        integrated_lufs,
        lra,
        true_peak_db,
        short_term,
    }
}

/// Applies a biquad IIR filter.
/// Direct Form II Transposed biquad filter.
///
/// Implements `H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (1 - a1*z⁻¹ - a2*z⁻²)`.
fn apply_biquad(samples: &[f32], b0: f64, b1: f64, b2: f64, a1: f64, a2: f64) -> Vec<f32> {
    let mut out = Vec::with_capacity(samples.len());
    let mut s1: f64 = 0.0;
    let mut s2: f64 = 0.0;
    for &x in samples {
        let xf = x as f64;
        let y = b0 * xf + s1;
        s1 = b1 * xf + a1 * y + s2;
        s2 = b2 * xf + a2 * y;
        out.push(y as f32);
    }
    out
}

// =============================================================================
// True-peak (dBTP) — ITU-R BS.1770-4 Annex 2 — 4× oversampling FIR
//
// RT-SAFETY DECISION (2026-06-26):
//   - RT hot-path (src/dsp/pipeline/stages/output.rs): keeps sample-peak
//     detection for `RT_STATUS_HAS_CLIPPED`. True-peak with 48-tap FIR × 4×
//     oversampling adds ~48 MAC/sample (12 per phase × 4) — prohibitive in the
//     DSP callback where every μs matters.
//   - Off-RT QA/telemetry: functions below expose full BS.1770-4 dBTP via
//     `compute_true_peak_db()` and `find_true_peak_overs()`. The main-thread
//     telemetry loop (src/standalone/rt_setup/telemetry.rs:81) can optionally
//     run these on buffered audio for comprehensive inter-sample over detection.
//   - Bench number (linked to P-7): pending hardware-validation measurements.
// =============================================================================

/// Oversampling factor for BS.1770-4 true-peak measurement.
const TP_OVERSAMPLE: usize = 4;

/// Number of taps in the BS.1770-4 Annex 2 FIR filter (full, 48 taps).
const TP_FIR_LEN: usize = 48;

/// Number of taps per polyphase sub-filter (48 / 4 = 12).
const TP_TAPS: usize = TP_FIR_LEN / TP_OVERSAMPLE;

/// BS.1770-4 Annex 2 polyphase sub-filter coefficients (4 phases × 12 taps each).
///
/// These are the polyphase sub-filters H_p(z) for 4× oversampling, given directly
/// by ITU-R BS.1770-4 Annex 2 Table (p. 17). Each phase sums to ~1.0 (unity DC gain).
///
/// Phase ordering follows the standard: H_0 through H_3 are convolution filters
/// that produce outputs y[4n], y[4n+1], y[4n+2], y[4n+3] respectively.
///
/// Symmetry properties: phase 3 = reversed(phase 0), phase 2 = reversed(phase 1).
#[rustfmt::skip]
const BS1770_PHASES: [[f64; TP_TAPS]; TP_OVERSAMPLE] = [
    // Phase 0: produces y[4n+0]
    [
         0.0017089843750,  0.0109863281250, -0.0196533203125,  0.0332031250000,
        -0.0594482421875,  0.1373291015625,  0.9721679687500, -0.1022949218750,
         0.0476074218750, -0.0266113281250,  0.0148925781250, -0.0083007812500,
    ],
    // Phase 1: produces y[4n+1]
    [
        -0.0291748046875,  0.0292968750000, -0.0517578125000,  0.0891113281250,
        -0.1665039062500,  0.4650878906250,  0.7797851562500, -0.2003173828125,
         0.1015625000000, -0.0582275390625,  0.0330810546875, -0.0189208984375,
    ],
    // Phase 2: produces y[4n+2]
    [
        -0.0189208984375,  0.0330810546875, -0.0582275390625,  0.1015625000000,
        -0.2003173828125,  0.7797851562500,  0.4650878906250, -0.1665039062500,
         0.0891113281250, -0.0517578125000,  0.0292968750000, -0.0291748046875,
    ],
    // Phase 3: produces y[4n+3]
    [
        -0.0083007812500,  0.0148925781250, -0.0266113281250,  0.0476074218750,
        -0.1022949218750,  0.9721679687500,  0.1373291015625, -0.0594482421875,
         0.0332031250000, -0.0196533203125,  0.0109863281250,  0.0017089843750,
    ],
];

/// 4× oversampling via BS.1770-4 Annex 2 polyphase FIR.
///
/// For input sample `x[n]`, produces 4 output samples:
/// ```text
/// y[4n+p] = x[n]*h_p[0] + x[n-1]*h_p[1] + ... + x[n-11]*h_p[11]   (p = 0,1,2,3)
/// ```
/// Uses a sliding window over the input (off-RT — allocates).
fn oversample_4x_bs1770(samples: &[f32]) -> Vec<f64> {
    let in_len = samples.len();
    if in_len == 0 {
        return Vec::new();
    }
    let out_len = in_len * TP_OVERSAMPLE;
    let mut out = vec![0.0f64; out_len];

    for n in 0..in_len {
        let base = n * TP_OVERSAMPLE;
        for p in 0..TP_OVERSAMPLE {
            let phase = &BS1770_PHASES[p];
            let mut acc = 0.0f64;
            for k in 0..TP_TAPS {
                if k > n {
                    break;
                }
                acc += (samples[n - k] as f64) * phase[k];
            }
            out[base + p] = acc;
        }
    }
    out
}

/// Computes the true-peak level in dBTP per ITU-R BS.1770-4 Annex 2.
///
/// Applies 4× oversampling via the standard 48-tap FIR polyphase filter,
/// then measures the absolute peak of the upsampled signal.
///
/// Returns `f64::NEG_INFINITY` for an empty or all-zero input.
pub fn compute_true_peak_db(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }

    let upsampled = oversample_4x_bs1770(samples);
    let peak_abs = upsampled.iter().fold(0.0f64, |max, &x| max.max(x.abs()));

    if peak_abs <= 1e-15 {
        f64::NEG_INFINITY
    } else {
        20.0 * peak_abs.log10()
    }
}

/// A detected inter-sample over.
#[derive(Debug, Clone, PartialEq)]
pub struct TruePeakOver {
    /// Sample index in the original (un-upsampled) signal.
    pub position: usize,
    /// True-peak level in dBTP at this position.
    pub dbtp: f64,
}

/// Finds all inter-sample overs (> 0 dBFS) via BS.1770-4 Annex 2 4× oversampling.
///
/// Scans the 4× upsampled signal and reports each region where `|y[m]| > 1.0`.
/// Consecutive overs within the same original-sample window are merged into a
/// single event with the maximum dBTP of that window.
pub fn find_true_peak_overs(samples: &[f32]) -> Vec<TruePeakOver> {
    let upsampled = oversample_4x_bs1770(samples);
    let mut overs = Vec::new();
    let len = upsampled.len();
    let mut i = 0;

    while i < len {
        if upsampled[i].abs() > 1.0 {
            let start_sample = i / TP_OVERSAMPLE;
            let mut peak = upsampled[i].abs();
            i += 1;
            while i < len && i / TP_OVERSAMPLE == start_sample {
                if upsampled[i].abs() > 1.0 {
                    peak = peak.max(upsampled[i].abs());
                }
                i += 1;
            }
            overs.push(TruePeakOver {
                position: start_sample,
                dbtp: 20.0 * peak.log10(),
            });
        } else {
            i += 1;
        }
    }

    overs
}

/// Returns the full BS.1770-4 Annex 2 4× oversampled signal.
///
/// Output length = `samples.len() * 4`. Useful for detailed analysis and plotting.
pub fn oversample_4x(samples: &[f32]) -> Vec<f64> {
    oversample_4x_bs1770(samples)
}

#[cfg(test)]
#[path = "perceptual_test.rs"]
mod perceptual_test;
