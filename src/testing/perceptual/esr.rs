// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Error-to-Signal Ratio (ESR) and Multi-Resolution STFT loss.

use crate::math::dsp::fft::FftPlanner;

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
            .map(|n| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * n as f64 / (ws - 1) as f64).cos()))
            .collect();

        let num_frames = (reference.len() - ws) / hop + 1;
        if num_frames == 0 {
            continue;
        }

        let num_bins = ws / 2 + 1;

        let mut buf_ref_re = vec![0.0f64; ws];
        let mut buf_ref_im = vec![0.0f64; ws];
        let mut buf_test_re = vec![0.0f64; ws];
        let mut buf_test_im = vec![0.0f64; ws];
        let mut mag_ref = vec![0.0f64; num_bins];
        let mut mag_test = vec![0.0f64; num_bins];

        let mut window_loss_sum = 0.0f64;

        for frame in 0..num_frames {
            let offset = frame * hop;

            for i in 0..ws {
                buf_ref_re[i] = reference[offset + i] as f64 * window[i];
                buf_test_re[i] = test[offset + i] as f64 * window[i];
                buf_ref_im[i] = 0.0;
                buf_test_im[i] = 0.0;
            }

            fft.process(&mut buf_ref_re, &mut buf_ref_im);
            fft.process(&mut buf_test_re, &mut buf_test_im);

            let mut frame_peak = 0.0f64;
            for i in 0..num_bins {
                mag_ref[i] = (buf_ref_re[i] * buf_ref_re[i] + buf_ref_im[i] * buf_ref_im[i]).sqrt();
                mag_test[i] =
                    (buf_test_re[i] * buf_test_re[i] + buf_test_im[i] * buf_test_im[i]).sqrt();
                frame_peak = frame_peak.max(mag_ref[i]).max(mag_test[i]);
            }

            let eps_frame = if frame_peak > eps_abs {
                frame_peak * 1e-4
            } else {
                eps_abs
            };

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
