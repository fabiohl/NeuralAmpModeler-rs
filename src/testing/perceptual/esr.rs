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
/// Returns `0.0` if both reference and test signals have zero energy.
/// Returns `f64::INFINITY` if the reference signal has zero energy but test has positive energy.
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
    if signal_power == 0.0 {
        if noise_power == 0.0 {
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
///
/// Returns [`f64::NEG_INFINITY`] if `esr == 0.0`. For positive subnormal or
/// small ESR values, returns the finite dB representation `10 * log10(esr)`.
pub fn esr_to_db(esr: f64) -> f64 {
    if esr == 0.0 {
        f64::NEG_INFINITY
    } else {
        10.0 * esr.log10()
    }
}

/// Computes Signal-to-Noise Ratio (SNR) in dB between reference and test.
///
/// Returns [`f64::INFINITY`] if the signals are bit-identical (`noise_power == 0.0`).
/// Returns [`f64::NEG_INFINITY`] if the reference signal has zero energy (`signal_power == 0.0`)
/// but noise is positive. Otherwise returns `10 * log10(signal_power / noise_power)`.
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
    if noise_power == 0.0 {
        return f64::INFINITY;
    }
    if signal_power == 0.0 {
        return f64::NEG_INFINITY;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_case_a_constant_1e4_one_ulp_error() {
        // (a) Constant signal 1e-4, 1 ULP f32 error -> SNR finite (~140 dB).
        // Before B1 fix, noise_power <= f64::EPSILON caused SNR = infinity.
        let r = vec![1e-4f32; 100];
        let t = vec![f32::from_bits(1e-4f32.to_bits() + 1); 100];
        let snr = compute_snr_db(&r, &t);
        assert!(
            snr.is_finite(),
            "SNR must be finite for 1 ULP error, got {snr}"
        );
        assert!(
            snr > 130.0 && snr < 150.0,
            "SNR out of expected range: {snr}"
        );
    }

    #[test]
    fn test_snr_case_b_identical_vectors() {
        // (b) Identical vectors -> SNR = infinity
        let r = vec![0.5f32, -0.2, 0.1];
        assert_eq!(compute_snr_db(&r, &r), f64::INFINITY);
    }

    #[test]
    fn test_snr_case_c_low_amplitude_small_relative_error() {
        // (c) Signal ≈ 1e-9 amplitude (signal_power < f64::EPSILON) with relative error 1e-6 -> SNR ≈ 120 dB.
        // Before B1 fix, noise_power (1e-30) <= f64::EPSILON caused SNR = infinity,
        // and signal_power (1e-18) <= f64::EPSILON caused ESR = 0.
        let r = vec![1e-9f32; 100];
        let t = vec![1e-9f32 * (1.0 + 1e-6); 100];
        let snr = compute_snr_db(&r, &t);
        assert!(
            snr.is_finite(),
            "SNR must be finite for low amplitude signal, got {snr}"
        );
        assert!(
            (snr - 120.0).abs() < 1.0,
            "SNR expected ≈ 120 dB, got {snr}"
        );

        let esr = compute_esr(&r, &t);
        assert!(esr > 0.0, "ESR must be positive, got {esr}");
        assert!(
            (esr - 1e-12).abs() < 1e-14,
            "ESR expected ≈ 1e-12, got {esr}"
        );
    }

    #[test]
    fn test_esr_case_d_zero_signal_zero_noise() {
        // (d) signal_power == 0 and noise_power == 0 -> ESR 0
        let r = vec![0.0f32; 10];
        let t = vec![0.0f32; 10];
        assert_eq!(compute_esr(&r, &t), 0.0);
    }

    #[test]
    fn test_esr_case_e_zero_signal_positive_noise() {
        // (e) signal_power == 0 and noise_power > 0 -> ESR infinity
        let r = vec![0.0f32; 10];
        let t = vec![0.1f32; 10];
        assert_eq!(compute_esr(&r, &t), f64::INFINITY);
    }

    #[test]
    fn test_snr_reproduce_convnet_silu_log_value() {
        // Reproduce log: noise_power ≈ 2.1e-16, signal_power ≈ 6.4e-4 -> SNR ≈ 124.9 dB.
        // // Measured: SNR=124.9 dB (convnet_silu)
        let n = 2048;
        let r_val = (6.44e-4f64 / n as f64).sqrt() as f32;
        let r = vec![r_val; n];
        let r_bits = r_val.to_bits();
        // 948 samples with +6 ULPs and 1100 samples with +5 ULPs
        // produces noise_power ≈ 2.088e-16 and SNR ≈ 124.9 dB.
        let mut t = Vec::with_capacity(n);
        for i in 0..n {
            let offset = if i < 948 { 6 } else { 5 };
            t.push(f32::from_bits(r_bits + offset));
        }

        let snr = compute_snr_db(&r, &t);
        assert!(
            snr.is_finite(),
            "SNR must be finite for convnet_silu parity, got {snr}"
        );
        assert!(
            (snr - 124.9).abs() < 0.1,
            "Expected SNR ≈ 124.9 dB, got {snr}"
        );
    }
}
