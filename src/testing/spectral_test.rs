// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for spectral fidelity measurement module.

#![cfg(test)]

use super::*;

// =============================================================================
// Common helpers
// =============================================================================

#[test]
fn test_next_power_of_two() {
    assert_eq!(next_power_of_two(0), 1);
    assert_eq!(next_power_of_two(1), 1);
    assert_eq!(next_power_of_two(3), 4);
    assert_eq!(next_power_of_two(1024), 1024);
    assert_eq!(next_power_of_two(1025), 2048);
}

// =============================================================================
// Farina sweep generation
// =============================================================================

#[test]
fn test_generate_farina_sweep_basic() {
    let sweep = generate_farina_sweep(20.0, 20000.0, 1.0, 48000);
    assert_eq!(sweep.len(), 48000);
    // Sweep should start near zero and grow in frequency
    // All samples within [-1, 1]
    for &s in &sweep {
        assert!((-1.0..=1.0).contains(&s));
    }
}

#[test]
fn test_farina_inverse_filter_length() {
    let sweep = generate_farina_sweep(20.0, 20000.0, 0.5, 48000);
    let inv = generate_farina_inverse_filter(&sweep, 20.0, 20000.0, 0.5, 48000);
    assert_eq!(inv.len(), sweep.len());
}

#[test]
fn test_farina_inverse_filter_nonzero() {
    let sweep = generate_farina_sweep(20.0, 20000.0, 0.5, 48000);
    let inv = generate_farina_inverse_filter(&sweep, 20.0, 20000.0, 0.5, 48000);
    let energy: f64 = inv.iter().map(|&x| x * x).sum();
    assert!(energy > 0.0, "Inverse filter must have energy");
    // Must be bounded after normalisation
    for &s in &inv {
        assert!((-1.0..=1.0).contains(&s));
    }
}

#[test]
fn test_deconvolution_delta() {
    // Circular convolution of a unit impulse with inv_filter returns the inv_filter
    let sweep = generate_farina_sweep(100.0, 10000.0, 0.2, 48000);
    let inv = generate_farina_inverse_filter(&sweep, 100.0, 10000.0, 0.2, 48000);

    let mut delta = vec![0.0f64; inv.len()];
    delta[0] = 1.0;
    let deconv = deconvolve_farina(&delta, &inv);

    assert_eq!(deconv.len(), inv.len());
    let mse: f64 = deconv
        .iter()
        .zip(inv.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / inv.len() as f64;
    assert!(
        mse < 1e-10,
        "Deconv of delta with inv_filter should equal inv_filter; MSE={mse:.2e}"
    );
}

// =============================================================================
// Farina measurement validation
// =============================================================================

/// Hard-clip distortion: known to produce odd-order harmonics.
/// Hard-clipping a sine at threshold 0.5 produces ~6.4% THD.
fn hard_clip_f64(x: f64, threshold: f64) -> f64 {
    x.clamp(-threshold, threshold)
}

#[test]
fn test_farina_hard_clip_thd() {
    let f1 = 20.0;
    let f2 = 8000.0;
    let duration_s = 0.5;
    let sr = 48000;

    let result = farina_measure(f1, f2, duration_s, sr, 5, |sweep| {
        sweep
            .iter()
            .map(|&x| hard_clip_f64(x, 0.5) as f32)
            .collect()
    });

    // Linear IR must exist
    assert!(!result.ir_linear.is_empty());
    assert!(!result.fr_magnitude_db.is_empty());

    // Hard-clip must produce measurable THD
    assert!(
        result.thd_total_percent > 1.0,
        "Hard-clip must produce THD > 1%; got {:.1}%",
        result.thd_total_percent
    );

    // THD should be in a plausible range (< 50% for mild clipping)
    assert!(
        result.thd_total_percent < 50.0,
        "THD {:.1}% unexpectedly high",
        result.thd_total_percent
    );

    // Should have detected multiple harmonic orders
    assert!(
        result.thd_by_order.len() >= 3,
        "Expected ≥3 harmonic orders, got {}",
        result.thd_by_order.len()
    );
}

#[test]
fn test_autocorrelation_quality() {
    // The sweep convolved with its inverse filter should approximate a unit delta
    let f1 = 20.0;
    let f2 = 16000.0;
    let duration_s = 1.0;
    let sr = 48000;
    let sweep = generate_farina_sweep(f1, f2, duration_s, sr);
    let inv = generate_farina_inverse_filter(&sweep, f1, f2, duration_s, sr);
    let auto = deconvolve_farina(&sweep, &inv);

    // Find peak
    let max_val = auto.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    assert!(
        max_val > 0.01,
        "Auto-correlation peak too weak: {max_val:.2e}"
    );

    // Find peak position
    let peak_idx = auto
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Check that non-peak values are small
    let noise_floor: f64 = auto
        .iter()
        .enumerate()
        .filter(|&(i, _)| i < peak_idx.saturating_sub(10) || i > peak_idx + 10)
        .map(|(_, &x)| x.abs())
        .fold(0.0f64, f64::max);

    let ratio_db = 20.0 * (noise_floor / max_val).log10();
    assert!(
        ratio_db < -20.0,
        "Auto-correlation side-lobe too high: peak={max_val:.2e}, noise_floor={noise_floor:.2e}, ratio={ratio_db:.1} dB"
    );
}

#[test]
fn test_farina_linear_no_distortion() {
    // A linear gain should produce negligible THD
    let f1 = 20.0;
    let f2 = 16000.0;
    let duration_s = 1.0;
    let sr = 48000;

    let result = farina_measure(f1, f2, duration_s, sr, 5, |sweep| {
        sweep.iter().map(|&x| (2.0 * x) as f32).collect()
    });

    // Linear system should have low THD
    assert!(
        result.thd_total_percent < 10.0,
        "Linear gain should have low THD; got {:.1}%",
        result.thd_total_percent
    );
}

// =============================================================================
// THD+N per AES17
// =============================================================================

#[test]
fn test_notch_biquad_design() {
    let notch = NotchBiquad::design(997.0, 5.0, 48000.0);
    // Coefficients should be finite
    assert!(notch.b0.is_finite());
    assert!(notch.b1.is_finite());
    assert!(notch.b2.is_finite());
    assert!(notch.a1.is_finite());
    assert!(notch.a2.is_finite());
}

#[test]
fn test_notch_attenuates_fundamental() {
    let notch = NotchBiquad::design(997.0, 5.0, 48000.0);
    let signal = generate_sine_f64(997.0, 48000, 48000, 1.0); // 1s of 997 Hz
    let notched = notch.apply(&signal);

    let rms_in: f64 = {
        let sq: f64 = signal.iter().map(|&x| x * x).sum();
        (sq / signal.len() as f64).sqrt()
    };
    let rms_out: f64 = {
        let sq: f64 = notched.iter().map(|&x| x * x).sum();
        (sq / notched.len() as f64).sqrt()
    };

    // Notch should attenuate at least 20 dB
    let attenuation = 20.0 * (rms_out / rms_in).log10();
    assert!(
        attenuation < -20.0,
        "Notch attenuation too weak: {:.1} dB",
        attenuation
    );
}

#[test]
fn test_thdn_linear() {
    // A linear gain should have near-zero THD+N
    let result = measure_thdn(997.0, 48000, 1.0, 5.0, 0, |sweep| {
        sweep.iter().map(|&x| (2.0 * x) as f32).collect()
    });

    assert!(
        result.thdn_percent < 2.0,
        "Linear system THD+N too high: {:.1}%",
        result.thdn_percent
    );
}

#[test]
fn test_thdn_hard_clip() {
    // Hard-clip produces measurable THD+N
    // 0.5 threshold with amplitude 1.0 sine → ~23% THD (heavy clipping)
    let result = measure_thdn(997.0, 48000, 1.0, 5.0, 0, |sweep| {
        sweep
            .iter()
            .map(|&x| hard_clip_f64(x, 0.5) as f32)
            .collect()
    });

    assert!(
        result.thdn_percent > 5.0,
        "Hard-clip THD+N too low: {:.1}%",
        result.thdn_percent
    );
    assert!(
        result.thdn_percent < 40.0,
        "Hard-clip THD+N too high: {:.1}%",
        result.thdn_percent
    );
}

// =============================================================================
// IMD SMPTE/DIN
// =============================================================================

#[test]
fn test_smpte_tones_level() {
    let tones = generate_smpte_tones(60.0, 7000.0, 4.0, 48000, 48000, 0.9);
    assert_eq!(tones.len(), 48000);
    // Signal should be bounded
    for &s in &tones {
        assert!((-1.0..=1.0).contains(&s));
    }
}

#[test]
fn test_smpte_imd_linear() {
    let result = measure_smpte_imd(60.0, 7000.0, 4.0, 48000, 0.5, 0, |tones| {
        tones.iter().map(|&x| (2.0 * x) as f32).collect()
    });

    // Linear system should have very low IMD
    assert!(
        result.imd_percent < 5.0,
        "Linear system IMD too high: {:.1}%",
        result.imd_percent
    );
}

#[test]
fn test_smpte_imd_hard_clip() {
    // Hard-clip should produce measurable IMD
    let result = measure_smpte_imd(60.0, 7000.0, 4.0, 48000, 0.5, 0, |tones| {
        tones
            .iter()
            .map(|&x| hard_clip_f64(x, 0.5) as f32)
            .collect()
    });

    assert!(
        result.imd_percent > 1.0,
        "Hard-clip IMD too low: {:.1}%",
        result.imd_percent
    );
}

// =============================================================================
// generate_sine_f64
// =============================================================================

#[test]
fn test_generate_sine_f64() {
    let sig = generate_sine_f64(440.0, 48000, 48000, 1.0);
    assert_eq!(sig.len(), 48000);
    for &s in &sig {
        assert!((-1.0..=1.0).contains(&s));
    }
    // Should cross zero (not all positive/negative)
    let has_positive = sig.iter().any(|&x| x > 0.1);
    let has_negative = sig.iter().any(|&x| x < -0.1);
    assert!(has_positive && has_negative);
}
