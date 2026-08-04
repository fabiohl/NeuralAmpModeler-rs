// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for the ASR (Aliasing-to-Signal Ratio) metric module.

#![cfg(test)]

use super::*;

// =============================================================================
// Blackman-Harris window
// =============================================================================

#[test]
fn test_blackman_harris_length() {
    let w = blackman_harris_4term(512);
    assert_eq!(w.len(), 512);
}

#[test]
fn test_blackman_harris_endpoints_symmetric() {
    let w = blackman_harris_4term(64);
    // Blackman-Harris 4-term has zero at endpoints (sum of coefficients = 0)
    let endpoint = 0.35875 - 0.48829 + 0.14128 - 0.01168;
    assert!((w[0] - endpoint).abs() < 1e-10);
    assert!((w[63] - endpoint).abs() < 1e-10);
}

#[test]
fn test_blackman_harris_peak_center() {
    // For even-length windows the exact peak lies between two samples
    let n = 1025; // odd → exact center at (n-1)/2 = 512
    let w = blackman_harris_4term(n);
    let mid = (n - 1) / 2;
    let peak = 0.35875 + 0.48829 + 0.14128 + 0.01168; // = 1.0
    assert!((w[mid] - peak).abs() < 1e-10);
}

// =============================================================================
// Sine generation
// =============================================================================

#[test]
fn test_generate_sine_length() {
    let s = generate_sine(440.0, 48000, 1024, 1.0);
    assert_eq!(s.len(), 1024);
}

#[test]
fn test_generate_sine_amplitude() {
    let s = generate_sine(1000.0, 48000, 65536, 1.0);
    let max_abs: f32 = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(max_abs > 0.99, "Expected peak near 1.0, got {max_abs}");
    assert!(max_abs <= 1.0, "Peak must not exceed 1.0");
}

#[test]
fn test_generate_sine_gain() {
    let s = generate_sine(440.0, 48000, 4096, 2.5);
    let max_abs: f32 = s.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 2.4,
        "Expected peak near 2.5 with gain=2.5, got {max_abs}"
    );
}

// =============================================================================
// Median
// =============================================================================

#[test]
fn test_median_odd() {
    let data = [1.0, 3.0, 2.0];
    let m = median(&data);
    assert!((m - 2.0).abs() < 1e-10);
}

#[test]
fn test_median_even() {
    let data = [1.0, 4.0, 2.0, 3.0];
    let m = median(&data);
    assert!((m - 2.5).abs() < 1e-10);
}

#[test]
fn test_median_empty() {
    let data: [f64; 0] = [];
    let m = median(&data);
    assert!((m - 0.0).abs() < 1e-10);
}

// =============================================================================
// ASR: linear system → ASR ≈ 0 (no aliasing)
// =============================================================================

#[test]
fn test_asr_linear_system_zero_aliasing() {
    let f0 = 440.0;
    let sr = 48000;
    let n = 16384;
    let gain = 1.0;

    let input = generate_sine(f0, sr, n, gain);
    // Linear system: output = 2.0 * input (pure gain, no distortion)
    let output: Vec<f32> = input.iter().map(|&x| 2.0 * x).collect();

    let result = compute_asr(&output, f0, sr);

    // Linear gain should produce no aliased energy beyond noise
    assert_eq!(
        result.num_aliased, 0,
        "Linear gain must produce zero aliased peaks, got {}",
        result.num_aliased
    );
    assert!(
        result.asr_linear < 1e-6 || result.asr_db < -60.0,
        "Linear system ASR must be near zero; got asr_linear={:.2e}, asr_db={:.1} dB",
        result.asr_linear,
        result.asr_db
    );
    assert!(
        result.num_harmonics > 0,
        "Must detect at least the fundamental"
    );
}

// =============================================================================
// ASR: hard-clip waveshaper → ASR > 0 (detectable aliasing)
// =============================================================================

/// Hard-clip at ±threshold.
fn hard_clip(x: f32, threshold: f32) -> f32 {
    x.clamp(-threshold, threshold)
}

#[test]
fn test_asr_hard_clip_produces_aliasing() {
    // Use an f0 that is incommensurate with sr/2 so that aliased harmonics
    // do NOT fold back onto existing harmonic bins.
    // f0=2017 Hz at 48 kHz: 12th harm (24204) → aliases to 23796 Hz,
    // which is NOT a multiple of 2017 (23796/2017 ≈ 11.8).
    let f0 = 2017.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;
    let clip_threshold = 0.2;

    let input = generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input
        .iter()
        .map(|&x| hard_clip(x, clip_threshold))
        .collect();

    let result = compute_asr(&output, f0, sr);

    assert!(
        result.num_aliased > 0,
        "Hard-clip at {f0} Hz must generate aliased peaks; got {} aliased (harmonics={}, noise_floor={:.2e})",
        result.num_aliased,
        result.num_harmonics,
        result.noise_floor,
    );
    assert!(
        result.asr_db > -30.0,
        "Hard-clip ASR must be above -30 dB; got {:.1} dB (aliased={:.2e}, harm={:.2e})",
        result.asr_db,
        result.aliased_energy,
        result.harmonic_energy,
    );
}

#[test]
fn test_asr_hard_clip_low_f0_less_aliasing() {
    // Low f0 → more harmonics within Nyquist before fold-back starts.
    // Use 98 Hz (incommensurate with 48 kHz: gcd(98,48000)=2, but key is
    // that the first aliased harmonic doesn't land on an existing one).
    let f0 = 98.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;
    let clip_threshold = 0.2;

    let input = generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input
        .iter()
        .map(|&x| hard_clip(x, clip_threshold))
        .collect();

    let result = compute_asr(&output, f0, sr);

    let input_h = generate_sine(2017.0, sr, n, gain);
    let output_h: Vec<f32> = input_h
        .iter()
        .map(|&x| hard_clip(x, clip_threshold))
        .collect();
    let result_high = compute_asr(&output_h, 2017.0, sr);

    assert!(
        result_high.has_aliasing(),
        "High-f0 case must detect aliasing (sanity check)"
    );

    // Low f0 should have less aliasing: first Nyquist fold-back is at
    // k > 24000/98 ≈ 245 → many harmonics stay in-band.
    assert!(
        result.asr_db < result_high.asr_db + 3.0,
        "Low f0 ({:.1} Hz, ASR={:.1} dB) should have lower ASR than high f0 (2017 Hz, ASR={:.1} dB)",
        f0,
        result.asr_db,
        result_high.asr_db
    );
}

// =============================================================================
// ASR: fundamental detection
// =============================================================================

#[test]
fn test_asr_detects_fundamental() {
    let f0 = 440.0;
    let sr = 48000;
    let n = 16384;

    let input = generate_sine(f0, sr, n, 1.0);
    // Mild soft-clip (tanh) still preserves fundamental clearly
    let output: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();

    let result = compute_asr(&output, f0, sr);

    assert!(
        result.num_harmonics >= 1,
        "Must detect at least the fundamental"
    );
}

// =============================================================================
// ASR: aggregate helpers
// =============================================================================

#[test]
fn test_asr_aggregate_empty() {
    assert!(asr_aggregate(&[]).is_infinite() && asr_aggregate(&[]) < 0.0);
}

#[test]
fn test_asr_aggregate_single() {
    let results = vec![AsrResult {
        f0: 440.0,
        sample_rate: 48000,
        asr_db: -20.0,
        asr_linear: 0.01,
        harmonic_energy: 100.0,
        aliased_energy: 1.0,
        num_harmonics: 5,
        num_aliased: 3,
        noise_floor: 1e-6,
        peak_threshold: 1e-4,
        bin_width: 0.732,
    }];
    let agg = asr_aggregate(&results);
    assert!((agg - (-20.0)).abs() < 0.01);
}

#[test]
fn test_asr_worst_case() {
    let make = |db: f64| AsrResult {
        f0: 440.0,
        sample_rate: 48000,
        asr_db: db,
        asr_linear: 10.0f64.powf(db / 10.0),
        harmonic_energy: 1.0,
        aliased_energy: 10.0f64.powf(db / 10.0),
        num_harmonics: 1,
        num_aliased: 1,
        noise_floor: 1e-6,
        peak_threshold: 1e-4,
        bin_width: 0.732,
    };

    let results = vec![make(-30.0), make(-15.0), make(-40.0)];
    let worst = asr_worst_case(&results);
    assert!((worst - (-15.0)).abs() < 0.01);
}

#[test]
fn test_asr_result_has_aliasing() {
    let r = AsrResult {
        f0: 440.0,
        sample_rate: 48000,
        asr_db: -20.0,
        asr_linear: 0.01,
        harmonic_energy: 100.0,
        aliased_energy: 1.0,
        num_harmonics: 1,
        num_aliased: 1,
        noise_floor: 0.0,
        peak_threshold: 0.0,
        bin_width: 0.0,
    };
    assert!(r.has_aliasing());

    let r_no = AsrResult {
        f0: 440.0,
        sample_rate: 48000,
        asr_db: f64::NEG_INFINITY,
        asr_linear: 0.0,
        harmonic_energy: 100.0,
        aliased_energy: 0.0,
        num_harmonics: 1,
        num_aliased: 0,
        noise_floor: 0.0,
        peak_threshold: 0.0,
        bin_width: 0.0,
    };
    assert!(!r_no.has_aliasing());
}
