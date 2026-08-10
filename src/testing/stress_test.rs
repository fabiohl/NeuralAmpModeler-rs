// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for stress signal generators (v1 and v2).
//!
//! Extracted from `stress.rs` per testing.md convention (files ≥ 300 LoC
//! must keep tests in a separate `_test.rs` file).

use super::*;

#[test]
fn test_v1_deterministic() {
    let a = generate_stress_signal_v1();
    let b = generate_stress_signal_v1();
    assert_eq!(a.len(), 2048);
    assert_eq!(a, b);
}

#[test]
fn test_v1_not_silent() {
    let sig = generate_stress_signal_v1();
    let power: f64 = sig.iter().map(|&x| (x as f64).powi(2)).sum();
    assert!(power > 1.0, "v1 signal is too quiet");
}

#[test]
fn test_v2_deterministic() {
    let a = generate_stress_signal_v2_default(48000);
    let b = generate_stress_signal_v2_default(48000);
    assert_eq!(a.len(), 48000 * 5);
    assert_eq!(a, b);
}

#[test]
fn test_v2_valid_sizes() {
    for &sr in SUPPORTED_SAMPLE_RATES {
        let sig = generate_stress_signal_v2_default(sr);
        assert_eq!(
            sig.len() as u32,
            (sr as f64 * STRESS_V2_DURATION) as u32,
            "wrong length for SR={sr}"
        );
        let power: f64 = sig.iter().map(|&x| (x as f64).powi(2)).sum();
        assert!(power > 10.0, "v2 signal too quiet for SR={sr}");
    }
}

#[test]
fn test_v2_non_silent_segments() {
    let sig = generate_stress_signal_v2_default(48000);
    let sr = 48000;

    // Check energy in each segment
    let segments = [
        (0, sr),
        (sr, 2 * sr),
        (2 * sr, (2 * sr + sr / 2)),
        ((2.5 * sr as f64) as usize, (3.5 * sr as f64) as usize),
        ((3.5 * sr as f64) as usize, (4.5 * sr as f64) as usize),
        ((4.5 * sr as f64) as usize, sig.len()),
    ];

    for (start, end) in segments {
        let power: f64 = sig[start..end].iter().map(|&x| (x as f64).powi(2)).sum();
        assert!(
            power > 1.0,
            "segment {start}..{end} has too little energy: {power}"
        );
    }
}

#[test]
fn test_finitude_check() {
    let clean = vec![0.0f32, 0.5, -0.2, 0.95];
    assert!(check_finitude(&clean));

    let with_nan = vec![0.0f32, f32::NAN, -0.2];
    assert!(!check_finitude(&with_nan));

    let with_inf = vec![0.0f32, f32::INFINITY, -0.2];
    assert!(!check_finitude(&with_inf));
}

#[test]
fn test_rms_and_peak_dbfs() {
    // 0 dBFS peak sine: RMS is 1/sqrt(2) ≈ -3.01 dBFS
    let sr = 48000;
    let n = 48000;
    let sine: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin() as f32)
        .collect();

    let rms = compute_rms_dbfs(&sine);
    let peak = compute_peak_dbfs(&sine);

    assert!(
        (rms - -3.01).abs() < 0.1,
        "RMS should be ~-3.01 dBFS, got {rms}"
    );
    assert!(
        (peak - 0.0).abs() < 0.1,
        "Peak should be ~0.0 dBFS, got {peak}"
    );

    // Silence
    let silence = vec![0.0f32; 100];
    assert_eq!(compute_rms_dbfs(&silence), f64::NEG_INFINITY);
    assert_eq!(compute_peak_dbfs(&silence), f64::NEG_INFINITY);
}

#[test]
fn test_evaluate_signal_energy() {
    let sig = generate_stress_signal_v1();
    let eval = evaluate_signal_energy(&sig, -80.0);
    assert!(eval.is_finite, "v1 signal must be 100% finite");
    assert!(eval.is_active, "v1 signal must be active above -80 dBFS");
    assert!(
        eval.rms_dbfs > -30.0,
        "v1 signal RMS energy expected > -30 dBFS, got {}",
        eval.rms_dbfs
    );
}

#[test]
fn test_block_invariance_helper() {
    use crate::loader::nam_json::LinearImplementation;
    use crate::models::linear::LinearModel;
    let sig = generate_stress_signal_v1();
    let res = verify_block_invariance_for_model(
        || Box::new(LinearModel::new(vec![1.0], 0.0, LinearImplementation::Direct).unwrap()),
        &sig,
        &[1, 8, 32, 64, 128, 512],
        64,
        1e-6,
    );

    assert!(
        res.is_invariant,
        "Linear passthrough model must be 100% block-size invariant"
    );
    assert_eq!(res.max_abs_error, 0.0);
    assert_eq!(res.errors_by_block_size.len(), 6);
}
