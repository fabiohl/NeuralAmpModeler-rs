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
