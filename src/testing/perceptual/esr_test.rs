// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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
