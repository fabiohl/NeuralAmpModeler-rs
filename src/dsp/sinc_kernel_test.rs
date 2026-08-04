// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_bessel_i0_known_values() {
    assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
    // I0(1) ≈ 1.2660658...
    assert!((bessel_i0(1.0) - 1.2660658).abs() < 1e-5);
}

#[test]
fn test_sinc_kaiser_dc_unity() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let dc: f64 = kernel.iter().sum();
    assert!((dc - 1.0).abs() < 1e-10, "DC gain must be unity, got {dc}");
}

#[test]
fn test_minimum_phase_causal() {
    let kernel = generate_sinc_kaiser(128, 0.5, 10.0);
    let min_ph = to_minimum_phase(&kernel);
    // Minimum phase concentrates energy at the start — the peak should be
    // in the first 10% of samples.
    let peak_pos = min_ph
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .unwrap()
        .0;
    assert!(
        peak_pos < kernel.len() / 4,
        "Peak should be at the start (minimum phase), found at {peak_pos}/{}",
        kernel.len()
    );
}

#[test]
fn test_minimum_phase_energy_concentration() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let min_ph = to_minimum_phase(&kernel);

    // In minimum phase, energy is concentrated at the start of the impulse.
    // We check that >50% of total energy is in the first 10% of samples.
    let n10 = min_ph.len() / 10;
    let early_energy: f64 = min_ph[..n10].iter().map(|x| x * x).sum();
    let total_energy: f64 = min_ph.iter().map(|x| x * x).sum();
    let ratio = early_energy / total_energy.max(1e-30);
    assert!(
        ratio > 0.5,
        "Minimum phase: >50% of energy should be in the first 10%, got {:.1}%",
        ratio * 100.0
    );

    // Verifies that the kernel is NOT symmetric (confirms it is not linear phase)
    let n = min_ph.len();
    let asym: f64 = (0..n / 2)
        .map(|i| (min_ph[i] - min_ph[n - 1 - i]).abs())
        .sum();
    assert!(
        asym > 0.01,
        "Kernel should be asymmetric (minimum phase), asymmetry={asym:.6}"
    );
}

#[test]
fn test_to_minimum_phase_dc_preservation() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let original_sum: f64 = kernel.iter().sum();
    assert!(
        (original_sum - 1.0).abs() < 1e-10,
        "Original DC must be 1.0, got {original_sum}"
    );

    let min_ph = to_minimum_phase(&kernel);
    let min_sum: f64 = min_ph.iter().sum();
    let sum_err = (min_sum - original_sum).abs();
    println!("original_sum={original_sum}, min_sum={min_sum}, err={sum_err}");
    assert!(
        sum_err < 0.05,
        "Min-phase DC sum must be close to original. Original={original_sum}, min={min_sum}, err={sum_err}"
    );
}

#[test]
fn test_to_minimum_phase_energy_rms() {
    let kernel = generate_sinc_kaiser(256, 0.5, 12.0);
    let original_rms: f64 =
        (kernel.iter().map(|x| x * x).sum::<f64>() / kernel.len() as f64).sqrt();

    let min_ph = to_minimum_phase(&kernel);
    let min_rms: f64 = (min_ph.iter().map(|x| x * x).sum::<f64>() / min_ph.len() as f64).sqrt();

    println!(
        "original_rms={original_rms}, min_rms={min_rms}, ratio={}",
        min_rms / original_rms
    );
    // Energy should be preserved (Parseval)
    let ratio = min_rms / original_rms;
    assert!(
        ratio > 0.9 && ratio < 1.1,
        "Min-phase RMS energy should be within 10% of original. Original={original_rms}, min={min_rms}"
    );
}

#[test]
fn test_polyphase_bank_dimensions() {
    let bank = generate_polyphase_bank(44100, 48000)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(bank.taps_per_phase, TAPS_PER_PHASE);
    // Verifies that all phases are accessible
    for p in 0..NUM_PHASES {
        let c = bank.phase_coeffs(p);
        assert_eq!(c.len(), TAPS_PER_PHASE);
    }
}

#[test]
fn test_polyphase_bank_phase_dc_unity() {
    let bank = generate_polyphase_bank(44100, 48000)
        .expect("construction should succeed for test-sized buffers");
    for p in 0..NUM_PHASES {
        let c = bank.phase_coeffs(p);
        let sum: f32 = c.iter().sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-5 || sum.abs() < 1e-9,
            "Phase {p} DC gain must be ~1.0, got {sum}"
        );
    }
}

#[test]
fn test_polyphase_bank_minphase_dc_unity() {
    let bank = generate_polyphase_bank(22050, 48000)
        .expect("construction should succeed for test-sized buffers");
    for p in 0..NUM_PHASES {
        let c = bank.phase_coeffs(p);
        let sum: f32 = c.iter().sum();
        assert!(
            (sum - 1.0f32).abs() < 1e-5 || sum.abs() < 1e-9,
            "Phase {p} DC gain must be ~1.0, got {sum}"
        );
    }
}

#[test]
fn test_aligned_coeffs_alignment() {
    let bank = generate_polyphase_bank(44100, 48000)
        .expect("construction should succeed for test-sized buffers");
    let ptr = bank.phase_ptr(0) as usize;
    assert_eq!(ptr % 64, 0, "Coefficients must be aligned to 64 bytes");
}
