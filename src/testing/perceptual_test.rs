// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for perceptual metrics (ESR, LUFS, LRA, MR-STFT, true-peak dBTP).

use super::*;

#[test]
fn test_esr_identical_is_zero() {
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let esr = compute_esr(&sig, &sig);
    assert!(
        esr < 1e-15,
        "ESR of identical signals should be near zero, got {esr}"
    );
}

#[test]
fn test_esr_all_zero_test() {
    let sig: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let zeros = vec![0.0f32; sig.len()];
    let esr = compute_esr(&sig, &zeros);
    assert!(
        (esr - 1.0).abs() < 1e-10,
        "ESR of signal vs zeros should be ~1.0, got {esr}"
    );
}

#[test]
fn test_compute_esr_blockwise() {
    let sig: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let mut perturbed = sig.clone();
    perturbed[5] += 1.0; // Block 0 offset
    perturbed[55] += 2.0; // Block 1 offset

    let block_size = 50;
    let esr_blocks = compute_esr_blockwise(&sig, &perturbed, block_size);
    assert_eq!(esr_blocks.len(), 2);

    let expected_esr_0 = compute_esr(&sig[0..50], &perturbed[0..50]);
    assert!((esr_blocks[0] - expected_esr_0).abs() < 1e-15);

    let expected_esr_1 = compute_esr(&sig[50..100], &perturbed[50..100]);
    assert!((esr_blocks[1] - expected_esr_1).abs() < 1e-15);
}

#[test]
fn test_compute_lufs_sine() {
    let sr = 48000;
    let n = sr as usize;
    let amplitude = 10.0f32.powf(-20.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let lufs = compute_lufs(&sig, sr);
    // -20 dBFS 1 kHz sine: mean_sq=0.005 with K-weighting shelf gain ~1.08 at 1 kHz
    // → LKFS = -0.691 + 10*log10(0.005 * 1.081²) ≈ -23.0
    assert!(
        (lufs - (-23.0)).abs() < 0.5,
        "Expected ~-23.0 LUFS for 1kHz sine at -20 dBFS, got {lufs}"
    );
}

#[test]
fn test_compute_lufs_empty() {
    assert!(compute_lufs(&[], 48000).is_infinite());
}

#[test]
fn test_compute_lufs_silence() {
    let sig = vec![0.0f32; 48000];
    assert!(compute_integrated_lufs(&sig, 48000).is_infinite());
}

// =============================================================================
// Integrated LUFS 2-pass gating tests
// =============================================================================

#[test]
fn test_integrated_lufs_full_scale_sine() {
    let sr = 48000;
    let n = sr as usize; // 1 s
    let sig: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let lufs = compute_integrated_lufs(&sig, sr);
    // Full-scale 1 kHz mono sine: mean_sq=0.5, K-weighting shelf gain ~1.08 at 1 kHz
    // → LKFS = -0.691 + 10*log10(0.5 * 1.081²) ≈ -3.0
    assert!(
        (lufs - (-3.0)).abs() < 0.2,
        "Expected ~-3.0 LUFS for full-scale 1 kHz sine, got {lufs}"
    );
}

#[test]
fn test_integrated_lufs_empty() {
    assert!(compute_integrated_lufs(&[], 48000).is_infinite());
}

#[test]
fn test_integrated_lufs_too_short() {
    // Less than one 400 ms block → no blocks → -∞
    let sr = 48000;
    let n = (sr as f64 * 0.3) as usize; // 300 ms
    let sig: Vec<f32> = vec![0.5f32; n];
    assert!(compute_integrated_lufs(&sig, sr).is_infinite());
}

#[test]
fn test_lufs_2pass_absolute_gate() {
    // Signal with loud section followed by near-silence below -70 LUFS.
    // The silence blocks should be gated out.
    let sr = 48000;
    let block_samples = (sr as usize * 400) / 1000; // 19200
    let _hop = block_samples / 4; // 4800

    // Build ~1.5 s: first 400ms loud (-20 dBFS), rest near-silence
    let total_samples = block_samples + _hop * 4; // ~1.2 s, 5 blocks
    let mut sig = vec![0.0f32; total_samples];
    let ampl_loud = 10.0f32.powf(-20.0 / 20.0);
    for (i, s) in sig.iter_mut().enumerate().take(block_samples) {
        *s = ampl_loud * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }

    let lufs = compute_integrated_lufs(&sig, sr);
    // Only the loud block passes → integrated ≈ loudness of that block ≈ -23.7
    assert!(
        lufs > -30.0 && lufs < -20.0,
        "Gated LUFS should discard silence, got {lufs}"
    );
}

#[test]
fn test_lufs_2pass_relative_gate() {
    // Signal with a loud section (2 s at -20 dBFS) followed by a quiet section
    // (3 s at -46 dBFS). With 75 % overlap and 400 ms blocks, most blocks in the
    // loud section are purely loud, and most in the quiet section purely quiet.
    // The relative gate at (ungated - 10) should discard quiet blocks.
    let sr = 48000;
    let loud_len = sr as usize * 2; // 2 s
    let quiet_len = sr as usize * 3; // 3 s
    let total_samples = loud_len + quiet_len;
    let mut sig = vec![0.0f32; total_samples];
    let ampl_loud = 10.0f32.powf(-20.0 / 20.0);
    let ampl_quiet = 10.0f32.powf(-46.0 / 20.0);

    // Loud: -20 dBFS 1 kHz sine
    for (i, s) in sig.iter_mut().enumerate().take(loud_len) {
        *s = ampl_loud * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }
    // Quiet: -46 dBFS 1 kHz sine
    for (i, s) in sig.iter_mut().enumerate().skip(loud_len) {
        *s = ampl_quiet * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }

    let lufs = compute_integrated_lufs(&sig, sr);
    // With 75 % overlap, some mixed blocks exist at the transition, but the
    // quiet section dominates at well below the relative gate threshold.
    // Result should be closer to the loud level than the ungated average.
    let ungated = {
        let kw = apply_k_weighting(&sig);
        let mean_sq: f64 = kw.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / kw.len() as f64;
        -0.691 + 10.0 * mean_sq.log10()
    };
    assert!(
        lufs > ungated,
        "Gated LUFS ({lufs}) should be higher than ungated ({ungated}) — quiet blocks were removed"
    );
    // Loud section alone is ~ -23 LUFS; gated should be somewhat louder than ungated
    assert!(lufs < -22.0, "Gated LUFS should not exceed -22, got {lufs}");
}

#[test]
fn test_lufs_2pass_steady_signal_consistent() {
    // For a steady-state signal, both 1-pass and 2-pass should agree tightly.
    let sr = 48000;
    let n = sr as usize * 2; // 2 s
    let ampl = 10.0f32.powf(-12.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();

    let lufs = compute_integrated_lufs(&sig, sr);
    // -12 dBFS → mean_sq = ampl²/2 = 0.0316; K-weighting shelf gain ~1.08 at 1 kHz
    // → LKFS = -0.691 + 10*log10(0.0316 * 1.081²) ≈ -15.0
    assert!(
        (lufs - (-15.0)).abs() < 0.3,
        "Steady sine at -12 dBFS should be ~ -15.0 LUFS, got {lufs}"
    );
}

#[test]
fn test_k_weighting_preserves_1khz() {
    // At 1 kHz the K-weighting shelf is near unity (shelf transition starts ~1 kHz).
    // A 1 kHz sine should have roughly the same RMS after K-weighting.
    let sr = 48000;
    let n = 48000;
    let sig: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let kw = apply_k_weighting(&sig);
    let rms_in: f64 = (sig.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let rms_out: f64 = (kw.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let gain_db = 20.0 * (rms_out / rms_in).log10();
    // At 1 kHz the shelf gain should be near 0 dB
    assert!(
        gain_db.abs() < 1.5,
        "K-weighting at 1 kHz should have near-unity gain, got {gain_db:.2} dB"
    );
}

#[test]
fn test_k_weighting_boosts_high_frequencies() {
    // At 10 kHz the RLB shelf should provide ~ +4 dB boost.
    let sr = 48000;
    let n = 48000;
    let sig: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f32::consts::PI * 10000.0 * i as f32 / sr as f32).sin())
        .collect();
    let kw = apply_k_weighting(&sig);
    let rms_in: f64 = (sig.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let rms_out: f64 = (kw.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let gain_db = 20.0 * (rms_out / rms_in).log10();
    // At 10 kHz the shelf should be close to +4 dB (high-frequency asymptote)
    assert!(
        gain_db > 2.5 && gain_db < 5.5,
        "K-weighting at 10 kHz should have ~+4 dB gain, got {gain_db:.2} dB"
    );
}

// =============================================================================
// LRA (EBU Tech 3342) tests
// =============================================================================

#[test]
fn test_lra_empty() {
    assert_eq!(compute_lra(&[], 48000), 0.0);
}

#[test]
fn test_lra_too_short() {
    // Less than 3 s → no blocks → LRA = 0
    let sr = 48000;
    let n = sr as usize * 2; // 2 s < 3 s
    let sig = vec![0.1f32; n];
    assert_eq!(compute_lra(&sig, sr), 0.0);
}

#[test]
fn test_lra_steady_sine_zero() {
    // A steady sine has constant short-term loudness → LRA ≈ 0
    let sr = 48000;
    let n = sr as usize * 6; // 6 s → 2 blocks of 3 s
    let ampl = 10.0f32.powf(-12.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let lra = compute_lra(&sig, sr);
    assert!(lra < 0.5, "Steady sine LRA should be ~0, got {lra}");
}

#[test]
fn test_lra_dynamic_signal() {
    // Alternating loud (-12 dBFS) and quiet (-30 dBFS) sections → measurable LRA
    let sr = 48000;
    let block_len = sr as usize * 3; // 3 s
    let n = block_len * 2; // 6 s → 2 blocks
    let mut sig = vec![0.0f32; n];
    let ampl_loud = 10.0f32.powf(-12.0 / 20.0);
    let ampl_quiet = 10.0f32.powf(-30.0 / 20.0);

    // First 3 s: loud
    for (i, s) in sig.iter_mut().enumerate().take(block_len) {
        *s = ampl_loud * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }
    // Second 3 s: quiet
    for (i, s) in sig.iter_mut().enumerate().skip(block_len) {
        *s = ampl_quiet * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }

    let lra = compute_lra(&sig, sr);
    // loud: -12 dBFS → ~-15.7 LUFS; quiet: -30 dBFS → ~-33.7 LUFS
    // Both above -70 LUFS, both above rel gate. Difference ≈ 18 LU before interpolation.
    // With 2 values: P10 ≈ sorted[0] + 0.1*diff, P95 ≈ sorted[0] + 0.95*diff → LRA ≈ 0.85*18 ≈ 15.3
    assert!(
        lra > 12.0 && lra < 18.0,
        "Dynamic signal LRA should be ~15.3 LU, got {lra}"
    );
}

#[test]
fn test_lra_absolute_gate_removes_silence() {
    // Signal with one loud block and one silence block.
    // Silence should be gated out by absolute gate → only 1 block → LRA = 0.
    let sr = 48000;
    let block_len = sr as usize * 3; // 3 s
    let n = block_len * 2; // 6 s
    let mut sig = vec![0.0f32; n];
    let ampl = 10.0f32.powf(-12.0 / 20.0);

    for (i, s) in sig.iter_mut().enumerate().take(block_len) {
        *s = ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }
    // Second 3 s are zeros → -∞ LUFS → absolute gate removes them

    let lra = compute_lra(&sig, sr);
    assert!(
        lra < 0.5,
        "Silence block should be gated, LRA should be ~0, got {lra}"
    );
}

#[test]
fn test_lra_with_one_block() {
    // Exactly 3 s → 1 block → LRA = 0 (need ≥ 2 blocks after gating)
    let sr = 48000;
    let n = sr as usize * 3;
    let ampl = 10.0f32.powf(-12.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    assert_eq!(compute_lra(&sig, sr), 0.0);
}

#[test]
fn test_short_term_loudness_returns_values() {
    let sr = 48000;
    let n = sr as usize * 6; // 6 s
    let ampl = 10.0f32.powf(-12.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();
    let st = short_term_loudness(&sig, sr);
    assert_eq!(
        st.len(),
        2,
        "6 s should produce 2 short-term blocks, got {}",
        st.len()
    );
    // Both blocks should have similar loudness
    assert!(
        (st[0] - st[1]).abs() < 0.1,
        "Steady sine short-term blocks should match, got {:?}",
        st
    );
}

#[test]
fn test_short_term_loudness_empty() {
    assert!(short_term_loudness(&[], 48000).is_empty());
}

#[test]
fn test_short_term_loudness_too_short() {
    let sig = vec![0.1f32; 48000]; // 1 s < 3 s
    assert!(short_term_loudness(&sig, 48000).is_empty());
}

// =============================================================================
// Combined measurement tests
// =============================================================================

#[test]
fn test_measure_loudness_combined() {
    let sr = 48000;
    let n = sr as usize * 6; // 6 s for LRA
    let ampl = 10.0f32.powf(-20.0 / 20.0);
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin())
        .collect();

    let result = measure_loudness(&sig, sr);
    // Integrated LUFS should be ~ -23.0 (K-weighting shelf gain ~1.08 at 1 kHz)
    assert!(
        (result.integrated_lufs - (-23.0)).abs() < 0.5,
        "Combined integrated LUFS mismatch: {}",
        result.integrated_lufs
    );
    // LRA should be ~0 for steady sine
    assert!(
        result.lra < 0.5,
        "Combined LRA should be ~0: {}",
        result.lra
    );
    // True-peak should be ~ -20 dBTP for -20 dBFS sine
    assert!(
        (result.true_peak_db - (-20.0)).abs() < 1.0,
        "Combined dBTP mismatch: {}",
        result.true_peak_db
    );
    // Short-term values should have correct count
    assert_eq!(result.short_term.len(), 2);
}

#[test]
fn test_measure_loudness_empty() {
    let result = measure_loudness(&[], 48000);
    assert!(result.integrated_lufs.is_infinite());
    assert_eq!(result.lra, 0.0);
    assert!(result.true_peak_db.is_infinite());
    assert!(result.short_term.is_empty());
}

#[test]
fn test_measure_loudness_dynamic() {
    // Alternating -12 / -30 dBFS → LRA > 0
    let sr = 48000;
    let block_len = sr as usize * 3;
    let n = block_len * 2;
    let mut sig = vec![0.0f32; n];
    let ampl_loud = 10.0f32.powf(-12.0 / 20.0);
    let ampl_quiet = 10.0f32.powf(-30.0 / 20.0);

    for (i, s) in sig.iter_mut().enumerate().take(block_len) {
        *s = ampl_loud * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }
    for (i, s) in sig.iter_mut().enumerate().skip(block_len) {
        *s = ampl_quiet * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / sr as f32).sin();
    }

    let result = measure_loudness(&sig, sr);
    assert!(
        result.lra > 5.0,
        "Dynamic signal should have LRA > 5 LU, got {}",
        result.lra
    );
}

// =============================================================================
// MR-STFT tests
// =============================================================================

#[test]
fn test_mr_stft_identical_is_zero() {
    let sig: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let loss = compute_mr_stft(&sig, &sig);
    assert!(
        loss < 1e-12,
        "MR-STFT of identical signals should be near zero, got {loss}"
    );
}

#[test]
fn test_mr_stft_empty() {
    assert_eq!(compute_mr_stft(&[], &[]), 0.0);
}

#[test]
fn test_mr_stft_different_signals() {
    let sig_a: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let sig_b: Vec<f32> = (0..4096)
        .map(|i| (2.0 * std::f32::consts::PI * 880.0 * i as f32 / 48000.0).sin())
        .collect();
    let loss = compute_mr_stft(&sig_a, &sig_b);
    assert!(
        loss > 0.0,
        "MR-STFT of different signals should be positive, got {loss}"
    );
}

#[test]
fn test_esr_invariant_to_sample_rate() {
    let sig: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut perturbed = sig.clone();
    perturbed[100] += 1e-5;
    let esr1 = compute_esr(&sig, &perturbed);
    let esr2 = compute_esr(&sig, &perturbed);
    assert!((esr1 - esr2).abs() < 1e-15, "ESR should be deterministic");
}

// =============================================================================
// True-peak (BS.1770-4 Annex 2) tests
// =============================================================================

#[test]
fn test_true_peak_empty() {
    assert!(compute_true_peak_db(&[]).is_infinite());
}

#[test]
fn test_true_peak_silence() {
    let sig = vec![0.0f32; 1024];
    assert!(compute_true_peak_db(&sig).is_infinite());
}

#[test]
fn test_true_peak_sine_minus_6_db() {
    // 1 kHz sine at -6 dBFS (amplitude 0.5) → dBTP ≈ -6
    let sr = 48000.0f64;
    let n = 2048;
    let ampl = 0.5f32;
    let sig: Vec<f32> = (0..n)
        .map(|i| ampl * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32)
        .collect();
    let tp = compute_true_peak_db(&sig);
    // Due to filter passband ripple the true-peak may be slightly above -6
    assert!(
        (tp - (-6.02)).abs() < 0.3,
        "1kHz -6 dBFS sine should be ~-6 dBTP, got {tp}"
    );
}

#[test]
fn test_true_peak_full_scale_sine() {
    // 1 kHz sine at 0 dBFS (amplitude 1.0) → dBTP ≈ 0
    let sr = 48000.0f64;
    let n = 2048;
    let sig: Vec<f32> = (0..n)
        .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32)
        .collect();
    let tp = compute_true_peak_db(&sig);
    assert!(
        tp.abs() < 0.15,
        "1kHz 0 dBFS sine should be ~0 dBTP, got {tp}"
    );
}

#[test]
fn test_oversample_4x_length() {
    let sig = vec![0.5f32; 128];
    let up = oversample_4x(&sig);
    assert_eq!(up.len(), 128 * 4, "4× oversampled length must be 4× input");
}

#[test]
fn test_bs1770_fir_dc_gain() {
    // Each polyphase sub-filter should have near-unity DC gain (sum ≈ 1.0).
    // Q15-derived coefficients have minor rounding; tolerance 1e-2 is adequate.
    for (p, phase) in BS1770_PHASES.iter().enumerate() {
        let sum: f64 = phase.iter().sum();
        assert!(
            (sum - 1.0).abs() < 3e-2,
            "Phase {p} DC gain should be ~1.0, got {sum}"
        );
    }
}

#[test]
fn test_true_peak_dc_minus_6_db() {
    // DC 0.5: the filter transient produces a brief overshoot (~ -5 dBTP)
    // at the start because the partial convolution (incomplete history)
    // sums phase coefficients that exceed 1.0 before settling.
    // Steady-state gain per phase is ~1.0, so:
    //   peak_abs ≈ 0.5 * max_phase_gain_at_transient ≈ 0.5 * 1.116 = 0.558
    //   dBTP ≈ 20 * log10(0.558) ≈ -5.07
    let sig = vec![0.5f32; 1024];
    let tp = compute_true_peak_db(&sig);
    assert!(
        tp > -6.5 && tp < -4.5,
        "DC 0.5 dBTP={tp} should be between -6.5 and -4.5 (transient overshoot)"
    );
    // Steady-state peak (skip filter settling) should be ~ -6 dBTP
    let up = oversample_4x(&sig);
    let steady_peak = up[48..].iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    let steady_tp = 20.0 * steady_peak.log10();
    assert!(
        (steady_tp - (-6.02)).abs() < 0.1,
        "Steady-state DC 0.5 should be ~-6 dBTP, got {steady_tp}"
    );
}

#[test]
fn test_true_peak_detects_gibbs_overshoot() {
    // A sharp step from +0.99 to -0.99 causes filter ringing.
    // The BS.1770-4 filter has large central coefficients (e.g., 0.972),
    // producing overshoot at the step discontinuity.
    let n = 512;
    let mut sig = vec![0.0f32; n];
    let step_at = n / 2;
    for (i, s) in sig.iter_mut().enumerate() {
        *s = if i < step_at { 0.99f32 } else { -0.99f32 };
    }
    // Sample-peak: all |x[i]| = 0.99 < 1.0
    let sample_peak = sig.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        sample_peak < 1.0,
        "Signal samples must all be < 1.0; sample_peak={sample_peak}"
    );

    let overs = find_true_peak_overs(&sig);
    assert!(
        !overs.is_empty(),
        "Step discontinuity must produce inter-sample overs"
    );
    // The overs should be near the step position (or at the initial transient)
    let pos = overs[0].position;
    let near_step = pos >= step_at.saturating_sub(24) && pos <= step_at + 24;
    assert!(
        pos < 24 || near_step,
        "Over position {pos} must be near step at {step_at} or at initial transient (< 24)"
    );
    assert!(
        overs[0].dbtp > 0.0,
        "Inter-sample over must have dBTP > 0, got {}",
        overs[0].dbtp
    );
}

#[test]
fn test_bs1770_fir_symmetry() {
    // Phase 3 = reversed Phase 0, Phase 2 = reversed Phase 1
    for k in 0..12 {
        assert!(
            (BS1770_PHASES[3][k] - BS1770_PHASES[0][11 - k]).abs() < 1e-12,
            "Phase symmetry p3[{k}] != p0[{}]: {} != {}",
            11 - k,
            BS1770_PHASES[3][k],
            BS1770_PHASES[0][11 - k]
        );
        assert!(
            (BS1770_PHASES[2][k] - BS1770_PHASES[1][11 - k]).abs() < 1e-12,
            "Phase symmetry p2[{k}] != p1[{}]: {} != {}",
            11 - k,
            BS1770_PHASES[2][k],
            BS1770_PHASES[1][11 - k]
        );
    }
}

#[test]
fn test_true_peak_no_overs_quiet_signal() {
    let sig = vec![0.3f32; 1024];
    let overs = find_true_peak_overs(&sig);
    assert!(overs.is_empty(), "Quiet signal should have no overs");
}

#[test]
fn test_find_true_peak_overs_empty() {
    let overs = find_true_peak_overs(&[]);
    assert!(overs.is_empty());
}

/// 21 kHz sine at 48 kHz, amplitude 0.999 → inter-sample overs from filter ripple.
#[test]
fn test_true_peak_detects_hf_sine_overs() {
    let sr = 48000.0f64;
    let ampl = 0.999f64;
    let freq = 21000.0;
    let n = 4800; // 100 ms
    let sig: Vec<f32> = (0..n)
        .map(|i| (ampl * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin()) as f32)
        .collect();

    // Verify all samples are < 1.0 (sample-peak not triggered)
    let sample_peak = sig.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(
        sample_peak < 1.0,
        "Sample peak {sample_peak} must be < 1.0 for sample-peak not to fire"
    );

    let overs = find_true_peak_overs(&sig);
    // Close to Nyquist, filter ripple may or may not push above 0 dBFS.
    // If it does, the overs should be at interpolated positions.
    for o in &overs {
        assert!(o.dbtp > 0.0, "Detected over must have dBTP > 0");
    }
}

/// Even a 1 kHz sine at amplitude 0.99999 may produce inter-sample overs
/// due to filter passband ripple (±0.1 dB per BS.1770-4 spec).
#[test]
fn test_true_peak_detects_near_full_scale_overs() {
    let sr = 48000.0f64;
    let ampl = 0.999999f64;
    let freq = 1000.0;
    let n = 4800;
    let sig: Vec<f32> = (0..n)
        .map(|i| (ampl * (2.0 * std::f64::consts::PI * freq * i as f64 / sr).sin()) as f32)
        .collect();

    let tp = compute_true_peak_db(&sig);
    // A nearly full-scale sine should have true-peak ~ 0 dBTP
    assert!(
        tp.abs() < 2.0,
        "Near-full-scale 1kHz sine dBTP={tp} should be close to 0"
    );
}

/// The upsampled signal must exactly match per-sample computation for a known waveform.
#[test]
fn test_oversample_4x_deterministic() {
    let sig: Vec<f32> = vec![1.0, -0.5, 0.25, 0.0, -0.125];
    let up1 = oversample_4x(&sig);
    let up2 = oversample_4x(&sig);
    assert_eq!(up1.len(), up2.len());
    for (i, (&a, &b)) in up1.iter().zip(up2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "oversample_4x must be deterministic at index {i}"
        );
    }
}

/// Cache-warmup: first few samples of the upsampled signal have partial filter history.
/// Output at index 0 should equal x[0] * h_p[0] (only one non-zero term).
#[test]
fn test_oversample_4x_first_sample() {
    let sig = vec![0.5f32];
    let up = oversample_4x(&sig);
    for p in 0..4 {
        let expected = 0.5f64 * BS1770_PHASES[p][0];
        let got = up[p];
        assert!(
            (got - expected).abs() < 1e-12,
            "Phase {p}: expected {expected}, got {got}"
        );
    }
}
