// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

/// Tests basic gain application.
/// Verifies bypass (1.0), amplification (2.0), and behavior with zeros.
#[test]
fn test_apply_gain_simd() {
    let mut buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let original = buffer;

    // Neutral: Gain 1.0 should not change the buffer.
    apply_gain_simd(&mut buffer, 1.0);
    assert_eq!(buffer, original);

    // Gain = 2.0: Multiplies all samples by 2.
    apply_gain_simd(&mut buffer, 2.0);
    assert_eq!(
        buffer,
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    );

    // Gain on Zeros: Multiplying zero by anything should result in zero.
    let mut zeros = [0.0; 25];
    apply_gain_simd(&mut zeros, 5.5);
    for &z in &zeros {
        assert_eq!(z, 0.0);
    }
}

/// Validates the combined gain staging calculation (user + model metadata).
/// Simulates the exact logic of `update_gain_multipliers` from pw_host.rs.
/// Ensures that the dB -> Linear conversion and application are correct.
#[test]
fn test_combined_gain_staging() {
    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let user_input_db: f32 = 6.0;
    let model_input_adj_db: f32 = -3.0;

    // In the new architecture, the Main Thread sends linear multipliers.
    let user_input_mult = lut.db_to_linear(user_input_db);
    let model_input_adj_mult = lut.db_to_linear(model_input_adj_db);

    // The RT thread simply multiplies them (zero powf/LUT in the hot-path).
    let gain_linear = user_input_mult * model_input_adj_mult;

    let mut buffer = [1.0f32; 16];
    apply_gain_simd(&mut buffer, gain_linear);

    let expected = lut.db_to_linear(3.0);
    for &sample in &buffer {
        assert!(
            (sample - expected).abs() < 1e-5,
            "Expected ~{expected:.5}, got {sample:.5}"
        );
    }
}

/// Verifies stability with extreme negative gain (-60dB ≈ 0.001)
/// and positive (+24dB ≈ 15.85) without Float32 underflow/overflow.
#[test]
fn test_extreme_gain_values() {
    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    // -60 dB → gain ≈ 0.001
    let gain_neg60 = lut.db_to_linear(-60.0);
    assert!(gain_neg60 > 0.0 && gain_neg60.is_finite());

    let mut buffer = [1.0f32; 20];
    apply_gain_simd(&mut buffer, gain_neg60);
    for &s in &buffer {
        assert!(
            s.is_finite() && s > 0.0 && s < 0.01,
            "Underflow at -60dB: {s}"
        );
    }

    // +24 dB → gain ≈ 15.85
    let gain_pos24 = lut.db_to_linear(24.0);
    assert!(gain_pos24 > 10.0 && gain_pos24.is_finite());

    let mut buffer2 = [0.5f32; 20];
    apply_gain_simd(&mut buffer2, gain_pos24);
    for &s in &buffer2 {
        assert!(
            s.is_finite() && s > 5.0 && s < 10.0,
            "Overflow at +24dB: {s}"
        );
    }
}

/// Verifies that `apply_gain_simd` with gain=1.0 is true-bypass bitwise (no bits changed).
#[test]
fn test_gain_true_bypass() {
    // Scenario 1: Aligned buffer (multiple of 8)
    let original_aligned: [f32; 16] = [
        0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2, 1.3, -1.4, 1.5, -1.6,
    ];
    let mut buf_aligned = original_aligned;
    apply_gain_simd(&mut buf_aligned, 1.0);
    assert_eq!(
        buf_aligned, original_aligned,
        "Gain 1.0 must preserve aligned buffer bitwise (true-bypass)"
    );

    // Scenario 2: Unaligned buffer (scalar tail)
    let original_tail: [f32; 13] = [
        0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.10, 0.11, -0.12, 0.13,
    ];
    let mut buf_tail = original_tail;
    apply_gain_simd(&mut buf_tail, 1.0);
    assert_eq!(
        buf_tail, original_tail,
        "Gain 1.0 must preserve unaligned buffer bitwise (true-bypass tail)"
    );

    // Scenario 3: Gain very close to 1.0 (within the bypass epsilon)
    let mut buf_eps = original_aligned;
    apply_gain_simd(&mut buf_eps, 1.0 + 1e-7);
    assert_eq!(
        buf_eps, original_aligned,
        "Gain ≈1.0 (within 1e-6) must trigger fast-path bypass"
    );
}

/// Roundtrip +6dB → -6dB must preserve the original signal (MSE < 1e-10).
#[test]
fn test_gain_roundtrip_6db() {
    let original: Vec<f32> = (0..256)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
        .collect();

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let gain_up = lut.db_to_linear(6.0); // +6 dB
    let gain_down = lut.db_to_linear(-6.0); // -6 dB

    let mut buffer = original.clone();
    apply_gain_simd(&mut buffer, gain_up);
    apply_gain_simd(&mut buffer, gain_down);

    let mse: f64 = original
        .iter()
        .zip(buffer.iter())
        .map(|(a, b)| {
            let d = (*a as f64) - (*b as f64);
            d * d
        })
        .sum::<f64>()
        / (original.len() as f64);

    assert!(
        mse < 1e-10,
        "Roundtrip +6dB/-6dB MSE={mse:.2e} exceeds 1e-10"
    );
}

/// Validates SIMD linear ramp against a scalar reference implementation.
#[test]
fn test_apply_ramp_simd() {
    let len = 37; // Non-multiple-of-8 size to test tail
    let mut buffer_simd = vec![1.0f32; len];
    let mut buffer_scalar = vec![1.0f32; len];

    let start = 0.5f32;
    let step = 0.01f32;

    // Scalar reference
    let mut m = start;
    for s in buffer_scalar.iter_mut() {
        *s *= m;
        m += step;
    }

    // SIMD implementation
    apply_ramp_simd(&mut buffer_simd, start, step);

    // Validates MSE between the two implementations
    for i in 0..len {
        assert!(
            (buffer_simd[i] - buffer_scalar[i]).abs() < 1e-6,
            "Ramp divergence at index {i}: SIMD={} Scalar={}",
            buffer_simd[i],
            buffer_scalar[i]
        );
    }
}
