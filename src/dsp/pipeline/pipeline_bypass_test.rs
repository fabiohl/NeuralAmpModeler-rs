// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::tests::run_pipeline_test;

/// Ensures that, if no effects are active, the input sound is exactly equal to the output.
#[test]
#[cfg(feature = "stereo")]
fn test_bypass_no_resampler_stereo() {
    let n = 64;
    // Creates a test sound for the left (L) and right (R) sides.
    let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    let input_r: Vec<f32> = (0..n).map(|i| (i as f32 + 50.0) * 0.01).collect();

    // Runs the lab with the same input and output rate (no Hz conversion).
    let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, false);

    assert_eq!(out_l.len(), n);
    assert_eq!(out_r.len(), n);
    // Checks if the sounds are identical, sample by sample.
    assert_eq!(
        out_l, input_l,
        "L channel must be identical to the original"
    );
    assert_eq!(
        out_r, input_r,
        "R channel must be identical to the original"
    );
}

/// TEST: Direct Sound (Bypass) in Mono.
/// Verifies that the system identifies identical sounds and keeps the output synchronized.
#[test]
fn test_bypass_no_resampler_mono() {
    let n = 64;
    let input_l: Vec<f32> = (0..n).map(|i| i as f32 * 0.01).collect();
    // For the mono test, we make the right side an exact copy of the left.
    let input_r = input_l.clone();

    let (out_l, out_r) = run_pipeline_test(48000, 48000, &input_l, &input_r, true);

    assert_eq!(out_l.len(), n);
    assert_eq!(out_r.len(), n);
    assert_eq!(out_l, input_l);
    // In mono mode, the right side (R) must come out exactly equal to the left (L).
    assert_eq!(
        out_r, input_l,
        "In mono mode, the R side must be a copy of L"
    );
}

/// TEST: Direct Sound with Quality Change (Resampling).
/// Tests whether the sound continues to pass correctly even when the "speed" (Hz rate) changes.
#[test]
#[cfg(feature = "stereo")]
fn test_bypass_with_resampler_stereo() {
    // Example: Sound enters at 44100Hz (CD) and leaves at 48000Hz (Video).
    let n = 256;
    let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let input_r: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();

    let (out_l, _out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, false);

    assert!(!out_l.is_empty());

    // Calculates the "strength" (energy) of the original sound.
    let mut energy_in = 0.0;
    for &x in &input_l {
        energy_in += x * x;
    }

    // Calculates the "strength" of the sound that came out after conversion.
    let mut energy_out = 0.0;
    for &x in &out_l {
        energy_out += x * x;
    }

    // The sound doesn't need to be identical (due to the mathematical conversion),
    // but it should keep a similar volume and not be silent.
    assert!(
        energy_out > energy_in * 0.5,
        "The output sound is too weak or silent after conversion"
    );
}

/// TEST: Mono with Quality Change (Resampling).
/// Ensures that, even after converting the Hz rate, both channels remain identical.
#[test]
fn test_bypass_with_resampler_mono() {
    let n = 256;
    let input_l: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let input_r = input_l.clone();

    let (out_l, out_r) = run_pipeline_test(44100, 48000, &input_l, &input_r, true);

    assert!(!out_l.is_empty());
    assert_eq!(
        out_l, out_r,
        "Even with Hz conversion, mono sound must be equal on both sides (L == R)"
    );
}
