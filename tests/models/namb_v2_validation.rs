// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Validation test for the `.namb` v2 format (Pre-transposed Weights).
//
//  Verifies that v2 export and corresponding loading maintain
//  absolute numerical parity with JSON loading (runtime transposition).

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{WeightsLayout, parse_nam_json};
use neural_amp_modeler_rs::loader::namb::parse_namb;
use neural_amp_modeler_rs::loader::namb_encoder::encode_namb;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

use crate::common::PrecisionGuard;
use crate::common::generate_sine_440hz as generate_sine;

/// Helper: resolves the absolute path to test fixtures.
fn model_path(filename: &str) -> PathBuf {
    neural_amp_modeler_rs::testing::fixtures::model_path(filename)
}

/// Computes the Mean Squared Error (MSE) using double precision (`f64`).
/// The use of `f64` is mandatory here to prevent rounding errors in the
/// accumulation from masking subtle divergences between weight layouts.
fn compute_mse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = (*x as f64) - (*y as f64);
            d * d
        })
        .sum();
    sum / (n as f64)
}

/// Validates NAMB v2 format parity with `GateMajorLstm` layout.
///
/// Traditionally, LSTM weights are loaded and transposed at runtime.
/// The v2 format allows weights to arrive already organized by gate (Gate-Major),
/// eliminating the need for memory rearrangement during initial loading.
#[test]
fn test_lstm_v2_gate_major_parity() {
    let path = model_path("BossLSTM-1x16.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    // 1. Build original model (with runtime transposition via JSON)
    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(1024);

    // 2. Encode to NAMB v2 using Gate-Major layout
    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::GateMajorLstm).unwrap();

    // 3. Decode the v2 binary and verify the layout was preserved
    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::GateMajorLstm);

    // 4. Build the v2 model (direct loading, no transposition)
    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(1024);

    // 5. Compare numerical output to guarantee absolute parity (MSE near zero).
    // PrecisionGuard ensures stable ActivationPrecision across both process() calls
    // — without it, a parallel writer thread could flip the global atomic between
    // the two inferences, causing spurious MSE divergence (S6.T01/F11 victims).
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[LSTM v2 Parity] MSE: {:.2e}", mse);
    assert!(
        mse < 1e-12,
        "Divergence detected in GateMajorLstm layout! MSE={:e}",
        mse
    );
}

/// Validates NAMB v2 format parity with `Interleaved4WaveNet` layout.
///
/// This layout organizes dilated convolution weights in blocks of 4 (tiling),
/// which is optimized for NeuralAmpModeler-rs AVX2 `fused_gemm_residual` kernels.
#[test]
fn test_wavenet_v2_interleaved4_parity() {
    let path = model_path("BossWN-nano.nam");
    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).unwrap();
    let original_data = parse_nam_json(&json_data).unwrap();

    // 1. Build original model (standard JSON loading)
    let mut model_orig = build_model(&original_data).unwrap();
    model_orig.prewarm(2048);

    // 2. Encode to NAMB v2 with 4-float interleaving (AVX2 tiling factor)
    let namb_v2 = encode_namb(&original_data, 2, WeightsLayout::Interleaved4WaveNet).unwrap();

    // 3. Load the v2 binary
    let v2_data = parse_namb(&namb_v2).unwrap();
    assert_eq!(v2_data.weights_layout, WeightsLayout::Interleaved4WaveNet);

    let mut model_v2 = build_model(&v2_data).unwrap();
    model_v2.prewarm(2048);

    // 4. Numerical validation.
    // See note in test_lstm_v2_gate_major_parity re: PrecisionGuard necessity.
    let _guard = PrecisionGuard::new(ActivationPrecision::Standard);
    let input = generate_sine(512);
    let mut out_orig = vec![0.0f32; 512];
    let mut out_v2 = vec![0.0f32; 512];

    model_orig.process(&input, &mut out_orig);
    model_v2.process(&input, &mut out_v2);

    let mse = compute_mse(&out_orig, &out_v2);
    println!("[WaveNet v2 Parity] MSE: {:.2e}", mse);
    assert!(
        mse < 1e-12,
        "Divergence detected in Interleaved4WaveNet layout! MSE={:e}",
        mse
    );
}
