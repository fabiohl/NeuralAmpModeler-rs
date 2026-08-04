// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Rust self-consistency (determinism) tests — universal gate.
//
//  # Objective
//
//  Validate that the Rust inference engine produces bitwise-identical results
//  across independent runs with the same model weights and inputs. This is the
//  **determinism invariant**: every architecture MUST produce identical output
//  from two independently-built instances processing the same signal.
//
//  # Pattern
//
//  Each test loads the same `.nam` model twice, builds two independent
//  `StaticModel` instances, runs prewarm, processes the same 440 Hz sine
//  test signal through both, and asserts that the MSE between the two output
//  buffers is exactly 0.0 (bitwise-identical). This generalizes the
//  `test_cabsim_bitwise_determinism` pattern to every supported architecture.
//
//  # Coverage
//
//  | Architecture            | Models tested                                      |
//  | ----------------------- | -------------------------------------------------- |
//  | WaveNet A1              | Standard (CH=16), Feather (CH=8), Nano (CH=4),     |
//  |                         | Lite (CH=12, deterministic post-P1 fix),            |
//  |                         | A1 Standard Official,                               |
//  |                         | Official (CH=3 free geom, dynamic path)            |
//  | WaveNet A2              | Full (CH=8), Lite (CH=3), Container (both),         |
//  |                         | SlimmableContainer A2 Example (CH=3→6)               |
//  | LSTM                    | 1×16, 2×8, Official                                |
//  | Linear                  | linear_test (RF=16)                                |
//  | CabSim (UPOLS)          | `tests/cabsim_golden.rs::test_cabsim_bitwise_determinism` |
//
//  These tests do not depend on C++ golden vectors.

use std::fs;
use std::path::Path;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;

use super::common;
use common::*;

/// Shared determinism assertion for any `.nam` model.
///
/// Loads the model twice from the same file, builds two independent
/// `StaticModel` instances, prewarms both, processes a 440 Hz sine signal
/// through each, and asserts `MSE == 0.0` (nothing less — determinism must
/// be absolute).
///
/// Returns silently on success; panics with diagnostic information on failure.
fn assert_model_determinism(path: &Path, label: &str) {
    let json_data =
        fs::read_to_string(path).unwrap_or_else(|e| panic!("[{label}] Failed to read model: {e}"));

    let model_data = parse_nam_json(&json_data)
        .unwrap_or_else(|e| panic!("[{label}] Failed in JSON parser: {e}"));

    let mut model_a = build_model(&model_data)
        .unwrap_or_else(|e| panic!("[{label}] Dispatcher failed (model_a): {e}"));
    let mut model_b = build_model(&model_data)
        .unwrap_or_else(|e| panic!("[{label}] Dispatcher failed (model_b): {e}"));

    model_a.prewarm(2048);
    model_b.prewarm(2048);

    let num_samples = if cfg!(debug_assertions) {
        512
    } else {
        GOLDEN_NUM_SAMPLES
    };

    let input = generate_sine_440hz(num_samples);
    let mut out_a = vec![0.0f32; num_samples];
    let mut out_b = vec![0.0f32; num_samples];

    process_in_blocks(&mut model_a, &input, &mut out_a, GOLDEN_BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut out_b, GOLDEN_BLOCK_SIZE);

    let mse = compute_mse(&out_a, &out_b);
    let mae = compute_max_abs_error(&out_a, &out_b);

    println!("[{label}] MSE={mse:.2e}, MaxAbsErr={mae:.2e}");

    assert!(
        mse == 0.0,
        "Rust engine non-deterministic [{label}]! MSE={mse:.6e}, MaxAbsErr={mae:.6e}"
    );
}

/// Helper: returns `true` if the model file exists, prints a SKIP message otherwise.
fn model_exists(filename: &str) -> bool {
    let path = model_path(filename);
    if !path.exists() {
        eprintln!("SKIP: {filename} not found at {path:?}. Skipping self-consistency.");
        return false;
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// WaveNet A1 — Standard, Feather, Nano, Lite, Official
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_auto_consistency_wavenet() {
    if !model_exists("BossWN-standard.nam") {
        return;
    }
    assert_model_determinism(&model_path("BossWN-standard.nam"), "WaveNet Standard CH=16");
}

#[test]
fn test_auto_consistency_wavenet_feather() {
    if !model_exists("BossWN-feather.nam") {
        return;
    }
    assert_model_determinism(&model_path("BossWN-feather.nam"), "WaveNet Feather CH=8");
}

#[test]
fn test_auto_consistency_wavenet_nano() {
    if !model_exists("BossWN-nano.nam") {
        return;
    }
    assert_model_determinism(&model_path("BossWN-nano.nam"), "WaveNet Nano CH=4");
}

#[test]
fn test_auto_consistency_wavenet_lite() {
    if !model_exists("EVH-5150-Lite.nam") {
        return;
    }
    assert_model_determinism(&model_path("EVH-5150-Lite.nam"), "WaveNet Lite CH=12");
}

#[test]
fn test_auto_consistency_wavenet_a1_standard() {
    if !model_exists("wavenet_a1_standard.nam") {
        return;
    }
    assert_model_determinism(
        &model_path("wavenet_a1_standard.nam"),
        "WaveNet A1 Standard Official",
    );
}

#[test]
fn test_auto_consistency_wavenet_official() {
    if !model_exists("wavenet_official.nam") {
        return;
    }
    assert_model_determinism(
        &model_path("wavenet_official.nam"),
        "WaveNet Official CH=3 (dynamic path)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// WaveNet A2 — Full, Lite, Container
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_auto_consistency_wavenet_a2_full() {
    if !model_exists("wavenet_a2_full.nam") {
        return;
    }
    assert_model_determinism(&model_path("wavenet_a2_full.nam"), "WaveNet A2-Full CH=8");
}

#[test]
fn test_auto_consistency_wavenet_a2_lite() {
    if !model_exists("wavenet_a2_lite.nam") {
        return;
    }
    assert_model_determinism(&model_path("wavenet_a2_lite.nam"), "WaveNet A2-Lite CH=3");
}

#[test]
fn test_auto_consistency_wavenet_a2_container() {
    if !model_exists("wavenet_a2_container.nam") {
        return;
    }
    assert_model_determinism(
        &model_path("wavenet_a2_container.nam"),
        "WaveNet A2 Container",
    );
}

#[test]
fn test_auto_consistency_a2_example_slimmable() {
    if !model_exists("a2_example.nam") {
        return;
    }
    assert_model_determinism(
        &model_path("a2_example.nam"),
        "SlimmableContainer A2 Example (CH=3→6)",
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// LSTM — 1×16, 2×8, Official
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_auto_consistency_lstm() {
    if !model_exists("BossLSTM-1x16.nam") {
        return;
    }
    assert_model_determinism(&model_path("BossLSTM-1x16.nam"), "LSTM 1×16");
}

#[test]
fn test_auto_consistency_lstm_2x8() {
    if !model_exists("BossLSTM-2x8.nam") {
        return;
    }
    assert_model_determinism(&model_path("BossLSTM-2x8.nam"), "LSTM 2×8");
}

#[test]
fn test_auto_consistency_lstm_official() {
    if !model_exists("lstm.nam") {
        return;
    }
    assert_model_determinism(&model_path("lstm.nam"), "LSTM Official");
}

// ═══════════════════════════════════════════════════════════════════════════
// Linear — FIR feedforward
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_auto_consistency_linear() {
    if !model_exists("linear_test.nam") {
        return;
    }
    assert_model_determinism(&model_path("linear_test.nam"), "Linear RF=16");
}
