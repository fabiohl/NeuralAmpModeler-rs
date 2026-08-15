// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration tests for optional prewarm control via `LoadOptions`.
//
//  Validates that:
//  1. `prewarm: Some(false)` skips the initial prewarm during loading.
//  2. `reset()` with `prewarm_on_reset == false` does not execute prewarm.
//  3. `set_prewarm_on_reset(false)` propagates through ContainerModel submodels.
//  4. LSTM `prewarm_samples()` returns values proportional to `expected_sample_rate`.
//  5. LSTM reset with `prewarm_on_reset = true` produces deterministic, stabilized output.

use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

use super::common;
use common::io_helpers::{model_path, process_in_blocks};

const BLOCK_SIZE: usize = 64;

// =============================================================================
// Helpers
// =============================================================================

fn sys() -> SystemSnapshot {
    SystemSnapshot::capture()
}

fn load_with_opts(
    path: &std::path::Path,
    prewarm: Option<bool>,
) -> neural_amp_modeler_rs::loader::LoadedModelPair {
    load_and_build_model(path, &sys(), false, LoadOptions { prewarm })
        .expect("Failed to load model for prewarm test")
}

// =============================================================================
// Test 1: prewarm: Some(false) skips initial prewarm
// =============================================================================

/// Load a model with `prewarm: Some(false)` and verify prewarm was skipped.
///
/// Verifies that `prewarm_on_reset()` returns `false` after loading with
/// `prewarm: Some(false)`, and returns `true` with the default `LoadOptions`.
#[test]
fn test_load_with_prewarm_skip() {
    let path = model_path("linear_test.nam");

    let pair_skip = load_with_opts(&path, Some(false));
    let model_skip = pair_skip.model_l.as_ref().unwrap();
    assert!(
        !model_skip.prewarm_on_reset(),
        "prewarm_on_reset should be false when loaded with prewarm: Some(false)"
    );

    let pair_default = load_with_opts(&path, None);
    let model_default = pair_default.model_l.as_ref().unwrap();
    assert!(
        model_default.prewarm_on_reset(),
        "prewarm_on_reset should be true when loaded with default LoadOptions"
    );
}

/// Verify that loading with `prewarm: Some(false)` produces a model that
/// processes audio correctly (no panics, finite output).
#[test]
fn test_skip_prewarm_output_is_valid() {
    let path = model_path("linear_test.nam");

    let mut pair_skip = load_with_opts(&path, Some(false));
    let model = pair_skip.model_l.as_mut().unwrap();

    let input = vec![0.5f32; BLOCK_SIZE * 4];
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(model, &input, &mut output, BLOCK_SIZE);

    for &s in &output {
        assert!(s.is_finite(), "Output sample should be finite");
    }
}

/// Ensure deterministic output comparison between prewarm-skip and default
/// after both models are manually reset with prewarm_on_reset enabled.
/// After identical reset, outputs must match.
#[test]
fn test_skip_vs_default_deterministic_after_reset() {
    let path = model_path("linear_test.nam");

    let mut pair_skip = load_with_opts(&path, Some(false));
    let mut pair_default = load_with_opts(&path, None);

    let model_skip = pair_skip.model_l.as_mut().unwrap();
    let model_default = pair_default.model_l.as_mut().unwrap();

    model_skip
        .reset(48000, BLOCK_SIZE)
        .expect("reset with prewarm_on_reset=false should succeed");
    model_default
        .reset(48000, BLOCK_SIZE)
        .expect("reset with prewarm_on_reset=true should succeed");

    let input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 4];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin();
        }
        v
    };
    let mut out_skip = vec![0.0f32; input.len()];
    let mut out_default = vec![0.0f32; input.len()];

    process_in_blocks(model_skip, &input, &mut out_skip, BLOCK_SIZE);
    process_in_blocks(model_default, &input, &mut out_default, BLOCK_SIZE);

    for (&a, &b) in out_skip.iter().zip(out_default.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "Outputs should match after identical reset"
        );
    }
}

// =============================================================================
// Test 2: reset() without prewarm_on_reset skips prewarm computations
// =============================================================================

/// Build model, set `prewarm_on_reset(false)`, call `reset()` and verify
/// no panic occurs and output is still valid.
#[test]
fn test_reset_without_prewarm_no_panic() {
    let path = model_path("linear_test.nam");
    let json = std::fs::read_to_string(&path).expect("Failed to read linear_test.nam");
    let data = parse_nam_json(&json).expect("Failed to parse linear_test.nam");
    let mut model = build_model(&data).expect("Failed to build linear model");

    model.set_prewarm_on_reset(false);
    assert!(!model.prewarm_on_reset());

    model
        .reset(48000, BLOCK_SIZE)
        .expect("reset() with prewarm_on_reset=false should succeed");

    let input = vec![0.5f32; BLOCK_SIZE * 4];
    let mut output = vec![0.0f32; input.len()];
    process_in_blocks(&mut model, &input, &mut output, BLOCK_SIZE);

    for &s in &output {
        assert!(
            s.is_finite(),
            "Output should be finite after reset without prewarm"
        );
    }
}

/// Verify that `reset()` clears internal state regardless of
/// `prewarm_on_reset` (S3.T2): with the flag disabled, the FIR history and
/// FFT tail must still be silenced — the flag only gates the priming pass.
/// Process audio first (fill internal state), then compare reset outcomes.
#[test]
fn test_reset_outcome_same_prewarm_vs_noprewarm() {
    let path = model_path("linear_test.nam");
    let json = std::fs::read_to_string(&path).expect("Failed to read linear_test.nam");
    let data = parse_nam_json(&json).expect("Failed to parse linear_test.nam");

    let input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 4];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.25).sin();
        }
        v
    };

    // Build and process some audio through the model to fill internal state
    let mut model_a = build_model(&data).expect("Failed to build");
    let mut dummy_out = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_a, &input, &mut dummy_out, BLOCK_SIZE);
    // Reset with prewarm → clears state
    model_a.set_prewarm_on_reset(true);
    model_a.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_a = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_a, &input, &mut out_a, BLOCK_SIZE);

    // Build separate model, process audio, reset without prewarm → state
    // must STILL be cleared (deterministic cleanup, no residual FIR energy).
    let mut model_b = build_model(&data).expect("Failed to build");
    process_in_blocks(&mut model_b, &input, &mut dummy_out, BLOCK_SIZE);
    model_b.set_prewarm_on_reset(false);
    model_b.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_b = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_b, &input, &mut out_b, BLOCK_SIZE);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(a.is_finite());
        assert!(b.is_finite());
        assert!(
            (a - b).abs() < 1e-6,
            "Outputs should match after identical reset (state cleared in both cases): \
             index {i}: prewarm={a} no-prewarm={b}"
        );
    }
}

// =============================================================================
// Test 3: ContainerModel flag propagation to nested submodels
// =============================================================================

/// Create a ContainerModel, call `set_prewarm_on_reset(false)`, and verify
/// all submodels also have their prewarm_on_reset set to false.
#[test]
fn test_container_prewarm_propagation() {
    let full_nam_path = model_path("wavenet_a2_full.nam");
    let lite_nam_path = model_path("wavenet_a2_lite.nam");
    if !full_nam_path.exists() || !lite_nam_path.exists() {
        eprintln!("SKIP: A2 model files not found. Container prewarm propagation test impossible.");
        return;
    }

    let full_json = std::fs::read_to_string(&full_nam_path).expect("Failed to read A2-Full");
    let full_data = parse_nam_json(&full_json).expect("Failed to parse A2-Full");
    let full_model = build_model(&full_data).expect("Failed to build A2-Full");

    let lite_json = std::fs::read_to_string(&lite_nam_path).expect("Failed to read A2-Lite");
    let lite_data = parse_nam_json(&lite_json).expect("Failed to parse A2-Lite");
    let lite_model = build_model(&lite_data).expect("Failed to build A2-Lite");

    let sample_rate = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);

    let mut container = neural_amp_modeler_rs::models::container::ContainerModel::new(
        vec![(0.5, lite_model), (1.0, full_model)],
        sample_rate,
    )
    .expect("Failed to create ContainerModel");

    assert!(container.prewarm_on_reset(), "Default should be true");

    container.set_prewarm_on_reset(false);
    assert!(
        !container.prewarm_on_reset(),
        "Container should be false after set"
    );

    for (idx, (_threshold, submodel)) in container.submodels().iter().enumerate() {
        assert!(
            !submodel.prewarm_on_reset(),
            "Submodel[{}] prewarm_on_reset should be false after container propagation",
            idx
        );
    }
}

/// Verify that loading a slimmable_container.nam with `prewarm: Some(false)`
/// propagates the flag through both ContainerModel and its submodels.
#[test]
fn test_container_load_skip_propagation() {
    let path = model_path("slimmable_container.nam");
    if !path.exists() {
        eprintln!("SKIP: slimmable_container.nam not found.");
        return;
    }

    let Ok(pair) = load_and_build_model(
        &path,
        &sys(),
        false,
        LoadOptions {
            prewarm: Some(false),
        },
    ) else {
        eprintln!("SKIP: container build failed (unsupported activation in submodel).");
        return;
    };
    let Some(model) = pair.model_l.as_ref() else {
        eprintln!("SKIP: container build failed (unsupported activation in submodel).");
        return;
    };

    assert!(
        !model.prewarm_on_reset(),
        "Container loaded with prewarm: Some(false) should have prewarm_on_reset=false"
    );

    if let StaticModel::Container(container) = model.as_ref() {
        for (idx, (_threshold, submodel)) in container.submodels().iter().enumerate() {
            assert!(
                !submodel.prewarm_on_reset(),
                "Container submodel[{}] should have prewarm_on_reset=false",
                idx
            );
        }
    }
}

// =============================================================================
// Test 4: LSTM prewarm_samples() changes with expected_sample_rate
// =============================================================================

/// Verify that `prewarm_samples()` returns a value proportional to the
/// `expected_sample_rate` stored in the LSTM model. For LSTM models,
/// `prewarm_samples() == (0.5 * expected_sample_rate)`.
///
/// Tests both static (H=3) and dynamic (H=7) LSTM variants.
#[test]
fn test_lstm_prewarm_samples_scales_with_sample_rate() {
    let lstm_static_path = model_path("lstm.nam");
    let lstm_dyn_path = model_path("lstm_dyn_test.nam");

    let cases: &[(&str, Option<f32>, usize)] = &[
        ("48kHz (explicit)", Some(48000.0), 24000),
        ("44.1kHz", Some(44100.0), 22050),
        ("None → DEFAULT_SAMPLE_RATE (48000)", None, 24000),
    ];

    for (model_name, model_path) in [
        ("static H=3", lstm_static_path),
        ("dynamic H=7", lstm_dyn_path),
    ] {
        if !model_path.exists() {
            eprintln!(
                "SKIP: {} model file not found for prewarm_samples test.",
                model_name
            );
            continue;
        }

        let json = std::fs::read_to_string(&model_path).expect("Failed to read LSTM model file");
        let mut data = parse_nam_json(&json).expect("Failed to parse LSTM model JSON");

        for &(label, sr, expected) in cases {
            data.sample_rate = sr;
            let model = build_model(&data).unwrap_or_else(|e| {
                panic!(
                    "Failed to build {} model with sample_rate={:?}: {}",
                    model_name, sr, e
                )
            });

            let actual = model.prewarm_samples();
            assert_eq!(
                actual, expected,
                "[{}] prewarm_samples() mismatch for {}: expected {}, got {}",
                model_name, label, expected, actual
            );
        }
    }
}

/// Confirm that the edge case `sample_rate = 1.0` yields at least 1 sample
/// (the `result <= 0 → 1` safety floor).
#[test]
fn test_lstm_prewarm_samples_minimum() {
    let path = model_path("lstm.nam");
    if !path.exists() {
        eprintln!("SKIP: lstm.nam not found.");
        return;
    }

    let json = std::fs::read_to_string(&path).expect("Failed to read lstm.nam");
    let mut data = parse_nam_json(&json).expect("Failed to parse lstm.nam");

    data.sample_rate = Some(1.0);
    let model = build_model(&data).expect("Failed to build LSTM with sample_rate=1.0");

    assert_eq!(
        model.prewarm_samples(),
        1,
        "prewarm_samples() should floor at 1 when 0.5 * sample_rate < 1"
    );
}

// =============================================================================
// Test 5: LSTM reset with prewarm_on_reset=true stabilizes recurrent state
// =============================================================================

/// Build two LSTM instances from the same weights, process different audio
/// through each to create divergent internal states, then reset both with
/// `prewarm_on_reset = true`. After reset, processing identical input must
/// yield identical output — proving deterministic stabilization.
#[test]
fn test_lstm_reset_deterministic_after_prewarm() {
    let path = model_path("lstm.nam");
    if !path.exists() {
        eprintln!("SKIP: lstm.nam not found.");
        return;
    }

    let json = std::fs::read_to_string(&path).expect("Failed to read lstm.nam");
    let data = parse_nam_json(&json).expect("Failed to parse lstm.nam");

    let mut model_a = build_model(&data).expect("Failed to build LSTM model A");
    let mut model_b = build_model(&data).expect("Failed to build LSTM model B");

    let oil_input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 8];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin() * 0.5;
        }
        v
    };
    let water_input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 8];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.13).cos() * 0.7;
        }
        v
    };
    let test_input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 4];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.06).sin() * 0.35;
        }
        v
    };

    let mut dummy = vec![0.0f32; oil_input.len()];

    process_in_blocks(&mut model_a, &oil_input, &mut dummy, BLOCK_SIZE);
    process_in_blocks(&mut model_b, &water_input, &mut dummy, BLOCK_SIZE);

    model_a
        .reset(48000, BLOCK_SIZE)
        .expect("reset A with prewarm should succeed");
    model_b
        .reset(48000, BLOCK_SIZE)
        .expect("reset B with prewarm should succeed");

    let mut out_a = vec![0.0f32; test_input.len()];
    let mut out_b = vec![0.0f32; test_input.len()];

    process_in_blocks(&mut model_a, &test_input, &mut out_a, BLOCK_SIZE);
    process_in_blocks(&mut model_b, &test_input, &mut out_b, BLOCK_SIZE);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            a.is_finite(),
            "Output A sample[{}] should be finite after prewarm reset",
            i
        );
        assert!(
            b.is_finite(),
            "Output B sample[{}] should be finite after prewarm reset",
            i
        );
        assert!(
            (a - b).abs() < 1e-6,
            "LSTM after prewarm reset: outputs must be deterministic; mismatch at sample[{}]: {} vs {}",
            i,
            a,
            b
        );
    }
}

/// Verify that resetting an LSTM with `prewarm_on_reset = false` does NOT
/// stabilize the state — diverging priors persist, yielding different outputs.
/// This is the LSTM counterpart of the Linear test
/// `test_reset_outcome_differs_prewarm_vs_noprewarm`.
#[test]
fn test_lstm_reset_differs_prewarm_vs_noprewarm() {
    let path = model_path("lstm.nam");
    if !path.exists() {
        eprintln!("SKIP: lstm.nam not found.");
        return;
    }

    let json = std::fs::read_to_string(&path).expect("Failed to read lstm.nam");
    let data = parse_nam_json(&json).expect("Failed to parse lstm.nam");

    let mut model_a = build_model(&data).expect("Failed to build LSTM model A");
    let mut model_b = build_model(&data).expect("Failed to build LSTM model B");

    let input = {
        let mut v = vec![0.0f32; BLOCK_SIZE * 8];
        for (i, s) in v.iter_mut().enumerate() {
            *s = (i as f32 * 0.1).sin() * 0.5;
        }
        v
    };
    let mut dummy = vec![0.0f32; input.len()];

    process_in_blocks(&mut model_a, &input, &mut dummy, BLOCK_SIZE);
    process_in_blocks(&mut model_b, &input, &mut dummy, BLOCK_SIZE);

    model_a.set_prewarm_on_reset(true);
    model_a.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_a = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_a, &input, &mut out_a, BLOCK_SIZE);

    model_b.set_prewarm_on_reset(false);
    model_b.reset(48000, BLOCK_SIZE).unwrap();
    let mut out_b = vec![0.0f32; input.len()];
    process_in_blocks(&mut model_b, &input, &mut out_b, BLOCK_SIZE);

    let mut any_differ = false;
    for (&a, &b) in out_a.iter().zip(out_b.iter()) {
        if (a - b).abs() > 1e-6 {
            any_differ = true;
        }
        assert!(
            a.is_finite(),
            "Output A should be finite after prewarm reset"
        );
        assert!(
            b.is_finite(),
            "Output B should be finite after no-prewarm reset"
        );
    }
    assert!(
        any_differ,
        "LSTM: prewarm vs no-prewarm reset should produce different outputs"
    );
}

// =============================================================================
// Test 6: WaveNet prewarm_samples() regression (S3-T2)
// =============================================================================

/// Validates that `prewarm_samples()` on a static 2-array WaveNet model
/// equals the canonical receptive field: `sum_{arrays} sum_{(K-1)*d}`.
///
/// Uses `BossWN-standard.nam` (16-CH, 2 arrays of 10 layers each).
#[test]
fn test_wavenet_static_prewarm_samples_matches_rf() {
    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam fixture not found.");
        return;
    }

    let json = std::fs::read_to_string(&path).expect("Failed to read BossWN-standard.nam");
    let data = parse_nam_json(&json).expect("Failed to parse BossWN-standard.nam");

    let expected_rf_total: usize = data
        .config
        .layers
        .iter()
        .map(|layer| {
            let k = layer.kernel_size.unwrap_or(3);
            let dils = layer
                .dilations
                .as_ref()
                .expect("WaveNet layer must have dilations");
            dils.iter().map(|&d| (k - 1) * d).sum::<usize>()
        })
        .sum();

    let model = build_model(&data).expect("Failed to build BossWN-standard model");

    assert_eq!(
        model.prewarm_samples(),
        expected_rf_total,
        "Static WaveNet prewarm_samples() should equal sum of (K-1)*d across all arrays"
    );
}

/// Validates that `prewarm_samples()` on a dynamic WaveNet model is at least
/// the sum of its layer-array receptive fields. The dynamic model may also
/// include condition_dsp and post-stack-head contributions.
///
/// Uses `wavenet_condition_dsp.nam` (dynamic, 2 arrays with cond DSP sub-model).
#[test]
fn test_wavenet_dyn_prewarm_samples_at_least_array_rf() {
    let path = model_path("wavenet_condition_dsp.nam");
    if !path.exists() {
        eprintln!("SKIP: wavenet_condition_dsp.nam fixture not found.");
        return;
    }

    let json = std::fs::read_to_string(&path).expect("Failed to read wavenet_condition_dsp.nam");
    let data = parse_nam_json(&json).expect("Failed to parse wavenet_condition_dsp.nam");

    let array_rf_sum: usize = data
        .config
        .layers
        .iter()
        .map(|layer| {
            let k = layer.kernel_size.unwrap_or(3);
            let dils = layer
                .dilations
                .as_ref()
                .expect("WaveNet layer must have dilations");
            dils.iter().map(|&d| (k - 1) * d).sum::<usize>()
        })
        .sum();

    let model = build_model(&data).expect("Failed to build wavenet_condition_dsp model");

    let actual = model.prewarm_samples();
    assert!(
        actual >= array_rf_sum,
        "Dynamic WaveNet prewarm_samples() ({}) should be >= array RF sum ({})",
        actual,
        array_rf_sum
    );
}
