// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration test suite for deterministic test signal injection and energy evaluation.
//!
//! Validates:
//! - Non-zero deterministic signal injection (sine sweep + transient impulse + harmonics)
//! - 100% numerical finitude (zero NaN/Inf in output samples)
//! - Minimum RMS output energy (> -80 dBFS for active supported models)
//! - Bit-by-bit output determinism across repeated inference executions

use std::fs;
use std::path::PathBuf;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::catalog::{ModelSupportKind, catalog_entries};
use neural_amp_modeler_rs::testing::stress::{evaluate_signal_energy, generate_stress_signal_v1};

fn resolve_fixture_path(path_str: &str) -> Option<PathBuf> {
    neural_amp_modeler_rs::testing::fixtures::resolve_repo_path(path_str)
}

#[test]
fn test_deterministic_signal_injection_and_energy_evaluation() {
    let input_signal = generate_stress_signal_v1();
    assert_eq!(input_signal.len(), 2048);

    // Evaluate input signal energy to ensure input is active
    let input_eval = evaluate_signal_energy(&input_signal, -80.0);
    assert!(input_eval.is_finite, "Input stress signal must be finite");
    assert!(
        input_eval.is_active,
        "Input stress signal must be active (> -80 dBFS), got {} dBFS",
        input_eval.rms_dbfs
    );

    let entries = catalog_entries();
    let mut tested_models = 0;

    for entry in entries {
        if entry.support != ModelSupportKind::Supported {
            continue;
        }

        let model_path = match resolve_fixture_path(entry.canonical_path) {
            Some(p) => p,
            None => continue,
        };

        // Read and parse JSON data
        let json_str = match fs::read_to_string(&model_path) {
            Ok(s) => s,
            Err(e) => {
                panic!("Failed to read file {}: {e}", entry.canonical_path);
            }
        };

        let model_data = match parse_nam_json(&json_str) {
            Ok(d) => d,
            Err(e) => {
                panic!("Failed to parse NAM JSON {}: {e}", entry.canonical_path);
            }
        };

        let mut model1 = match build_model(&model_data) {
            Ok(m) => m,
            Err(e) => {
                panic!("Failed to build model {}: {e}", entry.canonical_path);
            }
        };

        let mut model2 = match build_model(&model_data) {
            Ok(m) => m,
            Err(e) => {
                panic!("Failed to build model {}: {e}", entry.canonical_path);
            }
        };

        model1.prewarm(2048);
        model2.prewarm(2048);

        // Helper for chunked processing (respecting max_buffer_size limit)
        let process_buffered = |m: &mut Box<neural_amp_modeler_rs::models::StaticModel>,
                                input: &[f32],
                                output: &mut [f32]| {
            let chunk_size = 64;
            for (in_c, out_c) in input.chunks(chunk_size).zip(output.chunks_mut(chunk_size)) {
                m.process(in_c, out_c);
            }
        };

        // Run 1
        let mut output1 = vec![0.0f32; input_signal.len()];
        process_buffered(&mut model1, &input_signal, &mut output1);

        // Run 2
        let mut output2 = vec![0.0f32; input_signal.len()];
        process_buffered(&mut model2, &input_signal, &mut output2);

        // 1. Bit-by-bit determinism
        assert_eq!(
            output1, output2,
            "Inference output is not bit-by-bit deterministic for model {}",
            entry.canonical_path
        );

        // 2. Numerical finitude (100% finite, zero NaN or Inf)
        let eval = evaluate_signal_energy(&output1, -80.0);
        assert!(
            eval.is_finite,
            "Inference output contains NaN or Inf for model {}",
            entry.canonical_path
        );

        // 3. Minimum RMS energy (> -80 dBFS)
        assert!(
            eval.is_active,
            "Model {} produced output RMS energy below -80 dBFS threshold: {:.2} dBFS (peak {:.2} dBFS)",
            entry.canonical_path, eval.rms_dbfs, eval.peak_dbfs
        );

        tested_models += 1;
    }

    eprintln!(
        "Successfully validated deterministic signal injection and energy evaluation for {tested_models} supported models."
    );
    assert!(
        tested_models > 0,
        "At least one supported catalog model must be tested"
    );
}
