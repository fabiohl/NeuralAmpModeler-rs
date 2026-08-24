// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common;
use common::*;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use std::fs;
use std::path::PathBuf;

#[test]
fn test_nondist_models_validation() {
    let mut nondist_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        let community = neural_amp_modeler_rs::testing::fixtures::community_models_dir();
        if community.exists() {
            nondist_path = community;
        } else {
            println!(
                "[STATUS] SKIP_OPTIONAL: models_nondist_absent \
                 (tests/fixtures/models-nondist or third-party/community_models)"
            );
            return;
        }
    }

    let manifest_path = nondist_path.join("manifest.json");
    let manifest: Option<Vec<ManifestEntry>> = if manifest_path.exists() {
        let manifest_json =
            fs::read_to_string(&manifest_path).expect("Failed to read manifest.json");
        let entries: Vec<ManifestEntry> =
            serde_json::from_str(&manifest_json).expect("Failed to parse manifest.json");
        println!(
            "Loaded manifest.json with {} entries — using catalog as model discovery source.",
            entries.len()
        );
        Some(entries)
    } else {
        None
    };

    // Model discovery: manifest-driven when present, filesystem fallback otherwise.
    let models: Vec<PathBuf> = if let Some(ref manifest) = manifest {
        manifest
            .iter()
            .map(|e| nondist_path.join(&e.filename))
            .filter(|p| p.exists())
            .collect()
    } else {
        println!(
            "NOTE: No manifest.json found in {:?} — falling back to \
             filesystem discovery, skipping classification validation.",
            nondist_path
        );
        find_models_in_dir(&nondist_path)
    };

    if models.is_empty() {
        println!(
            "[STATUS] SKIP_OPTIONAL: models_nondist_empty ({:?})",
            nondist_path
        );
        return;
    }

    println!(
        "Found {} non-distributable models to validate.",
        models.len()
    );

    for model_file in models {
        let filename = model_file
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("Validating non-distributable model: {}", filename);

        // 1. Parsing & Loading
        let json_data = match fs::read_to_string(&model_file) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP (read error): {filename}: {e}");
                continue;
            }
        };
        let model_data = match parse_nam_json(&json_data) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIP (parse error): {filename}: {e}");
                continue;
            }
        };
        let model = match build_model(&model_data) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("SKIP (dispatch error): {filename}: {e}");
                continue;
            }
        };

        let mut model = model;

        // 2. Classification validation (manifest-driven)
        if let Some(ref manifest) = manifest
            && let Some(entry) = manifest.iter().find(|e| e.filename == filename)
            && let Some(ref expected) = entry.expected_class
        {
            if expected.starts_with("Unknown") {
                println!(
                    "  ∎ Classification skip: {filename} — expected_class \
                     is '{expected}' (unsupported architecture, skip assertion)"
                );
            } else {
                let class_label = model.class_label();
                assert_eq!(
                    class_label,
                    expected.as_str(),
                    "Classification mismatch for {filename}: \
                     expected '{expected}', got '{class_label}'"
                );
            }
        }

        // Prewarm the model
        model.prewarm(2048);

        // 3. Determinism (Self-Consistency)
        let mut model_dup = build_model(&model_data)
            .expect("Failed to build duplicate model for determinism check");
        model_dup.prewarm(2048);

        let num_samples = 512;
        let input = generate_sine_440hz(num_samples);
        let mut out_a = vec![0.0f32; num_samples];
        let mut out_b = vec![0.0f32; num_samples];

        process_in_blocks(&mut model, &input, &mut out_a, 64);
        process_in_blocks(&mut model_dup, &input, &mut out_b, 64);

        let mse = compute_mse(&out_a, &out_b);
        assert_eq!(
            mse, 0.0,
            "Model {} is non-deterministic (MSE={:.6e})",
            filename, mse
        );

        // 4. Block Size Invariance.
        // All architectures re-chunk internally in sub-blocks of ≤
        // `WAVENET_MAX_NUM_FRAMES` (64 frames). A2 gained internal chunking in T2.1,
        // so block sizes > 64 are safe across the board and fully exercise streaming
        // invariance.
        let block_sizes = [1, 16, 32, 64, 128, 256, 512];
        let mut ref_model = build_model(&model_data).unwrap();
        ref_model.prewarm(2048);
        let mut ref_output = vec![0.0f32; num_samples];
        process_in_blocks(&mut ref_model, &input, &mut ref_output, 1);

        for &bs in &block_sizes {
            let mut test_model = build_model(&model_data).unwrap();
            test_model.set_max_buffer_size(bs).unwrap();
            test_model.prewarm(2048);
            let mut test_output = vec![0.0f32; num_samples];
            process_in_blocks(&mut test_model, &input, &mut test_output, bs);

            let test_mse = compute_mse(&ref_output, &test_output);
            assert!(
                test_mse < 1e-7,
                "Block size invariance violated for {filename} at \
                 block_size={bs} (MSE={test_mse:.6e})"
            );
        }

        // 5. Stability / Finiteness Check
        for (i, &s) in out_a.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced non-finite output at sample index {} ({})",
                filename,
                i,
                s
            );
        }

        // 6. Denormal/Subnormal Silence Test
        let silence = vec![0.0f32; num_samples];
        let mut silence_output = vec![0.0f32; num_samples];

        // Build a fresh model to avoid bias from the previous sine wave
        let mut silence_model = build_model(&model_data).unwrap();
        silence_model.prewarm(2048);
        process_in_blocks(&mut silence_model, &silence, &mut silence_output, 64);

        for (i, &s) in silence_output.iter().enumerate() {
            assert!(
                s.is_finite(),
                "Model {} produced non-finite output under silence at index {} ({})",
                filename,
                i,
                s
            );
            assert!(
                s.abs() < 1.0,
                "Model {} output diverged under silence at index {} ({})",
                filename,
                i,
                s
            );
            assert!(
                s == 0.0 || s.is_normal(),
                "Model {} produced subnormal float value under silence at index {} (bits: 0x{:08X})",
                filename,
                i,
                s.to_bits()
            );
        }

        println!(
            "✓ Model {} validated successfully across all checks.",
            filename
        );
    }
}
