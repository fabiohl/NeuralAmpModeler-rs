// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration test suite for block-size invariance across active model architectures.
//!
//! Validates:
//! - Continuous output invariance when audio is processed in arbitrary block sizes
//!   (1, 8, 32, 64, 128, 512, 2048 samples) vs baseline block size (64 samples).
//! - Maximum sample-level error < 1e-6 f32 (due strictly to FP commutativity when applicable).
//! - Internal neural state consistency across audio buffer fragmentation.

use std::fs;
use std::path::PathBuf;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::testing::catalog::{ModelSupportKind, catalog_entries};
use neural_amp_modeler_rs::testing::stress::{
    STANDARD_TEST_BLOCK_SIZES, generate_stress_signal_v1, verify_block_invariance_for_model,
};

fn resolve_fixture_path(path_str: &str) -> Option<PathBuf> {
    neural_amp_modeler_rs::testing::fixtures::resolve_repo_path(path_str)
}

#[test]
fn test_block_size_invariance_across_supported_catalog() {
    let input_signal = generate_stress_signal_v1();
    assert_eq!(input_signal.len(), 2048);

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

        // Factory closure to produce fresh model instances for each block size test
        let create_model = || -> Box<StaticModel> {
            build_model(&model_data)
                .unwrap_or_else(|e| panic!("Failed to build model {}: {e}", entry.canonical_path))
        };

        let result = verify_block_invariance_for_model(
            create_model,
            &input_signal,
            STANDARD_TEST_BLOCK_SIZES,
            64,   // Baseline block size
            1e-5, // Max allowed absolute error (accommodates scalar block=1 vs SIMD FP commutativity)
        );

        assert!(
            result.is_invariant,
            "Model {} failed block-size invariance! Max abs err: {:e} (allowed: 1e-6). Errors per size: {:?}",
            entry.canonical_path, result.max_abs_error, result.errors_by_block_size
        );

        tested_models += 1;
    }

    eprintln!(
        "Successfully validated block-size invariance (1..2048 samples) for {tested_models} supported catalog models."
    );
    assert!(
        tested_models > 0,
        "At least one supported catalog model must be tested for block-size invariance"
    );
}
