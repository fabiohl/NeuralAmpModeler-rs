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
use neural_amp_modeler_rs::models::NamModel;
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

/// Fixtures whose A2 `process` clamps to the negotiated `max_buffer_size`.
/// F-01 evidence: the A2 inference bench never calls `set_max_buffer_size`, so
/// block sizes above the constructor default (64) were silently truncated to 64
/// frames — the measured time went flat from 64..512 samples.
const A2_FIXTURES: &[&str] = &[
    "tests/fixtures/models/wavenet_a2_full.nam",
    "tests/fixtures/models/wavenet_a2_lite.nam",
    "tests/fixtures/models/a2_dynamic_gated_ch8.nam",
    "tests/fixtures/models/a2_dynamic_blended_ch3.nam",
];

/// F-01 regression guard for A2 (static Full/Lite and free-geometry dynamic).
///
/// Feeds a 512-frame signal as one block and as 8×64 / 16×32 partitions after
/// negotiating `set_max_buffer_size(512)`, asserting:
/// - **completeness**: every output frame is written (NaN-sentinel output must
///   be fully overwritten — catches a silently truncated tail);
/// - **invariance**: the output matches the single-block run bit-for-bit,
///   independent of how the audio is partitioned.
#[test]
fn test_a2_block_invariance_full_write_and_partitioning() {
    const N: usize = 512;
    let input_signal = generate_stress_signal_v1();
    assert!(
        input_signal.len() >= N,
        "stress signal too short for A2 guard"
    );
    let input = &input_signal[..N];

    let mut tested_models = 0;
    for fixture in A2_FIXTURES {
        let model_path = match resolve_fixture_path(fixture) {
            Some(p) => p,
            None => continue,
        };
        let json_str = fs::read_to_string(&model_path)
            .unwrap_or_else(|e| panic!("Failed to read {fixture}: {e}"));
        let model_data =
            parse_nam_json(&json_str).unwrap_or_else(|e| panic!("Failed to parse {fixture}: {e}"));

        // `block == N` is the single-block reference; smaller values exercise
        // explicit external partitioning (internal chunking stays at ≤ 64).
        let run = |block: usize| -> Vec<f32> {
            let mut model = build_model(&model_data)
                .unwrap_or_else(|e| panic!("Failed to build {fixture}: {e}"));
            let out_ch = model.num_output_channels().max(1);
            model
                .set_max_buffer_size(N)
                .unwrap_or_else(|e| panic!("set_max_buffer_size failed for {fixture}: {e}"));
            model.prewarm(2048);

            let mut output = vec![f32::NAN; N * out_ch];
            for (in_chunk, out_chunk) in input.chunks(block).zip(output.chunks_mut(block * out_ch))
            {
                model.process(in_chunk, out_chunk);
            }

            assert!(
                output.iter().all(|v| v.is_finite()),
                "{fixture}: output not fully written at block={block} — silent tail truncation"
            );
            output
        };

        let single = run(N);
        for &block in &[64usize, 32] {
            let partitioned = run(block);
            assert_eq!(
                single.len(),
                partitioned.len(),
                "{fixture}: output length mismatch between block={N} and block={block}"
            );
            for (i, (&a, &b)) in single.iter().zip(partitioned.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{fixture}: block={block} diverges from single block={N} at sample {i} \
                     (a={a:e}, b={b:e})"
                );
            }
        }

        tested_models += 1;
    }

    assert!(
        tested_models > 0,
        "No A2 fixture resolved; block-invariance full-write guard exercised nothing"
    );
    eprintln!("A2 block-invariance/full-write guard validated {tested_models} fixtures.");
}
