// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Centralized fixture-path resolution for the crate's test infrastructure.
//!
//! Provides a single source of truth for locating test models and fixture
//! directories, with a multi-stage fallback chain:
//!
//! 1. `tests/fixtures/models-nondist/` — local non-distributable models
//! 2. `../third-party/nam_t3k/` — shared third-party model archive
//! 3. `tests/fixtures/models/` — version-controlled fixture models
//!
//! All paths are resolved relative to this crate's manifest directory, which
//! is the authoritative home of the fixture governance (manifest, freshness
//! gate, golden pipeline). Dependent crates and host integrations get the
//! same paths via this accessor under the `testing` feature gate.

use std::path::{Path, PathBuf};

use crate::loader::loaded_model_pair::{DEFAULT_INPUT_LEVEL_DBU, DEFAULT_LOUDNESS_DB};

/// Returns the root directory of all test fixtures.
///
/// Path: `{core_manifest}/tests/fixtures/`
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Resolves the path to a test model file by searching the fallback chain.
///
/// The search order is:
/// 1. `tests/fixtures/models-nondist/{name}` — if present
/// 2. `../third-party/nam_t3k/{name}` — if present
/// 3. `tests/fixtures/models/{name}` — default (version-controlled)
pub fn model_path(name: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let nondist = manifest.join("tests/fixtures/models-nondist").join(name);
    if nondist.exists() {
        return nondist;
    }

    let t3k = manifest.join("../third-party/nam_t3k").join(name);
    if t3k.exists() {
        return t3k;
    }

    manifest.join("tests/fixtures/models").join(name)
}

/// Returns the path to the models subdirectory under fixtures.
///
/// Path: `{core_manifest}/tests/fixtures/models/`
pub fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models")
}

/// Checks whether a named model fixture exists in any of the search locations.
pub fn has_model(name: &str) -> bool {
    model_path(name).exists()
}

/// Returns the canonical path for a golden binary fixture.
///
/// Path: `{core_manifest}/tests/fixtures/{name}`
pub fn golden_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// Returns the path to the stress-signal WAV directory.
///
/// Path: `{core_manifest}/tests/fixtures/`
pub fn stress_signal_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// Returns the path to the NAMCore C++ render binary built from source.
///
/// Search order:
/// 1. `{core_manifest}/build/namcore_render/tools/render` (default cmake output)
/// 2. `{core_manifest}/build/namcore_render/Release/render`
/// 3. `{core_manifest}/build/namcore_render/Debug/render`
/// 4. Any subdirectory of `build/namcore_render/` containing a `render` executable
///
/// Returns the first existing candidate, or the first path as default even if
/// it does not exist (caller should check existence with `Path::exists()`).
pub fn render_bin_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let build_dir = manifest.join("build/namcore_render");

    for candidate in &["tools/render", "Release/render", "Debug/render"] {
        let p = build_dir.join(candidate);
        if p.exists() {
            return p;
        }
    }

    if build_dir.exists() {
        for entry in std::fs::read_dir(&build_dir)
            .into_iter()
            .flatten()
            .flatten()
        {
            let p = entry.path();
            if p.is_dir() {
                let c = p.join("render");
                if c.exists() {
                    return c;
                }
            }
        }
    }

    build_dir.join("tools/render")
}

/// Computes calibration multipliers from a NAM model JSON file.
///
/// Reads `metadata.loudness` and `metadata.input_level_dbu` from the JSON,
/// computes dB adjustments against internal defaults, and converts to linear
/// multipliers via the gain LUT, mirroring the plugin's actual DSP pipeline
/// (see `loader/build.rs:177-183`).
///
/// Returns `(input_mult_adj, output_mult_adj)` where:
/// - `input_mult_adj` scales the input signal before the model
/// - `output_mult_adj` scales the output signal after the model
///
/// This is the documented recomputation path for parity testing: it uses the
/// same LUT-based `db_to_linear` as the production loader, so residuals
/// reflect only DSP divergence, not LUT-vs-exact mismatch.
pub fn calibration_multipliers_from_model_json(model_json_path: &Path) -> (f32, f32) {
    let file =
        std::fs::File::open(model_json_path).expect("Failed to open model JSON for calibration");
    let reader = std::io::BufReader::new(file);
    let model_data: serde_json::Value =
        serde_json::from_reader(reader).expect("Failed to parse model JSON for calibration");

    let metadata = model_data.get("metadata");

    let loudness = metadata
        .and_then(|m| m.get("loudness"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_LOUDNESS_DB);

    let input_level_dbu = metadata
        .and_then(|m| m.get("input_level_dbu"))
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(DEFAULT_INPUT_LEVEL_DBU);

    let input_db_adj = DEFAULT_INPUT_LEVEL_DBU - input_level_dbu;
    let output_db_adj = DEFAULT_LOUDNESS_DB - loudness;

    let lut = crate::math::dsp::gain_lut::get_gain_lut();
    let input_mult_adj = lut.db_to_linear(input_db_adj);
    let output_mult_adj = lut.db_to_linear(output_db_adj);

    (input_mult_adj, output_mult_adj)
}
