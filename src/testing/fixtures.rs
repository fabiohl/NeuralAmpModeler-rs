// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Centralized fixture-path resolution for the crate's test infrastructure.
//!
//! Single source of truth for locating test models, catalog paths, vendor
//! mirrors, goldens, and the C++ render binary. All resolution is relative to
//! this crate's manifest directory (`CARGO_MANIFEST_DIR`). Optional third-party
//! trees are never required: callers should treat a missing path as SKIP.
//!
//! ## Model basename search (`model_path`)
//!
//! 1. `$NAM_MODELS_DIR/{name}`
//! 2. `$NAM_THIRD_PARTY_DIR/community_models/{name}`
//! 3. `tests/fixtures/models-nondist/{name}`
//! 4. `third-party/community_models/{name}`
//! 5. `tests/fixtures/models/{name}` (default, may be absent)

use std::path::{Path, PathBuf};

use crate::loader::loaded_model_pair::{DEFAULT_INPUT_LEVEL_DBU, DEFAULT_LOUDNESS_DB};

/// Crate manifest directory (`CARGO_MANIFEST_DIR`).
pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Returns the root directory of all test fixtures.
///
/// Path: `{manifest}/tests/fixtures/`
pub fn fixture_dir() -> PathBuf {
    manifest_dir().join("tests/fixtures")
}

/// Repo-local third-party base (gitignored).
///
/// Override with `NAM_THIRD_PARTY_DIR`. Default: `{manifest}/third-party`.
pub fn third_party_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_THIRD_PARTY_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    manifest_dir().join("third-party")
}

/// NeuralAmpModelerCore vendor mirror directory.
///
/// Override with `NAM_CORE_DIR`. Default: `{third_party}/NeuralAmpModelerCore`.
pub fn nam_core_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_CORE_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    third_party_dir().join("NeuralAmpModelerCore")
}

/// NeuralAmpModelerPlugin vendor mirror directory.
///
/// Override with `NAM_PLUGIN_DIR`. Default: `{third_party}/NeuralAmpModelerPlugin`.
pub fn nam_plugin_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_PLUGIN_DIR") {
        let p = PathBuf::from(dir);
        if !p.as_os_str().is_empty() {
            return p;
        }
    }
    third_party_dir().join("NeuralAmpModelerPlugin")
}

/// Optional community / non-distributable model archive directory.
///
/// Default: `{third_party}/community_models` (often a symlink).
pub fn community_models_dir() -> PathBuf {
    third_party_dir().join("community_models")
}

/// Resolves a path string relative to the crate manifest when present on disk.
///
/// Used for catalog `canonical_path` / `aliases` entries such as
/// `tests/fixtures/models/….nam` or `third-party/community_models/….nam`.
/// Returns `None` when the file is absent (caller should SKIP).
pub fn resolve_repo_path(path_str: &str) -> Option<PathBuf> {
    let p = Path::new(path_str);
    if p.is_absolute() {
        return p.exists().then(|| p.to_path_buf());
    }
    let cand = manifest_dir().join(p);
    cand.exists().then_some(cand)
}

/// Resolves the path to a test model file by basename.
///
/// Search order:
/// 1. `$NAM_MODELS_DIR/{name}`
/// 2. `$NAM_THIRD_PARTY_DIR/community_models/{name}` (via [`third_party_dir`])
/// 3. `tests/fixtures/models-nondist/{name}`
/// 4. `third-party/community_models/{name}`
/// 5. `tests/fixtures/models/{name}` (default return even if missing)
pub fn model_path(name: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_MODELS_DIR") {
        let p = PathBuf::from(&dir).join(name);
        if p.exists() {
            return p;
        }
    }

    let community = community_models_dir().join(name);
    if community.exists() {
        return community;
    }

    let manifest = manifest_dir();
    let nondist = manifest.join("tests/fixtures/models-nondist").join(name);
    if nondist.exists() {
        return nondist;
    }

    manifest.join("tests/fixtures/models").join(name)
}

/// Returns the path to the models subdirectory under fixtures.
///
/// Path: `{manifest}/tests/fixtures/models/`
pub fn models_dir() -> PathBuf {
    manifest_dir().join("tests/fixtures/models")
}

/// Checks whether a named model fixture exists in any of the search locations.
pub fn has_model(name: &str) -> bool {
    model_path(name).exists()
}

/// Returns the canonical path for a golden binary fixture.
///
/// Path: `{manifest}/tests/fixtures/{name}`
pub fn golden_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// Returns the path to the stress-signal WAV directory.
///
/// Path: `{manifest}/tests/fixtures/`
pub fn stress_signal_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// Returns the path to the NAMCore C++ render binary built from source.
///
/// Search order:
/// 1. `{manifest}/build/namcore_render/tools/render` (default cmake output)
/// 2. `{manifest}/build/namcore_render/Release/render`
/// 3. `{manifest}/build/namcore_render/Debug/render`
/// 4. Any subdirectory of `build/namcore_render/` containing a `render` executable
///
/// Returns the first existing candidate, or the first path as default even if
/// it does not exist (caller should check existence with `Path::exists()`).
pub fn render_bin_path() -> PathBuf {
    let build_dir = manifest_dir().join("build/namcore_render");

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
/// multipliers via the gain LUT, mirroring the production loader pipeline
/// (see `loader/build.rs`).
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
