// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Disk validation for V1 and V2 golden catalogs.

use std::path::Path;

use super::golden::{GoldenGenEntry, V2Distribution, v2_catalog_entries, v2_sample_rates_for};

/// Outcome of a `validate_v2_catalog` run: disk presence of every expected
/// V2 model fixture and golden binary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogStatus {
    /// Number of catalog entries validated.
    pub entries_checked: usize,
    /// Number of fixture files checked (models + goldens).
    pub fixtures_checked: usize,
    /// Number of fixture files found on disk.
    pub present: usize,
    /// Required fixture paths absent from disk (hard preflight failure).
    pub missing_required: Vec<String>,
    /// Optional (non-distributable) fixture paths absent from disk.
    pub missing_optional: Vec<String>,
    /// Golden names with a documented skip reason — absence is expected.
    pub known_gaps: Vec<&'static str>,
    /// Number of known gap fixture files (goldens skipped due to known gaps).
    pub known_gap_files: usize,
}

impl CatalogStatus {
    /// True when no required fixture is missing.
    pub fn is_ok(&self) -> bool {
        self.missing_required.is_empty()
    }

    /// Emits a typed capability receipt for the V2 golden catalog.
    ///
    /// Uses the `MISSING-REQUIRED:` / `MISSING-OPTIONAL:` markers also emitted
    /// by the fixture catalog receipt, so `utils/tests-long.sh` can count
    /// failures from the log with a single grep.
    pub fn receipt(&self) -> String {
        self.receipt_for(
            "V2",
            "V2 Golden Catalog Capability Receipt",
            v2_catalog_entries().len(),
        )
    }

    /// Emits a typed capability receipt for the V1 golden catalog.
    ///
    /// Same marker contract as [`CatalogStatus::receipt`]: the shell preflight
    /// counts `MISSING-REQUIRED:` lines from the `catalog_preflight` log, so
    /// both receipts must keep that marker verbatim.
    pub fn receipt_v1(&self) -> String {
        self.receipt_for(
            "V1",
            "V1 Golden Catalog Capability Receipt",
            V1_GOLDEN_CATALOG.len(),
        )
    }

    fn receipt_for(&self, version: &str, title: &str, total_entries: usize) -> String {
        let mut lines = Vec::new();
        lines.push(format!("=== {title} ==="));
        lines.push(format!(
            "Validated {}/{} catalog entries, {} fixture files checked, {} present.",
            self.entries_checked, total_entries, self.fixtures_checked, self.present
        ));
        lines.push("-".repeat(100));
        if !self.known_gaps.is_empty() {
            lines.push(format!(
                "  KNOWN-GAP (expected absence, {}): {}",
                self.known_gaps.len(),
                self.known_gaps.join(", ")
            ));
        }
        if !self.missing_optional.is_empty() {
            lines.push(format!(
                "=== Optional (non-distributable) {version} fixtures absent ({} file(s)) ===",
                self.missing_optional.len()
            ));
            for path in &self.missing_optional {
                lines.push(format!(
                    "  MISSING-OPTIONAL: {path} ({version} — OptionalExternal, skip gracefully)"
                ));
            }
        }
        if !self.missing_required.is_empty() {
            lines.push(format!(
                "=== Required {version} fixtures absent ({} file(s)) ===",
                self.missing_required.len()
            ));
            for path in &self.missing_required {
                lines.push(format!(
                    "  MISSING-REQUIRED: {path} ({version} — RequiredLocal, preflight hard-fail)"
                ));
            }
        } else {
            lines.push(format!(
                "=== All {version} RequiredLocal fixtures present ✓ ==="
            ));
        }
        lines.push("-".repeat(100));
        if !self.missing_required.is_empty() {
            lines.push(format!(
                "Run './tests/fixtures/golden_gen_build.sh' to regenerate missing {version} \
                 golden vectors (C++ toolchain + NeuralAmpModelerCore required)."
            ));
        }
        lines.join("\n")
    }
}

/// Error surfaced when the V2 catalog cannot be validated.
#[derive(Debug)]
pub enum CatalogError {
    /// The fixtures root directory is not available at the resolved path.
    FixturesRootUnavailable(std::path::PathBuf),
    /// Internal catalog invariant violated (e.g. empty registry).
    CatalogInvariant(&'static str),
}

impl std::fmt::Display for CatalogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogError::FixturesRootUnavailable(path) => write!(
                f,
                "V2 catalog validation aborted: fixtures root not found at {path:?}. \
                 Run the suite from the crate root or restore tests/fixtures/."
            ),
            CatalogError::CatalogInvariant(msg) => {
                write!(f, "V2 catalog validation aborted: {msg}")
            }
        }
    }
}

impl std::error::Error for CatalogError {}

/// Validates the canonical V2 golden catalog against disk.
///
/// For every entry of the validated V2 subset (24):
/// - the model fixture must resolve via `fixtures::model_path` (required or
///   optional per `model_distribution`);
/// - every golden binary expected by the entry's sample-rate scope must exist
///   in `tests/fixtures/` (required or optional per `golden_distribution`);
/// - entries with a documented `skip_reason` (KnownGap) are recorded as
///   expected-absent and never fail the gate.
///
/// Validates a sequence of golden catalog entries against disk paths.
///
/// Invariant:
/// `status.present + status.missing_required.len() + status.missing_optional.len() + status.known_gap_files == status.fixtures_checked`
pub fn validate_v2_catalog_entries<F>(
    entries: &[&GoldenGenEntry],
    fixtures_root: &Path,
    model_exists: F,
) -> CatalogStatus
where
    F: Fn(&str) -> bool,
{
    let mut status = CatalogStatus::default();

    for &entry in entries {
        status.entries_checked += 1;

        // Model fixture check
        status.fixtures_checked += 1;
        let model_ok = model_exists(entry.nam_file);
        if model_ok {
            status.present += 1;
        } else {
            let path = format!("tests/fixtures/models/{}", entry.nam_file);
            match entry.model_distribution {
                V2Distribution::RequiredLocal => status.missing_required.push(path),
                V2Distribution::OptionalExternal => status.missing_optional.push(path),
            }
        }

        // Golden fixtures check
        let sample_rates = v2_sample_rates_for(entry.nam_file);
        if entry.skip_reason.is_some() {
            status.known_gaps.push(entry.golden_name);
            status.fixtures_checked += sample_rates.len();
            status.known_gap_files += sample_rates.len();
            continue;
        }

        for &sr in sample_rates {
            status.fixtures_checked += 1;
            let file = format!("{}_v2_{}.bin", entry.golden_name, sr);
            let path = fixtures_root.join(&file);
            if path.exists() {
                status.present += 1;
            } else {
                let rel = format!("tests/fixtures/{file}");
                match entry.golden_distribution {
                    V2Distribution::RequiredLocal => status.missing_required.push(rel),
                    V2Distribution::OptionalExternal => status.missing_optional.push(rel),
                }
            }
        }
    }

    status
}

/// Validates the canonical V2 golden catalog against disk.
///
/// For every entry of the validated V2 subset (24):
/// - the model fixture must resolve via `fixtures::model_path` (required or
///   optional per `model_distribution`);
/// - every golden binary expected by the entry's sample-rate scope must exist
///   in `tests/fixtures/` (required or optional per `golden_distribution`);
/// - entries with a documented `skip_reason` (KnownGap) are recorded as
///   expected-absent and never fail the gate.
///
/// Returns `Ok(CatalogStatus)` describing disk integrity, or `Err` when the
/// environment is not a valid crate checkout. Consumable by integration tests
/// (`tests/models/golden_vectors.rs::catalog_preflight`) and binaries.
pub fn validate_v2_catalog() -> Result<CatalogStatus, CatalogError> {
    let entries = v2_catalog_entries();
    if entries.is_empty() {
        return Err(CatalogError::CatalogInvariant(
            "GOLDEN_GEN_CATALOG defines no V2 entries (in_v2_catalog is all false)",
        ));
    }

    let fixtures_root = crate::testing::fixtures::fixture_dir();
    if !fixtures_root.is_dir() {
        return Err(CatalogError::FixturesRootUnavailable(fixtures_root));
    }

    let status = validate_v2_catalog_entries(&entries, &fixtures_root, |nam_file| {
        crate::testing::fixtures::model_path(nam_file).exists()
    });

    Ok(status)
}

// =============================================================================
// V1 Golden Catalog — single source of truth
// =============================================================================
//
// The former bash lists in `utils/tests-long.sh` (REQUIRED_GOLDEN_MODELS,
// NONDIST_GOLDEN_MODELS and REQUIRED_CABSIM_GOLDENS) are hereby eliminated:
// this registry is the ONLY definition of the v1 golden matrix (48 kHz `.bin`
// vectors) and its distribution policy. Disk presence is validated
// fail-closed by `validate_v1_goldens` / `catalog_preflight` before any timed
// long-suite phase, and by `check_freshness` (nam_freshness) against the
// `.golden_manifest.sha256` integrity manifest.

/// Single v1 golden registry entry: an expected `.bin` file under
/// `tests/fixtures/` and its distribution policy.
///
/// Distribution taxonomy mirrors the V2 registry:
/// `DistributedCore` → `RequiredLocal` (hard preflight failure when absent),
/// `LocalNonDistributable` → `OptionalExternal` (graceful typed skip).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V1GoldenEntry {
    /// Golden binary basename under `tests/fixtures/`.
    pub golden_file: &'static str,
    /// Distribution policy (RequiredLocal hard-fails preflight when absent).
    pub distribution: V2Distribution,
    /// Human-readable description (source model family / convolution scope).
    pub description: &'static str,
}

/// Canonical v1 golden registry — 13 entries (9 DistributedCore model goldens,
/// 1 LocalNonDistributable model golden, 3 CabSim convolution goldens).
pub static V1_GOLDEN_CATALOG: &[V1GoldenEntry] = &[
    // ── DistributedCore model goldens (v1, 48 kHz) — RequiredLocal ────────
    V1GoldenEntry {
        golden_file: "golden_wavenet_standard.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Standard (CH=16) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_feather.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Feather (CH=8) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_nano.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet Nano (CH=4) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a1_standard.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A1 Standard v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a2_full.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A2 Full (CH=8) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_wavenet_a2_lite.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "WaveNet A2 Lite (CH=3) v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_1x16.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM 1×16 v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_2x8.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM 2×8 v1 golden",
    },
    V1GoldenEntry {
        golden_file: "golden_lstm_official.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "LSTM Official v1 golden",
    },
    // ── LocalNonDistributable model golden (v1, 48 kHz) — OptionalExternal ─
    V1GoldenEntry {
        golden_file: "golden_wavenet_lite.bin",
        distribution: V2Distribution::OptionalExternal,
        description: "WaveNet Lite (CH=12, community) v1 golden",
    },
    // ── CabSim convolution goldens — RequiredLocal ────────────────────────
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_short.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim short IR convolution golden",
    },
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_medium.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim medium IR convolution golden",
    },
    V1GoldenEntry {
        golden_file: "golden_cabsim_cpp_long.bin",
        distribution: V2Distribution::RequiredLocal,
        description: "CabSim long IR convolution golden",
    },
];

/// Validates the canonical V1 golden catalog against disk.
///
/// For every entry of [`V1_GOLDEN_CATALOG`], checks that the `.bin` file exists
/// under `tests/fixtures/`. RequiredLocal entries (DistributedCore model
/// goldens + CabSim convolution goldens) absent from disk are hard preflight
/// failures; the single OptionalExternal entry (WaveNet Lite,
/// non-distributable) is recorded as a graceful missing-optional.
///
/// Returns `Ok(CatalogStatus)` describing disk integrity, or `Err` when the
/// environment is not a valid crate checkout. Consumable by integration tests
/// (`tests/models/golden_vectors.rs::catalog_preflight`) and binaries.
pub fn validate_v1_goldens() -> Result<CatalogStatus, CatalogError> {
    if V1_GOLDEN_CATALOG.is_empty() {
        return Err(CatalogError::CatalogInvariant(
            "V1_GOLDEN_CATALOG defines no entries",
        ));
    }

    let fixtures_root = crate::testing::fixtures::fixture_dir();
    if !fixtures_root.is_dir() {
        return Err(CatalogError::FixturesRootUnavailable(fixtures_root));
    }

    let mut status = CatalogStatus::default();

    for entry in V1_GOLDEN_CATALOG {
        status.entries_checked += 1;
        status.fixtures_checked += 1;
        let path = fixtures_root.join(entry.golden_file);
        if path.exists() {
            status.present += 1;
        } else {
            let rel = format!("tests/fixtures/{}", entry.golden_file);
            match entry.distribution {
                V2Distribution::RequiredLocal => status.missing_required.push(rel),
                V2Distribution::OptionalExternal => status.missing_optional.push(rel),
            }
        }
    }

    Ok(status)
}
