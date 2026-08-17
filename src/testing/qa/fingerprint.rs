// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance-baseline environment fingerprint (S3.T1, R-09) — serde JSON
//! replacing the bash heredoc + `sed` pair of
//! `tests-performance-regression.sh::generate_fingerprint` /
//! `compare_fingerprint` (`:81-215`).
//!
//! One `Fingerprint` records the environment of a bootstrapped performance
//! baseline with the **same ten JSON keys** the bash wrote, so files written
//! by either side stay readable by the other. JSON is written with serde:
//! values containing `"` (a rustc banner, `RUSTFLAGS` with
//! `-C target-cpu="..."`) are escaped instead of corrupting the file.
//!
//! Comparison is field by field, a literal port of the bash semantics:
//! - always compared: `cpu_model`, `cpu_microarchitecture`, `rustc_version`,
//!   `target_triple`, `rustflags`, `build_profile`;
//! - `frequency_governor`: the **current** governor must be `performance`
//!   (else incomparable regardless of the baseline value); when it is, it
//!   must match a recorded baseline value;
//! - `bench_core`: only when the baseline recorded one;
//! - `physical_cores`: only when the baseline recorded a non-zero count;
//! - `git_commit` is provenance only — never compared.
//!
//! The first mismatch wins, in the bash comparison order, with a typed
//! reason: `IncomparableEnvironment { field, baseline, current }` — the
//! future receipt `reason` of the perf gate (S3.T3, classified `NOT_VERIFIED`
//! by [`crate::testing::qa::classify`]). A missing baseline file yields
//! `MissingBaseline`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::env::EnvProbe;

/// Reason field of the CPU model comparison (JSON key, `cpu_model`).
pub const FIELD_CPU_MODEL: &str = "cpu_model";
/// Reason field of the canonical ISA comparison (`cpu_microarchitecture`).
pub const FIELD_CPU_MICROARCHITECTURE: &str = "cpu_microarchitecture";
/// Reason field of the physical core count comparison (`physical_cores`).
pub const FIELD_PHYSICAL_CORES: &str = "physical_cores";
/// Reason field of the rustc version comparison (`rustc_version`).
pub const FIELD_RUSTC_VERSION: &str = "rustc_version";
/// Reason field of the target triple comparison (`target_triple`).
pub const FIELD_TARGET_TRIPLE: &str = "target_triple";
/// Reason field of the RUSTFLAGS comparison (`rustflags`).
pub const FIELD_RUSTFLAGS: &str = "rustflags";
/// Reason field of the build profile comparison (`build_profile`).
pub const FIELD_BUILD_PROFILE: &str = "build_profile";
/// Reason field of the frequency governor comparison (`frequency_governor`).
pub const FIELD_FREQUENCY_GOVERNOR: &str = "frequency_governor";
/// Reason field of the bench core comparison (`bench_core`).
pub const FIELD_BENCH_CORE: &str = "bench_core";

/// Cargo profile the perf gate always benchmarks (the bash hardcodes
/// `build_profile="release"`; `cargo bench` is release-only).
pub const DEFAULT_BUILD_PROFILE: &str = "release";

/// Persisted environment fingerprint of a bootstrapped performance baseline.
///
/// Field names and JSON keys are byte-compatible with the bash heredoc of
/// `generate_fingerprint` (`tests-performance-regression.sh:97-110`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// CPU model name (`EnvProbe::cpu_model`; `unknown` when unavailable).
    pub cpu_model: String,
    /// Canonical ISA string (`EnvProbe::effective_isa`).
    pub cpu_microarchitecture: String,
    /// Physical core count (`EnvProbe::physical_cores`).
    pub physical_cores: u32,
    /// First line of `rustc --version`.
    pub rustc_version: String,
    /// `host:` triple of `rustc -vV`.
    pub target_triple: String,
    /// `RUSTFLAGS` value at baseline time (empty when unset).
    pub rustflags: String,
    /// Cargo profile of the benchmark (see [`DEFAULT_BUILD_PROFILE`]).
    pub build_profile: String,
    /// CPU frequency governor of CPU 0.
    pub frequency_governor: String,
    /// Producing git commit (provenance only — never compared).
    pub git_commit: String,
    /// Bench core pinning (`NAM_BENCH_CORE`; empty when unpinned).
    pub bench_core: String,
}

/// Typed failure of fingerprint I/O and comparison (S3.T1).
///
/// The variant names map 1:1 to the perf-gate reason strings of the bash
/// (`MISSING_BASELINE` / `INCOMPARABLE_ENVIRONMENT`), which
/// [`crate::testing::qa::classify::classify_regression_outcome`] classifies
/// as `NOT_VERIFIED`.
#[derive(Debug, thiserror::Error)]
pub enum FingerprintError {
    /// Baseline fingerprint file absent — nothing to compare.
    #[error("MISSING_BASELINE: no baseline fingerprint found")]
    MissingBaseline,
    /// Environment drift between baseline and current.
    #[error("INCOMPARABLE_ENVIRONMENT: {field} mismatch (baseline={baseline}, current={current})")]
    IncomparableEnvironment {
        /// Canonical field name (see the `FIELD_*` constants).
        field: &'static str,
        /// Value recorded by the baseline fingerprint.
        baseline: String,
        /// Value of the current environment.
        current: String,
    },
    /// The fingerprint file is not valid JSON / violates the schema.
    #[error("invalid fingerprint JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// Filesystem error while reading or writing the fingerprint.
    #[error("fingerprint I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl Fingerprint {
    /// Builds a fingerprint from an environment probe (`EnvProbe`, R-09) —
    /// the single probe shape shared with receipts and the dashboard.
    pub fn from_env_probe(probe: &EnvProbe, bench_core: &str) -> Self {
        Fingerprint {
            cpu_model: probe.cpu_model.clone(),
            cpu_microarchitecture: probe.effective_isa.to_string(),
            physical_cores: probe.physical_cores,
            rustc_version: probe.rustc_version.clone(),
            target_triple: probe.host_triple.clone(),
            rustflags: probe.rustflags.clone(),
            build_profile: DEFAULT_BUILD_PROFILE.to_string(),
            frequency_governor: probe.frequency_governor.clone(),
            git_commit: probe.git_commit.clone(),
            bench_core: bench_core.to_string(),
        }
    }

    /// Probes the live host and builds the fingerprint of the current
    /// environment.
    pub fn probe(bench_core: &str) -> Self {
        Self::from_env_probe(&EnvProbe::probe(), bench_core)
    }

    /// Serializes the fingerprint as pretty-printed JSON (UTF-8), like the
    /// quality contract — diffs stay reviewable.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parses a persisted fingerprint (JSON only, fail-closed).
    pub fn from_json_str(input: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(input)
    }

    /// Writes the fingerprint to `path`, creating parent directories.
    pub fn write_to_path(&self, path: &Path) -> Result<(), FingerprintError> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let json = self.to_json_pretty()?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Reads the fingerprint from `path`; [`FingerprintError::MissingBaseline`]
    /// when the file does not exist.
    pub fn read_from_path(path: &Path) -> Result<Self, FingerprintError> {
        if !path.exists() {
            return Err(FingerprintError::MissingBaseline);
        }
        let text = std::fs::read_to_string(path)?;
        Ok(Self::from_json_str(&text)?)
    }

    /// Compares this (current) environment against a stored baseline, field
    /// by field, mirroring `compare_fingerprint`
    /// (`tests-performance-regression.sh:115-215`).
    ///
    /// `Ok(())` means the environments are comparable. The first mismatch
    /// wins, in the bash comparison order, with the typed reason carrying the
    /// canonical field name plus both values.
    pub fn compare(&self, baseline: &Fingerprint) -> Result<(), FingerprintError> {
        if self.cpu_model != baseline.cpu_model {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_CPU_MODEL,
                baseline: baseline.cpu_model.clone(),
                current: self.cpu_model.clone(),
            });
        }
        if self.cpu_microarchitecture != baseline.cpu_microarchitecture {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_CPU_MICROARCHITECTURE,
                baseline: baseline.cpu_microarchitecture.clone(),
                current: self.cpu_microarchitecture.clone(),
            });
        }
        if self.rustc_version != baseline.rustc_version {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_RUSTC_VERSION,
                baseline: baseline.rustc_version.clone(),
                current: self.rustc_version.clone(),
            });
        }
        if self.target_triple != baseline.target_triple {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_TARGET_TRIPLE,
                baseline: baseline.target_triple.clone(),
                current: self.target_triple.clone(),
            });
        }
        if self.rustflags != baseline.rustflags {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_RUSTFLAGS,
                baseline: baseline.rustflags.clone(),
                current: self.rustflags.clone(),
            });
        }
        if self.build_profile != baseline.build_profile {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_BUILD_PROFILE,
                baseline: baseline.build_profile.clone(),
                current: self.build_profile.clone(),
            });
        }
        if self.frequency_governor != "performance" {
            // Bash: a current governor != `performance` is incomparable
            // regardless of the baseline value.
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_FREQUENCY_GOVERNOR,
                baseline: baseline.frequency_governor.clone(),
                current: self.frequency_governor.clone(),
            });
        }
        if !baseline.frequency_governor.is_empty()
            && self.frequency_governor != baseline.frequency_governor
        {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_FREQUENCY_GOVERNOR,
                baseline: baseline.frequency_governor.clone(),
                current: self.frequency_governor.clone(),
            });
        }
        if !baseline.bench_core.is_empty() && self.bench_core != baseline.bench_core {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_BENCH_CORE,
                baseline: baseline.bench_core.clone(),
                current: self.bench_core.clone(),
            });
        }
        if baseline.physical_cores != 0 && self.physical_cores != baseline.physical_cores {
            return Err(FingerprintError::IncomparableEnvironment {
                field: FIELD_PHYSICAL_CORES,
                baseline: baseline.physical_cores.to_string(),
                current: self.physical_cores.to_string(),
            });
        }
        Ok(())
    }

    /// Convenience for the perf gate: reads the baseline fingerprint at
    /// `path` and compares this fingerprint against it — fail-closed on both
    /// a missing baseline and any environment drift.
    pub fn compare_against_path(&self, path: &Path) -> Result<(), FingerprintError> {
        let baseline = Self::read_from_path(path)?;
        self.compare(&baseline)
    }
}

#[cfg(test)]
#[path = "fingerprint_test.rs"]
mod fingerprint_test;
