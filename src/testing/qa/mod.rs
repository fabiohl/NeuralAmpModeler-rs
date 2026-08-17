// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed quality-contract schema — the JSON-only authority of the QA gate.
//!
//! Defines the canonical machine-readable contract types that replace the
//! ASCII snapshot at `docs/quality-contract.txt` as the single source of
//! truth (finding R-01, sprint S1). Shell scripts never hand-serialize this
//! schema: serde does.
//!
//! The types mirror the illustrative schema of `.agents/TODO-refatora.md`
//! §R-01 and are consumed by the one-shot transcription (S1.T3) and by the
//! future `nam_quality` engine (S2). They are intentionally **not**
//! re-exported from the crate root: the public crate API stays free of
//! QA-contract types.
//!
//! Available only with the `testing` feature (the whole `testing` module is
//! feature-gated at `lib.rs`).

use serde::{Deserialize, Serialize};

/// Schema version of the persisted quality contract.
///
/// Bump (and gate reads) whenever the persisted shape changes. The verify
/// engine is fail-closed on any other version.
pub const SCHEMA_VERSION: u32 = 1;

/// Root of the persisted quality contract (`docs/quality-contract.json`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityContract {
    /// Schema version (see [`SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// ISO-8601 timestamp of the measured snapshot.
    pub generated_at: String,
    /// Machine/toolchain provenance of the measured snapshot.
    pub provenance: Provenance,
    /// Numeric acceptance envelopes applied by the verify engine.
    pub envelopes: Envelopes,
    /// Canonical fidelity entries (golden vectors + additional coverage).
    pub fidelity: Vec<FidelityEntry>,
    /// Real-time latency entries (Model Inference Core + DSP Infrastructure).
    pub performance: Vec<PerformanceEntry>,
}

impl QualityContract {
    /// Serializes the contract to pretty-printed JSON (UTF-8).
    ///
    /// The persisted artifact is always pretty-printed so diffs stay
    /// reviewable; compact form is only used in-memory.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Parses a persisted contract, fail-closed on unsupported schema versions.
    pub fn from_json_str(input: &str) -> Result<Self, QualityContractError> {
        let contract: QualityContract = serde_json::from_str(input)?;
        if contract.schema_version != SCHEMA_VERSION {
            return Err(QualityContractError::UnsupportedSchemaVersion {
                actual: contract.schema_version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(contract)
    }
}

/// Machine/toolchain provenance of the measured snapshot.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Git commit the snapshot was measured on (full or abbreviated).
    pub git_commit: String,
    /// Whether the tree was dirty at measurement time.
    pub git_dirty: bool,
    /// Dashboard run id of the measurement.
    pub run_id: String,
    /// Canonical ISA string, e.g. `x86-64-v3 (AVX2/FMA/F16C/BMI)`.
    pub effective_isa: String,
    /// CPU model name as reported by the host.
    pub cpu_model: String,
    /// `rustc --version` output of the toolchain used.
    pub rustc: String,
    /// Cargo profile used for the measurement (e.g. `release`).
    pub cargo_profile: String,
}

/// Numeric acceptance envelopes applied by the verify engine.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelopes {
    /// ESR envelope against the NAMCore (f32 SIMD) reference.
    pub esr_namcore: EsrNamcoreEnvelope,
    /// Maximum allowed SNR drop in dB (new < contract − 6.0 fails).
    pub snr_db_drop: f64,
    /// MR-STFT multiplier (new > contract × 10.0 fails).
    pub mrstft_mult: f64,
    /// Latency multiplier (new > max(contract × 1.10, contract + 0.05 µs) fails).
    pub latency_mult: f64,
    /// Absolute latency floor in microseconds.
    pub latency_floor_us: f64,
}

impl Envelopes {
    /// Canonical v1 policy — the measured envelope numbers of `verify_contract`
    /// (`utils/quality-dashboard.sh:2241-2250`), **not** the stale 10×/100×
    /// header comments (`:2036-2041`) that never were the gate.
    ///
    /// Measured policy: PERF-009
    ///
    /// Applied formulas (identical for the NAMCore and the f64 oracle):
    /// - ESR noise: `max(baseline × noise_mult, baseline + noise_floor_abs)`
    /// - ESR safety: `max(baseline × safety_mult, safety_floor_abs)`
    /// - SNR: fails when `new < baseline − snr_db_drop` dB
    /// - MR-STFT: fails when `new > baseline × mrstft_mult`
    /// - Latency: fails when `new > max(baseline × latency_mult, baseline + latency_floor_us)` µs
    pub fn policy_v1() -> Self {
        Self {
            esr_namcore: EsrNamcoreEnvelope {
                noise_mult: 3.0,
                noise_floor_abs: 5e-14,
                safety_mult: 10.0,
                safety_floor_abs: 1e-12,
            },
            snr_db_drop: 6.0,
            mrstft_mult: 10.0,
            latency_mult: 1.10,
            latency_floor_us: 0.05,
        }
    }
}

/// ESR envelope against the NAMCore reference (measured policy, PERF-009).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EsrNamcoreEnvelope {
    /// Baseline multiplier over the contract value.
    pub noise_mult: f64,
    /// Absolute floor added to the baseline envelope.
    pub noise_floor_abs: f64,
    /// Safety multiplier over the baseline envelope.
    pub safety_mult: f64,
    /// Absolute floor of the safety envelope.
    pub safety_floor_abs: f64,
}

/// One fidelity entry (golden vector or additional coverage).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FidelityEntry {
    /// Stable machine-readable id, e.g. `bosswn-standard@48000:live`.
    ///
    /// The verify engine matches by `id`, never by label prefix.
    pub id: String,
    /// Human-readable label (international English).
    pub label: String,
    /// ESR against the NAMCore reference.
    pub esr_namcore: f64,
    /// ESR against the ideal f64 oracle (`null` when not measured).
    pub esr_f64: Option<f64>,
    /// SNR in dB (`null` when not measured).
    pub snr_db: Option<f64>,
    /// MR-STFT metric value.
    pub mrstft: f64,
    /// Advisory entry: absence does not fail the gate.
    #[serde(default)]
    pub optional: bool,
}

/// One real-time latency entry (Model Inference Core or DSP Infrastructure).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerformanceEntry {
    /// Stable machine-readable id, e.g. `RT_WaveNet_Std_CH16`.
    pub id: String,
    /// Human-readable label (international English).
    pub label: String,
    /// Median block latency in microseconds.
    pub median_latency_us: f64,
}

/// Typed error for parsing/persisting a quality contract.
#[derive(Debug, thiserror::Error)]
pub enum QualityContractError {
    /// The payload is not valid JSON or violates the schema.
    #[error("invalid quality contract JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The payload declares a schema version this crate cannot interpret.
    #[error("unsupported quality contract schema version {actual}, expected {expected}")]
    UnsupportedSchemaVersion {
        /// Version found in the payload.
        actual: u32,
        /// Version this crate understands.
        expected: u32,
    },
}

/// JSONL fidelity metrics ingest (F-27) — `FidelityRecord` + canonical parser.
pub mod metrics;

/// Performance-baseline persistence (S3.T3) — replace-copy of top-level
/// Criterion series with nested sanitize (scenario 4 semantics).
pub mod baseline_store;

/// Single performance-status classifier (F-08) — `PASS` / `NOT_VERIFIED` /
/// `FAIL` semantics shared by the dashboard and the perf-gate.
pub mod classify;

/// Baseline coverage cross-check (F-24, S3.T2) — `executed_bench_ids` /
/// `missing_baseline_coverage` ported from the perf-gate bash.
pub mod coverage;

/// Single environment probe (R-09) — canonical ISA string + cpuinfo /
/// toolchain / governor / git state for receipts and fingerprints.
pub mod env;

/// Performance-baseline environment fingerprint (S3.T1) — serde JSON
/// persisted under `.performance-baselines/` plus the field-by-field
/// `MISSING_BASELINE` / `INCOMPARABLE_ENVIRONMENT` comparison.
pub mod fingerprint;

/// Label↔canonical-entity alias tables (old-report labels, catalog
/// label↔fixture projection, explicit `RT_*` bench table).
pub mod ids;

/// Contract verify engine — literal port of the bash `verify_contract`.
pub mod verify;

/// Human-facing dashboard renderer (S6.T1) — pure `QualityReport → String`
/// (ANSI/plain), consumed by `nam_quality render` and the dashboard wrapper.
pub mod render;

#[cfg(test)]
#[path = "qa_test.rs"]
mod qa_test;

#[cfg(test)]
#[path = "transcription.rs"]
mod transcription;
