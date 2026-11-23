// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Typed quality report model and error definitions.

use super::super::metrics::FidelityRecord;
use super::super::metrics::MetricValue;
use super::super::verify::{LatencyRecord, PhaseRecord};

/// Output style of the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    /// ANSI color escape sequences (terminal).
    Ansi,
    /// Plain text, no escape sequences (visual dump / golden snapshots).
    Plain,
}

/// Machine/toolchain header of the report (parsed from `build_metadata`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReportHeader {
    /// Full git commit hash (or `unknown`).
    pub git_commit: String,
    /// Whether the tree was dirty at measurement time.
    pub git_dirty: bool,
    /// Dashboard run id.
    pub run_id: String,
    /// Canonical ISA string of the engine dispatch active during measurement (R-09).
    pub effective_isa: String,
    /// Host CPU ISA probed from `/proc/cpuinfo` (present when it differs from or qualifies `effective_isa`).
    pub host_isa: Option<String>,
    /// Active engine SIMD backend label (`SIMD_MATH.name`), if recorded.
    pub engine_backend: Option<String>,
    /// `rustc --version` output.
    pub rustc: String,
    /// Cargo build profile (e.g. `release`).
    pub cargo_profile: String,
    /// Host target triple.
    pub target_triple: String,
    /// Optional ISO-8601 measurement timestamp (absent in the current sink).
    pub measured_at: Option<String>,
    /// Optional CPU model name (absent in the current `build_metadata`).
    pub cpu_model: Option<String>,
}

impl Default for ReportHeader {
    fn default() -> Self {
        Self {
            git_commit: "unknown".into(),
            git_dirty: false,
            run_id: String::new(),
            effective_isa: "unknown".into(),
            host_isa: None,
            engine_backend: None,
            rustc: "unknown".into(),
            cargo_profile: "release".into(),
            target_triple: "unknown".into(),
            measured_at: None,
            cpu_model: None,
        }
    }
}

/// One `f64_table` record (f64-oracle summary row).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F64TableRow {
    /// Oracle fixture filename (e.g. `BossWN-standard.nam`).
    pub filename: String,
    /// Model family label.
    pub family: String,
    /// Linear ESR against the f64 oracle.
    pub esr: MetricValue,
    /// ESR in dB.
    pub esr_db: MetricValue,
}

/// One `f64_decomp` record (error-source decomposition of one model).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F64Decomp {
    /// Model label.
    pub label: String,
    /// Architecture family (e.g. `LSTM`, `WaveNet`).
    pub architecture: String,
    /// Total ESR (f32 vs f64 oracle).
    pub esr_f32_vs_f64: MetricValue,
    /// F16C quantization term (omitted when unmeasured).
    pub esr_quant_f16c: Option<MetricValue>,
    /// BF16 quantization term (omitted when unmeasured).
    pub esr_quant_bf16: Option<MetricValue>,
    /// Activation-approximation term (omitted when unmeasured).
    pub esr_activation: Option<MetricValue>,
    /// Accumulation term (omitted when unmeasured).
    pub esr_accumulation: Option<MetricValue>,
    /// Combined term (F16C + Padé + F32).
    pub esr_combined: Option<MetricValue>,
}

/// One `activation` record (Fast Padé vs exact-grade SNR).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationRow {
    /// Model label.
    pub model: String,
    /// Reference oracle or test regime (`namcore`, `f64`, etc.).
    pub oracle: Option<String>,
    /// SNR of the Fast (Padé) path in dB.
    pub snr_fast_db: MetricValue,
    /// SNR of the Standard (exact) path in dB.
    pub snr_exact_db: MetricValue,
    /// Gain (`exact − fast`).
    pub gain_db: MetricValue,
}

/// One `isa` record (cross-ISA or self-consistency check).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsaRow {
    /// Model label.
    pub label: String,
    /// Reference ISA.
    pub ref_isa: String,
    /// ISA under test.
    pub test_isa: String,
    /// Mean squared error (always present).
    pub mse: MetricValue,
    /// Linear ESR (cross-ISA only).
    pub esr: Option<MetricValue>,
    /// Maximum absolute error (cross-ISA only).
    pub max_abs_err: Option<MetricValue>,
    /// Error budget (cross-ISA only).
    pub budget: Option<MetricValue>,
}

/// `coverage_matrix` record — per-axis coverage counts (governance).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoverageMatrix {
    /// NAMCore parity records.
    pub namcore_parity: u64,
    /// f64-oracle paired records.
    pub f64_oracle: u64,
    /// ISA-optimization records.
    pub isa_optimizations: u64,
    /// Spectral-baseline records.
    pub spectral_baselines: u64,
    /// Real-time performance records.
    pub rt_performance: u64,
}

/// `test_counts` record — phase-status tallies (governance, F-25 note).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TestCounts {
    /// Phases with PASS status.
    pub passed: u64,
    /// Phases with FAIL status.
    pub failed: u64,
    /// Ignored-test tallies (unused by the receipt, kept for schema parity).
    pub ignored: u64,
    /// Filtered-test tallies (unused by the receipt, kept for schema parity).
    pub filtered: u64,
    /// Phases with SKIP_CAPABILITY / SKIP_OPTIONAL_FIXTURE status.
    pub skip_capability: u64,
}

/// Typed superset of the dashboard report (all ingest kinds).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    /// Provenance header (`build_metadata`).
    pub header: ReportHeader,
    /// Phase outcome records.
    pub phases: Vec<PhaseRecord>,
    /// Fidelity metric records.
    pub fidelity: Vec<FidelityRecord>,
    /// Latency records.
    pub latency: Vec<LatencyRecord>,
    /// f64-oracle summary rows.
    pub f64_table: Vec<F64TableRow>,
    /// f64-oracle decomposition blocks.
    pub f64_decomp: Vec<F64Decomp>,
    /// Activation-precision rows.
    pub activation: Vec<ActivationRow>,
    /// ISA-parity rows.
    pub isa: Vec<IsaRow>,
    /// Coverage matrix (last record wins).
    pub coverage: Option<CoverageMatrix>,
    /// Test counts (last record wins).
    pub test_counts: Option<TestCounts>,
}

impl QualityReport {
    /// Status of one phase (last matching record; absent ⇒ `NOT_RUN`).
    pub fn phase_status(&self, phase_id: &str) -> &str {
        self.phases
            .iter()
            .rev()
            .find(|p| p.phase_id == phase_id)
            .map(|p| p.status.as_str())
            .unwrap_or("NOT_RUN")
    }

    /// Whether the performance domain is `NOT_VERIFIED` (never green).
    pub fn performance_not_verified(&self) -> bool {
        self.phase_status("regression_gate") != "PASS"
    }
}

/// Typed error of the report ingest.
#[derive(Debug, thiserror::Error)]
pub enum ReportError {
    /// A report line is not valid JSON.
    #[error("malformed report line {line}: {source}")]
    MalformedLine {
        /// 1-based line number in the report.
        line: usize,
        /// The underlying JSON parse error.
        source: serde_json::Error,
    },
    /// A coverage/test-counts record carries non-integer fields.
    #[error("record on line {line} must carry integer fields")]
    InvalidCounts {
        /// 1-based line number in the report.
        line: usize,
    },
}
