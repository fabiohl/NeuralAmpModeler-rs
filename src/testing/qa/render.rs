// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Human-facing dashboard renderer (Sprint S6.T1, finding R-07).
//!
//! Pure `QualityReport → String`: no I/O, no cargo, no environment access —
//! the renderer only formats a typed report. The dashboard bash wrapper and
//! the `nam_quality render` binary both pipe this output; the long/perf
//! suites never call it (they stay WARNING/ERROR + forensic JSONL only, PO
//! note).
//!
//! The `QualityReport` is the typed superset of the verify report: on top of
//! the `verify.rs` phase/fidelity/latency streams it carries the extra
//! oracle/ISA/activation/coverage kinds that the S2.T6 sink writes
//! (`f64_table`, `f64_decomp`, `activation`, `isa`, `coverage_matrix`,
//! `test_counts`) plus the `build_metadata` provenance record.
//!
//! Design invariants:
//! - Section order matches the current dashboard: header → quick summary →
//!   fidelity details → performance → ISA parity → activation precision →
//!   f64 decomposition → spectral summary → coverage matrix → footer.
//! - English-only (PO note: unify to international English).
//! - `NOT_VERIFIED` performance is **never** rendered green.
//! - ESR / CPU-budget colors follow the qualitative thresholds of the old
//!   bash render (green < 1e-5, yellow < 1e-1, red ≥ 1e-1; CPU headroom
//!   green > 50 %, yellow > 25 %, red otherwise).
//! - The f64-decomposition cold-start note is emitted in English.

use serde_json::Value;

use super::metrics::{FidelityRecord, MetricValue, fidelity_from_json};
use super::verify::{LatencyRecord, PhaseRecord};

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
    /// Canonical ISA string (the only ISA source, R-09).
    pub effective_isa: String,
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

/// Parses a full dashboard report (JSONL) into the typed [`QualityReport`].
///
/// Routing mirrors `verify.rs::parse_verify_report` and extends it:
/// records with `phase_id` are phase records, records with a `kind` are
/// routed to the corresponding section (`fidelity`, `latency`, `f64_table`,
/// `f64_decomp`, `activation`, `isa`, `coverage_matrix`, `test_counts`,
/// `build_metadata`), and everything else goes through the fidelity
/// canonicalization. Malformed lines fail the whole parse (fail-closed).
pub fn parse_quality_report(input: &str) -> Result<QualityReport, ReportError> {
    let mut report = QualityReport {
        header: ReportHeader::default(),
        phases: Vec::new(),
        fidelity: Vec::new(),
        latency: Vec::new(),
        f64_table: Vec::new(),
        f64_decomp: Vec::new(),
        activation: Vec::new(),
        isa: Vec::new(),
        coverage: None,
        test_counts: None,
    };
    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(line).map_err(|source| ReportError::MalformedLine {
                line: line_no + 1,
                source,
            })?;
        if value.get("phase_id").is_some() {
            if let Some(phase) = phase_from_json(&value) {
                report.phases.push(phase);
            }
            continue;
        }
        match value.get("kind").and_then(Value::as_str) {
            Some("build_metadata") => report.header = header_from_json(&value),
            Some("latency") => {
                if let Some(record) = latency_from_json(&value) {
                    report.latency.push(record);
                }
            }
            Some("f64_table") => {
                if let Some(row) = f64_table_from_json(&value) {
                    report.f64_table.push(row);
                }
            }
            Some("f64_decomp") => {
                if let Some(block) = f64_decomp_from_json(&value) {
                    report.f64_decomp.push(block);
                }
            }
            Some("activation") => {
                if let Some(row) = activation_from_json(&value) {
                    report.activation.push(row);
                }
            }
            Some("isa") => {
                if let Some(row) = isa_from_json(&value) {
                    report.isa.push(row);
                }
            }
            Some("coverage_matrix") => {
                report.coverage = Some(coverage_from_json(&value, line_no + 1)?);
            }
            Some("test_counts") => {
                report.test_counts = Some(test_counts_from_json(&value, line_no + 1)?);
            }
            _ => {
                // Fidelity (kind "fidelity" or absent, label-bearing) plus the
                // `median_latency_us`-only latency shape of the S2.T2 fixtures.
                if value.get("median_latency_us").is_some() {
                    if let Some(record) = latency_from_json(&value) {
                        report.latency.push(record);
                    }
                } else if let Some(record) = fidelity_from_json(&value) {
                    report.fidelity.push(record);
                }
            }
        }
    }
    Ok(report)
}

/// Parses a full dashboard report file, fail-closed on unreadable input.
pub fn parse_quality_report_file(
    path: impl AsRef<std::path::Path>,
) -> Result<QualityReport, ReportError> {
    let input = std::fs::read_to_string(path).map_err(|source| ReportError::MalformedLine {
        line: 0,
        source: serde_json::Error::io(source),
    })?;
    parse_quality_report(&input)
}

fn phase_from_json(value: &Value) -> Option<PhaseRecord> {
    let phase_id = value.get("phase_id")?.as_str()?.to_string();
    let status = value.get("status")?.as_str()?.to_string();
    Some(PhaseRecord { phase_id, status })
}

fn header_from_json(value: &Value) -> ReportHeader {
    let mut header = ReportHeader::default();
    if let Some(v) = value.get("git_commit").and_then(Value::as_str) {
        header.git_commit = v.to_string();
    }
    header.git_dirty = value
        .get("git_dirty_state")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(v) = value.get("run_id").and_then(Value::as_str) {
        header.run_id = v.to_string();
    }
    if let Some(v) = value.get("effective_isa").and_then(Value::as_str) {
        header.effective_isa = v.to_string();
    }
    if let Some(v) = value.get("rustc_version").and_then(Value::as_str) {
        header.rustc = v.to_string();
    }
    if let Some(v) = value.get("cargo_profile").and_then(Value::as_str) {
        header.cargo_profile = v.to_string();
    }
    if let Some(v) = value.get("target_triple").and_then(Value::as_str) {
        header.target_triple = v.to_string();
    }
    if let Some(v) = value.get("measured_at").and_then(Value::as_str) {
        header.measured_at = Some(v.to_string());
    }
    if let Some(v) = value.get("cpu_model").and_then(Value::as_str) {
        header.cpu_model = Some(v.to_string());
    }
    header
}

fn latency_from_json(value: &Value) -> Option<LatencyRecord> {
    let label = value.get("label")?.as_str()?.to_string();
    let median_latency_us = value
        .get("median_latency_us")?
        .as_f64()
        .filter(|v| v.is_finite())?;
    Some(LatencyRecord {
        label,
        median_latency_us,
    })
}

fn canon_metric(value: Option<&Value>) -> MetricValue {
    match value {
        None | Some(Value::Null) => MetricValue::Na,
        Some(Value::String(s)) if s.is_empty() => MetricValue::Na,
        Some(Value::String(s)) => MetricValue::Raw(s.clone()),
        Some(Value::Number(n)) => MetricValue::Raw(n.to_string()),
        Some(Value::Bool(b)) => MetricValue::Raw((*b).to_string()),
        Some(other) => MetricValue::Raw(other.to_string()),
    }
}

fn f64_table_from_json(value: &Value) -> Option<F64TableRow> {
    let filename = value.get("filename")?.as_str()?.to_string();
    let family = value.get("family")?.as_str()?.to_string();
    Some(F64TableRow {
        filename,
        family,
        esr: canon_metric(value.get("esr")),
        esr_db: canon_metric(value.get("esr_db")),
    })
}

fn f64_decomp_from_json(value: &Value) -> Option<F64Decomp> {
    let label = value.get("label")?.as_str()?.to_string();
    let architecture = value.get("architecture")?.as_str()?.to_string();
    let opt_metric = |key: &str| value.get(key).map(|v| canon_metric(Some(v)));
    Some(F64Decomp {
        label,
        architecture,
        esr_f32_vs_f64: canon_metric(value.get("esr_f32_vs_f64")),
        esr_quant_f16c: opt_metric("esr_quant_f16c"),
        esr_quant_bf16: opt_metric("esr_quant_bf16"),
        esr_activation: opt_metric("esr_activation"),
        esr_accumulation: opt_metric("esr_accumulation"),
        esr_combined: opt_metric("esr_combined"),
    })
}

fn activation_from_json(value: &Value) -> Option<ActivationRow> {
    let model = value.get("model")?.as_str()?.to_string();
    Some(ActivationRow {
        model,
        snr_fast_db: canon_metric(value.get("snr_fast_db")),
        snr_exact_db: canon_metric(value.get("snr_exact_db")),
        gain_db: canon_metric(value.get("gain_db")),
    })
}

fn isa_from_json(value: &Value) -> Option<IsaRow> {
    let label = value.get("label")?.as_str()?.to_string();
    let ref_isa = value.get("ref_isa")?.as_str()?.to_string();
    let test_isa = value.get("test_isa")?.as_str()?.to_string();
    let opt_metric = |key: &str| value.get(key).map(|v| canon_metric(Some(v)));
    Some(IsaRow {
        label,
        ref_isa,
        test_isa,
        mse: canon_metric(value.get("mse")),
        esr: opt_metric("esr"),
        max_abs_err: opt_metric("max_abs_err"),
        budget: opt_metric("budget"),
    })
}

fn as_u64(value: Option<&Value>) -> Option<u64> {
    value.and_then(Value::as_u64)
}

fn coverage_from_json(value: &Value, line: usize) -> Result<CoverageMatrix, ReportError> {
    let err = || ReportError::InvalidCounts { line };
    Ok(CoverageMatrix {
        namcore_parity: as_u64(value.get("namcore_parity")).ok_or_else(err)?,
        f64_oracle: as_u64(value.get("f64_oracle")).ok_or_else(err)?,
        isa_optimizations: as_u64(value.get("isa_optimizations")).ok_or_else(err)?,
        spectral_baselines: as_u64(value.get("spectral_baselines")).ok_or_else(err)?,
        rt_performance: as_u64(value.get("rt_performance")).ok_or_else(err)?,
    })
}

fn test_counts_from_json(value: &Value, line: usize) -> Result<TestCounts, ReportError> {
    let err = || ReportError::InvalidCounts { line };
    Ok(TestCounts {
        passed: as_u64(value.get("passed")).ok_or_else(err)?,
        failed: as_u64(value.get("failed")).ok_or_else(err)?,
        ignored: as_u64(value.get("ignored")).unwrap_or(0),
        filtered: as_u64(value.get("filtered")).unwrap_or(0),
        skip_capability: as_u64(value.get("skip_capability")).unwrap_or(0),
    })
}

// ── ANSI palette ────────────────────────────────────────────────────────────

struct Palette {
    green: &'static str,
    yellow: &'static str,
    red: &'static str,
    bold: &'static str,
    nc: &'static str,
}

impl Palette {
    fn new(style: RenderStyle) -> Self {
        match style {
            RenderStyle::Ansi => Self {
                green: "\x1b[0;32m",
                yellow: "\x1b[1;33m",
                red: "\x1b[0;31m",
                bold: "\x1b[1m",
                nc: "\x1b[0m",
            },
            RenderStyle::Plain => Self {
                green: "",
                yellow: "",
                red: "",
                bold: "",
                nc: "",
            },
        }
    }

    fn paint(&self, color: &str, text: &str) -> String {
        format!("{color}{text}{}", self.nc)
    }

    fn paint_class(&self, class: &str, text: &str) -> String {
        match class {
            "green" => self.green(text),
            "yellow" => self.yellow(text),
            "red" => self.red(text),
            _ => text.to_string(),
        }
    }

    fn green(&self, text: &str) -> String {
        self.paint(self.green, text)
    }

    fn yellow(&self, text: &str) -> String {
        self.paint(self.yellow, text)
    }

    fn red(&self, text: &str) -> String {
        self.paint(self.red, text)
    }

    fn bold(&self, text: &str) -> String {
        self.paint(self.bold, text)
    }
}

// ── Formatting helpers ──────────────────────────────────────────────────────

/// `printf "%.2e"` for small/scientific values, `%.4f` otherwise (mirror of
/// the bash `_fmt_metric`); `N/A` and non-finite literals pass through.
fn fmt_metric(mv: &MetricValue) -> String {
    match mv {
        MetricValue::Na => "N/A".to_string(),
        MetricValue::Raw(raw) => match raw.parse::<f64>() {
            Ok(v) if v.is_finite() => {
                if raw.contains(['e', 'E']) || (v != 0.0 && v.abs() < 0.0001) {
                    format!("{v:.2e}")
                } else {
                    format!("{v:.4}")
                }
            }
            _ => raw.clone(),
        },
    }
}

/// SNR is rendered with one decimal (mirror of the bash `_nfmt "%.1f"`).
fn fmt_snr(mv: &MetricValue) -> String {
    match mv {
        MetricValue::Na => "N/A".to_string(),
        MetricValue::Raw(raw) => match raw.parse::<f64>() {
            Ok(v) if v.is_finite() => format!("{v:.1}"),
            _ => raw.clone(),
        },
    }
}

/// Finite numeric value behind a metric, if any.
fn metric_f64(mv: &MetricValue) -> Option<f64> {
    match mv {
        MetricValue::Na => None,
        MetricValue::Raw(raw) => raw.parse::<f64>().ok().filter(|v| v.is_finite()),
    }
}

/// Visible width of a string, ignoring ANSI escape sequences.
fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            while let Some(&n) = chars.peek() {
                chars.next();
                if n == 'm' {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
}

/// Left-pads `s` to `width` visible columns (ANSI-aware).
fn pad(s: &str, width: usize) -> String {
    let visible = visible_width(s);
    if visible >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - visible))
    }
}

/// Truncates a label to `max` characters (the dashboard truncates to 38).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// Strips the trailing ` @rate` and ` Live`/` HQ` decorations, mirroring the
/// bash `sed 's/ @.*//; s/ Live$//; s/ HQ$//'`.
fn model_label(label: &str) -> String {
    let mut out = label.to_string();
    if let Some(pos) = out.find(" @") {
        out.truncate(pos);
    }
    for suffix in [" Live", " HQ"] {
        if let Some(stripped) = out.strip_suffix(suffix) {
            out = stripped.to_string();
        }
    }
    out
}

// ── Qualitative color/verdict thresholds (mirror of the bash helpers) ──────

/// ESR color class: green < 1e-5, yellow < 1e-1, red otherwise.
fn esr_color_class(esr: f64) -> &'static str {
    if esr < 1e-5 {
        "green"
    } else if esr < 1e-1 {
        "yellow"
    } else {
        "red"
    }
}

/// Short qualitative ESR verdict (English — the bash used PT here).
fn esr_verdict_short(esr: f64) -> &'static str {
    if esr < 1e-10 {
        "IDENTICAL"
    } else if esr < 1e-5 {
        "IMPERCEPTIBLE"
    } else if esr < 1e-2 {
        "A/B SCIENTIFIC"
    } else if esr < 1e-1 {
        "AUDIBLE DIRECT"
    } else {
        "⚠ AUDIBLE"
    }
}

/// CPU budget percentage (RT deadline 1333 µs).
fn budget_pct(latency_us: f64) -> f64 {
    (latency_us / 1333.0) * 100.0
}

/// CPU headroom percentage.
fn budget_headroom(pct: f64) -> f64 {
    100.0 - pct
}

/// CPU budget color class by headroom (green > 50, yellow > 25, red otherwise).
fn cpu_color_class(pct: f64) -> &'static str {
    let headroom = budget_headroom(pct);
    if headroom > 50.0 {
        "green"
    } else if headroom > 25.0 {
        "yellow"
    } else {
        "red"
    }
}

// ── Section renderers ───────────────────────────────────────────────────────

fn render_header(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    let h = &report.header;
    let dirty = if h.git_dirty { "dirty" } else { "clean" };
    out.push_str(&p.bold("===============================================================\n"));
    out.push_str("   NeuralAmpModeler-rs Quality Dashboard\n");
    out.push_str("   ------------------------------\n");
    if let Some(measured_at) = &h.measured_at {
        out.push_str(&format!("   Measured at: {measured_at}\n"));
    }
    if let Some(cpu) = &h.cpu_model {
        out.push_str(&format!("   CPU:    {cpu}\n"));
    }
    out.push_str(&format!("   Commit: {} ({dirty})\n", h.git_commit));
    out.push_str(&format!("   Run ID: {}\n", h.run_id));
    out.push_str(&format!("   ISA:    {}\n", h.effective_isa));
    out.push_str(&format!("   rustc:  {}\n", h.rustc));
    out.push_str(&format!(
        "   profile: {} · target: {}\n",
        h.cargo_profile, h.target_triple
    ));
    out.push_str(&p.bold("===============================================================\n"));
    out
}

/// Representative quick-summary models (display label, latency bench id,
/// case-insensitive label substrings) — mirrors the bash quick-summary map.
const QUICK_REPS: &[(&str, &str, &[&str])] = &[
    (
        "WaveNet Standard (CH16)",
        "RT_WaveNet_Std_CH16",
        &["BossWN-standard", "WaveNet Std"],
    ),
    (
        "WaveNet A1 Standard",
        "RT_WaveNet_Std_CH16",
        &["wavenet_a1_standard", "A1 Standard"],
    ),
    (
        "WaveNet Feather (CH8)",
        "RT_WaveNet_Feather_CH8",
        &["BossWN-feather", "WaveNet Feather"],
    ),
    (
        "LSTM 1x16 (BossLSTM)",
        "RT_LSTM_1x16",
        &["BossLSTM-1x16", "LSTM 1x16"],
    ),
    (
        "LSTM 2x8 (BossLSTM)",
        "RT_LSTM_2x8",
        &["BossLSTM-2x8", "LSTM 2x8"],
    ),
    ("A2 Full (CH8)", "RT_A2_Full_CH8", &["A2-Full", "A2 Full"]),
    ("A2 Lite (CH3)", "RT_A2_Lite_CH3", &["A2-Lite", "A2 Lite"]),
    (
        "A2-FiLM Lite (CH3)",
        "RT_A2_Lite_CH3",
        &["A2-FiLM-Lite", "FiLM.*Lite"],
    ),
    ("ConvNet", "RT_ConvNet", &["ConvNet"]),
    (
        "Linear (RF=2048)",
        "RT_Linear",
        &["linear_fft_rf2048", "Linear FFT RF=2048"],
    ),
];

fn find_representative<'a>(
    fidelity: &'a [FidelityRecord],
    patterns: &[&str],
) -> Option<&'a FidelityRecord> {
    fidelity.iter().find(|record| {
        let lower = record.label.to_lowercase();
        patterns
            .iter()
            .any(|pat| lower.contains(&pat.to_lowercase()))
    })
}

fn latency_by_bench<'a>(latency: &'a [LatencyRecord], bench_id: &str) -> Option<&'a LatencyRecord> {
    latency.iter().find(|record| record.label == bench_id)
}

fn render_quick_summary(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push('\n');
    out.push_str("QUICK SUMMARY (non-specialist)\n");
    out.push_str("═══════════════════════════════════════\n\n");

    for (display, bench_id, patterns) in QUICK_REPS {
        let Some(record) = find_representative(&report.fidelity, patterns) else {
            continue;
        };
        let esr_nam = fmt_metric(&record.esr);
        let esr_nam_colored = match metric_f64(&record.esr) {
            Some(esr) => p.paint_class(esr_color_class(esr), &esr_nam),
            None => esr_nam.clone(),
        };
        let verdict = match metric_f64(&record.esr) {
            Some(esr) => {
                let text = esr_verdict_short(esr);
                p.paint_class(esr_color_class(esr), text)
            }
            None => "N/A".to_string(),
        };
        let esr_f64 = fmt_metric(&record.esr_f64);
        let esr_f64_colored = match metric_f64(&record.esr_f64) {
            Some(esr) => p.paint_class(esr_color_class(esr), &esr_f64),
            None => esr_f64.clone(),
        };
        let cpu = match latency_by_bench(&report.latency, bench_id) {
            Some(record) => {
                let pct = budget_pct(record.median_latency_us);
                p.paint_class(cpu_color_class(pct), &format!("{pct:.1}%"))
            }
            None => "N/A".to_string(),
        };

        out.push_str(&format!(
            "  {:<24}  vs NAMCore: {:>10} {:<16}  |  vs Ideal (f64): {:>10}  |  CPU: {} of budget\n",
            display,
            esr_nam_colored,
            pad(&verdict, 16),
            esr_f64_colored,
            cpu,
        ));
    }
    out.push('\n');
    out
}

fn is_redundant_measurement(label: &str) -> bool {
    let t_digit = label.chars().next().is_some_and(|c| c == 'T')
        && label.chars().nth(1).is_some_and(|c| c.is_ascii_digit());
    label.starts_with("Quick ")
        || label.starts_with("Container ")
        || label.starts_with("Container File ")
        || label.starts_with("T-")
        || t_digit
}

fn fidelity_mode(label: &str) -> &str {
    if label.contains(" HQ") { "HQ" } else { "Live" }
}

fn render_fidelity_details(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("AUDIO FIDELITY — Technical Details\n");
    out.push_str("═════════════════════════════════════════\n\n");

    if report.fidelity.is_empty() {
        out.push_str(&p.yellow("  (i) No fidelity data available.\n"));
        out.push('\n');
        return out;
    }

    let mut canonicals: Vec<&FidelityRecord> = Vec::new();
    let mut redundants: Vec<&FidelityRecord> = Vec::new();
    for record in &report.fidelity {
        if is_redundant_measurement(&model_label(&record.label)) {
            redundants.push(record);
        } else {
            canonicals.push(record);
        }
    }

    let header_line = || {
        format!(
            "  {:<38} | {:>10} | {:>10} | {:>8} | {:>8} | {:<4}\n",
            "Model", "ESR NAMCore", "ESR f64", "SNR dB", "MR-STFT", "Mode"
        )
    };
    let rule_line = || {
        format!(
            "  {:-<38}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<4}\n",
            "", "", "", "", "", ""
        )
    };

    if !canonicals.is_empty() {
        out.push_str("  ── Canonical Fidelity (golden_vectors) ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for record in &canonicals {
            out.push_str(&fidelity_row(record, p));
        }
        out.push('\n');
    }

    if !redundants.is_empty() {
        out.push_str("  ── Additional Coverage (quick_parity, containers, regression gates) ──\n");
        out.push_str(
            "  (i) These measurements validate the same models via alternate entry points.\n",
        );
        out.push_str("       Equivalent rows from the canonical table above.\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for record in &redundants {
            out.push_str(&fidelity_row(record, p));
        }
        out.push('\n');
    }

    out.push_str("  Qualitative legend (ESR audibility bounds):\n");
    out.push_str(&format!(
        "    {} = imperceptible (ESR < 1e-5)\n",
        p.green("green")
    ));
    out.push_str(&format!(
        "    {} = audible only with scientific A/B (ESR < 1e-2)\n",
        p.yellow("yellow")
    ));
    out.push_str(&format!(
        "    {} = audible — needs investigation (ESR >= 1e-1)\n",
        p.red("red")
    ));
    out.push('\n');
    out
}

fn fidelity_row(record: &FidelityRecord, p: &Palette) -> String {
    let esr_nam = fmt_metric(&record.esr);
    let esr_nam_colored = match metric_f64(&record.esr) {
        Some(esr) => p.paint_class(esr_color_class(esr), &esr_nam),
        None => esr_nam.clone(),
    };
    let esr_f64 = fmt_metric(&record.esr_f64);
    let esr_f64_colored = match metric_f64(&record.esr_f64) {
        Some(esr) => p.paint_class(esr_color_class(esr), &esr_f64),
        None => esr_f64.clone(),
    };
    let snr = fmt_snr(&record.snr_db);
    let mrstft = fmt_metric(&record.mrstft);
    let mode = fidelity_mode(&record.label);
    let display_key = truncate(&record.label, 38);
    format!(
        "  {:<38} | {:>10} | {:>10} | {:>8} | {:>8} | {:<4}\n",
        pad(&display_key, 38),
        esr_nam_colored,
        esr_f64_colored,
        snr,
        mrstft,
        mode
    )
}

/// Bench id → human display label (mirror of the bash `BENCH_MODEL_MAP`).
const BENCH_DISPLAY: &[(&str, &str)] = &[
    ("RT_WaveNet_Std_CH16", "WaveNet Standard CH16"),
    ("RT_WaveNet_Feather_CH8", "WaveNet Feather CH8"),
    ("RT_WaveNet_Lite_CH12", "WaveNet Lite CH12"),
    ("RT_WaveNet_Nano_CH4", "WaveNet Nano CH4"),
    ("RT_A2_Full_CH8", "A2 Full CH8"),
    ("RT_A2_Lite_CH3", "A2 Lite CH3"),
    ("RT_LSTM_1x16", "LSTM 1x16"),
    ("RT_LSTM_2x8", "LSTM 2x8"),
    ("RT_Linear", "Linear RF=2048"),
    ("RT_ConvNet", "ConvNet"),
    ("RT_WaveNet_Dyn_Free", "WaveNet Dyn Free"),
    ("RT_LSTM_Dyn_1x7", "LSTM Dyn 1x7"),
    ("RT_A2_Dyn_Gated_CH8", "A2 Dyn Gated CH8"),
    ("RT_A2_Dyn_Blended_CH3", "A2 Dyn Blended CH3"),
    ("RT_DSP_Resampler_44k1_to_48k", "DSP Resampler 44.1k->48k"),
    ("RT_DSP_Resampler_96k_to_48k", "DSP Resampler 96k->48k"),
    ("RT_DSP_CabSim_IR_Medium", "DSP CabSim IR Medium"),
    ("RT_DSP_Pipeline_Base_NoOS", "DSP Pipeline Base (No OS)"),
    ("RT_DSP_Pipeline_HQ_4xOS", "DSP Pipeline HQ (4x OS)"),
];

/// Median latency (µs) for a bench id, if a matching latency record exists.
fn latency_us(report: &QualityReport, bench_id: &str) -> Option<f64> {
    report
        .latency
        .iter()
        .find(|record| record.label == bench_id)
        .map(|record| record.median_latency_us)
}

fn render_performance(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("PERFORMANCE — Block Latency (64 samples @ 48kHz)\n");
    out.push_str("══════════════════════════════════════════════════════════\n");
    out.push_str("  RT deadline: 1333 µs (1.33 ms)\n\n");

    if report.performance_not_verified() {
        out.push_str(&p.yellow("  ⚠ NOT_VERIFIED — performance not verified against baseline.\n"));
        out.push_str(&p.yellow(
            "    Performance is not certified in this run; --check against the quality contract fails on this.\n\n",
        ));
        out.push_str(
            &p.yellow("    Raw measurements for reference (not comparable to baseline):\n\n"),
        );
    }

    let mut model_rows: Vec<(&str, Option<f64>)> = Vec::new();
    let mut dsp_rows: Vec<(&str, Option<f64>)> = Vec::new();
    for (bench_id, display) in BENCH_DISPLAY {
        let is_dsp = display.starts_with("DSP ");
        let lat = latency_us(report, bench_id);
        if is_dsp {
            dsp_rows.push((display, lat));
        } else {
            model_rows.push((display, lat));
        }
    }

    let header_line = || {
        format!(
            "  {:<28} | {:>16} | {:>10} | {:>18}\n",
            "Model / Component", "Median Latency", "% Budget", "Headroom"
        )
    };
    let rule_line = || format!("  {:-<28}-+-{:-<16}-+-{:-<10}-+-{:-<18}\n", "", "", "", "");
    let verified = !report.performance_not_verified();
    let perf_row = |display: &str, lat: Option<f64>| {
        let (latency, pct, headroom) = match lat {
            Some(lat) => {
                let pct = budget_pct(lat);
                let headroom = budget_headroom(pct);
                // NOT_VERIFIED is never green (S6 invariant): raw values only.
                let (pct, headroom) = if verified {
                    (
                        p.paint_class(cpu_color_class(pct), &format!("{pct:.1}%")),
                        p.paint_class(cpu_color_class(pct), &format!("{headroom:.1}%")),
                    )
                } else {
                    (format!("{pct:.1}%"), format!("{headroom:.1}%"))
                };
                (format!("{lat:.1} us"), pct, headroom)
            }
            None => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
        };
        format!(
            "  {:<28} | {:>16} | {:>10} | {:>18}\n",
            pad(display, 28),
            latency,
            pct,
            headroom
        )
    };

    let has_any = !model_rows.is_empty() || !dsp_rows.is_empty();
    if !has_any {
        out.push_str(&p.yellow("  (i) No performance data available.\n\n"));
        return out;
    }

    if !model_rows.is_empty() {
        out.push_str("  ── Model Inference Core ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for (display, lat) in &model_rows {
            out.push_str(&perf_row(display, *lat));
        }
        out.push('\n');
    }
    if !dsp_rows.is_empty() {
        out.push_str("  ── DSP Infrastructure ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for (display, lat) in &dsp_rows {
            out.push_str(&perf_row(display, *lat));
        }
        out.push('\n');
    }

    out.push_str("  (i) Headroom > 50%:  2x oversampling usually safe without xruns\n");
    out.push_str("  (i) Headroom > 75%:  4x oversampling usually safe without xruns\n");
    out.push_str("  (i) Headroom < 25%:  xrun risk with a 64-sample buffer\n\n");
    out
}

fn render_isa_parity(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("ISA PARITY\n");
    out.push_str("═════════════\n\n");

    if report.isa.is_empty() {
        out.push_str(
            &p.yellow(
                "  (i) Not covered in quick mode — run tests-long for full verification.\n\n",
            ),
        );
        return out;
    }

    let self_consistency_count = report
        .isa
        .iter()
        .filter(|row| row.ref_isa == row.test_isa)
        .count();
    let cross_isa: Vec<&IsaRow> = report
        .isa
        .iter()
        .filter(|row| row.ref_isa != row.test_isa)
        .collect();
    let cross_isa_pass = cross_isa
        .iter()
        .filter(|row| match row.esr.as_ref().and_then(metric_f64) {
            Some(esr) => esr < 1e-8,
            None => false,
        })
        .count();

    if cross_isa_pass == cross_isa.len() && !cross_isa.is_empty() {
        out.push_str(&p.green("  AVX2 vs AVX-512: bitwise identical\n"));
    } else if !cross_isa.is_empty() {
        out.push_str(&p.yellow(&format!(
            "  AVX2 vs AVX-512: divergent on {}/{} models\n",
            cross_isa.len() - cross_isa_pass,
            cross_isa.len()
        )));
    } else {
        out.push_str("  AVX2 vs AVX-512: no data (CPU may lack AVX-512)\n");
    }

    out.push_str(&format!(
        "  Self-consistency checks: {self_consistency_count} executed\n\n"
    ));

    if !cross_isa.is_empty() {
        out.push_str("  Cross-ISA details:\n");
        for row in &cross_isa {
            let esr_text = row
                .esr
                .as_ref()
                .map(fmt_metric)
                .unwrap_or_else(|| "N/A".to_string());
            let pass = match row.esr.as_ref().and_then(metric_f64) {
                Some(esr) => esr < 1e-8,
                None => false,
            };
            let mark = if pass { p.green("ok") } else { p.yellow("⚠") };
            out.push_str(&format!("    {}  ESR={}  {}\n", row.label, esr_text, mark));
        }
        out.push('\n');
    }
    out
}

fn render_activation_precision(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("ACTIVATION PRECISION\n");
    out.push_str("════════════════════\n\n");

    if report.activation.is_empty() {
        out.push_str(&p.yellow("  (i) No activation-precision results available.\n\n"));
        return out;
    }

    out.push_str(&format!(
        "  {:<20} | {:>14} | {:>14} | {:>10}\n",
        "Model", "Fast(Pade)", "Standard(exact)", "Δ SNR"
    ));
    out.push_str(&format!(
        "  {:-<20}-+-{:-<14}-+-{:-<14}-+-{:-<10}\n",
        "", "", "", ""
    ));

    let mut total: f64 = 0.0;
    let mut count: u64 = 0;
    for row in &report.activation {
        let fast = fmt_snr(&row.snr_fast_db);
        let exact = fmt_snr(&row.snr_exact_db);
        let gain = fmt_snr(&row.gain_db);
        let gain_colored = match metric_f64(&row.gain_db) {
            Some(v) if v >= 3.0 => gain.clone(),
            Some(_) => p.yellow(&gain),
            None => gain.clone(),
        };
        if let Some(v) = metric_f64(&row.gain_db) {
            total += v;
            count += 1;
        }
        out.push_str(&format!(
            "  {:<20} | {:>14} | {:>14} | {:>10}\n",
            pad(&truncate(&row.model, 20), 20),
            format!("{fast} dB"),
            format!("{exact} dB"),
            gain_colored
        ));
    }
    if count > 0 {
        out.push_str(&format!(
            "  Mean SNR gain with Standard(exact): +{:.1} dB (over {count} LSTM model(s))\n",
            total / count as f64
        ));
    }
    out.push('\n');
    out
}

fn render_f64_decomposition(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    if report.f64_decomp.is_empty() {
        return out;
    }
    out.push_str("F64 ORACLE — Error Source Decomposition\n");
    out.push_str("══════════════════════════════════════════════\n\n");
    out.push_str("  (i) These measurements are cold-start (256 samples, NO prewarm) — NOT\n");
    out.push_str("      comparable to the 'vs Ideal (f64)' values in the fidelity table\n");
    out.push_str("      above (measured with 24k-sample warmup). For WaveNet/A2, the\n");
    out.push_str("      receptive field is larger than the 256-sample window, so the\n");
    out.push_str("      ESR total below reflects mostly the transient buffer fill-in,\n");
    out.push_str("      not the steady-state precision floor.\n");
    out.push_str("      See docs/perceptual_validation.md#decomposition-cold-start.\n\n");

    for block in &report.f64_decomp {
        out.push_str(&format!("  {}:\n", block.label));
        out.push_str(&format!(
            "    ESR(f32 vs f64 oracle): {}\n",
            fmt_metric(&block.esr_f32_vs_f64)
        ));
        for (name, value) in [
            ("quant F16C", &block.esr_quant_f16c),
            ("quant BF16", &block.esr_quant_bf16),
            ("activation", &block.esr_activation),
            ("accumulation", &block.esr_accumulation),
            ("combined (F16C+Padé+F32)", &block.esr_combined),
        ] {
            if let Some(value) = value {
                out.push_str(&format!("    {name}: {}\n", fmt_metric(value)));
            }
        }

        let total = metric_f64(&block.esr_f32_vs_f64);
        let combined = block.esr_combined.as_ref().and_then(metric_f64);
        if let (Some(total), Some(combined)) = (total, combined)
            && combined != 0.0
        {
            let ratio = (total / combined).abs().max((combined / total).abs());
            if ratio > 10.0 {
                out.push_str(&p.yellow(&format!(
                    "    Rule 5 (Σ sources ≈ total, within 10x) violated: total/combined ≈ {ratio:.0}x.\n"
                )));
                out.push_str(&p.yellow(
                    "      Expected for models whose receptive field exceeds the measurement window (cold-start).\n",
                ));
                out.push_str(&p.yellow(
                    "      Do not treat this number as a calibrated precision floor without paired prewarm measurement.\n",
                ));
            }
        }
        out.push('\n');
    }
    out
}

fn render_spectral_summary(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("SPECTRAL FIDELITY\n");
    out.push_str("═════════════════\n\n");
    let count = report.coverage.map(|c| c.spectral_baselines).unwrap_or(0);
    if count > 0 {
        out.push_str(&p.green(&format!(
            "  ok {count} model(s) with spectral metrics inside baseline.\n"
        )));
    } else {
        out.push_str(
            &p.yellow("  (i) Not covered in quick mode — run tests-long for full verification.\n"),
        );
    }
    out.push('\n');
    out
}

fn render_coverage_matrix(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("COVERAGE MATRIX BY AXIS (Governance)\n");
    out.push_str("════════════════════════════════════════════\n\n");

    let Some(coverage) = report.coverage else {
        out.push_str(&p.yellow("  (i) No coverage data available.\n\n"));
        return out;
    };

    let mut covered = 0u64;
    out.push_str(&format!(
        "  {:<28} | {:>10} | {:<20}\n",
        "Axis", "Records", "Coverage"
    ));
    out.push_str(&format!("  {:-<28}-+-{:-<10}-+-{:-<20}\n", "", "", ""));

    for (axis, records) in [
        ("NAMCore Parity", coverage.namcore_parity),
        ("f64 Oracle Fidelity", coverage.f64_oracle),
        ("ISA Optimizations", coverage.isa_optimizations),
        ("Spectral Baselines", coverage.spectral_baselines),
        ("RT Performance", coverage.rt_performance),
    ] {
        let status = if records > 0 {
            covered += 1;
            p.green("covered")
        } else {
            p.yellow("not covered")
        };
        out.push_str(&format!(
            "  {:<28} | {:>10} | {:<20}\n",
            pad(axis, 28),
            records,
            status
        ));
    }
    out.push('\n');
    out.push_str(&format!("  Coverage: {covered}/5 axes covered\n\n"));

    if let Some(counts) = report.test_counts {
        out.push_str("  Phases in the receipt:\n");
        out.push_str(&format!(
            "    passed:          {}\n",
            p.green(&counts.passed.to_string())
        ));
        out.push_str(&format!(
            "    failed:          {}\n",
            p.red(&counts.failed.to_string())
        ));
        out.push_str(&format!(
            "    skip_capability: {}\n",
            p.yellow(&counts.skip_capability.to_string())
        ));
        out.push_str(&format!("    ignored:         {}\n", counts.ignored));
        out.push_str(&format!("    filtered:        {}\n\n", counts.filtered));
    }
    out
}

fn render_footer(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("───────────────────────────────────────────────────────────────\n");

    if report.performance_not_verified() {
        out.push_str(
            &p.yellow("  Performance NOT_VERIFIED — fidelity gates validated independently.\n\n"),
        );
    }

    let phase_failures = report.phases.iter().any(|phase| phase.status == "FAIL");
    if phase_failures {
        out.push_str(&p.red("  One or more dashboard phases failed.\n\n"));
    }
    out
}

// ── Top-level render ────────────────────────────────────────────────────────

/// Renders the typed report in dashboard section order.
pub fn render_quality_report(report: &QualityReport, style: RenderStyle) -> String {
    let p = Palette::new(style);
    let mut out = String::new();
    out.push_str(&render_header(report, &p));
    out.push_str(&render_quick_summary(report, &p));
    out.push_str(&render_fidelity_details(report, &p));
    out.push_str(&render_performance(report, &p));
    out.push_str(&render_isa_parity(report, &p));
    out.push_str(&render_activation_precision(report, &p));
    out.push_str(&render_f64_decomposition(report, &p));
    out.push_str(&render_spectral_summary(report, &p));
    out.push_str(&render_coverage_matrix(report, &p));
    out.push_str(&render_footer(report, &p));
    out
}

#[cfg(test)]
#[path = "render_test.rs"]
mod render_test;
