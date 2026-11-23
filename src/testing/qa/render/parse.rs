// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! JSONL parser for the quality dashboard report.

use serde_json::Value;

use super::super::metrics::{MetricValue, fidelity_from_json};
use super::super::verify::{LatencyRecord, PhaseRecord, parse_record_counts};
use super::types::{
    ActivationRow, CoverageMatrix, F64Decomp, F64TableRow, IsaRow, QualityReport, ReportError,
    ReportHeader, TestCounts,
};

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
                // `median_latency_us`-only latency shape of legacy fixtures.
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
    let (observed_records, expected_records) = parse_record_counts(value);
    Some(PhaseRecord {
        phase_id,
        status,
        observed_records,
        expected_records,
    })
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
    if let Some(v) = value.get("host_isa").and_then(Value::as_str) {
        header.host_isa = Some(v.to_string());
    }
    if let Some(v) = value.get("engine_backend").and_then(Value::as_str) {
        header.engine_backend = Some(v.to_string());
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
    let oracle = value
        .get("oracle")
        .and_then(|v| v.as_str())
        .map(ToString::to_string);
    Some(ActivationRow {
        model,
        oracle,
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
