// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Transactional contract promotion and atomic persistence.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::exit;

use serde_json::Value;

use neural_amp_modeler_rs::testing::qa::QualityContract;
use neural_amp_modeler_rs::testing::qa::env::EnvProbe;
use neural_amp_modeler_rs::testing::qa::ids;
use neural_amp_modeler_rs::testing::qa::metrics::{FidelityRecord, MetricValue};
use neural_amp_modeler_rs::testing::qa::verify::{PROVENANCE_PHASE, parse_verify_report_file};

use crate::cli::{HELP_REQUEST, parse_flags, required, run_error, usage_error};
use crate::ingest::read_jsonl_lines;
use crate::third_party::detect_active_isa;

pub(crate) fn cmd_save(args: &[String]) {
    let flags = match parse_flags(args, &["contract", "receipt"], &[]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_save_help();
            exit(0);
        }
        Err(e) => usage_error("save", &e),
    };
    let contract_path = match required(&flags, "contract") {
        Ok(v) => v,
        Err(e) => usage_error("save", &e),
    };
    let receipt_path = match required(&flags, "receipt") {
        Ok(v) => v,
        Err(e) => usage_error("save", &e),
    };

    execute_save(contract_path, receipt_path);
    exit(0);
}

/// Reads and validates the phase receipt before saving a contract.
///
/// Refuses (`run_error`) if the provenance phase (`freshness`) or any fidelity
/// phase failed; when `require_perf` is true, also refuses if `regression_gate`
/// failed or did not complete with `PASS`.
fn validate_receipt_for_save(receipt_path: &str, require_perf: bool) {
    let receipt_lines = match read_jsonl_lines(receipt_path) {
        Ok(lines) => lines,
        Err(e) => run_error(format!("cannot read phase receipt {receipt_path}: {e}")),
    };
    let mut perf_status: Option<String> = None;
    for (i, line) in receipt_lines.iter().enumerate() {
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(source) => run_error(format!(
                "malformed phase receipt line {} in {receipt_path}: {source}",
                i + 1
            )),
        };
        // Provenance records (`kind: build_metadata`) carry no
        // `phase_id` and are irrelevant to the fidelity gate.
        let Some(phase_id) = value.get("phase_id").and_then(Value::as_str) else {
            continue;
        };
        let status = value.get("status").and_then(Value::as_str).unwrap_or("");
        if status == "FAIL" && phase_id == PROVENANCE_PHASE {
            run_error(format!(
                "contract NOT saved: provenance phase '{phase_id}' FAILed (--save requires fresh fixtures)"
            ));
        }
        if status == "FAIL" && phase_id != "regression_gate" {
            run_error(format!(
                "contract NOT saved: fidelity phase '{phase_id}' FAILed (--save requires all fidelity phases to pass)"
            ));
        }
        if phase_id == "regression_gate" {
            perf_status = Some(status.to_string());
        }
    }
    if require_perf && perf_status.as_deref() != Some("PASS") {
        let actual = perf_status.as_deref().unwrap_or("NOT_RUN");
        run_error(format!(
            "contract NOT saved: regression_gate status is '{actual}' (--save without --fidelity-only requires PASS)"
        ));
    }
}

fn parse_finite_metric(v: &MetricValue) -> Option<f64> {
    v.as_finite()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|x| x.is_finite())
}

pub(crate) fn save_dashboard_contract(
    contract_path: &str,
    receipt_path: &str,
    report_path: &str,
    run_id: &str,
    fidelity_only: bool,
    perf_passed: bool,
) {
    validate_receipt_for_save(receipt_path, !fidelity_only);
    if !fidelity_only && !perf_passed {
        run_error("contract NOT saved: performance regression_gate did not pass".to_string());
    }

    let mut contract = match fs::read_to_string(contract_path)
        .map_err(|e| format!("cannot read contract {contract_path}: {e}"))
        .and_then(|s| QualityContract::from_json_str(&s).map_err(|e| e.to_string()))
    {
        Ok(c) => c,
        Err(e) => run_error(e),
    };
    let report = match parse_verify_report_file(report_path) {
        Ok(r) => r,
        Err(e) => run_error(format!("cannot parse verify report {report_path}: {e}")),
    };

    let probe = EnvProbe::probe();
    let active_isa = detect_active_isa(&probe);
    contract.generated_at = neural_amp_modeler_rs::common::diagnostics::format::timestamp();
    contract.provenance.git_commit = probe.git_commit;
    contract.provenance.git_dirty = probe.git_dirty;
    contract.provenance.run_id = run_id.to_string();
    contract.provenance.effective_isa = active_isa.to_string();
    contract.provenance.cpu_model = probe.cpu_model;
    contract.provenance.rustc = probe.rustc_version;
    contract.provenance.cargo_profile = "release".to_string();
    contract.provenance.hardware_class = Some(probe.hardware_class.as_str().to_string());

    let mut report_by_label: HashMap<&str, &FidelityRecord> = HashMap::new();
    for record in &report.fidelity {
        report_by_label
            .entry(record.label.as_str())
            .or_insert(record);
    }
    for record in &report.fidelity {
        if let Some(canonical) = ids::resolve_fidelity_alias(&record.label) {
            report_by_label.entry(canonical).or_insert(record);
        }
    }

    let mut worst_esr: f64 = 0.0;
    for entry in &mut contract.fidelity {
        if let Some(record) = report_by_label.get(entry.label.as_str()).copied() {
            if let Some(esr) = parse_finite_metric(&record.esr) {
                entry.esr_namcore = esr;
            }
            if let Some(esr_f64) = parse_finite_metric(&record.esr_f64)
                && !record
                    .f64_oracle_provenance
                    .as_deref()
                    .is_some_and(|p| p.starts_with("proxy:"))
            {
                entry.esr_f64 = Some(esr_f64);
            } else if let Some(esr_f64) = parse_finite_metric(&record.esr_f64)
                && entry.esr_f64.is_some()
            {
                entry.esr_f64 = Some(esr_f64);
            }
            match &record.snr_db {
                MetricValue::Null => entry.snr_db = None,
                raw => {
                    if let Some(snr) = parse_finite_metric(raw) {
                        entry.snr_db = Some(snr);
                    }
                }
            }
            if let Some(mrstft) = parse_finite_metric(&record.mrstft) {
                entry.mrstft = mrstft;
            }
        } else if !entry.optional {
            run_error(format!(
                "contract NOT saved: missing mandatory fidelity record '{}'",
                entry.label
            ));
        }
        if !entry.optional && entry.esr_namcore.is_finite() {
            worst_esr = worst_esr.max(entry.esr_namcore);
        }
    }

    if !fidelity_only {
        for entry in &mut contract.performance {
            let Some(lat) = report
                .latency
                .iter()
                .find(|r| r.label == entry.id || r.label == entry.label)
            else {
                run_error(format!(
                    "contract NOT saved: missing latency measurement for '{}'",
                    entry.id
                ));
            };
            entry.median_latency_us = (lat.median_latency_us * 100.0).round() / 100.0;
        }
    }

    if let Some(ref mut alerts) = contract.monitoring_alerts {
        alerts.worst_esr_parity.measured = worst_esr;
    }

    let rendered = match contract.to_json_pretty() {
        Ok(s) => s + "\n",
        Err(e) => run_error(format!("cannot serialize contract: {e}")),
    };
    if let Err(e) = atomic_write(Path::new(contract_path), &rendered) {
        run_error(format!("contract save failed: {e}"));
    }
    eprintln!(
        "contract saved: {contract_path} (atomic, provenance={}, fidelity={}, performance={})",
        contract.provenance.git_commit,
        contract.fidelity.len(),
        if fidelity_only {
            "preserved".to_string()
        } else {
            contract.performance.len().to_string()
        }
    );
}

fn execute_save(contract_path: &str, receipt_path: &str) {
    // Validate the contract payload before touching anything (fail-closed:
    // a corrupt contract must never be promoted).
    let contract = match fs::read_to_string(contract_path)
        .map_err(|e| format!("cannot read contract {contract_path}: {e}"))
        .and_then(|s| QualityContract::from_json_str(&s).map_err(|e| e.to_string()))
    {
        Ok(c) => c,
        Err(e) => run_error(e),
    };

    // Fidelity gate: fidelity-domain phases must PASS; the performance domain
    // never blocks (PERF-006). Provenance is its own domain: a stale fixture
    // generator blocks the snapshot, but is reported as a provenance failure,
    // never as a fidelity failure.
    validate_receipt_for_save(receipt_path, false);

    let rendered = match contract.to_json_pretty() {
        Ok(s) => s + "\n",
        Err(e) => run_error(format!("cannot serialize contract: {e}")),
    };
    if let Err(e) = atomic_write(Path::new(contract_path), &rendered) {
        run_error(format!("contract save failed: {e}"));
    }
    eprintln!("contract saved: {contract_path} (atomic, fidelity phases verified)");
}

fn print_save_help() {
    println!("Usage: nam_quality save --contract <path> --receipt <path>");
    println!();
    println!("Transactionally promotes the validated contract JSON at <path>");
    println!("(temp file in the same directory + atomic rename). Refuses with exit 1");
    println!("when any fidelity phase (golden_vectors, reference_oracle_f64,");
    println!("quick_parity — or any phase other than regression_gate) FAILed in the");
    println!("phase receipt, or when the provenance phase (freshness) FAILed.");
    println!("Performance NOT_VERIFIED never blocks (PERF-006).");
}

/// Writes `content` to a temp sibling of `path` and renames it over `path`,
/// so readers never observe a partially written contract.
fn atomic_write(path: &Path, content: &str) -> io::Result<()> {
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let file_name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid contract path"))?;
    let mut tmp = parent
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tmp.push(format!(
        ".{}.tmp.{}",
        file_name.to_string_lossy(),
        std::process::id()
    ));
    fs::write(&tmp, content)?;
    match fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Creates `path` (with parent directories) and writes `content`; used by
/// `ingest --out` so the wrapper does not need to pre-create directories.
pub(crate) fn write_new_file(path: &str, content: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)
}
