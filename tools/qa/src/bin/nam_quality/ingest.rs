// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Report ingestion: merges phase receipts, fidelity metrics, and Criterion latencies.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::process::exit;

use serde_json::Value;

use neural_amp_modeler_rs::testing::qa::QualityContract;
use neural_amp_modeler_rs::testing::qa::criterion::{
    DEFAULT_DSP_MICRO_BATCH, extract_criterion_latencies, parse_bench_constants_env,
};
use neural_amp_modeler_rs::testing::qa::ids;

use crate::cli::{HELP_REQUEST, parse_flags, required, run_error, usage_error};
use crate::save::write_new_file;

/// Merges dashboard artifacts into one verify report (JSONL), fail-closed on
/// malformed lines. Lines are passed through verbatim — the verify parser
/// routes them by shape (`phase_id` / `median_latency_us` / fidelity `kind`).
pub(crate) fn cmd_ingest(args: &[String]) {
    let flags = match parse_flags(
        args,
        &[
            "receipt",
            "metrics",
            "latency",
            "criterion-dir",
            "bench-constants-env",
            "contract",
            "out",
        ],
        &[],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_ingest_help();
            exit(0);
        }
        Err(e) => usage_error("ingest", &e),
    };
    let receipt = match required(&flags, "receipt") {
        Ok(v) => v,
        Err(e) => usage_error("ingest", &e),
    };
    let out = flags.get("out");

    let rendered = match run_ingest(
        receipt,
        flags.get("metrics"),
        flags.get("latency"),
        flags.get("criterion-dir"),
        flags.get("bench-constants-env"),
        flags.get("contract"),
        out,
    ) {
        Ok(r) => r,
        Err(e) => run_error(e),
    };
    if out.is_none() {
        print!("{rendered}");
        eprintln!("ingest: report written to stdout");
    }
    exit(0);
}

pub(crate) fn run_ingest(
    receipt: &str,
    metrics_path: Option<&str>,
    latency_path: Option<&str>,
    criterion_dir: Option<&str>,
    bench_constants_env: Option<&str>,
    contract_path: Option<&str>,
    out: Option<&str>,
) -> Result<String, String> {
    let receipt_lines = read_jsonl_lines(receipt)
        .map_err(|e| format!("cannot read phase receipt {receipt}: {e}"))?;
    let mut phases = Vec::new();
    for (i, line) in receipt_lines.iter().enumerate() {
        let value: Value = serde_json::from_str(line).map_err(|source| {
            format!(
                "malformed phase receipt line {} in {receipt}: {source}",
                i + 1
            )
        })?;
        // Lines carrying `phase_id` must have a string `status`; provenance
        // records (`kind: build_metadata`) have no `phase_id` and pass
        // through — the verify parser skips them.
        if value.get("phase_id").is_some()
            && (value.get("phase_id").and_then(Value::as_str).is_none()
                || value.get("status").and_then(Value::as_str).is_none())
        {
            return Err(format!(
                "phase receipt line {} in {receipt} must have string `phase_id` and `status`",
                i + 1
            ));
        }
        phases.push(line.to_string());
    }

    let metrics = match metrics_path {
        Some(path) => {
            let lines = read_jsonl_lines(path)
                .map_err(|e| format!("cannot read metrics JSONL {path}: {e}"))?;
            for (i, line) in lines.iter().enumerate() {
                if let Err(source) = serde_json::from_str::<Value>(line) {
                    return Err(format!(
                        "malformed metrics JSONL line {} in {path}: {source}",
                        i + 1
                    ));
                }
            }
            lines
        }
        None => Vec::new(),
    };

    // Latency stream resolution:
    // If `--criterion-dir` is provided, ingest directly from Criterion's `estimates.json`.
    // Otherwise fallback to reading `--latency` stream.
    let latency: Vec<String> = if let Some(criterion_dir) = criterion_dir {
        let micro_batch = match bench_constants_env {
            Some(env_path) => {
                let content = fs::read_to_string(env_path)
                    .map_err(|e| format!("cannot read bench constants env {env_path}: {e}"))?;
                parse_bench_constants_env(&content).unwrap_or(DEFAULT_DSP_MICRO_BATCH)
            }
            None => DEFAULT_DSP_MICRO_BATCH,
        };

        let contract = match contract_path {
            Some(cpath) => {
                let content = fs::read_to_string(cpath)
                    .map_err(|e| format!("cannot read quality contract {cpath}: {e}"))?;
                let c: QualityContract = serde_json::from_str(&content)
                    .map_err(|e| format!("malformed quality contract JSON {cpath}: {e}"))?;
                Some(c)
            }
            None => None,
        };

        let entries =
            extract_criterion_latencies(Path::new(criterion_dir), micro_batch, contract.as_ref())
                .map_err(|e| format!("failed to extract Criterion latencies: {e}"))?;

        entries
            .into_iter()
            .map(|e| {
                serde_json::to_string(&e)
                    .map_err(|err| format!("cannot serialize latency entry: {err}"))
            })
            .collect::<Result<Vec<String>, String>>()?
    } else if let Some(lat_path) = latency_path {
        let lines = read_jsonl_lines(lat_path)
            .map_err(|e| format!("cannot read latency JSONL {lat_path}: {e}"))?;
        for (i, line) in lines.iter().enumerate() {
            if let Err(source) = serde_json::from_str::<Value>(line) {
                return Err(format!(
                    "malformed latency JSONL line {} in {lat_path}: {source}",
                    i + 1
                ));
            }
        }
        lines
    } else {
        Vec::new()
    };

    // Join the f64-oracle table (`kind: f64_table`) onto the fidelity
    // records: `verify_contract` reads `esr_f64` from the SAME record as the
    // NAMCore metrics, so the fixture ESR has to land on the fidelity label
    // (P0.T3 — the reference_oracle_f64 phase passed but its values never
    // reached the verify key).
    let metrics = join_f64_oracle_esr(&metrics);
    let telemetry_summary = synthesize_governance_records(&phases, &metrics, &latency);

    let mut rendered = String::new();
    for line in phases
        .iter()
        .chain(&metrics)
        .chain(&latency)
        .chain(&telemetry_summary)
    {
        rendered.push_str(line);
        rendered.push('\n');
    }

    if let Some(path) = out {
        write_new_file(path, &rendered)
            .map_err(|e| format!("cannot write report to {path}: {e}"))?;
        eprintln!(
            "ingest: report written to {path} ({} phase record(s), {} metric record(s), {} latency record(s))",
            phases.len(),
            metrics.len(),
            latency.len()
        );
    }

    Ok(rendered)
}

/// Joins the `f64_table` oracle ESR onto the fidelity records it measures.
///
/// The `reference_oracle_f64` phase sinks one `kind: "f64_table"` record per
/// golden fixture (`filename` + prewarm-paired `esr`), while the
/// golden/quick fidelity records carry the NAMCore metrics under the contract
/// label. `verify_contract` reads `esr_f64` from the fidelity record itself,
/// so the two streams must be joined here: a fidelity record whose label
/// family resolves (via `ids::resolve_f64_oracle_fixture`) to a measured
/// fixture receives that fixture's ESR as `esr_f64`.
///
/// Fail-closed: records without a resolution are passed through verbatim —
/// never fabricated, never coerced to `0.0`. Non-fidelity kinds (including
/// the `f64_table` rows themselves) survive untouched as the forensic stream.
fn join_f64_oracle_esr(metrics: &[String]) -> Vec<String> {
    if metrics.is_empty() {
        return metrics.to_vec();
    }
    let mut parsed: Vec<(usize, Value)> = Vec::with_capacity(metrics.len());
    let mut f64_by_fixture: HashMap<String, Value> = HashMap::new();
    for (idx, line) in metrics.iter().enumerate() {
        // `read_optional_stream` already validated every line; a parse miss
        // here is unreachable but must not corrupt the join.
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("kind").and_then(Value::as_str) == Some("f64_table")
            && let (Some(filename), Some(esr)) = (
                value.get("filename").and_then(Value::as_str),
                value.get("esr"),
            )
        {
            f64_by_fixture
                .entry(filename.to_string())
                .or_insert_with(|| esr.clone());
        }
        parsed.push((idx, value));
    }
    if f64_by_fixture.is_empty() {
        return metrics.to_vec();
    }

    let mut out: Vec<Option<String>> = metrics.iter().map(|l| Some(l.clone())).collect();
    for (idx, value) in parsed {
        let is_fidelity = match value.get("kind") {
            None => true,
            Some(Value::String(kind)) => kind == "fidelity",
            Some(_) => false,
        };
        let Some(label) = value.get("label").and_then(Value::as_str) else {
            continue;
        };
        if !is_fidelity || value.get("esr_f64").is_some() {
            continue;
        }
        // Alias-aware join: quick_parity records like `Quick ConvNet @48000
        // Live` resolve to the contract label `ConvNet Test @48000 Live`
        // (`ids::resolve_fidelity_alias`) — the family must be stripped from
        // the RESOLVED label or the fixture lookup misses.
        let resolved = ids::resolve_fidelity_alias(label).unwrap_or(label);
        let family = ids::fidelity_label_family(resolved);
        let Some(fixture) = ids::resolve_f64_oracle_fixture(family) else {
            continue;
        };
        let Some(esr) = f64_by_fixture.get(fixture) else {
            continue;
        };
        let Some(obj) = value.as_object() else {
            continue;
        };
        let mut joined = obj.clone();
        joined.insert("esr_f64".to_string(), esr.clone());
        let provenance = if ids::f64_oracle_fixture_is_substitute(family) {
            format!("proxy:{fixture}")
        } else {
            fixture.to_string()
        };
        joined.insert(
            "f64_oracle_provenance".to_string(),
            Value::String(provenance),
        );
        out[idx] = Some(serde_json::Value::Object(joined).to_string());
    }
    out.into_iter().flatten().collect()
}

/// Synthesizes `coverage_matrix` and `test_counts` records from the ingested
/// phase, metric, and latency streams when those summary records are not
/// already present in `metrics`.
fn synthesize_governance_records(
    phases: &[String],
    metrics: &[String],
    latency: &[String],
) -> Vec<String> {
    let mut has_coverage = false;
    let mut has_counts = false;
    let mut namcore_parity = 0u64;
    let mut f64_table = 0u64;
    let mut f64_joined = 0u64;
    let mut isa_optimizations = 0u64;
    let mut spectral_baselines = 0u64;

    for line in metrics {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        match v.get("kind").and_then(Value::as_str) {
            Some("coverage_matrix") => has_coverage = true,
            Some("test_counts") => has_counts = true,
            Some("f64_table") => f64_table += 1,
            Some("isa") => isa_optimizations += 1,
            Some("spectral_baseline") => spectral_baselines += 1,
            Some("fidelity") | None => {
                if v.get("label").and_then(Value::as_str).is_some() {
                    namcore_parity += 1;
                    if v.get("esr_f64").is_some_and(|e| !e.is_null()) {
                        f64_joined += 1;
                    }
                }
            }
            Some(_) => {}
        }
    }

    let mut passed = 0u64;
    let mut failed = 0u64;
    let mut skip_capability = 0u64;
    let mut has_phase_records = false;
    for line in phases {
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(status) = v.get("status").and_then(Value::as_str) else {
            continue;
        };
        if v.get("phase_id").is_none() {
            continue;
        }
        has_phase_records = true;
        match status {
            "PASS" | "PASSED" => passed += 1,
            "FAIL" | "FAILED" => failed += 1,
            "SKIP_CAPABILITY" | "NOT_VERIFIED" => skip_capability += 1,
            _ => {}
        }
    }

    let mut out = Vec::new();
    if !has_coverage
        && (namcore_parity > 0
            || f64_table > 0
            || f64_joined > 0
            || isa_optimizations > 0
            || spectral_baselines > 0
            || !latency.is_empty())
    {
        let rec = serde_json::json!({
            "kind": "coverage_matrix",
            "namcore_parity": namcore_parity,
            "f64_oracle": if f64_table > 0 { f64_table } else { f64_joined },
            "isa_optimizations": isa_optimizations,
            "spectral_baselines": spectral_baselines,
            "rt_performance": latency.len() as u64,
        });
        out.push(rec.to_string());
    }
    if !has_counts && has_phase_records {
        let rec = serde_json::json!({
            "kind": "test_counts",
            "passed": passed,
            "failed": failed,
            "ignored": 0,
            "filtered": 0,
            "skip_capability": skip_capability,
        });
        out.push(rec.to_string());
    }
    out
}

fn print_ingest_help() {
    println!("Usage: nam_quality ingest [--receipt <path>] [--metrics <path>]");
    println!("                             [--latency <path>] [--criterion-dir <path>]");
    println!("                             [--bench-constants-env <path>] [--contract <path>]");
    println!("                             [--out <path>]");
    println!();
    println!("Merges the dashboard artifacts into one verify report (JSONL):");
    println!(
        "  --receipt <path>              phase receipt (dashboard_phase_receipt schema) — required;"
    );
    println!(
        "                                every line must carry string `phase_id` and `status`."
    );
    println!(
        "  --metrics <path>              fidelity metrics JSONL (NAM_METRICS_JSONL shape) — optional."
    );
    println!("  --latency <path>              latency JSONL (kind `latency`) — optional.");
    println!(
        "  --criterion-dir <path>        directory containing Criterion benchmarks (extracts median point estimates) — optional."
    );
    println!(
        "  --bench-constants-env <path>  path to bench_constants.env for DSP_MICRO_BATCH factor — optional."
    );
    println!(
        "  --contract <path>             quality contract for benchmark batch_factor resolution — optional."
    );
    println!("  --out <path>                  report destination; stdout when omitted.");
    println!();
    println!("Fail-closed: any malformed line in a given input aborts with exit 1.");
}

pub(crate) fn read_jsonl_lines(path: &str) -> io::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}
