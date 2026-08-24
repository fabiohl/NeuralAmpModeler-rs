// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `nam_quality` — the JSON-only quality gate engine.
//!
//! The dashboard bash wrapper only orchestrates phases and delegates all
//! contract interpretation to this binary; there is **no** ASCII-contract
//! loader and no `jq` anywhere in the pipeline. The contract authority is
//! `docs/quality-contract.json`.
//!
//! Subcommands:
//! - `ingest`: merge the dashboard artifacts (phase receipt, fidelity metrics
//!   JSONL, latency JSONL) into one fail-closed verify report (JSONL). The
//!   wrapper calls this once per run; `verify` consumes the result.
//! - `verify --contract <path> --report <path>`: run the literal port of the
//!   bash `verify_contract` and print the verdict lines
//!   (`FIDELITY: OK/FAIL`, `PERFORMANCE: OK/FAIL/NOT_VERIFIED`,
//!   `CONTRACT VIOLATED`) mirroring the dashboard text. Exit 0 only when
//!   both domains pass and no oracle review is pending.
//! - `receipt append`: append one phase receipt record (serde — replaces the
//!   dashboard `printf '{`), byte-compatible with the old bash schema.
//! - `save --contract <path> --receipt <path>`: transactional promotion of a
//!   validated contract JSON (temp file + atomic rename). Refuses when a
//!   fidelity phase FAILed — the performance domain never blocks saving
//!   (PERF-006; `NOT_VERIFIED` performance must not block, bash comment at
//!   `quality-dashboard.sh:2599-2602`).
//!
//! Fail-closed CLI contract:
//! - exit 0: success;
//! - exit 1: run-time failure (unreadable input, invalid contract/report,
//!   violated gate, refused save);
//! - exit 2: usage error (unknown subcommand, unknown flag, missing required
//!   flag/value, malformed flag value). `--save`/`--check`-style wrappers
//!   must therefore validate their path arguments both in bash and here.
//!
//! Available only with the `testing` feature (`required-features`); without
//! it the binary does not exist — the public crate vocation stays intact.

use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::exit;

use serde::Serialize;
use serde_json::Value;

use neural_amp_modeler_rs::testing::qa::QualityContract;
use neural_amp_modeler_rs::testing::qa::classify::{
    classify_fresh_regression, classify_regression_outcome,
};
use neural_amp_modeler_rs::testing::qa::ids;
use neural_amp_modeler_rs::testing::qa::render::{
    RenderStyle, parse_quality_report_file, render_quality_report,
};
use neural_amp_modeler_rs::testing::qa::verify::{
    FidelityOutcome, FidelityVerdict, MalformedReason, Metric as MetricId, MetricOutcome,
    PerfResult, PerformanceVerdict, VerifyOutcome, parse_verify_report_file, verify_contract,
};

/// Status values accepted by `receipt append` — the documented receipt schema
/// (`utils/_lib.sh:80`) plus `NOT_VERIFIED`, which the dashboard writes for
/// `regression_gate` (`quality-dashboard.sh:502`).
const RECEIPT_STATUSES: [&str; 6] = [
    "PASS",
    "FAIL",
    "SKIP_CAPABILITY",
    "SKIP_OPTIONAL_FIXTURE",
    "NOT_RUN",
    "NOT_VERIFIED",
];

/// Mandatory fidelity phases — a FAIL here (or on any non-`regression_gate`
/// phase) refuses `save` and fails the fidelity domain of `verify`.
const FIDELITY_PHASES: [&str; 3] = ["golden_vectors", "reference_oracle_f64", "quick_parity"];

/// Sentinel used internally to forward a `--help` request from `parse_flags`.
const HELP_REQUEST: &str = "__help__";

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("missing subcommand");
        print_help();
        exit(2);
    }
    match args[0].as_str() {
        "-h" | "--help" => {
            print_help();
            exit(0);
        }
        "ingest" => cmd_ingest(&args[1..]),
        "verify" => cmd_verify(&args[1..]),
        "render" => cmd_render(&args[1..]),
        "classify" => cmd_classify(&args[1..]),
        "receipt" => cmd_receipt(&args[1..]),
        "save" => cmd_save(&args[1..]),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            exit(2);
        }
    }
}

fn print_help() {
    println!("NeuralAmpModeler-rs Quality Gate");
    println!();
    println!("The JSON-only verify engine: contracts come from");
    println!("docs/quality-contract.json; reports are JSONL. No ASCII loader, no jq.");
    println!();
    println!("Usage:");
    println!("  nam_quality ingest [--receipt <path>] [--metrics <path>] [--latency <path>]");
    println!("                          [--out <path>]");
    println!("  nam_quality verify --contract <path> --report <path> [--fidelity-only]");
    println!("  nam_quality render --report <path> [--ansi | --plain]");
    println!("  nam_quality classify --status <STATUS> --reason <text>");
    println!("  nam_quality receipt append --phase-id <id> --status <STATUS> --out <path>");
    println!("                          [--exit-code <n>] [--observed-records <n>]");
    println!("                          [--expected-records <n>] [--reason <text>]");
    println!("                          [--run-id <id>]");
    println!("  nam_quality save --contract <path> --receipt <path>");
    println!("  nam_quality --help");
    println!();
    println!("Exit codes: 0 success, 1 run-time failure (gate violated, unreadable input,");
    println!("refused save), 2 usage error (unknown subcommand/flag, missing argument).");
}

// ── Strict flag parsing (fail-closed: unknown flag ⇒ exit 2) ─────────────────

struct Flags {
    map: HashMap<String, String>,
}

impl Flags {
    fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }
}

/// Parses `--flag value` pairs; only `allowed` flags are accepted. Flags in
/// `boolean` are valueless switches (stored as `"true"`). A `-h`/`--help`
/// request surfaces as [`HELP_REQUEST`].
fn parse_flags(args: &[String], allowed: &[&str], boolean: &[&str]) -> Result<Flags, String> {
    let mut map = HashMap::new();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        if flag == "-h" || flag == "--help" {
            return Err(HELP_REQUEST.to_string());
        }
        let Some(name) = flag.strip_prefix("--") else {
            return Err(format!("{flag}: expected a --flag"));
        };
        if name.is_empty() {
            return Err(format!("{flag}: expected a flag name"));
        }
        if !allowed.contains(&name) {
            return Err(format!("unknown flag: --{name}"));
        }
        if boolean.contains(&name) {
            map.insert(name.to_string(), "true".to_string());
            continue;
        }
        let value = iter
            .next()
            .ok_or_else(|| format!("--{name} requires a value"))?;
        map.insert(name.to_string(), value.clone());
    }
    Ok(Flags { map })
}

/// Exits 2 with a usage message (fail-closed CLI contract).
fn usage_error(context: &str, message: &str) -> ! {
    eprintln!("{context}: {message}");
    eprintln!("Run `nam_quality --help` for usage.");
    exit(2);
}

/// Exits 1 with an `ERROR:` line (fail-closed run-time contract).
fn run_error(message: String) -> ! {
    eprintln!("ERROR: {message}");
    exit(1);
}

fn required<'a>(flags: &'a Flags, name: &str) -> Result<&'a str, String> {
    flags
        .get(name)
        .ok_or_else(|| format!("missing required flag --{name}"))
}

/// Parses a non-negative integer flag value.
///
/// T2.3: values produced by shell `grep -c` pipelines may carry multiple
/// lines. The FIRST non-empty, trimmed line must be the plain count — the
/// embedded newline is tolerated instead of failing the whole receipt. A
/// non-numeric value (including the `file:count` form of a multi-file grep,
/// which is rejected fail-closed) still fails with a usage error.
fn parse_u32(flags: &Flags, name: &str, context: &str) -> u32 {
    match flags.get(name) {
        None => 0,
        Some(v) => {
            let first_line = v
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or("");
            match first_line.parse() {
                Ok(n) => n,
                Err(_) => usage_error(
                    context,
                    &format!("--{name} must be a non-negative integer (got '{first_line}')"),
                ),
            }
        }
    }
}

// ── ingest ───────────────────────────────────────────────────────────────────

/// Merges dashboard artifacts into one verify report (JSONL), fail-closed on
/// malformed lines. Lines are passed through verbatim — the verify parser
/// routes them by shape (`phase_id` / `median_latency_us` / fidelity `kind`).
fn cmd_ingest(args: &[String]) {
    let flags = match parse_flags(args, &["receipt", "metrics", "latency", "out"], &[]) {
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

    let receipt_lines = match read_jsonl_lines(receipt) {
        Ok(lines) => lines,
        Err(e) => run_error(format!("cannot read phase receipt {receipt}: {e}")),
    };
    let mut phases = Vec::new();
    for (i, line) in receipt_lines.iter().enumerate() {
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(source) => run_error(format!(
                "malformed phase receipt line {} in {receipt}: {source}",
                i + 1
            )),
        };
        // Lines carrying `phase_id` must have a string `status`; provenance
        // records (`kind: build_metadata`, T0.4) have no `phase_id` and pass
        // through — the verify parser skips them.
        if value.get("phase_id").is_some()
            && (value.get("phase_id").and_then(Value::as_str).is_none()
                || value.get("status").and_then(Value::as_str).is_none())
        {
            run_error(format!(
                "phase receipt line {} in {receipt} must have string `phase_id` and `status`",
                i + 1
            ));
        }
        phases.push(line.to_string());
    }

    let metrics = read_optional_stream("metrics", &flags);
    let latency = read_optional_stream("latency", &flags);

    // Join the f64-oracle table (`kind: f64_table`) onto the fidelity
    // records: `verify_contract` reads `esr_f64` from the SAME record as the
    // NAMCore metrics, so the fixture ESR has to land on the fidelity label
    // (P0.T3 — the reference_oracle_f64 phase passed but its values never
    // reached the verify key).
    let metrics = join_f64_oracle_esr(&metrics);

    let mut rendered = String::new();
    for line in phases.iter().chain(&metrics).chain(&latency) {
        rendered.push_str(line);
        rendered.push('\n');
    }

    match out {
        Some(path) => {
            if let Err(e) = write_new_file(path, &rendered) {
                run_error(format!("cannot write report to {path}: {e}"));
            }
            eprintln!(
                "ingest: report written to {path} ({} phase record(s), {} metric record(s), {} latency record(s))",
                phases.len(),
                metrics.len(),
                latency.len()
            );
        }
        None => {
            print!("{rendered}");
            eprintln!(
                "ingest: report written to stdout ({} phase record(s), {} metric record(s), {} latency record(s))",
                phases.len(),
                metrics.len(),
                latency.len()
            );
        }
    }
    exit(0);
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
        out[idx] = Some(serde_json::Value::Object(joined).to_string());
    }
    out.into_iter().flatten().collect()
}

fn print_ingest_help() {
    println!("Usage: nam_quality ingest [--receipt <path>] [--metrics <path>]");
    println!("                             [--latency <path>] [--out <path>]");
    println!();
    println!("Merges the dashboard artifacts into one verify report (JSONL):");
    println!("  --receipt <path>  phase receipt (dashboard_phase_receipt schema) — required;");
    println!("                    every line must carry string `phase_id` and `status`.");
    println!("  --metrics <path>  fidelity metrics JSONL (NAM_METRICS_JSONL shape) — optional.");
    println!("  --latency <path>  latency JSONL (kind `latency`) — optional.");
    println!("  --out <path>      report destination; stdout when omitted.");
    println!();
    println!("Fail-closed: any malformed line in a given input aborts with exit 1.");
}

/// Reads one optional ingest stream. Absent flag ⇒ empty stream; present
/// flag with unreadable or malformed content ⇒ exit 1.
fn read_optional_stream(name: &str, flags: &Flags) -> Vec<String> {
    let Some(path) = flags.get(name) else {
        return Vec::new();
    };
    let lines = match read_jsonl_lines(path) {
        Ok(lines) => lines,
        Err(e) => run_error(format!("cannot read {name} JSONL {path}: {e}")),
    };
    for (i, line) in lines.iter().enumerate() {
        if let Err(source) = serde_json::from_str::<Value>(line) {
            run_error(format!(
                "malformed {name} JSONL line {} in {path}: {source}",
                i + 1
            ));
        }
    }
    lines
}

fn read_jsonl_lines(path: &str) -> io::Result<Vec<String>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect())
}

// ── verify ──────────────────────────────────────────────────────────────────

fn cmd_verify(args: &[String]) {
    let flags = match parse_flags(
        args,
        &["contract", "report", "fidelity-only"],
        &["fidelity-only"],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_verify_help();
            exit(0);
        }
        Err(e) => usage_error("verify", &e),
    };
    let contract_path = match required(&flags, "contract") {
        Ok(v) => v,
        Err(e) => usage_error("verify", &e),
    };
    let report_path = match required(&flags, "report") {
        Ok(v) => v,
        Err(e) => usage_error("verify", &e),
    };
    let fidelity_only = flags.get("fidelity-only").is_some();

    let contract = match fs::read_to_string(contract_path)
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

    let outcome = verify_contract(&contract, &report);
    print!(
        "{}",
        render_outcome(&contract, &report, &outcome, fidelity_only)
    );
    let exit_code = if fidelity_only {
        outcome.fidelity_exit_code()
    } else {
        outcome.exit_code()
    };
    exit(exit_code);
}

fn print_verify_help() {
    println!("Usage: nam_quality verify --contract <path> --report <path> [--fidelity-only]");
    println!();
    println!("Runs the literal port of the bash `verify_contract` against");
    println!("docs/quality-contract.json (JSON-only) and a JSONL report produced");
    println!("by `nam_quality ingest` (or by the test fixtures).");
    println!();
    println!("Options:");
    println!("  --fidelity-only   verifies only the fidelity domain (mandatory phases &");
    println!("                    metric envelopes); unverified performance does not fail.");
    println!();
    println!("Prints the verdict lines (`FIDELITY: OK/FAIL`, `PERFORMANCE: OK/");
    println!("FAIL/NOT_VERIFIED`, `CONTRACT VIOLATED`) and exits 0 only when the active");
    println!("domains pass and no oracle review is pending; exit 1 otherwise.");
}

/// Renders the verify outcome as plain text, mirroring the dashboard verdict
/// strings (`FIDELITY: OK`, `PERFORMANCE: NOT_VERIFIED`, `CONTRACT VIOLATED`,
/// `REVIEW_REQUIRED`) that scripts and operators grep for.
fn render_outcome(
    contract: &QualityContract,
    report: &neural_amp_modeler_rs::testing::qa::verify::VerifyReport,
    outcome: &VerifyOutcome,
    fidelity_only: bool,
) -> String {
    let mut out = String::new();
    out.push_str("QUALITY CONTRACT VERIFICATION\n");
    if fidelity_only {
        out.push_str("  [MODE] FIDELITY-ONLY VERIFICATION (latency checks skipped)\n");
    }
    out.push('\n');

    for phase in &report.phases {
        out.push_str(&format!("  phase {}: {}\n", phase.phase_id, phase.status));
    }
    for phase_id in FIDELITY_PHASES {
        let status = report
            .phases
            .iter()
            .rev()
            .find(|p| p.phase_id == phase_id)
            .map(|p| p.status.as_str())
            .unwrap_or("NOT_RUN");
        if status != "PASS" {
            out.push_str(&format!(
                "  PHASE_FAILED {phase_id}: status={status} (requires PASS)\n"
            ));
        }
    }
    out.push('\n');

    out.push_str(&format!(
        "  FIDELITY — {} model(s) in contract\n",
        contract.fidelity.len()
    ));
    for check in &outcome.fidelity_checks {
        match &check.outcome {
            FidelityOutcome::OptionalSkipped => {
                out.push_str(&format!(
                    "    ok {}: OPTIONAL_SKIPPED (absent from report)\n",
                    check.label
                ));
            }
            FidelityOutcome::MissingLabel => {
                out.push_str(&format!(
                    "    FAIL {}: MISSING_LABEL — mandatory contract entry not found in report\n",
                    check.label
                ));
            }
            FidelityOutcome::Measured(metrics) => {
                for m in metrics {
                    match &m.outcome {
                        MetricOutcome::Ok => {}
                        MetricOutcome::SafetyCeiling {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} above safety ceiling (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::NoiseEnvelope {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} above noise envelope (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::Envelope {
                            current,
                            limit,
                            baseline,
                        } => out.push_str(&format!(
                            "    FAIL {}: {} out of envelope (current={current:.3e}, limit={limit:.3e}, baseline={baseline:.3e})\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                        MetricOutcome::Malformed(reason) => out.push_str(&format!(
                            "    FAIL {}: {} malformed in report ({})\n",
                            check.label,
                            metric_name(m.metric),
                            malformed_reason(reason)
                        )),
                        MetricOutcome::Missing => out.push_str(&format!(
                            "    FAIL {}: {} missing from report\n",
                            check.label,
                            metric_name(m.metric)
                        )),
                    }
                }
            }
        }
    }
    out.push('\n');

    out.push_str(&format!(
        "  PERFORMANCE — {} benchmark(s) in contract\n",
        contract.performance.len()
    ));
    if fidelity_only {
        out.push_str("    [SKIPPED] performance checks skipped in fidelity-only mode\n");
    } else {
        for check in &outcome.perf_checks {
            match &check.result {
                PerfResult::Ok { median_us } => {
                    out.push_str(&format!(
                        "    ok {}: median={median_us:.3} µs\n",
                        check.label
                    ));
                }
                PerfResult::Regressed {
                    median_us,
                    limit_us,
                    baseline_us,
                } => {
                    out.push_str(&format!(
                        "    FAIL {}: latency regressed (current={median_us:.3} µs, limit={limit_us:.3} µs, baseline={baseline_us:.3} µs)\n",
                        check.label
                    ));
                }
                PerfResult::MissingLabel => {
                    out.push_str(&format!(
                        "    FAIL {}: MISSING_LABEL — contract entry not found in report\n",
                        check.label
                    ));
                }
                PerfResult::MissingLatency => {
                    out.push_str(&format!(
                        "    FAIL {}: MISSING_LATENCY — benchmark carried no latency data\n",
                        check.label
                    ));
                }
            }
        }
    }
    out.push('\n');

    match outcome.fidelity {
        FidelityVerdict::Ok => out.push_str("  FIDELITY: OK\n"),
        FidelityVerdict::Fail { violations } => {
            out.push_str(&format!("  FIDELITY: FAIL ({violations} violation(s))\n"));
            if outcome.review_required > 0 {
                out.push_str(&format!(
                    "  [GOVERNANCE] REVIEW_REQUIRED — NAMCore vs f64 divergence detected on {} model(s).\n",
                    outcome.review_required
                ));
            }
        }
    }
    if fidelity_only {
        out.push_str("  PERFORMANCE: NOT_VERIFIED (skipped in fidelity-only mode)\n");
    } else {
        match outcome.performance {
            PerformanceVerdict::Ok => out.push_str("  PERFORMANCE: OK\n"),
            PerformanceVerdict::NotVerified => {
                out.push_str("  PERFORMANCE: NOT_VERIFIED\n");
            }
            PerformanceVerdict::Fail { violations } => {
                out.push_str(&format!(
                    "  PERFORMANCE: FAIL ({violations} violation(s))\n"
                ));
            }
        }
    }

    if outcome.fidelity.is_ok()
        && (outcome.performance.is_ok() || fidelity_only)
        && outcome.review_required > 0
    {
        out.push_str(
            "  CONTRACT UNDER REVIEW — numeric metrics OK, but oracle divergence needs investigation.\n",
        );
        out.push('\n');
    } else {
        let is_violated = if fidelity_only {
            outcome.fidelity_exit_code() != 0
        } else {
            outcome.exit_code() != 0
        };
        if is_violated {
            out.push_str("  CONTRACT VIOLATED\n");
            out.push('\n');
        }
    }
    out
}

fn metric_name(metric: MetricId) -> &'static str {
    match metric {
        MetricId::EsrNamcore => "ESR_NAMCORE",
        MetricId::EsrF64 => "ESR_F64",
        MetricId::SnrDb => "SNR_DB",
        MetricId::Mrstft => "MRSTFT",
    }
}

fn malformed_reason(reason: &MalformedReason) -> String {
    match reason {
        MalformedReason::Missing => "N/A or empty in report".to_string(),
        MalformedReason::NonFinite(raw) => format!("non-finite literal {raw}"),
    }
}

// ── render ──────────────────────────────────────────────────────────────────

fn cmd_render(args: &[String]) {
    let flags = match parse_flags(args, &["report", "ansi", "plain"], &["ansi", "plain"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_render_help();
            exit(0);
        }
        Err(e) => usage_error("render", &e),
    };
    let report_path = match required(&flags, "report") {
        Ok(v) => v,
        Err(e) => usage_error("render", &e),
    };
    let style = match (flags.get("ansi").is_some(), flags.get("plain").is_some()) {
        (true, true) => usage_error("render", "--ansi and --plain are mutually exclusive"),
        (_, true) => RenderStyle::Plain,
        _ => RenderStyle::Ansi,
    };

    let report = match parse_quality_report_file(report_path) {
        Ok(r) => r,
        Err(e) => run_error(format!("cannot parse report {report_path}: {e}")),
    };
    print!("{}", render_quality_report(&report, style));
    exit(0);
}

fn print_render_help() {
    println!("Usage: nam_quality render --report <path> [--ansi | --plain]");
    println!();
    println!("Renders the human-facing quality dashboard from a JSONL report");
    println!("(the output of `nam_quality ingest`). The report is typed and");
    println!("never re-derived — the renderer only formats it.");
    println!();
    println!("  --report <path>  verify report (JSONL) produced by `ingest` — required.");
    println!("  --ansi           ANSI color escape sequences (default).");
    println!("  --plain          plain text, no escape sequences (visual dump only).");
}

// ── classify ────────────────────────────────────────────────────────────────

fn cmd_classify(args: &[String]) {
    let flags = match parse_flags(args, &["status", "reason", "reg-exit", "run-id-match"], &[]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_classify_help();
            exit(0);
        }
        Err(e) => usage_error("classify", &e),
    };
    let status = match required(&flags, "status") {
        Ok(v) => v,
        Err(e) => usage_error("classify", &e),
    };
    let reason = flags.get("reason").unwrap_or("");

    // T2.3: fresh-run classification — `--reg-exit <n>` and
    // `--run-id-match <0|1>` make the classifier stale-receipt-proof. When
    // omitted (legacy callers) the 3-way F-08 classifier is used.
    let reg_exit = flags.get("reg-exit");
    let run_id_match = flags.get("run-id-match");
    let outcome = match (reg_exit, run_id_match) {
        (Some(exit_str), Some(match_str)) => {
            let exit_code: i32 = match exit_str.parse() {
                Ok(n) => n,
                Err(_) => usage_error("classify", "--reg-exit must be an integer"),
            };
            let matches = match match_str {
                "1" | "true" => true,
                "0" | "false" => false,
                _ => usage_error("classify", "--run-id-match must be 0 or 1"),
            };
            classify_fresh_regression(exit_code, status, reason, matches)
        }
        (None, None) => classify_regression_outcome(status, reason),
        _ => usage_error(
            "classify",
            "--reg-exit and --run-id-match must be provided together",
        ),
    };
    println!("{}", outcome.as_str());
    exit(0);
}

fn print_classify_help() {
    println!("Usage: nam_quality classify --status <STATUS> --reason <text>");
    println!("                           [--reg-exit <n> --run-id-match <0|1>]");
    println!();
    println!("Classifies a regression receipt into the single performance status");
    println!("(`qa::classify`, F-08): PASS / NOT_VERIFIED / FAIL.");
    println!();
    println!("  --status <STATUS>  receipt status (e.g. PASS, FAIL, NOT_RUN).");
    println!("  --reason <text>    receipt reason (e.g. MISSING_BASELINE,");
    println!("                      INCOMPARABLE_ENVIRONMENT, REGRESSION_DETECTED).");
    println!("  --reg-exit <n>     exit code of the regression runner (T2.3).");
    println!("  --run-id-match     whether the receipt's run_id equals the current");
    println!("                     RUN_ID (0|1). Together they make the classifier");
    println!("                     stale-receipt-proof: a PASS from a previous run");
    println!("                     never validates a failed/aborted current run.");
}

// ── receipt append ──────────────────────────────────────────────────────────

/// Phase receipt record — serde serializes fields in declaration order, which
/// reproduces the bash `printf` schema of `dashboard_phase_receipt`
/// (`utils/_lib.sh:96`): `phase_id, status, exit_code, observed_records,
/// expected_records, reason, run_id`.
#[derive(Debug, Serialize)]
struct PhaseReceiptRecord<'a> {
    phase_id: &'a str,
    status: &'a str,
    exit_code: u32,
    observed_records: u32,
    expected_records: u32,
    reason: &'a str,
    run_id: &'a str,
}

fn cmd_receipt(args: &[String]) {
    if args.is_empty() {
        usage_error("receipt", "missing subcommand (append)");
    }
    match args[0].as_str() {
        "append" => cmd_receipt_append(&args[1..]),
        "-h" | "--help" => {
            print_receipt_help();
            exit(0);
        }
        other => usage_error("receipt", &format!("unknown subcommand: {other}")),
    }
}

fn print_receipt_help() {
    println!("Usage: nam_quality receipt append --phase-id <id> --status <STATUS>");
    println!("                                    --out <path>");
    println!("                                    [--exit-code <n>] [--observed-records <n>]");
    println!("                                    [--expected-records <n>] [--reason <text>]");
    println!("                                    [--run-id <id>]");
    println!();
    println!("Appends one phase receipt record (serde JSONL) to the dashboard phase");
    println!("receipt — the replacement for the bash `printf '{{'` of");
    println!(
        "`dashboard_phase_receipt`. Status values: {}",
        RECEIPT_STATUSES.join(", ")
    );
}

fn cmd_receipt_append(args: &[String]) {
    let flags = match parse_flags(
        args,
        &[
            "phase-id",
            "status",
            "exit-code",
            "observed-records",
            "expected-records",
            "reason",
            "run-id",
            "out",
        ],
        &[],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_receipt_help();
            exit(0);
        }
        Err(e) => usage_error("receipt append", &e),
    };
    let phase_id = match required(&flags, "phase-id") {
        Ok(v) => v.to_string(),
        Err(e) => usage_error("receipt append", &e),
    };
    let status = match required(&flags, "status") {
        Ok(v) => v.to_string(),
        Err(e) => usage_error("receipt append", &e),
    };
    if !RECEIPT_STATUSES.contains(&status.as_str()) {
        usage_error(
            "receipt append",
            &format!(
                "invalid status '{status}' (accepted: {})",
                RECEIPT_STATUSES.join(", ")
            ),
        );
    }
    let out = match required(&flags, "out") {
        Ok(v) => v.to_string(),
        Err(e) => usage_error("receipt append", &e),
    };
    let exit_code = parse_u32(&flags, "exit-code", "receipt append");
    let observed_records = parse_u32(&flags, "observed-records", "receipt append");
    let expected_records = parse_u32(&flags, "expected-records", "receipt append");
    let reason = flags.get("reason").unwrap_or("").to_string();
    let run_id = flags.get("run-id").unwrap_or("").to_string();

    let record = PhaseReceiptRecord {
        phase_id: &phase_id,
        status: &status,
        exit_code,
        observed_records,
        expected_records,
        reason: &reason,
        run_id: &run_id,
    };
    let line = match serde_json::to_string(&record) {
        Ok(l) => l,
        Err(e) => run_error(format!("cannot serialize phase receipt record: {e}")),
    };
    if let Err(e) = append_line(&out, &line) {
        run_error(format!("cannot append phase receipt to {out}: {e}"));
    }
    eprintln!("phase receipt: {phase_id} {status} appended to {out}");
    exit(0);
}

fn append_line(out: &str, line: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")
}

// ── save ────────────────────────────────────────────────────────────────────

fn cmd_save(args: &[String]) {
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

    // Validate the contract payload before touching anything (fail-closed:
    // a corrupt contract must never be promoted).
    let contract = match fs::read_to_string(contract_path)
        .map_err(|e| format!("cannot read contract {contract_path}: {e}"))
        .and_then(|s| QualityContract::from_json_str(&s).map_err(|e| e.to_string()))
    {
        Ok(c) => c,
        Err(e) => run_error(e),
    };

    // Fidelity gate: any phase FAIL outside the performance domain refuses
    // the save (mirror of the bash `--save` transactional gate; performance
    // NOT_VERIFIED/FAIL never blocks — PERF-006, dashboard :2599-2602).
    let receipt_lines = match read_jsonl_lines(receipt_path) {
        Ok(lines) => lines,
        Err(e) => run_error(format!("cannot read phase receipt {receipt_path}: {e}")),
    };
    for (i, line) in receipt_lines.iter().enumerate() {
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(source) => run_error(format!(
                "malformed phase receipt line {} in {receipt_path}: {source}",
                i + 1
            )),
        };
        // Provenance records (`kind: build_metadata`, T0.4) carry no
        // `phase_id` and are irrelevant to the fidelity gate.
        let Some(phase_id) = value.get("phase_id").and_then(Value::as_str) else {
            continue;
        };
        let status = value.get("status").and_then(Value::as_str).unwrap_or("");
        if status == "FAIL" && phase_id != "regression_gate" {
            run_error(format!(
                "contract NOT saved: fidelity phase '{phase_id}' FAILed (--save requires all fidelity phases to pass)"
            ));
        }
    }

    let rendered = match contract.to_json_pretty() {
        Ok(s) => s + "\n",
        Err(e) => run_error(format!("cannot serialize contract: {e}")),
    };
    if let Err(e) = atomic_write(Path::new(contract_path), &rendered) {
        run_error(format!("contract save failed: {e}"));
    }
    eprintln!("contract saved: {contract_path} (atomic, fidelity phases verified)");
    exit(0);
}

fn print_save_help() {
    println!("Usage: nam_quality save --contract <path> --receipt <path>");
    println!();
    println!("Transactionally promotes the validated contract JSON at <path>");
    println!("(temp file in the same directory + atomic rename). Refuses with exit 1");
    println!("when any fidelity phase (golden_vectors, reference_oracle_f64,");
    println!("quick_parity — or any phase other than regression_gate) FAILed in the");
    println!("phase receipt. Performance NOT_VERIFIED never blocks (PERF-006).");
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
fn write_new_file(path: &str, content: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, content)
}
