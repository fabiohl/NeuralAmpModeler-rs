// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance-gate binary (Sprint S3, R-03 / R-09 / F-24) — the Rust engine
//! behind `utils/tests-performance-regression.sh`.
//!
//! All performance-governance JSON lives here and in the `qa` modules
//! (`fingerprint`, `coverage`, `baseline_store`, `classify`); the shell
//! wrapper only orchestrates `taskset`/`cargo bench` and calls this binary.
//!
//! Subcommands:
//! - `probe`: write the current environment fingerprint (serde JSON).
//! - `compare`: compare the current environment against the stored baseline
//!   fingerprint — `MISSING_BASELINE` / `INCOMPARABLE_ENVIRONMENT` fail with
//!   exit 1.
//! - `coverage`: fail-closed baseline coverage cross-check (F-24): every
//!   `Benchmarking <id>:` line of the Criterion log must have a persisted
//!   `…/<id>/<baseline>/` series.
//! - `persist-baseline`: replace-copy `target/criterion/**/<baseline>` into
//!   `.performance-baselines/` (nested sanitized).
//! - `restore-baseline`: replace-copy the store back into the criterion root.
//! - `receipt append`: append one phase receipt record (serde JSONL,
//!   `dashboard_phase_receipt` schema).
//! - `receipt summary`: derive and append the `overall` verdict line.
//!
//! Exit codes:
//! - 0: success;
//! - 1: run-time failure (missing/incomparable baseline, coverage gap, I/O);
//! - 2: usage error (unknown subcommand/flag, malformed argument).

use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::exit;

use serde::Serialize;
use serde_json::Value;

use neural_amp_modeler_rs::testing::qa::baseline_store::{
    DEFAULT_BASELINE_NAME, persist_baseline, restore_baseline,
};
use neural_amp_modeler_rs::testing::qa::coverage::{
    BaselineCoverageGap, executed_bench_ids, missing_baseline_coverage,
};
use neural_amp_modeler_rs::testing::qa::fingerprint::Fingerprint;
use neural_amp_modeler_rs::testing::qa::fingerprint::FingerprintError;

/// Store directory of persisted baselines (gitignored).
const DEFAULT_BASELINE_DIR: &str = ".performance-baselines";
/// Criterion transient working area.
const DEFAULT_CRITERION_ROOT: &str = "target/criterion";
/// Regression phase receipt (bash `REGRESSION_RECEIPT`).
const DEFAULT_RECEIPT: &str = "target/logs/regression_phase_receipt.jsonl";

/// Status values accepted by `receipt append` — the documented receipt
/// schema (`utils/_lib.sh:80`) plus `NOT_VERIFIED` (dashboard writes it for
/// `regression_gate`); mirrors `nam_quality`.
const RECEIPT_STATUSES: [&str; 6] = [
    "PASS",
    "FAIL",
    "SKIP_CAPABILITY",
    "SKIP_OPTIONAL_FIXTURE",
    "NOT_RUN",
    "NOT_VERIFIED",
];

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
        "probe" => cmd_probe(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "coverage" => cmd_coverage(&args[1..]),
        "persist-baseline" => cmd_persist_baseline(&args[1..]),
        "restore-baseline" => cmd_restore_baseline(&args[1..]),
        "receipt" => cmd_receipt(&args[1..]),
        other => {
            eprintln!("unknown subcommand: {other}");
            print_help();
            exit(2);
        }
    }
}

fn print_help() {
    println!("NeuralAmpModeler-rs Performance Gate (Sprint S3, R-03 / R-09 / F-24)");
    println!();
    println!("Rust engine of the performance regression gate: fingerprint, coverage");
    println!("and baseline persistence are serde JSON — the shell wrapper only");
    println!("orchestrates taskset/cargo bench and calls this binary.");
    println!();
    println!("Usage:");
    println!("  nam_perf_gate probe [--out <path>] [--bench-core <core>]");
    println!("                      [--baseline-dir <dir>]");
    println!("  nam_perf_gate compare [--baseline <path>] [--bench-core <core>]");
    println!("                        [--baseline-dir <dir>]");
    println!("  nam_perf_gate coverage --log <path> [--root <dir>] [--baseline <name>]");
    println!("  nam_perf_gate persist-baseline [--baseline-dir <dir>] [--criterion-root <dir>]");
    println!("                                 [--baseline <name>]");
    println!("  nam_perf_gate restore-baseline [--baseline-dir <dir>] [--criterion-root <dir>]");
    println!("                                 [--baseline <name>]");
    println!("  nam_perf_gate receipt append --phase-id <id> --status <STATUS> [--out <path>]");
    println!("                               [--exit-code <n>] [--observed-records <n>]");
    println!("                               [--expected-records <n>] [--reason <text>]");
    println!("                               [--run-id <id>]");
    println!("  nam_perf_gate receipt summary [--out <path>]");
    println!("  nam_perf_gate --help");
    println!();
    println!("Defaults: baseline dir .performance-baselines/, criterion root");
    println!("target/criterion/, baseline name ci-baseline (NAM_BASELINE_NAME),");
    println!("bench core NAM_BENCH_CORE (empty = unpinned), receipt");
    println!("target/logs/regression_phase_receipt.jsonl.");
    println!();
    println!("Exit codes: 0 success, 1 run-time failure (MISSING_BASELINE,");
    println!("INCOMPARABLE_ENVIRONMENT, BASELINE_COVERAGE_GAP, I/O), 2 usage error.");
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

fn parse_flags(args: &[String], allowed: &[&str]) -> Result<Flags, String> {
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
        let value = iter
            .next()
            .ok_or_else(|| format!("--{name} requires a value"))?;
        map.insert(name.to_string(), value.clone());
    }
    Ok(Flags { map })
}

fn usage_error(context: &str, message: &str) -> ! {
    eprintln!("{context}: {message}");
    eprintln!("Run `nam_perf_gate --help` for usage.");
    exit(2);
}

fn run_error(message: String) -> ! {
    eprintln!("ERROR: {message}");
    exit(1);
}

fn required<'a>(flags: &'a Flags, name: &str) -> Result<&'a str, String> {
    flags
        .get(name)
        .ok_or_else(|| format!("missing required flag --{name}"))
}

fn parse_u32(flags: &Flags, name: &str, context: &str) -> u32 {
    match flags.get(name) {
        None => 0,
        Some(v) => match v.parse() {
            Ok(n) => n,
            Err(_) => usage_error(context, &format!("--{name} must be a non-negative integer")),
        },
    }
}

/// Bench core of the current invocation: `--bench-core` wins, then
/// `NAM_BENCH_CORE`, then unpinned (empty).
fn bench_core(flags: &Flags) -> String {
    match flags.get("bench-core") {
        Some(v) => v.to_string(),
        None => env::var("NAM_BENCH_CORE").unwrap_or_default(),
    }
}

fn baseline_name(flags: &Flags) -> String {
    match flags.get("baseline") {
        Some(v) => v.to_string(),
        None => env::var("NAM_BASELINE_NAME").unwrap_or_else(|_| DEFAULT_BASELINE_NAME.to_string()),
    }
}

// ── probe ────────────────────────────────────────────────────────────────────

fn cmd_probe(args: &[String]) {
    let flags = match parse_flags(args, &["out", "bench-core", "baseline-dir"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("probe", &e),
    };
    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let out = flags
        .get("out")
        .map(str::to_string)
        .unwrap_or_else(|| format!("{baseline_dir}/baseline-fingerprint.json"));

    let fingerprint = Fingerprint::probe(&bench_core(&flags));
    if let Err(e) = fingerprint.write_to_path(Path::new(&out)) {
        run_error(format!("cannot write fingerprint to {out}: {e}"));
    }
    match fingerprint.to_json_pretty() {
        Ok(json) => println!("{json}"),
        Err(e) => run_error(format!("cannot serialize fingerprint: {e}")),
    }
    eprintln!("fingerprint written to {out}");
    exit(0);
}

// ── compare ──────────────────────────────────────────────────────────────────

fn cmd_compare(args: &[String]) {
    let flags = match parse_flags(args, &["baseline", "bench-core", "baseline-dir"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("compare", &e),
    };
    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let baseline_path = flags
        .get("baseline")
        .map(str::to_string)
        .unwrap_or_else(|| format!("{baseline_dir}/baseline-fingerprint.json"));

    let current = Fingerprint::probe(&bench_core(&flags));
    let baseline = match Fingerprint::read_from_path(Path::new(&baseline_path)) {
        Ok(b) => b,
        Err(FingerprintError::MissingBaseline) => {
            println!("MISSING_BASELINE no baseline fingerprint at {baseline_path}");
            println!("Re-bootstrap the baseline with `--bootstrap-baseline` (human operation).");
            exit(1);
        }
        Err(e) => run_error(format!(
            "cannot read baseline fingerprint {baseline_path}: {e}"
        )),
    };
    match current.compare(&baseline) {
        Ok(()) => {
            println!("environment comparable: fingerprint matches {baseline_path}");
            exit(0);
        }
        Err(e) => {
            println!("{e}");
            println!("Performance comparison aborted: environment incompatible with baseline.");
            exit(1);
        }
    }
}

// ── coverage ─────────────────────────────────────────────────────────────────

fn cmd_coverage(args: &[String]) {
    let flags = match parse_flags(args, &["log", "root", "baseline"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("coverage", &e),
    };
    let log_path = match required(&flags, "log") {
        Ok(v) => v,
        Err(e) => usage_error("coverage", &e),
    };
    let criterion_root = flags
        .get("root")
        .unwrap_or(DEFAULT_CRITERION_ROOT)
        .to_string();
    let name = baseline_name(&flags);

    // An absent/unreadable log maps to empty text — the same blind gate as
    // the bash `sed` on a missing file (fail-closed, F-24).
    let log_text = fs::read_to_string(log_path).unwrap_or_default();
    match missing_baseline_coverage(&log_text, Path::new(&criterion_root), &name) {
        Err(BaselineCoverageGap) => {
            println!("BASELINE_COVERAGE_GAP no executed benchmark could be parsed from {log_path}");
            println!("The coverage cross-check is blind: nothing passes unverified.");
            exit(1);
        }
        Ok(missing) if !missing.is_empty() => {
            println!(
                "BASELINE_COVERAGE_GAP executed benchmark(s) without a saved baseline series:"
            );
            for id in &missing {
                println!("  {id}");
            }
            println!("A human operator must re-bootstrap the baseline.");
            exit(1);
        }
        Ok(_) => {
            let executed = executed_bench_ids(&log_text).len();
            println!(
                "coverage ok: all {executed} executed benchmark(s) have baseline series \
                 under {criterion_root}/<id>/{name}/"
            );
            exit(0);
        }
    }
}

// ── baseline persistence ─────────────────────────────────────────────────────

fn cmd_persist_baseline(args: &[String]) {
    let flags = match parse_flags(args, &["baseline-dir", "criterion-root", "baseline"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("persist-baseline", &e),
    };
    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let criterion_root = flags
        .get("criterion-root")
        .unwrap_or(DEFAULT_CRITERION_ROOT)
        .to_string();
    let name = baseline_name(&flags);

    match persist_baseline(Path::new(&baseline_dir), Path::new(&criterion_root), &name) {
        Ok(count) => {
            println!("persisted {count} baseline series from {criterion_root} to {baseline_dir}");
            exit(0);
        }
        Err(e) => run_error(format!("cannot persist baselines: {e}")),
    }
}

fn cmd_restore_baseline(args: &[String]) {
    let flags = match parse_flags(args, &["baseline-dir", "criterion-root", "baseline"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("restore-baseline", &e),
    };
    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let criterion_root = flags
        .get("criterion-root")
        .unwrap_or(DEFAULT_CRITERION_ROOT)
        .to_string();
    let name = baseline_name(&flags);

    match restore_baseline(Path::new(&baseline_dir), Path::new(&criterion_root), &name) {
        Ok(0) => {
            println!("no top-level '{name}' series found under {baseline_dir}");
            exit(0);
        }
        Ok(count) => {
            println!("restored {count} baseline series into {criterion_root}");
            exit(0);
        }
        Err(e) => run_error(format!("cannot restore baselines: {e}")),
    }
}

// ── receipt ──────────────────────────────────────────────────────────────────

/// Phase receipt record — serde serializes fields in declaration order, which
/// reproduces the bash `printf` schema of `dashboard_phase_receipt`
/// (`utils/_lib.sh:96`): `phase_id, status, exit_code, observed_records,
/// expected_records, reason, run_id` (byte-compatible with `nam_quality`).
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
        usage_error("receipt", "missing subcommand (append|summary)");
    }
    match args[0].as_str() {
        "append" => cmd_receipt_append(&args[1..]),
        "summary" => cmd_receipt_summary(&args[1..]),
        "-h" | "--help" => {
            print_receipt_help();
            exit(0);
        }
        other => usage_error("receipt", &format!("unknown subcommand: {other}")),
    }
}

fn print_receipt_help() {
    println!("Usage: nam_perf_gate receipt append --phase-id <id> --status <STATUS>");
    println!("                                    [--out <path>] [--exit-code <n>]");
    println!("                                    [--observed-records <n>]");
    println!("                                    [--expected-records <n>] [--reason <text>]");
    println!("                                    [--run-id <id>]");
    println!("       nam_perf_gate receipt summary [--out <path>]");
    println!();
    println!("Appends one phase receipt record (serde JSONL) to the regression phase");
    println!("receipt — the `dashboard_phase_receipt` schema, byte-compatible with");
    println!(
        "`nam_quality receipt append`. Status values: {}",
        RECEIPT_STATUSES.join(", ")
    );
    println!("summary derives the `overall` verdict (PASS iff every phase PASS).");
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
    let out = flags.get("out").unwrap_or(DEFAULT_RECEIPT).to_string();
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

fn cmd_receipt_summary(args: &[String]) {
    let flags = match parse_flags(args, &["out"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_receipt_help();
            exit(0);
        }
        Err(e) => usage_error("receipt summary", &e),
    };
    let out = flags.get("out").unwrap_or(DEFAULT_RECEIPT).to_string();

    let content = match fs::read_to_string(&out) {
        Ok(c) => c,
        Err(e) => run_error(format!("cannot read receipt {out}: {e}")),
    };
    let mut phase_count = 0usize;
    let mut pass_count = 0usize;
    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(source) => run_error(format!(
                "invalid receipt line {} in {out}: {source}",
                idx + 1
            )),
        };
        // Provenance records (`kind: build_metadata`, T0.4) have no
        // `phase_id` and are not phases.
        if value.get("phase_id").is_none() {
            continue;
        }
        phase_count += 1;
        if value.get("status").and_then(Value::as_str) == Some("PASS") {
            pass_count += 1;
        }
    }
    if phase_count == 0 {
        run_error(format!("no phase entries in {out} — nothing to summarize"));
    }
    let overall_status = if pass_count == phase_count {
        "PASS"
    } else {
        "FAIL"
    };
    let overall = PhaseReceiptRecord {
        phase_id: "overall",
        status: overall_status,
        exit_code: if overall_status == "PASS" { 0 } else { 1 },
        observed_records: phase_count as u32,
        expected_records: phase_count as u32,
        reason: "",
        run_id: "",
    };
    let line = match serde_json::to_string(&overall) {
        Ok(l) => l,
        Err(e) => run_error(format!("cannot serialize overall receipt: {e}")),
    };
    if let Err(e) = append_line(&out, &line) {
        run_error(format!("cannot append overall receipt to {out}: {e}"));
    }
    eprintln!(
        "regression receipt: overall {overall_status} ({pass_count}/{phase_count} phases PASS) -> {out}"
    );
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
