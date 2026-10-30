// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance-gate binary (R-03 / R-09 / F-24) — the Rust engine
//! behind the removed performance-gate script.
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
//! - `expected-count`: print the expected benchmark count for a suite.
//! - `verdict`: the machine regression verdict (R-3/B2) — parses Criterion's
//!   persisted `…/<id>/change/estimates.json` comparison artifacts, never the
//!   human console wording. `REGRESSION_DETECTED` / `REGRESSION_BLIND` fail
//!   with exit 1.
//! - `persist-baseline`: replace-copy `target/criterion/**/<baseline>` into
//!   `.performance-baselines/` (nested sanitized).
//! - `restore-baseline`: replace-copy the store back into the criterion root.
//! - `receipt append`: append one phase receipt record (serde JSONL,
//!   `dashboard_phase_receipt` schema).
//! - `receipt summary`: derive and append the `overall` verdict line.
//!
//! Exit codes:
//! - 0: success;
//! - 1: run-time failure (missing/incomparable baseline, coverage gap,
//!   machine regression / blind verdict, I/O);
//! - 2: usage error (unknown subcommand/flag, malformed argument);
//! - 3: `pick-core --require-isolated` found no online isolated CPU (`GAP`),
//!   or `run --check` found no performance baseline outside `--strict`
//!   (`GAP perf-baseline-missing`);
//! - 4: `run --check --strict` found no performance baseline (`FAIL`, typed
//!   reason `MISSING_BASELINE`).

use std::collections::HashMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;

use neural_amp_modeler_rs::testing::qa::baseline_store::{
    DEFAULT_BASELINE_NAME, persist_baseline, restore_baseline,
};
use neural_amp_modeler_rs::testing::qa::coverage::{
    BaselineCoverageGap, executed_bench_ids, expected_bench_ids_for_suite,
    missing_baseline_coverage, missing_expected_benchmarks,
};
use neural_amp_modeler_rs::testing::qa::fingerprint::Fingerprint;
use neural_amp_modeler_rs::testing::qa::fingerprint::FingerprintError;
use neural_amp_modeler_rs::testing::qa::pick_core::{
    DEFAULT_SYSFS_CPU_DIR, is_cpu_online, parse_cpu_list, resolve_bench_core,
};
use neural_amp_modeler_rs::testing::qa::verdict::{
    DEFAULT_NOISE_THRESHOLD, VerdictBlind, VerdictFinding, VerdictOptions,
    regression_verdict_with_options,
};

/// Store directory of persisted baselines (gitignored).
const DEFAULT_BASELINE_DIR: &str = ".performance-baselines";
/// Criterion transient working area.
const DEFAULT_CRITERION_ROOT: &str = "target/criterion";

fn default_criterion_root() -> String {
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        let trimmed = target_dir.trim();
        if !trimmed.is_empty() {
            return format!("{trimmed}/criterion");
        }
    }
    DEFAULT_CRITERION_ROOT.to_string()
}
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
        "pick-core" => cmd_pick_core(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "probe" => cmd_probe(&args[1..]),
        "compare" => cmd_compare(&args[1..]),
        "coverage" => cmd_coverage(&args[1..]),
        "expected-count" => cmd_expected_count(&args[1..]),
        "verdict" => cmd_verdict(&args[1..]),
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
    println!("NeuralAmpModeler-rs Performance Gate");
    println!();
    println!("Rust engine of the performance regression gate: fingerprint, coverage");
    println!("and baseline persistence are serde JSON — the shell wrapper only");
    println!("orchestrates taskset/cargo bench and calls this binary.");
    println!();
    println!("Usage:");
    println!(
        "  nam_perf_gate pick-core [--bench-core <core>] [--sysfs <dir>] [--strict] [--require-isolated]"
    );
    println!("  nam_perf_gate run [--check | --bootstrap-baseline] [--baseline <name>]");
    println!("                    [--baseline-dir <dir>] [--criterion-root <dir>]");
    println!("                    [--bench-core <core>] [--sysfs <dir>] [--suite <name>]");
    println!("                    [--manifest <path>] [--cooldown <seconds>]");
    println!("                    [--receipt <path>] [--log <path>] [--run-id <id>]");
    println!("                    [--noise <fraction>] [--abs-noise-floor-ns <nanoseconds>]");
    println!("                    [--check-stability] [--strict-stability]");
    println!("                    [--confirm-regressions] [--bench-exe <path>]");
    println!("                    [--strict]");
    println!("  nam_perf_gate probe [--out <path>] [--bench-core <core>]");
    println!("                      [--baseline-dir <dir>]");
    println!("  nam_perf_gate compare [--baseline <path>] [--bench-core <core>]");
    println!("                        [--baseline-dir <dir>]");
    println!("  nam_perf_gate coverage --log <path> [--root <dir>] [--baseline <name>]");
    println!("                         [--suite <name>] [--manifest <path>]");
    println!("  nam_perf_gate expected-count [--suite <name>] [--manifest <path>]");
    println!("  nam_perf_gate verdict --log <path> [--root <dir>] [--noise <fraction>]");
    println!("                        [--suite <name>] [--manifest <path>]");
    println!("                        [--check-stability] [--strict-stability]");
    println!("                        [--abs-noise-floor-ns <nanoseconds>]");
    println!("                        [--confirm-regressions] [--json]");
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
    println!("A --manifest file lists expected Criterion ids (one per line, blank");
    println!("lines and '#' comments ignored); a trailing 'Group/*' entry matches any");
    println!("executed id with that prefix, so a downstream suite declares group");
    println!("coverage without enumerating parametric ids. A group entry certifies that");
    println!("the group ran, not exhaustive per-id coverage — use exact ids when");
    println!("completeness matters. A non-regression_gate suite without --manifest fails");
    println!("closed.");
    println!();
    println!("Defaults: baseline dir .performance-baselines/, criterion root");
    println!("target/criterion/, baseline name ci-baseline (NAM_BASELINE_NAME),");
    println!("bench core NAM_BENCH_CORE (empty = unpinned), verdict noise band");
    println!("0.05, receipt target/logs/regression_phase_receipt.jsonl.");
    println!();
    println!("Exit codes: 0 success, 1 run-time failure / regression detected,");
    println!("2 usage error or unstable baseline (with --strict-stability),");
    println!("3 GAP (blind verdict, no isolated CPU, or missing baseline without");
    println!("--strict), 4 FAIL with reason MISSING_BASELINE (run --check --strict).");
}

// ── Strict flag parsing (fail-closed: unknown flag ⇒ exit 2) ─────────────────

struct Flags {
    map: HashMap<String, String>,
}

impl Flags {
    fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    fn get_bool(&self, name: &str) -> bool {
        self.map.get(name).map(|v| v == "true").unwrap_or(false)
    }
}

fn parse_flags(args: &[String], allowed: &[&str]) -> Result<Flags, String> {
    parse_flags_full(args, allowed, &[])
}

fn parse_flags_full(
    args: &[String],
    allowed_values: &[&str],
    allowed_bools: &[&str],
) -> Result<Flags, String> {
    let mut map = HashMap::new();
    let mut iter = args.iter().peekable();
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
        if allowed_bools.contains(&name) {
            let explicit_val = match iter.peek() {
                Some(arg) if *arg == "true" || *arg == "false" => Some((*arg).clone()),
                _ => None,
            };
            if let Some(val) = explicit_val {
                let _ = iter.next();
                map.insert(name.to_string(), val);
                continue;
            }
            if name == "check"
                && let Some(next) = iter.peek()
                && !next.starts_with("--")
            {
                let val = (*next).clone();
                let _ = iter.next();
                map.insert("check".to_string(), "true".to_string());
                map.insert("contract".to_string(), val);
                continue;
            }
            map.insert(name.to_string(), "true".to_string());
        } else if allowed_values.contains(&name) {
            let value = iter
                .next()
                .ok_or_else(|| format!("--{name} requires a value"))?;
            map.insert(name.to_string(), value.clone());
        } else {
            return Err(format!("unknown flag: --{name}"));
        }
    }
    Ok(Flags { map })
}

fn parse_f64(flags: &Flags, name: &str, context: &str) -> Option<f64> {
    match flags.get(name) {
        None => None,
        Some(v) => match v.trim().parse::<f64>() {
            Ok(n) if n.is_finite() && n >= 0.0 => Some(n),
            _ => usage_error(
                context,
                &format!("--{name} must be a non-negative number (got '{v}')"),
            ),
        },
    }
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

/// Parses a non-negative integer flag value.
///
/// Values produced by shell `grep -c` pipelines may carry multiple
/// lines. The FIRST non-empty, trimmed line must be the plain count — the
/// embedded newline is tolerated instead of failing the whole gate. A
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

/// Parses the verdict noise-band flag (a positive fraction, e.g. `0.05`),
/// defaulting to [`DEFAULT_NOISE_THRESHOLD`] when absent.
fn parse_noise(flags: &Flags, name: &str, context: &str) -> f64 {
    match flags.get(name) {
        None => DEFAULT_NOISE_THRESHOLD,
        Some(v) => match v.trim().parse::<f64>() {
            Ok(n) if n.is_finite() && n > 0.0 => n,
            _ => usage_error(
                context,
                &format!("--{name} must be a positive fraction (got '{v}')"),
            ),
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
    let flags = match parse_flags(args, &["log", "root", "baseline", "suite", "manifest"]) {
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
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let name = baseline_name(&flags);

    // An absent/unreadable log maps to empty text — the same blind gate as
    // the bash `sed` on a missing file (fail-closed, F-24).
    let log_text = fs::read_to_string(log_path).unwrap_or_default();

    if let Some(suite) = flags.get("suite") {
        let manifest_path = flags.get("manifest").map(Path::new);
        let expected_ids = match expected_bench_ids_for_suite(suite, manifest_path) {
            Ok(ids) => ids,
            Err(e) => run_error(e),
        };
        if suite != "regression_gate" {
            println!("(alternative scope: {suite})");
        }
        match missing_expected_benchmarks(&log_text, &expected_ids) {
            Err(BaselineCoverageGap) => {
                println!(
                    "BASELINE_COVERAGE_GAP no executed benchmark could be parsed from {log_path}"
                );
                println!("The coverage cross-check is blind: nothing passes unverified.");
                exit(1);
            }
            Ok(missing) if !missing.is_empty() => {
                println!(
                    "BASELINE_COVERAGE_GAP expected benchmark(s) were not executed in {log_path}:"
                );
                for id in &missing {
                    println!("  {id}");
                }
                exit(1);
            }
            Ok(_) => {}
        }
    }

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

// ── expected-count ───────────────────────────────────────────────────────────

fn cmd_expected_count(args: &[String]) {
    let flags = match parse_flags(args, &["suite", "manifest"]) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("expected-count", &e),
    };
    let suite = flags.get("suite").unwrap_or("regression_gate");
    let manifest_path = flags.get("manifest").map(Path::new);
    match expected_bench_ids_for_suite(suite, manifest_path) {
        Ok(ids) => {
            println!("{}", ids.len());
            exit(0);
        }
        Err(e) => run_error(e),
    }
}

// ── verdict ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct VerdictJsonReport {
    schema_version: &'static str,
    executed_count: usize,
    noise_band_relative: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    abs_noise_floor_ns: Option<f64>,
    verdict: &'static str,
    summary: VerdictJsonSummary,
    findings: Vec<VerdictJsonFinding>,
}

#[derive(Debug, Serialize)]
struct VerdictJsonSummary {
    regressions: usize,
    unstable_baselines: usize,
    sub_threshold_noise: usize,
    clean: usize,
}

#[derive(Debug, Serialize)]
struct VerdictJsonFinding {
    id: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ci_95: Option<[f64; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta_ns: Option<f64>,
    reason: String,
}

/// Prints the machine regression verdict over Criterion's persisted
/// `change/estimates.json` artifacts (R-3/B2).
///
/// Exit codes (Section 4):
/// - 0: SUCCESS (no regressions, sub-threshold noise ignored, clean run);
/// - 1: REGRESSION_DETECTED (at least one machine regression above noise & floor);
/// - 2: UNSTABLE_BASELINE (mean improvement exceeds stability band under --strict-stability);
/// - 3: REGRESSION_BLIND (missing/unparseable log or missing Criterion comparison estimates).
fn cmd_verdict(args: &[String]) {
    let flags = match parse_flags_full(
        args,
        &[
            "log",
            "root",
            "noise",
            "suite",
            "manifest",
            "abs-noise-floor-ns",
        ],
        &[
            "check-stability",
            "strict-stability",
            "confirm-regressions",
            "json",
        ],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_help();
            exit(0);
        }
        Err(e) => usage_error("verdict", &e),
    };
    let log_path = match required(&flags, "log") {
        Ok(v) => v,
        Err(e) => usage_error("verdict", &e),
    };
    let criterion_root = flags
        .get("root")
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let noise = parse_noise(&flags, "noise", "verdict");
    let strict_stability = flags.get_bool("strict-stability");
    let check_stability = strict_stability || flags.get_bool("check-stability");
    let abs_noise_floor_ns = parse_f64(&flags, "abs-noise-floor-ns", "verdict");
    let confirm_regressions = flags.get_bool("confirm-regressions");
    let json_output = flags.get_bool("json");

    // An absent/unreadable log maps to empty text — the same blind gate as
    // the bash `grep` on a missing file (fail-closed, R-3).
    let log_text = fs::read_to_string(log_path).unwrap_or_default();

    let expected_ids = if let Some(suite) = flags.get("suite") {
        let manifest_path = flags.get("manifest").map(Path::new);
        let ids = match expected_bench_ids_for_suite(suite, manifest_path) {
            Ok(ids) => ids,
            Err(e) => run_error(e),
        };
        if suite != "regression_gate" && !json_output {
            println!("(alternative scope: {suite})");
        }
        Some(ids)
    } else {
        None
    };

    let options = VerdictOptions {
        noise,
        expected_ids,
        check_stability,
        abs_noise_floor_ns,
    };

    let outcome_result =
        regression_verdict_with_options(&log_text, Path::new(&criterion_root), &options);

    let outcome = match outcome_result {
        Err(VerdictBlind) => {
            if json_output {
                let report = VerdictJsonReport {
                    schema_version: "1.1.0",
                    executed_count: 0,
                    noise_band_relative: noise,
                    abs_noise_floor_ns,
                    verdict: "FAILED",
                    summary: VerdictJsonSummary {
                        regressions: 0,
                        unstable_baselines: 0,
                        sub_threshold_noise: 0,
                        clean: 0,
                    },
                    findings: vec![VerdictJsonFinding {
                        id: String::new(),
                        status: "REGRESSION_BLIND",
                        relative_mean: None,
                        ci_95: None,
                        delta_ns: None,
                        reason: format!("no executed benchmark could be parsed from {log_path}"),
                    }],
                };
                if let Ok(json) = serde_json::to_string_pretty(&report) {
                    println!("{json}");
                }
            } else {
                println!("REGRESSION_BLIND no executed benchmark could be parsed from {log_path}");
                println!("The machine regression verdict is blind: nothing passes unverified.");
            }
            exit(3);
        }
        Ok(o) => o,
    };

    let flagged_file = Path::new(&criterion_root).join("flagged_regressions.txt");
    let mut flakes_cleared = Vec::new();
    if confirm_regressions {
        if flagged_file.exists() {
            let previously_flagged: std::collections::HashSet<String> =
                fs::read_to_string(&flagged_file)
                    .unwrap_or_default()
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect();
            let _ = fs::remove_file(&flagged_file);

            let current_reg_ids: std::collections::HashSet<String> =
                outcome.regressions().map(|f| f.id().to_string()).collect();
            for prev_id in &previously_flagged {
                if !current_reg_ids.contains(prev_id) {
                    flakes_cleared.push(prev_id.clone());
                }
            }
        } else {
            let reg_ids: Vec<&str> = outcome.regressions().map(|f| f.id()).collect();
            if !reg_ids.is_empty() {
                let text = reg_ids.join("\n") + "\n";
                let _ = fs::write(&flagged_file, text);
                if !json_output {
                    println!(
                        "[CONFIRMATION] Flagged {} regression(s) to {}",
                        reg_ids.len(),
                        flagged_file.display()
                    );
                }
            }
        }
    }

    let blind_count = outcome.blind_findings().count();
    let regression_count = outcome.regressions().count();
    let unstable_count = outcome.unstable_baselines().count();
    let noise_ignored_count = outcome.sub_threshold_noise().count();

    let has_failed =
        blind_count > 0 || regression_count > 0 || (strict_stability && unstable_count > 0);
    let verdict_str = if has_failed { "FAILED" } else { "PASSED" };
    let exit_code = if blind_count > 0 {
        3
    } else if regression_count > 0 {
        1
    } else if strict_stability && unstable_count > 0 {
        2
    } else {
        0
    };

    if json_output {
        let json_findings: Vec<VerdictJsonFinding> = outcome
            .findings
            .iter()
            .map(|f| match f {
                VerdictFinding::MissingExpectedBenchmark { id } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_BLIND",
                    relative_mean: None,
                    ci_95: None,
                    delta_ns: None,
                    reason: format!("expected benchmark {id} was not executed in {log_path}"),
                },
                VerdictFinding::MissingEstimates { id } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_BLIND",
                    relative_mean: None,
                    ci_95: None,
                    delta_ns: None,
                    reason: format!(
                        "missing comparison artifact at {criterion_root}/{id}/change/estimates.json"
                    ),
                },
                VerdictFinding::Regression {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    ..
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "REGRESSION_DETECTED",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: *delta_ns,
                    reason: "Relative mean change lies entirely above noise band".to_string(),
                },
                VerdictFinding::UnstableBaseline {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    ..
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "UNSTABLE_BASELINE",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: *delta_ns,
                    reason: "Mean improvement exceeds bilateral stability band".to_string(),
                },
                VerdictFinding::NoiseIgnored {
                    id,
                    point_estimate,
                    lower_bound,
                    upper_bound,
                    delta_ns,
                    floor_ns,
                } => VerdictJsonFinding {
                    id: id.clone(),
                    status: "NOISE_IGNORED",
                    relative_mean: Some(*point_estimate),
                    ci_95: Some([*lower_bound, *upper_bound]),
                    delta_ns: Some(*delta_ns),
                    reason: format!("Absolute delta within {:.1}ns noise floor", floor_ns),
                },
            })
            .collect();

        let summary = VerdictJsonSummary {
            regressions: regression_count,
            unstable_baselines: unstable_count,
            sub_threshold_noise: noise_ignored_count,
            clean: outcome
                .executed
                .saturating_sub(regression_count + unstable_count + noise_ignored_count),
        };
        let report = VerdictJsonReport {
            schema_version: "1.1.0",
            executed_count: outcome.executed,
            noise_band_relative: noise,
            abs_noise_floor_ns,
            verdict: verdict_str,
            summary,
            findings: json_findings,
        };
        match serde_json::to_string_pretty(&report) {
            Ok(s) => println!("{s}"),
            Err(e) => run_error(format!("cannot serialize verdict json: {e}")),
        }
        exit(exit_code);
    }

    for id in &flakes_cleared {
        println!("FLAKE_CLEARED {id} cleared upon re-execution");
    }

    for finding in &outcome.findings {
        match finding {
            VerdictFinding::MissingExpectedBenchmark { id } => {
                println!("REGRESSION_BLIND expected benchmark {id} was not executed in {log_path}");
            }
            VerdictFinding::MissingEstimates { id } => {
                println!(
                    "REGRESSION_BLIND {id} has no readable Criterion comparison \
                     artifact at {criterion_root}/{id}/change/estimates.json"
                );
            }
            VerdictFinding::Regression {
                id,
                point_estimate,
                lower_bound,
                upper_bound,
                noise,
                ..
            } => {
                println!(
                    "REGRESSION_DETECTED {id} relative mean {:+}% \
                     (CI [{:+}%, {:+}%]) lies entirely above the +{}% noise band",
                    point_estimate * 100.0,
                    lower_bound * 100.0,
                    upper_bound * 100.0,
                    noise * 100.0
                );
            }
            VerdictFinding::UnstableBaseline {
                id,
                point_estimate,
                lower_bound,
                upper_bound,
                noise,
                ..
            } => {
                println!(
                    "UNSTABLE_BASELINE {id} relative mean {:+}% \
                     (CI [{:+}%, {:+}%]) exceeds stability threshold (±{:.2}%)",
                    point_estimate * 100.0,
                    lower_bound * 100.0,
                    upper_bound * 100.0,
                    noise * 100.0
                );
            }
            VerdictFinding::NoiseIgnored {
                id,
                point_estimate,
                delta_ns,
                floor_ns,
                ..
            } => {
                println!(
                    "NOISE_IGNORED {id} relative mean {:+}% within absolute noise floor ({:.2} ns <= {:.2} ns)",
                    point_estimate * 100.0,
                    delta_ns,
                    floor_ns
                );
            }
        }
    }

    if exit_code == 0 {
        println!(
            "verdict ok: no machine regression among {} executed benchmark(s) \
             (noise band ±{:.2}%)",
            outcome.executed,
            noise * 100.0
        );
    }
    exit(exit_code);
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
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let name = baseline_name(&flags);

    match persist_baseline(Path::new(&baseline_dir), Path::new(&criterion_root), &name) {
        Ok(0) => run_error(format!(
            "no '{name}' baseline series found under {criterion_root} — nothing persisted (fail-closed)"
        )),
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
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let name = baseline_name(&flags);

    match restore_baseline(Path::new(&baseline_dir), Path::new(&criterion_root), &name) {
        Ok(0) => {
            println!("no top-level '{name}' series found under {baseline_dir}");
            exit(1);
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

    if let Err(e) = append_phase_receipt(
        &out,
        &phase_id,
        &status,
        exit_code,
        observed_records,
        expected_records,
        &reason,
        &run_id,
    ) {
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
        // Provenance records (`kind: build_metadata`) have no
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

#[expect(clippy::too_many_arguments)]
fn append_phase_receipt(
    out: &str,
    phase_id: &str,
    status: &str,
    exit_code: u32,
    observed_records: u32,
    expected_records: u32,
    reason: &str,
    run_id: &str,
) -> io::Result<()> {
    let record = PhaseReceiptRecord {
        phase_id,
        status,
        exit_code,
        observed_records,
        expected_records,
        reason,
        run_id,
    };
    let line = serde_json::to_string(&record).map_err(io::Error::other)?;
    append_line(out, &line)
}

fn format_duration_ms(ms: u64) -> String {
    if ms < 1000 {
        format!("{ms}ms")
    } else {
        let s = ms / 1000;
        let rem = (ms % 1000) / 100;
        format!("{s}.{rem}s")
    }
}

fn current_utc_iso() -> String {
    let mut now: libc::time_t = 0;
    // SAFETY: now is a valid stack-allocated time_t pointer.
    unsafe { libc::time(&mut now) };
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: now and tm are valid pointers; gmtime_r writes the UTC breakdown.
    unsafe { libc::gmtime_r(&now, &mut tm) };
    let mut buf = [0u8; 64];
    // SAFETY: buf is a 64-byte stack buffer and the format string is a valid C-string literal.
    let len = unsafe {
        libc::strftime(
            buf.as_mut_ptr() as *mut libc::c_char,
            buf.len(),
            c"%Y-%m-%dT%H:%M:%SZ".as_ptr(),
            &tm,
        )
    };
    if len > 0 {
        String::from_utf8_lossy(&buf[..len]).to_string()
    } else {
        "1970-01-01T00:00:00Z".to_string()
    }
}

fn prebuild_benchmark(suite_name: &str) -> io::Result<Option<PathBuf>> {
    eprintln!("  Pre-compiling benchmark '{suite_name}' prior to cooldown (all cores)...");
    let output = Command::new("cargo")
        .args([
            "bench",
            "--bench",
            suite_name,
            "--features",
            "testing",
            "--no-run",
            "--message-format=json",
        ])
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "cargo bench pre-compilation failed with exit status {}",
            output.status
        )));
    }

    let mut candidate_exe: Option<PathBuf> = None;
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(line) {
            let reason = val.get("reason").and_then(|v| v.as_str());
            let target_name = val
                .get("target")
                .and_then(|t| t.get("name"))
                .and_then(|v| v.as_str());
            let is_bench = val
                .get("target")
                .and_then(|t| t.get("kind"))
                .and_then(|v| v.as_array())
                .map(|kinds| kinds.iter().any(|k| k.as_str() == Some("bench")))
                .unwrap_or(false);

            if reason == Some("compiler-artifact")
                && target_name == Some(suite_name)
                && is_bench
                && let Some(exe) = val.get("executable").and_then(|v| v.as_str())
            {
                candidate_exe = Some(PathBuf::from(exe));
            }
        }
    }

    if candidate_exe.is_none() {
        let deps_dir = Path::new("target/release/deps");
        if deps_dir.is_dir() {
            let prefix = format!("{suite_name}-");
            if let Ok(entries) = fs::read_dir(deps_dir) {
                let mut matches: Vec<(SystemTime, PathBuf)> = Vec::new();
                for entry in entries.flatten() {
                    let path = entry.path();
                    let file_name = path.file_name().and_then(|s| s.to_str());
                    let is_match = file_name
                        .map(|name| name.starts_with(&prefix) && !name.contains('.'))
                        .unwrap_or(false);

                    if is_match && let Ok(meta) = entry.metadata() {
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::PermissionsExt;
                            if meta.permissions().mode() & 0o111 != 0 {
                                let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
                                matches.push((mtime, path));
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            let mtime = meta.modified().unwrap_or(UNIX_EPOCH);
                            matches.push((mtime, path));
                        }
                    }
                }
                matches.sort_by_key(|(mtime, _)| *mtime);
                if let Some((_, newest)) = matches.pop() {
                    candidate_exe = Some(newest);
                }
            }
        }
    }

    if let Some(ref exe) = candidate_exe {
        let test_out = Command::new(exe).args(["--bench", "--list"]).output();
        match test_out {
            Ok(out) if out.status.success() && !out.stdout.is_empty() => {
                eprintln!("  ✓ Benchmark executable ready: {}", exe.display());
                Ok(Some(exe.clone()))
            }
            _ => {
                eprintln!(
                    "  WARN: candidate executable {} failed criterion list validation",
                    exe.display()
                );
                Ok(None)
            }
        }
    } else {
        eprintln!(
            "  WARN: Could not isolate standalone benchmark binary — fallback to cargo bench"
        );
        Ok(None)
    }
}

fn apply_thermal_cooldown(cooldown: u64) {
    if cooldown > 0 {
        eprintln!("\n[INFO] Thermal cooldown: sleeping {cooldown}s before microbenchmarks...");
        std::thread::sleep(Duration::from_secs(cooldown));
    } else {
        eprintln!("\n[INFO] Thermal cooldown: bypassed (NAM_THERMAL_COOLDOWN_S=0)");
    }
}

fn run_benchmark_process(
    bench_exe: Option<&Path>,
    suite: &str,
    bench_core: Option<u32>,
    extra_args: &[&str],
    log_path: &str,
    baseline_name: &str,
    run_id: &str,
) -> io::Result<i32> {
    if let Some(parent) = Path::new(log_path).parent() {
        fs::create_dir_all(parent)?;
    }
    let date_str = current_utc_iso();
    let exe_display = bench_exe
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| format!("cargo bench --bench {suite}"));
    let header = format!(
        "# [NAM_REGRESSION_GATE] bench_exe={exe_display} baseline={baseline_name} run_id={run_id} date={date_str}\n"
    );
    fs::write(log_path, header)?;

    let mut cmd = match bench_core {
        Some(core) => {
            let mut c = Command::new("taskset");
            c.arg("-c").arg(core.to_string());
            if let Some(exe) = bench_exe {
                c.arg(exe).arg("--bench").args(extra_args);
            } else {
                c.args([
                    "cargo",
                    "bench",
                    "--bench",
                    suite,
                    "--features",
                    "testing",
                    "--",
                ])
                .args(extra_args);
            }
            c
        }
        None => {
            if let Some(exe) = bench_exe {
                let mut c = Command::new(exe);
                c.arg("--bench").args(extra_args);
                c
            } else {
                let mut c = Command::new("cargo");
                c.args(["bench", "--bench", suite, "--features", "testing", "--"])
                    .args(extra_args);
                c
            }
        }
    };

    let mut child = match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(child) => child,
        Err(e) if bench_core.is_some() && e.kind() == io::ErrorKind::NotFound => {
            eprintln!("WARN: taskset not found in PATH, running without CPU pinning");
            let mut fallback_cmd = if let Some(exe) = bench_exe {
                let mut c = Command::new(exe);
                c.arg("--bench").args(extra_args);
                c
            } else {
                let mut c = Command::new("cargo");
                c.args(["bench", "--bench", suite, "--features", "testing", "--"])
                    .args(extra_args);
                c
            };
            fallback_cmd
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?
        }
        Err(e) => return Err(e),
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("no child stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| io::Error::other("no child stderr"))?;

    let log_file = Arc::new(Mutex::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?,
    ));

    let log_out = Arc::clone(&log_file);
    let t_out = std::thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            println!("{line}");
            if let Ok(mut f) = log_out.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
    });

    let log_err = Arc::clone(&log_file);
    let t_err = std::thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            eprintln!("{line}");
            if let Ok(mut f) = log_err.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
    });

    let _ = t_out.join();
    let _ = t_err.join();
    let status = child.wait()?;
    Ok(status.code().unwrap_or(1))
}

// ── pick-core ────────────────────────────────────────────────────────────────

fn cmd_pick_core(args: &[String]) {
    let flags = match parse_flags_full(
        args,
        &["bench-core", "sysfs"],
        &["strict", "require-isolated"],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            println!(
                "Usage: nam_perf_gate pick-core [--bench-core <core>] [--sysfs <dir>] [--strict] [--require-isolated]"
            );
            exit(0);
        }
        Err(e) => usage_error("pick-core", &e),
    };

    let explicit_flag = flags.get("bench-core");
    let env_core = env::var("NAM_BENCH_CORE").ok();
    let explicit_str = explicit_flag.or(env_core.as_deref());
    let sysfs = flags.get("sysfs").map(Path::new);

    if flags.get_bool("require-isolated") {
        let base = sysfs.unwrap_or_else(|| Path::new(DEFAULT_SYSFS_CPU_DIR));
        let isolated = fs::read_to_string(base.join("isolated")).unwrap_or_default();
        let candidates = parse_cpu_list(&isolated);
        let host = Path::new(DEFAULT_SYSFS_CPU_DIR);
        let online = |core: u32| {
            let idx = core as usize;
            if base.join(format!("cpu{idx}")).is_dir() {
                is_cpu_online(base, idx)
            } else {
                is_cpu_online(host, idx)
            }
        };
        let explicit = explicit_str.and_then(|raw| {
            let trimmed = raw.trim();
            if trimmed.is_empty() || trimmed == "auto" {
                None
            } else {
                trimmed.parse::<u32>().ok()
            }
        });
        let selected = match explicit {
            Some(core) => (candidates.contains(&(core as usize)) && online(core)).then_some(core),
            None => candidates
                .into_iter()
                .find(|core| online(*core as u32))
                .map(|core| core as u32),
        };
        match selected {
            Some(core) => {
                println!("{core}");
                exit(0);
            }
            None => {
                println!("GAP");
                eprintln!("pick-core: no online isolated bench core found (--require-isolated)");
                exit(3);
            }
        }
    }

    let resolved = resolve_bench_core(explicit_str, sysfs);

    match resolved {
        Some(core) => {
            println!("{core}");
            exit(0);
        }
        None => {
            println!("GAP");
            if flags.get_bool("strict") {
                eprintln!("pick-core: no isolated or permitted bench core found (--strict)");
                exit(1);
            }
            exit(0);
        }
    }
}

// ── run ──────────────────────────────────────────────────────────────────────

fn cmd_run(args: &[String]) {
    let flags = match parse_flags_full(
        args,
        &[
            "baseline",
            "baseline-dir",
            "fingerprint",
            "criterion-root",
            "bench-core",
            "sysfs",
            "suite",
            "manifest",
            "cooldown",
            "receipt",
            "log",
            "run-id",
            "noise",
            "abs-noise-floor-ns",
            "bench-exe",
        ],
        &[
            "check",
            "bootstrap-baseline",
            "check-stability",
            "strict-stability",
            "confirm-regressions",
            "strict",
        ],
    ) {
        Ok(f) => f,
        Err(e) if e == HELP_REQUEST => {
            print_run_help();
            exit(0);
        }
        Err(e) => usage_error("run", &e),
    };

    let is_bootstrap = flags.get_bool("bootstrap-baseline");
    let is_check = flags.get_bool("check");
    if is_bootstrap && is_check {
        usage_error(
            "run",
            "cannot specify both --check and --bootstrap-baseline",
        );
    }
    let mode_bootstrap = is_bootstrap;

    let baseline_dir = flags
        .get("baseline-dir")
        .unwrap_or(DEFAULT_BASELINE_DIR)
        .to_string();
    let baseline = flags
        .get("baseline")
        .map(str::to_string)
        .unwrap_or_else(|| {
            env::var("NAM_BASELINE_NAME").unwrap_or_else(|_| DEFAULT_BASELINE_NAME.to_string())
        });
    let criterion_root = flags
        .get("criterion-root")
        .map(str::to_string)
        .unwrap_or_else(default_criterion_root);
    let suite = flags.get("suite").unwrap_or("regression_gate").to_string();
    let manifest = flags.get("manifest").map(str::to_string);
    let receipt_path = flags.get("receipt").unwrap_or(DEFAULT_RECEIPT).to_string();
    let log_path = flags
        .get("log")
        .unwrap_or("target/logs/regression-check.log")
        .to_string();
    let run_id = flags.get("run-id").map(str::to_string).unwrap_or_else(|| {
        env::var("NAM_RUN_ID").unwrap_or_else(|_| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            format!("{}-{}", now, std::process::id())
        })
    });
    let cooldown_secs = match flags.get("cooldown") {
        Some(v) => v.parse::<u64>().unwrap_or_else(|_| {
            eprintln!("\n[WARN] Invalid --cooldown='{v}', defaulting to 0s");
            0
        }),
        None => match env::var("NAM_THERMAL_COOLDOWN_S") {
            Ok(v) => v.parse::<u64>().unwrap_or_else(|_| {
                eprintln!("\n[WARN] Invalid NAM_THERMAL_COOLDOWN_S='{v}', defaulting to 0s");
                0
            }),
            Err(_) => 0,
        },
    };

    let is_strict = flags.get_bool("strict");
    let explicit_flag = flags.get("bench-core");
    let env_core = env::var("NAM_BENCH_CORE").ok();
    let explicit_str = explicit_flag.or(env_core.as_deref());
    let sysfs = flags.get("sysfs").map(Path::new);
    let resolved_core = resolve_bench_core(explicit_str, sysfs);

    if is_strict && resolved_core.is_none() {
        eprintln!("run: strict mode enabled and no bench core available (GAP)");
        exit(1);
    }

    let fingerprint_path = flags
        .get("fingerprint")
        .map(String::from)
        .unwrap_or_else(|| format!("{baseline_dir}/baseline-fingerprint.json"));

    // Baseline presence is a state decision (A2): decide it here, before the
    // benchmark pre-compilation (L-06). A missing baseline is a GAP in
    // non-strict mode (exit 3) and a typed FAIL in strict mode (exit 4),
    // never a performance regression.
    if !mode_bootstrap && !Path::new(&fingerprint_path).is_file() {
        println!("MISSING_BASELINE no baseline fingerprint at {fingerprint_path}");
        println!("Re-bootstrap the baseline with `--bootstrap-baseline` (human operation).");
        if is_strict {
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                4,
                0,
                1,
                "MISSING_BASELINE",
                &run_id,
            );
            exit(4);
        }
        println!("GAP perf-baseline-missing (non-strict mode)");
        exit(3);
    }

    let bench_exe: Option<PathBuf> = if let Some(exe) = flags.get("bench-exe") {
        let p = PathBuf::from(exe);
        if !p.exists() {
            run_error(format!(
                "specified --bench-exe does not exist: {}",
                p.display()
            ));
        }
        Some(p)
    } else {
        match prebuild_benchmark(&suite) {
            Ok(exe) => exe,
            Err(e) => {
                eprintln!("Pre-compilation of benchmark '{suite}' failed: {e}");
                if !mode_bootstrap {
                    let _ = append_phase_receipt(
                        &receipt_path,
                        "regression_check",
                        "FAIL",
                        1,
                        0,
                        1,
                        "BENCHMARK_PREBUILD_FAILED",
                        &run_id,
                    );
                }
                exit(1);
            }
        }
    };

    if mode_bootstrap {
        println!("\n[BOOTSTRAP] Creating new performance baseline...");
        println!("  ⚠ This operation must be performed by a human operator.");
        println!("  ⚠ Automated/CI/agent-driven execution is prohibited.\n");

        apply_thermal_cooldown(cooldown_secs);

        let t0 = Instant::now();
        let extra = ["--save-baseline", &baseline];
        let bench_rc = match run_benchmark_process(
            bench_exe.as_deref(),
            &suite,
            resolved_core,
            &extra,
            &log_path,
            &baseline,
            &run_id,
        ) {
            Ok(rc) => rc,
            Err(e) => run_error(format!("Failed to execute benchmark: {e}")),
        };
        if bench_rc != 0 {
            run_error(format!(
                "Benchmark execution failed during baseline bootstrap (rc={bench_rc})"
            ));
        }
        let dur_ms = t0.elapsed().as_millis() as u64;
        let dur_str = format_duration_ms(dur_ms);

        match persist_baseline(
            Path::new(&baseline_dir),
            Path::new(&criterion_root),
            &baseline,
        ) {
            Ok(0) => run_error(format!(
                "no '{baseline}' baseline series found under {criterion_root} — nothing persisted (fail-closed)"
            )),
            Ok(count) => {
                eprintln!("  ✓ Persisted {count} baseline series to {baseline_dir}");
            }
            Err(e) => run_error(format!("cannot persist baseline: {e}")),
        }

        let core_str = resolved_core.map(|c| c.to_string()).unwrap_or_default();
        let fingerprint = Fingerprint::probe(&core_str);
        if let Err(e) = fingerprint.write_to_path(Path::new(&fingerprint_path)) {
            run_error(format!(
                "cannot write fingerprint to {fingerprint_path}: {e}"
            ));
        }

        if let Err(e) = append_phase_receipt(
            &receipt_path,
            "regression_baseline_created",
            "PASS",
            0,
            1,
            1,
            "",
            &run_id,
        ) {
            run_error(format!("cannot append receipt to {receipt_path}: {e}"));
        }

        eprintln!("  ✓ Baseline '{baseline}' created and persisted in {dur_str}.");
        println!(
            "\n================================================================================"
        );
        println!("  Performance Bootstrap Artifacts saved:");
        println!("    - Fingerprint:   {fingerprint_path}");
        println!("    - Baseline Data: {baseline_dir}");
        println!("    - Receipt:       {receipt_path}");
        println!(
            "================================================================================\n"
        );
        exit(0);
    }

    // ── Check mode ───────────────────────────────────────────────────────────
    println!("\n[CHECK] Comparing against CI baseline...");

    let core_str = resolved_core.map(|c| c.to_string()).unwrap_or_default();
    let current_fingerprint = Fingerprint::probe(&core_str);

    let baseline_fingerprint = match Fingerprint::read_from_path(Path::new(&fingerprint_path)) {
        Ok(b) => b,
        Err(FingerprintError::MissingBaseline) => {
            println!("MISSING_BASELINE no baseline fingerprint at {fingerprint_path}");
            println!("Re-bootstrap the baseline with `--bootstrap-baseline` (human operation).");
            if is_strict {
                let _ = append_phase_receipt(
                    &receipt_path,
                    "regression_check",
                    "FAIL",
                    4,
                    0,
                    1,
                    "MISSING_BASELINE",
                    &run_id,
                );
                exit(4);
            }
            exit(3);
        }
        Err(e) => {
            eprintln!("cannot read baseline fingerprint {fingerprint_path}: {e}");
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "FINGERPRINT_ERROR",
                &run_id,
            );
            exit(1);
        }
    };

    if let Err(e) = current_fingerprint.compare(&baseline_fingerprint) {
        println!("{e}");
        println!("Performance comparison aborted: environment incompatible with baseline.");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            1,
            0,
            1,
            "INCOMPARABLE_ENVIRONMENT",
            &run_id,
        );
        exit(1);
    }

    if let Err(e) = restore_baseline(
        Path::new(&baseline_dir),
        Path::new(&criterion_root),
        &baseline,
    ) {
        eprintln!("cannot restore baseline: {e}");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            1,
            0,
            1,
            "BASELINE_RESTORE_FAILED",
            &run_id,
        );
        exit(1);
    }

    apply_thermal_cooldown(cooldown_secs);

    let t0 = Instant::now();
    let extra = ["--baseline", &baseline];
    let bench_rc = match run_benchmark_process(
        bench_exe.as_deref(),
        &suite,
        resolved_core,
        &extra,
        &log_path,
        &baseline,
        &run_id,
    ) {
        Ok(rc) => rc,
        Err(e) => run_error(format!("Failed to execute benchmark: {e}")),
    };
    let dur_ms = t0.elapsed().as_millis() as u64;
    let dur_str = format_duration_ms(dur_ms);

    if bench_rc != 0 {
        eprintln!("\n❌ Benchmark run failed (status={bench_rc})");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            bench_rc as u32,
            0,
            1,
            "Benchmark run failed",
            &run_id,
        );
        exit(1);
    }

    let log_text = fs::read_to_string(&log_path).unwrap_or_default();

    let manifest_p = manifest.as_deref().map(Path::new);
    let expected_ids = match expected_bench_ids_for_suite(&suite, manifest_p) {
        Ok(ids) => ids,
        Err(e) => {
            eprintln!("\n❌ {e}");
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "EXPECTED_IDS_ERROR",
                &run_id,
            );
            exit(1);
        }
    };

    match missing_expected_benchmarks(&log_text, &expected_ids) {
        Err(BaselineCoverageGap) => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: no executed benchmark could be parsed from {log_path}"
            );
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
            exit(1);
        }
        Ok(missing) if !missing.is_empty() => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: expected benchmark(s) were not executed in {log_path}:"
            );
            for id in &missing {
                eprintln!("  {id}");
            }
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
            exit(1);
        }
        Ok(_) => {}
    }

    match missing_baseline_coverage(&log_text, Path::new(&criterion_root), &baseline) {
        Err(BaselineCoverageGap) => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: no executed benchmark could be parsed from {log_path}"
            );
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
            exit(1);
        }
        Ok(missing) if !missing.is_empty() => {
            eprintln!(
                "\n❌ BASELINE_COVERAGE_GAP: executed benchmark(s) without a saved baseline series under {criterion_root}/<id>/{baseline}/:"
            );
            for id in &missing {
                eprintln!("  {id}");
            }
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "BASELINE_COVERAGE_GAP",
                &run_id,
            );
            exit(1);
        }
        Ok(_) => {}
    }

    let noise = parse_f64(&flags, "noise", "run").unwrap_or(DEFAULT_NOISE_THRESHOLD);
    let abs_noise_floor_ns = parse_f64(&flags, "abs-noise-floor-ns", "run");
    let check_stability = flags.get_bool("check-stability") || flags.get_bool("strict-stability");
    let strict_stability = flags.get_bool("strict-stability");
    let opts = VerdictOptions {
        noise,
        expected_ids: Some(expected_ids.clone()),
        check_stability,
        abs_noise_floor_ns,
    };

    let outcome = match regression_verdict_with_options(
        &log_text,
        Path::new(&criterion_root),
        &opts,
    ) {
        Err(VerdictBlind) => {
            eprintln!(
                "\n❌ REGRESSION VERDICT BLIND — Criterion comparison artifacts missing/unreadable"
            );
            let _ = append_phase_receipt(
                &receipt_path,
                "regression_check",
                "FAIL",
                1,
                0,
                1,
                "REGRESSION_BLIND",
                &run_id,
            );
            exit(1);
        }
        Ok(out) => out,
    };

    let regressions: Vec<_> = outcome.regressions().collect();
    if !regressions.is_empty() {
        for f in &regressions {
            eprintln!("  ❌ Regression: {:?}", f);
        }
        eprintln!("\n❌ PERFORMANCE REGRESSION DETECTED");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            1,
            0,
            1,
            "REGRESSION_DETECTED",
            &run_id,
        );
        exit(1);
    }

    let unstable: Vec<_> = outcome.unstable_baselines().collect();
    if strict_stability && !unstable.is_empty() {
        for f in &unstable {
            eprintln!("  ⚠️ Unstable baseline: {:?}", f);
        }
        eprintln!("\n❌ BILATERAL BASELINE INSTABILITY DETECTED (--strict-stability)");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            1,
            0,
            1,
            "UNSTABLE_BASELINE",
            &run_id,
        );
        exit(1);
    }

    let executed = executed_bench_ids(&log_text);
    let executed_count = executed.len();
    if executed_count < 1 {
        eprintln!("\n❌ NO BENCHMARK EXECUTED — refusing to certify observed==expected");
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_check",
            "FAIL",
            1,
            0,
            1,
            "NO_BENCHMARK_EXECUTED",
            &run_id,
        );
        exit(1);
    }

    let expected_count = expected_ids.len();
    if executed_count != expected_count {
        eprintln!(
            "\n❌ BENCHMARK COVERAGE MISMATCH — observed {executed_count} != expected {expected_count}"
        );
        let _ = append_phase_receipt(
            &receipt_path,
            "regression_baseline_coverage",
            "FAIL",
            1,
            executed_count as u32,
            expected_count as u32,
            "EXPECTED_BENCHMARK_MISSING",
            &run_id,
        );
        exit(1);
    }

    let _ = append_phase_receipt(
        &receipt_path,
        "regression_baseline_coverage",
        "PASS",
        0,
        executed_count as u32,
        expected_count as u32,
        "",
        &run_id,
    );

    eprintln!(
        "  ✓ Coverage cross-check: all {executed_count} executed benchmark(s) have baseline series."
    );
    eprintln!("  ✓ No performance regression detected in {dur_str} (machine verdict).");

    let _ = append_phase_receipt(
        &receipt_path,
        "regression_check",
        "PASS",
        0,
        1,
        1,
        "",
        &run_id,
    );

    println!("\n================================================================================");
    println!("  Performance Gate Artifacts saved:");
    println!("    - Benchmark Log: {log_path}");
    println!("    - Receipt:       {receipt_path}");
    println!("================================================================================\n");

    exit(0);
}

fn print_run_help() {
    println!("Usage:");
    println!("  nam_perf_gate run [--check | --bootstrap-baseline] [--baseline <name>]");
    println!("                    [--baseline-dir <dir>] [--criterion-root <dir>]");
    println!("                    [--bench-core <core>] [--sysfs <dir>] [--suite <name>]");
    println!("                    [--manifest <path>] [--cooldown <seconds>]");
    println!("                    [--receipt <path>] [--log <path>] [--run-id <id>]");
    println!("                    [--noise <fraction>] [--abs-noise-floor-ns <nanoseconds>]");
    println!("                    [--check-stability] [--strict-stability]");
    println!("                    [--confirm-regressions] [--bench-exe <path>]");
    println!("                    [--strict]");
}
