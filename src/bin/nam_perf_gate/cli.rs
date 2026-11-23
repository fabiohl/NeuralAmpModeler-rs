// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Strict fail-closed CLI flag parsing, defaults, and usage helpers.

use std::collections::HashMap;
use std::env;
use std::process::exit;

use neural_amp_modeler_rs::testing::qa::baseline_store::DEFAULT_BASELINE_NAME;
use neural_amp_modeler_rs::testing::qa::verdict::DEFAULT_NOISE_THRESHOLD;

/// Store directory of persisted baselines (gitignored).
pub(crate) const DEFAULT_BASELINE_DIR: &str = ".performance-baselines";
/// Criterion transient working area.
pub(crate) const DEFAULT_CRITERION_ROOT: &str = "target/criterion";

pub(crate) fn default_criterion_root() -> String {
    if let Ok(target_dir) = env::var("CARGO_TARGET_DIR") {
        let trimmed = target_dir.trim();
        if !trimmed.is_empty() {
            return format!("{trimmed}/criterion");
        }
    }
    DEFAULT_CRITERION_ROOT.to_string()
}
/// Regression phase receipt (bash `REGRESSION_RECEIPT`).
pub(crate) const DEFAULT_RECEIPT: &str = "target/logs/regression_phase_receipt.jsonl";

/// Status values accepted by `receipt append` — the documented receipt
/// schema (`utils/_lib.sh:80`) plus `NOT_VERIFIED` (dashboard writes it for
/// `regression_gate`); mirrors `nam_quality`.
pub(crate) const RECEIPT_STATUSES: [&str; 6] = [
    "PASS",
    "FAIL",
    "SKIP_CAPABILITY",
    "SKIP_OPTIONAL_FIXTURE",
    "NOT_RUN",
    "NOT_VERIFIED",
];

/// Sentinel used internally to forward a `--help` request from `parse_flags`.
pub(crate) const HELP_REQUEST: &str = "__help__";

pub(crate) fn print_help() {
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

pub(crate) struct Flags {
    map: HashMap<String, String>,
}

impl Flags {
    pub(crate) fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(String::as_str)
    }

    pub(crate) fn get_bool(&self, name: &str) -> bool {
        self.map.get(name).map(|v| v == "true").unwrap_or(false)
    }
}

pub(crate) fn parse_flags(args: &[String], allowed: &[&str]) -> Result<Flags, String> {
    parse_flags_full(args, allowed, &[])
}

pub(crate) fn parse_flags_full(
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

pub(crate) fn parse_f64(flags: &Flags, name: &str, context: &str) -> Option<f64> {
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

pub(crate) fn usage_error(context: &str, message: &str) -> ! {
    eprintln!("{context}: {message}");
    eprintln!("Run `nam_perf_gate --help` for usage.");
    exit(2);
}

pub(crate) fn run_error(message: String) -> ! {
    eprintln!("ERROR: {message}");
    exit(1);
}

pub(crate) fn required<'a>(flags: &'a Flags, name: &str) -> Result<&'a str, String> {
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
pub(crate) fn parse_u32(flags: &Flags, name: &str, context: &str) -> u32 {
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
pub(crate) fn parse_noise(flags: &Flags, name: &str, context: &str) -> f64 {
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
pub(crate) fn bench_core(flags: &Flags) -> String {
    match flags.get("bench-core") {
        Some(v) => v.to_string(),
        None => env::var("NAM_BENCH_CORE").unwrap_or_default(),
    }
}

pub(crate) fn baseline_name(flags: &Flags) -> String {
    match flags.get("baseline") {
        Some(v) => v.to_string(),
        None => env::var("NAM_BASELINE_NAME").unwrap_or_else(|_| DEFAULT_BASELINE_NAME.to_string()),
    }
}
