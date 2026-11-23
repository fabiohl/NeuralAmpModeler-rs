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

#[path = "nam_perf_gate/cli.rs"]
mod cli;
#[path = "nam_perf_gate/commands.rs"]
mod commands;
#[path = "nam_perf_gate/receipt.rs"]
mod receipt;
#[path = "nam_perf_gate/runner.rs"]
mod runner;
#[path = "nam_perf_gate/verdict.rs"]
mod verdict;

use std::env;
use std::process::exit;

use cli::print_help;
use commands::{
    cmd_compare, cmd_coverage, cmd_expected_count, cmd_persist_baseline, cmd_pick_core, cmd_probe,
    cmd_restore_baseline,
};
use receipt::cmd_receipt;
use runner::cmd_run;
use verdict::cmd_verdict;

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
