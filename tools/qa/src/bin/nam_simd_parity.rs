// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fail-closed reconciler for the mandatory AVX2 -> AVX-512 cross-ISA parity matrix.
//!
//! Exit codes:
//! - 0: all 10 mandatory comparisons passed with matching `run_id` and zero violations.
//! - 1: missing/unreadable/corrupt JSONL, stale `run_id`, missing/duplicate/unexpected cases,
//!   or any failed/skipped/out-of-budget comparison.
//! - 2: usage / CLI argument error.

use std::env;
use std::path::PathBuf;
use std::process::exit;

use serde::Serialize;

use neural_amp_modeler_rs::testing::qa::simd_parity::{
    SimdParityReconciliation, reconcile_simd_parity_file,
};

#[derive(Serialize)]
struct ReconcileSummaryJson<'a> {
    verdict: &'static str,
    run_id: &'a str,
    expected_cases: usize,
    observed_cases: usize,
    passed_cases: usize,
    violations: &'a [String],
}

fn print_help() {
    println!("NeuralAmpModeler-rs Cross-ISA Parity Matrix Reconciler");
    println!("Usage: nam_simd_parity --jsonl <FILE> [--run-id <ID>] [--json]");
    println!();
    println!("Options:");
    println!("  --jsonl <FILE>   Structured SIMD parity JSONL output to reconcile");
    println!("  --run-id <ID>    Expected run_id (default: NAM_RUN_ID from environment)");
    println!("  --json           Emit JSON summary instead of human-readable table");
    println!("  -h, --help       Print this help");
}

fn parse_args() -> Result<(PathBuf, String, bool), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut jsonl_path: Option<PathBuf> = None;
    let mut run_id: Option<String> = None;
    let mut json_out = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                exit(0);
            }
            "--jsonl" => {
                i += 1;
                let val = args.get(i).ok_or("--jsonl requires a file path")?;
                jsonl_path = Some(PathBuf::from(val));
            }
            "--run-id" => {
                i += 1;
                let val = args.get(i).ok_or("--run-id requires a value")?;
                let trimmed = val.trim();
                if trimmed.is_empty() {
                    return Err("--run-id must not be empty".to_owned());
                }
                run_id = Some(trimmed.to_owned());
            }
            "--json" => {
                json_out = true;
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }

    let path = jsonl_path.ok_or("missing required flag --jsonl <FILE>")?;
    let expected_run_id = match run_id {
        Some(id) => id,
        None => env::var("NAM_RUN_ID")
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .ok_or("missing --run-id and NAM_RUN_ID is unset/empty")?,
    };

    Ok((path, expected_run_id, json_out))
}

fn print_human_summary(res: &SimdParityReconciliation) {
    let verdict = if res.is_pass() { "PASS" } else { "FAIL" };
    println!(
        "SIMD Parity Reconciliation: verdict={verdict} run_id={} observed={}/{} passed={}/{}",
        res.run_id, res.observed_cases, res.expected_cases, res.passed_cases, res.expected_cases
    );
    for rec in &res.records {
        println!(
            "  {:<22} | {:<9} | {}->{} | {:?} | esr={}",
            rec.case_id,
            rec.precision.as_str(),
            rec.ref_isa,
            rec.test_isa,
            rec.status,
            rec.esr
                .map(|v| format!("{v:.3e}"))
                .unwrap_or_else(|| "-".to_owned()),
        );
    }
    if !res.violations.is_empty() {
        eprintln!("Reconciliation violations ({}):", res.violations.len());
        for v in &res.violations {
            eprintln!("  - {v}");
        }
    }
}

fn main() {
    let (path, expected_run_id, json_out) = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("nam_simd_parity: {e}");
            exit(2);
        }
    };

    let reconciliation = match reconcile_simd_parity_file(&path, &expected_run_id) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: nam_simd_parity: {e}");
            exit(1);
        }
    };

    if json_out {
        let payload = ReconcileSummaryJson {
            verdict: if reconciliation.is_pass() {
                "PASS"
            } else {
                "FAIL"
            },
            run_id: &reconciliation.run_id,
            expected_cases: reconciliation.expected_cases,
            observed_cases: reconciliation.observed_cases,
            passed_cases: reconciliation.passed_cases,
            violations: &reconciliation.violations,
        };
        let text = serde_json::to_string_pretty(&payload).unwrap_or_else(|e| {
            eprintln!("ERROR: nam_simd_parity: failed to serialize summary: {e}");
            exit(1);
        });
        println!("{text}");
    } else {
        print_human_summary(&reconciliation);
    }

    if !reconciliation.is_pass() {
        exit(1);
    }
}
