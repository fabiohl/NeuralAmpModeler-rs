// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Phase receipt serialization and append command.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::exit;

use serde::Serialize;

use crate::cli::{HELP_REQUEST, parse_flags, parse_u32, required, run_error, usage_error};

/// Status values accepted by `receipt append` — the documented receipt schema
/// (`utils/_lib.sh:80`) plus `NOT_VERIFIED`, which the dashboard writes for
/// `regression_gate` (`quality-dashboard.sh:502`).
pub(crate) const RECEIPT_STATUSES: [&str; 6] = [
    "PASS",
    "FAIL",
    "SKIP_CAPABILITY",
    "SKIP_OPTIONAL_FIXTURE",
    "NOT_RUN",
    "NOT_VERIFIED",
];

/// Phase receipt record — serde serializes fields in declaration order, which
/// reproduces the bash `printf` schema of `dashboard_phase_receipt`
/// (`utils/_lib.sh:96`): `phase_id, status, exit_code, observed_records,
/// expected_records, reason, run_id`.
#[derive(Debug, Serialize)]
pub(crate) struct PhaseReceiptRecord<'a> {
    pub(crate) phase_id: &'a str,
    pub(crate) status: &'a str,
    pub(crate) exit_code: u32,
    pub(crate) observed_records: u32,
    pub(crate) expected_records: u32,
    pub(crate) reason: &'a str,
    pub(crate) run_id: &'a str,
}

pub(crate) fn cmd_receipt(args: &[String]) {
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

pub(crate) fn append_line(out: &str, line: &str) -> io::Result<()> {
    if let Some(parent) = Path::new(out).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(out)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")
}
