// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Regression phase receipt append and summary commands.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::process::exit;

use serde::Serialize;
use serde_json::Value;

use crate::cli::{
    DEFAULT_RECEIPT, HELP_REQUEST, RECEIPT_STATUSES, parse_flags, parse_u32, required, run_error,
    usage_error,
};

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

pub(crate) fn cmd_receipt(args: &[String]) {
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
pub(crate) fn append_phase_receipt(
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
