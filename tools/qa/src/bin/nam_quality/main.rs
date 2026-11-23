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
//!   (`PROVENANCE: OK/FAIL/NOT_RUN`, `FIDELITY: OK/FAIL/NOT_VERIFIED`,
//!   `PERFORMANCE: OK/FAIL/NOT_VERIFIED`, `CONTRACT VIOLATED`) mirroring the
//!   dashboard text. Exit 0 only when both domains pass and no oracle review
//!   is pending.
//! - `receipt append`: append one phase receipt record (serde — replaces the
//!   dashboard `printf '{`), byte-compatible with the old bash schema.
//! - `save --contract <path> --receipt <path>`: transactional promotion of a
//!   validated contract JSON (temp file + atomic rename). Refuses when a
//!   fidelity phase FAILed or the provenance phase failed — the performance
//!   domain never blocks saving (PERF-006; `NOT_VERIFIED` performance must
//!   not block, bash comment at `quality-dashboard.sh:2599-2602`).
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

mod cli;
mod dashboard;
mod ingest;
mod receipt;
mod save;
mod third_party;
mod verify;

use std::env;
use std::process::exit;

use dashboard::cmd_dashboard;
use ingest::cmd_ingest;
use receipt::cmd_receipt;
use save::cmd_save;
use verify::{cmd_classify, cmd_render, cmd_verify};

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
        "dashboard" => cmd_dashboard(&args[1..]),
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
    println!("  nam_quality dashboard [--check <contract>] [--save] [--contract <path>]");
    println!("                          [--fidelity-only] [--bench-core <core>]");
    println!("                          [--run-id <id>] [--log-dir <dir>] [--criterion-dir <dir>]");
    println!("                          [--ansi | --plain]");
    println!("  nam_quality ingest [--receipt <path>] [--metrics <path>] [--latency <path>]");
    println!("                          [--criterion-dir <path>] [--bench-constants-env <path>]");
    println!("                          [--contract <path>] [--out <path>]");
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
