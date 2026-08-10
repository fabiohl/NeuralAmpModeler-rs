// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Binary CLI utility to generate and emit the machine-readable capability receipt.
//!
//! Output modes:
//! - Default / `--json`: Prints formatted JSON to stdout.
//! - `--table`: Prints human-readable ASCII/Markdown audit table to stdout.
//! - `--out <FILE>`: Writes structured JSON receipt to the specified file path.
//!
//! Exit codes:
//! - 0: All 51 catalog entries matched expected classification policies (zero unexpected failures).
//! - 1: One or more entries produced an unexpected `FAILED` result.

use std::env;
use std::fs;
use std::process::exit;

use neural_amp_modeler_rs::testing::receipt::generate_capability_receipt;

fn main() {
    let args: Vec<String> = env::args().collect();
    let receipt = generate_capability_receipt();

    let mut format_table = false;
    let mut output_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--table" => format_table = true,
            "--json" => format_table = false,
            "--out" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 1;
                } else {
                    eprintln!("Error: --out requires a file path argument");
                    exit(1);
                }
            }
            "--help" | "-h" => {
                println!("NeuralAmpModeler-rs Capability Receipt Emitter");
                println!("Usage: capability_receipt [--json | --table] [--out <FILE>]");
                exit(0);
            }
            arg => {
                eprintln!("Unknown argument: {arg}");
                exit(1);
            }
        }
        i += 1;
    }

    let json_output = receipt.render_json();

    if let Some(path) = output_path {
        if let Err(e) = fs::write(&path, &json_output) {
            eprintln!("Failed to write capability receipt to {path}: {e}");
            exit(1);
        }
        eprintln!("Successfully wrote capability receipt JSON to {path}");
    }

    if format_table {
        println!("{}", receipt.render_table());
    } else {
        println!("{json_output}");
    }

    if receipt.has_unexpected_failures() {
        eprintln!("CRITICAL: Capability receipt contains unexpected FAILED stages!");
        exit(1);
    }

    exit(0);
}
