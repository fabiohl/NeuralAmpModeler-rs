// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fail-closed typed reader for `skips-<NAM_RUN_ID>.jsonl` and fallback `skips.jsonl`.
//!
//! Exit codes:
//! - 0: skip log absent legitimately or parsed cleanly with all records valid.
//! - 1: unreadable file, directory, dangling symlink, malformed JSONL, invalid fields,
//!   or `run_id` mismatch against the active run.
//! - 2: usage / CLI argument error.

use std::env;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::exit;

use serde::Serialize;

use neural_amp_modeler_rs::testing::skip::{SkipRecord, skip_log_path};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFormat {
    Blocking,
    Counts,
    Json,
}

#[derive(Debug, Default, Serialize)]
struct SkipSummary {
    path: String,
    total: usize,
    optional: usize,
    blocking: usize,
}

fn print_help() {
    println!("NeuralAmpModeler-rs Skip Log Reader");
    println!(
        "Usage: nam_skips [--path <FILE>] [--log-dir <DIR>] [--run-id <ID>] [--no-run-id] [--format <blocking|counts|json>]"
    );
    println!();
    println!("Options:");
    println!(
        "  --path <FILE>       Explicit skip JSONL path (overrides --log-dir/--run-id derivation)"
    );
    println!("  --log-dir <DIR>     Directory for derived skip log path (default: target/logs)");
    println!("  --run-id <ID>       Expected run_id (default: NAM_RUN_ID from environment)");
    println!("  --no-run-id         Expect fallback skips.jsonl with absent run_id");
    println!("  --format <FMT>      Output format: blocking | counts | json (default: counts)");
    println!("  -h, --help          Print this help");
}

fn parse_args() -> Result<(PathBuf, Option<String>, OutputFormat), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut explicit_path: Option<PathBuf> = None;
    let mut log_dir = PathBuf::from("target/logs");
    let mut run_id_override: Option<Option<String>> = None;
    let mut format = OutputFormat::Counts;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_help();
                exit(0);
            }
            "--path" => {
                i += 1;
                let val = args.get(i).ok_or("--path requires a file path")?;
                explicit_path = Some(PathBuf::from(val));
            }
            "--log-dir" => {
                i += 1;
                let val = args.get(i).ok_or("--log-dir requires a directory path")?;
                log_dir = PathBuf::from(val);
            }
            "--run-id" => {
                i += 1;
                let val = args.get(i).ok_or("--run-id requires a value")?;
                let trimmed = val.trim();
                run_id_override = Some((!trimmed.is_empty()).then(|| trimmed.to_owned()));
            }
            "--no-run-id" => {
                run_id_override = Some(None);
            }
            "--format" => {
                i += 1;
                let val = args.get(i).ok_or("--format requires a value")?;
                format = match val.as_str() {
                    "blocking" => OutputFormat::Blocking,
                    "counts" => OutputFormat::Counts,
                    "json" => OutputFormat::Json,
                    other => return Err(format!("unknown format '{other}'")),
                };
            }
            other => return Err(format!("unknown argument '{other}'")),
        }
        i += 1;
    }

    let expected_run_id = match run_id_override {
        Some(val) => val,
        None => env::var("NAM_RUN_ID")
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty()),
    };

    let path = explicit_path.unwrap_or_else(|| skip_log_path(&log_dir, expected_run_id.as_deref()));

    Ok((path, expected_run_id, format))
}

fn read_and_validate_skips(
    path: &Path,
    expected_run_id: Option<&str>,
) -> Result<SkipSummary, String> {
    let mut summary = SkipSummary {
        path: path.display().to_string(),
        ..SkipSummary::default()
    };

    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(summary),
        Err(e) => {
            return Err(format!("cannot stat skip log {}: {e}", path.display()));
        }
    }

    let meta = fs::metadata(path).map_err(|e| {
        format!(
            "skip log {} is unreadable or a broken symlink: {e}",
            path.display()
        )
    })?;
    if meta.is_dir() {
        return Err(format!("skip log {} is a directory", path.display()));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read skip log {}: {e}", path.display()))?;

    if content.is_empty() {
        return Ok(summary);
    }

    for (idx, raw_line) in content.split_terminator('\n').enumerate() {
        let line_no = idx + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.trim().is_empty() {
            return Err(format!(
                "malformed empty JSONL line at {}:{}",
                path.display(),
                line_no
            ));
        }
        let record: SkipRecord = serde_json::from_str(line).map_err(|e| {
            format!(
                "malformed skip JSONL at {}:{}: {e}",
                path.display(),
                line_no
            )
        })?;

        if record.pid == 0 {
            return Err(format!(
                "invalid pid=0 in skip record at {}:{}",
                path.display(),
                line_no
            ));
        }
        if record.test.trim().is_empty() {
            return Err(format!(
                "empty 'test' field in skip record at {}:{}",
                path.display(),
                line_no
            ));
        }
        if record.reason.trim().is_empty() {
            return Err(format!(
                "empty 'reason' field in skip record at {}:{}",
                path.display(),
                line_no
            ));
        }

        match (expected_run_id, record.run_id.as_deref()) {
            (Some(exp), Some(act)) if act == exp => {}
            (Some(exp), Some(act)) => {
                return Err(format!(
                    "run_id mismatch at {}:{}: expected '{exp}', got '{act}'",
                    path.display(),
                    line_no
                ));
            }
            (Some(exp), None) => {
                return Err(format!(
                    "missing run_id at {}:{}: expected '{exp}'",
                    path.display(),
                    line_no
                ));
            }
            (None, Some(act)) => {
                return Err(format!(
                    "unexpected run_id '{act}' in isolated skip log at {}:{}",
                    path.display(),
                    line_no
                ));
            }
            (None, None) => {}
        }

        summary.total += 1;
        if record.optional {
            summary.optional += 1;
        } else {
            summary.blocking += 1;
        }
    }

    Ok(summary)
}

fn main() {
    let (path, expected_run_id, format) = match parse_args() {
        Ok(parsed) => parsed,
        Err(e) => {
            eprintln!("nam_skips: {e}");
            exit(2);
        }
    };

    let summary = match read_and_validate_skips(&path, expected_run_id.as_deref()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: nam_skips: {e}");
            exit(1);
        }
    };

    match format {
        OutputFormat::Blocking => println!("{}", summary.blocking),
        OutputFormat::Counts => println!(
            "total={} optional={} blocking={}",
            summary.total, summary.optional, summary.blocking
        ),
        OutputFormat::Json => {
            let json = serde_json::to_string(&summary).unwrap_or_else(|e| {
                eprintln!("ERROR: nam_skips: failed to serialize summary: {e}");
                exit(1);
            });
            println!("{json}");
        }
    }
}
