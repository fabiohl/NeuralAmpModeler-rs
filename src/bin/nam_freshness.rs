// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Lightweight CLI wrapper around `src/testing/freshness.rs`.
//!
//! Replaces the bulk of `utils/_lib.sh::check_freshness` with a portable Rust
//! implementation. Exit codes:
//!
//! - 0: freshness gate passed (or `warn-only` mode absorbed failures).
//! - 1: gate failed in the requested mode.
//! - 2: bad arguments.

use std::env;
use std::path::PathBuf;
use std::process::exit;
use std::str::FromStr;

use neural_amp_modeler_rs::testing::freshness::{
    FreshnessMode, check_artifact_freshness_mtime, check_freshness,
};

const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

struct Style {
    green: &'static str,
    yellow: &'static str,
    red: &'static str,
    bold: &'static str,
    reset: &'static str,
}

impl Style {
    fn no_color() -> Self {
        Self {
            green: "",
            yellow: "",
            red: "",
            bold: "",
            reset: "",
        }
    }

    fn color() -> Self {
        Self {
            green: GREEN,
            yellow: YELLOW,
            red: RED,
            bold: BOLD,
            reset: RESET,
        }
    }
}

fn print_help() {
    println!("Usage: nam_freshness [OPTIONS] [MODE]");
    println!();
    println!("Modes:");
    println!("  warn-only       Emit warnings but always exit 0");
    println!("  artifacts-hard  Fail on stale/missing/orphan artifacts (default)");
    println!("  hard-fail       Fail on artifact integrity or generator provenance drift");
    println!();
    println!("Options:");
    println!("  --root PATH     Root directory containing tests/fixtures (default: current dir)");
    println!("  --mtime SRC... -- ARTIFACT...");
    println!("                  Check whether artifacts are older than sources");
    println!("  --no-color      Disable ANSI colors");
    println!("  -h, --help      Print this help");
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        exit(2);
    }

    let mut mode = FreshnessMode::HardFail;
    let mut root = PathBuf::from(".");
    let mut no_color = env::var_os("NO_COLOR").is_some();
    let mut mtime_mode = false;
    let mut mtime_sources: Vec<PathBuf> = Vec::new();
    let mut mtime_artifacts: Vec<PathBuf> = Vec::new();
    let mut after_sep = false;

    let mut iter = args.into_iter().peekable();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_help();
                exit(0);
            }
            "--no-color" => no_color = true,
            "--root" => {
                let value = iter
                    .next()
                    .unwrap_or_else(|| die("--root requires a path", 2));
                root = PathBuf::from(value);
            }
            "--mtime" => {
                mtime_mode = true;
            }
            "--" => {
                after_sep = true;
            }
            _ => {
                if let Ok(parsed) = FreshnessMode::from_str(&arg) {
                    mode = parsed;
                } else if mtime_mode && !after_sep {
                    mtime_sources.push(PathBuf::from(arg));
                } else if mtime_mode {
                    mtime_artifacts.push(PathBuf::from(arg));
                } else {
                    eprintln!("unknown argument: {arg}");
                    exit(2);
                }
            }
        }
    }

    let s = if no_color {
        Style::no_color()
    } else {
        Style::color()
    };

    if mtime_mode {
        match check_artifact_freshness_mtime(&mtime_sources, &mtime_artifacts) {
            Ok(stale) if stale.is_empty() => {
                println!("{}✓{} artifact(s) newer than sources", s.green, s.reset);
                exit(0);
            }
            Ok(stale) => {
                println!(
                    "{}{}STALE:{} {} artifact(s) older than sources",
                    s.red,
                    s.bold,
                    s.reset,
                    stale.len()
                );
                for p in stale {
                    println!("  ▲ STALE: {}", p.display());
                }
                exit(1);
            }
            Err(e) => {
                eprintln!("{}❌{} freshness error: {e}", s.red, s.reset);
                exit(2);
            }
        }
    }

    match check_freshness(&root, mode) {
        Ok(outcome) => {
            // Print per-item diagnostics.
            for p in &outcome.missing {
                println!(
                    "  {}▲ MISSING:{} {} — expected file not found on disk",
                    s.yellow,
                    s.reset,
                    p.display()
                );
            }
            for p in &outcome.stale {
                println!(
                    "  {}▲ STALE:{} {} — hash changed",
                    s.yellow,
                    s.reset,
                    p.display()
                );
            }
            for p in &outcome.orphans {
                println!(
                    "  {}▲ ORPHAN:{} {} — model file not registered in freshness manifest",
                    s.yellow,
                    s.reset,
                    p.display()
                );
            }
            for p in &outcome.generator_drift {
                println!(
                    "  {}⚠ GENERATOR CHANGED:{} {} — fixtures may be stale; re-run golden_gen_build.sh",
                    s.yellow,
                    s.reset,
                    p.display()
                );
            }
            for drift in &outcome.toolchain_drift {
                println!("  {}⚠ TOOLCHAIN DRIFT:{} {}", s.yellow, s.reset, drift);
            }

            if outcome.is_ok() {
                let mut details = String::new();
                if outcome.artifact_integrity_ok {
                    details.push_str("artifact_integrity=OK ");
                }
                if outcome.generator_provenance_ok {
                    details.push_str("generator_provenance=OK ");
                } else {
                    details.push_str("generator_provenance=DRIFT ");
                }
                if outcome.toolchain_provenance_ok {
                    details.push_str("toolchain_provenance=OK");
                } else {
                    details.push_str("toolchain_provenance=DRIFT");
                }
                println!(
                    "{}✓{} Freshness gate passed ({}).",
                    s.green,
                    s.reset,
                    details.trim()
                );
                exit(0);
            }

            // Gate failed: print actionable summary.
            let prefix = format!("{}{}❌{}", s.red, s.bold, s.reset);
            if !outcome.missing.is_empty() {
                println!(
                    "  {prefix} {} expected file(s) missing. Run './tests/fixtures/golden_gen_build.sh' to generate missing golden vectors.",
                    outcome.missing.len()
                );
            }
            if !outcome.stale.is_empty() {
                println!(
                    "  {prefix} {} file(s) stale (artifact integrity). Run './tests/fixtures/golden_gen_build.sh' to regenerate fixtures and manifest.",
                    outcome.stale.len()
                );
            }
            if !outcome.orphans.is_empty() {
                println!(
                    "  {prefix} {} model(s) not registered in manifest. Add them to GOLDEN_GEN_CATALOG in src/testing/catalog.rs and regenerate.",
                    outcome.orphans.len()
                );
            }
            if !outcome.generator_drift.is_empty() {
                if mode.generator_hard() {
                    println!(
                        "  {prefix} {} generator(s) changed — fixture provenance stale. Re-run './tests/fixtures/golden_gen_build.sh'.",
                        outcome.generator_drift.len()
                    );
                } else {
                    println!(
                        "  {}⚠{} {} generator(s) changed — provenance drift (non-blocking in this mode).",
                        s.yellow,
                        s.reset,
                        outcome.generator_drift.len()
                    );
                }
            }
            if !outcome.toolchain_drift.is_empty() {
                println!(
                    "  {}⚠{} toolchain drift (informational, non-blocking).",
                    s.yellow, s.reset
                );
            }

            let should_fail = mode.artifact_hard() && !outcome.artifact_integrity_ok
                || mode.generator_hard() && !outcome.generator_provenance_ok;

            if should_fail {
                exit(1);
            }
            exit(0);
        }
        Err(e) => {
            println!("{}❌{} Freshness error: {e}", s.red, s.reset);
            exit(1);
        }
    }
}

fn die(message: &str, code: i32) -> ! {
    eprintln!("{message}");
    exit(code);
}
