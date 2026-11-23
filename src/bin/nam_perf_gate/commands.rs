// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Subcommands: probe, compare, coverage, expected-count, persist/restore baseline, pick-core.

use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

use neural_amp_modeler_rs::testing::qa::baseline_store::{persist_baseline, restore_baseline};
use neural_amp_modeler_rs::testing::qa::coverage::{
    BaselineCoverageGap, executed_bench_ids, expected_bench_ids_for_suite,
    missing_baseline_coverage, missing_expected_benchmarks,
};
use neural_amp_modeler_rs::testing::qa::fingerprint::{Fingerprint, FingerprintError};
use neural_amp_modeler_rs::testing::qa::pick_core::{
    DEFAULT_SYSFS_CPU_DIR, is_cpu_online, parse_cpu_list, resolve_bench_core,
};

use crate::cli::{
    DEFAULT_BASELINE_DIR, HELP_REQUEST, baseline_name, bench_core, default_criterion_root,
    parse_flags, parse_flags_full, print_help, required, run_error, usage_error,
};

// ── probe ────────────────────────────────────────────────────────────────────

pub(crate) fn cmd_probe(args: &[String]) {
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

pub(crate) fn cmd_compare(args: &[String]) {
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

pub(crate) fn cmd_coverage(args: &[String]) {
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

pub(crate) fn cmd_expected_count(args: &[String]) {
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

// ── baseline persistence ─────────────────────────────────────────────────────

pub(crate) fn cmd_persist_baseline(args: &[String]) {
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

pub(crate) fn cmd_restore_baseline(args: &[String]) {
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

// ── pick-core ────────────────────────────────────────────────────────────────

pub(crate) fn cmd_pick_core(args: &[String]) {
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
