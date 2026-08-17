// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Process-level tests of the `nam_perf_gate` CLI (Sprint S3.T3).
//!
//! Exercises the fail-closed CLI contract through `env!("CARGO_BIN_EXE_…")`
//! subprocesses — the same pattern as `quality_cli_test.rs`:
//! - exit 0: success;
//! - exit 1: run-time failure (missing/incomparable baseline, coverage gap,
//!   I/O);
//! - exit 2: usage error (unknown subcommand/flag, missing required flag).
//!
//! Every subcommand is pointed at temp paths — the repo's
//! `.performance-baselines/` and `target/logs/` are never touched.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use neural_amp_modeler_rs::testing::qa::env::EnvProbe;
use neural_amp_modeler_rs::testing::qa::fingerprint::{FIELD_CPU_MODEL, Fingerprint};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nam-perf-gate-test-{name}-{}-{n}",
        std::process::id()
    ))
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, content).unwrap();
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_nam_perf_gate")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("nam_perf_gate must run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

/// Acceptance (S3.T3): `compare` without a stored fingerprint fails with
/// exit 1 and the `MISSING_BASELINE` token.
#[test]
fn compare_without_fingerprint_exits_1_with_missing_baseline() {
    let store = temp_path("store");
    let out = run(&["compare", "--baseline-dir", store.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stdout(&out).contains("MISSING_BASELINE"),
        "unexpected stdout: {}",
        stdout(&out)
    );
}

/// `probe` writes the serde fingerprint (creating parent dirs) and a
/// subsequent `compare` against it passes — the self-comparison green path.
#[test]
fn probe_writes_fingerprint_and_compare_passes() {
    let probe = EnvProbe::probe();
    if probe.frequency_governor != "performance" {
        eprintln!(
            "skipping: current governor is '{}' (not 'performance')",
            probe.frequency_governor
        );
        return;
    }

    let path = temp_path("fp.json");
    let out = run(&[
        "probe",
        "--out",
        path.to_str().unwrap(),
        "--bench-core",
        "3",
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let content = fs::read_to_string(&path).unwrap();
    let parsed = Fingerprint::from_json_str(&content).expect("probe output must be valid JSON");
    assert_eq!(parsed.cpu_model, probe.cpu_model);
    assert_eq!(parsed.bench_core, "3");

    let out = run(&[
        "compare",
        "--baseline",
        path.to_str().unwrap(),
        "--bench-core",
        "3",
    ]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        stdout(&out),
        stderr(&out)
    );
    assert!(stdout(&out).contains("comparable"));
}

/// A baseline recorded on a different environment fails `compare` with the
/// typed `INCOMPARABLE_ENVIRONMENT` reason naming the first drifted field
/// (bash order: `cpu_model` first).
#[test]
fn compare_reports_incomparable_environment_with_field() {
    let mut probe = EnvProbe::probe();
    probe.cpu_model = "Intel(R) Xeon(R) Gold 6338".to_string();
    let baseline = Fingerprint::from_env_probe(&probe, "");
    let path = temp_path("mismatch.json");
    write_file(&path, &baseline.to_json_pretty().unwrap());

    let out = run(&["compare", "--baseline", path.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    let text = stdout(&out);
    assert!(
        text.contains("INCOMPARABLE_ENVIRONMENT"),
        "unexpected stdout: {text}"
    );
    assert!(
        text.contains(FIELD_CPU_MODEL),
        "reason must name the field: {text}"
    );
    assert!(
        text.contains("Intel(R) Xeon(R) Gold 6338"),
        "reason must carry the baseline value: {text}"
    );
}

/// A corrupt fingerprint file fails `compare` fail-closed with exit 1.
#[test]
fn compare_with_corrupt_fingerprint_exits_1() {
    let path = temp_path("corrupt.json");
    write_file(&path, "{not json");
    let out = run(&["compare", "--baseline", path.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("ERROR"));
}

/// `coverage` passes when every executed benchmark id has a series dir and
/// fails `BASELINE_COVERAGE_GAP` (listing ids) on gaps — F-24.
#[test]
fn coverage_ok_and_gap() {
    let crit = temp_path("crit");
    for id in ["RT_A", "RT_B"] {
        fs::create_dir_all(crit.join(id).join("ci-baseline")).unwrap();
    }
    let log = temp_path("crit.log");
    write_file(
        &log,
        "Benchmarking RT_A: Warming up for 1.0000 s\n\
         Benchmarking RT_B: Collecting 100 samples\n\
         Benchmarking RT_C: Warming up for 1.0000 s\n",
    );

    let out = run(&[
        "coverage",
        "--log",
        log.to_str().unwrap(),
        "--root",
        crit.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    let text = stdout(&out);
    assert!(
        text.contains("BASELINE_COVERAGE_GAP"),
        "unexpected stdout: {text}"
    );
    assert!(text.contains("RT_C"), "missing id must be listed: {text}");

    fs::create_dir_all(crit.join("RT_C").join("ci-baseline")).unwrap();
    let out = run(&[
        "coverage",
        "--log",
        log.to_str().unwrap(),
        "--root",
        crit.to_str().unwrap(),
    ]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "stdout: {} stderr: {}",
        stdout(&out),
        stderr(&out)
    );
    assert!(stdout(&out).contains("coverage ok"));
}

/// A log with no parseable `Benchmarking <id>:` line — or no log at all —
/// is the fail-closed blind gate (F-24).
#[test]
fn coverage_blind_gate_fails_closed() {
    let crit = temp_path("crit");
    fs::create_dir_all(crit.join("RT_A").join("ci-baseline")).unwrap();

    let garbage = temp_path("garbage.log");
    write_file(&garbage, "garbage log with no criterion lines\n");
    let out = run(&[
        "coverage",
        "--log",
        garbage.to_str().unwrap(),
        "--root",
        crit.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stdout(&out).contains("BASELINE_COVERAGE_GAP"));

    let absent = temp_path("absent.log");
    let out = run(&[
        "coverage",
        "--log",
        absent.to_str().unwrap(),
        "--root",
        crit.to_str().unwrap(),
    ]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "absent log must be the blind gate"
    );
}

/// `persist-baseline` / `restore-baseline` move top-level series between the
/// criterion root and the store, sanitizing nested dirs (scenario 4 CLI
/// surface).
#[test]
fn persist_and_restore_baselines_via_cli() {
    let criterion = temp_path("criterion");
    let store = temp_path("store");
    let marker = criterion
        .join("RT_Dummy")
        .join("ci-baseline")
        .join("marker.txt");
    write_file(&marker, "top-level");

    let out = run(&[
        "persist-baseline",
        "--criterion-root",
        criterion.to_str().unwrap(),
        "--baseline-dir",
        store.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(stdout(&out).contains("persisted 1 baseline series"));
    assert_eq!(
        fs::read_to_string(
            store
                .join("RT_Dummy")
                .join("ci-baseline")
                .join("marker.txt")
        )
        .unwrap(),
        "top-level"
    );

    // A nested leftover in the store is sanitized by restore.
    write_file(
        &store
            .join("RT_Dummy")
            .join("ci-baseline")
            .join("ci-baseline")
            .join("marker.txt"),
        "nested",
    );
    let out = run(&[
        "restore-baseline",
        "--criterion-root",
        criterion.to_str().unwrap(),
        "--baseline-dir",
        store.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    assert!(stdout(&out).contains("restored 1 baseline series"));
    assert_eq!(
        fs::read_to_string(
            criterion
                .join("RT_Dummy")
                .join("ci-baseline")
                .join("marker.txt")
        )
        .unwrap(),
        "top-level"
    );
    assert!(
        !criterion
            .join("RT_Dummy")
            .join("ci-baseline")
            .join("ci-baseline")
            .exists()
    );
}

/// `receipt append` writes the byte-compatible `dashboard_phase_receipt`
/// schema; `receipt summary` derives and appends the `overall` verdict
/// (PASS iff every phase PASS).
#[test]
fn receipt_append_and_summary() {
    let receipt = temp_path("regression_phase_receipt.jsonl");
    let out = run(&[
        "receipt",
        "append",
        "--phase-id",
        "regression_baseline_created",
        "--status",
        "PASS",
        "--exit-code",
        "0",
        "--out",
        receipt.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let out = run(&[
        "receipt",
        "append",
        "--phase-id",
        "regression_check",
        "--status",
        "FAIL",
        "--exit-code",
        "1",
        "--reason",
        "MISSING_BASELINE",
        "--out",
        receipt.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));

    let content = fs::read_to_string(&receipt).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("\"phase_id\":\"regression_baseline_created\""));
    assert!(lines[1].contains("\"status\":\"FAIL\""));
    assert!(lines[1].contains("\"reason\":\"MISSING_BASELINE\""));
    let value: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
    assert_eq!(value["exit_code"], 1);

    let out = run(&["receipt", "summary", "--out", receipt.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let content = fs::read_to_string(&receipt).unwrap();
    let overall: serde_json::Value = serde_json::from_str(content.lines().last().unwrap()).unwrap();
    assert_eq!(overall["phase_id"], "overall");
    assert_eq!(overall["status"], "FAIL");
    assert_eq!(overall["exit_code"], 1);

    // An all-PASS receipt derives overall PASS.
    let pass_only = temp_path("pass.jsonl");
    write_file(
        &pass_only,
        r#"{"phase_id":"regression_baseline_created","status":"PASS","exit_code":0,"observed_records":1,"expected_records":1,"reason":"","run_id":""}
"#,
    );
    let out = run(&["receipt", "summary", "--out", pass_only.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let content = fs::read_to_string(&pass_only).unwrap();
    let overall: serde_json::Value = serde_json::from_str(content.lines().last().unwrap()).unwrap();
    assert_eq!(overall["status"], "PASS");
    assert_eq!(overall["exit_code"], 0);
}

/// Fail-closed usage contract: unknown subcommand/flag and missing required
/// flags exit 2; `--help` exits 0.
#[test]
fn usage_errors_exit_2() {
    let no_args = run(&[]);
    assert_eq!(no_args.status.code(), Some(2));

    let unknown = run(&["frobnicate"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(stderr(&unknown).contains("unknown subcommand"));

    let bad_flag = run(&["compare", "--typo", "x"]);
    assert_eq!(bad_flag.status.code(), Some(2));

    let missing_log = run(&["coverage"]);
    assert_eq!(missing_log.status.code(), Some(2));
    assert!(stderr(&missing_log).contains("missing required flag --log"));

    let missing_status = run(&["receipt", "append", "--phase-id", "p"]);
    assert_eq!(missing_status.status.code(), Some(2));

    let help = run(&["--help"]);
    assert_eq!(help.status.code(), Some(0));
    assert!(stdout(&help).contains("nam_perf_gate"));
}
