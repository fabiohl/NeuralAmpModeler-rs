// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Process-level tests of the `nam_quality` CLI.
//!
//! Exercises the fail-closed CLI contract through `env!("CARGO_BIN_EXE_…")`
//! subprocesses — the same pattern as `receipt_test.rs`:
//! - exit 0: success;
//! - exit 1: run-time failure (gate violated, unreadable input, refused save);
//! - exit 2: usage error (unknown subcommand/flag, missing required flag).
//!
//! The `verify` fixtures are the acceptance scenarios, built from the
//! real `docs/quality-contract.json` (51 fidelity + 19 performance entries).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};

use neural_amp_modeler_rs::testing::qa::QualityContract;
use neural_amp_modeler_rs::testing::qa::verify::parse_verify_report;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nam-quality-test-{name}-{}-{n}",
        std::process::id()
    ))
}

fn write_file(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let mut f = fs::File::create(path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
}

fn contract_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/quality-contract.json")
}

fn load_contract() -> QualityContract {
    let content = fs::read_to_string(contract_path()).expect("read docs/quality-contract.json");
    QualityContract::from_json_str(&content).expect("contract must validate against the schema")
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_nam_quality")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("nam_quality must run")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).to_string()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).to_string()
}

// ── Report builders (mirror of the acceptance fixtures) ───────────────────────

fn phase_line(phase_id: &str, status: &str) -> String {
    format!(r#"{{"phase_id":"{phase_id}","status":"{status}"}}"#)
}

fn canonical_fidelity(entry: &neural_amp_modeler_rs::testing::qa::FidelityEntry) -> Option<Value> {
    let mut obj = json!({
        "kind": "fidelity",
        "label": entry.label,
        "esr": entry.esr_namcore,
        "snr_db": entry.snr_db,
        "mrstft": entry.mrstft,
    });
    if let Some(f64v) = entry.esr_f64 {
        obj["esr_f64"] = json!(f64v);
    }
    Some(obj)
}

fn canonical_latency(
    entry: &neural_amp_modeler_rs::testing::qa::PerformanceEntry,
) -> Option<Value> {
    Some(json!({
        "kind": "latency",
        "label": entry.label,
        "median_latency_us": entry.median_latency_us,
    }))
}

fn build_report(
    contract: &QualityContract,
    phases: &[(&str, &str)],
    fidelity: impl Fn(&neural_amp_modeler_rs::testing::qa::FidelityEntry) -> Option<Value>,
    latency: impl Fn(&neural_amp_modeler_rs::testing::qa::PerformanceEntry) -> Option<Value>,
) -> String {
    let mut lines: Vec<String> = phases
        .iter()
        .map(|(id, status)| phase_line(id, status))
        .collect();
    for entry in &contract.fidelity {
        if let Some(obj) = fidelity(entry) {
            lines.push(obj.to_string());
        }
    }
    for entry in &contract.performance {
        if let Some(obj) = latency(entry) {
            lines.push(obj.to_string());
        }
    }
    lines.join("\n")
}

const ALL_PASS: [(&str, &str); 4] = [
    ("golden_vectors", "PASS"),
    ("reference_oracle_f64", "PASS"),
    ("quick_parity", "PASS"),
    ("regression_gate", "PASS"),
];

const TARGET: &str = "bosslstm-1x16@48000:live";

fn write_report(report: &str) -> PathBuf {
    let path = temp_path("report.jsonl");
    write_file(&path, report);
    path
}

// ── CLI contract: exit codes and help ───────────────────────────────────────

#[test]
fn unknown_subcommand_and_flag_exit_2() {
    assert_eq!(run(&["--typo"]).status.code(), Some(2));
    assert_eq!(run(&["verify", "--bogus"]).status.code(), Some(2));
    assert_eq!(run(&["receipt", "nope"]).status.code(), Some(2));
    assert_eq!(run(&[""]).status.code(), Some(2), "empty subcommand");
}

#[test]
fn help_exits_0_and_lists_subcommands() {
    for args in [&["--help"][..], &["-h"][..], &["verify", "--help"][..]] {
        let out = run(args);
        assert_eq!(out.status.code(), Some(0), "{args:?}");
        assert!(!stdout(&out).is_empty(), "{args:?} must print help");
    }
    let help = stdout(&run(&["--help"]));
    for sub in ["ingest", "verify", "render", "receipt append", "save"] {
        assert!(help.contains(sub), "help must mention {sub}");
    }
}

#[test]
fn missing_required_flags_exit_2() {
    assert_eq!(run(&["verify"]).status.code(), Some(2));
    assert_eq!(run(&["verify", "--contract", "x"]).status.code(), Some(2));
    assert_eq!(run(&["receipt", "append"]).status.code(), Some(2));
    assert_eq!(run(&["save"]).status.code(), Some(2));
}

// ── verify: acceptance fixtures ─────────────────────────────────────────────

#[test]
fn verify_report_equals_contract_verdict_ok() {
    let contract = load_contract();
    let report = write_report(&build_report(
        &contract,
        &ALL_PASS,
        canonical_fidelity,
        canonical_latency,
    ));
    let out = run(&[
        "verify",
        "--contract",
        contract_path().to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("FIDELITY: OK"), "missing verdict: {text}");
    assert!(text.contains("PERFORMANCE: OK"), "missing verdict: {text}");
    assert!(!text.contains("CONTRACT VIOLATED"));
}

#[test]
fn verify_esr_above_safety_fidelity_fail() {
    let contract = load_contract();
    let report = write_report(&build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["esr"] = json!("1e-3");
            }
            Some(obj)
        },
        canonical_latency,
    ));
    let out = run(&[
        "verify",
        "--contract",
        contract_path().to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    let text = stdout(&out);
    assert!(text.contains("FIDELITY: FAIL (1 violation(s))"), "{text}");
    assert!(
        text.contains("ESR_NAMCORE"),
        "violation detail expected: {text}"
    );
    assert!(text.contains("CONTRACT VIOLATED"), "{text}");
}

#[test]
fn verify_regression_gate_not_verified_is_performance_domain() {
    let contract = load_contract();
    for gate in ["FAIL", "NOT_VERIFIED"] {
        let phases = [
            ("golden_vectors", "PASS"),
            ("reference_oracle_f64", "PASS"),
            ("quick_parity", "PASS"),
            ("regression_gate", gate),
        ];
        let report = write_report(&build_report(
            &contract,
            &phases,
            canonical_fidelity,
            canonical_latency,
        ));
        let out = run(&[
            "verify",
            "--contract",
            contract_path().to_str().unwrap(),
            "--report",
            report.to_str().unwrap(),
        ]);
        assert_eq!(out.status.code(), Some(1), "{gate}");
        let text = stdout(&out);
        assert!(text.contains("FIDELITY: OK"), "PERF-006: {text}");
        assert!(
            text.contains("PERFORMANCE: NOT_VERIFIED"),
            "missing NOT_VERIFIED verdict: {text}"
        );
        assert!(text.contains("CONTRACT VIOLATED"), "{text}");
    }
}

#[test]
fn verify_f64_violation_triggers_review_required() {
    let contract = load_contract();
    let report = write_report(&build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["esr_f64"] = json!("1e-3");
            }
            Some(obj)
        },
        canonical_latency,
    ));
    let out = run(&[
        "verify",
        "--contract",
        contract_path().to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    let text = stdout(&out);
    assert!(text.contains("REVIEW_REQUIRED"), "{text}");
    assert!(text.contains("CONTRACT VIOLATED"), "{text}");
}

#[test]
fn verify_unreadable_inputs_exit_1() {
    let out = run(&[
        "verify",
        "--contract",
        "/nonexistent/contract.json",
        "--report",
        "/nonexistent/report.jsonl",
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("ERROR:"), "{}", stderr(&out));

    // Malformed contract JSON is fail-closed too.
    let bad_contract = temp_path("bad.json");
    write_file(&bad_contract, "{not json}");
    let report = write_report(&build_report(
        &load_contract(),
        &ALL_PASS,
        canonical_fidelity,
        canonical_latency,
    ));
    let out = run(&[
        "verify",
        "--contract",
        bad_contract.to_str().unwrap(),
        "--report",
        report.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
}

// ── ingest ──────────────────────────────────────────────────────────────────

const PHASE_RECEIPT_FIXTURE: &str = r#"{"phase_id":"golden_vectors","status":"PASS","exit_code":0,"observed_records":51,"expected_records":51,"reason":"","run_id":"run-1"}
{"kind":"build_metadata","cargo_profile":"release","target_triple":"x86_64-unknown-linux-gnu","rustflags":"","rustc_version":"rustc 1.94.0","git_commit":"abc","git_dirty_state":false,"run_id":"run-1","effective_isa":"x86-64-v3 (AVX2/FMA/F16C/BMI)"}
{"phase_id":"regression_gate","status":"NOT_VERIFIED","exit_code":1,"observed_records":0,"expected_records":10,"reason":"MISSING_BASELINE","run_id":"run-1"}
"#;

const METRICS_FIXTURE: &str = r#"{"kind":"fidelity","label":"Boss WN Standard @48000 Live","esr":"2.31e-14","esr_db":"-138.7","snr_db":"110.7","mse":"5.3e-13","mrstft":"2.8e-5"}
{"kind":"fidelity","label":"BossLSTM-1x16 @48000 Live","esr":"1.02e-14","esr_db":"-139.8","snr_db":"111.2","mse":"4.1e-13","mrstft":"2.6e-5"}
"#;

const LATENCY_FIXTURE: &str = r#"{"kind":"latency","label":"WaveNet Standard CH16","median_latency_us":36.9}
"#;

#[test]
fn ingest_merges_streams_into_parseable_report() {
    let receipt = temp_path("receipt.jsonl");
    let metrics = temp_path("metrics.jsonl");
    let latency = temp_path("latency.jsonl");
    let out = temp_path("report.jsonl");
    write_file(&receipt, PHASE_RECEIPT_FIXTURE);
    write_file(&metrics, METRICS_FIXTURE);
    write_file(&latency, LATENCY_FIXTURE);

    let run_out = run(&[
        "ingest",
        "--receipt",
        receipt.to_str().unwrap(),
        "--metrics",
        metrics.to_str().unwrap(),
        "--latency",
        latency.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(
        run_out.status.code(),
        Some(0),
        "stderr: {}",
        stderr(&run_out)
    );

    let content = fs::read_to_string(&out).unwrap();
    let report = parse_verify_report(&content).expect("merged report must parse");
    assert_eq!(
        report.phases.len(),
        2,
        "build_metadata must pass through and be skipped"
    );
    assert_eq!(report.phases[0].phase_id, "golden_vectors");
    assert_eq!(report.phases[1].status, "NOT_VERIFIED");
    assert_eq!(report.fidelity.len(), 2);
    assert_eq!(report.fidelity[0].label, "Boss WN Standard @48000 Live");
    assert_eq!(report.latency.len(), 1);
    assert_eq!(report.latency[0].median_latency_us, 36.9);
    assert!(
        content.contains(r#""kind":"build_metadata""#),
        "provenance record must survive the merge verbatim"
    );

    // stdout mode carries the same bytes.
    let run_out = run(&[
        "ingest",
        "--receipt",
        receipt.to_str().unwrap(),
        "--metrics",
        metrics.to_str().unwrap(),
        "--latency",
        latency.to_str().unwrap(),
    ]);
    assert_eq!(run_out.status.code(), Some(0));
    assert_eq!(stdout(&run_out), content);
}

#[test]
fn ingest_fails_closed_on_malformed_lines() {
    let receipt = temp_path("receipt.jsonl");
    write_file(&receipt, "not json\n");
    let out = run(&["ingest", "--receipt", receipt.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("malformed phase receipt line 1"));

    // Missing phase_id/status in a receipt line is fail-closed too.
    write_file(&receipt, r#"{"phase_id":1,"status":"PASS"}"#);
    let out = run(&["ingest", "--receipt", receipt.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));

    // Malformed metrics stream aborts the whole ingest.
    let metrics = temp_path("metrics.jsonl");
    write_file(&metrics, "{\"broken\":");
    let out = run(&[
        "ingest",
        "--receipt",
        receipt.to_str().unwrap(),
        "--metrics",
        metrics.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
}

#[test]
fn ingest_missing_receipt_is_usage_error() {
    assert_eq!(run(&["ingest"]).status.code(), Some(2));
    let out = run(&["ingest", "--receipt", "/nonexistent.jsonl"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "unreadable receipt is a run failure"
    );
}

// ── render ──────────────────────────────────────────────────────────────────

fn report_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qa/report.jsonl")
}

#[test]
fn render_cli_contract_exit_codes() {
    // Missing required --report → usage error.
    assert_eq!(run(&["render"]).status.code(), Some(2));
    assert_eq!(run(&["render", "--bogus"]).status.code(), Some(2));
    // Mutually exclusive styles → usage error.
    assert_eq!(
        run(&["render", "--report", "x", "--ansi", "--plain"])
            .status
            .code(),
        Some(2)
    );
    // Unreadable report → run-time failure.
    assert_eq!(
        run(&["render", "--report", "/nonexistent/report.jsonl"])
            .status
            .code(),
        Some(1)
    );
}

#[test]
fn render_plain_matches_committed_golden() {
    let out = run(&[
        "render",
        "--report",
        report_fixture_path().to_str().unwrap(),
        "--plain",
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("QUICK SUMMARY (non-specialist)"), "{text}");
    assert!(text.contains("AUDIO FIDELITY"), "{text}");
    assert!(text.contains("PERFORMANCE — Block Latency"), "{text}");
    // Plain render carries no ANSI escape sequences.
    assert!(
        !text.contains('\u{1b}'),
        "plain render must have no ANSI: {text}"
    );
}

// ── classify ────────────────────────────────────────────────────────────────

#[test]
fn classify_subcommand_delegates_to_the_single_classifier() {
    let cases: &[(&[&str], &str)] = &[
        (&["--status", "PASS", "--reason", ""], "PASS"),
        (
            &["--status", "FAIL", "--reason", "MISSING_BASELINE"],
            "NOT_VERIFIED",
        ),
        (
            &["--status", "FAIL", "--reason", "INCOMPARABLE_ENVIRONMENT"],
            "NOT_VERIFIED",
        ),
        (
            &["--status", "FAIL", "--reason", "REGRESSION_DETECTED"],
            "FAIL",
        ),
        (&["--status", "SKIP_CAPABILITY", "--reason", "x"], "FAIL"),
    ];
    for (args, expected) in cases {
        let mut full = vec!["classify"];
        full.extend_from_slice(args);
        let out = run(&full);
        assert_eq!(out.status.code(), Some(0), "{args:?}");
        assert_eq!(stdout(&out).trim(), *expected, "{args:?}");
    }
    // Missing --status is a usage error.
    assert_eq!(run(&["classify"]).status.code(), Some(2));
}

// ── receipt append ──────────────────────────────────────────────────────────

#[test]
fn receipt_append_writes_bash_compatible_serde_line() {
    let out = temp_path("phase_receipt.jsonl");
    let run_out = run(&[
        "receipt",
        "append",
        "--phase-id",
        "golden_vectors",
        "--status",
        "PASS",
        "--exit-code",
        "0",
        "--observed-records",
        "51",
        "--expected-records",
        "51",
        "--reason",
        "ok",
        "--run-id",
        "run-9",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(run_out.status.code(), Some(0), "{}", stderr(&run_out));

    // Field order and schema must reproduce the bash `printf` of
    // `dashboard_phase_receipt` (utils/_lib.sh:96).
    let line = fs::read_to_string(&out).unwrap();
    assert_eq!(
        line.trim(),
        r#"{"phase_id":"golden_vectors","status":"PASS","exit_code":0,"observed_records":51,"expected_records":51,"reason":"ok","run_id":"run-9"}"#
    );
    let obj: Value = serde_json::from_str(line.trim()).unwrap();
    for field in [
        "phase_id",
        "status",
        "exit_code",
        "observed_records",
        "expected_records",
        "reason",
        "run_id",
    ] {
        assert!(obj.get(field).is_some(), "missing schema field {field}");
    }

    // Appending accumulates lines; a second call must not clobber the first.
    let run_out = run(&[
        "receipt",
        "append",
        "--phase-id",
        "quick_parity",
        "--status",
        "PASS",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(run_out.status.code(), Some(0));
    let content = fs::read_to_string(&out).unwrap();
    assert_eq!(content.lines().count(), 2);

    // Unknown status and bad numbers are usage errors (exit 2).
    let bad = run(&[
        "receipt",
        "append",
        "--phase-id",
        "x",
        "--status",
        "MAYBE",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(bad.status.code(), Some(2));
    let bad = run(&[
        "receipt",
        "append",
        "--phase-id",
        "x",
        "--status",
        "PASS",
        "--exit-code",
        "many",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(bad.status.code(), Some(2));
}

// ── save ────────────────────────────────────────────────────────────────────

fn copy_contract(dest: &Path) {
    fs::copy(contract_path(), dest).expect("copy contract to temp path");
}

fn receipt_with(statuses: &[(&str, &str)]) -> PathBuf {
    let path = temp_path("save_receipt.jsonl");
    let mut content = String::new();
    for (phase_id, status) in statuses {
        content.push_str(&phase_line(phase_id, status));
        content.push('\n');
    }
    write_file(&path, &content);
    path
}

#[test]
fn save_promotes_contract_atomically_when_fidelity_ok() {
    // regression_gate NOT_VERIFIED/FAIL must NOT block saving (PERF-006);
    // the provenance record without phase_id must be ignored.
    for gate in ["NOT_VERIFIED", "FAIL", "PASS"] {
        let contract = temp_path("save_contract.json");
        copy_contract(&contract);
        let receipt = receipt_with(&[
            ("golden_vectors", "PASS"),
            ("reference_oracle_f64", "PASS"),
            ("quick_parity", "PASS"),
            ("regression_gate", gate),
        ]);
        let content = format!(
            "{}{{\"kind\":\"build_metadata\",\"run_id\":\"x\"}}\n",
            fs::read_to_string(&receipt).unwrap()
        );
        write_file(&receipt, &content);
        let out = run(&[
            "save",
            "--contract",
            contract.to_str().unwrap(),
            "--receipt",
            receipt.to_str().unwrap(),
        ]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "gate={gate}: stderr {}",
            stderr(&out)
        );
        let saved = fs::read_to_string(&contract).unwrap();
        QualityContract::from_json_str(&saved).expect("saved file must be a valid contract");
        assert!(
            !fs::read_dir(contract.parent().unwrap()).unwrap().any(|e| e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp.")),
            "no temp file may be left behind"
        );
    }
}

#[test]
fn save_refuses_on_fidelity_phase_fail_and_leaves_file_untouched() {
    let contract = temp_path("save_contract.json");
    copy_contract(&contract);
    let original = fs::read(&contract).unwrap();
    let receipt = receipt_with(&[
        ("golden_vectors", "FAIL"),
        ("reference_oracle_f64", "PASS"),
        ("quick_parity", "PASS"),
        ("regression_gate", "PASS"),
    ]);
    let out = run(&[
        "save",
        "--contract",
        contract.to_str().unwrap(),
        "--receipt",
        receipt.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        stderr(&out).contains("contract NOT saved"),
        "{}",
        stderr(&out)
    );
    assert_eq!(
        fs::read(&contract).unwrap(),
        original,
        "refused save must not touch the contract"
    );
}

#[test]
fn save_rejects_invalid_contract_and_missing_flags() {
    // Invalid contract payload → exit 1, no write.
    let contract = temp_path("bad_contract.json");
    write_file(&contract, "{not json}");
    let receipt = receipt_with(&[("golden_vectors", "PASS")]);
    let out = run(&[
        "save",
        "--contract",
        contract.to_str().unwrap(),
        "--receipt",
        receipt.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(1));
    assert_eq!(fs::read_to_string(&contract).unwrap(), "{not json}");

    // Missing flags → exit 2.
    assert_eq!(run(&["save", "--contract", "x"]).status.code(), Some(2));
    assert_eq!(run(&["save"]).status.code(), Some(2));
}
