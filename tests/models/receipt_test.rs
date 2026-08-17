// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration test suite for the machine-readable capability receipt and skip classification.
//!
//! Validates:
//! - Exact count invariants: 51 canonical entries, 61 catalog paths, 45 supported, 6 unsupported.
//! - Zero unexpected FAILED stages (`receipt.has_unexpected_failures() == false`).
//! - JSON serialization roundtrip and schema validity.
//! - ASCII/Markdown audit table rendering.
//! - Long-suite JSONL audit receipt (Sprint S3-T04): schema roundtrip, log
//!   test counting, summary verdict derivation, and the `nam_long_receipt`
//!   CLI end-to-end (append → summary → validate).

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use neural_amp_modeler_rs::testing::receipt::{
    CapabilityReceipt, LongAuditReceipt, LongPhaseReceipt, LongPhaseStatus,
    count_tests_executed_from_log, generate_capability_receipt,
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nam-receipt-test-{name}-{}-{n}",
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

fn mk_receipt(phase_id: &str, status: LongPhaseStatus) -> LongPhaseReceipt {
    LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: format!("phase {phase_id}"),
        status,
        duration_ms: 1000,
        tests_executed: 2,
        gaps: vec![],
        timestamp: "2026-08-14T00:00:00Z".to_string(),
    }
}

#[test]
fn test_capability_receipt_generation_and_invariants() {
    let receipt = generate_capability_receipt();

    // 1. Count Invariants
    assert_eq!(receipt.total_canonical_models, 51);
    assert_eq!(receipt.total_catalog_paths, 61);
    assert_eq!(receipt.supported_count, 46);
    assert_eq!(receipt.unsupported_count, 5);
    assert_eq!(receipt.entries.len(), 51);

    // 2. Zero unexpected FAILED stages
    assert!(
        !receipt.has_unexpected_failures(),
        "Capability receipt contains unexpected FAILED stages! Table:\n{}",
        receipt.render_table()
    );

    // 3. JSON roundtrip & validity
    let json_str = receipt.render_json();
    assert!(!json_str.is_empty());
    let parsed: CapabilityReceipt =
        serde_json::from_str(&json_str).expect("Receipt JSON must be valid deserializable JSON");
    assert_eq!(parsed.total_canonical_models, 51);
    assert_eq!(parsed.entries.len(), 51);

    // 4. Audit Table Rendering
    let table = receipt.render_table();
    assert!(table.contains("=== Fixture Catalog Capability Receipt ==="));
    assert!(table.contains("Total Canonical: 51"));

    eprintln!("Successfully validated capability receipt for all 51 canonical model identities.");
}

// ── Long-duration audit suite receipt (Sprint S3-T04) ──────────────────────

#[test]
fn test_long_phase_receipt_jsonl_schema_and_roundtrip() {
    let entry = LongPhaseReceipt {
        phase_id: "phase4".to_string(),
        name: "RT Deadline Gate (deterministic)".to_string(),
        status: LongPhaseStatus::Inconclusive,
        duration_ms: 42_000,
        tests_executed: 3,
        gaps: vec!["inconclusive_environment".to_string()],
        timestamp: "2026-08-14T00:00:00Z".to_string(),
    };

    let line = entry.render_jsonl_line();
    let parsed: LongPhaseReceipt =
        serde_json::from_str(&line).expect("LongPhaseReceipt line must be valid schema JSON");
    assert_eq!(parsed, entry);

    // Schema fields required by the invariant: phase_id, name, status,
    // duration_ms, tests_executed, gaps, timestamp.
    let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
    for field in [
        "phase_id",
        "name",
        "status",
        "duration_ms",
        "tests_executed",
        "gaps",
        "timestamp",
    ] {
        assert!(obj.get(field).is_some(), "missing schema field: {field}");
    }

    assert_eq!(
        LongPhaseStatus::from_str(obj["status"].as_str().unwrap()).unwrap(),
        LongPhaseStatus::Inconclusive
    );
}

#[test]
fn test_long_receipt_jsonl_parse_rejects_invalid_line() {
    let a = mk_receipt("phase1", LongPhaseStatus::Passed);
    let bad = format!("{}\n{{not json}}\n", a.render_jsonl_line());
    let err = LongAuditReceipt::parse_jsonl(&bad).unwrap_err();
    assert!(err.to_string().contains("invalid JSONL line 2"), "{err}");
}

#[test]
fn test_long_receipt_counts_executed_tests_from_log() {
    let log = temp_path("count.log");
    write_file(
        &log,
        "test result: ok. 123 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2s\n\
         test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.5s\n",
    );
    assert_eq!(count_tests_executed_from_log(&log), 123 + 2 + 1);

    // Bench fallback: criterion-style `time:` lines count when no test-result
    // line exists (mirrors _lib.sh::assert_ran_tests).
    let bench = temp_path("bench.log");
    write_file(
        &bench,
        "soak_test/bench1  time:   [1.1 ms 1.2 ms 1.3 ms]\n\
         soak_test/bench2  time:   [2.1 ms 2.2 ms 2.3 ms]\n",
    );
    assert_eq!(count_tests_executed_from_log(&bench), 2);

    assert_eq!(count_tests_executed_from_log(&temp_path("missing.log")), 0);
}

#[test]
fn test_long_receipt_summary_derives_verdicts() {
    let all_pass = LongAuditReceipt {
        phases: vec![
            mk_receipt("phase1", LongPhaseStatus::Passed),
            mk_receipt("phase2", LongPhaseStatus::Passed),
        ],
    };
    assert_eq!(all_pass.summary_receipt().status, LongPhaseStatus::Passed);
    assert!(all_pass.summary_receipt().gaps.is_empty());

    let with_fail = LongAuditReceipt {
        phases: vec![mk_receipt("phase2", LongPhaseStatus::Failed)],
    };
    assert_eq!(with_fail.summary_receipt().status, LongPhaseStatus::Failed);

    let with_gaps = LongAuditReceipt {
        phases: vec![
            mk_receipt("phase4", LongPhaseStatus::Inconclusive),
            mk_receipt("phase5", LongPhaseStatus::SkipCapability),
        ],
    };
    let summary = with_gaps.summary_receipt();
    assert_eq!(summary.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(
        summary.gaps,
        vec!["phase4:INCONCLUSIVE", "phase5:SKIP_CAPABILITY"]
    );
    assert_eq!(summary.duration_ms, 2000);
    assert_eq!(summary.tests_executed, 4);
}

#[test]
fn test_long_receipt_cli_end_to_end() {
    let out = temp_path("audit.jsonl");
    let log = temp_path("phase1.log");
    write_file(
        &log,
        "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n",
    );
    let bin = env!("CARGO_BIN_EXE_nam_long_receipt");

    let run = |args: &[&str]| {
        Command::new(bin)
            .args(args)
            .output()
            .expect("nam_long_receipt must run")
    };

    // Phase 1: PASSED (tests counted from the log).
    let out_run = run(&[
        "append",
        "--phase-id",
        "phase1",
        "--name",
        "Soak Tests",
        "--status",
        "PASSED",
        "--duration-ms",
        "42000",
        "--log",
        log.to_str().unwrap(),
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(
        out_run.status.success(),
        "append phase1 failed: {:?}",
        out_run
    );

    // Phase 5: INCONCLUSIVE (gap marker auto-declared).
    let out_run = run(&[
        "append",
        "--phase-id",
        "phase5",
        "--name",
        "RT Jitter Characterization",
        "--status",
        "INCONCLUSIVE",
        "--duration-ms",
        "1000",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(
        out_run.status.success(),
        "append phase5 failed: {:?}",
        out_run
    );

    // Summary derives COMPLETED_WITH_GAPS and appends the overall line.
    let sum_run = run(&["summary", "--out", out.to_str().unwrap()]);
    assert!(sum_run.status.success(), "summary failed: {:?}", sum_run);

    // S5: the summary also prints the human one-liners (WARNING/ERROR +
    // verdicts) that utils/tests-long.sh echoes verbatim.
    let human = String::from_utf8_lossy(&sum_run.stdout);
    assert!(
        human.contains("OVERALL: COMPLETED_WITH_GAPS"),
        "missing OVERALL line: {human}"
    );
    assert!(
        human.contains("WARNING: phase5"),
        "missing WARNING line: {human}"
    );
    assert!(
        human.contains("FIDELITY: OK"),
        "missing FIDELITY line: {human}"
    );
    assert!(
        human.contains("PERF_REGRESSION: NOT_RUN"),
        "missing PERF_REGRESSION line: {human}"
    );

    // Validate: every line is valid JSON with the receipt schema.
    let val_run = run(&["validate", "--out", out.to_str().unwrap()]);
    assert!(val_run.status.success(), "validate failed: {:?}", val_run);
    let stdout = String::from_utf8_lossy(&val_run.stdout);
    assert!(
        stdout.contains("VALID: 3 receipt line(s)"),
        "unexpected validate output: {stdout}"
    );

    // File-level contract: 3 lines, all valid schema JSON; overall line last.
    let content = fs::read_to_string(&out).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 3, "unexpected receipt content: {content}");
    let audit = LongAuditReceipt::parse_jsonl(&content)
        .expect("every receipt line must parse as LongPhaseReceipt");
    assert_eq!(audit.phases.len(), 3);
    let overall = audit.phases.last().unwrap();
    assert_eq!(overall.phase_id, "overall");
    assert_eq!(overall.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(overall.tests_executed, 5);
    assert_eq!(overall.gaps, vec!["phase5:INCONCLUSIVE"]);

    // Bad status must fail with usage error (exit 2).
    let bad_run = run(&[
        "append",
        "--phase-id",
        "phaseX",
        "--name",
        "x",
        "--status",
        "MAYBE",
        "--duration-ms",
        "1",
    ]);
    assert_eq!(bad_run.status.code(), Some(2), "bad status must exit 2");
}

// S4.T2: `count-log` is the F-21 counter behind `_lib.sh::assert_ran_tests`.
#[test]
fn test_long_receipt_count_log_mirrors_f21_cases() {
    let bin = env!("CARGO_BIN_EXE_nam_long_receipt");
    let run = |path: &std::path::Path| {
        let out = Command::new(bin)
            .args(["count-log", "--log", path.to_str().unwrap()])
            .output()
            .expect("nam_long_receipt count-log must run");
        assert!(
            out.status.success(),
            "count-log failed: {:?}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };

    let pass = temp_path("pass.log");
    write_file(&pass, "test result: ok. 50 passed. 2 failed.\n");
    assert_eq!(run(&pass), "52");

    let zero = temp_path("zero.log");
    write_file(&zero, "test result: ok. 0 passed. 0 failed.\n");
    assert_eq!(run(&zero), "0");

    let skip = temp_path("skip.log");
    write_file(&skip, "running tests...\nall filtered out (early return)\n");
    assert_eq!(run(&skip), "0");

    let absent = temp_path("absent.log");
    assert_eq!(run(&absent), "0");

    let bench = temp_path("bench.log");
    write_file(&bench, "bench time: [1.2 ms]\nbench time: [3.4 ms]\n");
    assert_eq!(run(&bench), "2");

    let measured = temp_path("meas.log");
    write_file(&measured, "x 5 measured\n");
    assert_eq!(run(&measured), "5");

    // Usage error: missing --log exits 2.
    let bad = Command::new(bin)
        .args(["count-log"])
        .output()
        .expect("nam_long_receipt count-log must run");
    assert_eq!(
        bad.status.code(),
        Some(2),
        "count-log without --log must exit 2"
    );
}
