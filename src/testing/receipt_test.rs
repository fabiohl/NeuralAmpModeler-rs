// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nam-long-receipt-{}-{n}", std::process::id()))
}

fn write_temp(content: &str) -> PathBuf {
    let path = temp_path();
    let mut f = fs::File::create(&path).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    path
}

#[test]
fn status_parses_all_canonical_values() {
    for s in [
        "PASSED",
        "FAILED",
        "SKIPPED",
        "INCONCLUSIVE",
        "SKIP_CAPABILITY",
        "NOT_RUN",
        "SIMULATED",
        "COMPLETED_WITH_GAPS",
    ] {
        let parsed = LongPhaseStatus::from_str(s).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(parsed.as_str(), s);
    }
    assert!(LongPhaseStatus::from_str("nope").is_err());
    assert!(LongPhaseStatus::from_str("").is_err());
}

#[test]
fn gap_classification_matches_runner_semantics() {
    assert!(!LongPhaseStatus::Passed.is_gap());
    assert!(!LongPhaseStatus::Failed.is_gap());
    assert!(!LongPhaseStatus::CompletedWithGaps.is_gap());
    assert!(LongPhaseStatus::Skipped.is_gap());
    assert!(LongPhaseStatus::Inconclusive.is_gap());
    assert!(LongPhaseStatus::SkipCapability.is_gap());
    assert!(LongPhaseStatus::NotRun.is_gap());
    assert!(LongPhaseStatus::Simulated.is_gap());
    assert_eq!(LongPhaseStatus::Inconclusive.gap_id(), Some("inconclusive"));
    assert_eq!(LongPhaseStatus::Simulated.gap_id(), Some("simulated"));
    assert_eq!(LongPhaseStatus::Passed.gap_id(), None);
}

#[test]
fn receipt_line_roundtrips_with_schema() {
    let entry = LongPhaseReceipt {
        phase_id: "phase4".to_string(),
        name: "RT Deadline Gate (deterministic)".to_string(),
        status: LongPhaseStatus::Inconclusive,
        duration_ms: 42_000,
        tests_executed: 3,
        gaps: vec!["inconclusive_environment".to_string()],
        timestamp: "2026-08-14T00:00:00Z".to_string(),
        ..Default::default()
    };
    let line = entry.render_jsonl_line();
    let parsed = LongPhaseReceipt::parse_jsonl_line(&line).unwrap();
    assert_eq!(parsed, entry);
    assert!(line.contains("\"phase_id\":\"phase4\""));
    assert!(line.contains("\"status\":\"INCONCLUSIVE\""));
    assert!(line.contains("\"duration_ms\":42000"));
    assert!(line.contains("\"tests_executed\":3"));
    assert!(line.contains("\"gaps\":[\"inconclusive_environment\"]"));
    assert!(line.contains("\"timestamp\":\"2026-08-14T00:00:00Z\""));
}

#[test]
fn jsonl_parser_accepts_stream_and_rejects_bad_lines() {
    let a = LongPhaseReceipt {
        phase_id: "phase1".to_string(),
        name: "Soak".to_string(),
        status: LongPhaseStatus::Passed,
        duration_ms: 100,
        tests_executed: 5,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let b = LongPhaseReceipt {
        phase_id: "phase2".to_string(),
        name: "Defense".to_string(),
        status: LongPhaseStatus::Skipped,
        duration_ms: 200,
        tests_executed: 0,
        gaps: vec!["skipped".to_string()],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let input = format!("{}\n{}\n", a.render_jsonl_line(), b.render_jsonl_line());
    let parsed = LongAuditReceipt::parse_jsonl(&input).unwrap();
    assert_eq!(parsed.phases, vec![a.clone(), b.clone()]);
    assert_eq!(parsed.tests_executed_total(), 5);
    assert_eq!(parsed.duration_ms_total(), 300);

    let bad = format!("{}\n{{not json}}\n", a.render_jsonl_line());
    let err = LongAuditReceipt::parse_jsonl(&bad).unwrap_err();
    assert!(
        err.to_string().contains("invalid JSONL line 2"),
        "unexpected error: {err}"
    );
}

#[test]
fn jsonl_parser_skips_empty_lines() {
    let a = LongPhaseReceipt {
        phase_id: "phase1".to_string(),
        name: "Soak".to_string(),
        status: LongPhaseStatus::Passed,
        duration_ms: 1,
        tests_executed: 1,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let input = format!("\n{}\n\n", a.render_jsonl_line());
    let parsed = LongAuditReceipt::parse_jsonl(&input).unwrap();
    assert_eq!(parsed.phases.len(), 1);
}

#[test]
fn tests_executed_counts_pass_fail_and_measured_only() {
    let log = write_temp(
        "test result: ok. 123 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2s\n\
         test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.5s\n\
         test result: ok. 0 passed; 0 failed; 0 ignored; 10 measured; 0 filtered out; finished in 3.0s\n\
         running 200 tests (not a result line)\n",
    );
    assert_eq!(count_tests_executed_from_log(&log), 123 + 2 + 1 + 10);
}

#[test]
fn tests_executed_falls_back_to_benchmark_lines() {
    let log = write_temp(
        "soak_test/bench1  time:   [1.1 ms 1.2 ms 1.3 ms]\n\
         soak_test/bench2  time:   [2.1 ms 2.2 ms 2.3 ms]\n",
    );
    assert_eq!(count_tests_executed_from_log(&log), 2);
    let empty = write_temp("nothing to see here\n");
    assert_eq!(count_tests_executed_from_log(&empty), 0);
    let missing = temp_path();
    assert_eq!(count_tests_executed_from_log(&missing), 0);
}

// Test execution counting cases ported from `utils/tests-long.sh:700-723`:
// `_lib.sh::assert_ran_tests` now delegates to this function, so the shell
// asserts exercise the same inputs.
#[test]
fn f21_cases_from_long_suite() {
    let pass = write_temp("test result: ok. 50 passed. 2 failed.\n");
    assert_eq!(count_tests_executed_from_log(&pass), 52);
    let zero = write_temp("test result: ok. 0 passed. 0 failed.\n");
    assert_eq!(count_tests_executed_from_log(&zero), 0);
    let skip = write_temp("running tests...\nall filtered out (early return)\n");
    assert_eq!(count_tests_executed_from_log(&skip), 0);
    assert_eq!(count_tests_executed_from_log(&temp_path()), 0);
    let bench = write_temp("bench time: [1.2 ms]\nbench time: [3.4 ms]\n");
    assert_eq!(count_tests_executed_from_log(&bench), 2);
    let measured = write_temp("x 5 measured\n");
    assert_eq!(count_tests_executed_from_log(&measured), 5);
}

#[test]
fn gap_markers_are_detected_in_canonical_order() {
    // All structured markers recognized; details (`reason=`,
    // trailing text after `MISSING-REQUIRED:`) are attached to the gap
    // entry so the receipt carries WHY the phase deviated.
    let log = write_temp(
        "preflight: INCONCLUSIVE_ENVIRONMENT\n\
         [STATUS] INCONCLUSIVE\n\
         [STATUS] SKIP_CAPABILITY\n\
         [STATUS] SKIP_OPTIONAL reason=\"models_nondist_absent\"\n\
         [STATUS] KNOWN_GAP id=\"condition_lstm_cpp_crash\" reason=\"C++ render limitation\"\n\
         MISSING-REQUIRED: wavenet_lite\n\
         AVX512_OPT_IN: NOT_RUN (default runner without --features avx512)\n\
         SKIP: old-style free print not yet converted\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec![
            "inconclusive_environment",
            "skip_capability",
            "skip_optional:models_nondist_absent",
            "known_gap:condition_lstm_cpp_crash",
            "inconclusive",
            "missing_required:wavenet_lite",
            "avx512_opt_in_not_run",
            "legacy_skip:old-style free print not yet converted",
        ]
    );
    let clean = write_temp("all good\n");
    assert!(detect_gap_markers(&clean).is_empty());
}

#[test]
fn unreadable_log_is_fail_closed() {
    // A missing/unreadable phase log must surface as a gap, never be
    // silently promoted to a clean PASSED with `gaps: []`.
    let missing = temp_path();
    assert_eq!(detect_gap_markers(&missing), vec!["log_unreadable"]);
}

#[test]
fn all_skip_occurrences_accumulate_per_phase() {
    // Every distinct skip of a phase log is accumulated — the
    // detector never collapses a family to its first line.
    let log = write_temp(
        "[STATUS] SKIP_OPTIONAL: models_nondist_absent\n\
         test result: ok. 3 passed; 0 failed\n\
         [STATUS] SKIP_OPTIONAL reason=\"optional_fixture_missing:lstm_2x24.nam\"\n\
         SKIP: legacy print A\n\
         SKIP: legacy print B\n\
         [STATUS] SKIP_CAPABILITY: model_not_found:BossWN-standard.nam\n\
         test result: ok. 5 passed; 0 failed\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec![
            "skip_capability:model_not_found:BossWN-standard.nam",
            "skip_optional:models_nondist_absent",
            "skip_optional:optional_fixture_missing:lstm_2x24.nam",
            "legacy_skip:legacy print A",
            "legacy_skip:legacy print B",
        ]
    );
}

#[test]
fn colon_form_skip_markers_are_recognized() {
    // Canonical grammar `[STATUS] SKIP_<MOTIVO>: <detalhes>` — the
    // colon dialect and the attribute dialect parse to the same gap ids.
    let log = write_temp(
        "[STATUS] SKIP_CAPABILITY: avx512_cpu_unsupported:Zen1\n\
         [STATUS] SKIP_OPTIONAL: models_nondist_empty\n\
         [STATUS] INCONCLUSIVE: governor not performance\n\
         [STATUS] KNOWN_GAP: condition_lstm_cpp_crash\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec![
            "skip_capability:avx512_cpu_unsupported:Zen1",
            "skip_optional:models_nondist_empty",
            "known_gap:condition_lstm_cpp_crash",
            "inconclusive:governor not performance",
        ]
    );
    // The attribute dialect keeps parsing identically.
    let attr = write_temp("[STATUS] SKIP_CAPABILITY reason=\"avx512_not_compiled:x\"\n");
    assert_eq!(
        detect_gap_markers(&attr),
        vec!["skip_capability:avx512_not_compiled:x"]
    );
    // A marker occurring MID-line (`.contains` match) still yields the
    // colon detail — the detector never drops the reason.
    let midline = write_temp("NOTE: [STATUS] SKIP_OPTIONAL: models_nondist_absent\n");
    assert_eq!(
        detect_gap_markers(&midline),
        vec!["skip_optional:models_nondist_absent"]
    );
}

#[test]
fn typed_skip_changes_receipt_verdict_to_completed_with_gaps() {
    // Regression guard: ANY typed skip emission alters the phase receipt
    // so the suite-level verdict is COMPLETED_WITH_GAPS — never a clean
    // PASSED with `gaps: []`.
    let log = write_temp(
        "[STATUS] SKIP_OPTIONAL: model_not_found:linear_test.nam\n\
         test result: ok. 2 passed; 0 failed\n",
    );
    let gaps = detect_gap_markers(&log);
    assert_eq!(gaps, vec!["skip_optional:model_not_found:linear_test.nam"]);

    let phase = LongPhaseReceipt {
        phase_id: "phase2".to_string(),
        name: "Property-Based, Parity & Golden Vectors in Release".to_string(),
        status: LongPhaseStatus::Passed,
        duration_ms: 1000,
        tests_executed: 2,
        gaps: gaps.clone(),
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let summary = receipt.summary_receipt();
    assert_eq!(summary.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(summary.gaps, vec!["phase2:PASSED"]);
    assert!(receipt.strict_verdict().is_err());
    assert_eq!(
        detect_gap_markers(&write_temp("all good\n")),
        Vec::<String>::new()
    );
}

#[test]
fn avx512_opt_in_declaration_counts_as_declared_gap() {
    // A default local long-suite run compiles without `avx512`, so
    // the isa_parity subphase executes zero cross-ISA cases. The explicit
    // `AVX512_OPT_IN: NOT_RUN` declaration must surface as a typed gap —
    // the suite verdict becomes COMPLETED_WITH_GAPS, never a clean PASSED.
    let log = write_temp("AVX512_OPT_IN: NOT_RUN (default runner without --features avx512)\n");
    assert_eq!(detect_gap_markers(&log), vec!["avx512_opt_in_not_run"]);

    let run = write_temp("AVX512_OPT_IN: RUN (cross-ISA matrix compiled and exercised)\n");
    assert!(
        detect_gap_markers(&run).is_empty(),
        "the RUN declaration must NOT be classified as a gap"
    );

    let phase = LongPhaseReceipt {
        phase_id: "phase2".to_string(),
        name: "Property-Based, Parity & Golden Vectors in Release".to_string(),
        status: LongPhaseStatus::Passed,
        duration_ms: 1000,
        tests_executed: 42,
        gaps: detect_gap_markers(&log),
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let summary = receipt.summary_receipt();
    assert_eq!(summary.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(summary.gaps, vec!["phase2:PASSED"]);
}

#[test]
fn summary_derives_verdict_and_gaps() {
    let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status,
        duration_ms: 1000,
        tests_executed: 2,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let all_pass = LongAuditReceipt {
        phases: vec![
            mk("phase1", LongPhaseStatus::Passed),
            mk("phase2", LongPhaseStatus::Passed),
        ],
    };
    let s = all_pass.summary_receipt();
    assert_eq!(s.phase_id, "overall");
    assert_eq!(s.status, LongPhaseStatus::Passed);
    assert_eq!(s.duration_ms, 2000);
    assert_eq!(s.tests_executed, 4);
    assert!(s.gaps.is_empty());

    let with_fail = LongAuditReceipt {
        phases: vec![
            mk("phase1", LongPhaseStatus::Passed),
            mk("phase2", LongPhaseStatus::Failed),
        ],
    };
    assert_eq!(with_fail.summary_receipt().status, LongPhaseStatus::Failed);

    let with_gaps = LongAuditReceipt {
        phases: vec![
            mk("phase4", LongPhaseStatus::Inconclusive),
            mk("phase5", LongPhaseStatus::SkipCapability),
        ],
    };
    let s = with_gaps.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(
        s.gaps,
        vec!["phase4:INCONCLUSIVE", "phase5:SKIP_CAPABILITY"]
    );

    let mut push = with_gaps.clone();
    push.push_summary();
    assert_eq!(push.phases.len(), 3);
    assert_eq!(push.phases[2].status, LongPhaseStatus::CompletedWithGaps);
    push.push_summary();
    assert_eq!(
        push.phases.len(),
        3,
        "push_summary must replace the overall line"
    );
}

#[test]
fn passed_phase_with_log_markers_counts_as_declared_gap() {
    // S5: the bash PHASE_STATUS overrides are gone, so a PASSED phase
    // whose gaps list carries typed log markers (INCONCLUSIVE_ENVIRONMENT
    // / [STATUS] *) must still yield COMPLETED_WITH_GAPS — never a clean
    // PASSED verdict (the runner's "exit-0 with internal bypass SHALL NOT
    // be promoted to PASS" invariant).
    let mk = |phase_id: &str, status: LongPhaseStatus, gaps: &[&str]| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status,
        duration_ms: 1000,
        tests_executed: 2,
        gaps: gaps.iter().map(|s| s.to_string()).collect(),
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let bypassed = LongAuditReceipt {
        phases: vec![
            mk(
                "phase5",
                LongPhaseStatus::Passed,
                &["inconclusive_environment"],
            ),
            mk("phase6", LongPhaseStatus::Passed, &["inconclusive"]),
        ],
    };
    let s = bypassed.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(s.gaps, vec!["phase5:PASSED", "phase6:PASSED"]);

    let clean = LongAuditReceipt {
        phases: vec![mk("phase5", LongPhaseStatus::Passed, &[])],
    };
    assert_eq!(clean.summary_receipt().status, LongPhaseStatus::Passed);
}

#[test]
fn human_summary_lines_flag_only_alarms_and_verdicts() {
    // S5: the human gets WARNING/ERROR alarms + the verdict lines;
    // quiet phases stay silent.
    let mk =
        |phase_id: &str, name: &str, status: LongPhaseStatus, gaps: &[&str]| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: name.to_string(),
            status,
            duration_ms: 1000,
            tests_executed: 2,
            gaps: gaps.iter().map(|s| s.to_string()).collect(),
            timestamp: "t".to_string(),
            ..Default::default()
        };
    let receipt = LongAuditReceipt {
        phases: vec![
            mk("phase1", "Soak Tests", LongPhaseStatus::Passed, &[]),
            mk("phase2", "Defense", LongPhaseStatus::Failed, &[]),
            mk(
                "phase5",
                "RT Deadline Gate (deterministic)",
                LongPhaseStatus::Passed,
                &["inconclusive_environment"],
            ),
            mk(
                "phase6",
                "RT Jitter Characterization",
                LongPhaseStatus::Passed,
                &["skip_capability"],
            ),
        ],
    };
    let lines = receipt.human_summary_lines();
    assert!(
        lines.iter().any(|l| l == "ERROR: phase2 Defense — FAILED"),
        "missing ERROR line: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l == "WARNING: phase5 RT Deadline Gate (deterministic) — PASSED (gaps: inconclusive_environment)"),
        "missing deadline WARNING line: {lines:?}"
    );
    assert!(
        lines
            .iter()
            .any(|l| l
                == "WARNING: phase6 RT Jitter Characterization — PASSED (gaps: skip_capability)"),
        "missing jitter WARNING line: {lines:?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("phase1")),
        "quiet phase must not appear: {lines:?}"
    );
    assert!(
        lines.iter().any(|l| l == "OVERALL: FAILED"),
        "the FAILED phase dominates the verdict: {lines:?}"
    );
    assert!(lines.iter().any(|l| l == "FIDELITY: FAIL"), "{lines:?}");
    assert!(
        lines.iter().any(|l| l == "RT_DEADLINE: INCONCLUSIVE"),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|l| l == "RT_JITTER: SKIP_CAPABILITY"),
        "{lines:?}"
    );
    assert!(
        lines.iter().any(|l| l == "PERF_REGRESSION: NOT_RUN"),
        "{lines:?}"
    );
}

#[test]
fn verdict_lines_preserve_pre_s5_mappings() {
    let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status,
        duration_ms: 1000,
        tests_executed: 1,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };

    // Deadline: FAILED → FAIL; SKIPPED/PASSED → PASS; absent → PASS.
    let failed = LongAuditReceipt {
        phases: vec![mk("phase5", LongPhaseStatus::Failed)],
    };
    assert_eq!(failed.rt_deadline_verdict(), "FAIL");
    for status in [
        LongPhaseStatus::Passed,
        LongPhaseStatus::Skipped,
        LongPhaseStatus::NotRun,
    ] {
        assert_eq!(
            LongAuditReceipt {
                phases: vec![mk("phase5", status)]
            }
            .rt_deadline_verdict(),
            "PASS",
            "deadline {status} must map to PASS"
        );
    }
    assert_eq!(
        LongAuditReceipt { phases: vec![] }.rt_deadline_verdict(),
        "NOT_RUN"
    );

    // Jitter: PASSED → PASS; FAILED → FAIL; SKIPPED → INCONCLUSIVE.
    let mk_jitter = |status: LongPhaseStatus| LongAuditReceipt {
        phases: vec![mk("phase6", status)],
    };
    assert_eq!(
        mk_jitter(LongPhaseStatus::Passed).rt_jitter_verdict(),
        "PASS"
    );
    assert_eq!(
        mk_jitter(LongPhaseStatus::Failed).rt_jitter_verdict(),
        "FAIL"
    );
    assert_eq!(
        mk_jitter(LongPhaseStatus::Skipped).rt_jitter_verdict(),
        "INCONCLUSIVE"
    );
    assert_eq!(
        LongAuditReceipt { phases: vec![] }.rt_jitter_verdict(),
        "NOT_RUN"
    );

    // Fidelity: FAIL only on a failed non-performance phase; performance
    // failures keep FIDELITY OK (the PERF-006 split).
    let perf_failed = LongAuditReceipt {
        phases: vec![
            mk("phase5", LongPhaseStatus::Failed),
            mk("phase6", LongPhaseStatus::Failed),
            mk("phase1", LongPhaseStatus::Passed),
        ],
    };
    assert_eq!(perf_failed.fidelity_verdict(), "OK");
    let fidelity_failed = LongAuditReceipt {
        phases: vec![mk("phase2", LongPhaseStatus::Failed)],
    };
    assert_eq!(fidelity_failed.fidelity_verdict(), "FAIL");
    let preflight_failed = LongAuditReceipt {
        phases: vec![mk("preflight-catalog", LongPhaseStatus::Failed)],
    };
    assert_eq!(preflight_failed.fidelity_verdict(), "FAIL");
}

#[test]
fn simulated_receipt_never_prints_green_verdicts() {
    // F-07: `--simulate` / `--dry-run` pre-registers every phase as SIMULATED
    // with zero tests executed. The derived overall line must be SIMULATED
    // and every verdict line must read NOT_RUN — an `OK`/`PASS` would be a
    // vacuous green on a run that measured nothing.
    let mk = |phase_id: &str| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status: LongPhaseStatus::Simulated,
        duration_ms: 0,
        tests_executed: 0,
        gaps: vec!["simulated".to_string()],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let receipt = LongAuditReceipt {
        phases: vec![
            mk("preflight-render"),
            mk("phase1"),
            mk("phase5"),
            mk("phase6"),
        ],
    };
    assert!(receipt.is_simulated_only());
    assert_eq!(receipt.summary_receipt().status, LongPhaseStatus::Simulated);
    assert!(receipt.strict_verdict().is_err());

    let lines = receipt.human_summary_lines();
    assert!(lines.iter().any(|l| l == "OVERALL: SIMULATED"), "{lines:?}");
    assert!(lines.iter().any(|l| l == "FIDELITY: NOT_RUN"), "{lines:?}");
    assert!(
        lines.iter().any(|l| l == "RT_DEADLINE: NOT_RUN"),
        "{lines:?}"
    );
    assert!(lines.iter().any(|l| l == "RT_JITTER: NOT_RUN"), "{lines:?}");
    assert!(
        lines.iter().any(|l| l == "PERF_REGRESSION: NOT_RUN"),
        "{lines:?}"
    );
    assert!(
        !lines
            .iter()
            .any(|l| l.ends_with(": OK") || l.ends_with(": PASS")),
        "simulated receipt must never print a green verdict: {lines:?}"
    );

    // A mixed receipt is not "simulated only": the SIMULATED phases stay
    // declared gaps (COMPLETED_WITH_GAPS), and a simulated fidelity-class
    // phase still suppresses the vacuous `FIDELITY: OK`.
    let mixed = LongAuditReceipt {
        phases: vec![
            LongPhaseReceipt {
                phase_id: "phase2".to_string(),
                name: "Defense".to_string(),
                status: LongPhaseStatus::Passed,
                duration_ms: 10,
                tests_executed: 3,
                gaps: vec![],
                timestamp: "t".to_string(),
                ..Default::default()
            },
            mk("phase6"),
        ],
    };
    assert!(!mixed.is_simulated_only());
    assert_eq!(
        mixed.summary_receipt().status,
        LongPhaseStatus::CompletedWithGaps
    );
    assert_eq!(mixed.fidelity_verdict(), "OK");
    assert_eq!(mixed.rt_jitter_verdict(), "NOT_RUN");
}

#[test]
fn partial_simulated_receipt_never_prints_green_verdicts() {
    // F-07 robustness: even an all-SIMULATED receipt that omits the
    // performance/fidelity phases (e.g. only phase5) must report NOT_RUN on
    // every verdict line — a missing phase is not a green verdict.
    let only_phase5 = LongAuditReceipt {
        phases: vec![mk_phase(
            "phase5",
            LongPhaseStatus::Simulated,
            0,
            &["simulated"],
        )],
    };
    assert!(only_phase5.is_simulated_only());
    assert_eq!(
        only_phase5.summary_receipt().status,
        LongPhaseStatus::Simulated
    );
    assert_eq!(only_phase5.fidelity_verdict(), "NOT_RUN");
    assert_eq!(only_phase5.rt_deadline_verdict(), "NOT_RUN");
    assert_eq!(only_phase5.rt_jitter_verdict(), "NOT_RUN");

    let only_phase6 = LongAuditReceipt {
        phases: vec![mk_phase(
            "phase6",
            LongPhaseStatus::Simulated,
            0,
            &["simulated"],
        )],
    };
    assert_eq!(only_phase6.fidelity_verdict(), "NOT_RUN");
    assert_eq!(only_phase6.rt_jitter_verdict(), "NOT_RUN");

    let lines = only_phase5.human_summary_lines();
    assert!(
        !lines
            .iter()
            .any(|l| l.ends_with(": OK") || l.ends_with(": PASS")),
        "partial simulated receipt must never print a green verdict: {lines:?}"
    );
}

#[test]
fn preflight_ids_are_canonical_and_roundtrip() {
    for id in PREFLIGHT_PHASE_IDS {
        assert!(is_preflight_id(id), "{id} must be a preflight id");
        let entry = LongPhaseReceipt {
            phase_id: id.to_string(),
            name: id.to_string(),
            status: LongPhaseStatus::Passed,
            duration_ms: 0,
            tests_executed: 0,
            gaps: vec![],
            timestamp: "t".to_string(),
            ..Default::default()
        };
        let line = entry.render_jsonl_line();
        assert_eq!(LongPhaseReceipt::parse_jsonl_line(&line).unwrap(), entry);
    }
    assert!(is_preflight_id("preflight-future-step"));
    assert!(!is_preflight_id("phase1"));
    assert!(!is_preflight_id("overall"));
    assert!(!is_preflight_id("preflight")); // bare prefix without id
}

#[test]
fn jsonl_parser_accepts_canonical_preflight_ids_and_rejects_unknown() {
    let mk = |phase_id: &str| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status: LongPhaseStatus::Passed,
        duration_ms: 1,
        tests_executed: 0,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    let canonical = format!(
        "{}\n{}\n",
        mk("preflight-render").render_jsonl_line(),
        mk("preflight-catalog").render_jsonl_line()
    );
    let parsed = LongAuditReceipt::parse_jsonl(&canonical).unwrap();
    assert_eq!(parsed.preflight_entries().count(), 2);
    assert_eq!(parsed.phases.len(), 2);

    // Typo (preflight-catlog) must be rejected fail-closed.
    let typo = mk("preflight-catlog").render_jsonl_line();
    let err = LongAuditReceipt::parse_jsonl(&typo).unwrap_err();
    assert!(
        err.to_string()
            .contains("unknown preflight identifier 'preflight-catlog'"),
        "unexpected error: {err}"
    );
    // A non-preflight id is unaffected by the canonical preflight set.
    let phase = mk("phase1").render_jsonl_line();
    assert_eq!(
        LongAuditReceipt::parse_jsonl(&phase).unwrap().phases.len(),
        1
    );
}

#[test]
fn preflight_failure_drives_overall_failed_verdict() {
    let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status,
        duration_ms: 1000,
        tests_executed: 1,
        gaps: vec![],
        timestamp: "t".to_string(),
        ..Default::default()
    };
    // Preflight failure acceptance: an aborted preflight leaves its FAILED line and
    // the summary derives `overall FAILED` — even with all timed phases
    // green, because the suite never reached them.
    let aborted = LongAuditReceipt {
        phases: vec![
            mk("preflight-catalog", LongPhaseStatus::Failed),
            mk("preflight-render", LongPhaseStatus::Passed),
        ],
    };
    assert_eq!(aborted.preflight_entries().count(), 2);
    let s = aborted.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::Failed);
    assert_eq!(s.duration_ms, 2000);
    assert_eq!(s.tests_executed, 2);
    assert!(s.gaps.is_empty());

    let all_pass = LongAuditReceipt {
        phases: vec![
            mk("preflight-catalog", LongPhaseStatus::Passed),
            mk("preflight-meta", LongPhaseStatus::Passed),
        ],
    };
    assert_eq!(all_pass.summary_receipt().status, LongPhaseStatus::Passed);
}

// ═══════════════════════════════════════════════════════════════════════
// Fixture-based tests: synthetic logs drive the marker parser and
// the receipt classifier through every status/gap branch.
// ═══════════════════════════════════════════════════════════════════════

fn mk_phase(
    phase_id: &str,
    status: LongPhaseStatus,
    tests_executed: u64,
    gaps: &[&str],
) -> LongPhaseReceipt {
    LongPhaseReceipt {
        phase_id: phase_id.to_string(),
        name: phase_id.to_string(),
        status,
        duration_ms: 1000,
        tests_executed,
        gaps: gaps.iter().map(|s| s.to_string()).collect(),
        timestamp: "t".to_string(),
        ..Default::default()
    }
}

fn mk_canonical_clean_receipt() -> LongAuditReceipt {
    let mut phases = Vec::new();
    for &id in &CANONICAL_ROSTER {
        let tests = if id.starts_with("phase") { 10 } else { 0 };
        phases.push(mk_phase(id, LongPhaseStatus::Passed, tests, &[]));
    }
    LongAuditReceipt { phases }
}

#[test]
fn clean_log_classifies_as_passed_with_empty_gaps() {
    // Fixture: a fully clean execution — no markers, real tests executed.
    let log = write_temp(
        "test result: ok. 42 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n\
         all assertions green\n",
    );
    assert!(detect_gap_markers(&log).is_empty());
    assert_eq!(count_tests_executed_from_log(&log), 42);

    let phase = mk_phase("phase1", LongPhaseStatus::Passed, 42, &[]);
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let s = receipt.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::Passed);
    assert!(s.gaps.is_empty());
    // Partial receipt is accepted in diagnostic summary mode, but rejected
    // in strict mode (which requires the complete 14-phase canonical roster).
    assert!(receipt.strict_verdict().is_err());

    let canonical = mk_canonical_clean_receipt();
    assert_eq!(canonical.summary_receipt().status, LongPhaseStatus::Passed);
    assert_eq!(canonical.strict_verdict(), Ok(()));
}

#[test]
fn skip_optional_nondist_log_classifies_as_completed_with_gaps() {
    // Fixture: SKIP_OPTIONAL (models-nondist absent) must surface as a
    // typed gap and downgrade the overall verdict — never a silent PASSED.
    let log = write_temp(
        "[STATUS] SKIP_OPTIONAL reason=\"models_nondist_absent\"\n\
         test result: ok. 5 passed; 0 failed\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec!["skip_optional:models_nondist_absent"]
    );

    let phase = mk_phase(
        "phase2",
        LongPhaseStatus::Passed,
        5,
        &["skip_optional:models_nondist_absent"],
    );
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let s = receipt.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(s.gaps, vec!["phase2:PASSED"]);
    assert!(receipt.strict_verdict().is_err());
}

#[test]
fn known_gap_condition_lstm_registers_gap_with_correct_id() {
    // Fixture: the upstream C++ condition_lstm gap must be recorded with
    // its stable id (`known_gap:condition_lstm_cpp_crash`).
    let log = write_temp(
        "[STATUS] KNOWN_GAP id=\"condition_lstm_cpp_crash\" \
         reason=\"C++ render tool limitation (LSTM condition_dsp channel mismatch)\"\n\
         test result: ok. 1 passed; 0 failed\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec!["known_gap:condition_lstm_cpp_crash"]
    );

    let phase = mk_phase(
        "phase2",
        LongPhaseStatus::Passed,
        1,
        &["known_gap:condition_lstm_cpp_crash"],
    );
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let s = receipt.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(s.gaps, vec!["phase2:PASSED"]);
    assert!(receipt.strict_verdict().is_err());
}

#[test]
fn zero_test_pass_is_detected_as_inconsistency() {
    // Fixture: a timed phase that PASSED with zero executed tests is a
    // mandatory-subphase gate violation — it must not be promoted.
    let zero = mk_phase("phase2", LongPhaseStatus::Passed, 0, &[]);
    let receipt = LongAuditReceipt { phases: vec![zero] };
    assert_eq!(receipt.zero_test_passes().count(), 1);
    let s = receipt.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert_eq!(s.gaps, vec!["phase2:ZERO_TESTS"]);
    assert!(receipt.strict_verdict().is_err());

    // Preflight lines with zero tests are NOT zero-test passes.
    let preflight = LongAuditReceipt {
        phases: vec![mk_phase(
            "preflight-render",
            LongPhaseStatus::Passed,
            0,
            &[],
        )],
    };
    assert_eq!(preflight.zero_test_passes().count(), 0);
    assert_eq!(preflight.summary_receipt().status, LongPhaseStatus::Passed);
    assert!(preflight.strict_verdict().is_err());
}

#[test]
fn legacy_skip_lines_surface_as_transitional_gap() {
    // Transitional fallback: unconverted free-form
    // `SKIP:` prints are still recognized — never masked into a PASSED.
    let log = write_temp(
        "SKIP: Model file not found\n\
         test result: ok. 3 passed; 0 failed\n",
    );
    assert_eq!(
        detect_gap_markers(&log),
        vec!["legacy_skip:Model file not found"]
    );

    let phase = mk_phase(
        "phase2",
        LongPhaseStatus::Passed,
        3,
        &["legacy_skip:Model file not found"],
    );
    let receipt = LongAuditReceipt {
        phases: vec![phase],
    };
    let s = receipt.summary_receipt();
    assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
    assert!(receipt.strict_verdict().is_err());
}

#[test]
fn strict_verdict_rejects_every_gap_condition() {
    // Strict mode fails on each gap family and passes
    // only on a fully clean receipt.
    let cases: Vec<Vec<LongPhaseReceipt>> = vec![
        // gap status
        vec![mk_phase("phase4", LongPhaseStatus::Inconclusive, 1, &[])],
        // skip status
        vec![mk_phase("phase2", LongPhaseStatus::Skipped, 0, &[])],
        // not-run status
        vec![mk_phase("phase6", LongPhaseStatus::NotRun, 0, &[])],
        // simulated pre-registration (T2.1 / F-03: never passes strict)
        vec![mk_phase(
            "phase1",
            LongPhaseStatus::Simulated,
            0,
            &["simulated"],
        )],
        // PASSED with typed markers
        vec![mk_phase(
            "phase5",
            LongPhaseStatus::Passed,
            1,
            &["inconclusive_environment"],
        )],
        // zero-test pass
        vec![mk_phase("phase2", LongPhaseStatus::Passed, 0, &[])],
        // failed phase
        vec![mk_phase("phase3", LongPhaseStatus::Failed, 1, &[])],
    ];
    for phases in cases {
        let receipt = LongAuditReceipt { phases };
        assert!(
            receipt.strict_verdict().is_err(),
            "strict must reject gap condition: {}",
            receipt.summary_receipt().status
        );
    }

    let clean = mk_canonical_clean_receipt();
    assert_eq!(clean.strict_verdict(), Ok(()));
    assert_eq!(clean.summary_receipt().status, LongPhaseStatus::Passed);
}

#[test]
fn test_strict_verdict_canonical_roster_matrix() {
    // 1. Empty receipt
    let empty = LongAuditReceipt { phases: vec![] };
    assert!(empty.strict_verdict().is_err());

    // 2. Only whitespace from JSONL
    let parsed_ws = LongAuditReceipt::parse_jsonl("   \n\t\n").unwrap();
    assert!(parsed_ws.strict_verdict().is_err());

    // 3. Only overall
    let only_overall = LongAuditReceipt {
        phases: vec![mk_phase("overall", LongPhaseStatus::Passed, 100, &[])],
    };
    assert!(only_overall.strict_verdict().is_err());

    // 4. Prefix up to phase3
    let mut prefix_phases = Vec::new();
    for &id in &CANONICAL_ROSTER {
        if id == "phase4" {
            break;
        }
        let tests = if id.starts_with("phase") { 10 } else { 0 };
        prefix_phases.push(mk_phase(id, LongPhaseStatus::Passed, tests, &[]));
    }
    let prefix = LongAuditReceipt {
        phases: prefix_phases,
    };
    assert!(prefix.strict_verdict().is_err());

    // 5. Repeated/duplicate phase
    let mut dup_phases = mk_canonical_clean_receipt().phases;
    dup_phases.insert(0, mk_phase("phase1", LongPhaseStatus::Passed, 5, &[]));
    let dup = LongAuditReceipt { phases: dup_phases };
    assert!(dup.strict_verdict().is_err());

    // 6. Unknown phase ID
    let mut unknown_phases = mk_canonical_clean_receipt().phases;
    unknown_phases.insert(0, mk_phase("phaseX", LongPhaseStatus::Passed, 5, &[]));
    let unknown = LongAuditReceipt {
        phases: unknown_phases,
    };
    assert!(unknown.strict_verdict().is_err());

    // 7. phase4 with zero tests
    let mut zero_phase4 = mk_canonical_clean_receipt().phases;
    for p in &mut zero_phase4 {
        if p.phase_id == "phase4" {
            p.tests_executed = 0;
        }
    }
    let zero_p4_receipt = LongAuditReceipt {
        phases: zero_phase4,
    };
    assert!(zero_p4_receipt.strict_verdict().is_err());

    // 8. Full canonical PASSED
    let full_passed = mk_canonical_clean_receipt();
    assert_eq!(full_passed.strict_verdict(), Ok(()));

    // 9. Full canonical with one FAILED phase
    let mut failed_phase = mk_canonical_clean_receipt().phases;
    for p in &mut failed_phase {
        if p.phase_id == "phase3" {
            p.status = LongPhaseStatus::Failed;
        }
    }
    let failed_receipt = LongAuditReceipt {
        phases: failed_phase,
    };
    assert!(failed_receipt.strict_verdict().is_err());

    // 10. SIMULATED receipt
    let mut simulated_phases = Vec::new();
    for &id in &CANONICAL_ROSTER {
        simulated_phases.push(mk_phase(id, LongPhaseStatus::Simulated, 0, &["simulated"]));
    }
    let simulated_receipt = LongAuditReceipt {
        phases: simulated_phases,
    };
    assert!(simulated_receipt.strict_verdict().is_err());
}

#[test]
fn gap_family_prefix_matching_survives_details() {
    // Verdict helpers match the canonical family even when the gap
    // entry carries a `:detail` suffix from the marker grammar.
    let with_detail = LongAuditReceipt {
        phases: vec![mk_phase(
            "phase6",
            LongPhaseStatus::Passed,
            1,
            &["skip_capability:Single core affinity"],
        )],
    };
    assert_eq!(with_detail.rt_jitter_verdict(), "SKIP_CAPABILITY");

    let with_detail_deadline = LongAuditReceipt {
        phases: vec![mk_phase(
            "phase5",
            LongPhaseStatus::Passed,
            1,
            &["inconclusive_environment:preflight_failed"],
        )],
    };
    assert_eq!(with_detail_deadline.rt_deadline_verdict(), "INCONCLUSIVE");
}

#[test]
fn simulated_preregistration_derives_simulated_and_rejects_strict() {
    // T2.1 / F-03 + F-07: `--simulate` pre-registers every phase as
    // SIMULATED with zero executed tests. The summary must derive SIMULATED
    // (never PASSED, never a green verdict) and `strict_verdict` must reject
    // it fail-closed.
    let phases = vec![
        mk_phase(
            "preflight-render",
            LongPhaseStatus::Simulated,
            0,
            &["simulated"],
        ),
        mk_phase("phase1", LongPhaseStatus::Simulated, 0, &["simulated"]),
        mk_phase("phase7", LongPhaseStatus::Simulated, 0, &["simulated"]),
    ];
    let receipt = LongAuditReceipt { phases };
    let summary = receipt.summary_receipt();
    assert_eq!(summary.status, LongPhaseStatus::Simulated);
    assert_eq!(
        summary.gaps,
        vec![
            "preflight-render:SIMULATED",
            "phase1:SIMULATED",
            "phase7:SIMULATED"
        ]
    );
    assert!(receipt.has_declared_gaps());
    assert!(receipt.strict_verdict().is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// Referenced-artifact provenance fields (S4-T4.1) and fail-closed semantic
// verification (S4-T4.2): the unified receipt schema must stay backward
// compatible with historical goldens and must reject tampering structurally.
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn provenance_fields_are_optional_and_backward_compatible() {
    // A line without provenance must NOT serialize the optional keys (keeps the
    // historical JSONL byte-identical) and must round-trip to `None`.
    let bare = mk_phase("phase2", LongPhaseStatus::Passed, 1, &[]);
    let bare_line = bare.render_jsonl_line();
    assert!(!bare_line.contains("oracle_sha256"));
    assert!(!bare_line.contains("fixture_sha256"));
    assert!(!bare_line.contains("artifact_sha256"));

    // An old golden line (predating the fields) parses with `None` provenance.
    let old_golden = r#"{"phase_id":"phase2","name":"phase2","status":"PASSED","duration_ms":1000,"tests_executed":1,"gaps":[],"timestamp":"t"}"#;
    let parsed_old = LongPhaseReceipt::parse_jsonl_line(old_golden).unwrap();
    assert_eq!(parsed_old.oracle_sha256, None);
    assert_eq!(parsed_old.fixture_sha256, None);
    assert_eq!(parsed_old.artifact_sha256, None);

    // Populated provenance survives the round-trip and is emitted.
    let with_hashes = bare.clone().with_referenced_artifacts(
        Some("aa".repeat(32)),
        Some("bb".repeat(32)),
        Some("cc".repeat(32)),
    );
    let line = with_hashes.render_jsonl_line();
    assert!(line.contains("\"oracle_sha256\":\"aaaa"));
    assert!(line.contains("\"fixture_sha256\":\"bbbb"));
    assert!(line.contains("\"artifact_sha256\":\"cccc"));
    assert_eq!(
        LongPhaseReceipt::parse_jsonl_line(&line).unwrap(),
        with_hashes
    );
}

#[test]
fn semantic_verdict_accepts_clean_and_consistent_gap_receipts() {
    let clean = mk_canonical_clean_receipt();
    assert_eq!(clean.semantic_verdict(), Ok(()));
    assert_eq!(clean.campaign_strict_verdict(), Ok(()));

    // A declared COMPLETED_WITH_GAPS that matches a real gap phase is
    // semantically consistent (it is rejected only by the strict policy).
    let mut gapped = mk_canonical_clean_receipt().phases;
    for p in &mut gapped {
        if p.phase_id == "phase2" {
            p.status = LongPhaseStatus::Skipped;
            p.gaps = vec!["skipped".to_string()];
        }
        if p.phase_id == "overall" {
            p.status = LongPhaseStatus::CompletedWithGaps;
        }
    }
    let gapped = LongAuditReceipt { phases: gapped };
    assert_eq!(gapped.semantic_verdict(), Ok(()));
    assert!(gapped.campaign_strict_verdict().is_err());
}

#[test]
fn semantic_verdict_rejects_forged_overall_verdict() {
    // A forged/injected green verdict over real gap evidence must be rejected:
    // the declared terminal `overall` disagrees with the derived verdict.
    let mut phases = mk_canonical_clean_receipt().phases;
    for p in &mut phases {
        if p.phase_id == "phase3" {
            p.status = LongPhaseStatus::Skipped;
            p.gaps = vec!["skipped".to_string()];
        }
    }
    // terminal `overall` is still PASSED (forged).
    let forged = LongAuditReceipt { phases };
    let err = forged.semantic_verdict().unwrap_err();
    assert!(err.contains("disagrees"), "unexpected error: {err}");
}

#[test]
fn semantic_verdict_rejects_missing_and_duplicate_phases() {
    let mut missing = mk_canonical_clean_receipt().phases;
    missing.retain(|p| p.phase_id != "phase4");
    let missing = LongAuditReceipt { phases: missing };
    assert!(missing.semantic_verdict().unwrap_err().contains("missing"));

    let mut duplicated = mk_canonical_clean_receipt().phases;
    duplicated.insert(0, mk_phase("phase1", LongPhaseStatus::Passed, 5, &[]));
    let duplicated = LongAuditReceipt { phases: duplicated };
    assert!(
        duplicated
            .semantic_verdict()
            .unwrap_err()
            .contains("duplicate")
    );
}

#[test]
fn semantic_verdict_rejects_injected_overall_comment_line() {
    // The classic substring-search bypass: an `OVERALL: PASSED` line pasted
    // into a comment/echo is not a receipt line and must fail closed at parse
    // time (never promoted to a green verdict).
    let clean = mk_canonical_clean_receipt().render_jsonl();
    let tampered = format!("{clean}# OVERALL: PASSED\n");
    let err = LongAuditReceipt::parse_jsonl(&tampered).unwrap_err();
    assert!(
        err.to_string().contains("invalid JSONL line 15"),
        "unexpected error: {err}"
    );
}
