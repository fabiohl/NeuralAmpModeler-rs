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
    assert_eq!(LongPhaseStatus::Inconclusive.gap_id(), Some("inconclusive"));
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
    };
    let b = LongPhaseReceipt {
        phase_id: "phase2".to_string(),
        name: "Defense".to_string(),
        status: LongPhaseStatus::Skipped,
        duration_ms: 200,
        tests_executed: 0,
        gaps: vec!["skipped".to_string()],
        timestamp: "t".to_string(),
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

// F-21 acceptance cases ported from `utils/tests-long.sh:700-723` (S4.T2):
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
    // T3.2/T3.3: all structured markers recognized; details (`reason=`,
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
    // T3.3: a missing/unreadable phase log must surface as a gap, never be
    // silently promoted to a clean PASSED with `gaps: []`.
    let missing = temp_path();
    assert_eq!(detect_gap_markers(&missing), vec!["log_unreadable"]);
}

#[test]
fn all_skip_occurrences_accumulate_per_phase() {
    // T2.1: every distinct skip of a phase log is accumulated — the
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
    // T2.1 canonical grammar `[STATUS] SKIP_<MOTIVO>: <detalhes>` — the
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
    // T2.1 regression: ANY typed skip emission alters the phase receipt
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
    // T2.4: a default local long-suite run compiles without `avx512`, so
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
        "PASS"
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
        "PASS"
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
    };
    // S6-T03 acceptance: an aborted preflight leaves its FAILED line and
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
// T3.4 — fixture-based tests: synthetic logs drive the marker parser and
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
    }
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
    assert_eq!(receipt.strict_verdict(), Ok(()));
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
    assert_eq!(preflight.strict_verdict(), Ok(()));
}

#[test]
fn legacy_skip_lines_surface_as_transitional_gap() {
    // T3.2 rollback: during the transition, unconverted free-form
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
    // T3.3 acceptance: strict mode fails on each gap family and passes
    // only on a fully clean receipt.
    let cases: Vec<Vec<LongPhaseReceipt>> = vec![
        // gap status
        vec![mk_phase("phase4", LongPhaseStatus::Inconclusive, 1, &[])],
        // skip status
        vec![mk_phase("phase2", LongPhaseStatus::Skipped, 0, &[])],
        // not-run status
        vec![mk_phase("phase6", LongPhaseStatus::NotRun, 0, &[])],
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

    let clean = LongAuditReceipt {
        phases: vec![
            mk_phase("phase1", LongPhaseStatus::Passed, 10, &[]),
            mk_phase("phase2", LongPhaseStatus::Passed, 5, &[]),
        ],
    };
    assert_eq!(clean.strict_verdict(), Ok(()));
    assert_eq!(clean.summary_receipt().status, LongPhaseStatus::Passed);
}

#[test]
fn gap_family_prefix_matching_survives_details() {
    // T3.3: verdict helpers match the canonical family even when the gap
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
