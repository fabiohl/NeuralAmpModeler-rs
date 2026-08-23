// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the contract verify engine (S2.T2 literal port).
//!
//! The acceptance fixtures use the real `docs/quality-contract.json` (51
//! fidelity + 19 performance entries) with a report generated from it, so
//! the four acceptance scenarios exercise the full contract surface.

use std::assert_matches;
use std::path::PathBuf;

use serde_json::{Value, json};

use super::super::{Envelopes, FidelityEntry};
use super::*;

/// All phases PASS.
const ALL_PASS: [(&str, &str); 4] = [
    ("golden_vectors", "PASS"),
    ("reference_oracle_f64", "PASS"),
    ("quick_parity", "PASS"),
    ("regression_gate", "PASS"),
];

fn load_contract() -> QualityContract {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("docs/quality-contract.json");
    let content = std::fs::read_to_string(&path).expect("read docs/quality-contract.json");
    QualityContract::from_json_str(&content).expect("contract must validate against the schema")
}

fn phase_line(phase_id: &str, status: &str) -> String {
    format!(r#"{{"phase_id":"{phase_id}","status":"{status}"}}"#)
}

/// Canonical fidelity object for `entry` (`None` drops the entry).
fn canonical_fidelity(entry: &FidelityEntry) -> Option<Value> {
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

/// Canonical latency object for `entry` (`None` drops the entry).
fn canonical_latency(entry: &PerformanceEntry) -> Option<Value> {
    Some(json!({
        "kind": "latency",
        "label": entry.label,
        "median_latency_us": entry.median_latency_us,
    }))
}

/// Builds a report from the contract: phase records + per-entry fidelity and
/// latency objects (closure `None` drops the entry).
fn build_report(
    contract: &QualityContract,
    phases: &[(&str, &str)],
    fidelity: impl Fn(&FidelityEntry) -> Option<Value>,
    latency: impl Fn(&PerformanceEntry) -> Option<Value>,
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

fn verify(contract: &QualityContract, report: &str) -> VerifyOutcome {
    let report = parse_verify_report(report).expect("fixture report must parse");
    verify_contract(contract, &report)
}

/// Finds the metric check of one fidelity entry by id.
fn metric_check<'a>(outcome: &'a VerifyOutcome, id: &str, metric: Metric) -> &'a MetricOutcome {
    let check = outcome
        .fidelity_checks
        .iter()
        .find(|c| c.id == id)
        .unwrap_or_else(|| panic!("no fidelity check for {id}"));
    let FidelityOutcome::Measured(metrics) = &check.outcome else {
        panic!("{id} was not measured: {:?}", check.outcome);
    };
    metrics
        .iter()
        .find(|m| m.metric == metric)
        .map(|m| &m.outcome)
        .unwrap_or_else(|| panic!("no {metric:?} check for {id}"))
}

const TARGET: &str = "bosslstm-1x16@48000:live";

// ── Acceptance fixtures ─────────────────────────────────────────────────────

#[test]
fn acceptance_fixture_1_report_equals_contract_verdict_ok() {
    let contract = load_contract();
    let report = build_report(&contract, &ALL_PASS, canonical_fidelity, canonical_latency);
    let outcome = verify(&contract, &report);

    assert!(outcome.fidelity.is_ok(), "fidelity must be OK: {outcome:?}");
    assert!(
        outcome.performance.is_ok(),
        "performance must be OK: {outcome:?}"
    );
    assert_eq!(outcome.review_required, 0);
    assert_eq!(outcome.exit_code(), 0);
    assert_eq!(outcome.fidelity_checks.len(), contract.fidelity.len());
    assert_eq!(outcome.perf_checks.len(), contract.performance.len());
}

#[test]
fn acceptance_fixture_2_esr_above_safety_fidelity_fail() {
    let contract = load_contract();
    let report = build_report(
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
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    assert_eq!(outcome.exit_code(), 1);
    assert_matches!(
        metric_check(&outcome, TARGET, Metric::EsrNamcore),
        MetricOutcome::SafetyCeiling { .. },
        "1e-3 is above the rounded safety ceiling"
    );
    // NAMCore violated while f64 stayed ok → oracle divergence review.
    assert_eq!(outcome.review_required, 1);
}

#[test]
fn acceptance_fixture_3_regression_gate_not_pass_is_performance_not_verified() {
    let contract = load_contract();
    for gate in ["FAIL", "NOT_VERIFIED"] {
        let phases = [
            ("golden_vectors", "PASS"),
            ("reference_oracle_f64", "PASS"),
            ("quick_parity", "PASS"),
            ("regression_gate", gate),
        ];
        let report = build_report(&contract, &phases, canonical_fidelity, canonical_latency);
        let outcome = verify(&contract, &report);

        assert!(
            outcome.fidelity.is_ok(),
            "regression_gate must never flip FIDELITY (PERF-006): {outcome:?}"
        );
        assert!(
            outcome.performance.is_not_verified(),
            "regression_gate={gate} must yield PERFORMANCE: NOT_VERIFIED"
        );
        assert_eq!(outcome.exit_code(), 1);
    }
}

#[test]
fn acceptance_fixture_4_f64_violates_namcore_ok_review_required() {
    let contract = load_contract();
    let report = build_report(
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
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { .. });
    assert_eq!(
        outcome.review_required, 1,
        "NAMCore ok + f64 violated → review"
    );
    assert_eq!(outcome.exit_code(), 1);
    assert_eq!(
        metric_check(&outcome, TARGET, Metric::EsrNamcore),
        &MetricOutcome::Ok
    );
    assert_matches!(
        metric_check(&outcome, TARGET, Metric::EsrF64),
        MetricOutcome::SafetyCeiling { .. },
        "f64 ESR above its rounded safety ceiling"
    );
}

// ── Phases (PERF-006 domain separation) ────────────────────────────────────

#[test]
fn mandatory_phase_fail_and_not_run_count_without_double_counting() {
    let contract = load_contract();
    // golden_vectors FAIL (counted once by the FAIL scan); quick_parity
    // absent (NOT_RUN, counted once); regression_gate absent → NOT_VERIFIED.
    let phases = [("golden_vectors", "FAIL"), ("reference_oracle_f64", "PASS")];
    let report = build_report(&contract, &phases, canonical_fidelity, canonical_latency);
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 2 });
    assert!(
        outcome.performance.is_not_verified(),
        "absent regression_gate is NOT_RUN → performance not certified"
    );
    assert_eq!(outcome.exit_code(), 1);
}

#[test]
fn non_mandatory_phase_fail_still_counts_as_fidelity_violation() {
    let contract = load_contract();
    let phases = [
        ("golden_vectors", "PASS"),
        ("reference_oracle_f64", "PASS"),
        ("quick_parity", "PASS"),
        ("regression_gate", "PASS"),
        ("freshness", "FAIL"),
    ];
    let report = build_report(&contract, &phases, canonical_fidelity, canonical_latency);
    let outcome = verify(&contract, &report);

    assert_matches!(
        outcome.fidelity,
        FidelityVerdict::Fail { violations: 1 },
        "any non-regression phase FAIL belongs to the fidelity domain (bash scan)"
    );
}

// ── Fidelity matching ──────────────────────────────────────────────────────

#[test]
fn missing_label_is_fail_closed_and_optional_is_skipped() {
    let contract = load_contract();

    // Dropping only the optional entry → no violation.
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            if e.id == "evh-5150-lite@48000:live" {
                None
            } else {
                canonical_fidelity(e)
            }
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);
    assert!(
        outcome.fidelity.is_ok(),
        "optional absence must not fail: {outcome:?}"
    );
    assert!(outcome.fidelity_checks.iter().any(|c| {
        c.id == "evh-5150-lite@48000:live" && c.outcome == FidelityOutcome::OptionalSkipped
    }));
    assert_eq!(outcome.exit_code(), 0);

    // Dropping a mandatory entry → MISSING_LABEL fail-closed.
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            if e.id == "bosswn-standard@48000:live" {
                None
            } else {
                canonical_fidelity(e)
            }
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);
    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    let check = outcome
        .fidelity_checks
        .iter()
        .find(|c| c.id == "bosswn-standard@48000:live")
        .expect("check must exist");
    assert_eq!(check.outcome, FidelityOutcome::MissingLabel);
    assert_eq!(outcome.exit_code(), 1);
}

#[test]
fn old_report_quick_parity_label_resolves_via_ids_alias() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == "convnet-test@48000:live" {
                obj["label"] = json!("Quick ConvNet @48000 Live");
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert!(
        outcome.fidelity.is_ok(),
        "alias must resolve to the canonical entry"
    );
    assert_eq!(outcome.exit_code(), 0);
}

// ── Metric envelopes ───────────────────────────────────────────────────────

#[test]
fn non_finite_esr_is_malformed_and_fail_closed() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["esr"] = json!("inf");
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    assert_matches!(
        metric_check(&outcome, TARGET, Metric::EsrNamcore),
        MetricOutcome::Malformed(MalformedReason::NonFinite(raw)) if raw == "inf"
    );
    // NAMCore malformed + f64 ok → divergence review.
    assert_eq!(outcome.review_required, 1);
}

#[test]
fn missing_metrics_are_malformed() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["esr"] = json!("");
                // Absent `snr_db` (foreign/corrupt writer) must stay
                // fail-closed — distinct from the canonical `null` encoding
                // of perfect parity (P0.T3).
                obj.as_object_mut().unwrap().remove("snr_db");
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 2 });
    assert_eq!(
        metric_check(&outcome, TARGET, Metric::EsrNamcore),
        &MetricOutcome::Malformed(MalformedReason::Missing)
    );
    assert_eq!(
        metric_check(&outcome, TARGET, Metric::SnrDb),
        &MetricOutcome::Malformed(MalformedReason::Missing)
    );
}

/// A `null` SNR is the canonical sink's representation of non-finite SNR —
/// bit-identical perfect parity (`+∞` dB), which is above any envelope floor
/// (P0.T3). It must not fail the contract; the non-finite *literal*
/// sentinels (`"inf"`, `"-inf"`, `"nan"`) still fail closed.
#[test]
fn null_snr_from_perfect_parity_is_ok() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["snr_db"] = json!(null);
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert!(outcome.fidelity.is_ok(), "fidelity must be OK: {outcome:?}");
    assert_eq!(
        metric_check(&outcome, TARGET, Metric::SnrDb),
        &MetricOutcome::Ok
    );
}

/// A non-finite SNR *literal* never comes from the canonical sink — its
/// presence signals a foreign/corrupt writer and stays fail-closed.
#[test]
fn non_finite_snr_literal_is_malformed() {
    let contract = load_contract();
    for sentinel in ["inf", "-inf", "nan"] {
        let report = build_report(
            &contract,
            &ALL_PASS,
            |e| {
                let mut obj = canonical_fidelity(e)?;
                if e.id == TARGET {
                    obj["snr_db"] = json!(sentinel);
                }
                Some(obj)
            },
            canonical_latency,
        );
        let outcome = verify(&contract, &report);
        assert_matches!(
            metric_check(&outcome, TARGET, Metric::SnrDb),
            MetricOutcome::Malformed(MalformedReason::NonFinite(raw)) if raw == sentinel
        );
        assert_matches!(
            outcome.fidelity,
            FidelityVerdict::Fail { violations: 1 },
            "{sentinel}"
        );
    }
}

#[test]
fn f64_baseline_present_but_not_measured_is_missing() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                obj["esr_f64"] = json!(null);
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    assert_eq!(
        metric_check(&outcome, TARGET, Metric::EsrF64),
        &MetricOutcome::Missing
    );
    assert_eq!(
        outcome.review_required, 0,
        "unmeasured f64 is MISSING, not an oracle divergence"
    );
}

#[test]
fn snr_regression_violates_the_exact_floor() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                // Contract SNR is 110.7 dB → exact floor 104.7; 103.7 fails.
                obj["snr_db"] = json!("103.7");
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    let MetricOutcome::Envelope {
        limit, baseline, ..
    } = metric_check(&outcome, TARGET, Metric::SnrDb)
    else {
        panic!("expected exact SNR envelope violation");
    };
    assert_eq!(*baseline, 110.7);
    assert!(
        (*limit - 104.7).abs() < 1e-9,
        "SNR compares the exact limit, no %.2e rounding: {limit}"
    );
}

#[test]
fn mrstft_regression_violates_the_exact_ceiling() {
    let contract = load_contract();
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                // Contract MR-STFT 2.8e-5 → exact ceiling 2.8e-4; 10.5× fails.
                obj["mrstft"] = json!(e.mrstft * 10.5);
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert_matches!(outcome.fidelity, FidelityVerdict::Fail { violations: 1 });
    assert_matches!(
        metric_check(&outcome, TARGET, Metric::Mrstft),
        MetricOutcome::Envelope { .. }
    );
}

#[test]
fn directional_divergence_triggers_review_without_envelope_violation() {
    let contract = load_contract();
    // NAMCore ESR improves (×0.84) while f64 ESR degrades (×1.16): ratios
    // 0.84 and 1.16 → rn < 0.85 && rf > 1.15 → REVIEW_REQUIRED, while both
    // oracles stay inside their noise envelopes (3×).
    let report = build_report(
        &contract,
        &ALL_PASS,
        |e| {
            let mut obj = canonical_fidelity(e)?;
            if e.id == TARGET {
                let esr = e.esr_namcore;
                let f64v = e.esr_f64.expect("TARGET has an f64 baseline");
                obj["esr"] = json!(format!("{}", esr * 0.84));
                obj["esr_f64"] = json!(format!("{}", f64v * 1.16));
            }
            Some(obj)
        },
        canonical_latency,
    );
    let outcome = verify(&contract, &report);

    assert!(
        outcome.fidelity.is_ok(),
        "both oracles inside their envelopes"
    );
    assert_eq!(
        outcome.review_required, 1,
        "opposite directional moves need review"
    );
    assert_eq!(outcome.exit_code(), 1);
}

// ── Performance domain ─────────────────────────────────────────────────────

#[test]
fn latency_regression_and_missing_label_are_performance_violations() {
    let contract = load_contract();

    // One latency far above the exact envelope → PERFORMANCE: FAIL.
    let report = build_report(&contract, &ALL_PASS, canonical_fidelity, |e| {
        let mut obj = canonical_latency(e)?;
        if e.id == "RT_WaveNet_Std_CH16" {
            obj["median_latency_us"] = json!(200.0);
        }
        Some(obj)
    });
    let outcome = verify(&contract, &report);
    assert!(outcome.fidelity.is_ok());
    assert_matches!(
        outcome.performance,
        PerformanceVerdict::Fail { violations: 1 }
    );
    assert_eq!(outcome.exit_code(), 1);
    let check = outcome
        .perf_checks
        .iter()
        .find(|c| c.id == "RT_WaveNet_Std_CH16")
        .expect("perf check must exist");
    let PerfResult::Regressed {
        median_us,
        limit_us,
        baseline_us,
    } = &check.result
    else {
        panic!("expected latency regression");
    };
    assert_eq!(*median_us, 200.0);
    assert_eq!(*baseline_us, 36.9);
    assert!(
        (*limit_us - 40.59).abs() < 1e-9,
        "latency compares the exact limit max(36.9×1.10, 36.9+0.05): {limit_us}"
    );

    // One latency record missing → MISSING_LABEL fail-closed.
    let report = build_report(&contract, &ALL_PASS, canonical_fidelity, |e| {
        if e.id == "RT_WaveNet_Std_CH16" {
            None
        } else {
            canonical_latency(e)
        }
    });
    let outcome = verify(&contract, &report);
    assert_matches!(
        outcome.performance,
        PerformanceVerdict::Fail { violations: 1 }
    );
    let check = outcome
        .perf_checks
        .iter()
        .find(|c| c.id == "RT_WaveNet_Std_CH16")
        .expect("perf check must exist");
    assert_eq!(check.result, PerfResult::MissingLabel);
}

#[test]
fn latency_label_normalization_matches_contract() {
    let contract = load_contract();
    // "×"→"x" normalization: report label with the character variant.
    let report = build_report(&contract, &ALL_PASS, canonical_fidelity, |e| {
        let mut obj = canonical_latency(e)?;
        if e.id == "RT_WaveNet_Std_CH16" {
            obj["label"] = json!(e.label.replace(' ', "  "));
        }
        Some(obj)
    });
    let outcome = verify(&contract, &report);
    assert!(
        outcome.performance.is_ok(),
        "normalized label must match: {outcome:?}"
    );
}

/// The Criterion bench labels of `regression_gate.rs` differ from the
/// contract performance ids for `RT_Linear` and the DSP benches — the verify
/// must resolve them through `ids::resolve_rt_contract_id`, otherwise the
/// dashboard reports `MISSING_LABEL` for benches that ran and passed
/// (P0.T3).
#[test]
fn latency_bench_labels_resolve_to_contract_ids() {
    let contract = load_contract();
    let report = build_report(&contract, &ALL_PASS, canonical_fidelity, |e| {
        let bench_label = match e.id.as_str() {
            "RT_Linear_RF2048" => "RT_Linear",
            "RT_DSP_Resampler_44k_to_48k" => "RT_DSP_Resampler_44k1_to_48k",
            "RT_DSP_Pipeline_Base" => "RT_DSP_Pipeline_Base_NoOS",
            "RT_DSP_Pipeline_HQ" => "RT_DSP_Pipeline_HQ_4xOS",
            id => id,
        };
        Some(json!({
            "kind": "latency",
            "label": bench_label,
            "median_latency_us": e.median_latency_us,
        }))
    });
    let outcome = verify(&contract, &report);
    assert!(
        outcome.performance.is_ok(),
        "bench labels must resolve via the RT_* alias table: {outcome:?}"
    );
    assert_eq!(outcome.perf_checks.len(), contract.performance.len());
    assert!(
        outcome
            .perf_checks
            .iter()
            .all(|c| matches!(c.result, PerfResult::Ok { .. }))
    );
}

// ── Envelope arithmetic (literal port notes) ───────────────────────────────

#[test]
fn esr_limits_apply_printf_2e_rounding() {
    let env = Envelopes::policy_v1();

    let (noise, safety) = esr_limits(1e-13, &env.esr_namcore);
    assert_eq!(noise, 3e-13, "noise max(b×3, b+5e-14) prints as 3.00e-13");
    assert_eq!(safety, 1e-12, "safety max(b×10, 1e-12) prints as 1.00e-12");

    // 6.11999999e-14 prints as 6.12e-14 → the rounded gate is looser than
    // the raw limit, and a current between the two passes (bash behavior).
    let (noise, _) = esr_limits(1.11999999e-14, &env.esr_namcore);
    assert_eq!(noise, 6.12e-14);
    assert!(noise > 6.11999999e-14);
    assert!(6.119999995e-14 < noise);
}

#[test]
fn round_printf_2e_rounds_three_significant_digits() {
    assert_eq!(round_printf_2e(7.31e-14), 7.31e-14);
    assert_eq!(round_printf_2e(8.5e-12), 8.5e-12);
    assert_eq!(round_printf_2e(2.55e-11), 2.55e-11);
    // Carry: 9.997e-15 prints as 1.00e-14.
    assert_eq!(round_printf_2e(9.997e-15), 1e-14);
    assert_eq!(round_printf_2e(0.0), 0.0);
    assert_eq!(round_printf_2e(f64::INFINITY), f64::INFINITY);
}

// ── Report parser ──────────────────────────────────────────────────────────

#[test]
fn report_parser_routes_kinds_and_skips_unknown() {
    let input = r#"{"phase_id":"golden_vectors","status":"PASS"}
{"kind":"fidelity","label":"A","esr":"1","esr_f64":"2"}
{"kind":"latency","label":"RT_X","median_latency_us":3.5}
{"kind":"build_metadata","git_commit":"abc"}
{"phase_id":"regression_gate","status":"FAIL"}"#;
    let report = parse_verify_report(input).expect("report must parse");

    assert_eq!(report.phases.len(), 2);
    assert_eq!(report.phases[0].phase_id, "golden_vectors");
    assert_eq!(report.phases[1].status, "FAIL");
    assert_eq!(report.fidelity.len(), 1);
    assert_eq!(report.fidelity[0].label, "A");
    assert_eq!(report.latency.len(), 1);
    assert_eq!(report.latency[0].median_latency_us, 3.5);
}

#[test]
fn report_parser_fails_closed_on_bad_records() {
    assert_matches!(
        parse_verify_report("not json\n"),
        Err(VerifyError::MalformedLine { line: 1, .. })
    );
    assert_matches!(
        parse_verify_report(r#"{"phase_id":1,"status":"PASS"}"#),
        Err(VerifyError::InvalidPhaseRecord { line: 1 })
    );
    assert_matches!(
        parse_verify_report(r#"{"kind":"latency","label":"X","median_latency_us":"slow"}"#),
        Err(VerifyError::InvalidLatencyRecord { line: 1 })
    );
    assert_matches!(
        parse_verify_report(r#"{"kind":"latency","label":"X","median_latency_us":null}"#),
        Err(VerifyError::InvalidLatencyRecord { line: 1 })
    );
}
