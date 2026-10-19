// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests of the machine regression verdict (B2 / R-3) over synthetic
//! Criterion `change/estimates.json` artifacts.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::json;

use super::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp criterion root (no two tests collide).
fn temp_root() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nam-verdict-{}-{n}", std::process::id()))
}

/// Criterion `change/estimates.json` body for one benchmark.
fn change_json(point: f64, lower: f64, upper: f64) -> String {
    json!({
        "mean": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": lower,
                "upper_bound": upper
            },
            "point_estimate": point,
            "standard_error": 0.001
        },
        "median": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": lower,
                "upper_bound": upper
            },
            "point_estimate": point,
            "standard_error": 0.001
        }
    })
    .to_string()
}

fn write_change(root: &Path, id: &str, body: &str) {
    let path = change_estimates_path(root, id);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, body).unwrap();
}

fn crit_log(ids: &[&str]) -> String {
    ids.iter()
        .map(|id| format!("Benchmarking {id}: Warming up for 1.0000 s\n"))
        .collect()
}

/// A clean run: every executed benchmark has a comparison artifact whose
/// mean-change CI stays inside the ±5% noise band → no findings.
#[test]
fn clean_run_within_noise_has_no_findings() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.012, 0.004, 0.020));
    write_change(&root, "RT_B", &change_json(-0.030, -0.045, -0.015));

    let outcome = regression_verdict(&crit_log(&["RT_A", "RT_B"]), &root, 0.05).unwrap();
    assert_eq!(outcome.executed, 2);
    assert!(outcome.findings.is_empty(), "{:?}", outcome.findings);
}

/// A regression: the whole mean-change CI sits above +5% → machine finding
/// with the exact estimate and bounds.
#[test]
fn regression_when_whole_ci_above_noise_band() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.020, 0.010, 0.030));
    write_change(&root, "RT_B", &change_json(0.080, 0.061, 0.099));

    let outcome = regression_verdict(&crit_log(&["RT_A", "RT_B"]), &root, 0.05).unwrap();
    assert_eq!(
        outcome.findings,
        vec![VerdictFinding::Regression {
            id: "RT_B".to_string(),
            point_estimate: 0.080,
            lower_bound: 0.061,
            upper_bound: 0.099,
            noise: 0.05,
            delta_ns: None,
        }]
    );
}

/// Whole-CI improvements and CIs crossing the band are never regressions.
#[test]
fn improvement_and_straddling_ci_are_not_regressions() {
    let root = temp_root();
    write_change(&root, "RT_IMP", &change_json(-0.090, -0.110, -0.070));
    write_change(&root, "RT_ST", &change_json(0.060, -0.010, 0.090));

    let outcome = regression_verdict(&crit_log(&["RT_IMP", "RT_ST"]), &root, 0.05).unwrap();
    assert!(outcome.findings.is_empty(), "{:?}", outcome.findings);
}

/// A lower bound exactly on the noise band is not a regression (`>`, like
/// Criterion's `compare_to_threshold`).
#[test]
fn lower_bound_exactly_at_noise_is_not_regression() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.052, 0.05, 0.10));

    let outcome = regression_verdict(&crit_log(&["RT_A"]), &root, 0.05).unwrap();
    assert!(outcome.findings.is_empty(), "{:?}", outcome.findings);
}

/// An executed benchmark without a readable `change/estimates.json` makes the
/// verdict blind for that id — nothing is certified (R-3 fail-closed).
#[test]
fn missing_change_estimates_is_a_blind_finding() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.010, 0.001, 0.020));

    let outcome = regression_verdict(&crit_log(&["RT_A", "RT_B"]), &root, 0.05).unwrap();
    assert_eq!(
        outcome.findings,
        vec![VerdictFinding::MissingEstimates {
            id: "RT_B".to_string(),
        }]
    );
}

/// A corrupt comparison artifact is the same blind signal as an absent one.
#[test]
fn corrupt_change_estimates_is_a_blind_finding() {
    let root = temp_root();
    write_change(&root, "RT_A", "{not criterion change estimates");

    let outcome = regression_verdict(&crit_log(&["RT_A"]), &root, 0.05).unwrap();
    assert_eq!(
        outcome.findings,
        vec![VerdictFinding::MissingEstimates {
            id: "RT_A".to_string(),
        }]
    );
}

/// An empty log, a garbage log, or a log with only empty ids is the blind
/// gate (same semantics as coverage's `BaselineCoverageGap`).
#[test]
fn empty_or_garbage_log_is_the_blind_gate() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.010, 0.001, 0.020));

    assert_eq!(regression_verdict("", &root, 0.05), Err(VerdictBlind));
    assert_eq!(
        regression_verdict("garbage with no Benchmarking lines\n", &root, 0.05),
        Err(VerdictBlind)
    );
    assert_eq!(
        regression_verdict(
            "Benchmarking : no id\nBenchmarking   : blank\n",
            &root,
            0.05
        ),
        Err(VerdictBlind),
        "empty/whitespace-only ids are dropped like bash word splitting"
    );
}

/// An expected benchmark absent from the log produces a `MissingExpectedBenchmark` finding.
#[test]
fn missing_expected_benchmark_is_reported_in_findings() {
    let root = temp_root();
    write_change(&root, "RT_A", &change_json(0.010, 0.001, 0.020));

    let expected = vec!["RT_A".to_string(), "RT_MISSING".to_string()];
    let outcome =
        regression_verdict_with_expected(&crit_log(&["RT_A"]), &root, 0.05, Some(&expected))
            .unwrap();

    assert_eq!(outcome.executed, 1);
    assert_eq!(
        outcome.findings,
        vec![VerdictFinding::MissingExpectedBenchmark {
            id: "RT_MISSING".to_string(),
        }]
    );
}

fn write_new(root: &Path, id: &str, point_estimate_ns: f64) {
    let path = new_estimates_path(root, id);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    let body = change_json(
        point_estimate_ns,
        point_estimate_ns * 0.99,
        point_estimate_ns * 1.01,
    );
    fs::write(path, body).unwrap();
}

/// When check_stability is enabled, improvements exceeding the noise band
/// are flagged as UnstableBaseline.
#[test]
fn stability_check_flags_improvements_as_unstable_baseline() {
    let root = temp_root();
    // Improved by -15.59% (CI [-15.74%, -14.84%])
    write_change(&root, "RT_WAVENET", &change_json(-0.1559, -0.1574, -0.1484));
    write_new(&root, "RT_WAVENET", 12890.0); // 12.89 µs

    // Default options: stability checking is disabled
    let default_outcome = regression_verdict_with_options(
        &crit_log(&["RT_WAVENET"]),
        &root,
        &VerdictOptions::default(),
    )
    .unwrap();
    assert!(default_outcome.findings.is_empty());

    // With check_stability = true
    let options = VerdictOptions {
        check_stability: true,
        ..VerdictOptions::default()
    };
    let outcome =
        regression_verdict_with_options(&crit_log(&["RT_WAVENET"]), &root, &options).unwrap();
    assert_eq!(outcome.findings.len(), 1);
    match &outcome.findings[0] {
        VerdictFinding::UnstableBaseline {
            id,
            point_estimate,
            lower_bound,
            upper_bound,
            delta_ns,
            ..
        } => {
            assert_eq!(id, "RT_WAVENET");
            assert_eq!(*point_estimate, -0.1559);
            assert_eq!(*lower_bound, -0.1574);
            assert_eq!(*upper_bound, -0.1484);
            assert!(delta_ns.is_some());
            let d = delta_ns.unwrap();
            // Expected: 12890 * (-0.1559 / (1 - 0.1559)) ≈ -2380.5 ns
            assert!(d < -2300.0 && d > -2400.0, "unexpected delta_ns: {d}");
        }
        other => panic!("expected UnstableBaseline, got: {other:?}"),
    }
}

/// When abs_noise_floor_ns is configured, relative regressions whose absolute
/// delta is within the floor are classified as NoiseIgnored.
#[test]
fn absolute_noise_floor_ignores_subthreshold_microbenchmark_noise() {
    let root = temp_root();
    // +10.49% change on a 16.04 ns microbenchmark (delta = +1.52 ns <= 2.0 ns floor)
    write_change(
        &root,
        "Delay_Mono/64_samp_d0",
        &change_json(0.1049, 0.1041, 0.1058),
    );
    write_new(&root, "Delay_Mono/64_samp_d0", 16.04);

    let options = VerdictOptions {
        abs_noise_floor_ns: Some(2.0),
        ..VerdictOptions::default()
    };

    let outcome =
        regression_verdict_with_options(&crit_log(&["Delay_Mono/64_samp_d0"]), &root, &options)
            .unwrap();
    assert_eq!(outcome.findings.len(), 1);
    match &outcome.findings[0] {
        VerdictFinding::NoiseIgnored {
            id,
            point_estimate,
            delta_ns,
            floor_ns,
            ..
        } => {
            assert_eq!(id, "Delay_Mono/64_samp_d0");
            assert_eq!(*point_estimate, 0.1049);
            assert!(*delta_ns <= 2.0 && *delta_ns >= 1.5);
            assert_eq!(*floor_ns, 2.0);
        }
        other => panic!("expected NoiseIgnored, got: {other:?}"),
    }
}

/// When relative change is a regression and absolute delta exceeds abs_noise_floor_ns,
/// it is flagged as Regression with delta_ns recorded.
#[test]
fn absolute_noise_floor_flags_regression_when_delta_exceeds_floor() {
    let root = temp_root();
    // +10% change on a 100 ns benchmark (delta = 9.09 ns > 2.0 ns floor)
    write_change(&root, "RT_Slow", &change_json(0.10, 0.08, 0.12));
    write_new(&root, "RT_Slow", 100.0);

    let options = VerdictOptions {
        abs_noise_floor_ns: Some(2.0),
        ..VerdictOptions::default()
    };

    let outcome =
        regression_verdict_with_options(&crit_log(&["RT_Slow"]), &root, &options).unwrap();
    assert_eq!(outcome.findings.len(), 1);
    match &outcome.findings[0] {
        VerdictFinding::Regression {
            id,
            point_estimate,
            delta_ns,
            ..
        } => {
            assert_eq!(id, "RT_Slow");
            assert_eq!(*point_estimate, 0.10);
            assert!(delta_ns.is_some());
            assert!(delta_ns.unwrap() > 2.0);
        }
        other => panic!("expected Regression, got: {other:?}"),
    }
}

/// When abs_noise_floor_ns is configured but new/estimates.json is missing,
/// the gate fails closed by reporting Regression with delta_ns = None.
#[test]
fn absolute_noise_floor_without_new_estimates_fails_closed() {
    let root = temp_root();
    write_change(&root, "RT_MissingNew", &change_json(0.10, 0.08, 0.12));

    let options = VerdictOptions {
        abs_noise_floor_ns: Some(2.0),
        ..VerdictOptions::default()
    };

    let outcome =
        regression_verdict_with_options(&crit_log(&["RT_MissingNew"]), &root, &options).unwrap();
    assert_eq!(outcome.findings.len(), 1);
    match &outcome.findings[0] {
        VerdictFinding::Regression { id, delta_ns, .. } => {
            assert_eq!(id, "RT_MissingNew");
            assert_eq!(*delta_ns, None);
        }
        other => panic!("expected Regression, got: {other:?}"),
    }
}
