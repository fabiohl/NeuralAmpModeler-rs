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
