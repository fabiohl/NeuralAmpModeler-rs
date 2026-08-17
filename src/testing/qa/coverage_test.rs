// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the baseline coverage cross-check (S3.T2) — the acceptance
//! cases of `utils/tests-long.sh:869-899` mirrored verbatim (F-24), plus the
//! sed-parity corners.

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp dir (no two tests collide on the same criterion root).
fn temp_root() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nam-coverage-{}-{n}", std::process::id()))
}

/// The four-line Criterion log of the acceptance block.
fn crit_log() -> &'static str {
    "Benchmarking RT_A: Warming up for 1.0000 s\n\
     Benchmarking RT_A: Collecting 100 samples\n\
     Benchmarking RT_B: Warming up for 1.0000 s\n\
     Benchmarking RT_C: Warming up for 1.0000 s\n"
}

/// The acceptance cases of `utils/tests-long.sh:869-899`, verbatim (F-24).
#[test]
fn tests_long_f24_cases_are_mirrored() {
    let crit = temp_root();
    for id in ["RT_A", "RT_B"] {
        fs::create_dir_all(crit.join(id).join("ci-baseline")).unwrap();
    }

    // missing_baseline_coverage: parse ok -> RT_C listed without a series.
    let missing = missing_baseline_coverage(crit_log(), &crit, "ci-baseline").unwrap();
    assert_eq!(missing, vec!["RT_C".to_string()]);

    // executed_bench_ids dedups (RT_A twice) and sorts.
    assert_eq!(executed_bench_ids(crit_log()), ["RT_A", "RT_B", "RT_C"]);

    // Full coverage -> empty missing list.
    fs::create_dir_all(crit.join("RT_C").join("ci-baseline")).unwrap();
    assert_eq!(
        missing_baseline_coverage(crit_log(), &crit, "ci-baseline").unwrap(),
        Vec::<String>::new()
    );

    // Unparseable log -> Err (blind gate, fail-closed).
    assert_eq!(
        missing_baseline_coverage(
            "garbage log with no criterion lines\n",
            &crit,
            "ci-baseline"
        ),
        Err(BaselineCoverageGap)
    );

    // Absent log maps to empty text (the bash `sed` of a missing file) -> Err.
    assert_eq!(
        missing_baseline_coverage("", &crit, "ci-baseline"),
        Err(BaselineCoverageGap)
    );
}

/// A line without the `Benchmarking ` prefix or without a colon after the
/// id never matches the sed — only real Criterion lines yield ids.
#[test]
fn sed_requires_prefix_and_colon() {
    let log = "Warming up for 1.0000 s\n\
               Benchmarking RT_A without a colon\n\
               Benchmarking RT_B: Collecting 100 samples\n";
    assert_eq!(executed_bench_ids(log), ["RT_B"]);
}

/// The id capture stops at the **first** colon (`[^:]*`) and keeps the raw
/// spacing (no trimming) exactly like the sed. A trailing space still
/// collates like the bare id in both sort schemes (`sort -u` and bytes).
#[test]
fn id_capture_stops_at_first_colon_and_is_raw() {
    let log = "Benchmarking RT_X_1:2:3: more\nBenchmarking RT_A : x\n";
    assert_eq!(executed_bench_ids(log), ["RT_A ", "RT_X_1"]);
}

/// Sorting is lexicographic byte order (`sort -u`) over the deduplicated set.
#[test]
fn ids_are_deduplicated_and_sorted() {
    let log =
        "Benchmarking RT_C: a\nBenchmarking RT_A: b\nBenchmarking RT_C: c\nBenchmarking RT_B: d\n";
    assert_eq!(executed_bench_ids(log), ["RT_A", "RT_B", "RT_C"]);
}

/// Missing ids are listed in executed (sorted) order, space-joined by the
/// caller — the bash echoes them `missing:+:missing` without reordering.
#[test]
fn missing_list_preserves_sorted_order() {
    let crit = temp_root();
    fs::create_dir_all(crit.join("RT_B").join("ci-baseline")).unwrap();
    let missing = missing_baseline_coverage(crit_log(), &crit, "ci-baseline").unwrap();
    assert_eq!(missing, ["RT_A", "RT_C"]);
}

/// A custom baseline name is honored (`${3:-ci-baseline}` has no effect when
/// the argument is provided).
#[test]
fn custom_baseline_name_is_used() {
    let crit = temp_root();
    fs::create_dir_all(crit.join("RT_A").join("ci-baseline")).unwrap();
    fs::create_dir_all(crit.join("RT_A").join("nightly")).unwrap();
    fs::create_dir_all(crit.join("RT_B").join("nightly")).unwrap();
    let missing = missing_baseline_coverage(crit_log(), &crit, "nightly").unwrap();
    assert_eq!(missing, ["RT_C"]);
}

/// The series check is `[ -d ]`: a **file** named `ci-baseline` does not
/// count as coverage.
#[test]
fn a_file_named_like_the_series_is_not_coverage() {
    let crit = temp_root();
    fs::create_dir_all(crit.join("RT_A")).unwrap();
    fs::write(crit.join("RT_A").join("ci-baseline"), "not a dir").unwrap();
    fs::create_dir_all(crit.join("RT_B").join("ci-baseline")).unwrap();
    let missing = missing_baseline_coverage(crit_log(), &crit, "ci-baseline").unwrap();
    assert_eq!(missing, ["RT_A", "RT_C"]);
}

/// Empty/whitespace-only id captures are dropped by bash word splitting
/// (`for id in $ids`) — both for the blind-gate check and the missing list.
#[test]
fn empty_id_captures_are_dropped_like_bash_word_splitting() {
    let only_empty = "Benchmarking : no id\n";
    assert_eq!(executed_bench_ids(only_empty), [""]);
    assert_eq!(
        missing_baseline_coverage(only_empty, &temp_root(), "ci-baseline"),
        Err(BaselineCoverageGap)
    );

    let crit = temp_root();
    fs::create_dir_all(crit.join("RT_A").join("ci-baseline")).unwrap();
    let mixed = "Benchmarking : no id\nBenchmarking RT_A: ok\nBenchmarking    : also empty\n";
    assert_eq!(
        missing_baseline_coverage(mixed, &crit, "ci-baseline").unwrap(),
        Vec::<String>::new(),
        "empty captures must not be listed as missing"
    );
}
