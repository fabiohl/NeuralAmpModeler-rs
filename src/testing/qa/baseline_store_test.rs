// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for the baseline store (S3.T3) — the acceptance of
//! `tests/scripts/test_regression_guard.sh` scenario 4 mirrored verbatim
//! (nested sanitize + replace-copy never nests), plus persist/restore
//! roundtrips.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Unique temp root (no two tests collide on the same store).
fn temp_root() -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("nam-baseline-store-{}-{n}", std::process::id()))
}

fn write_marker(path: &Path, content: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, content).unwrap();
}

fn read_marker(path: &Path) -> String {
    fs::read_to_string(path).unwrap()
}

fn nested_count(root: &Path, baseline_name: &str) -> usize {
    let mut nested = Vec::new();
    walk_collect(root, baseline_name, 0, 3, &mut nested);
    nested.len()
}

/// The acceptance of `tests/scripts/test_regression_guard.sh` scenario 4,
/// verbatim: a deep nested tree is sanitized (top-level marker preserved),
/// restore replace-copies the top-level series, and a second restore still
/// keeps a single flat baseline layer.
#[test]
fn scenario4_nested_sanitize_and_replace_copy_never_nest() {
    let work = temp_root();
    let baseline_dir = work.join("fake-baselines");
    let criterion = work.join("fake-criterion");
    let name = "ci-baseline";

    // Baseline dir: deep nest + top-level marker.
    write_marker(
        &baseline_dir
            .join("RT_Dummy")
            .join(name)
            .join(name)
            .join(name)
            .join("marker.txt"),
        "nested-deep",
    );
    write_marker(
        &baseline_dir.join("RT_Dummy").join(name).join("marker.txt"),
        "top-level",
    );
    // Criterion root: stale dest + already-nested tree.
    write_marker(
        &criterion
            .join("RT_Dummy")
            .join(name)
            .join(name)
            .join("marker.txt"),
        "already-nested",
    );
    write_marker(
        &criterion.join("RT_Dummy").join(name).join("marker.txt"),
        "stale-dest",
    );

    // sanitize_nested_baselines on the baseline dir.
    sanitize_nested_baselines(&baseline_dir, name);
    assert_eq!(
        nested_count(&baseline_dir, name),
        0,
        "nested baseline dirs must be removed"
    );
    assert_eq!(
        read_marker(&baseline_dir.join("RT_Dummy").join(name).join("marker.txt")),
        "top-level",
        "top-level marker must survive the sanitize"
    );

    // restore_baseline: replace-copy into the criterion root.
    let restored = restore_baseline(&baseline_dir, &criterion, name).unwrap();
    assert_eq!(restored, 1);
    assert_eq!(
        nested_count(&criterion, name),
        0,
        "restore must not re-introduce nested baseline dirs"
    );
    assert!(
        criterion
            .join("RT_Dummy")
            .join(name)
            .join("marker.txt")
            .is_file()
    );
    assert_eq!(
        read_marker(&criterion.join("RT_Dummy").join(name).join("marker.txt")),
        "top-level",
        "restore must replace-copy the top-level content (stale-dest gone)"
    );

    // Second restore must still not nest, with exactly one top-level series.
    let second = restore_baseline(&baseline_dir, &criterion, name).unwrap();
    assert_eq!(second, 1);
    assert_eq!(nested_count(&criterion, name), 0);
    assert_eq!(list_top_level_baselines(&criterion, name).len(), 1);

    let _ = fs::remove_dir_all(&work);
}

/// `persist_baseline` replace-copies criterion top-level series into the
/// store, sanitizing nested dirs on both sides; the store is the same flat
/// single layer.
#[test]
fn persist_replaces_and_sanitizes_both_sides() {
    let work = temp_root();
    let baseline_dir = work.join("store");
    let criterion = work.join("criterion");
    let name = "ci-baseline";

    write_marker(&criterion.join("RT_A").join(name).join("marker.txt"), "a");
    write_marker(&criterion.join("RT_B").join(name).join("marker.txt"), "b");
    // Nested leftovers in both roots.
    write_marker(
        &criterion
            .join("RT_A")
            .join(name)
            .join(name)
            .join("marker.txt"),
        "nested",
    );
    write_marker(
        &baseline_dir
            .join("RT_A")
            .join(name)
            .join(name)
            .join("marker.txt"),
        "stale-nested",
    );

    let persisted = persist_baseline(&baseline_dir, &criterion, name).unwrap();
    assert_eq!(persisted, 2);
    assert_eq!(nested_count(&criterion, name), 0);
    assert_eq!(nested_count(&baseline_dir, name), 0);
    assert_eq!(
        read_marker(&baseline_dir.join("RT_A").join(name).join("marker.txt")),
        "a"
    );
    assert_eq!(
        read_marker(&baseline_dir.join("RT_B").join(name).join("marker.txt")),
        "b"
    );
    assert_eq!(list_top_level_baselines(&baseline_dir, name).len(), 2);

    // Re-persist replaces the stale store content, never duplicating.
    write_marker(
        &criterion.join("RT_A").join(name).join("marker.txt"),
        "a-v2",
    );
    persist_baseline(&baseline_dir, &criterion, name).unwrap();
    assert_eq!(
        read_marker(&baseline_dir.join("RT_A").join(name).join("marker.txt")),
        "a-v2"
    );
    assert_eq!(list_top_level_baselines(&baseline_dir, name).len(), 2);

    let _ = fs::remove_dir_all(&work);
}

/// A missing baseline dir restores 0 series without touching the criterion
/// root; a missing criterion root persists 0 series.
#[test]
fn missing_roots_are_graceful() {
    let work = temp_root();
    let baseline_dir = work.join("store");
    let criterion = work.join("criterion");
    let name = "ci-baseline";

    fs::create_dir_all(&criterion).unwrap();
    assert_eq!(
        restore_baseline(&baseline_dir, &criterion, name).unwrap(),
        0
    );
    assert!(
        criterion.is_dir(),
        "missing store must not wipe the criterion root"
    );

    assert_eq!(
        persist_baseline(&baseline_dir, &criterion, name).unwrap(),
        0
    );
    assert!(baseline_dir.is_dir(), "persist creates the store dir");

    let _ = fs::remove_dir_all(&work);
}

/// A custom baseline name is honored end to end.
#[test]
fn custom_baseline_name_roundtrips() {
    let work = temp_root();
    let baseline_dir = work.join("store");
    let criterion = work.join("criterion");
    write_marker(
        &criterion.join("RT_A").join("nightly").join("marker.txt"),
        "n",
    );
    write_marker(
        &criterion
            .join("RT_A")
            .join("ci-baseline")
            .join("marker.txt"),
        "ignore-me",
    );

    assert_eq!(
        persist_baseline(&baseline_dir, &criterion, "nightly").unwrap(),
        1
    );
    assert!(
        baseline_dir
            .join("RT_A")
            .join("nightly")
            .join("marker.txt")
            .is_file()
    );
    assert!(!baseline_dir.join("RT_A").join("ci-baseline").exists());
    assert_eq!(list_top_level_baselines(&baseline_dir, "nightly").len(), 1);
    assert!(list_top_level_baselines(&baseline_dir, "ci-baseline").is_empty());

    let _ = fs::remove_dir_all(&work);
}

/// Files inside a series are preserved by the replace-copy (not just the
/// marker) — a mini Criterion series with an estimate JSON survives.
#[test]
fn series_content_is_preserved_recursively() {
    let work = temp_root();
    let baseline_dir = work.join("store");
    let criterion = work.join("criterion");
    let name = "ci-baseline";

    write_marker(
        &criterion.join("RT_DSP").join(name).join("estimates.json"),
        r#"{"mean": 1.5}"#,
    );
    write_marker(
        &criterion.join("RT_DSP").join(name).join("sample.json"),
        r#"{"t": [1, 2]}"#,
    );

    persist_baseline(&baseline_dir, &criterion, name).unwrap();
    let criterion2 = work.join("criterion2");
    restore_baseline(&baseline_dir, &criterion2, name).unwrap();
    assert_eq!(
        read_marker(&criterion2.join("RT_DSP").join(name).join("estimates.json")),
        r#"{"mean": 1.5}"#
    );
    assert_eq!(
        read_marker(&criterion2.join("RT_DSP").join(name).join("sample.json")),
        r#"{"t": [1, 2]}"#
    );

    let _ = fs::remove_dir_all(&work);
}
