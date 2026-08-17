// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance-baseline persistence (S3.T3) — literal port of
//! `list_top_level_baselines` / `sanitize_nested_baselines` /
//! `replace_copy_dir` / `persist_baseline` / `restore_baseline`
//! (`tests-performance-regression.sh:219-318`).
//!
//! Criterion's `target/criterion/` is a **transient** working area: the
//! persisted store (`.performance-baselines/`, gitignored) holds only the
//! top-level `…/<bench>/<baseline>/` series. Persist/restore use
//! replace-copy of top-level series only; nested
//! `<baseline>/<baseline>/…` paths (historical `cp -a` into an existing
//! dest) are sanitized and never re-copied — the same semantics the bash
//! tests in `tests/scripts/test_regression_guard.sh` scenario 4 exercise.
//!
//! The Rust implementation replaces the fragile `cp -a` tree dance: every
//! copy is recursive and explicit, and nested series are removed after each
//! copy so a stale nested tree can never be re-introduced.

use std::fs;
use std::io;
use std::path::PathBuf;

/// Default Criterion baseline name (bash `NAM_BASELINE_NAME` default).
pub const DEFAULT_BASELINE_NAME: &str = "ci-baseline";

/// Top-level baseline series under `root`: every `root/<id>/<baseline>/`
/// directory, in sorted order — the bash
/// `find "$root" -mindepth 2 -maxdepth 2 -type d -name "$BASELINE_NAME" | sort`.
/// A missing root yields an empty list.
pub fn list_top_level_baselines(root: &std::path::Path, baseline_name: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let id_dir = entry.path();
        if !id_dir.is_dir() {
            continue;
        }
        let series = id_dir.join(baseline_name);
        if series.is_dir() {
            out.push(series);
        }
    }
    out.sort();
    out
}

/// Removes nested baseline dirs (depth ≥ 3, deepest first) under `root` —
/// the bash `find "$root" -mindepth 3 -type d -name "$BASELINE_NAME"` with
/// `sort -rn` by path length. Returns the number of removed dirs. Top-level
/// series (depth 2) are never touched.
pub fn sanitize_nested_baselines(root: &std::path::Path, baseline_name: &str) -> u32 {
    if !root.is_dir() {
        return 0;
    }
    let mut nested = Vec::new();
    walk_collect(root, baseline_name, 0, 3, &mut nested);
    nested.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    let mut removed = 0;
    for path in nested {
        if path.is_dir() {
            let _ = fs::remove_dir_all(&path);
            removed += 1;
        }
    }
    removed
}

/// Persists every top-level series of `criterion_root` into `baseline_dir`
/// (replace-copy, nested sanitized on both sides) — the bash
/// `persist_baseline`. Returns the number of persisted series.
pub fn persist_baseline(
    baseline_dir: &std::path::Path,
    criterion_root: &std::path::Path,
    baseline_name: &str,
) -> io::Result<usize> {
    fs::create_dir_all(baseline_dir)?;
    let mut count = 0;
    if criterion_root.is_dir() {
        sanitize_nested_baselines(criterion_root, baseline_name);
        for series in list_top_level_baselines(criterion_root, baseline_name) {
            let rel = series
                .strip_prefix(criterion_root)
                .expect("series paths come from list_top_level_baselines");
            replace_copy_dir(&series, &baseline_dir.join(rel), baseline_name)?;
            count += 1;
        }
        sanitize_nested_baselines(baseline_dir, baseline_name);
    }
    Ok(count)
}

/// Restores every top-level series of `baseline_dir` into `criterion_root`
/// (replace-copy, nested sanitized on both sides) — the bash
/// `restore_baseline`. Returns the number of restored series; a missing
/// baseline dir yields 0 without touching the criterion root.
pub fn restore_baseline(
    baseline_dir: &std::path::Path,
    criterion_root: &std::path::Path,
    baseline_name: &str,
) -> io::Result<usize> {
    if !baseline_dir.is_dir() {
        return Ok(0);
    }
    sanitize_nested_baselines(baseline_dir, baseline_name);
    if criterion_root.exists() {
        fs::remove_dir_all(criterion_root)?;
    }
    fs::create_dir_all(criterion_root)?;
    let mut count = 0;
    for series in list_top_level_baselines(baseline_dir, baseline_name) {
        let rel = series
            .strip_prefix(baseline_dir)
            .expect("series paths come from list_top_level_baselines");
        replace_copy_dir(&series, &criterion_root.join(rel), baseline_name)?;
        count += 1;
    }
    Ok(count)
}

/// Replace-copy of one series dir: removes any existing `dest`, copies
/// `src` recursively into it, then removes nested baseline dirs inside the
/// copy — the bash `replace_copy_dir` (`rm -rf dest`, `cp -a src dest`,
/// `find dest -mindepth 1 -type d -name "$BASELINE_NAME" -exec rm -rf {} +`).
fn replace_copy_dir(
    src: &std::path::Path,
    dest: &std::path::Path,
    baseline_name: &str,
) -> io::Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest)?;
    }
    if let Some(parent) = dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    copy_recursive(src, dest)?;
    let mut nested = Vec::new();
    walk_collect(dest, baseline_name, 0, 1, &mut nested);
    nested.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for path in nested {
        let _ = fs::remove_dir_all(&path);
    }
    Ok(())
}

/// Recursively copies `src` into `dest` (dirs and files; empty dirs
/// preserved) — the Rust `cp -a` of the replace-copy.
fn copy_recursive(src: &std::path::Path, dest: &std::path::Path) -> io::Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dest.join(entry.file_name());
        if from.is_dir() {
            copy_recursive(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Collects every directory named `baseline_name` at depth ≥ `min_depth`
/// (root has depth 0) into `out`.
fn walk_collect(
    dir: &std::path::Path,
    baseline_name: &str,
    depth: usize,
    min_depth: usize,
    out: &mut Vec<PathBuf>,
) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let child_depth = depth + 1;
        if path
            .file_name()
            .map(|n| n == baseline_name)
            .unwrap_or(false)
            && child_depth >= min_depth
        {
            out.push(path.clone());
        }
        walk_collect(&path, baseline_name, child_depth, min_depth, out);
    }
}

#[cfg(test)]
#[path = "baseline_store_test.rs"]
mod baseline_store_test;
