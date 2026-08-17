// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Baseline coverage cross-check (F-24, S3.T2) — literal port of
//! `executed_bench_ids` / `missing_baseline_coverage`
//! (`tests-performance-regression.sh:229-249`).
//!
//! The Rust only classifies the Criterion log: every `Benchmarking <id>:`
//! line yields the id up to the first `:` (the same mental
//! `sed -n 's/^Benchmarking \([^:]*\):.*/\1/p'`), and coverage means that
//! each executed id has a persisted `…/<id>/<baseline>/` series directory
//! under the criterion root. Criterion's `has regressed` t-test remains the
//! statistical signal — this module never re-runs statistics.
//!
//! Fail-closed (F-24): a log with no parseable id makes the cross-check
//! blind, so `missing_baseline_coverage` returns
//! [`crate::testing::qa::coverage::BaselineCoverageGap`] — nothing passes
//! unverified. Word splitting mirrors the bash `for id in $ids`
//! (empty/whitespace-only captures are dropped), and an absent/unreadable
//! log maps to empty text, i.e. the same blind gate.
//!
//! Documented divergence: `sort -u` collates with the active locale (leading
//! whitespace is ignorable), while the port sorts by raw bytes — the orders
//! coincide for every realistic Criterion id (no leading whitespace), and
//! the Rust order is deterministic regardless of the host locale.

use std::path::Path;

/// Fail-closed marker (F-24): the log carried no parseable
/// `Benchmarking <id>:` line, so the coverage cross-check is blind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BaselineCoverageGap;

/// Extracts the benchmark ids executed by a Criterion log — the same mental
/// `sed -n 's/^Benchmarking \([^:]*\):.*/\1/p'` of `executed_bench_ids`:
/// every line starting with `Benchmarking ` contributes the id up to the
/// **first** `:` (a line without a colon does not match), ids are
/// deduplicated and lexicographically sorted (`sort -u`).
///
/// The captures are returned raw, exactly like the sed output: the caller
/// applies the bash word-splitting semantics when iterating
/// ([`missing_baseline_coverage`] does).
pub fn executed_bench_ids(log_text: &str) -> Vec<String> {
    let mut ids: Vec<String> = log_text
        .lines()
        .filter_map(|line| line.strip_prefix("Benchmarking "))
        .filter_map(|rest| rest.split_once(':').map(|(id, _)| id.to_string()))
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

/// `true` when `id` survives bash word splitting (`for id in $ids`):
/// empty and whitespace-only captures are dropped.
fn is_word(id: &str) -> bool {
    !id.trim().is_empty()
}

/// Cross-checks baseline coverage of an executed Criterion log (F-24).
///
/// Literal port of `missing_baseline_coverage`
/// (`tests-performance-regression.sh:234-249`):
/// - no parseable id ⇒ [`Err(BaselineCoverageGap)`](BaselineCoverageGap)
///   (the bash `[ -z "$ids" ]` blind gate — an absent/unparseable log fails
///   closed);
/// - otherwise ⇒ `Ok(missing)` with the ids that have **no**
///   `<criterion_root>/<id>/<baseline_name>/` directory, in
///   [`executed_bench_ids`] order (`Ok(empty)` = full coverage).
pub fn missing_baseline_coverage(
    log_text: &str,
    criterion_root: &Path,
    baseline_name: &str,
) -> Result<Vec<String>, BaselineCoverageGap> {
    let ids = executed_bench_ids(log_text);
    let words: Vec<&str> = ids
        .iter()
        .map(String::as_str)
        .filter(|id| is_word(id))
        .collect();
    if words.is_empty() {
        return Err(BaselineCoverageGap);
    }
    Ok(words
        .into_iter()
        .filter(|id| !criterion_root.join(id).join(baseline_name).is_dir())
        .map(str::to_string)
        .collect())
}

#[cfg(test)]
#[path = "coverage_test.rs"]
mod coverage_test;
