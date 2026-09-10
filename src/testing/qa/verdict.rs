// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Machine regression verdict over Criterion's persisted comparison JSON
//! (B2 / R-3) — the replacement for the bash
//! `grep -qiE 'has regressed' "$LOG_FILE"` text-marker detector of
//! `tests-performance-regression.sh:197`.
//!
//! When `cargo bench -- --baseline <name>` compares against a restored
//! series, Criterion persists the bootstrapped relative change of every
//! benchmark in `target/criterion/<id>/change/estimates.json` (serde shape
//! of `criterion::estimate::ChangeEstimates`: relative `mean`/`median`
//! estimates with 95% confidence intervals). This module reads that machine
//! artifact — never the human console wording — and classifies a benchmark as
//! regressed when the **whole** mean-change confidence interval lies above the
//! positive noise band (the second half of Criterion's own "regressed"
//! decision in `report.rs::compare_to_threshold`; Criterion additionally
//! requires `p < 0.05`, but the p-value is not persisted).
//!
//! Fail-closed (the R-3 invariant): a log with no parseable benchmark id, or
//! an executed benchmark without a readable `change/estimates.json`, makes the
//! verdict blind — nothing passes unverified. Truncated/empty logs and a
//! redacted Criterion wording can therefore never turn green: the machine
//! artifact is missing in every such case.
//!
//! Documented divergence: the persisted JSON carries the change confidence
//! interval but not the t-test p-value, so the classifier is the CI side of
//! Criterion's rule only. Any comparison Criterion would print as "has
//! regressed" necessarily has `lower_bound > noise` (Criterion requires the
//! whole CI beyond the noise band), so the machine gate never misses a
//! Criterion regression; it can only (theoretically) flag a change whose
//! p-value happened to stay above 0.05 while the whole CI exceeded the band —
//! for the bootstrap CIs and t-distribution Criterion derives from the same
//! samples, such a disagreement does not occur in practice. The dashboard's
//! single status classifier ([`crate::testing::qa::classify`]) still decides
//! `PASS` / `NOT_VERIFIED` / `FAIL` from the receipt the shell writes.

use std::path::Path;

use serde::Deserialize;

use super::coverage::executed_bench_ids;

/// Noise band of `benches/regression_gate.rs` (`noise_threshold(0.05)`):
/// a relative mean change whose whole 95% CI lies above `+5%` is a machine
/// regression.
pub const DEFAULT_NOISE_THRESHOLD: f64 = 0.05;

/// Serde shape of `criterion::estimate::ConfidenceInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct ConfidenceInterval {
    /// Confidence level of the interval (0.95 for Criterion defaults).
    pub confidence_level: f64,
    /// Lower bound of the interval.
    pub lower_bound: f64,
    /// Upper bound of the interval.
    pub upper_bound: f64,
}

/// Serde shape of `criterion::estimate::Estimate`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Estimate {
    /// Confidence interval of the estimate.
    pub confidence_interval: ConfidenceInterval,
    /// Point estimate.
    pub point_estimate: f64,
    /// Standard error of the estimate.
    pub standard_error: f64,
}

/// Serde shape of `criterion::estimate::ChangeEstimates` — the content of
/// `…/<id>/change/estimates.json`, the relative change of the current run
/// against the compared baseline.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ChangeEstimates {
    /// Relative change of the mean (point estimate `new/base − 1`).
    pub mean: Estimate,
    /// Relative change of the median (point estimate `new/base − 1`).
    pub median: Estimate,
}

/// Fail-closed marker (R-3 / B2): the Criterion log carried no parseable
/// `Benchmarking <id>:` line, so the machine regression verdict is blind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerdictBlind;

/// One machine finding of the regression verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum VerdictFinding {
    /// An executed benchmark has no readable Criterion `change/estimates.json`
    /// — Criterion produced no machine comparison for it (baseline incomplete,
    /// interrupted run, or a compare error), so nothing is certified.
    MissingEstimates {
        /// Executed Criterion benchmark id (from the log).
        id: String,
    },
    /// Machine regression: the whole mean-change CI lies above the positive
    /// noise band.
    Regression {
        /// Executed Criterion benchmark id (from the log).
        id: String,
        /// Relative mean change point estimate (`new/base − 1`).
        point_estimate: f64,
        /// Lower bound of the mean-change confidence interval.
        lower_bound: f64,
        /// Upper bound of the mean-change confidence interval.
        upper_bound: f64,
        /// Noise band used by the verdict.
        noise: f64,
    },
}

/// Outcome of the machine verdict over the executed benchmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictOutcome {
    /// Number of executed benchmark ids parsed from the log (deduplicated).
    pub executed: usize,
    /// Machine findings, in executed (sorted) order.
    pub findings: Vec<VerdictFinding>,
}

/// Classifies one relative mean-change estimate against the noise band —
/// a literal port of Criterion's `compare_to_threshold` bounds logic
/// (`report.rs`): the whole CI beyond `+noise` is a regression, the whole CI
/// below `−noise` an improvement, anything else is within noise.
pub fn classify_mean_change(mean: &Estimate, noise: f64) -> ChangeClass {
    let ci = &mean.confidence_interval;
    if ci.lower_bound > noise {
        ChangeClass::Regressed
    } else if ci.upper_bound < -noise {
        ChangeClass::Improved
    } else {
        ChangeClass::WithinNoise
    }
}

/// Mean-change class of a single benchmark (see [`classify_mean_change`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeClass {
    /// Whole mean-change CI above `+noise` — machine regression.
    Regressed,
    /// Whole mean-change CI below `−noise` — machine improvement.
    Improved,
    /// CI crosses or stays inside the noise band — not regressed.
    WithinNoise,
}

/// Path of the Criterion comparison artifact of one benchmark id:
/// `<criterion_root>/<id>/change/estimates.json`.
pub fn change_estimates_path(criterion_root: &Path, id: &str) -> std::path::PathBuf {
    criterion_root
        .join(id)
        .join("change")
        .join("estimates.json")
}

/// Reads the Criterion `change/estimates.json` of one benchmark id.
///
/// `None` when the file is absent, unreadable, or not valid
/// `ChangeEstimates` JSON — every such case is the fail-closed blind signal
/// (an artifact Criterion produced is expected after a compared run).
fn read_change_estimates(criterion_root: &Path, id: &str) -> Option<ChangeEstimates> {
    let text = std::fs::read_to_string(change_estimates_path(criterion_root, id)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Machine regression verdict over a Criterion log and its persisted
/// comparison artifacts (B2 / R-3).
///
/// - no executed id survives bash word splitting ⇒
///   [`Err(VerdictBlind)`](VerdictBlind) — same blind gate as coverage;
/// - otherwise ⇒ `Ok(outcome)` with a finding per executed id whose
///   `change/estimates.json` is missing (blind for that id) or whose
///   mean-change CI is entirely above the noise band (regression).
pub fn regression_verdict(
    log_text: &str,
    criterion_root: &Path,
    noise: f64,
) -> Result<VerdictOutcome, VerdictBlind> {
    let ids: Vec<String> = executed_bench_ids(log_text)
        .into_iter()
        .filter(|id| !id.trim().is_empty())
        .collect();
    if ids.is_empty() {
        return Err(VerdictBlind);
    }
    let executed = ids.len();
    let mut findings = Vec::new();
    for id in ids {
        match read_change_estimates(criterion_root, &id) {
            None => findings.push(VerdictFinding::MissingEstimates { id }),
            Some(change) => {
                if classify_mean_change(&change.mean, noise) == ChangeClass::Regressed {
                    let ci = change.mean.confidence_interval;
                    findings.push(VerdictFinding::Regression {
                        id,
                        point_estimate: change.mean.point_estimate,
                        lower_bound: ci.lower_bound,
                        upper_bound: ci.upper_bound,
                        noise,
                    });
                }
            }
        }
    }
    Ok(VerdictOutcome { executed, findings })
}

#[cfg(test)]
#[path = "verdict_test.rs"]
mod verdict_test;
