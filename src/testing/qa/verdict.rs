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

use serde::{Deserialize, Serialize};

use super::coverage::{executed_bench_ids, expected_id_matches};

/// Noise band of `benches/regression_gate.rs` (`noise_threshold(0.05)`):
/// a relative mean change whose whole 95% CI lies above `+5%` is a machine
/// regression.
pub const DEFAULT_NOISE_THRESHOLD: f64 = 0.05;

/// Serde shape of `criterion::estimate::ConfidenceInterval`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
pub struct ConfidenceInterval {
    /// Confidence level of the interval (0.95 for Criterion defaults).
    pub confidence_level: f64,
    /// Lower bound of the interval.
    pub lower_bound: f64,
    /// Upper bound of the interval.
    pub upper_bound: f64,
}

/// Serde shape of `criterion::estimate::Estimate`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
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
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VerdictFinding {
    /// An executed benchmark has no readable Criterion `change/estimates.json`
    /// — Criterion produced no machine comparison for it (baseline incomplete,
    /// interrupted run, or a compare error), so nothing is certified.
    MissingEstimates {
        /// Executed Criterion benchmark id (from the log).
        id: String,
    },
    /// An expected benchmark was not executed in the Criterion run.
    MissingExpectedBenchmark {
        /// Expected Criterion benchmark id that was absent from the log.
        id: String,
    },
    /// Machine regression: the whole mean-change CI lies above the positive
    /// noise band (and exceeds absolute noise floor if configured).
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
        /// Absolute mean difference in nanoseconds, if available.
        delta_ns: Option<f64>,
    },
    /// Bilateral baseline instability: the whole mean-change CI lies below
    /// the negative noise band (e.g. improved beyond noise), indicating
    /// baseline bimodal variation or environmental drift.
    UnstableBaseline {
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
        /// Absolute mean difference in nanoseconds, if available.
        delta_ns: Option<f64>,
    },
    /// Relative change exceeded the noise band, but the absolute delta in
    /// nanoseconds was at or below `abs_noise_floor_ns`.
    NoiseIgnored {
        /// Executed Criterion benchmark id (from the log).
        id: String,
        /// Relative mean change point estimate (`new/base − 1`).
        point_estimate: f64,
        /// Lower bound of the mean-change confidence interval.
        lower_bound: f64,
        /// Upper bound of the mean-change confidence interval.
        upper_bound: f64,
        /// Measured absolute delta in nanoseconds.
        delta_ns: f64,
        /// Absolute noise floor threshold in nanoseconds.
        floor_ns: f64,
    },
}

impl VerdictFinding {
    /// Benchmark id associated with this finding.
    pub fn id(&self) -> &str {
        match self {
            Self::MissingEstimates { id }
            | Self::MissingExpectedBenchmark { id }
            | Self::Regression { id, .. }
            | Self::UnstableBaseline { id, .. }
            | Self::NoiseIgnored { id, .. } => id,
        }
    }

    /// True if this finding represents a confirmed machine regression.
    pub fn is_regression(&self) -> bool {
        matches!(self, Self::Regression { .. })
    }

    /// True if this finding represents a blind verdict condition.
    pub fn is_blind(&self) -> bool {
        matches!(
            self,
            Self::MissingEstimates { .. } | Self::MissingExpectedBenchmark { .. }
        )
    }

    /// True if this finding represents bilateral baseline instability.
    pub fn is_unstable_baseline(&self) -> bool {
        matches!(self, Self::UnstableBaseline { .. })
    }

    /// True if this finding represents sub-threshold noise ignored by absolute floor.
    pub fn is_noise_ignored(&self) -> bool {
        matches!(self, Self::NoiseIgnored { .. })
    }
}

/// Options for configuring the machine regression verdict.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictOptions {
    /// Noise band of the verdict (e.g. 0.05 for ±5%).
    pub noise: f64,
    /// Optional expected benchmark ids.
    pub expected_ids: Option<Vec<String>>,
    /// Check bilateral baseline stability (flag improvements exceeding the noise band).
    pub check_stability: bool,
    /// Absolute noise floor in nanoseconds. Microbenchmark relative regressions whose
    /// absolute mean delta is at or below this floor are reported as sub-threshold noise
    /// rather than regressions.
    pub abs_noise_floor_ns: Option<f64>,
}

impl Default for VerdictOptions {
    fn default() -> Self {
        Self {
            noise: DEFAULT_NOISE_THRESHOLD,
            expected_ids: None,
            check_stability: false,
            abs_noise_floor_ns: None,
        }
    }
}

/// Outcome of the machine verdict over the executed benchmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictOutcome {
    /// Number of executed benchmark ids parsed from the log (deduplicated).
    pub executed: usize,
    /// Machine findings, in executed (sorted) order.
    pub findings: Vec<VerdictFinding>,
}

impl VerdictOutcome {
    /// Iterator over confirmed regression findings.
    pub fn regressions(&self) -> impl Iterator<Item = &VerdictFinding> {
        self.findings.iter().filter(|f| f.is_regression())
    }

    /// Iterator over blind verdict findings.
    pub fn blind_findings(&self) -> impl Iterator<Item = &VerdictFinding> {
        self.findings.iter().filter(|f| f.is_blind())
    }

    /// Iterator over unstable baseline findings.
    pub fn unstable_baselines(&self) -> impl Iterator<Item = &VerdictFinding> {
        self.findings.iter().filter(|f| f.is_unstable_baseline())
    }

    /// Iterator over sub-threshold noise findings.
    pub fn sub_threshold_noise(&self) -> impl Iterator<Item = &VerdictFinding> {
        self.findings.iter().filter(|f| f.is_noise_ignored())
    }
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

/// Path of the Criterion current-run artifact of one benchmark id:
/// `<criterion_root>/<id>/new/estimates.json`.
pub fn new_estimates_path(criterion_root: &Path, id: &str) -> std::path::PathBuf {
    criterion_root.join(id).join("new").join("estimates.json")
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

/// Reads the Criterion `new/estimates.json` of one benchmark id if present.
fn read_new_estimates(criterion_root: &Path, id: &str) -> Option<ChangeEstimates> {
    let text = std::fs::read_to_string(new_estimates_path(criterion_root, id)).ok()?;
    serde_json::from_str(&text).ok()
}

/// Computes absolute delta in nanoseconds (`mean_current - mean_baseline`),
/// if `new/estimates.json` is available.
///
/// Since relative change `r = (T_new - T_base) / T_base`, we have:
/// `1 + r = T_new / T_base` => `T_base = T_new / (1 + r)`
/// `delta = T_new - T_base = T_new * (r / (1 + r))`.
fn compute_delta_ns(criterion_root: &Path, id: &str, relative_mean: f64) -> Option<f64> {
    let new_est = read_new_estimates(criterion_root, id)?;
    let t_new = new_est.mean.point_estimate;
    let denom = 1.0 + relative_mean;
    if denom > 0.0 {
        Some(t_new * (relative_mean / denom))
    } else {
        None
    }
}

/// Machine regression verdict over a Criterion log, with custom verdict options.
pub fn regression_verdict_with_options(
    log_text: &str,
    criterion_root: &Path,
    options: &VerdictOptions,
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

    if let Some(expected) = &options.expected_ids {
        for exp_id in expected {
            if !expected_id_matches(exp_id, &ids) {
                findings.push(VerdictFinding::MissingExpectedBenchmark { id: exp_id.clone() });
            }
        }
    }

    for id in ids {
        match read_change_estimates(criterion_root, &id) {
            None => findings.push(VerdictFinding::MissingEstimates { id }),
            Some(change) => {
                let ci = change.mean.confidence_interval;
                let rel_point = change.mean.point_estimate;
                let delta_ns = compute_delta_ns(criterion_root, &id, rel_point);
                match classify_mean_change(&change.mean, options.noise) {
                    ChangeClass::Regressed => {
                        let noise_floor_matched = options
                            .abs_noise_floor_ns
                            .zip(delta_ns)
                            .filter(|&(floor_ns, delta)| delta <= floor_ns);
                        if let Some((floor_ns, delta)) = noise_floor_matched {
                            findings.push(VerdictFinding::NoiseIgnored {
                                id,
                                point_estimate: rel_point,
                                lower_bound: ci.lower_bound,
                                upper_bound: ci.upper_bound,
                                delta_ns: delta,
                                floor_ns,
                            });
                            continue;
                        }
                        findings.push(VerdictFinding::Regression {
                            id,
                            point_estimate: rel_point,
                            lower_bound: ci.lower_bound,
                            upper_bound: ci.upper_bound,
                            noise: options.noise,
                            delta_ns,
                        });
                    }
                    ChangeClass::Improved => {
                        if options.check_stability {
                            findings.push(VerdictFinding::UnstableBaseline {
                                id,
                                point_estimate: rel_point,
                                lower_bound: ci.lower_bound,
                                upper_bound: ci.upper_bound,
                                noise: options.noise,
                                delta_ns,
                            });
                        }
                    }
                    ChangeClass::WithinNoise => {}
                }
            }
        }
    }
    Ok(VerdictOutcome { executed, findings })
}

/// Machine regression verdict over a Criterion log, with optional expected benchmark ids
/// cross-check.
pub fn regression_verdict_with_expected(
    log_text: &str,
    criterion_root: &Path,
    noise: f64,
    expected_ids: Option<&[String]>,
) -> Result<VerdictOutcome, VerdictBlind> {
    let options = VerdictOptions {
        noise,
        expected_ids: expected_ids.map(|s| s.to_vec()),
        check_stability: false,
        abs_noise_floor_ns: None,
    };
    regression_verdict_with_options(log_text, criterion_root, &options)
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
    regression_verdict_with_expected(log_text, criterion_root, noise, None)
}
