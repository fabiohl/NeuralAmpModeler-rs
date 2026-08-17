// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single performance-status classifier — literal port of the bash
//! `classify_regression_outcome` (F-08 / EP-05; the bash copy was removed,
//! `utils/_lib.sh` now delegates nothing here — the dashboard
//! inlines the same 3-way case over the Rust-written receipt).
//!
//! One classifier: the dashboard and the perf-gate consume the same enum.
//! The perf gate records `FAIL` with a typed reason; only `MISSING_BASELINE`
//! and `INCOMPARABLE_ENVIRONMENT` classify as `NOT_VERIFIED`. Everything else
//! — real regressions, benchmark failures, empty receipts, and
//! `SKIP_CAPABILITY` — is fail-closed `FAIL` (never promoted).

/// Single performance verification outcome (F-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressionOutcome {
    /// Receipt status `PASS`.
    Pass,
    /// `FAIL` with reason `MISSING_BASELINE` | `INCOMPARABLE_ENVIRONMENT`:
    /// performance cannot be certified, never counted as `PASS`.
    NotVerified,
    /// Everything else — including empty receipts — fail-closed.
    Fail,
}

impl RegressionOutcome {
    /// The canonical dashboard string of the outcome.
    pub fn as_str(self) -> &'static str {
        match self {
            RegressionOutcome::Pass => "PASS",
            RegressionOutcome::NotVerified => "NOT_VERIFIED",
            RegressionOutcome::Fail => "FAIL",
        }
    }
}

/// Classifies a regression receipt into the single performance status.
///
/// Literal port of the original bash case:
/// `PASS:*` → PASS; `*:MISSING_BASELINE|*:INCOMPARABLE_ENVIRONMENT` →
/// NOT_VERIFIED; anything else → FAIL.
pub fn classify_regression_outcome(status: &str, reason: &str) -> RegressionOutcome {
    match (status, reason) {
        ("PASS", _) => RegressionOutcome::Pass,
        (_, "MISSING_BASELINE" | "INCOMPARABLE_ENVIRONMENT") => RegressionOutcome::NotVerified,
        _ => RegressionOutcome::Fail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 7 acceptance cases of `utils/tests-long.sh:860-866`, verbatim.
    #[test]
    fn tests_long_classify_cases_match_expect_str() {
        assert_eq!(
            classify_regression_outcome("PASS", ""),
            RegressionOutcome::Pass
        );
        assert_eq!(
            classify_regression_outcome("FAIL", "MISSING_BASELINE"),
            RegressionOutcome::NotVerified
        );
        assert_eq!(
            classify_regression_outcome("FAIL", "INCOMPARABLE_ENVIRONMENT"),
            RegressionOutcome::NotVerified
        );
        assert_eq!(
            classify_regression_outcome("FAIL", "REGRESSION_DETECTED"),
            RegressionOutcome::Fail
        );
        assert_eq!(
            classify_regression_outcome("FAIL", "Benchmark run failed"),
            RegressionOutcome::Fail
        );
        assert_eq!(classify_regression_outcome("", ""), RegressionOutcome::Fail);
        assert_eq!(
            classify_regression_outcome("SKIP_CAPABILITY", "whatever"),
            RegressionOutcome::Fail
        );
    }

    #[test]
    fn outcome_strings_are_the_dashboard_verdicts() {
        assert_eq!(RegressionOutcome::Pass.as_str(), "PASS");
        assert_eq!(RegressionOutcome::NotVerified.as_str(), "NOT_VERIFIED");
        assert_eq!(RegressionOutcome::Fail.as_str(), "FAIL");
    }

    #[test]
    fn any_status_with_missing_baseline_reason_is_not_verified() {
        assert_eq!(
            classify_regression_outcome("FAIL", "MISSING_BASELINE"),
            RegressionOutcome::NotVerified
        );
        assert_eq!(
            classify_regression_outcome("NOT_RUN", "MISSING_BASELINE"),
            RegressionOutcome::NotVerified
        );
    }

    #[test]
    fn pass_with_garbage_reason_is_still_pass() {
        assert_eq!(
            classify_regression_outcome("PASS", "MISSING_BASELINE"),
            RegressionOutcome::Pass
        );
    }
}
