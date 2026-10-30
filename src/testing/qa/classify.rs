// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Single performance-status classifier — literal port of the bash
//! `classify_regression_outcome` (the bash copy was removed;
//! `utils/_lib.sh` now delegates nothing here — the dashboard
//! inlines the same 3-way case over the Rust-written receipt).
//!
//! One classifier: the dashboard and the perf-gate consume the same enum.
//! The perf gate records `FAIL` with a typed reason; only `MISSING_BASELINE`
//! and `INCOMPARABLE_ENVIRONMENT` classify as `NOT_VERIFIED`. Everything else
//! — real regressions, benchmark failures, empty receipts, and
//! `SKIP_CAPABILITY` — is fail-closed `FAIL` (never promoted).

/// Single performance verification outcome.
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

/// Fail-closed fresh-run classification — the dashboard's
/// `regression_gate` phase.
///
/// A receipt from a **previous** execution can never validate the current
/// run, and a failed benchmark run can never be green:
/// - `reg_exit == 0` ⇒ the runner certified the run; only a **fresh** `PASS`
///   receipt (`run_id_matches`) proves it. A stale `PASS` from an older run
///   (or an absent one) is `FAIL`.
/// - `reg_exit != 0` ⇒ the phase is `FAIL` immediately — unless the **fresh**
///   receipt from THIS run declares the typed `NOT_VERIFIED` reasons
///   (`MISSING_BASELINE` / `INCOMPARABLE_ENVIRONMENT`), which are deliberate
///   states recorded by the runner, not aborts.
pub fn classify_fresh_regression(
    reg_exit: i32,
    status: &str,
    reason: &str,
    run_id_matches: bool,
) -> RegressionOutcome {
    if reg_exit == 0 {
        if run_id_matches && status == "PASS" {
            RegressionOutcome::Pass
        } else {
            RegressionOutcome::Fail
        }
    } else if run_id_matches && matches!(reason, "MISSING_BASELINE" | "INCOMPARABLE_ENVIRONMENT") {
        RegressionOutcome::NotVerified
    } else {
        RegressionOutcome::Fail
    }
}
