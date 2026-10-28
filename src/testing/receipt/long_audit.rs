// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Ordered container and evaluation engine for the long-suite audit receipt.

use std::fs;
use std::path::Path;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::long_phase::{
    CANONICAL_ROSTER, LongPhaseReceipt, LongPhaseStatus, PERFORMANCE_PHASE_IDS,
    PREFLIGHT_PHASE_IDS, is_preflight_id,
};
use crate::common::diagnostics::format::timestamp as iso_timestamp;

/// Current UTC time formatted as ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// Public wrapper around the crate-internal diagnostics formatter so CLI
/// binaries (separate crates) can stamp receipt lines.
pub fn now_iso8601() -> String {
    iso_timestamp()
}

/// Typed error for long-audit receipt I/O and JSONL validation.
#[derive(Debug, thiserror::Error)]
pub enum LongReceiptError {
    /// A line of the JSONL stream is not valid JSON (line numbers are 1-based).
    #[error("invalid JSONL line {line}: {source}")]
    InvalidJsonLine {
        /// 1-based line number of the offending entry.
        line: usize,
        /// Underlying JSON parse error.
        #[source]
        source: serde_json::Error,
    },
    /// Filesystem error while reading or writing the receipt.
    #[error("receipt I/O error: {source}")]
    Io {
        /// Underlying filesystem error.
        #[source]
        source: std::io::Error,
    },
    /// A `preflight-*` line uses an identifier outside [`PREFLIGHT_PHASE_IDS`]
    /// (typoed preflight ids would silently drop the abort trace).
    #[error("unknown preflight identifier '{id}' (canonical: {canonical})")]
    UnknownPreflightId {
        /// The offending `phase_id`.
        id: String,
        /// Comma-joined canonical preflight identifiers.
        canonical: String,
    },
}

impl From<std::io::Error> for LongReceiptError {
    fn from(source: std::io::Error) -> Self {
        LongReceiptError::Io { source }
    }
}

/// Ordered container for the long-suite audit receipt — one entry per emitted
/// JSONL line, in emission order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LongAuditReceipt {
    /// Phase entries in emission order (terminated by the `overall` line).
    pub phases: Vec<LongPhaseReceipt>,
}

/// `true` when a `gaps` entry belongs to the canonical gap family `id`
/// (exact match or `id:<detail>` suffix — gap identifiers may carry typed detail suffixes).
fn gap_has_id(gap: &str, id: &str) -> bool {
    gap == id
        || gap
            .strip_prefix(id)
            .is_some_and(|rest| rest.starts_with(':'))
}

/// `true` when a timed phase (`phase<N>`) reports `PASSED` while executing zero
/// tests/benchmarks — a mandatory subphase gate violation: a `#[cfg]`
/// filter mismatch must never be promoted to a clean `PASSED`.
fn is_zero_test_pass(p: &LongPhaseReceipt) -> bool {
    p.phase_id.starts_with("phase") && p.status == LongPhaseStatus::Passed && p.tests_executed == 0
}

/// Returns true if a phase is selected by the optional `--only` list.
/// If `only` is None or empty, all phases are selected.
/// Comparison is case-insensitive and normalizes underscores to hyphens.
fn is_phase_selected_by_only(phase_id: &str, name: &str, only: Option<&[String]>) -> bool {
    match only {
        None => true,
        Some(list) => {
            if list.is_empty() {
                return true;
            }
            list.iter().any(|item| {
                let item_norm = item.to_lowercase().replace('_', "-");
                let pid_norm = phase_id.to_lowercase().replace('_', "-");
                let pname_norm = name.to_lowercase().replace('_', "-");
                item_norm == pid_norm || item_norm == pname_norm
            })
        }
    }
}

impl LongAuditReceipt {
    /// Parses a JSONL stream; every non-empty line must be valid JSON with the
    /// `LongPhaseReceipt` schema (fail-closed validation). Preflight lines
    /// (`preflight-*`) must additionally use a canonical identifier from
    /// [`PREFLIGHT_PHASE_IDS`].
    pub fn parse_jsonl(input: &str) -> Result<Self, LongReceiptError> {
        let mut phases = Vec::new();
        for (idx, line) in input.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let entry = LongPhaseReceipt::parse_jsonl_line(line).map_err(|source| {
                LongReceiptError::InvalidJsonLine {
                    line: idx + 1,
                    source,
                }
            })?;
            if is_preflight_id(&entry.phase_id)
                && !PREFLIGHT_PHASE_IDS.contains(&entry.phase_id.as_str())
            {
                return Err(LongReceiptError::UnknownPreflightId {
                    id: entry.phase_id,
                    canonical: PREFLIGHT_PHASE_IDS.join(", "),
                });
            }
            phases.push(entry);
        }
        Ok(LongAuditReceipt { phases })
    }

    /// Parses the receipt file at `path` (fail-closed on missing/invalid lines).
    pub fn parse_jsonl_file(path: &Path) -> Result<Self, LongReceiptError> {
        let content = fs::read_to_string(path)?;
        Self::parse_jsonl(&content)
    }

    /// Renders the whole receipt as JSONL (one line per entry, trailing newline).
    pub fn render_jsonl(&self) -> String {
        let mut out = String::new();
        for phase in &self.phases {
            out.push_str(&phase.render_jsonl_line());
            out.push('\n');
        }
        out
    }

    /// Phase entries excluding the suite-level `overall` line (aggregations
    /// must never double-count the summary entry).
    pub fn phase_entries(&self) -> impl Iterator<Item = &LongPhaseReceipt> {
        self.phases.iter().filter(|p| p.phase_id != "overall")
    }

    /// Preflight entries (`preflight-*`), in emission order — the steps that
    /// run ahead of Phase 1 and abort the suite on failure (preflight isolation gate).
    pub fn preflight_entries(&self) -> impl Iterator<Item = &LongPhaseReceipt> {
        self.phases.iter().filter(|p| is_preflight_id(&p.phase_id))
    }

    /// Total executed tests/benchmarks across all phases (excluding `overall`).
    pub fn tests_executed_total(&self) -> u64 {
        self.phase_entries().map(|p| p.tests_executed).sum()
    }

    /// Total wall-clock duration across all phases, in milliseconds
    /// (excluding `overall`).
    pub fn duration_ms_total(&self) -> u64 {
        self.phase_entries().map(|p| p.duration_ms).sum()
    }

    /// `true` when the receipt was produced by simulated dry-run:
    /// it carries at least one phase entry and every entry is `SIMULATED`
    /// (no test was executed). Used to suppress every green (`OK`/`PASS`)
    /// verdict line — a simulated receipt must never *look* like a passing
    /// audit (F-07).
    pub fn is_simulated_only(&self) -> bool {
        let mut any = false;
        for p in self.phase_entries() {
            any = true;
            if p.status != LongPhaseStatus::Simulated {
                return false;
            }
        }
        any
    }

    /// Derived suite-level `overall` receipt line.
    ///
    /// Verdict semantics mirror the runner's final summary:
    /// - any `FAILED` phase or preflight ⇒ `FAILED`;
    /// - otherwise an all-`SIMULATED` receipt ⇒ `SIMULATED` (the
    ///   simulated pre-registration where no test ran);
    /// - otherwise any **declared gap** ⇒ `COMPLETED_WITH_GAPS`. A declared
    ///   gap is a gap status (SKIPPED / INCONCLUSIVE / SKIP_CAPABILITY /
    ///   NOT_RUN), a `PASSED` phase whose `gaps` list is non-empty, or a
    ///   `PASSED` timed phase that executed zero tests (zero-test-pass
    ///   invariant — typed log markers
    ///   (`detect_gap_markers`) are the only carrier of a measurement bypass, and
    ///   "exit-0 with internal bypass" must never be promoted to a clean `PASSED`
    ///   verdict);
    /// - otherwise ⇒ `PASSED`.
    ///
    /// Preflight entries (`preflight-*`) participate in the verdict
    /// like any other phase: an aborted preflight leaves its `FAILED` line and
    /// the derived `overall FAILED` — a trace that survives the abort because
    /// `utils/tests-long.sh` emits it before exiting. `gaps` lists every
    /// declared-gap entry as `phase_id:STATUS`.
    /// Derived suite-level `overall` receipt line, optionally evaluated against
    /// an `--only` phase selection filter.
    pub fn summary_receipt_with_only(&self, only: Option<&[String]>) -> LongPhaseReceipt {
        let has_declared_gap = |p: &LongPhaseReceipt| {
            let selected = is_phase_selected_by_only(&p.phase_id, &p.name, only);
            if selected {
                p.status.is_gap()
                    || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty())
                    || is_zero_test_pass(p)
            } else {
                // When outside --only, SKIPPED is the expected status, not a gap.
                if p.status == LongPhaseStatus::Skipped {
                    false
                } else {
                    p.status.is_gap()
                        || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty())
                        || is_zero_test_pass(p)
                }
            }
        };

        let status = if self
            .phase_entries()
            .any(|p| p.status == LongPhaseStatus::Failed)
        {
            LongPhaseStatus::Failed
        } else if self.is_simulated_only() {
            LongPhaseStatus::Simulated
        } else if self.phase_entries().any(has_declared_gap) {
            LongPhaseStatus::CompletedWithGaps
        } else {
            LongPhaseStatus::Passed
        };

        let mut gaps = Vec::new();
        for p in self.phase_entries().filter(|p| has_declared_gap(p)) {
            if is_zero_test_pass(p) {
                let entry = format!("{}:ZERO_TESTS", p.phase_id);
                if !gaps.contains(&entry) {
                    gaps.push(entry);
                }
            }
            if p.gaps.is_empty() {
                if !is_zero_test_pass(p) {
                    let entry = format!("{}:{}", p.phase_id, p.status.as_str());
                    if !gaps.contains(&entry) {
                        gaps.push(entry);
                    }
                }
            } else {
                for g in &p.gaps {
                    let entry = if g.starts_with(&format!("{}:", p.phase_id)) {
                        g.clone()
                    } else {
                        format!("{}:{}", p.phase_id, g)
                    };
                    if !gaps.contains(&entry) {
                        gaps.push(entry);
                    }
                }
            }
        }

        LongPhaseReceipt {
            phase_id: "overall".to_string(),
            name: "Long Audit Suite".to_string(),
            status,
            duration_ms: self.duration_ms_total(),
            tests_executed: self.tests_executed_total(),
            gaps,
            timestamp: now_iso8601(),
            ..Default::default()
        }
    }

    /// Derived suite-level `overall` receipt line without filtering.
    pub fn summary_receipt(&self) -> LongPhaseReceipt {
        self.summary_receipt_with_only(None)
    }

    /// Appends the suite-level `overall` line, optionally evaluating against `--only`.
    pub fn push_summary_with_only(&mut self, only: Option<&[String]>) {
        self.phases.retain(|p| p.phase_id != "overall");
        let summary = self.summary_receipt_with_only(only);
        self.phases.push(summary);
    }

    /// Appends the suite-level `overall` line, replacing any previous one.
    pub fn push_summary(&mut self) {
        self.push_summary_with_only(None);
    }

    /// Timed phases (`phase<N>`) that report `PASSED` with zero executed tests
    /// (zero-test-pass invariant — mandatory subphase gate violations).
    pub fn zero_test_passes(&self) -> impl Iterator<Item = &LongPhaseReceipt> {
        self.phase_entries().filter(|p| is_zero_test_pass(p))
    }

    /// `true` when any phase declares a gap: a gap status, a `PASSED` phase
    /// with non-empty log markers, or a zero-test `PASSED` timed phase.
    pub fn has_declared_gaps(&self) -> bool {
        self.phase_entries().any(|p| {
            p.status.is_gap()
                || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty())
                || is_zero_test_pass(p)
        })
    }

    /// Fail-closed **semantic** verification against a caller-supplied canonical
    /// roster, independent of the gap/failure policy.
    ///
    /// Rejects a receipt that cannot be trusted structurally or that lies about
    /// its own verdict:
    /// 1. no entries;
    /// 2. last entry is not the terminal `overall` record;
    /// 3. an unknown or duplicated phase id (a duplicated `overall` included);
    /// 4. a missing mandatory roster phase;
    /// 5. a timed phase (`phase<N>`) that passed with zero executed tests;
    /// 6. a declared terminal `OVERALL` verdict that disagrees with the verdict
    ///    *derived* from the phase/gap evidence in the same receipt (this is the
    ///    class of tampering where a forged `overall PASSED` line is prepended
    ///    or a green verdict is pasted over gap evidence).
    ///
    /// The roster is a parameter so downstream producers (a plugin host, a
    /// standalone host) can reuse the exact schema and semantics with their own
    /// phase matrix without the engine hardcoding a consumer.
    /// Fail-closed **semantic** verification against a caller-supplied canonical
    /// roster and optional `--only` phase selection filter.
    pub fn semantic_verdict_with_only(
        &self,
        roster: &[&str],
        only: Option<&[String]>,
    ) -> Result<(), String> {
        if self.phases.is_empty() {
            return Err("audit strictly invalid: empty receipt".to_string());
        }

        // 1. Terminal 'overall' entry check
        let last = self.phases.last().unwrap();
        if last.phase_id != "overall" {
            return Err(format!(
                "audit strictly invalid: terminal phase must be 'overall', found '{}'",
                last.phase_id
            ));
        }

        // Validate only items if provided
        if let Some(list) = only {
            for item in list {
                let item_norm = item.to_lowercase().replace('_', "-");
                let found = roster.iter().any(|r| {
                    let r_norm = r.to_lowercase().replace('_', "-");
                    r_norm == item_norm
                }) || self.phases.iter().any(|p| {
                    let p_norm = p.name.to_lowercase().replace('_', "-");
                    p_norm == item_norm
                });
                if !found {
                    return Err(format!(
                        "audit strictly invalid: unknown phase '{item}' in --only (not in canonical roster)"
                    ));
                }
            }
        }

        // 2. Canonical membership + duplicate detection
        let mut seen = std::collections::BTreeSet::new();
        for p in &self.phases {
            if !roster.contains(&p.phase_id.as_str()) {
                return Err(format!(
                    "audit strictly invalid: unknown phase '{}' (not in canonical roster)",
                    p.phase_id
                ));
            }
            if !seen.insert(p.phase_id.as_str()) {
                return Err(format!(
                    "audit strictly invalid: duplicate phase '{}'",
                    p.phase_id
                ));
            }
        }

        // 3. Every mandatory roster phase present
        for expected in roster {
            if !seen.contains(expected) {
                return Err(format!(
                    "audit strictly invalid: missing required canonical phase '{expected}'"
                ));
            }
        }

        // 4. Subphase checks:
        // For phases selected by --only:
        //   - If claiming PASSED, must have executed > 0 tests (timed phases).
        // For phases outside --only:
        //   - MUST have status SKIPPED!
        for p in self.phase_entries() {
            let selected = is_phase_selected_by_only(&p.phase_id, &p.name, only);
            if selected {
                if is_zero_test_pass(p) {
                    return Err(format!(
                        "audit strictly invalid: timed phase '{}' PASSED with 0 executed tests",
                        p.phase_id
                    ));
                }
            } else if p.status != LongPhaseStatus::Skipped {
                return Err(format!(
                    "audit strictly invalid: phase '{}' outside --only was expected to have status SKIPPED, found {}",
                    p.phase_id,
                    p.status.as_str()
                ));
            }
        }

        // 5. Declared terminal verdict must equal the verdict derived from the
        //    phase/gap evidence (with --only applied) — a receipt must never certify a
        //    verdict it does not back (forged/injected OVERALL line).
        let derived = self.summary_receipt_with_only(only).status;
        if last.status != derived {
            return Err(format!(
                "audit strictly invalid: declared OVERALL={} disagrees with derived verdict {}",
                last.status.as_str(),
                derived.as_str()
            ));
        }

        Ok(())
    }

    /// Fail-closed semantic verification against a caller-supplied canonical roster.
    pub fn semantic_verdict_with_roster(&self, roster: &[&str]) -> Result<(), String> {
        self.semantic_verdict_with_only(roster, None)
    }

    /// Fail-closed semantic verification against the engine's canonical
    /// 14-phase roster ([`CANONICAL_ROSTER`]).
    pub fn semantic_verdict(&self) -> Result<(), String> {
        self.semantic_verdict_with_roster(&CANONICAL_ROSTER)
    }

    /// Roster-parameterized strict campaign verdict with optional `--only` filter.
    pub fn campaign_strict_verdict_with_only(
        &self,
        roster: &[&str],
        only: Option<&[String]>,
    ) -> Result<(), String> {
        // Structural + declared-vs-derived consistency first (fail-closed).
        self.semantic_verdict_with_only(roster, only)?;

        let last = self.phases.last().unwrap();

        // Terminal 'overall' phase status check
        if last.status != LongPhaseStatus::Passed {
            return Err(format!(
                "audit FAILED: terminal 'overall' status is {}",
                last.status
            ));
        }

        // Evaluate summary verdict & gaps
        let summary = self.summary_receipt_with_only(only);
        match summary.status {
            LongPhaseStatus::Passed => Ok(()),
            LongPhaseStatus::Failed => Err("audit FAILED: one or more phases failed".to_string()),
            LongPhaseStatus::Simulated => Err(
                "audit SIMULATED: no test was executed — strict-pre-release rejects simulated receipts"
                    .to_string(),
            ),
            _ => Err(format!(
                "audit COMPLETED_WITH_GAPS: {} declared gap(s) — strict-pre-release rejects gaps",
                summary.gaps.len()
            )),
        }
    }

    /// Roster-parameterized strict campaign verdict. Consumers (a plugin host, a
    /// standalone host) reuse the exact schema and semantics with their own
    /// phase matrix: semantic structural verification against `roster`, then a
    /// clean `OVERALL: PASSED` requirement (no gaps, no failures, no simulation).
    pub fn campaign_strict_verdict_with_roster(&self, roster: &[&str]) -> Result<(), String> {
        self.campaign_strict_verdict_with_only(roster, None)
    }

    /// Fail-closed strict campaign verdict: enforces the canonical 14-phase roster,
    /// presence of terminal 'overall' record, no duplicate or unknown phases,
    /// tests_executed > 0 for timed phases (`phase1`..`phase7`), and zero gaps/failures.
    pub fn campaign_strict_verdict(&self) -> Result<(), String> {
        self.campaign_strict_verdict_with_roster(&CANONICAL_ROSTER)
    }

    /// Fail-closed strict-pre-release verdict: enforces canonical campaign roster
    /// and clean PASSED status.
    pub fn strict_verdict(&self) -> Result<(), String> {
        self.campaign_strict_verdict()
    }

    /// First phase entry with the given `phase_id` (`overall` excluded).
    pub fn phase_by_id(&self, phase_id: &str) -> Option<&LongPhaseReceipt> {
        self.phase_entries().find(|p| p.phase_id == phase_id)
    }

    /// Human `FIDELITY` verdict: `OK` unless a fidelity-class phase failed.
    ///
    /// Fidelity-class = every phase outside [`PERFORMANCE_PHASE_IDS`]
    /// (preflights included). Declared gaps never downgrade FIDELITY to
    /// `FAIL` — that is what the `OVERALL: COMPLETED_WITH_GAPS` verdict is
    /// for (the pre-S5 `ANY_FIDELITY_FAILED` semantics).
    ///
    /// A fidelity-class phase that was never executed (`SIMULATED`)
    /// reports `NOT_RUN` instead of the vacuous
    /// `OK`: a green verdict requires tests that actually ran (F-07). An
    /// all-`SIMULATED` receipt reports `NOT_RUN` unconditionally, even when
    /// it carries no fidelity-class entry.
    pub fn fidelity_verdict(&self) -> &'static str {
        if self.is_simulated_only() {
            return "NOT_RUN";
        }
        let simulated_fidelity = self.phase_entries().any(|p| {
            p.status == LongPhaseStatus::Simulated
                && !PERFORMANCE_PHASE_IDS.contains(&p.phase_id.as_str())
        });
        if simulated_fidelity {
            return "NOT_RUN";
        }
        let failed_fidelity = self.phase_entries().any(|p| {
            p.status == LongPhaseStatus::Failed
                && !PERFORMANCE_PHASE_IDS.contains(&p.phase_id.as_str())
        });
        if failed_fidelity { "FAIL" } else { "OK" }
    }

    /// Human `RT_DEADLINE` verdict from the `phase5` receipt line.
    ///
    /// Mirrors the pre-S5 bash mapping (FAILED → FAIL, SKIPPED → PASS,
    /// anything else → PASS) plus the typed gap carrier: a `PASSED` phase
    /// whose log carried the `inconclusive_environment` marker is reported
    /// `INCONCLUSIVE`. A `SIMULATED` phase (never executed) reports
    /// `NOT_RUN`, never a vacuous `PASS`. The bash override that used to
    /// patch `PHASE_STATUS` is gone (S5) — the marker is authoritative, and
    /// "exit-0 with internal measurement bypass" must not be promoted to
    /// PASS. If phase5 was never executed (None), returns `NOT_RUN`.
    pub fn rt_deadline_verdict(&self) -> &'static str {
        if self.is_simulated_only() {
            return "NOT_RUN";
        }
        match self.phase_by_id("phase5") {
            Some(p) => match p.status {
                LongPhaseStatus::Failed => "FAIL",
                LongPhaseStatus::Inconclusive => "INCONCLUSIVE",
                LongPhaseStatus::Simulated => "NOT_RUN",
                LongPhaseStatus::Passed
                    if p.gaps
                        .iter()
                        .any(|g| gap_has_id(g, "inconclusive_environment")) =>
                {
                    "INCONCLUSIVE"
                }
                _ => "PASS",
            },
            None => "NOT_RUN",
        }
    }

    /// Human `RT_JITTER` verdict from the `phase6` receipt line.
    ///
    /// Mirrors the pre-S5 bash mapping (PASSED → PASS, INCONCLUSIVE →
    /// INCONCLUSIVE, SKIP_CAPABILITY → SKIP_CAPABILITY, FAILED → FAIL,
    /// SKIPPED → INCONCLUSIVE), with the typed log markers as the carrier
    /// for bypasses now that the bash `PHASE_STATUS` overrides are gone. A
    /// `SIMULATED` phase (never executed) reports `NOT_RUN`, never a vacuous
    /// `PASS`. If phase6 was never executed (None), returns `NOT_RUN`.
    pub fn rt_jitter_verdict(&self) -> &'static str {
        if self.is_simulated_only() {
            return "NOT_RUN";
        }
        match self.phase_by_id("phase6") {
            None => "NOT_RUN",
            Some(p) => match p.status {
                LongPhaseStatus::Failed => "FAIL",
                LongPhaseStatus::Inconclusive | LongPhaseStatus::Skipped => "INCONCLUSIVE",
                LongPhaseStatus::SkipCapability => "SKIP_CAPABILITY",
                LongPhaseStatus::Simulated => "NOT_RUN",
                LongPhaseStatus::Passed
                    if p.gaps.iter().any(|g| gap_has_id(g, "skip_capability")) =>
                {
                    "SKIP_CAPABILITY"
                }
                LongPhaseStatus::Passed if p.gaps.iter().any(|g| gap_has_id(g, "inconclusive")) => {
                    "INCONCLUSIVE"
                }
                _ => "PASS",
            },
        }
    }

    /// Evaluates the performance regression gate verdict by inspecting
    /// `target/logs/regression_phase_receipt.jsonl` (or a custom path).
    ///
    /// Fail-closed semantics:
    /// - Absent file => `NOT_RUN`
    /// - Expired file (> 24 hours without matching run id) => `NOT_RUN`
    /// - Corrupted JSON / empty receipt with zero phases => `FAIL` (fail-closed)
    /// - Any phase or overall status `FAIL` (or exit_code != 0) => `FAIL`
    /// - Mismatched non-empty run_id (when `NAM_RUN_ID` is present in env) => `NOT_RUN`
    /// - All phases `PASS` (and exit_code == 0) => `PASS`
    pub fn perf_regression_verdict_at(
        path: &Path,
        max_age: Option<Duration>,
        expected_run_id: Option<&str>,
    ) -> &'static str {
        if !path.exists() {
            return "NOT_RUN";
        }

        // Check file recency (default 24 hours)
        let recency_ok = match max_age {
            Some(max_dur) => {
                if let Ok(mtime) = fs::metadata(path).and_then(|m| m.modified()) {
                    if let Ok(elapsed) = std::time::SystemTime::now().duration_since(mtime) {
                        elapsed <= max_dur
                    } else if let Ok(drift) = mtime.duration_since(std::time::SystemTime::now()) {
                        drift < Duration::from_secs(300)
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            None => true,
        };

        let content = match fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return "FAIL",
        };

        let mut phase_count = 0usize;
        let mut fail_detected = false;
        let mut matched_run_id = false;

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => return "FAIL",
            };

            // Skip non-phase records (e.g. build_metadata)
            if value.get("phase_id").is_none() {
                continue;
            }

            phase_count += 1;

            if let Some(exp) = expected_run_id.filter(|s| !s.is_empty()) {
                let run_id = value.get("run_id").and_then(|v| v.as_str()).unwrap_or("");
                if !run_id.is_empty() {
                    if run_id == exp {
                        matched_run_id = true;
                    } else {
                        continue;
                    }
                }
            }

            let status = value.get("status").and_then(|v| v.as_str()).unwrap_or("");
            let exit_code = value.get("exit_code").and_then(|v| v.as_u64()).unwrap_or(0);

            if status != "PASS" || exit_code != 0 {
                fail_detected = true;
            }
        }

        if phase_count == 0 {
            return "FAIL";
        }

        if let Some(exp) = expected_run_id {
            if !exp.is_empty() && !matched_run_id {
                return "NOT_RUN";
            }
        } else if !recency_ok {
            return "NOT_RUN";
        }

        if fail_detected { "FAIL" } else { "PASS" }
    }

    /// Evaluates the performance regression verdict from the default receipt file
    /// (`target/logs/regression_phase_receipt.jsonl`).
    pub fn perf_regression_verdict(&self) -> &'static str {
        if self.is_simulated_only() {
            return "NOT_RUN";
        }
        let env_run_id = std::env::var("NAM_RUN_ID").ok();
        Self::perf_regression_verdict_at(
            Path::new("target/logs/regression_phase_receipt.jsonl"),
            Some(Duration::from_secs(24 * 3600)),
            env_run_id.as_deref(),
        )
    }

    /// Human one-line summary printed by `nam_long_receipt summary` (S5),
    /// with an optional custom regression receipt file path and optional `--only` filter.
    pub fn human_summary_lines_with_only(
        &self,
        reg_path: Option<&Path>,
        only: Option<&[String]>,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        for p in self.phase_entries() {
            let selected = is_phase_selected_by_only(&p.phase_id, &p.name, only);
            if p.status == LongPhaseStatus::Failed {
                lines.push(format!("ERROR: {} {} — FAILED", p.phase_id, p.name));
            } else if selected
                && (p.status.is_gap()
                    || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty()))
            {
                let gap_suffix = if p.gaps.is_empty() {
                    String::new()
                } else {
                    format!(" (gaps: {})", p.gaps.join(", "))
                };
                lines.push(format!(
                    "WARNING: {} {} — {}{}",
                    p.phase_id,
                    p.name,
                    p.status.as_str(),
                    gap_suffix
                ));
            }
        }
        let overall = self.summary_receipt_with_only(only);
        let perf_verdict =
            if self.is_simulated_only() || !is_phase_selected_by_only("phase8", "perf", only) {
                "NOT_RUN"
            } else if let Some(p) = reg_path {
                Self::perf_regression_verdict_at(
                    p,
                    Some(Duration::from_secs(24 * 3600)),
                    std::env::var("NAM_RUN_ID").ok().as_deref(),
                )
            } else {
                self.perf_regression_verdict()
            };

        if perf_verdict == "FAIL" {
            lines.push("ERROR: perf_regression — FAILED".to_string());
        }

        let overall_status = if perf_verdict == "FAIL" && overall.status != LongPhaseStatus::Failed
        {
            LongPhaseStatus::Failed
        } else {
            overall.status
        };

        let rt_deadline_verdict = if is_phase_selected_by_only("phase5", "rt-deadline", only) {
            self.rt_deadline_verdict()
        } else {
            "NOT_RUN"
        };

        let rt_jitter_verdict = if is_phase_selected_by_only("phase6", "rt-jitter", only) {
            self.rt_jitter_verdict()
        } else {
            "NOT_RUN"
        };

        lines.push(format!("OVERALL: {}", overall_status.as_str()));
        lines.push(format!("FIDELITY: {}", self.fidelity_verdict()));
        lines.push(format!("RT_DEADLINE: {}", rt_deadline_verdict));
        lines.push(format!("RT_JITTER: {}", rt_jitter_verdict));
        lines.push(format!("PERF_REGRESSION: {}", perf_verdict));
        lines
    }

    /// Human one-line summary printed by `nam_long_receipt summary` (S5),
    /// with an optional custom regression receipt file path for testing.
    pub fn human_summary_lines_with_regression_path(&self, reg_path: Option<&Path>) -> Vec<String> {
        self.human_summary_lines_with_only(reg_path, None)
    }

    /// Human one-line summary printed by `nam_long_receipt summary` (S5).
    ///
    /// Replaces the giant ASCII table + top-N block of `utils/tests-long.sh`:
    /// the forensic data lives in the JSONL, and the human gets only alarms
    /// (WARNING/ERROR) plus the verdict lines. The runner echoes these lines
    /// verbatim and maps `OVERALL:` to its exit code — it never reclassifies
    /// logs.
    ///
    /// A simulated receipt (all phases `SIMULATED`) yields
    /// `OVERALL: SIMULATED` and `NOT_RUN` on every verdict line — no `OK` /
    /// `PASS` is ever printed for a run that executed nothing (F-07).
    pub fn human_summary_lines(&self) -> Vec<String> {
        self.human_summary_lines_with_only(None, None)
    }
}
