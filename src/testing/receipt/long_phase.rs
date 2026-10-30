// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Phase statuses, phase receipt records, and canonical rosters for the long audit suite.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// Status classification of an individual long-suite phase (or overall verdict).
///
/// Preflight phases use `Passed` / `Failed` only (an abort is never
/// SIMULATED).
/// `CompletedWithGaps` is only used by the suite-level `overall` line.
/// `Simulated` is only used by simulated pre-registration
/// (no test executed; strict mode always rejects it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LongPhaseStatus {
    /// Phase completed with zero failures and no declared gaps.
    Passed,
    /// Phase failed (test failure, gate promotion, or non-zero exit).
    Failed,
    /// Phase was skipped by the runner (legacy exit code 77 convention;
    /// current phases convey skips via typed `[STATUS]` log markers that the
    /// receipt detects as gaps — `Skipped` remains parseable for old lines).
    Skipped,
    /// Phase ran but could not certify its assertion (environment telemetry).
    Inconclusive,
    /// Phase bypassed itself due to a missing capability (typed log marker).
    SkipCapability,
    /// Phase was never executed.
    NotRun,
    /// Phase pre-registered without execution (simulation).
    Simulated,
    /// Suite verdict: audit completed, but with declared gaps.
    CompletedWithGaps,
}

impl LongPhaseStatus {
    /// Canonical serialized name (e.g. `SKIP_CAPABILITY`).
    pub fn as_str(self) -> &'static str {
        match self {
            LongPhaseStatus::Passed => "PASSED",
            LongPhaseStatus::Failed => "FAILED",
            LongPhaseStatus::Skipped => "SKIPPED",
            LongPhaseStatus::Inconclusive => "INCONCLUSIVE",
            LongPhaseStatus::SkipCapability => "SKIP_CAPABILITY",
            LongPhaseStatus::NotRun => "NOT_RUN",
            LongPhaseStatus::Simulated => "SIMULATED",
            LongPhaseStatus::CompletedWithGaps => "COMPLETED_WITH_GAPS",
        }
    }

    /// Returns `true` when this status declares a gap for the audit verdict
    /// (the long suite's `HAS_GAPS` semantics: skipped / inconclusive /
    /// capability-skipped / not-run / simulated phases).
    pub fn is_gap(self) -> bool {
        matches!(
            self,
            LongPhaseStatus::Skipped
                | LongPhaseStatus::Inconclusive
                | LongPhaseStatus::SkipCapability
                | LongPhaseStatus::NotRun
                | LongPhaseStatus::Simulated
        )
    }

    /// Canonical gap identifier for gap-declaring statuses (`None` otherwise).
    pub fn gap_id(self) -> Option<&'static str> {
        match self {
            LongPhaseStatus::Skipped => Some("skipped"),
            LongPhaseStatus::Inconclusive => Some("inconclusive"),
            LongPhaseStatus::SkipCapability => Some("skip_capability"),
            LongPhaseStatus::NotRun => Some("not_run"),
            LongPhaseStatus::Simulated => Some("simulated"),
            _ => None,
        }
    }
}

impl FromStr for LongPhaseStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "PASSED" => Ok(LongPhaseStatus::Passed),
            "FAILED" => Ok(LongPhaseStatus::Failed),
            "SKIPPED" => Ok(LongPhaseStatus::Skipped),
            "INCONCLUSIVE" => Ok(LongPhaseStatus::Inconclusive),
            "SKIP_CAPABILITY" => Ok(LongPhaseStatus::SkipCapability),
            "NOT_RUN" => Ok(LongPhaseStatus::NotRun),
            "SIMULATED" => Ok(LongPhaseStatus::Simulated),
            "COMPLETED_WITH_GAPS" => Ok(LongPhaseStatus::CompletedWithGaps),
            other => Err(format!(
                "invalid long-suite status '{other}' (expected one of: \
                 PASSED, FAILED, SKIPPED, INCONCLUSIVE, SKIP_CAPABILITY, NOT_RUN, SIMULATED, COMPLETED_WITH_GAPS)"
            )),
        }
    }
}

impl std::fmt::Display for LongPhaseStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Typed root cause attached to a `FAILED` receipt line.
///
/// The human summary and the strict validator use this instead of re-reading
/// phase logs, so a failure is explained by its cause (generator drift,
/// missing baseline, timeout, …) rather than by a free-form string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LongReason {
    /// Fixture generator provenance drifted from the committed manifest.
    GeneratorDrift,
    /// The performance baseline required by the phase is absent.
    MissingBaseline,
    /// The phase executed tests and at least one failed.
    TestFailure,
    /// The phase exceeded its timeout.
    Timeout,
    /// A preflight step failed before the timed phases.
    PreflightFailure,
}

impl LongReason {
    /// Canonical serialized name (e.g. `GENERATOR_DRIFT`).
    pub fn as_str(self) -> &'static str {
        match self {
            LongReason::GeneratorDrift => "GENERATOR_DRIFT",
            LongReason::MissingBaseline => "MISSING_BASELINE",
            LongReason::TestFailure => "TEST_FAILURE",
            LongReason::Timeout => "TIMEOUT",
            LongReason::PreflightFailure => "PREFLIGHT_FAILURE",
        }
    }
}

impl FromStr for LongReason {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "GENERATOR_DRIFT" => Ok(LongReason::GeneratorDrift),
            "MISSING_BASELINE" => Ok(LongReason::MissingBaseline),
            "TEST_FAILURE" => Ok(LongReason::TestFailure),
            "TIMEOUT" => Ok(LongReason::Timeout),
            "PREFLIGHT_FAILURE" => Ok(LongReason::PreflightFailure),
            other => Err(format!(
                "invalid long-suite reason '{other}' (expected one of: \
                 GENERATOR_DRIFT, MISSING_BASELINE, TEST_FAILURE, TIMEOUT, PREFLIGHT_FAILURE)"
            )),
        }
    }
}

impl std::fmt::Display for LongReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Single JSONL line: structured audit receipt entry for one long-suite phase.
///
/// The optional `*_sha256` fields carry the provenance of the **referenced
/// artifacts** that backed the phase's verdict (e.g. the render oracle, the
/// input fixture and the produced artifact). They are producer-neutral and
/// fully optional: a receipt written before these fields existed still parses
/// (`#[serde(default)]`) and a line that does not declare provenance omits them
/// entirely (`skip_serializing_if`), keeping the JSONL byte-compatible with the
/// historical schema. Producers that can prove provenance (e.g. a C++/f64
/// parity phase) populate them on whichever line owns the evidence — typically
/// the `overall` line — never encoding a specific host or downstream project in
/// the field names.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LongPhaseReceipt {
    /// Stable phase identifier (e.g. `phase1`..`phase7`, `overall`).
    pub phase_id: String,
    /// Human-readable phase name (mirrors `tests-long.sh` run_phase labels).
    pub name: String,
    /// Typed phase status.
    pub status: LongPhaseStatus,
    /// Wall-clock duration of the phase in milliseconds.
    pub duration_ms: u64,
    /// Number of tests/benchmarks actually executed by the phase
    /// (parsed from `test result:` lines of the phase log).
    pub tests_executed: u64,
    /// Declared gap markers for this phase (typed log markers + explicit gaps
    /// + the phase status itself when it is a gap status).
    pub gaps: Vec<String>,
    /// Optional, non-distributable fixture skips. Reported and counted for
    /// traceability, but never part of the blocking gap set under `--strict`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub optional_gaps: Vec<String>,
    /// ISO-8601 emission timestamp (UTC).
    pub timestamp: String,
    /// Optional SHA-256 of the oracle/render artifact referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_sha256: Option<String>,
    /// Optional git commit of the C++ oracle source tree (NAMCore) that backed
    /// the phase verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_core_commit: Option<String>,
    /// Optional C++ compiler identity used to build the oracle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_compiler: Option<String>,
    /// Optional SHA-256 of the input fixture referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture_sha256: Option<String>,
    /// Optional SHA-256 of the produced artifact referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_sha256: Option<String>,
    /// Optional run identifier exported by the runner (`NAM_RUN_ID`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Optional `git rev-parse HEAD` captured when the line was emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    /// Optional tracked-tree dirtiness captured when the line was emitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_dirty: Option<bool>,
    /// Optional typed root cause; mandatory for `FAILED` under strict validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<LongReason>,
}

impl Default for LongPhaseReceipt {
    /// Neutral default used to fill the optional provenance fields at
    /// construction sites that predate them (`..Default::default()`).
    fn default() -> Self {
        LongPhaseReceipt {
            phase_id: String::new(),
            name: String::new(),
            status: LongPhaseStatus::Passed,
            duration_ms: 0,
            tests_executed: 0,
            gaps: Vec::new(),
            optional_gaps: Vec::new(),
            timestamp: String::new(),
            oracle_sha256: None,
            oracle_core_commit: None,
            oracle_compiler: None,
            fixture_sha256: None,
            artifact_sha256: None,
            run_id: None,
            git_sha: None,
            git_dirty: None,
            reason: None,
        }
    }
}

impl LongPhaseReceipt {
    /// Builds a receipt entry with the optional referenced-artifact provenance
    /// hashes attached from three `Option<String>` values, omitting `None`.
    pub fn with_referenced_artifacts(
        mut self,
        oracle_sha256: Option<String>,
        fixture_sha256: Option<String>,
        artifact_sha256: Option<String>,
    ) -> Self {
        self.oracle_sha256 = oracle_sha256;
        self.fixture_sha256 = fixture_sha256;
        self.artifact_sha256 = artifact_sha256;
        self
    }
}

impl LongPhaseReceipt {
    /// Serializes this entry as a single JSONL line (no trailing newline).
    pub fn render_jsonl_line(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parses one JSONL line back into a receipt entry.
    pub fn parse_jsonl_line(line: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(line)
    }
}

/// Canonical preflight step identifiers emitted by `utils/tests-long.sh`
/// ahead of Phase 1 (preflight isolation gate): render binary, fixture/V1/V2 catalog,
/// package exclusion, freshness, SIMD capability probe. An abort in any of them exits the suite before a single timed
/// phase, so each step must still leave a machine-readable `preflight-*` line
/// (plus the `overall` verdict).
pub const PREFLIGHT_PHASE_IDS: [&str; 5] = [
    "preflight-render",
    "preflight-catalog",
    "preflight-package",
    "preflight-freshness",
    "preflight-simd-probe",
];

/// Canonical 14-phase roster of the long audit suite (preflights, timed phases, overall).
///
/// In strict mode (`--strict`), every canonical phase must be present
/// in the receipt without duplicates or unknown IDs, with 'overall' as the
/// final entry, and with timed phases (`phase1`..`phase7`) executing > 0 tests.
pub const CANONICAL_ROSTER: [&str; 14] = [
    "preflight-render",
    "preflight-catalog",
    "preflight-package",
    "preflight-freshness",
    "preflight-simd-probe",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "overall",
];

/// `true` when `phase_id` belongs to the preflight namespace (`preflight-*`).
pub fn is_preflight_id(phase_id: &str) -> bool {
    phase_id.starts_with("preflight-")
}

/// Phase ids of the long suite's performance-class phases: RT Deadline =
/// `phase5`, RT Jitter = `phase6`, Perf Gate = `phase8`. Kept as the canonical
/// public matrix for consumers that classify performance phases separately
/// (the human `FIDELITY` verdict uses [`FIDELITY_PHASE_IDS`] instead). The
/// phase matrix itself is an invariant of `utils/tests-long.sh`.
pub const PERFORMANCE_PHASE_IDS: [&str; 3] = ["phase5", "phase6", "phase8"];

/// Phase ids that back the human `FIDELITY` verdict. Preflights whose object is
/// provenance or packaging (`preflight-render`, `preflight-package`,
/// `preflight-simd-probe`) are deliberately excluded: their failure is reported
/// by `PROVENANCE`/the overall verdict, not as a fidelity failure.
pub const FIDELITY_PHASE_IDS: [&str; 5] =
    ["preflight-catalog", "phase1", "phase3", "phase4", "phase7"];
