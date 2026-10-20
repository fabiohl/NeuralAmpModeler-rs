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
/// `Simulated` is only used by `--simulate` / `--dry-run` pre-registration
/// (no test executed; strict-pre-release always rejects it).
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
    /// Phase pre-registered without execution (`--simulate` / `--dry-run`).
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
    /// ISO-8601 emission timestamp (UTC).
    pub timestamp: String,
    /// Optional SHA-256 of the oracle/render artifact referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oracle_sha256: Option<String>,
    /// Optional SHA-256 of the input fixture referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture_sha256: Option<String>,
    /// Optional SHA-256 of the produced artifact referenced by the phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_sha256: Option<String>,
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
            timestamp: String::new(),
            oracle_sha256: None,
            fixture_sha256: None,
            artifact_sha256: None,
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
/// package exclusion, freshness, catalog↔test coherence, SIMD capability
/// probe. An abort in any of them exits the suite before a single timed
/// phase, so each step must still leave a machine-readable `preflight-*` line
/// (plus the `overall` verdict).
pub const PREFLIGHT_PHASE_IDS: [&str; 6] = [
    "preflight-render",
    "preflight-catalog",
    "preflight-package",
    "preflight-freshness",
    "preflight-meta",
    "preflight-simd-probe",
];

/// Canonical 14-phase roster of the long audit suite.
///
/// In strict mode (`--strict`), every canonical phase must be present
/// in the receipt without duplicates or unknown IDs, with 'overall' as the
/// final entry, and with timed phases (`phase1`..`phase7`) executing > 0 tests.
pub const CANONICAL_ROSTER: [&str; 14] = [
    "preflight-render",
    "preflight-catalog",
    "preflight-package",
    "preflight-freshness",
    "preflight-meta",
    "preflight-simd-probe",
    "phase1",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "overall",
];

/// `true` when `phase_id` belongs to the preflight namespace (`preflight-*`).
pub fn is_preflight_id(phase_id: &str) -> bool {
    phase_id.starts_with("preflight-")
}

/// Phase ids of the long suite's performance-class phases (the bash
/// `PHASE_CLASS` table, moved to the receipt in S5 so the human summary
/// derives `FIDELITY` without reclassifying logs): RT Deadline = `phase5`,
/// RT Jitter = `phase6`. The phase matrix itself is an invariant of
/// `utils/tests-long.sh` (soak → defense → proptests → heap → deadline →
/// jitter → loom).
pub const PERFORMANCE_PHASE_IDS: [&str; 2] = ["phase5", "phase6"];
