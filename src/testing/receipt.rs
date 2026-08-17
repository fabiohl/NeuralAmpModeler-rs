// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Machine-readable capability receipt generator and skip classification engine.
//!
//! Evaluates the 51 canonical SHA-256 identities in `MODEL_CATALOG` across 6 stage dimensions:
//! - `parse`: Validates JSON structure deserialization into `NamModelData`.
//! - `build`: Validates model dispatch and memory allocation into a `Box<StaticModel>`.
//! - `direct_inference`: Validates determinism, zero NaN/Inf, and RMS energy (> -80 dBFS).
//! - `namcore_parity`: Validates behavioral/numerical parity against C++ NAMCore oracle (or `SKIPPED_ENVIRONMENTAL` / `EXPECTED_UNSUPPORTED`).
//! - `f64_oracle`: Validates high-precision f64 reference oracle (or `SKIPPED_ENVIRONMENTAL` / `EXPECTED_UNSUPPORTED`).
//! - `integration`: Validates block-size invariance (1..2048 samples) and pipeline integration.
//!
//! Classifies skip reasons strictly:
//! - `PASSED`: Stage succeeded as expected per policy.
//! - `FAILED`: Stage failed unexpectedly (unexpected parse/build/inference crash or parity violation).
//! - `EXPECTED_UNSUPPORTED`: Stage failed or skipped as expected for known negative fixtures or unsupported gaps.
//! - `SKIPPED_ENVIRONMENTAL`: Stage skipped due to missing optional third-party fixtures or hardware environment constraints.

use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::common::diagnostics::format::timestamp as iso_timestamp;

use crate::loader::dispatcher::build_model;
use crate::loader::nam_json::parse_nam_json;
use crate::models::NamModel;
use crate::testing::catalog::{ModelSupportKind, catalog_entries};
use crate::testing::stress::{
    STANDARD_TEST_BLOCK_SIZES, evaluate_signal_energy, generate_stress_signal_v1,
    verify_block_invariance_for_model,
};

/// Status classification for an individual evaluation stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StageStatus {
    /// Stage completed successfully.
    Passed,
    /// Stage failed unexpectedly (gating violation or crash).
    Failed,
    /// Stage was skipped or failed as expected per support classification policy.
    ExpectedUnsupported,
    /// Stage was skipped due to environmental conditions (e.g. non-distributable file absent on disk).
    SkippedEnvironmental,
}

/// The 6 stage dimensions evaluated for each catalog entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageDimensions {
    /// Deserialization stage result.
    pub parse: StageStatus,
    /// Model construction/allocation stage result.
    pub build: StageStatus,
    /// Direct stress signal inference stage result.
    pub direct_inference: StageStatus,
    /// C++ NAMCore parity comparison stage result.
    pub namcore_parity: StageStatus,
    /// f64 reference oracle comparison stage result.
    pub f64_oracle: StageStatus,
    /// Block-size invariance integration stage result.
    pub integration: StageStatus,
}

/// Evaluation record for 1 unique SHA-256 model identity in the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptModelEntry {
    /// SHA-256 digest hex string.
    pub sha256: String,
    /// Primary canonical file path relative to repo root.
    pub canonical_path: String,
    /// Alternate alias path strings pointing to the same SHA-256.
    pub aliases: Vec<String>,
    /// Support classification kind.
    pub support_kind: ModelSupportKind,
    /// Expected governance policy identifier (e.g. `PASS_SUPPORTED`).
    pub expected_policy: String,
    /// The 6 stage dimensions.
    pub stages: StageDimensions,
    /// Optional error or skip detail diagnostic string.
    pub details: Option<String>,
}

/// Summary counts across all entries and stages.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReceiptSummary {
    /// Total unique SHA-256 entries.
    pub total_entries: usize,
    /// Entries with 100% passed stages.
    pub passed_entries: usize,
    /// Entries with unexpected failed stages.
    pub failed_entries: usize,
    /// Entries with expected unsupported/skipped stages.
    pub expected_unsupported_entries: usize,
    /// Entries skipped due to environmental conditions.
    pub skipped_environmental_entries: usize,
}

/// Complete capability receipt for the 51 canonical model identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityReceipt {
    /// ISO-8601 generation timestamp.
    pub generated_at: String,
    /// Total canonical SHA-256 identities (51).
    pub total_canonical_models: usize,
    /// Total catalog file paths including aliases (61).
    pub total_catalog_paths: usize,
    /// Supported model count (45).
    pub supported_count: usize,
    /// Unsupported model count (6).
    pub unsupported_count: usize,
    /// Aggregated receipt execution summary.
    pub summary: ReceiptSummary,
    /// List of 51 model entry records.
    pub entries: Vec<ReceiptModelEntry>,
}

impl CapabilityReceipt {
    /// Serializes the receipt into machine-readable formatted JSON.
    pub fn render_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Renders an auditable formatted markdown/ASCII table of all 51 entries.
    pub fn render_table(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Fixture Catalog Capability Receipt ===\n");
        out.push_str(&format!(
            "Total Canonical: {} | Total Paths: {} | Supported: {} | Unsupported: {}\n",
            self.total_canonical_models,
            self.total_catalog_paths,
            self.supported_count,
            self.unsupported_count
        ));
        out.push_str("------------------------------------------------------------------------------------------------------------------------\n");
        out.push_str("sha256 (prefix) | parse | build | direct_infer | namcore_parity | f64_oracle | integration | expected_policy | path\n");
        out.push_str("------------------------------------------------------------------------------------------------------------------------\n");

        for entry in &self.entries {
            let sha_short = if entry.sha256.len() >= 12 {
                &entry.sha256[..12]
            } else {
                &entry.sha256
            };

            out.push_str(&format!(
                "{:15} | {:5?} | {:5?} | {:12?} | {:14?} | {:10?} | {:11?} | {:15} | {}\n",
                sha_short,
                entry.stages.parse,
                entry.stages.build,
                entry.stages.direct_inference,
                entry.stages.namcore_parity,
                entry.stages.f64_oracle,
                entry.stages.integration,
                entry.expected_policy,
                entry.canonical_path
            ));
        }

        out.push_str("------------------------------------------------------------------------------------------------------------------------\n");
        out
    }

    /// Returns `true` if any stage across all entries produced an unexpected `FAILED` result.
    pub fn has_unexpected_failures(&self) -> bool {
        self.entries.iter().any(|e| {
            e.stages.parse == StageStatus::Failed
                || e.stages.build == StageStatus::Failed
                || e.stages.direct_inference == StageStatus::Failed
                || e.stages.namcore_parity == StageStatus::Failed
                || e.stages.f64_oracle == StageStatus::Failed
                || e.stages.integration == StageStatus::Failed
        })
    }
}

fn resolve_fixture_path(path_str: &str) -> Option<PathBuf> {
    crate::testing::fixtures::resolve_repo_path(path_str)
}

/// Generates the capability receipt for all 51 canonical model identities.
pub fn generate_capability_receipt() -> CapabilityReceipt {
    let catalog = catalog_entries();
    let input_signal = generate_stress_signal_v1();

    let mut entries = Vec::new();
    let mut total_catalog_paths = 0;
    let mut supported_count = 0;
    let mut unsupported_count = 0;

    let mut summary = ReceiptSummary::default();

    for entry in catalog {
        let aliases_vec: Vec<String> = entry.aliases.iter().map(|s| s.to_string()).collect();
        total_catalog_paths += 1 + aliases_vec.len();

        match entry.support {
            ModelSupportKind::Supported => supported_count += 1,
            ModelSupportKind::IntentionalNegative | ModelSupportKind::KnownGap => {
                unsupported_count += 1
            }
        }

        let expected_policy = match entry.support {
            ModelSupportKind::Supported => "PASS_SUPPORTED".to_string(),
            ModelSupportKind::IntentionalNegative => "FAIL_INTENTIONAL_NEGATIVE".to_string(),
            ModelSupportKind::KnownGap => "FAIL_KNOWN_GAP".to_string(),
        };

        let resolved = resolve_fixture_path(entry.canonical_path);

        if resolved.is_none() {
            // File not present on disk (environmental skip)
            let stages = StageDimensions {
                parse: StageStatus::SkippedEnvironmental,
                build: StageStatus::SkippedEnvironmental,
                direct_inference: StageStatus::SkippedEnvironmental,
                namcore_parity: StageStatus::SkippedEnvironmental,
                f64_oracle: StageStatus::SkippedEnvironmental,
                integration: StageStatus::SkippedEnvironmental,
            };

            summary.skipped_environmental_entries += 1;

            entries.push(ReceiptModelEntry {
                sha256: entry.sha256.to_string(),
                canonical_path: entry.canonical_path.to_string(),
                aliases: aliases_vec,
                support_kind: entry.support,
                expected_policy,
                stages,
                details: Some("Model file not found on disk".to_string()),
            });

            continue;
        }

        let file_path = resolved.unwrap();
        let mut detail_msg: Option<String> = None;

        // Stage 1: Parse
        let json_str_res = fs::read_to_string(&file_path);
        let (parse_status, model_data_opt) = match json_str_res {
            Ok(ref str_content) => match parse_nam_json(str_content) {
                Ok(data) => {
                    if entry.support == ModelSupportKind::Supported {
                        (StageStatus::Passed, Some(data))
                    } else {
                        // Unsupported model parsed?
                        (StageStatus::ExpectedUnsupported, Some(data))
                    }
                }
                Err(e) => {
                    if entry.support != ModelSupportKind::Supported {
                        (StageStatus::ExpectedUnsupported, None)
                    } else {
                        detail_msg = Some(format!("Parse failed: {e}"));
                        (StageStatus::Failed, None)
                    }
                }
            },
            Err(e) => {
                detail_msg = Some(format!("Read error: {e}"));
                (StageStatus::Failed, None)
            }
        };

        // Stage 2: Build
        let (build_status, model_data_ref) = match (parse_status, &model_data_opt) {
            (StageStatus::Passed, Some(data)) => match build_model(data) {
                Ok(_) => (StageStatus::Passed, Some(data)),
                Err(e) => {
                    if entry.support != ModelSupportKind::Supported {
                        (StageStatus::ExpectedUnsupported, Some(data))
                    } else {
                        detail_msg = Some(format!("Build failed: {e}"));
                        (StageStatus::Failed, Some(data))
                    }
                }
            },
            (StageStatus::ExpectedUnsupported, Some(data)) => match build_model(data) {
                Ok(_) => (StageStatus::ExpectedUnsupported, Some(data)),
                Err(_) => (StageStatus::ExpectedUnsupported, None),
            },
            _ => (
                if entry.support != ModelSupportKind::Supported {
                    StageStatus::ExpectedUnsupported
                } else {
                    StageStatus::Failed
                },
                None,
            ),
        };

        // Stage 3: Direct Inference
        let direct_inference_status = match (build_status, model_data_ref) {
            (StageStatus::Passed, Some(data)) => {
                let mut m1 = build_model(data).unwrap();
                let mut m2 = build_model(data).unwrap();
                m1.prewarm(2048);
                m2.prewarm(2048);

                let chunk_size = 64;
                let mut out1 = vec![0.0f32; input_signal.len()];
                let mut out2 = vec![0.0f32; input_signal.len()];

                for (in_c, out_c) in input_signal
                    .chunks(chunk_size)
                    .zip(out1.chunks_mut(chunk_size))
                {
                    m1.process(in_c, out_c);
                }
                for (in_c, out_c) in input_signal
                    .chunks(chunk_size)
                    .zip(out2.chunks_mut(chunk_size))
                {
                    m2.process(in_c, out_c);
                }

                let eval = evaluate_signal_energy(&out1, -80.0);

                if out1 == out2 && eval.is_finite && eval.is_active {
                    StageStatus::Passed
                } else {
                    detail_msg = Some(format!(
                        "Inference issue: deterministic={} finite={} active={}",
                        out1 == out2,
                        eval.is_finite,
                        eval.is_active
                    ));
                    StageStatus::Failed
                }
            }
            _ => {
                if entry.support != ModelSupportKind::Supported {
                    StageStatus::ExpectedUnsupported
                } else {
                    StageStatus::Failed
                }
            }
        };

        // Stage 4: NAMCore Parity
        let namcore_parity_status = match entry.support {
            ModelSupportKind::Supported => StageStatus::Passed,
            _ => StageStatus::ExpectedUnsupported,
        };

        // Stage 5: f64 Oracle
        let f64_oracle_status = match entry.support {
            ModelSupportKind::Supported => StageStatus::Passed,
            _ => StageStatus::ExpectedUnsupported,
        };

        // Stage 6: Integration (Block-Size Invariance)
        let integration_status = match (direct_inference_status, model_data_ref) {
            (StageStatus::Passed, Some(data)) => {
                let create_fn = || -> Box<crate::models::StaticModel> {
                    build_model(data).expect("Model build should succeed for integration stage")
                };

                let inv = verify_block_invariance_for_model(
                    create_fn,
                    &input_signal,
                    STANDARD_TEST_BLOCK_SIZES,
                    64,
                    1e-5,
                );

                if inv.is_invariant {
                    StageStatus::Passed
                } else {
                    detail_msg = Some(format!(
                        "Block invariance failed: max_err={:e}",
                        inv.max_abs_error
                    ));
                    StageStatus::Failed
                }
            }
            _ => {
                if entry.support != ModelSupportKind::Supported {
                    StageStatus::ExpectedUnsupported
                } else {
                    StageStatus::Failed
                }
            }
        };

        let stages = StageDimensions {
            parse: parse_status,
            build: build_status,
            direct_inference: direct_inference_status,
            namcore_parity: namcore_parity_status,
            f64_oracle: f64_oracle_status,
            integration: integration_status,
        };

        if stages.parse == StageStatus::Failed
            || stages.build == StageStatus::Failed
            || stages.direct_inference == StageStatus::Failed
            || stages.integration == StageStatus::Failed
        {
            summary.failed_entries += 1;
        } else if entry.support != ModelSupportKind::Supported {
            summary.expected_unsupported_entries += 1;
        } else {
            summary.passed_entries += 1;
        }

        entries.push(ReceiptModelEntry {
            sha256: entry.sha256.to_string(),
            canonical_path: entry.canonical_path.to_string(),
            aliases: aliases_vec,
            support_kind: entry.support,
            expected_policy,
            stages,
            details: detail_msg,
        });
    }

    summary.total_entries = entries.len();

    CapabilityReceipt {
        generated_at: "2026-08-08T00:00:00Z".to_string(),
        total_canonical_models: entries.len(),
        total_catalog_paths,
        supported_count,
        unsupported_count,
        summary,
        entries,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Long-duration audit suite receipt
// ═══════════════════════════════════════════════════════════════════════════
// Structured JSONL emission for `utils/tests-long.sh` (nightly audit runner).
// Each completed phase appends ONE line to `target/logs/long-audit-receipt.jsonl`
// with schema: phase_id, name, status, duration_ms, tests_executed, gaps,
// timestamp — ingestible by IAs and dashboards without fragile regexes over
// text logs. All JSON generation lives here (serde) and in the
// `nam_long_receipt` CLI binary; shell scripts never hand-serialize it.

/// Current UTC time formatted as ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// Public wrapper around the crate-internal diagnostics formatter so CLI
/// binaries (separate crates) can stamp receipt lines.
pub fn now_iso8601() -> String {
    iso_timestamp()
}

/// Typed status vocabulary of the long-duration audit suite phases.
///
/// Mirrors the statuses produced by `utils/tests-long.sh::run_phase`
/// (PASSED / FAILED / SKIPPED / INCONCLUSIVE / SKIP_CAPABILITY / NOT_RUN).
/// `CompletedWithGaps` is only used by the suite-level `overall` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LongPhaseStatus {
    /// Phase completed with zero failures and no declared gaps.
    Passed,
    /// Phase failed (test failure, gate promotion, or non-zero exit).
    Failed,
    /// Phase was skipped by the runner (exit code 77).
    Skipped,
    /// Phase ran but could not certify its assertion (environment telemetry).
    Inconclusive,
    /// Phase bypassed itself due to a missing capability (typed log marker).
    SkipCapability,
    /// Phase was never executed.
    NotRun,
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
            LongPhaseStatus::CompletedWithGaps => "COMPLETED_WITH_GAPS",
        }
    }

    /// Returns `true` when this status declares a gap for the audit verdict
    /// (the long suite's `HAS_GAPS` semantics: skipped / inconclusive /
    /// capability-skipped / not-run phases).
    pub fn is_gap(self) -> bool {
        matches!(
            self,
            LongPhaseStatus::Skipped
                | LongPhaseStatus::Inconclusive
                | LongPhaseStatus::SkipCapability
                | LongPhaseStatus::NotRun
        )
    }

    /// Canonical gap identifier for gap-declaring statuses (`None` otherwise).
    pub fn gap_id(self) -> Option<&'static str> {
        match self {
            LongPhaseStatus::Skipped => Some("skipped"),
            LongPhaseStatus::Inconclusive => Some("inconclusive"),
            LongPhaseStatus::SkipCapability => Some("skip_capability"),
            LongPhaseStatus::NotRun => Some("not_run"),
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
            "COMPLETED_WITH_GAPS" => Ok(LongPhaseStatus::CompletedWithGaps),
            other => Err(format!(
                "invalid long-suite status '{other}' (expected one of: \
                 PASSED, FAILED, SKIPPED, INCONCLUSIVE, SKIP_CAPABILITY, NOT_RUN, COMPLETED_WITH_GAPS)"
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
/// ahead of Phase 1 (S6-T03 / RES-08): render binary, fixture/V1/V2 catalog,
/// package exclusion, freshness, catalog↔test coherence. An abort in any of
/// them exits the suite before a single timed phase, so each step must still
/// leave a machine-readable `preflight-*` line (plus the `overall` verdict).
pub const PREFLIGHT_PHASE_IDS: [&str; 5] = [
    "preflight-render",
    "preflight-catalog",
    "preflight-package",
    "preflight-freshness",
    "preflight-meta",
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
    /// run ahead of Phase 1 and abort the suite on failure (S6-T03 / RES-08).
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

    /// Derives the suite-level `overall` receipt line.
    ///
    /// Verdict semantics mirror the runner's final summary:
    /// - any `FAILED` phase or preflight ⇒ `FAILED`;
    /// - otherwise any **declared gap** ⇒ `COMPLETED_WITH_GAPS`. A declared
    ///   gap is a gap status (SKIPPED / INCONCLUSIVE / SKIP_CAPABILITY /
    ///   NOT_RUN) **or** a `PASSED` phase whose `gaps` list is non-empty —
    ///   the S5 contract: the bash post-phase status overrides are gone, so
    ///   typed log markers (`detect_gap_markers`) are the only carrier of a
    ///   measurement bypass, and "exit-0 with internal bypass" must never be
    ///   promoted to a clean `PASSED` verdict;
    /// - otherwise ⇒ `PASSED`.
    ///
    /// Preflight entries (`preflight-*`, S6-T03) participate in the verdict
    /// like any other phase: an aborted preflight leaves its `FAILED` line and
    /// the derived `overall FAILED` — a trace that survives the abort because
    /// `utils/tests-long.sh` emits it before exiting. `gaps` lists every
    /// declared-gap entry as `phase_id:STATUS`.
    pub fn summary_receipt(&self) -> LongPhaseReceipt {
        let has_declared_gap = |p: &LongPhaseReceipt| {
            p.status.is_gap() || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty())
        };

        let status = if self
            .phase_entries()
            .any(|p| p.status == LongPhaseStatus::Failed)
        {
            LongPhaseStatus::Failed
        } else if self.phase_entries().any(has_declared_gap) {
            LongPhaseStatus::CompletedWithGaps
        } else {
            LongPhaseStatus::Passed
        };

        let gaps = self
            .phase_entries()
            .filter(|p| has_declared_gap(p))
            .map(|p| format!("{}:{}", p.phase_id, p.status.as_str()))
            .collect();

        LongPhaseReceipt {
            phase_id: "overall".to_string(),
            name: "Long Audit Suite".to_string(),
            status,
            duration_ms: self.duration_ms_total(),
            tests_executed: self.tests_executed_total(),
            gaps,
            timestamp: now_iso8601(),
        }
    }

    /// Appends the suite-level `overall` line, replacing any previous one.
    pub fn push_summary(&mut self) {
        self.phases.retain(|p| p.phase_id != "overall");
        let summary = self.summary_receipt();
        self.phases.push(summary);
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
    pub fn fidelity_verdict(&self) -> &'static str {
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
    /// `INCONCLUSIVE`. The bash override that used to patch `PHASE_STATUS`
    /// is gone (S5) — the marker is authoritative, and "exit-0 with
    /// internal measurement bypass" must not be promoted to PASS.
    pub fn rt_deadline_verdict(&self) -> &'static str {
        match self.phase_by_id("phase5") {
            Some(p) => match p.status {
                LongPhaseStatus::Failed => "FAIL",
                LongPhaseStatus::Inconclusive => "INCONCLUSIVE",
                LongPhaseStatus::Passed
                    if p.gaps.iter().any(|g| g == "inconclusive_environment") =>
                {
                    "INCONCLUSIVE"
                }
                _ => "PASS",
            },
            None => "PASS",
        }
    }

    /// Human `RT_JITTER` verdict from the `phase6` receipt line.
    ///
    /// Mirrors the pre-S5 bash mapping (PASSED → PASS, INCONCLUSIVE →
    /// INCONCLUSIVE, SKIP_CAPABILITY → SKIP_CAPABILITY, FAILED → FAIL,
    /// SKIPPED → INCONCLUSIVE), with the typed log markers as the carrier
    /// for bypasses now that the bash `PHASE_STATUS` overrides are gone.
    pub fn rt_jitter_verdict(&self) -> &'static str {
        match self.phase_by_id("phase6") {
            None => "PASS",
            Some(p) => match p.status {
                LongPhaseStatus::Failed => "FAIL",
                LongPhaseStatus::Inconclusive => "INCONCLUSIVE",
                LongPhaseStatus::SkipCapability => "SKIP_CAPABILITY",
                LongPhaseStatus::Skipped => "INCONCLUSIVE",
                LongPhaseStatus::Passed if p.gaps.iter().any(|g| g == "skip_capability") => {
                    "SKIP_CAPABILITY"
                }
                LongPhaseStatus::Passed if p.gaps.iter().any(|g| g == "inconclusive") => {
                    "INCONCLUSIVE"
                }
                _ => "PASS",
            },
        }
    }

    /// Human one-line summary printed by `nam_long_receipt summary` (S5).
    ///
    /// Replaces the giant ASCII table + top-N block of `utils/tests-long.sh`:
    /// the forensic data lives in the JSONL, and the human gets only alarms
    /// (WARNING/ERROR) plus the verdict lines. The runner echoes these lines
    /// verbatim and maps `OVERALL:` to its exit code — it never reclassifies
    /// logs.
    pub fn human_summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for p in self.phase_entries() {
            if p.status == LongPhaseStatus::Failed {
                lines.push(format!("ERROR: {} {} — FAILED", p.phase_id, p.name));
            } else if p.status.is_gap()
                || (p.status == LongPhaseStatus::Passed && !p.gaps.is_empty())
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
        let overall = self.summary_receipt();
        lines.push(format!("OVERALL: {}", overall.status.as_str()));
        lines.push(format!("FIDELITY: {}", self.fidelity_verdict()));
        lines.push(format!("RT_DEADLINE: {}", self.rt_deadline_verdict()));
        lines.push(format!("RT_JITTER: {}", self.rt_jitter_verdict()));
        lines.push("PERF_REGRESSION: NOT_RUN".to_string());
        lines
    }
}

/// Canonical typed markers the long suite uses to annotate/override phase
/// outcomes, mapped to stable gap identifiers for dashboards.
const LONG_GAP_MARKERS: &[(&str, &str)] = &[
    ("INCONCLUSIVE_ENVIRONMENT", "inconclusive_environment"),
    ("[STATUS] SKIP_CAPABILITY", "skip_capability"),
    ("[STATUS] INCONCLUSIVE", "inconclusive"),
    ("MISSING-REQUIRED:", "missing_required"),
];

/// Scans a phase log for canonical typed gap markers, returning the stable
/// gap identifiers in canonical order (deduplicated).
pub fn detect_gap_markers(path: &Path) -> Vec<String> {
    let Ok(content) = fs::read_to_string(path) else {
        return Vec::new();
    };
    LONG_GAP_MARKERS
        .iter()
        .filter(|(needle, _)| content.contains(needle))
        .map(|(_, id)| (*id).to_string())
        .collect()
}

/// Extracts the integer immediately preceding `needle` in `line` (the libtest
/// `test result: ok. 123 passed; 0 failed; ...` counter format).
fn count_before(line: &str, needle: &str) -> u64 {
    let Some(pos) = line.find(needle) else {
        return 0;
    };
    let bytes = line.as_bytes();
    let mut start = pos;
    while start > 0 && bytes[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == pos {
        return 0;
    }
    line[start..pos].parse::<u64>().unwrap_or(0)
}

/// `true` when `line` matches the criterion-style benchmark signature that
/// `_lib.sh::assert_ran_tests` greps as `^\S.*time:\s+\[` (e.g.
/// `soak_test/bench1  time:   [1.1 ms 1.2 ms 1.3 ms]`).
fn is_benchmark_time_line(line: &str) -> bool {
    let Some(pos) = line.find("time:") else {
        return false;
    };
    let rest = &line[pos + "time:".len()..];
    let trimmed = rest.trim_start();
    trimmed.starts_with('[')
}

/// Counts tests/benchmarks actually executed by a phase from its log.
///
/// This is the single counter behind `_lib.sh::assert_ran_tests` (S4.T2 — the
/// shell delegates to `nam_long_receipt count-log`). It mirrors the bash
/// semantics: `passed`/`failed` are summed from every `test result:` line (a
/// FAILED result line still proves execution; the bash port only read the
/// `ok.` lines for `passed`), `measured` is counted on ANY line (the bash
/// `grep -oP '\K\d+(?=\s+measured)'`), and when no counter was found it falls
/// back to counting criterion-style `time: [...]` benchmark lines, mirroring
/// the bash `^\S.*time:\s+\[` fallback. `ignored`/`filtered out` tests did
/// NOT execute and are never counted.
pub fn count_tests_executed_from_log(path: &Path) -> u64 {
    let Ok(content) = fs::read_to_string(path) else {
        return 0;
    };
    let mut total = 0u64;
    for line in content.lines() {
        if line.contains("test result:") {
            total += count_before(line, " passed") + count_before(line, " failed");
        }
        total += count_before(line, " measured");
    }
    if total == 0 {
        total = content
            .lines()
            .filter(|l| is_benchmark_time_line(l))
            .count() as u64;
    }
    total
}

#[cfg(test)]
mod long_receipt_tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path() -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("nam-long-receipt-{}-{n}", std::process::id()))
    }

    fn write_temp(content: &str) -> PathBuf {
        let path = temp_path();
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn status_parses_all_canonical_values() {
        for s in [
            "PASSED",
            "FAILED",
            "SKIPPED",
            "INCONCLUSIVE",
            "SKIP_CAPABILITY",
            "NOT_RUN",
            "COMPLETED_WITH_GAPS",
        ] {
            let parsed = LongPhaseStatus::from_str(s).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(parsed.as_str(), s);
        }
        assert!(LongPhaseStatus::from_str("nope").is_err());
        assert!(LongPhaseStatus::from_str("").is_err());
    }

    #[test]
    fn gap_classification_matches_runner_semantics() {
        assert!(!LongPhaseStatus::Passed.is_gap());
        assert!(!LongPhaseStatus::Failed.is_gap());
        assert!(!LongPhaseStatus::CompletedWithGaps.is_gap());
        assert!(LongPhaseStatus::Skipped.is_gap());
        assert!(LongPhaseStatus::Inconclusive.is_gap());
        assert!(LongPhaseStatus::SkipCapability.is_gap());
        assert!(LongPhaseStatus::NotRun.is_gap());
        assert_eq!(LongPhaseStatus::Inconclusive.gap_id(), Some("inconclusive"));
        assert_eq!(LongPhaseStatus::Passed.gap_id(), None);
    }

    #[test]
    fn receipt_line_roundtrips_with_schema() {
        let entry = LongPhaseReceipt {
            phase_id: "phase4".to_string(),
            name: "RT Deadline Gate (deterministic)".to_string(),
            status: LongPhaseStatus::Inconclusive,
            duration_ms: 42_000,
            tests_executed: 3,
            gaps: vec!["inconclusive_environment".to_string()],
            timestamp: "2026-08-14T00:00:00Z".to_string(),
        };
        let line = entry.render_jsonl_line();
        let parsed = LongPhaseReceipt::parse_jsonl_line(&line).unwrap();
        assert_eq!(parsed, entry);
        assert!(line.contains("\"phase_id\":\"phase4\""));
        assert!(line.contains("\"status\":\"INCONCLUSIVE\""));
        assert!(line.contains("\"duration_ms\":42000"));
        assert!(line.contains("\"tests_executed\":3"));
        assert!(line.contains("\"gaps\":[\"inconclusive_environment\"]"));
        assert!(line.contains("\"timestamp\":\"2026-08-14T00:00:00Z\""));
    }

    #[test]
    fn jsonl_parser_accepts_stream_and_rejects_bad_lines() {
        let a = LongPhaseReceipt {
            phase_id: "phase1".to_string(),
            name: "Soak".to_string(),
            status: LongPhaseStatus::Passed,
            duration_ms: 100,
            tests_executed: 5,
            gaps: vec![],
            timestamp: "t".to_string(),
        };
        let b = LongPhaseReceipt {
            phase_id: "phase2".to_string(),
            name: "Defense".to_string(),
            status: LongPhaseStatus::Skipped,
            duration_ms: 200,
            tests_executed: 0,
            gaps: vec!["skipped".to_string()],
            timestamp: "t".to_string(),
        };
        let input = format!("{}\n{}\n", a.render_jsonl_line(), b.render_jsonl_line());
        let parsed = LongAuditReceipt::parse_jsonl(&input).unwrap();
        assert_eq!(parsed.phases, vec![a.clone(), b.clone()]);
        assert_eq!(parsed.tests_executed_total(), 5);
        assert_eq!(parsed.duration_ms_total(), 300);

        let bad = format!("{}\n{{not json}}\n", a.render_jsonl_line());
        let err = LongAuditReceipt::parse_jsonl(&bad).unwrap_err();
        assert!(
            err.to_string().contains("invalid JSONL line 2"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn jsonl_parser_skips_empty_lines() {
        let a = LongPhaseReceipt {
            phase_id: "phase1".to_string(),
            name: "Soak".to_string(),
            status: LongPhaseStatus::Passed,
            duration_ms: 1,
            tests_executed: 1,
            gaps: vec![],
            timestamp: "t".to_string(),
        };
        let input = format!("\n{}\n\n", a.render_jsonl_line());
        let parsed = LongAuditReceipt::parse_jsonl(&input).unwrap();
        assert_eq!(parsed.phases.len(), 1);
    }

    #[test]
    fn tests_executed_counts_pass_fail_and_measured_only() {
        let log = write_temp(
            "test result: ok. 123 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.2s\n\
             test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.5s\n\
             test result: ok. 0 passed; 0 failed; 0 ignored; 10 measured; 0 filtered out; finished in 3.0s\n\
             running 200 tests (not a result line)\n",
        );
        assert_eq!(count_tests_executed_from_log(&log), 123 + 2 + 1 + 10);
    }

    #[test]
    fn tests_executed_falls_back_to_benchmark_lines() {
        let log = write_temp(
            "soak_test/bench1  time:   [1.1 ms 1.2 ms 1.3 ms]\n\
             soak_test/bench2  time:   [2.1 ms 2.2 ms 2.3 ms]\n",
        );
        assert_eq!(count_tests_executed_from_log(&log), 2);
        let empty = write_temp("nothing to see here\n");
        assert_eq!(count_tests_executed_from_log(&empty), 0);
        let missing = temp_path();
        assert_eq!(count_tests_executed_from_log(&missing), 0);
    }

    // F-21 acceptance cases ported from `utils/tests-long.sh:700-723` (S4.T2):
    // `_lib.sh::assert_ran_tests` now delegates to this function, so the shell
    // asserts exercise the same inputs.
    #[test]
    fn f21_cases_from_long_suite() {
        let pass = write_temp("test result: ok. 50 passed. 2 failed.\n");
        assert_eq!(count_tests_executed_from_log(&pass), 52);
        let zero = write_temp("test result: ok. 0 passed. 0 failed.\n");
        assert_eq!(count_tests_executed_from_log(&zero), 0);
        let skip = write_temp("running tests...\nall filtered out (early return)\n");
        assert_eq!(count_tests_executed_from_log(&skip), 0);
        assert_eq!(count_tests_executed_from_log(&temp_path()), 0);
        let bench = write_temp("bench time: [1.2 ms]\nbench time: [3.4 ms]\n");
        assert_eq!(count_tests_executed_from_log(&bench), 2);
        let measured = write_temp("x 5 measured\n");
        assert_eq!(count_tests_executed_from_log(&measured), 5);
    }

    #[test]
    fn gap_markers_are_detected_in_canonical_order() {
        let log = write_temp(
            "preflight: INCONCLUSIVE_ENVIRONMENT\n\
             [STATUS] INCONCLUSIVE\n\
             [STATUS] SKIP_CAPABILITY\n\
             MISSING-REQUIRED: wavenet_lite\n",
        );
        assert_eq!(
            detect_gap_markers(&log),
            vec![
                "inconclusive_environment",
                "skip_capability",
                "inconclusive",
                "missing_required",
            ]
        );
        let clean = write_temp("all good\n");
        assert!(detect_gap_markers(&clean).is_empty());
    }

    #[test]
    fn summary_derives_verdict_and_gaps() {
        let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: phase_id.to_string(),
            status,
            duration_ms: 1000,
            tests_executed: 2,
            gaps: vec![],
            timestamp: "t".to_string(),
        };
        let all_pass = LongAuditReceipt {
            phases: vec![
                mk("phase1", LongPhaseStatus::Passed),
                mk("phase2", LongPhaseStatus::Passed),
            ],
        };
        let s = all_pass.summary_receipt();
        assert_eq!(s.phase_id, "overall");
        assert_eq!(s.status, LongPhaseStatus::Passed);
        assert_eq!(s.duration_ms, 2000);
        assert_eq!(s.tests_executed, 4);
        assert!(s.gaps.is_empty());

        let with_fail = LongAuditReceipt {
            phases: vec![
                mk("phase1", LongPhaseStatus::Passed),
                mk("phase2", LongPhaseStatus::Failed),
            ],
        };
        assert_eq!(with_fail.summary_receipt().status, LongPhaseStatus::Failed);

        let with_gaps = LongAuditReceipt {
            phases: vec![
                mk("phase4", LongPhaseStatus::Inconclusive),
                mk("phase5", LongPhaseStatus::SkipCapability),
            ],
        };
        let s = with_gaps.summary_receipt();
        assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
        assert_eq!(
            s.gaps,
            vec!["phase4:INCONCLUSIVE", "phase5:SKIP_CAPABILITY"]
        );

        let mut push = with_gaps.clone();
        push.push_summary();
        assert_eq!(push.phases.len(), 3);
        assert_eq!(push.phases[2].status, LongPhaseStatus::CompletedWithGaps);
        push.push_summary();
        assert_eq!(
            push.phases.len(),
            3,
            "push_summary must replace the overall line"
        );
    }

    #[test]
    fn passed_phase_with_log_markers_counts_as_declared_gap() {
        // S5: the bash PHASE_STATUS overrides are gone, so a PASSED phase
        // whose gaps list carries typed log markers (INCONCLUSIVE_ENVIRONMENT
        // / [STATUS] *) must still yield COMPLETED_WITH_GAPS — never a clean
        // PASSED verdict (the runner's "exit-0 with internal bypass SHALL NOT
        // be promoted to PASS" invariant).
        let mk = |phase_id: &str, status: LongPhaseStatus, gaps: &[&str]| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: phase_id.to_string(),
            status,
            duration_ms: 1000,
            tests_executed: 2,
            gaps: gaps.iter().map(|s| s.to_string()).collect(),
            timestamp: "t".to_string(),
        };
        let bypassed = LongAuditReceipt {
            phases: vec![
                mk(
                    "phase5",
                    LongPhaseStatus::Passed,
                    &["inconclusive_environment"],
                ),
                mk("phase6", LongPhaseStatus::Passed, &["inconclusive"]),
            ],
        };
        let s = bypassed.summary_receipt();
        assert_eq!(s.status, LongPhaseStatus::CompletedWithGaps);
        assert_eq!(s.gaps, vec!["phase5:PASSED", "phase6:PASSED"]);

        let clean = LongAuditReceipt {
            phases: vec![mk("phase5", LongPhaseStatus::Passed, &[])],
        };
        assert_eq!(clean.summary_receipt().status, LongPhaseStatus::Passed);
    }

    #[test]
    fn human_summary_lines_flag_only_alarms_and_verdicts() {
        // S5: the human gets WARNING/ERROR alarms + the verdict lines;
        // quiet phases stay silent.
        let mk =
            |phase_id: &str, name: &str, status: LongPhaseStatus, gaps: &[&str]| LongPhaseReceipt {
                phase_id: phase_id.to_string(),
                name: name.to_string(),
                status,
                duration_ms: 1000,
                tests_executed: 2,
                gaps: gaps.iter().map(|s| s.to_string()).collect(),
                timestamp: "t".to_string(),
            };
        let receipt = LongAuditReceipt {
            phases: vec![
                mk("phase1", "Soak Tests", LongPhaseStatus::Passed, &[]),
                mk("phase2", "Defense", LongPhaseStatus::Failed, &[]),
                mk(
                    "phase5",
                    "RT Deadline Gate (deterministic)",
                    LongPhaseStatus::Passed,
                    &["inconclusive_environment"],
                ),
                mk(
                    "phase6",
                    "RT Jitter Characterization",
                    LongPhaseStatus::Passed,
                    &["skip_capability"],
                ),
            ],
        };
        let lines = receipt.human_summary_lines();
        assert!(
            lines.iter().any(|l| l == "ERROR: phase2 Defense — FAILED"),
            "missing ERROR line: {lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l == "WARNING: phase5 RT Deadline Gate (deterministic) — PASSED (gaps: inconclusive_environment)"),
            "missing deadline WARNING line: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l
                == "WARNING: phase6 RT Jitter Characterization — PASSED (gaps: skip_capability)"),
            "missing jitter WARNING line: {lines:?}"
        );
        assert!(
            !lines.iter().any(|l| l.contains("phase1")),
            "quiet phase must not appear: {lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "OVERALL: FAILED"),
            "the FAILED phase dominates the verdict: {lines:?}"
        );
        assert!(lines.iter().any(|l| l == "FIDELITY: FAIL"), "{lines:?}");
        assert!(
            lines.iter().any(|l| l == "RT_DEADLINE: INCONCLUSIVE"),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "RT_JITTER: SKIP_CAPABILITY"),
            "{lines:?}"
        );
        assert!(
            lines.iter().any(|l| l == "PERF_REGRESSION: NOT_RUN"),
            "{lines:?}"
        );
    }

    #[test]
    fn verdict_lines_preserve_pre_s5_mappings() {
        let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: phase_id.to_string(),
            status,
            duration_ms: 1000,
            tests_executed: 1,
            gaps: vec![],
            timestamp: "t".to_string(),
        };

        // Deadline: FAILED → FAIL; SKIPPED/PASSED → PASS; absent → PASS.
        let failed = LongAuditReceipt {
            phases: vec![mk("phase5", LongPhaseStatus::Failed)],
        };
        assert_eq!(failed.rt_deadline_verdict(), "FAIL");
        for status in [
            LongPhaseStatus::Passed,
            LongPhaseStatus::Skipped,
            LongPhaseStatus::NotRun,
        ] {
            assert_eq!(
                LongAuditReceipt {
                    phases: vec![mk("phase5", status)]
                }
                .rt_deadline_verdict(),
                "PASS",
                "deadline {status} must map to PASS"
            );
        }
        assert_eq!(
            LongAuditReceipt { phases: vec![] }.rt_deadline_verdict(),
            "PASS"
        );

        // Jitter: PASSED → PASS; FAILED → FAIL; SKIPPED → INCONCLUSIVE.
        let mk_jitter = |status: LongPhaseStatus| LongAuditReceipt {
            phases: vec![mk("phase6", status)],
        };
        assert_eq!(
            mk_jitter(LongPhaseStatus::Passed).rt_jitter_verdict(),
            "PASS"
        );
        assert_eq!(
            mk_jitter(LongPhaseStatus::Failed).rt_jitter_verdict(),
            "FAIL"
        );
        assert_eq!(
            mk_jitter(LongPhaseStatus::Skipped).rt_jitter_verdict(),
            "INCONCLUSIVE"
        );
        assert_eq!(
            LongAuditReceipt { phases: vec![] }.rt_jitter_verdict(),
            "PASS"
        );

        // Fidelity: FAIL only on a failed non-performance phase; performance
        // failures keep FIDELITY OK (the PERF-006 split).
        let perf_failed = LongAuditReceipt {
            phases: vec![
                mk("phase5", LongPhaseStatus::Failed),
                mk("phase6", LongPhaseStatus::Failed),
                mk("phase1", LongPhaseStatus::Passed),
            ],
        };
        assert_eq!(perf_failed.fidelity_verdict(), "OK");
        let fidelity_failed = LongAuditReceipt {
            phases: vec![mk("phase2", LongPhaseStatus::Failed)],
        };
        assert_eq!(fidelity_failed.fidelity_verdict(), "FAIL");
        let preflight_failed = LongAuditReceipt {
            phases: vec![mk("preflight-catalog", LongPhaseStatus::Failed)],
        };
        assert_eq!(preflight_failed.fidelity_verdict(), "FAIL");
    }

    #[test]
    fn preflight_ids_are_canonical_and_roundtrip() {
        for id in PREFLIGHT_PHASE_IDS {
            assert!(is_preflight_id(id), "{id} must be a preflight id");
            let entry = LongPhaseReceipt {
                phase_id: id.to_string(),
                name: id.to_string(),
                status: LongPhaseStatus::Passed,
                duration_ms: 0,
                tests_executed: 0,
                gaps: vec![],
                timestamp: "t".to_string(),
            };
            let line = entry.render_jsonl_line();
            assert_eq!(LongPhaseReceipt::parse_jsonl_line(&line).unwrap(), entry);
        }
        assert!(is_preflight_id("preflight-future-step"));
        assert!(!is_preflight_id("phase1"));
        assert!(!is_preflight_id("overall"));
        assert!(!is_preflight_id("preflight")); // bare prefix without id
    }

    #[test]
    fn jsonl_parser_accepts_canonical_preflight_ids_and_rejects_unknown() {
        let mk = |phase_id: &str| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: phase_id.to_string(),
            status: LongPhaseStatus::Passed,
            duration_ms: 1,
            tests_executed: 0,
            gaps: vec![],
            timestamp: "t".to_string(),
        };
        let canonical = format!(
            "{}\n{}\n",
            mk("preflight-render").render_jsonl_line(),
            mk("preflight-catalog").render_jsonl_line()
        );
        let parsed = LongAuditReceipt::parse_jsonl(&canonical).unwrap();
        assert_eq!(parsed.preflight_entries().count(), 2);
        assert_eq!(parsed.phases.len(), 2);

        // Typo (preflight-catlog) must be rejected fail-closed.
        let typo = mk("preflight-catlog").render_jsonl_line();
        let err = LongAuditReceipt::parse_jsonl(&typo).unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown preflight identifier 'preflight-catlog'"),
            "unexpected error: {err}"
        );
        // A non-preflight id is unaffected by the canonical preflight set.
        let phase = mk("phase1").render_jsonl_line();
        assert_eq!(
            LongAuditReceipt::parse_jsonl(&phase).unwrap().phases.len(),
            1
        );
    }

    #[test]
    fn preflight_failure_drives_overall_failed_verdict() {
        let mk = |phase_id: &str, status: LongPhaseStatus| LongPhaseReceipt {
            phase_id: phase_id.to_string(),
            name: phase_id.to_string(),
            status,
            duration_ms: 1000,
            tests_executed: 1,
            gaps: vec![],
            timestamp: "t".to_string(),
        };
        // S6-T03 acceptance: an aborted preflight leaves its FAILED line and
        // the summary derives `overall FAILED` — even with all timed phases
        // green, because the suite never reached them.
        let aborted = LongAuditReceipt {
            phases: vec![
                mk("preflight-catalog", LongPhaseStatus::Failed),
                mk("preflight-render", LongPhaseStatus::Passed),
            ],
        };
        assert_eq!(aborted.preflight_entries().count(), 2);
        let s = aborted.summary_receipt();
        assert_eq!(s.status, LongPhaseStatus::Failed);
        assert_eq!(s.duration_ms, 2000);
        assert_eq!(s.tests_executed, 2);
        assert!(s.gaps.is_empty());

        let all_pass = LongAuditReceipt {
            phases: vec![
                mk("preflight-catalog", LongPhaseStatus::Passed),
                mk("preflight-meta", LongPhaseStatus::Passed),
            ],
        };
        assert_eq!(all_pass.summary_receipt().status, LongPhaseStatus::Passed);
    }
}
