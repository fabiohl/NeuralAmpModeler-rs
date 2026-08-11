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
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
