// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Criterion `estimates.json` ingestion for real-time benchmark latencies (PR-09).
//!
//! Replaces the textual time-center awk parser with typed extraction of
//! `median.point_estimate` (nanoseconds) from Criterion's `estimates.json`
//! (`target/criterion/<label>/new/estimates.json`).
//!
//! Extracted nanoseconds are converted to microseconds (`µs = ns / 1000.0`)
//! and divided **once** by the benchmark's canonical `batch_factor`
//! (`DSP_MICRO_BATCH` for micro-benchmarks, or `batch_factor` from the contract).

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::QualityContract;
use super::ids::RT_BENCH_TABLE;

/// Micro-benchmarks that process `DSP_MICRO_BATCH` blocks per Criterion sample.
pub const MICRO_BATCH_BENCH_LABELS: &[&str] = &[
    "RT_DSP_Resampler_44k1_to_48k",
    "RT_DSP_Resampler_96k_to_48k",
    "RT_DSP_CabSim_IR_Medium",
];

/// Default `DSP_MICRO_BATCH` factor when `target/bench_constants.env` is missing.
pub const DEFAULT_DSP_MICRO_BATCH: u64 = 64;

/// One extracted latency measurement for the verify report stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriterionLatencyEntry {
    /// Always `"latency"` for verify report stream.
    pub kind: String,
    /// Benchmark label (e.g. `RT_WaveNet_Std_CH16`).
    pub label: String,
    /// Median block latency in microseconds.
    pub median_latency_us: f64,
}

/// Typed error for Criterion median extraction.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CriterionExtractionError {
    /// Failed to read the file from filesystem.
    #[error("cannot read Criterion estimates file '{path}': {source_error}")]
    IoError {
        /// Path that failed to read.
        path: String,
        /// Underlying IO error details.
        source_error: String,
    },

    /// The required estimates file does not exist.
    #[error("Criterion estimates artifact missing at '{path}'")]
    MissingFile {
        /// Missing file path.
        path: String,
    },

    /// The JSON content is malformed and cannot be parsed.
    #[error("malformed Criterion estimates JSON in '{path}': {source_error}")]
    MalformedJson {
        /// Path with malformed JSON.
        path: String,
        /// Underlying JSON parser error details.
        source_error: String,
    },

    /// The JSON does not contain the required `median.point_estimate` field.
    #[error("missing 'median.point_estimate' in '{path}'")]
    MissingMedian {
        /// Path with missing median field.
        path: String,
    },

    /// The median point estimate is NaN or infinite.
    #[error("non-finite median point estimate ({value}) in '{path}'")]
    NonFiniteMedian {
        /// Path with non-finite median.
        path: String,
        /// Parsed non-finite value.
        value: f64,
    },

    /// Invalid batch factor (must be >= 1).
    #[error("invalid batch factor {factor} for benchmark '{bench_label}': must be >= 1")]
    InvalidBatchFactor {
        /// Benchmark label.
        bench_label: String,
        /// The invalid batch factor value.
        factor: u64,
    },
}

/// Parses `median.point_estimate` (in nanoseconds) from a Criterion `estimates.json` string.
///
/// Fail-closed: returns an error if `median` or `point_estimate` is missing, null, or not finite.
pub fn parse_criterion_median_ns(
    json_str: &str,
    path_context: &str,
) -> Result<f64, CriterionExtractionError> {
    let value: serde_json::Value =
        serde_json::from_str(json_str).map_err(|e| CriterionExtractionError::MalformedJson {
            path: path_context.to_string(),
            source_error: e.to_string(),
        })?;

    let median_obj =
        value
            .get("median")
            .ok_or_else(|| CriterionExtractionError::MissingMedian {
                path: path_context.to_string(),
            })?;

    let point_estimate = median_obj
        .get("point_estimate")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| CriterionExtractionError::MissingMedian {
            path: path_context.to_string(),
        })?;

    if !point_estimate.is_finite() {
        return Err(CriterionExtractionError::NonFiniteMedian {
            path: path_context.to_string(),
            value: point_estimate,
        });
    }

    Ok(point_estimate)
}

/// Reads `DSP_MICRO_BATCH` from `bench_constants.env` if present and valid.
pub fn parse_bench_constants_env(content: &str) -> Option<u64> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("DSP_MICRO_BATCH=") {
            let val_str = rest.trim_matches(|c| c == '"' || c == '\'');
            match val_str.parse::<u64>() {
                Ok(n) if n >= 1 => return Some(n),
                _ => {}
            }
        }
    }
    None
}

/// Resolves the batch factor for a benchmark label.
pub fn resolve_batch_factor(
    bench_label: &str,
    micro_batch: u64,
    contract: Option<&QualityContract>,
) -> Result<u64, CriterionExtractionError> {
    if micro_batch == 0 {
        return Err(CriterionExtractionError::InvalidBatchFactor {
            bench_label: bench_label.to_string(),
            factor: 0,
        });
    }

    let contract_factor = contract.and_then(|c| {
        c.performance
            .iter()
            .find(|p| p.id == bench_label || p.label == bench_label)
            .and_then(|entry| entry.batch_factor)
    });

    if let Some(bf) = contract_factor {
        if bf == 0 {
            return Err(CriterionExtractionError::InvalidBatchFactor {
                bench_label: bench_label.to_string(),
                factor: 0,
            });
        }
        return Ok(bf);
    }

    if MICRO_BATCH_BENCH_LABELS.contains(&bench_label) {
        Ok(micro_batch)
    } else {
        Ok(1)
    }
}

/// Extracts real median latencies for all canonical benchmarks in `RT_BENCH_TABLE`
/// from Criterion's `estimates.json` files under `criterion_root`.
pub fn extract_criterion_latencies(
    criterion_root: &Path,
    micro_batch: u64,
    contract: Option<&QualityContract>,
) -> Result<Vec<CriterionLatencyEntry>, CriterionExtractionError> {
    let mut entries = Vec::with_capacity(RT_BENCH_TABLE.len());

    for bench in RT_BENCH_TABLE {
        let label = bench.bench_label;
        let estimates_path = criterion_root
            .join(label)
            .join("new")
            .join("estimates.json");
        if !estimates_path.is_file() {
            return Err(CriterionExtractionError::MissingFile {
                path: estimates_path.display().to_string(),
            });
        }

        let content =
            fs::read_to_string(&estimates_path).map_err(|e| CriterionExtractionError::IoError {
                path: estimates_path.display().to_string(),
                source_error: e.to_string(),
            })?;

        let median_ns = parse_criterion_median_ns(&content, &estimates_path.display().to_string())?;
        let batch_factor = resolve_batch_factor(label, micro_batch, contract)?;

        let median_us = (median_ns / 1_000.0) / (batch_factor as f64);

        entries.push(CriterionLatencyEntry {
            kind: "latency".to_string(),
            label: label.to_string(),
            median_latency_us: median_us,
        });
    }

    Ok(entries)
}

/// Formats a list of extracted latency entries into JSONL lines.
pub fn criterion_latencies_to_jsonl(
    entries: &[CriterionLatencyEntry],
) -> Result<String, serde_json::Error> {
    let mut out = String::new();
    for entry in entries {
        let json_line = serde_json::to_string(entry)?;
        out.push_str(&json_line);
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
#[path = "criterion_test.rs"]
mod criterion_test;
