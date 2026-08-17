// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! JSONL fidelity metrics ingest (F-27) — `FidelityRecord` + canonical parser.
//!
//! Ports the canonical JSONL fidelity stream semantics of
//! `parse_jsonl_fidelity` (`utils/quality-dashboard.sh:530-579`) and the
//! `_is_finite_num` numeric gate (`:217-225`, findings F-01/F-28) to Rust,
//! jq-free.
//!
//! The stream shape matches the sink emitted by
//! `tests/common/validation.rs:567-575`: one JSON object per line with
//! `label`, `kind`, and the metric fields `esr`, `esr_db`, `snr_db`, `mse`,
//! `mrstft`. Metrics arrive either as numbers or as explicit string
//! sentinels (`"inf"`, `"-inf"`, `"nan"`) — never `null` when written by the
//! sink, but `null` and empty strings are normalized to the `N/A` state on
//! read exactly like the bash `canon` helper.
//!
//! Non-finite values are **preserved** as raw text at ingest time and
//! rejected only by `is_finite_num` at verify time — fail-closed, never
//! coerced to `0.0`.
//!
//! Available only with the `testing` feature (the whole `testing` module is
//! feature-gated at `lib.rs`).

use serde_json::Value;

/// Canonical metric value of a JSONL fidelity record.
///
/// Mirrors the bash/jq `canon` normalization: JSON `null` and the empty
/// string map to [`Na`](Self::Na); every other JSON value is preserved as
/// raw stream text (strings verbatim, numbers in their JSON rendering).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetricValue {
    /// `null` or empty-string metric — the `N/A` sentinel of the bash parse.
    Na,
    /// Raw stream text, preserved verbatim (`"1.5e2"`, `"inf"`, numbers, …).
    Raw(String),
}

impl MetricValue {
    /// The raw text behind a non-`N/A` value.
    pub fn as_raw(&self) -> Option<&str> {
        match self {
            MetricValue::Na => None,
            MetricValue::Raw(raw) => Some(raw),
        }
    }

    /// Fail-closed accessor for the `--check` path: only finite values pass.
    ///
    /// Non-finite sentinels (`"inf"`, `"nan"`, …) and `N/A` return `None` —
    /// they are never coerced to `0.0` (invariant: fail-closed on
    /// non-finite in the verify path).
    pub fn as_finite(&self) -> Option<&str> {
        self.as_raw().filter(|raw| is_finite_num(raw))
    }
}

/// One canonical fidelity record of the JSONL metric stream.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FidelityRecord {
    /// Model label; missing, null, empty, or `"null"` labels are dropped.
    pub label: String,
    /// Stream kind — `"fidelity"` or absent; other kinds are skipped until
    /// the sink extension (S2.T6).
    pub kind: Option<String>,
    /// Linear ESR against the NAMCore reference.
    pub esr: MetricValue,
    /// ESR in dB.
    pub esr_db: MetricValue,
    /// SNR in dB.
    pub snr_db: MetricValue,
    /// Mean squared error.
    pub mse: MetricValue,
    /// MR-STFT metric value.
    pub mrstft: MetricValue,
    /// Paired f64-oracle ESR (`"N/A"` when not measured).
    ///
    /// The canonical sink does not emit this field yet (S2.T6); the verify
    /// report may carry it as `"esr_f64"`.
    pub esr_f64: MetricValue,
}

/// Canonicalizes one JSON object into a fidelity record, if it is one.
///
/// Applies the kind filter of `parse_jsonl_fidelity` (accepts `"fidelity"`
/// or an absent kind; non-string kinds and other kinds are skipped) and the
/// label drop rule (missing, null, empty, or `"null"` labels are dropped).
/// Returns `None` for records that do not survive either rule.
pub(crate) fn fidelity_from_json(value: &Value) -> Option<FidelityRecord> {
    let kind = match value.get("kind") {
        None => None,
        Some(Value::String(kind)) => Some(kind.clone()),
        Some(_) => return None,
    };
    if let Some(kind) = kind.as_deref()
        && kind != "fidelity"
    {
        return None;
    }
    let label = match value.get("label") {
        Some(Value::String(label)) if !label.is_empty() && label != "null" => label.clone(),
        _ => return None,
    };
    Some(FidelityRecord {
        label,
        kind,
        esr: canon_metric(value.get("esr")),
        esr_db: canon_metric(value.get("esr_db")),
        snr_db: canon_metric(value.get("snr_db")),
        mse: canon_metric(value.get("mse")),
        mrstft: canon_metric(value.get("mrstft")),
        esr_f64: canon_metric(value.get("esr_f64")),
    })
}

/// Parses a canonical JSONL fidelity stream, jq-free.
///
/// Faithful port of `parse_jsonl_fidelity` (`quality-dashboard.sh:530`):
/// - records whose `kind` is neither `"fidelity"` nor absent are skipped
///   (other kinds land with S2.T6; non-string `kind` values are skipped too
///   — the canonical sink only ever emits `"fidelity"`);
/// - null/empty metric fields normalize to [`MetricValue::Na`];
/// - records with a missing, null, empty, or `"null"` label are dropped;
/// - blank lines are ignored.
///
/// Fail-closed divergence from the bash: a malformed JSON line fails the
/// whole parse (the bash jq pipeline could silently yield partial data).
pub fn parse_fidelity_jsonl(input: &str) -> Result<Vec<FidelityRecord>, MetricsError> {
    let mut records = Vec::new();
    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(line).map_err(|source| MetricsError::MalformedLine {
                line: line_no + 1,
                source,
            })?;
        if let Some(record) = fidelity_from_json(&value) {
            records.push(record);
        }
    }
    Ok(records)
}

/// Parses a JSONL metric file, fail-closed on unreadable input.
pub fn parse_fidelity_jsonl_file(
    path: impl AsRef<std::path::Path>,
) -> Result<Vec<FidelityRecord>, MetricsError> {
    let input = std::fs::read_to_string(path)?;
    parse_fidelity_jsonl(&input)
}

/// Typed error of the JSONL metrics ingest.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    /// A stream line is not valid JSON.
    #[error("malformed JSONL line {line}: {source}")]
    MalformedLine {
        /// 1-based line number in the stream.
        line: usize,
        /// The underlying JSON parse error.
        source: serde_json::Error,
    },
    /// The metrics file could not be read.
    #[error("cannot read metrics JSONL: {0}")]
    Io(#[from] std::io::Error),
}

/// Port of `_is_finite_num` (`quality-dashboard.sh:217-225`, F-01/F-28).
///
/// Accepts the canonical decimal/scientific grammar
/// `[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?` and rejects the
/// non-finite sentinels (`inf`, `-inf`, `+inf`, `infinity`, `-infinity`,
/// `nan`, `-nan` — case-insensitive), empty strings, and anything else.
///
/// This is a syntactic gate, faithful to the bash original: it never parses
/// or coerces the value. In particular `"1e400"` is accepted here — the
/// verify engine owns any f64 overflow concern, exactly like the awk
/// comparisons of the dashboard.
pub fn is_finite_num(v: &str) -> bool {
    if v.is_empty() {
        return false;
    }
    if matches!(
        v.to_ascii_lowercase().as_str(),
        "inf" | "-inf" | "+inf" | "infinity" | "-infinity" | "nan" | "-nan"
    ) {
        return false;
    }
    let bytes = v.as_bytes();
    let mut i = 0;
    if matches!(bytes[0], b'+' | b'-') {
        i += 1;
    }
    let mut int_digits = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        int_digits += 1;
        i += 1;
    }
    let mut frac_digits = 0;
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            frac_digits += 1;
            i += 1;
        }
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < bytes.len() && matches!(bytes[i], b'e' | b'E') {
        i += 1;
        if i < bytes.len() && matches!(bytes[i], b'+' | b'-') {
            i += 1;
        }
        let mut exp_digits = 0;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            exp_digits += 1;
            i += 1;
        }
        if exp_digits == 0 {
            return false;
        }
    }
    i == bytes.len()
}

fn canon_metric(value: Option<&Value>) -> MetricValue {
    match value {
        None | Some(Value::Null) => MetricValue::Na,
        Some(Value::String(s)) if s.is_empty() => MetricValue::Na,
        Some(Value::String(s)) => MetricValue::Raw(s.clone()),
        Some(Value::Number(n)) => MetricValue::Raw(n.to_string()),
        Some(Value::Bool(b)) => MetricValue::Raw((*b).to_string()),
        Some(other) => MetricValue::Raw(other.to_string()),
    }
}

#[cfg(test)]
#[path = "metrics_test.rs"]
mod metrics_test;
