// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fault-injection tests for the enriched structured loader logging (T5.1).
//!
//! Feeds synthetically corrupted payloads (NaN/Inf weights, invalid
//! `sample_rate`, NAMB CRC corruption) through the full `load_and_build_model`
//! path and verifies that:
//!
//! 1. The `LogBuffer` captures the expected structured rejection message
//!    (affected field, value, byte offset).
//! 2. The returned `LoadError` variant and the emitted `NamErrorCode` are the
//!    correct (specific, non-generic) ones.

use crate::common::diagnostics::{LoggerConfig, NamLogger, SystemSnapshot};
use crate::loader::nam_json::{NamConfig, NamModelData, WeightsLayout};
use crate::loader::namb_encoder::encode_namb;
use crate::loader::{LoadError, LoadOptions, load_and_build_model};
use std::path::PathBuf;
use std::sync::Mutex;

/// Serializes the LogBuffer-sensitive tests in this module against each other
/// (the global ring buffer is shared with every other test in the process).
static LOG_TESTS_MUTEX: Mutex<()> = Mutex::new(());

/// Ensures the global `NamLogger` is installed once (OnceLock-safe) with a
/// `Debug` level filter so WARN/ERROR records reach the `LogBuffer`.
fn ensure_logger() {
    let _ = NamLogger::init(LoggerConfig {
        level_filter: log::LevelFilter::Debug,
        emit_stderr: false,
    });
}

/// Snapshot of all records currently held in the global `LogBuffer`.
fn log_snapshot() -> Vec<crate::common::diagnostics::LogRecord> {
    NamLogger::log_buffer()
        .map(|b| b.snapshot())
        .unwrap_or_default()
}

/// Returns `true` if the buffer contains a record whose level and message
/// match the given predicates.
fn buffer_has(
    snapshot: &[crate::common::diagnostics::LogRecord],
    level: &str,
    needle: &str,
) -> bool {
    snapshot
        .iter()
        .any(|r| r.level == level && r.message.contains(needle))
}

fn temp_path(tag: &str, ext: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "nam_loader_log_{}_{}.{}",
        std::process::id(),
        tag,
        ext
    ))
}

/// Minimal finite LSTM JSON — the fault-injection base payload.
fn lstm_json(weights: &str, sample_rate: &str) -> String {
    format!(
        r#"{{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {{ "num_layers": 1, "hidden_size": 8, "layers": [] }},
        "weights": [{}],
        "sample_rate": {}
    }}"#,
        weights, sample_rate
    )
}

fn write_temp(path: &PathBuf, bytes: &[u8]) {
    std::fs::write(path, bytes).expect("temp file must be writable");
}

/// Synthetic LSTM model with the given weights (used to build NAMB payloads).
fn synthetic_lstm(weights: Vec<f32>) -> NamModelData {
    NamModelData {
        version: Some("0.5.0".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: Some(1.0),
            num_layers: Some(1),
            hidden_size: Some(8),
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights,
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// JSON weight `1e39` saturates to `+Inf` in f32 — the weights visitor must
/// reject it with a structured WARN (field/value/offset) and the loader must
/// return `LoadError::NonFiniteWeights` with the specific `NamJsonWeightNotFinite`
/// code (previously swallowed into the generic `NamJsonParseError`).
#[test]
fn test_json_nan_inf_weight_rejected_with_structured_log() {
    let _guard = LOG_TESTS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    ensure_logger();

    let json = lstm_json("0.1, 1e39, 0.2", "48000");
    let path = temp_path("json_inf_weight", "nam");
    write_temp(&path, json.as_bytes());

    let sys = SystemSnapshot::capture();
    let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
    std::fs::remove_file(&path).ok();

    let err = res.expect_err("NaN/Inf weight must be rejected");
    assert!(
        matches!(err, LoadError::NonFiniteWeights),
        "expected LoadError::NonFiniteWeights, got: {err:?}"
    );

    let snap = log_snapshot();
    assert!(
        buffer_has(
            &snap,
            "WARN",
            "[Loader] Invalid field rejected: field='weights[1]'"
        ),
        "LogBuffer must capture the structured WARN for weights[1], got:\n{}",
        snap.iter()
            .map(|r| format!("[{}] {}", r.level, r.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        buffer_has(&snap, "WARN", "offset_bytes=4"),
        "LogBuffer WARN must carry the byte offset (index 1 × 4 = 4)"
    );
    assert!(
        buffer_has(&snap, "ERROR", "NamJsonWeightNotFinite"),
        "LogBuffer must capture the specific NamErrorCode, got:\n{}",
        snap.iter()
            .map(|r| format!("[{}] {}", r.level, r.message))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// `sample_rate: 1e39` saturates to `+Inf` — the sample-rate visitor must
/// reject with a structured WARN and the loader must map to the specific
/// `NamJsonInvalidSampleRate` code.
#[test]
fn test_json_non_finite_sample_rate_rejected_with_structured_log() {
    let _guard = LOG_TESTS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    ensure_logger();

    let json = lstm_json("0.1, 0.2", "1e39");
    let path = temp_path("json_inf_sr", "nam");
    write_temp(&path, json.as_bytes());

    let sys = SystemSnapshot::capture();
    let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
    std::fs::remove_file(&path).ok();

    let err = res.expect_err("non-finite sample_rate must be rejected");
    assert!(
        matches!(err, LoadError::UnsupportedArchitecture(_)),
        "expected LoadError::UnsupportedArchitecture, got: {err:?}"
    );

    let snap = log_snapshot();
    assert!(
        buffer_has(
            &snap,
            "WARN",
            "[Loader] Invalid field rejected: field='sample_rate'"
        ),
        "LogBuffer must capture the structured WARN for sample_rate"
    );
    assert!(
        buffer_has(&snap, "ERROR", "NamJsonInvalidSampleRate"),
        "LogBuffer must capture the specific NamErrorCode"
    );
}

/// NAMB binary weight section with a NaN float — the binary parser must emit
/// the structured WARN (with the absolute byte offset) and the loader must map
/// to `NambNonFiniteWeight` / `LoadError::NonFiniteWeights`.
#[test]
fn test_namb_nan_weight_rejected_with_structured_log() {
    let _guard = LOG_TESTS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    ensure_logger();

    let model = synthetic_lstm(vec![0.1, f32::NAN, 0.2]);
    let namb = encode_namb(&model, 2, WeightsLayout::Original).expect("encode must succeed");
    let path = temp_path("namb_nan_weight", "namb");
    write_temp(&path, &namb);

    let sys = SystemSnapshot::capture();
    let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
    std::fs::remove_file(&path).ok();

    let err = res.expect_err("NAMB NaN weight must be rejected");
    assert!(
        matches!(err, LoadError::NonFiniteWeights),
        "expected LoadError::NonFiniteWeights, got: {err:?}"
    );

    let snap = log_snapshot();
    assert!(
        buffer_has(
            &snap,
            "WARN",
            "[Loader] Invalid field rejected: field='weights[1]'"
        ),
        "LogBuffer must capture the structured WARN for the NaN weight slot"
    );
    assert!(
        buffer_has(&snap, "WARN", "value=NaN"),
        "LogBuffer WARN must carry the offending value (NaN)"
    );
    assert!(
        buffer_has(&snap, "ERROR", "NambNonFiniteWeight"),
        "LogBuffer must capture the specific NAMB error code"
    );
}

/// NAMB with a corrupted weight byte (CRC mismatch) — the CRC gate must emit
/// the structured WARN and the loader must map to `NambCrc32Mismatch`.
#[test]
fn test_namb_crc_mismatch_rejected_with_structured_log() {
    let _guard = LOG_TESTS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    ensure_logger();

    let model = synthetic_lstm(vec![0.1, 0.2, 0.3]);
    let mut namb = encode_namb(&model, 2, WeightsLayout::Original).expect("encode must succeed");

    // Corrupt the last weight byte (inside the CRC-covered region, after the
    // header + JSON metadata gap). The CRC check runs before the binary
    // weights are read, so this deterministically surfaces CrcMismatch.
    let last = namb.len() - 1;
    namb[last] ^= 0xFF;

    let path = temp_path("namb_crc_mismatch", "namb");
    write_temp(&path, &namb);

    let sys = SystemSnapshot::capture();
    let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
    std::fs::remove_file(&path).ok();

    let err = res.expect_err("NAMB CRC mismatch must be rejected");
    assert!(
        matches!(err, LoadError::NambCrc32Mismatch),
        "expected LoadError::NambCrc32Mismatch, got: {err:?}"
    );

    let snap = log_snapshot();
    assert!(
        buffer_has(
            &snap,
            "WARN",
            "[Loader] Invalid CRC rejected: field='crc32'"
        ),
        "LogBuffer must capture the structured CRC WARN"
    );
    assert!(
        buffer_has(&snap, "ERROR", "NambCrc32Mismatch"),
        "LogBuffer must capture the specific NAMB CRC error code"
    );
}

/// Metadata float `1e39` saturates to `+Inf` and is caught by the post-parse
/// `validate_metadata_floats` gate (F-14) — structured WARN + the specific
/// `InvalidMetadata` code.
#[test]
fn test_json_non_finite_metadata_rejected_with_structured_log() {
    let _guard = LOG_TESTS_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
    ensure_logger();

    let json = r#"{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": { "num_layers": 1, "hidden_size": 8, "layers": [] },
        "weights": [0.1],
        "sample_rate": 48000,
        "metadata": { "input_level_dbu": 1e39 }
    }"#;
    let path = temp_path("json_inf_metadata", "nam");
    write_temp(&path, json.as_bytes());

    let sys = SystemSnapshot::capture();
    let res = load_and_build_model(&path, &sys, false, LoadOptions::default());
    std::fs::remove_file(&path).ok();

    let err = res.expect_err("non-finite metadata must be rejected");
    assert!(
        matches!(err, LoadError::InvalidMetadata(_)),
        "expected LoadError::InvalidMetadata, got: {err:?}"
    );

    let snap = log_snapshot();
    assert!(
        buffer_has(
            &snap,
            "WARN",
            "[Loader] Invalid field rejected: field='input_level_dbu'"
        ),
        "LogBuffer must capture the structured metadata WARN"
    );
    assert!(
        buffer_has(&snap, "ERROR", "InvalidMetadata"),
        "LogBuffer must capture the InvalidMetadata code"
    );
}
