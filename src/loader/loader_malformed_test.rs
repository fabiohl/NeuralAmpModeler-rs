// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Negative regression tests for hard-limits validation in the model parser and loader.
//!
//! Feeds the parser and loader with forged, exaggerated, or malformed
//! inputs and asserts that the library rejects them gracefully (Soft Reject)
//! within 50 ms, never panicking.

use std::time::Instant;

/// Helper: builds a minimal valid WaveNet JSON, then replaces `layers` with the given array.
fn wavenet_json_with_layers(layers_json: &str) -> String {
    format!(
        r#"{{
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": {{
            "layers": {},
            "head": null,
            "head_scale": 0.02
        }},
        "weights": [0.1],
        "sample_rate": 48000
    }}"#,
        layers_json
    )
}

/// Helper: builds a minimal valid LSTM JSON, then replaces `num_layers`/`hidden_size`.
fn lstm_json(hidden_size: usize) -> String {
    format!(
        r#"{{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {{
            "num_layers": 2,
            "hidden_size": {},
            "layers": []
        }},
        "weights": [0.1, 0.2]
    }}"#,
        hidden_size
    )
}

// ── JSON Parser Negative Tests ──

/// 9 layers — one above MAX_LAYERS (8).
#[test]
fn test_reject_too_many_layers() {
    let layers: Vec<String> = (0..9)
        .map(|_| {
            r#"{"input_size":1,"condition_size":1,"head_size":4,"channels":8,"kernel_size":3,"dilations":[1],"activation":"Tanh","gated":false,"head_bias":false}"#.to_string()
        })
        .collect();
    let layers_json = format!("[{}]", layers.join(","));
    let json = wavenet_json_with_layers(&layers_json);
    let t0 = Instant::now();
    let result = crate::loader::nam_json::parse_nam_json(&json);
    let elapsed = t0.elapsed();
    assert!(result.is_err(), "9 layers must be rejected");
    assert!(
        elapsed.as_micros() < 50_000,
        "rejection took {} µs, expected < 50 ms",
        elapsed.as_micros()
    );
}

/// Exactly at MAX_LAYERS (8) — must succeed.
#[test]
fn test_accept_max_layers() {
    let layers: Vec<String> = (0..8)
        .map(|_| {
            r#"{"input_size":1,"condition_size":1,"head_size":4,"channels":8,"kernel_size":3,"dilations":[1],"activation":"Tanh","gated":false,"head_bias":false}"#.to_string()
        })
        .collect();
    let layers_json = format!("[{}]", layers.join(","));
    let json = wavenet_json_with_layers(&layers_json);
    let result = crate::loader::nam_json::parse_nam_json(&json);
    assert!(result.is_ok(), "8 layers must be accepted");
}

/// hidden_size = 1024 (double MAX_HIDDEN_SIZE = 512) — must reject.
#[test]
fn test_reject_large_hidden_size() {
    let json = lstm_json(1024);
    let t0 = Instant::now();
    let result = crate::loader::nam_json::parse_nam_json(&json);
    let elapsed = t0.elapsed();
    assert!(result.is_err(), "hidden_size=1024 must be rejected");
    assert!(
        elapsed.as_micros() < 50_000,
        "rejection took {} µs, expected < 50 ms",
        elapsed.as_micros()
    );
}

/// hidden_size = 513 (one above MAX_HIDDEN_SIZE) — must reject.
#[test]
fn test_reject_hidden_size_just_above_limit() {
    let json = lstm_json(513);
    let result = crate::loader::nam_json::parse_nam_json(&json);
    assert!(result.is_err(), "hidden_size=513 must be rejected");
}

/// hidden_size = 512 (at MAX_HIDDEN_SIZE) — must succeed.
#[test]
fn test_accept_hidden_size_at_limit() {
    let json = lstm_json(512);
    let result = crate::loader::nam_json::parse_nam_json(&json);
    assert!(result.is_ok(), "hidden_size=512 must be accepted");
}

/// Both limits breached — too many layers AND hidden_size > MAX_HIDDEN_SIZE.
/// The layer check happens first and short-circuits.
#[test]
fn test_reject_both_limits_breached() {
    let layers: Vec<String> = (0..12)
        .map(|_| {
            r#"{"input_size":1,"condition_size":1,"head_size":4,"channels":8,"kernel_size":3,"dilations":[1],"activation":"Tanh","gated":false,"head_bias":false}"#.to_string()
        })
        .collect();
    let layers_json = format!("[{}]", layers.join(","));
    let json = format!(
        r#"{{
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {{
            "num_layers": 12,
            "hidden_size": 9999,
            "layers": {}
        }},
        "weights": [0.1]
    }}"#,
        layers_json
    );
    let t0 = Instant::now();
    let result = crate::loader::nam_json::parse_nam_json(&json);
    let elapsed = t0.elapsed();
    assert!(result.is_err(), "combined limits breach must be rejected");
    assert!(
        elapsed.as_micros() < 50_000,
        "rejection took {} µs, expected < 50 ms",
        elapsed.as_micros()
    );
}

/// 0 layers — must succeed (valid degenerate case).
#[test]
fn test_accept_zero_layers() {
    let json = wavenet_json_with_layers("[]");
    let result = crate::loader::nam_json::parse_nam_json(&json);
    assert!(result.is_ok(), "0 layers must be accepted");
}

// ── IR Loader Negative Tests ──

/// WAV file with sample count exceeding MAX_IR_LENGTH (192000).
/// Generates a PCM16 mono WAV with 200000 samples programmatically.
#[test]
fn test_reject_ir_too_many_samples() {
    let path = std::path::Path::new("/tmp/nam_rs_test_oversized_ir.wav");

    let num_samples: u32 = 200_000;
    let data_size = num_samples * 2;
    let file_size: u32 = 36 + data_size;
    let sample_rate: u32 = 48_000;
    let byte_rate = sample_rate * 2;
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    let silence = vec![0u8; (num_samples * 2) as usize];
    buf.extend_from_slice(&silence);

    std::fs::write(path, &buf).ok();

    let t0 = Instant::now();
    let result = crate::dsp::cabsim::loader::CabSimIr::load(path, 0, false);
    let elapsed = t0.elapsed();
    std::fs::remove_file(path).ok();

    assert!(result.is_err(), "200k-sample IR must be rejected");
    assert!(
        elapsed.as_micros() < 50_000,
        "IR rejection took {} µs, expected < 50 ms",
        elapsed.as_micros()
    );
}

/// WAV with exactly MAX_IR_LENGTH samples — must succeed.
#[test]
fn test_accept_ir_at_limit() {
    let path = std::path::Path::new("/tmp/nam_rs_test_ir_at_limit.wav");

    let num_samples: u32 = 192_000;
    let data_size = num_samples * 2;
    let file_size: u32 = 36 + data_size;
    let sample_rate: u32 = 48_000;
    let byte_rate = sample_rate * 2;
    let block_align: u16 = 2;
    let bits_per_sample: u16 = 16;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    let silence = vec![0u8; (num_samples * 2) as usize];
    buf.extend_from_slice(&silence);

    std::fs::write(path, &buf).ok();

    let result = crate::dsp::cabsim::loader::CabSimIr::load(path, 0, false);
    std::fs::remove_file(path).ok();

    assert!(
        result.is_ok(),
        "192k-sample IR must be accepted (at MAX_IR_LENGTH)"
    );
}

/// Multiple quick rejections — ensures the rejection path is stable
/// under repeated abuse.
#[test]
fn test_repeated_rejections_are_stable() {
    let layers: Vec<String> = (0..20)
        .map(|_| {
            r#"{"input_size":1,"condition_size":1,"head_size":4,"channels":8,"kernel_size":3,"dilations":[1],"activation":"Tanh","gated":false,"head_bias":false}"#.to_string()
        })
        .collect();
    let layers_json = format!("[{}]", layers.join(","));
    let json = wavenet_json_with_layers(&layers_json);

    for _ in 0..50 {
        let result = crate::loader::nam_json::parse_nam_json(&json);
        assert!(result.is_err(), "must reject on every iteration");
    }
}
