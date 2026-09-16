// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! State compatibility contract regression tests for `ProcessingParams`.
//!
//! Verifies that snapshots of `ProcessingParams` serialized by previous versions
//! of this crate can be deserialized by newer versions without error or panic.
//! Missing fields are populated with their defaults via `#[serde(default)]`, and
//! unknown future fields (like `_schema_version`) are safely ignored.

// Contrato público: ProcessingParams é backward-compatible via serde(default).

use neural_amp_modeler_rs::ProcessingParams;

const FIXTURE_0_8_0: &str = include_str!("fixtures/state_compat/processing_params_0_8_0.json");
const FIXTURE_0_7_4: &str = include_str!("fixtures/state_compat/processing_params_0_7_4.json");
const FIXTURE_LEGACY_MINIMAL: &str =
    include_str!("fixtures/state_compat/processing_params_legacy_minimal.json");

#[test]
fn test_processing_params_roundtrip_v0_8_0() {
    // Contrato público: ProcessingParams é backward-compatible via serde(default).
    let deserialized: Result<ProcessingParams, _> = serde_json::from_str(FIXTURE_0_8_0);
    assert!(
        deserialized.is_ok(),
        "Failed to deserialize v0.8.0 fixture: {:?}",
        deserialized.err()
    );

    let params = deserialized.unwrap();
    let default_params = ProcessingParams::default();
    assert_eq!(
        params, default_params,
        "Deserialized v0.8.0 fixture does not match default parameters"
    );

    // Verify roundtrip serialization/deserialization
    let serialized = serde_json::to_string(&params).expect("Serialization failed");
    let roundtripped: ProcessingParams =
        serde_json::from_str(&serialized).expect("Roundtrip deserialization failed");
    assert_eq!(params, roundtripped);
}

#[test]
fn test_processing_params_roundtrip_v0_7_4() {
    // Contrato público: ProcessingParams é backward-compatible via serde(default).
    let deserialized: Result<ProcessingParams, _> = serde_json::from_str(FIXTURE_0_7_4);
    assert!(
        deserialized.is_ok(),
        "Failed to deserialize v0.7.4 fixture: {:?}",
        deserialized.err()
    );

    let params = deserialized.unwrap();
    let default_params = ProcessingParams::default();
    assert_eq!(
        params, default_params,
        "Deserialized v0.7.4 fixture does not match default parameters"
    );

    let serialized = serde_json::to_string(&params).expect("Serialization failed");
    let roundtripped: ProcessingParams =
        serde_json::from_str(&serialized).expect("Roundtrip deserialization failed");
    assert_eq!(params, roundtripped);
}

#[test]
fn test_processing_params_legacy_minimal() {
    // Contrato público: ProcessingParams é backward-compatible via serde(default).
    let deserialized: Result<ProcessingParams, _> = serde_json::from_str(FIXTURE_LEGACY_MINIMAL);
    assert!(
        deserialized.is_ok(),
        "Failed to deserialize legacy minimal fixture: {:?}",
        deserialized.err()
    );

    let params = deserialized.unwrap();
    let default_params = ProcessingParams::default();
    assert_eq!(
        params, default_params,
        "Omitted fields should fallback to Default values via serde(default)"
    );
}

#[test]
fn test_processing_params_ignores_unknown_fields() {
    // Unknown or extra fields in future state payloads must not cause deserialization failures.
    let json_with_future_fields = r#"{
        "_schema_version": "9.9.9",
        "input_gain_db": 3.5,
        "output_gain_db": -1.0,
        "future_field_string": "unknown_value",
        "future_field_numeric": 42.0,
        "future_field_object": { "nested": true }
    }"#;

    let deserialized: Result<ProcessingParams, _> = serde_json::from_str(json_with_future_fields);
    assert!(
        deserialized.is_ok(),
        "Deserialization with unknown fields must succeed: {:?}",
        deserialized.err()
    );

    let params = deserialized.unwrap();
    assert_eq!(params.input_gain_db, 3.5);
    assert_eq!(params.output_gain_db, -1.0);
    // All omitted fields must take their default values
    assert_eq!(params.gate_threshold_db, -70.0);
    assert!(!params.bypass);
}
