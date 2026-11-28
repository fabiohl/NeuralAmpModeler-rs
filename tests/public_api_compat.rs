// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Public API compatibility pins.
//!
//! These tests record decisions from Sprint S5 (T5.2): enum discriminants,
//! wire names and diagnostic codes that downstream hosts may persist,
//! serialize or match against must not drift silently. Additive changes are
//! allowed; changing an existing pin is a SemVer decision, not a refactor.

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::nam_json::WeightsLayout;
use neural_amp_modeler_rs::models::a2::activations::ActivationType;
use neural_amp_modeler_rs::models::a2::gating::GatingMode;

#[test]
fn weights_layout_discriminants_and_wire_names_are_stable() {
    assert_eq!(WeightsLayout::Original as u8, 0);
    assert_eq!(WeightsLayout::GateMajorLstm as u8, 1);
    assert_eq!(WeightsLayout::Interleaved4WaveNet as u8, 2);

    assert_eq!(
        serde_json::to_string(&WeightsLayout::Original).unwrap(),
        "\"Original\""
    );
    assert_eq!(
        serde_json::to_string(&WeightsLayout::GateMajorLstm).unwrap(),
        "\"GateMajorLstm\""
    );
    assert_eq!(
        serde_json::to_string(&WeightsLayout::Interleaved4WaveNet).unwrap(),
        "\"Interleaved4WaveNet\""
    );
}

#[test]
fn a2_enum_wire_names_are_stable() {
    assert_eq!(
        serde_json::to_string(&GatingMode::None).unwrap(),
        "\"none\""
    );
    assert_eq!(
        serde_json::to_string(&GatingMode::Gated).unwrap(),
        "\"gated\""
    );
    assert_eq!(
        serde_json::to_string(&GatingMode::Blended).unwrap(),
        "\"blended\""
    );

    assert_eq!(
        serde_json::to_string(&ActivationType::Tanh).unwrap(),
        r#"{"type":"Tanh"}"#
    );
    assert_eq!(
        serde_json::to_string(&ActivationType::HardTanh).unwrap(),
        r#"{"type":"HardTanh"}"#
    );
}

#[test]
fn diagnostic_codes_are_stable() {
    let pins = [
        (NamErrorCode::ModelTooLarge, "E1304", "MODEL_TOO_LARGE"),
        (
            NamErrorCode::NamJsonWeightNotFinite,
            "E1211",
            "NAM_JSON_WEIGHT_NOT_FINITE",
        ),
        (
            NamErrorCode::UnsupportedArchitecture,
            "E1300",
            "UNSUPPORTED_ARCHITECTURE",
        ),
        (
            NamErrorCode::HostFormatContractViolation,
            "E2304",
            "HOST_FORMAT_CONTRACT_VIOLATION",
        ),
    ];
    for (code, numeric, mnemonic) in pins {
        assert_eq!(code.code(), numeric);
        assert_eq!(code.mnemonic(), mnemonic);
    }
}

/// R1.8: the `.nam` loader's JSON classification preserves the v4.0.0 public
/// `LoadError` variants, their consumer-visible match arms and their `Display`
/// text. Each case is driven through the real byte-loading path, so the
/// loader mapping itself is pinned (not just the derive-generated messages).
#[test]
fn load_error_legacy_json_classification_is_stable() {
    use neural_amp_modeler_rs::SystemSnapshot;
    use neural_amp_modeler_rs::loader::{LoadError, LoadOptions, load_and_build_model_from_bytes};

    fn classify(err: &LoadError) -> &'static str {
        match err {
            LoadError::ModelTooLarge => "too_large",
            LoadError::NonFiniteWeights => "non_finite",
            LoadError::UnsupportedArchitecture(_) => "unsupported",
            _ => "other",
        }
    }

    fn load_err(json: &str) -> LoadError {
        let sys = SystemSnapshot::capture();
        load_and_build_model_from_bytes(json.as_bytes(), &sys, false, LoadOptions::default())
            .expect_err("fixture must be rejected")
    }

    // Training payload exceeds the JSON budget -> legacy `ModelTooLarge`.
    let over_budget = format!(
        r#"{{"version":"0.5.4","architecture":"LSTM","config":{{"layers":[]}},"weights":[],"metadata":{{"training":{{"p":"{}","q":"{}"}}}}}}"#,
        "x".repeat(600_000),
        "y".repeat(600_000)
    );
    let too_large = load_err(&over_budget);
    assert_eq!(classify(&too_large), "too_large");
    assert_eq!(
        too_large.to_string(),
        "Model file exceeds maximum allowed size"
    );
    assert_eq!(too_large.code(), NamErrorCode::ModelTooLarge);

    // Non-finite weight -> legacy `NonFiniteWeights`.
    let non_finite = load_err(
        r#"{"version":"0.5.0","architecture":"LSTM","config":{"num_layers":1,"input_size":1,"hidden_size":8},"weights":[1e39],"sample_rate":48000}"#,
    );
    assert_eq!(classify(&non_finite), "non_finite");
    assert_eq!(
        non_finite.to_string(),
        "Non-finite weights detected in model (NaN or Inf)"
    );
    assert_eq!(non_finite.code(), NamErrorCode::NamJsonWeightNotFinite);

    // Topology-limit rejection -> legacy `UnsupportedArchitecture(String)`.
    let unsupported = load_err(
        r#"{"version":"0.5.0","architecture":"LSTM","config":{"num_layers":1,"input_size":1,"hidden_size":513},"weights":[],"sample_rate":48000}"#,
    );
    assert_eq!(classify(&unsupported), "unsupported");
    assert!(
        unsupported
            .to_string()
            .contains("hidden_size=513 exceeds maximum"),
        "topology cause must survive in Display: {unsupported}"
    );
    assert_eq!(unsupported.code(), NamErrorCode::UnsupportedArchitecture);

    // Malformed version -> the v4.0.0 dedicated payload shape.
    let invalid_version = load_err(
        r#"{"version":"not-semver","architecture":"LSTM","config":{"layers":[]},"weights":[],"sample_rate":48000}"#,
    );
    match &invalid_version {
        LoadError::UnsupportedArchitecture(payload) => {
            assert_eq!(payload, "Invalid version format: not-semver");
        }
        other => panic!("expected UnsupportedArchitecture, got {other:?}"),
    }
    assert_eq!(
        invalid_version.to_string(),
        "Unsupported model architecture or topology: Invalid version format: not-semver"
    );
}
