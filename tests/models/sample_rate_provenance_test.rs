// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Sample-rate provenance across nested model graphs (DEC-01 / DEC-01b).
//!
//! Pins the C++ NAMcore rule that a composite node inherits the *effective*
//! declared rate of its children (never materializing the 48 kHz default mid
//! recursion), that a `SlimmableContainer` without its own rate adopts its
//! first known submodel rate, and that `condition_dsp` rates are compared
//! strictly (including the unknown/known mismatch).

use neural_amp_modeler_rs::common::diagnostics::{NamErrorCode, SystemSnapshot};
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model_from_bytes};
use neural_amp_modeler_rs::models::StaticModel;

use super::common;
use common::io_helpers::model_path;

fn sys() -> SystemSnapshot {
    SystemSnapshot::capture()
}

fn linear_json(sample_rate: Option<f32>) -> String {
    let sr = sample_rate
        .map(|s| format!("\"sample_rate\": {s},"))
        .unwrap_or_default();
    format!(
        r#"{{"version":"0.7.0","architecture":"Linear",{sr}"config":{{"receptive_field":1,"bias":false}},"weights":[1.0]}}"#
    )
}

fn sequential_json(sample_rate: Option<f32>, children: &[String]) -> String {
    let sr = sample_rate
        .map(|s| format!("\"sample_rate\": {s},"))
        .unwrap_or_default();
    let models = children.join(",");
    format!(
        r#"{{"version":"0.7.0","architecture":"Sequential",{sr}"config":{{"models":[{models}]}},"weights":[]}}"#
    )
}

fn container_json(sample_rate: Option<f32>, submodel: &str) -> String {
    let sr = sample_rate
        .map(|s| format!("\"sample_rate\": {s},"))
        .unwrap_or_default();
    format!(
        r#"{{"version":"0.7.0","architecture":"SlimmableContainer",{sr}"config":{{"submodels":[{{"max_value":1.0,"model":{submodel}}}]}},"weights":[]}}"#
    )
}

fn load(json: &str) -> Result<neural_amp_modeler_rs::loader::LoadedModelPair, String> {
    load_and_build_model_from_bytes(json.as_bytes(), &sys(), false, LoadOptions::default())
        .map_err(|e| e.to_string())
}

/// `pair.sample_rate` must equal the built model's own expected rate.
fn assert_rate_invariant(pair: &neural_amp_modeler_rs::loader::LoadedModelPair) {
    let model = pair.model_l.as_ref().expect("model present");
    let model_rate = match model.as_ref() {
        StaticModel::Sequential(m) => m.expected_sample_rate(),
        StaticModel::Container(m) => m.sample_rate() as f32,
        _ => pair.sample_rate as f32,
    };
    assert_eq!(
        pair.sample_rate as f32, model_rate,
        "pair.sample_rate must equal model.expected_sample_rate()"
    );
}

#[test]
fn nested_sequential_inherits_grandchild_rate() {
    // Root unknown; child Sequential unknown with grandchild 44.1k; sibling 44.1k.
    let nested = sequential_json(None, &[linear_json(Some(44100.0))]);
    let root = sequential_json(None, &[nested, linear_json(Some(44100.0))]);
    let pair = load(&root).expect("must resolve to 44100");
    assert_eq!(pair.sample_rate, 44100);
    assert_rate_invariant(&pair);
}

#[test]
fn nested_sequential_conflicting_rates_rejected() {
    // Child Sequential carries 44.1k, sibling declares 48k → conflict.
    let nested = sequential_json(None, &[linear_json(Some(44100.0))]);
    let root = sequential_json(None, &[nested, linear_json(Some(48000.0))]);
    assert!(
        load(&root).is_err(),
        "conflicting known rates must be rejected"
    );
}

#[test]
fn all_unknown_falls_back_to_default() {
    let root = sequential_json(None, &[linear_json(None), linear_json(None)]);
    let pair = load(&root).expect("all-unknown must be accepted");
    assert_eq!(pair.sample_rate, 48000, "DEC-01 fallback is 48000");
    assert_rate_invariant(&pair);
}

#[test]
fn container_unknown_adopts_submodel_rate() {
    // F-16: an unknown-rate container with a 44.1k submodel must be accepted.
    let root = container_json(None, &linear_json(Some(44100.0)));
    let pair = load(&root).expect("unknown container + 44.1k submodel must be accepted");
    assert_eq!(pair.sample_rate, 44100);
    assert_rate_invariant(&pair);
}

#[test]
fn container_known_rate_conflicting_with_submodel_rejected() {
    let root = container_json(Some(48000.0), &linear_json(Some(44100.0)));
    assert!(
        load(&root).is_err(),
        "container 48k + submodel 44.1k must be rejected"
    );
}

#[test]
fn nested_sequential_inherits_grandchild_rate_through_container() {
    // Container(unknown) → Sequential(unknown) → Linear(44.1k): the container
    // must adopt the rate resolved deep inside the nested Sequential, never the
    // 48 kHz default materialized mid-recursion (DEC-01b).
    let seq = sequential_json(None, &[linear_json(Some(44100.0))]);
    let root = container_json(None, &seq);
    let pair = load(&root).expect("container must inherit the nested Sequential rate");
    assert_eq!(pair.sample_rate, 44100);
    assert_rate_invariant(&pair);
}

#[test]
fn condition_dsp_a2_mismatch_rejected() {
    let path = model_path("wavenet_a2_lite.nam");
    assert!(path.exists(), "wavenet_a2_lite.nam is a required fixture");
    let json = std::fs::read_to_string(&path).expect("read wavenet_a2_lite.nam");

    // The A2 model is 48 kHz; a condition_dsp at the same rate is accepted.
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    value["config"]["condition_dsp"] = serde_json::json!({
        "version": "0.5.4",
        "architecture": "Linear",
        "config": {"receptive_field": 1, "bias": false},
        "weights": [0.5],
        "sample_rate": 48000.0
    });
    load(&value.to_string()).expect("A2 condition_dsp at matching 48k must build");

    // A concrete mismatch (44.1k vs 48k) must be rejected.
    value["config"]["condition_dsp"]["sample_rate"] = serde_json::json!(44100.0);
    let err = load(&value.to_string()).expect_err("A2 condition_dsp 44.1k must be rejected");
    assert!(
        err.contains("E1315")
            || err.to_lowercase().contains("sample rate")
            || err.contains("SampleRate"),
        "A2 condition_dsp mismatch must report a sample-rate error, got: {err}"
    );
}

#[test]
fn condition_dsp_unknown_rate_rejected_against_known_wavenet() {
    let path = model_path("wavenet_condition_dsp.nam");
    if !path.exists() {
        eprintln!("SKIP: wavenet_condition_dsp.nam fixture not found.");
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read fixture");

    // Sanity: the unmodified fixture (48000 / 48000) is accepted.
    load(&json).expect("fixture with matching rates must build");

    // Strip the condition_dsp declared rate → unknown must never match a known
    // WaveNet rate (C++ strict comparison includes the unknown sentinel).
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    value["config"]["condition_dsp"]
        .as_object_mut()
        .expect("condition_dsp object")
        .remove("sample_rate");
    assert!(
        load(&value.to_string()).is_err(),
        "condition_dsp without a rate must be rejected by the strict comparison"
    );

    // A concrete mismatch (44.1k vs 48k) is rejected too.
    let mut value2: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    value2["config"]["condition_dsp"]["sample_rate"] = serde_json::json!(44100.0);
    assert!(
        load(&value2.to_string()).is_err(),
        "condition_dsp 44.1k vs WaveNet 48k must be rejected"
    );
}

#[test]
fn condition_dsp_mismatch_carries_e1315() {
    let path = model_path("wavenet_condition_dsp.nam");
    if !path.exists() {
        eprintln!("SKIP: wavenet_condition_dsp.nam fixture not found.");
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read fixture");
    let mut value: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    value["config"]["condition_dsp"]["sample_rate"] = serde_json::json!(44100.0);

    // `parse_nam_json` + `build_model` keeps the typed code downcastable.
    let data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&value.to_string())
        .expect("parse mutated fixture");
    let err = match neural_amp_modeler_rs::loader::dispatcher::build_model(&data) {
        Ok(_) => panic!("expected E1315"),
        Err(e) => e,
    };
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonSampleRateMismatch),
        "expected E1315, got: {err}"
    );
}
