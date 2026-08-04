// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Smoke tests for B.1.2 fixtures — verifies ConvNet, WaveNetDyn, and LstmDyn
//  models parse and load correctly as their expected StaticModel variants.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::NamModelData;
use neural_amp_modeler_rs::models::StaticModel;
use std::path::Path;

fn fixture_path(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/models")
        .join(name)
        .to_string_lossy()
        .into_owned()
}

fn load(name: &str) -> anyhow::Result<Box<StaticModel>> {
    let content = std::fs::read_to_string(fixture_path(name))?;
    let data: NamModelData = serde_json::from_str(&content)?;
    build_model(&data)
}

#[test]
fn convnet_test_fixture_loads() {
    let model = load("convnet_test.nam").expect("Failed to load ConvNet");
    assert!(
        matches!(model.as_ref(), StaticModel::ConvNet(_)),
        "Expected ConvNet model"
    );
    assert!(model.receptive_field() > 0, "ConvNet RF should be > 0");
}

#[test]
fn wavenet_dyn_fixture_loads() {
    let model = load("wavenet_dyn_free.nam").expect("Failed to load WaveNetDyn");
    assert!(
        matches!(model.as_ref(), StaticModel::WavenetDyn(_)),
        "Expected WavenetDyn model"
    );
    assert!(model.receptive_field() > 0);
}

#[test]
fn lstm_dyn_fixture_loads() {
    let model = load("lstm_dyn_test.nam").expect("Failed to load LSTM-Dyn");
    assert!(
        matches!(model.as_ref(), StaticModel::LstmDyn(_)),
        "Expected LstmDyn model"
    );
    assert!(model.num_output_channels() > 0);
}
