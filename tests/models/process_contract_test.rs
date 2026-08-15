// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! S3.T3 — `NamModel::process` length-contract battery.
//!
//! Every architecture family must survive asymmetric caller buffers
//! (`output.len() = input.len() - 1`) without panicking, in debug and in
//! `--release`. The engine clamps to `n = input.len().min(output.len())`
//! (or `output.len() / out_channels` for multi-channel families) and never
//! indexes past the output.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::a2::WaveNetA2Dyn;
use neural_amp_modeler_rs::models::a2::activations::ActivationType;
use neural_amp_modeler_rs::models::a2::gating::GatingMode;
use neural_amp_modeler_rs::models::a2::params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
};
use neural_amp_modeler_rs::models::container::ContainerModel;
use neural_amp_modeler_rs::models::slimmable::SlimmableModel;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

use super::common;
use common::model_builders::{build_soak_wavenet, build_synth_a2};

fn load_named_model(filename: &str) -> Option<Box<StaticModel>> {
    let path = neural_amp_modeler_rs::testing::fixtures::model_path(filename);
    if !path.exists() {
        eprintln!("SKIP: {filename} fixture not found.");
        return None;
    }
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("Failed to read {filename}: {e}");
    });
    let data = parse_nam_json(&json).unwrap_or_else(|e| {
        panic!("Failed to parse {filename}: {e}");
    });
    Some(build_model(&data).unwrap_or_else(|e| {
        panic!("Failed to build {filename}: {e}");
    }))
}

fn make_synth_a2_dyn() -> WaveNetA2Dyn {
    let activations = vec![
        ActivationType::LeakyReLU {
            negative_slope: A2_LEAKY_SLOPE,
        };
        A2_NUM_LAYERS
    ];
    let gating = vec![GatingMode::None; A2_NUM_LAYERS];
    let secondary = vec![None; A2_NUM_LAYERS];
    WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        A2_HEAD_KERNEL_SIZE,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        activations,
        gating,
        secondary,
    )
    .expect("synthetic WaveNetA2Dyn")
}

/// Runs one asymmetric-buffer case through the full `StaticModel::process`
/// dispatch path. Returns the number of samples actually written.
fn run_short_output_case(model: &mut StaticModel, label: &str) {
    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output = vec![0.0f32; input.len() - 1];

    // Must not panic: the contract clamps to the shorter buffer.
    NamModel::process(model, &input, &mut output);

    assert!(
        output.iter().all(|s| s.is_finite()),
        "{label}: non-finite output with output.len() = input.len() - 1"
    );
}

#[test]
fn process_short_output_lstm_no_panic() {
    let mut model = StaticModel::Lstm1x8(Box::default());
    run_short_output_case(&mut model, "LSTM 1x8");

    let mut model2 = StaticModel::Lstm2x8(Box::default());
    run_short_output_case(&mut model2, "LSTM 2x8");

    let mut model_dyn = StaticModel::LstmDyn(Box::new(
        neural_amp_modeler_rs::models::lstm::LstmModelDyn::new(2, 8)
            .expect("LstmModelDyn allocation"),
    ));
    run_short_output_case(&mut model_dyn, "LSTM Dyn");
}

#[test]
fn process_short_output_wavenet_no_panic() {
    let mut model = StaticModel::WavenetStandard(Box::new(build_soak_wavenet()));
    run_short_output_case(&mut model, "WaveNet Standard");
}

#[test]
fn process_short_output_convnet_no_panic() {
    let Some(mut model) = load_named_model("convnet_test.nam") else {
        return;
    };
    run_short_output_case(&mut model, "ConvNet");
}

#[test]
fn process_short_output_a2_no_panic() {
    let mut model = StaticModel::WavenetA2Lite(Box::new(build_synth_a2::<3>(0.01)));
    run_short_output_case(&mut model, "A2 Lite");
}

#[test]
fn process_short_output_a2_dyn_no_panic() {
    let mut model = StaticModel::WavenetA2Dyn(Box::new(make_synth_a2_dyn()));
    run_short_output_case(&mut model, "A2 Dyn");
}

#[test]
fn process_short_output_a2_cascade_no_panic() {
    // A synthetic `WaveNetA2Dyn` is not a valid cascade array: its
    // `condition_dsp_output` is empty and `cascade_layer_loop` indexes it.
    // Only a loader-built cascade has the condition buffers sized.
    for name in ["wavenet_condition_dsp.nam", "a2_example.nam"] {
        if let Some(mut model) = load_named_model(name)
            && matches!(model.as_ref(), StaticModel::WavenetA2Cascade(_))
        {
            run_short_output_case(&mut model, "A2 Cascade");
            return;
        }
    }
    eprintln!("SKIP: no WavenetA2Cascade fixture available for process-contract.");
}

#[test]
fn process_short_output_wavenet_dyn_no_panic() {
    let Some(mut model) = load_named_model("wavenet_dyn_free.nam") else {
        return;
    };
    assert!(
        matches!(model.as_ref(), StaticModel::WavenetDyn(_)),
        "wavenet_dyn_free.nam must load as WavenetDyn"
    );
    run_short_output_case(&mut model, "WaveNet Dyn");
}

#[test]
fn process_short_output_container_no_panic() {
    let make_lstm = || Box::new(StaticModel::Lstm1x8(Box::default()));
    let container = ContainerModel::new(vec![(0.5, make_lstm()), (1.0, make_lstm())], 48000)
        .expect("ContainerModel creation");
    let mut model = StaticModel::Container(Box::new(container));

    // Plain path.
    run_short_output_case(&mut model, "Container");

    // Crossfade path with asymmetric buffers: pending tier switch in progress.
    if let StaticModel::Container(ref mut c) = model {
        c.set_slimmable_size(0.2, None);
        assert!(c.is_crossfading());
    }
    let input: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut output = vec![0.0f32; input.len() - 1];
    NamModel::process(&mut model, &input, &mut output);
    assert!(
        output.iter().all(|s| s.is_finite()),
        "Container crossfade: non-finite output with asymmetric buffers"
    );
}
