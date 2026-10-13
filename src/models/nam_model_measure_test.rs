// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Tests for [`NamModel::measure_cost_ns_per_sample`]: the estimate must be
//! finite and positive, and the measurement must restore the model to its
//! freshly-built, stabilized condition (bit-exact reference-block output
//! before vs. after measurement).

use super::{NamModel, StaticModel};
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::models::linear::LinearModel;
use crate::models::lstm::LstmModel1;

/// Reference block length used for the state-neutrality oracle.
const FRAMES: usize = 256;

/// Restores the freshly-built, stabilized condition through the engine's split
/// pass (the same restoration `measure_cost_ns_per_sample` performs).
fn canonicalize(model: &mut StaticModel) {
    model.prewarm_reset();
    while !model.prewarm_complete() {
        if model.prewarm_step(usize::MAX) == 0 {
            break;
        }
    }
}

/// Runs a deterministic reference block and returns the flattened output.
fn reference_block(model: &mut StaticModel) -> Vec<f32> {
    let input: Vec<f32> = (0..FRAMES)
        .map(|i| ((i as f32) * 0.013).sin() * 0.3)
        .collect();
    let ni = model.num_input_channels();
    let no = model.num_output_channels();
    if ni == 1 && no == 1 {
        let mut output = vec![0.0f32; FRAMES];
        model.process(&input, &mut output);
        output
    } else {
        let inputs: Vec<&[f32]> = (0..ni).map(|_| input.as_slice()).collect();
        let mut outputs: Vec<Vec<f32>> = (0..no).map(|_| vec![0.0f32; FRAMES]).collect();
        let mut out_refs: Vec<&mut [f32]> = outputs.iter_mut().map(|v| v.as_mut_slice()).collect();
        model
            .process_multichannel(&inputs, &mut out_refs)
            .expect("reference geometry matches declared channels");
        outputs.into_iter().flatten().collect()
    }
}

fn measure_restores_state(model: &mut StaticModel) -> f32 {
    canonicalize(model);
    let before = reference_block(model);
    let cost = model.measure_cost_ns_per_sample(FRAMES);
    assert!(cost.is_finite(), "measured cost must be finite");
    let after = reference_block(model);
    assert_eq!(
        before, after,
        "measurement must be state-neutral: identical reference output before/after"
    );
    cost
}

fn linear_mono() -> StaticModel {
    StaticModel::Linear(Box::new(
        LinearModel::new(
            vec![0.25, -0.5, 0.75, 0.1],
            0.05,
            LinearImplementation::Direct,
        )
        .expect("mono linear model"),
    ))
}

/// 2-in / 2-out Linear model: exercises the multichannel measurement branch.
fn linear_stereo() -> StaticModel {
    let topo = LinearTopology {
        in_channels: 2,
        out_channels: 2,
        receptive_field: 2,
        has_bias: false,
        implementation: LinearImplementation::Direct,
    };
    // `fan = max(in, out) = 2` kernels of `receptive_field = 2` taps each.
    let weights = vec![0.7, 0.0, 0.0, 0.7];
    let biases = vec![0.0, 0.0];
    StaticModel::Linear(Box::new(
        LinearModel::new_with_topology(topo, weights, biases).expect("stereo linear model"),
    ))
}

/// A recurrent LSTM exercises the full clear-state + prewarm restoration path.
// LstmModel1::new() is constructor-based (recurrent state zeroed at
// construction); the box_default lint still fires on the explicit pattern.
#[expect(
    clippy::box_default,
    reason = "constructor-based LSTM, not field Default"
)]
fn lstm_mono() -> StaticModel {
    StaticModel::Lstm1x3(Box::new(LstmModel1::<3, 4, 12>::new()))
}

#[test]
fn measure_cost_is_positive_and_state_neutral_linear() {
    let mut model = linear_mono();
    let cost = measure_restores_state(&mut model);
    assert!(cost > 0.0, "mono linear cost must be positive, got {cost}");
}

#[test]
fn measure_cost_is_positive_and_state_neutral_lstm() {
    let mut model = lstm_mono();
    let cost = measure_restores_state(&mut model);
    assert!(cost > 0.0, "LSTM cost must be positive, got {cost}");
}

#[test]
fn measure_cost_is_positive_and_state_neutral_stereo() {
    let mut model = linear_stereo();
    let cost = measure_restores_state(&mut model);
    assert!(
        cost > 0.0,
        "stereo linear cost must be positive, got {cost}"
    );
}

/// The min-of-k estimate is stable across repeated calibrations on the same
/// (freshly restored) model; a generous factor absorbs dev-host scheduler
/// noise without letting a runaway error slip through.
#[test]
fn measure_cost_is_reproducible() {
    let mut model = linear_mono();
    let first = model.measure_cost_ns_per_sample(FRAMES);
    assert!(first > 0.0);
    for _ in 0..3 {
        let again = model.measure_cost_ns_per_sample(FRAMES);
        assert!(again > 0.0);
        assert!(
            again <= first * 16.0,
            "repeated calibration drifted too far: first={first} again={again}"
        );
    }
}
