// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Channel-geometry contract for [`NamModel`].
//!
//! Every model declares the geometry its multichannel entry points require
//! through [`NamModel::num_input_channels`] / [`NamModel::num_output_channels`].
//! The declaration must match the geometry NAMcore builds for the same
//! topology (`DSP::NumInputChannels` / `DSP::NumOutputChannels`), so the safe
//! helpers can reject mismatched host buffers before any pointer is
//! dereferenced.
//!
//! The table below fixes the expected `(input, output)` pair for the
//! multichannel families: Linear (all four channel topologies), an A2 cascade
//! with `head_size = 2`, a 1 -> 2 -> 1 Sequential chain and a Container.

use neural_amp_modeler_rs::loader::nam_json::{LinearImplementation, LinearTopology};
use neural_amp_modeler_rs::models::a2::WaveNetA2Cascade;
use neural_amp_modeler_rs::models::a2::WaveNetA2Dyn;
use neural_amp_modeler_rs::models::a2::activations::ActivationType;
use neural_amp_modeler_rs::models::a2::gating::GatingMode;
use neural_amp_modeler_rs::models::a2::params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
};
use neural_amp_modeler_rs::models::container::ContainerModel;
use neural_amp_modeler_rs::models::linear::LinearModel;
use neural_amp_modeler_rs::models::sequential::SequentialModel;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

/// Receptive field shared by every synthetic Linear kernel in this battery.
const RF: usize = 4;

/// Builds a Linear model with the given geometry and deterministic weights.
fn make_linear(in_ch: usize, out_ch: usize) -> LinearModel {
    let topo = LinearTopology {
        in_channels: in_ch,
        out_channels: out_ch,
        receptive_field: RF,
        has_bias: false,
        implementation: LinearImplementation::Direct,
    };
    let weights = vec![0.25f32; topo.num_kernels() * RF];
    LinearModel::new_with_topology(topo, weights, Vec::new()).expect("synthetic Linear model")
}

/// Builds a synthetic A2 dynamic array with the requested head size.
fn make_a2_dyn(head_size: usize) -> WaveNetA2Dyn {
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
        head_size,
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

/// Extracts the declared `(input, output)` geometry from any `NamModel`.
fn geometry<M: NamModel>(model: &M) -> (usize, usize) {
    (model.num_input_channels(), model.num_output_channels())
}

#[test]
fn linear_geometry_matches_declared_channels() {
    let cases = [(1, 1), (1, 2), (2, 1), (2, 2)];
    for (in_ch, out_ch) in cases {
        let model = make_linear(in_ch, out_ch);
        assert_eq!(
            geometry(&model),
            (in_ch, out_ch),
            "Linear {in_ch}->{out_ch} geometry mismatch"
        );
    }
}

#[test]
fn static_model_delegates_linear_geometry() {
    let model = StaticModel::Linear(Box::new(make_linear(2, 1)));
    assert_eq!(geometry(&model), (2, 1));
}

#[test]
fn a2_cascade_head2_reports_two_output_channels() {
    let cascade = WaveNetA2Cascade::try_new(vec![make_a2_dyn(2), make_a2_dyn(2)], None, 1)
        .expect("synthetic A2 cascade");
    let model = StaticModel::WavenetA2Cascade(Box::new(cascade));
    assert_eq!(geometry(&model), (1, 2));
}

#[test]
fn sequential_chain_reports_first_input_last_output() {
    let stages = vec![
        StaticModel::Linear(Box::new(make_linear(1, 2))),
        StaticModel::Linear(Box::new(make_linear(2, 1))),
    ];
    let chain = SequentialModel::new(stages, vec![None, None], Some(48_000.0))
        .expect("synthetic 1->2->1 chain");
    assert_eq!(geometry(&chain), (1, 1));
}

#[test]
fn container_delegates_active_submodel_geometry() {
    let submodel = Box::new(StaticModel::Linear(Box::new(make_linear(1, 2))));
    let container = ContainerModel::new(vec![(1.0, submodel)], 48_000).expect("container");
    assert_eq!(geometry(&container), (1, 2));
}
