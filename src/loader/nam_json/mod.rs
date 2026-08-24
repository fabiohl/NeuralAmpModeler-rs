// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Parser for the .nam format (JSON)
//!
//! Loads tensors and metadata outside the RT path.

pub mod activation_parser;
pub mod data;
pub mod error;
pub mod model;
pub mod parse;
pub mod topology;
pub(crate) mod validation;

// Re-export validation constants for integration tests (F2 — adversarial dimension fuzzing)
pub use validation::{
    MAX_A2_DYN_BOTTLENECK, MAX_A2_DYN_CHANNELS, MAX_CONVNET_CHANNELS, MAX_CONVNET_KERNEL_SIZE,
    MAX_DILATION, MAX_DILATIONS_PER_ARRAY, MAX_HEAD_SIZE, MAX_HIDDEN_SIZE, MAX_KERNEL_SIZE,
    MAX_LAYERS, MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_LAYERS, MAX_RECEPTIVE_FIELD, MAX_TOTAL_STATE_FRAMES,
    MAX_WAVENET_ARRAYS, MAX_WAVENET_FREE_CHANNELS,
};

pub use activation_parser::{
    LayerActivationConfig, parse_activations_from_json, parse_gating_modes_from_json,
    parse_layer_activations, parse_secondary_activations_from_json,
};
pub use data::{
    JsonError, LinearImplementation, NamConfig, NamDate, NamLayerConfig, NamMetadata, NamModelData,
    SlimmableConfig, SlimmableKwargs, WeightsLayout,
};
pub use parse::parse_nam_json;
#[cfg(test)]
pub(crate) use topology::parse_semver;
pub use topology::{
    A2TopologyResult, A2TopologyVectors, ConvNetFormat, ConvNetTopology, FreeWavenetGeometry,
    NamWavenetTopology, WavenetTopologyResult, get_convnet_topology, get_linear_topology,
    get_lstm_topology, get_wavenet_topology, is_a2_shape, validate_a2_layer_topology,
};

#[cfg(test)]
#[path = "../nam_json_test.rs"]
mod nam_json_test;

#[cfg(test)]
#[path = "validation_test.rs"]
mod validation_test;
