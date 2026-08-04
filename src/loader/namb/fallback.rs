// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fallback model configuration for legacy `.namb` files.

use super::super::nam_json::{NamConfig, NamLayerConfig, NamModelData, WeightsLayout};

/// Creates a "fallback" dataset.
/// Useful for old .namb files that do not describe their own structure.
pub fn make_fallback_model_data() -> NamModelData {
    NamModelData {
        version: None,
        architecture: "WaveNet".to_string(), // legacy .namb files are always WaveNet Standard
        config: make_standard_wavenet_config(),
        weights: Vec::new(),
        sample_rate: None,
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// Defines the standard "template" for the WaveNet algorithm.
/// It's like defining the number of neurons and connections of a standard digital brain.
pub fn make_standard_wavenet_config() -> NamConfig {
    // Dilations: defines the "reach" of the algorithm's memory (essential for capturing timbre).
    let std_dilations = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    // First processing layer.
    let l0 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),   // "Width" of the internal processing.
        kernel_size: Some(3), // Number of neighboring samples analyzed at each step.
        dilations: Some(std_dilations.clone()),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(false),
        ..Default::default()
    };

    // Second layer (usually identical to the first in Standard models).
    let l1 = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: Some(8),
        channels: Some(16),
        kernel_size: Some(3),
        dilations: Some(std_dilations),
        activation: Some("Tanh".to_string()),
        gated: Some(false),
        head_bias: Some(true),
        ..Default::default()
    };

    NamConfig {
        layers: vec![l0, l1],
        head: Some(serde_json::Value::Null),
        head_scale: Some(0.02), // Final volume adjustment to ensure consistency.
        num_layers: None,
        hidden_size: None,
        receptive_field: None,
        bias: None,
        submodels: None,
        ..Default::default()
    }
}
