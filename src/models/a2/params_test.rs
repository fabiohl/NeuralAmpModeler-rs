// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_layer_params_a2_construction() {
    let _params = LayerParamsA2 {
        condition_size: 1,
        channels: 16,
        bottleneck: 16,
        kernel_size: 3,
        dilation: 1,
        activation: ActivationType::Tanh,
        gating_mode: GatingMode::None,
        groups_input: 1,
        groups_input_mixin: 1,
        layer1x1: Layer1x1Params {
            active: false,
            groups: 1,
        },
        head1x1: Head1x1Params {
            active: false,
            out_channels: 16,
            groups: 1,
        },
        secondary_activation: ActivationType::Sigmoid,
        conv_pre_film: FiLMConfig::default(),
        conv_post_film: FiLMConfig::default(),
        input_mixin_pre_film: FiLMConfig::default(),
        input_mixin_post_film: FiLMConfig::default(),
        activation_pre_film: FiLMConfig::default(),
        activation_post_film: FiLMConfig::default(),
        layer1x1_post_film: FiLMConfig::default(),
        head1x1_post_film: FiLMConfig::default(),
    };
}

#[test]
fn test_layer_array_params_a2_construction() {
    let _params = LayerArrayParamsA2 {
        input_size: 1,
        condition_size: 1,
        head_size: 1,
        head_kernel_size: 1,
        channels: 16,
        bottleneck: 16,
        kernel_sizes: vec![3, 3],
        dilations: vec![1, 2],
        activations: vec![ActivationType::Tanh, ActivationType::Tanh],
        gating_modes: vec![GatingMode::None, GatingMode::None],
        head_bias: true,
        groups_input: 1,
        groups_input_mixin: 1,
        layer1x1: Layer1x1Params {
            active: false,
            groups: 1,
        },
        head1x1: Head1x1Params {
            active: false,
            out_channels: 16,
            groups: 1,
        },
        secondary_activations: vec![ActivationType::Sigmoid, ActivationType::Sigmoid],
        conv_pre_film: FiLMConfig::default(),
        conv_post_film: FiLMConfig::default(),
        input_mixin_pre_film: FiLMConfig::default(),
        input_mixin_post_film: FiLMConfig::default(),
        activation_pre_film: FiLMConfig::default(),
        activation_post_film: FiLMConfig::default(),
        layer1x1_post_film: FiLMConfig::default(),
        head1x1_post_film: FiLMConfig::default(),
    };
}

#[test]
fn test_head_params_construction() {
    let _params = HeadParams {
        in_channels: 16,
        channels: 16,
        out_channels: 1,
        kernel_sizes: vec![1, 1],
        activation: ActivationType::Tanh,
    };
}
