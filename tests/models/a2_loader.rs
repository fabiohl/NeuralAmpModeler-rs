// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  A2 Loader Tests.
//
//  Covers constant parity with C++ `a2_fast.h`, topology detection (`is_a2_shape`,
//  `is_wavenet_a2` dispatch), fixture loading, real inference sanity, forward‑compatibility
//  error handling, and A1 model regression.

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    A2TopologyResult, NamConfig, NamLayerConfig, NamModelData, WavenetTopologyResult,
    WeightsLayout, get_wavenet_topology, is_a2_shape, parse_nam_json,
};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::a2::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS,
    A2_VALID_CHANNELS, a2_weight_count,
};
use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_snr_db};
use neural_amp_modeler_rs::testing::wav::{read_wav_f32, write_wav_f32};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn model_path(filename: &str) -> PathBuf {
    neural_amp_modeler_rs::testing::fixtures::model_path(filename)
}

fn make_a2_data(channels: u8, dilations: Vec<usize>) -> NamModelData {
    NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(channels as usize),
                kernel_size: None,
                kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                dilations: Some(dilations),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
                bottleneck: Some(channels as usize),
                ..Default::default()
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

fn make_unrecognized_a2_like_data(channels: usize) -> NamModelData {
    NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(channels),
                kernel_size: None,
                dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
                ..Default::default()
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

// =============================================================================
// 1. Constants cross‑check with C++ a2_fast.h
// =============================================================================

#[test]
fn test_a2_constants_match_cpp_reference() {
    assert_eq!(A2_NUM_LAYERS, 23, "kNumLayers mismatch");

    assert_eq!(A2_HEAD_KERNEL_SIZE, 16, "kHeadKernelSize mismatch");

    assert!(
        (A2_LEAKY_SLOPE - 0.01f32).abs() < f32::EPSILON,
        "kLeakySlope mismatch: got {}, expected 0.01",
        A2_LEAKY_SLOPE
    );

    let cpp_kernel_sizes: [usize; 23] = [
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6,
    ];
    assert_eq!(A2_KERNEL_SIZES, cpp_kernel_sizes, "kKernelSizes mismatch");

    let cpp_dilations: [usize; 23] = [
        1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239,
    ];
    assert_eq!(A2_DILATIONS, cpp_dilations, "kDilations mismatch");

    let cpp_valid_channels: [u8; 2] = [3, 8];
    assert_eq!(
        A2_VALID_CHANNELS, cpp_valid_channels,
        "valid channels mismatch"
    );
}

// =============================================================================
// 2. is_a2_shape: accepts valid topologies, rejects others
// =============================================================================

#[test]
fn test_is_a2_shape_accepts_channels_3() {
    let data = make_a2_data(3, A2_DILATIONS.to_vec());
    let result = is_a2_shape(&data);
    assert!(result.is_some(), "should be A2 shape");
    assert_eq!(
        result.unwrap(),
        neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(3),
        "should return channels=3"
    );
}

#[test]
fn test_is_a2_shape_accepts_channels_8() {
    let data = make_a2_data(8, A2_DILATIONS.to_vec());
    let result = is_a2_shape(&data);
    assert!(result.is_some(), "should be A2 shape");
    assert_eq!(
        result.unwrap(),
        neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(8),
        "should return channels=8"
    );
}

#[test]
fn test_is_a2_shape_rejects_channels_4() {
    let data = make_a2_data(4, A2_DILATIONS.to_vec());
    assert_eq!(
        is_a2_shape(&data),
        Some(neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::Dynamic),
        "channels=4 should be Dynamic A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_channels_16() {
    let data = make_a2_data(16, A2_DILATIONS.to_vec());
    assert_eq!(
        is_a2_shape(&data),
        Some(neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::Dynamic),
        "channels=16 should be Dynamic A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_wrong_dilations() {
    let wrong_dils: Vec<usize> = vec![
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4,
    ];
    let data = make_a2_data(8, wrong_dils);
    assert_eq!(
        is_a2_shape(&data),
        Some(A2TopologyResult::Dynamic),
        "dilations not matching A2_DILATIONS should route to Dynamic"
    );
}

#[test]
fn test_is_a2_shape_accepts_multiple_layers() {
    // Multi-array A2 topologies are now accepted.
    // The first layer must have valid A2 shape characteristics.
    let data = NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(1),
                    head_size: None,
                    channels: Some(8),
                    kernel_size: None,
                    kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                    dilations: Some(A2_DILATIONS.to_vec()),
                    activation: Some("LeakyReLU".to_string()),
                    gated: None,
                    head_bias: None,
                    ..Default::default()
                },
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(1),
                    head_size: None,
                    channels: Some(8),
                    kernel_size: None,
                    kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                    dilations: Some(A2_DILATIONS.to_vec()),
                    activation: Some("LeakyReLU".to_string()),
                    gated: None,
                    head_bias: None,
                    ..Default::default()
                },
            ],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };
    assert!(
        is_a2_shape(&data).is_some(),
        "multi-array A2 models with valid shape must now be accepted"
    );
}

#[test]
fn test_is_a2_shape_rejects_non_wavenet_architecture() {
    let data = NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(8),
                kernel_size: None,
                dilations: Some(A2_DILATIONS.to_vec()),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
                ..Default::default()
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };
    assert!(
        is_a2_shape(&data).is_none(),
        "LSTM architecture should not match A2 shape"
    );
}

#[test]
fn test_is_a2_shape_rejects_wrong_dilations_length() {
    let short_dils: Vec<usize> = vec![1, 2, 3];
    let data = make_a2_data(8, short_dils);
    assert_eq!(
        is_a2_shape(&data),
        Some(A2TopologyResult::Dynamic),
        "dilations with wrong length should route to Dynamic"
    );
}

#[test]
fn test_a2_shape_does_not_match_standard_wavenet() {
    let a1_std_dils = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let data = NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    channels: Some(16),
                    dilations: Some(a1_std_dils.clone()),
                    activation: Some("Tanh".to_string()),
                    input_size: None,
                    condition_size: None,
                    head_size: None,
                    kernel_size: None,
                    gated: Some(false),
                    head_bias: Some(false),
                    ..Default::default()
                },
                NamLayerConfig {
                    channels: Some(16),
                    dilations: Some(a1_std_dils),
                    activation: Some("Tanh".to_string()),
                    input_size: None,
                    condition_size: None,
                    head_size: None,
                    kernel_size: None,
                    gated: Some(false),
                    head_bias: Some(true),
                    ..Default::default()
                },
            ],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };

    assert!(
        is_a2_shape(&data).is_none(),
        "Standard WaveNet topology should not match A2 shape"
    );
    assert!(
        matches!(get_wavenet_topology(&data), WavenetTopologyResult::Known(_)),
        "Standard WaveNet should still be detected by get_wavenet_topology"
    );
}

// =============================================================================
// 3. Dispatch table: A1 with high version must NOT be misrouted to A2
// =============================================================================

#[test]
fn test_dispatch_table_a1_high_version_not_misrouted() {
    let a1_std_dils: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let a1_lite_dils_0: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64];
    let a1_lite_dils_1: Vec<usize> = vec![128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512];

    struct Case {
        desc: &'static str,
        data: NamModelData,
        expect_a2_shape: Option<neural_amp_modeler_rs::loader::nam_json::A2TopologyResult>,
        expect_wavenet_a2: bool,
    }

    let cases = vec![
        Case {
            desc: "A1 Standard v0.5.4 — baseline A1",
            data: NamModelData {
                version: Some("0.5.4".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                            ..Default::default()
                        },
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(true),
                            ..Default::default()
                        },
                    ],
                    head: None,
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Standard v0.7.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.7.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                            ..Default::default()
                        },
                        NamLayerConfig {
                            channels: Some(16),
                            dilations: Some(a1_std_dils.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(true),
                            ..Default::default()
                        },
                    ],
                    head: None,
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Lite v0.8.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.8.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(12),
                            dilations: Some(a1_lite_dils_0.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                            ..Default::default()
                        },
                        NamLayerConfig {
                            channels: Some(12),
                            dilations: Some(a1_lite_dils_1.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(true),
                            ..Default::default()
                        },
                    ],
                    head: None,
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A1 Feather v0.9.0 — high version, NOT A2",
            data: NamModelData {
                version: Some("0.9.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![
                        NamLayerConfig {
                            channels: Some(8),
                            dilations: Some(a1_lite_dils_0.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(false),
                            ..Default::default()
                        },
                        NamLayerConfig {
                            channels: Some(8),
                            dilations: Some(a1_lite_dils_1.clone()),
                            activation: Some("Tanh".to_string()),
                            input_size: None,
                            condition_size: None,
                            head_size: None,
                            kernel_size: None,
                            gated: Some(false),
                            head_bias: Some(true),
                            ..Default::default()
                        },
                    ],
                    head: None,
                    head_scale: None,
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "A2 real CH=3 (Lite shape)",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(3),
                        dilations: Some(A2_DILATIONS.to_vec()),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                        kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                        bottleneck: Some(3),
                        ..Default::default()
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: Some(
                neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(3),
            ),
            expect_wavenet_a2: true,
        },
        Case {
            desc: "A2 real CH=8 (Full shape)",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(8),
                        dilations: Some(A2_DILATIONS.to_vec()),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                        kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                        bottleneck: Some(8),
                        ..Default::default()
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: Some(
                neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(8),
            ),
            expect_wavenet_a2: true,
        },
        Case {
            desc: "Ambiguous: CH=3, wrong dils, LeakyReLU — A2 via activation",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(3),
                        dilations: Some(vec![1, 2, 4, 8]),
                        activation: Some("LeakyReLU".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                        ..Default::default()
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: true,
        },
        Case {
            desc: "Ambiguous: CH=8, wrong dils, Tanh, high version — NOT A2",
            data: NamModelData {
                version: Some("0.7.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(8),
                        dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                        activation: Some("Tanh".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                        ..Default::default()
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
        Case {
            desc: "Ambiguous: CH=4 (A1 Nano channels), single layer, high version, Tanh — NOT A2",
            data: NamModelData {
                version: Some("0.6.0".to_string()),
                architecture: "WaveNet".to_string(),
                config: NamConfig {
                    layers: vec![NamLayerConfig {
                        channels: Some(4),
                        dilations: Some(vec![1, 2, 4, 8, 16, 32, 64]),
                        activation: Some("Tanh".to_string()),
                        input_size: Some(1),
                        condition_size: Some(1),
                        head_size: None,
                        kernel_size: None,
                        gated: None,
                        head_bias: None,
                        ..Default::default()
                    }],
                    head: None,
                    head_scale: Some(1.0),
                    num_layers: None,
                    hidden_size: None,
                    receptive_field: None,
                    bias: None,
                    submodels: None,
                    ..Default::default()
                },
                weights: vec![],
                sample_rate: Some(48000.0),
                metadata: None,
                weights_layout: WeightsLayout::Original,
            },
            expect_a2_shape: None,
            expect_wavenet_a2: false,
        },
    ];

    for case in &cases {
        let shape_result = is_a2_shape(&case.data);
        let wavenet_a2_result = case.data.is_wavenet_a2();
        assert_eq!(
            shape_result, case.expect_a2_shape,
            "[{}] is_a2_shape mismatch: expected {:?}, got {:?}",
            case.desc, case.expect_a2_shape, shape_result,
        );
        assert_eq!(
            wavenet_a2_result, case.expect_wavenet_a2,
            "[{}] is_wavenet_a2 mismatch: expected {}, got {}",
            case.desc, case.expect_wavenet_a2, wavenet_a2_result,
        );
    }
}

// =============================================================================
// 4. Fixture loading (real .nam files from tests/fixtures/models/)
// =============================================================================

#[test]
fn test_a2_full_fixture_loads() {
    let json = fs::read_to_string(model_path("wavenet_a2_full.nam")).unwrap();
    let data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&json)
        .expect("Failed to parse wavenet_a2_full.nam");

    assert_eq!(data.architecture, "WaveNet");
    assert_eq!(data.weights.len(), a2_weight_count::<8>());

    let ch = is_a2_shape(&data).expect("Should be recognized as A2 shape");
    assert_eq!(
        ch,
        neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(8)
    );

    let _model = build_model(&data).expect("Should dispatch to A2-Full");
}

#[test]
fn test_a2_lite_fixture_loads() {
    let json = fs::read_to_string(model_path("wavenet_a2_lite.nam")).unwrap();
    let data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&json)
        .expect("Failed to parse wavenet_a2_lite.nam");

    assert_eq!(data.architecture, "WaveNet");
    assert_eq!(data.weights.len(), a2_weight_count::<3>());

    let ch = is_a2_shape(&data).expect("Should be recognized as A2 shape");
    assert_eq!(
        ch,
        neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::KnownFastPath(3)
    );

    let _model = build_model(&data).expect("Should dispatch to A2-Lite");
}

// =============================================================================
// 5. Real inference sanity: A2 models produce finite output
// =============================================================================

/// With zero-weights (default construction), the A2 Lite pipeline
/// must produce silence — input × zero weights → zero output.
/// Enforces the rigid dynamic envelope: |output| < 1e-7 for
/// constant 0.01 input through zero-weighted net.
#[test]
fn test_a2_lite_inference_produces_finite_output() {
    use neural_amp_modeler_rs::models::a2::WaveNetA2;

    let mut model = WaveNetA2::<3>::new().unwrap();
    model.prewarm();
    let input = [0.01f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "A2-Lite output must be finite");
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        assert!(
            abs_s < 1e-7,
            "[A2-Lite] sample {i}: |{s}| >= 1e-7 — zero-weight net produced non-zero output"
        );
    }
    assert!(
        max_abs_out < 1e-7,
        "[A2-Lite] max_abs={max_abs_out} — silence expected from zero-weight model"
    );
}

/// With zero-weights (default construction), the A2 Full pipeline
/// must produce silence — input × zero weights → zero output.
/// Enforces the rigid dynamic envelope: |output| < 1e-7 for
/// constant 0.01 input through zero-weighted net.
#[test]
fn test_a2_full_inference_produces_finite_output() {
    use neural_amp_modeler_rs::models::a2::WaveNetA2;

    let mut model = WaveNetA2::<8>::new().unwrap();
    model.prewarm();
    let input = [0.01f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    let mut max_abs_out = 0.0f32;
    for (i, &s) in output.iter().enumerate() {
        assert!(s.is_finite(), "A2-Full output must be finite");
        let abs_s = s.abs();
        max_abs_out = max_abs_out.max(abs_s);
        assert!(
            abs_s < 1e-7,
            "[A2-Full] sample {i}: |{s}| >= 1e-7 — zero-weight net produced non-zero output"
        );
    }
    assert!(
        max_abs_out < 1e-7,
        "[A2-Full] max_abs={max_abs_out} — silence expected from zero-weight model"
    );
}

// =============================================================================
// 6. Forward‑compatibility: unrecognized A2‑like shapes return clear errors
// =============================================================================

#[test]
fn test_unrecognized_a2_shape_returns_clear_error() {
    let data = make_unrecognized_a2_like_data(5);
    assert!(
        data.is_wavenet_a2(),
        "model should be detected as A2 via activation (LeakyReLU, non-Tanh)"
    );
    let result = build_model(&data);
    assert!(
        result.is_err(),
        "unrecognized A2 shape must produce an error, not a silent bypass"
    );
    let err_msg = format!("{}", result.err().unwrap());
    assert!(
        err_msg.contains("not recognized") || err_msg.contains("shape"),
        "Error should mention topology not being recognized: {err_msg}",
    );
}

// =============================================================================
// 7. Strict A2 shape rejection
// =============================================================================

/// bottleneck != channels must be rejected by is_a2_shape (prevents
/// silent misroute to fast-path that assumes bottleneck==channels).
#[test]
fn test_is_a2_shape_rejects_bottleneck_neq_channels() {
    let json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": 1.0,
            "head": null,
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "bottleneck": 16,
                "kernel_sizes": [6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6],
                "dilations": [1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239],
                "activation": [{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01}],
                "gating_mode": [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
                "layer1x1": {"active": true, "groups": 1},
                "head1x1": null,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
                "groups_input": 1,
                "groups_input_mixin": 1
            }]
        },
        "weights": []
    }"#;
    let data = parse_nam_json(json).expect("JSON parse failed");
    assert_eq!(
        is_a2_shape(&data),
        Some(neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::Dynamic),
        "bottleneck=16 != channels=8 must be Dynamic A2 shape"
    );
}

/// Gated activation (gating_mode not all "none") must be rejected.
#[test]
fn test_is_a2_shape_rejects_gated_activation() {
    let json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": 1.0,
            "head": null,
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "bottleneck": 8,
                "kernel_sizes": [6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6],
                "dilations": [1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239],
                "activation": [{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01}],
                "gating_mode": ["gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated","gated"],
                "layer1x1": {"active": true, "groups": 1},
                "head1x1": null,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
                "groups_input": 1,
                "groups_input_mixin": 1
            }]
        },
        "weights": []
    }"#;
    let data = parse_nam_json(json).expect("JSON parse failed");
    assert_eq!(
        is_a2_shape(&data),
        Some(neural_amp_modeler_rs::loader::nam_json::A2TopologyResult::Dynamic),
        "gating_mode=all gated must be Dynamic A2 shape"
    );
}

/// Active FiLM conditioning is now accepted by is_a2_shape (FiLM
/// weight loading happens in set_weights, so the shape detector no longer rejects
/// models with active FiLM config entries).
#[test]
fn test_is_a2_shape_accepts_active_film() {
    let json = r#"{
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "in_channels": 1,
            "head_scale": 1.0,
            "head": null,
            "layers": [{
                "input_size": 1,
                "condition_size": 1,
                "channels": 8,
                "bottleneck": 8,
                "kernel_sizes": [6,6,6,6,6,6,6,6,6,6,6,6,6,6,15,15,6,6,6,6,6,6,6],
                "dilations": [1,3,7,17,41,101,239,1,3,7,17,41,101,239,1,13,1,3,7,17,41,101,239],
                "activation": [{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01},{"type":"LeakyReLU","negative_slope":0.01}],
                "gating_mode": [null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null,null],
                "layer1x1": {"active": true, "groups": 1},
                "head1x1": null,
                "head": {"out_channels": 1, "kernel_size": 16, "bias": true},
                "groups_input": 1,
                "groups_input_mixin": 1,
                "conv_pre_film": {"active": true, "shared": false, "num_inputs": 1}
            }]
        },
        "weights": []
    }"#;
    let data = parse_nam_json(json).expect("JSON parse failed");
    assert!(
        is_a2_shape(&data).is_some(),
        "active FiLM (conv_pre_film) must now be accepted by is_a2_shape"
    );
}

// =============================================================================
// 8. Regression: A1 models continue to load and infer
// =============================================================================

#[test]
fn test_regression_a1_wavenet_standard() {
    let path = model_path("BossWN-standard.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model =
        build_model(&model_data).expect("Dispatcher failed to build WaveNet A1 Standard model");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "WaveNet A1 output contains non-finite values (NaN/Inf)"
        );
    }
}

#[test]
fn test_regression_a1_lstm() {
    let path = model_path("BossLSTM-1x16.nam");

    if !path.exists() {
        return;
    }

    let json_data = fs::read_to_string(&path).expect("Failed to read JSON file");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    let mut model = build_model(&model_data).expect("Dispatcher failed to build LSTM A1 model");

    model.prewarm(2048);

    let input = [0.0f32; 64];
    let mut output = [0.0f32; 64];
    model.process(&input, &mut output);

    for &s in output.iter() {
        assert!(
            s.is_finite(),
            "LSTM A1 output contains non-finite values (NaN/Inf)"
        );
    }
}

// ---------------------------------------------------------------------------
// Adversarial A2 topology validation (F-02 / F-03 / F-04)
// ---------------------------------------------------------------------------

fn make_a2_data_with_topology(
    channels: u8,
    kernel_sizes: Vec<usize>,
    dilations: Vec<usize>,
) -> NamModelData {
    NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                head_size: None,
                channels: Some(channels as usize),
                kernel_size: None,
                kernel_sizes: Some(kernel_sizes),
                dilations: Some(dilations),
                activation: Some("LeakyReLU".to_string()),
                gated: None,
                head_bias: None,
                bottleneck: Some(channels as usize),
                ..Default::default()
            }],
            head: None,
            head_scale: Some(1.0),
            num_layers: None,
            hidden_size: None,
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    }
}

/// F-03: a zero kernel size would underflow `(kernel - 1)` in the A2 dynamic
/// builder — it must be rejected before any allocation, without panicking.
#[test]
fn test_a2_rejects_zero_kernel_size() {
    let data = make_a2_data_with_topology(8, vec![0], vec![1]);
    let err = match build_model(&data) {
        Ok(_) => panic!("kernel_sizes=[0] must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("kernel_sizes[0] is 0") || err.contains("must be >= 1"),
        "error should describe the zero kernel, got: {err}"
    );
}

/// F-02: diverging `kernel_sizes`/`dilations` lengths must be rejected at the
/// semantic layer (previously `assert_eq!` in `WaveNetA2Dyn::new` panicked).
#[test]
fn test_a2_rejects_divergent_vector_lengths() {
    let data = make_a2_data_with_topology(8, vec![3, 3, 3], vec![1, 2]);
    let err = match build_model(&data) {
        Ok(_) => panic!("divergent kernel_sizes/dilations lengths must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("kernel_sizes length (3) != dilations length (2)"),
        "error should describe the length divergence, got: {err}"
    );
}

/// F-02: activation-array length divergence must be rejected (no silent
/// fallback to default LeakyReLU).
#[test]
fn test_a2_rejects_divergent_activation_array_length() {
    let layer = NamLayerConfig {
        input_size: Some(1),
        condition_size: Some(1),
        head_size: None,
        channels: Some(8),
        kernel_size: None,
        kernel_sizes: Some(vec![6, 6]),
        dilations: Some(vec![1, 2]),
        activation: Some("LeakyReLU".to_string()),
        gated: None,
        head_bias: None,
        bottleneck: Some(8),
        layer_raw: Some(serde_json::json!({
            "activation": ["LeakyReLU"], // length 1 != num_layers 2
            "gating_mode": ["none", "none"],
            "secondary_activation": [null, null],
        })),
        ..Default::default()
    };
    let data = NamModelData {
        version: Some("0.6.0".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![layer],
            head: None,
            head_scale: Some(1.0),
            ..Default::default()
        },
        weights: vec![],
        sample_rate: Some(48000.0),
        metadata: None,
        weights_layout: WeightsLayout::Original,
    };
    let err = match build_model(&data) {
        Ok(_) => panic!("activation-array length divergence must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("array length (1) != num_layers (2)") || err.contains("topology divergence"),
        "error should describe a topology divergence, got: {err}"
    );
}

/// F-04: a massive layer count (100k elements) must be rejected before any
/// mmap'd MirroredBuffer is allocated (each layer allocates one).
#[test]
fn test_a2_rejects_massive_layer_count_without_allocations() {
    let kernel_sizes = vec![6usize; 100_000];
    let dilations = vec![1usize; 100_000];
    let data = make_a2_data_with_topology(8, kernel_sizes, dilations);
    let err = match build_model(&data) {
        Ok(_) => panic!("100k layers must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("exceeding maximum") || err.contains("layers"),
        "error should describe the layer cap, got: {err}"
    );
}

/// Regression test: synthetic A2 Dynamic model with head kernel size 16 must build
/// and process without weight count mismatch.
#[test]
fn test_a2_dynamic_synthetic_model_with_head_kernel_size_16_builds() {
    let channels = 4usize;
    let bottleneck = 4usize;
    let head_k = 16usize;

    let mut total_weights = channels;
    for &ksize in A2_KERNEL_SIZES.iter() {
        total_weights += channels * bottleneck * ksize;
        total_weights += bottleneck;
        total_weights += bottleneck;
        total_weights += bottleneck * channels;
        total_weights += channels;
    }
    total_weights += head_k * channels;
    total_weights += 1;
    total_weights += 1;

    let data = NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(1),
                channels: Some(channels),
                bottleneck: Some(bottleneck),
                kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                dilations: Some(A2_DILATIONS.to_vec()),
                activation: Some("LeakyReLU".to_string()),
                gated: Some(true),
                head_bias: Some(true),
                layer_raw: Some(serde_json::json!({
                    "head": {
                        "out_channels": 1,
                        "kernel_size": head_k,
                        "bias": true
                    }
                })),
                ..Default::default()
            }],
            head: None,
            head_scale: Some(0.02),
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    };

    let mut model = build_model(&data).expect("Synthetic A2-Dynamic model build must succeed");
    model.prewarm(64);
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

// =============================================================================
// 7. condition_dsp in A2 fast-path candidate (T5.2)
// =============================================================================

fn find_namcore_render_bin() -> Option<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join("build/namcore_render/tools/render"),
        manifest_dir.join("build/namcore_render/Release/render"),
        manifest_dir.join("build/namcore_render/Debug/render"),
        manifest_dir.join("build/namcore_render/render"),
    ];
    for path in &candidates {
        if path.exists() && path.is_file() {
            return Some(path.clone());
        }
    }
    None
}

/// A2 fast-shape candidate with condition_dsp at 44.1 kHz vs WaveNet at 48 kHz
/// must be rejected with `NamErrorCode::NamJsonSampleRateMismatch` (E1315).
#[test]
fn test_a2_fast_shape_condition_dsp_sample_rate_mismatch_rejected() {
    let json_str = fs::read_to_string(model_path("wavenet_a2_lite.nam"))
        .expect("read wavenet_a2_lite.nam fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse JSON");
    val["config"]["condition_dsp"] = serde_json::json!({
        "version": "0.5.4",
        "architecture": "Linear",
        "config": {"receptive_field": 1, "bias": false},
        "weights": [0.5],
        "sample_rate": 44100.0
    });
    let data = parse_nam_json(&val.to_string()).expect("parse mutated nam json");
    assert_eq!(
        is_a2_shape(&data),
        Some(A2TopologyResult::Dynamic),
        "A2 model with non-null condition_dsp must route to Dynamic"
    );
    let err = match build_model(&data) {
        Ok(_) => panic!("expected NamJsonSampleRateMismatch error"),
        Err(e) => e,
    };
    assert_eq!(
        err.downcast_ref::<NamErrorCode>(),
        Some(&NamErrorCode::NamJsonSampleRateMismatch),
        "condition_dsp with mismatched sample rate must fail with E1315"
    );
}

/// A2 fast-shape candidate with condition_dsp at 48 kHz routes to Dynamic
/// and successfully builds `WaveNetA2Dyn` executing `condition_dsp`.
#[test]
fn test_a2_fast_shape_condition_dsp_routes_to_dynamic_and_builds() {
    let json_str = fs::read_to_string(model_path("wavenet_a2_lite.nam"))
        .expect("read wavenet_a2_lite.nam fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse JSON");
    val["config"]["condition_dsp"] = serde_json::json!({
        "version": "0.5.4",
        "architecture": "Linear",
        "config": {"receptive_field": 1, "bias": false},
        "weights": [0.5],
        "sample_rate": 48000.0
    });
    let data = parse_nam_json(&val.to_string()).expect("parse mutated nam json");
    assert_eq!(
        is_a2_shape(&data),
        Some(A2TopologyResult::Dynamic),
        "A2 model with condition_dsp must route to Dynamic"
    );
    let mut model = build_model(&data).expect("build model must succeed");
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

/// A2 fast-shape candidate with condition_dsp produces sample-accurate audio parity
/// against the upstream C++ `NeuralAmpModelerCore` render binary.
#[test]
fn test_a2_fast_shape_condition_dsp_cpp_render_parity() {
    let render_bin = match find_namcore_render_bin() {
        Some(bin) => bin,
        None => {
            eprintln!(
                "[STATUS] SKIP_OPTIONAL_FIXTURE: namcore_render binary not found, skipping C++ parity"
            );
            return;
        }
    };

    let json_str = fs::read_to_string(model_path("wavenet_a2_lite.nam"))
        .expect("read wavenet_a2_lite.nam fixture");
    let mut val: serde_json::Value = serde_json::from_str(&json_str).expect("parse JSON");
    val["config"]["condition_dsp"] = serde_json::json!({
        "version": "0.5.4",
        "architecture": "Linear",
        "config": {"receptive_field": 1, "bias": false},
        "weights": [0.5],
        "sample_rate": 48000.0
    });
    let mutated_json = val.to_string();

    let num_samples = 2048;
    let sample_rate = 48000;
    let mut input_signal = vec![0.0f32; num_samples];
    for (i, sample) in input_signal.iter_mut().enumerate() {
        *sample =
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / (sample_rate as f32)).sin();
    }

    let tmp_dir = std::env::temp_dir();
    let in_wav_path = tmp_dir.join("nam_test_a2_cond_dsp_in.wav");
    let out_wav_path = tmp_dir.join("nam_test_a2_cond_dsp_out.wav");
    let nam_path = tmp_dir.join("nam_test_a2_cond_dsp.nam");

    write_wav_f32(&in_wav_path, &input_signal, sample_rate).expect("write input wav");
    fs::write(&nam_path, &mutated_json).expect("write mutated nam");

    let cmd_status = Command::new(&render_bin)
        .arg(&nam_path)
        .arg(&in_wav_path)
        .arg(&out_wav_path)
        .output();

    let output = match cmd_status {
        Ok(out) if out.status.success() => out,
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            panic!("C++ render failed for A2 condition_dsp: {stderr}");
        }
        Err(e) => {
            panic!("Failed to execute C++ render: {e}");
        }
    };
    let _ = output;

    let (cpp_output, cpp_sr) = read_wav_f32(&out_wav_path).expect("read cpp output wav");
    assert_eq!(cpp_sr, sample_rate);
    assert_eq!(cpp_output.len(), num_samples);

    let data = parse_nam_json(&mutated_json).expect("parse in rust");
    let mut rust_model = build_model(&data).expect("build model in rust");
    rust_model.reset(sample_rate, 64).expect("reset");
    let mut rust_output = vec![0.0f32; num_samples];
    for (in_chunk, out_chunk) in input_signal.chunks(64).zip(rust_output.chunks_mut(64)) {
        rust_model.process(in_chunk, out_chunk);
    }

    let esr = compute_esr(&cpp_output, &rust_output);
    let snr_db = compute_snr_db(&cpp_output, &rust_output);

    // Measured: ESR = 4.8722e-14, SNR = 133.12 dB (oracle parity against NeuralAmpModelerCore render tool)
    assert!(
        esr < 1e-4,
        "ESR too high: {esr} (must match C++ render output)"
    );
    assert!(
        snr_db > 80.0,
        "SNR too low: {snr_db} dB (must match C++ render output)"
    );

    // Cleanup temp files
    let _ = fs::remove_file(&in_wav_path);
    let _ = fs::remove_file(&out_wav_path);
    let _ = fs::remove_file(&nam_path);
}
