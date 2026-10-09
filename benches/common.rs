// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared benchmark utilities.
//!
//! Provides deterministic signal generators, synthetic model-data builders,
//! and model-loader helpers used across the bench suite.
//!
//! This module is compiled into multiple bench binaries; individual functions
//! may appear unused in some binaries — this is expected and silenced with
//! `#![allow(dead_code)]` below.

// Benchmark common utilities shared across multiple bench targets.
#![allow(dead_code)]
#![allow(unused_imports)]

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::{
    NamConfig, NamLayerConfig, NamModelData, parse_nam_json,
};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::lstm::lstm_weight_count;
use std::fs;
use std::path::PathBuf;

/// Generates `num_samples` of a deterministic 440 Hz sine wave at 48 kHz.
/// Used as the standard excitation signal across inference benches.
pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    const F0: f64 = 440.0;
    const SR: f64 = 48_000.0;
    let omega = 2.0 * std::f64::consts::PI * F0 / SR;
    (0..num_samples)
        .map(|i| ((i as f64 * omega).sin()) as f32)
        .collect()
}

/// Builds synthetic `NamModelData` for an LSTM with the given layer count and
/// hidden size, using near-zero weights (0.01) to benchmark dispatch and process
/// paths without external fixture files.
pub fn make_lstm_data(num_layers: usize, hidden_size: usize) -> NamModelData {
    let total_weights = lstm_weight_count(num_layers, hidden_size);
    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "LSTM".to_string(),
        config: NamConfig {
            layers: vec![],
            head: None,
            head_scale: None,
            num_layers: Some(num_layers),
            hidden_size: Some(hidden_size),
            receptive_field: None,
            bias: None,
            submodels: None,
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: neural_amp_modeler_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Builds synthetic `NamModelData` for a free-geometry WaveNet Dynamic model
/// (CH=5, K=3, COND=3) that forces routing to `WaveNetModelDyn` instead of a
/// const-generic SKU.
pub fn make_wavenet_dyn_data() -> NamModelData {
    let channels = 5usize;
    let kernel_size = 3usize;
    let condition_size = 3usize;
    let head_1 = 5usize;
    let head_2 = 1usize;
    let dilations = [vec![1, 2, 4, 8, 16], vec![1, 2, 4, 8, 16]];
    let num_layers_per_array = 5usize;

    let array1_rechannel = channels;
    let array2_rechannel = channels * channels;
    let per_layer = channels * kernel_size * channels
        + channels
        + condition_size * channels
        + channels * channels
        + channels;
    let array1_head = channels * head_1;
    let array2_head = channels * head_2 + head_2;
    let total_weights = array1_rechannel
        + num_layers_per_array * per_layer
        + array1_head
        + array2_rechannel
        + num_layers_per_array * per_layer
        + array2_head
        + 1;

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![
                NamLayerConfig {
                    input_size: Some(1),
                    condition_size: Some(condition_size),
                    head_size: Some(head_1),
                    channels: Some(channels),
                    kernel_size: Some(kernel_size),
                    dilations: Some(dilations[0].clone()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(false),
                    ..Default::default()
                },
                NamLayerConfig {
                    input_size: Some(channels),
                    condition_size: Some(condition_size),
                    head_size: Some(head_2),
                    channels: Some(channels),
                    kernel_size: Some(kernel_size),
                    dilations: Some(dilations[1].clone()),
                    activation: Some("Tanh".to_string()),
                    gated: Some(false),
                    head_bias: Some(true),
                    ..Default::default()
                },
            ],
            head: None,
            head_scale: Some(0.02),
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: neural_amp_modeler_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Builds synthetic `NamModelData` for a WaveNet A2 Dynamic model (CH=4, gated)
/// that forces routing to `WaveNetA2Dyn` (CH=4 is not in the {3,8} catalog).
pub fn make_wavenet_a2_dyn_data() -> NamModelData {
    use neural_amp_modeler_rs::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES};

    let channels = 4usize;
    let bottleneck = 4usize;
    let head_k = neural_amp_modeler_rs::models::a2::params::A2_HEAD_KERNEL_SIZE;

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

    NamModelData {
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
        weights_layout: neural_amp_modeler_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Group count shared by every FiLM slot of the synthetic grouped-FiLM
/// A2-Dynamic fixtures (exact divisor of `condition_size=4` and of every
/// slot width, so the grouped row layout is fully populated).
const SYNTH_A2_DYN_FILM_GROUPS: u32 = 2;

/// Builds synthetic `NamModelData` for a WaveNet A2 Dynamic control model
/// (CH=4, `condition_size=4`, Linear `condition_dsp`) with **no FiLM slots**.
///
/// Identical twin of [`make_wavenet_a2_dyn_film_grouped_data`] minus the FiLM
/// weights: same topology, same conditioning path, same weight stream order.
/// The latency delta between the two targets isolates the grouped-FiLM
/// hot-path cost from the `condition_dsp` cost.
///
/// `condition_size=4` requires the Linear `condition_dsp` (receptive field 8):
/// the single condition output is broadcast to 4 channels at runtime, matching
/// the multi-channel condition contract the FiLM layers read.
pub fn make_wavenet_a2_dyn_cond_dsp_data() -> NamModelData {
    make_wavenet_a2_dyn_cond_data(false)
}

/// Builds synthetic `NamModelData` for a WaveNet A2 Dynamic model (CH=4) with
/// **grouped FiLM** (all 8 insertion slots active, `shift=true`,
/// `groups=2`) plus a Linear `condition_dsp`.
///
/// CH=4 is not in the A2 const-generic dispatch table ({3, 8}), forcing
/// routing to `WaveNetA2Dyn`. All 8 FiLM slots run the grouped
/// `cond_to_scale_shift` + global-row modulation path with per-layer weight
/// extents `film_weight_count_generic(groups, 4, slot_width, true)` — the
/// exact stream layout `load_film_for_layer_dynamic` consumes.
///
/// Control counterpart: [`make_wavenet_a2_dyn_cond_dsp_data`].
pub fn make_wavenet_a2_dyn_film_grouped_data() -> NamModelData {
    make_wavenet_a2_dyn_cond_data(true)
}

/// Shared builder for the synthetic conditioned A2-Dynamic pair above.
///
/// Weight stream order mirrors `WaveNetA2Dyn::load_weights_inner`:
/// rechannel → per layer (conv_w, conv_b, mixin_w, l1x1_w, l1x1_b,
/// head1x1_w, head1x1_b, then FiLM w/b per active slot in `FILM_KEYS`
/// order) → head_w, head_b, head_scale.
fn make_wavenet_a2_dyn_cond_data(with_grouped_film: bool) -> NamModelData {
    use neural_amp_modeler_rs::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES};

    let channels = 4usize;
    let bottleneck = 4usize;
    let cond_size = 4usize;
    let head_k = neural_amp_modeler_rs::models::a2::params::A2_HEAD_KERNEL_SIZE;
    let head1x1_out = 4usize;
    let h1_in = bottleneck;
    let groups = SYNTH_A2_DYN_FILM_GROUPS as usize;

    // FiLM slot widths in `FILM_KEYS` order (conv_pre, conv_post,
    // input_mixin_pre, input_mixin_post, activation_pre, activation_post,
    // layer1x1_post, head1x1_post) — see `load_film_for_layer_dynamic`.
    let film_slot_widths = [
        channels,
        bottleneck,
        cond_size,
        bottleneck,
        bottleneck,
        bottleneck,
        channels,
        head1x1_out,
    ];

    let mut total_weights = channels;
    for &ksize in A2_KERNEL_SIZES.iter() {
        total_weights += channels * bottleneck * ksize;
        total_weights += bottleneck;
        total_weights += bottleneck * cond_size;
        total_weights += bottleneck * channels;
        total_weights += channels;
        total_weights += head1x1_out * h1_in;
        total_weights += head1x1_out;
        if with_grouped_film {
            for &slot_ch in &film_slot_widths {
                total_weights += slot_ch * 2 * cond_size / groups;
                total_weights += slot_ch * 2;
            }
        }
    }
    total_weights += head_k * head1x1_out;
    total_weights += 1;
    total_weights += 1;

    let mut layer_raw = serde_json::json!({
        "head": {
            "out_channels": 1,
            "kernel_size": head_k,
            "bias": true
        },
        "head1x1": {
            "active": true,
            "out_channels": head1x1_out,
            "groups": 1
        },
        "layer1x1": {
            "active": true,
            "groups": 1
        }
    });
    if with_grouped_film {
        let film = serde_json::json!({
            "active": true,
            "shift": true,
            "groups": SYNTH_A2_DYN_FILM_GROUPS
        });
        let obj = layer_raw
            .as_object_mut()
            .expect("synthetic A2-Dyn layer_raw is a JSON object");
        for key in [
            "conv_pre_film",
            "conv_post_film",
            "input_mixin_pre_film",
            "input_mixin_post_film",
            "activation_pre_film",
            "activation_post_film",
            "layer1x1_post_film",
            "head1x1_post_film",
        ] {
            obj.insert(key.to_string(), film.clone());
        }
    }

    let cond_dsp = serde_json::to_value(make_linear_data(8, true))
        .expect("synthetic Linear condition_dsp must serialize to JSON");

    NamModelData {
        version: Some("0.5.4".to_string()),
        architecture: "WaveNet".to_string(),
        config: NamConfig {
            layers: vec![NamLayerConfig {
                input_size: Some(1),
                condition_size: Some(cond_size),
                channels: Some(channels),
                bottleneck: Some(bottleneck),
                kernel_sizes: Some(A2_KERNEL_SIZES.to_vec()),
                dilations: Some(A2_DILATIONS.to_vec()),
                activation: Some("LeakyReLU".to_string()),
                head_bias: Some(true),
                layer_raw: Some(layer_raw),
                ..Default::default()
            }],
            head: None,
            head_scale: Some(0.02),
            condition_dsp: Some(cond_dsp),
            ..Default::default()
        },
        weights: vec![0.01; total_weights],
        weights_layout: neural_amp_modeler_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Builds synthetic `NamModelData` for a `Linear` FIR model with the given
/// receptive field and optional bias scalar.
///
/// `implementation` is left unset, so the loader resolves it to
/// `LinearImplementation::Auto`: a receptive field at or above
/// `FFT_AUTO_THRESHOLD` (256) selects the partitioned-FFT production path,
/// below it the time-domain Direct path. Weights are a deterministic
/// decaying sinusoid; the trailing bias (when enabled) is zero.
pub fn make_linear_data(receptive_field: usize, bias: bool) -> NamModelData {
    let mut weights: Vec<f32> = (0..receptive_field)
        .map(|i| (i as f32 * 0.001).sin() * 0.5)
        .collect();
    if bias {
        weights.push(0.0);
    }
    NamModelData {
        version: Some("0.7.0".to_string()),
        architecture: "Linear".to_string(),
        config: NamConfig {
            receptive_field: Some(receptive_field),
            bias: Some(bias),
            ..Default::default()
        },
        weights,
        weights_layout: neural_amp_modeler_rs::loader::nam_json::WeightsLayout::Original,
        sample_rate: Some(48000.0),
        metadata: None,
    }
}

/// Builds and prewarms the synthetic `Linear` RF=2048 model that exercises the
/// partitioned-FFT (`LinearMode::Fft`) production path, without a fixture file.
pub fn load_and_prewarm_linear_fft_rf2048() -> neural_amp_modeler_rs::models::StaticModel {
    let data = make_linear_data(2048, true);
    let mut model = build_model(&data).expect("synthetic Linear RF=2048 model must build");
    model.prewarm(2048);
    *model
}

/// Resolves the path to a model fixture file, preferring `models-nondist`
/// or `third-party/community_models` when present and falling back to `tests/fixtures/models`.
pub fn model_path(filename: &str) -> PathBuf {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nondist = base.join("tests/fixtures/models-nondist").join(filename);
    if nondist.exists() {
        return nondist;
    }
    let community = std::env::var("NAM_THIRD_PARTY_DIR")
        .map(|d| PathBuf::from(d).join("community_models").join(filename))
        .unwrap_or_else(|_| base.join("third-party/community_models").join(filename));
    if community.exists() {
        return community;
    }
    base.join("tests/fixtures/models").join(filename)
}

/// Loads and prewarms a model fixture (2048 samples). Returns `None` if the
/// file is missing or fails to parse — callers should `return` to skip the
/// benchmark silently when this happens.
pub fn load_and_prewarm(filename: &str) -> Option<neural_amp_modeler_rs::models::StaticModel> {
    let path = model_path(filename);
    if !path.exists() {
        return None;
    }
    let json_data = fs::read_to_string(&path).ok()?;
    let model_data = parse_nam_json(&json_data).ok()?;
    let mut model = build_model(&model_data).ok()?;
    model.prewarm(2048);
    Some(*model)
}

/// Loads and prewarms a model fixture (2048 samples). Panics with an explicit
/// error message if the file is missing or fails to parse/build — required for
/// regression gate tests where missing fixtures are prohibited.
pub fn load_and_prewarm_required(filename: &str) -> neural_amp_modeler_rs::models::StaticModel {
    let path = model_path(filename);
    if !path.exists() {
        panic!(
            "Required benchmark model fixture '{}' not found at {:?}",
            filename, path
        );
    }
    let json_data = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "Failed to read required benchmark model fixture '{}': {e}",
            filename
        )
    });
    let model_data = parse_nam_json(&json_data).unwrap_or_else(|e| {
        panic!(
            "Failed to parse JSON for required benchmark model fixture '{}': {e}",
            filename
        )
    });
    let mut model = build_model(&model_data).unwrap_or_else(|e| {
        panic!(
            "Failed to build model for required benchmark model fixture '{}': {e}",
            filename
        )
    });
    model.prewarm(2048);
    *model
}

/// Loads and parses a model fixture into `NamModelData` without building the
/// model. Returns `None` if the file is missing or fails to parse. Used by
/// benches that need `model_data` downstream (e.g. prewarm cost with
/// `iter_with_setup`).
pub fn load_model_data(filename: &str) -> Option<NamModelData> {
    let path = model_path(filename);
    let json_data = fs::read_to_string(&path).ok()?;
    parse_nam_json(&json_data).ok()
}

/// Creates deterministic f32-only test data for a given (in_len, out_len) pair.
/// Generates in_frames (sinusoidal input), flat weights (sinusoidal f32),
/// and zero-initialized out_frames.
pub fn make_f32_test_data(in_len: usize, out_len: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let in_frames: Vec<f32> = (0..in_len).map(|i| (i as f32 * 0.17).sin()).collect();
    let weights: Vec<f32> = (0..in_len * out_len)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    let out_frames = vec![0.0f32; out_len];
    (in_frames, weights, out_frames)
}

/// Generates a synthetic impulse response (exponentially decaying sinusoid)
/// for CabSim/Linear benchmarks, avoiding external fixture dependencies.
pub fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
    const SR: f32 = 48000.0;
    (0..len)
        .map(|n| {
            let t = n as f32 / SR;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}
