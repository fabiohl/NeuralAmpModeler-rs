// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Numerical parity tests: dynamic (runtime-dimensional) ↔ const-generic WaveNet.
//!
//! Validates that `WaveNetModelDyn` produces identical outputs to the
//! corresponding const-generic `WaveNetModel<CH, K, HEAD>` for the four
//! catalog geometries (Standard, Lite, Feather, Nano).
//!
//! Each test:
//! 1. Builds a const-generic model with deterministic synthetic weights (0.01)
//! 2. Builds an equivalent dynamic model using the same weights
//! 3. Prewarms both models
//! 4. Feeds identical random input through both
//! 5. Asserts bit-exact output match

use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::loader::dispatcher::wavenet::layout::transpose_conv1d_interleaved_8wide;
use crate::loader::dispatcher::wavenet::layout::transpose_conv1d_interleaved_16wide;
use crate::math::common::AlignedVec;
use crate::models::a2::activations::ActivationType;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d::Conv1d;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::dense::DenseLayer;
use crate::models::wavenet::dense_dyn::DenseLayerDyn;
use crate::models::wavenet::layer::WaveNetLayer;
use crate::models::wavenet::layer_array::WaveNetLayerArray;
use crate::models::wavenet::layer_array_dyn::WaveNetLayerArrayDyn;
use crate::models::wavenet::layer_dyn::WaveNetLayerDyn;
use crate::models::wavenet::model::WaveNetModel;
use crate::models::wavenet::model_dyn::WaveNetModelDyn;

/// Weight value used throughout for deterministic synthetic model construction.
const SYNTHETIC_WEIGHT: f32 = 0.01;

/// Dilations for the test models (3 layers).
const TEST_DILATIONS: [usize; 3] = [1, 2, 4];

/// Helper: interleave f32 weights into the `[OUT/4][K][IN][4]` layout used by Conv1d.
fn make_conv1d_weights(in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<f32> {
    let raw_weights = vec![SYNTHETIC_WEIGHT; out_ch * k * in_ch];
    let interleave_width = select_interleave_width(out_ch);
    let num_blocks = out_ch.div_ceil(interleave_width);
    let interleaved_len = num_blocks * k * in_ch * interleave_width;
    let mut weights = AlignedVec::new(interleaved_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    match interleave_width {
        16 => {
            transpose_conv1d_interleaved_16wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        8 => {
            transpose_conv1d_interleaved_8wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        _ => {
            crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
                &raw_weights,
                &mut weights,
                in_ch,
                out_ch,
                k,
            );
        }
    }
    weights
}

/// Helper: create f32 dense weights.
fn make_dense_weights(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![SYNTHETIC_WEIGHT; out_ch * in_ch])
        .expect("allocation should succeed for test-sized buffers")
}

/// Helper: create a bias AlignedVec of given size filled with zeros.
fn make_bias(len: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.0; len]).expect("allocation should succeed for test-sized buffers")
}

/// Builds a const-generic `WaveNetModel<CH, K, HEAD>` with synthetic weights.
fn build_const_generic_model<const CH: usize, const K: usize, const HEAD: usize>()
-> WaveNetModel<CH, K, HEAD> {
    let rf1 = *TEST_DILATIONS.iter().max().unwrap_or(&1) * (K - 1);
    let rf2 = rf1;

    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, CH, K> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv1d_weights(CH, CH, K),
                bias: make_bias(CH),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: make_dense_weights(1, CH),
                bias: make_bias(CH),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: make_dense_weights(CH, CH),
                bias: make_bias(CH),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let layers_1: Vec<WaveNetLayer<1, CH, K>> =
        TEST_DILATIONS.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(CH, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_1 = layers_1.len();

    let array1 = WaveNetLayerArray::<1, 1, CH, K, HEAD> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: DenseLayer {
            weights: make_dense_weights(1, CH),
            bias: make_bias(CH),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: make_dense_weights(CH, HEAD),
            bias: make_bias(HEAD),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf1,
        block_size: CH,
        block_buffer: AlignedVec::from_vec(vec![0.0; CH * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, HEAD, K> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv1d_weights(HEAD, HEAD, K),
                bias: make_bias(HEAD),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: make_dense_weights(1, HEAD),
                bias: make_bias(HEAD),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: make_dense_weights(HEAD, HEAD),
                bias: make_bias(HEAD),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let layers_2: Vec<WaveNetLayer<1, HEAD, K>> =
        TEST_DILATIONS.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(HEAD, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_2 = layers_2.len();

    let array2 = WaveNetLayerArray::<CH, 1, HEAD, K, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: DenseLayer {
            weights: make_dense_weights(CH, HEAD),
            bias: make_bias(HEAD),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: make_dense_weights(HEAD, 1),
            bias: make_bias(1),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf2,
        block_size: HEAD,
        block_buffer: AlignedVec::from_vec(vec![0.0; HEAD * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
        prewarm_on_reset: true,
    }
}

/// Builds a dynamic `WaveNetModelDyn` equivalent to `WaveNetModel<CH, K, HEAD>`
/// using the same synthetic weights.
fn build_dynamic_model(ch: usize, k: usize, head: usize) -> WaveNetModelDyn {
    let rf = TEST_DILATIONS.iter().max().unwrap_or(&1) * (k - 1);

    let make_conv1d_dyn = |in_ch: usize, out_ch: usize, dilation: usize| -> Conv1dDyn {
        let interleave_width = select_interleave_width(out_ch);
        Conv1dDyn {
            weights: make_conv1d_weights(in_ch, out_ch, k),
            bias: make_bias(out_ch),
            do_bias: false,
            dilation,
            in_ch,
            out_ch,
            num_blocks: out_ch.div_ceil(interleave_width),
            interleave_width,
            kernel: k,
        }
    };

    let make_dense_dyn = |in_ch: usize, out_ch: usize, do_bias: bool| -> DenseLayerDyn {
        DenseLayerDyn {
            in_ch,
            out_ch,
            weights: make_dense_weights(in_ch, out_ch),
            bias: make_bias(out_ch),
            do_bias,
        }
    };

    // Array 1 layers: COND=1, CH=ch, K=k
    let layers_1: Vec<WaveNetLayerDyn> = TEST_DILATIONS
        .iter()
        .map(|&d| {
            WaveNetLayerDyn::new(
                ch,
                make_conv1d_dyn(ch, ch, d),
                make_dense_dyn(1, ch, false),
                make_dense_dyn(ch, ch, false),
            )
            .unwrap()
        })
        .collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(ch, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_1 = layers_1.len();

    let array1 = WaveNetLayerArrayDyn {
        in_ch: 1,
        cond: 1,
        ch,
        k,
        head,
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: make_dense_dyn(1, ch, false),
        head_rechannel: make_dense_dyn(ch, head, false),
        array_outputs: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf,
        block_size: ch,
        block_buffer: AlignedVec::from_vec(vec![0.0; ch * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    // Array 2 layers: COND=1, CH=head, K=k
    let layers_2: Vec<WaveNetLayerDyn> = TEST_DILATIONS
        .iter()
        .map(|&d| {
            WaveNetLayerDyn::new(
                head,
                make_conv1d_dyn(head, head, d),
                make_dense_dyn(1, head, false),
                make_dense_dyn(head, head, false),
            )
            .unwrap()
        })
        .collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(head, rf, i).expect("Failed to create WaveNetLayerState"))
        .collect();
    let num_layers_2 = layers_2.len();

    let array2 = WaveNetLayerArrayDyn {
        in_ch: ch,
        cond: 1,
        ch: head,
        k,
        head: 1,
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: make_dense_dyn(ch, head, false),
        head_rechannel: make_dense_dyn(head, 1, true),
        array_outputs: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf,
        block_size: head,
        block_buffer: AlignedVec::from_vec(vec![0.0; head * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    WaveNetModelDyn {
        ch,
        k,
        head,
        arrays: vec![array1, array2],
        head_scale: 0.02,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        slimmable_capable: false,
        allowed_channels: None,
        pending_slim_channel: None,
    }
}

/// Magnitude comparison of two model outputs using max absolute error.
/// For bit-exact parity, max_error should be 0.0.
fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

macro_rules! parity_test {
    ($name:ident, $ch:literal, $k:literal, $head:literal) => {
        #[test]
        fn $name() {
            let mut const_model = build_const_generic_model::<$ch, $k, $head>();
            let mut dyn_model = build_dynamic_model($ch, $k, $head);

            const_model.prewarm();
            dyn_model.prewarm();

            let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

            let mut const_out = vec![0.0f32; input.len()];
            let mut dyn_out = vec![0.0f32; input.len()];

            const_model.process(&input, &mut const_out);
            dyn_model.process(&input, &mut dyn_out);

            let err = max_abs_error(&const_out, &dyn_out);
            assert!(
                err < 1e-7,
                "Parity failed for CH={}, K={}, HEAD={}: max_abs_error = {:.3e}",
                $ch,
                $k,
                $head,
                err
            );
        }
    };
}

// Catalog geometries: Standard, Lite, Feather, Nano
parity_test!(test_dynamic_parity_standard, 16, 3, 8);
parity_test!(test_dynamic_parity_lite, 12, 3, 6);
parity_test!(test_dynamic_parity_feather, 8, 3, 4);
parity_test!(test_dynamic_parity_nano, 4, 3, 2);

/// Determinism: two identically built and prewarmed dynamic models must produce
/// identical output for the same input.
#[test]
fn test_dynamic_determinism() {
    let mut model_a = build_dynamic_model(4, 3, 2);
    let mut model_b = build_dynamic_model(4, 3, 2);
    model_a.prewarm();
    model_b.prewarm();

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

    let mut out_a = vec![0.0f32; input.len()];
    let mut out_b = vec![0.0f32; input.len()];

    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    let err = max_abs_error(&out_a, &out_b);
    assert!(
        err < 1e-7,
        "Dynamic model is not deterministic across instances: max_abs_error = {:.3e}",
        err
    );
}

/// All buffers remain finite after prewarm and processing.
#[test]
fn test_dynamic_no_nan() {
    let mut model = build_dynamic_model(4, 3, 2);
    model.prewarm();

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut output = vec![0.0f32; input.len()];
    model.process(&input, &mut output);

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite output at sample {}: {}", i, v);
    }
}
