// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common;
use common::{compute_mse, generate_sine_440hz};

use neural_amp_modeler_rs::loader::dispatcher::wavenet::select_interleave_width;
use neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide;
use neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_8wide;
use neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_16wide;
use neural_amp_modeler_rs::math::common::AlignedVec;
use neural_amp_modeler_rs::models::a2::activations::ActivationType;
use neural_amp_modeler_rs::models::wavenet::{
    Conv1d, DenseLayer, WAVENET_MAX_NUM_FRAMES, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState,
    WaveNetModel,
};

fn test_dense_f32(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.01f32; out_ch * in_ch])
        .expect("allocation should succeed for test-sized buffers")
}

fn make_conv_weights(in_ch: usize, out_ch: usize, k: usize) -> AlignedVec<f32> {
    let raw_weights = vec![0.01f32; out_ch * k * in_ch];
    let interleave_width = select_interleave_width(out_ch);
    let num_blocks = out_ch.div_ceil(interleave_width);
    let padded_total = num_blocks * interleave_width * in_ch * k;
    let mut weights = AlignedVec::new(padded_total, 0.0)
        .expect("allocation should succeed for test-sized buffers");
    match interleave_width {
        16 => {
            transpose_conv1d_interleaved_16wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        8 => {
            transpose_conv1d_interleaved_8wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
        _ => {
            transpose_conv1d_interleaved_4wide(&raw_weights, &mut weights, in_ch, out_ch, k);
        }
    }
    weights
}

fn build_tiny_lite_wavenet() -> WaveNetModel<12, 3, 6> {
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 12, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv_weights(12, 12, 3),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: test_dense_f32(1, 12),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(12, 12),
                bias: AlignedVec::from_vec(vec![0.0; 12])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(12 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(12 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 6, 3> {
        WaveNetLayer {
            conv1d: Conv1d {
                weights: make_conv_weights(6, 6, 3),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: test_dense_f32(1, 6),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(6, 6),
                bias: AlignedVec::from_vec(vec![0.0; 6])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(6 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(6 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let dilations_1: Vec<usize> = (0..10).map(|i| 1 << i).collect();
    let dilations_2: Vec<usize> = (0..10).map(|i| 1 << i).collect();

    let rf1: usize = dilations_1.iter().map(|&d| (3 - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (3 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 12, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| {
            WaveNetLayerState::new(12, (3 - 1) * dilations_1[i], i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let num_layers_1 = layers_1.len();
    let array1 = WaveNetLayerArray::<1, 1, 12, 3, 6> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        rechannel: DenseLayer {
            weights: test_dense_f32(1, 12),
            bias: AlignedVec::from_vec(vec![0.0; 12])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(12, 6),
            bias: AlignedVec::from_vec(vec![0.0; 6])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 12 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 12 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf1,
        block_size: 12,
        block_buffer: AlignedVec::from_vec(vec![0.0; 12 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
    };

    let layers_2: Vec<WaveNetLayer<1, 6, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| {
            WaveNetLayerState::new(6, (3 - 1) * dilations_2[i], i)
                .expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let num_layers_2 = layers_2.len();
    let array2 = WaveNetLayerArray::<12, 1, 6, 3, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        rechannel: DenseLayer {
            weights: test_dense_f32(12, 6),
            bias: AlignedVec::from_vec(vec![0.0; 6])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(6, 1),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf2,
        block_size: 6,
        block_buffer: AlignedVec::from_vec(vec![0.0; 6 * WAVENET_MAX_NUM_FRAMES])
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

#[test]
// T1.2 fix: MirroredBuffer::new_aligned guarantees size_elements % channels == 0,
// so the ring-buffer wrap period aligns exactly with the mirror period.
// T1.3: hardened to assert! < 1e-7 (synthetic model gives MSE~1e-20 post-fix).
fn test_wavenet_lite_block_invariance() {
    let num_samples = 16384;
    let input = generate_sine_440hz(num_samples);

    let mut ref_model = build_tiny_lite_wavenet();
    ref_model.prewarm();
    let mut ref_output = vec![0.0f32; num_samples];
    process_in_blocks_lite(&mut ref_model, &input, &mut ref_output, 1);

    let block_sizes = [16, 32, 64];

    for &bs in &block_sizes {
        let mut test_model = build_tiny_lite_wavenet();
        test_model.prewarm();
        let mut test_output = vec![0.0f32; num_samples];
        process_in_blocks_lite(&mut test_model, &input, &mut test_output, bs);

        let test_mse = compute_mse(&ref_output, &test_output);

        eprintln!("P1 block invariance: Lite CH=12 bs=1 vs bs={bs} MSE={test_mse:.6e}");

        assert!(
            test_mse < 1e-7,
            "Block invariance violated (CH=12 ring-buffer desync): bs={bs} MSE={test_mse:.6e}"
        );
    }
}

fn process_in_blocks_lite(
    model: &mut WaveNetModel<12, 3, 6>,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}
