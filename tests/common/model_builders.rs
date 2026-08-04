// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared synthetic model builders for integration tests.
//!
//! Centralizes duplicated builder functions that were previously scattered across
//! `soak_test.rs` and `wavenet_prewarm_edge.rs`.

use neural_amp_modeler_rs::math::common::AlignedVec;
use neural_amp_modeler_rs::models::a2::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, WaveNetA2, a2_weight_count,
};
use neural_amp_modeler_rs::models::wavenet::{
    Conv1d, DenseLayer, WAVENET_MAX_NUM_FRAMES, WaveNetLayer, WaveNetLayerArray, WaveNetLayerState,
    WaveNetModel,
};

// =============================================================================
// build_soak_wavenet — WaveNetModel<16, 3, 8> (Standard topology)
// =============================================================================

/// Helper to build a synthetic WaveNetModel<16, 3, 8> for soak testing.
///
/// Uses 10 dilations matching the Standard topology (Array1 only).
/// All weights are initialized with small values (0.01) so the audio does not
/// explode immediately, allowing the FPU to process real values across all
/// layers for millions of iterations.
pub fn build_soak_wavenet() -> WaveNetModel<16, 3, 8> {
    let make_layer = |dilation: usize| -> WaveNetLayer<1, 16, 3> {
        let _raw_weights = vec![0.01f32; 16 * 3 * 16];
        let conv_weights = {
            let mut fw = AlignedVec::new(_raw_weights.len(), 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
                &_raw_weights,
                &mut fw,
                16,
                16,
                3,
            );
            fw
        };
        WaveNetLayer {
            conv1d: Conv1d {
                weights: conv_weights,
                bias: AlignedVec::from_vec(vec![0.001; 16])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: true,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 16])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 16])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 16 * 16])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 16])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(16 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(16 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    // Standard topology: 10 dilations per array
    let dilations: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let rf: usize = dilations.iter().map(|&d| (3 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 16, 3>> = dilations.iter().map(|&d| make_layer(d)).collect();
    let states_1: Vec<WaveNetLayerState> = dilations
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            WaveNetLayerState::new(16, (3 - 1) * d, i).expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let layers_1_len = layers_1.len();
    let array1 = WaveNetLayerArray::<1, 1, 16, 3, 8> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 16])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 16])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 8 * 16])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf,
        block_size: 16,
        block_buffer: AlignedVec::from_vec(vec![0.0; 16 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: layers_1_len,
    };

    // Array 2: Final spectral refinement (CH=8, K=3)
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 8, 3> {
        let _raw_weights = vec![0.01f32; 8 * 3 * 8];
        let conv_weights = {
            let mut fw = AlignedVec::new(_raw_weights.len(), 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
                &_raw_weights,
                &mut fw,
                8,
                8,
                3,
            );
            fw
        };
        WaveNetLayer {
            conv1d: Conv1d {
                weights: conv_weights,
                bias: AlignedVec::from_vec(vec![0.001; 8])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: true,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 8])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 8])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 8 * 8])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 8])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(8 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(8 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let layers_2: Vec<WaveNetLayer<1, 8, 3>> = [1, 2].iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| {
            WaveNetLayerState::new(8, 2 * (3 - 1), i).expect("Failed to create WaveNetLayerState")
        })
        .collect();

    let layers_2_len = layers_2.len();
    let array2 = WaveNetLayerArray::<16, 1, 8, 3, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 16 * 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 8])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 8])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: 2,
        block_size: 8,
        block_buffer: AlignedVec::from_vec(vec![0.0; 8 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: layers_2_len,
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.1,
        receptive_field_size: rf,
        prewarm_on_reset: true,
    }
}

// =============================================================================
// build_k5_large_rf_wavenet — WaveNetModel<4, 5, 2> (large RF, K=5)
// =============================================================================

/// Builds a WaveNetModel<4, 5, 2> with K=5 and max dilation 512.
pub fn build_k5_large_rf_wavenet() -> WaveNetModel<4, 5, 2> {
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 5> {
        let raw_weights = vec![0.01f32; 4 * 5 * 4];
        let mut weights = AlignedVec::new(5 * 4 * 4, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            4,
            4,
            5,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 4 * 4])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(4 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(4 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 5> {
        let raw_weights = vec![0.01f32; 2 * 5 * 2];
        let mut weights = AlignedVec::new(5 * 2 * 4, 0.0f32)
            .expect("allocation should succeed for test-sized buffers");
        neural_amp_modeler_rs::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            2,
            2,
            5,
        );
        WaveNetLayer {
            conv1d: Conv1d {
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            input_mixin: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: AlignedVec::from_vec(vec![0.01f32; 2 * 2])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::from_vec(vec![0.0; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            scratch_mixin: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    let dilations_1 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dilations_2 = [1, 2, 4, 8];

    let rf1: usize = dilations_1.iter().map(|&d| (5 - 1) * d).sum();
    let rf2: usize = dilations_2.iter().map(|&d| (5 - 1) * d).sum();

    let layers_1: Vec<WaveNetLayer<1, 4, 5>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array1 = WaveNetLayerArray::<1, 1, 4, 5, 2> {
        layers: layers_1,
        states: states_1,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 4])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 4])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 2 * 4])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 2])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf1,
        block_size: 4,
        block_buffer: AlignedVec::from_vec(vec![0.0; 4 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: dilations_1.len(),
    };

    let layers_2: Vec<WaveNetLayer<1, 2, 5>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let array2 = WaveNetLayerArray::<4, 1, 2, 5, 1> {
        layers: layers_2,
        states: states_2,
        rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 4 * 2])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 2])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: AlignedVec::from_vec(vec![0.01f32; 2])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true,
        },
        array_outputs: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::from_vec(vec![0.0; WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf2,
        block_size: 2,
        block_buffer: AlignedVec::from_vec(vec![0.0; 2 * WAVENET_MAX_NUM_FRAMES])
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: dilations_2.len(),
    };

    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        receptive_field_size: rf1.max(rf2),
        prewarm_on_reset: true,
    }
}

// =============================================================================
// A2 Architecture — WaveNetA2<CH> synthetic helpers
// =============================================================================

/// A2 receptive field from canonical constants (matching `a2_receptive_field()` in model.rs).
pub fn a2_rf() -> usize {
    let mut rf = 0usize;
    for i in 0..23 {
        rf += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
    }
    rf + (A2_HEAD_KERNEL_SIZE - 1)
}

/// Assembles synthetic A2 weight stream with exact count.
pub fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
    let num_weights = a2_weight_count::<CH>();
    let mut w = Vec::with_capacity(num_weights);

    w.extend(std::iter::repeat_n(weight_val, CH));

    for &k in &A2_KERNEL_SIZES {
        w.extend(std::iter::repeat_n(weight_val, CH * CH * k));
        w.extend(std::iter::repeat_n(0.0f32, CH));
        w.extend(std::iter::repeat_n(weight_val, CH));
        w.extend(std::iter::repeat_n(weight_val, CH * CH));
        w.extend(std::iter::repeat_n(0.0f32, CH));
    }

    w.extend(std::iter::repeat_n(weight_val, 16 * CH));
    w.push(0.0);
    w.push(0.02);

    assert_eq!(w.len(), num_weights, "A2 weight count mismatch");
    w
}

/// Builds a WaveNetA2<CH> with all weights set to `weight_val` and zero biases.
/// Uses the exact weight count expected by `set_weights`.
pub fn build_synth_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
    let weights = a2_synth_weights::<CH>(weight_val);
    let mut model = WaveNetA2::<CH>::new().expect("Failed to create WaveNetA2");
    model
        .set_weights(&weights)
        .expect("Failed to set A2 synthetic weights");
    model
}
