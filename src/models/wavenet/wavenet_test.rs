// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::models::a2::activations::ActivationType;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d::Conv1d;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::dense::DenseLayer;
use crate::models::wavenet::layer::WaveNetLayer;
use crate::models::wavenet::layer_array::WaveNetLayerArray;
use crate::models::wavenet::model::*;

/// Helper: create f32 synthetic dense weights for test models.
fn test_dense_f32(in_ch: usize, out_ch: usize) -> AlignedVec<f32> {
    AlignedVec::from_vec(vec![0.01f32; out_ch * in_ch])
        .expect("allocation should succeed for test-sized buffers")
}

/// Builds a minimal WaveNetModel<4, 3, 2> for tests with static, controlled data.
/// This function serves as a "mock" (simulated model) for unit tests.
fn build_tiny_wavenet() -> WaveNetModel<4, 3, 2> {
    // Layer factory for Array 1 (Main Array).
    // Generics: <COND=1, CH=4, K=3>.
    // In WaveNet, each layer is a functional unit that processes the dilated signal.
    let make_layer_a1 = |dilation: usize| -> WaveNetLayer<1, 4, 3> {
        let raw_weights = vec![0.01f32; 4 * 3 * 4];
        let mut weights =
            AlignedVec::new(48, 0.0f32).expect("allocation should succeed for test-sized buffers");
        crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            4, // IN
            4, // OUT
            3, // K
        );
        WaveNetLayer {
            // Dilated Causal Convolution enables capturing long temporal dependencies
            // without linearly increasing the number of parameters.
            conv1d: Conv1d {
                // Dimensions: OUT * K * IN = 4 * 3 * 4.
                // Here, IN=CH because the layer receives the signal from previous layers.
                weights,
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation,
            },
            // input_mixin injects conditioning (e.g., timbre metadata) into the signal.
            // Dimensions: OUT * IN = 4 * 1.
            input_mixin: DenseLayer {
                weights: test_dense_f32(1, 4),
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(4, 4),
                bias: AlignedVec::from_vec(vec![0.0; 4])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(4 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(4 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    // Array2: CH=2 (=HEAD), layers with COND=1, CH=2
    // Layer factory for Array 2 (Head Array).
    // Generics: <COND=1, CH=2, K=3>.
    // This array usually has fewer channels and focuses on final audio refinement.
    let make_layer_a2 = |dilation: usize| -> WaveNetLayer<1, 2, 3> {
        let raw_weights = vec![0.01f32; 2 * 3 * 2];
        let mut weights =
            AlignedVec::new(24, 0.0f32).expect("allocation should succeed for test-sized buffers");
        crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
            &raw_weights,
            &mut weights,
            2, // IN
            2, // OUT
            3, // K
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
                weights: test_dense_f32(1, 2),
                // Dimensions: OUT * IN = 2 * 1.
                bias: AlignedVec::from_vec(vec![0.0; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            one_by_one: DenseLayer {
                weights: test_dense_f32(2, 2),
                // Dimensions: OUT * IN = 2 * 2.
                bias: AlignedVec::from_vec(vec![0.0; 2])
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
            },
            activation: ActivationType::Tanh,
            scratch_mixin: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
            scratch_conv: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("scratch alloc"),
        }
    };

    // Define the dilation pattern. Exponential growth (1, 2, 4...)
    // is what allows WaveNet to have a vast receptive field with few layers.
    let dilations_1 = [1, 2, 4];
    let dilations_2 = [1, 2, 4];

    // Receptive Field (RF) calculation: determines how many past samples influence the present.
    // Simplified formula: max_dilation * (kernel_size - 1).
    let rf1 = *dilations_1.iter().max().unwrap_or(&1) * (3 - 1);
    let rf2 = *dilations_2.iter().max().unwrap_or(&1) * (3 - 1);

    // Manual array construction with explicit const generics.
    // Array1 (Main Receptive Field): Ensures primary feature extraction.
    // For each dilation, we build a layer and allocate its internal state (historical buffer).
    let layers_1: Vec<WaveNetLayer<1, 4, 3>> =
        dilations_1.iter().map(|&d| make_layer_a1(d)).collect();
    let states_1: Vec<WaveNetLayerState> = (0..layers_1.len())
        .map(|i| WaveNetLayerState::new(4, rf1, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let num_layers_1 = layers_1.len();
    let array1 = WaveNetLayerArray::<1, 1, 4, 3, 2> {
        layers: layers_1,
        states: states_1,
        effective_layers: num_layers_1,
        // Rechannel: Projects raw input (Mono/Stereo) to the internal dimension (Channels).
        rechannel: DenseLayer {
            weights: test_dense_f32(1, 4),
            bias: AlignedVec::from_vec(vec![0.0; 4])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(4, 2),
            bias: AlignedVec::from_vec(vec![0.0; 2])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        // Pre-allocated output buffers to ensure RT-Safety (Zero Alloc in the loop).
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
    };

    // Array2 (Head Definition): The secondary array acts on refined final predictions.
    // IN=4(=CH of array1), COND=1, CH=2(=HEAD1), K=3, HEAD2=1
    // `head_rechannel` defines the final data transition.
    let layers_2: Vec<WaveNetLayer<1, 2, 3>> =
        dilations_2.iter().map(|&d| make_layer_a2(d)).collect();
    let states_2: Vec<WaveNetLayerState> = (0..layers_2.len())
        .map(|i| WaveNetLayerState::new(2, rf2, i).expect("Failed to create WaveNetLayerState"))
        .collect();

    let num_layers_2 = layers_2.len();
    let array2 = WaveNetLayerArray::<4, 1, 2, 3, 1> {
        layers: layers_2,
        states: states_2,
        effective_layers: num_layers_2,
        // Projects Array 1 output (HEAD1=2) to Array 2 dimension (CH2=2).
        rechannel: DenseLayer {
            weights: test_dense_f32(4, 2),
            bias: AlignedVec::from_vec(vec![0.0; 2])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayer {
            weights: test_dense_f32(2, 1),
            bias: AlignedVec::from_vec(vec![0.0; 1])
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: true, // Enable bias for final DC offset correction.
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
    };

    // WaveNetModel orchestrates the array cascade and applies the final gain.
    WaveNetModel {
        array1,
        array2,
        head_scale: 0.02,
        // The global RF is the largest among the arrays (usually Array1's RF dominates).
        receptive_field_size: rf1.max(rf2),
        prewarm_on_reset: true,
    }
}

#[path = "conv1d_dyn_test.rs"]
mod conv1d_dyn_test;
#[path = "conv1d_test.rs"]
mod conv1d_test;
#[path = "dense_test.rs"]
mod dense_test;
#[path = "dynamic_parity_test.rs"]
mod dynamic_parity_test;
#[path = "post_stack_head_integration_test.rs"]
mod post_stack_head_integration_test;
#[path = "wavenet_ch12_diagnostic_test.rs"]
mod wavenet_ch12_diagnostic_test;
#[path = "wavenet_sub_test.rs"]
mod wavenet_sub_test;

#[test]
fn wavenet_ringbuffer_alignment() {
    // MirroredBuffer size MUST be multiple of channels
    // for all SKU-relevant channel counts post-P1 fix.
    // CH=12 and CH=6 were the pre-fix offenders (1024%12=4, 1024%6=4).
    let rf = 128;
    for &ch in &[2, 3, 4, 6, 8, 12, 16] {
        let state = WaveNetLayerState::new(ch, rf, 0)
            .unwrap_or_else(|e| panic!("WaveNetLayerState::new failed for CH={ch}: {e}"));
        let size = state.layer_buffer.size();
        assert!(
            size.is_multiple_of(ch),
            "Ringbuffer size ({size}) not multiple of channels ({ch}); \
             alignment guard-rail violated"
        );
        assert!(
            size >= rf
                + crate::models::wavenet::common::LAYER_ARRAY_BUFFER_PADDING
                    * crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES,
            "Ringbuffer size ({size}) too small for RF={rf} with CH={ch}"
        );
    }
}
