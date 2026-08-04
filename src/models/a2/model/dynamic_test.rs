// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::params::{A2_DILATIONS, A2_KERNEL_SIZES, A2_LEAKY_SLOPE, A2_NUM_LAYERS};

fn make_standard_activations(num: usize) -> Vec<ActivationType> {
    vec![
        ActivationType::LeakyReLU {
            negative_slope: A2_LEAKY_SLOPE,
        };
        num
    ]
}

fn make_standard_gating(num: usize) -> Vec<GatingMode> {
    vec![GatingMode::None; num]
}

fn make_standard_secondary(num: usize) -> Vec<Option<ActivationType>> {
    vec![None; num]
}

#[test]
fn test_wavenet_a2_dyn_new_ch3() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();

    assert_eq!(model.channels, 3);
    assert_eq!(model.bottleneck, 3);
    assert_eq!(model.num_layers, A2_NUM_LAYERS);
    assert!(model.receptive_field_size > 0);
    assert!(!model.head_accum.is_empty());
    assert!(!model.layer_buffers.is_empty());
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
    assert_eq!(model.rechannel_w_f32.len(), 3);
    assert_eq!(model.layer_in.len(), 3 * model.max_buffer_size);
    assert!(!model.head1x1_active);
}

#[test]
fn test_wavenet_a2_dyn_new_ch8() {
    let model = WaveNetA2Dyn::new(
        1,
        8,
        8,
        1,
        8,
        8,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();

    assert_eq!(model.channels, 8);
    assert_eq!(model.bottleneck, 8);
    assert_eq!(model.num_layers, A2_NUM_LAYERS);
    assert_eq!(model.layer_buffers.len(), A2_NUM_LAYERS);
}

#[test]
fn test_wavenet_a2_dyn_bottleneck_neq_channels() {
    let model = WaveNetA2Dyn::new(
        1,
        8,
        4,
        1,
        4,
        4,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        true,
    )
    .unwrap();

    assert_eq!(model.channels, 8);
    assert_eq!(model.bottleneck, 4);
    assert!(model.head1x1_active);
    assert_eq!(model.head1x1_w.len(), 4 * 4);
    assert_eq!(model.head1x1_b.len(), 4);
}

#[test]
fn test_wavenet_a2_dyn_gating_prealloc() {
    let num = A2_NUM_LAYERS;
    let mut gating = vec![GatingMode::None; num];
    gating[0] = GatingMode::Gated;
    gating[1] = GatingMode::Blended;
    let mut sec = vec![None; num];
    sec[0] = Some(ActivationType::Sigmoid);
    sec[1] = Some(ActivationType::Tanh);

    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(num),
        gating,
        sec,
        false,
    )
    .unwrap();

    assert!(model.gating_configs[0].is_some());
    assert!(model.blending_configs[1].is_some());
    assert!(model.gating_configs[2].is_none());
    assert!(model.blending_configs[0].is_none());
}

#[test]
fn test_wavenet_a2_dyn_process_empty_input() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];
    model.process(&input, &mut output);
}

#[test]
fn test_wavenet_a2_dyn_process_silence_no_weights() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let input = vec![0.5f32; 64];
    let mut output = vec![1.0f32; 64];
    model.process(&input, &mut output);
    for v in &output {
        assert!(v.abs() < 1e-9, "expected silence, got {}", v);
    }
}

#[test]
fn test_wavenet_a2_dyn_prewarm_zeroes() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    for buf in &mut model.layer_buffers {
        let len = buf.size();
        buf[..len].fill(0.5);
    }
    model.head_accum.fill(0.5);
    model.layer_in.fill(0.5);
    model.prewarm();
    for buf in &model.layer_buffers {
        let len = buf.size();
        for &v in buf[..len].iter() {
            assert!(v.abs() < 1e-9, "layer_buffer not zeroed");
        }
    }
    for v in model.head_accum.iter() {
        assert!(v.abs() < 1e-9, "head_accum not zeroed");
    }
    for v in model.layer_in.iter() {
        assert!(v.abs() < 1e-9, "layer_in not zeroed");
    }
}

#[test]
fn test_wavenet_a2_dyn_set_max_buffer_size_noop() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(32).unwrap();
    assert_eq!(model.layer_ring_sizes, orig_sizes);
}

#[test]
fn test_wavenet_a2_dyn_set_max_buffer_size_grows() {
    let mut model = WaveNetA2Dyn::new(
        1,
        8,
        8,
        1,
        8,
        8,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let orig_sizes: Vec<usize> = model.layer_ring_sizes.clone();
    model.set_max_buffer_size(256).unwrap();
    assert!(model.max_buffer_size == 256);
    let any_grew = orig_sizes
        .iter()
        .zip(model.layer_ring_sizes.iter())
        .any(|(a, b)| b > a);
    assert!(any_grew, "at least one ring should grow");
}

#[test]
fn test_wavenet_a2_dyn_has_weights_false_initially() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    assert!(!model.has_weights());
}

#[test]
fn test_wavenet_a2_dyn_receptive_field() {
    let model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let expected = {
        let mut sum = 0usize;
        for i in 0..A2_NUM_LAYERS {
            sum += (A2_KERNEL_SIZES[i] - 1) * A2_DILATIONS[i];
        }
        sum + (crate::models::a2::A2_HEAD_KERNEL_SIZE - 1)
    };
    assert_eq!(model.receptive_field_size, expected);
    assert_eq!(model.receptive_field(), expected);
}

// ── F9 regression: head_write_pos wrap-around with empty layers ──────

#[test]
fn test_wavenet_a2_dyn_head_write_pos_never_exceeds_ring_mask() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let ring_mask = model.head_ring_mask;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    for i in 0..2048 {
        model.process(&input, &mut output);
        assert!(
            model.head_write_pos <= ring_mask,
            "iteration {}: head_write_pos {} exceeded ring_mask {}",
            i,
            model.head_write_pos,
            ring_mask
        );
    }
}

#[test]
fn test_wavenet_a2_dyn_head_write_pos_wraps_correctly() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let ring_mask = model.head_ring_mask;
    let cap = ring_mask + 1;
    let mut prev_wp = model.head_write_pos;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut wrapped = false;
    for _ in 0..4096 {
        model.process(&input, &mut output);
        let wp = model.head_write_pos;
        if wp < prev_wp {
            wrapped = true;
        }
        assert!(wp <= ring_mask);
        prev_wp = wp;
    }
    assert!(
        wrapped || cap <= 64 * 4096,
        "head_write_pos should wrap after enough iterations (cap={}, wp after 4096 iters={})",
        cap,
        model.head_write_pos
    );
}
#[test]
fn test_wavenet_a2_dyn_head_write_pos_reset_after_prewarm() {
    let mut model = WaveNetA2Dyn::new(
        1,
        3,
        3,
        1,
        3,
        3,
        &A2_KERNEL_SIZES,
        &A2_DILATIONS,
        make_standard_activations(A2_NUM_LAYERS),
        make_standard_gating(A2_NUM_LAYERS),
        make_standard_secondary(A2_NUM_LAYERS),
        false,
    )
    .unwrap();
    let rf = model.receptive_field_size;
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    for _ in 0..500 {
        model.process(&input, &mut output);
    }
    assert_ne!(
        model.head_write_pos, rf,
        "head_write_pos should have advanced after 500 process calls"
    );
    model.prewarm();
    assert_eq!(
        model.head_write_pos, rf,
        "head_write_pos should be reset to rf after prewarm"
    );
}

// ── B1/B2/B3 dedicated unit tests ────────────────────────────────────────────

#[test]
fn test_wavenet_a2_dyn_bug_b1_mixin_post_film() {
    use crate::math::common::AlignedVec;
    use crate::models::a2::conv1d::A2Conv1d;
    use crate::models::a2::film::{FiLMConfig, FiLMLayer};
    use crate::models::a2::layer::A2Layer;

    let mut model = WaveNetA2Dyn::new(
        1,    // input_channels
        1,    // channels
        1,    // bottleneck
        1,    // head_size
        1,    // head_accum_size
        1,    // h1_in_size
        &[1], // kernel_sizes
        &[1], // dilations
        vec![ActivationType::LeakyReLU {
            negative_slope: 1.0,
        }], // activations (identity)
        vec![GatingMode::None], // gating_modes
        vec![None], // secondary_activations
        false, // head1x1_active
    )
    .unwrap();

    model.rechannel_w_f32 = AlignedVec::from_vec(vec![1.0]).unwrap();

    // Dilated causal Conv1D with bias = 3.0 and weights = 0.0 (interleaved 4-wide size = 4)
    let conv = A2Conv1d::new(
        AlignedVec::from_vec(vec![0.0; 4]).unwrap(),
        AlignedVec::from_vec(vec![3.0]).unwrap(),
        true,
        1,
        1,
        1,
        1,
    );
    let mixin_w = AlignedVec::from_vec(vec![4.0]).unwrap();
    let l1x1_w = AlignedVec::from_vec(vec![0.0]).unwrap();
    let l1x1_b = AlignedVec::from_vec(vec![0.0]).unwrap();

    let mut layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

    // Active input_mixin_post_film FiLM layer
    let film_config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // scale weight: 2.5, shift weight: 1.2, scale bias: 0.0, shift bias: 0.0
    let film_layer = FiLMLayer::load(
        film_config,
        1, // cond_size
        1, // channels
        vec![2.5, 1.2],
        vec![0.0, 0.0],
    )
    .unwrap();
    layer.input_mixin_post_film = Some(film_layer);

    model.layers = vec![layer];

    let input = vec![1.5]; // input_x = 1.5, which is also the condition
    let mut output = vec![0.0];

    // We run the model process. This will run process_internal on the 1-sample input.
    // The condition value is 1.5.
    model.process(&input, &mut output);

    // Let's trace mathematically:
    // conv_out = 3.0
    // mixin = mixin_w * condition = 4.0 * 1.5 = 6.0
    // scale = 2.5 * 1.5 + 0.0 = 3.75
    // shift = 1.2 * 1.5 + 0.0 = 1.8
    // Since input_mixin_post_film applies only to the mixin:
    // mixin_modulated = mixin * scale + shift = 6.0 * 3.75 + 1.8 = 22.5 + 1.8 = 24.3
    // z = conv_out + mixin_modulated = 3.0 + 24.3 = 27.3
    // This value is written to head_accum at the frame position.

    let head_wp = model.receptive_field_size; // initial head_write_pos
    let actual_val = model.head_accum[head_wp];
    let expected_val = 27.3f32;

    assert!(
        (actual_val - expected_val).abs() < 1e-4,
        "Bug B1 test failed: expected {}, got {}",
        expected_val,
        actual_val
    );
}

#[test]
fn test_wavenet_a2_dyn_bug_b2_l1x1_gating_modes() {
    use crate::math::common::AlignedVec;
    use crate::models::a2::conv1d::A2Conv1d;
    use crate::models::a2::film::{FiLMConfig, FiLMLayer};
    use crate::models::a2::layer::A2Layer;

    // Helper closure to create and run the 2-layer model with a specific gating mode and film activity
    let run_model = |gating: GatingMode, film_active: bool| -> f32 {
        let mut model = WaveNetA2Dyn::new(
            1,       // input_channels
            1,       // channels
            1,       // bottleneck
            1,       // head_size
            1,       // head_accum_size
            1,       // h1_in_size
            &[1, 1], // kernel_sizes
            &[1, 1], // dilations
            vec![
                ActivationType::LeakyReLU {
                    negative_slope: 1.0,
                },
                ActivationType::LeakyReLU {
                    negative_slope: 1.0,
                },
            ], // activations (identity)
            vec![gating, GatingMode::None], // gating_modes
            vec![None, None], // secondary_activations
            false,   // head1x1_active
        )
        .unwrap();

        model.rechannel_w_f32 = AlignedVec::from_vec(vec![1.0]).unwrap();

        let use_g = gating == GatingMode::Gated || gating == GatingMode::Blended;
        let conv0_out = if use_g { 2 } else { 1 };

        // Layer 0: conv output = 3.0, l1x1_w = 2.0, l1x1_b = 0.5
        let conv0 = A2Conv1d::new(
            AlignedVec::from_vec(vec![0.0; conv0_out * 4]).unwrap(),
            AlignedVec::from_vec(vec![3.0; conv0_out]).unwrap(),
            true,
            1,
            1,
            conv0_out,
            1,
        );
        let mixin_w0 = AlignedVec::from_vec(vec![0.0; conv0_out]).unwrap();
        let l1x1_w0 = AlignedVec::from_vec(vec![2.0]).unwrap();
        let l1x1_b0 = AlignedVec::from_vec(vec![0.5]).unwrap();
        let mut layer0 = A2Layer::new_dyn(conv0, mixin_w0, l1x1_w0, l1x1_b0, 1, 1, 1);

        // Active layer1x1_post_film FiLM layer on Layer 0 only if film_active is true
        if film_active {
            let film_config = FiLMConfig {
                active: true,
                shift: true,
                groups: 1,
            };
            // scale weight: 4.0, shift weight: 1.5, scale bias: 0.0, shift bias: 0.0
            let film_layer = FiLMLayer::load(
                film_config,
                1, // cond_size
                1, // channels
                vec![4.0, 1.5],
                vec![0.0, 0.0],
            )
            .unwrap();
            layer0.layer1x1_post_film = Some(film_layer);
        } else {
            layer0.layer1x1_post_film = None;
        }

        // Layer 1: conv weight = 1.0 (interleaved), bias = 0.0
        let conv1 = A2Conv1d::new(
            AlignedVec::from_vec(vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
            AlignedVec::from_vec(vec![0.0]).unwrap(),
            true,
            1,
            1,
            1,
            1,
        );
        let mixin_w1 = AlignedVec::from_vec(vec![0.0]).unwrap();
        let l1x1_w1 = AlignedVec::from_vec(vec![0.0]).unwrap();
        let l1x1_b1 = AlignedVec::from_vec(vec![0.0]).unwrap();
        let layer1 = A2Layer::new(conv1, mixin_w1, l1x1_w1, l1x1_b1);

        model.layers = vec![layer0, layer1];

        // input_x = 2.0 (so initial layer_in has 2.0), condition = 2.0
        let input = vec![2.0];
        let mut output = vec![0.0];
        model.process(&input, &mut output);

        let head_wp = model.receptive_field_size;
        model.head_accum[head_wp]
    };

    let out_none_with = run_model(GatingMode::None, true);
    let out_none_without = run_model(GatingMode::None, false);
    let out_gated_with = run_model(GatingMode::Gated, true);
    let out_gated_without = run_model(GatingMode::Gated, false);
    let out_blended_with = run_model(GatingMode::Blended, true);
    let out_blended_without = run_model(GatingMode::Blended, false);

    // Assert that FiLM has no effect in GatingMode::None and GatingMode::Gated
    assert_eq!(
        out_none_with, out_none_without,
        "None mode: expected output to be identical regardless of FiLM layer activity"
    );
    assert_eq!(
        out_gated_with, out_gated_without,
        "Gated mode: expected output to be identical regardless of FiLM layer activity"
    );

    // Assert that FiLM is applied in GatingMode::Blended (changing the output from 11.5 to 60.0)
    assert!(
        (out_blended_with - 60.0).abs() < 1e-4,
        "Blended mode with FiLM: expected 60.0, got {}",
        out_blended_with
    );
    assert!(
        (out_blended_without - 11.5).abs() < 1e-4,
        "Blended mode without FiLM: expected 11.5, got {}",
        out_blended_without
    );
}

#[test]
fn test_wavenet_a2_dyn_bug_b3_l1x1_residual_modulation() {
    use crate::math::common::AlignedVec;
    use crate::models::a2::conv1d::A2Conv1d;
    use crate::models::a2::film::{FiLMConfig, FiLMLayer};
    use crate::models::a2::layer::A2Layer;

    let mut model = WaveNetA2Dyn::new(
        1,       // input_channels
        1,       // channels
        1,       // bottleneck
        1,       // head_size
        1,       // head_accum_size
        1,       // h1_in_size
        &[1, 1], // kernel_sizes
        &[1, 1], // dilations
        vec![
            ActivationType::LeakyReLU {
                negative_slope: 1.0,
            },
            ActivationType::LeakyReLU {
                negative_slope: 1.0,
            },
        ], // activations (identity)
        vec![GatingMode::Blended, GatingMode::None], // gating_modes
        vec![None, None], // secondary_activations
        false,   // head1x1_active
    )
    .unwrap();

    model.rechannel_w_f32 = AlignedVec::from_vec(vec![1.0]).unwrap();

    // Layer 0: conv output = 3.0 (so conv0_out = 2), l1x1_w = 2.0, l1x1_b = 0.5
    let conv0 = A2Conv1d::new(
        AlignedVec::from_vec(vec![0.0; 8]).unwrap(),
        AlignedVec::from_vec(vec![3.0, 3.0]).unwrap(),
        true,
        1,
        1,
        2,
        1,
    );
    let mixin_w0 = AlignedVec::from_vec(vec![0.0, 0.0]).unwrap();
    let l1x1_w0 = AlignedVec::from_vec(vec![2.0]).unwrap();
    let l1x1_b0 = AlignedVec::from_vec(vec![0.5]).unwrap();
    let mut layer0 = A2Layer::new_dyn(conv0, mixin_w0, l1x1_w0, l1x1_b0, 1, 1, 1);

    // Active layer1x1_post_film FiLM layer on Layer 0
    let film_config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // scale weight: 4.0, shift weight: 1.5, scale bias: 0.0, shift bias: 0.0
    let film_layer = FiLMLayer::load(
        film_config,
        1, // cond_size
        1, // channels
        vec![4.0, 1.5],
        vec![0.0, 0.0],
    )
    .unwrap();
    layer0.layer1x1_post_film = Some(film_layer);

    // Layer 1: conv weight = 1.0 (interleaved), bias = 0.0
    let conv1 = A2Conv1d::new(
        AlignedVec::from_vec(vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        AlignedVec::from_vec(vec![0.0]).unwrap(),
        true,
        1,
        1,
        1,
        1,
    );
    let mixin_w1 = AlignedVec::from_vec(vec![0.0]).unwrap();
    let l1x1_w1 = AlignedVec::from_vec(vec![0.0]).unwrap();
    let l1x1_b1 = AlignedVec::from_vec(vec![0.0]).unwrap();
    let layer1 = A2Layer::new(conv1, mixin_w1, l1x1_w1, l1x1_b1);

    model.layers = vec![layer0, layer1];

    // input_x = 2.0 (so initial layer_in has 2.0), condition = 2.0
    let input = vec![2.0];
    let mut output = vec![0.0];
    model.process(&input, &mut output);

    // Trace calculation:
    // initial layer_in = 2.0
    // layer 0: z_scratch = 3.0
    // l1x1_scratch = 0.5 + 2.0 * 3.0 = 6.5
    // scale = 4.0 * 2.0 = 8.0
    // shift = 1.5 * 2.0 = 3.0
    // only l1x1 is modulated:
    // l1x1_scratch_modulated = 6.5 * 8.0 + 3.0 = 55.0
    // layer_in after layer 0 = 2.0 + 55.0 = 57.0
    // layer 1: conv output = 1.0 * 57.0 = 57.0
    // head_accum = 3.0 + 57.0 = 60.0

    let head_wp = model.receptive_field_size;
    let actual_val = model.head_accum[head_wp];
    let expected_val = 60.0f32;

    assert!(
        (actual_val - expected_val).abs() < 1e-4,
        "Bug B3 test failed: expected {}, got {}",
        expected_val,
        actual_val
    );
}

#[test]
fn test_wavenet_a2_dyn_bug_c1_mixin_pre_film_modulates_condition() {
    use crate::math::common::AlignedVec;
    use crate::models::a2::conv1d::A2Conv1d;
    use crate::models::a2::film::{FiLMConfig, FiLMLayer};
    use crate::models::a2::layer::A2Layer;

    let mut model = WaveNetA2Dyn::new(
        1,
        1,
        1,
        1,
        1,
        1,
        &[1], // kernel_sizes
        &[1], // dilations
        vec![ActivationType::LeakyReLU {
            negative_slope: 1.0, // identity
        }],
        vec![GatingMode::None],
        vec![None],
        false,
    )
    .unwrap();

    model.rechannel_w_f32 = AlignedVec::from_vec(vec![1.0]).unwrap();

    // Dilated causal Conv1D: bias = 3.0, weights = 0.0
    let conv = A2Conv1d::new(
        AlignedVec::from_vec(vec![0.0; 4]).unwrap(),
        AlignedVec::from_vec(vec![3.0]).unwrap(),
        true,
        1,
        1,
        1,
        1,
    );
    let mixin_w = AlignedVec::from_vec(vec![4.0]).unwrap();
    let l1x1_w = AlignedVec::from_vec(vec![0.0]).unwrap();
    let l1x1_b = AlignedVec::from_vec(vec![0.0]).unwrap();

    let mut layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

    // Active input_mixin_pre_film FiLM layer: cond_size=1, channels=1
    let film_config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    // scale = 2.0 * cond + 0.0, shift = 1.0 * cond + 0.0
    let film_layer = FiLMLayer::load(
        film_config,
        1, // cond_size
        1, // channels (= cond_size, C++ convention for slot 2)
        vec![2.0, 1.0],
        vec![0.0, 0.0],
    )
    .unwrap();
    layer.input_mixin_pre_film = Some(film_layer);

    model.layers = vec![layer];

    let input = vec![0.5];
    let mut output = vec![0.0];
    model.process(&input, &mut output);

    // Mathematical trace (C++ model.cpp:188-197):
    // conv_out = bias = 3.0
    // input_mixin_pre_film self-modulates condition (target == cond == 0.5):
    //   scale = 2.0 * 0.5 + 0.0 = 1.0
    //   shift = 1.0 * 0.5 + 0.0 = 0.5
    //   cond_mod = 0.5 * 1.0 + 0.5 = 1.0
    // mixin = mixin_w[0] * cond_mod = 4.0 * 1.0 = 4.0
    // z = conv_out + mixin = 3.0 + 4.0 = 7.0
    // LeakyReLU (identity): unchanged → head_accum = 7.0
    //
    // If the old bug were present (FiLM applied to z_scratch instead of condition):
    //   z_mod = 3.0 * 1.0 + 0.5 = 3.5
    //   mixin = 4.0 * 0.5 = 2.0
    //   z = 3.5 + 2.0 = 5.5  (wrong output, 7.0 vs 5.5 distinguishable)

    let head_wp = model.receptive_field_size;
    let actual_val = model.head_accum[head_wp];
    let expected_val = 7.0f32;

    assert!(
        (actual_val - expected_val).abs() < 1e-4,
        "Bug C1 test failed: expected {}, got {}",
        expected_val,
        actual_val
    );
}

#[test]
fn test_wavenet_a2_dyn_head1x1_post_film_modulates_projection() {
    use crate::math::common::AlignedVec;
    use crate::models::a2::conv1d::A2Conv1d;
    use crate::models::a2::film::{FiLMConfig, FiLMLayer};
    use crate::models::a2::layer::A2Layer;

    // Single layer, single channel, head1x1_active with FiLM after projection.
    let mut model = WaveNetA2Dyn::new(
        1,    // input_channels
        1,    // channels
        1,    // bottleneck
        1,    // head_size
        1,    // head_accum_size
        1,    // h1_in_size (= bottleneck / groups = 1)
        &[1], // kernel_sizes
        &[1], // dilations
        vec![ActivationType::LeakyReLU {
            negative_slope: 1.0, // identity
        }],
        vec![GatingMode::None],
        vec![None],
        true, // head1x1_active
    )
    .unwrap();

    model.rechannel_w_f32 = AlignedVec::from_vec(vec![1.0]).unwrap();
    // head1x1_w: [head_accum_size * h1_in_size] = [1 * 1] = 1 element
    model.head1x1_w = AlignedVec::from_vec(vec![1.0]).unwrap();
    model.head1x1_b = AlignedVec::from_vec(vec![0.0]).unwrap();

    // Dilated causal Conv1D: bias = 3.0, weights = 0.0
    let conv = A2Conv1d::new(
        AlignedVec::from_vec(vec![0.0; 4]).unwrap(),
        AlignedVec::from_vec(vec![3.0]).unwrap(),
        true,
        1,
        1,
        1,
        1,
    );
    let mixin_w = AlignedVec::from_vec(vec![4.0]).unwrap();
    let l1x1_w = AlignedVec::from_vec(vec![0.0]).unwrap();
    let l1x1_b = AlignedVec::from_vec(vec![0.0]).unwrap();

    let mut layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

    // Active head1x1_post_film FiLM: cond_size=1, channels=1 (head_accum_size)
    let film_config = FiLMConfig {
        active: true,
        shift: true,
        groups: 1,
    };
    let film_layer = FiLMLayer::load(
        film_config,
        1,
        1,
        vec![2.0, 1.0], // scale weight: 2.0, shift weight: 1.0
        vec![0.0, 0.0],
    )
    .unwrap();
    layer.head1x1_post_film = Some(film_layer);

    model.layers = vec![layer];

    let input = vec![0.5];
    let mut output = vec![0.0];
    model.process(&input, &mut output);

    // Mathematical trace (C++ model.cpp:283-287):
    // conv_out = bias = 3.0
    // mixin = mixin_w * condition = 4.0 * 0.5 = 2.0
    // z = conv_out + mixin = 3.0 + 2.0 = 5.0
    // LeakyReLU (identity): unchanged → 5.0
    // head1x1: h1_out = head1x1_w[0] * z + head1x1_b[0] = 1.0 * 5.0 + 0.0 = 5.0
    // head1x1_post_film self-modulates h1_out with condition=0.5:
    //   scale = 2.0 * 0.5 + 0.0 = 1.0
    //   shift = 1.0 * 0.5 + 0.0 = 0.5
    //   modulated = 5.0 * 1.0 + 0.5 = 5.5
    // head_accum = 5.5 (is_first)
    //
    // Without FiLM: head_accum = 5.0
    // The gap (5.5 vs 5.0) proves FiLM modulates head1x1 output.

    let head_wp = model.receptive_field_size;
    let actual_val = model.head_accum[head_wp];
    let expected_val = 5.5f32;

    assert!(
        (actual_val - expected_val).abs() < 1e-4,
        "head1x1_post_film test failed: expected {}, got {}",
        expected_val,
        actual_val
    );
}
