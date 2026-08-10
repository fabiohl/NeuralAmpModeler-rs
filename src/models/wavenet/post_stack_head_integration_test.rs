// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Integration tests for Post-stack Head flow.
//!
//! Validates the full flow of PostStackHead within WaveNetModelDyn:
//! arrays → PostStackHead → head_scale.
//!
//! Golden values are hand-computed from the simplified geometry:
//!   CH=1, K=1, HEAD=1, 1 layer, dilation=1, with known weights.

use crate::loader::nam_json::model::HeadConfig;
use crate::math::common::AlignedVec;
use crate::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
use crate::models::wavenet::dense_dyn::DenseLayerDyn;
use crate::models::wavenet::layer_array_dyn::WaveNetLayerArrayDyn;
use crate::models::wavenet::layer_dyn::WaveNetLayerDyn;
use crate::models::wavenet::model_dyn::WaveNetModelDyn;
use crate::models::wavenet::post_stack_head::PostStackHead;

/// Building blocks for the golden-test model.
///
/// **Model trace (single frame, input x):**
///
///   1. rechannel(weight=0.5): state[0] = 0.5·x
///   2. input_mixin(weight=0.0):  mixin = 0.0·x = 0.0
///   3. conv1d(weight=2.0, kernel=1) + mixin:  conv = 2.0·state[0] + 0.0 = x
///   4. tanh_and_overwrite: head_accum = tanh(x)
///   5. one_by_one(weight=0.0, residual): output = state[0] = 0.5·x  (→ array_outputs)
///   6. head_rechannel(weight=1.0): head_out = 1.0·tanh(x) = tanh(x)
///   7. PostStackHead(weight=3.0, kernel=1, ReLU): scratch = ReLU(3.0·tanh(x))
///   8. head_scale(0.5): final = 0.5·ReLU(3.0·tanh(x))
///
/// With kernel=1 everywhere there is no temporal dependency, so multi-frame
/// processing is equivalent to per-frame independent evaluation.
fn build_minimal_model_with_head() -> WaveNetModelDyn {
    let ch: usize = 1;
    let k: usize = 1;
    let head: usize = 1;

    // max_dilation=1, kernel=1 ⇒ RF = 1*(1-1) = 0
    let rf: usize = 0;

    // Interleaved 4-wide: [b=0][k=0][in_c=0][lane0..3] = [2.0, 0, 0, 0]
    let mut conv_weights =
        AlignedVec::new(4, 0.0f32).expect("allocation should succeed for test-sized buffers");
    conv_weights[0] = 2.0;

    let conv1d = Conv1dDyn {
        weights: conv_weights,
        bias: AlignedVec::new(ch, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: false,
        dilation: 1,
        in_ch: ch,
        out_ch: ch,
        num_blocks: 1,
        interleave_width: 4,
        kernel: k,
    };

    let input_mixin = DenseLayerDyn {
        in_ch: 1,
        out_ch: ch,
        weights: AlignedVec::new(ch, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        bias: AlignedVec::new(ch, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: false,
    };

    let one_by_one = DenseLayerDyn {
        in_ch: ch,
        out_ch: ch,
        weights: AlignedVec::new(ch * ch, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        bias: AlignedVec::new(ch, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        do_bias: false,
    };

    let layer = WaveNetLayerDyn::new(ch, conv1d, input_mixin, one_by_one).unwrap();

    let array = WaveNetLayerArrayDyn {
        in_ch: 1,
        cond: 1,
        ch,
        k,
        head,
        layers: vec![layer],
        states: vec![WaveNetLayerState::new(ch, rf, 0).expect("create array state")],
        effective_layers: 1,
        rechannel: DenseLayerDyn {
            in_ch: 1,
            out_ch: ch,
            weights: AlignedVec::from_vec(vec![0.5f32])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::new(ch, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        head_rechannel: DenseLayerDyn {
            in_ch: ch,
            out_ch: head,
            weights: AlignedVec::from_vec(vec![1.0f32])
                .expect("allocation should succeed for test-sized buffers"),
            bias: AlignedVec::new(head, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            do_bias: false,
        },
        array_outputs: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::new(head * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size: rf,
        block_size: ch,
        block_buffer: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
    };

    // PostStackHead: kernel=1, in_ch=1, out_ch=1, weight=3.0, ReLU
    let head_config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: Some("ReLU".to_string()),
        kernel_size: Some(1),
    };
    let mut post_stack_head =
        PostStackHead::from_config(&head_config, 1).expect("create post-stack head");
    // Set weights: single lane 0 = 3.0
    post_stack_head.set_weights(&[3.0f32, 0.0, 0.0, 0.0]);

    WaveNetModelDyn {
        ch,
        k,
        head,
        arrays: vec![array],
        head_scale: 0.5,
        receptive_field_size: 0, // array_rf(0) + head_kernel(1) - 1 = 0
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        post_stack_head: Some(post_stack_head),
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        slimmable_capable: false,
        allowed_channels: None,
        pending_slim_channel: None,
    }
}

/// Golden formula for the minimal model:
///   output(x) = head_scale * ReLU( head_weight * tanh( rechannel_w * conv_w * x ) )
///
/// With chosen weights: output(x) = 0.5 * ReLU( 3.0 * tanh(x) )
#[inline]
fn golden_output(x: f32) -> f32 {
    0.5 * (3.0 * x.tanh()).max(0.0)
}

#[test]
fn test_post_stack_head_golden_single_frame() {
    let mut model = build_minimal_model_with_head();
    model.prewarm();

    let input = [0.5f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);

    let expected = golden_output(0.5);
    assert!(
        (output[0] - expected).abs() < 1e-5,
        "Golden mismatch: expected {:.9}, got {:.9}",
        expected,
        output[0]
    );
}

#[test]
fn test_post_stack_head_golden_multi_frame() {
    let mut model = build_minimal_model_with_head();
    model.prewarm();

    // Mix of positive (ReLU-active) and negative (ReLU-zero) inputs
    let input = [0.5f32, -0.5, 1.0, -0.3, 0.8];
    let mut output = [0.0f32; 5];
    model.process(&input, &mut output);

    for (i, &inp) in input.iter().enumerate() {
        let expected = golden_output(inp);
        assert!(
            (output[i] - expected).abs() < 1e-5,
            "Golden mismatch at frame {i} (input={inp}): expected {expected:.9}, got {:.9}",
            output[i]
        );
    }
}

/// Negative inputs produce zero output through ReLU in the PostStackHead.
#[test]
fn test_post_stack_head_golden_negative_vanishes() {
    let mut model = build_minimal_model_with_head();
    model.prewarm();

    let input = [-2.0f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);

    assert!(
        output[0].abs() < 1e-8,
        "Negative input via ReLU head should vanish, got {}",
        output[0]
    );
}

/// Two identical models with PostStackHead must produce identical output.
#[test]
fn test_post_stack_head_determinism() {
    let mut model_a = build_minimal_model_with_head();
    let mut model_b = build_minimal_model_with_head();
    model_a.prewarm();
    model_b.prewarm();

    let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut out_a = vec![0.0f32; input.len()];
    let mut out_b = vec![0.0f32; input.len()];

    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Non-deterministic output at sample {i}: {a} vs {b}"
        );
    }
}

/// All outputs must be finite after prewarm and processing.
#[test]
fn test_post_stack_head_no_nan() {
    let mut model = build_minimal_model_with_head();
    model.prewarm();

    let input: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut output = vec![0.0f32; input.len()];
    model.process(&input, &mut output);

    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite output at sample {i}: {v}");
    }
}

/// Prewarm should not produce NaN in PostStackHead internal buffers.
#[test]
fn test_post_stack_head_prewarm_no_nan() {
    let mut model = build_minimal_model_with_head();
    model.prewarm();

    let head = model.post_stack_head.as_ref().unwrap();
    for &v in head.state.layer_buffer.iter() {
        assert!(v.is_finite(), "NaN/Inf in head state after prewarm");
    }
}

/// prewarm_samples must include PostStackHead kernel contribution.
/// Without head: prewarm_samples = array_rf (kernel=1 ⇒ 0)
/// With    head: prewarm_samples = array_rf + head_kernel - 1 = 0 + 1 - 1 = 0
/// For kernel > 1 this grows. We test with a separate model using kernel=3.
#[test]
fn test_post_stack_head_prewarm_samples_includes_kernel() {
    use crate::models::NamModel;

    // Without head
    let mut model_no_head = build_minimal_model_with_head();
    model_no_head.post_stack_head = None;
    assert_eq!(model_no_head.prewarm_samples(), 0);

    // With kernel=1 head
    let model_k1 = build_minimal_model_with_head();
    assert_eq!(model_k1.prewarm_samples(), 0);

    // With kernel=3 head: array_rf=0 + 3 - 1 = 2
    let mut model_k3 = build_minimal_model_with_head();
    let head_config_k3 = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: Some("Tanh".to_string()),
        kernel_size: Some(3),
    };
    let mut head_k3 = PostStackHead::from_config(&head_config_k3, 1).expect("create kernel=3 head");
    // Populate weights to avoid zero-data
    let weights = vec![1.0f32; 12]; // num_blocks=1, k=3, in_ch=1, 4 lanes = 12
    head_k3.set_weights(&weights);
    model_k3.post_stack_head = Some(head_k3);
    model_k3.receptive_field_size = 2; // 0 + 3 - 1
    assert_eq!(model_k3.prewarm_samples(), 2);
}

/// Multi-frame processing with kernel=1 and two-arrays model exercises
/// the cascaded head pattern through PostStackHead.
#[test]
fn test_post_stack_head_multi_array_determinism() {
    fn build_two_array_model_with_head() -> WaveNetModelDyn {
        const CH: usize = 2;
        const K: usize = 1;
        const HEAD: usize = 1;
        const RF: usize = 0;

        let make_dense = |in_ch: usize, out_ch: usize, do_bias: bool| -> DenseLayerDyn {
            DenseLayerDyn {
                in_ch,
                out_ch,
                weights: AlignedVec::from_vec(vec![0.01f32; out_ch * in_ch])
                    .expect("allocation should succeed for test-sized buffers"),
                bias: AlignedVec::new(out_ch, 0.0f32)
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias,
            }
        };

        let make_conv = |in_ch: usize, out_ch: usize| -> Conv1dDyn {
            let raw = vec![0.01f32; out_ch * K * in_ch];
            let num_blocks = out_ch.div_ceil(4);
            let mut weights = AlignedVec::new(num_blocks * K * in_ch * 4, 0.0f32)
                .expect("allocation should succeed for test-sized buffers");
            crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
                &raw,
                &mut weights,
                in_ch,
                out_ch,
                K,
            );
            Conv1dDyn {
                weights,
                bias: AlignedVec::new(out_ch, 0.0f32)
                    .expect("allocation should succeed for test-sized buffers"),
                do_bias: false,
                dilation: 1,
                in_ch,
                out_ch,
                num_blocks,
                interleave_width: 4,
                kernel: K,
            }
        };

        let layer_a1 = WaveNetLayerDyn::new(
            CH,
            make_conv(CH, CH),
            make_dense(1, CH, false),
            make_dense(CH, CH, false),
        )
        .unwrap();

        let array1 = WaveNetLayerArrayDyn {
            in_ch: 1,
            cond: 1,
            ch: CH,
            k: K,
            head: HEAD,
            layers: vec![layer_a1],
            states: vec![WaveNetLayerState::new(CH, RF, 0).expect("state")],
            effective_layers: 1,
            rechannel: make_dense(1, CH, false),
            head_rechannel: make_dense(CH, HEAD, false),
            array_outputs: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_accum: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_outputs: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            receptive_field_size: RF,
            block_size: CH,
            block_buffer: AlignedVec::new(CH * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
        };

        let layer_a2 = WaveNetLayerDyn::new(
            HEAD,
            make_conv(HEAD, HEAD),
            make_dense(1, HEAD, false),
            make_dense(HEAD, HEAD, false),
        )
        .unwrap();

        let array2 = WaveNetLayerArrayDyn {
            in_ch: CH,
            cond: 1,
            ch: HEAD,
            k: K,
            head: 1,
            layers: vec![layer_a2],
            states: vec![WaveNetLayerState::new(HEAD, RF, 0).expect("state")],
            effective_layers: 1,
            rechannel: make_dense(CH, HEAD, false),
            head_rechannel: make_dense(HEAD, 1, true),
            array_outputs: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_accum: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            head_outputs: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            receptive_field_size: RF,
            block_size: HEAD,
            block_buffer: AlignedVec::new(HEAD * WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
        };

        // PostStackHead: Tanh, kernel=1, weight=1.0
        let head_config = HeadConfig {
            channels: Some(1),
            bias: Some(false),
            out_channels: Some(1),
            activation: Some("Tanh".to_string()),
            kernel_size: Some(1),
        };
        let mut post = PostStackHead::from_config(&head_config, 1).expect("create post-stack head");
        post.set_weights(&[1.0f32, 0.0, 0.0, 0.0]);

        WaveNetModelDyn {
            ch: CH,
            k: K,
            head: HEAD,
            arrays: vec![array1, array2],
            head_scale: 0.02,
            receptive_field_size: RF, // RF + head_kernel - 1 with kernel=1
            condition_dsp: None,
            condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            post_stack_head: Some(post),
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0f32)
                .expect("allocation should succeed for test-sized buffers"),
            prewarm_on_reset: true,
            slimmable_capable: false,
            allowed_channels: None,
            pending_slim_channel: None,
        }
    }

    let mut model_a = build_two_array_model_with_head();
    let mut model_b = build_two_array_model_with_head();
    model_a.prewarm();
    model_b.prewarm();

    let input: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();
    let mut out_a = vec![0.0f32; input.len()];
    let mut out_b = vec![0.0f32; input.len()];

    model_a.process(&input, &mut out_a);
    model_b.process(&input, &mut out_b);

    for (i, (&a, &b)) in out_a.iter().zip(out_b.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-7,
            "Multi-array non-determinism at sample {i}: {a} vs {b}"
        );
    }

    for &v in &out_a {
        assert!(v.is_finite(), "NaN/Inf in multi-array output");
    }
}
