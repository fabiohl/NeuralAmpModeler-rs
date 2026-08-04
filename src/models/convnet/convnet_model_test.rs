// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;

fn build_single_block_model() -> ConvNetModel {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: None,
    }
}

#[test]
fn test_single_block_process() {
    let mut model = build_single_block_model();

    let input = [0.5f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
}

#[test]
fn test_head_scale() {
    let mut model = build_single_block_model();
    model.head_scale = 2.0;

    let input = [0.5f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    let expected = 2.0 * 0.5f32.tanh();
    assert!((output[0] - expected).abs() < 1e-4);
}

#[test]
fn test_linear_head_flat_cpp_parity() {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");
    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);
    block.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let head_weight = AlignedVec::from_vec(vec![1.0f32]).expect("head_weight alloc");
    let head_bias = AlignedVec::from_vec(vec![0.0f32]).expect("head_bias alloc");

    let linear_head = LinearHead {
        weight: head_weight,
        bias: head_bias,
        in_ch: 1,
        out_ch: 1,
    };

    let mut model = ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: Some(linear_head),
    };

    let input = [0.5f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);

    let expected = 0.5f32.tanh();
    assert!(
        (output[0] - expected).abs() < 1e-4,
        "FlatCpp parity: head_scale=1.0 must produce identity gain. output={}, expected={}",
        output[0],
        expected
    );
}

#[test]
fn test_empty_model_outputs_silence() {
    let mut model = ConvNetModel {
        blocks: vec![],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: None,
    };

    let input = [0.5f32, -0.3];
    let mut output = [1.0f32; 2];
    model.process(&input, &mut output);
    assert_eq!(output, [0.0, 0.0]);
}

#[test]
fn test_empty_input_noop() {
    let mut model = build_single_block_model();

    let input: [f32; 0] = [];
    let mut output: [f32; 0] = [];

    model.process(&input, &mut output);
}

#[test]
fn test_prewarm_no_panic() {
    let mut model = build_single_block_model();
    model.prewarm();

    let input = [0.0f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

#[test]
fn test_two_block_chain() {
    let mut block0 =
        ConvNetBlock::new(1, 2, 1, 1, false, ActivationType::ReLU, 0).expect("block 0");
    let weights0 = vec![1.0f32, 2.0, 0.0, 0.0];
    block0.set_conv_weights(&weights0);
    block0
        .set_bn_params(&[1.0f32, 1.0], &[0.0f32, 0.0])
        .unwrap();

    let mut block1 =
        ConvNetBlock::new(2, 1, 1, 1, false, ActivationType::Tanh, 1).expect("block 1");
    let weights1 = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
    block1.set_conv_weights(&weights1);
    block1.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let model = ConvNetModel {
        blocks: vec![block0, block1],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(2 * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: None,
    };

    let mut model = model;
    let input = [2.0f32];
    let mut output = [0.0f32];

    model.process(&input, &mut output);

    let _b0_c0: f32 = 2.0 * 1.0;
    let _b0_c1: f32 = 2.0 * 2.0;
    let b1_out: f32 = 4.0 * 0.5 + 2.0 * 0.5;
    let expected = b1_out.tanh();
    assert!(
        (output[0] - expected).abs() < 5e-4,
        "output[0]={}, expected={}, diff={}",
        output[0],
        expected,
        (output[0] - expected).abs()
    );
}

#[test]
fn test_post_stack_head_integration() {
    use crate::loader::nam_json::model::HeadConfig;
    use crate::models::wavenet::PostStackHead;

    let mut block = ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);
    block.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let head_config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: Some("Tanh".to_string()),
        kernel_size: Some(1),
    };
    let head = PostStackHead::from_config(&head_config, 1).expect("head");

    let model = ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: Some(head),
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: None,
    };

    let mut model = model;
    model
        .post_stack_head
        .as_mut()
        .unwrap()
        .set_weights(&[1.0, 0.0, 0.0, 0.0]);

    let input = [0.5f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!((output[0] - 0.5f32.tanh()).abs() < 1e-4);
}

#[test]
fn test_prewarm_with_head() {
    use crate::loader::nam_json::model::HeadConfig;

    let mut block = ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("block");
    block.set_conv_weights(&[1.0, 0.0, 0.0, 0.0]);
    block.set_bn_params(&[1.0f32], &[0.0f32]).unwrap();

    let head_config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: None,
        kernel_size: Some(1),
    };
    let head = PostStackHead::from_config(&head_config, 1).expect("head");

    let mut model = ConvNetModel {
        blocks: vec![block],
        head_scale: 1.0,
        receptive_field_size: 0,
        post_stack_head: Some(head),
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        linear_head: None,
    };

    model
        .post_stack_head
        .as_mut()
        .unwrap()
        .set_weights(&[1.0, 0.0, 0.0, 0.0]);
    model.prewarm();

    let input = [0.0f32];
    let mut output = [0.0f32];
    model.process(&input, &mut output);
    assert!(output[0].is_finite());
}

#[test]
fn test_convnet_prewarm_fixed_point_invariant() {
    fn create_model() -> ConvNetModel {
        let mut block =
            ConvNetBlock::new(1, 1, 2, 2, false, ActivationType::Tanh, 0).expect("create block");

        let weights = vec![1.0f32, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        block.set_conv_weights(&weights);
        block.set_bn_params(&[1.0f32], &[0.5f32]).unwrap();

        let rf = block.receptive_field();
        ConvNetModel {
            blocks: vec![block],
            head_scale: 1.0,
            receptive_field_size: rf,
            post_stack_head: None,
            head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
                .expect("head_output_scratch"),
            scratch_a: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).expect("scratch_a"),
            scratch_b: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).expect("scratch_b"),
            prewarm_on_reset: true,
            linear_head: None,
        }
    }

    let mut model_a = create_model();
    model_a.prewarm();

    const NUM_TEST: usize = 64;
    let zeros = vec![0.0f32; NUM_TEST];
    let mut out_a = vec![0.0f32; NUM_TEST];
    model_a.process(&zeros, &mut out_a);

    let dc_a = out_a[0];
    for (i, &val) in out_a.iter().enumerate().skip(1) {
        assert!(
            (val - dc_a).abs() < 1e-6,
            "post-prewarm output drift: out_a[{}]={} vs out_a[0]={}, diff={:.2e}",
            i,
            val,
            dc_a,
            (val - dc_a).abs()
        );
    }

    let mut model_b = create_model();
    const CONVERGE: usize = 256;
    let zeros_b = vec![0.0f32; CONVERGE];
    let mut out_b = vec![0.0f32; CONVERGE];
    model_b.process(&zeros_b, &mut out_b);

    let dc_b_steady = out_b[CONVERGE - 1];
    assert!(
        (dc_a - dc_b_steady).abs() < 1e-4,
        "prewarm fixed point = {} diverges from explicit convergence = {}, diff = {:.2e}",
        dc_a,
        dc_b_steady,
        (dc_a - dc_b_steady).abs()
    );
}

#[test]
fn test_struct_alignment() {
    assert_eq!(std::mem::align_of::<ConvNetModel>(), 64);
}
