// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;

fn build_identity_block() -> ConvNetBlock {
    ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block")
}

#[test]
fn test_construction_defaults() {
    let block = build_identity_block();
    assert_eq!(block.conv.in_ch, 1);
    assert_eq!(block.conv.out_ch, 1);
    assert_eq!(block.conv.kernel, 1);
    assert_eq!(block.receptive_field(), 0);
}

#[test]
fn test_receptive_field_computation() {
    let block =
        ConvNetBlock::new(1, 4, 3, 8, false, ActivationType::ReLU, 0).expect("create block");
    assert_eq!(block.receptive_field(), (3 - 1) * 8);
}

#[test]
fn test_process_identity_tanh() {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    let input = [0.5f32];
    let mut output = [0.0f32];

    unsafe {
        block.process_block(&input, &mut output, 1);
    }

    let expected = 0.5f32.tanh();
    assert!((output[0] - expected).abs() < 1e-4);
}

#[test]
fn test_process_with_relu() {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::ReLU, 0).expect("create block");

    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    let input = [0.5f32, -0.3, 0.8, -1.0];
    let mut output = [0.0f32; 4];

    unsafe {
        block.process_block(&input, &mut output, 4);
    }

    assert!((output[0] - 0.5).abs() < 1e-4);
    assert!((output[1] - 0.0).abs() < 1e-4); // RELU clamps
    assert!((output[2] - 0.8).abs() < 1e-4);
    assert!((output[3] - 0.0).abs() < 1e-4);
}

#[test]
fn test_batch_norm_integration() {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, false, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![1.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![2.0f32];
    let bn_offset = vec![1.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    let input = [0.5f32];
    let mut output = [0.0f32];

    unsafe {
        block.process_block(&input, &mut output, 1);
    }

    let conv_out: f32 = 0.5;
    let bn_out: f32 = conv_out * 2.0 + 1.0;
    let expected = bn_out.tanh();
    assert!((output[0] - expected).abs() < 1e-4);
}

#[test]
fn test_multi_frame_causal() {
    let mut block =
        ConvNetBlock::new(1, 1, 2, 1, false, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![0.5f32, 0.0, 0.0, 0.0, 0.5f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    let input = [1.0f32, 2.0, 3.0];
    let mut output = [0.0f32; 3];

    unsafe {
        block.process_block(&input, &mut output, 3);
    }

    let v0: f32 = 0.5 * 1.0;
    let expected_0 = v0.tanh();
    assert!((output[0] - expected_0).abs() < 1e-4);

    let v1: f32 = 0.5 * 1.0 + 0.5 * 2.0;
    let expected_1 = v1.tanh();
    assert!((output[1] - expected_1).abs() < 1e-4);

    let v2: f32 = 0.5 * 2.0 + 0.5 * 3.0;
    let expected_2 = v2.tanh();
    assert!((output[2] - expected_2).abs() < 2e-4);
}

#[test]
fn test_conv_bias() {
    let mut block =
        ConvNetBlock::new(1, 1, 1, 1, true, ActivationType::Tanh, 0).expect("create block");

    let weights = vec![2.0f32, 0.0, 0.0, 0.0];
    block.set_conv_weights(&weights);

    let bias = vec![0.5f32];
    block.set_conv_bias(&bias);

    let bn_scale = vec![1.0f32];
    let bn_offset = vec![0.0f32];
    block.set_bn_params(&bn_scale, &bn_offset).unwrap();

    let input = [1.0f32];
    let mut output = [0.0f32];

    unsafe {
        block.process_block(&input, &mut output, 1);
    }

    let v: f32 = 1.0 * 2.0 + 0.5;
    let expected = v.tanh();
    assert!((output[0] - expected).abs() < 2e-4);
}

#[test]
fn test_struct_alignment() {
    assert_eq!(std::mem::align_of::<ConvNetBlock>(), 64);
}
