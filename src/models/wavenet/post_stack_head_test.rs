// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::loader::nam_json::model::HeadConfig;

#[test]
fn test_parse_activation_known_names() {
    assert_eq!(parse_activation("Tanh"), ActivationType::Tanh);
    assert_eq!(parse_activation("HardTanh"), ActivationType::HardTanh);
    assert_eq!(parse_activation("FastTanh"), ActivationType::FastTanh);
    assert_eq!(parse_activation("ReLU"), ActivationType::ReLU);
    assert_eq!(parse_activation("Sigmoid"), ActivationType::Sigmoid);
    assert_eq!(parse_activation("SiLU"), ActivationType::SiLU);
    assert_eq!(parse_activation("HardSwish"), ActivationType::HardSwish);
    assert_eq!(parse_activation("Softsign"), ActivationType::Softsign);
}

#[test]
fn test_parse_activation_unknown_falls_back_to_tanh() {
    assert_eq!(parse_activation("UnknownAct"), ActivationType::Tanh);
    assert_eq!(parse_activation(""), ActivationType::Tanh);
}

#[test]
fn test_construction_defaults() {
    let config = HeadConfig {
        channels: None,
        bias: None,
        out_channels: None,
        activation: None,
        kernel_size: None,
    };

    let head = PostStackHead::from_config(&config, 8).expect("should build head");
    assert_eq!(head.in_channels(), 8);
    assert_eq!(head.out_channels(), 1);
    assert_eq!(head.receptive_field(), 3);
    assert!(!head.conv.do_bias);
    assert_eq!(head.activation, ActivationType::Tanh);
}

#[test]
fn test_construction_explicit() {
    let config = HeadConfig {
        channels: Some(16),
        bias: Some(true),
        out_channels: Some(2),
        activation: Some("ReLU".to_string()),
        kernel_size: Some(5),
    };

    let head = PostStackHead::from_config(&config, 8).expect("should build head");
    assert_eq!(head.in_channels(), 16);
    assert_eq!(head.out_channels(), 2);
    assert_eq!(head.receptive_field(), 5);
    assert!(head.conv.do_bias);
    assert_eq!(head.activation, ActivationType::ReLU);
}

/// Builds a PostStackHead with identity weights (in_ch=1, out_ch=1, kernel=1).
/// No bias, Tanh activation.
fn build_identity_head() -> PostStackHead {
    let in_ch = 1;
    let out_ch: usize = 1;
    let kernel: usize = 1;

    let num_blocks = out_ch.div_ceil(4);
    let weights_len = num_blocks * kernel * in_ch * 4;
    let mut weights = AlignedVec::new(weights_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    weights[0] = 1.0;

    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    PostStackHead {
        conv: Conv1dDyn {
            weights,
            bias,
            do_bias: false,
            dilation: 1,
            in_ch,
            out_ch,
            num_blocks,
            interleave_width: 4,
            kernel,
        },
        activation: ActivationType::Tanh,
        state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
        scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
    }
}

#[test]
fn test_process_single_frame_identity() {
    let mut head = build_identity_head();

    let input = [0.5f32];
    let mut output = [0.0f32];

    unsafe {
        head.process_block(&input, &mut output, 1);
    }

    assert!((output[0] - 0.462117).abs() < 1e-4); // tanh(0.5)
}

#[test]
fn test_process_multi_frame() {
    let mut head = build_identity_head();

    let input = [0.1f32, -0.2, 0.3, -0.4, 0.5];
    let mut output = [0.0f32; 5];

    unsafe {
        head.process_block(&input, &mut output, 5);
    }

    for (i, &inp) in input.iter().enumerate() {
        let expected = inp.tanh();
        assert!(
            (output[i] - expected).abs() < 1e-4,
            "frame {i}: expected {expected}, got {}",
            output[i]
        );
    }
}

#[test]
fn test_prewarm_no_nan() {
    let mut head = build_identity_head();
    head.prewarm();

    let input = [0.0f32];
    let mut output = [0.0f32];
    unsafe {
        head.process_block(&input, &mut output, 1);
    }

    assert!(output[0].is_finite());
}

#[test]
fn test_deterministic() {
    let mut head1 = build_identity_head();
    let mut head2 = build_identity_head();

    head1.prewarm();
    head2.prewarm();

    let input = [0.3f32, -0.1, 0.7, 0.2, -0.9];
    let mut out1 = [0.0f32; 5];
    let mut out2 = [0.0f32; 5];

    unsafe {
        head1.process_block(&input, &mut out1, 5);
        head2.process_block(&input, &mut out2, 5);
    }

    assert_eq!(out1, out2);
}

#[test]
fn test_process_with_weights_and_activation() {
    let in_ch: usize = 1;
    let out_ch: usize = 1;
    let kernel: usize = 3;
    let num_blocks = out_ch.div_ceil(4);
    let weights_len = num_blocks * kernel * in_ch * 4;

    let mut weights = AlignedVec::new(weights_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    weights[0] = 0.5;
    weights[4] = 0.3;
    weights[8] = 0.2;

    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let mut head = PostStackHead {
        conv: Conv1dDyn {
            weights,
            bias,
            do_bias: false,
            dilation: 1,
            in_ch,
            out_ch,
            num_blocks,
            interleave_width: 4,
            kernel,
        },
        activation: ActivationType::Tanh,
        state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
        scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
    };

    // Frame 0: k=0 reads buf[-2]=0, k=1 reads buf[-1]=0, k=2 reads buf[+0]=1.0
    //   output = 0.5*0 + 0.3*0 + 0.2*1.0 = 0.2 => tanh(0.2)
    // Frame 1: k=0 reads buf[-1]=0, k=1 reads buf[+0]=1.0, k=2 reads buf[+1]=2.0
    //   output = 0.5*0 + 0.3*1.0 + 0.2*2.0 = 0.7 => tanh(0.7)
    // Frame 2: k=0 reads buf[+0]=1.0, k=1 reads buf[+1]=2.0, k=2 reads buf[+2]=3.0
    //   output = 0.5*1.0 + 0.3*2.0 + 0.2*3.0 = 1.7 => tanh(1.7)
    let input = [1.0f32, 2.0, 3.0];
    let mut output = [0.0f32; 3];

    unsafe {
        head.process_block(&input, &mut output, 3);
    }

    let expected = [0.2f32.tanh(), 0.7f32.tanh(), 1.7f32.tanh()];
    for (i, &exp) in expected.iter().enumerate() {
        assert!(
            (output[i] - exp).abs() < 1e-4,
            "frame {i}: expected {exp}, got {}",
            output[i]
        );
    }
}

#[test]
fn test_set_weights_and_bias() {
    let config = HeadConfig {
        channels: Some(1),
        bias: Some(false),
        out_channels: Some(1),
        activation: None,
        kernel_size: Some(1),
    };

    let mut head = PostStackHead::from_config(&config, 1).expect("create head");

    let new_weights = vec![1.0f32, 0.0, 0.0, 0.0];
    head.set_weights(&new_weights);

    let new_bias = vec![0.0f32];
    head.set_bias(&new_bias);

    let input = [0.5f32];
    let mut output = [0.0f32];
    unsafe {
        head.process_block(&input, &mut output, 1);
    }

    assert!((output[0] - 0.462117).abs() < 1e-4); // tanh(0.5)
}

#[test]
fn test_multi_channel_in_out() {
    let in_ch: usize = 2;
    let out_ch: usize = 3;
    let kernel: usize = 1;
    let num_blocks = out_ch.div_ceil(4);
    let weights_len = num_blocks * kernel * in_ch * 4;

    let mut weights = AlignedVec::new(weights_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    // Block 0: out_ch 0,1,2,3 (but out_ch=3 so only 0,1,2)
    // Weights layout: [b][k][in_c][lane]
    // b=0, k=0, in_c=0, lane=0 => weight for out_c=0, in_c=0
    weights[0] = 1.0; // out_c=0, in_c=0
    weights[1] = 0.0; // out_c=0, in_c=1
    // in_c=1: [b=0][k=0][in_c=1][lane=0]
    weights[4] = 0.5; // out_c=0, in_c=1
    // in_c=0: [b=0][k=0][in_c=0][lane=1]
    weights[1] = 0.0; // out_c=1, in_c=0
    weights[5] = 2.0; // out_c=1, in_c=1
    // in_c=0: [b=0][k=0][in_c=0][lane=2]
    weights[2] = 0.3; // out_c=2, in_c=0
    weights[6] = 0.0; // out_c=2, in_c=1

    let bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let mut head = PostStackHead {
        conv: Conv1dDyn {
            weights,
            bias,
            do_bias: false,
            dilation: 1,
            in_ch,
            out_ch,
            num_blocks,
            interleave_width: 4,
            kernel,
        },
        activation: ActivationType::Tanh,
        state: WaveNetLayerState::new(in_ch, kernel, 0).expect("create state"),
        scratch: AlignedVec::new(out_ch * WAVENET_MAX_NUM_FRAMES, 0.0f32)
            .expect("allocation should succeed for test-sized buffers"),
    };

    // Frame 0: in=[A=1.0, B=0.5]
    //   out[0] = 1.0*1.0 + 0.5*0.5 = 1.0 + 0.25 = 1.25 => tanh(1.25)
    //   out[1] = 0.0*1.0 + 2.0*0.5 = 1.0 => tanh(1.0)
    //   out[2] = 0.3*1.0 + 0.0*0.5 = 0.3 => tanh(0.3)
    let input = [1.0f32, 0.5];
    let mut output = [0.0f32; 3];

    unsafe {
        head.process_block(&input, &mut output, 1);
    }

    assert!((output[0] - 1.25f32.tanh()).abs() < 1e-4);
    assert!((output[1] - 1.0f32.tanh()).abs() < 1e-4);
    assert!((output[2] - 0.3f32.tanh()).abs() < 1e-4);
}
