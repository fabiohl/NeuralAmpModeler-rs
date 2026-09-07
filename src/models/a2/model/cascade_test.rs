// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::activations::ActivationType;
use crate::models::a2::conv1d::A2Conv1d;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::layer::A2Layer;
use crate::models::a2::model::dynamic::WaveNetA2Dyn;

fn make_activations(num: usize) -> Vec<ActivationType> {
    vec![
        ActivationType::LeakyReLU {
            negative_slope: 0.01,
        };
        num
    ]
}

fn make_gating(num: usize) -> Vec<GatingMode> {
    vec![GatingMode::None; num]
}

fn make_secondary(num: usize) -> Vec<Option<ActivationType>> {
    vec![None; num]
}

fn make_test_dyn_array(
    in_ch: usize,
    channels: usize,
    head_size: usize,
    head_k: usize,
) -> WaveNetA2Dyn {
    let mut model = WaveNetA2Dyn::new(
        in_ch,
        channels,
        channels,
        head_size,
        channels,
        channels,
        head_k,
        &[1],
        &[1],
        make_activations(1),
        make_gating(1),
        make_secondary(1),
    )
    .expect("synthetic dynamic model creation failed");

    let num_blocks = channels.div_ceil(4);
    let total_padded = num_blocks * 4 * channels;
    let conv = A2Conv1d::new(
        AlignedVec::from_vec(vec![0.5; total_padded]).unwrap(),
        AlignedVec::from_vec(vec![0.0; channels]).unwrap(),
        true,
        1,
        channels,
        channels,
        1,
    );
    let mixin_w = AlignedVec::from_vec(vec![1.0; channels]).unwrap();
    let l1x1_w = AlignedVec::from_vec(vec![0.5; channels * channels]).unwrap();
    let l1x1_b = AlignedVec::from_vec(vec![0.0; channels]).unwrap();
    let layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);
    model.layers = vec![layer];

    // Initialize head_rechannel weights, biases and scale for multichannel heads
    if head_size > 1 {
        let hw_len = head_size * head_k * model.head_accum_size;
        let mut hw = vec![0.0f32; hw_len];
        for (i, v) in hw.iter_mut().enumerate() {
            *v = (i as f32 + 1.0) * 0.01;
        }
        model.head_rechannel_w = AlignedVec::from_vec(hw).unwrap();

        let mut hb = vec![0.0f32; head_size];
        for (i, v) in hb.iter_mut().enumerate() {
            *v = (i as f32) * 0.05;
        }
        model.head_rechannel_b = AlignedVec::from_vec(hb).unwrap();

        let hs = vec![1.0f32; head_size];
        model.head_rechannel_scale = AlignedVec::from_vec(hs).unwrap();
    }

    model
}

#[test]
fn test_cascade_residual_projection_row_major() {
    // 3 input channels -> 4 output channels
    let mut arr = make_test_dyn_array(3, 4, 1, 1);

    // Row-major matrix: c * src_channels + ic
    // Row 0 (c=0): [1.0, 2.0, 3.0]
    // Row 1 (c=1): [4.0, 5.0, 6.0]
    // Row 2 (c=2): [0.1, 0.2, 0.3]
    // Row 3 (c=3): [10.0, 20.0, 30.0]
    #[rustfmt::skip]
    let weights = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        0.1, 0.2, 0.3,
        10.0, 20.0, 30.0,
    ];
    arr.rechannel_w_f32 = AlignedVec::from_vec(weights).unwrap();

    // 2 frames of 3 channels each
    #[rustfmt::skip]
    let residual = vec![
        0.5, -1.0, 2.0,  // frame 0
        1.0,  0.0, 1.0,  // frame 1
    ];

    arr.cascade_write_residual_input(&residual, 2, 3);

    // Frame 0:
    // c=0: 0.5*1.0 + (-1.0)*2.0 + 2.0*3.0 = 0.5 - 2.0 + 6.0 = 4.5
    // c=1: 0.5*4.0 + (-1.0)*5.0 + 2.0*6.0 = 2.0 - 5.0 + 12.0 = 9.0
    // c=2: 0.5*0.1 + (-1.0)*0.2 + 2.0*0.3 = 0.05 - 0.2 + 0.6 = 0.45
    // c=3: 0.5*10.0 + (-1.0)*20.0 + 2.0*30.0 = 5.0 - 20.0 + 60.0 = 45.0
    assert!((arr.layer_in[0] - 4.5).abs() < 1e-6);
    assert!((arr.layer_in[1] - 9.0).abs() < 1e-6);
    assert!((arr.layer_in[2] - 0.45).abs() < 1e-6);
    assert!((arr.layer_in[3] - 45.0).abs() < 1e-6);

    // Frame 1:
    // c=0: 1.0*1.0 + 0.0*2.0 + 1.0*3.0 = 4.0
    // c=1: 1.0*4.0 + 0.0*5.0 + 1.0*6.0 = 10.0
    // c=2: 1.0*0.1 + 0.0*0.2 + 1.0*0.3 = 0.4
    // c=3: 1.0*10.0 + 0.0*20.0 + 1.0*30.0 = 40.0
    assert!((arr.layer_in[4] - 4.0).abs() < 1e-6);
    assert!((arr.layer_in[5] - 10.0).abs() < 1e-6);
    assert!((arr.layer_in[6] - 0.4).abs() < 1e-6);
    assert!((arr.layer_in[7] - 40.0).abs() < 1e-6);
}

#[test]
fn test_process_rechannel_prescale_row_major() {
    let mut arr = make_test_dyn_array(3, 4, 1, 1);

    #[rustfmt::skip]
    let weights = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        0.1, 0.2, 0.3,
        10.0, 20.0, 30.0,
    ];
    arr.rechannel_w_f32 = AlignedVec::from_vec(weights).unwrap();

    let input = vec![0.5, -1.0, 2.0];
    arr.rechannel_prescale(&input, 0, 1);

    assert!((arr.layer_in[0] - 4.5).abs() < 1e-6);
    assert!((arr.layer_in[1] - 9.0).abs() < 1e-6);
    assert!((arr.layer_in[2] - 0.45).abs() < 1e-6);
    assert!((arr.layer_in[3] - 45.0).abs() < 1e-6);
}

#[test]
fn test_cascade_multichannel_stride_chunk_invariance() {
    // Array 0: 1 in_ch -> 3 ch, head_size = 3
    let mut arr0_a = make_test_dyn_array(1, 3, 3, 1);
    arr0_a.rechannel_w_f32 = AlignedVec::from_vec(vec![0.5, -0.3, 0.8]).unwrap();

    // Array 1: 3 in_ch -> 4 ch, head_size = 8
    let mut arr1_a = make_test_dyn_array(3, 4, 8, 1);
    #[rustfmt::skip]
    let rw1 = vec![
        0.1, 0.2, 0.3,
        -0.4, 0.5, -0.6,
        0.7, -0.8, 0.9,
        -0.1, 0.2, -0.3,
    ];
    arr1_a.rechannel_w_f32 = AlignedVec::from_vec(rw1.clone()).unwrap();

    let mut cascade_full =
        WaveNetA2Cascade::try_new(vec![arr0_a, arr1_a], None, 1).expect("cascade creation failed");
    cascade_full.set_max_buffer_size(512).unwrap();

    let mut arr0_b = make_test_dyn_array(1, 3, 3, 1);
    arr0_b.rechannel_w_f32 = AlignedVec::from_vec(vec![0.5, -0.3, 0.8]).unwrap();
    let mut arr1_b = make_test_dyn_array(3, 4, 8, 1);
    arr1_b.rechannel_w_f32 = AlignedVec::from_vec(rw1).unwrap();

    let mut cascade_chunked =
        WaveNetA2Cascade::try_new(vec![arr0_b, arr1_b], None, 1).expect("cascade creation failed");
    cascade_chunked.set_max_buffer_size(512).unwrap();

    // 256 input frames
    let num_frames = 256;
    let input: Vec<f32> = (0..num_frames)
        .map(|i| (i as f32 * 0.05).sin() * 0.5)
        .collect();

    // Head size of last array is 8, so output is 256 * 8 floats
    let mut out_full = vec![0.0f32; num_frames * 8];
    cascade_full.process(&input, &mut out_full);

    // Process chunked in 4 chunks of 64 frames (matching WAVENET_MAX_NUM_FRAMES)
    let mut out_chunked = vec![0.0f32; num_frames * 8];
    for chunk_idx in 0..4 {
        let f_start = chunk_idx * 64;
        let f_end = f_start + 64;
        let o_start = f_start * 8;
        let o_end = f_end * 8;
        cascade_chunked.process(&input[f_start..f_end], &mut out_chunked[o_start..o_end]);
    }

    // Output must match bit-for-bit across all channels and frames
    let mut max_diff = 0.0f32;
    for i in 0..(num_frames * 8) {
        let diff = (out_full[i] - out_chunked[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        assert_eq!(
            out_full[i].to_bits(),
            out_chunked[i].to_bits(),
            "Mismatch at sample {} (frame {}, channel {}): full={}, chunked={}",
            i,
            i / 8,
            i % 8,
            out_full[i],
            out_chunked[i]
        );
    }
    assert_eq!(max_diff, 0.0);

    // Also verify that non-zero output was produced (not all silence)
    let non_zero_count = out_full.iter().filter(|&&x| x != 0.0).count();
    assert!(
        non_zero_count > 0,
        "Cascade produced all zeroes; test inputs/weights may be inactive"
    );
}

#[test]
fn test_cascade_output_buffer_dirty_memory_clearing() {
    let arr0 = make_test_dyn_array(1, 3, 3, 1);
    let arr1 = make_test_dyn_array(3, 4, 8, 1);
    let mut cascade =
        WaveNetA2Cascade::try_new(vec![arr0, arr1], None, 1).expect("cascade creation failed");

    let num_frames = 128;
    let input = vec![0.2f32; num_frames];

    // Pre-fill output buffer with sentinel dirty data
    let mut output = vec![999.0f32; num_frames * 8];
    cascade.process(&input, &mut output);

    // No output value should remain the sentinel 999.0
    for (i, &val) in output.iter().enumerate() {
        assert_ne!(
            val, 999.0,
            "Sample {} in multichannel output buffer was not overwritten/cleared",
            i
        );
    }
}
