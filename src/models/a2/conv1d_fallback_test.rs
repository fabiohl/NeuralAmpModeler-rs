// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_a2_fallback_kernel6_all_ones() {
    let in_ch: usize = 2;
    let out_ch: usize = 4;
    let kernel: usize = 6;
    let dilation: usize = 1;

    let num_blocks: usize = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let weights = vec![1.0f32; total_padded];

    let bias = vec![0.5f32; out_ch];
    let layer_buffer = vec![1.0f32; 32];
    let mut out = vec![0.0f32; out_ch];

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        10,
        None,
        &mut out,
    );

    let expected = 0.5 + kernel as f32 * in_ch as f32 * 1.0;
    for &val in &out {
        assert!((val - expected).abs() < 1e-4);
    }
}

#[test]
fn test_a2_fallback_kernel15_all_ones() {
    let in_ch: usize = 8;
    let out_ch: usize = 8;
    let kernel: usize = 15;
    let dilation: usize = 13;

    let num_blocks: usize = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let weights = vec![1.0f32; total_padded];

    let bias = vec![0.0f32; out_ch];
    let buf_frames = kernel * dilation + 1000 + 8;
    let layer_buffer = vec![1.0f32; buf_frames * in_ch];
    let mut out = vec![0.0f32; out_ch];

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        false,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        1000,
        None,
        &mut out,
    );

    let expected = kernel as f32 * in_ch as f32;
    for &val in &out {
        assert!((val - expected).abs() < 1e-4);
    }
}

#[test]
fn test_a2_fallback_dilated_tap_selection() {
    let in_ch: usize = 3;
    let out_ch: usize = 3;
    let kernel: usize = 2;
    let dilation: usize = 5;

    let num_blocks: usize = out_ch.div_ceil(4);
    let total_padded = num_blocks * 4 * in_ch * kernel;

    let weights = vec![1.0f32; total_padded];

    let bias = vec![0.0f32; out_ch];
    let buf_size = 100 * in_ch;
    let mut layer_buffer = vec![0.0f32; buf_size];

    layer_buffer[10 * in_ch] = 2.0;
    layer_buffer[10 * in_ch + 1] = 3.0;
    layer_buffer[10 * in_ch + 2] = 4.0;

    layer_buffer[15 * in_ch] = -1.0;
    layer_buffer[15 * in_ch + 1] = -2.0;
    layer_buffer[15 * in_ch + 2] = -3.0;

    let mut out = vec![0.0f32; out_ch];
    let frame_idx = 15;
    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        false,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut out,
    );

    let expected = (2.0 + 3.0 + 4.0) + (-1.0 + -2.0 + -3.0);
    for &val in &out {
        assert!(
            (val - expected).abs() < 1e-4,
            "expected {}, got {}",
            expected,
            val
        );
    }
}
