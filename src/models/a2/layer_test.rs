// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::Avx2Math;
use crate::models::a2::A2_DILATIONS;

fn make_conv_weights(ch: usize, kernel: usize, seed: u32) -> (AlignedVec<f32>, AlignedVec<f32>) {
    let num_blocks = ch.div_ceil(4);
    let total_w = num_blocks * 4 * ch * kernel;
    let mut weights =
        AlignedVec::new(total_w, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut state = seed;
    for i in 0..total_w {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
        weights[i] = v;
    }
    let mut bias =
        AlignedVec::new(ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    for i in 0..ch {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bias[i] = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    }
    (weights, bias)
}

fn make_f32_vec(len: usize, seed: u32) -> AlignedVec<f32> {
    let mut v =
        AlignedVec::new(len, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut state = seed;
    for i in 0..len {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        v[i] = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    v
}

fn make_history(num_cols: usize, ch: usize, seed: u32) -> Vec<f32> {
    let mut v = vec![0.0f32; num_cols * ch];
    let mut state = seed;
    for val in &mut v {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    v
}

/// End-to-end parity test: CH=3, kernel=6, dilation=101, all frames.
#[test]
fn test_a2_layer_ch3_kernel6_parity() {
    let ch = 3usize;
    let kernel = 6;
    let dilation = A2_DILATIONS[5]; // 101
    let num_frames = 16;

    let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 42);
    let conv = A2Conv1d::new(
        conv_w.clone(),
        conv_bias.clone(),
        true,
        dilation,
        ch,
        ch,
        kernel,
    );
    let mixin_w = make_f32_vec(ch, 100);
    let l1x1_w = make_f32_vec(ch * ch, 200);
    let l1x1_b = make_f32_vec(ch, 300);

    let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w.clone(), l1x1_b.clone());

    // History buffer: needs enough columns for lookback + write buffer.
    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, ch, 77);
    let hist_write_pos = max_lookback + 4; // wp after prewarm, before ring-write

    let input_cond: Vec<f32> = (0..num_frames).map(|i| (i as f32).sin() * 0.5).collect();

    let mut layer_in_simd = vec![0.0f32; num_frames * ch];
    let mut layer_in_scalar = vec![0.0f32; num_frames * ch];
    let mut head_simd = vec![0.0f32; (num_frames + 1) * ch];
    let mut head_scalar = vec![0.0f32; (num_frames + 1) * ch];

    // SIMD path: process frame-by-frame.
    {
        let mut z_buf = vec![0.0f32; ch];
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f; // = wp (ring-write position) + f
            let lin_slice = &mut layer_in_simd[f * ch..(f + 1) * ch];
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head_simd,
                f,
                &mut z_buf,
                lin_slice,
                f == 0,
                f == num_frames - 1,
            );
        }
    }

    // Scalar reference path.
    {
        for f in 0..num_frames {
            let lin_slice = &mut layer_in_scalar[f * ch..(f + 1) * ch];
            let frame_idx = hist_write_pos + f;
            a2_layer_single_frame_scalar_ref(
                &layer.conv.standard_inner().weights,
                &layer.conv.standard_inner().bias,
                layer.conv.standard_inner().do_bias,
                dilation,
                kernel,
                &history,
                frame_idx,
                &mixin_w,
                input_cond[f],
                &l1x1_w,
                &l1x1_b,
                &mut head_scalar,
                f,
                lin_slice,
                f == 0,
                f == num_frames - 1,
            );
        }
    }

    // Compare head accumulator.
    for c in 0..ch * num_frames {
        let diff = (head_simd[c] - head_scalar[c]).abs();
        assert!(
            diff < 1e-5,
            "head[{}]: simd={}, scalar={}, diff={}",
            c,
            head_simd[c],
            head_scalar[c],
            diff
        );
    }

    // Compare layer_in (only non-last layers updated).
    for c in 0..ch * (num_frames - 1) {
        let diff = (layer_in_simd[c] - layer_in_scalar[c]).abs();
        assert!(
            diff < 2e-5,
            "layer_in[{}]: simd={}, scalar={}, diff={}",
            c,
            layer_in_simd[c],
            layer_in_scalar[c],
            diff
        );
    }
}

/// End-to-end parity test: CH=8, kernel=15, dilation=13, all frames.
#[test]
fn test_a2_layer_ch8_kernel15_parity() {
    let ch = 8usize;
    let kernel = 15;
    let dilation = A2_DILATIONS[15]; // 13
    let num_frames = 16;

    let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 123);
    let conv = A2Conv1d::new(
        conv_w.clone(),
        conv_bias.clone(),
        true,
        dilation,
        ch,
        ch,
        kernel,
    );
    let mixin_w = make_f32_vec(ch, 400);
    let l1x1_w = make_f32_vec(ch * ch, 500);
    let l1x1_b = make_f32_vec(ch, 600);

    let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w.clone(), l1x1_b.clone());

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, ch, 88);
    let hist_write_pos = max_lookback + 4;

    let input_cond: Vec<f32> = (0..num_frames)
        .map(|i| (i as f32 * 0.7).cos() * 0.5)
        .collect();

    let mut layer_in_simd = vec![0.0f32; num_frames * ch];
    let mut layer_in_scalar = vec![0.0f32; num_frames * ch];
    let mut head_simd = vec![0.0f32; (num_frames + 1) * ch];
    let mut head_scalar = vec![0.0f32; (num_frames + 1) * ch];

    {
        let mut z_buf = vec![0.0f32; ch];
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f;
            let lin_slice = &mut layer_in_simd[f * ch..(f + 1) * ch];
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head_simd,
                f,
                &mut z_buf,
                lin_slice,
                f == 0,
                f == num_frames - 1,
            );
        }
    }

    {
        for f in 0..num_frames {
            let lin_slice = &mut layer_in_scalar[f * ch..(f + 1) * ch];
            let frame_idx = hist_write_pos + f;
            a2_layer_single_frame_scalar_ref(
                &layer.conv.standard_inner().weights,
                &layer.conv.standard_inner().bias,
                layer.conv.standard_inner().do_bias,
                dilation,
                kernel,
                &history,
                frame_idx,
                &mixin_w,
                input_cond[f],
                &l1x1_w,
                &l1x1_b,
                &mut head_scalar,
                f,
                lin_slice,
                f == 0,
                f == num_frames - 1,
            );
        }
    }

    for c in 0..ch * num_frames {
        let diff = (head_simd[c] - head_scalar[c]).abs();
        assert!(
            diff < 1e-5,
            "CH=8 head[{}]: simd={}, scalar={}, diff={}",
            c,
            head_simd[c],
            head_scalar[c],
            diff
        );
    }

    for c in 0..ch * (num_frames - 1) {
        let diff = (layer_in_simd[c] - layer_in_scalar[c]).abs();
        assert!(
            diff < 2e-5,
            "CH=8 layer_in[{}]: simd={}, scalar={}, diff={}",
            c,
            layer_in_simd[c],
            layer_in_scalar[c],
            diff
        );
    }
}

/// Verify first layer assigns to head (not accumulates), middle layers accumulate, last skips l1x1.
#[test]
fn test_a2_layer_first_middle_last_behavior() {
    let ch = 3usize;
    let kernel = 6;
    let dilation = 1;
    let num_frames = 4;

    let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 99);
    let conv = A2Conv1d::new(
        conv_w.clone(),
        conv_bias.clone(),
        true,
        dilation,
        ch,
        ch,
        kernel,
    );
    let mixin_w = make_f32_vec(ch, 101);
    let l1x1_w = make_f32_vec(ch * ch, 201);
    let l1x1_b = make_f32_vec(ch, 301);

    let layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

    let hist_cols = (kernel - 1) * dilation + num_frames + 4;
    let history = make_history(hist_cols, ch, 55);
    let hist_write_pos = (kernel - 1) * dilation + 2;

    let input_cond: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];

    // Test first layer: head should be assigned, not accumulated.
    {
        let mut head = vec![99.0f32; (num_frames + 1) * ch];
        let mut layer_in = vec![1.0f32; num_frames * ch];
        let mut z_buf = vec![0.0f32; ch];
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f;
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head,
                f,
                &mut z_buf,
                &mut layer_in[f * ch..(f + 1) * ch],
                true,
                f == num_frames - 1,
            );
        }
        // is_first=true means head values were ASSIGNED (overwritten), so they differ from 99.0.
        for f in 0..num_frames {
            for c in 0..ch {
                assert!(
                    (head[f * ch + c] - 99.0).abs() > 1e-3,
                    "first layer should assign head, not accumulate old value"
                );
            }
        }
    }

    // Test middle layer: head should accumulate.
    // Run is_first=false twice on the SAME head buffer; the second pass should change values.
    {
        let mut head = vec![1.0f32; (num_frames + 1) * ch];
        let mut head_copy = head.clone();
        let mut layer_in = vec![0.0f32; num_frames * ch];
        let mut z_buf = vec![0.0f32; ch];

        // First pass (is_first=true): head = layer output
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f;
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head,
                f,
                &mut z_buf,
                &mut layer_in[f * ch..(f + 1) * ch],
                true,
                false,
            );
        }
        head_copy.copy_from_slice(&head);

        // Second pass (is_first=false): should add more to head
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f;
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head,
                f,
                &mut z_buf,
                &mut layer_in[f * ch..(f + 1) * ch],
                false,
                false,
            );
        }

        // After second pass, head should differ from first pass (at least some frames/channels).
        let mut any_changed = false;
        for f in 0..num_frames {
            for c in 0..ch {
                if (head[f * ch + c] - head_copy[f * ch + c]).abs() > 1e-3 {
                    any_changed = true;
                }
            }
        }
        assert!(
            any_changed,
            "middle layer (is_first=false) should accumulate, but all values unchanged"
        );
    }

    // Test last layer: layer_in should NOT be updated.
    {
        let mut head = vec![0.0f32; (num_frames + 1) * ch];
        let mut layer_in = vec![1.0f32; num_frames * ch];
        let mut z_buf = vec![0.0f32; ch];
        for f in 0..num_frames {
            let frame_idx = hist_write_pos + f;
            layer.process_single_frame::<Avx2Math>(
                &history,
                frame_idx,
                input_cond[f],
                &mut head,
                f,
                &mut z_buf,
                &mut layer_in[f * ch..(f + 1) * ch],
                true,
                true, // is_last=true → skip l1x1
            );
        }
        for f in 0..num_frames {
            for c in 0..ch {
                assert!(
                    (layer_in[f * ch + c] - 1.0).abs() < 1e-6,
                    "last layer should skip l1x1 residual"
                );
            }
        }
    }
}

/// Test that mixin_w contributes to output (relaxed tolerance due to LeakyReLU nonlinearity).
#[test]
fn test_a2_layer_mixin_contribution() {
    let ch = 3usize;
    let kernel = 6;
    let dilation = 1;

    // Zero conv weights and bias to isolate mixin.
    let num_blocks = ch.div_ceil(4);
    let total_w = num_blocks * 4 * ch * kernel;
    let conv_w =
        AlignedVec::new(total_w, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let conv_bias =
        AlignedVec::new(ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let conv = A2Conv1d::new(
        conv_w, conv_bias, false, // no bias — pure conv with zero weights outputs 0
        dilation, ch, ch, kernel,
    );
    let mixin_w = AlignedVec::from_vec(vec![0.1f32, 0.2, 0.3])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w =
        AlignedVec::new(ch * ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let l1x1_b =
        AlignedVec::new(ch, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let layer = A2Layer::new(conv, mixin_w.clone(), l1x1_w, l1x1_b);

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + 8;
    let history = vec![0.0f32; hist_cols * ch];
    let hist_write_pos = max_lookback + 2;
    let frame_idx = hist_write_pos;

    // With zero conv: output = mixin_w[c] * cond, LeakyReLU, then head assign.
    // With cond=2.0: z = 0 + mixin_w[c]*2.0 = mixin_w[c]*2.0 > 0, so LeakyReLU is identity.
    // Head = z = mixin_w[c]*2.0.
    let mut head = vec![0.0f32; ch];
    let mut layer_in = vec![0.0f32; ch];
    let mut z_buf = vec![0.0f32; ch];

    layer.process_single_frame::<Avx2Math>(
        &history,
        frame_idx,
        2.0,
        &mut head,
        0,
        &mut z_buf,
        &mut layer_in,
        true,
        true,
    );

    for c in 0..ch {
        let expected = mixin_w[c] * 2.0;
        assert!(
            (head[c] - expected).abs() < 1e-5,
            "ch {}: head={}, expected={}",
            c,
            head[c],
            expected
        );
    }
}

/// Test that layer with zero weights and known input produces deterministic output.
#[test]
fn test_a2_layer_zero_weights_deterministic() {
    let ch = 3usize;
    let kernel = 6;
    let dilation = 1;

    let num_blocks = ch.div_ceil(4);
    let total_w = num_blocks * 4 * ch * kernel;
    let conv_w =
        AlignedVec::new(total_w, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let conv_bias =
        AlignedVec::new(ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let conv = A2Conv1d::new(conv_w, conv_bias.clone(), false, dilation, ch, ch, kernel);
    let mixin_w = AlignedVec::from_vec(vec![1.0f32, 2.0, 3.0])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w =
        AlignedVec::new(ch * ch, 1.0f32).expect("allocation should succeed for test-sized buffers");
    let l1x1_b = AlignedVec::from_vec(vec![0.5f32, 0.5, 0.5])
        .expect("allocation should succeed for test-sized buffers");

    let layer = A2Layer::new(conv, mixin_w, l1x1_w, l1x1_b);

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + 8;
    let history = vec![0.0f32; hist_cols * ch];
    let hist_write_pos = max_lookback + 2;

    let mut head = vec![0.0f32; ch];
    let mut layer_in = vec![0.0f32; ch];
    let mut z_buf = vec![0.0f32; ch];

    layer.process_single_frame::<Avx2Math>(
        &history,
        hist_write_pos,
        0.5,
        &mut head,
        0,
        &mut z_buf,
        &mut layer_in,
        true,
        false, // not last → l1x1 applied
    );

    // Conv output = 0 (zero weights, no bias, no mixin in conv).
    // Mixin: z[c] = 0 + mixin_w[c] * 0.5 = [0.5, 1.0, 1.5]
    // LeakyReLU: all positive → identity
    // Head: [0.5, 1.0, 1.5]
    assert!((head[0] - 0.5).abs() < 1e-5);
    assert!((head[1] - 1.0).abs() < 1e-5);
    assert!((head[2] - 1.5).abs() < 1e-5);

    // L1x1: layer_in[c] += 0.5 + sum_u(l1x1_w[u*3+c] * z[u])
    // l1x1_w is all 1.0, col-major: [u*3+c] = 1.0 for all u,c
    // sum_u(1.0 * z[u]) = 0.5+1.0+1.5 = 3.0
    // layer_in = [0.5+3.0, 0.5+3.0, 0.5+3.0] = [3.5, 3.5, 3.5]
    assert!((layer_in[0] - 3.5).abs() < 1e-5);
    assert!((layer_in[1] - 3.5).abs() < 1e-5);
    assert!((layer_in[2] - 3.5).abs() < 1e-5);
}
