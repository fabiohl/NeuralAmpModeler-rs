// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::Avx2Math;
use crate::models::a2::A2_DILATIONS;
use crate::models::a2::conv1d_fallback::a2_conv1d_single_frame_fallback;
use crate::models::a2::grouped_conv1d::{
    grouped_conv1d_single_frame_ref, make_test_weights_grouped,
};
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;
#[cfg(debug_assertions)]
use std::panic::{AssertUnwindSafe, catch_unwind};

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

#[test]
fn test_a2_layer_new_dyn_populates_conv_ch_for_standard_ch3_and_ch8() {
    for &ch in &[3usize, 8usize] {
        for &kernel in &[6usize, 15usize] {
            let (conv_w, conv_bias) = make_conv_weights(ch, kernel, 42 + ch as u32);
            let conv = A2Conv1d::new(conv_w, conv_bias, true, 7, ch, ch, kernel);
            let layer = A2Layer::new_dyn(
                conv,
                make_f32_vec(ch, 10),
                make_f32_vec(ch * ch, 20),
                make_f32_vec(ch, 30),
                ch,
                ch,
                1,
            );
            assert!(
                layer.conv_ch().is_some(),
                "expected conv_ch populated for ch={ch}, kernel={kernel}"
            );
        }
    }

    // Gated / bottleneck != channels or cond_size > 1 must leave conv_ch as None.
    let (conv_w_gated, conv_b_gated) = make_conv_weights(8, 6, 99);
    let conv_gated = A2Conv1d::new(conv_w_gated, conv_b_gated, true, 1, 8, 8, 6);
    let layer_bn = A2Layer::new_dyn(
        conv_gated.clone(),
        make_f32_vec(8, 10),
        make_f32_vec(32, 20),
        make_f32_vec(8, 30),
        8,
        4,
        1,
    );
    assert!(layer_bn.conv_ch().is_none());

    let layer_cond2 = A2Layer::new_dyn(
        conv_gated,
        make_f32_vec(16, 10),
        make_f32_vec(64, 20),
        make_f32_vec(8, 30),
        8,
        8,
        2,
    );
    assert!(layer_cond2.conv_ch().is_none());
}

fn make_logical_weights(out_ch: usize, in_ch: usize, kernel: usize, seed: u32) -> Vec<f32> {
    let mut weights = vec![0.0f32; out_ch * kernel * in_ch];
    let mut state = seed;
    for weight in &mut weights {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *weight = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    weights
}

fn layout_weights(
    logical: &[f32],
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
    width: usize,
) -> AlignedVec<f32> {
    let padded = out_ch.div_ceil(width) * width * in_ch * kernel;
    let mut weights =
        AlignedVec::new(padded, 0.0f32).expect("allocation should succeed for test-sized buffers");
    for out_c in 0..out_ch {
        for k in 0..kernel {
            for ic in 0..in_ch {
                let src = (out_c * kernel + k) * in_ch + ic;
                let dst =
                    ((out_c / width) * kernel + k) * in_ch * width + ic * width + out_c % width;
                weights[dst] = logical[src];
            }
        }
    }
    weights
}

fn make_standard_layer_with_width(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    width: usize,
    seed: u32,
) -> (A2Layer, AlignedVec<f32>) {
    let logical = make_logical_weights(out_ch, in_ch, kernel, seed);
    let weights = layout_weights(&logical, out_ch, in_ch, kernel, width);
    let oracle_weights = layout_weights(&logical, out_ch, in_ch, kernel, 4);
    let conv = Conv1dDyn::try_from_parts(
        weights,
        make_f32_vec(out_ch, seed.wrapping_add(1)),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        width,
    )
    .expect("test conv must satisfy the padded layout contract");
    let layer = A2Layer::new_base(
        A2Conv1d::Standard(conv),
        None,
        make_f32_vec(out_ch, seed.wrapping_add(2)),
        1,
        make_f32_vec(out_ch * out_ch, seed.wrapping_add(3)),
        make_f32_vec(out_ch, seed.wrapping_add(4)),
        1,
    );
    (layer, oracle_weights)
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar oracle mirroring the production frame parameters"
)]
fn scalar_post_conv(
    layer: &A2Layer,
    z_buf: &mut [f32],
    input_cond: f32,
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch = layer.channels();
    for c in 0..ch {
        z_buf[c] += layer.mixin_w[c] * input_cond;
    }
    for value in z_buf.iter_mut().take(ch) {
        if *value < 0.0 {
            *value *= A2_LEAKY_SLOPE;
        }
    }
    let head_off = head_col * ch;
    if is_first {
        head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[..ch]);
    } else {
        for c in 0..ch {
            head_accum[head_off + c] += z_buf[c];
        }
    }
    if !is_last {
        for c in 0..ch {
            let mut sum = layer.l1x1_b[c];
            for u in 0..ch {
                sum += layer.l1x1_w[u * ch + c] * z_buf[u];
            }
            layer_in_out[c] += sum;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar oracle mirroring the production frame parameters"
)]
fn scalar_standard_frame(
    layer: &A2Layer,
    oracle_weights: &[f32],
    layer_history: &[f32],
    frame_idx: usize,
    input_cond: f32,
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) -> Vec<f32> {
    let conv = layer.conv.standard_inner();
    let mut z_buf = vec![0.0f32; conv.out_ch];
    a2_conv1d_single_frame_fallback(
        oracle_weights,
        &conv.bias,
        conv.do_bias,
        conv.dilation,
        conv.in_ch,
        conv.out_ch,
        conv.kernel,
        layer_history,
        frame_idx,
        None,
        &mut z_buf,
    );
    scalar_post_conv(
        layer,
        &mut z_buf,
        input_cond,
        head_accum,
        head_col,
        layer_in_out,
        is_first,
        is_last,
    );
    z_buf
}

#[expect(
    clippy::too_many_arguments,
    reason = "scalar oracle mirroring the production frame parameters"
)]
fn scalar_grouped_frame(
    layer: &A2Layer,
    layer_history: &[f32],
    frame_idx: usize,
    input_cond: f32,
    head_accum: &mut [f32],
    head_col: usize,
    layer_in_out: &mut [f32],
    is_first: bool,
    is_last: bool,
) -> Vec<f32> {
    let A2Conv1d::Grouped(conv) = &layer.conv else {
        panic!("grouped scalar oracle requires a grouped layer");
    };
    let mut z_buf = vec![0.0f32; conv.out_ch()];
    grouped_conv1d_single_frame_ref(
        conv.weights(),
        conv.bias(),
        conv.do_bias(),
        conv.dilation(),
        conv.in_ch(),
        conv.out_ch(),
        conv.kernel(),
        conv.groups(),
        layer_history,
        frame_idx,
        &mut z_buf,
    );
    scalar_post_conv(
        layer,
        &mut z_buf,
        input_cond,
        head_accum,
        head_col,
        layer_in_out,
        is_first,
        is_last,
    );
    z_buf
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "{label}[{index}]: actual={actual}, expected={expected}, diff={diff}"
        );
    }
}

struct NegativeFrame {
    layer: A2Layer,
    history: Vec<f32>,
    frame_idx: usize,
    head_col: usize,
    head: Vec<f32>,
    z_buf: Vec<f32>,
    layer_in_out: Vec<f32>,
}

impl NegativeFrame {
    fn standard(ch: usize, kernel: usize, dilation: usize, seed: u32) -> Self {
        let (layer, _) = make_standard_layer_with_width(ch, ch, kernel, dilation, 4, seed);
        Self::with_layer(layer, ch, kernel, dilation, seed)
    }

    fn grouped(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        dilation: usize,
        groups: usize,
        seed: u32,
    ) -> Self {
        let (raw_weights, raw_bias) =
            make_test_weights_grouped(in_ch, out_ch, kernel, groups, seed);
        let conv = A2Conv1d::new_grouped(
            &raw_weights,
            &raw_bias,
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
        )
        .expect("test grouped conv must be sound");
        let layer = A2Layer::new_base(
            conv,
            None,
            make_f32_vec(out_ch, seed.wrapping_add(1)),
            1,
            make_f32_vec(out_ch * out_ch, seed.wrapping_add(2)),
            make_f32_vec(out_ch, seed.wrapping_add(3)),
            1,
        );
        Self::with_layer(layer, in_ch, kernel, dilation, seed)
    }

    fn with_layer(layer: A2Layer, in_ch: usize, kernel: usize, dilation: usize, seed: u32) -> Self {
        let lookback = (kernel - 1) * dilation;
        let frame_idx = lookback + 2;
        let out_ch = layer.channels();
        let head_col = 1;
        Self {
            layer,
            history: make_history(frame_idx + 2, in_ch, seed.wrapping_add(10)),
            frame_idx,
            head_col,
            head: vec![7.5f32; (head_col + 1) * out_ch],
            z_buf: vec![0.0f32; out_ch],
            layer_in_out: vec![0.25f32; out_ch],
        }
    }

    fn expect(&mut self, expected: A2LayerError) {
        let head_before = self.head.clone();
        let z_before = self.z_buf.clone();
        let layer_in_before = self.layer_in_out.clone();

        let err = self
            .layer
            .try_process_single_frame::<Avx2Math>(
                &self.history,
                self.frame_idx,
                0.37,
                &mut self.head,
                self.head_col,
                &mut self.z_buf,
                &mut self.layer_in_out,
                false,
                false,
            )
            .expect_err("negative case must be rejected");
        assert_eq!(err, expected);
        assert_eq!(&*self.head, head_before.as_slice());
        assert_eq!(&*self.z_buf, z_before.as_slice());
        assert_eq!(&*self.layer_in_out, layer_in_before.as_slice());

        let invoke_legacy = || {
            self.layer.process_single_frame::<Avx2Math>(
                &self.history,
                self.frame_idx,
                0.37,
                &mut self.head,
                self.head_col,
                &mut self.z_buf,
                &mut self.layer_in_out,
                false,
                false,
            );
        };
        #[cfg(debug_assertions)]
        {
            assert!(
                catch_unwind(AssertUnwindSafe(invoke_legacy)).is_err(),
                "legacy wrapper must debug_assert on invalid input"
            );
        }
        #[cfg(not(debug_assertions))]
        {
            let mut invoke_legacy = invoke_legacy;
            invoke_legacy();
        }

        assert_eq!(&*self.head, head_before.as_slice());
        assert_eq!(&*self.z_buf, z_before.as_slice());
        assert_eq!(&*self.layer_in_out, layer_in_before.as_slice());
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "parameter sweep over the production frame geometry"
)]
fn assert_standard_valid(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    width: usize,
    frame_idx: usize,
    head_col: usize,
    seed: u32,
    is_first: bool,
    is_last: bool,
) {
    let (layer, oracle_weights) =
        make_standard_layer_with_width(in_ch, out_ch, kernel, dilation, width, seed);
    let history = make_history(frame_idx + 2, in_ch, seed.wrapping_add(20));
    let mut head_simd = vec![13.0f32; (head_col + 1) * out_ch];
    let mut head_ref = head_simd.clone();
    let mut z_simd = vec![0.0f32; out_ch];
    let mut layer_in_simd = vec![0.25f32; out_ch];
    let mut layer_in_ref = layer_in_simd.clone();

    layer
        .try_process_single_frame::<Avx2Math>(
            &history,
            frame_idx,
            0.37,
            &mut head_simd,
            head_col,
            &mut z_simd,
            &mut layer_in_simd,
            is_first,
            is_last,
        )
        .expect("valid frame must be accepted");

    let z_ref = scalar_standard_frame(
        &layer,
        &oracle_weights,
        &history,
        frame_idx,
        0.37,
        &mut head_ref,
        head_col,
        &mut layer_in_ref,
        is_first,
        is_last,
    );

    assert_close("z_buf", &z_simd, &z_ref, 1e-5);
    assert_close("head_accum", &head_simd, &head_ref, 1e-5);
    assert_close("layer_in_out", &layer_in_simd, &layer_in_ref, 2e-5);
}

fn assert_grouped_valid(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    seed: u32,
) {
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, seed);
    let conv = A2Conv1d::new_grouped(
        &raw_weights,
        &raw_bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("test grouped conv must be sound");
    let layer = A2Layer::new_base(
        conv,
        None,
        make_f32_vec(out_ch, seed.wrapping_add(1)),
        1,
        make_f32_vec(out_ch * out_ch, seed.wrapping_add(2)),
        make_f32_vec(out_ch, seed.wrapping_add(3)),
        1,
    );
    let lookback = (kernel - 1) * dilation;
    let frame_idx = lookback + 2;
    let head_col = 1;
    let history = make_history(frame_idx + 2, in_ch, seed.wrapping_add(20));
    let mut head_simd = vec![13.0f32; (head_col + 1) * out_ch];
    let mut head_ref = head_simd.clone();
    let mut z_simd = vec![0.0f32; out_ch];
    let mut layer_in_simd = vec![0.25f32; out_ch];
    let mut layer_in_ref = layer_in_simd.clone();

    layer
        .try_process_single_frame::<Avx2Math>(
            &history,
            frame_idx,
            0.37,
            &mut head_simd,
            head_col,
            &mut z_simd,
            &mut layer_in_simd,
            false,
            false,
        )
        .expect("valid grouped frame must be accepted");

    let z_ref = scalar_grouped_frame(
        &layer,
        &history,
        frame_idx,
        0.37,
        &mut head_ref,
        head_col,
        &mut layer_in_ref,
        false,
        false,
    );

    assert_close("grouped z_buf", &z_simd, &z_ref, 2e-5);
    assert_close("grouped head_accum", &head_simd, &head_ref, 2e-5);
    assert_close("grouped layer_in_out", &layer_in_simd, &layer_in_ref, 2e-5);
}

#[test]
fn test_a2_layer_rejects_empty_channels() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 11);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.in_ch = 0;
    frame.expect(A2LayerError::EmptyChannels {
        in_ch: 0,
        out_ch: 3,
    });

    let mut frame = NegativeFrame::standard(3, 6, 2, 111);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.out_ch = 0;
    frame.expect(A2LayerError::EmptyChannels {
        in_ch: 3,
        out_ch: 0,
    });
}

#[test]
fn test_a2_layer_rejects_unsupported_kernel_sizes() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 12);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.kernel = 0;
    frame.expect(A2LayerError::UnsupportedKernel {
        kernel: 0,
        max: MAX_KERNEL,
    });

    let mut frame = NegativeFrame::standard(3, 6, 2, 13);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.kernel = MAX_KERNEL + 1;
    conv.weights = make_f32_vec(4 * 3 * (MAX_KERNEL + 1), 7);
    frame.expect(A2LayerError::UnsupportedKernel {
        kernel: MAX_KERNEL + 1,
        max: MAX_KERNEL,
    });
}

#[test]
fn test_a2_layer_rejects_frame_before_receptive_field() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 14);
    let lookback = (6 - 1) * 2;
    frame.frame_idx = lookback - 1;
    frame.history = make_history(lookback + 2, 3, 141);
    frame.expect(A2LayerError::FrameBeforeReceptiveField {
        frame_idx: lookback - 1,
        lookback,
    });
}

#[test]
fn test_a2_layer_rejects_short_history() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 15);
    let required = (frame.frame_idx + 1) * 3;
    frame.history.truncate(required - 1);
    frame.expect(A2LayerError::HistoryTooShort {
        len: required - 1,
        required,
        frame_idx: frame.frame_idx,
    });
}

#[test]
fn test_a2_layer_rejects_short_output_scratch() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 16);
    frame.z_buf.truncate(2);
    frame.expect(A2LayerError::OutputScratchTooSmall { len: 2, out_ch: 3 });
}

#[test]
fn test_a2_layer_rejects_short_residual_buffer() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 17);
    frame.layer_in_out.truncate(2);
    frame.expect(A2LayerError::ResidualBufferTooSmall { len: 2, out_ch: 3 });
}

#[test]
fn test_a2_layer_rejects_short_head_accumulator() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 18);
    frame.head.truncate(3);
    frame.expect(A2LayerError::HeadAccumTooShort {
        len: 3,
        head_col: 1,
        required: 6,
    });
}

#[test]
fn test_a2_layer_rejects_short_mixin_weights() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 19);
    frame.layer.mixin_w = make_f32_vec(2, 191);
    frame.expect(A2LayerError::MixinWeightsTooShort { len: 2, out_ch: 3 });
}

#[test]
fn test_a2_layer_rejects_mixin_width_mismatch() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 20);
    frame.layer.mixin_w = make_f32_vec(4, 201);
    frame.expect(A2LayerError::MixinWidthMismatch { len: 4, out_ch: 3 });
}

#[test]
fn test_a2_layer_rejects_unsupported_mixin_groups() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 21);
    frame.layer.mixin_groups = 2;
    frame.expect(A2LayerError::UnsupportedMixinGroups { groups: 2 });
}

#[test]
fn test_a2_layer_rejects_unsupported_residual_groups() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 22);
    frame.layer.l1x1_groups = 2;
    frame.expect(A2LayerError::UnsupportedResidualGroups { groups: 2 });
}

#[test]
fn test_a2_layer_rejects_residual_width_mismatch() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 23);
    frame.layer.l1x1_b = make_f32_vec(4, 231);
    frame.expect(A2LayerError::ResidualWidthMismatch {
        residual: 4,
        out_ch: 3,
    });

    let mut frame = NegativeFrame::standard(3, 6, 2, 232);
    frame.layer.l1x1_b = make_f32_vec(2, 232);
    frame.expect(A2LayerError::ResidualWidthMismatch {
        residual: 2,
        out_ch: 3,
    });
}

#[test]
fn test_a2_layer_rejects_short_residual_weights_when_not_last() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 24);
    frame.layer.l1x1_w = make_f32_vec(8, 241);
    frame.expect(A2LayerError::ResidualWeightsTooShort {
        len: 8,
        required: 9,
    });
}

#[test]
fn test_a2_layer_rejects_unsupported_interleave_width() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 25);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.interleave_width = 5;
    conv.weights = make_f32_vec(5 * 3 * 6, 7);
    frame.expect(A2LayerError::UnsupportedInterleaveWidth { width: 5 });
}

#[test]
fn test_a2_layer_rejects_short_conv_weights() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 26);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    let required = conv.weights.len();
    conv.weights = make_f32_vec(required - 1, 261);
    frame.expect(A2LayerError::ConvWeightsTooShort {
        len: required - 1,
        required,
    });
}

#[test]
fn test_a2_layer_rejects_short_conv_bias() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 27);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.bias = make_f32_vec(2, 271);
    frame.expect(A2LayerError::ConvBiasTooShort { len: 2, out_ch: 3 });
}

#[test]
fn test_a2_layer_rejects_history_size_overflow() {
    let mut frame = NegativeFrame::standard(1, 1, 0, 28);
    frame.frame_idx = usize::MAX;
    frame.expect(A2LayerError::SizeOverflow);
}

#[test]
fn test_a2_layer_rejects_head_size_overflow() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 29);
    frame.head_col = usize::MAX;
    frame.expect(A2LayerError::SizeOverflow);
}

#[test]
fn test_a2_layer_rejects_lookback_size_overflow() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 30);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.kernel = MAX_KERNEL;
    conv.dilation = isize::MAX as usize;
    frame.expect(A2LayerError::SizeOverflow);
}

#[test]
fn test_a2_layer_rejects_dilation_beyond_isize() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 31);
    let A2Conv1d::Standard(conv) = &mut frame.layer.conv else {
        panic!("standard layer expected");
    };
    conv.kernel = 1;
    conv.dilation = isize::MAX as usize + 1;
    frame.expect(A2LayerError::SizeOverflow);
}

#[test]
fn test_a2_layer_rejects_grouped_dilation_beyond_isize_with_kernel_one() {
    let mut frame = NegativeFrame::grouped(6, 4, 1, 2, 2, 34);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    conv.dilation = isize::MAX as usize + 1;
    frame.expect(A2LayerError::SizeOverflow);
}

#[test]
fn test_a2_layer_rejects_active_head_projection() {
    let mut frame = NegativeFrame::standard(3, 6, 2, 32);
    frame.layer.head1x1_active = true;
    frame.expect(A2LayerError::HeadProjectionActive);
}

#[test]
fn test_a2_layer_rejects_grouped_zero_groups() {
    let mut frame = NegativeFrame::grouped(6, 4, 3, 2, 2, 33);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    conv.groups = 0;
    frame.expect(A2LayerError::InvalidGroupedTopology {
        groups: 0,
        in_ch: 6,
        out_ch: 4,
    });
}

#[test]
fn test_a2_layer_rejects_grouped_non_divisor_groups() {
    let mut frame = NegativeFrame::grouped(6, 4, 3, 2, 2, 34);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    conv.groups = 3;
    frame.expect(A2LayerError::InvalidGroupedTopology {
        groups: 3,
        in_ch: 6,
        out_ch: 4,
    });
}

#[test]
fn test_a2_layer_rejects_short_grouped_weights() {
    let mut frame = NegativeFrame::grouped(6, 4, 3, 2, 2, 35);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    let required = conv.weights().len();
    conv.weights = make_f32_vec(required - 1, 351);
    frame.expect(A2LayerError::GroupedWeightsTooShort {
        len: required - 1,
        required,
    });

    let mut frame = NegativeFrame::grouped(4, 4, 3, 5, 4, 352);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    let required = conv.weights().len();
    conv.weights = make_f32_vec(required - 1, 3521);
    frame.expect(A2LayerError::GroupedWeightsTooShort {
        len: required - 1,
        required,
    });
}

#[test]
fn test_a2_layer_rejects_grouped_unsound_derived_layout() {
    let mut frame = NegativeFrame::grouped(6, 4, 3, 2, 2, 36);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    conv.in_per_group += 1;
    frame.expect(A2LayerError::GroupedUnsound);
}

#[test]
fn test_a2_layer_rejects_short_grouped_bias() {
    let mut frame = NegativeFrame::grouped(6, 4, 3, 2, 2, 37);
    let A2Conv1d::Grouped(conv) = &mut frame.layer.conv else {
        panic!("grouped layer expected");
    };
    conv.bias = make_f32_vec(3, 371);
    frame.expect(A2LayerError::ConvBiasTooShort { len: 3, out_ch: 4 });
}

#[test]
fn test_a2_layer_standard_i_not_o_matches_scalar_oracle() {
    assert_standard_valid(3, 5, 6, 2, 4, 12, 1, 401, true, false);
    assert_standard_valid(5, 3, 6, 2, 4, 12, 1, 402, false, false);
}

#[test]
fn test_a2_layer_simd_width_tails_match_scalar_oracle() {
    for &(width, out_ch) in &[(4usize, 6usize), (8, 10), (16, 20)] {
        assert_standard_valid(
            3,
            out_ch,
            6,
            2,
            width,
            12,
            1,
            410 + width as u32,
            true,
            false,
        );
    }
}

#[test]
fn test_a2_layer_kernel_one_tap_first_and_last_layer_matches_scalar_oracle() {
    assert_standard_valid(3, 5, 1, 5, 8, 0, 0, 420, true, true);
}

#[test]
fn test_a2_layer_first_and_last_frame_matches_scalar_oracle() {
    let kernel = 6;
    let dilation = 2;
    let lookback = (kernel - 1) * dilation;
    assert_standard_valid(3, 4, kernel, dilation, 4, lookback, 1, 430, true, false);
    assert_standard_valid(3, 4, kernel, dilation, 4, 31, 1, 431, false, false);
}

#[test]
fn test_a2_layer_non_first_last_layer_matches_scalar_oracle() {
    assert_standard_valid(3, 4, 6, 2, 4, 12, 1, 432, false, true);
}

#[test]
fn test_a2_layer_zero_dilation_matches_scalar_oracle() {
    assert_standard_valid(3, 4, 3, 0, 4, 0, 1, 440, true, false);
}

#[test]
fn test_a2_layer_last_layer_accepts_short_residual_weights() {
    let in_ch = 3;
    let out_ch = 3;
    let kernel = 6;
    let dilation = 2;
    let (mut layer, oracle_weights) =
        make_standard_layer_with_width(in_ch, out_ch, kernel, dilation, 4, 450);
    layer.l1x1_w = make_f32_vec(1, 451);
    let lookback = (kernel - 1) * dilation;
    let frame_idx = lookback + 2;
    let history = make_history(frame_idx + 2, in_ch, 452);
    let head_col = 1;
    let mut head_simd = vec![13.0f32; (head_col + 1) * out_ch];
    let mut head_ref = head_simd.clone();
    let mut z_simd = vec![0.0f32; out_ch];
    let mut layer_in_simd = vec![0.25f32; out_ch];
    let mut layer_in_ref = layer_in_simd.clone();

    layer
        .try_process_single_frame::<Avx2Math>(
            &history,
            frame_idx,
            0.37,
            &mut head_simd,
            head_col,
            &mut z_simd,
            &mut layer_in_simd,
            true,
            true,
        )
        .expect("last layer must skip the residual weights precondition");

    let z_ref = scalar_standard_frame(
        &layer,
        &oracle_weights,
        &history,
        frame_idx,
        0.37,
        &mut head_ref,
        head_col,
        &mut layer_in_ref,
        true,
        true,
    );

    assert_close("last z_buf", &z_simd, &z_ref, 1e-5);
    assert_close("last head_accum", &head_simd, &head_ref, 1e-5);
    assert_eq!(layer_in_simd, layer_in_ref);
}

#[test]
fn test_a2_layer_grouped_general_matches_scalar_oracle() {
    assert_grouped_valid(6, 4, 3, 2, 2, 460);
    assert_grouped_valid(8, 8, 2, 1, 4, 461);
}

#[test]
fn test_a2_layer_grouped_depthwise_matches_scalar_oracle() {
    assert_grouped_valid(4, 4, 3, 5, 4, 470);
    assert_grouped_valid(8, 8, 6, 3, 8, 471);
}
