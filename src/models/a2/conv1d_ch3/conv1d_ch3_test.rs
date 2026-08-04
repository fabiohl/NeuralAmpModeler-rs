// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::math::common::Avx2Math;
use crate::models::a2::A2_DILATIONS;
use crate::models::a2::conv1d_fallback::a2_conv1d_single_frame_fallback;
use crate::models::wavenet::conv1d_dyn::Conv1dDyn;

fn make_ch3_test_weights(kernel: usize, seed: u32) -> (AlignedVec<f32>, AlignedVec<f32>) {
    let in_ch = 3usize;
    let out_ch = 3usize;
    let num_blocks = 1; // out_ch.div_ceil(4) = 1 for CH=3
    let total_w = num_blocks * 4 * in_ch * kernel;
    let mut weights =
        AlignedVec::new(total_w, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let mut state = seed;
    for i in 0..total_w {
        // Lane 3 stays 0 (CH=3 only has 3 output channels).
        if i % 4 == 3 {
            continue;
        }
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
        weights[i] = v;
    }

    let mut bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    for i in 0..out_ch {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        bias[i] = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    }

    (weights, bias)
}

fn make_ch3_conv_dyn(
    weights: AlignedVec<f32>,
    bias: AlignedVec<f32>,
    do_bias: bool,
    dilation: usize,
    kernel: usize,
) -> Conv1dDyn {
    let in_ch = 3usize;
    let out_ch = 3usize;
    Conv1dDyn {
        weights,
        bias,
        do_bias,
        dilation,
        in_ch,
        out_ch,
        num_blocks: 1,
        interleave_width: 4,
        kernel,
    }
}

/// K=6 parity: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_parity() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5]; // 101

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 99u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6 channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=15 parity: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k15_parity() {
    let kernel = 15;
    let dilation = A2_DILATIONS[15]; // 13

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 77u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15 channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=6 with mixin: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_with_mixin() {
    let kernel = 6;
    let dilation = A2_DILATIONS[3]; // 17

    let (weights, bias) = make_ch3_test_weights(kernel, 555);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 31u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let mut state = 88u32;
    let mixin: Vec<f32> = (0..3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, Some(&mixin));
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        Some(&mixin),
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6+mixin channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// K=15 with mixin: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k15_with_mixin() {
    let kernel = 15;
    let dilation = A2_DILATIONS[14]; // 1

    let (weights, bias) = make_ch3_test_weights(kernel, 777);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 13u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let mut state = 99u32;
    let mixin: Vec<f32> = (0..3)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, Some(&mixin));
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        true,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        Some(&mixin),
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15+mixin channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// All K=6 dilations: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_all_dilations() {
    let kernel = 6;

    let dilation_set: Vec<usize> = A2_DILATIONS
        .iter()
        .filter(|&&d| {
            let idx = A2_DILATIONS.iter().position(|&x| x == d).unwrap();
            crate::models::a2::A2_KERNEL_SIZES[idx] == 6
        })
        .copied()
        .collect();

    let (weights, bias) = make_ch3_test_weights(kernel, 111);

    let buf_frames = 4096;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 17u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    for &dilation in &dilation_set {
        let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

        let frame_idx = 3500;

        let mut ch3_out = vec![0.0f32; 3];
        let mut scalar_out = vec![0.0f32; 3];

        unsafe {
            conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &weights,
            &bias,
            true,
            dilation,
            3,
            3,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..3 {
            let diff = (ch3_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "dilation={} channel {}: ch3_unrolled={}, scalar={}, diff={}",
                dilation,
                c,
                ch3_out[c],
                scalar_out[c],
                diff
            );
        }
    }
}

/// No bias: unrolled CH=3 vs scalar fallback.
#[test]
fn test_ch3_unrolled_k6_no_bias() {
    let kernel = 6;
    let dilation = A2_DILATIONS[7]; // 1

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), false, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 55u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut scalar_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &weights,
        &bias,
        false,
        dilation,
        3,
        3,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..3 {
        let diff = (ch3_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "no_bias channel {}: ch3_unrolled={}, scalar={}, diff={}",
            c,
            ch3_out[c],
            scalar_out[c],
            diff
        );
    }
}

/// Consistency: independent runs produce identical results.
#[test]
fn test_ch3_unrolled_k6_deterministic() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5];

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 44u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut out1 = vec![0.0f32; 3];
    let mut out2 = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut out1, frame_idx, None);
        conv.process_single_ch3_unrolled(&layer_buffer, &mut out2, frame_idx, None);
    }

    for c in 0..3 {
        assert_eq!(
            out1[c], out2[c],
            "deterministic failure: run1[{}]={} != run2[{}]={}",
            c, out1[c], c, out2[c]
        );
    }
}

/// Unrolled K=6 vs generic SIMD path: results must match within tolerance.
#[test]
fn test_ch3_unrolled_k6_vs_generic() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5];

    let (weights, bias) = make_ch3_test_weights(kernel, 42);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 99u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut generic_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut generic_out, frame_idx, None);
    }

    for c in 0..3 {
        let diff = (ch3_out[c] - generic_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=6 ch3 vs generic channel {}: ch3_unrolled={}, generic={}, diff={}",
            c,
            ch3_out[c],
            generic_out[c],
            diff
        );
    }
}

/// Unrolled K=15 vs generic SIMD path: results must match within tolerance.
#[test]
fn test_ch3_unrolled_k15_vs_generic() {
    let kernel = 15;
    let dilation = A2_DILATIONS[15];

    let (weights, bias) = make_ch3_test_weights(kernel, 123);

    let conv = make_ch3_conv_dyn(weights.clone(), bias.clone(), true, dilation, kernel);

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * 3];
        let mut state = 77u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = kernel * dilation + 64;

    let mut ch3_out = vec![0.0f32; 3];
    let mut generic_out = vec![0.0f32; 3];

    unsafe {
        conv.process_single_ch3_unrolled(&layer_buffer, &mut ch3_out, frame_idx, None);
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut generic_out, frame_idx, None);
    }

    for c in 0..3 {
        let diff = (ch3_out[c] - generic_out[c]).abs();
        assert!(
            diff < 1e-5,
            "K=15 ch3 vs generic channel {}: ch3_unrolled={}, generic={}, diff={}",
            c,
            ch3_out[c],
            generic_out[c],
            diff
        );
    }
}

// =============================================================================
// Tests for A2Conv1dCh3 (f32-native path) and layer_forward_ch3_block
// =============================================================================

use crate::models::a2::conv1d_ch3::{
    A2Conv1dCh3, conv1d_ch3_single_frame_ref, layer_forward_ch3_block, layer_forward_ch3_scalar_ref,
};
use crate::models::a2::film::FilmBlock;

fn make_ch3_f32_weights(kernel: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut raw = vec![0.0f32; 3 * 3 * kernel];
    let mut bias = vec![0.0f32; 3];
    let mut state = seed;
    for v in raw.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    for v in bias.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    }
    (raw, bias)
}

fn make_f32_layer_buffer(num_frames: usize, seed: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; num_frames * 3];
    let mut state = seed;
    for v in buf.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    buf
}

/// A2Conv1dCh3 K=6 vs scalar ref — pure f32, no f16 conversion.
#[test]
fn test_a2conv1dch3_k6_vs_scalar_ref() {
    let kernel = 6;
    let dilation = A2_DILATIONS[5];
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 10);
    let conv = A2Conv1dCh3::new(&raw_w, 3, 3, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let buf_frames = kernel * dilation + 512;
    let layer_buffer = make_f32_layer_buffer(buf_frames, 99);
    let frame_idx = kernel * dilation + 64;

    let mut fast_out = [0.0f32; 4];
    let mut ref_out = [0.0f32; 4];
    unsafe {
        super::conv1d_ch3_f32_dispatch(&conv, &layer_buffer, frame_idx, &mut fast_out);
    }
    conv1d_ch3_single_frame_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &layer_buffer,
        frame_idx,
        &mut ref_out,
    );
    for c in 0..3 {
        let d = (fast_out[c] - ref_out[c]).abs();
        assert!(
            d < 1e-6,
            "K=6 f32 ch[{}]: fast={} ref={} diff={}",
            c,
            fast_out[c],
            ref_out[c],
            d
        );
    }
}

/// A2Conv1dCh3 K=15 vs scalar ref.
#[test]
fn test_a2conv1dch3_k15_vs_scalar_ref() {
    let kernel = 15;
    let dilation = A2_DILATIONS[15];
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 20);
    let conv = A2Conv1dCh3::new(&raw_w, 3, 3, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let buf_frames = kernel * dilation + 512;
    let layer_buffer = make_f32_layer_buffer(buf_frames, 77);
    let frame_idx = kernel * dilation + 64;

    let mut fast_out = [0.0f32; 4];
    let mut ref_out = [0.0f32; 4];
    unsafe {
        super::conv1d_ch3_f32_dispatch(&conv, &layer_buffer, frame_idx, &mut fast_out);
    }
    conv1d_ch3_single_frame_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &layer_buffer,
        frame_idx,
        &mut ref_out,
    );
    for c in 0..3 {
        let d = (fast_out[c] - ref_out[c]).abs();
        assert!(
            d < 1e-6,
            "K=15 f32 ch[{}]: fast={} ref={} diff={}",
            c,
            fast_out[c],
            ref_out[c],
            d
        );
    }
}

/// A2Conv1dCh3 zero history → output equals bias only.
#[test]
fn test_a2conv1dch3_zero_history_gives_bias() {
    let kernel = 6;
    let dilation = 1;
    let raw_w = vec![0.0f32; 3 * 3 * kernel];
    let bias = vec![0.1f32, 0.2, 0.3];
    let conv = A2Conv1dCh3::new(&raw_w, 3, 3, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let layer_buffer = vec![0.0f32; (kernel + 4) * 3];
    let frame_idx = kernel;
    let mut out = [0.0f32; 4];
    unsafe {
        super::conv1d_ch3_f32_dispatch(&conv, &layer_buffer, frame_idx, &mut out);
    }
    for c in 0..3 {
        assert!(
            (out[c] - bias[c]).abs() < 1e-7,
            "bias ch[{}]: got {} expected {}",
            c,
            out[c],
            bias[c]
        );
    }
}

/// layer_forward_ch3_block K=6, 16 frames (even) — AVX2 pairs only, no tail.
#[test]
fn test_layer_fwd_ch3_k6_even_frames_parity() {
    const CH: usize = 3;
    let kernel = 6;
    let dilation = A2_DILATIONS[5];
    let num_frames = 16usize;
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 30);
    let conv = A2Conv1dCh3::new(&raw_w, CH, CH, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let max_lookback = (kernel - 1) * dilation;
    let layer_buffer = make_f32_layer_buffer(max_lookback + num_frames + 8, 55);
    let frame_start = max_lookback;
    let mixin_w = [0.1f32, -0.2, 0.3];
    let l1x1_w: Vec<f32> = (0..CH * CH).map(|i| (i as f32 + 1.0) * 0.05).collect();
    let l1x1_b = [0.01f32, -0.01, 0.02];
    let input_cond: Vec<f32> = (0..num_frames)
        .map(|i| (i as f32 * 0.3).sin() * 0.5)
        .collect();
    let mut head_fast = vec![0.0f32; 256 * CH];
    let mut head_ref = vec![0.0f32; 256 * CH];
    let mut lin_fast = vec![0.0f32; num_frames * CH];
    let mut lin_ref = vec![0.0f32; num_frames * CH];
    let head_col = 8usize;
    let mut fb = FilmBlock::empty();
    unsafe {
        layer_forward_ch3_block(
            &conv,
            &mixin_w,
            &l1x1_w,
            &l1x1_b,
            &mut fb,
            false,
            &layer_buffer,
            frame_start,
            num_frames,
            &input_cond,
            &mut head_fast,
            head_col,
            &mut lin_fast,
            true,
            false,
        );
    }
    layer_forward_ch3_scalar_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &mixin_w,
        &l1x1_w,
        &l1x1_b,
        &layer_buffer,
        frame_start,
        num_frames,
        &input_cond,
        &mut head_ref,
        head_col,
        &mut lin_ref,
        true,
        false,
    );
    for f in 0..num_frames {
        for c in 0..CH {
            let hd = (head_fast[(head_col + f) * CH + c] - head_ref[(head_col + f) * CH + c]).abs();
            assert!(
                hd < 1e-5,
                "head[f={} c={}]: fast={} ref={} diff={}",
                f,
                c,
                head_fast[(head_col + f) * CH + c],
                head_ref[(head_col + f) * CH + c],
                hd
            );
            let ld = (lin_fast[f * CH + c] - lin_ref[f * CH + c]).abs();
            assert!(
                ld < 1e-5,
                "lin[f={} c={}]: fast={} ref={} diff={}",
                f,
                c,
                lin_fast[f * CH + c],
                lin_ref[f * CH + c],
                ld
            );
        }
    }
}

/// layer_forward_ch3_block K=15, 17 frames (odd) — exercises scalar tail frame.
#[test]
fn test_layer_fwd_ch3_k15_odd_frames_parity() {
    const CH: usize = 3;
    let kernel = 15;
    let dilation = A2_DILATIONS[15];
    let num_frames = 17usize;
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 40);
    let conv = A2Conv1dCh3::new(&raw_w, CH, CH, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let max_lookback = (kernel - 1) * dilation;
    let layer_buffer = make_f32_layer_buffer(max_lookback + num_frames + 8, 66);
    let frame_start = max_lookback;
    let mixin_w = [-0.1f32, 0.3, 0.2];
    let l1x1_w: Vec<f32> = (0..CH * CH).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let l1x1_b = [0.05f32, 0.05, -0.05];
    let input_cond: Vec<f32> = (0..num_frames).map(|i| (i as f32).cos() * 0.4).collect();
    let mut head_fast = vec![0.0f32; 256 * CH];
    let mut head_ref = vec![0.0f32; 256 * CH];
    let mut lin_fast = vec![0.0f32; num_frames * CH];
    let mut lin_ref = vec![0.0f32; num_frames * CH];
    let head_col = 4usize;
    let mut fb = FilmBlock::empty();
    unsafe {
        layer_forward_ch3_block(
            &conv,
            &mixin_w,
            &l1x1_w,
            &l1x1_b,
            &mut fb,
            false,
            &layer_buffer,
            frame_start,
            num_frames,
            &input_cond,
            &mut head_fast,
            head_col,
            &mut lin_fast,
            false,
            false,
        );
    }
    layer_forward_ch3_scalar_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &mixin_w,
        &l1x1_w,
        &l1x1_b,
        &layer_buffer,
        frame_start,
        num_frames,
        &input_cond,
        &mut head_ref,
        head_col,
        &mut lin_ref,
        false,
        false,
    );
    for f in 0..num_frames {
        for c in 0..CH {
            let hd = (head_fast[(head_col + f) * CH + c] - head_ref[(head_col + f) * CH + c]).abs();
            assert!(
                hd < 1e-5,
                "K15 head[f={} c={}]: fast={} ref={} diff={}",
                f,
                c,
                head_fast[(head_col + f) * CH + c],
                head_ref[(head_col + f) * CH + c],
                hd
            );
        }
    }
}

/// layer_forward_ch3_block is_last=true skips l1x1 (layer_in unchanged).
#[test]
fn test_layer_fwd_ch3_is_last_skips_l1x1() {
    const CH: usize = 3;
    let kernel = 6;
    let dilation = 1;
    let num_frames = 8;
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 50);
    let conv = A2Conv1dCh3::new(&raw_w, CH, CH, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let max_lookback = (kernel - 1) * dilation;
    let layer_buffer = make_f32_layer_buffer(max_lookback + num_frames + 4, 11);
    let frame_start = max_lookback;
    let mixin_w = [0.1f32, 0.1, 0.1];
    let l1x1_w = vec![1.0f32; CH * CH];
    let l1x1_b = [0.5f32, 0.5, 0.5];
    let input_cond = vec![0.5f32; num_frames];
    let mut head = vec![0.0f32; 64 * CH];
    let mut lin = vec![99.0f32; num_frames * CH];
    let mut fb = FilmBlock::empty();
    unsafe {
        layer_forward_ch3_block(
            &conv,
            &mixin_w,
            &l1x1_w,
            &l1x1_b,
            &mut fb,
            false,
            &layer_buffer,
            frame_start,
            num_frames,
            &input_cond,
            &mut head,
            0,
            &mut lin,
            true,
            true,
        );
    }
    for &v in &lin {
        assert!(
            (v - 99.0).abs() < 1e-7,
            "is_last=true: layer_in should be unchanged, got {}",
            v
        );
    }
}

/// layer_forward_ch3_block is_first=true assigns head (overwrites sentinel).
#[test]
fn test_layer_fwd_ch3_is_first_assigns_head() {
    const CH: usize = 3;
    let kernel = 6;
    let dilation = 1;
    let num_frames = 8;
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 60);
    let conv = A2Conv1dCh3::new(&raw_w, CH, CH, kernel, dilation, &bias)
        .expect("construction should succeed for test-sized buffers");
    let max_lookback = (kernel - 1) * dilation;
    let layer_buffer = make_f32_layer_buffer(max_lookback + num_frames + 4, 22);
    let frame_start = max_lookback;
    let mixin_w = [0.2f32, -0.1, 0.3];
    let l1x1_w = vec![0.0f32; CH * CH];
    let l1x1_b = [0.0f32; CH];
    let input_cond = vec![0.5f32; num_frames];
    let mut head = vec![999.0f32; 64 * CH];
    let mut lin = vec![0.0f32; num_frames * CH];
    let mut fb = FilmBlock::empty();
    unsafe {
        layer_forward_ch3_block(
            &conv,
            &mixin_w,
            &l1x1_w,
            &l1x1_b,
            &mut fb,
            false,
            &layer_buffer,
            frame_start,
            num_frames,
            &input_cond,
            &mut head,
            0,
            &mut lin,
            true,
            false,
        );
    }
    for f in 0..num_frames {
        for c in 0..CH {
            let v = head[f * CH + c];
            assert!(
                (v - 999.0).abs() > 1e-3,
                "is_first: head[f={} c={}]={} should differ from 999",
                f,
                c,
                v
            );
        }
    }
}

/// A2Conv1dCh3 K=6 all A2 dilations — parity vs scalar ref.
#[test]
fn test_a2conv1dch3_k6_all_dilations() {
    let kernel = 6;
    let (raw_w, bias) = make_ch3_f32_weights(kernel, 70);
    let layer_buffer = make_f32_layer_buffer(4096, 88);

    for (idx, &ksize) in crate::models::a2::A2_KERNEL_SIZES.iter().enumerate() {
        if ksize != 6 {
            continue;
        }
        let dilation = A2_DILATIONS[idx];
        let conv = A2Conv1dCh3::new(&raw_w, 3, 3, kernel, dilation, &bias)
            .expect("construction should succeed for test-sized buffers");
        let frame_idx = 3500usize;
        let mut fast_out = [0.0f32; 4];
        let mut ref_out = [0.0f32; 4];
        unsafe {
            super::conv1d_ch3_f32_dispatch(&conv, &layer_buffer, frame_idx, &mut fast_out);
        }
        conv1d_ch3_single_frame_ref(
            &conv.weights,
            &conv.bias,
            dilation,
            kernel,
            &layer_buffer,
            frame_idx,
            &mut ref_out,
        );
        for c in 0..3 {
            let d = (fast_out[c] - ref_out[c]).abs();
            assert!(
                d < 1e-6,
                "dilation={} ch[{}]: fast={} ref={} diff={}",
                dilation,
                c,
                fast_out[c],
                ref_out[c],
                d
            );
        }
    }
}
