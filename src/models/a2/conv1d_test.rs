// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::{AlignedVec, Avx2Math};
use crate::models::a2::A2_DILATIONS;
use crate::models::a2::conv1d_fallback::a2_conv1d_single_frame_fallback;

fn make_test_weights(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    seed: u32,
) -> (AlignedVec<f32>, AlignedVec<f32>) {
    let num_blocks = out_ch.div_ceil(4);
    let total_w = num_blocks * 4 * in_ch * kernel;
    let mut weights =
        AlignedVec::new(total_w, 0.0f32).expect("allocation should succeed for test-sized buffers");

    let mut state = seed;
    for i in 0..total_w {
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

#[test]
fn test_a2_conv1d_kernel6_parity_single_frame() {
    let in_ch = 3;
    let out_ch = 8;
    let kernel = 6;
    let dilation = A2_DILATIONS[1];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 42);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 99u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = 400;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &conv.standard_inner().weights,
        &conv.standard_inner().bias,
        conv.standard_inner().do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "channel {}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_kernel15_parity_single_frame() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 15;
    let dilation = A2_DILATIONS[15];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 123);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = 4096;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 77u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = 3500;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &conv.standard_inner().weights,
        &conv.standard_inner().bias,
        conv.standard_inner().do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "channel {}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_first_layer_in_ch_1() {
    let in_ch = 1;
    let out_ch = 3;
    let kernel = 6;
    let dilation = A2_DILATIONS[0];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 7);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = 256;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 13u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = 200;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &conv.standard_inner().weights,
        &conv.standard_inner().bias,
        conv.standard_inner().do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "channel {}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_with_mixin_kernel6() {
    let in_ch = 3;
    let out_ch = 8;
    let kernel = 6;
    let dilation = A2_DILATIONS[5];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 555);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 31u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let mut state = 88u32;
    let mixin: Vec<f32> = (0..out_ch)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(
            &layer_buffer,
            &mut simd_out,
            frame_idx,
            Some(&mixin),
        );
    }

    a2_conv1d_single_frame_fallback(
        &conv.standard_inner().weights,
        &conv.standard_inner().bias,
        conv.standard_inner().do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        Some(&mixin),
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "channel {} with mixin: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_all_dilations_kernel6() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 6;

    let dilation_set: Vec<usize> = A2_DILATIONS
        .iter()
        .filter(|&&d| {
            let idx = A2_DILATIONS.iter().position(|&x| x == d).unwrap();
            crate::models::a2::A2_KERNEL_SIZES[idx] == 6
        })
        .copied()
        .collect();

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 111);

    let buf_frames = 4096;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 17u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    for &dilation in &dilation_set {
        let conv = A2Conv1d::new(
            weights.clone(),
            bias.clone(),
            true,
            dilation,
            in_ch,
            out_ch,
            kernel,
        );

        let frame_idx = 3500;

        let mut simd_out = vec![0.0f32; out_ch];
        let mut scalar_out = vec![0.0f32; out_ch];

        unsafe {
            conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
        }

        a2_conv1d_single_frame_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            frame_idx,
            None,
            &mut scalar_out,
        );

        for c in 0..out_ch {
            let diff = (simd_out[c] - scalar_out[c]).abs();
            assert!(
                diff < 1e-5,
                "dilation={} channel {}: simd={}, scalar={}, diff={}",
                dilation,
                c,
                simd_out[c],
                scalar_out[c],
                diff
            );
        }
    }
}

#[test]
fn test_a2_conv1d_block_processing_kernel15() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 15;
    let dilation = 13;

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 777);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = 4096;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 43u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    for num_frames in [1, 2, 3, 4, 7, 16, 31, 64] {
        let buffer_start = 3500 - kernel * dilation;
        let mut simd_block = vec![0.0f32; num_frames * out_ch];
        let mut scalar_block = vec![0.0f32; num_frames * out_ch];

        unsafe {
            conv.process_block::<crate::math::common::Avx2Math>(
                &layer_buffer,
                &mut simd_block,
                buffer_start,
                num_frames,
                None,
            );
        }

        crate::models::a2::conv1d_fallback::a2_conv1d_block_fallback(
            &conv.standard_inner().weights,
            &conv.standard_inner().bias,
            conv.standard_inner().do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            &layer_buffer,
            buffer_start,
            num_frames,
            None,
            &mut scalar_block,
        );

        for (i, (&s, &f)) in simd_block.iter().zip(scalar_block.iter()).enumerate() {
            let diff = (s - f).abs();
            assert!(
                diff < 1e-5,
                "num_frames={} idx={}: simd={}, scalar={}, diff={}",
                num_frames,
                i,
                s,
                f,
                diff
            );
        }
    }
}

#[test]
fn test_a2_conv1d_kernel6_non_multiple_of_4_output() {
    let in_ch = 3;
    let out_ch = 7;
    let kernel = 6;
    let dilation = A2_DILATIONS[3];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 22);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    let buf_frames = 512;
    let layer_buffer = {
        let mut buf = vec![0.0f32; buf_frames * in_ch];
        let mut state = 91u32;
        for val in &mut buf {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        }
        buf
    };

    let frame_idx = 400;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    a2_conv1d_single_frame_fallback(
        &conv.standard_inner().weights,
        &conv.standard_inner().bias,
        conv.standard_inner().do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        &layer_buffer,
        frame_idx,
        None,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "channel {} (out_ch=7): simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

// =============================================================================
// Grouped conv integration tests — A2Conv1d enum
// =============================================================================

use crate::models::a2::grouped_conv1d::{make_layer_buffer, make_test_weights_grouped};

#[test]
fn test_a2_conv1d_grouped_groups2_parity() {
    let in_ch = 6;
    let out_ch = 4;
    let kernel = 6;
    let dilation = A2_DILATIONS[1];
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);

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
    .expect("construction should succeed for test-sized buffers");

    let buf_frames = 512;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 99);
    let frame_idx = 400;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
        match &conv {
            A2Conv1d::Grouped(g) => &g.weights,
            _ => panic!("expected Grouped variant"),
        },
        &raw_bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "groups=2 channel {}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_grouped_depthwise_parity() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 6;
    let dilation = A2_DILATIONS[3];
    let groups = 4; // depthwise: groups == channels

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 555);

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
    .expect("construction should succeed for test-sized buffers");

    assert!(conv.is_depthwise());
    assert_eq!(conv.groups(), 4);

    let buf_frames = 512;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 31);
    let frame_idx = kernel * dilation + 64;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(&layer_buffer, &mut simd_out, frame_idx, None);
    }

    crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
        match &conv {
            A2Conv1d::Grouped(g) => &g.weights,
            _ => panic!("expected Grouped variant"),
        },
        &raw_bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "depthwise ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_a2_conv1d_standard_groups_is_1() {
    let in_ch = 3;
    let out_ch = 8;
    let kernel = 6;
    let dilation = A2_DILATIONS[0];

    let (weights, bias) = make_test_weights(in_ch, out_ch, kernel, 7);

    let conv = A2Conv1d::new(
        weights.clone(),
        bias.clone(),
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
    );

    assert_eq!(conv.groups(), 1);
    assert!(!conv.is_depthwise());
    assert_eq!(conv.kernel_size(), kernel);
    assert_eq!(conv.dilation(), dilation);
    assert_eq!(conv.in_ch(), in_ch);
    assert_eq!(conv.out_ch(), out_ch);
}

#[test]
fn test_a2_conv1d_grouped_with_mixin() {
    let in_ch = 6;
    let out_ch = 8;
    let kernel = 6;
    let dilation = A2_DILATIONS[5];
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 777);

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
    .expect("construction should succeed for test-sized buffers");

    let buf_frames = kernel * dilation + 512;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 44);

    let mut state = 88u32;
    let mixin: Vec<f32> = (0..out_ch)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0
        })
        .collect();

    let frame_idx = kernel * dilation + 64;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    unsafe {
        conv.process_single_frame::<Avx2Math>(
            &layer_buffer,
            &mut simd_out,
            frame_idx,
            Some(&mixin),
        );
    }

    crate::models::a2::grouped_conv1d::grouped_conv1d_single_frame_ref(
        match &conv {
            A2Conv1d::Grouped(g) => &g.weights,
            _ => panic!("expected Grouped variant"),
        },
        &raw_bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut scalar_out,
    );

    for c in 0..out_ch {
        let scalar_with_mixin = scalar_out[c] + mixin[c];
        let diff = (simd_out[c] - scalar_with_mixin).abs();
        assert!(
            diff < 1e-5,
            "groups=2 with mixin ch={}: simd={}, scalar+mixin={}, diff={}",
            c,
            simd_out[c],
            scalar_with_mixin,
            diff
        );
    }
}
