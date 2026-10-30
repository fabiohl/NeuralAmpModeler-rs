// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::common::diagnostics::NamErrorCode;

#[test]
fn test_grouped_conv1d_groups2_single_frame() {
    let in_ch = 6;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);

    let conv = A2GroupedConv1d::new(
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

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }

    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
            "groups=2 ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_grouped_conv1d_groups4_single_frame() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 2;
    let dilation = 1;
    let groups = 4;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 123);

    let conv = A2GroupedConv1d::new(
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

    let buf_frames = 256;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 77);

    let frame_idx = 200;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }

    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
            "groups=4 ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_grouped_conv1d_depthwise() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 5;
    let groups = 4;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 555);

    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 31);

    let frame_idx = kernel * dilation + 64;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }

    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
fn test_grouped_conv1d_large_kernel_dilation() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 15;
    let dilation = 128;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 777);

    let conv = A2GroupedConv1d::new(
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

    let buf_frames = 4096;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 13);

    let frame_idx = 3500;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }

    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
            "large kernel ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_grouped_conv1d_with_mixin() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 3;
    let dilation = 4;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 888);

    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 17);

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
    let mut scalar_out_ref = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `mixin` = `out_ch`, `frame_idx` within history); they outlive the
    // call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(
            &conv,
            &layer_buffer,
            &mut simd_out,
            frame_idx,
            Some(&mixin),
        );
    }

    // Check via scalar reference + manual mixin
    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut scalar_out_ref,
    );
    for c in 0..out_ch {
        scalar_out[c] = scalar_out_ref[c] + mixin[c];
    }

    for c in 0..out_ch {
        let diff = (simd_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "mixin ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_grouped_conv1d_block() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 5;
    let dilation = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 333);

    let conv = A2GroupedConv1d::new(
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

    let buf_frames = 2048;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 11);

    for num_frames in [1, 2, 3, 4, 7, 16, 31] {
        let buffer_start = 1500 - kernel * dilation;
        let mut simd_block = vec![0.0f32; num_frames * out_ch];
        let mut scalar_block = vec![0.0f32; num_frames * out_ch];

        // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
        // `simd_block` = `num_frames` × `out_ch`, `buffer_start` within history); they outlive
        // the call; AVX2+FMA via `#[target_feature]`/dispatch.
        unsafe {
            conv.process_block(
                &layer_buffer,
                &mut simd_block,
                buffer_start,
                num_frames,
                None,
            );
        }

        grouped_conv1d_block_ref(
            &conv.weights,
            &conv.bias,
            conv.do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            &layer_buffer,
            buffer_start,
            num_frames,
            &mut scalar_block,
        );

        for (i, (&s, &f)) in simd_block.iter().zip(scalar_block.iter()).enumerate() {
            let diff = (s - f).abs();
            assert!(
                diff < 1e-5,
                "block nf={} idx={}: simd={}, scalar={}, diff={}",
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
fn test_grouped_conv1d_groups1_delegates_correctly() {
    // groups == 1 should produce the same result as the standard A2Conv1d
    let in_ch = 8;
    let out_ch = 6;
    let kernel = 6;
    let dilation = 13;
    let groups = 1;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);

    let conv = A2GroupedConv1d::new(
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

    let mut grouped_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `grouped_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut grouped_out, frame_idx, None);
    }

    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
        let diff = (grouped_out[c] - scalar_out[c]).abs();
        assert!(
            diff < 1e-5,
            "groups=1 ch={}: grouped={}, scalar={}, diff={}",
            c,
            grouped_out[c],
            scalar_out[c],
            diff
        );
    }
}

#[test]
fn test_grouped_conv1d_no_bias() {
    let in_ch = 8;
    let out_ch = 8;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 111);

    let conv = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        false, // do_bias = false
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("construction should succeed for test-sized buffers");

    let buf_frames = 256;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 33);

    let frame_idx = 200;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut scalar_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to the kernel's contract (`layer_buffer` = buf_frames × `in_ch`,
    // `simd_out` = `out_ch`, `frame_idx` within history); they outlive the call; AVX2+FMA via `#[target_feature]`/dispatch.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }
    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
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
            "no_bias ch={}: simd={}, scalar={}, diff={}",
            c,
            simd_out[c],
            scalar_out[c],
            diff
        );
    }
}

// =============================================================================
// #[should_panic] tests — SIMD contract validation
// =============================================================================
// These tests ensure that broken shapes / misaligned dims passed to unsafe
// SIMD kernel functions trigger debug_assert! panics instead of producing
// silent out-of-bounds memory accesses (General Protection Fault / segfault).

#[test]
#[should_panic(expected = "out_frame")]
fn should_panic_process_single_frame_out_frame_too_short() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(512, in_ch, 99);
    let frame_idx = 400;
    let mut out_frame = vec![0.0f32; out_ch - 1]; // too short by 1

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "out_frame").
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut out_frame, frame_idx, None);
    }
}

#[test]
#[should_panic(expected = "frame_idx")]
fn should_panic_process_single_frame_frame_idx_too_low() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 64;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(4096, in_ch, 99);
    let frame_idx = 1; // far too low: lookback = 64 * 2 = 128
    let mut out_frame = vec![0.0f32; out_ch];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "frame_idx").
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut out_frame, frame_idx, None);
    }
}

#[test]
#[should_panic(expected = "mixin len")]
fn should_panic_process_single_frame_mixin_too_short() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(512, in_ch, 99);
    let frame_idx = 400;
    let mut out_frame = vec![0.0f32; out_ch];
    let mixin = vec![0.0f32; out_ch - 1];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "mixin len").
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut out_frame, frame_idx, Some(&mixin));
    }
}

#[test]
#[should_panic(expected = "layer_buffer")]
fn should_panic_process_single_frame_layer_buffer_too_small() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let buffer_frames = 10; // tiny buffer
    let layer_buffer = make_layer_buffer(buffer_frames, in_ch, 99);
    let frame_idx = 400; // way beyond buffer
    let mut out_frame = vec![0.0f32; out_ch];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "layer_buffer").
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut out_frame, frame_idx, None);
    }
}

#[test]
#[should_panic(expected = "simd: out_frame")]
fn should_panic_simd_kernel_out_frame_too_short() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(512, in_ch, 99);
    let frame_idx = 400;
    let mut out_frame = vec![0.0f32; out_ch - 1];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "simd: out_frame").
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut out_frame, frame_idx, None);
    }
}

#[test]
#[should_panic(expected = "simd: frame_idx")]
fn should_panic_simd_kernel_frame_idx_too_low() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 15;
    let dilation = 128;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(4096, in_ch, 99);
    let frame_idx = 50; // lookback = 128 * 14 = 1792
    let mut out_frame = vec![0.0f32; out_ch];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "simd: frame_idx").
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut out_frame, frame_idx, None);
    }
}

#[test]
#[should_panic(expected = "simd: mixin")]
fn should_panic_simd_kernel_mixin_too_short() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(512, in_ch, 99);
    let frame_idx = 400;
    let mut out_frame = vec![0.0f32; out_ch];
    let mixin = vec![0.0f32; out_ch - 1];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "simd: mixin").
    unsafe {
        grouped_conv1d_single_frame_simd(
            &conv,
            &layer_buffer,
            &mut out_frame,
            frame_idx,
            Some(&mixin),
        );
    }
}

#[test]
#[should_panic(expected = "depthwise: out_frame")]
fn should_panic_depthwise_out_frame_too_short() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 4;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(512, in_ch, 99);
    let frame_idx = 400;
    let mut out_frame = vec![0.0f32; out_ch - 1];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "depthwise: out_frame").
    unsafe {
        process_single_frame_depthwise_avx2(&conv, &layer_buffer, &mut out_frame, frame_idx);
    }
}

#[test]
#[should_panic(expected = "depthwise: frame_idx")]
fn should_panic_depthwise_frame_idx_too_low() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 5;
    let dilation = 32;
    let groups = 4;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(4096, in_ch, 99);
    let frame_idx = 10; // lookback = 32 * 4 = 128
    let mut out_frame = vec![0.0f32; out_ch];

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "depthwise: frame_idx").
    unsafe {
        process_single_frame_depthwise_avx2(&conv, &layer_buffer, &mut out_frame, frame_idx);
    }
}

#[test]
#[should_panic(expected = "process_block: block len")]
fn should_panic_process_block_block_too_small() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(4096, in_ch, 99);
    let num_frames = 4;
    let mut block = vec![0.0f32; num_frames * out_ch - 1]; // 1 element short
    let buffer_start = 1500;

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "process_block: block len").
    unsafe {
        conv.process_block(&layer_buffer, &mut block, buffer_start, num_frames, None);
    }
}

#[test]
#[should_panic(expected = "process_block: mixin")]
fn should_panic_process_block_mixin_too_short() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let dilation = 2;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 43);
    let conv = A2GroupedConv1d::new(
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
    let layer_buffer = make_layer_buffer(4096, in_ch, 99);
    let num_frames = 4;
    let mut block = vec![0.0f32; num_frames * out_ch];
    let mixin = vec![0.0f32; num_frames * out_ch - 1];
    let buffer_start = 1500;

    // SAFETY: test-only call; buffers deliberately under-sized to exercise the kernel's
    // debug_assert; the test is marked #[should_panic] (expected "process_block: mixin").
    unsafe {
        conv.process_block(
            &layer_buffer,
            &mut block,
            buffer_start,
            num_frames,
            Some(&mixin),
        );
    }
}

#[test]
fn new_rejects_mismatched_weight_len() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let wrong_weights = &raw_weights[..raw_weights.len() - 1]; // truncate by 1

    let err = A2GroupedConv1d::new(
        wrong_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("mismatched weight stream must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::WeightCountMismatch,
        "weight count inconsistent with geometry must map to WeightCountMismatch: {err}"
    );
}

#[test]
fn new_rejects_zero_groups() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 0;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, 2, 42);
    let err = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("zero groups must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::InvalidModelTopology,
        "zero groups must map to InvalidModelTopology: {err}"
    );
}

#[test]
fn new_rejects_in_ch_not_divisible_by_groups() {
    let in_ch = 7;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let err = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("in_ch not divisible by groups must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::InvalidModelTopology,
        "non-divisor in_ch/groups must map to InvalidModelTopology: {err}"
    );
}

#[test]
fn new_rejects_out_ch_not_divisible_by_groups() {
    let in_ch = 8;
    let out_ch = 5;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let err = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("out_ch not divisible by groups must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::InvalidModelTopology,
        "non-divisor out_ch/groups must map to InvalidModelTopology: {err}"
    );
}

/// F-RES2-11: `kernel` above `MAX_KERNEL` (16) must be rejected by the
/// constructor itself — the SIMD kernels read taps through a fixed
/// 16-entry pointer array, and the loader's upstream ceiling must not be
/// the only line of defense for the public constructor path.
#[test]
fn new_rejects_kernel_above_max_kernel() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 17; // MAX_KERNEL + 1
    let groups = 2;

    // Weights sized exactly for the geometry so only the kernel ceiling fires.
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let err = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("kernel above MAX_KERNEL must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::InvalidModelTopology,
        "kernel above MAX_KERNEL must map to InvalidModelTopology: {err}"
    );
}

/// Companion of `new_rejects_kernel_above_max_kernel`: zero kernel is the
/// other endpoint of the `Conv1dDyn::try_from_parts` ceiling contract.
#[test]
fn new_rejects_zero_kernel() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 0;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let err = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("zero kernel must be rejected, not panic");
    assert_eq!(
        err,
        NamErrorCode::InvalidModelTopology,
        "zero kernel must map to InvalidModelTopology: {err}"
    );
}

/// The ceiling is inclusive: `kernel == MAX_KERNEL` with a consistent weight
/// stream must still construct (no valid fixture rejection).
#[test]
fn new_accepts_kernel_at_max_kernel() {
    let in_ch = 4;
    let out_ch = 4;
    let kernel = 16; // == MAX_KERNEL
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let conv = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("kernel == MAX_KERNEL must construct with a consistent weight stream");
    assert_eq!(conv.kernel, kernel);
}

// -----------------------------------------------------------------------------
// R-18 — Bias contract tests for A2GroupedConv1d::new
// -----------------------------------------------------------------------------

#[test]
fn new_rejects_missing_bias_when_do_bias_true() {
    // R-18: bias ausente com do_bias=true deve ser rejeitado com WeightCountMismatch
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;
    let (raw_weights, _) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);

    let err = A2GroupedConv1d::new(
        &raw_weights,
        &[], // missing bias
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("missing bias with do_bias=true must be rejected");
    assert_eq!(err, NamErrorCode::WeightCountMismatch);
}

#[test]
fn new_rejects_short_bias_when_do_bias_true() {
    // R-18: bias curto (len < out_ch) com do_bias=true deve ser rejeitado com WeightCountMismatch
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let short_bias = &raw_bias[..out_ch - 1]; // 3 elements instead of 4

    let err = A2GroupedConv1d::new(
        &raw_weights,
        short_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("short bias with do_bias=true must be rejected");
    assert_eq!(err, NamErrorCode::WeightCountMismatch);
}

#[test]
fn new_rejects_long_bias_when_do_bias_true() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let mut long_bias = raw_bias.clone();
    long_bias.push(0.5); // 5 elements instead of 4

    let err = A2GroupedConv1d::new(
        &raw_weights,
        &long_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .err()
    .expect("oversized bias with do_bias=true must be rejected");
    assert_eq!(err, NamErrorCode::WeightCountMismatch);
}

#[test]
fn new_accepts_exact_bias_and_matches_reference() {
    // R-18: bias exato com do_bias=true constrói e mantém paridade com referência
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;
    let dilation = 2;
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 77);

    let conv = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("exact bias with do_bias=true must construct");
    assert_eq!(conv.bias.len(), out_ch);

    let buf_frames = 64;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 55);
    let frame_idx = 30;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut ref_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to kernel contract (`layer_buffer` = buf_frames * in_ch,
    // `simd_out` = out_ch, `frame_idx` within history); pointers remain valid for call.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }
    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut ref_out,
    );

    for c in 0..out_ch {
        assert!(
            (simd_out[c] - ref_out[c]).abs() < 1e-5,
            "channel {c} mismatch: simd={}, ref={}",
            simd_out[c],
            ref_out[c]
        );
    }
}

#[test]
fn new_accepts_missing_bias_when_do_bias_false() {
    // R-18: bias ausente com do_bias=false constrói com bias pre-alocado seguro
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;
    let dilation = 1;
    let (raw_weights, _) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 99);

    let conv = A2GroupedConv1d::new(
        &raw_weights,
        &[], // missing bias is valid when do_bias=false
        false,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("missing bias with do_bias=false must construct");
    assert_eq!(conv.bias.len(), out_ch);
    assert!(conv.bias.iter().all(|&b| b == 0.0));

    let buf_frames = 64;
    let layer_buffer = make_layer_buffer(buf_frames, in_ch, 12);
    let frame_idx = 20;

    let mut simd_out = vec![0.0f32; out_ch];
    let mut ref_out = vec![0.0f32; out_ch];

    // SAFETY: buffers sized to kernel contract (`layer_buffer` = buf_frames * in_ch,
    // `simd_out` = out_ch, `frame_idx` within history); pointers remain valid for call.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut simd_out, frame_idx, None);
    }
    grouped_conv1d_single_frame_ref(
        &conv.weights,
        &conv.bias,
        conv.do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
        &layer_buffer,
        frame_idx,
        &mut ref_out,
    );

    for c in 0..out_ch {
        assert!(
            (simd_out[c] - ref_out[c]).abs() < 1e-5,
            "channel {c} mismatch with do_bias=false: simd={}, ref={}",
            simd_out[c],
            ref_out[c]
        );
    }
}

#[test]
fn new_rejects_geometry_arithmetic_overflow() {
    // Hostile geometry with huge dimensions must fail safely via checked math
    let in_ch = usize::MAX / 2;
    let out_ch = usize::MAX / 2;
    let kernel = 4;
    let groups = 1;

    let res = A2GroupedConv1d::new(&[], &[], false, 1, in_ch, out_ch, kernel, groups);
    assert!(res.is_err());
}

// =============================================================================
// T05.1 — Frozen bit-exact goldens (recorded before the SoA repack)
// =============================================================================

fn run_frozen_case(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    seed: u32,
    frame_idx: usize,
    do_bias: bool,
) -> Vec<u32> {
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, seed);
    let conv = A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("frozen golden case must construct");

    let layer_buffer = make_layer_buffer(frame_idx + 32, in_ch, seed ^ 0x5a5a_5a5a);
    let mut out = vec![0.0f32; out_ch];
    // SAFETY: `layer_buffer` covers the full lookback of `frame_idx`, `out`
    // has `out_ch` elements, and both outlive the call.
    unsafe {
        grouped_conv1d_single_frame_simd(&conv, &layer_buffer, &mut out, frame_idx, None);
    }
    out.into_iter().map(f32::to_bits).collect()
}

#[test]
fn test_grouped_conv1d_frozen_golden_bits() {
    // Configurations absent from the pre-T05.1 matrix: out_per_group % 4 != 0,
    // in_per_group > 4, kernel == 1, depthwise remainder `ch % 8 != 0`, and
    // depthwise multi-vector `ch > 8` with an odd kernel.
    let general_odd = run_frozen_case(6, 9, 4, 3, 3, 2026, 300, true);
    let general_k1 = run_frozen_case(10, 6, 1, 1, 2, 2028, 64, false);
    let depthwise_tail = run_frozen_case(6, 6, 2, 1, 6, 2027, 64, true);
    let depthwise_wide = run_frozen_case(16, 16, 5, 2, 16, 2029, 64, true);

    // Measured on the pre-SoA SIMD kernel (2026-10-28). These bit patterns are
    // the byte-for-byte contract the tap-major repack must reproduce.
    assert_eq!(
        general_odd,
        vec![
            3204644338, 3207609294, 1042346023, 1046741304, 1051990957, 1038983794, 3193255661,
            3196502571, 1017089004
        ]
    );
    assert_eq!(
        general_k1,
        vec![
            3198560733, 1045108557, 1046439687, 3158382694, 3190153347, 3192983711
        ]
    );
    assert_eq!(
        depthwise_tail,
        vec![
            1017000571, 3165386787, 3160250785, 3192770068, 1034789874, 3180270534
        ]
    );
    assert_eq!(
        depthwise_wide,
        vec![
            3199588362, 1042091507, 1055593546, 3200365944, 1050952083, 3205512766, 1049894644,
            3181343879, 3180673444, 3181944322, 3159922061, 3190867284, 1045042188, 1035457602,
            3189191373, 1030845720
        ]
    );
}

// =============================================================================
// T05.2/T05.3 — Bit-exact production dispatch after the tap-major repack
// =============================================================================

fn old_general_tree(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
) -> Vec<f32> {
    let in_pg = conv.in_per_group;
    let out_pg = conv.out_per_group;
    let nb = conv.num_blocks_per_group;
    let k_count = conv.kernel;
    let mut out = vec![0.0f32; conv.out_ch];

    for g in 0..conv.groups {
        for b in 0..nb {
            let out_c = g * out_pg + b * 4;
            let mut acc = [0.0f32; 4];
            for (lane, slot) in acc.iter_mut().enumerate() {
                let idx = out_c + lane;
                if idx < conv.out_ch {
                    *slot = if conv.do_bias { conv.bias[idx] } else { 0.0 };
                    if let Some(m) = mixin {
                        *slot += m[idx];
                    }
                }
            }

            for k in 0..k_count {
                let taps_back = k_count - 1 - k;
                let col = frame_idx - conv.dilation * taps_back;
                for ic in 0..in_pg {
                    let h = layer_buffer[col * conv.in_ch + g * in_pg + ic];
                    let w_base = g * (nb * k_count * in_pg * 4)
                        + b * (k_count * in_pg * 4)
                        + k * (in_pg * 4)
                        + ic * 4;
                    for (lane, slot) in acc.iter_mut().enumerate() {
                        *slot = conv.weights[w_base + lane].mul_add(h, *slot);
                    }
                }
            }

            for (lane, &val) in acc.iter().enumerate() {
                if out_c + lane < conv.out_ch {
                    out[out_c + lane] = val;
                }
            }
        }
    }
    out
}

fn old_depthwise_tree(
    conv: &A2GroupedConv1d,
    layer_buffer: &[f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0.0f32; conv.out_ch];
    for c in 0..conv.out_ch {
        let mut even = 0.0f32;
        let mut odd = 0.0f32;
        let mut k = 0;
        while k + 1 < conv.kernel {
            let tap0 = frame_idx - conv.dilation * (conv.kernel - 1 - k);
            let tap1 = frame_idx - conv.dilation * (conv.kernel - 1 - (k + 1));
            let w0 = conv.weights[c * (conv.kernel * 4) + k * 4];
            let w1 = conv.weights[c * (conv.kernel * 4) + (k + 1) * 4];
            even = w0.mul_add(layer_buffer[tap0 * conv.in_ch + c], even);
            odd = w1.mul_add(layer_buffer[tap1 * conv.in_ch + c], odd);
            k += 2;
        }
        if k < conv.kernel {
            let tap = frame_idx - conv.dilation * (conv.kernel - 1 - k);
            let w = conv.weights[c * (conv.kernel * 4) + k * 4];
            even = w.mul_add(layer_buffer[tap * conv.in_ch + c], even);
        }
        let mut acc = even + odd;
        if conv.do_bias {
            acc += conv.bias[c];
        }
        if let Some(m) = mixin {
            acc += m[c];
        }
        out[c] = acc;
    }
    out
}

fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (c, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{label}: channel {c} not bit-exact (actual={a}, expected={e})"
        );
    }
}

fn build_conv(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    seed: u32,
    do_bias: bool,
) -> A2GroupedConv1d {
    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, seed);
    A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        do_bias,
        dilation,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("bit-exact test conv must construct")
}

#[test]
fn test_grouped_conv1d_bit_exact_vs_legacy_tree_general() {
    let conv = build_conv(6, 9, 4, 3, 3, 2026, true);
    let frame_idx = 300;
    let layer_buffer = make_layer_buffer(frame_idx + 32, conv.in_ch, 77);
    let mixin: Vec<f32> = (0..conv.out_ch).map(|c| c as f32 * 0.125 - 0.5).collect();

    let mut actual = vec![0.0f32; conv.out_ch];
    // SAFETY: `layer_buffer` covers the full lookback of `frame_idx`, `actual`
    // has `out_ch` elements and `mixin` has `out_ch` elements.
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut actual, frame_idx, Some(&mixin));
    }
    let expected = old_general_tree(&conv, &layer_buffer, frame_idx, Some(&mixin));
    assert_bits_eq("general groups=3", &actual, &expected);
}

#[test]
fn test_grouped_conv1d_bit_exact_vs_legacy_tree_depthwise_tail() {
    let conv = build_conv(6, 6, 2, 1, 6, 2027, true);
    let frame_idx = 64;
    let layer_buffer = make_layer_buffer(frame_idx + 32, conv.in_ch, 78);
    let mixin: Vec<f32> = (0..conv.out_ch)
        .map(|c| 0.25 - c as f32 * 0.03125)
        .collect();

    let mut actual = vec![0.0f32; conv.out_ch];
    // SAFETY: as above; this shape exercises the `ch % 8` scalar tail.
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut actual, frame_idx, Some(&mixin));
    }
    let expected = old_depthwise_tree(&conv, &layer_buffer, frame_idx, Some(&mixin));
    assert_bits_eq("depthwise ch=6", &actual, &expected);
}

#[test]
fn test_grouped_conv1d_bit_exact_vs_legacy_tree_depthwise_wide() {
    let conv = build_conv(16, 16, 5, 2, 16, 2029, true);
    let frame_idx = 64;
    let layer_buffer = make_layer_buffer(frame_idx + 32, conv.in_ch, 79);

    let mut actual = vec![0.0f32; conv.out_ch];
    // SAFETY: as above; this shape exercises two full 8-lane vectors and the
    // odd-kernel final tap.
    unsafe {
        conv.process_single_frame(&layer_buffer, &mut actual, frame_idx, None);
    }
    let expected = old_depthwise_tree(&conv, &layer_buffer, frame_idx, None);
    assert_bits_eq("depthwise ch=16", &actual, &expected);
}
