// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

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
#[should_panic(expected = "raw_weights len")]
fn should_panic_new_mismatched_weight_len() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    let wrong_weights = &raw_weights[..raw_weights.len() - 1]; // truncate by 1

    A2GroupedConv1d::new(
        wrong_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("construction should succeed for test-sized buffers");
}

#[test]
#[should_panic(expected = "groups must be > 0")]
fn should_panic_new_zero_groups() {
    let in_ch = 8;
    let out_ch = 4;
    let kernel = 3;
    let groups = 0;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, 2, 42);
    A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("construction should succeed for test-sized buffers");
}

#[test]
#[should_panic(expected = "in_ch")]
fn should_panic_new_in_ch_not_divisible_by_groups() {
    let in_ch = 7;
    let out_ch = 4;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("construction should succeed for test-sized buffers");
}

#[test]
#[should_panic(expected = "out_ch")]
fn should_panic_new_out_ch_not_divisible_by_groups() {
    let in_ch = 8;
    let out_ch = 5;
    let kernel = 3;
    let groups = 2;

    let (raw_weights, raw_bias) = make_test_weights_grouped(in_ch, out_ch, kernel, groups, 42);
    A2GroupedConv1d::new(
        &raw_weights,
        &raw_bias,
        true,
        2,
        in_ch,
        out_ch,
        kernel,
        groups,
    )
    .expect("construction should succeed for test-sized buffers");
}
