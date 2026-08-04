// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn make_test_weights(channels: usize) -> (AlignedVec<f32>, f32, f32) {
    let k = A2HeadConv::HEAD_KERNEL_SIZE;
    let mut w = AlignedVec::new(k * channels, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let mut state: u32 = 42;
    for i in 0..k * channels {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        w[i] = ((state as f32) / (u32::MAX as f32)) * 0.5 - 0.25;
    }
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let bias = ((state as f32) / (u32::MAX as f32)) * 0.2 - 0.1;
    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let scale = ((state as f32) / (u32::MAX as f32)) * 0.5 + 0.75;
    (w, bias, scale)
}

fn make_test_history(channels: usize, cols: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; channels * cols];
    let mut state: u32 = 99;
    for val in &mut buf {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = ((state as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
    }
    buf
}

#[test]
fn test_a2_head_conv_ch3_parity() {
    let ch = 3;
    let (w, bias, scale) = make_test_weights(ch);
    let head = A2HeadConv::new(w.clone(), bias, scale, ch);

    let ring_size = 64;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 50;
    let num_frames = 8;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    head.process(&history, write_pos, ring_mask, num_frames, &mut output);
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-6,
            "CH=3 frame {}: proc={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_conv_ch8_parity() {
    let ch = 8;
    let (w, bias, scale) = make_test_weights(ch);
    let head = A2HeadConv::new(w.clone(), bias, scale, ch);

    let ring_size = 64;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 45;
    let num_frames = 16;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    head.process(&history, write_pos, ring_mask, num_frames, &mut output);
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-6,
            "CH=8 frame {}: proc={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_conv_ring_wraparound() {
    let ch = 3;
    let (w, bias, scale) = make_test_weights(ch);
    let head = A2HeadConv::new(w.clone(), bias, scale, ch);

    let ring_size = 32;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 5;
    let num_frames = 16;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    head.process(&history, write_pos, ring_mask, num_frames, &mut output);
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-6,
            "wrap frame {}: proc={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_conv_known_values_ch3() {
    let ch = 3;
    let k = A2HeadConv::HEAD_KERNEL_SIZE;

    let mut w =
        AlignedVec::new(k * ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let bias = 0.5;
    let scale = 2.0;

    // w[0][0] = 1.0, all others = 0
    w[0] = 1.0;

    let ring_size = 32;
    let ring_mask = ring_size - 1;
    let mut history = vec![0.0f32; ch * ring_size];

    let write_pos: usize = 20;
    let num_frames = 2;

    // At frame 0, tap 0 (oldest = K-1 back) should read col = write_pos - num_frames + 0 - (K-1 - 0) = write_pos - num_frames - (K-1)
    let col_expected =
        (write_pos as isize - num_frames as isize - (k as isize - 1)) as usize & ring_mask;
    history[col_expected * ch] = 3.0;

    let head = A2HeadConv::new(w, bias, scale, ch);
    let mut output = vec![0.0f32; num_frames];
    head.process(&history, write_pos, ring_mask, num_frames, &mut output);

    let expected = (bias + 1.0 * 3.0) * scale; // 0.5 + 3.0 = 3.5 * 2.0 = 7.0
    assert!(
        (output[0] - expected).abs() < 1e-6,
        "got {} expected {}",
        output[0],
        expected
    );

    // Frame 1 should be 0.5 * 2.0 = 1.0 (no non-zero history hit)
    let expected_f1 = bias * scale;
    assert!(
        (output[1] - expected_f1).abs() < 1e-6,
        "frame1: got {} expected {}",
        output[1],
        expected_f1
    );
}

#[test]
fn test_a2_head_conv_all_taps_ch8() {
    let ch = 8;
    let k = A2HeadConv::HEAD_KERNEL_SIZE;

    let mut w =
        AlignedVec::new(k * ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let bias = 0.1;
    let scale = 1.5;

    // Each tap has weight 1.0 for channel 0
    for t in 0..k {
        w[t * ch] = 1.0;
    }

    let ring_size = 64;
    let ring_mask = ring_size - 1;
    let mut history = vec![0.0f32; ch * ring_size];

    let write_pos: usize = 35;
    let num_frames = 1;

    // Fill all K lookback columns with value 1.0 at channel 0
    for t in 0..k {
        let col = (write_pos as isize - num_frames as isize - (k as isize - 1 - t as isize))
            as usize
            & ring_mask;
        history[col * ch] = 1.0;
    }

    let head = A2HeadConv::new(w, bias, scale, ch);
    let mut output = vec![0.0f32; num_frames];
    head.process(&history, write_pos, ring_mask, num_frames, &mut output);

    let expected = (bias + k as f32 * 1.0 * 1.0) * scale;
    assert!(
        (output[0] - expected).abs() < 1e-6,
        "all taps: got {} expected {}",
        output[0],
        expected
    );
}

#[test]
fn test_a2_head_conv_ch3_stepping_write_pos() {
    let ch = 3;
    let (w, bias, scale) = make_test_weights(ch);
    let head = A2HeadConv::new(w.clone(), bias, scale, ch);

    let ring_size = 64;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let num_frames = 4;

    for wp in [num_frames, num_frames + 10, num_frames + 20] {
        let write_pos = wp;
        let mut output = vec![0.0f32; num_frames];
        let mut scalar = vec![0.0f32; num_frames];

        head.process(&history, write_pos, ring_mask, num_frames, &mut output);
        a2_head_block_scalar_ref(
            &w,
            bias,
            scale,
            ch,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut scalar,
        );

        for f in 0..num_frames {
            let diff = (output[f] - scalar[f]).abs();
            assert!(
                diff < 1e-6,
                "wp={} frame {}: proc={}, ref={}, diff={}",
                wp,
                f,
                output[f],
                scalar[f],
                diff
            );
        }
    }
}

#[test]
fn test_a2_head_ch8_avx2_parity() {
    let ch = 8;
    let (w, bias, scale) = make_test_weights(ch);

    let ring_size = 64;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 45;
    let num_frames = 16;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    unsafe {
        head_process_ch8_avx2(
            &w,
            bias,
            scale,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut output,
        );
    }
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-5,
            "AVX2 CH=8 frame {}: avx2={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_ch8_avx2_parity_large_block() {
    let ch = 8;
    let (w, bias, scale) = make_test_weights(ch);

    let ring_size = 256;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 150;
    let num_frames = 128;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    unsafe {
        head_process_ch8_avx2(
            &w,
            bias,
            scale,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut output,
        );
    }
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-5,
            "AVX2 CH=8 large frame {}: avx2={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_ch8_avx2_wraparound() {
    let ch = 8;
    let (w, bias, scale) = make_test_weights(ch);

    let ring_size = 32;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 5;
    let num_frames = 16;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    unsafe {
        head_process_ch8_avx2(
            &w,
            bias,
            scale,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut output,
        );
    }
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-5,
            "AVX2 CH=8 wrap frame {}: avx2={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}

#[test]
fn test_a2_head_ch3_sse_parity_wraparound() {
    let ch = 3;
    let (w, bias, scale) = make_test_weights(ch);

    let ring_size = 32;
    let ring_mask = ring_size - 1;
    let history = make_test_history(ch, ring_size);
    let write_pos: usize = 3;
    let num_frames = 16;

    let mut output = vec![0.0f32; num_frames];
    let mut scalar = vec![0.0f32; num_frames];

    unsafe {
        head_process_ch3_sse(
            &w,
            bias,
            scale,
            &history,
            write_pos,
            ring_mask,
            num_frames,
            &mut output,
        );
    }
    a2_head_block_scalar_ref(
        &w,
        bias,
        scale,
        ch,
        &history,
        write_pos,
        ring_mask,
        num_frames,
        &mut scalar,
    );

    for f in 0..num_frames {
        let diff = (output[f] - scalar[f]).abs();
        assert!(
            diff < 1e-5,
            "SSE CH=3 wrap frame {}: sse={}, ref={}, diff={}",
            f,
            output[f],
            scalar[f],
            diff
        );
    }
}
