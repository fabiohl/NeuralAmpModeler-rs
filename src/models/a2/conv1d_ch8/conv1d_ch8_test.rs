// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// Bit-exact verification against scalar reference uses explicit range loops.
#![allow(clippy::needless_range_loop)]

use super::*;
use crate::math::common::AlignedVec;
use crate::models::a2::film::{FiLMConfig, FiLMLayer, FilmBlock};

fn make_random_weights(kernel: usize, seed: u32) -> (AlignedVec<f32>, AlignedVec<f32>) {
    let mut w = AlignedVec::new(kernel * 64, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    let mut bias =
        AlignedVec::new(8, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut state = seed;
    for val in w.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = (state as f32 / u32::MAX as f32) * 0.5 - 0.25;
    }
    for val in bias.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = (state as f32 / u32::MAX as f32) * 0.2 - 0.1;
    }
    (w, bias)
}

fn make_history(cols: usize, seed: u32) -> Vec<f32> {
    let mut buf = vec![0.0f32; cols * 8];
    let mut state = seed;
    for val in buf.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *val = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    buf
}

fn make_cond(num_frames: usize) -> Vec<f32> {
    (0..num_frames)
        .map(|i| (i as f32 * 0.7).sin() * 0.5)
        .collect()
}

// ── Conv-only parity tests ─────────────────────────────────────────

#[test]
fn test_conv1d_ch8_k6_parity() {
    let kernel = 6;
    let dilation = 101; // A2_DILATIONS[5]
    let (w, b) = make_random_weights(kernel, 42);
    let num_frames = 16;

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 77);
    let frame_start = max_lookback + 4;

    let mut z_simd = vec![0.0f32; num_frames * 8];
    let mut z_ref = vec![0.0f32; num_frames * 8];

    // SAFETY: `w` (kernel*64), `b` (8), `history` (hist_cols*8) and `z_simd` (num_frames*8) are sized
    // to the `conv1d_ch8_t8_avx2` contract and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        conv1d_ch8_t8_avx2(
            &w,
            &b,
            dilation,
            kernel,
            &history,
            frame_start,
            num_frames,
            &mut z_simd,
        );
    }

    conv1d_ch8_block_ref(
        &w,
        &b,
        dilation,
        kernel,
        &history,
        frame_start,
        num_frames,
        &mut z_ref,
    );

    for i in 0..num_frames * 8 {
        let diff = (z_simd[i] - z_ref[i]).abs();
        assert!(
            diff < 5e-5,
            "z[{}]: simd={}, ref={}, diff={}",
            i,
            z_simd[i],
            z_ref[i],
            diff
        );
    }
}

#[test]
fn test_conv1d_ch8_k15_parity() {
    let kernel = 15;
    let dilation = 13; // A2_DILATIONS[15]
    let (w, b) = make_random_weights(kernel, 99);
    let num_frames = 16;

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 88);
    let frame_start = max_lookback + 4;

    let mut z_simd = vec![0.0f32; num_frames * 8];
    let mut z_ref = vec![0.0f32; num_frames * 8];

    // SAFETY: `w` (kernel*64), `b` (8), `history` (hist_cols*8) and `z_simd` (num_frames*8) are sized
    // to the `conv1d_ch8_t8_avx2` contract and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        conv1d_ch8_t8_avx2(
            &w,
            &b,
            dilation,
            kernel,
            &history,
            frame_start,
            num_frames,
            &mut z_simd,
        );
    }

    conv1d_ch8_block_ref(
        &w,
        &b,
        dilation,
        kernel,
        &history,
        frame_start,
        num_frames,
        &mut z_ref,
    );

    for i in 0..num_frames * 8 {
        let diff = (z_simd[i] - z_ref[i]).abs();
        assert!(
            diff < 5e-5,
            "z[{}]: simd={}, ref={}, diff={}",
            i,
            z_simd[i],
            z_ref[i],
            diff
        );
    }
}

#[test]
fn test_conv1d_ch8_t8_tail_parity() {
    let kernel = 6;
    let dilation = 1;
    let (w, b) = make_random_weights(kernel, 55);
    // Odd number of frames to exercise scalar tail.
    let num_frames = 7;

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 33);
    let frame_start = max_lookback + 2;

    let mut z_simd = vec![0.0f32; num_frames * 8];
    let mut z_ref = vec![0.0f32; num_frames * 8];

    // SAFETY: `w` (kernel*64), `b` (8), `history` (hist_cols*8) and `z_simd` (num_frames*8) are sized
    // to the `conv1d_ch8_t8_avx2` contract and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        conv1d_ch8_t8_avx2(
            &w,
            &b,
            dilation,
            kernel,
            &history,
            frame_start,
            num_frames,
            &mut z_simd,
        );
    }

    conv1d_ch8_block_ref(
        &w,
        &b,
        dilation,
        kernel,
        &history,
        frame_start,
        num_frames,
        &mut z_ref,
    );

    for i in 0..num_frames * 8 {
        let diff = (z_simd[i] - z_ref[i]).abs();
        assert!(
            diff < 5e-5,
            "z[{}]: simd={}, ref={}, diff={}",
            i,
            z_simd[i],
            z_ref[i],
            diff
        );
    }
}

#[test]
fn test_conv1d_ch8_z_1_frame() {
    // Single frame: no T=8 tiling, pure scalar tail path.
    let kernel = 15;
    let dilation = 239;
    let (w, b) = make_random_weights(kernel, 111);
    let num_frames = 1;

    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 44);
    let frame_start = max_lookback + 1;

    let mut z_simd = vec![0.0f32; num_frames * 8];
    let mut z_ref = vec![0.0f32; num_frames * 8];

    // SAFETY: `w` (kernel*64), `b` (8), `history` (hist_cols*8) and `z_simd` (num_frames*8) are sized
    // to the `conv1d_ch8_t8_avx2` contract and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        conv1d_ch8_t8_avx2(
            &w,
            &b,
            dilation,
            kernel,
            &history,
            frame_start,
            num_frames,
            &mut z_simd,
        );
    }

    conv1d_ch8_block_ref(
        &w,
        &b,
        dilation,
        kernel,
        &history,
        frame_start,
        num_frames,
        &mut z_ref,
    );

    for i in 0..num_frames * 8 {
        let diff = (z_simd[i] - z_ref[i]).abs();
        assert!(
            diff < 5e-5,
            "z[{}]: simd={}, ref={}, diff={}",
            i,
            z_simd[i],
            z_ref[i],
            diff
        );
    }
}

#[test]
fn test_conv1d_ch8_a2conv1dch8_constructor() {
    // Verify A2Conv1dCh8 correctly permutes from NAM JSON order to col-major-per-tap.
    let kernel = 6;
    let raw_len = 8 * 8 * kernel; // 384
    let mut raw = vec![0.0f32; raw_len];
    // Fill with known values: raw[out * 8 * K + in * K + k] = out * 100 + in * 10 + k
    for out in 0..8 {
        for inp in 0..8 {
            for k in 0..kernel {
                raw[out * 8 * kernel + inp * kernel + k] =
                    (out as f32) * 100.0 + (inp as f32) * 10.0 + (k as f32);
            }
        }
    }
    let bias = AlignedVec::from_vec(vec![0.0f32; 8])
        .expect("allocation should succeed for test-sized buffers");
    let conv = A2Conv1dCh8::new(&raw, 8, 8, kernel, 1, &bias)
        .expect("construction should succeed for test-sized buffers");

    // Verify: conv.weights[k * 64 + in * 8 + out] == raw[out * 8 * K + in * K + k]
    for out in 0..8 {
        for inp in 0..8 {
            for k in 0..kernel {
                let expected = (out as f32) * 100.0 + (inp as f32) * 10.0 + (k as f32);
                let actual = conv.weights[k * 64 + inp * 8 + out];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "w[k={} in={} out={}]: expected {}, got {}",
                    k,
                    inp,
                    out,
                    expected,
                    actual
                );
            }
        }
    }
}

// ── Full-layer forward parity tests ────────────────────────────────

#[test]
fn test_layer_forward_ch8_k6_parity() {
    let kernel = 6;
    let dilation = 7; // A2_DILATIONS[2]
    let (w, b) = make_random_weights(kernel, 42);
    let conv = A2Conv1dCh8::new(&w, 8, 8, kernel, dilation, &b)
        .expect("construction should succeed for test-sized buffers");

    let mut mixin_w_vec =
        AlignedVec::new(8, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut l1x1_w_vec =
        AlignedVec::new(64, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut l1x1_b_vec =
        AlignedVec::new(8, 0.0f32).expect("allocation should succeed for test-sized buffers");
    let mut state: u32 = 100;
    for v in mixin_w_vec.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (state as f32 / u32::MAX as f32) * 0.5 - 0.25;
    }
    for v in l1x1_w_vec.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (state as f32 / u32::MAX as f32) * 0.8 - 0.4;
    }
    for v in l1x1_b_vec.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (state as f32 / u32::MAX as f32) * 0.2 - 0.1;
    }

    let num_frames = 16;
    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 77);
    let frame_start = max_lookback + 4;
    let cond = make_cond(num_frames);

    let mut head_simd = vec![0.0f32; (num_frames + 1) * 8];
    let mut head_ref = vec![0.0f32; (num_frames + 1) * 8];
    let mut layer_in_simd = vec![0.0f32; num_frames * 8];
    let mut layer_in_ref = vec![0.0f32; num_frames * 8];
    let mut fb = FilmBlock::empty();

    // SAFETY: `history`, `cond`, `head_simd` ((num_frames+1)*8), `layer_in_simd` (num_frames*8),
    // `mixin_w_vec`, `l1x1_w_vec` and `l1x1_b_vec` are sized to `layer_forward_ch8_block`'s contract
    // and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        layer_forward_ch8_block(
            &conv,
            &mixin_w_vec,
            &l1x1_w_vec,
            &l1x1_b_vec,
            &mut fb,
            false,
            &history,
            frame_start,
            num_frames,
            &cond,
            &mut head_simd,
            0,
            &mut layer_in_simd,
            true,  // is_first
            false, // is_last
        );
    }

    layer_forward_ch8_scalar_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &mixin_w_vec,
        &l1x1_w_vec,
        &l1x1_b_vec,
        &history,
        frame_start,
        num_frames,
        &cond,
        &mut head_ref,
        0,
        &mut layer_in_ref,
        true,
        false,
    );

    // Head comparison.
    for i in 0..num_frames * 8 {
        let diff = (head_simd[i] - head_ref[i]).abs();
        assert!(
            diff < 1e-4,
            "head[{}]: simd={}, ref={}, diff={}",
            i,
            head_simd[i],
            head_ref[i],
            diff
        );
    }

    // Layer_in comparison.
    for i in 0..num_frames * 8 {
        let diff = (layer_in_simd[i] - layer_in_ref[i]).abs();
        assert!(
            diff < 1e-4,
            "layer_in[{}]: simd={}, ref={}, diff={}",
            i,
            layer_in_simd[i],
            layer_in_ref[i],
            diff
        );
    }
}

#[test]
fn test_layer_forward_ch8_k15_last_layer_parity() {
    let kernel = 15;
    let dilation = 13;
    let (w, b) = make_random_weights(kernel, 88);
    let conv = A2Conv1dCh8::new(&w, 8, 8, kernel, dilation, &b)
        .expect("construction should succeed for test-sized buffers");

    let mixin_w_vec = AlignedVec::from_vec(vec![0.1f32; 8])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w_vec = AlignedVec::from_vec(vec![0.5f32; 64])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_b_vec = AlignedVec::from_vec(vec![0.0f32; 8])
        .expect("allocation should succeed for test-sized buffers");

    let num_frames = 16;
    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 55);
    let frame_start = max_lookback + 4;
    let cond = make_cond(num_frames);

    let mut head_simd = vec![0.0f32; (num_frames + 1) * 8];
    let mut head_ref = vec![0.0f32; (num_frames + 1) * 8];
    let mut layer_in_simd = vec![1.0f32; num_frames * 8];
    let mut layer_in_ref = vec![1.0f32; num_frames * 8];
    let mut fb = FilmBlock::empty();

    // SAFETY: `history`, `cond`, `head_simd` ((num_frames+1)*8), `layer_in_simd` (num_frames*8),
    // `mixin_w_vec`, `l1x1_w_vec` and `l1x1_b_vec` are sized to `layer_forward_ch8_block`'s contract
    // and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        layer_forward_ch8_block(
            &conv,
            &mixin_w_vec,
            &l1x1_w_vec,
            &l1x1_b_vec,
            &mut fb,
            false,
            &history,
            frame_start,
            num_frames,
            &cond,
            &mut head_simd,
            0,
            &mut layer_in_simd,
            false, // not first → accumulate
            true,  // is_last → skip l1x1
        );
    }

    layer_forward_ch8_scalar_ref(
        &conv.weights,
        &conv.bias,
        dilation,
        kernel,
        &mixin_w_vec,
        &l1x1_w_vec,
        &l1x1_b_vec,
        &history,
        frame_start,
        num_frames,
        &cond,
        &mut head_ref,
        0,
        &mut layer_in_ref,
        false,
        true,
    );

    // Head comparison.
    for i in 0..num_frames * 8 {
        let diff = (head_simd[i] - head_ref[i]).abs();
        assert!(
            diff < 1e-4,
            "head[{}]: simd={}, ref={}, diff={}",
            i,
            head_simd[i],
            head_ref[i],
            diff
        );
    }

    // Layer_in MUST be unchanged (is_last = true → skip l1x1).
    for i in 0..num_frames * 8 {
        assert!(
            (layer_in_simd[i] - 1.0).abs() < 1e-6,
            "last layer should not update layer_in[{}], got {}",
            i,
            layer_in_simd[i]
        );
    }
}

#[test]
fn test_layer_forward_ch8_middle_layer_accumulates() {
    let kernel = 6;
    let dilation = 1;
    let (w, b) = make_random_weights(kernel, 33);
    let conv = A2Conv1dCh8::new(&w, 8, 8, kernel, dilation, &b)
        .expect("construction should succeed for test-sized buffers");

    let mixin_w_vec = AlignedVec::from_vec(vec![0.1f32; 8])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w_vec = AlignedVec::from_vec(vec![0.0f32; 64])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_b_vec = AlignedVec::from_vec(vec![0.0f32; 8])
        .expect("allocation should succeed for test-sized buffers");

    let num_frames = 8;
    let max_lookback = (kernel - 1) * dilation;
    let hist_cols = max_lookback + num_frames + 8;
    let history = make_history(hist_cols, 22);
    let frame_start = max_lookback + 2;
    let cond = make_cond(num_frames);

    // First pass: is_first=true → assign.
    let mut head = vec![0.0f32; num_frames * 8];
    let mut head_copy = vec![0.0f32; num_frames * 8];
    let mut layer_in = vec![0.0f32; num_frames * 8];
    let mut fb = FilmBlock::empty();

    // SAFETY: `history`, `cond`, `head` ((num_frames)*8), `layer_in` (num_frames*8),
    // `mixin_w_vec`, `l1x1_w_vec` and `l1x1_b_vec` are sized to `layer_forward_ch8_block`'s contract
    // and outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        layer_forward_ch8_block(
            &conv,
            &mixin_w_vec,
            &l1x1_w_vec,
            &l1x1_b_vec,
            &mut fb,
            false,
            &history,
            frame_start,
            num_frames,
            &cond,
            &mut head,
            0,
            &mut layer_in,
            true,
            true,
        );
    }
    head_copy.copy_from_slice(&head);

    // Second pass: is_first=false → accumulate. Head values should change.
    // SAFETY: `history`, `cond`, `head` (num_frames*8), `layer_in` (num_frames*8), `mixin_w_vec`,
    // `l1x1_w_vec` and `l1x1_b_vec` are sized to `layer_forward_ch8_block`'s contract and outlive
    // the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        layer_forward_ch8_block(
            &conv,
            &mixin_w_vec,
            &l1x1_w_vec,
            &l1x1_b_vec,
            &mut fb,
            false,
            &history,
            frame_start,
            num_frames,
            &cond,
            &mut head,
            0,
            &mut layer_in,
            false,
            true,
        );
    }

    let mut changed = false;
    for i in 0..num_frames * 8 {
        if (head[i] - head_copy[i]).abs() > 1e-6 {
            changed = true;
        }
    }
    assert!(
        changed,
        "Middle layer (is_first=false) should accumulate, but head unchanged"
    );
}

fn make_film_identity_8(groups: u32) -> FiLMLayer {
    let config = FiLMConfig {
        active: true,
        shift: true,
        groups,
    };
    let cond_size = 1usize;
    let channels = 8usize;
    let w_count = groups as usize * (8 / groups as usize * 2) * (1 / groups as usize).max(1);
    let _ = w_count;
    let (w_count, b_count) = if cond_size > 1 {
        (0, 0)
    } else {
        let ch_per_group = channels / groups as usize;
        let rows = ch_per_group * 2 * (cond_size / groups as usize).max(1) * groups as usize;
        (rows, channels * 2)
    };
    let weights = vec![0.0f32; w_count];
    let mut bias = vec![0.0f32; b_count];
    for c in 0..channels {
        bias[c] = 1.0;
    }
    FiLMLayer::load(config, cond_size, channels, weights, bias)
        .expect("identity FiLM should load for test-sized buffers")
}

/// No-FiLM fast path is bit-exact vs the general block kernel.
///
/// Covers the Sprint 3 hoist: `mask == 0` must produce identical head and
/// layer_in as the `Option`-testing path for first/middle/last layers.
#[test]
fn test_ch8_no_film_fast_path_bit_exact() {
    for (is_first, is_last) in [(true, false), (false, false), (false, true)] {
        let kernel = 6;
        let dilation = 101;
        let (w, b) = make_random_weights(kernel, 7);
        let conv = A2Conv1dCh8::new(&w, 8, 8, kernel, dilation, &b)
            .expect("construction should succeed for test-sized buffers");
        let mixin_w_vec = AlignedVec::from_vec(vec![0.13f32; 8])
            .expect("allocation should succeed for test-sized buffers");
        let l1x1_w_vec = AlignedVec::from_vec(vec![0.31f32; 64])
            .expect("allocation should succeed for test-sized buffers");
        let l1x1_b_vec = AlignedVec::from_vec(vec![0.02f32; 8])
            .expect("allocation should succeed for test-sized buffers");
        let num_frames = 16;
        let max_lookback = (kernel - 1) * dilation;
        let history = make_history(max_lookback + num_frames + 8, 21);
        let frame_start = max_lookback + 4;
        let cond = make_cond(num_frames);

        let mut head_fast = vec![0.5f32; (num_frames + 1) * 8];
        let mut head_ref = head_fast.clone();
        let mut lin_fast = vec![0.25f32; num_frames * 8];
        let mut lin_ref = lin_fast.clone();
        let mut fb = FilmBlock::empty();

        // SAFETY: buffers sized to `layer_forward_ch8_block`'s contract and
        // outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
        unsafe {
            layer_forward_ch8_block_no_film(
                &conv,
                &mixin_w_vec,
                &l1x1_w_vec,
                &l1x1_b_vec,
                &history,
                frame_start,
                num_frames,
                &cond,
                &mut head_fast,
                0,
                &mut lin_fast,
                is_first,
                is_last,
            );
            layer_forward_ch8_block(
                &conv,
                &mixin_w_vec,
                &l1x1_w_vec,
                &l1x1_b_vec,
                &mut fb,
                false,
                &history,
                frame_start,
                num_frames,
                &cond,
                &mut head_ref,
                0,
                &mut lin_ref,
                is_first,
                is_last,
            );
        }

        for i in 0..num_frames * 8 {
            assert!(
                head_fast[i] == head_ref[i],
                "head[{i}] fast={} ref={} (first={is_first} last={is_last})",
                head_fast[i],
                head_ref[i]
            );
            assert!(
                lin_fast[i] == lin_ref[i],
                "layer_in[{i}] fast={} ref={} (first={is_first} last={is_last})",
                lin_fast[i],
                lin_ref[i]
            );
        }
    }
}

/// Active FiLM path stays on the general kernel (mask != 0) and the hoisted
/// presence mask matches the `Option` discriminants.
#[test]
fn test_ch8_film_active_mask_matches_options() {
    let kernel = 6;
    let dilation = 101;
    let (w, b) = make_random_weights(kernel, 13);
    let conv = A2Conv1dCh8::new(&w, 8, 8, kernel, dilation, &b)
        .expect("construction should succeed for test-sized buffers");
    let mixin_w_vec = AlignedVec::from_vec(vec![0.11f32; 8])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_w_vec = AlignedVec::from_vec(vec![0.29f32; 64])
        .expect("allocation should succeed for test-sized buffers");
    let l1x1_b_vec = AlignedVec::from_vec(vec![0.01f32; 8])
        .expect("allocation should succeed for test-sized buffers");
    let num_frames = 16;
    let max_lookback = (kernel - 1) * dilation;
    let history = make_history(max_lookback + num_frames + 8, 31);
    let frame_start = max_lookback + 4;
    let cond = make_cond(num_frames);

    let film = make_film_identity_8(1);
    let mut film = film;
    let mut head = vec![0.0f32; (num_frames + 1) * 8];
    let mut layer_in = vec![0.0f32; num_frames * 8];
    let mut fb = FilmBlock {
        conv_pre_film: None,
        conv_post_film: None,
        input_mixin_pre_film: None,
        input_mixin_post_film: None,
        activation_pre_film: None,
        activation_post_film: Some(&mut film),
        layer1x1_post_film: None,
        head1x1_post_film: None,
    };
    assert_eq!(fb.active_mask(), 1 << 4);

    // SAFETY: buffers sized to `layer_forward_ch8_block`'s contract and
    // outlive the call; AVX2+FMA is guaranteed by `#[target_feature]`.
    unsafe {
        layer_forward_ch8_block(
            &conv,
            &mixin_w_vec,
            &l1x1_w_vec,
            &l1x1_b_vec,
            &mut fb,
            false,
            &history,
            frame_start,
            num_frames,
            &cond,
            &mut head,
            0,
            &mut layer_in,
            true,
            false,
        );
    }
    assert!(head.iter().any(|v| v.is_finite()));
}
