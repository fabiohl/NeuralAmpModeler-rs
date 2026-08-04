// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::MAX_KERNEL_FRAMES;
use crate::models::a2::params::A2_LEAKY_SLOPE;

/// Scalar reference for a single-frame CH=8 dilated conv.
///
/// Replicates the exact math using 64-bit accumulators for maximum precision,
/// matching the semantics of the AVX2 broadcast-FMA kernel.
pub fn conv1d_ch8_single_frame_ref(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    debug_assert!(out_frame.len() >= 8);
    debug_assert!(weights.len() >= kernel * 64);

    out_frame[..8].copy_from_slice(&bias[..8]);

    for k in 0..kernel {
        let wk_base = k * 64;
        let taps_back = kernel - 1 - k;
        let col = frame_idx.wrapping_sub(dilation * taps_back);
        let src_base = col * 8;

        for inp in 0..8 {
            let hv = layer_buffer[src_base + inp];
            let w_base = wk_base + inp * 8;
            for out in 0..8 {
                out_frame[out] += weights[w_base + out] * hv;
            }
        }
    }
}

/// Scalar reference for a block of `num_frames` consecutive frames.
pub fn conv1d_ch8_block_ref(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    z_out: &mut [f32],
) {
    debug_assert!(z_out.len() >= num_frames * 8);

    for f in 0..num_frames {
        conv1d_ch8_single_frame_ref(
            weights,
            bias,
            dilation,
            kernel,
            layer_buffer,
            frame_start + f,
            &mut z_out[f * 8..(f + 1) * 8],
        );
    }
}

/// Scalar reference for the full CH=8 layer forward pass.
///
/// Matches `layer_forward_ch8_block` semantics exactly. Used as oracle
/// for parity testing.
#[expect(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "Audio DSP kernel with many dimension parameters and explicit SIMD indexing — struct consolidation would add indirection overhead in the hot path"
)]
pub fn layer_forward_ch8_scalar_ref(
    conv_weights: &[f32],
    conv_bias: &[f32],
    dilation: usize,
    kernel: usize,
    mixin_w: &[f32],
    l1x1_w: &[f32],
    l1x1_b: &[f32],
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    input_cond: &[f32],
    head_accum: &mut [f32],
    head_col: usize,
    layer_in: &mut [f32],
    is_first: bool,
    is_last: bool,
) {
    let ch: usize = 8;
    debug_assert!(num_frames <= MAX_KERNEL_FRAMES);
    let mut z_buf = [0.0f32; MAX_KERNEL_FRAMES * 8];

    conv1d_ch8_block_ref(
        conv_weights,
        conv_bias,
        dilation,
        kernel,
        layer_buffer,
        frame_start,
        num_frames,
        &mut z_buf[..num_frames * ch],
    );

    for f in 0..num_frames {
        let off = f * ch;
        for c in 0..ch {
            z_buf[off + c] += mixin_w[c] * input_cond[f];
            if z_buf[off + c] < 0.0 {
                z_buf[off + c] *= A2_LEAKY_SLOPE;
            }
        }
        let head_off = (head_col + f) * ch;
        if is_first {
            head_accum[head_off..head_off + ch].copy_from_slice(&z_buf[off..off + ch]);
        } else {
            for c in 0..ch {
                head_accum[head_off + c] += z_buf[off + c];
            }
        }
    }

    if !is_last {
        for f in 0..num_frames {
            let off = f * ch;
            for c in 0..ch {
                let mut sum = l1x1_b[c];
                for u in 0..ch {
                    sum += l1x1_w[u * ch + c] * z_buf[off + u];
                }
                layer_in[off + c] += sum;
            }
        }
    }
}
