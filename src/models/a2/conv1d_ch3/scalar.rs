// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CH=3 A2 scalar reference implementations (oracle for parity tests).
//!
//! These functions are bit-exact with the SIMD kernels but use pure f32 scalar math.

use crate::models::a2::params::A2_LEAKY_SLOPE;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

/// Maximum frames per kernel invocation.
/// Guaranteed by `process()` internal chunking.
const MAX_KERNEL_FRAMES: usize = WAVENET_MAX_NUM_FRAMES;

// Scalar reference for A2Conv1dCh3 — oracle for parity tests
// =============================================================================

/// Scalar reference for a single-frame CH=3 dilated conv (f32 col-major weights).
///
/// Reads from `weights` in the same col-major-per-tap layout as `A2Conv1dCh3`.
pub fn conv1d_ch3_single_frame_ref(
    weights: &[f32],
    bias: &[f32],
    dilation: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    const CH: usize = 3;
    debug_assert!(out_frame.len() >= CH);
    debug_assert!(weights.len() >= kernel * 16);

    out_frame[..CH].copy_from_slice(&bias[..CH]);

    for k in 0..kernel {
        let taps_back = kernel - 1 - k;
        let col = frame_idx.wrapping_sub(dilation * taps_back);
        let src_base = col * CH;
        let wk_base = k * 16;

        for inp in 0..CH {
            let hv = layer_buffer[src_base + inp];
            let w_base = wk_base + inp * 4;
            for out in 0..CH {
                out_frame[out] += weights[w_base + out] * hv;
            }
        }
    }
}

// Scalar reference for full layer forward (oracle)
// =============================================================================

/// Scalar reference for the full CH=3 layer forward pass using f32 native weights.
///
/// Used as oracle in parity tests against `layer_forward_ch3_block`.
#[expect(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "Audio DSP kernel with many dimension parameters and explicit SIMD indexing — struct consolidation would add indirection overhead in the hot path"
)]
pub fn layer_forward_ch3_scalar_ref(
    conv_weights: &[f32], // col-major-per-tap f32 weights (A2Conv1dCh3 layout)
    conv_bias: &[f32],    // [4] bias (3 valid + 1 zero)
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
    const CH: usize = 3;
    const CH_PAD: usize = 4;
    debug_assert!(num_frames <= MAX_KERNEL_FRAMES); // process() guarantees ≤ MAX_KERNEL_FRAMES
    let mut z_buf = [0.0f32; MAX_KERNEL_FRAMES * CH_PAD];

    for f in 0..num_frames {
        let frame_idx = frame_start + f;
        let z_slice = &mut z_buf[f * CH_PAD..(f + 1) * CH_PAD];
        conv1d_ch3_single_frame_ref(
            conv_weights,
            conv_bias,
            dilation,
            kernel,
            layer_buffer,
            frame_idx,
            z_slice,
        );
    }

    for f in 0..num_frames {
        let z_off = f * CH_PAD;

        // Mixin
        for c in 0..CH {
            z_buf[z_off + c] += mixin_w[c] * input_cond[f];
        }

        // LeakyReLU
        for c in 0..CH {
            if z_buf[z_off + c] < 0.0 {
                z_buf[z_off + c] *= A2_LEAKY_SLOPE;
            }
        }

        // Head accumulate
        let head_off = (head_col + f) * CH;
        if is_first {
            head_accum[head_off..head_off + CH].copy_from_slice(&z_buf[z_off..z_off + CH]);
        } else {
            for c in 0..CH {
                head_accum[head_off + c] += z_buf[z_off + c];
            }
        }

        // L1x1 residual
        if !is_last {
            let lin_off = f * CH;
            for c in 0..CH {
                let mut sum = l1x1_b[c];
                for u in 0..CH {
                    sum += l1x1_w[u * CH + c] * z_buf[z_off + u];
                }
                layer_in[lin_off + c] += sum;
            }
        }
    }
}
