// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(clippy::too_many_arguments)]

//! Scalar reference implementation for the A2 dilated causal Conv1D.
//!
//! This is the oracle used in parity tests: every SIMD kernel must produce
//! the same numerical result (within floating-point tolerance) as this
//! unoptimized scalar computation.

/// Scalar reference for a single-frame A2 dilated causal Conv1D.
///
/// Replicates the exact math of `Conv1dDyn::process_single_frame`:
/// 1. Initialize accumulators with bias (and optional mixin).
/// 2. For each output channel block of 4:
///    a. For each kernel tap k (dilated lookback):
///       - Compute tap position: `frame_idx + dilation * (k + 1 - K)`
///       - Load `in_ch` input values from that position.
///       - Compute 4 dot products (one per output lane) with interleaved weights.
/// 3. Store results.
///
/// Weights are in `[OUT/4][KERNEL][IN][4]` interleaved layout (same as `Conv1dDyn`).
///
/// # Parameters
/// * `weights` — interleaved f32 weights in `[OUT/4][KERNEL][IN][4]` format.
/// * `bias` — f32 bias vector of length `out_ch`.
/// * `do_bias` — whether to add bias to the accumulator.
/// * `dilation` — temporal dilation factor.
/// * `in_ch` — number of input channels.
/// * `out_ch` — number of output channels.
/// * `kernel` — kernel size (6 or 15 for A2).
/// * `layer_buffer` — input buffer (f32 mirror buffer slice).
/// * `frame_idx` — current frame index within `layer_buffer`.
/// * `mixin` — optional conditioning vector of length `out_ch`.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 dilated convolution fallback kernel requiring many shape/stride parameters when SIMD path is unavailable"
)]
pub fn a2_conv1d_single_frame_fallback(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    mixin: Option<&[f32]>,
    out_frame: &mut [f32],
) {
    debug_assert!(out_frame.len() >= out_ch);

    let num_blocks = out_ch.div_ceil(4);

    for b in 0..num_blocks {
        let out_c = b * 4;

        let (mv0, mv1, mv2, mv3) = if let Some(m) = mixin {
            let m0 = if out_c < m.len() { m[out_c] } else { 0.0 };
            let m1 = if out_c + 1 < m.len() {
                m[out_c + 1]
            } else {
                0.0
            };
            let m2 = if out_c + 2 < m.len() {
                m[out_c + 2]
            } else {
                0.0
            };
            let m3 = if out_c + 3 < m.len() {
                m[out_c + 3]
            } else {
                0.0
            };
            (m0, m1, m2, m3)
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        let mut r0 = if do_bias { bias[out_c] } else { 0.0 } + mv0;
        let mut r1 = if do_bias && out_c + 1 < out_ch {
            bias[out_c + 1]
        } else {
            0.0
        } + mv1;
        let mut r2 = if do_bias && out_c + 2 < out_ch {
            bias[out_c + 2]
        } else {
            0.0
        } + mv2;
        let mut r3 = if do_bias && out_c + 3 < out_ch {
            bias[out_c + 3]
        } else {
            0.0
        } + mv3;

        for k in 0..kernel {
            let offset = (dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            let in_slice_start = ((frame_idx as isize) + offset) as usize * in_ch;

            let w_start = (b * kernel + k) * in_ch * 4;

            for in_c in 0..in_ch {
                let w_idx = w_start + in_c * 4;
                let input_val = layer_buffer[in_slice_start + in_c];

                let w0 = weights[w_idx];
                let w1 = weights[w_idx + 1];
                let w2 = weights[w_idx + 2];
                let w3 = weights[w_idx + 3];

                r0 += w0 * input_val;
                r1 += w1 * input_val;
                r2 += w2 * input_val;
                r3 += w3 * input_val;
            }
        }

        if out_c + 3 < out_ch {
            out_frame[out_c] = r0;
            out_frame[out_c + 1] = r1;
            out_frame[out_c + 2] = r2;
            out_frame[out_c + 3] = r3;
        } else {
            if out_c < out_ch {
                out_frame[out_c] = r0;
            }
            if out_c + 1 < out_ch {
                out_frame[out_c + 1] = r1;
            }
            if out_c + 2 < out_ch {
                out_frame[out_c + 2] = r2;
            }
            if out_c + 3 < out_ch {
                out_frame[out_c + 3] = r3;
            }
        }
    }
}

/// Scalar reference for a block of `num_frames` consecutive frames.
///
/// Iterates the single-frame fallback for each consecutive frame index.
#[expect(
    clippy::too_many_arguments,
    reason = "A2 dilated convolution fallback kernel requiring many shape/stride parameters when SIMD path is unavailable"
)]
pub fn a2_conv1d_block_fallback(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    layer_buffer: &[f32],
    buffer_start: usize,
    num_frames: usize,
    mixin: Option<&[f32]>,
    block: &mut [f32],
) {
    debug_assert!(block.len() >= num_frames * out_ch);

    for i in 0..num_frames {
        let out_slice = &mut block[i * out_ch..(i + 1) * out_ch];
        let m = mixin.map(|full| &full[i * out_ch..(i + 1) * out_ch]);
        a2_conv1d_single_frame_fallback(
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            layer_buffer,
            buffer_start + i,
            m,
            out_slice,
        );
    }
}
#[cfg(test)]
#[path = "conv1d_fallback_test.rs"]
mod tests;
