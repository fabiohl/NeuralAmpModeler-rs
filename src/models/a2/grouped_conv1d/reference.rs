// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Scalar reference implementations for grouped dilated causal Conv1D.
//!
//! These functions serve as precision oracles for parity testing against the
//! SIMD kernels.

/// Scalar reference for a single-frame grouped dilated conv with 64-bit accumulation.
///
/// Uses the same grouped-interleaved-4-wide weight layout as `A2GroupedConv1d`.
/// Each weight read uses f64 multiplication + accumulation for maximal precision.
pub fn grouped_conv1d_single_frame_ref(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    groups: usize,
    layer_buffer: &[f32],
    frame_idx: usize,
    out_frame: &mut [f32],
) {
    let in_per_group = in_ch / groups;
    let out_per_group = out_ch / groups;
    let num_blocks_per_group = out_per_group.div_ceil(4);

    for out_c in 0..out_ch {
        out_frame[out_c] = if do_bias { bias[out_c] } else { 0.0 };
    }

    for g in 0..groups {
        let group_in_start = g * in_per_group;
        let group_out_start = g * out_per_group;

        for b in 0..num_blocks_per_group {
            for k in 0..kernel {
                let taps_back = kernel - 1 - k;
                let col = frame_idx.wrapping_sub(dilation * taps_back);
                let src_base = col * in_ch;

                let wk_group_base = g * (num_blocks_per_group * kernel * in_per_group * 4);

                for ic in 0..in_per_group {
                    let hv = layer_buffer[src_base + group_in_start + ic];
                    let w_base = wk_group_base
                        + b * (kernel * in_per_group * 4)
                        + k * (in_per_group * 4)
                        + ic * 4;

                    let w0 = weights[w_base] as f64;
                    let w1 = weights[w_base + 1] as f64;
                    let w2 = weights[w_base + 2] as f64;
                    let w3 = weights[w_base + 3] as f64;
                    let hv64 = hv as f64;

                    {
                        let out0 = group_out_start + b * 4;
                        if out0 < out_ch {
                            out_frame[out0] = (out_frame[out0] as f64 + w0 * hv64) as f32;
                        }
                        if out0 + 1 < out_ch {
                            out_frame[out0 + 1] = (out_frame[out0 + 1] as f64 + w1 * hv64) as f32;
                        }
                        if out0 + 2 < out_ch {
                            out_frame[out0 + 2] = (out_frame[out0 + 2] as f64 + w2 * hv64) as f32;
                        }
                        if out0 + 3 < out_ch {
                            out_frame[out0 + 3] = (out_frame[out0 + 3] as f64 + w3 * hv64) as f32;
                        }
                    }
                }
            }
        }
    }
}

/// Scalar reference for a block of consecutive frames.
pub fn grouped_conv1d_block_ref(
    weights: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    groups: usize,
    layer_buffer: &[f32],
    frame_start: usize,
    num_frames: usize,
    block: &mut [f32],
) {
    for f in 0..num_frames {
        grouped_conv1d_single_frame_ref(
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            kernel,
            groups,
            layer_buffer,
            frame_start + f,
            &mut block[f * out_ch..(f + 1) * out_ch],
        );
    }
}
