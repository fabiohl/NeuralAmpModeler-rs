// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::super::WeightCursor;
use super::super::checked_arith;
use super::traits::{ConvWeightsOutput, DenseWeightsOutput};
use crate::math::common::AlignedVec;

// =============================================================================
// Unified weight reading for static WaveNet models
// =============================================================================

pub(crate) fn read_conv1d_weights_typed<T: ConvWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    k_size: usize,
    dilation: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let width = select_interleave_width(out_size);
    let num_blocks = out_size.div_ceil(width);
    let padded_total =
        checked_arith::checked_conv_padded_total(num_blocks, width, in_size, k_size)?;
    let interleaved = cursor.is_interleaved4();

    let f32_weights = if interleaved {
        let total_4wide = checked_arith::checked_mul4(out_size.div_ceil(4), 4, in_size, k_size)?;
        let raw = cursor.read_slice(total_4wide)?;
        let mut weights = AlignedVec::new(padded_total, 0.0f32)?;
        if width != 4 {
            let mut tmp = AlignedVec::new(total_4wide, 0.0f32)?;
            tmp.copy_from_slice(raw);
            match width {
                16 => transpose_4wide_to_16wide(&tmp, &mut weights, in_size, out_size, k_size),
                8 => transpose_4wide_to_8wide(&tmp, &mut weights, in_size, out_size, k_size),
                _ => transpose_conv1d_interleaved_4wide(
                    &tmp,
                    &mut weights,
                    in_size,
                    out_size,
                    k_size,
                ),
            }
        } else {
            weights.copy_from_slice(raw);
        }
        weights
    } else {
        let total = checked_arith::checked_conv_total(out_size, in_size, k_size)?;
        let raw = cursor.read_slice(total)?;
        let mut weights = AlignedVec::new(padded_total, 0.0f32)?;
        match width {
            16 => transpose_conv1d_interleaved_16wide(raw, &mut weights, in_size, out_size, k_size),
            8 => transpose_conv1d_interleaved_8wide(raw, &mut weights, in_size, out_size, k_size),
            _ => transpose_conv1d_interleaved_4wide(raw, &mut weights, in_size, out_size, k_size),
        }
        weights
    };

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())?
    } else {
        AlignedVec::new(out_size, 0.0)?
    };

    Ok(T::from_parts(
        f32_weights,
        bias,
        do_bias,
        dilation,
        in_size,
        out_size,
        k_size,
    ))
}

pub(crate) fn read_dense_weights_typed<T: DenseWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let total = checked_arith::checked_dense_total(out_size, in_size)?;
    let raw = cursor.read_slice(total)?;
    let mut f32_weights = AlignedVec::new(total, 0.0f32)?;
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        f32_weights.copy_from_slice(raw);
    } else {
        transpose_dense_layer_f32(raw, &mut f32_weights, in_size, out_size);
    }

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())?
    } else {
        AlignedVec::new(out_size, 0.0)?
    };

    Ok(T::from_parts(f32_weights, bias, do_bias, in_size, out_size))
}

pub(crate) fn read_dense_head_weights_typed<T: DenseWeightsOutput>(
    cursor: &mut WeightCursor<'_>,
    in_size: usize,
    out_size: usize,
    do_bias: bool,
) -> anyhow::Result<T> {
    let total = checked_arith::checked_dense_total(out_size, in_size)?;
    let raw = cursor.read_slice(total)?;
    let mut f32_weights = AlignedVec::new(total, 0.0f32)?;
    let interleaved = cursor.is_interleaved4();

    if interleaved {
        f32_weights.copy_from_slice(raw);
    } else {
        transpose_dense_layer_f32(raw, &mut f32_weights, in_size, out_size);
    }

    let bias = if do_bias {
        AlignedVec::from_vec(cursor.read_slice(out_size)?.to_vec())?
    } else {
        AlignedVec::new(out_size, 0.0)?
    };

    Ok(T::from_parts(f32_weights, bias, do_bias, in_size, out_size))
}

// =============================================================================
// Transposition (full f32 precision — no quantization)
// =============================================================================

/// Rearranges convolution layer weights into the "Interleaved 4-Wide" format
/// preserving full f32 precision.
/// This technique groups data in blocks of 4, allowing the processor to execute
/// calculations in "batch" (SIMD), processing 4 audio channels at once.
pub fn transpose_conv1d_interleaved_4wide(
    raw: &[f32],
    weights: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    debug_assert!(
        raw.len() >= out_ch * in_ch * kernel,
        "raw slice too short for transposition"
    );
    let num_blocks = out_ch.div_ceil(4);
    let padded_total = num_blocks * 4 * in_ch * kernel;
    debug_assert!(
        weights.len() >= padded_total,
        "weights slice too short for interleaved output"
    );
    // Target layout: [block][kernel][in_ch][lane], 4 lanes per block. Each block
    // holds 4 output channels, zero-padded past `out_ch` so an SIMD dot over the
    // lane dimension never strides beyond the block boundary.
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..4 {
                    let out_c = b * 4 + lane;
                    let target_idx = b * (kernel * in_ch * 4) + k * (in_ch * 4) + in_c * 4 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw[raw_idx];
                    } else {
                        weights[target_idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Rearranges convolution layer weights into the "Interleaved 8-Wide" format
/// preserving full f32 precision.
/// Analogue to `transpose_conv1d_interleaved_4wide` but groups 8 output channels per block.
pub fn transpose_conv1d_interleaved_8wide(
    raw: &[f32],
    weights: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    debug_assert!(
        raw.len() >= out_ch * in_ch * kernel,
        "raw slice too short for transposition"
    );
    let num_blocks = out_ch.div_ceil(8);
    let padded_total = num_blocks * 8 * in_ch * kernel;
    debug_assert!(
        weights.len() >= padded_total,
        "weights slice too short for interleaved output"
    );
    // Target layout: [block][kernel][in_ch][lane], 8 lanes per block (analogue to
    // the 4-wide variant, grouping 8 output channels per block for wider SIMD).
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..8 {
                    let out_c = b * 8 + lane;
                    let target_idx = b * (kernel * in_ch * 8) + k * (in_ch * 8) + in_c * 8 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw[raw_idx];
                    } else {
                        weights[target_idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Rearranges convolution layer weights into the "Interleaved 16-Wide" format
/// preserving full f32 precision.
/// Groups 16 output channels per block for AVX-512 dot_product_16x_f32.
pub fn transpose_conv1d_interleaved_16wide(
    raw: &[f32],
    weights: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    debug_assert!(
        raw.len() >= out_ch * in_ch * kernel,
        "raw slice too short for transposition"
    );
    let num_blocks = out_ch.div_ceil(16);
    let padded_total = num_blocks * 16 * in_ch * kernel;
    debug_assert!(
        weights.len() >= padded_total,
        "weights slice too short for interleaved output"
    );
    // Target layout: [block][kernel][in_ch][lane], 16 lanes per block, feeding
    // `dot_product_16x_f32` (AVX-512). Lanes ≥ out_ch are zero-padded.
    for b in 0..num_blocks {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                for lane in 0..16 {
                    let out_c = b * 16 + lane;
                    let target_idx =
                        b * (kernel * in_ch * 16) + k * (in_ch * 16) + in_c * 16 + lane;
                    if out_c < out_ch {
                        let raw_idx = (out_c * in_ch + in_c) * kernel + k;
                        weights[target_idx] = raw[raw_idx];
                    } else {
                        weights[target_idx] = 0.0;
                    }
                }
            }
        }
    }
}

/// Converts 4-wide interleaved weights to 8-wide interleaved format.
/// Used when .nam file stores weights in 4-wide layout but the model benefits from 8-wide.
pub(crate) fn transpose_4wide_to_8wide(
    src: &[f32],
    dst: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    let num_blocks_4 = out_ch.div_ceil(4);
    // Merge pairs of consecutive 4-wide blocks into one 8-wide block: lanes 0..4
    // come from src_b = 2*b and lanes 4..8 from src_b = 2*b+1, zero-filling when
    // the source block index exceeds `num_blocks_4`.
    for b in 0..out_ch.div_ceil(8) {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                let dst_base = b * kernel * in_ch * 8 + k * in_ch * 8 + in_c * 8;
                for lane in 0..4 {
                    let src_b = 2 * b;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + lane] = src[src_idx];
                    } else {
                        dst[dst_base + lane] = 0.0;
                    }
                }
                for lane in 0..4 {
                    let src_b = 2 * b + 1;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + 4 + lane] = src[src_idx];
                    } else {
                        dst[dst_base + 4 + lane] = 0.0;
                    }
                }
            }
        }
    }
}

/// Converts 4-wide interleaved weights to 16-wide interleaved format.
/// Used when .nam file stores weights in 4-wide layout but the model benefits from 16-wide.
pub(crate) fn transpose_4wide_to_16wide(
    src: &[f32],
    dst: &mut [f32],
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
) {
    let num_blocks_4 = out_ch.div_ceil(4);
    // Merge groups of four consecutive 4-wide blocks into one 16-wide block: lane
    // groups 0..4, 4..8, 8..12, 12..16 come from src_b = 4*b, 4*b+1, 4*b+2, 4*b+3,
    // zero-filling any source block beyond `num_blocks_4`.
    for b in 0..out_ch.div_ceil(16) {
        for k in 0..kernel {
            for in_c in 0..in_ch {
                let dst_base = b * kernel * in_ch * 16 + k * in_ch * 16 + in_c * 16;
                for lane in 0..4 {
                    let src_b = 4 * b;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + lane] = src[src_idx];
                    } else {
                        dst[dst_base + lane] = 0.0;
                    }
                }
                for lane in 0..4 {
                    let src_b = 4 * b + 1;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + 4 + lane] = src[src_idx];
                    } else {
                        dst[dst_base + 4 + lane] = 0.0;
                    }
                }
                for lane in 0..4 {
                    let src_b = 4 * b + 2;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + 8 + lane] = src[src_idx];
                    } else {
                        dst[dst_base + 8 + lane] = 0.0;
                    }
                }
                for lane in 0..4 {
                    let src_b = 4 * b + 3;
                    if src_b < num_blocks_4 {
                        let src_idx = src_b * kernel * in_ch * 4 + k * in_ch * 4 + in_c * 4 + lane;
                        dst[dst_base + 12 + lane] = src[src_idx];
                    } else {
                        dst[dst_base + 12 + lane] = 0.0;
                    }
                }
            }
        }
    }
}

/// Selects the optimal interleaving width based on output channel count.
/// Must be used consistently by both the weight loader (to store weights
/// in the correct layout) and the convolution processors (to read them back).
#[inline]
pub fn select_interleave_width(out_ch: usize) -> usize {
    if out_ch.is_multiple_of(16) || out_ch == 12 {
        16
    } else if out_ch.is_multiple_of(8) {
        8
    } else {
        4
    }
}

/// Rearranges dense layer weights into transposed format, keeping full f32 precision.
/// Stores result in column-major layout: `f32_weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c]`.
fn transpose_dense_layer_f32(raw: &[f32], weights: &mut [f32], in_size: usize, out_size: usize) {
    debug_assert!(
        raw.len() >= out_size * in_size,
        "raw slice too short for dense transposition"
    );
    debug_assert!(
        weights.len() >= in_size * out_size,
        "weights slice too short for dense output"
    );
    for out_c in 0..out_size {
        for in_c in 0..in_size {
            weights[in_c * out_size + out_c] = raw[out_c * in_size + in_c];
        }
    }
}

#[cfg(test)]
#[path = "layout_test.rs"]
mod layout_test;
