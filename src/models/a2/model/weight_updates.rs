// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared preflight and in-place coefficient packing for A2 runtime updates.

use crate::models::a2::conv1d::A2Conv1d;
use crate::models::a2::conv1d_ch::A2Conv1dCh;
use crate::models::a2::layer::{A2ConvCh, A2Layer};
use crate::models::a2::weights_layout::{transpose_conv1d_interleaved_4wide, transpose_dense_f32};

/// Nonallocating errors from [`super::WaveNetA2::update_weights`] and [`super::dynamic::WaveNetA2Dyn::update_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum A2WeightUpdateError {
    /// Construction/loading must finish before runtime updates.
    #[error("A2 weights have not been loaded")]
    NotLoaded,
    /// The loaded topology is not supported by the update API.
    #[error("Unsupported A2 runtime weight-update topology")]
    UnsupportedTopology,
    /// The stream must have exactly the prepared serialized weight count.
    #[error("A2 runtime update needs {expected} weights, got {actual}")]
    WeightCountMismatch {
        /// Required number of serialized f32 coefficients.
        expected: usize,
        /// Supplied number of serialized f32 coefficients.
        actual: usize,
    },
    /// Every coefficient, including head scale, must be finite.
    #[error("A2 runtime update contains a non-finite weight at index {index}")]
    NonFiniteWeight {
        /// Position in the NAM serialized weight stream.
        index: usize,
    },
}

#[derive(Clone, Copy)]
pub(super) struct LayerShape {
    pub channels: usize,
    pub bottleneck: usize,
    pub conv_out: usize,
    pub condition: usize,
    pub kernel: usize,
    pub dilation: usize,
}

/// Validates every destination and computes the serialized count with checked
/// arithmetic. A None result must be handled before any coefficient is changed.
pub(super) fn layer_weight_count(layer: &A2Layer, shape: LayerShape) -> Option<usize> {
    let LayerShape {
        channels,
        bottleneck,
        conv_out,
        condition,
        kernel,
        dilation,
    } = shape;
    if channels == 0 || bottleneck == 0 || conv_out == 0 || condition == 0 || kernel == 0 {
        return None;
    }
    let A2Conv1d::Standard(conv) = &layer.conv else {
        return None;
    };
    let raw_conv = conv_out.checked_mul(channels)?.checked_mul(kernel)?;
    let padded = conv_out
        .div_ceil(4)
        .checked_mul(4)?
        .checked_mul(channels)?
        .checked_mul(kernel)?;
    let mixin = conv_out.checked_mul(condition)?;
    let residual = channels.checked_mul(bottleneck)?;
    let specialized = match &layer.conv_ch {
        None => true,
        Some(A2ConvCh::Ch3(c)) if channels == 3 && conv_out == 3 => ch_valid(c, kernel, dilation),
        Some(A2ConvCh::Ch8(c)) if channels == 8 && conv_out == 8 => ch_valid(c, kernel, dilation),
        _ => false,
    };
    let valid = conv.in_ch == channels
        && conv.out_ch == conv_out
        && conv.kernel == kernel
        && conv.dilation == dilation
        && conv.do_bias
        && conv.interleave_width == 4
        && conv.num_blocks == conv_out.div_ceil(4)
        && conv.weights.len() == padded
        && conv.bias.len() == conv_out
        && specialized
        && layer.mixin_w.len() == mixin
        && layer.mixin_groups == 1
        && layer.l1x1_w.len() == residual
        && layer.l1x1_b.len() == channels
        && layer.l1x1_groups == 1
        && !layer.head1x1_active
        && layer.head1x1_w.is_empty()
        && layer.head1x1_b.is_empty()
        && layer.conv_pre_film.is_none()
        && layer.conv_post_film.is_none()
        && layer.input_mixin_pre_film.is_none()
        && layer.input_mixin_post_film.is_none()
        && layer.activation_pre_film.is_none()
        && layer.activation_post_film.is_none()
        && layer.layer1x1_post_film.is_none()
        && layer.head1x1_post_film.is_none();
    if !valid {
        return None;
    }
    raw_conv
        .checked_add(conv_out)?
        .checked_add(mixin)?
        .checked_add(residual)?
        .checked_add(channels)
}

/// Only called after all layers, stream length and finiteness passed preflight.
pub(super) fn write_layer(
    layer: &mut A2Layer,
    shape: LayerShape,
    weights: &[f32],
    pos: &mut usize,
) {
    let raw = take(weights, pos, shape.channels * shape.conv_out * shape.kernel);
    let bias = take(weights, pos, shape.conv_out);
    if let A2Conv1d::Standard(conv) = &mut layer.conv {
        conv.weights.fill(0.0);
        transpose_conv1d_interleaved_4wide(
            raw,
            &mut conv.weights,
            shape.channels,
            shape.conv_out,
            shape.kernel,
        );
        conv.bias.copy_from_slice(bias);
    }
    match &mut layer.conv_ch {
        Some(A2ConvCh::Ch3(conv)) => update_ch(conv, raw, bias),
        Some(A2ConvCh::Ch8(conv)) => update_ch(conv, raw, bias),
        None => {}
    }
    transpose_dense_f32(
        take(weights, pos, shape.conv_out * shape.condition),
        &mut layer.mixin_w,
        shape.condition,
        shape.conv_out,
    );
    transpose_dense_f32(
        take(weights, pos, shape.channels * shape.bottleneck),
        &mut layer.l1x1_w,
        shape.bottleneck,
        shape.channels,
    );
    layer
        .l1x1_b
        .copy_from_slice(take(weights, pos, shape.channels));
}

pub(super) fn validate_stream(weights: &[f32], expected: usize) -> Result<(), A2WeightUpdateError> {
    if weights.len() != expected {
        return Err(A2WeightUpdateError::WeightCountMismatch {
            expected,
            actual: weights.len(),
        });
    }
    if let Some(index) = weights.iter().position(|v| !v.is_finite()) {
        return Err(A2WeightUpdateError::NonFiniteWeight { index });
    }
    Ok(())
}

fn ch_valid<const CH: usize>(conv: &A2Conv1dCh<CH>, kernel: usize, dilation: usize) -> bool {
    let pad = CH.div_ceil(4) * 4;
    conv.kernel == kernel
        && conv.dilation == dilation
        && conv.weights.len() == kernel * pad * pad
        && conv.bias.len() == pad
}

fn update_ch<const CH: usize>(conv: &mut A2Conv1dCh<CH>, raw: &[f32], bias: &[f32]) {
    let pad = CH.div_ceil(4) * 4;
    conv.weights.fill(0.0);
    for out in 0..CH {
        for input in 0..CH {
            for tap in 0..conv.kernel {
                conv.weights[tap * pad * pad + input * pad + out] =
                    raw[(out * CH + input) * conv.kernel + tap];
            }
        }
    }
    conv.bias.fill(0.0);
    conv.bias[..CH].copy_from_slice(bias);
}

pub(super) fn take<'a>(weights: &'a [f32], pos: &mut usize, count: usize) -> &'a [f32] {
    let start = *pos;
    *pos += count;
    &weights[start..*pos]
}
