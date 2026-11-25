// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! In-place runtime updates for the canonical static A2 weight layout.

use super::WaveNetA2;
use crate::models::a2::a2_weight_count;
use crate::models::a2::conv1d::A2Conv1d;
use crate::models::a2::conv1d_ch::A2Conv1dCh;
use crate::models::a2::layer::{A2ConvCh, A2Layer};
use crate::models::a2::params::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS,
};
use crate::models::a2::weights_layout::{
    transpose_conv1d_interleaved_4wide, transpose_dense_f32, transpose_head_w,
};

/// Nonallocating errors from [`WaveNetA2::update_weights`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum A2WeightUpdateError {
    /// Construction/loading must finish before runtime updates.
    #[error("A2 weights have not been loaded")]
    NotLoaded,
    /// Only canonical CH=3/8 static models without FiLM are supported.
    #[error("A2 runtime update requires the canonical CH=3/8 topology without FiLM")]
    UnsupportedTopology,
    /// The stream must have exactly the prepared canonical weight count.
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

impl<const CH: usize> WaveNetA2<CH> {
    /// Replaces canonical A2 coefficients in place, preserving streaming history.
    ///
    /// Call [`Self::set_weights`] off-thread first. This operation accepts the
    /// same NAM serialized order for CH=3/8 models **without FiLM** and refreshes
    /// both the generic interleaved and CH-specialized convolution layouts.
    /// It does not resize buffers, prewarm, reset histories/cursors, or change
    /// the negotiated block size. The next processing call uses the new weights
    /// against the existing history. The caller must serialize updates and
    /// processing (the exclusive mutable borrow enforces this in safe Rust).
    ///
    /// # Real-time safety
    /// No allocation, deallocation, locks, I/O or logging, including error paths.
    /// Work is bounded by the fixed topology and serialized weight count. Error
    /// formatting is left to the caller and should be done off the audio thread.
    ///
    /// # Errors
    /// Returns [`A2WeightUpdateError`] for unloaded/unsupported models, incorrect
    /// stream length, or NaN/Inf coefficients. All checks finish **before any
    /// coefficient is written**: rejection preserves both weights and history.
    pub fn update_weights(&mut self, weights: &[f32]) -> Result<(), A2WeightUpdateError> {
        if CH != 3 && CH != 8 {
            return Err(A2WeightUpdateError::UnsupportedTopology);
        }
        if self.layers.is_empty() || self.head_conv.is_none() {
            return Err(A2WeightUpdateError::NotLoaded);
        }
        if !self.runtime_update_topology_valid() {
            return Err(A2WeightUpdateError::UnsupportedTopology);
        }
        let expected = a2_weight_count::<CH>();
        if weights.len() != expected {
            return Err(A2WeightUpdateError::WeightCountMismatch {
                expected,
                actual: weights.len(),
            });
        }
        if let Some(index) = weights.iter().position(|value| !value.is_finite()) {
            return Err(A2WeightUpdateError::NonFiniteWeight { index });
        }

        let mut pos = 0;
        self.rechannel_w_f32
            .copy_from_slice(take(weights, &mut pos, CH));
        for (layer, kernel) in self.layers.iter_mut().zip(A2_KERNEL_SIZES) {
            let raw = take(weights, &mut pos, CH * CH * kernel);
            let bias = take(weights, &mut pos, CH);
            // Preflight proved the canonical variant and every destination size.
            if let A2Conv1d::Standard(conv) = &mut layer.conv {
                conv.weights.fill(0.0);
                transpose_conv1d_interleaved_4wide(raw, &mut conv.weights, CH, CH, kernel);
                conv.bias.copy_from_slice(bias);
            }
            match &mut layer.conv_ch {
                Some(A2ConvCh::Ch3(conv)) => update_ch(conv, raw, bias),
                Some(A2ConvCh::Ch8(conv)) => update_ch(conv, raw, bias),
                None => {}
            }
            layer.mixin_w.copy_from_slice(take(weights, &mut pos, CH));
            transpose_dense_f32(take(weights, &mut pos, CH * CH), &mut layer.l1x1_w, CH, CH);
            layer.l1x1_b.copy_from_slice(take(weights, &mut pos, CH));
        }
        if let Some(head) = &mut self.head_conv {
            transpose_head_w(
                take(weights, &mut pos, CH * A2_HEAD_KERNEL_SIZE),
                &mut head.head_w,
                CH,
                A2_HEAD_KERNEL_SIZE,
            );
            head.head_b = weights[pos];
            head.head_scale = weights[pos + 1];
        }
        Ok(())
    }

    fn runtime_update_topology_valid(&self) -> bool {
        self.layers.len() == A2_NUM_LAYERS
            && self.rechannel_w_f32.len() == CH
            && self.head_conv.as_ref().is_some_and(|h| {
                h.num_channels == CH
                    && h.kernel_size == A2_HEAD_KERNEL_SIZE
                    && h.head_w.len() == CH * A2_HEAD_KERNEL_SIZE
            })
            && self
                .layers
                .iter()
                .enumerate()
                .all(|(i, layer)| layer_valid::<CH>(layer, A2_KERNEL_SIZES[i], A2_DILATIONS[i]))
    }
}

fn layer_valid<const CH: usize>(layer: &A2Layer, kernel: usize, dilation: usize) -> bool {
    let A2Conv1d::Standard(conv) = &layer.conv else {
        return false;
    };
    let pad = CH.div_ceil(4) * 4;
    let specialized = match &layer.conv_ch {
        None => true,
        Some(A2ConvCh::Ch3(c)) if CH == 3 => ch_valid(c, kernel, dilation),
        Some(A2ConvCh::Ch8(c)) if CH == 8 => ch_valid(c, kernel, dilation),
        _ => false,
    };
    conv.in_ch == CH
        && conv.out_ch == CH
        && conv.kernel == kernel
        && conv.dilation == dilation
        && conv.do_bias
        && conv.interleave_width == 4
        && conv.num_blocks == CH.div_ceil(4)
        && conv.weights.len() == pad * CH * kernel
        && conv.bias.len() == CH
        && specialized
        && layer.mixin_w.len() == CH
        && layer.mixin_groups == 1
        && layer.l1x1_w.len() == CH * CH
        && layer.l1x1_b.len() == CH
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
        && layer.head1x1_post_film.is_none()
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

fn take<'a>(weights: &'a [f32], pos: &mut usize, count: usize) -> &'a [f32] {
    let start = *pos;
    *pos += count;
    &weights[start..*pos]
}

#[cfg(test)]
#[path = "update_weights_test.rs"]
mod tests;
