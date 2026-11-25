// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! In-place runtime updates for the canonical static A2 weight layout.

use super::WaveNetA2;
use crate::models::a2::model::weight_updates::{
    A2WeightUpdateError, LayerShape, layer_weight_count, take, validate_stream, write_layer,
};
use crate::models::a2::weights_layout::transpose_head_w;
use crate::models::a2::{
    A2_DILATIONS, A2_HEAD_KERNEL_SIZE, A2_KERNEL_SIZES, A2_NUM_LAYERS, a2_weight_count,
};

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
    /// stream length, or NaN/Inf coefficients. The loader validates the layout
    /// off-thread. Runtime stream checks finish before any coefficient is written.
    pub fn update_weights(&mut self, weights: &[f32]) -> Result<(), A2WeightUpdateError> {
        if CH != 3 && CH != 8 {
            return Err(A2WeightUpdateError::UnsupportedTopology);
        }
        validate_stream(weights, self.runtime_weight_layout?)?;

        let mut pos = 0;
        self.rechannel_w_f32
            .copy_from_slice(take(weights, &mut pos, CH));
        for (i, layer) in self.layers.iter_mut().enumerate() {
            write_layer(layer, static_shape::<CH>(i), weights, &mut pos);
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

    pub(in crate::models::a2::model) fn prepare_runtime_weight_layout(&mut self) {
        self.runtime_weight_layout = if (CH == 3 || CH == 8) && self.runtime_update_topology_valid()
        {
            Ok(a2_weight_count::<CH>())
        } else {
            Err(A2WeightUpdateError::UnsupportedTopology)
        };
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
                .all(|(i, layer)| layer_weight_count(layer, static_shape::<CH>(i)).is_some())
    }
}

fn static_shape<const CH: usize>(i: usize) -> LayerShape {
    LayerShape {
        channels: CH,
        bottleneck: CH,
        conv_out: CH,
        condition: 1,
        kernel: A2_KERNEL_SIZES[i],
        dilation: A2_DILATIONS[i],
    }
}

#[cfg(test)]
#[path = "update_weights_test.rs"]
mod tests;
