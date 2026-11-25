// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! State-preserving weight updates for dense, mono-output dynamic A2 arrays.

use super::WaveNetA2Dyn;
use crate::models::a2::gating::GatingMode;
use crate::models::a2::model::weight_updates::{
    A2WeightUpdateError, LayerShape, layer_weight_count, take, validate_stream, write_layer,
};
use crate::models::a2::weights_layout::transpose_head_w;

impl WaveNetA2Dyn {
    /// Replaces loaded dynamic A2 coefficients without allocation or reset.
    ///
    /// Supports dense mono-output arrays with arbitrary loaded channel counts,
    /// bottlenecks, input/condition widths, kernels, dilations and gating modes.
    /// Grouped projections/convolutions, FiLM and head1x1 are not supported.
    /// The stream uses exactly [`Self::set_weights`] order, including head bias
    /// and scale. An embedded condition DSP is preserved and is not updated.
    ///
    /// All checks precede all writes. Success preserves streaming history,
    /// buffer addresses, cursors, prewarm state and negotiated block size.
    /// Calls require exclusive access and should occur at block boundaries.
    /// Work is proportional to the loaded weight count; this API does not smooth
    /// changes, run a hypernetwork or provide a parametric-format loader.
    ///
    /// # Real-time safety
    /// No allocation/deallocation, locks, I/O or logging, including errors.
    /// Prepare the model and caller-owned weight storage off the audio thread.
    ///
    /// # Errors
    /// Returns [`A2WeightUpdateError`] without changing the model for unloaded or
    /// unsupported topologies, incorrect stream length or non-finite weights.
    pub fn update_weights(&mut self, weights: &[f32]) -> Result<(), A2WeightUpdateError> {
        let expected = self.runtime_weight_count()?;
        validate_stream(weights, expected)?;
        let mut pos = 0;
        self.rechannel_w_f32.copy_from_slice(take(
            weights,
            &mut pos,
            self.input_channels * self.channels,
        ));
        for i in 0..self.num_layers {
            let shape = self.update_layer_shape(i);
            write_layer(&mut self.layers[i], shape, weights, &mut pos);
        }
        if let Some(head) = &mut self.head_conv {
            transpose_head_w(
                take(weights, &mut pos, head.head_w.len()),
                &mut head.head_w,
                self.head_accum_size,
                self.head_kernel_size,
            );
            head.head_b = weights[pos];
            head.head_scale = weights[pos + 1];
        }
        Ok(())
    }

    /// Returns the exact serialized count accepted by [`Self::update_weights`].
    ///
    /// Validates the loaded destinations without allocation. Hosts can use this
    /// off-thread to size replacement streams. This excludes nested condition-DSP
    /// weights, exactly as [`Self::set_weights`] does.
    ///
    /// # Errors
    /// Returns [`A2WeightUpdateError::NotLoaded`] or
    /// [`A2WeightUpdateError::UnsupportedTopology`].
    pub fn runtime_weight_count(&self) -> Result<usize, A2WeightUpdateError> {
        if self.layers.is_empty() {
            return Err(A2WeightUpdateError::NotLoaded);
        }
        self.checked_runtime_weight_count()
            .ok_or(A2WeightUpdateError::UnsupportedTopology)
    }

    fn checked_runtime_weight_count(&self) -> Option<usize> {
        if self.head_size != 1
            || self.head1x1_active
            || self.head_accum_size != self.bottleneck
            || self.groups_input != 1
            || self.mixin_groups != 1
            || self.l1x1_groups != 1
            || self.layers.len() != self.num_layers
            || self.kernel_sizes.len() != self.num_layers
            || self.dilations.len() != self.num_layers
            || self.gating_modes.len() != self.num_layers
            || self.input_channels == 0
            || self.channels == 0
            || self.head_kernel_size == 0
        {
            return None;
        }
        let mut total = self.input_channels.checked_mul(self.channels)?;
        if self.rechannel_w_f32.len() != total {
            return None;
        }
        for (i, layer) in self.layers.iter().enumerate() {
            total = total.checked_add(layer_weight_count(layer, self.update_layer_shape(i))?)?;
        }
        let head = self.head_conv.as_ref()?;
        let head_weights = self.head_accum_size.checked_mul(self.head_kernel_size)?;
        if head.num_channels != self.head_accum_size
            || head.kernel_size != self.head_kernel_size
            || head.head_w.len() != head_weights
        {
            return None;
        }
        total.checked_add(head_weights)?.checked_add(2)
    }

    fn update_layer_shape(&self, i: usize) -> LayerShape {
        let mult = if self.gating_modes[i] == GatingMode::None {
            1
        } else {
            2
        };
        LayerShape {
            channels: self.channels,
            bottleneck: self.bottleneck,
            conv_out: self.bottleneck.saturating_mul(mult),
            condition: self.condition_size,
            kernel: self.kernel_sizes[i],
            dilation: self.dilations[i],
        }
    }
}

#[cfg(test)]
#[path = "update_weights_test.rs"]
mod tests;
