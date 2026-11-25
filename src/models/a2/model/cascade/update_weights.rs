// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Atomic coefficient updates across all arrays of a loaded A2 cascade.

use super::WaveNetA2Cascade;
use crate::models::a2::model::weight_updates::{A2WeightUpdateError, validate_stream};

impl WaveNetA2Cascade {
    /// Updates all loaded arrays and the cascade scale without allocation/reset.
    ///
    /// `weights` concatenates each array's [`WaveNetA2Dyn::set_weights`](
    /// crate::models::a2::WaveNetA2Dyn::set_weights) stream, in array order.
    /// The model-level `head_scale` is supplied separately: serialized files may
    /// include that optional trailing scalar, while array streams do not.
    /// This explicit boundary avoids guessing from the supplied stream length.
    /// Mono array streams still contain their own head bias/scale, as required
    /// by that array's loader. Multichannel array output scales are preserved.
    ///
    /// Each array validates its destinations during weight loading. The cached
    /// results and full stream are checked before any writes. Errors in a later
    /// array leave all earlier arrays and the cascade
    /// scale unchanged. Successful updates preserve all causal/condition-DSP
    /// state, ring cursors, buffer addresses and negotiated block sizes.
    ///
    /// # Real-time safety
    /// No heap allocation/deallocation, locks, I/O or logging, including errors.
    /// Exclusive access is required. Work scales with the loaded weight count;
    /// call at block boundaries with prepared caller-owned storage.
    ///
    /// # Errors
    /// Returns [`A2WeightUpdateError`] for unloaded/unsupported array layouts,
    /// incorrect stream length, non-finite weights or a non-finite global scale.
    pub fn update_weights(
        &mut self,
        weights: &[f32],
        head_scale: f32,
    ) -> Result<(), A2WeightUpdateError> {
        let expected = self.runtime_weight_count()?;
        validate_stream(weights, expected)?;
        if !head_scale.is_finite() {
            return Err(A2WeightUpdateError::NonFiniteHeadScale);
        }
        let mut pos = 0;
        for array in &mut self.arrays {
            array.write_runtime_weights(weights, &mut pos);
        }
        self.head_scale = head_scale;
        Ok(())
    }

    /// Returns the exact concatenated array-stream size, excluding global scale.
    ///
    /// Sums the counts validated during loading, without inspecting layers.
    /// The array list is public, so this uses the current arrays on each call.
    /// This excludes nested condition-DSP weights, which are not replaced.
    ///
    /// # Errors
    /// Returns [`A2WeightUpdateError`] for unloaded/unsupported array layouts or
    /// if the combined serialized count overflows `usize`.
    pub fn runtime_weight_count(&self) -> Result<usize, A2WeightUpdateError> {
        if self.arrays.is_empty() {
            return Err(A2WeightUpdateError::NotLoaded);
        }
        self.arrays.iter().try_fold(0usize, |count, array| {
            count
                .checked_add(array.runtime_weight_count()?)
                .ok_or(A2WeightUpdateError::UnsupportedTopology)
        })
    }
}

#[cfg(test)]
#[path = "update_weights_test.rs"]
mod tests;
