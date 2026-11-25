// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Validated, allocation-free host-precomputed FiLM modulation.

use super::FiLMLayer;

/// Nonallocating errors from FiLM modulation updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FiLMModulationError {
    /// Every layer must already have a loaded activation-pre FiLM projection.
    #[error("Activation-pre FiLM is not prepared for every layer")]
    NotPrepared,
    /// Public FiLM geometry no longer matches its prepared destination storage.
    #[error("FiLM modulation geometry does not match its prepared storage")]
    InvalidTopology,
    /// Both streams must have exactly one value per prepared feature channel.
    #[error("FiLM needs {expected} scales/shifts, got {scales} scales and {shifts} shifts")]
    SizeMismatch {
        /// Prepared modulation feature count.
        expected: usize,
        /// Supplied scale count.
        scales: usize,
        /// Supplied shift count.
        shifts: usize,
    },
    /// Scale coefficients must be finite.
    #[error("FiLM scale at index {index} is not finite")]
    NonFiniteScale {
        /// Index in the supplied scale stream.
        index: usize,
    },
    /// Shift coefficients must be finite.
    #[error("FiLM shift at index {index} is not finite")]
    NonFiniteShift {
        /// Index in the supplied shift stream.
        index: usize,
    },
}

impl FiLMLayer {
    /// Installs final scale/shift vectors without evaluating a control network.
    ///
    /// Subsequent [`Self::process`] calls apply `input * scale + shift` using
    /// these values and skip the condition projection. These are final scales,
    /// not scale deltas; a neutral update uses ones and zeros. Projection
    /// weights/biases remain intact. The override persists until replaced or
    /// [`Self::use_conditioned_modulation`] is called, including across cloning.
    /// The projection's `shift` flag does not restrict precomputed shifts.
    ///
    /// No allocation/deallocation, locks, I/O, logging or state reset occurs.
    /// Call through the same exclusive reference used for processing, at a
    /// block boundary. Control encoding, evaluation and smoothing belong to the
    /// caller. The existing unsafe processing preconditions remain in force.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] for destination mismatch, wrong lengths
    /// or non-finite coefficients. Rejection preserves the previous mode and
    /// coefficients; all validation precedes all writes.
    pub fn set_precomputed_modulation(
        &mut self,
        scales: &[f32],
        shifts: &[f32],
    ) -> Result<(), FiLMModulationError> {
        let expected = self.precomputed_channels()?;
        validate_modulation(scales, shifts, expected)?;
        self.write_precomputed(scales, shifts);
        Ok(())
    }

    /// Resumes the loaded condition projection on the next processing call.
    ///
    /// Does not allocate, reset streaming state, or modify projection weights.
    pub fn use_conditioned_modulation(&mut self) {
        if self.precomputed_modulation {
            // Grouped projections can leave trailing rows unwritten. Restore
            // the zeroed scratch state that a freshly loaded projection has.
            self.scale_shift_buf.fill(0.0);
            self.precomputed_modulation = false;
        }
    }

    /// Whether processing currently uses host-precomputed coefficients.
    pub fn has_precomputed_modulation(&self) -> bool {
        self.precomputed_modulation
    }

    pub(crate) fn precomputed_channels(&self) -> Result<usize, FiLMModulationError> {
        if self.channels == 0 || self.channels.checked_mul(2) != Some(self.scale_shift_buf.len()) {
            return Err(FiLMModulationError::InvalidTopology);
        }
        Ok(self.channels)
    }

    // Only called after all destination and coefficient checks succeeded.
    pub(crate) fn write_precomputed(&mut self, scales: &[f32], shifts: &[f32]) {
        self.scale_shift_buf[..self.channels].copy_from_slice(scales);
        self.scale_shift_buf[self.channels..].copy_from_slice(shifts);
        self.precomputed_modulation = true;
    }
}

pub(crate) fn validate_modulation(
    scales: &[f32],
    shifts: &[f32],
    expected: usize,
) -> Result<(), FiLMModulationError> {
    if scales.len() != expected || shifts.len() != expected {
        return Err(FiLMModulationError::SizeMismatch {
            expected,
            scales: scales.len(),
            shifts: shifts.len(),
        });
    }
    if let Some(index) = scales.iter().position(|v| !v.is_finite()) {
        return Err(FiLMModulationError::NonFiniteScale { index });
    }
    if let Some(index) = shifts.iter().position(|v| !v.is_finite()) {
        return Err(FiLMModulationError::NonFiniteShift { index });
    }
    Ok(())
}

#[cfg(test)]
#[path = "modulation_test.rs"]
mod tests;
