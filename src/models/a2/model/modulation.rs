// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Atomic block-level activation-pre FiLM updates across loaded A2 layers.

use super::{WaveNetA2, cascade::WaveNetA2Cascade, dynamic::WaveNetA2Dyn};
use crate::models::a2::film::modulation::validate_modulation;
use crate::models::a2::{A2Layer, FiLMModulationError};

fn layer_count(layers: &[A2Layer]) -> Result<usize, FiLMModulationError> {
    if layers.is_empty() {
        return Err(FiLMModulationError::NotPrepared);
    }
    layers.iter().try_fold(0usize, |count, layer| {
        let film = layer
            .activation_pre_film
            .as_ref()
            .ok_or(FiLMModulationError::NotPrepared)?;
        let channels = film.precomputed_channels()?;
        if channels != layer.conv.out_ch() {
            return Err(FiLMModulationError::InvalidTopology);
        }
        count
            .checked_add(channels)
            .ok_or(FiLMModulationError::InvalidTopology)
    })
}

fn write_layers(layers: &mut [A2Layer], scales: &[f32], shifts: &[f32], pos: &mut usize) {
    for layer in layers {
        if let Some(film) = &mut layer.activation_pre_film {
            let end = *pos + film.channels;
            film.write_precomputed(&scales[*pos..end], &shifts[*pos..end]);
            *pos = end;
        }
    }
}

fn resume_layers(layers: &mut [A2Layer]) {
    for layer in layers {
        if let Some(film) = &mut layer.activation_pre_film {
            film.use_conditioned_modulation();
        }
    }
}

impl<const CH: usize> WaveNetA2<CH> {
    /// Returns the flat feature count for prepared activation-pre FiLM updates.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] if any layer has no loaded activation-pre
    /// FiLM projection or its prepared dimensions are inconsistent.
    pub fn activation_modulation_size(&self) -> Result<usize, FiLMModulationError> {
        layer_count(&self.layers)
    }

    /// Sets block-constant activation-pre FiLM for all layers atomically.
    ///
    /// Scales and shifts each concatenate one vector per layer, in layer order.
    /// Every layer must already contain activation-pre FiLM loaded off-thread.
    /// Final scales use one for identity; shifts use zero. Projection weights,
    /// histories, cursors and prepared block size are preserved. The override
    /// persists across processing/reset until replaced or resumed. The caller
    /// owns control encoding, evaluation and smoothing. No allocation/free,
    /// locking, I/O or logging occurs, including rejection paths.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] before changing any layer if a destination
    /// is unprepared/inconsistent, lengths differ or a coefficient is non-finite.
    pub fn set_activation_modulation(
        &mut self,
        scales: &[f32],
        shifts: &[f32],
    ) -> Result<(), FiLMModulationError> {
        validate_modulation(scales, shifts, self.activation_modulation_size()?)?;
        let mut pos = 0;
        write_layers(&mut self.layers, scales, shifts, &mut pos);
        Ok(())
    }

    /// Resumes all loaded activation-pre condition projections without reset.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] without changing any layer if preparation
    /// or geometry is incomplete. This operation never allocates or frees.
    pub fn resume_activation_conditioning(&mut self) -> Result<(), FiLMModulationError> {
        self.activation_modulation_size()?;
        resume_layers(&mut self.layers);
        Ok(())
    }
}

impl WaveNetA2Dyn {
    /// Returns the concatenated activation-pre feature count (before gating).
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] for incomplete/inconsistent preparation.
    pub fn activation_modulation_size(&self) -> Result<usize, FiLMModulationError> {
        layer_count(&self.layers)
    }

    /// Sets final activation-pre scales/shifts in layer order, atomically.
    ///
    /// Shares [`WaveNetA2::set_activation_modulation`]'s lifecycle and real-time
    /// contract. Feature widths may differ from residual widths with bottlenecks
    /// or gating. Other FiLM insertion points and condition DSPs are preserved.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] before any writes on invalid input/state.
    pub fn set_activation_modulation(
        &mut self,
        scales: &[f32],
        shifts: &[f32],
    ) -> Result<(), FiLMModulationError> {
        validate_modulation(scales, shifts, self.activation_modulation_size()?)?;
        let mut pos = 0;
        write_layers(&mut self.layers, scales, shifts, &mut pos);
        Ok(())
    }

    /// Resumes loaded activation-pre condition projections without allocation.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] without mutation for unprepared topology.
    pub fn resume_activation_conditioning(&mut self) -> Result<(), FiLMModulationError> {
        self.activation_modulation_size()?;
        resume_layers(&mut self.layers);
        Ok(())
    }
}

impl WaveNetA2Cascade {
    /// Returns the concatenated activation-pre feature count across all arrays.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] if any array/layer is unprepared or invalid.
    pub fn activation_modulation_size(&self) -> Result<usize, FiLMModulationError> {
        if self.arrays.is_empty() {
            return Err(FiLMModulationError::NotPrepared);
        }
        self.arrays.iter().try_fold(0usize, |count, a| {
            count
                .checked_add(layer_count(&a.layers)?)
                .ok_or(FiLMModulationError::InvalidTopology)
        })
    }

    /// Sets all activation-pre scales/shifts in array/layer order, atomically.
    ///
    /// Shares [`WaveNetA2::set_activation_modulation`]'s real-time contract.
    /// Validation covers every array before any layer is changed. Other FiLM
    /// slots, condition DSPs, temporal state and head scales are preserved.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] before any writes on invalid input/state.
    pub fn set_activation_modulation(
        &mut self,
        scales: &[f32],
        shifts: &[f32],
    ) -> Result<(), FiLMModulationError> {
        validate_modulation(scales, shifts, self.activation_modulation_size()?)?;
        let mut pos = 0;
        for a in &mut self.arrays {
            write_layers(&mut a.layers, scales, shifts, &mut pos);
        }
        Ok(())
    }

    /// Resumes every loaded activation-pre projection without allocation/reset.
    ///
    /// # Errors
    /// Returns [`FiLMModulationError`] before mutation if any layer is unprepared.
    pub fn resume_activation_conditioning(&mut self) -> Result<(), FiLMModulationError> {
        self.activation_modulation_size()?;
        for a in &mut self.arrays {
            resume_layers(&mut a.layers);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "modulation_test.rs"]
mod tests;
