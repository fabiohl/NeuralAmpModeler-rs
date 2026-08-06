// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `ContainerModel` — A bundle of pre-trained submodels selected by a quality threshold.
//!
//! Each submodel covers values up to (but not including) its `max_value`.
//! The last submodel is the fallback for values at or above the last threshold.
//!
//! ## Architecture
//!
//! This implements the official NAM `SlimmableContainer` architecture
//! (registered since file version 0.7.0). It is the format used by the
//! mainstream A2 distribution (nano + standard bundle).
//!
//! ## Crossfade
//!
//! Submodel switching uses a linear crossfade (32 ms) to prevent audible
//! clicks. During the transition, input is processed through both submodels
//! and outputs are blended linearly. The scratch buffer is pre-allocated
//! (set_max_buffer_size) for zero alloc on the hot path.

use std::cmp::Ordering;

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};
use crate::common::spsc::RT_STATUS_SLIMMABLE_RESET_FAILED;

const CROSSFADE_DURATION_MS: f32 = 32.0;

/// A bundle of pre-trained submodels selected by a quality threshold.
///
/// Each submodel covers values up to (but not including) its `max_value`.
/// The last submodel is the fallback for values at or above the last threshold.
///
/// ## Architecture
///
/// This implements the official NAM `SlimmableContainer` architecture
/// (registered since file version 0.7.0). It is the format used by the
/// mainstream A2 distribution (nano + standard bundle).
pub struct ContainerModel {
    submodels: Vec<(f32, Box<StaticModel>)>,
    active_index: usize,
    pending_index: Option<usize>,
    crossfade_elapsed: usize,
    crossfade_duration: usize,
    scratch_buffer: Vec<f32>,
    sample_rate: u32,
    max_buffer_size: usize,
    prewarm_on_reset: bool,
}

impl ContainerModel {
    /// Creates a new `ContainerModel`.
    ///
    /// # Requirements (enforced)
    ///
    /// - `submodels` must be non-empty.
    /// - Sorted by `max_value` ascending.
    /// - Last `max_value` must be `>= 1.0`.
    pub fn new(
        mut submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        if submodels.is_empty() {
            anyhow::bail!("ContainerModel: no submodels provided");
        }

        submodels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for (i, (max_value, _)) in submodels.iter().enumerate() {
            if !max_value.is_finite() || *max_value < 0.0 {
                anyhow::bail!(
                    "ContainerModel: submodel[{}] has invalid max_value={} \
                     (must be finite and >= 0.0)",
                    i,
                    max_value
                );
            }
        }

        for w in submodels.windows(2) {
            if w[1].0 <= w[0].0 {
                anyhow::bail!("ContainerModel: submodels must be sorted by ascending max_value");
            }
        }
        let last = submodels
            .last()
            .ok_or_else(|| anyhow::anyhow!("ContainerModel: submodels list is empty"))?;
        if last.0 < 1.0 {
            anyhow::bail!("ContainerModel: last submodel max_value must be >= 1.0");
        }

        let active_index = submodels.len() - 1;
        let crossfade_duration =
            (CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize;
        let default_buf = 4096usize;

        let mut container = Self {
            submodels,
            active_index,
            pending_index: None,
            crossfade_elapsed: 0,
            crossfade_duration: crossfade_duration.max(1),
            scratch_buffer: vec![0.0f32; default_buf],
            sample_rate,
            max_buffer_size: default_buf,
            prewarm_on_reset: true,
        };

        for (_, model) in &mut container.submodels {
            model.set_max_buffer_size(default_buf)?;
        }

        container.prewarm(4096);

        Ok(container)
    }

    /// Returns the list of `(max_value, StaticModel ref)` for diagnostics.
    pub fn submodels(&self) -> &[(f32, Box<StaticModel>)] {
        &self.submodels
    }

    /// Mutable access to submodels (for testing purposes only).
    pub fn submodels_mut(&mut self) -> &mut [(f32, Box<StaticModel>)] {
        &mut self.submodels
    }

    /// Returns the index of the currently active submodel.
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// Sets the active submodel index directly, bypassing crossfade (testing only).
    pub fn set_active_index(&mut self, idx: usize) {
        assert!(idx < self.submodels.len());
        self.active_index = idx;
        self.pending_index = None;
        self.crossfade_elapsed = 0;
    }

    /// Returns the index of the pending submodel during crossfade, if any.
    pub fn pending_index(&self) -> Option<usize> {
        self.pending_index
    }

    /// Returns `true` if a crossfade is in progress.
    pub fn is_crossfading(&self) -> bool {
        self.pending_index.is_some()
    }

    /// Returns the sample rate used to validate submodels.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub(crate) fn active(&self) -> &StaticModel {
        &self.submodels[self.active_index].1
    }
}

impl NamModel for ContainerModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len();

        if let Some(pending_idx) = self.pending_index {
            let active_idx = self.active_index;

            self.submodels[pending_idx]
                .1
                .process(input, &mut self.scratch_buffer[..n]);
            self.submodels[active_idx].1.process(input, output);

            let t = (self.crossfade_elapsed as f32 / self.crossfade_duration as f32).min(1.0);
            // SAFETY: scratch_buffer has the same length as output (both sized to n).
            unsafe {
                crate::math::dsp::gain::crossfade_blend_mono(output, &self.scratch_buffer[..n], t);
            }

            self.crossfade_elapsed += n;
            if self.crossfade_elapsed >= self.crossfade_duration {
                self.active_index = pending_idx;
                self.pending_index = None;
                self.crossfade_elapsed = 0;
            }
        } else {
            self.submodels[self.active_index].1.process(input, output);
        }
    }

    fn prewarm(&mut self, num_samples: usize) {
        for (_, model) in &mut self.submodels {
            model.prewarm(num_samples);
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.sample_rate = sample_rate;
        self.max_buffer_size = max_buffer_size;
        self.scratch_buffer.resize(max_buffer_size, 0.0);
        self.crossfade_duration =
            (CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize;
        for (_, model) in &mut self.submodels {
            model.set_max_buffer_size(max_buffer_size)?;
        }
        self.submodels[self.active_index]
            .1
            .reset(sample_rate, max_buffer_size)
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        self.max_buffer_size = max_buf;
        self.scratch_buffer.resize(max_buf, 0.0);
        for (_, model) in &mut self.submodels {
            model.set_max_buffer_size(max_buf)?;
        }
        Ok(())
    }

    fn prewarm_samples(&self) -> usize {
        self.active().prewarm_samples()
    }

    fn prewarm_on_reset(&self) -> bool {
        self.prewarm_on_reset
    }

    fn set_prewarm_on_reset(&mut self, val: bool) {
        self.prewarm_on_reset = val;
        for (_, model) in &mut self.submodels {
            model.set_prewarm_on_reset(val);
        }
    }
}

impl SlimmableModel for ContainerModel {
    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        if self.submodels.len() <= 1 {
            return Box::new([]);
        }
        self.submodels[..self.submodels.len() - 1]
            .iter()
            .map(|(max_value, _)| *max_value as f64)
            .collect()
    }

    fn set_slimmable_size(
        &mut self,
        val: f32,
        rt_status: Option<&crate::common::spsc::RtStatusFlags>,
    ) {
        let mut next = self.submodels.len() - 1;

        for (i, (max_value, _)) in self.submodels.iter().enumerate() {
            if val < *max_value {
                next = i;
                break;
            }
        }

        if next == self.active_index && self.pending_index.is_none() {
            return;
        }

        if self.pending_index == Some(next) {
            return;
        }

        debug_assert!(
            self.scratch_buffer.len() >= self.max_buffer_size,
            "scratch_buffer capacity invariant violated: {} < {}",
            self.scratch_buffer.len(),
            self.max_buffer_size
        );

        if let Some(idx) = self.pending_index {
            self.active_index = idx;
        }

        if let Err(_e) = self.submodels[next]
            .1
            .reset(self.sample_rate, self.max_buffer_size)
        {
            if let Some(s) = rt_status {
                s.set_flag(RT_STATUS_SLIMMABLE_RESET_FAILED);
            }
            return;
        }

        self.pending_index = Some(next);
        self.crossfade_elapsed = 0;
    }
}

impl super::sealed::Sealed for ContainerModel {}

#[cfg(test)]
#[path = "container_test.rs"]
mod tests;
