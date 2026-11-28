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
//!
//! ## Real-time transition protocol
//!
//! The adaptive FSM calls `SlimmableModel::set_slimmable_size` from the
//! audio callback (`rt_status = Some`), so that path may not run a full
//! child `reset()`: for a nested `Sequential` that allocates (saved-flag
//! `Vec`, recreated pointer tables) and may log. Instead the RT path walks an
//! explicit state machine using the family-provided split-stabilization
//! primitives, which are documented as zero-allocation and lock-free:
//!
//! 1. **Idle → Preparing**: `prewarm_reset()` clears the target's temporal
//!    state and arms its zero-feed budget (no allocation, no logging).
//! 2. **Preparing**: every `process` call advances the armed budget by at
//!    most one host block, so the callback's added work is bounded by the
//!    work it already does for the active submodel. Output stays on the
//!    active submodel; the crossfade is never used to hide prewarm.
//! 3. **Preparing → Crossfading**: once the target converges, the usual
//!    32 ms linear crossfade starts. The target's scratch was sized by the
//!    off-RT `reset`/construction (`set_max_buffer_size` on every member), so
//!    the transition never allocates.
//!
//! A request that arrives while preparing supersedes the in-flight target
//! (latest wins). A request that arrives during the crossfade is queued and
//! honored only after the crossfade completes: the active index never jumps
//! mid-blend, preserving output continuity. Cancelling back to the current
//! active submodel aborts an in-flight preparation. No submodel is ever
//! replaced or dropped here — switch state only borrows the container's
//! permanent members — so the GC cascade is untouched by tier changes.

use std::cmp::Ordering;

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};
use crate::common::CROSSFADE_DURATION_MS;
use crate::common::diagnostics::NamErrorCode;
use crate::dsp::pipeline::MAX_RESAMP_BUF;
use crate::math::dsp::gain::crossfade_blend_ramp_mono_simd;

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
    /// Target whose zero-phase prewarm is in flight (`prewarm_step` budget
    /// advanced per audio block). Mutually exclusive with `pending_index`.
    preparing_index: Option<usize>,
    /// Latest requested target received while a crossfade was in flight.
    /// Honored only after the crossfade completes (replacement policy).
    requested_index: Option<usize>,
    crossfade_elapsed: usize,
    crossfade_duration: usize,
    scratch_buffer: Vec<f32>,
    sample_rate: u32,
    max_buffer_size: usize,
    prewarm_on_reset: bool,
}

/// Structural validation failure for the submodel spec passed to
/// [`ContainerModel::new`] / [`ContainerModel::new_typed`].
///
/// Carries enough detail for the `anyhow` constructor to preserve its
/// human-readable messages, while the typed constructor collapses every
/// variant to [`NamErrorCode::InvalidModelTopology`]. Both constructors share
/// a single validation pass ([`ContainerModel::validate_submodels`]) — no
/// drift between the two error paths.
#[derive(Debug)]
enum SpecError {
    /// `submodels` was empty.
    Empty,
    /// A `max_value` was non-finite or negative.
    InvalidMaxValue {
        /// Position of the offending submodel.
        index: usize,
        /// The rejected `max_value`.
        value: f32,
    },
    /// `max_value`s were not strictly ascending.
    NotAscending,
    /// The last `max_value` was `< 1.0`.
    LastBelowUnity,
}

impl SpecError {
    /// Maps every structural violation to the model-topology error code
    /// (E1305).
    fn code(&self) -> NamErrorCode {
        NamErrorCode::InvalidModelTopology
    }
}

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "ContainerModel: no submodels provided"),
            Self::InvalidMaxValue { index, value } => write!(
                f,
                "ContainerModel: submodel[{}] has invalid max_value={} \
                 (must be finite and >= 0.0)",
                index, value
            ),
            Self::NotAscending => write!(
                f,
                "ContainerModel: submodels must be sorted by ascending max_value"
            ),
            Self::LastBelowUnity => {
                write!(f, "ContainerModel: last submodel max_value must be >= 1.0")
            }
        }
    }
}

impl std::error::Error for SpecError {}

impl ContainerModel {
    /// Creates a new `ContainerModel` with a structured error type.
    ///
    /// Typed-error counterpart of [`new`](Self::new): identical validation and
    /// construction, but failures are reported as [`NamErrorCode`] instead of
    /// an opaque `anyhow` error, so a caller can triage them without a
    /// downcast. The `anyhow` constructor remains available; its replacement
    /// is deferred to the next breaking release.
    ///
    /// # Requirements (enforced)
    ///
    /// - `submodels` must be non-empty.
    /// - Sorted by `max_value` ascending (the constructor sorts in place).
    /// - Last `max_value` must be `>= 1.0`.
    ///
    /// # Errors
    ///
    /// - [`NamErrorCode::InvalidModelTopology`] (E1305) when `submodels` is
    ///   empty, a `max_value` is non-finite or negative, the `max_value`s are
    ///   not strictly ascending (duplicate or out-of-order), or the last
    ///   `max_value` is `< 1.0`.
    /// - [`NamErrorCode::OutOfMemory`] (E5000) when a submodel cannot allocate
    ///   its internal buffers (`set_max_buffer_size`).
    pub fn new_typed(
        mut submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
    ) -> Result<Self, NamErrorCode> {
        Self::validate_submodels(&mut submodels).map_err(|e| e.code())?;
        Self::assemble(submodels, sample_rate, true)
    }

    /// Creates a new `ContainerModel`.
    ///
    /// # Requirements (enforced)
    ///
    /// - `submodels` must be non-empty.
    /// - Sorted by `max_value` ascending.
    /// - Last `max_value` must be `>= 1.0`.
    ///
    /// # Errors
    ///
    /// Returns an error when any of the enforced requirements above is
    /// violated, namely:
    /// - `submodels` is empty;
    /// - any `max_value` is non-finite or negative;
    /// - `max_value` values are not strictly ascending (duplicate or
    ///   out-of-order);
    /// - the last `max_value` is `< 1.0`.
    ///
    /// Also propagates the error from `set_max_buffer_size` when a submodel
    /// cannot allocate its internal buffers (out of memory).
    ///
    /// For a typed-error counterpart (no `anyhow`), see
    /// [`new_typed`](Self::new_typed).
    pub fn new(
        mut submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        Self::validate_submodels(&mut submodels)?;
        Self::assemble(submodels, sample_rate, true).map_err(Into::into)
    }

    /// Typed-error constructor that honours the loader's `LoadOptions.prewarm`
    /// decision: when `prewarm` is `false` the construction-time stabilization
    /// pass is skipped, so a subsequent host reset (or the loader's
    /// `Some(false)` path) observes the raw cleared state (T5.4-5).
    pub(crate) fn new_with_prewarm(
        mut submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
        prewarm: bool,
    ) -> Result<Self, NamErrorCode> {
        Self::validate_submodels(&mut submodels).map_err(|e| e.code())?;
        Self::assemble(submodels, sample_rate, prewarm)
    }

    /// Validates the submodel ordering/value invariants shared by [`new`] and
    /// [`new_typed`](Self::new_typed), sorting `submodels` ascending by
    /// `max_value` in place first.
    fn validate_submodels(submodels: &mut [(f32, Box<StaticModel>)]) -> Result<(), SpecError> {
        if submodels.is_empty() {
            return Err(SpecError::Empty);
        }

        submodels.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        for (i, (max_value, _)) in submodels.iter().enumerate() {
            if !max_value.is_finite() || *max_value < 0.0 {
                return Err(SpecError::InvalidMaxValue {
                    index: i,
                    value: *max_value,
                });
            }
        }

        for w in submodels.windows(2) {
            if w[1].0 <= w[0].0 {
                return Err(SpecError::NotAscending);
            }
        }

        if let Some((last_max, _)) = submodels.last()
            && *last_max < 1.0
        {
            return Err(SpecError::LastBelowUnity);
        }

        Ok(())
    }

    /// Builds the container from an already-validated submodel list.
    ///
    /// Allocation failures while sizing each submodel's scratch buffers are
    /// reported as [`NamErrorCode::OutOfMemory`], per the
    /// `NamModel::set_max_buffer_size` contract.
    fn assemble(
        submodels: Vec<(f32, Box<StaticModel>)>,
        sample_rate: u32,
        prewarm: bool,
    ) -> Result<Self, NamErrorCode> {
        let active_index = submodels.len() - 1;
        let crossfade_duration =
            (CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize;
        // The scratch must accommodate the largest pipeline block (`MAX_RESAMP_BUF`),
        // even when no explicit `set_max_buffer_size` call precedes processing.
        let default_buf = 4096usize.max(MAX_RESAMP_BUF);

        let mut container = Self {
            submodels,
            active_index,
            pending_index: None,
            preparing_index: None,
            requested_index: None,
            crossfade_elapsed: 0,
            crossfade_duration: crossfade_duration.max(1),
            scratch_buffer: vec![0.0f32; default_buf],
            sample_rate,
            max_buffer_size: default_buf,
            prewarm_on_reset: true,
        };

        for (_, model) in &mut container.submodels {
            model
                .set_max_buffer_size(default_buf)
                .map_err(|_| NamErrorCode::OutOfMemory)?;
        }
        // T5.4-5: the construction-time stabilization pass is governed by the
        // loader's `LoadOptions.prewarm` (threaded through
        // `ContainerModel::new_with_prewarm`, `build.rs`). `new`/`new_typed`
        // keep the historical prewarm for direct builders and C++ golden
        // references; `prewarm == false` (from `LoadOptions::Some(false)`)
        // skips it so the raw cleared state is observable.
        if prewarm {
            container.prewarm(4096);
        }

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
        self.preparing_index = None;
        self.requested_index = None;
        self.crossfade_elapsed = 0;
    }

    /// Returns the index of the pending submodel during crossfade, if any.
    pub fn pending_index(&self) -> Option<usize> {
        self.pending_index
    }

    /// Returns the index whose split prewarm is in flight, if any.
    pub fn preparing_index(&self) -> Option<usize> {
        self.preparing_index
    }

    /// Returns the target queued behind an in-flight crossfade, if any.
    pub fn requested_index(&self) -> Option<usize> {
        self.requested_index
    }

    /// Returns `true` if a split prewarm is converging.
    pub fn is_preparing(&self) -> bool {
        self.preparing_index.is_some()
    }

    /// Returns `true` if a crossfade is in progress.
    pub fn is_crossfading(&self) -> bool {
        self.pending_index.is_some()
    }

    /// Returns the sample rate used to validate submodels.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the currently active submodel (the one producing the output).
    ///
    /// Exposed so hosts and reset-contract tests can inspect the active
    /// member's family without reaching into the private submodel list.
    pub fn active(&self) -> &StaticModel {
        &self.submodels[self.active_index].1
    }
}

impl NamModel for ContainerModel {
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());

        // Phase 1: converge the target's split prewarm with a per-block budget
        // bounded by the block the callback is already paying for. Output
        // stays on the active submodel; the crossfade income is not used to
        // hide this work.
        if let Some(preparing_idx) = self.preparing_index {
            if n > 0 {
                let remaining = self.submodels[preparing_idx].1.prewarm_step(n);
                if remaining == 0 {
                    self.preparing_index = None;
                    self.pending_index = Some(preparing_idx);
                    self.crossfade_elapsed = 0;
                }
            }
            self.submodels[self.active_index]
                .1
                .process(&input[..n], &mut output[..n]);
            return;
        }

        if let Some(pending_idx) = self.pending_index {
            // Scratch capacity guard: never slice the scratch beyond its length.
            // If the block is larger than the scratch, abort the crossfade for
            // this block (process only the active submodel) without panicking;
            // the crossfade resumes on a subsequent fitting block.
            if n > self.scratch_buffer.len() {
                self.submodels[self.active_index]
                    .1
                    .process(&input[..n], &mut output[..n]);
                return;
            }

            let active_idx = self.active_index;

            // The pending model must render the whole block: it becomes the
            // active model at the end (possibly mid-block). The outgoing
            // model only needs to render the frames that still belong to the
            // ramp, so its state never advances past the frames it produced.
            self.submodels[pending_idx]
                .1
                .process(&input[..n], &mut self.scratch_buffer[..n]);

            let remaining = self
                .crossfade_duration
                .saturating_sub(self.crossfade_elapsed);
            let ramp_len = n.min(remaining);
            if ramp_len > 0 {
                self.submodels[active_idx]
                    .1
                    .process(&input[..ramp_len], &mut output[..ramp_len]);

                // Per-sample ramp: fixes the block-constant envelope that left
                // an audible step when a block was as long as the crossfade.
                let t_start = self.crossfade_elapsed as f32 / self.crossfade_duration as f32;
                let t_step = 1.0 / self.crossfade_duration as f32;
                crossfade_blend_ramp_mono_simd(
                    &mut output[..ramp_len],
                    &self.scratch_buffer[..ramp_len],
                    t_start,
                    t_step,
                );
            }
            if ramp_len < n {
                // The ramp ended inside this block: the remaining frames are
                // fully owned by the incoming model.
                output[ramp_len..n].copy_from_slice(&self.scratch_buffer[ramp_len..n]);
            }

            self.crossfade_elapsed += n;
            if self.crossfade_elapsed >= self.crossfade_duration {
                self.active_index = pending_idx;
                self.pending_index = None;
                self.crossfade_elapsed = 0;

                // Replacement policy: the active index never jumps mid-blend.
                // A request queued during the crossfade is honored now, from
                // the newly active submodel, as a fresh preparation.
                if let Some(next) = self.requested_index.take()
                    && next != self.active_index
                {
                    self.begin_rt_transition(next);
                }
            }
        } else {
            self.submodels[self.active_index].1.process(input, output);
        }
    }

    /// Channel geometry delegated to the active submodel; the container does
    /// not alter the geometry of its members.
    #[inline(always)]
    fn num_input_channels(&self) -> usize {
        self.active().num_input_channels()
    }

    /// Channel geometry delegated to the active submodel.
    #[inline(always)]
    fn num_output_channels(&self) -> usize {
        self.active().num_output_channels()
    }

    fn prewarm(&mut self, num_samples: usize) {
        for (_, model) in &mut self.submodels {
            model.prewarm(num_samples);
        }
    }

    /// Zero phase of the deferred split flow: every submodel runs its own
    /// zero phase; their feeds stay armed and drain through
    /// [`prewarm_step`](Self::prewarm_step). Real-time safe.
    fn prewarm_reset(&mut self) {
        for (_, model) in &mut self.submodels {
            model.prewarm_reset();
        }
    }

    /// Advances every armed submodel feed within a single aggregate
    /// `samples`-frame budget and returns the maximum pending count
    /// (0 = all converged). Real-time safe.
    ///
    /// The budget is split across the still-incomplete submodels (ceiling
    /// split, so the shares sum to `samples`), guaranteeing that a nested
    /// container cannot multiply the per-callback work by the number of
    /// members. A child may consume less than its share when it converges
    /// early; unused budget is not reused, keeping the ceiling conservative.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        let active = self
            .submodels
            .iter()
            .filter(|(_, m)| !m.prewarm_complete())
            .count();
        if active == 0 {
            return 0;
        }
        let mut budget = samples;
        let mut remaining_active = active;
        let mut rem = 0usize;
        for (_, model) in &mut self.submodels {
            if model.prewarm_complete() {
                continue;
            }
            let share =
                budget / remaining_active + usize::from(!budget.is_multiple_of(remaining_active));
            let pending = model.prewarm_step(share);
            budget = budget.saturating_sub(share);
            remaining_active -= 1;
            rem = rem.max(pending);
        }
        rem
    }

    /// Deferred pass pending? (`true` when no submodel has work left.)
    fn prewarm_complete(&self) -> bool {
        self.submodels.iter().all(|(_, m)| m.prewarm_complete())
    }

    /// Clears every submodel's temporal state (all members, not only the
    /// active one) by delegating to each child's [`NamModel::clear_state`].
    /// Real-time safe.
    fn clear_state(&mut self) {
        for (_, model) in &mut self.submodels {
            model.clear_state();
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.sample_rate = sample_rate;
        self.max_buffer_size = max_buffer_size;
        self.scratch_buffer.resize(max_buffer_size, 0.0);
        self.crossfade_duration =
            ((CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize).max(1);
        // A host reset is the off-RT preparation point: it sizes every
        // member's scratch, so later RT transitions never allocate. It also
        // cancels every in-flight transition component (preparation, armed
        // crossfade, queued request and the fade clock) *before* touching the
        // inactive members, so no post-reset block can resume a blend against
        // state that was just cleared. The active member is the tier that
        // survives a reset.
        self.preparing_index = None;
        self.pending_index = None;
        self.requested_index = None;
        self.crossfade_elapsed = 0;
        for (_, model) in &mut self.submodels {
            model.set_max_buffer_size(max_buffer_size)?;
        }
        // `set_max_buffer_size` is idempotent (equality is a no-op), so reset
        // semantics must not rely on it to wipe an inactive member. Clear the
        // inactive members explicitly; only the active member is prewarmed.
        for (idx, (_, model)) in self.submodels.iter_mut().enumerate() {
            if idx != self.active_index {
                model.clear_state();
            }
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
        let next = self.index_for_size(val);

        // Off-RT path (tests, direct builders, host housekeeping): keep the
        // historical full-reset semantics, including the immediate jump to a
        // pending target when a second request arrives mid-crossfade.
        if rt_status.is_none() {
            if next == self.active_index && self.pending_index.is_none() {
                self.preparing_index = None;
                self.requested_index = None;
                return;
            }
            if self.pending_index == Some(next) {
                self.requested_index = None;
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
            self.preparing_index = None;
            self.requested_index = None;

            if let Err(_e) = self.submodels[next]
                .1
                .reset(self.sample_rate, self.max_buffer_size)
            {
                return;
            }

            self.pending_index = Some(next);
            self.crossfade_elapsed = 0;
            return;
        }

        // Real-time path: never run a child `reset()`. The off-RT
        // `reset`/construction already sized every member's scratch, so the
        // transition reduces to the zero-allocation split-prewarm primitives.
        if next == self.active_index && self.pending_index.is_none() {
            // Desired equals the current output: cancel any in-flight
            // preparation and any queued replacement.
            self.preparing_index = None;
            self.requested_index = None;
            return;
        }
        if self.pending_index == Some(next) {
            self.requested_index = None;
            return;
        }
        if self.preparing_index == Some(next) {
            self.requested_index = None;
            return;
        }
        if self.pending_index.is_some() {
            // Continuity: finish the blend already in flight, then honor the
            // latest request from the newly active submodel.
            self.requested_index = Some(next);
            return;
        }

        // Idle or preparing: supersede with the latest target (latest wins).
        self.requested_index = None;
        self.begin_rt_transition(next);
    }
}

impl ContainerModel {
    /// Submodel index selected by a normalized quality value (C++
    /// `_get_index_for_slimmable_size`): the first member whose `max_value`
    /// is strictly greater than `val`, else the last member.
    fn index_for_size(&self, val: f32) -> usize {
        let mut next = self.submodels.len() - 1;
        for (i, (max_value, _)) in self.submodels.iter().enumerate() {
            if val < *max_value {
                next = i;
                break;
            }
        }
        next
    }

    /// Arms the RT-safe preparation of `next` (zero heap, zero I/O).
    ///
    /// The family split-stabilization primitive clears temporal state and
    /// arms a bounded zero-feed budget; `process` drains it one host block at
    /// a time before the crossfade starts. Members created without a
    /// split-stabilization implementation (infallible defaults) arm nothing
    /// and transition immediately, matching a `clear_state`-only reset.
    fn begin_rt_transition(&mut self, next: usize) {
        let target = &mut self.submodels[next].1;
        if target.prewarm_on_reset() {
            target.prewarm_reset();
            self.preparing_index = Some(next);
        } else {
            target.clear_state();
            self.preparing_index = None;
            self.pending_index = Some(next);
            self.crossfade_elapsed = 0;
        }
    }
}

impl super::sealed::Sealed for ContainerModel {}

#[cfg(test)]
#[path = "container_test.rs"]
mod tests;
