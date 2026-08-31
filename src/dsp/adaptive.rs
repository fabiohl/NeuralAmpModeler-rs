// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Adaptive compute: soft-degrade under CPU pressure (graceful fallback).
//!
//! Monitors DSP block latency and gracefully reduces model complexity
//! when CPU pressure is detected, preventing audible xruns.
//!
//! ## Cohesion justification
//!
//! This file is intentionally kept as a single module (not split) because:
//!
//! - `AdaptiveState` and `CrossfadePhase` are internal implementation details of
//!   the hysteresis FSM — exposing them in separate modules would break
//!   encapsulation of the adaptive compute unit.
//! - `AdaptiveComputeMode` is the sole user-facing configuration type driving the
//!   FSM behavior; it has zero consumers outside the adaptive degradation pipeline.
//! - All threshold constants (`THRESHOLD_*`, `DEGRADE_CONSECUTIVE`,
//!   `RECOVER_CONSECUTIVE`) are private and tightly
//!   coupled to `AdaptiveCompute::update()`. `CROSSFADE_DURATION_MS` is the
//!   shared crate-wide tuning constant (`crate::common`), imported here.
//! - The entire module forms a single algorithmic unit (hysteresis FSM +
//!   crossfade + layer reduction) with no independent consumers for any
//!   sub-component.
//! - Splitting would create artificial module boundaries that increase cognitive
//!   overhead without enabling reuse, test isolation, or independent compilation.
//!
//! ## Hysteresis FSM
//!
//! - **Full → Reduced**: 3 consecutive blocks with `latency > 0.70 * budget` (Conservative)
//!   or `latency > 0.55 * budget` (Aggressive).
//! - **Reduced → Minimal**: 3 consecutive blocks with `latency > 0.85 * budget` (Conservative)
//!   or `latency > 0.70 * budget` (Aggressive).
//! - **Reverse path**: 5 consecutive blocks below recovery threshold.
//!
//! ## Crossfade
//!
//! 32 ms linear ramp between states to avoid audible artifacts.

use crate::common::CROSSFADE_DURATION_MS;
use crate::common::spsc::RtStatusFlags;
use crate::common::unlikely;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

/// Manual slim override — forces a fixed quality level, bypassing the FSM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum SlimOverride {
    /// FSM decides quality level automatically.
    #[default]
    Auto = 0,
    /// Force full-quality processing regardless of CPU pressure.
    ForceFull = 1,
    /// Force lite (reduced) processing regardless of CPU pressure.
    ForceLite = 2,
}

impl SlimOverride {
    /// Creates a `SlimOverride` from a floating-point value.
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            1 => Self::ForceFull,
            2 => Self::ForceLite,
            _ => Self::Auto,
        }
    }
}

/// Adaptive compute mode — user-facing parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum AdaptiveComputeMode {
    /// Adaptive compute disabled (passthrough).
    #[default]
    Off = 0,
    /// Conservative threshold — late degradation.
    Conservative = 1,
    /// Aggressive threshold — early degradation.
    Aggressive = 2,
}

impl AdaptiveComputeMode {
    /// Creates an `AdaptiveComputeMode` from a floating-point value.
    pub fn from_f32(val: f32) -> Self {
        match val.round() as i32 {
            1 => Self::Conservative,
            2 => Self::Aggressive,
            _ => Self::Off,
        }
    }

    /// Converts the mode to its corresponding floating-point representation.
    pub fn to_f32(self) -> f32 {
        self as u8 as f32
    }
}

/// Degradation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdaptiveState {
    /// Full processing — all model layers active.
    Full = 0,
    /// Reduced processing — fewer layers.
    Reduced = 1,
    /// Minimal processing — maximum reduction.
    Minimal = 2,
}

/// Crossfade phase between degradation stages.
///
/// Duration and progress are tracked by `crossfade_total` and `crossfade_elapsed`
/// on `AdaptiveCompute` directly; no payload is needed in the variant.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CrossfadePhase {
    /// No crossfade in progress.
    Idle,
    /// Actively crossfading; see `crossfade_total` / `crossfade_elapsed`.
    Active,
}

/// Hysteresis-driven adaptive compute FSM.
///
/// Uses asymmetric hysteresis:
/// - Upgrade (degrade): 3 consecutive blocks above threshold.
/// - Downgrade (recover): 5 consecutive blocks below threshold.
pub struct AdaptiveCompute {
    /// Current degradation state.
    state: AdaptiveState,
    /// Previous degradation state (during crossfade).
    prev_state: AdaptiveState,
    /// Consecutive overload counter for degradation.
    overload_counter: u32,
    /// Consecutive underload counter for recovery.
    recovery_counter: u32,
    /// Crossfade phase.
    crossfade: CrossfadePhase,
    /// Current user mode.
    mode: AdaptiveComputeMode,
    /// Crossfade total samples (fixed duration for linear progress).
    crossfade_total: usize,
    /// Crossfade elapsed samples.
    crossfade_elapsed: usize,
    /// Manual slim override (Auto = FSM decides).
    slim_override: SlimOverride,
    /// Original (full-quality) channel count of the loaded WaveNet model.
    /// `None` until a WaveNet model is detected via `set_wavenet_full_ch`.
    wavenet_full_ch: Option<usize>,
    /// Current slimmable channel count of the active WaveNet model.
    /// Tracks the last value applied via `take_slimmable_rebuild`.
    wavenet_slim_ch_current: Option<usize>,
    /// Whether the loaded WaveNet model explicitly declares slimmability support.
    /// Conventional WaveNets (default) set this to `false` and maintain full channel size.
    wavenet_slimmable_capable: bool,
}

const DEGRADE_CONSECUTIVE: u32 = 3;
const RECOVER_CONSECUTIVE: u32 = 5;

const THRESHOLD_FULL_TO_REDUCED_CONSERVATIVE: f32 = 0.70;
const THRESHOLD_REDUCED_TO_MINIMAL_CONSERVATIVE: f32 = 0.85;
const THRESHOLD_FULL_TO_REDUCED_AGGRESSIVE: f32 = 0.55;
const THRESHOLD_REDUCED_TO_MINIMAL_AGGRESSIVE: f32 = 0.70;

impl AdaptiveCompute {
    /// Creates a new `AdaptiveCompute` instance with the given mode.
    #[cold]
    pub fn new(mode: AdaptiveComputeMode) -> Self {
        Self {
            state: AdaptiveState::Full,
            prev_state: AdaptiveState::Full,
            overload_counter: 0,
            recovery_counter: 0,
            crossfade: CrossfadePhase::Idle,
            mode,
            crossfade_total: 0,
            crossfade_elapsed: 0,
            slim_override: SlimOverride::Auto,
            wavenet_full_ch: None,
            wavenet_slim_ch_current: None,
            wavenet_slimmable_capable: false,
        }
    }

    /// Resets the FSM to its initial Full state: zeroes the hysteresis
    /// counters and aborts any in-flight crossfade, preserving the user
    /// configuration (`mode`, `slim_override`, WaveNet slimmable tracking).
    ///
    /// The RT status degrade flags are cleared to match the reverted state
    /// (the main thread uses them to report degradation).
    ///
    /// RT-safe: zero allocations.
    #[inline(always)]
    pub fn reset(&mut self, rt_status: &RtStatusFlags) {
        self.state = AdaptiveState::Full;
        self.prev_state = AdaptiveState::Full;
        self.overload_counter = 0;
        self.recovery_counter = 0;
        self.crossfade = CrossfadePhase::Idle;
        self.crossfade_total = 0;
        self.crossfade_elapsed = 0;
        rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
        rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
    }

    /// Sets the user-facing adaptive compute mode.
    #[inline]
    pub fn set_mode(&mut self, mode: AdaptiveComputeMode, rt_status: &RtStatusFlags) {
        self.mode = mode;
        if mode == AdaptiveComputeMode::Off {
            self.state = AdaptiveState::Full;
            self.prev_state = AdaptiveState::Full;
            self.overload_counter = 0;
            self.recovery_counter = 0;
            self.crossfade = CrossfadePhase::Idle;
            self.crossfade_total = 0;
            self.crossfade_elapsed = 0;
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
            rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
        }
    }

    /// Returns the current adaptive compute mode.
    pub fn mode(&self) -> AdaptiveComputeMode {
        self.mode
    }

    /// Sets the manual slim override, forcing a fixed quality level.
    #[inline]
    pub fn set_slim_override(&mut self, ov: SlimOverride) {
        self.slim_override = ov;
    }

    /// Returns the current slim override.
    pub fn slim_override(&self) -> SlimOverride {
        self.slim_override
    }

    /// Returns the current degradation state of the FSM, respecting manual override.
    pub fn state(&self) -> AdaptiveState {
        match self.slim_override {
            SlimOverride::ForceFull => AdaptiveState::Full,
            SlimOverride::ForceLite => AdaptiveState::Reduced,
            SlimOverride::Auto => self.state,
        }
    }

    /// Returns the crossfade gain (`crossfade_elapsed / total`).
    /// `sample_rate` is used to compute the crossfade duration in samples.
    /// Returns the linear multiplier applied to the degradation path.
    /// Call this during the output stage.
    #[inline(always)]
    pub fn crossfade_multiplier(&mut self, _sample_rate: u32, frame_advance: usize) -> f32 {
        match self.crossfade {
            CrossfadePhase::Idle => {
                if self.state == AdaptiveState::Full {
                    0.0
                } else {
                    1.0
                }
            }
            CrossfadePhase::Active => {
                let total = self.crossfade_total;
                let progress = if total == 0 {
                    1.0
                } else {
                    (self.crossfade_elapsed as f32 / total as f32).min(1.0)
                };

                if self.crossfade_elapsed + frame_advance >= total {
                    self.crossfade = CrossfadePhase::Idle;
                    self.crossfade_elapsed = 0;
                    1.0
                } else {
                    self.crossfade_elapsed += frame_advance;
                    progress
                }
            }
        }
    }

    /// Returns `true` if a crossfade is in progress.
    #[inline(always)]
    pub fn is_crossfading(&self) -> bool {
        !matches!(self.crossfade, CrossfadePhase::Idle)
    }

    /// Feeds the FSM with the latest latency measurement.
    ///
    /// `latency_us` — processing time of this block in microseconds.
    /// `budget_us` — budget for this block in microseconds.
    /// `sample_rate` — sample rate for crossfade duration.
    pub fn update(
        &mut self,
        latency_us: u64,
        budget_us: u64,
        sample_rate: u32,
        rt_status: &RtStatusFlags,
    ) {
        if self.mode == AdaptiveComputeMode::Off
            || budget_us == 0
            || self.slim_override != SlimOverride::Auto
        {
            return;
        }

        // ── Step 1: Compute latency/budget ratio and hysteresis thresholds ──
        let ratio = latency_us as f32 / budget_us as f32;
        let (full_to_reduced, reduced_to_minimal) = match self.mode {
            AdaptiveComputeMode::Conservative => (
                THRESHOLD_FULL_TO_REDUCED_CONSERVATIVE,
                THRESHOLD_REDUCED_TO_MINIMAL_CONSERVATIVE,
            ),
            AdaptiveComputeMode::Aggressive => (
                THRESHOLD_FULL_TO_REDUCED_AGGRESSIVE,
                THRESHOLD_REDUCED_TO_MINIMAL_AGGRESSIVE,
            ),
            AdaptiveComputeMode::Off => return,
        };

        // Recovery thresholds use 50% hysteresis to prevent oscillation
        let recovery_reduced = full_to_reduced * 0.5;
        let recovery_minimal = reduced_to_minimal * 0.5;

        // ── Step 2: FSM state transition with consecutive-sample confirmation ──
        match self.state {
            AdaptiveState::Full => {
                if ratio > full_to_reduced {
                    // Overload detected — increment counter toward degradation
                    self.overload_counter = self.overload_counter.saturating_add(1);
                    self.recovery_counter = 0;
                    if unlikely(self.overload_counter >= DEGRADE_CONSECUTIVE) {
                        self.transition_to(AdaptiveState::Reduced, sample_rate, rt_status);
                    }
                } else {
                    self.overload_counter = 0;
                }
            }
            AdaptiveState::Reduced => {
                if ratio > reduced_to_minimal {
                    // Further degradation — escalate to Minimal
                    self.overload_counter = self.overload_counter.saturating_add(1);
                    self.recovery_counter = 0;
                    if unlikely(self.overload_counter >= DEGRADE_CONSECUTIVE) {
                        self.transition_to(AdaptiveState::Minimal, sample_rate, rt_status);
                    }
                } else if ratio < recovery_reduced {
                    // Headroom recovered — potentially restore Full
                    self.recovery_counter = self.recovery_counter.saturating_add(1);
                    self.overload_counter = 0;
                    if self.recovery_counter >= RECOVER_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Full, sample_rate, rt_status);
                        self.recovery_counter = 0;
                    }
                } else {
                    // Within steady-state band — reset counters
                    self.overload_counter = 0;
                    self.recovery_counter = 0;
                }
            }
            AdaptiveState::Minimal => {
                if ratio < recovery_minimal {
                    // Headroom recovered — restore to Reduced
                    self.recovery_counter = self.recovery_counter.saturating_add(1);
                    if self.recovery_counter >= RECOVER_CONSECUTIVE {
                        self.transition_to(AdaptiveState::Reduced, sample_rate, rt_status);
                        self.recovery_counter = 0;
                    }
                } else {
                    self.recovery_counter = 0;
                }
            }
        }
    }

    fn transition_to(
        &mut self,
        new_state: AdaptiveState,
        sample_rate: u32,
        rt_status: &RtStatusFlags,
    ) {
        if self.state == new_state {
            return;
        }

        let current_progress = match self.crossfade {
            CrossfadePhase::Active if self.crossfade_total > 0 => {
                (self.crossfade_elapsed as f32 / self.crossfade_total as f32).min(1.0)
            }
            _ => 0.0,
        };

        if !matches!(self.crossfade, CrossfadePhase::Active) {
            self.prev_state = self.state;
        }
        self.state = new_state;
        self.overload_counter = 0;
        self.recovery_counter = 0;

        let crossfade_samples =
            (CROSSFADE_DURATION_MS / 1000.0 * sample_rate as f32).round() as usize;
        let new_total = crossfade_samples.max(1);
        self.crossfade = CrossfadePhase::Active;
        self.crossfade_total = new_total;
        self.crossfade_elapsed = (current_progress * new_total as f32).round() as usize;

        rt_status
            .degrade_transitions_total
            .fetch_add(1, Ordering::Relaxed);

        match new_state {
            AdaptiveState::Full => {
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
            AdaptiveState::Reduced => {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.clear_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
            AdaptiveState::Minimal => {
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_REDUCED);
                rt_status.set_flag(crate::common::spsc::RT_STATUS_DEGRADE_MINIMAL);
            }
        }
    }

    /// Fraction of WaveNet layers to skip: 0.0 (Full), 0.25 (Reduced), 0.50 (Minimal).
    pub fn wavenet_skip_fraction(&self) -> f32 {
        match self.state() {
            AdaptiveState::Full => 0.0,
            AdaptiveState::Reduced => 0.25,
            AdaptiveState::Minimal => 0.50,
        }
    }

    /// Effective number of WaveNet layers after reduction.
    #[inline(always)]
    pub fn wavenet_effective_layers(&self, total_layers: usize) -> usize {
        let skip = (total_layers as f32 * self.wavenet_skip_fraction()).round() as usize;
        total_layers.saturating_sub(skip).max(1)
    }

    /// Effective LSTM layers: 2 (Full), 1 (Reduced), 0 = passthrough (Minimal).
    pub fn lstm_effective_layers(&self, total_layers: usize) -> usize {
        match self.state() {
            AdaptiveState::Full => total_layers,
            AdaptiveState::Reduced => total_layers.min(1), // Keep at most 1
            AdaptiveState::Minimal => 0,                   // Passthrough
        }
    }

    /// Slimmable quality mapping for `SlimmableContainer` submodel selection.
    ///
    /// Returns a value in `[0.0, 1.0]` to be passed to `SlimmableModel::set_slimmable_size()`:
    /// - `Full` → `1.0` (highest-quality submodel — fallback, last in ordered list).
    /// - `Reduced` → `0.25` (mid-quality submodel — typically A2-Lite for 2-model containers).
    /// - `Minimal` → `0.0` (cheapest submodel — first in ordered list).
    ///
    /// For a typical 2-model container (Lite threshold 0.5, Full threshold 1.0):
    /// - `1.0` selects the Full model (≥0.5, falls through to last).
    /// - `0.25` selects the Lite model (<0.5, hits first threshold).
    /// - `0.0` selects the Lite model (<0.5).
    ///
    /// The container's internal threshold logic (`val < max_value`) picks the first
    /// submodel whose `max_value` is strictly greater than `val`.
    pub fn slimmable_size(&self) -> f32 {
        match self.slim_override {
            SlimOverride::ForceFull => 1.0,
            SlimOverride::ForceLite => 0.25,
            SlimOverride::Auto => match self.state {
                AdaptiveState::Full => 1.0,
                AdaptiveState::Reduced => 0.25,
                AdaptiveState::Minimal => 0.0,
            },
        }
    }

    /// Target channel count for WaveNet slimming, based on FSM state + override.
    ///
    /// Maps the adaptive degradation level to a fraction of the original channels:
    /// - `Full` → 100% (no reduction).
    /// - `Reduced` → 75% (3/4 of original channels).
    /// - `Minimal` → 50% (1/2 of original channels).
    /// - `ForceFull` → 100%, `ForceLite` → 75%.
    ///
    /// Clamped to `[4, original_ch]` (4 is the minimum viable channel count).
    pub fn wavenet_slimmable_ch_target(&self) -> Option<usize> {
        let original = self.wavenet_full_ch?;
        if !self.wavenet_slimmable_capable {
            return Some(original);
        }
        let fraction = match self.slim_override {
            SlimOverride::ForceFull => 1.0,
            SlimOverride::ForceLite => 0.75,
            SlimOverride::Auto => match self.state {
                AdaptiveState::Full => 1.0,
                AdaptiveState::Reduced => 0.75,
                AdaptiveState::Minimal => 0.50,
            },
        };
        let target = (original as f32 * fraction).round() as usize;
        Some(target.max(4).min(original))
    }

    /// Records the full-quality channel count of the currently loaded WaveNet model
    /// and whether it explicitly declares slimmability capability.
    ///
    /// Conventional WaveNet models without slimmability metadata pass `capable = false`,
    /// preventing `take_slimmable_rebuild` from altering their array widths.
    pub fn set_wavenet_full_ch(&mut self, ch: usize, capable: bool) {
        self.wavenet_full_ch = Some(ch);
        self.wavenet_slim_ch_current = Some(ch);
        self.wavenet_slimmable_capable = capable;
    }

    /// Returns `Some(target_ch)` if a WaveNet slimmable rebuild is needed,
    /// i.e., the FSM/override demands a different channel count than what's
    /// currently loaded. Clears the pending flag and updates tracking.
    ///
    /// Returns `None` if the model is not slimmable-capable.
    /// Must be called **before** DSP (in `process_events` / `receive_commands`)
    /// to perform the allocation-intensive `slice_channels` off the hot-path.
    pub fn take_slimmable_rebuild(&mut self) -> Option<usize> {
        if !self.wavenet_slimmable_capable {
            return None;
        }
        let target = self.wavenet_slimmable_ch_target()?;
        let current = self.wavenet_slim_ch_current?;
        if current != target && target >= 4 {
            self.wavenet_slim_ch_current = Some(target);
            Some(target)
        } else {
            None
        }
    }

    /// Returns the previous degradation state.
    #[inline(always)]
    pub fn prev_state(&self) -> AdaptiveState {
        self.prev_state
    }

    /// Returns the current crossfade gain (`crossfade_elapsed / total`) without side effects.
    #[inline(always)]
    pub fn current_crossfade_multiplier(&self) -> f32 {
        match self.crossfade {
            CrossfadePhase::Idle => {
                if self.state() == AdaptiveState::Full {
                    0.0
                } else {
                    1.0
                }
            }
            CrossfadePhase::Active => {
                let total = self.crossfade_total;
                if total == 0 {
                    1.0
                } else {
                    (self.crossfade_elapsed as f32 / total as f32).min(1.0)
                }
            }
        }
    }

    /// Returns the fraction of WaveNet layers to skip for a given state: 0.0 (Full), 0.25 (Reduced), 0.50 (Minimal).
    #[inline(always)]
    pub fn wavenet_skip_fraction_for_state(&self, state: AdaptiveState) -> f32 {
        match state {
            AdaptiveState::Full => 0.0,
            AdaptiveState::Reduced => 0.25,
            AdaptiveState::Minimal => 0.50,
        }
    }

    /// Effective number of WaveNet layers after reduction for a given state, respecting overrides.
    #[inline(always)]
    pub fn wavenet_effective_layers_for_state(
        &self,
        state: AdaptiveState,
        total_layers: usize,
    ) -> usize {
        let state_to_use = match self.slim_override {
            SlimOverride::ForceFull => AdaptiveState::Full,
            SlimOverride::ForceLite => AdaptiveState::Reduced,
            SlimOverride::Auto => state,
        };
        let skip = (total_layers as f32 * self.wavenet_skip_fraction_for_state(state_to_use))
            .round() as usize;
        total_layers.saturating_sub(skip).max(1)
    }
}

#[cfg(test)]
#[path = "adaptive_test.rs"]
mod adaptive_test;
