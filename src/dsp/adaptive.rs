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
#[non_exhaustive]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
    /// Array channel widths of the full-quality WaveNet model.
    /// Only populated by off-RT registrations; the RT mono path uses
    /// [`Self::wavenet_rt_target_ch`] instead so it never allocates.
    wavenet_array_channels: Vec<usize>,
    /// Per-array allowed channels metadata (`allowed_channels`).
    wavenet_allowed_channels: Option<Vec<Vec<usize>>>,
    /// Whether [`Self::wavenet_allowed_channels`] belongs to the active model.
    ///
    /// The RT registration path clears this flag without freeing the metadata
    /// (dropping a nested `Vec` would deallocate on the audio thread). A later
    /// off-RT registration replaces and frees it safely.
    wavenet_allowed_channels_active: bool,
    /// Current target channel counts of the active WaveNet model.
    /// Tracks the last value applied via `take_slimmable_rebuild`.
    /// An empty vector means "no applied target" (legacy `None`).
    wavenet_targets_current: Vec<usize>,
    /// Inline single-array state used by the RT registration path.
    ///
    /// `true` means the active model was registered via
    /// [`Self::set_wavenet_full_ch_rt`]: the mono channel count lives in
    /// [`Self::wavenet_full_ch`] and the applied target in
    /// [`Self::wavenet_rt_target_ch`], so the vector bookkeeping is bypassed
    /// and no allocation/deallocation is required. The stale vectors are
    /// reclaimed by the next off-RT registration.
    wavenet_mono_rt: bool,
    /// Applied single-array target for the RT mono path (`None` = no target).
    wavenet_rt_target_ch: Option<usize>,
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
            wavenet_array_channels: Vec::new(),
            wavenet_allowed_channels: None,
            wavenet_allowed_channels_active: false,
            wavenet_targets_current: Vec::new(),
            wavenet_mono_rt: false,
            wavenet_rt_target_ch: None,
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
    ///
    /// Note: The `_sample_rate` parameter is currently unused because crossfade
    /// duration is established internally during state transitions (32 ms window).
    /// It is retained for API compatibility and scheduled for review in v0.8.
    ///
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

    /// Returns the slimmable ratio corresponding to current FSM state and override.
    ///
    /// - Full: 1.0 (100% capacity)
    /// - Reduced: 0.75 (3/4 capacity)
    /// - Minimal: 0.50 (1/2 capacity)
    /// - ForceFull: 1.0, ForceLite: 0.75
    pub fn wavenet_slimmable_ratio(&self) -> f32 {
        match self.slim_override {
            SlimOverride::ForceFull => 1.0,
            SlimOverride::ForceLite => 0.75,
            SlimOverride::Auto => match self.state {
                AdaptiveState::Full => 1.0,
                AdaptiveState::Reduced => 0.75,
                AdaptiveState::Minimal => 0.50,
            },
        }
    }

    /// Target channel counts per array for WaveNet slimming, based on FSM state + override.
    ///
    /// Uses `allowed_channels` when present: for each array, calculates the tier index
    /// via `slimmable_tier_index(ratio, allowed.len())`. Fixed arrays (`allowed.is_empty()`)
    /// retain their full channel count.
    ///
    /// When `allowed_channels` is absent, scales channels by `ratio` and clamps to `[1, orig_ch]`.
    pub fn wavenet_slimmable_targets(&self) -> Option<Vec<usize>> {
        if self.wavenet_mono_rt {
            return Some(vec![self.wavenet_scalar_target(0)?]);
        }
        if !self.wavenet_slimmable_capable {
            return if self.wavenet_array_channels.is_empty() {
                self.wavenet_full_ch.map(|ch| vec![ch])
            } else {
                Some(self.wavenet_array_channels.clone())
            };
        }
        let ratio = self.wavenet_slimmable_ratio();
        if let Some(ref per_array) = self.wavenet_allowed_channels
            && self.wavenet_allowed_channels_active
        {
            let mut targets = Vec::with_capacity(per_array.len());
            for (i, allowed) in per_array.iter().enumerate() {
                if allowed.is_empty() {
                    let ch = self
                        .wavenet_array_channels
                        .get(i)
                        .copied()
                        .unwrap_or_else(|| self.wavenet_full_ch.unwrap_or(1));
                    targets.push(ch);
                } else {
                    let idx =
                        crate::models::slimmable::slimmable_tier_index(ratio as f64, allowed.len());
                    targets.push(allowed[idx]);
                }
            }
            Some(targets)
        } else {
            let mut targets = Vec::with_capacity(self.wavenet_array_channels.len());
            for &orig in &self.wavenet_array_channels {
                let t = ((orig as f32) * ratio).round() as usize;
                targets.push(t.clamp(1, orig));
            }
            if targets.is_empty() {
                let orig = self.wavenet_full_ch?;
                let t = ((orig as f32) * ratio).round() as usize;
                targets.push(t.clamp(1, orig));
            }
            Some(targets)
        }
    }

    /// Primary scalar channel target query (returns first array's target channels).
    ///
    /// Scalar, allocation-free variant of [`Self::wavenet_slimmable_targets`]
    /// for the mono/uniform-channel WaveNet fast path. Safe to call on the RT
    /// thread.
    pub fn wavenet_slimmable_ch_target(&self) -> Option<usize> {
        self.wavenet_scalar_target(0)
    }

    /// Returns the currently applied first-array channel count without cloning
    /// the target vector.
    ///
    /// Falls back to the registered full-quality width when no target has been
    /// applied yet. Safe to call on the RT thread.
    #[inline(always)]
    pub fn wavenet_current_ch(&self) -> Option<usize> {
        if self.wavenet_mono_rt {
            return self.wavenet_rt_target_ch.or(self.wavenet_full_ch);
        }
        self.wavenet_targets_current
            .first()
            .copied()
            .or(self.wavenet_full_ch)
    }

    /// Number of per-array targets the active configuration exposes.
    ///
    /// Zero means the legacy `wavenet_slimmable_targets()` would return `None`.
    #[inline(always)]
    fn wavenet_targets_len(&self) -> usize {
        if self.wavenet_mono_rt {
            return usize::from(self.wavenet_full_ch.is_some());
        }
        if self.wavenet_slimmable_capable {
            if self.wavenet_allowed_channels_active
                && let Some(per_array) = self.wavenet_allowed_channels.as_ref()
            {
                return per_array.len();
            }
            if !self.wavenet_array_channels.is_empty() {
                return self.wavenet_array_channels.len();
            }
            return usize::from(self.wavenet_full_ch.is_some());
        }
        if self.wavenet_array_channels.is_empty() {
            return usize::from(self.wavenet_full_ch.is_some());
        }
        self.wavenet_array_channels.len()
    }

    /// Scalar target channel count for a single array index.
    ///
    /// Mirrors the per-index logic of [`Self::wavenet_slimmable_targets`] but
    /// performs no heap allocation, making it suitable for the RT path.
    fn wavenet_scalar_target(&self, index: usize) -> Option<usize> {
        if self.wavenet_mono_rt {
            if index != 0 {
                return None;
            }
            let orig = self.wavenet_full_ch?;
            if !self.wavenet_slimmable_capable {
                return Some(orig);
            }
            let ratio = self.wavenet_slimmable_ratio();
            let t = ((orig as f32) * ratio).round() as usize;
            return Some(t.clamp(1, orig));
        }
        if self.wavenet_slimmable_capable {
            let ratio = self.wavenet_slimmable_ratio();
            if self.wavenet_allowed_channels_active
                && let Some(per_array) = self.wavenet_allowed_channels.as_ref()
            {
                let allowed = per_array.get(index)?;
                if allowed.is_empty() {
                    return Some(
                        self.wavenet_array_channels
                            .get(index)
                            .copied()
                            .unwrap_or_else(|| self.wavenet_full_ch.unwrap_or(1)),
                    );
                }
                let idx =
                    crate::models::slimmable::slimmable_tier_index(ratio as f64, allowed.len());
                return allowed.get(idx).copied();
            }
            if let Some(&orig) = self.wavenet_array_channels.get(index) {
                let t = ((orig as f32) * ratio).round() as usize;
                return Some(t.clamp(1, orig));
            }
            if index == 0 {
                let orig = self.wavenet_full_ch?;
                let t = ((orig as f32) * ratio).round() as usize;
                return Some(t.clamp(1, orig));
            }
            return None;
        }
        if self.wavenet_array_channels.is_empty() {
            return if index == 0 {
                self.wavenet_full_ch
            } else {
                None
            };
        }
        self.wavenet_array_channels.get(index).copied()
    }

    /// Whether the currently applied targets already equal the requested ones.
    #[inline]
    fn wavenet_current_matches_targets(&self) -> bool {
        if self.wavenet_mono_rt {
            return self.wavenet_rt_target_ch == self.wavenet_scalar_target(0);
        }
        let len = self.wavenet_targets_len();
        if len != self.wavenet_targets_current.len() {
            return false;
        }
        for (i, &applied) in self.wavenet_targets_current.iter().enumerate() {
            match self.wavenet_scalar_target(i) {
                Some(target) if target == applied => {}
                _ => return false,
            }
        }
        true
    }

    /// Records the requested targets as applied, reusing the preallocated
    /// `wavenet_targets_current` buffer (no allocation, no deallocation).
    #[inline]
    fn store_targets_current_in_place(&mut self) {
        if self.wavenet_mono_rt {
            self.wavenet_rt_target_ch = self.wavenet_scalar_target(0);
            return;
        }
        let len = self.wavenet_targets_len();
        if self.wavenet_targets_current.capacity() < len {
            // Growing here would allocate on the RT thread. Registrations keep
            // the capacity sized for their array count, so this is a contract
            // violation rather than a runtime condition.
            debug_assert!(
                false,
                "RT slimmable confirm requires preallocated target capacity"
            );
            return;
        }
        self.wavenet_targets_current.clear();
        let mut written = 0usize;
        // SAFETY: `capacity >= len` was checked above, so there is room for
        // exactly `len` initialized `usize` values. `wavenet_scalar_target`
        // reads only the channel metadata fields, never `wavenet_targets_current`,
        // so the raw writes cannot alias an active read. The length is updated
        // only after the written prefix is initialized; a `None` target leaves
        // the already-written prefix valid (mirroring the previous `push` loop).
        unsafe {
            let ptr = self.wavenet_targets_current.as_mut_ptr();
            for i in 0..len {
                let Some(target) = self.wavenet_scalar_target(i) else {
                    break;
                };
                ptr.add(i).write(target);
                written += 1;
            }
            self.wavenet_targets_current.set_len(written);
        }
    }

    /// Peeks the first-array slimmable rebuild target without allocating.
    ///
    /// Scalar counterpart of [`Self::peek_slimmable_rebuild`] for the
    /// mono/uniform-channel fast path. Returns `Some(ch)` only when the FSM or
    /// override demands a different width than the applied one. RT-safe.
    #[inline]
    pub fn peek_slimmable_ch_rebuild(&self) -> Option<usize> {
        if !self.wavenet_slimmable_capable || self.wavenet_current_matches_targets() {
            return None;
        }
        self.wavenet_scalar_target(0)
    }

    /// Commits a successful first-array rebuild without allocating.
    ///
    /// Scalar counterpart of [`Self::take_slimmable_rebuild`] for the
    /// mono/uniform-channel fast path. RT-safe.
    #[inline]
    pub fn take_slimmable_ch_rebuild(&mut self) -> Option<usize> {
        let target = self.peek_slimmable_ch_rebuild()?;
        self.store_targets_current_in_place();
        Some(target)
    }

    /// Records the full-quality channel count of the currently loaded WaveNet model
    /// and whether it explicitly declares slimmability capability.
    ///
    /// Conventional WaveNet models without slimmability metadata pass `capable = false`,
    /// preventing `take_slimmable_rebuild` from altering their array widths.
    ///
    /// Off-RT only: allocates/frees the tracking buffers.
    pub fn set_wavenet_full_ch(&mut self, ch: usize, capable: bool) {
        self.wavenet_full_ch = Some(ch);
        self.wavenet_array_channels = vec![ch];
        self.wavenet_allowed_channels = None;
        self.wavenet_allowed_channels_active = false;
        self.wavenet_targets_current = if capable { vec![ch] } else { Vec::new() };
        self.wavenet_mono_rt = false;
        self.wavenet_rt_target_ch = None;
        self.wavenet_slimmable_capable = capable;
    }

    /// RT-safe counterpart of [`Self::set_wavenet_full_ch`] for the
    /// mono/uniform-channel fast path.
    ///
    /// Stores the channel count inline (no `Vec` growth) and invalidates the
    /// inactive `allowed_channels` metadata by flag, leaving its buffers to be
    /// reclaimed by the next off-RT registration. The vector bookkeeping is
    /// bypassed until then, so this call performs zero heap allocation and zero
    /// deallocation.
    #[inline]
    pub fn set_wavenet_full_ch_rt(&mut self, ch: usize, capable: bool) {
        self.wavenet_full_ch = Some(ch);
        self.wavenet_mono_rt = true;
        self.wavenet_rt_target_ch = if capable { Some(ch) } else { None };
        self.wavenet_allowed_channels_active = false;
        self.wavenet_slimmable_capable = capable;
    }

    /// Forgets the WaveNet topology used for slimmable rebuilds.
    ///
    /// Installed when the active model is not a dynamically slimmable WaveNet
    /// (e.g. LSTM, a fixed preset WaveNet, or no model at all): a stale
    /// topology would otherwise let the FSM publish channel requests derived
    /// from a model that is no longer loaded.
    ///
    /// RT-safe: clears the inline scalar state and deactivates the vector
    /// bookkeeping by flag. The retained `Vec` buffers are reclaimed by the
    /// next off-RT registration, so this call performs zero heap allocation and
    /// zero deallocation.
    #[inline]
    pub fn clear_wavenet_registration_rt(&mut self) {
        self.wavenet_full_ch = None;
        self.wavenet_targets_current.clear();
        self.wavenet_mono_rt = true;
        self.wavenet_rt_target_ch = None;
        self.wavenet_allowed_channels_active = false;
        self.wavenet_slimmable_capable = false;
    }

    /// Registers full WaveNet topology and allowed_channels metadata.
    ///
    /// Off-RT only: takes and frees the metadata buffers.
    pub fn set_wavenet_slimmable_info(
        &mut self,
        array_channels: Vec<usize>,
        allowed_channels: Option<Vec<Vec<usize>>>,
        capable: bool,
    ) {
        self.wavenet_full_ch = array_channels.first().copied();
        let mut array_channels = array_channels;
        if array_channels.is_empty() {
            array_channels.reserve(1);
        }
        let mut targets = Vec::with_capacity(array_channels.len().max(1));
        if capable {
            targets.extend_from_slice(&array_channels);
        }
        self.wavenet_array_channels = array_channels;
        self.wavenet_allowed_channels = allowed_channels;
        self.wavenet_allowed_channels_active = self.wavenet_allowed_channels.is_some();
        self.wavenet_targets_current = targets;
        self.wavenet_mono_rt = false;
        self.wavenet_rt_target_ch = None;
        self.wavenet_slimmable_capable = capable;
    }

    /// Registers WaveNet topology directly from a loaded [`crate::models::wavenet::WaveNetModelDyn`].
    pub fn set_wavenet_from_model(&mut self, model: &crate::models::wavenet::WaveNetModelDyn) {
        let array_channels = model.arrays.iter().map(|a| a.ch).collect();
        self.set_wavenet_slimmable_info(
            array_channels,
            model.allowed_channels.clone(),
            model.slimmable_capable,
        );
    }

    /// Peeks the WaveNet slimmable rebuild targets without consuming the
    /// request: returns `Some(targets)` when the FSM/override demands
    /// different channel counts than what's currently applied.
    ///
    /// Transactional contract: pair with
    /// [`confirm_slimmable_rebuild`](Self::confirm_slimmable_rebuild) after a
    /// successful rebuild. If the rebuild fails, skip the confirm and the
    /// same targets are offered again on the next peek — the request is only
    /// consumed on success. Must be called off-RT.
    pub fn peek_slimmable_rebuild(&self) -> Option<Vec<usize>> {
        if !self.wavenet_slimmable_capable {
            return None;
        }
        let targets = self.wavenet_slimmable_targets()?;
        let pending = if self.wavenet_mono_rt {
            !self.wavenet_current_matches_targets()
        } else {
            self.wavenet_targets_current != targets
        };
        if pending { Some(targets) } else { None }
    }

    /// Commits a successfully applied rebuild: records `applied` as the
    /// current channel counts so subsequent peeks report no pending work.
    /// Call only after the rebuild actually succeeded.
    ///
    /// Off-RT only: allocates the new tracking buffer.
    pub fn confirm_slimmable_rebuild(&mut self, applied: &[usize]) {
        if self.wavenet_mono_rt {
            self.wavenet_rt_target_ch = applied.first().copied();
            return;
        }
        let mut targets = Vec::with_capacity(applied.len().max(1));
        targets.extend_from_slice(applied);
        self.wavenet_targets_current = targets;
    }

    /// Returns `Some(targets)` if a WaveNet slimmable rebuild is needed,
    /// i.e., the FSM/override demands different channel counts than what's
    /// currently loaded. Consumes the request immediately (marks the targets
    /// as current even if the caller later fails to rebuild).
    ///
    /// Legacy convenience for tests and non-transactional callers. Production
    /// consumers that can fail the rebuild must use
    /// [`peek_slimmable_rebuild`](Self::peek_slimmable_rebuild) +
    /// [`confirm_slimmable_rebuild`](Self::confirm_slimmable_rebuild) so a
    /// failed rebuild retries instead of being silently dropped.
    /// Must be called off-RT (e.g., in command drain before DSP).
    pub fn take_slimmable_rebuild(&mut self) -> Option<Vec<usize>> {
        let targets = self.peek_slimmable_rebuild()?;
        self.confirm_slimmable_rebuild(&targets);
        Some(targets)
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

    /// Returns `(t_start, t_end)` for per-sample crossfade ramp interpolation
    /// over `frame_advance` host-rate samples without mutating the crossfade clock.
    ///
    /// During an active crossfade, `t_end` equals the `t_start` of the next
    /// block when `crossfade_multiplier` advances the clock by `frame_advance`
    /// at the end of the output stage.
    #[inline(always)]
    pub fn crossfade_ramp_bounds(&self, frame_advance: usize) -> (f32, f32) {
        match self.crossfade {
            CrossfadePhase::Idle => {
                if self.state() == AdaptiveState::Full {
                    (0.0, 0.0)
                } else {
                    (1.0, 1.0)
                }
            }
            CrossfadePhase::Active => {
                let total = self.crossfade_total;
                if total == 0 {
                    (1.0, 1.0)
                } else {
                    let t_start = (self.crossfade_elapsed as f32 / total as f32).min(1.0);
                    let t_end = (self
                        .crossfade_elapsed
                        .saturating_add(frame_advance)
                        .min(total) as f32
                        / total as f32)
                        .min(1.0);
                    (t_start, t_end)
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
