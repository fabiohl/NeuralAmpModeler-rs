// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Gate logic with Dynamic Hysteresis for DSP optimizations.
//!
//! This module implements a Finite State Machine (FSM) to detect
//! silence or mono signal with temporal and amplitude hysteresis (Schmitt Trigger).
//! The goal is to prevent "chattering" (fast state oscillation) and audible
//! artifacts (clicks/zipper noise) when switching between processing modes.

use crate::math::common::SimdMath;

/// Configuration parameters for the Gate and Hysteresis logic.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(align(128))] // Avoids false sharing on the SPSC channel
pub struct GateParams {
    /// Threshold to open the gate (transition from Silence to Signal), in dB.
    pub threshold_open_db: f32,
    /// Threshold to close the gate (transition from Signal to Silence), in dB.
    pub threshold_close_db: f32,
    /// Number of frames (samples) to wait in silence before closing the gate.
    pub hold_frames: usize,
    /// Number of frames (samples) to perform the fade-in/fade-out (smoothing).
    pub fade_frames: usize,
    /// Pre-computed: `1.0 / fade_frames as f32` to avoid division in the hotpath.
    pub inv_fade_frames: f32,
    /// Absolute tolerance between L/R channels for Mono signal detection.
    pub mono_epsilon: f32,
}

impl GateParams {
    /// Creates new gate parameters by computing `inv_fade_frames = 1.0 / fade_frames`.
    pub fn new(
        threshold_open_db: f32,
        threshold_close_db: f32,
        hold_frames: usize,
        fade_frames: usize,
        mono_epsilon: f32,
    ) -> Self {
        let div = fade_frames.max(1) as f32;
        Self {
            threshold_open_db,
            threshold_close_db,
            hold_frames,
            fade_frames,
            inv_fade_frames: 1.0 / div,
            mono_epsilon,
        }
    }
}

impl Default for GateParams {
    fn default() -> Self {
        Self::new(-70.0, -80.0, 2048, 256, 1e-4)
    }
}

/// Represents the possible states of the gate with hysteresis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateState {
    /// Gate open: normal processing.
    Open,
    /// Starting transition to closed (fade-out).
    FadingOut,
    /// Gate closed: full bypass/silence.
    Closed,
    /// Starting transition to open (fade-in).
    FadingIn,
}

/// Dynamic Hysteresis FSM Implementation (Schmitt Trigger).
pub struct DynamicHysteresis {
    state: GateState,
    hold_counter: usize,
    fade_counter: usize,
    current_multiplier: f32,
    ramp_start_multiplier: f32,
    ramp_samples: usize,
}

impl Default for DynamicHysteresis {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicHysteresis {
    /// Creates a new instance in the initial Open state.
    #[cold]
    pub fn new() -> Self {
        Self {
            state: GateState::Open,
            hold_counter: 0,
            fade_counter: 0,
            current_multiplier: 1.0,
            ramp_start_multiplier: 1.0,
            ramp_samples: 0,
        }
    }

    /// Returns the current state of the FSM.
    pub fn state(&self) -> GateState {
        self.state
    }

    /// Returns the current gain multiplier (0.0 to 1.0).
    pub fn multiplier(&self) -> f32 {
        self.current_multiplier
    }

    /// Returns `true` when the gate is in a steady state (no active fade ramp).
    #[inline]
    pub fn is_steady(&self) -> bool {
        self.ramp_samples == 0
    }

    /// Decides whether the noise gate should open, close, or remain as is,
    /// based on the current audio volume.
    ///
    /// # Parameters
    /// - `value`: The currently detected volume.
    /// - `threshold_open`: The volume threshold to "open" the gate.
    /// - `threshold_close`: The volume below which the gate should start to "close".
    /// - `params`: Timing settings (how long to wait and how slow to close).
    /// - `n_samples`: Number of sound samples being processed now.
    pub fn update(
        &mut self,
        value: f32,
        threshold_open: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        self.ramp_start_multiplier = self.current_multiplier;
        match self.state {
            GateState::Open => self.update_open(value, threshold_close, params, n_samples),
            GateState::FadingOut => {
                self.update_fading_out(value, threshold_open, params, n_samples)
            }
            GateState::Closed => self.update_closed(value, threshold_open, params, n_samples),
            GateState::FadingIn => self.update_fading_in(value, threshold_close, params, n_samples),
        }
    }

    fn update_open(
        &mut self,
        value: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value < threshold_close {
            self.hold_counter += n_samples;
            if self.hold_counter >= params.hold_frames {
                if params.fade_frames == 0 {
                    self.state = GateState::Closed;
                    self.current_multiplier = 0.0;
                    self.fade_counter = 0;
                    self.ramp_samples = 0;
                } else {
                    self.state = GateState::FadingOut;
                    self.fade_counter = params.fade_frames;
                    self.ramp_samples = 0;
                }
            } else {
                self.ramp_samples = 0;
            }
        } else {
            self.hold_counter = 0;
            self.ramp_samples = 0;
        }
    }

    fn update_fading_out(
        &mut self,
        value: f32,
        threshold_open: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value >= threshold_open {
            self.state = GateState::FadingIn;
            if self.fade_counter + n_samples < params.fade_frames {
                self.fade_counter += n_samples;
                self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
                self.ramp_samples = n_samples;
            } else {
                self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
                self.state = GateState::Open;
                self.current_multiplier = 1.0;
                self.fade_counter = params.fade_frames;
                self.hold_counter = 0;
            }
        } else if self.fade_counter > n_samples {
            self.fade_counter -= n_samples;
            self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
            self.ramp_samples = n_samples;
        } else {
            self.ramp_samples = self.fade_counter;
            self.state = GateState::Closed;
            self.current_multiplier = 0.0;
            self.fade_counter = 0;
        }
    }

    fn update_closed(
        &mut self,
        value: f32,
        threshold_open: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value >= threshold_open {
            if params.fade_frames == 0 {
                self.state = GateState::Open;
                self.current_multiplier = 1.0;
                self.fade_counter = 0;
                self.hold_counter = 0;
                self.ramp_samples = 0;
            } else {
                self.state = GateState::FadingIn;
                self.fade_counter = 0;
                self.hold_counter = 0;
                if n_samples < params.fade_frames {
                    self.fade_counter += n_samples;
                    self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
                    self.ramp_samples = n_samples;
                } else {
                    self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
                    self.state = GateState::Open;
                    self.current_multiplier = 1.0;
                    self.fade_counter = params.fade_frames;
                }
            }
        } else {
            self.ramp_samples = 0;
        }
    }

    fn update_fading_in(
        &mut self,
        value: f32,
        threshold_close: f32,
        params: &GateParams,
        n_samples: usize,
    ) {
        if value < threshold_close {
            self.state = GateState::FadingOut;
            if self.fade_counter > n_samples {
                self.fade_counter -= n_samples;
                self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
                self.ramp_samples = n_samples;
            } else {
                self.ramp_samples = self.fade_counter;
                self.state = GateState::Closed;
                self.current_multiplier = 0.0;
                self.fade_counter = 0;
            }
        } else if self.fade_counter + n_samples < params.fade_frames {
            self.fade_counter += n_samples;
            self.current_multiplier = self.fade_counter as f32 * params.inv_fade_frames;
            self.ramp_samples = n_samples;
        } else {
            self.ramp_samples = params.fade_frames.saturating_sub(self.fade_counter);
            self.state = GateState::Open;
            self.current_multiplier = 1.0;
            self.fade_counter = params.fade_frames;
            self.hold_counter = 0;
        }
    }

    /// Applies the current volume (gain) to the sound.
    /// If the gate is opening or closing, it performs a smooth change (ramp).
    /// If it is fully open or closed, it applies a constant volume.
    pub fn apply_gain_rt(&self, buffer: &mut [f32], n_samples: usize) {
        if self.ramp_samples == 0 {
            // Volume is stable (not in the middle of a fade).
            if self.current_multiplier == 0.0 {
                // Total silence.
                buffer.fill(0.0);
            } else if (self.current_multiplier - 1.0).abs() > 1e-6 {
                // Applies a constant volume (e.g., 50%).
                crate::math::dsp::gain::apply_gain_simd(buffer, self.current_multiplier);
            }
            // If volume is 1.0 (100%), nothing needs to be done.
            return;
        }

        let start_mult = self.ramp_start_multiplier;
        let end_mult = self.current_multiplier;

        if self.ramp_samples >= n_samples {
            // The smooth volume change will span the entire block.
            if (start_mult - end_mult).abs() < 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(buffer, end_mult);
            } else {
                // Computes the volume "step" for each sound sample.
                // NOTE: If n_samples = 1, step = (end - start) / 1.0.
                // The resulting value will be applied to the single sample, which is the
                // expected behavior for sample-accurate transitions in host block subdivision.
                let step = (end_mult - start_mult) / (n_samples as f32);
                crate::math::dsp::gain::apply_ramp_simd(buffer, start_mult, step);
            }
        } else {
            // Special case: the volume change finishes before the end of the block.
            // We split the block into two parts: the ramp and the final constant volume.
            let (ramp_part, const_part) = buffer.split_at_mut(self.ramp_samples);

            if (start_mult - end_mult).abs() < 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(ramp_part, end_mult);
            } else {
                let step = (end_mult - start_mult) / (self.ramp_samples as f32);
                crate::math::dsp::gain::apply_ramp_simd(ramp_part, start_mult, step);
            }

            // Fills the rest of the block with the stabilized final volume.
            if end_mult == 0.0 {
                const_part.fill(0.0);
            } else if (end_mult - 1.0).abs() > 1e-6 {
                crate::math::dsp::gain::apply_gain_simd(const_part, end_mult);
            }
        }
    }

    /// Does the same as the function above, but for stereo sound (left and right channels).
    /// Processing is done jointly for improved speed.
    pub fn apply_gain_rt_stereo<M: SimdMath>(
        &self,
        left: &mut [f32],
        right: &mut [f32],
        n_samples: usize,
    ) {
        if self.ramp_samples == 0 {
            // Stable volume for both channels.
            if self.current_multiplier == 0.0 {
                left.fill(0.0);
                right.fill(0.0);
            } else if (self.current_multiplier - 1.0).abs() > 1e-6 {
                // SAFETY: `M::apply_gain_stereo` operates on mutable f32
                // slices `left` and `right` of equal length (n_samples).
                // The SIMD kernel only reads the multiplier and writes
                // in-place within slice bounds.
                unsafe { M::apply_gain_stereo(left, right, self.current_multiplier) };
            }
            return;
        }

        let start_mult = self.ramp_start_multiplier;
        let end_mult = self.current_multiplier;

        if self.ramp_samples >= n_samples {
            // Smooth change across both channels for the entire block.
            if (start_mult - end_mult).abs() < 1e-6 {
                // SAFETY: same invariants as apply_gain_stereo above.
                unsafe { M::apply_gain_stereo(left, right, end_mult) };
            } else {
                // NOTE: With n_samples = 1, the 1-sample "ramp"
                // results in a direct jump to the target value, which is accepted by design.
                let step = (end_mult - start_mult) / (n_samples as f32);
                // SAFETY: `M::apply_ramp_stereo` operates on mutable f32
                // slices `left` and `right` of equal length (n_samples).
                // The SIMD kernel reads start_mult and step, writes
                // in-place within slice bounds.
                unsafe { M::apply_ramp_stereo(left, right, start_mult, step) };
            }
        } else {
            // Smooth change finishes before the end of the block for both channels.
            let (ramp_l, const_l) = left.split_at_mut(self.ramp_samples);
            let (ramp_r, const_r) = right.split_at_mut(self.ramp_samples);

            if (start_mult - end_mult).abs() < 1e-6 {
                // SAFETY: ramp_l and ramp_r have equal length (self.ramp_samples).
                // Same SIMD invariants as the global-ramp path.
                unsafe { M::apply_gain_stereo(ramp_l, ramp_r, end_mult) };
            } else {
                let step = (end_mult - start_mult) / (self.ramp_samples as f32);
                // SAFETY: ramp_l and ramp_r have equal length (self.ramp_samples).
                // Same SIMD invariants as apply_ramp_stereo above.
                unsafe { M::apply_ramp_stereo(ramp_l, ramp_r, start_mult, step) };
            }

            // Finalizes the rest of the block with stable volume.
            if end_mult == 0.0 {
                const_l.fill(0.0);
                const_r.fill(0.0);
            } else if (end_mult - 1.0).abs() > 1e-6 {
                // SAFETY: const_l and const_r have equal length (the
                // remainder after splitting off ramp_samples). Same
                // SIMD invariants as apply_gain_stereo.
                unsafe { M::apply_gain_stereo(const_l, const_r, end_mult) };
            }
        }
    }
}

#[cfg(test)]
#[path = "gate_test.rs"]
mod gate_test;
