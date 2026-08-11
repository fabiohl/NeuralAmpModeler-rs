// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Smoothing filter for audio parameters.
//!
//! Implements a 1-pole IIR filter (Low-pass) to avoid clicks and zipper noise
//! when changing gains during real-time processing.

/// Parameter smoother based on a 1-pole IIR filter.
/// y\[n\] = α * target + (1 - α) * y\[n-1\]
#[derive(Debug, Clone, Copy)]
pub struct ParamSmoother {
    current: f32,
    target: f32,
    alpha: f32,
}

impl ParamSmoother {
    /// Creates a new smoother with initial value and alpha coefficient.
    ///
    /// # Parameters
    /// * `initial_value`: Initial value (and initial target).
    /// * `sample_rate`: Sampling rate (fs).
    /// * `cutoff_hz`: Cutoff frequency (fc). Recommended ~20Hz for gains.
    pub fn new(initial_value: f32, sample_rate: f32, cutoff_hz: f32) -> Self {
        let alpha = if sample_rate > 0.0 {
            // α = 1 - exp(-2π * fc / fs)
            1.0 - (-(2.0 * std::f32::consts::PI * cutoff_hz) / sample_rate).exp()
        } else {
            1.0
        };

        Self {
            current: initial_value,
            target: initial_value,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Updates the target value of the parameter.
    #[inline]
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }

    /// Jumps immediately to the target value (without smoothing).
    #[inline]
    pub fn snap_to_target(&mut self) {
        self.current = self.target;
    }

    /// Advances one sample and returns the smoothed value.
    ///
    /// Called per-sample in the output gain smoothing path.
    /// Micro-opt: `#[inline]` eliminates the function-call
    /// overhead for this hot-path 1-pole IIR tick.
    #[inline]
    pub fn tick(&mut self) -> f32 {
        let diff = self.current - self.target;
        // Threshold proportional to the target: 2-5x faster convergence for higher values.
        let threshold = 1e-6 * self.target.abs().max(1.0);
        if diff.abs() < threshold {
            self.current = self.target;
        } else {
            let next = self.alpha * self.target + (1.0 - self.alpha) * self.current;
            if next == self.current {
                // Precision stall detection in f32: if the step is smaller than
                // the smallest representable variation, forces a snap to the target.
                self.current = self.target;
            } else {
                self.current = next;
                // Fade-to-zero guard (RT-Safety §2.1).
                //
                // With DAZ/FTZ active in MXCSR (set at boot and periodically
                // reaffirmed by the host/runtime processor), actual f32 subnormals
                // (abs < ~1.18e-38) are never created — FPU hardware flushes
                // them to zero automatically.  Therefore this check is not
                // about denormal protection (which DAZ/FTZ already provides).
                //
                // The threshold 1e-15 is ~17 orders of magnitude above the
                // subnormal boundary.  It serves a *sonic* purpose: kill the
                // inaudible tail of the smoother to prevent a theoretically
                // infinite decay when values get so small they no longer
                // produce any audible output (< -300 dBFS).
                if self.current.abs() < 1e-15 {
                    self.current = 0.0;
                }
            }
        }
        self.current
    }

    /// Returns the current value (last computed).
    #[inline]
    pub fn current_value(&self) -> f32 {
        self.current
    }

    /// Returns the target value.
    #[inline]
    pub fn target_value(&self) -> f32 {
        self.target
    }

    /// Returns the current value (peek).
    #[inline]
    pub fn peek(&self) -> f32 {
        self.current
    }

    /// Returns the IIR coefficient α.
    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Sets the current value.
    #[inline]
    pub fn set(&mut self, val: f32) {
        self.current = val;
    }
}

#[cfg(test)]
#[path = "smoother_test.rs"]
mod tests;
