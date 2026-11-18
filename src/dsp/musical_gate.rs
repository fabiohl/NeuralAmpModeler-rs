// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Reference-compatible musical noise gate (quadratic downward expander).
//!
//! Follows the DSP equations and timing constants of `AudioDSPTools/dsp/NoiseGate`:
//! - RMS/power envelope time constant: `10 ms` (`alpha = 0.5^(1 / (0.01 * fs))`)
//! - Quadratic gain-reduction slope: `ratio = 0.1` (`-0.1 * (level_db - threshold)^2`)
//! - Attack/open slew time: `5 ms` (across `[max_gr, 0.0]`)
//! - Hold time above/before closing: `10 ms`
//! - Release/close slew time: `50 ms` (across `[0.0, max_gr]`)
//! - Loudness floor: `-120.0 dB` (`1e-12` power)
//!
//! Real-time constraints:
//! - Per-channel state and gain scratch (`Box<[f32; MAX_RESAMP_BUF]>`) are pre-allocated off-RT.
//! - Sample-rate coefficients (`alpha`, `beta`, `dt`, inverse ramp times) are computed only
//!   inside `MusicalNoiseGate::prepare` off the per-sample hot path using pure IEEE-754 math.
//! - Per-sample `log10` and `10^x` evaluations use exact-range IEEE-754 bit reduction plus
//!   branchless Horner polynomials with zero `libm` calls, zero integer division, and zero locks.

use crate::dsp::pipeline::MAX_RESAMP_BUF;

/// Loudness envelope time constant in seconds (`NoiseGate.cpp`: `time = 0.01`).
pub const MUSICAL_GATE_TIME_SEC: f64 = 0.01;
/// Quadratic gain-reduction slope factor (`NoiseGate.cpp`: `ratio = 0.1`).
pub const MUSICAL_GATE_RATIO: f64 = 0.1;
/// Open/attack slew time in seconds (`NoiseGate.cpp`: `openTime = 0.005`).
pub const MUSICAL_GATE_OPEN_TIME_SEC: f64 = 0.005;
/// Hold duration in seconds (`NoiseGate.cpp`: `holdTime = 0.01`).
pub const MUSICAL_GATE_HOLD_TIME_SEC: f64 = 0.01;
/// Close/release slew time in seconds (`NoiseGate.cpp`: `closeTime = 0.05`).
pub const MUSICAL_GATE_CLOSE_TIME_SEC: f64 = 0.05;
/// Minimum tracked loudness in dB power (`NoiseGate.h`: `-120.0`).
pub const MINIMUM_LOUDNESS_DB: f64 = -120.0;
/// Minimum tracked loudness power (`10^(-120/10)`).
pub const MINIMUM_LOUDNESS_POWER: f64 = 1e-12;
/// Maximum tracked loudness power (`NoiseGate.cpp`: clamped to `1000.0`).
pub const MAXIMUM_LOUDNESS_POWER: f64 = 1000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriggerState {
    Moving,
    Holding,
}

#[derive(Debug, Clone, Copy)]
struct MusicalGateChannel {
    state: TriggerState,
    level: f64,
    last_gain_reduction_db: f64,
    time_held: f64,
}

impl MusicalGateChannel {
    fn reset(&mut self, initial_gr_db: f64) {
        self.state = TriggerState::Moving;
        self.level = MINIMUM_LOUDNESS_POWER;
        self.last_gain_reduction_db = initial_gr_db;
        self.time_held = 0.0;
    }
}

/// Computes `log10(x)` for positive finite `x in [1e-12, 1000.0]` with zero `libm` calls.
///
/// Reduces `x = 2^e * m` into `m in [sqrt(0.5), sqrt(2))` using IEEE-754 exponent
/// extraction and evaluates `ln(m) = 2 * z * (1 + z^2/3 + z^4/5 + z^6/7 + z^8/9)` where
/// `z = (m - 1) / (m + 1)`, `|z| <= 0.171573` (max error `< 1e-9`).
#[inline(always)]
fn fast_log10_pos_f64(x: f64) -> f64 {
    const SQRT_2: f64 = core::f64::consts::SQRT_2;
    const LN_2: f64 = core::f64::consts::LN_2;
    const LOG10_E: f64 = core::f64::consts::LOG10_E;

    let bits = x.to_bits();
    let mut e = ((bits >> 52) as i64) - 1023;
    let mut m = f64::from_bits((bits & 0x000f_ffff_ffff_ffff_u64) | (1023_u64 << 52));
    if m >= SQRT_2 {
        m *= 0.5;
        e += 1;
    }
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let poly =
        1.0 + z2 * ((1.0 / 3.0) + z2 * ((1.0 / 5.0) + z2 * ((1.0 / 7.0) + z2 * (1.0 / 9.0))));
    let ln_m = 2.0 * z * poly;
    ((e as f64) * LN_2 + ln_m) * LOG10_E
}

/// Computes `10^x` for `x <= 0.0` with zero `libm` calls.
///
/// Rewrites `10^x = 2^(k + f)` (`|f| <= 0.5`), reconstructs `2^k` via IEEE-754 exponent
/// shifting, and evaluates `e^(f * ln(2))` with a 7th-order Taylor polynomial (error `< 3e-10`).
#[inline(always)]
fn fast_pow10_nonpos_f64(x: f64) -> f64 {
    const LOG2_10: f64 = core::f64::consts::LOG2_10;
    const LN_2: f64 = core::f64::consts::LN_2;

    if x >= 0.0 {
        return 1.0;
    }
    let p = x * LOG2_10;
    if p <= -1021.0 {
        return 0.0;
    }
    // For p < 0, `(p - 0.5) as i64` yields the nearest integer k with p - k in [-0.5, 0.5].
    let k = (p - 0.5) as i64;
    let f = p - (k as f64);
    let u = f * LN_2;
    let exp_u = 1.0
        + u * (1.0
            + u * (0.5
                + u * ((1.0 / 6.0)
                    + u * ((1.0 / 24.0)
                        + u * ((1.0 / 120.0) + u * ((1.0 / 720.0) + u * (1.0 / 5040.0)))))));
    let scale_bits = ((k + 1023) as u64) << 52;
    exp_u * f64::from_bits(scale_bits)
}

#[inline(always)]
fn gain_reduction_db(level_db: f64, threshold_db: f64) -> f64 {
    if level_db < threshold_db {
        let diff = level_db - threshold_db;
        -MUSICAL_GATE_RATIO * diff * diff
    } else {
        0.0
    }
}

/// Dual-channel musical noise gate (quadratic downward expander) with pre-allocated scratch.
pub struct MusicalNoiseGate {
    sample_rate: u32,
    alpha: f64,
    beta: f64,
    dt: f64,
    inv_open_samples: f64,
    inv_close_samples: f64,
    threshold_db: f64,
    max_gain_reduction_db: f64,
    d_open: f64,
    d_close: f64,
    enabled: bool,
    channels: [MusicalGateChannel; 2],
    gains_l: Box<[f32; MAX_RESAMP_BUF]>,
    gains_r: Box<[f32; MAX_RESAMP_BUF]>,
}

impl MusicalNoiseGate {
    /// Allocates per-channel state and gain buffers off-RT for `sample_rate`.
    pub fn new(sample_rate: u32) -> Self {
        let mut gate = Self {
            sample_rate: 0,
            alpha: 0.0,
            beta: 1.0,
            dt: 0.0,
            inv_open_samples: 0.0,
            inv_close_samples: 0.0,
            threshold_db: -90.0,
            max_gain_reduction_db: 0.0,
            d_open: 0.0,
            d_close: 0.0,
            enabled: false,
            channels: [
                MusicalGateChannel {
                    state: TriggerState::Moving,
                    level: MINIMUM_LOUDNESS_POWER,
                    last_gain_reduction_db: 0.0,
                    time_held: 0.0,
                },
                MusicalGateChannel {
                    state: TriggerState::Moving,
                    level: MINIMUM_LOUDNESS_POWER,
                    last_gain_reduction_db: 0.0,
                    time_held: 0.0,
                },
            ],
            gains_l: Box::new([1.0f32; MAX_RESAMP_BUF]),
            gains_r: Box::new([1.0f32; MAX_RESAMP_BUF]),
        };
        gate.prepare(sample_rate);
        gate
    }

    /// Recomputes sample-rate-dependent constants off the per-sample loop and resets state.
    ///
    /// Uses `fast_pow10_nonpos_f64` instead of `f64::powf` so neither `.text` nor `.text.cold`
    /// pulls in `libm` symbols under RT codegen verification.
    #[cold]
    #[inline(never)]
    pub fn prepare(&mut self, sample_rate: u32) {
        let sr_u32 = sample_rate.max(1);
        let fs = sr_u32 as f64;
        self.sample_rate = sr_u32;
        self.alpha =
            fast_pow10_nonpos_f64(-core::f64::consts::LOG10_2 / (MUSICAL_GATE_TIME_SEC * fs));
        self.beta = 1.0 - self.alpha;
        self.dt = 1.0 / fs;
        self.inv_open_samples = 1.0 / (MUSICAL_GATE_OPEN_TIME_SEC * fs);
        self.inv_close_samples = 1.0 / (MUSICAL_GATE_CLOSE_TIME_SEC * fs);
        self.recompute_threshold_slopes();
        self.reset();
    }

    /// Re-prepares coefficients only when `sample_rate` changes.
    #[inline]
    pub fn ensure_sample_rate(&mut self, sample_rate: u32) {
        let sr_u32 = sample_rate.max(1);
        if sr_u32 != self.sample_rate {
            self.prepare(sr_u32);
        }
    }

    /// Returns whether the expander is actively enabled.
    #[inline]
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the configured open/knee threshold in dB.
    #[inline]
    pub const fn threshold_db(&self) -> f64 {
        self.threshold_db
    }

    /// Sets gate threshold in dB and bypass enable state using arithmetic-only coefficient math.
    #[inline]
    pub fn set_threshold(&mut self, threshold_db: f64, enabled: bool) {
        let clamped = if threshold_db.is_finite() {
            threshold_db.clamp(MINIMUM_LOUDNESS_DB, 0.0)
        } else {
            MINIMUM_LOUDNESS_DB
        };
        let was_enabled = self.enabled;
        self.enabled = enabled;
        if (clamped - self.threshold_db).abs() > 1e-9 || enabled != was_enabled {
            self.threshold_db = clamped;
            self.recompute_threshold_slopes();
            if !enabled {
                self.reset_open();
            } else if !was_enabled {
                self.reset();
            }
        }
    }

    #[inline]
    fn recompute_threshold_slopes(&mut self) {
        let max_gr = gain_reduction_db(MINIMUM_LOUDNESS_DB, self.threshold_db);
        self.max_gain_reduction_db = max_gr;
        self.d_open = -max_gr * self.inv_open_samples;
        self.d_close = max_gr * self.inv_close_samples;
        for ch in &mut self.channels {
            if ch.last_gain_reduction_db < max_gr {
                ch.last_gain_reduction_db = max_gr;
            }
        }
    }

    /// Resets per-channel envelope and gain-reduction state cleanly.
    pub fn reset(&mut self) {
        let init_gr = if self.enabled {
            self.max_gain_reduction_db
        } else {
            0.0
        };
        self.channels[0].reset(init_gr);
        self.channels[1].reset(init_gr);
    }

    /// Resets channels to full open (`0.0 dB`, `1.0x`) for clean bypass transitions.
    pub fn reset_open(&mut self) {
        for ch in &mut self.channels {
            ch.state = TriggerState::Moving;
            ch.level = MINIMUM_LOUDNESS_POWER;
            ch.last_gain_reduction_db = 0.0;
            ch.time_held = 0.0;
        }
    }

    /// Returns whether the musical gate is actively attenuating audio (`> 0.5 dB` reduction).
    #[inline]
    pub fn is_attenuating(&self, process_mono: bool) -> bool {
        if !self.enabled {
            return false;
        }
        let l_att = self.channels[0].last_gain_reduction_db < -0.5;
        if process_mono {
            l_att
        } else {
            l_att || self.channels[1].last_gain_reduction_db < -0.5
        }
    }

    /// Processes unity-scaled input trigger samples (pre-model / pre-calibration) into
    /// pre-allocated per-sample gain scratch buffers.
    #[inline]
    pub fn process_trigger(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        n_samples: usize,
        process_mono: bool,
    ) {
        self.process_trigger_scaled(in_l, in_r, n_samples, process_mono, 1.0);
    }

    /// Processes input trigger samples scaled by `trigger_scale` (e.g. `user_input_gain_mult`,
    /// post user input gain and pre model `cal_in`) into pre-allocated scratch buffers.
    #[inline]
    pub fn process_trigger_scaled(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        n_samples: usize,
        process_mono: bool,
        trigger_scale: f64,
    ) {
        if !self.enabled {
            return;
        }
        let scale = if trigger_scale.is_finite() {
            trigger_scale
        } else {
            0.0
        };
        let n = n_samples.min(MAX_RESAMP_BUF).min(in_l.len());
        self.process_channel_trigger(0, &in_l[..n], scale);
        if !process_mono {
            let nr = n.min(in_r.len());
            self.process_channel_trigger(1, &in_r[..nr], scale);
        }
    }

    #[inline]
    fn process_channel_trigger(&mut self, ch_idx: usize, samples: &[f32], trigger_scale: f64) {
        let alpha = self.alpha;
        let beta = self.beta;
        let dt = self.dt;
        let threshold_db = self.threshold_db;
        let max_gr = self.max_gain_reduction_db;
        let d_open = self.d_open;
        let d_close = self.d_close;

        let ch = &mut self.channels[ch_idx];
        let gains = if ch_idx == 0 {
            &mut self.gains_l[..samples.len()]
        } else {
            &mut self.gains_r[..samples.len()]
        };

        let mut level = ch.level;
        let mut last_gr = ch.last_gain_reduction_db;
        let mut time_held = ch.time_held;
        let mut state = ch.state;

        for (out_g, &s_f32) in gains.iter_mut().zip(samples.iter()) {
            let x = if s_f32.is_finite() {
                (s_f32 as f64) * trigger_scale
            } else {
                0.0
            };
            level = (alpha * level + beta * (x * x))
                .clamp(MINIMUM_LOUDNESS_POWER, MAXIMUM_LOUDNESS_POWER);
            let level_db = 10.0 * fast_log10_pos_f64(level);

            match state {
                TriggerState::Holding => {
                    last_gr = 0.0;
                    if level_db < threshold_db {
                        time_held += dt;
                        if time_held >= MUSICAL_GATE_HOLD_TIME_SEC {
                            state = TriggerState::Moving;
                        }
                    } else {
                        time_held = 0.0;
                    }
                    *out_g = 1.0;
                }
                TriggerState::Moving => {
                    let target_gr = gain_reduction_db(level_db, threshold_db);
                    if target_gr > last_gr {
                        let d_gain = (0.5 * (target_gr - last_gr)).clamp(0.0, d_open);
                        last_gr += d_gain;
                        if last_gr >= 0.0 {
                            last_gr = 0.0;
                            state = TriggerState::Holding;
                            time_held = 0.0;
                            *out_g = 1.0;
                            continue;
                        }
                    } else if target_gr < last_gr {
                        let d_gain = (0.5 * (target_gr - last_gr)).clamp(d_close, 0.0);
                        last_gr += d_gain;
                        if last_gr < max_gr {
                            last_gr = max_gr;
                        }
                    }
                    // Matches C++ `_DBToLevel(gainReductionDB) = pow(10, gainReductionDB / 10)`.
                    *out_g = fast_pow10_nonpos_f64(last_gr * 0.1) as f32;
                }
            }
        }

        ch.level = level;
        ch.last_gain_reduction_db = last_gr;
        ch.time_held = time_held;
        ch.state = state;
    }

    /// Applies precomputed per-sample expander gains to model output (post-model, pre-CabSim IR).
    #[inline]
    pub fn apply_gain(
        &self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        n_out: usize,
        process_mono: bool,
    ) {
        if !self.enabled {
            return;
        }
        let nl = n_out.min(MAX_RESAMP_BUF).min(out_l.len());
        for (y, &g) in out_l[..nl].iter_mut().zip(self.gains_l[..nl].iter()) {
            *y *= g;
        }
        if !process_mono {
            let nr = n_out.min(MAX_RESAMP_BUF).min(out_r.len());
            for (y, &g) in out_r[..nr].iter_mut().zip(self.gains_r[..nr].iter()) {
                *y *= g;
            }
        }
    }

    /// Returns a read-only view of left-channel gains computed in the last block.
    #[cfg(test)]
    pub(crate) fn last_gains_l(&self, n: usize) -> &[f32] {
        let len = n.min(MAX_RESAMP_BUF);
        &self.gains_l[..len]
    }
}

#[cfg(test)]
#[path = "musical_gate_test.rs"]
mod tests;
