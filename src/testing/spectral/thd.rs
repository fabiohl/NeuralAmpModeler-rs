// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! THD+N (AES17) and IMD (SMPTE/DIN) measurement routines.

use super::next_power_of_two;
use crate::math::dsp::fft::FftPlanner;
use std::f64::consts::TAU;

/// Result of an AES17 THD+N measurement.
#[derive(Debug, Clone)]
pub struct ThdnResult {
    /// Fundamental frequency in Hz.
    pub f0: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// THD+N in percent: 100 · RMS(notched) / RMS(total).
    pub thdn_percent: f64,
    /// THD+N in dB relative to fundamental.
    pub thdn_db: f64,
    /// RMS of the notched signal (everything except fundamental).
    pub rms_notched: f64,
    /// RMS of the total signal.
    pub rms_total: f64,
}

/// Second-order biquad notch filter coefficients.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NotchBiquad {
    pub(crate) b0: f64,
    pub(crate) b1: f64,
    pub(crate) b2: f64,
    pub(crate) a1: f64,
    pub(crate) a2: f64,
}

impl NotchBiquad {
    /// Designs a biquad notch filter.
    ///
    /// `f0` = notch frequency (Hz), `q` = quality factor (typ. 1–5 per AES17),
    /// `fs` = sample rate (Hz).
    pub(crate) fn design(f0: f64, q: f64, fs: f64) -> Self {
        let w0 = TAU * f0 / fs;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Applies the notch filter in-place to `signal`.
    pub(crate) fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0f64; signal.len()];
        let mut x1 = 0.0f64;
        let mut x2 = 0.0f64;
        let mut y1 = 0.0f64;
        let mut y2 = 0.0f64;

        for (i, &x0) in signal.iter().enumerate() {
            let y0 = self.b0 * x0 + self.b1 * x1 + self.b2 * x2 - self.a1 * y1 - self.a2 * y2;
            out[i] = y0;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
        }
        out
    }
}

/// Generates a pure sine at `f0` Hz with amplitude 1.0.
pub fn generate_sine_f64(f0: f64, sample_rate: u32, num_samples: usize, gain: f64) -> Vec<f64> {
    let sr = sample_rate as f64;
    let omega = TAU * f0 / sr;
    (0..num_samples)
        .map(|i| (i as f64 * omega).sin() * gain)
        .collect()
}

/// Measures THD+N per AES17.
///
/// 1. Generates a `f0` Hz pure tone (typ. 997 Hz per AES17).
/// 2. Processes through `process_fn`.
/// 3. Notch-filters the fundamental (Q ≈ 5).
/// 4. THD+N = 100% · RMS(notched) / RMS(total).
///
/// `stability_blocks` = number of extra blocks to discard at start to avoid
/// transient artefacts from model warm-up (0 for static functions).
pub fn measure_thdn<F>(
    f0: f64,
    sample_rate: u32,
    duration_s: f64,
    notch_q: f64,
    stability_blocks: usize,
    process_fn: F,
) -> ThdnResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let n_total = (sample_rate as f64 * duration_s).ceil() as usize + stability_blocks;
    let input = generate_sine_f64(f0, sample_rate, n_total, 1.0);

    let output_f32 = process_fn(&input);
    assert_eq!(output_f32.len(), input.len());

    // Discard stability blocks and convert to f64
    let start = stability_blocks;
    let output: Vec<f64> = output_f32[start..].iter().map(|&x| x as f64).collect();
    let n = output.len();

    // RMS of total signal
    let rms_total = {
        let sum_sq: f64 = output.iter().map(|&x| x * x).sum();
        (sum_sq / n as f64).sqrt()
    };

    // Notch filter the fundamental
    let notch = NotchBiquad::design(f0, notch_q, sample_rate as f64);
    let notched = notch.apply(&output);

    // Discard first 2000 samples to skip biquad transient
    let settle = 2000.min(n.saturating_sub(1));
    let notched_stable = &notched[settle..];

    // RMS of notched signal (stable portion)
    let rms_notched = {
        let sum_sq: f64 = notched_stable.iter().map(|&x| x * x).sum();
        (sum_sq / notched_stable.len() as f64).sqrt()
    };

    let thdn_percent = if rms_total > 1e-15 {
        100.0 * rms_notched / rms_total
    } else {
        0.0
    };

    let thdn_db = if thdn_percent > 1e-15 {
        20.0 * (thdn_percent / 100.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    ThdnResult {
        f0,
        sample_rate,
        thdn_percent,
        thdn_db,
        rms_notched,
        rms_total,
    }
}

/// Result of an IMD SMPTE/DIN measurement.
#[derive(Debug, Clone)]
pub struct SmpteImdResult {
    /// Low frequency in Hz (60 Hz per SMPTE).
    pub f_low: f64,
    /// High frequency in Hz (7 kHz per SMPTE).
    pub f_high: f64,
    /// Amplitude ratio (4:1 per SMPTE).
    pub ratio: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// IMD in percent: 100 · RMS(sidebands) / RMS(carrier).
    pub imd_percent: f64,
    /// IMD in dB: 20 · log10(imd_percent / 100).
    pub imd_db: f64,
    /// Individual sideband amplitudes (normalised to carrier).
    pub sideband_percents: Vec<(i32, f64)>,
}

/// Generates an SMPTE/DIN two-tone test signal.
///
/// `f_low` and `f_high` are the two tone frequencies.
/// `ratio` is the amplitude ratio (f_high : f_low). Standard SMPTE is 4:1.
/// `gain` scales the combined signal such that peak ≤ gain.
pub fn generate_smpte_tones(
    f_low: f64,
    f_high: f64,
    ratio: f64,
    sample_rate: u32,
    num_samples: usize,
    gain: f64,
) -> Vec<f64> {
    let sr = sample_rate as f64;
    let omega_low = TAU * f_low / sr;
    let omega_high = TAU * f_high / sr;

    let amp_high = gain * ratio / (ratio + 1.0);
    let amp_low = gain / (ratio + 1.0);

    (0..num_samples)
        .map(|i| {
            let t = i as f64;
            amp_low * (t * omega_low).sin() + amp_high * (t * omega_high).sin()
        })
        .collect()
}

/// Measures SMPTE IMD.
///
/// 1. Generates two-tone signal (60 Hz + 7 kHz, 4:1).
/// 2. Processes through `process_fn`.
/// 3. Computes FFT of output, identifies carrier and sidebands.
/// 4. IMD(%) = 100 · √(Σ|sidebands|²) / |carrier|
///
/// `stability_blocks` = number of extra blocks discarded from start.
pub fn measure_smpte_imd<F>(
    f_low: f64,
    f_high: f64,
    ratio: f64,
    sample_rate: u32,
    duration_s: f64,
    stability_blocks: usize,
    process_fn: F,
) -> SmpteImdResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let n_signal = (sample_rate as f64 * duration_s).ceil() as usize + stability_blocks;
    let input = generate_smpte_tones(f_low, f_high, ratio, sample_rate, n_signal, 0.9);

    let output_f32 = process_fn(&input);
    assert_eq!(output_f32.len(), input.len());

    let start = stability_blocks;
    let output: Vec<f64> = output_f32[start..].iter().map(|&x| x as f64).collect();
    let n = output.len();
    let n_fft = next_power_of_two(n);
    let sr = sample_rate as f64;
    let bin_width = sr / n_fft as f64;

    // Apply 4-term Blackman-Harris window
    let window: Vec<f64> = crate::testing::aliasing::blackman_harris_4term(n);
    let mut re = vec![0.0f64; n_fft];
    let mut im = vec![0.0f64; n_fft];
    for (i, &x) in output.iter().enumerate() {
        re[i] = x * window[i];
    }

    let fft = FftPlanner::<f64>::new(n_fft);
    fft.process(&mut re, &mut im);

    let mag: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| (r * r + i * i).sqrt())
        .collect();

    // Find the carrier bin (closest to f_high)
    let carrier_bin = (f_high / bin_width).round() as usize;
    let carrier_mag = if carrier_bin < mag.len() {
        mag[carrier_bin]
    } else {
        0.0
    };

    // Search for sidebands at f_high ± n·f_low
    let max_sideband_order = 6i32;
    let bin_search = (2.0 / bin_width).ceil() as usize;
    let mut sideband_percents: Vec<(i32, f64)> = Vec::new();

    for n_order in 1..=max_sideband_order {
        for &sign in &[-1i32, 1i32] {
            let expected_freq = f_high + sign as f64 * n_order as f64 * f_low;
            if expected_freq <= 0.0 || expected_freq >= (sr / 2.0 - bin_width) {
                continue;
            }

            let expected_bin = (expected_freq / bin_width).round() as usize;
            // Find the actual peak in a small range around expected_bin
            let start_bin = expected_bin.saturating_sub(bin_search);
            let end_bin = (expected_bin + bin_search).min(mag.len() - 1);

            if start_bin >= end_bin {
                continue;
            }

            let mut _peak_bin = start_bin;
            let mut peak_val = 0.0f64;
            for b in start_bin + 1..end_bin {
                if mag[b] > mag[b - 1] && mag[b] > mag[b + 1] && mag[b] > peak_val {
                    _peak_bin = b;
                    peak_val = mag[b];
                }
            }

            if peak_val > 0.0 && carrier_mag > 1e-15 {
                let pct = 100.0 * peak_val / carrier_mag;
                sideband_percents.push((n_order * sign, pct));
            }
        }
    }

    let imd_percent = if carrier_mag > 1e-15 {
        let sum_sq: f64 = sideband_percents
            .iter()
            .map(|(_, p)| (p / 100.0).powi(2))
            .sum();
        100.0 * sum_sq.sqrt()
    } else {
        0.0
    };

    let imd_db = if imd_percent > 1e-15 {
        20.0 * (imd_percent / 100.0).log10()
    } else {
        f64::NEG_INFINITY
    };

    SmpteImdResult {
        f_low,
        f_high,
        ratio,
        sample_rate,
        imd_percent,
        imd_db,
        sideband_percents,
    }
}
