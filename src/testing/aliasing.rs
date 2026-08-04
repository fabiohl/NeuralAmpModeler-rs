// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Aliasing-to-Signal Ratio (ASR) metric — Sato & Smith, DAFx 2025.
//!
//! Measures the energy ratio of aliased (non-harmonic) components to harmonic
//! components when a pure sine wave is processed through a nonlinear system.
//!
//! ```text
//! ASR = Σ E_aliased / Σ E_harmonic
//! ```
//!
//! Low ASR → good anti-aliasing; high ASR → significant fold-back distortion.
//! Used to fingerprint the aliasing behaviour of neural amp models and to gate
//! regressions in anti-aliasing quality.

use crate::math::dsp::fft::RfftPlanner;

// =============================================================================
// Musical pitch table — standard scientific pitch notation (A4 = 440 Hz)
// =============================================================================

/// Musical pitch table from E2 to E5 plus stress tone ≥ 2 kHz.
pub const MUSICAL_PITCHES: &[(f64, &str)] = &[
    (82.41, "E2"),
    (110.00, "A2"),
    (146.83, "D3"),
    (196.00, "G3"),
    (246.94, "B3"),
    (329.63, "E4"),
    (440.00, "A4"),
    (587.33, "D5"),
    (659.25, "E5"),
];

/// High-frequency stress tone to expose aliasing aggressively.
///
/// Uses 2017 Hz (incommensurate with 48 kHz Nyquist) to ensure aliased
/// harmonics do not fold back coincidentally onto existing harmonic bins.
pub const STRESS_F0: f64 = 2017.0;

/// High gain for stress testing: +12 dB to drive nonlinearities harder.
pub const HIGH_GAIN: f64 = 4.0;

// =============================================================================
// ASR Result
// =============================================================================

/// Result of an ASR measurement for a single test tone.
#[derive(Debug, Clone)]
pub struct AsrResult {
    /// Fundamental frequency in Hz.
    pub f0: f64,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// ASR value in dB: `10 · log10(E_aliased / E_harmonic)`.
    ///
    /// Returns `f64::NEG_INFINITY` if no aliased energy is detected
    /// (i.e., the system behaves linearly at this frequency).
    pub asr_db: f64,
    /// ASR on a linear scale: `E_aliased / E_harmonic`.
    pub asr_linear: f64,
    /// Total energy in detected harmonic bins (squared magnitude sum).
    pub harmonic_energy: f64,
    /// Total energy in detected aliased (non-harmonic) bins (squared magnitude sum).
    pub aliased_energy: f64,
    /// Number of detected harmonic peaks.
    pub num_harmonics: usize,
    /// Number of detected aliased peaks.
    pub num_aliased: usize,
    /// Noise floor estimate (median magnitude across all frequency bins).
    pub noise_floor: f64,
    /// Peak detection threshold (in magnitude units).
    pub peak_threshold: f64,
    /// Frequency resolution (bin width) in Hz.
    pub bin_width: f64,
}

impl AsrResult {
    /// Returns `true` if any aliasing was detected (aliased energy > 0).
    pub fn has_aliasing(&self) -> bool {
        self.aliased_energy > f64::EPSILON
    }
}

// =============================================================================
// Signal Generation
// =============================================================================

/// Generates a pure sine waveform at `f0` Hz.
///
/// `gain` is a linear multiplier applied to the sine (1.0 = 0 dBFS peak).
pub fn generate_sine(f0: f64, sample_rate: u32, num_samples: usize, gain: f64) -> Vec<f32> {
    let sr = sample_rate as f64;
    let omega = 2.0 * std::f64::consts::PI * f0 / sr;
    (0..num_samples)
        .map(|i| ((i as f64 * omega).sin() * gain) as f32)
        .collect()
}

/// Canonical 440 Hz sine generator at 48 kHz, 0 dBFS.
///
/// Preferred entry point for any code that needs a reference A4 sine
/// (test signals, benchmarks, validation). Eliminates redundant per-crate
/// implementations.
pub fn generate_sine_440hz(num_samples: usize) -> Vec<f32> {
    generate_sine(440.0, 48_000, num_samples, 1.0)
}

/// Returns the 4-term Blackman-Harris window coefficients.
///
/// Coefficients (Nuttall 1981):
/// - a0 = 0.35875
/// - a1 = 0.48829
/// - a2 = 0.14128
/// - a3 = 0.01168
///
/// Highest side-lobe rejection among common 4-term windows (−92 dB).
pub fn blackman_harris_4term(n: usize) -> Vec<f64> {
    let n_minus_1 = (n - 1) as f64;
    (0..n)
        .map(|i| {
            let x = 2.0 * std::f64::consts::PI * i as f64 / n_minus_1;
            0.35875 - 0.48829 * x.cos() + 0.14128 * (2.0 * x).cos() - 0.01168 * (3.0 * x).cos()
        })
        .collect()
}

// =============================================================================
// Core ASR Computation
// =============================================================================

/// Computes the ASR for a pure-sine input and the corresponding system output.
///
/// # Arguments
/// * `output` - The system's output signal (must be power-of-two length ≥ 1024).
/// * `f0` - Fundamental frequency of the input sine in Hz.
/// * `sample_rate` - Sample rate in Hz.
///
/// # Panics
/// Panics if `output.len()` is not a power of two, or if it's shorter than 1024.
pub fn compute_asr(output: &[f32], f0: f64, sample_rate: u32) -> AsrResult {
    let n = output.len();
    assert!(
        n.is_power_of_two(),
        "Signal length {n} must be power of two"
    );
    assert!(n >= 1024, "Signal too short ({n}) for reliable ASR");
    assert!(f0 > 0.0, "f0 must be positive");

    let bin_width = sample_rate as f64 / n as f64;
    let nyquist = sample_rate as f64 / 2.0;
    let num_bins = n / 2 + 1;

    // 1. Apply Blackman-Harris window
    let window = blackman_harris_4term(n);
    let windowed: Vec<f64> = output
        .iter()
        .enumerate()
        .map(|(i, &s)| s as f64 * window[i])
        .collect();

    // 2. Real FFT
    let mut rfft = RfftPlanner::<f64>::new(n);
    let mut re = vec![0.0f64; num_bins];
    let mut im = vec![0.0f64; num_bins];
    rfft.process_forward(&windowed, &mut re, &mut im);

    // 3. Magnitude spectrum
    let mag: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| (r * r + i * i).sqrt())
        .collect();

    // 4. Noise floor: median of all bin magnitudes (robust to outliers)
    let noise_floor = median(&mag);

    // 5. Peak detection threshold
    //    Sato & Smith (DAFx 2025) propose 6× the median noise floor as a
    //    robust peak-detection threshold for harmonic/aliasing separation in
    //    ASR measurements. The factor accounts for spectral leakage smearing
    //    adjacent bins and for statistical noise-floor fluctuations inherent
    //    to windowed FFT magnitude spectra of finite-length signals.
    let max_mag = mag.iter().cloned().fold(0.0f64, f64::max);
    let peak_threshold = (noise_floor * 6.0).max(max_mag * 1e-4);

    // 6. Find local maxima in magnitude spectrum (skip DC at bin 0)
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..num_bins - 1 {
        if mag[i] > mag[i - 1] && mag[i] > mag[i + 1] && mag[i] > peak_threshold {
            peaks.push((i, mag[i]));
        }
    }

    // 7. Classify peaks: harmonic vs aliased
    //    Tolerance: 1.5 bins in Hz. This accounts for:
    //    (a) spectral leakage — energy from sharp harmonic peaks smears into
    //        neighboring FFT bins (especially with high-CF Blackman-Harris
    //        windowing on high-Q harmonics),
    //    (b) bin-width granularity — at low FFT sizes a single bin may be
    //        several Hz wide, so a fixed fractional-bin tolerance decouples
    //        classification accuracy from N (the FFT length),
    //    (c) non-integer harmonic alignment — the harmonic frequency
    //        (k × f0) does not generally coincide with an FFT bin center,
    //        requiring a ± tolerance to avoid misclassifying true harmonics
    //        as aliased components.
    //    The 1.5-bin width matches Sato & Smith (DAFx 2025): a ±1-bin margin
    //    plus ½-bin slack for the leakage shoulder.
    let tolerance = 1.5 * bin_width;
    let mut harmonic_energy = 0.0f64;
    let mut aliased_energy = 0.0f64;
    let mut num_harmonics = 0usize;
    let mut num_aliased = 0usize;

    let max_harm = (nyquist / f0).floor() as usize;

    for &(bin_idx, peak_mag) in &peaks {
        let freq_bin = bin_idx as f64 * bin_width;

        // Try to match to a harmonic k·f0 for k ∈ [1, max_harm]
        let mut is_harmonic = false;
        for k in 1..=max_harm {
            let expected = k as f64 * f0;
            if (freq_bin - expected).abs() <= tolerance {
                is_harmonic = true;
                break;
            }
        }

        let energy = peak_mag * peak_mag;
        if is_harmonic {
            harmonic_energy += energy;
            num_harmonics += 1;
        } else {
            aliased_energy += energy;
            num_aliased += 1;
        }
    }

    // 8. Compute ASR
    let asr_linear = if harmonic_energy > f64::EPSILON {
        aliased_energy / harmonic_energy
    } else {
        f64::INFINITY
    };

    let asr_db = if asr_linear <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        10.0 * asr_linear.log10()
    };

    AsrResult {
        f0,
        sample_rate,
        asr_db,
        asr_linear,
        harmonic_energy,
        aliased_energy,
        num_harmonics,
        num_aliased,
        noise_floor,
        peak_threshold,
        bin_width,
    }
}

/// Returns the aggregate ASR (mean of linear ASR values across results).
///
/// Arithmetic-averages the linear ASR values (converts to dB for reporting).
pub fn asr_aggregate(results: &[AsrResult]) -> f64 {
    if results.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = results.iter().map(|r| r.asr_linear).sum();
    let mean = sum / results.len() as f64;
    if mean <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        10.0 * mean.log10()
    }
}

/// Returns the worst-case ASR (maximum linear value) across results, in dB.
pub fn asr_worst_case(results: &[AsrResult]) -> f64 {
    results
        .iter()
        .map(|r| r.asr_db)
        .fold(f64::NEG_INFINITY, f64::max)
}

// =============================================================================
// Helpers
// =============================================================================

/// Computes the median of a slice of `f64`.
pub(crate) fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) * 0.5
    } else {
        sorted[mid]
    }
}

#[cfg(test)]
#[path = "aliasing_test.rs"]
mod tests;
