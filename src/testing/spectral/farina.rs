// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Farina exponential sine sweep analysis and deconvolution routines.

use super::next_power_of_two;
use crate::math::dsp::fft::FftPlanner;
use std::f64::consts::TAU;

/// Farina sweep + deconvolution result.
#[derive(Debug, Clone)]
pub struct FarinaResult {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Start frequency of the sweep in Hz.
    pub f1: f64,
    /// End frequency of the sweep in Hz.
    pub f2: f64,
    /// Sweep duration in seconds.
    pub duration_s: f64,
    /// Linear impulse response (time-domain), length = n_samples.
    pub ir_linear: Vec<f64>,
    /// Frequency response magnitude (dB), length = ir_linear.len()/2 + 1.
    pub fr_magnitude_db: Vec<f64>,
    /// Frequency response phase (radians), length = ir_linear.len()/2 + 1.
    pub fr_phase_rad: Vec<f64>,
    /// Frequency axis in Hz, length = ir_linear.len()/2 + 1.
    pub freq_axis: Vec<f64>,
    /// THD per harmonic order: `Vec<(order, thd_percent)>`.
    /// `order = 1` is the linear component (THD = 0 by definition).
    /// Higher orders are distortion.
    pub thd_by_order: Vec<(u32, f64)>,
    /// Overall THD (%) summed from orders ≥ 2.
    pub thd_total_percent: f64,
}

/// Generates an exponential sine sweep with the Farina method.
///
/// The sweep covers `[f1, f2]` Hz over `duration_s` seconds at `sample_rate` Hz.
/// Returns the sweep signal with peak amplitude ≤ 1.0.
pub fn generate_farina_sweep(f1: f64, f2: f64, duration_s: f64, sample_rate: u32) -> Vec<f64> {
    assert!(f1 > 0.0 && f2 > f1 && duration_s > 0.0 && sample_rate > 0);
    let n = (sample_rate as f64 * duration_s).ceil() as usize;
    let sr = sample_rate as f64;
    let t_scale = sr * duration_s;
    let omega1 = TAU * f1;
    let ln_ratio = (f2 / f1).ln();

    let mut sweep = Vec::with_capacity(n);
    for i in 0..n {
        let t_norm = i as f64 / t_scale;
        // Instantaneous phase
        let phase = omega1 * duration_s / ln_ratio * ((t_norm * ln_ratio).exp_m1());
        sweep.push(phase.sin());
    }
    sweep
}

/// Generates the inverse filter for a Farina sweep.
///
/// Computed directly in the frequency domain:
///   F\[k\] = conj(S\[k\]) / (|S\[k\]|² + ε)
///   f\[n\] = IFFT(F\[k\])  (first n samples)
///
/// This is the mathematically exact matched filter, equivalent to the
/// time-reversed sweep with amplitude compensation.
pub fn generate_farina_inverse_filter(
    sweep: &[f64],
    _f1: f64,
    _f2: f64,
    _duration_s: f64,
    _sample_rate: u32,
) -> Vec<f64> {
    let n = sweep.len();
    let n_fft = next_power_of_two(n);

    let fft = FftPlanner::<f64>::new(n_fft);

    let mut s_re = vec![0.0f64; n_fft];
    let mut s_im = vec![0.0f64; n_fft];
    s_re[..n].copy_from_slice(sweep);

    fft.process(&mut s_re, &mut s_im);

    // H[k] = conj(S[k]) / (|S[k]|² + ε)
    let eps = 1e-10;
    for i in 0..n_fft {
        let mag_sq = s_re[i] * s_re[i] + s_im[i] * s_im[i] + eps;
        s_re[i] /= mag_sq;
        s_im[i] = -s_im[i] / mag_sq;
    }

    fft.process_inverse(&mut s_re, &mut s_im);

    // Take first n samples as the inverse filter
    let mut inv: Vec<f64> = s_re[..n].to_vec();

    // Normalise
    let max_abs = inv.iter().map(|&x| x.abs()).fold(0.0f64, f64::max);
    if max_abs > 1e-10 {
        let scale = 0.95 / max_abs;
        for v in &mut inv {
            *v *= scale;
        }
    }

    inv
}

/// Deconvolves the system output `y[n]` with the inverse filter `f[n]` via FFT.
///
/// Uses circular convolution (same length as input), as required by the
/// Farina method. Both signals must have the same length.
///
/// Returns the deconvolved impulse response `h[n] = y[n] ⊛ f[n]`.
pub(crate) fn deconvolve_farina(y: &[f64], inv_filter: &[f64]) -> Vec<f64> {
    assert_eq!(
        y.len(),
        inv_filter.len(),
        "y and inv_filter must have same length for Farina deconvolution"
    );
    let n = next_power_of_two(y.len());

    let fft = FftPlanner::<f64>::new(n);

    let mut y_re = vec![0.0f64; n];
    let mut y_im = vec![0.0f64; n];
    let mut f_re = vec![0.0f64; n];
    let mut f_im = vec![0.0f64; n];

    y_re[..y.len()].copy_from_slice(y);
    f_re[..inv_filter.len()].copy_from_slice(inv_filter);

    fft.process(&mut y_re, &mut y_im);
    fft.process(&mut f_re, &mut f_im);

    // Pointwise complex multiply
    for i in 0..n {
        let yr = y_re[i];
        let yi = y_im[i];
        let fr = f_re[i];
        let fi = f_im[i];
        y_re[i] = yr * fr - yi * fi;
        y_im[i] = yr * fi + yi * fr;
    }

    fft.process_inverse(&mut y_re, &mut y_im);

    // Truncate to original length (circular convolution)
    y_re.truncate(y.len());
    y_re
}

/// Runs the full Farina measurement pipeline.
///
/// 1. Generates exponential sweep from `f1` to `f2` over `duration_s`.
/// 2. Processes the sweep through `process_fn` to obtain system output.
/// 3. Deconvolves with inverse filter to obtain separated impulse responses.
/// 4. Extracts linear IR and THD per harmonic order.
///
/// `process_fn` receives the sweep (f64) and must return the system output (f32 or f64).
/// The `max_harmonics` parameter controls how many harmonic orders to extract.
pub fn farina_measure<F>(
    f1: f64,
    f2: f64,
    duration_s: f64,
    sample_rate: u32,
    max_harmonics: u32,
    process_fn: F,
) -> FarinaResult
where
    F: FnOnce(&[f64]) -> Vec<f32>,
{
    let sweep = generate_farina_sweep(f1, f2, duration_s, sample_rate);
    let n = sweep.len();
    let inv_filter = generate_farina_inverse_filter(&sweep, f1, f2, duration_s, sample_rate);

    let output_f32 = process_fn(&sweep);
    assert_eq!(
        output_f32.len(),
        n,
        "process_fn output length {} != sweep length {n}",
        output_f32.len()
    );
    let output: Vec<f64> = output_f32.iter().map(|&x| x as f64).collect();

    let deconv = deconvolve_farina(&output, &inv_filter);

    // The deconvolution result has the linear IR at the start (delay ≈ 0)
    // and harmonic distortion kernels spaced apart.
    // The time lag for harmonic order k relative to the linear IR is:
    //   Δt_k = T · ln(k) / ln(f2/f1)
    let sr = sample_rate as f64;
    let ln_ratio = (f2 / f1).ln();
    let _delay_linear = inv_filter.len() as f64 / sr; // approximate

    // Extract linear IR (from start to first harmonic boundary)
    // The 2nd harmonic appears at: half_range = T·ln(2)/ln(f2/f1) seconds
    let half_point_s = duration_s * (1.5f64).ln() / ln_ratio;
    let half_point_samples = (half_point_s * sr).round() as usize;
    let ir_len = if half_point_samples < n {
        half_point_samples
    } else {
        n / 2
    };

    let ir_linear: Vec<f64> = deconv[..ir_len.min(deconv.len())].to_vec();

    // Compute frequency response from linear IR via FFT
    let fr_n = next_power_of_two(ir_linear.len());
    let mut fr_re = vec![0.0f64; fr_n];
    let mut fr_im = vec![0.0f64; fr_n];
    fr_re[..ir_linear.len()].copy_from_slice(&ir_linear);

    let fft = FftPlanner::<f64>::new(fr_n);
    fft.process(&mut fr_re, &mut fr_im);

    let num_bins = fr_n / 2 + 1;
    let mut fr_magnitude_db = vec![0.0f64; num_bins];
    let mut fr_phase_rad = vec![0.0f64; num_bins];
    let mut freq_axis = vec![0.0f64; num_bins];

    for i in 0..num_bins {
        freq_axis[i] = i as f64 * sr / fr_n as f64;
        let mag = (fr_re[i] * fr_re[i] + fr_im[i] * fr_im[i]).sqrt();
        let mag_db = if mag > 1e-15 {
            20.0 * mag.log10()
        } else {
            -300.0
        };
        fr_magnitude_db[i] = mag_db;
        fr_phase_rad[i] = fr_im[i].atan2(fr_re[i]);
    }

    // Extract THD per harmonic order
    // For each order k, extract the IR segment at time lag Δt_k
    // Time lag for harmonic order k: Δt_k = T · ln(k) / ln(f2/f1)
    let window_len = ir_len;
    let mut harmonic_energies: Vec<(u32, f64)> = Vec::new();

    for k in 1..=max_harmonics {
        let lag_s = duration_s * (k as f64).ln() / ln_ratio;
        let lag_samples = (lag_s * sr).round() as isize;

        let start = lag_samples.max(0) as usize;
        let end = (start + window_len).min(deconv.len());

        if end <= start || start >= deconv.len() {
            break;
        }

        let segment = &deconv[start..end];
        let energy: f64 = segment.iter().map(|&x| x * x).sum();
        harmonic_energies.push((k, energy));
    }

    let fund_energy = harmonic_energies.first().map(|(_, e)| *e).unwrap_or(0.0);

    let mut thd_by_order: Vec<(u32, f64)> = Vec::new();
    for (k, energy) in &harmonic_energies {
        let thd = if *k == 1 {
            0.0
        } else if fund_energy > 1e-20 {
            100.0 * (energy / fund_energy).sqrt()
        } else {
            0.0
        };
        thd_by_order.push((*k, thd));
    }

    // Compute total THD (sum of orders ≥ 2).
    let thd_total_percent = if thd_by_order.len() > 1 {
        let sum_sq: f64 = thd_by_order
            .iter()
            .skip(1)
            .map(|(_, thd)| (*thd / 100.0).powi(2))
            .sum();
        100.0 * sum_sq.sqrt()
    } else {
        0.0
    };

    // Recalculate THD for order 1 (should be 0 or very small as it's the linear part)
    if let Some((1, _)) = thd_by_order.first() {}

    FarinaResult {
        sample_rate,
        f1,
        f2,
        duration_s,
        ir_linear,
        fr_magnitude_db,
        fr_phase_rad,
        freq_axis,
        thd_by_order,
        thd_total_percent,
    }
}
