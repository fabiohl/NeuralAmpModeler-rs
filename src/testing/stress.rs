// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Deterministic stress signal generators for numerical cross-validation.
//!
//! Provides v1 (legacy 2048-sample, 48 kHz) for fast CI and v2 (multi-component,
//! 5-second, multi-sample-rate) for comprehensive drift detection.

use super::mushra::{Mulberry32, fnv1a32};

// =============================================================================
// Constants
// =============================================================================

/// Default sample rates supported for multi-SR generation.
pub const SUPPORTED_SAMPLE_RATES: &[u32] = &[44100, 48000, 88200, 96000, 192000];

/// Default duration of v2 stress signal in seconds.
pub const STRESS_V2_DURATION: f64 = 5.0;

/// Clamp ceiling for stress signal output (leaves headroom).
pub const STRESS_CLAMP_CEILING: f32 = 0.95;

/// Seed string for deterministic PRNG.
const PRNG_SEED: &str = "nam-rs-stress-v2";

// =============================================================================
// v1 — Legacy stress signal (2048 samples @ 48 kHz)
// =============================================================================

/// Generates the legacy deterministic multi-component stress signal (2048 samples @ 48 kHz).
///
/// Components:
/// - Low-E guitar harmonics (82/165/330/659 Hz)
/// - Linear chirp 220 Hz → 3520 Hz
/// - Transient impulse (+0.9) at 25%
/// - Attack–sustain–release envelope with fade-to-silence
///
/// Bit-for-bit identical to the original Python implementation.
pub fn generate_stress_signal_v1() -> Vec<f32> {
    let n = 2048;
    let sr = 48000.0f64;
    let attack_end = (0.002 * sr) as usize;
    let release_beg = n - (0.005 * sr) as usize;
    let t_total = n as f64 / sr;

    (0..n)
        .map(|i| {
            let t = i as f64 / sr;

            let env = if i < attack_end {
                i as f64 / attack_end as f64
            } else if i >= release_beg {
                (n - 1 - i) as f64 / (n - release_beg) as f64
            } else {
                1.0
            };

            let guitar = 0.40 * (2.0 * std::f64::consts::PI * 82.41 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 164.81 * t).sin()
                + 0.15 * (2.0 * std::f64::consts::PI * 329.63 * t).sin()
                + 0.08 * (2.0 * std::f64::consts::PI * 659.25 * t).sin();

            let f0: f64 = 220.0;
            let f1: f64 = 3520.0;
            let chirp_phase =
                2.0 * std::f64::consts::PI * (f0 * t + (f1 - f0) * t * t / (2.0 * t_total));
            let chirp = 0.30 * chirp_phase.sin();

            let impulse = if i == n / 4 { 0.9 } else { 0.0 };

            let sample = env * (guitar + chirp) + impulse;
            sample.clamp(-1.0, 1.0) as f32
        })
        .collect()
}

// =============================================================================
// v2 — Multi-component stress signal (5 seconds, multi-SR)
// =============================================================================

/// Generates the Stress Signal v2 — a 5-second multi-component signal for
/// comprehensive numerical drift detection.
///
/// Components (deterministic via seeded PRNG):
/// - 0.0–1.0s: Single note Low-E (82.41 Hz) with bend ½-tone + vibrato
/// - 1.0–2.0s: Power chord E2+E3+B3 with ADSR envelope
/// - 2.0–2.5s: Palm-mute attack-release (16 hits, 120 BPM)
/// - 2.5–3.5s: Pinch harmonic train + saw sweep 200→3500 Hz
/// - 3.5–4.5s: Bass amp: low-A (55 Hz) with 5 harmonics + transient pluck
/// - 4.5–5.0s: Slow chord ringing decay (C-E-G, exponential fade)
pub fn generate_stress_signal_v2(seed: &str, sample_rate: u32) -> Vec<f32> {
    let sr = sample_rate as f64;
    let duration = STRESS_V2_DURATION;
    let n = (sr * duration) as usize;
    let mut out = vec![0.0f64; n];

    let seed_hash = fnv1a32(seed.as_bytes());
    let mut rng = Mulberry32::new(seed_hash);

    // --- 0.0–1.0s: Single note Low-E (82.41 Hz) with bend + vibrato (GA) ---
    {
        let t0 = 0.0;
        let t1 = 1.0;
        let i0 = (t0 * sr) as usize;
        let i1 = (t1 * sr).min(n as f64) as usize;
        let freq = 82.41;
        let harmonics = [1.0f64, 0.5, 0.34, 0.22, 0.14, 0.09];
        let bend_cents = 50.0; // ½-tone bend

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;
            let bend_factor = 2.0f64.powf(bend_cents * local_t / 1200.0);
            let f = freq * bend_factor;
            let vibrato = 0.0015 * freq * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
            let env = (1.0f64).min(local_t * 120.0) * (-2.2 * local_t).exp();

            let mut s = 0.0f64;
            for (k, &amp) in harmonics.iter().enumerate() {
                let h = (k + 1) as f64;
                let phase = 2.0 * std::f64::consts::PI * f * h * t + vibrato * h;
                s += amp * phase.sin();
            }
            *sample += s * env * 0.9;
        }
    }

    // --- 1.0–2.0s: Power chord E2+E3+B3 (FRG) ---
    {
        let t0 = 1.0;
        let t1 = 2.0;
        let i0 = (t0 * sr) as usize;
        let i1 = (t1 * sr).min(n as f64) as usize;
        let voices = [
            (82.41f64, 0.6),  // E2
            (164.81f64, 0.5), // E3
            (246.94f64, 0.4), // B3
        ];
        let harmonics = [1.0f64, 0.5, 0.3, 0.18, 0.1];

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;

            let attack = (local_t * 500.0).min(1.0);
            let sustain = 0.7;
            let release = if local_t > 0.85 {
                (1.0 - (local_t - 0.85) / 0.15).max(0.0)
            } else {
                1.0
            };
            let env = attack * sustain * release;

            let mut s = 0.0f64;
            for &(freq, amp) in &voices {
                for (k, &ha) in harmonics.iter().enumerate() {
                    let h = (k + 1) as f64;
                    let phase = 2.0 * std::f64::consts::PI * freq * h * t;
                    s += amp * ha * phase.sin();
                }
            }
            *sample += s * env * 0.85;
        }
    }

    // --- 2.0–2.5s: Palm-mute attack-release (P) ---
    {
        let t0 = 2.0;
        let t1 = 2.5;
        let i0 = (t0 * sr) as usize;
        let i1 = (t1 * sr).min(n as f64) as usize;
        let bpm = 120.0;
        let beat_interval = 60.0 / bpm / 4.0; // 16th note
        let attack_s = 0.002;
        let decay_s = 0.020;
        let freq = 110.0; // A2

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;

            let hit_idx = (local_t / beat_interval).floor() as usize;
            let hit_t = local_t - hit_idx as f64 * beat_interval;

            let env = if hit_t < attack_s {
                hit_t / attack_s
            } else if hit_t < attack_s + decay_s {
                let d = (hit_t - attack_s) / decay_s;
                (-4.0 * d).exp()
            } else {
                0.0
            };

            let noise = (rng.next_f32() as f64 * 2.0 - 1.0) * 0.05;
            let s = (2.0 * std::f64::consts::PI * freq * t).sin() * 0.7 + noise;
            *sample += s * env;
        }
    }

    // --- 2.5–3.5s: Pinch harmonic train + saw sweep (P) ---
    {
        let t0 = 2.5;
        let t1 = 3.5;
        let i0 = (t0 * sr) as usize;
        let i1 = (t1 * sr).min(n as f64) as usize;
        let harmonics = [4, 5, 6, 7]; // harmonic numbers
        let base_freq = 82.41;
        let harmonics_per_block = 0.25; // seconds per harmonic

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;
            let h_idx = (local_t / harmonics_per_block).floor() as usize % harmonics.len();
            let h_num = harmonics[h_idx] as f64;

            let env = (-3.0 * (local_t % harmonics_per_block)).exp();

            let swipe = 200.0 + (3500.0 - 200.0) * (local_t / (t1 - t0));
            let saw = saw_wave(swipe, t, sr, 6);

            let harmonic: f64 = (2.0 * std::f64::consts::PI * base_freq * h_num * t).sin();
            *sample += (harmonic * 0.4 + saw * 0.15) * env * 0.8;
        }
    }

    // --- 3.5–4.5s: Bass amp: low-A (55 Hz) (BA) ---
    {
        let t0 = 3.5;
        let t1 = 4.5;
        let i0 = (t0 * sr) as usize;
        let i1 = (t1 * sr).min(n as f64) as usize;
        let freq = 55.0;
        let harmonics = [1.0f64, 0.7, 0.5, 0.35, 0.2];

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;

            let env = if local_t < 0.01 {
                local_t / 0.01
            } else {
                (-0.5 * local_t).exp()
            };

            let mut s = 0.0f64;
            for (k, &amp) in harmonics.iter().enumerate() {
                let h = (k + 1) as f64;
                let phase = 2.0 * std::f64::consts::PI * freq * h * t;
                s += amp * phase.sin();
            }
            *sample += s * env * 0.9;
        }
    }

    // --- 4.5–5.0s: Slow chord ringing decay C-E-G (PA) ---
    {
        let t0 = 4.5;
        let _t1 = 5.0;
        let i0 = (t0 * sr) as usize;
        let i1 = n;
        let voices = [
            (261.63f64, 0.5), // C4
            (329.63f64, 0.4), // E4
            (392.00f64, 0.3), // G4
        ];
        let harmonics = [1.0f64, 0.4, 0.2, 0.1];

        for (i, sample) in out.iter_mut().enumerate().take(i1).skip(i0) {
            let t = i as f64 / sr;
            let local_t = t - t0;

            let env = (-6.0 * local_t).exp();

            let mut s = 0.0f64;
            for &(freq, amp) in &voices {
                for (k, &ha) in harmonics.iter().enumerate() {
                    let h = (k + 1) as f64;
                    let phase = 2.0 * std::f64::consts::PI * freq * h * t;
                    s += amp * ha * phase.sin();
                }
            }
            *sample += s * env * 0.6;
        }
    }

    // Clamp final output
    out.into_iter()
        .map(|s| s.clamp(-(STRESS_CLAMP_CEILING as f64), STRESS_CLAMP_CEILING as f64) as f32)
        .collect()
}

/// Generates stress signal v2 with default seed for the given sample rate.
pub fn generate_stress_signal_v2_default(sample_rate: u32) -> Vec<f32> {
    generate_stress_signal_v2(PRNG_SEED, sample_rate)
}

// =============================================================================
// Signal Energy & Numerical Finitude Evaluation
// =============================================================================

/// Result of evaluating numerical finitude and RMS energy of an audio buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyEvaluation {
    /// Root-mean-square energy in dBFS (`20 · log10(RMS)`).
    /// Returns `f64::NEG_INFINITY` if signal is strictly silent.
    pub rms_dbfs: f64,
    /// Absolute peak level in dBFS (`20 · log10(max |sample|)`).
    pub peak_dbfs: f64,
    /// Whether all samples in the buffer are finite (zero NaN or Inf).
    pub is_finite: bool,
    /// Whether RMS energy meets or exceeds the minimum active threshold (e.g., -80.0 dBFS).
    pub is_active: bool,
}

/// Verifies that 100% of samples in `samples` are finite numbers (no NaN or Inf).
pub fn check_finitude(samples: &[f32]) -> bool {
    samples.iter().all(|s| s.is_finite())
}

/// Computes the RMS energy in dBFS for the given audio buffer.
pub fn compute_rms_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64).powi(2)).sum();
    let mean_sq = sum_sq / samples.len() as f64;
    let rms = mean_sq.sqrt();
    if rms <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        20.0 * rms.log10()
    }
}

/// Computes the peak level in dBFS for the given audio buffer.
pub fn compute_peak_dbfs(samples: &[f32]) -> f64 {
    if samples.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max_abs = samples
        .iter()
        .map(|&s| (s as f64).abs())
        .fold(0.0f64, f64::max);
    if max_abs <= f64::EPSILON {
        f64::NEG_INFINITY
    } else {
        20.0 * max_abs.log10()
    }
}

/// Evaluates finitude, RMS energy, and peak level of `samples` against `min_rms_dbfs`.
///
/// Default active threshold is typically -80.0 dBFS.
pub fn evaluate_signal_energy(samples: &[f32], min_rms_dbfs: f64) -> EnergyEvaluation {
    let is_finite = check_finitude(samples);
    let rms_dbfs = compute_rms_dbfs(samples);
    let peak_dbfs = compute_peak_dbfs(samples);
    let is_active = rms_dbfs >= min_rms_dbfs;

    EnergyEvaluation {
        rms_dbfs,
        peak_dbfs,
        is_finite,
        is_active,
    }
}

// =============================================================================
// Block-Size Invariance Verification
// =============================================================================

/// Standard block sizes used for block-size invariance testing: 1, 8, 32, 64, 128, 512, 2048.
pub const STANDARD_TEST_BLOCK_SIZES: &[usize] = &[1, 8, 32, 64, 128, 512, 2048];

/// Result of evaluating block-size invariance across arbitrary block sizes.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockInvarianceResult {
    /// Maximum absolute sample difference observed across all block size comparisons.
    pub max_abs_error: f32,
    /// Baseline block size used as ground truth (typically 64).
    pub baseline_block_size: usize,
    /// Errors for each tested block size relative to baseline output: `(block_size, max_abs_err)`.
    pub errors_by_block_size: Vec<(usize, f32)>,
    /// Whether all tested block sizes stayed within `max_allowed_error` (typically 1e-6 f32).
    pub is_invariant: bool,
}

/// Evaluates block-size invariance for a model instance by processing `input_signal`
/// in blocks of size `baseline_block_size` vs each size in `block_sizes`.
///
/// Returns a `BlockInvarianceResult` comparing continuous output signals sample-by-sample.
pub fn verify_block_invariance_for_model<M: crate::models::NamModel + ?Sized, F: Fn() -> Box<M>>(
    create_model: F,
    input_signal: &[f32],
    block_sizes: &[usize],
    baseline_block_size: usize,
    max_allowed_error: f32,
) -> BlockInvarianceResult {
    let sizes = if block_sizes.is_empty() {
        STANDARD_TEST_BLOCK_SIZES
    } else {
        block_sizes
    };

    // 1. Process baseline signal
    let mut baseline_model = create_model();
    let _ = baseline_model.set_max_buffer_size(baseline_block_size);
    baseline_model.prewarm(2048);
    let mut baseline_output = vec![0.0f32; input_signal.len()];

    for (in_chunk, out_chunk) in input_signal
        .chunks(baseline_block_size)
        .zip(baseline_output.chunks_mut(baseline_block_size))
    {
        baseline_model.process(in_chunk, out_chunk);
    }

    let mut overall_max_err = 0.0f32;
    let mut errors_by_size = Vec::new();

    // 2. Process each test block size
    for &bs in sizes {
        let mut test_model = create_model();
        let _ = test_model.set_max_buffer_size(bs);
        test_model.prewarm(2048);
        let mut test_output = vec![0.0f32; input_signal.len()];

        for (in_chunk, out_chunk) in input_signal.chunks(bs).zip(test_output.chunks_mut(bs)) {
            test_model.process(in_chunk, out_chunk);
        }

        let mut max_err_for_bs = 0.0f32;
        for (&b_sample, &t_sample) in baseline_output.iter().zip(test_output.iter()) {
            let diff = (b_sample - t_sample).abs();
            if diff > max_err_for_bs {
                max_err_for_bs = diff;
            }
        }

        if max_err_for_bs > overall_max_err {
            overall_max_err = max_err_for_bs;
        }

        errors_by_size.push((bs, max_err_for_bs));
    }

    let is_invariant = overall_max_err <= max_allowed_error;

    BlockInvarianceResult {
        max_abs_error: overall_max_err,
        baseline_block_size,
        errors_by_block_size: errors_by_size,
        is_invariant,
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Sawtooth wave with `num_harmonics`.
fn saw_wave(freq: f64, t: f64, _sr: f64, num_harmonics: usize) -> f64 {
    let mut s = 0.0;
    for h in 1..=num_harmonics {
        let phase = 2.0 * std::f64::consts::PI * freq * h as f64 * t;
        s += phase.sin() / h as f64;
    }
    s * 2.0 / std::f64::consts::PI
}

#[cfg(test)]
#[path = "stress_test.rs"]
mod stress_test;
