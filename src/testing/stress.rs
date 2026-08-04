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
