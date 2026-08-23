// SPDX-License-Identifier: MIT
// Copyright (c) 2022–2025 t3k-mushra contributors
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// License exception: This file is the sole MIT-licensed source in NeuralAmpModeler-rs. All
// other source files use Apache-2.0. The MIT license is inherited from the
// upstream project t3k-mushra <https://github.com/tone-3000/t3k-mushra> and
// cannot be re-licensed.
//
// Ported from t3k-mushra <https://github.com/tone-3000/t3k-mushra>
// Original source: src/demo/generateSampleAudio.ts, src/lib/internal/prng.ts
// See NOTICE.txt for attribution.

//! Audio primitives ported from t3k-mushra (MIT-licensed).
//!
//! This module provides deterministic PRNG (FNV-1a + Mulberry32) and five
//! audio DSP primitives for synthesizing test stimuli and MUSHRA-compliant
//! quality variants.

// =============================================================================
// PRNG — FNV-1a + Mulberry32 (bit-parity with t3k-mushra TypeScript)
// =============================================================================

/// FNV-1a 32-bit hash.
pub fn fnv1a32(input: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c9dc5;
    for &byte in input {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

/// Deterministic PRNG matching the TypeScript `mulberry32` from t3k-mushra.
///
/// Produces `f32` values in `[0, 1)`.
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    /// Creates a new PRNG with the given seed.
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// Returns the next pseudorandom `f32` in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        let mut s = self.state;
        s = (s ^ (s >> 16)).wrapping_mul(0x5bd1e995);
        s = (s ^ (s >> 13)).wrapping_mul(0xcabca125);
        self.state = s;
        (s ^ (s >> 16)) as f32 / 4294967296.0f32
    }

    /// Returns the next pseudorandom `f64` in `[-1, 1]`.
    pub fn next_f64_bipolar(&mut self) -> f64 {
        let a = self.next_f32() as f64;
        let b = self.next_f32() as f64;
        (a * 2.0 - 1.0) + (b * 2.0 - 1.0) * 0.5
    }

    /// Returns the internal state (for snapshot/parity tests).
    pub fn state(&self) -> u32 {
        self.state
    }
}

// =============================================================================
// Audio Primitives
// =============================================================================

/// Synthesizes a guitar-like pluck tone with 6 harmonics and vibrato.
///
/// Matches `synthTone(freq)` from `generateSampleAudio.ts:18-34`.
pub fn synth_tone(freq: f64, sr: u32, duration_s: f64) -> Vec<f32> {
    let n = (sr as f64 * duration_s).ceil() as usize;
    let sr_f64 = sr as f64;
    let harmonics = [1.0f64, 0.5, 0.34, 0.22, 0.14, 0.09];
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let t = i as f64 / sr_f64;
        let env = (1.0f64).min(t * 120.0) * (-2.2f64 * t).exp();
        let vibrato = 0.003 * freq * (2.0 * std::f64::consts::PI * 5.0 * t).sin();
        let mut sample = 0.0f64;
        for (k, &amp) in harmonics.iter().enumerate() {
            let h = (k + 1) as f64;
            let phase = 2.0 * std::f64::consts::PI * freq * h * t + vibrato * h;
            sample += amp * phase.sin();
        }
        out.push((sample * env) as f32);
    }
    out
}

/// 1-pole IIR low-pass filter.
///
/// Matches `lowPass(input, cutoff)` from `generateSampleAudio.ts:37-48`.
pub fn low_pass_1pole(input: &[f32], cutoff_hz: f32, sr: u32) -> Vec<f32> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let dt = 1.0f64 / (sr as f64);
    let rc = 1.0f64 / (2.0 * std::f64::consts::PI * cutoff_hz as f64);
    let a = dt / (rc + dt);
    let mut out = Vec::with_capacity(n);
    let mut y: f64 = 0.0;
    for &x in input {
        y += a * (x as f64 - y);
        out.push(y as f32);
    }
    out
}

/// Symmetric soft-clip via tanh.
///
/// Matches `softClip(input, drive)` from `generateSampleAudio.ts:58-64`.
pub fn soft_clip(input: &[f32], drive: f32) -> Vec<f32> {
    let scale = 1.0f64 / (drive as f64).tanh();
    input
        .iter()
        .map(|&x| ((x as f64 * drive as f64).tanh() * scale) as f32)
        .collect()
}

/// Adds white noise using the deterministic PRNG.
///
/// Matches `addNoise(input, amount)` from `generateSampleAudio.ts:50-56`.
pub fn add_noise(input: &[f32], amount: f32, rng: &mut Mulberry32) -> Vec<f32> {
    let a = amount as f64;
    input
        .iter()
        .map(|&x| {
            let noise = rng.next_f64_bipolar() * a;
            ((x as f64) + noise) as f32
        })
        .collect()
}

/// Applies linear gain scaling.
///
/// Matches `gain(input, g)` from `generateSampleAudio.ts:66-70`.
pub fn apply_gain(input: &[f32], g: f32) -> Vec<f32> {
    let g = g as f64;
    input.iter().map(|&x| (x as f64 * g) as f32).collect()
}

// =============================================================================
// MUSHRA Variants
// =============================================================================

/// Label for a MUSHRA quality variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MushraVariantLabel {
    /// Bit-identical hidden reference
    HiddenRef,
    /// Reference + low noise (transparent)
    Excellent,
    /// Low-pass 9 kHz + moderate noise
    Good,
    /// Low-pass 5 kHz + soft-clip
    Fair,
    /// Low-pass 2.5 kHz + gain + heavy noise
    Poor,
    /// Low-pass 3.5 kHz anchor (canonical MUSHRA)
    Anchor,
}

/// A single MUSHRA test variant (stimulus + label).
#[derive(Debug, Clone)]
pub struct MushraVariant {
    /// MUSHRA quality label
    pub label: MushraVariantLabel,
    /// Audio samples
    pub samples: Vec<f32>,
    /// Human-readable description of the processing applied
    pub description: &'static str,
}

/// Builds the 6 MUSHRA-compliant quality variants from a reference signal.
///
/// Scheme from `generateSampleAudio.ts:132-139`:
///
/// ```text
/// hidden-ref  → reference bit-identical
/// excellent   → reference + noise(0.001)
/// good        → lowpass(9 kHz) + noise(0.002)
/// fair        → lowpass(5 kHz) + softclip(drive=1.6)
/// poor        → lowpass(2.5 kHz) + gain(0.9) + noise(0.01)
/// anchor      → lowpass(3.5 kHz)  [MUSHRA canonical anchor]
/// ```
pub fn build_mushra_variants(reference: &[f32], sr: u32) -> [MushraVariant; 6] {
    let mut rng = Mulberry32::new(fnv1a32(b"nam-rs-mushra-variants"));

    let excellent = {
        let noisy = add_noise(reference, 0.001, &mut rng);
        MushraVariant {
            label: MushraVariantLabel::Excellent,
            description: "reference + noise(0.001)",
            samples: noisy,
        }
    };

    let good = {
        let lp = low_pass_1pole(reference, 9000.0, sr);
        let noisy = add_noise(&lp, 0.002, &mut rng);
        MushraVariant {
            label: MushraVariantLabel::Good,
            description: "lowpass(9kHz) + noise(0.002)",
            samples: noisy,
        }
    };

    let fair = {
        let lp = low_pass_1pole(reference, 5000.0, sr);
        let clipped = soft_clip(&lp, 1.6);
        MushraVariant {
            label: MushraVariantLabel::Fair,
            description: "lowpass(5kHz) + softclip(drive=1.6)",
            samples: clipped,
        }
    };

    let poor = {
        let lp = low_pass_1pole(reference, 2500.0, sr);
        let gained = apply_gain(&lp, 0.9);
        let noisy = add_noise(&gained, 0.01, &mut rng);
        MushraVariant {
            label: MushraVariantLabel::Poor,
            description: "lowpass(2.5kHz) + gain(0.9) + noise(0.01)",
            samples: noisy,
        }
    };

    let anchor = {
        let lp = low_pass_1pole(reference, 3500.0, sr);
        MushraVariant {
            label: MushraVariantLabel::Anchor,
            description: "lowpass(3.5kHz) [MUSHRA canonical anchor]",
            samples: lp,
        }
    };

    let hidden_ref = MushraVariant {
        label: MushraVariantLabel::HiddenRef,
        description: "reference bit-identical",
        samples: reference.to_vec(),
    };

    [hidden_ref, excellent, good, fair, poor, anchor]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulberry32_determinism() {
        let mut rng1 = Mulberry32::new(42);
        let mut rng2 = Mulberry32::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_f32(), rng2.next_f32());
        }
    }

    #[test]
    fn test_mulberry32_range() {
        let mut rng = Mulberry32::new(12345);
        for _ in 0..1000 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v), "out of range: {v}");
        }
    }

    #[test]
    fn test_fnv1a32_known_vector() {
        assert_eq!(fnv1a32(b"hello"), 0x4f9f2cab);
        assert_eq!(fnv1a32(b""), 0x811c9dc5);
    }

    #[test]
    fn test_soft_clip_identity() {
        // Very small signal with drive=0.01 should be nearly linear (tanh(x)/tanh(0.01) ≈ x/0.01 + small)
        let input: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.0001).collect();
        let output = soft_clip(&input, 0.01);
        for (x, y) in input.iter().zip(output.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "soft_clip not linear for small signal: x={x}, y={y}"
            );
        }
    }
}
