// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Golden tests for the Linear architecture using direct convolution as
//  mathematical oracle.
//
//  The Linear model is a simple FIR filter: `output[n] = bias + Σ w[k]·x[n-k]`.
//  The mathematical oracle computes this directly (forward-time weights,
//  scalar accumulator) and compares against the Rust engine (reversed weights,
//  SIMD dot product via `MirroredBuffer`). No golden `.bin` files or C++
//  reference are needed — this is a pure mathematical oracle.
//
//  This follows the same pattern as `tests/cabsim_golden.rs`.

use super::common;

use common::generate_sine_440hz;
use neural_amp_modeler_rs::loader::nam_json::LinearImplementation;
use neural_amp_modeler_rs::models::linear::LinearModel;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};

// ── Simple LCG PRNG (no rand dependency) ──

struct SimplePcg {
    state: u64,
    inc: u64,
}

impl SimplePcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1442695040888963407),
            inc: 1,
        }
    }

    fn next_f32(&mut self) -> f32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(self.inc | 1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let rotated = (xorshifted >> rot) | (xorshifted << ((-(rot as i32)) & 31));
        (rotated as f32) * 2.3283064e-10
    }

    fn next_f32_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

// ── Mathematical Oracle: Direct FIR Convolution ──

/// Direct FIR convolution oracle for Linear architecture.
///
/// `weights_forward` are in forward-time order (as stored in JSON, before the
/// engine reverses them internally). `bias` is added after the dot product.
///
/// ```text
/// output[n] = bias + Σ_{k=0}^{min(n, R-1)} weights_forward[k] * input[n-k]
/// ```
fn linear_reference(input: &[f32], weights_forward: &[f32], bias: f32) -> Vec<f32> {
    let r = weights_forward.len();
    if r == 0 {
        return vec![bias; input.len()];
    }
    let mut output = vec![0.0f32; input.len()];
    for n in 0..input.len() {
        let mut sum = 0.0f32;
        let max_k = n.min(r - 1);
        for k in 0..=max_k {
            sum += weights_forward[k] * input[n - k];
        }
        output[n] = bias + sum;
    }
    output
}

/// Synthesize deterministic FIR weights (forward-time order).
fn synth_weights_fir(len: usize, rng: &mut SimplePcg) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / len as f32;
            (std::f32::consts::TAU * 3.0 * t).cos() * (-2.0 * t).exp() * 0.9
                + 0.01 * rng.next_f32_signed()
        })
        .collect()
}

/// Synthesize a deterministic multi-component input signal.
fn synth_signal(len: usize, rng: &mut SimplePcg) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / 48000.0;
            0.45 * (std::f32::consts::TAU * 110.0 * t).sin()
                + 0.30 * (std::f32::consts::TAU * 440.0 * t).sin()
                + 0.15 * (std::f32::consts::TAU * 880.0 * t).sin()
                + 0.03 * rng.next_f32_signed()
        })
        .collect()
}

// ── Golden verification ──

/// Builds a Linear model, prewarms it, processes the input, and compares the
/// engine output against the mathematical oracle reference.
#[track_caller]
fn verify_linear_golden(
    label: &str,
    weights_forward: &[f32],
    bias: f32,
    input: &[f32],
    max_esr: f64,
) {
    let reference = linear_reference(input, weights_forward, bias);

    let mut model = StaticModel::Linear(Box::new(
        LinearModel::new(
            weights_forward.to_vec(),
            bias,
            LinearImplementation::default(),
        )
        .expect("Linear model creation"),
    ));
    model.prewarm(input.len());
    let mut output = vec![0.0f32; input.len()];
    model.process(input, &mut output);

    assert_eq!(
        reference.len(),
        output.len(),
        "[{label}] Length mismatch: ref={}, out={}",
        reference.len(),
        output.len()
    );

    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;
    let mut max_abs_diff = 0.0f32;

    for (r, o) in reference.iter().zip(output.iter()) {
        let err = (o - r).abs();
        if err > max_abs_diff {
            max_abs_diff = err;
        }
        sum_sq_err += (*r as f64 - *o as f64).powi(2);
        sum_sq_ref += (*r as f64) * (*r as f64);
    }

    let esr = if sum_sq_ref < 1e-30 {
        0.0
    } else {
        sum_sq_err / sum_sq_ref
    };

    // Compute SNR for diagnostics
    let snr = if sum_sq_err <= f64::EPSILON {
        f64::INFINITY
    } else {
        10.0 * (sum_sq_ref / sum_sq_err).log10()
    };

    println!();
    println!("[Linear Golden × Direct Convolution — {label}]");
    println!(
        "  ESR    = {esr:.4e}       (threshold < {max_esr:.1e})  {}",
        if esr < max_esr { "✓" } else { "✗" }
    );
    if max_abs_diff > 0.0 {
        println!("  MaxErr = {max_abs_diff:.2e}");
    }
    if snr.is_finite() {
        println!("  SNR    = {snr:.1} dB");
    } else {
        println!("  SNR    = ∞ dB");
    }
    println!("  Samples = {} @ 48 kHz", input.len());

    assert!(
        esr < max_esr,
        "[{label}] ESR={esr:.4e} exceeds threshold {max_esr:.1e} (max sample diff={max_abs_diff:.2e})"
    );
}

// ── Tests ──

#[test]
fn test_linear_golden_short() {
    let mut rng = SimplePcg::new(42);
    let weights = synth_weights_fir(3, &mut rng);
    let bias = 0.1;
    let input = synth_signal(256, &mut rng);
    verify_linear_golden("golden_short_RF=3", &weights, bias, &input, 1e-12);
}

#[test]
fn test_linear_golden_medium() {
    let mut rng = SimplePcg::new(137);
    let weights = synth_weights_fir(16, &mut rng);
    let bias = -0.05;
    let input = synth_signal(1024, &mut rng);
    verify_linear_golden("golden_medium_RF=16", &weights, bias, &input, 1e-12);
}

#[test]
fn test_linear_golden_identity() {
    // Unit weight RF=1 → passthrough
    let weights = vec![1.0];
    let bias = 0.0;
    let mut rng = SimplePcg::new(99);
    let input = synth_signal(512, &mut rng);
    verify_linear_golden("golden_identity_RF=1", &weights, bias, &input, 1e-12);
}

#[test]
fn test_linear_golden_bias_only() {
    // All-zero weights, bias only → constant output
    let weights = vec![0.0; 8];
    let bias = 0.42;
    let mut rng = SimplePcg::new(77);
    let input = synth_signal(256, &mut rng);
    verify_linear_golden("golden_bias_only_RF=8", &weights, bias, &input, 1e-12);
}

#[test]
fn test_linear_golden_known_weights() {
    // Explicit known weights from linear_test.nam (RF=4, bias=0.1)
    let weights_forward = vec![0.125, 0.25, 0.5, 1.0];
    let bias = 0.1;
    let input = generate_sine_440hz(512);
    verify_linear_golden("known_weights_RF=4", &weights_forward, bias, &input, 1e-12);
}

#[test]
#[ignore = "heavy: long receptive field golden test"]
fn test_linear_golden_long() {
    let mut rng = SimplePcg::new(31337);
    let weights = synth_weights_fir(128, &mut rng);
    let bias = 0.02;
    let input = synth_signal(4096, &mut rng);
    verify_linear_golden("golden_long_RF=128", &weights, bias, &input, 1e-12);
}

#[test]
#[ignore = "heavy: very long receptive field golden test"]
fn test_linear_golden_stress() {
    let mut rng = SimplePcg::new(999983);
    let weights = synth_weights_fir(512, &mut rng);
    let bias = 0.01;
    let input = synth_signal(8192, &mut rng);
    verify_linear_golden("golden_stress_RF=512", &weights, bias, &input, 1e-12);
}

#[test]
fn test_linear_bitwise_determinism() {
    let mut rng = SimplePcg::new(777);
    let weights = synth_weights_fir(16, &mut rng);
    let bias = 0.05;
    let input = synth_signal(512, &mut rng);

    let mut model1 = StaticModel::Linear(Box::new(
        LinearModel::new(weights.clone(), bias, LinearImplementation::default())
            .expect("Linear model creation"),
    ));
    let mut model2 = StaticModel::Linear(Box::new(
        LinearModel::new(weights, bias, LinearImplementation::default())
            .expect("Linear model creation"),
    ));

    model1.prewarm(input.len());
    model2.prewarm(input.len());

    let mut out1 = vec![0.0f32; input.len()];
    let mut out2 = vec![0.0f32; input.len()];

    model1.process(&input, &mut out1);
    model2.process(&input, &mut out2);

    assert_eq!(out1.len(), out2.len());
    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "bitwise determinism failed at sample {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_linear_reset_reproduces() {
    let mut rng = SimplePcg::new(256);
    let weights_forward = synth_weights_fir(8, &mut rng);
    let bias = 0.1;
    let input = synth_signal(128, &mut rng);
    let reference = linear_reference(&input, &weights_forward, bias);

    let mut model = StaticModel::Linear(Box::new(
        LinearModel::new(
            weights_forward.clone(),
            bias,
            LinearImplementation::default(),
        )
        .expect("Linear model creation"),
    ));

    model.prewarm(input.len());
    let mut out1 = vec![0.0f32; input.len()];
    model.process(&input, &mut out1);

    model.reset(48000, 512).unwrap();
    model.prewarm(input.len());
    let mut out2 = vec![0.0f32; input.len()];
    model.process(&input, &mut out2);

    for (i, ((vr, o1), o2)) in reference
        .iter()
        .zip(out1.iter())
        .zip(out2.iter())
        .enumerate()
    {
        assert!(
            (o1 - o2).abs() < 1e-6,
            "reset-reproduce failed at sample {i}: {o1:.8e} vs {o2:.8e}"
        );
        assert!(
            (o1 - vr).abs() < 1e-6,
            "reset-reproduce: out1 diverged from oracle at sample {i}: out1={o1:.8e}, ref={vr:.8e}"
        );
    }
}
