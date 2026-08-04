// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Golden convolution tests for the IR Cabsim engine.
//
//  Verifies that the UPOLS convolution engine produces deterministic,
//  bit-identical output for fixed synthetic IRs and input signals.
//  The direct convolution reference serves as the golden baseline.

use super::common;
use common::conv_helpers::{direct_convolve, process_full_signal};
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;

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

// ── Synthetic IR generator ──

fn synth_ir_deterministic(
    len: usize,
    freq: f32,
    decay: f32,
    sample_rate: u32,
    rng: &mut SimplePcg,
) -> Vec<f32> {
    let noise_level = 0.02;
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
                + noise_level * rng.next_f32_signed()
        })
        .collect()
}

// ── Synthetic signal generator ──

fn synth_signal_deterministic(len: usize, rng: &mut SimplePcg) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / 48000.0;
            0.7 * (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.35 * (std::f32::consts::TAU * 554.37 * t).sin()
                + 0.18 * (std::f32::consts::TAU * 880.0 * t).sin()
                + 0.05 * rng.next_f32_signed()
        })
        .collect()
}

// ── Golden verification ──

fn golden_verify(label: &str, ir: &[f32], signal: &[f32], block_size: usize, max_sample_diff: f32) {
    let mut engine = ConvEngine::new(ir, block_size)
        .expect("construction should succeed for test-sized buffers");
    let upols_output = process_full_signal(&mut engine, signal);

    let ref_output = direct_convolve(ir, signal);

    assert!(
        upols_output.len() >= ref_output.len() - engine.partition_size(),
        "{}: UPOLS output {} is too short compared to ref {}",
        label,
        upols_output.len(),
        ref_output.len()
    );

    let min_len = upols_output.len().min(ref_output.len());
    let upols_slice = &upols_output[..min_len];
    let ref_slice = &ref_output[..min_len];

    let mut max_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut sum_sq_ref = 0.0f64;

    for (u, r) in upols_slice.iter().zip(ref_slice.iter()) {
        let err = (u - r).abs();
        if err > max_err {
            max_err = err;
        }
        sum_sq_err += (err as f64) * (err as f64);
        sum_sq_ref += (*r as f64) * (*r as f64);
    }

    let esr = if sum_sq_ref < 1e-30 {
        0.0
    } else {
        sum_sq_err / sum_sq_ref
    };

    assert!(
        max_err <= max_sample_diff,
        "{}: max sample diff {:.2e} exceeds threshold {:.2e}",
        label,
        max_err,
        max_sample_diff
    );
    assert!(
        esr < 1e-5,
        "{}: ESR {:.2e} exceeds 1e-5 threshold",
        label,
        esr
    );
}

// ── Tests ──

#[test]
fn test_cabsim_golden_short() {
    let mut rng = SimplePcg::new(42);
    let ir = synth_ir_deterministic(64, 600.0, 12.0, 48000, &mut rng);
    let signal = synth_signal_deterministic(256, &mut rng);

    golden_verify("golden_short", &ir, &signal, 64, 5e-5);
}

#[test]
fn test_cabsim_golden_medium() {
    let mut rng = SimplePcg::new(137);
    let ir = synth_ir_deterministic(512, 350.0, 6.0, 48000, &mut rng);
    let signal = synth_signal_deterministic(1024, &mut rng);

    golden_verify("golden_medium", &ir, &signal, 64, 1e-4);
}

#[test]
#[ignore = "heavy: long IR golden parity test"]
fn test_cabsim_golden_long() {
    let mut rng = SimplePcg::new(31337);
    let ir = synth_ir_deterministic(8192, 200.0, 2.0, 48000, &mut rng);
    let signal = synth_signal_deterministic(16384, &mut rng);

    golden_verify("golden_long", &ir, &signal, 64, 2e-4);
}

#[test]
#[ignore = "heavy: very long IR + large signal golden test"]
fn test_cabsim_golden_stress() {
    let mut rng = SimplePcg::new(999983);
    let ir = synth_ir_deterministic(32768, 150.0, 1.5, 48000, &mut rng);
    let signal = synth_signal_deterministic(65536, &mut rng);

    golden_verify("golden_stress", &ir, &signal, 256, 5e-4);
}

#[test]
fn test_cabsim_bitwise_determinism() {
    let mut rng = SimplePcg::new(777);
    let ir = synth_ir_deterministic(128, 440.0, 8.0, 48000, &mut rng);
    let signal = synth_signal_deterministic(512, &mut rng);

    let mut engine1 =
        ConvEngine::new(&ir, 64).expect("construction should succeed for test-sized buffers");
    let mut engine2 =
        ConvEngine::new(&ir, 64).expect("construction should succeed for test-sized buffers");

    let out1 = process_full_signal(&mut engine1, &signal);
    let out2 = process_full_signal(&mut engine2, &signal);

    assert_eq!(out1.len(), out2.len());
    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "bitwise determinism failed at sample {}: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_cabsim_passthrough_golden() {
    let engine =
        ConvEngine::new(&[], 64).expect("construction should succeed for test-sized buffers");
    assert!(engine.is_passthrough());

    let mut rng = SimplePcg::new(42);
    let signal = synth_signal_deterministic(256, &mut rng);

    let mut engine2 =
        ConvEngine::new(&[], 64).expect("construction should succeed for test-sized buffers");
    let output = process_full_signal(&mut engine2, &signal);

    let min_len = signal.len().min(output.len());
    let ref_slice = &signal[..min_len];
    let out_slice = &output[..min_len];

    let mut max_err = 0.0f32;
    for (r, o) in ref_slice.iter().zip(out_slice.iter()) {
        let err = (r - o).abs();
        if err > max_err {
            max_err = err;
        }
    }
    assert!(
        max_err < 1e-10,
        "Passthrough golden: max diff = {:.2e}",
        max_err
    );
}
