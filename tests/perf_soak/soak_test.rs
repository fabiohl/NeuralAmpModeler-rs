// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Numerical Stability Suite (Soak Test) for NeuralAmpModeler-rs.
//
//  These tests are designed to run for millions of iterations, checking
//  numerical drift, filter stability, absence of NaNs/Infs, and circular
//  buffer integrity under long-duration execution.
//
//  Execution: `cargo test --release -- --ignored --nocapture`

use neural_amp_modeler_rs::common::params::AdaptiveComputeMode;
use neural_amp_modeler_rs::common::spsc::{
    RT_STATUS_DEGRADE_MINIMAL, RT_STATUS_DEGRADE_REDUCED, RtStatusFlags,
};
use neural_amp_modeler_rs::dsp::adaptive::{AdaptiveCompute, AdaptiveState};
use neural_amp_modeler_rs::dsp::gate::*;
use neural_amp_modeler_rs::dsp::mirror_buf::*;
use neural_amp_modeler_rs::dsp::resampler::*;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::common::half::{f16_bits_to_f32, f32_to_f16_bits};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::models::a2::{A2_KERNEL_SIZES, WaveNetA2, a2_weight_count};
use neural_amp_modeler_rs::models::lstm::*;

use super::common;
use common::alloc_audit::{TrackingGuard, get_alloc_count};
use common::io_helpers::model_path;
use common::model_builders::build_soak_wavenet;
use std::fs;
use std::sync::atomic::Ordering;
use std::time::Instant;

/// Simple deterministic PRNG (Linear Congruential Generator - LCG).
///
/// Implemented manually to keep the project free of rand crate dependencies
/// used only for tests. The goal is to provide a reproducible noise stream
/// for diagnosing failures in soak tests.
struct SimplePcg {
    state: u64,
}

impl SimplePcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generates the next f32 in the range [-1.0, 1.0].
    fn next_f32(&mut self) -> f32 {
        // Simplified variant of PCG-XSH-RR for 32-bit output.
        let old_state = self.state;
        self.state = old_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        let res = (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31));
        // Normalization to the floating-point audio range
        (res as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Soak Test: Infinite Silence Processing via WaveNet.
///
/// The goal is to verify whether error accumulation in feedback models (if any)
/// or f16 precision instability in SIMD kernels causes NaNs or audio
/// "blow-ups" after millions of iterations.
///
/// // Measured: WaveNet silence residue is ~3.58e-5 (−89 dBFS) — NOT a bug.
/// // Origin: conv1d biases (0.001 F32 → tanh(0.001) ≈ 0.001) propagate through
/// // 10+2 layers of one_by_one/rechannel dense projections and accumulate in the
/// // head. F16/BF16 weight quantization contributes a secondary drift (scales
/// // the bias propagation slightly). No denormal involvement — DAZ/FTZ is active
/// // (src/math/common/ops.rs:163, src/clap/processor/mod.rs:268).
/// //
/// // This is faithful to NAMCore C++ v0.5.3 (NAM/dsp.h:67: "don't expect the
/// // model to be outputting zeroes after this"). Biases make tanh(bias) ≠ 0 by
/// // design. Forcing silence to zero would diverge from the C++ bible and break
/// // parity. The interaction with noise-gate/true-bypass belongs to the gate
/// // layer (src/dsp/gate.rs), not the model.
#[test]
#[ignore] // Run manually via --ignored
fn test_wavenet_silence_soak() {
    let mut model = build_soak_wavenet();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        // black_box prevents the compiler from noticing that the input is always zero
        // and optimizing the entire inference loop into a no-op.
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            // Basic numerical integrity check (RT Safety)
            assert!(
                v.is_finite(),
                "Numerical Instability (NaN/Inf) after {} frames",
                processed
            );
            // Ensures the model is not gaining energy out of nowhere (divergence)
            assert!(
                (-2.0..=2.0).contains(&v),
                "Output diverged excessively: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);

    assert!(
        max_val.abs() < 5e-5,
        "Silence residue drifted above expected ceiling (5e-5): max={}",
        max_val
    );
    assert!(
        max_val.abs() > 1e-6,
        "Silence residue unexpectedly near-zero ({}): possible unintended zeroing",
        max_val
    );
}

/// Decomposition Test: isolates the sources of the ~3.6e-5 silence residue.
///
/// Strategy:
/// 1. Run soak wavenet with silence → measure total residue
/// 2. Zero out conv1d biases → re-measure (quantization-only residue)
/// 3. Report the conv1d-bias vs quantization decomposition
///
/// // Measured: conv1d biases dominate 100% of the residue.
/// // With synthetic model (weights=0.01, conv1d_bias=0.001, head_scale=0.1,
/// // 10 layers array1 + 2 layers array2):
/// //   Total residue:        3.58e-5  (−89 dBFS)
/// //   Conv1D bias contrib:  3.58e-5  (−89 dBFS) ← 100% of total
/// //   Quantization drift:   0.00     (zero — 0×weight = 0 regardless of F16 error)
#[test]
fn test_wavenet_silence_decomposition() {
    let input = vec![0.0f32; 64];

    // ── Phase 1: measure total residue ────────────────────────────
    let mut model = build_soak_wavenet();
    model.prewarm();
    let mut output = vec![0.0f32; 64];
    model.process(&input, &mut output);

    let mut total_max = 0.0f32;
    let mut total_sum = 0.0f64;
    for &v in &output {
        total_max = total_max.max(v.abs());
        total_sum += v.abs() as f64;
    }
    let total_mean = total_sum / output.len() as f64;

    println!("--- WaveNet Silence Decomposition ---");
    println!(
        "Phase 1 — Total residue: max={:.6e}, mean={:.6e} ({} dBFS)",
        total_max,
        total_mean,
        if total_max > 0.0 {
            20.0 * (total_max as f64).log10()
        } else {
            f64::NEG_INFINITY
        }
    );

    // Sanity: residue must be in expected range (not zero, not blown)
    assert!(
        total_max > 1e-6,
        "Total residue unexpectedly near-zero ({:.6e})",
        total_max
    );
    assert!(
        total_max < 1e-3,
        "Total residue unexpectedly large ({:.6e})",
        total_max
    );

    // ── Phase 2: zero out conv1d biases → isolate quantization ────
    let mut model_clean = build_soak_wavenet();
    for layer in &mut model_clean.array1.layers {
        for v in &mut *layer.conv1d.bias {
            *v = 0.0;
        }
    }
    for layer in &mut model_clean.array2.layers {
        for v in &mut *layer.conv1d.bias {
            *v = 0.0;
        }
    }
    model_clean.prewarm();

    let mut output_clean = vec![0.0f32; 64];
    model_clean.process(&input, &mut output_clean);

    let mut quant_max = 0.0f32;
    for &v in &output_clean {
        quant_max = quant_max.max(v.abs());
    }

    let bias_contrib = total_max - quant_max;
    let bias_pct = if total_max > 0.0 {
        (bias_contrib / total_max) * 100.0
    } else {
        0.0
    };

    println!(
        "Phase 2 — Quantization-only residue: max={:.6e} ({} dBFS)",
        quant_max,
        if quant_max > 0.0 {
            20.0 * (quant_max as f64).log10()
        } else {
            f64::NEG_INFINITY
        }
    );
    println!(
        "Conv1D bias contribution (diff): {:.6e} ({:.0}% of total)",
        bias_contrib, bias_pct
    );

    // Acceptance: conv1d biases must dominate (>50% of total)
    assert!(
        bias_pct > 50.0,
        "Expected conv1d biases to dominate residue, but they account for only {:.0}% (bias={:.6e}, quant={:.6e})",
        bias_pct,
        bias_contrib,
        quant_max
    );

    println!("Diagnosis confirmed: conv1d biases → tanh(bias) ≠ 0 → head accumulation.");
    println!("This is faithful to NAMCore C++ (dsp.h:67). No action required.");
}

/// Soak Test: White Noise Processing via WaveNet.
///
/// Unlike silence, white noise excites all filter coefficients
/// and all activation branches (Tanh) simultaneously. This is vital
/// for detecting numerical overflows in deep layers that only occur
/// under high signal energy.
#[test]
#[ignore]
fn test_wavenet_noise_soak() {
    let mut model = build_soak_wavenet();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(42);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detected under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let rms = (output.iter().map(|x| (x * x) as f64).sum::<f64>() / output.len() as f64).sqrt();
        assert!(
            rms > 0.0001 && rms < 10.0,
            "RMS do loop out of range: {} at {} frames",
            rms,
            processed
        );
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- WaveNet Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: LSTM state stability under silence.
///
/// LSTMs have recurrent states (cell state) that can accumulate residual errors.
/// This test monitors internal state sanity over millions of iterations.
#[test]
#[ignore]
fn test_lstm_silence_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "NaN/Inf detected in LSTM output after {} frames",
                processed
            );
        }
        // Checks divergence of internal states in the 2-layer architecture
        for &v in model.layer1.cell_state.iter() {
            assert!(v.is_finite(), "Layer 1 LSTM internal state corrupted");
        }
        for &v in model.layer2.cell_state.iter() {
            assert!(v.is_finite(), "Layer 2 LSTM internal state corrupted");
        }

        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
}

/// Soak Test: LSTM resilience under stochastic input.
///
/// Random signals force constant forgetting and updating of the LSTM
/// gates. This test ensures that the "Forget/Input/Output" logic
/// does not cause infinite drift under signal stress conditions.
#[test]
#[ignore]
fn test_lstm_noise_soak() {
    let mut model = LstmModel2::<16, 17, 32, 64>::new();
    let mut pcg = SimplePcg::new(1337);

    for gate in 0..4 {
        for ih in 0..17 {
            for h in 0..16 {
                model.layer1.input_hidden_weights.0[gate][ih][h] = pcg.next_f32() * 0.1 - 0.05;
            }
        }
    }
    for gate in 0..4 {
        for ih in 0..32 {
            for h in 0..16 {
                model.layer2.input_hidden_weights.0[gate][ih][h] = pcg.next_f32() * 0.1 - 0.05;
            }
        }
    }
    for i in 0..64 {
        model.layer1.bias.0[i] = pcg.next_f32() * 0.1 - 0.05;
        model.layer2.bias.0[i] = pcg.next_f32() * 0.1 - 0.05;
    }
    model.head_bias = pcg.next_f32() * 0.1;
    for i in 0..16 {
        model.head_weights[i] = pcg.next_f32() * 0.1 - 0.05;
    }

    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "LSTM diverged under noise after {} frames",
                processed
            );
        }
        let rms = (output.iter().map(|x| (x * x) as f64).sum::<f64>() / output.len() as f64).sqrt();
        assert!(
            rms > 0.0001 && rms < 10.0,
            "LSTM RMS do loop out of range: {} at {} frames",
            rms,
            processed
        );
        processed += 64;
    }
    let duration = start.elapsed();

    println!("--- LSTM Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
}

/// Soak Test: Resampler stability under long-duration asynchronous conversions.
///
/// The resampler uses polyphase interpolation and 24.40 fixed-point phase accumulation.
/// Small rounding errors in the phase increment can accumulate over millions of samples,
/// causing "clicks" or NaNs. This test validates Upsampling and Downsampling scenarios
/// with strict temporal conservation (no silent truncation of output samples).
///
/// Invariants:
/// - Output buffer sized with `NamResampler::min_output_samples` — no truncation.
/// - `processed_out` deviates from `(processed_in * out_sr / in_sr).round()` by at most
///   the documented sinc filter group delay (≤ TAPS_PER_PHASE samples per stage).
#[test]
#[ignore]
fn test_resampler_drift_soak() {
    test_resampler_drift_soak_scenario(22050, 48000, "Upsampling");
    test_resampler_drift_soak_scenario(96000, 48000, "Downsampling");
}

fn test_resampler_drift_soak_scenario(in_sr: u32, out_sr: u32, label: &str) {
    const TAPS_PER_PHASE: usize = 64;
    const BLOCK_SIZE: usize = 1024;
    const NUM_SAMPLES: usize = 50_000_000;

    let mut resampler = NamResampler::new(in_sr, out_sr, 64).unwrap();
    let mut pcg = SimplePcg::new(777);

    let out_cap = NamResampler::min_output_samples(BLOCK_SIZE, in_sr, out_sr);

    let mut processed_in: usize = 0;
    let mut processed_out: usize = 0;

    let mut in_l = vec![0.0f32; BLOCK_SIZE];
    let mut in_r = vec![0.0f32; BLOCK_SIZE];
    let mut out_l = vec![0.0f32; out_cap];
    let mut out_r = vec![0.0f32; out_cap];

    let start = Instant::now();
    while processed_in < NUM_SAMPLES {
        for i in 0..BLOCK_SIZE {
            in_l[i] = pcg.next_f32();
            in_r[i] = pcg.next_f32();
        }
        let progress = resampler.process_input(
            std::hint::black_box(&in_l),
            std::hint::black_box(&in_r),
            std::hint::black_box(&mut out_l),
            std::hint::black_box(&mut out_r),
        );
        let n = progress.samples_written;
        let r = progress.samples_read;

        assert!(
            r > 0,
            "{}: zero samples consumed at processed_in={}",
            label,
            processed_in
        );
        assert!(
            n > 0,
            "{}: zero samples produced at processed_in={}",
            label,
            processed_in
        );
        assert!(
            n <= out_cap,
            "{}: output overflow: {} > cap {}",
            label,
            n,
            out_cap
        );

        processed_in += r;
        processed_out += n;

        for i in 0..n {
            assert!(out_l[i].is_finite(), "NaN in Resampler Out L ({label})");
            assert!(out_r[i].is_finite(), "NaN in Resampler Out R ({label})");
        }
        let rms_l = (out_l[..n].iter().map(|x| (x * x) as f64).sum::<f64>() / n as f64).sqrt();
        let rms_r = (out_r[..n].iter().map(|x| (x * x) as f64).sum::<f64>() / n as f64).sqrt();
        assert!(
            rms_l > 0.0001 && rms_l < 10.0,
            "Resampler {label} L RMS do loop out of range: {}",
            rms_l
        );
        assert!(
            rms_r > 0.0001 && rms_r < 10.0,
            "Resampler {label} R RMS do loop out of range: {}",
            rms_r
        );
    }

    let expected_out = (processed_in as f64 * out_sr as f64 / in_sr as f64).round() as usize;
    let delta = processed_out.max(expected_out) - processed_out.min(expected_out);
    let tolerance = TAPS_PER_PHASE;

    let duration = start.elapsed();

    println!("╔══════════════════════════════════════════════════════╗");
    println!(
        "║  Resampler Drift Soak ({}→{}) - {label:<8}   ║",
        in_sr, out_sr
    );
    println!("╠══════════════════════════════════════════════════════╣");
    println!(
        "║  Duration:     {}                     ║",
        format_args!("{:>12?}", duration)
    );
    println!(
        "║  Samples In:     {:>12}                         ║",
        processed_in
    );
    println!(
        "║  Expected Out:   {:>12}                         ║",
        expected_out
    );
    println!(
        "║  Actual Out:     {:>12}                         ║",
        processed_out
    );
    println!("║  Delta:          {:>12}                         ║", delta);
    println!(
        "║  Tolerance:      {:>12} (TAPS_PER_PHASE)        ║",
        tolerance
    );
    println!(
        "║  Verdict:        {:>12}                         ║",
        if delta <= tolerance { "PASS" } else { "FAIL" }
    );
    println!("╚══════════════════════════════════════════════════════╝");

    assert!(
        delta <= tolerance,
        "Temporal conservation violated ({}→{}): delta={} exceeds tolerance={} (TAPS_PER_PHASE)\n\
         in={} expected={} actual={}",
        in_sr,
        out_sr,
        delta,
        tolerance,
        processed_in,
        expected_out,
        processed_out
    );

    let min_expected_out = processed_in.saturating_mul(out_sr as usize) / in_sr as usize;
    let max_expected_out = min_expected_out + tolerance;
    assert!(
        processed_out >= min_expected_out && processed_out <= max_expected_out,
        "{}→{}: output {} outside [{}, {}] — possible silent truncation",
        in_sr,
        out_sr,
        processed_out,
        min_expected_out,
        max_expected_out
    );

    if in_sr == 22050 && out_sr == 48000 {
        assert!(
            processed_out > 108_000_000,
            "22050→48000 soak: output {} has stalled far below expected ~108.8M — suspected truncation bug",
            processed_out
        );
    }
}

/// Soak Test: MirroredBuffer memory integrity.
///
/// MirroredBuffer uses mmap to mirror memory, enabling continuous linear
/// accesses that cross the buffer boundary. This test verifies that mirroring
/// remains consistent after billions of writes.
#[test]
#[ignore]
#[cfg(target_os = "linux")]
fn test_mirror_buf_long_run() {
    let mut buf = MirroredBuffer::<f32>::new(1024 * 1024).expect("Failed to create MirroredBuffer"); // 1M elements
    let size = buf.size();
    let num_cycles = 100_000_000;
    let mut pos = 0;
    let chunk = 64;

    let start = Instant::now();
    for i in 0..(num_cycles / chunk) {
        // Simulates write and advance in the mirrored buffer
        for j in 0..chunk {
            std::hint::black_box(&mut buf)[pos + j] = std::hint::black_box((i * chunk + j) as f32);
        }

        // Checks integrity at the boundary: the value written at the end
        // must appear identically at the beginning (mmap mirroring).
        if pos + chunk >= size {
            let offset = (pos + chunk) - size;
            for j in 0..offset {
                assert_eq!(
                    buf[j],
                    buf[size + j],
                    "Critical memory mirroring failure in MirroredBuffer at index {}",
                    j
                );
            }
        }

        pos += chunk;
        if pos >= size {
            pos -= size;
        }
    }
    let duration = start.elapsed();

    println!("--- MirroredBuffer Long Run ---");
    println!("Duration: {:?}", duration);
    println!("Cycles (samples): {}", num_cycles);
}

/// Soak Test: Noise Gate State Machine (FSM) endurance.
///
/// This test forces rapid alternations between the `Open`, `Hold`, and `Release` states
/// using a stochastic binary signal. The goal is to ensure that the gain
/// multiplier never leaves the [0, 1] range and that the FSM does not lock up
/// in invalid states after millions of transitions.
#[test]
#[ignore]
fn test_gate_fsm_endurance() {
    let mut gate = DynamicHysteresis::new();
    let params = GateParams::default();
    let mut pcg = SimplePcg::new(999);
    let num_alternations = 10_000_000; // 10 million transition cycles

    // Typical real-world thresholds (-60dB and -80dB)
    let threshold_open = -60.0f32.powf(10.0 / 20.0);
    let threshold_close = -80.0f32.powf(10.0 / 20.0);

    let start = Instant::now();
    for _ in 0..num_alternations {
        // Aggressively alternates between full signal (1.0) and silence (0.0)
        let val = if pcg.next_f32() > 0.0 { 1.0 } else { 0.0 };
        gate.update(
            std::hint::black_box(val),
            threshold_open,
            threshold_close,
            &params,
            64,
        );

        // The gate's gain multiplier must be strictly well-behaved [0.0, 1.0]
        let m = std::hint::black_box(gate.multiplier());
        assert!(
            (0.0..=1.0).contains(&m),
            "Gate multiplier diverged: {} outside [0, 1]",
            m
        );
    }
    let duration = start.elapsed();

    println!("--- Gate FSM Endurance ---");
    println!("Duration: {:?}", duration);
    println!("Alternations: {}", num_alternations);
}

// =============================================================================
// A2 Architecture Soak Tests — WaveNetA2 numerical stability
// =============================================================================

/// Assembles synthetic A2 weight stream.
fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
    let num = a2_weight_count::<CH>();
    let mut w = Vec::with_capacity(num);
    w.extend(std::iter::repeat_n(weight_val, CH));
    for &k in &A2_KERNEL_SIZES {
        w.extend(std::iter::repeat_n(weight_val, CH * CH * k));
        w.extend(std::iter::repeat_n(0.0f32, CH));
        w.extend(std::iter::repeat_n(weight_val, CH));
        w.extend(std::iter::repeat_n(weight_val, CH * CH));
        w.extend(std::iter::repeat_n(0.0f32, CH));
    }
    w.extend(std::iter::repeat_n(weight_val, 16 * CH));
    w.push(0.0);
    w.push(0.02);
    assert_eq!(w.len(), num);
    w
}

/// Builds a synthetically-weighted A2 model.
fn build_soak_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
    let weights = a2_synth_weights::<CH>(weight_val);
    let mut model = WaveNetA2::<CH>::new().expect("Failed to create WaveNetA2");
    model.set_weights(&weights).expect("A2 set_weights failed");
    model
}

/// Soak Test: A2-Full silence processing — 10M frames.
#[test]
#[ignore]
fn test_a2_full_silence_soak() {
    let mut model = build_soak_a2::<8>(0.01);
    model.prewarm();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(v.is_finite(), "A2-Full NaN/Inf after {} frames", processed);
            assert!(
                (-2.0..=2.0).contains(&v),
                "A2-Full diverged: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Full Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Lite silence processing — 10M frames.
#[test]
#[ignore]
fn test_a2_lite_silence_soak() {
    let mut model = build_soak_a2::<3>(0.01);
    model.prewarm();
    let input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(v.is_finite(), "A2-Lite NaN/Inf after {} frames", processed);
            assert!(
                (-2.0..=2.0).contains(&v),
                "A2-Lite diverged: {} after {} frames",
                v,
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Lite Silence Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Full white noise processing — 10M frames.
#[test]
#[ignore]
fn test_a2_full_noise_soak() {
    let mut model = build_soak_a2::<8>(0.01);
    model.prewarm();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(42);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "A2-Full NaN/Inf under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let rms = (output.iter().map(|x| (x * x) as f64).sum::<f64>() / output.len() as f64).sqrt();
        assert!(
            rms > 0.0001 && rms < 10.0,
            "A2-Full Noise RMS do loop out of range: {} at {} frames",
            rms,
            processed
        );
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Full Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

/// Soak Test: A2-Lite white noise processing — 10M frames.
#[test]
#[ignore]
fn test_a2_lite_noise_soak() {
    let mut model = build_soak_a2::<3>(0.01);
    model.prewarm();
    let mut input = vec![0.0f32; 64];
    let mut output = vec![0.0f32; 64];
    let mut pcg = SimplePcg::new(1337);
    let num_frames = 10_000_000;
    let mut processed = 0;
    let mut min_val = 0.0f32;
    let mut max_val = 0.0f32;

    let start = Instant::now();
    while processed < num_frames {
        for v in &mut input {
            *v = pcg.next_f32();
        }
        model.process(
            std::hint::black_box(&input),
            std::hint::black_box(&mut output),
        );
        for &v in &output {
            assert!(
                v.is_finite(),
                "A2-Lite NaN/Inf under noise after {} frames",
                processed
            );
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        let rms = (output.iter().map(|x| (x * x) as f64).sum::<f64>() / output.len() as f64).sqrt();
        assert!(
            rms > 0.0001 && rms < 10.0,
            "A2-Lite Noise RMS do loop out of range: {} at {} frames",
            rms,
            processed
        );
        processed += 64;
    }
    let duration = start.elapsed();
    println!("--- A2-Lite Noise Soak ---");
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Min output: {}", min_val);
    println!("Max output: {}", max_val);
}

// =============================================================================
// Adaptive Compute FSM Endurance Soak
// =============================================================================

/// Soak Test: Adaptive Compute FSM endurance.
///
/// Forces millions of alternating overload/recovery cycles to stress the
/// hysteresis FSM, crossfade logic, and telemetry (RtStatusFlags).
/// Verifies that transitions are deterministic, the FSM never locks up,
/// and the degrade_transitions_total counter is consistent.
///
/// Runs with both Conservative and Aggressive modes under randomized
/// latency patterns.
#[test]
#[ignore]
fn test_adaptive_fsm_endurance() {
    const NUM_CYCLES: usize = 2_000_000;
    const SAMPLE_RATE: u32 = 48000;
    const BLOCK_SIZE: usize = 64;

    let mut pcg = SimplePcg::new(12345);

    for (mode, label) in [
        (AdaptiveComputeMode::Conservative, "Conservative"),
        (AdaptiveComputeMode::Aggressive, "Aggressive"),
    ] {
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);

        let (full_to_reduced, reduced_to_minimal) = match mode {
            AdaptiveComputeMode::Conservative => (0.70, 0.85),
            AdaptiveComputeMode::Aggressive => (0.55, 0.70),
            AdaptiveComputeMode::Off => unreachable!(),
        };

        let recovery_reduced = full_to_reduced * 0.5;
        let recovery_minimal = reduced_to_minimal * 0.5;

        let start = Instant::now();

        for i in 0..NUM_CYCLES {
            let budget_us = 1000 + (pcg.next_f32().abs() * 9000.0) as u64;
            let ratio = match i % 12 {
                0..=3 => full_to_reduced * 1.2,    // Overload zone (degrade trigger)
                4..=6 => recovery_reduced * 0.8,   // Recovery zone
                7..=8 => reduced_to_minimal * 1.2, // Extreme overload (minimal trigger)
                9..=11 => recovery_minimal * 0.8,  // Deep recovery
                _ => unreachable!(),
            };

            let latency_us = ((budget_us as f64) * ratio as f64).ceil() as u64;
            let latency_us = std::hint::black_box(latency_us);

            adaptive.update(latency_us, budget_us, SAMPLE_RATE, &rt_status);

            // Advance crossfade if active
            if adaptive.is_crossfading() {
                let mult = adaptive.crossfade_multiplier(SAMPLE_RATE, BLOCK_SIZE);
                assert!(
                    (0.0..=1.0).contains(&mult),
                    "Crossfade multiplier {} out of [0, 1]",
                    mult
                );
            }

            // State and flags consistency
            let state = adaptive.state();
            let is_reduced = rt_status.check_flag(RT_STATUS_DEGRADE_REDUCED);
            let is_minimal = rt_status.check_flag(RT_STATUS_DEGRADE_MINIMAL);

            match state {
                AdaptiveState::Full => {
                    assert!(
                        !is_reduced && !is_minimal,
                        "{}: Full state but flags set (R={}, M={}) at cycle {}",
                        label,
                        is_reduced,
                        is_minimal,
                        i
                    );
                }
                AdaptiveState::Reduced => {
                    assert!(
                        is_reduced && !is_minimal,
                        "{}: Reduced state but flags wrong (R={}, M={}) at cycle {}",
                        label,
                        is_reduced,
                        is_minimal,
                        i
                    );
                }
                AdaptiveState::Minimal => {
                    assert!(
                        is_reduced && is_minimal,
                        "{}: Minimal state but flags wrong (R={}, M={}) at cycle {}",
                        label,
                        is_reduced,
                        is_minimal,
                        i
                    );
                }
            }

            // slimmable_size() bounds
            let sz = adaptive.slimmable_size();
            assert!(
                (0.0..=1.0).contains(&sz),
                "{}: slimmable_size {} out of [0, 1] at cycle {}",
                label,
                sz,
                i
            );

            // wavenet_effective_layers bounds
            let el = adaptive.wavenet_effective_layers(8);
            assert!(
                (1..=8).contains(&el),
                "{}: wavenet_effective_layers(8) = {} out of [1, 8] at cycle {}",
                label,
                el,
                i
            );
        }

        let duration = start.elapsed();
        let transitions = rt_status.degrade_transitions_total.load(Ordering::Relaxed);

        println!("--- Adaptive FSM Endurance ({}) ---", label);
        println!("Duration: {:?}", duration);
        println!("Cycles: {}", NUM_CYCLES);
        println!("Total transitions: {}", transitions);

        // Under such aggressive jitter, transitions MUST have occurred
        assert!(
            transitions > 0,
            "{}: No transitions occurred under aggressive jitter in {} cycles",
            label,
            NUM_CYCLES
        );

        // Transitions should be reasonable (not stuck in infinite loop)
        assert!(
            transitions < NUM_CYCLES as u32 / 3,
            "{}: Excessive transitions {} detected — possible FSM oscillation",
            label,
            transitions
        );
    }
}

/// Soak Test: Adaptive Compute FSM deterministic transition endurance.
///
/// Forces repeated complete degradation cycles (Full→Reduced→Minimal→Reduced→Full)
/// to stress the full hysteresis FSM, crossfade logic, and flag consistency under
/// a known deterministic pattern. Each cycle produces 4 transitions.
#[test]
#[ignore]
fn test_adaptive_fsm_transition_cycles() {
    const CYCLES: usize = 50_000;
    const SAMPLE_RATE: u32 = 48000;
    const BLOCK_SIZE: usize = 64;
    const BUDGET_US: u64 = 1000;

    for (mode, label) in [
        (AdaptiveComputeMode::Conservative, "Conservative"),
        (AdaptiveComputeMode::Aggressive, "Aggressive"),
    ] {
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(mode);

        let (full_to_reduced, reduced_to_minimal) = match mode {
            AdaptiveComputeMode::Conservative => (0.70, 0.85),
            AdaptiveComputeMode::Aggressive => (0.55, 0.70),
            AdaptiveComputeMode::Off => unreachable!(),
        };

        let recovery_reduced = full_to_reduced * 0.5;
        let recovery_minimal = reduced_to_minimal * 0.5;

        let overload_reduced = BUDGET_US as f64 * full_to_reduced * 1.1;
        let overload_minimal = BUDGET_US as f64 * reduced_to_minimal * 1.1;
        let undervolt_reduced = BUDGET_US as f64 * recovery_reduced * 0.9;
        let undervolt_minimal = BUDGET_US as f64 * recovery_minimal * 0.9;

        let start = Instant::now();

        for _cycle in 0..CYCLES {
            // Phase 1: Degrade Full → Reduced (3 overloads)
            for _ in 0..3 {
                adaptive.update(
                    overload_reduced.ceil() as u64,
                    BUDGET_US,
                    SAMPLE_RATE,
                    &rt_status,
                );
                if adaptive.is_crossfading() {
                    let _ = adaptive.crossfade_multiplier(SAMPLE_RATE, BLOCK_SIZE);
                }
            }

            // Phase 2: Degrade Reduced → Minimal (3 overloads)
            for _ in 0..3 {
                adaptive.update(
                    overload_minimal.ceil() as u64,
                    BUDGET_US,
                    SAMPLE_RATE,
                    &rt_status,
                );
                if adaptive.is_crossfading() {
                    let _ = adaptive.crossfade_multiplier(SAMPLE_RATE, BLOCK_SIZE);
                }
            }

            // Phase 3: Recover Minimal → Reduced (5 undervolts)
            for _ in 0..5 {
                adaptive.update(
                    undervolt_minimal.floor() as u64,
                    BUDGET_US,
                    SAMPLE_RATE,
                    &rt_status,
                );
                if adaptive.is_crossfading() {
                    let _ = adaptive.crossfade_multiplier(SAMPLE_RATE, BLOCK_SIZE);
                }
            }

            // Phase 4: Recover Reduced → Full (5 undervolts)
            for _ in 0..5 {
                adaptive.update(
                    undervolt_reduced.floor() as u64,
                    BUDGET_US,
                    SAMPLE_RATE,
                    &rt_status,
                );
                if adaptive.is_crossfading() {
                    let _ = adaptive.crossfade_multiplier(SAMPLE_RATE, BLOCK_SIZE);
                }
            }

            // Verify invariants at end of each cycle
            let state = adaptive.state();
            assert_eq!(
                state,
                AdaptiveState::Full,
                "{}: expected Full at end of cycle {}, got {:?}",
                label,
                _cycle,
                state
            );
        }

        let duration = start.elapsed();
        let transitions = rt_status.degrade_transitions_total.load(Ordering::Relaxed);

        // 4 transitions per cycle
        let expected = CYCLES as u32 * 4;
        assert_eq!(
            transitions, expected,
            "{}: expected {} transitions, got {}",
            label, expected, transitions
        );

        println!("--- Adaptive FSM Transition Cycles ({}) ---", label);
        println!("Duration: {:?}", duration);
        println!("Cycles: {}", CYCLES);
        println!("Transitions: {} (expected {})", transitions, expected);
    }
}

// =============================================================================
// T1.4 — WaveNet Drift Source Decomposition
// =============================================================================

/// Padé [5,4] scalar approximation of tanh. Max error ~2.32e-3.
fn pade_tanh_scalar(x: f32) -> f32 {
    let x = x.clamp(-4.0, 4.0);
    let x2 = x * x;
    let num = x * ((x2 + 105.0) * x2 + 945.0);
    let den = (15.0 * x2 + 420.0) * x2 + 945.0;
    (num / den).clamp(-1.0, 1.0)
}

/// Applies tanh to a slice. Uses Padé [5,4] or exact `f32::tanh`.
fn apply_tanh_scalar(slice: &mut [f32], use_exact: bool) {
    if use_exact {
        for v in slice {
            *v = v.tanh();
        }
    } else {
        for v in slice {
            *v = pade_tanh_scalar(*v);
        }
    }
}

/// Scalar Dense (GEMV): output = weights * input + bias (+ optional residual).
#[expect(
    clippy::too_many_arguments,
    reason = "Performance soak test helper requiring many configuration parameters for comprehensive stress testing"
)]
fn dense_scalar(
    w: &[f32],
    bias: &[f32],
    do_bias: bool,
    in_ch: usize,
    out_ch: usize,
    input: &[f32],
    residual: Option<&[f32]>,
    output: &mut [f32],
    nf: usize,
) {
    for f in 0..nf {
        let o = f * out_ch;
        let i = f * in_ch;
        for oc in 0..out_ch {
            let mut s = if do_bias && oc < bias.len() {
                bias[oc]
            } else {
                0.0
            };
            for ic in 0..in_ch {
                s += input[i + ic] * w[oc * in_ch + ic];
            }
            if let Some(r) = residual {
                s += r[o + oc];
            }
            output[o + oc] = s;
        }
    }
}

/// Scalar Dense with F16 (u16) weights.
#[expect(
    clippy::too_many_arguments,
    reason = "Performance soak test helper requiring many configuration parameters for comprehensive stress testing"
)]
fn dense_scalar_f16(
    w: &[u16],
    bias: &[f32],
    do_bias: bool,
    in_ch: usize,
    out_ch: usize,
    input: &[f32],
    residual: Option<&[f32]>,
    output: &mut [f32],
    nf: usize,
) {
    for f in 0..nf {
        let o = f * out_ch;
        let i = f * in_ch;
        for oc in 0..out_ch {
            let mut s = if do_bias && oc < bias.len() {
                bias[oc]
            } else {
                0.0
            };
            for ic in 0..in_ch {
                s += input[i + ic] * f16_bits_to_f32(w[oc * in_ch + ic]);
            }
            if let Some(r) = residual {
                s += r[o + oc];
            }
            output[o + oc] = s;
        }
    }
}

/// Scalar dilated causal Conv1D with f32 weights.
#[expect(
    clippy::too_many_arguments,
    reason = "Performance soak test helper requiring many configuration parameters for comprehensive stress testing"
)]
fn conv1d_scalar(
    w: &[f32],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    ring_buf: &[f32],
    buf_pos: usize,
    mixin: Option<&[f32]>,
    output: &mut [f32],
    nf: usize,
) {
    for f in 0..nf {
        let fi = buf_pos + f;
        let o = f * out_ch;
        for oc in 0..out_ch {
            let mut s = if do_bias && oc < bias.len() {
                bias[oc]
            } else {
                0.0
            };
            if let Some(m) = mixin
                && f * out_ch + oc < m.len()
            {
                s += m[f * out_ch + oc];
            }
            output[o + oc] = s;
        }
        for tk in 0..k {
            let off = (dilation as isize) * ((tk as isize) + 1 - (k as isize));
            let pos = ((fi as isize) + off) as usize * in_ch;
            for oc in 0..out_ch {
                let mut s = 0.0f32;
                for ic in 0..in_ch {
                    s += ring_buf[pos + ic] * w[oc * k * in_ch + tk * in_ch + ic];
                }
                output[o + oc] += s;
            }
        }
    }
}

/// Scalar dilated causal Conv1D with F16 (u16) weights.
#[expect(
    clippy::too_many_arguments,
    reason = "Performance soak test helper requiring many configuration parameters for comprehensive stress testing"
)]
fn conv1d_scalar_f16(
    w: &[u16],
    bias: &[f32],
    do_bias: bool,
    dilation: usize,
    in_ch: usize,
    out_ch: usize,
    k: usize,
    ring_buf: &[f32],
    buf_pos: usize,
    mixin: Option<&[f32]>,
    output: &mut [f32],
    nf: usize,
) {
    for f in 0..nf {
        let fi = buf_pos + f;
        let o = f * out_ch;
        for oc in 0..out_ch {
            let mut s = if do_bias && oc < bias.len() {
                bias[oc]
            } else {
                0.0
            };
            if let Some(m) = mixin
                && f * out_ch + oc < m.len()
            {
                s += m[f * out_ch + oc];
            }
            output[o + oc] = s;
        }
        for tk in 0..k {
            let off = (dilation as isize) * ((tk as isize) + 1 - (k as isize));
            let pos = ((fi as isize) + off) as usize * in_ch;
            for oc in 0..out_ch {
                let mut s = 0.0f32;
                for ic in 0..in_ch {
                    s += ring_buf[pos + ic] * f16_bits_to_f32(w[oc * k * in_ch + tk * in_ch + ic]);
                }
                output[o + oc] += s;
            }
        }
    }
}

/// Scalar reference engine for WaveNet Standard (CH=16, K=3, HEAD=8, 10+2 layers).
///
/// Mirrors `build_soak_wavenet()` topology with configurable precision:
/// - `use_f32` → f32 weights, else F16 (u16→f32 decoded)
/// - `use_exact` → `f32::tanh()`, else Padé [5,4]
///
/// Ring buffers are pre-filled with zeros (equivalent to silence prewarm).
fn wavenet_standard_scalar(use_f32: bool, use_exact: bool, input: &[f32], nf: usize) -> Vec<f32> {
    let fw = 0.01f32;
    let uw = f32_to_f16_bits(fw);

    // ── Topology ─────────────────────────────────────────────
    let dil_a1: [usize; 10] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512];
    let dil_a2: [usize; 2] = [1, 2];
    let rf_a1: usize = dil_a1.iter().map(|&d| 2 * d).sum();
    let rf_a2: usize = dil_a2.iter().map(|&d| 2 * d).sum();

    // ── Weights: Array1 layers ────────────────────────────────
    let a1_mix_f32: Vec<f32> = vec![fw; 16];
    let a1_mix_u16: Vec<u16> = vec![uw; 16];
    let a1_o2o_f32: Vec<f32> = vec![fw; 16 * 16];
    let a1_o2o_u16: Vec<u16> = vec![uw; 16 * 16];
    let a1_cb: Vec<f32> = vec![0.001; 16];
    let a1_cw_f32: Vec<f32> = vec![fw; 16 * 3 * 16];
    let a1_cw_u16: Vec<u16> = vec![uw; 16 * 3 * 16];

    // ── Weights: Array1 head ──────────────────────────────────
    let a1_rec_f32: Vec<f32> = vec![fw; 16];
    let a1_rec_u16: Vec<u16> = vec![uw; 16];
    let a1_hd_f32: Vec<f32> = vec![fw; 8 * 16];
    let a1_hd_u16: Vec<u16> = vec![uw; 8 * 16];

    // ── Weights: Array2 layers ────────────────────────────────
    let a2_mix_f32: Vec<f32> = vec![fw; 8];
    let a2_mix_u16: Vec<u16> = vec![uw; 8];
    let a2_o2o_f32: Vec<f32> = vec![fw; 8 * 8];
    let a2_o2o_u16: Vec<u16> = vec![uw; 8 * 8];
    let a2_cb: Vec<f32> = vec![0.001; 8];
    let a2_cw_f32: Vec<f32> = vec![fw; 8 * 3 * 8];
    let a2_cw_u16: Vec<u16> = vec![uw; 8 * 3 * 8];

    // ── Weights: Array2 head ──────────────────────────────────
    let a2_rec_f32: Vec<f32> = vec![fw; 8 * 16];
    let a2_rec_u16: Vec<u16> = vec![uw; 8 * 16];
    let a2_hd_f32: Vec<f32> = vec![fw; 8];
    let a2_hd_u16: Vec<u16> = vec![uw; 8];

    // ── Ring buffers ──────────────────────────────────────────
    let make_bufs = |n: usize, ch: usize, rf: usize| -> Vec<Vec<f32>> {
        (0..n).map(|_| vec![0.0f32; (rf + nf) * ch]).collect()
    };
    let mut a1b: Vec<Vec<f32>> = make_bufs(11, 16, rf_a1);
    let mut a2b: Vec<Vec<f32>> = make_bufs(3, 8, rf_a2);

    let mut a1h = vec![0.0f32; nf * 16];
    let mut a2h = vec![0.0f32; nf * 8];
    let mut a1ho = vec![0.0f32; nf * 8];

    // helper: select f32 or u16 reference
    macro_rules! dsel {
        ($fw:expr, $uw:expr, $bias:expr, $db:expr, $in:expr, $out:expr, $src:expr, $res:expr, $dst:expr) => {
            if use_f32 {
                dense_scalar($fw, $bias, $db, $in, $out, $src, $res, $dst, nf);
            } else {
                dense_scalar_f16($uw, $bias, $db, $in, $out, $src, $res, $dst, nf);
            }
        };
    }

    // ── Array1 rechannel ─────────────────────────────────────
    let rf0 = 2 * dil_a1[0];
    dsel!(
        &a1_rec_f32,
        &a1_rec_u16,
        &[0.0f32; 16],
        false,
        1,
        16,
        input,
        None,
        &mut a1b[0][rf0 * 16..(rf0 + nf) * 16]
    );

    // ── Array1 layers ─────────────────────────────────────────
    for li in 0..10 {
        let dil = dil_a1[li];
        let rfi = 2 * dil;
        let rfo = if li < 9 { 2 * dil_a1[li + 1] } else { rfi };
        let (l, r) = a1b.split_at_mut(li + 1);
        let ib = &l[li];
        let ob = &mut r[0];

        // Per-frame mixin: COND=1 × CH
        let mut mixin = vec![0.0f32; nf * 16];
        for ff in 0..nf {
            let cv = input[ff];
            for oc in 0..16 {
                let mw = if use_f32 {
                    a1_mix_f32[oc]
                } else {
                    f16_bits_to_f32(a1_mix_u16[oc])
                };

                mixin[ff * 16 + oc] = mw * cv;
            }
        }
        let mut co = vec![0.0f32; nf * 16];
        if use_f32 {
            conv1d_scalar(
                &a1_cw_f32,
                &a1_cb,
                true,
                dil,
                16,
                16,
                3,
                ib,
                rfi,
                Some(&mixin),
                &mut co,
                nf,
            );
        } else {
            conv1d_scalar_f16(
                &a1_cw_u16,
                &a1_cb,
                true,
                dil,
                16,
                16,
                3,
                ib,
                rfi,
                Some(&mixin),
                &mut co,
                nf,
            );
        }
        apply_tanh_scalar(&mut co, use_exact);

        let res = &ib[rfi * 16..(rfi + nf) * 16];
        dsel!(
            &a1_o2o_f32,
            &a1_o2o_u16,
            &[0.0f32; 16],
            false,
            16,
            16,
            &co,
            Some(res),
            &mut ob[rfo * 16..(rfo + nf) * 16]
        );

        if li == 0 {
            a1h.copy_from_slice(&co);
        } else {
            for (h, c) in a1h.iter_mut().zip(co.iter()) {
                *h += c;
            }
        }
    }

    // ── Array1 head ──────────────────────────────────────────
    dsel!(
        &a1_hd_f32,
        &a1_hd_u16,
        &[0.0f32; 8],
        false,
        16,
        8,
        &a1h,
        None,
        &mut a1ho
    );

    // ── Array1 output ────────────────────────────────────────
    let a1o_rf = 2 * dil_a1[9];
    let a1o = &a1b[10][a1o_rf * 16..(a1o_rf + nf) * 16];

    // ── Array2 ───────────────────────────────────────────────
    a2h.copy_from_slice(&a1ho);

    let rf2_0 = 2 * dil_a2[0];
    dsel!(
        &a2_rec_f32,
        &a2_rec_u16,
        &[0.0f32; 8],
        false,
        16,
        8,
        a1o,
        None,
        &mut a2b[0][rf2_0 * 8..(rf2_0 + nf) * 8]
    );

    for li in 0..2 {
        let dil = dil_a2[li];
        let rfi = 2 * dil;
        let rfo = if li < 1 { 2 * dil_a2[li + 1] } else { rfi };
        let (l, r) = a2b.split_at_mut(li + 1);
        let ib = &l[li];
        let ob = &mut r[0];

        let mut mixin = vec![0.0f32; nf * 8];
        for ff in 0..nf {
            let cv = input[ff];
            for oc in 0..8 {
                let mw = if use_f32 {
                    a2_mix_f32[oc]
                } else {
                    f16_bits_to_f32(a2_mix_u16[oc])
                };
                mixin[ff * 8 + oc] = mw * cv;
            }
        }
        let mut co = vec![0.0f32; nf * 8];
        if use_f32 {
            conv1d_scalar(
                &a2_cw_f32,
                &a2_cb,
                true,
                dil,
                8,
                8,
                3,
                ib,
                rfi,
                Some(&mixin),
                &mut co,
                nf,
            );
        } else {
            conv1d_scalar_f16(
                &a2_cw_u16,
                &a2_cb,
                true,
                dil,
                8,
                8,
                3,
                ib,
                rfi,
                Some(&mixin),
                &mut co,
                nf,
            );
        }
        apply_tanh_scalar(&mut co, use_exact);

        let res = &ib[rfi * 8..(rfi + nf) * 8];
        dsel!(
            &a2_o2o_f32,
            &a2_o2o_u16,
            &[0.0f32; 8],
            false,
            8,
            8,
            &co,
            Some(res),
            &mut ob[rfo * 8..(rfo + nf) * 8]
        );

        for (h, c) in a2h.iter_mut().zip(co.iter()) {
            *h += c;
        }
    }

    // ── Array2 head → output ─────────────────────────────────
    let mut out = vec![0.0f32; nf];
    dsel!(
        &a2_hd_f32,
        &a2_hd_u16,
        &[0.0f32; 1],
        true,
        8,
        1,
        &a2h,
        None,
        &mut out
    );

    for v in &mut out {
        *v *= 0.1;
    }
    out
}

/// Computes Error-to-Signal Ratio (linear).
fn compute_esr(reference: &[f32], test: &[f32]) -> f64 {
    let sig: f64 = reference.iter().map(|&x| (x as f64).powi(2)).sum();
    let noise: f64 = reference
        .iter()
        .zip(test.iter())
        .map(|(r, t)| ((r - t) as f64).powi(2))
        .sum();
    if sig <= f64::EPSILON {
        if noise <= f64::EPSILON {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        noise / sig
    }
}

fn esr_to_db(esr: f64) -> f64 {
    if esr <= 0.0 {
        f64::INFINITY
    } else {
        10.0 * esr.log10()
    }
}

/// T1.4 — Drift Source Decomposition for WaveNet Standard (CH=16).
///
/// Measures the contribution of each precision-loss source to the total
/// ESR against a full-precision (f32 weights + exact tanh) scalar reference:
///   (a) F16 weight quantization
///   (b) tanh Padé [5,4] approximation
///   (c) f32 accumulation (residual)
///
/// Does NOT alter the production engine — all measurement is via a
/// self-contained scalar reference engine in this test file.
#[test]
#[ignore]
fn test_wavenet_drift_decomposition() {
    const NF: usize = 64;
    let mut input = vec![0.0f32; NF];
    for (i, item) in input.iter_mut().enumerate() {
        let t = i as f32 / 48000.0;
        let freq = 200.0 + (8000.0 - 200.0) * (i as f32 / NF as f32);
        *item = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5;
    }

    // 1. Reference: f32 weights + exact tanh
    let r_ref = wavenet_standard_scalar(true, true, &input, NF);
    // 2. F16 weights + exact tanh → isolates weight quantization
    let r_quant = wavenet_standard_scalar(false, true, &input, NF);
    // 3. F32 weights + Padé tanh → isolates tanh approximation
    let r_tanh = wavenet_standard_scalar(true, false, &input, NF);
    // 4. F16 weights + Padé tanh → production-equivalent
    let r_prod = wavenet_standard_scalar(false, false, &input, NF);

    let e_quant = compute_esr(&r_ref, &r_quant);
    let e_tanh = compute_esr(&r_ref, &r_tanh);
    let e_total = compute_esr(&r_ref, &r_prod);
    let e_resid = e_total - e_quant - e_tanh;

    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  T1.4 — WaveNet Drift Decomposition (Standard, CH=16)  ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Source                                  ESR       dB   ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  (a) F16 weight quantization     {:.4e}  {:6.1}  ║",
        e_quant,
        esr_to_db(e_quant)
    );
    println!(
        "║  (b) tanh Padé [5,4] approx      {:.4e}  {:6.1}  ║",
        e_tanh,
        esr_to_db(e_tanh)
    );
    println!(
        "║  (c) f32 accumulation (residual) {:.4e}  {:6.1}  ║",
        e_resid,
        if e_resid > 0.0 {
            esr_to_db(e_resid)
        } else {
            f64::NEG_INFINITY
        }
    );
    println!("╠══════════════════════════════════════════════════════════╣");
    println!(
        "║  Total (a+b+c)                   {:.4e}  {:6.1}  ║",
        e_total,
        esr_to_db(e_total)
    );
    println!("╚══════════════════════════════════════════════════════════╝");

    // Sanity checks
    assert!(e_quant > 0.0, "Quantization ESR should be positive");
    assert!(e_total > 0.0, "Total ESR should be positive");
    assert!(
        e_total < 1e-5,
        "Total ESR ({:.4e}) unexpectedly large — possible scalar engine bug",
        e_total
    );

    println!();
    println!("// Measured: WaveNet Standard (CH=16, K=3, HEAD=8, 10+2 layers)");
    println!(
        "//   weight_quant_esr={:.4e} ({:.1} dB)",
        e_quant,
        esr_to_db(e_quant)
    );
    println!(
        "//   tanh_approx_esr= {:.4e} ({:.1} dB)",
        e_tanh,
        esr_to_db(e_tanh)
    );
    println!(
        "//   accum_residual=  {:.4e} ({:.1} dB)",
        e_resid,
        if e_resid > 0.0 {
            esr_to_db(e_resid)
        } else {
            f64::NEG_INFINITY
        }
    );
    println!(
        "//   total_esr=       {:.4e} ({:.1} dB)",
        e_total,
        esr_to_db(e_total)
    );
    println!("//   Recommendation: P2 — weight quantization dominates; S4 exact mode");
    println!("//   should prioritize higher-precision weight storage.");
    println!("//   Tanh Padé contribution is negligible for this topology (small");
    println!("//   weights keep activations in the linear tanh region).");
}

// =============================================================================
// B.3.1 — Soak Tests for New Paths (ConvNet, WaveNetDyn, LstmDyn, WaveNetA2Dyn)
// =============================================================================

fn run_model_soak(
    fixture: &str,
    label: &str,
    num_frames: usize,
    block_size: usize,
    output_is_multichannel: bool,
) {
    let path = model_path(fixture);
    assert!(path.exists(), "Fixture {} not found at {:?}", fixture, path);

    let json_data = fs::read_to_string(&path).expect("Failed to read model file");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse JSON");
    let mut model = build_model(&model_data).expect("Failed to build model");

    let out_ch = if output_is_multichannel {
        match model.as_ref() {
            StaticModel::ConvNet(c) => c.out_channels(),
            _ => 1,
        }
    } else {
        1
    };

    model.prewarm(model.prewarm_samples());

    let out_len = if output_is_multichannel {
        block_size * out_ch
    } else {
        block_size
    };

    let mut input = vec![0.0f32; block_size];
    let mut output = vec![0.0f32; out_len];
    let mut pcg = SimplePcg::new(42);

    let mut processed = 0;
    let mut zero_alloc_checked = false;

    let start = Instant::now();
    while processed < num_frames {
        let is_noise = (processed / 100_000) % 2 == 0;
        if is_noise {
            for v in &mut input {
                *v = pcg.next_f32();
            }
        } else {
            input.fill(0.0);
        }

        // Zero-alloc audit: verify during one noise frame (post-prewarm)
        if !zero_alloc_checked {
            {
                let _guard = TrackingGuard::new();
                model.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                );
            }
            assert_eq!(
                get_alloc_count(),
                0,
                "{} — heap allocation detected during process() call",
                label
            );
            zero_alloc_checked = true;
        } else {
            model.process(
                std::hint::black_box(&input),
                std::hint::black_box(&mut output),
            );
        }

        for &v in &output {
            assert!(
                v.is_finite(),
                "{} NaN/Inf after {} frames",
                label,
                processed
            );
            assert!(
                v == 0.0 || v.is_normal(),
                "{} subnormal at frame {} (bits: 0x{:08X})",
                label,
                processed,
                v.to_bits()
            );
        }
        if is_noise {
            let rms =
                (output.iter().map(|x| (x * x) as f64).sum::<f64>() / output.len() as f64).sqrt();
            assert!(
                rms > 0.0001 && rms < 10.0,
                "{} RMS do loop out of range: {} at {} frames",
                label,
                rms,
                processed
            );
        }

        processed += block_size;
    }

    let duration = start.elapsed();
    println!("--- {} Soak ---", label);
    println!("Duration: {:?}", duration);
    println!("Frames processed: {}", processed);
    println!("Zero-alloc verified: ✓");
}

/// Soak Test: ConvNet endurance — 10M frames.
///
/// ConvNet operates on the full buffer (not sub-blocks). The output buffer
/// must be `block_size × out_channels`.
#[test]
#[ignore]
fn test_convnet_soak() {
    run_model_soak("convnet_test.nam", "ConvNet", 10_000_000, 64, true);
}

/// Soak Test: WaveNetModelDyn endurance — 10M frames.
#[test]
#[ignore]
fn test_wavenet_dyn_soak() {
    run_model_soak("wavenet_dyn_free.nam", "WaveNetDyn", 10_000_000, 64, false);
}

/// Soak Test: LstmModelDyn endurance — 10M frames.
#[test]
#[ignore]
fn test_lstm_dyn_soak() {
    run_model_soak("lstm_dyn_test.nam", "LstmDyn", 10_000_000, 64, false);
}

/// Soak Test: WaveNetA2Dyn (Gated) endurance — 10M frames.
#[test]
#[ignore]
fn test_a2_dyn_gated_soak() {
    run_model_soak(
        "a2_dynamic_gated_ch8.nam",
        "A2Dyn-Gated",
        10_000_000,
        64,
        false,
    );
}

/// Soak Test: WaveNetA2Dyn (Blended) endurance — 10M frames.
#[test]
#[ignore]
fn test_a2_dyn_blended_soak() {
    run_model_soak(
        "a2_dynamic_blended_ch3.nam",
        "A2Dyn-Blended",
        10_000_000,
        64,
        false,
    );
}
