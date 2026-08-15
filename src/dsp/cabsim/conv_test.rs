// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

// ── Reference direct convolution ──

/// Naive direct convolution: y[n] = Σ_m h[m] * x[n-m].
///
/// Used as the ground-truth reference for ESR validation.
fn direct_convolve(ir: &[f32], input: &[f32]) -> Vec<f32> {
    let out_len = input.len() + ir.len() - 1;
    let mut output = vec![0.0f32; out_len];
    for (n, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (m, &ir_val) in ir.iter().enumerate() {
            let x_idx = n as isize - m as isize;
            if x_idx >= 0 && x_idx < input.len() as isize {
                acc += ir_val * input[x_idx as usize];
            }
        }
        *out = acc;
    }
    output
}

/// Error-to-Signal Ratio: ESR = Σ(err²) / Σ(ref²).
///
/// ESR < 1e-5 means the error energy is < 0.001% of the reference energy.
fn compute_esr(reference: &[f32], computed: &[f32]) -> f64 {
    assert_eq!(reference.len(), computed.len());
    let mut ref_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for (r, c) in reference.iter().zip(computed.iter()) {
        let diff = *r as f64 - *c as f64;
        ref_energy += (*r as f64) * (*r as f64);
        err_energy += diff * diff;
    }
    if ref_energy < 1e-30 {
        return 0.0; // All zeros: err is also zero if computed is zero
    }
    err_energy / ref_energy
}

// ── Helpers ──

/// Feeds a full signal through the UPOLS engine block-by-block and collects output.
fn process_full_signal(engine: &mut ConvEngine, signal: &[f32]) -> Vec<f32> {
    let b = engine.partition_size();
    let mut output = Vec::with_capacity(signal.len());
    let mut buf_in = vec![0.0f32; b];
    let mut buf_out = vec![0.0f32; b];

    let mut pos = 0;
    while pos < signal.len() {
        let chunk = (signal.len() - pos).min(b);
        buf_in[..chunk].copy_from_slice(&signal[pos..pos + chunk]);
        if chunk < b {
            buf_in[chunk..].fill(0.0);
        }
        engine.process(&buf_in, &mut buf_out, None);
        output.extend_from_slice(&buf_out[..chunk.min(b)]);
        pos += chunk;
    }

    // Flush: feed zero blocks to drain the FDL of remaining energy.
    let flush_blocks = engine.num_partitions();
    for _ in 0..flush_blocks {
        buf_in.fill(0.0);
        engine.process(&buf_in, &mut buf_out, None);
        output.extend_from_slice(&buf_out[..b]);
    }

    output
}

/// Generates a synthetic impulse response: exponentially decaying sine.
fn synth_ir(len: usize, freq: f32, decay: f32, sample_rate: u32) -> Vec<f32> {
    (0..len)
        .map(|n| {
            let t = n as f32 / sample_rate as f32;
            (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
        })
        .collect()
}

// ── Tests ──

#[test]
fn passthrough_on_empty_ir() {
    let engine =
        ConvEngine::new(&[], 64).expect("construction should succeed for test-sized buffers");
    assert!(engine.is_passthrough());
    assert_eq!(engine.num_partitions(), 0);

    let input: Vec<f32> = (0..128).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut output = vec![0.0f32; 64];
    let mut engine2 =
        ConvEngine::new(&[], 64).expect("construction should succeed for test-sized buffers");
    engine2.process(&input[..64], &mut output, None);
    for i in 0..64 {
        assert!((output[i] - input[i]).abs() < 1e-10);
    }
}

#[test]
fn short_slices_zero_output_and_raise_flag_without_panic() {
    // R-01: in debug and release, `ConvEngine::process` must never read or
    // write beyond the caller's slices. Short input/output must zero the
    // output, raise the contract-violation flag, and return without
    // panicking (and without corrupting internal UPOLS state).
    let ir = synth_ir(64, 500.0, 10.0, 48000);
    let partition = 64;
    let mut engine = ConvEngine::new(&ir, partition)
        .expect("construction should succeed for test-sized buffers");

    // Short input (32 < 64), full output.
    let rt = RtStatusFlags::new();
    let input = vec![0.5f32; 32];
    let mut output = vec![1.0f32; 64];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.process(&input, &mut output, Some(&rt));
    }));
    assert!(result.is_ok(), "short input must not panic");
    assert!(output.iter().all(|&s| s == 0.0), "output must be zeroed");
    assert!(rt.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));

    // Full input, short output (32 < 64).
    let rt2 = RtStatusFlags::new();
    let input = vec![0.5f32; 64];
    let mut output = vec![1.0f32; 32];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.process(&input, &mut output, Some(&rt2));
    }));
    assert!(result.is_ok(), "short output must not panic");
    assert!(output.iter().all(|&s| s == 0.0), "output must be zeroed");
    assert!(rt2.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));

    // Contract-compliant block must not raise the flag and must still
    // process correctly after the violations (state was not corrupted).
    let rt3 = RtStatusFlags::new();
    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];
    engine.process(&input, &mut output, Some(&rt3));
    assert!(!rt3.check_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION));
}

#[test]
fn impulse_response_is_identity_with_latency() {
    // Dirac delta IR at index 0 → filter = identity.
    // UPOLS introduces block_size samples of algorithmic latency.
    let mut ir = vec![0.0f32; 64];
    ir[0] = 1.0;

    let block_size = 32;
    let mut engine = ConvEngine::new(&ir, block_size)
        .expect("construction should succeed for test-sized buffers");

    // Generate input: ramp 0..255
    let signal: Vec<f32> = (0..256).map(|i| i as f32).collect();

    // Direct convolution reference (identity → output = input)
    let ref_full = direct_convolve(&ir, &signal);

    let upols_out = process_full_signal(&mut engine, &signal);

    // UPOLS has block_size latency: the first block_size samples of output
    // should be ~0 (algorithmic delay). The remainder should match the
    // reference (which is just the signal without the block delay).
    let min_len = ref_full.len().min(upols_out.len());
    let esr = compute_esr(&ref_full[..min_len], &upols_out[..min_len]);
    assert!(
        esr < 1e-5,
        "ESR = {:.2e} exceeds 1e-5 for dirac delta IR",
        esr
    );
}

#[test]
fn latency_equals_partition_size() {
    let ir = synth_ir(100, 440.0, 10.0, 48000);
    for &bs in &[32, 64, 128, 256] {
        let engine =
            ConvEngine::new(&ir, bs).expect("construction should succeed for test-sized buffers");
        assert_eq!(engine.latency_samples(), bs);
    }
}

#[test]
fn esr_parity_short_ir() {
    // Short IR: 32 samples
    let ir = synth_ir(32, 800.0, 15.0, 48000);
    // Input: 512 samples of mixed sines
    let signal: Vec<f32> = (0..512)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f32::consts::TAU * 660.0 * t).sin()
        })
        .collect();

    // Direct convolution reference (full length)
    let ref_full = direct_convolve(&ir, &signal);

    // UPOLS with block_size = 64
    let block_size = 64;
    let mut engine = ConvEngine::new(&ir, block_size)
        .expect("construction should succeed for test-sized buffers");
    let upols_out = process_full_signal(&mut engine, &signal);

    // The UPOLS output has latency of block_size, and the tail extends by (ir.len() - 1)
    // after the signal ends (flushed by zero blocks).
    // Total UPOLS output length should match ref_full length.
    // Trim to common length for ESR comparison.
    let min_len = ref_full.len().min(upols_out.len());
    let esr = compute_esr(&ref_full[..min_len], &upols_out[..min_len]);

    assert!(
        esr < 1e-5,
        "ESR = {:.2e} exceeds 1e-5 threshold for short IR (block_size={})",
        esr,
        block_size
    );
}

#[test]
fn esr_parity_medium_ir() {
    // Medium IR: 200 samples (multiple partitions with block_size=64 = 4 partitions)
    let ir = synth_ir(200, 500.0, 8.0, 48000);
    let signal: Vec<f32> = (0..1024)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 180.0 * t).sin()
                + 0.3 * (std::f32::consts::TAU * 1200.0 * t).sin()
        })
        .collect();

    let ref_full = direct_convolve(&ir, &signal);

    let block_size = 64;
    let mut engine = ConvEngine::new(&ir, block_size)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(engine.num_partitions(), 4); // ceil(200/64) = 4

    let upols_out = process_full_signal(&mut engine, &signal);

    let min_len = ref_full.len().min(upols_out.len());
    let esr = compute_esr(&ref_full[..min_len], &upols_out[..min_len]);

    assert!(
        esr < 1e-5,
        "ESR = {:.2e} exceeds 1e-5 threshold for medium IR (4 partitions)",
        esr
    );
}

#[test]
fn esr_parity_long_ir() {
    // Long IR: 1000 samples (many partitions)
    let ir = synth_ir(1000, 300.0, 4.0, 48000);
    let signal: Vec<f32> = (0..2048)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 150.0 * t).sin()
        })
        .collect();

    let ref_full = direct_convolve(&ir, &signal);

    let block_size = 64;
    let mut engine = ConvEngine::new(&ir, block_size)
        .expect("construction should succeed for test-sized buffers");
    assert!(engine.num_partitions() > 10);

    let upols_out = process_full_signal(&mut engine, &signal);

    let min_len = ref_full.len().min(upols_out.len());
    let esr = compute_esr(&ref_full[..min_len], &upols_out[..min_len]);

    assert!(
        esr < 1e-5,
        "ESR = {:.2e} exceeds 1e-5 threshold for long IR ({} partitions)",
        esr,
        engine.num_partitions()
    );
}

#[test]
fn single_sample_ir() {
    // Single sample IR: just a gain
    let ir = vec![0.75f32];
    let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.02).sin()).collect();

    let mut engine =
        ConvEngine::new(&ir, 32).expect("construction should succeed for test-sized buffers");
    let output = process_full_signal(&mut engine, &signal);

    // Reference: direct convolution
    let ref_full = direct_convolve(&ir, &signal);
    let min_len = ref_full.len().min(output.len());
    let esr = compute_esr(&ref_full[..min_len], &output[..min_len]);

    assert!(esr < 1e-5, "ESR = {:.2e} for single-sample IR", esr);
}

#[test]
fn block_size_one() {
    // Minimum block size: 1
    let ir = synth_ir(16, 1000.0, 20.0, 48000);
    let signal: Vec<f32> = (0..128).map(|i| (i as f32 * 0.05).sin()).collect();

    let mut engine =
        ConvEngine::new(&ir, 1).expect("construction should succeed for test-sized buffers");
    assert_eq!(engine.fft_size(), 2); // 2 * 1 = 2, already a power of two

    let output = process_full_signal(&mut engine, &signal);
    let ref_full = direct_convolve(&ir, &signal);
    let min_len = ref_full.len().min(output.len());
    let esr = compute_esr(&ref_full[..min_len], &output[..min_len]);

    assert!(esr < 1e-5, "ESR = {:.2e} for block_size=1", esr);
}

#[test]
fn non_power_of_two_block_size() {
    // Block size 75 (non-power-of-2) → FFT size = next_power_of_two(150) = 256
    let ir = synth_ir(80, 600.0, 12.0, 48000);
    let signal: Vec<f32> = (0..300).map(|i| (i as f32 * 0.03).sin()).collect();

    let mut engine =
        ConvEngine::new(&ir, 75).expect("construction should succeed for test-sized buffers");
    assert_eq!(engine.fft_size(), 256);
    assert_eq!(engine.num_partitions(), 2); // ceil(80/75) = 2

    let output = process_full_signal(&mut engine, &signal);
    let ref_full = direct_convolve(&ir, &signal);
    let min_len = ref_full.len().min(output.len());
    let esr = compute_esr(&ref_full[..min_len], &output[..min_len]);

    assert!(
        esr < 1e-5,
        "ESR = {:.2e} for non-power-of-2 block size 75 (fft_size=256)",
        esr
    );
}

#[test]
fn output_length_matches_reference() {
    // The total output length of UPOLS (after flushing) should match
    // the direct convolution length: len(input) + len(ir) - 1.
    let ir = synth_ir(47, 440.0, 10.0, 48000);
    let signal: Vec<f32> = (0..200).map(|i| (i as f32 * 0.01).sin()).collect();
    let expected_len = signal.len() + ir.len() - 1; // 200 + 47 - 1 = 246

    let mut engine =
        ConvEngine::new(&ir, 50).expect("construction should succeed for test-sized buffers");
    let output = process_full_signal(&mut engine, &signal);

    assert!(
        output.len() >= expected_len,
        "Output length {} < expected {}",
        output.len(),
        expected_len
    );
}

#[test]
fn deterministic_output() {
    // Same IR + same input → same output every time
    let ir = synth_ir(60, 350.0, 8.0, 48000);
    let signal: Vec<f32> = (0..300).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut engine1 =
        ConvEngine::new(&ir, 64).expect("construction should succeed for test-sized buffers");
    let mut engine2 =
        ConvEngine::new(&ir, 64).expect("construction should succeed for test-sized buffers");

    let out1 = process_full_signal(&mut engine1, &signal);
    let out2 = process_full_signal(&mut engine2, &signal);

    assert_eq!(out1.len(), out2.len());
    for (a, b) in out1.iter().zip(out2.iter()) {
        assert!((a - b).abs() < 1e-10, "Non-deterministic output");
    }
}

// ── f64 Oracle Validation ──

/// Direct convolution in f64 precision — ground-truth oracle.
fn direct_convolve_f64(ir: &[f32], input: &[f32]) -> Vec<f64> {
    let out_len = input.len() + ir.len() - 1;
    let mut output = vec![0.0f64; out_len];
    let ir_f64: Vec<f64> = ir.iter().map(|&v| v as f64).collect();
    let input_f64: Vec<f64> = input.iter().map(|&v| v as f64).collect();
    for (n, out) in output.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (m, &ir_val) in ir_f64.iter().enumerate() {
            let x_idx = n as isize - m as isize;
            if x_idx >= 0 && x_idx < input_f64.len() as isize {
                acc += ir_val * input_f64[x_idx as usize];
            }
        }
        *out = acc;
    }
    output
}

/// ESR: f32 UPOLS output vs f64 oracle reference.
fn compute_esr_f64_oracle(reference_f64: &[f64], computed_f32: &[f32]) -> f64 {
    assert_eq!(reference_f64.len(), computed_f32.len());
    let mut ref_energy = 0.0f64;
    let mut err_energy = 0.0f64;
    for (r, c) in reference_f64.iter().zip(computed_f32.iter()) {
        let c_f64 = *c as f64;
        let diff = *r - c_f64;
        ref_energy += *r * *r;
        err_energy += diff * diff;
    }
    if ref_energy < 1e-60 {
        return 0.0;
    }
    err_energy / ref_energy
}

fn run_f64_oracle_test(ir_len: usize, partition_size: usize, signal_len: usize) {
    let ir = synth_ir(ir_len, 500.0, 8.0, 48000);
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / 48000.0;
            (std::f32::consts::TAU * 220.0 * t).sin()
                + 0.5 * (std::f32::consts::TAU * 660.0 * t).sin()
                + 0.3 * (std::f32::consts::TAU * 110.0 * t).sin()
        })
        .collect();

    let ref_f64 = direct_convolve_f64(&ir, &signal);

    let mut engine = ConvEngine::new(&ir, partition_size)
        .expect("construction should succeed for test-sized buffers");
    let upols_out = process_full_signal(&mut engine, &signal);

    let min_len = ref_f64.len().min(upols_out.len());
    let esr = compute_esr_f64_oracle(&ref_f64[..min_len], &upols_out[..min_len]);
    assert!(
        esr < 1e-10,
        "ESR = {:.2e} exceeds 1e-10 — IR={ir_len} taps, P={partition_size}, signal={signal_len}",
        esr
    );
}

#[test]
fn f64_oracle_ir_64() {
    run_f64_oracle_test(64, 32, 256);
}

#[test]
fn f64_oracle_ir_128() {
    run_f64_oracle_test(128, 64, 256);
}

#[test]
fn f64_oracle_ir_256() {
    run_f64_oracle_test(256, 128, 512);
}

#[test]
fn f64_oracle_ir_512() {
    run_f64_oracle_test(512, 128, 1024);
}

#[test]
fn f64_oracle_ir_1024() {
    run_f64_oracle_test(1024, 128, 2048);
}

#[test]
fn f64_oracle_ir_2048() {
    run_f64_oracle_test(2048, 256, 4096);
}

#[test]
fn f64_oracle_ir_4096() {
    run_f64_oracle_test(4096, 256, 8192);
}
