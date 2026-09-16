// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Zero-Dependency Synthetic Neural Model Example (`synthetic_model`)
//!
//! Demonstrates constructing an in-memory neural network model ([`StaticModel::Lstm1x3`])
//! entirely through the public API without reading external `.nam` or `.namb` files from disk.
//!
//! ## Overview
//!
//! In headless test harnesses, continuous integration, or host environments requiring a fallback
//! neural profile, callers may wish to instantiate and process audio through a valid model
//! immediately after `cargo add` or `git clone`:
//! - **In-Memory Construction**: Directly instantiates [`Lstm1x3`] (1 LSTM layer × 3 hidden units).
//! - **Deterministic Weights**: Configures gate biases and projection weights for stable acoustic throughput.
//! - **Zero Disk I/O**: Runs end-to-end without requiring any sample file or model weights on disk.
//! - **Real-Time Pipeline Compliance**: Primes internal recurrent states via [`NamModel::reset`] and
//!   executes hot-path block processing via [`NamModel::process`].
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example synthetic_model
//! ```

use std::f32::consts::PI;
use std::time::Instant;

use neural_amp_modeler_rs::models::lstm::Lstm1x3;
use neural_amp_modeler_rs::prelude::*;

/// Target audio sample rate.
const SAMPLE_RATE: u32 = 48000;
/// Real-time audio quantum block size.
const BLOCK_SIZE: usize = 64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — In-Memory Synthetic Model Demo      ");
    println!("============================================================");

    // 1. Construct a minimal valid 1-layer, 3-hidden-unit LSTM model in memory.
    println!("\n[1/3] Constructing In-Memory Synthetic Model (Lstm1x3)");
    let mut lstm = Lstm1x3::new();

    // Configure deterministic gate biases to ensure transparent non-zero signal flow:
    // Gates layout in LstmLayer:
    //   Indices  0..3 : Input gate (i)  -> set bias ~1.5 (sigmoid ~0.82)
    //   Indices  3..6 : Forget gate (f) -> set bias ~2.0 (sigmoid ~0.88)
    //   Indices  6..9 : Cell candidate (g)
    //   Indices  9..12: Output gate (o) -> set bias ~1.5 (sigmoid ~0.82)
    lstm.layer.bias[0] = 1.5;
    lstm.layer.bias[1] = 1.5;
    lstm.layer.bias[2] = 1.5;

    lstm.layer.bias[3] = 2.0;
    lstm.layer.bias[4] = 2.0;
    lstm.layer.bias[5] = 2.0;

    lstm.layer.bias[9] = 1.5;
    lstm.layer.bias[10] = 1.5;
    lstm.layer.bias[11] = 1.5;

    // Set non-zero input projection weights for cell candidate state (gate 2, input 0):
    lstm.layer.input_hidden_weights.0[2][0][0] = 0.8;
    lstm.layer.input_hidden_weights.0[2][0][1] = 0.4;
    lstm.layer.input_hidden_weights.0[2][0][2] = 0.2;

    // Linear head projection weights and bias:
    lstm.head_weights_f32 = [0.5, 0.3, 0.2];
    lstm.head_bias = 0.0;

    // Wrap in the canonical StaticModel enum:
    let mut model = StaticModel::Lstm1x3(Box::new(lstm));
    println!("  Architecture : LSTM 1 Layer × 3 Hidden Units");
    println!("  Disk Access  : 0 bytes (100% In-Memory Synthetic Weights)");

    // 2. Initialize and prewarm model state.
    model.reset(SAMPLE_RATE, BLOCK_SIZE)?;
    println!("  Prewarm      : Completed via model.reset()");

    // 3. Generate a synthetic test signal (220 Hz A3 fundamental + harmonics).
    println!("\n[2/3] Generating Synthetic Test Audio Signal");
    let duration_secs = 1.0;
    let total_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let input_audio = generate_test_tone(total_samples, SAMPLE_RATE);
    let mut output_audio = vec![0.0f32; total_samples];

    println!(
        "  Duration     : {:.2} seconds ({} samples)",
        duration_secs, total_samples
    );
    println!("  Block Size   : {} samples per quantum", BLOCK_SIZE);

    // 4. Process audio quantum-by-quantum through the neural model.
    println!("\n[3/3] Processing Audio through Neural Inference Loop");
    let start_time = Instant::now();

    for (in_chunk, out_chunk) in input_audio
        .chunks_exact(BLOCK_SIZE)
        .zip(output_audio.chunks_exact_mut(BLOCK_SIZE))
    {
        model.process(in_chunk, out_chunk);
    }

    let elapsed = start_time.elapsed();
    let throughput = (total_samples as f64 / elapsed.as_secs_f64()) / 1000.0;

    // 5. Verify signal energy and compute metrics.
    let in_peak = compute_peak(&input_audio);
    let out_peak = compute_peak(&output_audio);
    let in_rms = compute_rms(&input_audio);
    let out_rms = compute_rms(&output_audio);

    println!("\n[Performance & Statistics]");
    println!("  Execution Time   : {:.2?}", elapsed);
    println!("  Throughput       : {:.2} kSamples/sec", throughput);

    println!("\n[Audio Energy Breakdown]");
    println!(
        "  Input Peak       : {:.4} ({:.2} dBFS)",
        in_peak,
        20.0 * in_peak.max(1e-6).log10()
    );
    println!(
        "  Output Peak      : {:.4} ({:.2} dBFS)",
        out_peak,
        20.0 * out_peak.max(1e-6).log10()
    );
    println!(
        "  Input RMS        : {:.4} ({:.2} dBFS)",
        in_rms,
        20.0 * in_rms.max(1e-6).log10()
    );
    println!(
        "  Output RMS       : {:.4} ({:.2} dBFS)",
        out_rms,
        20.0 * out_rms.max(1e-6).log10()
    );

    // Verifiable assertions
    assert!(
        out_peak > 0.001,
        "Output peak amplitude must be non-zero (measured: {})",
        out_peak
    );
    assert!(
        out_rms > 0.001,
        "Output RMS level must be non-zero (measured: {})",
        out_rms
    );

    println!("\n[Status] In-memory synthetic model inference PASSED (OK).");
    Ok(())
}

/// Generates a test audio signal with fundamental and harmonic content.
fn generate_test_tone(samples: usize, sample_rate: u32) -> Vec<f32> {
    let mut buf = Vec::with_capacity(samples);
    let dt = 1.0 / sample_rate as f32;

    for i in 0..samples {
        let t = i as f32 * dt;
        let sig = 0.5 * (2.0 * PI * 220.0 * t).sin()
            + 0.25 * (2.0 * PI * 440.0 * t).sin()
            + 0.125 * (2.0 * PI * 660.0 * t).sin();
        buf.push(sig);
    }
    buf
}

/// Computes the peak absolute amplitude of a sample buffer.
fn compute_peak(buf: &[f32]) -> f32 {
    buf.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
}

/// Computes the Root Mean Square (RMS) energy level of a sample buffer.
fn compute_rms(buf: &[f32]) -> f32 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = buf.iter().map(|s| s * s).sum();
    (sum_sq / buf.len() as f32).sqrt()
}
