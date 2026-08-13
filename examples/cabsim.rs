// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Standalone Cabinet Impulse Response (IR) Simulation Example (`cabsim`)
//!
//! Demonstrates the standalone usage of the `CabSim` convolution module
//! in `NeuralAmpModeler-rs`.
//!
//! ## Overview
//!
//! The `CabSim` module provides real-time, frequency-domain impulse response (IR)
//! convolution for guitar speaker cabinet simulation:
//! - **UPOLS Engine (`ConvEngine`)**: Uniform-Partitioned Overlap-Save algorithm
//!   pre-calculating FFTs of IR partitions for zero-allocation hot-path convolution.
//! - **Variable-Block Adapter (`CabSimAdapter`)**: Accumulates sub-blocks of arbitrary
//!   size (produced by DAW sample-accurate automation) into fixed partitions.
//! - **IR Loader (`CabSimIr`)**: Reads mono WAV files (PCM16, PCM24, Float32),
//!   resamples to active sample rate, and normalizes peak level off-RT.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example cabsim -- /path/to/cabinet_ir.wav
//! ```

use std::f32::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use neural_amp_modeler_rs::dsp::cabsim::adapter::CabSimAdapter;
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
use neural_amp_modeler_rs::dsp::cabsim::loader::CabSimIr;

/// Target audio sample rate.
const SAMPLE_RATE: u32 = 48000;
/// Fixed partition size for UPOLS convolution engine.
const PARTITION_SIZE: usize = 256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — Cabinet IR Simulation (CabSim)      ");
    println!("============================================================");

    // 1. Resolve IR file path from argument or generate synthetic IR.
    let (ir_samples, ir_source_label) = match std::env::args().nth(1) {
        Some(arg_path) => {
            let path = PathBuf::from(arg_path);
            if !path.exists() {
                eprintln!(
                    "\nError: Specified IR file \"{}\" does not exist.",
                    path.display()
                );
                std::process::exit(1);
            }
            println!("\n[1/3] Loading IR WAV File");
            println!("  File Path : {}", path.display());
            let ir = CabSimIr::load(&path, SAMPLE_RATE, true)?;
            println!("  Original Rate : {} Hz", ir.original_rate);
            println!("  Loaded Rate   : {} Hz", ir.sample_rate);
            println!("  Samples Count : {}", ir.samples.len());
            println!("  Normalized    : {}", ir.normalized);
            (
                ir.samples.clone(),
                format!(
                    "WAV: {}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                ),
            )
        }
        None => {
            println!("\n[Notice] No WAV IR path supplied as CLI argument.");
            println!("Generating synthetic 12-inch guitar cabinet impulse response...");
            let syn_ir = generate_synthetic_cab_ir(SAMPLE_RATE, 2048);
            (
                syn_ir,
                "Synthetic 12\" Guitar Cab IR (2048 samples)".to_string(),
            )
        }
    };

    // 2. Initialize UPOLS frequency-domain convolution engine.
    println!("\n[2/3] Building UPOLS Convolution Engine");
    let conv_engine = ConvEngine::new(&ir_samples, PARTITION_SIZE)
        .map_err(|e| format!("Failed to build ConvEngine: {:?}", e))?;

    let partition_size = conv_engine.partition_size();
    let num_partitions = conv_engine.num_partitions();
    let fft_size = conv_engine.fft_size();
    let latency_samples = conv_engine.latency_samples();
    let latency_ms = (latency_samples as f32 / SAMPLE_RATE as f32) * 1000.0;

    println!("  IR Source Label  : {}", ir_source_label);
    println!("  IR Total Samples : {}", ir_samples.len());
    println!("  Partition Size   : {} samples", partition_size);
    println!("  Num Partitions   : {}", num_partitions);
    println!("  FFT Block Size   : {} bins", fft_size);
    println!(
        "  Algorithmic Latency : {} samples ({:.2} ms)",
        latency_samples, latency_ms
    );

    // 3. Wrap with CabSimAdapter for variable block size support.
    let mut adapter = CabSimAdapter::new(Box::new(conv_engine))
        .map_err(|e| format!("Failed to create CabSimAdapter: {:?}", e))?;

    // 4. Generate a test audio signal (raw guitar DI style: 110 Hz A2 fundamental + harmonics).
    let duration_secs = 2.0;
    let total_samples = (SAMPLE_RATE as f32 * duration_secs) as usize;
    let input_audio = generate_guitar_di_signal(total_samples, SAMPLE_RATE);
    let mut output_audio = vec![0.0f32; total_samples];

    println!("\n[3/3] Processing Audio through CabSim Adapter");
    println!(
        "  Audio Duration   : {:.2} seconds ({} samples)",
        duration_secs, total_samples
    );

    // Simulate variable sub-block buffer sizes as produced by DAW hosts (e.g. 64, 128, 96, 256).
    let sub_block_sizes = [64, 128, 96, 256, 192, 128];
    let mut block_idx = 0;
    let mut offset = 0;

    let start_time = Instant::now();

    while offset < total_samples {
        let block_size = sub_block_sizes[block_idx % sub_block_sizes.len()];
        let len = block_size.min(total_samples - offset);

        let in_slice = &input_audio[offset..offset + len];
        let out_slice = &mut output_audio[offset..offset + len];

        adapter.process_variable(in_slice, out_slice, None);

        offset += len;
        block_idx += 1;
    }

    let elapsed = start_time.elapsed();
    let throughput = (total_samples as f64 / elapsed.as_secs_f64()) / 1000.0;

    // 5. Compute signal metrics.
    let in_peak = compute_peak(&input_audio);
    let out_peak = compute_peak(&output_audio);
    let in_rms = compute_rms(&input_audio);
    let out_rms = compute_rms(&output_audio);

    println!("\n[Performance & Statistics]");
    println!("  Execution Time   : {:.2?}", elapsed);
    println!("  Sub-Blocks Ran   : {} variable-length calls", block_idx);
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

    println!("\n[Status] Cabinet IR convolution completed successfully.");
    Ok(())
}

/// Generates a realistic synthetic speaker cabinet impulse response (2048 samples).
///
/// Models a 12-inch guitar speaker resonant response using damped low-frequency sine oscillations
/// and exponential energy decay.
fn generate_synthetic_cab_ir(sample_rate: u32, length: usize) -> Vec<f32> {
    let mut ir = Vec::with_capacity(length);
    let dt = 1.0 / sample_rate as f32;

    for i in 0..length {
        let t = i as f32 * dt;
        // Primary speaker cone resonance ~90 Hz + 4.5 kHz high-frequency roll-off dampening
        let resonance = (2.0 * PI * 90.0 * t).sin();
        let body = (2.0 * PI * 220.0 * t).sin() * 0.5;
        let decay = (-t * 80.0).exp(); // ~50ms T60 decay
        let sample = (resonance + body) * decay;
        ir.push(sample);
    }

    // Normalize IR so that total energy sum yields ~0 dB passband gain.
    let sum_abs: f32 = ir.iter().map(|s| s.abs()).sum();
    if sum_abs > 0.0 {
        for s in &mut ir {
            *s /= sum_abs;
        }
    }
    ir
}

/// Generates a synthetic raw guitar DI signal (110 Hz A2 string fundamental with harmonics).
fn generate_guitar_di_signal(samples: usize, sample_rate: u32) -> Vec<f32> {
    let mut buf = Vec::with_capacity(samples);
    let dt = 1.0 / sample_rate as f32;

    for i in 0..samples {
        let t = i as f32 * dt;
        let sig = 0.5 * (2.0 * PI * 110.0 * t).sin()
            + 0.3 * (2.0 * PI * 220.0 * t).sin()
            + 0.15 * (2.0 * PI * 330.0 * t).sin();
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
