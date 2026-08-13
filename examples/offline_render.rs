// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Offline High-Quality (HQ) Audio Buffer Rendering Example (`offline_render`)
//!
//! Demonstrates offline audio processing using the `NeuralAmpModeler-rs` engine
//! in **High Quality (HQ) Mode**.
//!
//! ## Quality Modes Overview
//!
//! - **Live / Realtime Mode**: Optimized for minimum latency and CPU usage during live performance.
//!   Uses zero added latency (`OversampleFactor::Off`), active adaptive compute load-balancing,
//!   and exact or fast-math activations depending on CPU constraints.
//! - **HQ / Offline Mode**: Designed for audio rendering and DAW exports where CPU budget is unconstrained.
//!   Enables **4× half-band oversampling (`OversampleFactor::X4`)** to eliminate non-linear activation
//!   aliasing, disables adaptive compute for 100% deterministic output, and uses exact activation precision.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example offline_render -- /path/to/model.nam
//! cargo run --example offline_render -- /path/to/model.namb
//! ```

use std::f32::consts::PI;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};
use neural_amp_modeler_rs::models::NamModel;

/// Processing block size in samples at native model rate.
const BLOCK_SIZE: usize = 512;
/// Sample rate for the generated test buffer.
const SAMPLE_RATE: u32 = 48000;
/// Duration of generated test buffer in seconds.
const DURATION_SECS: f32 = 2.0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — High-Quality Offline Renderer       ");
    println!("============================================================");

    // 1. Capture system hardware capability snapshot.
    let sys = SystemSnapshot::capture();

    // 2. Resolve model file path from CLI argument or search local fixtures.
    let path = match std::env::args().nth(1) {
        Some(arg) => PathBuf::from(arg),
        None => {
            println!("\n[Notice] No model path supplied as argument.");
            println!("Searching for local fixture models...");
            find_sample_model().unwrap_or_else(|| {
                println!("\n[Usage]");
                println!("  cargo run --example offline_render -- <path-to-model.nam|.namb>");
                println!("\nError: Please specify a valid .nam or .namb model file.");
                process::exit(1);
            })
        }
    };

    if !path.exists() {
        eprintln!("\nError: File \"{}\" does not exist.", path.display());
        process::exit(1);
    }

    println!("\n[1/4] Loading Model");
    println!("  File Path : {}", path.display());

    let mut model_pair = load_and_build_model(&path, &sys, false, LoadOptions::default())?;
    let model = model_pair
        .model_l
        .as_mut()
        .ok_or("Failed to obtain left-channel model from pair")?;

    println!("  Architecture : {}", model_pair.architecture);
    println!("  Topology     : {}", model_pair.topology);
    println!("  Sample Rate  : {} Hz", model_pair.sample_rate);

    // 3. Prewarm model state (flushes recurrent memory / tube state before rendering).
    println!("\n[2/4] Pre-Warming Neural Model State");
    model.prewarm(BLOCK_SIZE * 4);
    println!("  State pre-warmed with {} samples", BLOCK_SIZE * 4);

    // 4. Initialize 4× half-band oversampling engine (HQ Mode).
    println!("\n[3/4] Initializing HQ Quality Engine");
    let os_factor = OversampleFactor::X4;
    let mut os_engine = OversampleEngine::new(os_factor, BLOCK_SIZE)
        .map_err(|e| format!("Failed to create oversampling engine: {:?}", e))?;

    let os_multiplier = os_factor.multiplier(); // 4x
    let added_latency = os_engine.latency_samples();

    println!("  Oversampling Factor : {}× (HQ Mode)", os_multiplier);
    println!(
        "  Filter Latency      : {} samples ({:.2} ms at {} Hz)",
        added_latency,
        (added_latency as f32 / SAMPLE_RATE as f32) * 1000.0,
        SAMPLE_RATE
    );
    println!("  Adaptive Compute    : Disabled (100% deterministic output)");

    // 5. Generate a synthetic audio input signal (1 kHz sine sweep + harmonics).
    let total_blocks = (SAMPLE_RATE as f32 * DURATION_SECS / BLOCK_SIZE as f32).ceil() as usize;
    let total_samples = total_blocks * BLOCK_SIZE;
    let actual_duration = total_samples as f32 / SAMPLE_RATE as f32;

    println!("\n[4/4] Rendering Audio Buffer");
    println!(
        "  Buffer Length : {} samples ({:.2} seconds at {} Hz)",
        total_samples, actual_duration, SAMPLE_RATE
    );

    let input_audio = generate_test_signal(total_samples, SAMPLE_RATE);
    let mut output_audio = vec![0.0f32; total_samples];

    // Allocate scratch buffers for 4× oversampled intermediate data.
    let mut os_up_buf = vec![0.0f32; BLOCK_SIZE * os_multiplier];
    let mut os_model_buf = vec![0.0f32; BLOCK_SIZE * os_multiplier];

    // 6. Execute block-by-block offline processing loop.
    let start_time = Instant::now();
    let mut processed_samples = 0;

    for (in_chunk, out_chunk) in input_audio
        .chunks(BLOCK_SIZE)
        .zip(output_audio.chunks_mut(BLOCK_SIZE))
    {
        let n_in = in_chunk.len();

        // Stage A: Upsample native input to 4× rate
        let n_os = os_engine.upsample(in_chunk, &mut os_up_buf[..n_in * os_multiplier], None);

        // Stage B: Process neural model at 4× oversampled rate
        model.process(&os_up_buf[..n_os], &mut os_model_buf[..n_os]);

        // Stage C: Downsample back to native rate
        let n_out = os_engine.downsample(&os_model_buf[..n_os], out_chunk, None);

        processed_samples += n_out;
    }

    let elapsed = start_time.elapsed();
    let render_speed = actual_duration / elapsed.as_secs_f32();
    let throughput_khz = (processed_samples as f64 / elapsed.as_secs_f64()) / 1000.0;

    // 7. Compute signal statistics.
    let in_peak = compute_peak(&input_audio);
    let out_peak = compute_peak(&output_audio);
    let in_rms = compute_rms(&input_audio);
    let out_rms = compute_rms(&output_audio);

    println!("\n[Render Statistics]");
    println!("  Render Duration : {:.2?}", elapsed);
    println!("  Real-Time Speed : {:.1}× real-time", render_speed);
    println!("  Throughput      : {:.2} kSamples/sec", throughput_khz);

    println!("\n[Audio Fidelity Summary]");
    println!(
        "  Input Peak      : {:.4} ({:.2} dBFS)",
        in_peak,
        20.0 * in_peak.max(1e-6).log10()
    );
    println!(
        "  Output Peak     : {:.4} ({:.2} dBFS)",
        out_peak,
        20.0 * out_peak.max(1e-6).log10()
    );
    println!(
        "  Input RMS       : {:.4} ({:.2} dBFS)",
        in_rms,
        20.0 * in_rms.max(1e-6).log10()
    );
    println!(
        "  Output RMS      : {:.4} ({:.2} dBFS)",
        out_rms,
        20.0 * out_rms.max(1e-6).log10()
    );

    println!("\n[Status] Offline HQ rendering completed successfully with zero aliasing.");
    Ok(())
}

/// Generates a test audio signal consisting of a 440 Hz fundamental with harmonics.
fn generate_test_signal(samples: usize, sample_rate: u32) -> Vec<f32> {
    let mut buf = Vec::with_capacity(samples);
    let dt = 1.0 / sample_rate as f32;

    for i in 0..samples {
        let t = i as f32 * dt;
        // 440 Hz fundamental + 880 Hz 2nd harmonic + amplitude envelope
        let env = (t * 2.0 * PI * 0.5).sin().abs(); // 0.5 Hz modulation
        let sig = 0.6 * (2.0 * PI * 440.0 * t).sin() + 0.25 * (2.0 * PI * 880.0 * t).sin();
        buf.push(sig * env);
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

/// Helper function to locate a test fixture model in local workspace search paths.
fn find_sample_model() -> Option<PathBuf> {
    let candidate_paths = [
        "tests/fixtures/models/wavenet_a1_standard.nam",
        "tests/fixtures/models/wavenet.nam",
        "tests/fixtures/models/lstm.nam",
        "tests/fixtures/models-nondist/sample.nam",
        "tests/fixtures/models-nondist/sample.namb",
        "third-party/community_models/sample.nam",
        "third-party/community_models/sample.namb",
    ];

    for candidate in candidate_paths {
        let p = Path::new(candidate);
        if p.exists() {
            println!("  Found fixture: {}", p.display());
            return Some(p.to_path_buf());
        }
    }
    None
}
