// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Clean Off-RT Model Loading Example (`load_model`)
//!
//! This example demonstrates how to load and initialize Neural Amp Modeler
//! (`.nam` JSON or `.namb` binary profile) model files outside the real-time
//! audio processing thread.
//!
//! ## Overview
//!
//! Model loading involves file I/O, JSON parsing or binary deserialization,
//! memory allocations, and SIMD kernel prewarming. Therefore, model loading
//! MUST ALWAYS take place off the real-time audio thread. Once loaded, the
//! resulting `LoadedModelPair` can be passed to the DSP processing loop.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example load_model -- /path/to/model.nam
//! cargo run --example load_model -- /path/to/model.namb
//! ```

use std::path::{Path, PathBuf};
use std::process;

use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — Off-RT Model Loader Demonstration  ");
    println!("============================================================");

    // 1. Capture host CPU capabilities and SIMD feature snapshot (AVX2/AVX-512, topology).
    let sys = SystemSnapshot::capture();
    println!("\n[System Info]");
    println!("  CPU SIMD Features : AVX2/FMA baseline enabled");
    println!("  System Snapshot   : Captured hardware capability profile");

    // 2. Resolve model file path from CLI argument or search for common test fixtures.
    let path = match std::env::args().nth(1) {
        Some(arg) => PathBuf::from(arg),
        None => {
            println!("\n[Notice] No model path supplied as argument.");
            println!("Searching for local fixture models...");
            find_sample_model().unwrap_or_else(|| {
                println!("\n[Usage]");
                println!("  cargo run --example load_model -- <path-to-model.nam|.namb>");
                println!("\nError: Please specify a valid .nam or .namb model file.");
                process::exit(1);
            })
        }
    };

    if !path.exists() {
        eprintln!(
            "\nError: File standard path \"{}\" does not exist.",
            path.display()
        );
        process::exit(1);
    }

    println!("\n[Loading Model]");
    println!("  Target File : {}", path.display());

    let file_size = std::fs::metadata(&path)?.len();
    println!(
        "  File Size   : {} bytes ({:.2} KiB)",
        file_size,
        file_size as f64 / 1024.0
    );

    // 3. Configure load options (e.g. prewarm behavior).
    let options = LoadOptions::default();

    // 4. Perform off-RT loading and compilation of model data.
    let start_time = std::time::Instant::now();
    let model_pair = match load_and_build_model(&path, &sys, false, options) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!(
                "\n[Error] Failed to load model from \"{}\":",
                path.display()
            );
            eprintln!("  {:#}", err);
            process::exit(1);
        }
    };
    let elapsed = start_time.elapsed();

    // 5. Query and display model metadata & execution parameters.
    let info = model_pair.model_info(&path);

    println!("\n[Load Results]");
    println!("  Parse & Build Time : {:.2?}", elapsed);
    println!("  Architecture       : {}", model_pair.architecture);
    println!("  Topology           : {}", model_pair.topology);
    println!("  Sample Rate        : {} Hz", model_pair.sample_rate);
    println!("  Receptive Field    : {} samples", info.receptive_field);
    println!("  Weights Layout     : {}", model_pair.weights_layout);
    println!(
        "  Left Channel Model : Ready ({})",
        if model_pair.model_r.is_some() {
            "Stereo L"
        } else {
            "Mono"
        }
    );
    println!(
        "  Right Channel Model: {}",
        if model_pair.model_r.is_some() {
            "Ready (Stereo R)"
        } else {
            "None (Mono Load)"
        }
    );

    println!("\n[Gain Staging Metadata]");
    if let Some(loudness) = model_pair.loudness() {
        println!("  Loudness           : {:.2} dB", loudness);
    } else {
        println!("  Loudness           : Not specified in metadata");
    }

    if let Some(input_level) = model_pair.input_level_dbu() {
        println!("  Input Level        : {:.2} dBu", input_level);
    } else {
        println!("  Input Level        : Default (12.0 dBu assumed)");
    }

    if let Some(output_level) = model_pair.output_level_dbu() {
        println!("  Output Level       : {:.2} dBu", output_level);
    } else {
        println!("  Output Level       : Not specified in metadata");
    }

    println!("  Input Adj Mult     : {:.6}", model_pair.input_mult_adj);
    println!("  Output Adj Mult    : {:.6}", model_pair.output_mult_adj);

    println!("\n[Status] Model successfully loaded off-RT and ready for real-time DSP execution.");
    Ok(())
}

/// Helper function to locate a test fixture model in local workspace search paths.
fn find_sample_model() -> Option<PathBuf> {
    let candidate_paths = [
        "tests/fixtures/models/wavenet_a1_standard.nam",
        "tests/fixtures/models/wavenet.nam",
        "tests/fixtures/models/lstm.nam",
        "tests/fixtures/models-nondist/sample.nam",
        "tests/fixtures/models-nondist/sample.namb",
        "../third-party/nam_t3k/models/sample.nam",
        "../third-party/nam_t3k/models/sample.namb",
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
