// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! # Logging & Diagnostic Support Bundle Example (`diagnostics`)
//!
//! Demonstrates the logging facade (`NamLogger`), circular log buffer (`LogBuffer`),
//! system environment snapshot (`SystemSnapshot`), and support bundle rendering (`DiagnosticBundle`).
//!
//! ## Overview
//!
//! - **Unified Logging (`NamLogger`)**: Routes off-RT `log::*` macros to a global ring buffer (`LogBuffer`)
//!   and stderr, ensuring zero log allocations on the real-time audio hot path.
//! - **Privacy-Preserving Diagnostic Bundles (`DiagnosticBundle`)**: Captures OS metadata, CPU ISA support,
//!   memory limits, and recent execution traces with path redaction (`~` substitution for `$HOME`).
//! - **Error Diagnostics (`NamErrorCode`)**: Attaches structured error codes and key-value diagnostic parameters.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example diagnostics
//! ```

use log::{LevelFilter, debug, error, info, warn};
use neural_amp_modeler_rs::common::diagnostics::bundle::DiagnosticBundle;
use neural_amp_modeler_rs::common::diagnostics::error_codes::NamErrorCode;
use neural_amp_modeler_rs::common::diagnostics::logger::{LoggerConfig, NamLogger};
use neural_amp_modeler_rs::common::diagnostics::system_info::SystemSnapshot;
use neural_amp_modeler_rs::install_panic_hook;

fn main() {
    println!("============================================================");
    println!("  NeuralAmpModeler-rs — Diagnostics & Support Bundle Demo   ");
    println!("============================================================");

    // 1. Install zero-alloc panic hook for crash reporting.
    install_panic_hook("diagnostics-example");
    println!("\n[1/5] Installed Zero-Alloc Panic Hook");

    // 2. Initialize global logging facade (NamLogger).
    let config = LoggerConfig {
        level_filter: LevelFilter::Debug,
        emit_stderr: false, // Set to false here so example output is controlled in main
    };

    let _logger = NamLogger::init(config).expect("Failed to initialize NamLogger");
    println!("[2/5] Initialized NamLogger facade (LevelFilter::Debug)");

    // 3. Emit representative off-RT logs across various DSP subsystem components.
    println!("\n[3/5] Emitting Simulated Off-RT Logs");
    info!("Engine initialized with target rate: 48000 Hz, block size: 64 samples");
    debug!("AVX2 + FMA SIMD math kernels enabled unconditionally (x86-64-v3 baseline)");
    info!("Loading model from file: /home/user/models/vintage_cranked_amp.nam");
    info!("Parsed WaveNet model: 3 layers, 16 channels, receptive field = 63 samples");
    warn!(
        "IR sample rate mismatch: WAV rate = 44100 Hz, engine rate = 48000 Hz (triggering resampler)"
    );
    info!("Sinc polyphase resampler initialized: ratio = 1.0884 (44100 -> 48000 Hz)");
    error!(
        "Failed to open optional cabinet profile: /home/user/cabs/missing_cab.wav (File not found)"
    );

    // 4. Query global LogBuffer ring buffer.
    if let Some(log_buf) = NamLogger::log_buffer() {
        println!("  LogBuffer Capacity : {} entries", log_buf.capacity());
        println!("  LogBuffer Length   : {} records stored", log_buf.len());
    }

    // 5. Capture & render System Snapshot.
    println!("\n[4/5] System Hardware & Platform Snapshot");
    let sys = SystemSnapshot::capture();
    println!("  OS Name / Kernel   : {} | {}", sys.os, sys.kernel);
    println!("  CPU Architecture   : {}", sys.arch);
    println!("  CPU Features       : {:?}", sys.features);
    println!("  Engine Version     : {}", sys.version);

    // 6. Capture & render Redacted Diagnostic Bundle (default for public pastebin/issues).
    println!("\n[5/5] Rendering Diagnostic Support Bundles");

    println!("\n--- A. Redacted Support Bundle (Default for Public Sharing) ---");
    let err_params = vec![
        (
            "model_path",
            "/home/user/models/vintage_cranked_amp.nam".to_string(),
        ),
        ("target_rate", "48000".to_string()),
    ];
    let redacted_bundle =
        DiagnosticBundle::capture_with_error(NamErrorCode::FileNotFound, err_params.clone());
    let redacted_text = redacted_bundle.render();
    println!("{}", redacted_text);

    println!("\n--- B. Unredacted Support Bundle (Full Developer Mode: with_full(true)) ---");
    let full_bundle = DiagnosticBundle::capture_with_error(NamErrorCode::FileNotFound, err_params)
        .with_full(true);
    let full_text = full_bundle.render();
    println!("{}", full_text);

    println!("\n[Status] Diagnostics demonstration completed successfully.");
}
