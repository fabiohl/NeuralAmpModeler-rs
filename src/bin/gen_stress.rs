// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLI binary to generate deterministic stress signal WAV files.
//!
//! Replaces the Python block in `tests/fixtures/golden_gen_build.sh`.
//!
//! ## Usage
//! ```bash
//! cargo run --release --bin gen_stress -- --version v2 --sample-rate 48000 --output stress_signal.wav
//! cargo run --release --bin gen_stress -- --version v1 --output stress_signal_v1.wav
//! ```

use std::path::PathBuf;

fn main() {
    let args = parse_args();
    let signal = generate_signal(&args);
    write_output(&signal, &args);
}

struct Args {
    version: String,
    sample_rate: u32,
    output: PathBuf,
    seed: String,
}

fn parse_args() -> Args {
    let mut version = String::from("v2");
    let mut sample_rate: u32 = 48000;
    let mut output = PathBuf::from("stress_signal.wav");
    let mut seed = String::from("nam-rs-stress-v2");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--version" => {
                i += 1;
                if i < args.len() {
                    version = args[i].clone();
                }
            }
            "--sample-rate" => {
                i += 1;
                if i < args.len() {
                    sample_rate = args[i].parse().unwrap_or(48000);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = PathBuf::from(&args[i]);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].clone();
                }
            }
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let valid_rates = [44100u32, 48000, 88200, 96000, 192000];
    if !valid_rates.contains(&sample_rate) {
        eprintln!(
            "Warning: unsupported sample rate {sample_rate}. Supported: {:?}",
            valid_rates
        );
    }

    Args {
        version,
        sample_rate,
        output,
        seed,
    }
}

fn print_help() {
    eprintln!(
        r#"gen_stress — Deterministic stress signal WAV generator

Usage:
  gen_stress [OPTIONS]

Options:
  --version VERSION       Stress signal version: "v1" or "v2" (default: v2)
  --sample-rate RATE      Sample rate in Hz (default: 48000)
  --output PATH           Output WAV path (default: stress_signal.wav)
  --seed STRING           PRNG seed string (default: "nam-rs-stress-v2")
  --help, -h              Show this help message

Supported sample rates: 44100, 48000, 88200, 96000, 192000

Examples:
  gen_stress --version v2 --sample-rate 48000 --output stress_signal.wav
  gen_stress --version v1 --output stress_signal_v1.wav"#
    );
}

fn generate_signal(args: &Args) -> Vec<f32> {
    match args.version.as_str() {
        "v1" => {
            let sig = neural_amp_modeler_rs::testing::stress::generate_stress_signal_v1();
            eprintln!("Generated v1 stress signal: {} samples @ 48 kHz", sig.len());
            sig
        }
        "v2" => {
            let sig = neural_amp_modeler_rs::testing::stress::generate_stress_signal_v2(
                &args.seed,
                args.sample_rate,
            );
            eprintln!(
                "Generated v2 stress signal: {} samples @ {} Hz (seed: {})",
                sig.len(),
                args.sample_rate,
                args.seed
            );
            sig
        }
        other => {
            eprintln!("Unknown version: {other}. Use 'v1' or 'v2'.");
            std::process::exit(1);
        }
    }
}

fn write_output(signal: &[f32], args: &Args) {
    let sr = if args.version == "v1" {
        48000
    } else {
        args.sample_rate
    };

    neural_amp_modeler_rs::testing::wav::write_wav_f32(&args.output, signal, sr)
        .expect("Failed to write WAV file");

    eprintln!("Wrote: {}", args.output.display());
}
