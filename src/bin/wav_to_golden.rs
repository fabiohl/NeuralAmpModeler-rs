// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! CLI binary to convert rendered WAV output to `.golden.bin` format.
//!
//! Replaces the Python block in `tests/fixtures/golden_gen_build.sh` for
//! WAV → golden.bin conversion.
//!
//! ## Usage
//! ```bash
//! cargo run --release --bin wav_to_golden -- --input rendered.wav --reference stress.wav --output golden.bin
//! ```

use std::path::PathBuf;

fn main() {
    let args = parse_args();

    let (ref_samples, ref_sr) = neural_amp_modeler_rs::testing::wav::read_wav_f32(&args.reference)
        .expect("Failed to read reference WAV");
    let (test_samples, test_sr) = neural_amp_modeler_rs::testing::wav::read_wav_f32(&args.input)
        .expect("Failed to read input WAV");

    if ref_sr != test_sr {
        eprintln!(
            "Warning: sample rate mismatch — reference {} Hz, input {} Hz",
            ref_sr, test_sr
        );
    }

    let n_samples = ref_samples.len().min(test_samples.len());
    let ref_trimmed = &ref_samples[..n_samples];
    let test_trimmed = &test_samples[..n_samples];

    // Write golden.bin format: [u32 N] [f32×N input] [f32×N output]
    let mut buf = Vec::with_capacity(4 + n_samples * 8);

    buf.extend_from_slice(&(n_samples as u32).to_le_bytes());

    #[cfg(target_endian = "little")]
    {
        let ref_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(ref_trimmed.as_ptr() as *const u8, n_samples * 4) };
        buf.extend_from_slice(ref_bytes);

        let test_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(test_trimmed.as_ptr() as *const u8, n_samples * 4)
        };
        buf.extend_from_slice(test_bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &s in ref_trimmed {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        for &s in test_trimmed {
            buf.extend_from_slice(&s.to_le_bytes());
        }
    }

    std::fs::write(&args.output, &buf).expect("Failed to write golden.bin");

    let size_mb = buf.len() as f64 / 1_048_576.0;
    eprintln!(
        "Wrote {}: {} samples ({} input + {} output), {:.1} MB",
        args.output.display(),
        n_samples,
        n_samples,
        n_samples,
        size_mb
    );
}

struct Args {
    input: PathBuf,
    reference: PathBuf,
    output: PathBuf,
}

fn parse_args() -> Args {
    let mut input = PathBuf::from("rendered.wav");
    let mut reference = PathBuf::from("stress.wav");
    let mut output = PathBuf::from("golden.bin");

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                if i < args.len() {
                    input = PathBuf::from(&args[i]);
                }
            }
            "--reference" => {
                i += 1;
                if i < args.len() {
                    reference = PathBuf::from(&args[i]);
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output = PathBuf::from(&args[i]);
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

    Args {
        input,
        reference,
        output,
    }
}

fn print_help() {
    eprintln!(
        r#"wav_to_golden — Convert rendered WAV to .golden.bin format

Usage:
  wav_to_golden [OPTIONS]

Options:
  --input PATH        Rendered output WAV from NeuralAmpModelerCore (default: rendered.wav)
  --reference PATH    Reference stress signal WAV (default: stress.wav)
  --output PATH       Output .golden.bin file (default: golden.bin)
  --help, -h          Show this help message

Output format:
  [u32 num_samples LE] [f32×N input] [f32×N output LE]

Example:
  wav_to_golden --input rendered.wav --reference stress.wav --output golden_wavenet_standard.bin"#
    );
}
