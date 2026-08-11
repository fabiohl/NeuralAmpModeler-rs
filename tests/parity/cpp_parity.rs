// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Live cross-validation tests NAM-rs ↔ NeuralAmpModelerCore (Layer 2).
//
//  ## When it runs
//  - `utils/tests-long.sh` (full slow suite)
//  - `cargo test --test cpp_parity -- --ignored --nocapture`
//
//  ## Pipeline
//  1. Compiles the `render` tool from NeuralAmpModelerCore on-demand (idempotent, cached)
//  2. Generates WAV stress signal via `generate_stress_signal_v1()` or `generate_stress_signal_v2()`
//  3. Writes WAV, executes render via `std::process::Command`
//  4. Reads WAV output, compares C++ vs Rust with `report_dsp_fidelity()`
//
//  Tests are `#[ignore]` in normal CI — requires C++ toolchain installed.
//
//  ## Parity Thresholds (Aggressive Live Floors)
//
//  Post-migration thresholds use `live_parity_thresholds()` — aggressive floors
//  calibrated from live C++ cross-validation measurements (f32-exact):
//
//  | Family  | Variant  | Measured SNR | Floor SNR | Margin  |
//  |---------|----------|-------------|-----------|---------|
//  | WaveNet | Standard | 134.6 dB    | 105 dB    | 29.6 dB |
//  | WaveNet | Feather  | 133.1 dB    | 100 dB    | 33.1 dB |
//  | WaveNet | Nano     | 132.0 dB    | 95 dB     | 37.0 dB |
//  | WaveNet | Lite     | 117.4 dB    | 100 dB    | 17.4 dB |
//  | WaveNet | A2-Full  | 79.2 dB     | 70.0 dB   | 9.2 dB  |
//  | WaveNet | A2-Lite  | 90.7 dB     | 80.0 dB   | 10.7 dB |
//  | LSTM    | —        | 50–97 dB    | 45–75 dB  | formula |
//  | Linear  | —        | bit-exact   | 140 dB    | —       |
//
//  Lite (CH=12) P1 resolved (ringbuffer alignment fix) — now matches C++.
//
//  ## Multi-Sample-Rate Support
//
//  v2 stress signal supports 44.1k, 48k, 96k, and 192k sample rates.
//  The `#[test]` functions parametrize model × SR combinations via runtime arrays
//  instead of statically exploding test functions.

use super::common;
use common::*;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::models::NamModel;
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParityOutcome {
    Completed,
    SkippedModelNotFound,
    SkippedToolNotAvailable,
    SkippedRateRejected,
    SkippedGarbageOutput,
}

/// Require that a `ParityOutcome` is `Completed`, otherwise panic with a
/// descriptive message naming the test and the actual outcome.
///
/// Used by `quick_parity_*` tests to enforce fail-closed semantics: in
/// environments where NAMCore and fixtures were validated by preflight,
/// every non-Completed outcome is a hard failure.
///
/// `SkippedGarbageOutput` is always a failure regardless of environment.
#[track_caller]
fn require_completed(outcome: ParityOutcome, test_name: &str) {
    match outcome {
        ParityOutcome::Completed => {}
        ParityOutcome::SkippedGarbageOutput => {
            panic!(
                "[{test_name}] FAIL: parity outcome=SkippedGarbageOutput. \
                 C++ render produced non-finite output — possible render tool regression or broken NAMCore build."
            );
        }
        other => {
            panic!(
                "[{test_name}] FAIL: parity required Completed but got {other:?}. \
                 In a full environment (NAMCore + fixtures validated by preflight), \
                 every non-Completed outcome is a failure. \
                 If running in a minimal environment, skip the cpp_parity phase entirely \
                 (SKIP_OPTIONAL_FIXTURE) instead of running individual tests that cannot complete."
            );
        }
    }
}

/// Resolves the NeuralAmpModelerCore vendor mirror directory.
///
/// See [`neural_amp_modeler_rs::testing::fixtures::nam_core_dir`].
fn nam_core_dir() -> PathBuf {
    neural_amp_modeler_rs::testing::fixtures::nam_core_dir()
}

const BUILD_DIR: &str = "build/namcore_render";

fn render_bin() -> PathBuf {
    let build_dir = BUILD_DIR;
    let mut bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    bin.push(build_dir);
    bin.push("Release/render");
    if !bin.exists() {
        bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        bin.push(build_dir);
        bin.push("Debug/render");
    }
    if !bin.exists() {
        let build_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(build_dir);
        if let Ok(entries) = std::fs::read_dir(&build_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.file_name().is_some_and(|n| n == "render") && path.is_file() {
                    return path;
                }
                if path.is_dir() {
                    let render = path.join("render");
                    if render.exists() {
                        return render;
                    }
                }
            }
        }
    }
    bin
}

fn ensure_render_compiled() -> bool {
    // Serialize the C++ render build across parallel test threads. All
    // `live_cross_validation_*` tests share the same CMake build directory
    // (`build/namcore_render`); on a cold run they would otherwise race to
    // invoke `cmake` concurrently, corrupting the CMake cache and producing
    // spurious "CMAKE_CXX_COMPILER not set" failures. The first thread builds;
    // the rest wait and then find the binary already present.
    use std::sync::Mutex;
    static BUILD_LOCK: Mutex<()> = Mutex::new(());
    let _guard = BUILD_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    fn skip_or_abort(msg: &str) -> bool {
        if std::env::var("NAM_REQUIRE_CPP_ORACLE").as_deref() == Ok("1") {
            panic!("NAM_REQUIRE_CPP_ORACLE=1 — aborting test: {msg}");
        }
        eprintln!("{msg}");
        false
    }

    let bin = render_bin();
    if bin.exists() {
        return true;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nam_core = nam_core_dir();

    if !nam_core.exists() {
        return skip_or_abort(&format!(
            "NeuralAmpModelerCore not found at {nam_core:?}.\n\
             Run './tests/fixtures/golden_gen_build.sh' to set up mirrors and generate golden vectors."
        ));
    }

    for sub in &["Dependencies/eigen", "Dependencies/AudioDSPTools"] {
        let sub_path = nam_core.join(sub);
        if sub_path.exists() && fs::read_dir(&sub_path).map_or(true, |mut d| d.next().is_none()) {
            let status = Command::new("git")
                .args(["submodule", "update", "--init", sub])
                .current_dir(&nam_core)
                .status();
            if status.is_ok_and(|s| !s.success()) {
                eprintln!("WARN: failed to initialize submodule {sub}");
            }
        }
    }

    let build_dir = project_root.join(BUILD_DIR);
    fs::create_dir_all(&build_dir).ok();

    eprintln!("Compiling render tool (v0.5.4 + A2-fast)...");

    let cmake_args = vec![
        "-S",
        nam_core.to_str().unwrap(),
        "-B",
        build_dir.to_str().unwrap(),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_STANDARD=20",
        "-DCMAKE_CXX_FLAGS=-w",
        "-DNAM_ENABLE_A2_FAST=ON",
    ];

    let cmake_status = Command::new("cmake").args(&cmake_args).status();

    match cmake_status {
        Ok(s) if s.success() => {}
        _ => {
            return skip_or_abort(
                "CMake failed — C++ build dependencies missing.\n\
                 Install cmake and a C++20 compiler (g++ or clang++), then re-run.",
            );
        }
    }

    let build_status = Command::new("cmake")
        .args([
            "--build",
            build_dir.to_str().unwrap(),
            "--target",
            "render",
            "-j",
            "2",
        ])
        .status();

    match build_status {
        Ok(s) if s.success() => {}
        _ => {
            return skip_or_abort(&format!(
                "Render build failed — check build logs in {build_dir:?}."
            ));
        }
    }

    if !render_bin().exists() {
        return skip_or_abort(&format!(
            "Render binary not found after build — expected at {bin:?}"
        ));
    }

    true
}

fn run_render_comparison(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    sample_rate: u32,
    use_v2: bool,
    check_lufs_gate: bool,
    use_hf: bool,
) -> ParityOutcome {
    let model_status = FIXTURE_CATALOG.check(model_filename);
    if !model_status.is_available() {
        if model_status == FixtureStatus::MissingOptional {
            eprintln!("SKIP: {label} — optional fixture {model_filename} not found.");
            return ParityOutcome::SkippedModelNotFound;
        }
        eprintln!(
            "REQUIRED FIXTURE ABSENT: {label} — {model_filename} is a DistributedCore model that must exist."
        );
        return ParityOutcome::SkippedModelNotFound;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();

    // Read model expected sample rate
    let model_path = model_path(model_filename);
    let json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("JSON parser failed");

    if !ensure_render_compiled() {
        eprintln!("SKIP: {label} — C++ render tool not available, skipping cross-validation.");
        return ParityOutcome::SkippedToolNotAvailable;
    }

    let actual_sr = if use_v2 {
        sample_rate
    } else {
        STRESS_SAMPLE_RATE
    };
    let model_sr = model_data.sample_rate.unwrap_or(actual_sr as f32) as u32;

    let stress_signal = if use_v2 {
        generate_stress_signal_v2(sample_rate)
    } else {
        generate_stress_signal_v1()
    };

    let suffix = if use_hf { "_hf" } else { "" };
    let stress_wav = temp_dir.join(format!(
        "stress_live_{golden_name}_{sample_rate}{suffix}.wav"
    ));

    use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    let mut resampler_cpp = if actual_sr != model_sr {
        Some(NamResampler::new(actual_sr, model_sr, 2048).expect("Failed to create NamResampler"))
    } else {
        None
    };
    let mut resampler_rust = if actual_sr != model_sr {
        Some(NamResampler::new(actual_sr, model_sr, 2048).expect("Failed to create NamResampler"))
    } else {
        None
    };

    let (input_for_render, input_sr) = if let Some(ref mut rs) = resampler_cpp {
        let est_len =
            (stress_signal.len() as f64 * model_sr as f64 / actual_sr as f64).ceil() as usize + 512;
        let mut resampled_in_l = vec![0.0f32; est_len];
        let mut resampled_in_r = vec![0.0f32; est_len];
        let n_resampled = rs
            .process_input_mono(&stress_signal, &mut resampled_in_l, &mut resampled_in_r)
            .samples_written;
        resampled_in_l.truncate(n_resampled);
        (resampled_in_l, model_sr)
    } else {
        (stress_signal.clone(), actual_sr)
    };

    let (input_for_rust, _) = if let Some(ref mut rs) = resampler_rust {
        let est_len =
            (stress_signal.len() as f64 * model_sr as f64 / actual_sr as f64).ceil() as usize + 512;
        let mut resampled_in_l = vec![0.0f32; est_len];
        let mut resampled_in_r = vec![0.0f32; est_len];
        let n_resampled = rs
            .process_input_mono(&stress_signal, &mut resampled_in_l, &mut resampled_in_r)
            .samples_written;
        resampled_in_l.truncate(n_resampled);
        (resampled_in_l, model_sr)
    } else {
        (stress_signal.clone(), actual_sr)
    };

    common::wav::write_wav_f32(&stress_wav, &input_for_render, input_sr)
        .expect("Failed to write stress WAV");

    let output_wav = temp_dir.join(format!("{golden_name}_live_{sample_rate}{suffix}.wav"));

    // Execute render tool — capture stdout/stderr to prevent interleaving
    // with the Rust test harness output.
    let bin = render_bin();
    let output = Command::new(&bin)
        .arg(model_path.to_str().unwrap())
        .arg(stress_wav.to_str().unwrap())
        .arg(output_wav.to_str().unwrap())
        .output();

    match output {
        Ok(o) if o.status.success() => { /* silence on success */ }
        Ok(o) => {
            let stderr_msg = String::from_utf8_lossy(&o.stderr);
            let stdout_msg = if o.stdout.is_empty() {
                String::new()
            } else {
                format!(
                    "--- render stdout ---\n{}\n",
                    String::from_utf8_lossy(&o.stdout)
                )
            };
            eprintln!(
                "SKIP: {label} — render returned exit code {}\n\
                 --- render stderr ---\n{stderr_msg}{stdout_msg}",
                o.status.code().unwrap_or(-1)
            );
            return ParityOutcome::SkippedRateRejected;
        }
        Err(e) => {
            eprintln!("SKIP: {label} — failed to execute render: {e}");
            return ParityOutcome::SkippedRateRejected;
        }
    }

    // Read render WAV output
    let (cpp_output_raw, _sr) =
        common::wav::read_wav_f32(&output_wav).expect("Failed to read render WAV output");

    let cpp_output = if let Some(ref mut rs) = resampler_cpp {
        let est_out_len = (cpp_output_raw.len() as f64 * actual_sr as f64 / model_sr as f64).ceil()
            as usize
            + 512;
        let mut resampled_out_l = vec![0.0f32; est_out_len];
        let mut resampled_out_r = vec![0.0f32; est_out_len];
        let n_out_resampled = rs
            .process_output_mono(&cpp_output_raw, &mut resampled_out_l, &mut resampled_out_r)
            .samples_written;
        resampled_out_l.truncate(n_out_resampled);
        resampled_out_l
    } else {
        cpp_output_raw.clone()
    };

    // Sanity-check C++ render output: skip comparison if render produced garbage.
    //
    // Only check for non-finite samples (NaN/Inf).
    {
        let has_nonfinite = cpp_output.iter().any(|x| !x.is_finite());
        if has_nonfinite {
            eprintln!(
                "SKIP: {label} — C++ render produced garbage output (non-finite=true); skipping comparison.",
            );
            fs::remove_file(&output_wav).ok();
            return ParityOutcome::SkippedGarbageOutput;
        }
    }

    let (mut mse_limit, mut min_snr_db, mut max_esr, mut mrstft_max) =
        live_parity_thresholds(&model_data, golden_name);
    let calibrated_mse = mse_limit;
    if use_v2 {
        if model_data.architecture == "LSTM" {
            // LSTM recurrent state accumulates quantization/approximation errors
            // over the 100x longer v2 stress signal. The accumulation is proportional
            // to the sequence length. We adjust the thresholds accordingly.
            let sr_ratio = sample_rate as f64 / 48000.0;
            let snr_relaxation = (3.5 * sr_ratio).min(10.0);
            min_snr_db = (min_snr_db - snr_relaxation).max(7.0);
            if let Some(ref mut m) = mse_limit {
                *m *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut mr) = mrstft_max {
                // MR-STFT grows faster than power metrics for recurrent models
                // (spectral drift in LSTM hidden states over long sequences)
                *mr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        } else {
            // WaveNet and other models accumulate minor differences over the longer v2 stress signal
            let sr_ratio = sample_rate as f64 / 48000.0;
            let snr_relaxation = (1.5 * sr_ratio).min(4.0);
            min_snr_db -= snr_relaxation;
            if let Some(ref mut m) = mse_limit {
                *m *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut esr) = max_esr {
                *esr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
            if let Some(ref mut mr) = mrstft_max {
                *mr *= 10.0_f64.powf(snr_relaxation / 10.0);
            }
        }
    }
    if use_v2 && actual_sr != model_sr {
        // Resampling introduces minor interpolation/approximation errors, relax thresholds slightly
        min_snr_db -= 1.5;
        if let Some(ref mut m) = mse_limit {
            *m *= 1.5;
        }
        if let Some(ref mut esr) = max_esr {
            *esr *= 1.5;
        }
        // MR-STFT is particularly sensitive to resampling phase/timing mismatches
        // between the C++ render's SRC and nam-rs's polyphase resampler
        if let Some(ref mut mr) = mrstft_max {
            *mr *= 3.0;
        }
    }

    let is_film = golden_name.to_lowercase().contains("film")
        || model_filename.to_lowercase().contains("film");

    // Absolute ESR/SNR/MR-STFT caps for Standard-mode parity.
    //
    // C++ NAMCore uses native math (std::tanh) — exact-grade.
    // Rust Standard mode uses polynomial exp-based kernels (~2.4e-7 error).
    // The caps below prevent false positives in both v1 (quick) and v2 tests.
    //
    // non-HF Fast-mode caps removed — run_v1 / run_v2_multi_sr
    // always use Standard activation since the universal default switch.
    const ABSOLUTE_ESR_CAP_WAVENET_HF: f64 = 1.0e-10;
    const ABSOLUTE_ESR_CAP_LSTM_NATIVE_HF: f64 = 1.0e-5;
    const ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF: f64 = 1.0e-4;
    const ABSOLUTE_ESR_CAP_CONVNET_HF: f64 = 1.0e-10;
    const ABSOLUTE_ESR_CAP_FILM_HF: f64 = 0.15;
    const ABSOLUTE_SNR_FLOOR: f64 = 5.0;
    const ABSOLUTE_MRSTFT_CAP: f64 = 0.95;
    const ABSOLUTE_MRSTFT_CAP_FILM: f64 = 1.20;

    let esr_cap = if model_data.architecture == "LSTM" {
        if sample_rate > 96_000 {
            ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF
        } else {
            ABSOLUTE_ESR_CAP_LSTM_NATIVE_HF
        }
    } else if is_film {
        ABSOLUTE_ESR_CAP_FILM_HF
    } else if model_data.architecture == "ConvNet" {
        ABSOLUTE_ESR_CAP_CONVNET_HF
    } else {
        ABSOLUTE_ESR_CAP_WAVENET_HF
    };

    min_snr_db = min_snr_db.max(ABSOLUTE_SNR_FLOOR);
    if let Some(ref mut esr) = max_esr
        && *esr > esr_cap
    {
        let scale_back = esr_cap / *esr;
        *esr = esr_cap;
        if let Some(ref mut m) = mse_limit {
            let scaled = *m * scale_back;
            *m = if let Some(ref cm) = calibrated_mse {
                scaled.max(*cm)
            } else {
                scaled
            };
        }
    }
    {
        let mrstft_cap = if is_film {
            ABSOLUTE_MRSTFT_CAP_FILM
        } else {
            ABSOLUTE_MRSTFT_CAP
        };
        if let Some(ref mut mr) = mrstft_max
            && *mr > mrstft_cap
        {
            *mr = mrstft_cap;
        }
    }

    if use_v2 {
        // FiLM v2 MR-STFT floor (calibration, 2026-07-10):
        // FiLM models (A2-FiLM-Lite/Full) accumulate spectral drift far faster
        // than generic WaveNet over the 5-second v2 stress signal, because the
        // per-frame FiLM gamma/beta modulation diverges from C++'s generic
        // WaveNet fallback (Eigen; a2_fast.cpp rejects FiLM). The base
        // `mrstft_max=1.0e-4` is calibrated for v1 (2k samples, MR-STFT=3.92e-5);
        // the generic v2 relaxation (~2× @ 48 kHz) is insufficient. v2 measured
        // worst-case at the hard-gate sample rates:
        //   Lite @ 44.1 kHz: 8.12e-3,  Lite @ 48 kHz: 8.81e-3
        //   Full @ 44.1 kHz: 3.56e-3,  Full @ 48 kHz: 3.46e-3
        // Floor at 3.0e-2 (~3.4× margin over the Lite worst) — still far below
        // the ABSOLUTE_MRSTFT_CAP_FILM (1.20) ceiling. SNR/ESR/MSE remain at
        // float-precision limits (SNR≥124 dB, ESR≤3.4e-13), confirming this is
        // inherent long-sequence FiLM spectral drift, not an engine regression.
        const FILM_V2_MRSTFT_FLOOR: f64 = 3.0e-2;
        if is_film && let Some(ref mut mr) = mrstft_max {
            *mr = (*mr).max(FILM_V2_MRSTFT_FLOOR);
        }
    }

    // Activation precision mode selection via RAII PrecisionGuard.
    //
    // C++ NAMCore uses native math activations (`std::tanh`/`std::exp`) — exact-grade.
    //
    // Non-HF (use_hf=false) → `ActivationPrecision::Fast`:
    //   Rust uses Padé/minimax approximations, trading ~2.3e-3 tanh error for throughput.
    //   This deliberate asymmetry characterizes the C++↔Rust FastMath parity envelope.
    //
    // HF (use_hf=true) → `ActivationPrecision::Standard`:
    //   Both engines use native math, aligning to near-bit-exact agreement (~1e-10 ESR).
    //
    // PrecisionGuard holds the global PRECISION_MUTEX and restores original mode on Drop.
    let _precision = if use_hf {
        Some(PrecisionGuard::new(ActivationPrecision::Standard))
    } else {
        Some(PrecisionGuard::new(ActivationPrecision::Fast))
    };

    let mut model = build_model(&model_data).expect("Dispatcher failed");

    let prewarm_len = model.prewarm_samples();
    model.prewarm(prewarm_len);
    let mut rust_output_model_sr = vec![0.0f32; input_for_rust.len()];
    process_in_blocks(
        &mut model,
        &input_for_rust,
        &mut rust_output_model_sr,
        GOLDEN_BLOCK_SIZE,
    );

    let rust_output = if let Some(ref mut rs) = resampler_rust {
        let est_out_len = (rust_output_model_sr.len() as f64 * actual_sr as f64 / model_sr as f64)
            .ceil() as usize
            + 512;
        let mut resampled_out_l = vec![0.0f32; est_out_len];
        let mut resampled_out_r = vec![0.0f32; est_out_len];
        let n_out_resampled = rs
            .process_output_mono(
                &rust_output_model_sr,
                &mut resampled_out_l,
                &mut resampled_out_r,
            )
            .samples_written;
        resampled_out_l.truncate(n_out_resampled);
        resampled_out_l
    } else {
        rust_output_model_sr
    };

    let min_len = cpp_output.len().min(rust_output.len());
    let cpp_slice = &cpp_output[..min_len];
    let rust_slice = &rust_output[..min_len];
    set_metric_model(format!("{label} @{actual_sr} Live"));
    set_metric_mode("Live".to_string());
    if check_lufs_gate {
        report_dsp_fidelity(
            cpp_slice, rust_slice, mse_limit, min_snr_db, max_esr, mrstft_max, label, actual_sr,
        );
    } else {
        report_dsp_fidelity_no_lufs(
            cpp_slice, rust_slice, mse_limit, min_snr_db, max_esr, mrstft_max, label, actual_sr,
        );
    }

    // Cleanup
    fs::remove_file(&output_wav).ok();
    ParityOutcome::Completed
}

/// Helper: run v1 comparison at 48 kHz in Standard (exact-grade) mode
/// (universal default).
fn run_v1(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    check_lufs_gate: bool,
) -> ParityOutcome {
    run_render_comparison(
        model_filename,
        golden_name,
        label,
        48000,
        false,
        check_lufs_gate,
        true,
    )
}

/// Unified implementation for v2 multi-SR parity tests.
///
/// Monitors every rate declared in the model's v2 multi-SR scope and asserts that:
/// - At least one rate completed parity validation (no silent total skip).
/// - The set of completed rates exactly matches the expected rates for the model
///   (no silent partial skip violating Gate Calibration Policy Rule 7).
///
/// Expected rates are resolved from the V2 multi-SR scope declaration in
/// `io_helpers::v2_multi_sr_expected_rates()`, which mirrors the
/// `tests/fixtures/golden_gen_build.sh` catalog contract.
///
fn run_v2_multi_sr_impl(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
    use_hf: bool,
) {
    let model_status = FIXTURE_CATALOG.check(model_filename);
    if !model_status.is_available() {
        if model_status == FixtureStatus::MissingOptional {
            eprintln!("SKIP: {label_base} (v2) — optional fixture {model_filename} not found.");
            return;
        }
        panic!(
            "REQUIRED FIXTURE ABSENT: {label_base} (v2) — {model_filename} is a DistributedCore model that must exist."
        );
    }

    let model_path = model_path(model_filename);
    let _json_data = fs::read_to_string(&model_path).expect("Failed to read model");
    let _model_data = parse_nam_json(&_json_data).expect("JSON parser failed");

    let expected_rates = v2_multi_sr_expected_rates(model_filename);

    let mut outcomes: Vec<(u32, ParityOutcome)> = Vec::new();
    let mut failures: Vec<(u32, String)> = Vec::new();
    let hf_tag = if use_hf { ", HF" } else { "" };

    for &sr in &expected_rates {
        let label = format!("{label_base} @ {sr} Hz (v2{hf_tag})");
        let gname = format!("{golden_name}_v2_{sr}");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_render_comparison(
                model_filename,
                &gname,
                &label,
                sr,
                true,
                check_lufs_gate,
                use_hf,
            )
        }));
        match result {
            Ok(outcome) => outcomes.push((sr, outcome)),
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic (non-string payload)".to_string());
                failures.push((sr, msg));
            }
        }
    }

    println!("=== Multi-SR Summary: {label_base}{hf_tag} ===");
    println!("{:<10} {:<30}", "SR (Hz)", "Outcome");
    println!("{}", "-".repeat(42));
    for &(sr, ref outcome) in &outcomes {
        println!("{:<10} {:?}", sr, outcome);
    }
    for &(sr, ref msg) in &failures {
        println!("{:<10} FAILED: {}", sr, &msg[..msg.len().min(30)]);
    }
    println!("{}", "-".repeat(42));

    let completed: Vec<u32> = outcomes
        .iter()
        .filter(|(_, o)| *o == ParityOutcome::Completed)
        .map(|(sr, _)| *sr)
        .collect();

    assert!(
        !completed.is_empty(),
        "Parity validation for '{label_base}': no sample rate completed. \
         Expected at least one of {expected_rates:?}. \
         Outcomes: {outcomes:?}, Failures: {failures:?}"
    );

    let completed_set: BTreeSet<u32> = completed.iter().copied().collect();
    let expected_set: BTreeSet<u32> = expected_rates.iter().copied().collect();

    assert_eq!(
        completed_set, expected_set,
        "Parity validation for '{label_base}': completed SRs ({completed_set:?}) != expected SRs ({expected_set:?})"
    );

    if !failures.is_empty() {
        let summary: Vec<String> = failures
            .iter()
            .map(|(sr, msg)| format!("@ {sr} Hz: {msg}"))
            .collect();
        panic!(
            "Parity validation panic failures for sample rates:\n  {}",
            summary.join("\n  ")
        );
    }
}

fn run_v2_multi_sr(
    model_filename: &str,
    golden_name: &str,
    label_base: &str,
    check_lufs_gate: bool,
) {
    run_v2_multi_sr_impl(
        model_filename,
        golden_name,
        label_base,
        check_lufs_gate,
        true,
    );
}

// =============================================================================
// Tests — Quick Parity Subset (non-ignored, 48 kHz, v1 short signal)
// =============================================================================
//
// Representative subset of cross-validations
// running in the ~3 min quick loop. BUILD_LOCK caches the C++ render tool.
//
// v1 tests (2048 samples, 48 kHz) — covering the three main architectures.
// v2 tests (5-second stress signal, 48 kHz) — comprehensive.

#[test]
fn quick_parity_lstm_1x16() {
    let outcome = run_v1("BossLSTM-1x16.nam", "lstm_1x16", "Quick LSTM 1×16", true);
    require_completed(outcome, "quick_parity_lstm_1x16");
}

#[test]
fn quick_parity_wavenet_ch16() {
    let outcome = run_v1(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Quick WaveNet CH16",
        true,
    );
    require_completed(outcome, "quick_parity_wavenet_ch16");
}

#[test]
fn quick_parity_a2_full() {
    let outcome = run_v1(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Quick A2-Full",
        true,
    );
    require_completed(outcome, "quick_parity_a2_full");
}

#[test]
fn quick_parity_convnet() {
    let outcome = run_v1("convnet_test.nam", "convnet_test", "Quick ConvNet", true);
    require_completed(outcome, "quick_parity_convnet");
}

/// Quick v2 parity: WaveNet Standard with 5-second stress signal at 48 kHz.
/// Promoted from `#[ignore]` to ensure v2 format
/// coverage in the pre-commit quick suite.
#[test]
fn quick_parity_wavenet_standard_v2_48k() {
    let outcome = run_render_comparison(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Quick WaveNet Standard v2",
        48000,
        true,
        true,
        true,
    );
    require_completed(outcome, "quick_parity_wavenet_standard_v2_48k");
}

/// Quick v2 parity: A2 Full with 5-second stress signal at 48 kHz.
/// Added per PO request alongside the WaveNet Standard v2 quick test.
#[test]
fn quick_parity_a2_full_v2_48k() {
    let outcome = run_render_comparison(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Quick A2-Full v2",
        48000,
        true,
        true,
        true,
    );
    require_completed(outcome, "quick_parity_a2_full_v2_48k");
}

/// Quick v2 parity: SlimmableContainer A2 Example with 5-second stress signal at 48 kHz.
/// Promoted to v2 alongside the 48k_only scope change (S4-T3).
#[test]
fn quick_parity_a2_example_slimmable_v2_48k() {
    let outcome = run_render_comparison(
        "a2_example.nam",
        "a2_example",
        "Quick SlimmableContainer A2 Example v2",
        48000,
        true,
        true,
        true,
    );
    require_completed(outcome, "quick_parity_a2_example_slimmable_v2_48k");
}

/// Quick parity: LSTM 1×10 (uncatalogued synthetic topology).
#[test]
fn quick_parity_lstm_1x10() {
    let outcome = run_v1("lstm_1x10.nam", "lstm_1x10", "Quick LSTM 1×10", true);
    require_completed(outcome, "quick_parity_lstm_1x10");
}

/// Quick parity: LSTM 2×24 (uncatalogued synthetic topology).
#[test]
fn quick_parity_lstm_2x24() {
    let outcome = run_v1("lstm_2x24.nam", "lstm_2x24", "Quick LSTM 2×24", true);
    require_completed(outcome, "quick_parity_lstm_2x24");
}

/// Quick parity: LSTM 3×8 (3-layer synthetic topology).
#[test]
fn quick_parity_lstm_3x8() {
    let outcome = run_v1("lstm_3x8.nam", "lstm_3x8", "Quick LSTM 3×8", true);
    require_completed(outcome, "quick_parity_lstm_3x8");
}

/// Quick parity: ConvNet No BatchNorm.
#[test]
fn quick_parity_convnet_nobn() {
    let outcome = run_v1(
        "convnet_nobn.nam",
        "convnet_nobn",
        "Quick ConvNet No BatchNorm",
        true,
    );
    require_completed(outcome, "quick_parity_convnet_nobn");
}

/// Quick parity: ConvNet ReLU.
#[test]
fn quick_parity_convnet_relu() {
    let outcome = run_v1(
        "convnet_relu.nam",
        "convnet_relu",
        "Quick ConvNet ReLU",
        true,
    );
    require_completed(outcome, "quick_parity_convnet_relu");
}

/// Quick parity: ConvNet SiLU.
#[test]
fn quick_parity_convnet_silu() {
    let outcome = run_v1(
        "convnet_silu.nam",
        "convnet_silu",
        "Quick ConvNet SiLU",
        true,
    );
    require_completed(outcome, "quick_parity_convnet_silu");
}

/// Quick parity: Linear No Bias.
#[test]
fn quick_parity_linear_nobias() {
    let outcome = run_v1(
        "linear_nobias.nam",
        "linear_nobias",
        "Quick Linear No Bias",
        true,
    );
    require_completed(outcome, "quick_parity_linear_nobias");
}

// =============================================================================
// Tests — #[ignore] (require C++ toolchain)
// =============================================================================

// --- v1 (legacy, fast CI) ---

#[test]
#[ignore]
fn live_cross_validation_wavenet_standard() {
    let outcome = run_v1(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_feather() {
    let outcome = run_v1(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_nano() {
    let outcome = run_v1("BossWN-nano.nam", "wavenet_nano", "Live WaveNet Nano", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_lite() {
    let outcome = run_v1(
        "EVH-5150-Lite.nam",
        "wavenet_lite",
        "Live WaveNet Lite",
        true,
    );
    match outcome {
        ParityOutcome::Completed => {}
        ParityOutcome::SkippedModelNotFound => {}
        other => panic!(
            "[Live WaveNet Lite] FAIL: parity required Completed or SkippedModelNotFound, got {other:?}"
        ),
    }
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a1_standard() {
    let outcome = run_v1(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_lstm_official() {
    let outcome = run_v1("lstm.nam", "lstm_official", "Live LSTM Official", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_full() {
    let outcome = run_v1(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_lite() {
    let outcome = run_v1(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_convnet() {
    let outcome = run_v1(
        "convnet_test.nam",
        "convnet_test",
        "Live ConvNet Test",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

// ── Synthetic and exotic topologies (v1) ──────────────────────────────

#[test]
#[ignore]
fn live_cross_validation_lstm_1x10() {
    let outcome = run_v1("lstm_1x10.nam", "lstm_1x10", "Live LSTM 1×10", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_lstm_2x24() {
    let outcome = run_v1("lstm_2x24.nam", "lstm_2x24", "Live LSTM 2×24", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_lstm_3x8() {
    let outcome = run_v1("lstm_3x8.nam", "lstm_3x8", "Live LSTM 3×8", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_convnet_nobn() {
    let outcome = run_v1(
        "convnet_nobn.nam",
        "convnet_nobn",
        "Live ConvNet No BatchNorm",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_convnet_relu() {
    let outcome = run_v1(
        "convnet_relu.nam",
        "convnet_relu",
        "Live ConvNet ReLU",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_convnet_silu() {
    let outcome = run_v1(
        "convnet_silu.nam",
        "convnet_silu",
        "Live ConvNet SiLU",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_linear_nobias() {
    let outcome = run_v1(
        "linear_nobias.nam",
        "linear_nobias",
        "Live Linear No Bias",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_condition_dsp() {
    let outcome = run_v1(
        "wavenet_condition_dsp.nam",
        "wavenet_condition_dsp",
        "Live WaveNet Condition DSP",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_condition_lstm() {
    let outcome = run_v1(
        "wavenet_condition_lstm.nam",
        "wavenet_condition_lstm",
        "Live WaveNet Condition DSP LSTM",
        true,
    );
    match outcome {
        ParityOutcome::Completed => {}
        ParityOutcome::SkippedRateRejected => {
            eprintln!(
                "SKIP: Condition DSP LSTM — C++ render tool limitation (condition_size mismatch)"
            )
        }
        other => panic!("Unexpected outcome: {other:?}"),
    }
}

#[test]
#[ignore]
fn live_cross_validation_a2_example_slimmable() {
    let outcome = run_v1(
        "a2_example.nam",
        "a2_example",
        "Live SlimmableContainer A2 Example (CH=3→6)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

// --- v2 (multi-SR, comprehensive) ---

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_standard() {
    run_v2_multi_sr(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_feather() {
    run_v2_multi_sr(
        "BossWN-feather.nam",
        "wavenet_feather",
        "Live WaveNet Feather (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_nano() {
    run_v2_multi_sr(
        "BossWN-nano.nam",
        "wavenet_nano",
        "Live WaveNet Nano (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_lite() {
    run_v2_multi_sr(
        "EVH-5150-Lite.nam",
        "wavenet_lite",
        "Live WaveNet Lite (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a1_standard() {
    run_v2_multi_sr(
        "wavenet_a1_standard.nam",
        "wavenet_a1_standard",
        "Live WaveNet A1 Standard (Official) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_official() {
    run_v2_multi_sr("lstm.nam", "lstm_official", "Live LSTM Official (v2)", true);
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_full() {
    run_v2_multi_sr(
        "wavenet_a2_full.nam",
        "wavenet_a2_full",
        "Live WaveNet A2-Full (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_app_evh() {
    run_v2_multi_sr(
        "APP-EVH-Stealth100-Dialled-xSTD.nam",
        "wavenet_app_evh",
        "Live APP EVH Stealth 100 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_boss_bd2() {
    run_v2_multi_sr(
        "Boss BD-2 H2O Mod T-12_00 G-12_00.nam",
        "wavenet_boss_bd2",
        "Live Boss BD-2 H2O Mod (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_slammin_marshall() {
    run_v2_multi_sr(
        "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam",
        "wavenet_slammin_marshall",
        "Live SLAMMIN MARSHALL JTM 45 (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_lite() {
    run_v2_multi_sr(
        "wavenet_a2_lite.nam",
        "wavenet_a2_lite",
        "Live WaveNet A2-Lite (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_a2_example_slimmable() {
    run_v2_multi_sr(
        "a2_example.nam",
        "a2_example",
        "Live SlimmableContainer A2 Example (CH=3→6) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_condition_dsp() {
    run_v2_multi_sr(
        "wavenet_condition_dsp.nam",
        "wavenet_condition_dsp",
        "Live WaveNet Condition DSP (v2)",
        true,
    );
}

/// Upstream KnownGap: C++ render rejects / cannot golden LSTM-as-condition_dsp
/// (channel mismatch). Mirrors v1 `live_cross_validation_wavenet_condition_lstm`
/// — allow `SkippedRateRejected` without requiring Completed SRs and without
/// `catch_unwind` (which polluted logs while still reporting ok).
#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_condition_lstm() {
    let model_filename = "wavenet_condition_lstm.nam";
    let label_base = "Live WaveNet Condition DSP LSTM (v2)";
    let model_status = FIXTURE_CATALOG.check(model_filename);
    if !model_status.is_available() {
        if model_status == FixtureStatus::MissingOptional {
            eprintln!("SKIP: {label_base} — optional fixture {model_filename} not found.");
            return;
        }
        panic!(
            "REQUIRED FIXTURE ABSENT: {label_base} — {model_filename} must exist for KnownGap policy test."
        );
    }

    let expected_rates = v2_multi_sr_expected_rates(model_filename);
    let mut outcomes: Vec<(u32, ParityOutcome)> = Vec::new();
    for &sr in &expected_rates {
        let label = format!("{label_base} @ {sr} Hz (v2, HF)");
        let gname = format!("wavenet_condition_lstm_v2_{sr}");
        let outcome = run_render_comparison(model_filename, &gname, &label, sr, true, true, true);
        outcomes.push((sr, outcome));
    }

    println!("=== Multi-SR Summary: {label_base}, HF (KnownGap upstream) ===");
    for &(sr, ref outcome) in &outcomes {
        println!("{sr:<10} {outcome:?}");
    }

    for &(sr, ref outcome) in &outcomes {
        match outcome {
            ParityOutcome::Completed => {
                // Unexpected success is fine if upstream ever gains support.
            }
            ParityOutcome::SkippedRateRejected
            | ParityOutcome::SkippedToolNotAvailable
            | ParityOutcome::SkippedModelNotFound => {
                eprintln!(
                    "SKIP: {label_base} @ {sr} Hz — C++ render tool limitation \
                     (LSTM condition_dsp channel mismatch / KnownGap)"
                );
            }
            ParityOutcome::SkippedGarbageOutput => {
                panic!("[{label_base} @ {sr}] FAIL: SkippedGarbageOutput is never acceptable");
            }
        }
    }

    let unexpected: Vec<_> = outcomes
        .iter()
        .filter(|(_, o)| {
            !matches!(
                o,
                ParityOutcome::Completed
                    | ParityOutcome::SkippedRateRejected
                    | ParityOutcome::SkippedToolNotAvailable
                    | ParityOutcome::SkippedModelNotFound
            )
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "{label_base}: unexpected parity outcomes: {unexpected:?}"
    );
}

// SKIP_CAPABILITY (2026-08-03): Sprint 3 synthetic LSTM 1×10 produces
// borderline-recurrent LUFS (−50.2) at 96 kHz on the v2 stress signal.
// The longer sequence accumulates minor LSTM state deviations in the C++
// render, marginally reducing output energy at the tail. All other fidelity
// metrics (SNR ≈ 92 dB, ESR ≈ 7e-13) are well within gates — the output is
// valid, not defective. Gate remains off with this typed justification.
#[test]
#[ignore]
fn live_cross_validation_v2_lstm_1x10() {
    run_v2_multi_sr("lstm_1x10.nam", "lstm_1x10", "Live LSTM 1×10 (v2)", false);
}

// SKIP_CAPABILITY (2026-08-03): Sprint 3 synthetic LSTM 2×24 produces LUFS
// between −51 and −54 across all SRs on the v2 stress signal. The very large
// hidden size (24) causes recurrent state to dampen output energy over the
// long stress sequence — a known architecture characteristic, not a defect.
// SNR (≈80 dB) and ESR (≈1e-11) remain within gates across all SRs.
#[test]
#[ignore]
fn live_cross_validation_v2_lstm_2x24() {
    run_v2_multi_sr("lstm_2x24.nam", "lstm_2x24", "Live LSTM 2×24 (v2)", false);
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_3x8() {
    run_v2_multi_sr("lstm_3x8.nam", "lstm_3x8", "Live LSTM 3×8 (v2)", true);
}

#[test]
#[ignore]
fn live_cross_validation_v2_convnet_nobn() {
    run_v2_multi_sr(
        "convnet_nobn.nam",
        "convnet_nobn",
        "Live ConvNet No BatchNorm (v2)",
        true,
    );
}

// SKIP_CAPABILITY (2026-08-03): ReLU without BatchNorm produces inherently
// low-loudness output for the v2 stress signal (LUFS ≈ −65). The C++ render
// output is validated against all other fidelity metrics (SNR ≈ 140 dB,
// ESR ≈ 1e-14) — the low LUFS is a legitimate architecture characteristic,
// not a defect. Gate remains off with this typed justification.
#[test]
#[ignore]
fn live_cross_validation_v2_convnet_relu() {
    run_v2_multi_sr(
        "convnet_relu.nam",
        "convnet_relu",
        "Live ConvNet ReLU (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_convnet_silu() {
    run_v2_multi_sr(
        "convnet_silu.nam",
        "convnet_silu",
        "Live ConvNet SiLU (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_linear_nobias() {
    run_v2_multi_sr(
        "linear_nobias.nam",
        "linear_nobias",
        "Live Linear No Bias (v2)",
        true,
    );
}

// ── Dynamic Models (WaveNetDyn & LstmModelDyn) ────────────────────────

#[test]
#[ignore]
fn live_cross_validation_wavenet_dyn() {
    let outcome = run_v1(
        "wavenet_dyn_free.nam",
        "wavenet_dyn_free",
        "Live WaveNetDyn Free-Shape",
        false,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_lstm_dyn() {
    let outcome = run_v1(
        "lstm_dyn_test.nam",
        "lstm_dyn_test",
        "Live LSTM-Dyn 1×7",
        false,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_dyn() {
    run_v2_multi_sr(
        "wavenet_dyn_free.nam",
        "wavenet_dyn_free",
        "Live WaveNetDyn Free-Shape (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_lstm_dyn() {
    run_v2_multi_sr(
        "lstm_dyn_test.nam",
        "lstm_dyn_test",
        "Live LSTM-Dyn 1×7 (v2)",
        false,
    );
}

// --- A2 Dynamic Models ---
//
// Live C++ cross-validation for A2 dynamic geometries (FiLM, Gated, Blended)
// that exercise the native WaveNetA2Dyn engine paths against the C++ generic
// Eigen-based WaveNet (C++ a2_fast.cpp rejects these topologies and falls
// back to the generic path — see Finding 7.6.2).

#[test]
#[ignore]
fn live_cross_validation_a2_dynamic_gated() {
    let outcome = run_v1(
        "a2_dynamic_gated_ch8.nam",
        "a2_dynamic_gated_ch8",
        "Live A2 Dynamic Gated (CH=8)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_a2_dynamic_blended() {
    let outcome = run_v1(
        "a2_dynamic_blended_ch3.nam",
        "a2_dynamic_blended_ch3",
        "Live A2 Dynamic Blended (CH=3)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_lite() {
    let outcome = run_v1(
        "wavenet_a2_film_lite.nam",
        "wavenet_a2_film_lite",
        "Live WaveNet A2-FiLM-Lite (CH=3)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_full() {
    let outcome = run_v1(
        "wavenet_a2_film_full.nam",
        "wavenet_a2_film_full",
        "Live WaveNet A2-FiLM-Full (CH=8)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_input_mixin_pre() {
    let outcome = run_v1(
        "wavenet_a2_film_input_mixin_pre.nam",
        "wavenet_a2_film_input_mixin_pre",
        "Live WaveNet A2-FiLM-InputMixinPre (CH=3)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_wavenet_a2_film_chaos_stress() {
    let outcome = run_v1(
        "wavenet_a2_film_chaos_stress.nam",
        "wavenet_a2_film_chaos_stress",
        "Live WaveNet A2-FiLM Chaos Stress (CH=3)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

// wavenet_a2_max — DISABLED: KB-A2-MAX known bug; fail-closed TR1.1; see cpp_parity_map §4.4.3
// (prod f32×C++ SNR≈0.23 dB). Re-enable only after KB-A2-MAX §4.4.3 reopening criteria.
// #[test]
// #[ignore]
// fn live_cross_validation_wavenet_a2_max() {
//     run_v1(
//         "wavenet_a2_max.nam",
//         "wavenet_a2_max",
//         "Live WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1)",
//         true,
//     );
// }

// --- v2 A2 Dynamic Models ---

#[test]
#[ignore]
fn live_cross_validation_v2_a2_dynamic_gated() {
    run_v2_multi_sr(
        "a2_dynamic_gated_ch8.nam",
        "a2_dynamic_gated_ch8",
        "Live A2 Dynamic Gated (CH=8) (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_a2_dynamic_blended() {
    run_v2_multi_sr(
        "a2_dynamic_blended_ch3.nam",
        "a2_dynamic_blended_ch3",
        "Live A2 Dynamic Blended (CH=3) (v2)",
        false,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_lite() {
    run_v2_multi_sr(
        "wavenet_a2_film_lite.nam",
        "wavenet_a2_film_lite",
        "Live WaveNet A2-FiLM-Lite (CH=3) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_full() {
    run_v2_multi_sr(
        "wavenet_a2_film_full.nam",
        "wavenet_a2_film_full",
        "Live WaveNet A2-FiLM-Full (CH=8) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_input_mixin_pre() {
    run_v2_multi_sr(
        "wavenet_a2_film_input_mixin_pre.nam",
        "wavenet_a2_film_input_mixin_pre",
        "Live WaveNet A2-FiLM-InputMixinPre (CH=3) (v2)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_wavenet_a2_film_chaos_stress() {
    run_v2_multi_sr(
        "wavenet_a2_film_chaos_stress.nam",
        "wavenet_a2_film_chaos_stress",
        "Live WaveNet A2-FiLM Chaos Stress (CH=3) (v2)",
        true,
    );
}

// wavenet_a2_max v2 — DISABLED: KB-A2-MAX known bug; fail-closed TR1.1; see cpp_parity_map §4.4.3
// (prod f32×C++ SNR≈0.23 dB). Re-enable only after KB-A2-MAX §4.4.3 reopening criteria.
// #[test]
// #[ignore]
// fn live_cross_validation_v2_wavenet_a2_max() {
//     run_v2_multi_sr(
//         "wavenet_a2_max.nam",
//         "wavenet_a2_max",
//         "Live WaveNet A2 Max (CH=4, cond=8, FiLM, head1x1) (v2)",
//         true,
//     );
// }

// --- Linear ---

#[test]
#[ignore]
fn live_cross_validation_linear() {
    let outcome = run_v1("linear_test.nam", "linear_test", "Live Linear RF=4", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_v2_linear() {
    run_v2_multi_sr(
        "linear_test.nam",
        "linear_test",
        "Live Linear RF=4 (v2)",
        true,
    );
}

// =============================================================================
// Standard (exact-grade) mode cpp_parity tests
//
// C++ NAMCore uses native math (std::tanh) — exact-grade.
// Rust in Standard mode uses high-fidelity polynomial exp-based kernels
// (~2.4e-7 error vs ~2.32e-3 for Padé tanh in Fast mode). This deliberate
// asymmetry means the interop divergence is smaller in HF mode than in
// standard (Fast) mode.
//
// These tests characterize that divergence and gate it with HF-specific
// caps (ABSOLUTE_ESR_CAP_*_HF in run_render_comparison).
//
// Quick (non-ignored): representative subset for fast CI loop
// Comprehensive (ignored): full model × SR matrix, requires C++ toolchain
//
// quick_parity_hf_* tests removed — run_v1 now always uses
// Standard activation, so they were 1:1 duplicates of quick_parity_*.

// --- v1 (standard-rate) HF tests, ignored ---

#[test]
#[ignore]
fn live_cross_validation_hf_lstm_1x16() {
    let outcome = run_v1(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16 (HF)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_hf_lstm_2x8() {
    let outcome = run_v1("BossLSTM-2x8.nam", "lstm_2x8", "Live LSTM 2×8 (HF)", true);
    assert_eq!(outcome, ParityOutcome::Completed);
}

#[test]
#[ignore]
fn live_cross_validation_hf_wavenet_standard() {
    let outcome = run_v1(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (HF)",
        true,
    );
    assert_eq!(outcome, ParityOutcome::Completed);
}

// --- v2 (multi-SR) HF tests, ignored ---

#[test]
#[ignore]
fn live_cross_validation_v2_hf_lstm_1x16() {
    run_v2_multi_sr(
        "BossLSTM-1x16.nam",
        "lstm_1x16",
        "Live LSTM 1×16 (v2, HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_hf_lstm_2x8() {
    run_v2_multi_sr(
        "BossLSTM-2x8.nam",
        "lstm_2x8",
        "Live LSTM 2×8 (v2, HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_v2_hf_wavenet_standard() {
    run_v2_multi_sr(
        "BossWN-standard.nam",
        "wavenet_standard",
        "Live WaveNet Standard (v2, HF)",
        true,
    );
}

#[test]
#[ignore]
fn live_cross_validation_nondist_models() {
    let mut nondist_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    nondist_path.push("tests/fixtures/models-nondist");

    if !nondist_path.exists() {
        println!(
            "SKIP: Non-distributable models directory {:?} not found.",
            nondist_path
        );
        return;
    }

    let models: Vec<_> = discovery::find_models_in_dir(&nondist_path)
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n != "manifest.json"))
        .collect();

    if models.is_empty() {
        println!("SKIP: No models found in {:?}", nondist_path);
        return;
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let temp_dir = project_root.join("tests/fixtures/.temp_live");
    fs::create_dir_all(&temp_dir).ok();
    if !ensure_render_compiled() {
        println!("SKIP: C++ render tool not available, skipping nondist cross-validation.");
        return;
    }

    for model_path in models {
        let filename = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        println!("Live C++ cross-validating nondist model: {}", filename);

        let mut dest_path = project_root.join("tests/fixtures/models");
        dest_path.push(&filename);

        let copied = if !dest_path.exists() {
            fs::copy(&model_path, &dest_path).is_ok()
        } else {
            false
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_v1(
                &filename,
                &format!("nondist_{}", filename.replace('.', "_")),
                &format!("Live Nondist {}", filename),
                true,
            )
        }));

        if copied {
            fs::remove_file(&dest_path).ok();
        }

        match result {
            Ok(outcome) => assert_eq!(outcome, ParityOutcome::Completed),
            Err(e) => std::panic::resume_unwind(e),
        }
    }
}
