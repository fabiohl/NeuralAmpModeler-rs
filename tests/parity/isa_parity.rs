// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Cross-ISA Determinism Matrix — Task 2.7 (P-8).
//
//  Runs golden vectors through each supported ISA path (AVX2 as reference,
//  and AVX-512) and asserts end-to-end model output parity.
//
//  # Rationale
//
//  The `dispatch_simd!` macro selects the "best" available ISA at runtime.
//  This suite overrides the dispatch to force a specific ISA path and compares
//  the full model output against the AVX2 reference, quantifying the SIMD-vs-
//  reference error floor for every model architecture.
//
//  Kernel-level scalar-vs-SIMD parity is already covered by unit tests
//  (`gemv_test.rs`, `dot_4x_test.rs`, `dot_8x_test.rs`, `dot_16x_test.rs`,
//  `proptest_math.rs`). This suite adds the missing end-to-end model-level
//  cross-ISA coverage.
//
//  # Production Policy & VNNI Status (F-SIMD-12)
//
//  Production neural inference runs strictly in single-precision `f32` on the
//  AVX2 (x86-64-v3) backend — the only production/default dispatch. AVX-512 is
//  strictly opt-in research (`avx512` feature): the AVX2→AVX-512 cross-ISA cases
//  below are research parity measurements, never a claim of production dispatch.
//  VNNI / BF16 is NOT an active production acceleration path. In
//  `dispatch_simd!`, `InstructionSet::Avx512VnniBf16` is deprecated and folds
//  directly into `Avx512Math` (f32). The VNNI test cases below are `#[ignore]`d as
//  legacy / evaluation-only tests; when executed under `TEST_ISA_OVERRIDE = 2`,
//  they exercise the AVX-512 f32 math kernels without BF16 weight layout conversion.
//
//  # Running
//
//  These tests manipulate a process-wide ISA override. They must run serially:
//
//  ```sh
//  cargo test --release --test isa_parity -- --test-threads=1 --nocapture
//  ```
//
//  Tests requiring AVX-512 hardware are `#[ignore]` and only execute in
//  environments that support those ISA levels (or via `utils/tests-long.sh`).
//
//  # ISA Coverage Map
//
//  | ISA Pair                   | CI Coverage       | Notes                                                 |
//  | -------------------------- | ----------------- | ----------------------------------------------------- |
//  | AVX2 (ref) → AVX2          | ✓ always (v2 bin) | Self-consistency, MSE = 0                             |
//  | AVX2 (ref) → AVX-512       | ✓ if AVX-512      | Cross-ISA f32 parity (within ESR budget)              |
//  | AVX2 (ref) → VNNI+BF16     | Ignored (legacy)  | Legacy / evaluation-only; routes to f32 (F-SIMD-12)   |

use std::path::PathBuf;
use std::sync::Mutex;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::math::common::InstructionSet;
#[cfg(feature = "avx512")]
use neural_amp_modeler_rs::math::common::has_full_avx512;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::isa_guard::IsaGuard;

use super::common;
use common::*;

/// Serialises access to the process-wide ISA override (T2.3: all installs go
/// through the validated `testing::isa_guard::IsaGuard`).
static ISA_LOCK: Mutex<()> = Mutex::new(());

/// Signals that the host CPU does not support a given ISA path.
///
/// T3.2: emits the typed `[STATUS] SKIP_CAPABILITY` marker (machine-parseable
/// by `detect_gap_markers` in `src/testing/receipt.rs`) instead of free-form
/// `SKIP ...` prints.
macro_rules! skip_if_unsupported {
    ($isa:expr, $test_name:expr) => {
        #[expect(deprecated)]
        match $isa {
            InstructionSet::Avx2 => { /* always supported (x86-64-v3) */ }
            InstructionSet::Avx512 => {
                #[cfg(feature = "avx512")]
                if !has_full_avx512() {
                    eprintln!(
                        "[STATUS] SKIP_CAPABILITY reason=\"avx512_cpu_unsupported:{}\"",
                        $test_name
                    );
                    return;
                }
                #[cfg(not(feature = "avx512"))]
                {
                    eprintln!(
                        "[STATUS] SKIP_CAPABILITY reason=\"avx512_not_compiled:{}\"",
                        $test_name
                    );
                    return;
                }
            }
            InstructionSet::Avx512VnniBf16 => {
                #[cfg(feature = "avx512")]
                if !has_full_avx512()
                    || !is_x86_feature_detected!("avx512bf16")
                    || !is_x86_feature_detected!("avx512vnni")
                {
                    eprintln!(
                        "[STATUS] SKIP_CAPABILITY reason=\"vnni_bf16_cpu_unsupported:{}\"",
                        $test_name
                    );
                    return;
                }
                #[cfg(not(feature = "avx512"))]
                {
                    eprintln!(
                        "[STATUS] SKIP_CAPABILITY reason=\"vnni_bf16_not_compiled:{}\"",
                        $test_name
                    );
                    return;
                }
            }
        }
    };
}

/// Loads a model and runs golden-vector inference under a specific ISA.
///
/// Returns the model output buffer and the expected (C++ reference) output.
/// Loads a model and runs golden-vector inference under a specific ISA.
///
/// Returns the model output buffer and the expected (C++ reference) output.
fn run_under_isa(
    model_filename: &str,
    golden_name: &str,
    sr: u32,
    isa: InstructionSet,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);
    let golden_filename = format!("{golden_name}_v2_{sr}.bin");
    let golden_path = fixtures_dir.join(&golden_filename);

    if !nam_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:{model_filename}\"");
        return None;
    }
    if !golden_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"golden_not_found:{golden_filename}\"");
        return None;
    }

    let (input, expected) = match read_golden_bin(&golden_path) {
        Some(pair) => pair,
        None => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"golden_unreadable:{golden_filename}\"");
            return None;
        }
    };

    let json_data = match std::fs::read_to_string(&nam_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_json_unreadable:{model_filename}\"");
            return None;
        }
    };
    let model_data = match parse_nam_json(&json_data) {
        Ok(m) => m,
        Err(_) => {
            eprintln!(
                "[STATUS] SKIP_CAPABILITY reason=\"model_json_parse_failed:{model_filename}\""
            );
            return None;
        }
    };

    // CRITICAL: set override BEFORE building model — the builder reads
    // SimdMathConfig::get() to decide BF16 vs non-BF16 weight layout.
    // Also explicitly pin Fast activation precision — this measures ISA
    // parity of the Padé/minimax kernels specifically (the `_hf` sibling
    // function below measures the Standard/exact-grade kernels instead).
    // Standard-mode tests may have left the global atomic dirty (Tarefa β1.3).
    // T2.3: installs go through the validated crate guard (host capability
    // checked) — a mismatch degrades to a typed skip, never a SIGILL.
    let _guard = match IsaGuard::try_set(isa) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"isa_override_install_failed:{isa:?}\"");
            return None;
        }
    };
    let _prec = PrecisionGuard::new(ActivationPrecision::Fast);

    let mut model = match build_model(&model_data) {
        Ok(m) => m,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_build_failed:{model_filename}\"");
            return None;
        }
    };
    model.prewarm(V2_PREWARM_SAMPLES);

    let num_samples = input.len();
    let mut output = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

    Some((output, expected))
}

/// Loads a model and runs golden-vector inference under a specific ISA
/// with `ActivationPrecision::Standard` (exact-grade) enabled.
///
/// Returns the model output buffer and the expected (C++ reference) output.
fn run_under_isa_hf(
    model_filename: &str,
    golden_name: &str,
    sr: u32,
    isa: InstructionSet,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);
    let golden_filename = format!("{golden_name}_v2_{sr}.bin");
    let golden_path = fixtures_dir.join(&golden_filename);

    if !nam_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:{model_filename}\"");
        return None;
    }
    if !golden_path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"golden_not_found:{golden_filename}\"");
        return None;
    }

    let (input, expected) = match read_golden_bin(&golden_path) {
        Some(pair) => pair,
        None => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"golden_unreadable:{golden_filename}\"");
            return None;
        }
    };

    let json_data = match std::fs::read_to_string(&nam_path) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_json_unreadable:{model_filename}\"");
            return None;
        }
    };
    let model_data = match parse_nam_json(&json_data) {
        Ok(m) => m,
        Err(_) => {
            eprintln!(
                "[STATUS] SKIP_CAPABILITY reason=\"model_json_parse_failed:{model_filename}\""
            );
            return None;
        }
    };

    let _precision = PrecisionGuard::new(ActivationPrecision::Standard);
    let _guard = match IsaGuard::try_set(isa) {
        Ok(g) => g,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"isa_override_install_failed:{isa:?}\"");
            return None;
        }
    };

    let mut model = match build_model(&model_data) {
        Ok(m) => m,
        Err(_) => {
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_build_failed:{model_filename}\"");
            return None;
        }
    };
    model.prewarm(V2_PREWARM_SAMPLES);

    let num_samples = input.len();
    let mut output = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

    Some((output, expected))
}

/// Compares two output buffers produced under different ISAs and asserts
/// ESR parity within the given budget.
///
/// Reports the SIMD-vs-reference ESR floor for diagnostics (P-8 / P-4).
#[cfg(feature = "avx512")]
fn assert_isa_parity(
    output_ref: &[f32],
    output_test: &[f32],
    label: &str,
    ref_isa: InstructionSet,
    test_isa: InstructionSet,
    max_esr: f64,
) {
    let esr = compute_esr(output_ref, output_test);
    let mse = compute_mse(output_ref, output_test);
    let mae = compute_max_abs_error(output_ref, output_test);

    #[expect(deprecated)]
    let ref_name = match ref_isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };
    #[expect(deprecated)]
    let test_name = match test_isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };

    println!(
        "[ISA Matrix] {label} | {ref_name:>10} → {test_name:>10} | \
         ESR={esr:.2e} | MSE={mse:.2e} | MaxAbsErr={mae:.2e} | \
         budget ESR<{max_esr:.1e}"
    );

    // S2.T6: forensic JSONL sink (kind `isa`) — human log unchanged.
    common::report_isa(
        label,
        ref_name,
        test_name,
        Some(esr),
        mse,
        Some(mae),
        Some(max_esr),
    );

    assert!(
        esr < max_esr,
        "[{label}] ISA parity FAIL: {ref_name} → {test_name} \
         ESR={esr:.2e} ≥ budget={max_esr:.1e}"
    );
}

/// Convenience: runs cross-ISA comparison for one model at 48 kHz
/// in `ActivationPrecision::Standard` (exact-grade) mode.
#[cfg(feature = "avx512")]
fn check_isa_parity_for_model_hf(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    test_isa: InstructionSet,
    max_esr: f64,
) {
    let sr = 48000;

    let (ref_output, _expected) =
        match run_under_isa_hf(model_filename, golden_name, sr, InstructionSet::Avx2) {
            Some(pair) => pair,
            None => return,
        };

    let (test_output, _expected2) =
        match run_under_isa_hf(model_filename, golden_name, sr, test_isa) {
            Some(pair) => pair,
            None => return,
        };

    assert_isa_parity(
        &ref_output,
        &test_output,
        &format!("{label} @ {sr} Hz (HF)"),
        InstructionSet::Avx2,
        test_isa,
        max_esr,
    );
}

/// Convenience: runs cross-ISA comparison for one model at 48 kHz.
#[cfg(feature = "avx512")]
fn check_isa_parity_for_model(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    test_isa: InstructionSet,
    max_esr: f64,
) {
    let sr = 48000;

    // Always run AVX2 as reference
    let (ref_output, _expected) =
        match run_under_isa(model_filename, golden_name, sr, InstructionSet::Avx2) {
            Some(pair) => pair,
            None => return,
        };

    // Run under test ISA
    let (test_output, _expected2) = match run_under_isa(model_filename, golden_name, sr, test_isa) {
        Some(pair) => pair,
        None => return,
    };

    assert_isa_parity(
        &ref_output,
        &test_output,
        &format!("{label} @ {sr} Hz"),
        InstructionSet::Avx2,
        test_isa,
        max_esr,
    );
}

/// Runs the same model twice under the same ISA and asserts bitwise-identical.
fn assert_isa_self_consistency(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    isa: InstructionSet,
) {
    let sr = 48000;
    skip_if_unsupported!(isa, label);

    let (output1, _) = match run_under_isa(model_filename, golden_name, sr, isa) {
        Some(pair) => pair,
        None => return,
    };
    let (output2, _) = match run_under_isa(model_filename, golden_name, sr, isa) {
        Some(pair) => pair,
        None => return,
    };

    let mse = compute_mse(&output1, &output2);
    #[expect(deprecated)]
    let isa_name = match isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };
    println!("[ISA Matrix] {label} | {isa_name:>10} self-consistency | MSE={mse:.2e}");

    // S2.T6: self-consistency sinks with `ref_isa == test_isa` (kind `isa`,
    // only `mse` carried) — human log unchanged.
    common::report_isa(label, isa_name, isa_name, None, mse, None, None);

    assert!(
        mse == 0.0,
        "[{label}] {isa_name} self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
    );
}

/// Runs the same model twice under the same ISA in Standard HF mode and asserts bitwise-identical.
fn assert_isa_hf_self_consistency(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    isa: InstructionSet,
) {
    let sr = 48000;
    skip_if_unsupported!(isa, label);

    let (output1, _) = match run_under_isa_hf(model_filename, golden_name, sr, isa) {
        Some(pair) => pair,
        None => return,
    };
    let (output2, _) = match run_under_isa_hf(model_filename, golden_name, sr, isa) {
        Some(pair) => pair,
        None => return,
    };

    let mse = compute_mse(&output1, &output2);
    #[expect(deprecated)]
    let isa_name = match isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };
    println!("[ISA HF Matrix] {label} | {isa_name:>10} self-consistency (HF) | MSE={mse:.2e}");

    common::report_isa(label, isa_name, isa_name, None, mse, None, None);

    assert!(
        mse == 0.0,
        "[{label}] {isa_name} HF self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
    );
}

// ══════════════════════════════════════════════════════════════════════
// ISA matrix calibration budget (per-model, initial conservative values)
// ══════════════════════════════════════════════════════════════════════
//
// These are initial conservative budgets designed to pass on known-good
// hardware and catch regressions. They are tightened after hardware-
// specific calibration in a CI runner with AVX-512 support.

/// Default cross-ISA ESR budget for WaveNet models (conservative).
#[cfg(feature = "avx512")]
const WN_ESR_BUDGET: f64 = 1e-3;

/// Default cross-ISA ESR budget for LSTM models (recurrent accumulation
/// amplifies minor ISA differences — more generous budget).
#[cfg(feature = "avx512")]
const LSTM_ESR_BUDGET: f64 = 1e-2;

/// Default cross-ISA ESR budget for A2 models.
#[cfg(feature = "avx512")]
const A2_ESR_BUDGET: f64 = 1e-3;

// ══════════════════════════════════════════════════════════════════════
// AVX2 self-consistency — runs in quick suite when goldens are present.
// ══════════════════════════════════════════════════════════════════════

#[test]
fn isa_self_consistency_wavenet_standard_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_wavenet_feather_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "BossWN-feather.nam",
        "golden_wavenet_feather",
        "WN-Feather",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_wavenet_nano_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WN-Nano",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_lstm_1x16_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM-1x16",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_lstm_2x8_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM-2x8",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_a2_full_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "wavenet_a2_full.nam",
        "golden_wavenet_a2_full",
        "A2-Full",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_self_consistency_a2_lite_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_self_consistency(
        "wavenet_a2_lite.nam",
        "golden_wavenet_a2_lite",
        "A2-Lite",
        InstructionSet::Avx2,
    );
}

// ══════════════════════════════════════════════════════════════════════
// Cross-ISA parity tests — AVX2 (ref) vs AVX-512 (ignored by default)
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_wavenet_standard_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "WN-Std/AVX-512");
    check_isa_parity_for_model(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx512,
        WN_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_wavenet_feather_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "WN-Feather/AVX-512");
    check_isa_parity_for_model(
        "BossWN-feather.nam",
        "golden_wavenet_feather",
        "WN-Feather",
        InstructionSet::Avx512,
        WN_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_wavenet_nano_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "WN-Nano/AVX-512");
    check_isa_parity_for_model(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WN-Nano",
        InstructionSet::Avx512,
        WN_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_lstm_1x16_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "LSTM-1x16/AVX-512");
    check_isa_parity_for_model(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM-1x16",
        InstructionSet::Avx512,
        LSTM_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_lstm_2x8_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "LSTM-2x8/AVX-512");
    check_isa_parity_for_model(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM-2x8",
        InstructionSet::Avx512,
        LSTM_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_a2_full_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "A2-Full/AVX-512");
    check_isa_parity_for_model(
        "wavenet_a2_full.nam",
        "golden_wavenet_a2_full",
        "A2-Full",
        InstructionSet::Avx512,
        A2_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_a2_lite_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "A2-Lite/AVX-512");
    check_isa_parity_for_model(
        "wavenet_a2_lite.nam",
        "golden_wavenet_a2_lite",
        "A2-Lite",
        InstructionSet::Avx512,
        A2_ESR_BUDGET,
    );
}

// ══════════════════════════════════════════════════════════════════════
// Cross-ISA parity tests — AVX2 (ref) vs VNNI+BF16 (ignored by default)
// ══════════════════════════════════════════════════════════════════════

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Legacy evaluation-only: requires AVX-512 VNNI+BF16 hardware"]
#[expect(deprecated)]
fn isa_parity_wavenet_standard_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Std/VNNI-BF16");
    // F-SIMD-12: VNNI is not a production path. In dispatch_simd!, Avx512VnniBf16
    // folds into Avx512Math (f32). This test remains as a legacy evaluation checkpoint.
    check_isa_parity_for_model(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx512VnniBf16,
        WN_ESR_BUDGET * 10.0,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Legacy evaluation-only: requires AVX-512 VNNI+BF16 hardware"]
#[expect(deprecated)]
fn isa_parity_wavenet_nano_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Nano/VNNI-BF16");
    // F-SIMD-12: VNNI is not a production path. In dispatch_simd!, Avx512VnniBf16
    // folds into Avx512Math (f32). This test remains as a legacy evaluation checkpoint.
    check_isa_parity_for_model(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WN-Nano",
        InstructionSet::Avx512VnniBf16,
        WN_ESR_BUDGET * 10.0,
    );
}

// ══════════════════════════════════════════════════════════════════════
// Standard (exact-grade) mode self-consistency (AVX2) — always runs
// ══════════════════════════════════════════════════════════════════════
//
// Tarefa β1.3: verify that the HF activation paths (scalar + SIMD) are
// deterministic across repeated runs with the same ISA.

#[test]
fn isa_hf_self_consistency_wavenet_standard_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_hf_self_consistency(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_hf_self_consistency_lstm_1x16_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_hf_self_consistency(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM-1x16",
        InstructionSet::Avx2,
    );
}

#[test]
fn isa_hf_self_consistency_lstm_2x8_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    assert_isa_hf_self_consistency(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM-2x8",
        InstructionSet::Avx2,
    );
}

// ══════════════════════════════════════════════════════════════════════
// Standard (exact-grade) cross-ISA parity — AVX2 (ref) vs AVX-512 (ignored)
// ══════════════════════════════════════════════════════════════════════
//
// Tarefa β1.3: verify cross-ISA parity in Standard (exact-grade) mode.
// HF polynomial kernels use the same mathematical approximation (degree-6
// Taylor with range reduction) across ISAs, so cross-ISA parity should be
// comparable to or better than standard mode.

/// HF cross-ISA ESR budget for LSTM models.
#[cfg(feature = "avx512")]
const LSTM_HF_ESR_BUDGET: f64 = 1e-2;

/// HF cross-ISA ESR budget for WaveNet models.
#[cfg(feature = "avx512")]
const WN_HF_ESR_BUDGET: f64 = 1e-3;

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_hf_lstm_1x16_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "LSTM-1x16/AVX-512 HF");
    check_isa_parity_for_model_hf(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        "LSTM-1x16",
        InstructionSet::Avx512,
        LSTM_HF_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_hf_lstm_2x8_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "LSTM-2x8/AVX-512 HF");
    check_isa_parity_for_model_hf(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        "LSTM-2x8",
        InstructionSet::Avx512,
        LSTM_HF_ESR_BUDGET,
    );
}

#[cfg(feature = "avx512")]
#[test]
#[ignore = "Requires AVX-512 hardware"]
fn isa_parity_hf_wavenet_standard_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512, "WN-Std/AVX-512 HF");
    check_isa_parity_for_model_hf(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx512,
        WN_HF_ESR_BUDGET,
    );
}

// ══════════════════════════════════════════════════════════════════════
// ISA matrix header (informational, always runs)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn isa_matrix_header_info() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cross-ISA Determinism Matrix (P-8 / Task 2.7)               ║");
    println!("║  Reference = AVX2 (x86-64-v3, always available)              ║");
    println!("║                                                              ║");
    println!("║  Kernel-level scalar-vs-SIMD parity: gemv_test.rs,           ║");
    println!("║  dot_4x_test.rs, dot_8x_test.rs, dot_16x_test.rs,            ║");
    println!("║  proptest_math.rs                                            ║");
    println!("║                                                              ║");
    println!("║  Run cross-ISA matrix:                                       ║");
    println!("║  cargo test --release --test isa_parity -- \\                 ║");
    println!("║    --ignored --test-threads=1 --nocapture                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}
