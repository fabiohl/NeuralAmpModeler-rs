// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Cross-ISA Determinism Matrix — Task 2.7 (P-8).
//
//  Runs golden vectors through each supported ISA path (AVX2 as reference,
//  AVX-512, and AVX-512 VNNI+BF16) and asserts end-to-end model output parity.
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
//  # Running
//
//  These tests manipulate a process-wide ISA override. They must run serially:
//
//  ```sh
//  cargo test --release --test isa_parity -- --test-threads=1 --nocapture
//  ```
//
//  Tests requiring AVX-512 or VNNI+BF16 hardware are `#[ignore]` and only
//  execute in environments that support those ISA levels.
//
//  # ISA Coverage Map
//
//  | ISA Pair                   | CI Coverage | Notes                        |
//  | -------------------------- | ----------- | ---------------------------- |
//  | AVX2 (ref) → AVX2          | ✓ always    | Self-consistency, ESR = 0    |
//  | AVX2 (ref) → AVX-512       | ✓ if AVX-512| Cross-ISA parity             |
//  | AVX2 (ref) → VNNI+BF16     | ✓ if VNNI   | Includes BF16 quantisation   |

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::Ordering;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::math::activations::ActivationPrecision;
use neural_amp_modeler_rs::math::common::{InstructionSet, TEST_ISA_OVERRIDE, encode_isa_override};
use neural_amp_modeler_rs::models::NamModel;

use super::common;
use common::*;

/// Serialises access to the process-wide `TEST_ISA_OVERRIDE`.
static ISA_LOCK: Mutex<()> = Mutex::new(());

/// Clears the ISA override.
fn clear_override() {
    TEST_ISA_OVERRIDE.store(u8::MAX, Ordering::SeqCst);
}

/// Sets the ISA override for the current test and returns a guard that
/// clears it on drop. The `ISA_LOCK` must already be held.
struct IsaGuard;

impl IsaGuard {
    fn set(isa: InstructionSet) -> Self {
        TEST_ISA_OVERRIDE.store(encode_isa_override(isa), Ordering::SeqCst);
        IsaGuard
    }
}

impl Drop for IsaGuard {
    fn drop(&mut self) {
        clear_override();
    }
}

/// Signals that the host CPU does not support a given ISA path.
macro_rules! skip_if_unsupported {
    ($isa:expr, $test_name:expr) => {
        match $isa {
            InstructionSet::Avx2 => { /* always supported (x86-64-v3) */ }
            InstructionSet::Avx512 => {
                if !is_x86_feature_detected!("avx512f") {
                    eprintln!("SKIP {}: AVX-512 not supported on this CPU", $test_name);
                    return;
                }
            }
            InstructionSet::Avx512VnniBf16 => {
                if !is_x86_feature_detected!("avx512bf16")
                    || !is_x86_feature_detected!("avx512vnni")
                {
                    eprintln!("SKIP {}: VNNI+BF16 not supported on this CPU", $test_name);
                    return;
                }
            }
        }
    };
}

/// Loads a model and runs golden-vector inference under a specific ISA.
///
/// Returns the model output buffer and the expected (C++ reference) output.
fn run_under_isa(
    model_filename: &str,
    golden_name: &str,
    sr: u32,
    isa: InstructionSet,
) -> (Vec<f32>, Vec<f32>) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);
    let golden_filename = format!("{golden_name}_v2_{sr}.bin");
    let golden_path = fixtures_dir.join(&golden_filename);

    assert!(nam_path.exists(), "Model file not found: {nam_path:?}");
    assert!(
        golden_path.exists(),
        "Golden vector not found: {golden_path:?}. Run './tests/fixtures/golden_gen_build.sh'."
    );

    let (input, expected) = read_golden_bin(&golden_path)
        .unwrap_or_else(|| panic!("Failed to read golden {golden_filename}"));

    let json_data = std::fs::read_to_string(&nam_path).expect("Failed to read model JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    // CRITICAL: set override BEFORE building model — the builder reads
    // SimdMathConfig::get() to decide BF16 vs non-BF16 weight layout.
    // Also explicitly pin Fast activation precision — this measures ISA
    // parity of the Padé/minimax kernels specifically (the `_hf` sibling
    // function below measures the Standard/exact-grade kernels instead).
    // Standard-mode tests may have left the global atomic dirty (Tarefa β1.3).
    let _guard = IsaGuard::set(isa);
    let _prec = PrecisionGuard::new(ActivationPrecision::Fast);

    let mut model = build_model(&model_data).unwrap_or_else(|e| panic!("Build failed: {e}"));
    model.prewarm(V2_PREWARM_SAMPLES);

    let num_samples = input.len();
    let mut output = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

    (output, expected)
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
) -> (Vec<f32>, Vec<f32>) {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);
    let golden_filename = format!("{golden_name}_v2_{sr}.bin");
    let golden_path = fixtures_dir.join(&golden_filename);

    assert!(nam_path.exists(), "Model file not found: {nam_path:?}");
    assert!(
        golden_path.exists(),
        "Golden vector not found: {golden_path:?}. Run './tests/fixtures/golden_gen_build.sh'."
    );

    let (input, expected) = read_golden_bin(&golden_path)
        .unwrap_or_else(|| panic!("Failed to read golden {golden_filename}"));

    let json_data = std::fs::read_to_string(&nam_path).expect("Failed to read model JSON");
    let model_data = parse_nam_json(&json_data).expect("Failed in JSON parser");

    let _precision = PrecisionGuard::new(ActivationPrecision::Standard);
    let _guard = IsaGuard::set(isa);

    let mut model = build_model(&model_data).unwrap_or_else(|e| panic!("Build failed: {e}"));
    model.prewarm(V2_PREWARM_SAMPLES);

    let num_samples = input.len();
    let mut output = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

    (output, expected)
}

/// Compares two output buffers produced under different ISAs and asserts
/// ESR parity within the given budget.
///
/// Reports the SIMD-vs-reference ESR floor for diagnostics (P-8 / P-4).
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

    let ref_name = match ref_isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };
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

    assert!(
        esr < max_esr,
        "[{label}] ISA parity FAIL: {ref_name} → {test_name} \
         ESR={esr:.2e} ≥ budget={max_esr:.1e}"
    );
}

/// Convenience: runs cross-ISA comparison for one model at 48 kHz
/// in `ActivationPrecision::Standard` (exact-grade) mode.
fn check_isa_parity_for_model_hf(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    test_isa: InstructionSet,
    max_esr: f64,
) {
    let sr = 48000;

    let (ref_output, _expected) =
        run_under_isa_hf(model_filename, golden_name, sr, InstructionSet::Avx2);

    let (test_output, _expected2) = run_under_isa_hf(model_filename, golden_name, sr, test_isa);

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
        run_under_isa(model_filename, golden_name, sr, InstructionSet::Avx2);

    // Run under test ISA
    let (test_output, _expected2) = run_under_isa(model_filename, golden_name, sr, test_isa);

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

    let (output1, _) = run_under_isa(model_filename, golden_name, sr, isa);
    let (output2, _) = run_under_isa(model_filename, golden_name, sr, isa);

    let mse = compute_mse(&output1, &output2);
    let isa_name = match isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
    };
    println!("[ISA Matrix] {label} | {isa_name:>10} self-consistency | MSE={mse:.2e}");

    assert!(
        mse == 0.0,
        "[{label}] {isa_name} self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
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
const WN_ESR_BUDGET: f64 = 1e-3;

/// Default cross-ISA ESR budget for LSTM models (recurrent accumulation
/// amplifies minor ISA differences — more generous budget).
const LSTM_ESR_BUDGET: f64 = 1e-2;

/// Default cross-ISA ESR budget for A2 models.
const A2_ESR_BUDGET: f64 = 1e-3;

// ══════════════════════════════════════════════════════════════════════
// AVX2 self-consistency — requires golden v2 vectors (generated via
// golden_gen_build.sh). Ignored by default; run with `--ignored` in
// Phase 2 / tests-long.sh when goldens are present.
// ══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
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

#[test]
#[ignore = "Requires AVX-512 VNNI+BF16 hardware"]
fn isa_parity_wavenet_standard_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Std/VNNI-BF16");
    // VNNI+BF16 introduces bf16 quantisation on top of AVX-512 → larger budget
    check_isa_parity_for_model(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        InstructionSet::Avx512VnniBf16,
        WN_ESR_BUDGET * 10.0,
    );
}

#[test]
#[ignore = "Requires AVX-512 VNNI+BF16 hardware"]
fn isa_parity_wavenet_nano_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Nano/VNNI-BF16");
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
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
fn isa_hf_self_consistency_wavenet_standard_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    let sr = 48000;
    let (output1, _) = run_under_isa_hf(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        sr,
        InstructionSet::Avx2,
    );
    let (output2, _) = run_under_isa_hf(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        sr,
        InstructionSet::Avx2,
    );
    let mse = compute_mse(&output1, &output2);
    println!("[ISA HF Matrix] WN-Std AVX2 self-consistency (HF) | MSE={mse:.2e}");
    assert!(
        mse == 0.0,
        "WN-Std AVX2 HF self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
    );
}

#[test]
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
fn isa_hf_self_consistency_lstm_1x16_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    let sr = 48000;
    let (output1, _) = run_under_isa_hf(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        sr,
        InstructionSet::Avx2,
    );
    let (output2, _) = run_under_isa_hf(
        "BossLSTM-1x16.nam",
        "golden_lstm_1x16",
        sr,
        InstructionSet::Avx2,
    );
    let mse = compute_mse(&output1, &output2);
    println!("[ISA HF Matrix] LSTM-1x16 AVX2 self-consistency (HF) | MSE={mse:.2e}");
    assert!(
        mse == 0.0,
        "LSTM-1x16 AVX2 HF self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
    );
}

#[test]
#[ignore = "requires golden v2 binaries — generated via golden_gen_build.sh"]
fn isa_hf_self_consistency_lstm_2x8_avx2() {
    let _lock = ISA_LOCK.lock().unwrap();
    let sr = 48000;
    let (output1, _) = run_under_isa_hf(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        sr,
        InstructionSet::Avx2,
    );
    let (output2, _) = run_under_isa_hf(
        "BossLSTM-2x8.nam",
        "golden_lstm_2x8",
        sr,
        InstructionSet::Avx2,
    );
    let mse = compute_mse(&output1, &output2);
    println!("[ISA HF Matrix] LSTM-2x8 AVX2 self-consistency (HF) | MSE={mse:.2e}");
    assert!(
        mse == 0.0,
        "LSTM-2x8 AVX2 HF self-consistency FAIL: MSE={mse:.6e} (expected 0.0)"
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
const LSTM_HF_ESR_BUDGET: f64 = 1e-2;

/// HF cross-ISA ESR budget for WaveNet models.
const WN_HF_ESR_BUDGET: f64 = 1e-3;

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
