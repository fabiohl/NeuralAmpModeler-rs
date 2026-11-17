// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Cross-ISA Determinism Matrix.
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
//  # Scope: self-consistency vs. real cross-ISA parity
//
//  The local QA dashboard (`utils/quality-dashboard.sh`) runs only the AVX2-vs-
//  AVX2 self-consistency cases under the phase name `isa_self_consistency` and
//  declares the cross-ISA matrix as an explicit `SKIP_CAPABILITY` gap
//  (`isa_parity_cross_isa` / `cross_isa_matrix_requires_avx512_remote`). The
//  real inter-ISA matrix (AVX2 vs AVX-512) is restricted to the multi-target
//  environment via `utils/remote-simd-gate.sh` (`--features avx512 --include-ignored`)
//  and the long suite — a local `isa_self_consistency: PASS` is internal
//  determinism (MSE=0 tautology), never a claim of cross-ISA parity.
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
#[cfg(nam_avx512)]
use neural_amp_modeler_rs::math::common::has_full_avx512;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::isa_guard::IsaGuard;
#[cfg(nam_avx512)]
use neural_amp_modeler_rs::testing::qa::simd_parity::{
    SimdActivationPrecision, SimdParityCaseSpec, SimdParityRecord, SimdParityStatus,
    WN_CROSS_ISA_ESR_BUDGET, append_simd_parity_record, find_mandatory_case,
};

use super::common;
use common::*;

/// Serialises access to the process-wide ISA override (all installs go
/// through the validated `testing::isa_guard::IsaGuard`).
static ISA_LOCK: Mutex<()> = Mutex::new(());

/// Signals that the host CPU does not support a given ISA path.
///
/// Emits the typed `[STATUS] SKIP_CAPABILITY` marker (machine-parseable
/// by `detect_gap_markers` in `src/testing/receipt.rs`) instead of free-form
/// `SKIP ...` prints.
macro_rules! skip_if_unsupported {
    ($isa:expr, $test_name:expr) => {
        #[expect(deprecated)]
        let unsupported_reason: Option<&'static str> = match $isa {
            InstructionSet::Avx2 => None,
            InstructionSet::Avx512 => {
                #[cfg(nam_avx512)]
                {
                    (!has_full_avx512()).then_some("avx512_cpu_unsupported")
                }
                #[cfg(not(nam_avx512))]
                {
                    Some("avx512_not_compiled")
                }
            }
            InstructionSet::Avx512VnniBf16 => {
                #[cfg(nam_avx512)]
                {
                    (!has_full_avx512()
                        || !is_x86_feature_detected!("avx512bf16")
                        || !is_x86_feature_detected!("avx512vnni"))
                    .then_some("vnni_bf16_cpu_unsupported")
                }
                #[cfg(not(nam_avx512))]
                {
                    Some("vnni_bf16_not_compiled")
                }
            }
            _ => None,
        };
        if let Some(reason) = unsupported_reason {
            eprintln!(
                "[STATUS] SKIP_CAPABILITY reason=\"{}:{}\"",
                reason, $test_name
            );
            return;
        }
    };
}

enum IsaRunError {
    MissingFixture(String),
    Execution(String),
}

impl IsaRunError {
    fn reason(&self) -> &str {
        match self {
            Self::MissingFixture(r) | Self::Execution(r) => r,
        }
    }
}

/// Loads a model and runs golden-vector inference under a specific ISA and activation precision.
///
/// Returns the model output buffer and the expected (C++ reference) output.
fn run_under_isa_with_precision(
    model_filename: &str,
    golden_name: &str,
    sr: u32,
    isa: InstructionSet,
    precision: ActivationPrecision,
) -> Result<(Vec<f32>, Vec<f32>), IsaRunError> {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let nam_path = model_path(model_filename);
    let golden_filename = format!("{golden_name}_v2_{sr}.bin");
    let golden_path = fixtures_dir.join(&golden_filename);

    if !nam_path.exists() {
        return Err(IsaRunError::MissingFixture(format!(
            "model_not_found:{model_filename}"
        )));
    }
    if !golden_path.exists() {
        return Err(IsaRunError::MissingFixture(format!(
            "golden_not_found:{golden_filename}"
        )));
    }

    let (input, expected) = read_golden_bin(&golden_path)
        .ok_or_else(|| IsaRunError::Execution(format!("golden_unreadable:{golden_filename}")))?;

    let json_data = std::fs::read_to_string(&nam_path)
        .map_err(|_| IsaRunError::Execution(format!("model_json_unreadable:{model_filename}")))?;
    let model_data = parse_nam_json(&json_data)
        .map_err(|_| IsaRunError::Execution(format!("model_json_parse_failed:{model_filename}")))?;

    // CRITICAL: set override BEFORE building model — the builder reads
    // SimdMathConfig::get() to decide weight layout, and precision must be pinned
    // explicitly because standard-mode tests may have left the global atomic dirty.
    let _guard = IsaGuard::try_set(isa)
        .map_err(|_| IsaRunError::Execution(format!("isa_override_install_failed:{isa:?}")))?;
    let _prec = PrecisionGuard::new(precision);

    let mut model = build_model(&model_data)
        .map_err(|_| IsaRunError::Execution(format!("model_build_failed:{model_filename}")))?;
    model.prewarm(V2_PREWARM_SAMPLES);

    let num_samples = input.len();
    let mut output = vec![0.0f32; num_samples];
    process_in_blocks(&mut model, &input, &mut output, V2_TEST_BLOCK_SIZE);

    Ok((output, expected))
}

#[cfg(nam_avx512)]
fn active_parity_sink_and_run_id() -> Option<(PathBuf, String)> {
    let path = std::env::var("NAM_SIMD_PARITY_JSONL").ok()?;
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    let run_id = std::env::var("NAM_RUN_ID")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .expect("NAM_RUN_ID must be non-empty when NAM_SIMD_PARITY_JSONL is set");
    Some((PathBuf::from(trimmed), run_id))
}

#[cfg(nam_avx512)]
struct ParityMetricsPayload {
    esr: f64,
    mse: f64,
    max_abs_err: f64,
    budget: f64,
    bitwise_mismatches: usize,
    total_samples: usize,
}

#[cfg(nam_avx512)]
fn maybe_emit_mandatory_record(
    spec: &SimdParityCaseSpec,
    status: SimdParityStatus,
    metrics: Option<ParityMetricsPayload>,
    detail: Option<String>,
) {
    let Some((path, run_id)) = active_parity_sink_and_run_id() else {
        return;
    };
    let (esr, mse, max_abs_err, budget, bitwise_mismatches, total_samples) = match metrics {
        Some(m) => (
            Some(m.esr),
            Some(m.mse),
            Some(m.max_abs_err),
            Some(m.budget),
            Some(m.bitwise_mismatches),
            Some(m.total_samples),
        ),
        None => (None, None, None, None, None, None),
    };
    let record = SimdParityRecord {
        run_id,
        case_id: spec.case_id.to_owned(),
        model: spec.model.to_owned(),
        fixture: spec.fixture.to_owned(),
        precision: spec.precision,
        sample_rate: spec.sample_rate,
        ref_isa: spec.ref_isa.to_owned(),
        test_isa: spec.test_isa.to_owned(),
        status,
        esr,
        mse,
        max_abs_err,
        budget,
        bitwise_mismatches,
        total_samples,
        detail,
    };
    append_simd_parity_record(&path, &record).unwrap_or_else(|e| {
        panic!(
            "Failed to persist mandatory SIMD parity record for {}: {e}",
            spec.case_id
        )
    });
}

/// Compares two output buffers produced under different ISAs and asserts
/// ESR parity within the given budget.
#[cfg(nam_avx512)]
fn evaluate_isa_parity(
    output_ref: &[f32],
    output_test: &[f32],
    label: &str,
    ref_name: &str,
    test_name: &str,
    max_esr: f64,
) -> (bool, f64, f64, f64, common::BitwiseComparison) {
    let esr = compute_esr(output_ref, output_test);
    let mse = compute_mse(output_ref, output_test);
    let mae = compute_max_abs_error(output_ref, output_test);
    let bitwise = compare_samples_bitwise(output_ref, output_test);

    println!(
        "[ISA Matrix] {label} | {ref_name:>10} → {test_name:>10} | \
         ESR={esr:.2e} | MSE={mse:.2e} | MaxAbsErr={mae:.2e} | \
         bitwise mismatches={}/{} (first idx={:?}, ref={:?}, test={:?}) | \
         budget ESR<{max_esr:.1e}",
        bitwise.mismatch_count,
        bitwise.total_samples,
        bitwise.first_mismatch_idx,
        bitwise.first_ref_bits.map(|b| format!("0x{b:08X}")),
        bitwise.first_test_bits.map(|b| format!("0x{b:08X}")),
    );

    // Forensic JSONL sink (kind `isa`) — dashboard compatibility.
    common::report_isa(
        label,
        ref_name,
        test_name,
        Some(esr),
        mse,
        Some(mae),
        Some(max_esr),
    );

    let passed = esr.is_finite() && mse.is_finite() && mae.is_finite() && esr < max_esr;
    (passed, esr, mse, mae, bitwise)
}

/// Executes one mandatory AVX2 -> AVX-512 comparison case from [`MANDATORY_AVX512_PARITY_CASES`].
#[cfg(nam_avx512)]
fn run_mandatory_avx512_case(case_id: &str) {
    let spec = find_mandatory_case(case_id).expect("mandatory SIMD parity case must be registered");
    if !has_full_avx512() {
        eprintln!(
            "[STATUS] SKIP_CAPABILITY reason=\"avx512_cpu_unsupported:{}\"",
            spec.label
        );
        maybe_emit_mandatory_record(
            spec,
            SimdParityStatus::SkippedCapability,
            None,
            Some("avx512_cpu_unsupported".to_owned()),
        );
        return;
    }

    let precision = match spec.precision {
        SimdActivationPrecision::Fast => ActivationPrecision::Fast,
        SimdActivationPrecision::Standard => ActivationPrecision::Standard,
    };

    let (ref_output, _) = match run_under_isa_with_precision(
        spec.model,
        spec.golden,
        spec.sample_rate,
        InstructionSet::Avx2,
        precision,
    ) {
        Ok(pair) => pair,
        Err(err) => {
            let status = match err {
                IsaRunError::MissingFixture(_) => SimdParityStatus::MissingFixture,
                IsaRunError::Execution(_) => SimdParityStatus::Error,
            };
            let reason = err.reason().to_owned();
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"{reason}\"");
            maybe_emit_mandatory_record(spec, status, None, Some(reason.clone()));
            panic!(
                "[{}] mandatory AVX-512 parity case aborted during AVX2 reference run: {reason}",
                spec.case_id
            );
        }
    };

    let (test_output, _) = match run_under_isa_with_precision(
        spec.model,
        spec.golden,
        spec.sample_rate,
        InstructionSet::Avx512,
        precision,
    ) {
        Ok(pair) => pair,
        Err(err) => {
            let status = match err {
                IsaRunError::MissingFixture(_) => SimdParityStatus::MissingFixture,
                IsaRunError::Execution(_) => SimdParityStatus::Error,
            };
            let reason = err.reason().to_owned();
            eprintln!("[STATUS] SKIP_CAPABILITY reason=\"{reason}\"");
            maybe_emit_mandatory_record(spec, status, None, Some(reason.clone()));
            panic!(
                "[{}] mandatory AVX-512 parity case aborted during AVX-512 run: {reason}",
                spec.case_id
            );
        }
    };

    let (passed, esr, mse, mae, bitwise) = evaluate_isa_parity(
        &ref_output,
        &test_output,
        spec.label,
        spec.ref_isa,
        spec.test_isa,
        spec.max_esr,
    );
    let detail =
        (!passed).then(|| format!("ESR={esr:.2e} violated budget ESR<{:.1e}", spec.max_esr));
    maybe_emit_mandatory_record(
        spec,
        if passed {
            SimdParityStatus::Passed
        } else {
            SimdParityStatus::Failed
        },
        Some(ParityMetricsPayload {
            esr,
            mse,
            max_abs_err: mae,
            budget: spec.max_esr,
            bitwise_mismatches: bitwise.mismatch_count,
            total_samples: bitwise.total_samples,
        }),
        detail,
    );

    assert!(
        passed,
        "[{}] ISA parity FAIL: {} → {} ESR={esr:.2e} ≥ budget={:.1e}",
        spec.case_id, spec.ref_isa, spec.test_isa, spec.max_esr
    );
}

/// Legacy evaluation-only VNNI+BF16 check (folds into `Avx512Math` f32; excluded from mandatory gate).
#[cfg(nam_avx512)]
fn run_legacy_vnnibf16_case(model_filename: &str, golden_name: &str, label: &str, max_esr: f64) {
    let sr = 48_000;
    let Ok((ref_output, _)) = run_under_isa_with_precision(
        model_filename,
        golden_name,
        sr,
        InstructionSet::Avx2,
        ActivationPrecision::Fast,
    ) else {
        return;
    };
    #[expect(deprecated)]
    let Ok((test_output, _)) = run_under_isa_with_precision(
        model_filename,
        golden_name,
        sr,
        InstructionSet::Avx512VnniBf16,
        ActivationPrecision::Fast,
    ) else {
        return;
    };
    let (passed, esr, _, _, _) = evaluate_isa_parity(
        &ref_output,
        &test_output,
        &format!("{label} @ {sr} Hz"),
        "AVX2",
        "VNNI+BF16",
        max_esr,
    );
    assert!(
        passed,
        "[{label}] legacy VNNI+BF16 parity FAIL: ESR={esr:.2e} ≥ budget={max_esr:.1e}"
    );
}

/// Runs the same model twice under the same ISA and precision, asserting bitwise identity.
fn assert_isa_self_consistency_with_precision(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    isa: InstructionSet,
    precision: ActivationPrecision,
) {
    let sr = 48000;
    skip_if_unsupported!(isa, label);

    let (output1, _) =
        match run_under_isa_with_precision(model_filename, golden_name, sr, isa, precision) {
            Ok(pair) => pair,
            Err(err) => {
                eprintln!("[STATUS] SKIP_CAPABILITY reason=\"{}\"", err.reason());
                return;
            }
        };
    let (output2, _) =
        match run_under_isa_with_precision(model_filename, golden_name, sr, isa, precision) {
            Ok(pair) => pair,
            Err(err) => {
                eprintln!("[STATUS] SKIP_CAPABILITY reason=\"{}\"", err.reason());
                return;
            }
        };

    let bitwise = compare_samples_bitwise(&output1, &output2);
    let mse = compute_mse(&output1, &output2);
    #[expect(deprecated)]
    let isa_name = match isa {
        InstructionSet::Avx2 => "AVX2",
        InstructionSet::Avx512 => "AVX-512",
        InstructionSet::Avx512VnniBf16 => "VNNI+BF16",
        _ => "UNKNOWN",
    };
    let suffix = match precision {
        ActivationPrecision::Fast => "self-consistency",
        ActivationPrecision::Standard => "self-consistency (HF)",
        _ => "self-consistency (custom)",
    };
    println!(
        "[ISA Matrix] {label} | {isa_name:>10} {suffix} | MSE={mse:.2e} | \
         bitwise mismatches={}/{}",
        bitwise.mismatch_count, bitwise.total_samples
    );

    common::report_isa(label, isa_name, isa_name, None, mse, None, None);

    assert_eq!(
        bitwise.mismatch_count, 0,
        "[{label}] {isa_name} {suffix} FAIL: {} bitwise mismatches across {} samples",
        bitwise.mismatch_count, bitwise.total_samples
    );
    assert!(
        mse == 0.0,
        "[{label}] {isa_name} {suffix} FAIL: MSE={mse:.6e} (expected 0.0)"
    );
}

fn assert_isa_self_consistency(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    isa: InstructionSet,
) {
    assert_isa_self_consistency_with_precision(
        model_filename,
        golden_name,
        label,
        isa,
        ActivationPrecision::Fast,
    );
}

fn assert_isa_hf_self_consistency(
    model_filename: &str,
    golden_name: &str,
    label: &str,
    isa: InstructionSet,
) {
    assert_isa_self_consistency_with_precision(
        model_filename,
        golden_name,
        label,
        isa,
        ActivationPrecision::Standard,
    );
}

// ══════════════════════════════════════════════════════════════════════
// AVX2 self-consistency — runs in quick suite when goldens are present.
// ══════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_wavenet_standard_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("wn-standard-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_wavenet_feather_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("wn-feather-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_wavenet_nano_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("wn-nano-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_lstm_1x16_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("lstm-1x16-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_lstm_2x8_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("lstm-2x8-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_a2_full_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("a2-full-fast");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_a2_lite_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("a2-lite-fast");
}

// ══════════════════════════════════════════════════════════════════════
// Cross-ISA parity tests — AVX2 (ref) vs VNNI+BF16 (ignored by default)
// ══════════════════════════════════════════════════════════════════════

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA legada (requer AVX-512 VNNI+BF16)"]
fn isa_parity_wavenet_standard_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Std/VNNI-BF16");
    // F-SIMD-12: VNNI is not a production path. In dispatch_simd!, Avx512VnniBf16
    // folds into Avx512Math (f32). This test remains as a legacy evaluation checkpoint.
    run_legacy_vnnibf16_case(
        "BossWN-standard.nam",
        "golden_wavenet_standard",
        "WN-Std",
        WN_CROSS_ISA_ESR_BUDGET * 10.0,
    );
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA legada (requer AVX-512 VNNI+BF16)"]
fn isa_parity_wavenet_nano_avx2_vs_vnnibf16() {
    let _lock = ISA_LOCK.lock().unwrap();
    skip_if_unsupported!(InstructionSet::Avx512VnniBf16, "WN-Nano/VNNI-BF16");
    // F-SIMD-12: VNNI is not a production path. In dispatch_simd!, Avx512VnniBf16
    // folds into Avx512Math (f32). This test remains as a legacy evaluation checkpoint.
    run_legacy_vnnibf16_case(
        "BossWN-nano.nam",
        "golden_wavenet_nano",
        "WN-Nano",
        WN_CROSS_ISA_ESR_BUDGET * 10.0,
    );
}

// ══════════════════════════════════════════════════════════════════════
// Standard (exact-grade) mode self-consistency (AVX2) — always runs
// ══════════════════════════════════════════════════════════════════════
//
// Verify that the HF activation paths (scalar + SIMD) are
// deterministic across repeated runs with the same ISA.

#[test]
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
#[ignore = "long: paridade ISA (auto-consistência AVX2/HF)"]
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
// Verify cross-ISA parity in Standard (exact-grade) mode.
// HF polynomial kernels use the same mathematical approximation (degree-6
// Taylor with range reduction) across ISAs, so cross-ISA parity should be
// comparable to or better than standard mode.

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_hf_lstm_1x16_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("lstm-1x16-standard");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_hf_lstm_2x8_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("lstm-2x8-standard");
}

#[cfg(nam_avx512)]
#[test]
#[ignore = "long: paridade ISA (requer hardware AVX-512)"]
fn isa_parity_hf_wavenet_standard_avx2_vs_avx512() {
    let _lock = ISA_LOCK.lock().unwrap();
    run_mandatory_avx512_case("wn-standard-standard");
}

// ══════════════════════════════════════════════════════════════════════
// ISA matrix header (informational, always runs)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn isa_matrix_header_info() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Cross-ISA Determinism Matrix                                ║");
    println!("║  Reference = AVX2 (x86-64-v3, always available)              ║");
    println!("║                                                              ║");
    println!("║  Scope: local runner = AVX2 self-consistency only            ║");
    println!("║  (isa_self_consistency). Cross-ISA matrix = remote gate      ║");
    println!("║  (utils/remote-simd-gate.sh, --features avx512) / long suite ║");
    println!("║                                                              ║");
    println!("║  Kernel-level scalar-vs-SIMD parity: gemv_test.rs,           ║");
    println!("║  dot_4x_test.rs, dot_8x_test.rs, dot_16x_test.rs,            ║");
    println!("║  proptest_math.rs                                            ║");
    println!("║                                                              ║");
    println!("║  Run cross-ISA matrix:                                       ║");
    println!("║  cargo test --release --test parity isa_parity -- \\                 ║");
    println!("║    --ignored --test-threads=1 --nocapture                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}
