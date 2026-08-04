// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Meta-tests that enforce threshold calibration discipline.
//!
//! Ensures every model with a committed golden `.bin` fixture has a
//! calibrated entry in `get_calibrated_threshold()` — never silently
//! falling back to heuristic thresholds.
//!
//! Part of the calibration infrastructure.
//!
//! All meta-tests implement the Gate Calibration Policy from
//! docs/perceptual_validation.md (Rules 1–5): validated reference,
//! below placebo, provenance comment, link to independent measurement,
//! sanity-check Σ sources ≈ total.
//
//  ## Investigation (E2.1 — 2026-06-24)
//
//  Fidelity Margin ≤ 0.5 dB in `live_cross_validation_lstm_dyn_1x7 (v2)`
//  and `live_cross_validation_linear (v2)` was investigated via `git bisect`
//  between current HEAD and pre-Épico-B commit `ff8a500` ("épico a concluido").
//
//  **Result: PRE-EXISTENT.** The Fidelity Margin values (0.5, 0.4, -0.9, -0.8 dB)
//  are bit-identical between commits. They originate from golden_vectors tests
//  (WaveNet A2-Full/Lite and LSTM models) where SNR(anchor) — the C++ model's
//  own signal degradation — is close to the Rust/C++ parity SNR. The SIMD
//  kernel rewrites (Épicos B/C: fused dot-product accumulate, unified GEMV,
//  1-div tanh) introduced zero regressão in Fidelity Margin.
//
//  The cpp_parity tests originally cited (LSTM-Dyn 1×7 v2, Linear v2) show
//  Fidelity Margin > 40 dB at both commits. The actual low-margin entries
//  belong to the golden_vectors suite (WaveNet A2-Full/Lite, LSTM Official),
//  which were already at those values before the kernel rewrites.

use std::fs;
use std::path::PathBuf;

use super::common;
use common::A2_ESR_LIMIT;
use common::LSTM_ESR_LIMIT;
use common::WAVENET_ESR_LIMIT;
use common::validation::MRSTFT_SOFT_THRESHOLD;
use common::validation::get_calibrated_threshold;

/// Maps a committed golden `.bin` filename to the `model_name` key
/// used in `get_calibrated_threshold()`. Returns `None` for `.bin`
/// files that don't participate in DSP fidelity validation (e.g.
/// cabsim goldens use their own oracles).
///
/// For v2 multi-SR files (`golden_{name}_v2_{sr}.bin`), the suffix
/// is stripped to resolve the base model name.
fn golden_bin_to_model_name(filename: &str) -> Option<&str> {
    // Strip v2 multi-SR suffix: `golden_X_v2_44100.bin` → `golden_X`
    let base = if let Some(suffix_start) = filename.find("_v2_") {
        &filename[..suffix_start]
    } else {
        filename.strip_suffix(".bin").unwrap_or(filename)
    };
    // Handle `.bin` extension removal if not already stripped
    let base = base.strip_suffix(".bin").unwrap_or(base);

    match base {
        "golden_wavenet_standard" => Some("BossWN-standard"),
        "golden_lstm_1x16" => Some("BossLSTM-1x16"),
        "golden_lstm_2x8" => Some("BossLSTM-2x8"),
        "golden_wavenet_a1_standard" => Some("wavenet_a1_standard"),
        "golden_lstm_official" => Some("lstm (Official)"),
        "golden_wavenet_feather" => Some("BossWN-feather"),
        "golden_wavenet_nano" => Some("BossWN-nano"),
        "golden_wavenet_lite" => Some("EVH-5150-Lite"),
        "golden_wavenet_a2_full" => Some("wavenet_a2_full"),
        "golden_wavenet_a2_lite" => Some("wavenet_a2_lite"),
        "golden_wavenet_a2_film_chaos_stress" => Some("wavenet_a2_film_chaos_stress"),
        "golden_wavenet_official" => Some("wavenet_official"),
        // Nondist production models — validated by cpp_parity + golden vectors
        "golden_wavenet_app_evh" => Some("APP-EVH-Stealth100-Dialled-xSTD"),
        "golden_wavenet_boss_bd2" => Some("Boss BD-2 H2O Mod T-12_00 G-12_00"),
        "golden_wavenet_slammin_marshall" => Some("SLAMMIN MARSHALL JTM 45 REISSUE"),
        // Linear FFT — partitioned convolution engine validated against direct FIR oracle
        // and C++ golden vectors (NeuralAmpModelerCore `nam::Linear` dsp.cpp:255-301).
        // Near-bit-exact FFT round-trip; mrstft_max ≤ 0.05 (conservative).
        "golden_linear_fft_rf2048" => Some("linear_fft_rf2048"),
        "golden_linear_fft_rf4096" => Some("linear_fft_rf4096"),
        "golden_linear_fft_rf8192" => Some("linear_fft_rf8192"),
        // cabsim goldens use their own oracle (convolution / C++ parity),
        // not topology_thresholds.
        _ => None,
    }
}

/// Every model with a committed golden `.bin` fixture MUST have a
/// calibrated entry in `get_calibrated_threshold()`. This test
/// prevents silent fallback to heuristic thresholds.
#[test]
fn test_all_golden_models_have_calibrated_thresholds() {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let mut tested_count = 0;

    for entry in fs::read_dir(&fixtures_dir).expect("Failed to read fixtures directory") {
        let entry = entry.expect("Failed to read fixture entry");
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();

        if !filename_str.ends_with(".bin") {
            continue;
        }

        if let Some(model_name) = golden_bin_to_model_name(&filename_str) {
            let threshold = get_calibrated_threshold(model_name);
            assert!(
                threshold.is_some(),
                "Model '{model_name}' has committed golden '{filename_str}' \
                 but NO calibrated entry in get_calibrated_threshold(). \
                 Add an entry with // Measured: SNR=..., ESR=... and recalibrated floors."
            );

            let (_mse, snr_db, esr_opt, _mrstft) = threshold.unwrap();
            // Reject fully-neutralized thresholds (SNR ≤ 0 and no ESR gate)
            if let Some(esr) = esr_opt {
                assert!(
                    snr_db > 0.0 || esr < 1.0,
                    "Model '{model_name}' has both SNR ≤ 0 dB and ESR ≥ 1.0 \
                     — effectively a neutralized/placebo gate. \
                     Thresholds must trace a real measurement."
                );
            } else {
                assert!(
                    snr_db > 0.0,
                    "Model '{model_name}' has SNR ≤ 0 dB with no ESR gate \
                     — effectively a neutralized/placebo gate."
                );
            }

            tested_count += 1;
        }
    }

    assert!(
        tested_count >= 10,
        "Expected ≥ 10 golden models with calibrated thresholds, found {tested_count}"
    );
}

/// Verify that every calibrated entry in `get_calibrated_threshold()`
/// has a `// Measured:` comment documenting ESR and MR-STFT
/// next to the match arm.
///
/// This reads the source file of `validation.rs` and checks for the
/// comment pattern near each distinct match arm.
///
/// Uses the **first** string pattern in each match arm to locate the
/// arm boundary — for `|`-grouped arms this is closest to the
/// `// Measured:` comment at the top of the block.
#[test]
fn test_all_calibrated_entries_have_measurement_comments() {
    let validation_src =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/common/validation.rs");

    let source =
        fs::read_to_string(&validation_src).expect("Failed to read tests/common/validation.rs");

    // One representative model name per match arm in get_calibrated_threshold().
    // These are the FIRST pattern in each arm — closest to the `// Measured:` comment.
    let models: &[&str] = &[
        "BossWN-standard",
        "BossWN-feather",
        "BossWN-nano",
        "wavenet_a1_standard",
        "wavenet_official",
        "BossLSTM-1x16",
        "BossLSTM-2x8",
        "lstm (Official)",
        "EVH-5150-Lite",
        "wavenet_a2_full",
        "wavenet_a2_lite",
        "wavenet_condition_dsp",
        // First nondist model — the `// Measured:` comment is above this pattern
        "APP-EVH-Stealth100-Dialled-xSTD",
        "wavenet_a2_film_lite",
        "wavenet_a2_film_full",
        "wavenet_a2_film_chaos_stress",
        "wavenet_a2_film_input_mixin_pre",
        "a2_dynamic_gated_ch8",
        "a2_dynamic_blended_ch3",
        "wavenet_dyn_free",
        "lstm_dyn_test",
        "a2_example",
        "convnet_test",
        // Linear FFT — partitioned convolution
        "linear_fft_rf2048",
        "linear_fft_rf4096",
        "linear_fft_rf8192",
    ];

    for &model in models {
        let model_line = source
            .lines()
            .enumerate()
            .find(|(_, l)| {
                let t = l.trim();
                t.starts_with('"') && t.contains(model)
            })
            .map(|(i, _)| i)
            .unwrap_or_else(|| {
                panic!(
                    "Could not find match arm for model '{model}' in validation.rs \
                     (expected a line containing '\"{model}\"' as a match arm pattern)"
                )
            });

        let mut found_measured = false;
        for offset in 1..=15 {
            if model_line >= offset {
                let prev_line = source.lines().nth(model_line - offset).unwrap_or("");
                if prev_line.contains("// Measured:") {
                    found_measured = true;
                    break;
                }
            }
        }

        assert!(
            found_measured,
            "Calibrated entry for model '{model}' at line {} is missing \
             '// Measured: ESR=..., MRSTFT=...' comment within 15 lines above. \
             Add the measurement documentation.",
            model_line + 1,
        );
    }
}

/// Every model with a committed golden `.bin` MUST NOT have any
/// placebo (neutralized) threshold component. The basic
/// check (which only rejects when `SNR ≤ 0 AND ESR ≥ 1`), this test
/// treats each dimension independently — any single neutralized
/// component makes the gate a placebo.
///
/// ## What constitutes a placebo threshold:
///
/// 1. `snr_db ≤ 0.0` → SNR gate is neutralized (never catches
///    regressions). Fails regardless of ESR or MSE values.
/// 2. `max_esr ≥ 1.0` → ESR gate is neutralized (never catches
///    regressions). Fails regardless of SNR or MSE values.
/// 3. `mse_limit = None` → MSE gate explicitly not applicable.
///    This is acceptable ONLY if the remaining SNR and ESR gates
///    are "rigid" enough to compensate (SNR ≥ 40 dB AND ESR < 0.1).
///    The A2 Full/Lite/a2_example models intentionally use `None`
///    because their ESR gates are ultra-strict, making
///    MSE redundant. (MseGate::NotApplicable Explicite.)
/// 4. `mrstft_max ≥ 0.5` → MR-STFT gate is neutralized (never catches
///    spectral regressions). MR-STFT is a relative metric bounded [0,1];
///    a threshold ≥ 0.5 would allow severe spectral divergence.
///
/// ## Principle: "todo golden pode falhar"
///
/// A golden test **must** be able to fail — that is the whole point
/// of a gate. A self-golden (output validated against itself) and a
/// neutralized threshold (SNR ≤ 0, ESR ≥ 1, or MSE ≥ 1e29 without
/// rigid SNR+ESR) are **not gates** — they are placebos that grant
/// a false sense of confidence.
///
/// Part of the anti-placebo meta-test.
#[test]
fn test_all_thresholds_anti_placebo() {
    let fixtures_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");

    let mut tested_count = 0;

    for entry in fs::read_dir(&fixtures_dir).expect("Failed to read fixtures directory") {
        let entry = entry.expect("Failed to read fixture entry");
        let filename = entry.file_name();
        let filename_str = filename.to_string_lossy();

        if !filename_str.ends_with(".bin") {
            continue;
        }

        if let Some(model_name) = golden_bin_to_model_name(&filename_str) {
            let threshold = get_calibrated_threshold(model_name);
            assert!(
                threshold.is_some(),
                "Model '{model_name}' has NO calibrated entry for anti-placebo check."
            );

            let (mse_limit, snr_db, esr_opt, mrstft_opt) = threshold.unwrap();

            // Rule 1: SNR ≤ 0 → placebo, regardless of other gates.
            assert!(
                snr_db > 0.0,
                "Model '{model_name}' has SNR = {snr_db} dB ≤ 0 — \
                 placebo gate. SNR must be > 0 to catch regressions."
            );

            // Rule 2: ESR ≥ 1.0 → placebo, regardless of other gates.
            if let Some(esr) = esr_opt {
                assert!(
                    esr < 1.0,
                    "Model '{model_name}' has ESR = {esr} ≥ 1.0 — \
                     placebo gate. ESR must be < 1.0 to catch regressions."
                );
            }

            // Rule 3: MSE gate explicitly not applicable (None) without rigid SNR+ESR → placebo.
            // A2 Full/Lite/a2_example intentionally use None (was 1e30)
            // because their ESR gates are ultra-strict, making MSE redundant.
            if mse_limit.is_none() {
                let esr_rigid = esr_opt.is_some() && esr_opt.unwrap() < 0.1;
                assert!(
                    snr_db >= 40.0 && esr_rigid,
                    "Model '{model_name}' has mse_limit = None \
                     (MSE gate not applicable — ESR primary) but lacks rigid SNR/ESR \
                     compensation (SNR = {snr_db} dB, ESR = {esr_opt:?}). \
                     For MSE gate to be not applicable, SNR must be ≥ 40 dB and ESR must be < 0.1."
                );
            }

            // Rule 4: MR-STFT ≥ 0.5 → placebo.
            // MR-STFT is a relative metric bounded [0,1]; values approaching 1.0
            // indicate spectral collapse. A gate with threshold ≥ 0.5 would never
            // catch meaningful regressions.
            // F64 oracle recalibration: LSTM models are no longer exempt —
            // their MR-STFT thresholds are now recalibrated from v1 measurements
            // (format floor + minimal recurrent drift, all < 0.5) with v2 relaxation
            // handled in cpp_parity.rs via rate-dependent multiplier and ABSOLUTE_MRSTFT_CAP.
            if let Some(mrstft) = mrstft_opt {
                assert!(
                    mrstft < 0.5,
                    "Model '{model_name}' has MR-STFT = {mrstft} ≥ 0.5 — \
                     placebo gate. MR-STFT must be < 0.5 to catch \
                     spectral regressions. Calibrate from real measurements."
                );
            }

            tested_count += 1;
        }
    }

    assert!(
        tested_count >= 10,
        "Expected ≥ 10 golden models in anti-placebo check, found {tested_count}"
    );
}

/// Meta-teste: nenhum gate do oráculo pode ser ≥ linha de placebo.
///
/// Reads the oracle ESR limits (WAVENET_ESR_LIMIT, LSTM_ESR_LIMIT, A2_ESR_LIMIT)
/// from the shared `tests/common/constants.rs` module and asserts that ALL are
/// strictly less than 1.0 — the project's placebo threshold (Rule 2, AC-9):
/// "ESR ≥ 1.0 = placebo gate that never catches regressions."
///
/// This is the guard that prevents future calibration rounds from raising
/// oracle limits back into the placebo zone.
/// (before limits are fixed) and pass after.
///
/// Uses compile-time const assertions: if any limit is raised >= 1.0,
/// compilation fails with a clear message — not just a test failure.
#[test]
fn test_oracle_gates_below_placebo_threshold() {
    const {
        assert!(
            WAVENET_ESR_LIMIT < 1.0,
            "WaveNet oracle gate is placebo (≥ 1.0)"
        );
        assert!(LSTM_ESR_LIMIT < 1.0, "LSTM oracle gate is placebo (≥ 1.0)");
        assert!(A2_ESR_LIMIT < 1.0, "A2 oracle gate is placebo (≥ 1.0)");
    }
}

/// Eixo B isolation guard-rail.
///
/// Ensures that no structural test (Phase 1, debug, `STRUCTURAL_TESTS`)
/// references golden `.bin` files or `golden_vectors` test functions.
/// Structural tests run with debug assertions ON, where float codegen
/// is a "phantom" (no `-O`, no FMA contraction, no auto-vectorization).
/// Any test that compares floats against a reference `.bin` or
/// `golden_vectors` must run in Phase 2 (release) — the Eixo B principle
/// from `docs/testing.md` §2.
///
/// This meta-test reads `utils/tests-quick.sh`, extracts the
/// `STRUCTURAL_TESTS` array, and checks each corresponding `tests/*.rs`
/// source file for forbidden patterns. It acts as an automated gate
/// that prevents the class of regression where a measurement oracle
/// is accidentally placed in the debug phase.
///
/// ## Exclusions
///
/// - `threshold_calibration`: meta-test that validates the golden catalog
///   metadata — references `.bin` filenames for catalog integrity, not
///   float comparison.
/// - `parity_primitives`: references `.bin` for PRNG bit-parity
///   (Mulberry32 vs TypeScript) and MR-STFT algorithm verification
///   (vs Python) — structural parity, not production float measurement.
/// - `lstm_activation_precision` (TODO Eixo B): pre-existing violation —
///   measures SNR of LSTM models against C++ golden `.bin` files. Runs
///   non-ignored in Phase 1 debug. Should be moved to Phase 2 release.
#[test]
fn test_structural_tests_contain_no_bin_references() {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let quick_script = project_root.join("utils").join("tests-quick.sh");

    let script = fs::read_to_string(&quick_script).expect("Failed to read utils/tests-quick.sh");

    let start_marker = "STRUCT_TESTS=(";
    let start = script
        .find(start_marker)
        .expect("STRUCT_TESTS=() array not found in utils/tests-quick.sh");
    let content_start = start + start_marker.len();
    let rest = &script[content_start..];

    let end = rest
        .find("\n)")
        .expect("Could not find closing ')' of STRUCTURAL_TESTS array in utils/tests-quick.sh");
    let array_content = &rest[..end];

    let test_names: Vec<&str> = array_content.split_whitespace().collect();

    assert!(
        !test_names.is_empty(),
        "STRUCTURAL_TESTS array is empty — check parsing of utils/tests-quick.sh"
    );

    const EXCLUDED: &[&str] = &[
        "threshold_calibration", // meta-test — catalog integrity, not float comparison
        "parity_primitives",     // PRNG bit-parity + MR-STFT algorithm verification
        "lstm_activation_precision", // TODO: pre-existing violation — measures SNR vs C++ golden
    ];

    let mut violations: Vec<String> = Vec::new();

    for &test_name in &test_names {
        if EXCLUDED.contains(&test_name) {
            continue;
        }

        const ENTRY_POINTS: &[&str] = &["models", "perf_soak", "parity", "clap", "rt_constraints"];
        let mut test_file = None;
        for entry in ENTRY_POINTS {
            let candidate = project_root
                .join("tests")
                .join(entry)
                .join(format!("{test_name}.rs"));
            if candidate.exists() {
                test_file = Some(candidate);
                break;
            }
        }
        // Legacy flat layout fallback
        if test_file.is_none() {
            let legacy = project_root.join("tests").join(format!("{test_name}.rs"));
            if legacy.exists() {
                test_file = Some(legacy);
            }
        }
        let Some(test_file) = test_file else {
            continue;
        };

        let source = fs::read_to_string(&test_file).unwrap_or_else(|e| {
            panic!(
                "Failed to read {test_path}: {e}",
                test_path = test_file.display()
            )
        });

        // Filter out comment lines to avoid false positives from
        // doc-comments that mention `.bin` without actually loading them
        // (e.g., "No golden .bin files or C++" in linear_golden.rs).
        let non_comment: String = source
            .lines()
            .filter(|line| {
                let trimmed = line.trim_start();
                !trimmed.starts_with("//") && !trimmed.starts_with("///")
            })
            .collect::<Vec<_>>()
            .join("\n");

        if non_comment.contains(".bin") {
            violations.push(format!(
                "{test_name}.rs references .bin files (forbidden in Phase 1 structural tests)"
            ));
        }

        if non_comment.contains("golden_vectors") {
            violations.push(format!(
                "{test_name}.rs references golden_vectors (forbidden in Phase 1 structural tests)"
            ));
        }
    }

    if !violations.is_empty() {
        panic!(
            "Eixo B isolation guard-rail FAILED:\n\
             \n\
             The following structural tests (Phase 1, debug) contain references\n\
             to .bin files or golden_vectors — they MUST run in Phase 2 (release),\n\
             per the Eixo B principle in docs/testing.md §2:\n\
             \n{}\n\n\
             Resolution: remove these tests from STRUCTURAL_TESTS in\n\
             utils/tests-quick.sh and add them to the Phase 2 release invocation.\n",
            violations
                .iter()
                .map(|v| format!("  ✗ {v}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Meta-teste: limite do soft gate de MR-STFT é calibrado.
///
/// Verifica que o `MRSTFT_SOFT_THRESHOLD` (gate brando informacional para taxas
/// de amostragem não-padrão) é:
///
/// 1. **Abaixo do teto anti-placebo:** ≤ 0.5 (Rule 4 do anti-placebo).
///    MR-STFT é uma métrica limitada em [0, 1]; valores ≥ 0.5 indicam colapso
///    espectral. O soft gate não pode nunca exceder o teto de placebo.
///
/// 2. **Não-zero:** > 0.0. Um soft gate de 0.0 seria inútil — toda divergência
///    espectral geraria falso-positivo. O gate deve ser calibrado com margem
///    acima dos modelos calibrados e abaixo do teto anti-placebo.
///
/// 3. **Documentado:** a definição de `pub const MRSTFT_SOFT_THRESHOLD` em
///    `tests/common/validation.rs` deve ter um comentário `// Measured:` nas
///    proximidades, documentando a proveniência da calibração.
///
/// O gate brando opera em taxas não-padrão (≠ 44.1/48 kHz) onde os hard gates
/// por-modelo não se aplicam. Ele é puramente informacional — não causa falha
/// de teste — mas serve como guard-rail global de sanidade espectral.
#[test]
fn test_mrstft_soft_threshold_is_calibrated() {
    // Rule 1: below anti-placebo ceiling (≤ 0.5)
    // Rule 2: non-zero (must be a real gate, not a trivial bypass)
    const {
        assert!(
            MRSTFT_SOFT_THRESHOLD <= 0.5,
            "MRSTFT_SOFT_THRESHOLD exceeds anti-placebo ceiling 0.5"
        );
        assert!(
            MRSTFT_SOFT_THRESHOLD > 0.0,
            "MRSTFT_SOFT_THRESHOLD ≤ 0 — trivial bypass"
        );
    }

    // Rule 3: documented with measurement provenance
    // Read the validation.rs source and check for a `// Measured:` comment
    // near the `MRSTFT_SOFT_THRESHOLD` definition.
    let validation_src =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/common/validation.rs");

    let source =
        fs::read_to_string(&validation_src).expect("Failed to read tests/common/validation.rs");

    let threshold_line = source
        .lines()
        .enumerate()
        .find(|(_, l)| {
            let t = l.trim();
            t.starts_with("pub const MRSTFT_SOFT_THRESHOLD")
        })
        .map(|(i, _)| i)
        .expect("Could not find 'pub const MRSTFT_SOFT_THRESHOLD' definition in validation.rs");

    let mut found_measured = false;
    for offset in 1..=25 {
        if threshold_line >= offset {
            let prev_line = source.lines().nth(threshold_line - offset).unwrap_or("");
            if prev_line.contains("// Measured:") {
                found_measured = true;
                break;
            }
        }
    }

    assert!(
        found_measured,
        "MRSTFT_SOFT_THRESHOLD at line {} is missing '// Measured:' comment within 25 lines above. \
         Add calibration provenance documentation.",
        threshold_line + 1,
    );
}

/// Meta-test: every `set_activation_tls(` call-site in
/// `tests/**/*.rs` (outside `tests/common/precision.rs`) must have
/// `PrecisionGuard::new` in the same function.
///
/// This meta-test reads all test source files and enforces the rule that
/// any function calling `set_activation_tls()` is protected by a
/// `PrecisionGuard`, guarding against race conditions when tests run
/// in parallel.
///
/// ## Mechanism
///
/// Parses Rust source files by tracking brace depth to identify function
/// boundaries, then verifies that every function body containing
/// `set_activation_tls(` also contains `PrecisionGuard::new`.
#[test]
fn test_all_set_activation_calls_are_guarded() {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tests_dir = project_root.join("tests");
    let precision_rs = tests_dir.join("common").join("precision.rs");

    let mut rs_files: Vec<PathBuf> = Vec::new();
    fn collect_rs(dir: &PathBuf, out: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    collect_rs(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
    }
    collect_rs(&tests_dir, &mut rs_files);

    let mut violations: Vec<String> = Vec::new();

    for file_path in &rs_files {
        if *file_path == precision_rs {
            continue;
        }

        let source = fs::read_to_string(file_path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {e}", file_path.display()));

        if !source.contains("set_activation_tls(") {
            continue;
        }

        let mut depth: i32 = 0;
        let mut in_fn = false;
        let mut has_set = false;
        let mut has_guard = false;
        let mut fn_start_line: usize = 0;

        for (i, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if !in_fn
                && depth == 0
                && (trimmed.starts_with("fn ") || trimmed.starts_with("pub fn "))
            {
                in_fn = true;
                has_set = false;
                has_guard = false;
                fn_start_line = i + 1;
            }

            if !in_fn {
                depth += line.matches('{').count() as i32;
                depth -= line.matches('}').count() as i32;
                continue;
            }

            depth += line.matches('{').count() as i32;
            depth -= line.matches('}').count() as i32;

            if line.contains("set_activation_tls(") {
                has_set = true;
            }
            if line.contains("PrecisionGuard::new") {
                has_guard = true;
            }

            if depth == 0 {
                if has_set && !has_guard {
                    violations.push(format!(
                        "{}:{} — set_activation_tls() without PrecisionGuard::new in the same function",
                        file_path
                            .strip_prefix(&project_root)
                            .unwrap_or(file_path.as_path())
                            .display(),
                        fn_start_line,
                    ));
                }
                in_fn = false;
            }
        }
    }

    if !violations.is_empty() {
        panic!(
            "Guard-rail FAILED: unprotected set_activation_tls() call-sites found:\n\
             \n{}\n\n\
             Every test function that calls set_activation_tls() must also contain\n\
             PrecisionGuard::new, acquired BEFORE TrackingGuard, to prevent race conditions\n\
             when tests run in parallel.\n",
            violations
                .iter()
                .map(|v| format!("  ✗ {v}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

/// Meta-teste anti-let_: nenhum wrapper em cpp_parity.rs pode
/// descartar silenciosamente o `ParityOutcome` retornado por
/// `run_render_comparison`.
///
/// Analisa o código-fonte de `tests/parity/cpp_parity.rs` e falha se encontrar
/// qualquer ocorrência do padrão `let _ = run_render_comparison`. Todas as
/// chamadas a `run_render_comparison` devem capturar o `ParityOutcome` retornado
/// e tomar decisões explícitas sobre ele (assert, SKIP-COVERAGE, etc).
#[test]
fn test_no_silent_let_underscore_in_cpp_parity_wrappers() {
    let cpp_parity_src =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/parity/cpp_parity.rs");

    let source =
        fs::read_to_string(&cpp_parity_src).expect("Failed to read tests/parity/cpp_parity.rs");

    assert!(
        !source.contains("let _ = run_render_comparison"),
        "cpp_parity.rs contains silent discard of ParityOutcome via \
         'let _ = run_render_comparison'. All calls to run_render_comparison \
         must capture the returned ParityOutcome and assert on it."
    );
}

/// Checks that a skip_reason string contains a date annotation
/// in the format `(YYYY-MM-DD)`, proving the skip was reviewed.
fn skip_reason_has_date(reason: &str) -> bool {
    let bytes = reason.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'('
            && let Some(close_offset) = bytes[i..].iter().position(|&b| b == b')')
        {
            let inside = &bytes[i + 1..i + close_offset];
            if inside.len() == 10
                && inside[0].is_ascii_digit()
                && inside[1].is_ascii_digit()
                && inside[2].is_ascii_digit()
                && inside[3].is_ascii_digit()
                && inside[4] == b'-'
                && inside[5].is_ascii_digit()
                && inside[6].is_ascii_digit()
                && inside[7] == b'-'
                && inside[8].is_ascii_digit()
                && inside[9].is_ascii_digit()
            {
                return true;
            }
            i += close_offset + 1;
            continue;
        }
        i += 1;
    }
    false
}

/// Maps a catalog entry golden_name to the model name key used
/// in `get_calibrated_threshold()`. Returns `None` for entries that only
/// need date-check validation (skip_reason).
///
/// Covers ALL entries in `golden_gen_build.sh` CATALOG, including models
/// that don't have committed `.bin` fixtures (e.g. A2-FiLM, dynamic engines).
fn catalog_entry_to_model_name<'a>(_nam_file: &str, golden_name: &'a str) -> Option<&'a str> {
    let base = golden_name.strip_prefix("golden_").unwrap_or(golden_name);
    match base {
        "wavenet_standard" => Some("BossWN-standard"),
        "wavenet_lite" => Some("EVH-5150-Lite"),
        "wavenet_feather" => Some("BossWN-feather"),
        "wavenet_nano" => Some("BossWN-nano"),
        "wavenet_a1_standard" => Some("wavenet_a1_standard"),
        "wavenet_official" => Some("wavenet_official"),
        "lstm_1x16" => Some("BossLSTM-1x16"),
        "lstm_2x8" => Some("BossLSTM-2x8"),
        "lstm_official" => Some("lstm (Official)"),
        "wavenet_a2_full" => Some("wavenet_a2_full"),
        "wavenet_a2_lite" => Some("wavenet_a2_lite"),
        "wavenet_condition_dsp" => Some("wavenet_condition_dsp"),
        // condition_lstm has skip_reason → only date check, not calibration check
        "wavenet_condition_lstm" => None,
        "a2_example" => Some("a2_example"),
        "wavenet_app_evh" => Some("APP-EVH-Stealth100-Dialled-xSTD"),
        "wavenet_boss_bd2" => Some("Boss BD-2 H2O Mod T-12_00 G-12_00"),
        "wavenet_slammin_marshall" => Some("SLAMMIN MARSHALL JTM 45 REISSUE"),
        "wavenet_dyn_free" => Some("wavenet_dyn_free"),
        "lstm_dyn_test" => Some("lstm_dyn_test"),
        "convnet_test" => Some("convnet_test"),
        "wavenet_a2_max" => Some("wavenet_a2_max"),
        "a2_dynamic_gated_ch8" => Some("a2_dynamic_gated_ch8"),
        "a2_dynamic_blended_ch3" => Some("a2_dynamic_blended_ch3"),
        "wavenet_a2_film_lite" => Some("wavenet_a2_film_lite"),
        "wavenet_a2_film_full" => Some("wavenet_a2_film_full"),
        "wavenet_a2_film_chaos_stress" => Some("wavenet_a2_film_chaos_stress"),
        "wavenet_a2_film_input_mixin_pre" => Some("wavenet_a2_film_input_mixin_pre"),
        "linear_fft_rf2048" => Some("linear_fft_rf2048"),
        "linear_fft_rf4096" => Some("linear_fft_rf4096"),
        "linear_fft_rf8192" => Some("linear_fft_rf8192"),
        "linear_fft_rf320" => Some("linear_fft_rf320"),
        _ => None,
    }
}

/// Auditoria Anti-Placebo Estendida ao CATALOG.
///
/// Extends `test_all_thresholds_anti_placebo` beyond `.bin` fixtures to
/// cover ALL entries in the `golden_gen_build.sh` CATALOG, including models
/// without golden binaries (e.g. FiLM, dynamic engines).
///
/// # Rules enforced:
///
/// 1. **Entries with `skip_reason`**: the reason must contain a date annotation
///    in `(YYYY-MM-DD)` format, proving the skip is reviewed and not perpetual.
///
/// 2. **Entries without `skip_reason`**: must have a `Some` calibrated entry in
///    `get_calibrated_threshold()` that passes the anti-placebo Rules 1–4:
///    SNR > 0, ESR < 1.0, MSE-None compensation, MR-STFT < 0.5.
///
/// # Catalog parsing (from golden_gen_build.sh):
///
/// Format: `.nam_file:golden_name:label:v2_scope[:skip_srs[:skip_reason]]`
///
/// Part of the anti-placebo audit: close the gap where models without `.bin` fixtures
/// (condition_lstm) escaped anti-placebo audit entirely.
#[test]
fn test_catalog_anti_placebo_audit() {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let catalog_script = project_root
        .join("tests")
        .join("fixtures")
        .join("golden_gen_build.sh");

    let script = fs::read_to_string(&catalog_script).expect("Failed to read golden_gen_build.sh");

    let cat_start = script
        .find("CATALOG=(")
        .expect("Could not find CATALOG=() in golden_gen_build.sh");

    let rest = &script[cat_start..];
    let cat_end = rest
        .find("\n)")
        .expect("Could not find closing ')' of CATALOG array");

    let cat_body = &rest[..=cat_end];
    let mut lines = cat_body.lines();

    let mut tested_skip = 0usize;
    let mut tested_calibrated = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for line in lines.by_ref() {
        let trimmed = line.trim();
        if trimmed == "CATALOG=(" || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed == ")" {
            break;
        }

        let inner = trimmed.trim_matches('"').trim();
        let mut parts = inner.splitn(6, ':');
        let nam_file = parts.next().unwrap_or("");
        let golden_name = parts.next().unwrap_or("");
        let label = parts.next().unwrap_or("");
        let _v2_scope = parts.next().unwrap_or("");
        let _skip_srs = parts.next().unwrap_or("");
        let skip_reason = parts.next().unwrap_or("");

        if !skip_reason.is_empty() {
            tested_skip += 1;
            if !skip_reason_has_date(skip_reason) {
                failures.push(format!(
                    "CATALOG entry '{label}' ({nam_file}/{golden_name}) has skip_reason=\"{skip_reason}\" \
                     without (YYYY-MM-DD) date.\n  \
                     Add a date annotation (e.g. \"(2026-07-11)\") to prove the skip is reviewed."
                ));
            }
        } else {
            match catalog_entry_to_model_name(nam_file, golden_name) {
                None => {
                    failures.push(format!(
                        "CATALOG entry '{label}' ({nam_file}/{golden_name}) has no skip_reason \
                         but golden_name \"{golden_name}\" is not recognized by \
                         catalog_entry_to_model_name().\n  \
                         Add a mapping in catalog_entry_to_model_name() or set skip_reason."
                    ));
                }
                Some(model_name) => {
                    let threshold = get_calibrated_threshold(model_name);
                    match threshold {
                        None => {
                            failures.push(format!(
                                "CATALOG entry '{label}' ({nam_file}/{golden_name}) mapped to model \
                                 '{model_name}' but get_calibrated_threshold() returned None.\n  \
                                 Add a calibrated entry in get_calibrated_threshold() for '{model_name}'."
                            ));
                        }
                        Some((mse_limit, snr_db, esr_opt, mrstft_opt)) => {
                            tested_calibrated += 1;

                            if snr_db <= 0.0 {
                                failures.push(format!(
                                    "CATALOG '{label}' ({model_name}): SNR={snr_db} dB ≤ 0 — placebo gate."
                                ));
                            }

                            if let Some(esr) = esr_opt
                                && esr >= 1.0
                            {
                                failures.push(format!(
                                        "CATALOG '{label}' ({model_name}): ESR={esr} ≥ 1.0 — placebo gate."
                                    ));
                            }

                            if mse_limit.is_none() {
                                let esr_rigid = esr_opt.is_some() && esr_opt.unwrap() < 0.1;
                                if !(snr_db >= 40.0 && esr_rigid) {
                                    failures.push(format!(
                                        "CATALOG '{label}' ({model_name}): MSE=None without rigid SNR/ESR \
                                         compensation (SNR={snr_db} dB, ESR={esr_opt:?})."
                                    ));
                                }
                            }

                            if let Some(mrstft) = mrstft_opt
                                && mrstft >= 0.5
                            {
                                failures.push(format!(
                                        "CATALOG '{label}' ({model_name}): MR-STFT={mrstft} ≥ 0.5 — placebo gate."
                                    ));
                            }
                        }
                    }
                }
            }
        }
    }

    assert!(
        tested_skip + tested_calibrated > 0,
        "No CATALOG entries parsed — check golden_gen_build.sh format"
    );

    if !failures.is_empty() {
        panic!(
            "CATALOG anti-placebo audit FAILED ({} failure(s)):\n\n{}\n\n\
             {} entries checked ({} with skip_reason, {} calibrated).\n\
             \n\
             Fixes needed:\n\
             - For skip_reason entries: add (YYYY-MM-DD) date annotation.\n\
             - For uncalibrated entries: add a match arm in get_calibrated_threshold()\n\
               with real measurement data and // Measured: comment.\n",
            failures.len(),
            failures
                .iter()
                .map(|f| format!("  ✗ {f}"))
                .collect::<Vec<_>>()
                .join("\n\n"),
            tested_skip + tested_calibrated,
            tested_skip,
            tested_calibrated,
        );
    }
}

/// Meta-teste: qualidade-contract.txt não pode conter rótulos sintéticos.
///
/// Self-tests que degradam o sinal para validar gates de regressão (ex:
/// `test_mrstft_hard_gate_catches_regression`) emitem labels com `(synthetic)`.
/// Essas labels contaminavam o `--save` do dashboard e eram carregadas pelo
/// `--check` como parte do contrato de qualidade — uma violação da integridade
/// da baseline.
///
/// Este meta-teste audita `docs/quality-contract.txt` e falha se QUALQUER linha
/// contiver `(synthetic)`, garantindo que o contrato só contém medições reais
/// de fidelidade.
///
/// Parte da trava de segurança permanente para completar o expurgo.
#[test]
fn test_quality_contract_no_synthetic_labels() {
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let contract_file = project_root.join("docs").join("quality-contract.txt");

    let content =
        fs::read_to_string(&contract_file).expect("Failed to read docs/quality-contract.txt");

    assert!(
        !content.contains("(synthetic)"),
        "docs/quality-contract.txt contains labels with '(synthetic)' — \
         self-test degradation entries have contaminated the quality baseline.\n\
         Remove all lines containing '(synthetic)' from the contract file.\n\
         (The source was fixed; this meta-test is the permanent guard.)"
    );
}
