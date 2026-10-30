// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use serial_test::serial;

#[test]
fn test_effective_instruction_set_default_avx2() {
    #[cfg(not(nam_avx512))]
    {
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "effective_instruction_set() must return Avx2 when avx512 feature is off"
        );
        assert_eq!(
            SIMD_MATH.instruction_set,
            InstructionSet::Avx2,
            "SIMD_MATH.instruction_set must be Avx2 when avx512 feature is off"
        );
        assert!(!SIMD_MATH.is_avx512, "SIMD_MATH.is_avx512 must be false");
    }
}

#[test]
#[serial]
fn test_override_ignored_without_avx512_feature() {
    #[cfg(not(nam_avx512))]
    {
        let prev = TEST_ISA_OVERRIDE.swap(1, Ordering::SeqCst);
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "Overriding to 1 (AVX-512) must be ignored and fallback to Avx2 without feature"
        );
        assert_eq!(
            decode_isa_override(1),
            None,
            "decode_isa_override(1) must return None without avx512 feature"
        );

        TEST_ISA_OVERRIDE.store(2, Ordering::SeqCst);
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "Overriding to 2 (VNNI) must be ignored and fallback to Avx2 without feature"
        );
        assert_eq!(
            decode_isa_override(2),
            None,
            "decode_isa_override(2) must return None without avx512 feature"
        );

        TEST_ISA_OVERRIDE.store(0, Ordering::SeqCst);
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "Overriding to 0 (Avx2) returns Avx2"
        );
        assert_eq!(decode_isa_override(0), Some(InstructionSet::Avx2));

        TEST_ISA_OVERRIDE.store(prev, Ordering::SeqCst);
    }
}

#[test]
#[cfg(nam_avx512)]
fn test_decode_isa_override_with_avx512_feature() {
    assert_eq!(decode_isa_override(0), Some(InstructionSet::Avx2));
    assert_eq!(decode_isa_override(1), Some(InstructionSet::Avx512));
    #[expect(deprecated)]
    {
        assert_eq!(decode_isa_override(2), Some(InstructionSet::Avx512VnniBf16));
    }
    assert_eq!(decode_isa_override(3), None);
    assert_eq!(decode_isa_override(255), None);
}

// ── Capability matrix (pure, hardware-independent) ───────────────────────

#[test]
fn avx512_capability_matrix_requires_all_four_flags() {
    // F+VL without BW/DQ is the partial-subset SIGILL vector (F-ROB-03).
    assert!(!avx512_capability_complete(true, true, false, false));
    // Any single missing flag must reject.
    assert!(!avx512_capability_complete(false, true, true, true));
    assert!(!avx512_capability_complete(true, false, true, true));
    assert!(!avx512_capability_complete(true, true, true, false));
    // Empty set and full set bounds.
    assert!(!avx512_capability_complete(false, false, false, false));
    assert!(avx512_capability_complete(true, true, true, true));
}

#[cfg(nam_avx512)]
#[test]
fn detect_falls_back_to_avx2_without_full_capability() {
    // Deterministic per-host: with the avx512 feature on, runtime
    // detection must return Avx512 only when ALL of F+VL+BW+DQ are
    // Present; any partial subset yields the Avx2 fallback.
    let detected = detect_best_simd().instruction_set;
    if has_full_avx512() {
        assert_eq!(
            detected,
            InstructionSet::Avx512,
            "full AVX-512 capability set must select Avx512 under the avx512 feature"
        );
    } else {
        assert_eq!(
            detected,
            InstructionSet::Avx2,
            "partial AVX-512 capability must fall back to Avx2 (no SIGILL risk)"
        );
    }
}

#[cfg(nam_avx512)]
#[test]
fn missing_features_reports_only_absent_flags() {
    if has_full_avx512() {
        assert!(missing_avx512_features().is_empty());
    } else {
        let missing = missing_avx512_features();
        assert!(
            !missing.is_empty(),
            "a partial host must report missing flags"
        );
        assert!(
            missing.len() <= 4,
            "only the F/VL/BW/DQ flags can be reported, got {missing:?}"
        );
        // Consistency: the pure matrix and the runtime report agree.
        let present = |name: &str| !missing.contains(&name);
        let expect_full =
            present("avx512f") && present("avx512vl") && present("avx512bw") && present("avx512dq");
        assert_eq!(
            has_full_avx512(),
            expect_full,
            "missing-feature report must agree with has_full_avx512()"
        );
    }
}

// ── Validated override install and atomic concurrency safety ──────────────

#[test]
#[serial]
fn override_avx2_is_always_acceptable() {
    let prev = set_test_isa_override(InstructionSet::Avx2).expect("AVX2 override must install");
    assert_eq!(effective_instruction_set(), InstructionSet::Avx2);
    assert_eq!(
        clear_test_isa_override(),
        encode_isa_override(InstructionSet::Avx2)
    );
    // Restore the environment as it was before the test.
    TEST_ISA_OVERRIDE.store(prev, Ordering::SeqCst);
}

#[cfg(not(nam_avx512))]
#[test]
#[serial]
fn override_rejects_avx512_without_feature() {
    let err = set_test_isa_override(InstructionSet::Avx512).expect_err(
        "without the avx512 feature, forcing Avx512 must be rejected, not silently ignored",
    );
    assert!(matches!(
        err,
        IsaOverrideError::UnsupportedIsa {
            isa: InstructionSet::Avx512,
            ..
        }
    ));
    assert_eq!(
        effective_instruction_set(),
        InstructionSet::Avx2,
        "a rejected override must leave dispatch on the detected baseline"
    );
}

#[cfg(nam_avx512)]
#[test]
#[serial]
fn override_avx512_requires_full_capability() {
    let result = set_test_isa_override(InstructionSet::Avx512);
    match result {
        Ok(prev) => {
            assert!(
                has_full_avx512(),
                "a successful AVX-512 override implies the full F+VL+BW+DQ matrix"
            );
            assert_eq!(effective_instruction_set(), InstructionSet::Avx512);
            clear_test_isa_override();
            TEST_ISA_OVERRIDE.store(prev, Ordering::SeqCst);
        }
        Err(e) => {
            assert!(!has_full_avx512());
            assert!(matches!(
                e,
                IsaOverrideError::UnsupportedIsa {
                    isa: InstructionSet::Avx512,
                    ..
                }
            ));
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx2,
                "a rejected AVX-512 override must leave dispatch on AVX2"
            );
        }
    }
}

#[cfg(nam_avx512)]
#[test]
#[serial]
fn override_vnni_requires_bf16_and_vnni_on_top_of_full_avx512() {
    #[expect(deprecated)]
    let result = set_test_isa_override(InstructionSet::Avx512VnniBf16);
    match result {
        Ok(prev) => {
            assert!(has_full_avx512());
            assert!(
                is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vnni"),
                "a successful VNNI override requires bf16+vnni on top of the full matrix"
            );
            clear_test_isa_override();
            TEST_ISA_OVERRIDE.store(prev, Ordering::SeqCst);
        }
        Err(_) => {
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx2,
                "a rejected VNNI override must leave dispatch on AVX2"
            );
        }
    }
}

#[test]
#[serial]
fn concurrent_override_installs_never_produce_invalid_state() {
    // Reads via effective_instruction_set() stay lock-free (Relaxed)
    // and concurrent installs/clears (SeqCst swap) must never leave the
    // atomic in a state that decodes to an invalid ISA or panics.
    let handles: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                for _ in 0..2_000 {
                    if i % 2 == 0 {
                        let _ = set_test_isa_override(InstructionSet::Avx2);
                    } else {
                        let _ = clear_test_isa_override();
                    }
                    let isa = effective_instruction_set();
                    assert!(
                        decode_isa_override(encode_isa_override(isa)).is_some(),
                        "effective_instruction_set() must always decode to a valid ISA"
                    );
                }
            })
        })
        .collect();
    for handle in handles {
        handle
            .join()
            .expect("concurrent override thread must not panic");
    }
    clear_test_isa_override();
}
