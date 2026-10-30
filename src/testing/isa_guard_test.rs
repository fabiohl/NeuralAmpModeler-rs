// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::{SIMD_MATH, clear_test_isa_override, effective_instruction_set};
use serial_test::serial;

#[test]
#[serial]
fn test_force_avx2_guard_lifecycle() {
    let initial_isa = effective_instruction_set();
    {
        let _guard = ForceAvx2Guard::new();
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "effective_instruction_set() must return InstructionSet::Avx2 while guard is active"
        );
    }
    assert_eq!(
        effective_instruction_set(),
        initial_isa,
        "effective_instruction_set() must revert to initial state after guard drops"
    );
}

#[test]
#[serial]
fn test_generic_isa_guard_avx2_lifecycle() {
    let initial_isa = effective_instruction_set();
    {
        let _guard = IsaGuard::try_set(InstructionSet::Avx2)
            .expect("AVX2 override must always be installable");
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx2,
            "effective_instruction_set() must return InstructionSet::Avx2 while guard is active"
        );
    }
    assert_eq!(
        effective_instruction_set(),
        initial_isa,
        "effective_instruction_set() must revert to initial state after guard drops"
    );
}

#[test]
#[serial]
#[cfg(nam_avx512)]
fn test_force_avx512_guard_try_new_requires_full_capability() {
    // Acceptance: try_new() on a host without the full F+VL+BW+DQ
    // matrix returns a structured error — never a crash/SIGILL.
    let initial_isa = effective_instruction_set();
    match ForceAvx512Guard::try_new() {
        Ok(guard) => {
            assert!(
                crate::math::common::has_full_avx512(),
                "a successful try_new() implies the full host capability set"
            );
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx512,
                "effective_instruction_set() must return Avx512 while the guard is active"
            );
            drop(guard);
            assert_eq!(
                effective_instruction_set(),
                initial_isa,
                "guard drop must restore the previous override"
            );
        }
        Err(e) => {
            assert!(
                !crate::math::common::has_full_avx512(),
                "a rejected try_new() implies a partial capability subset"
            );
            assert!(matches!(
                e,
                IsaOverrideError::UnsupportedIsa {
                    isa: InstructionSet::Avx512,
                    ..
                }
            ));
            assert_eq!(
                effective_instruction_set(),
                initial_isa,
                "a rejected try_new() must never touch the override"
            );
        }
    }
}

#[test]
#[serial]
#[cfg(nam_avx512)]
fn test_generic_isa_guard_avx512_lifecycle() {
    let initial_isa = effective_instruction_set();
    let result = IsaGuard::try_set(InstructionSet::Avx512);
    match result {
        Ok(guard) => {
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx512,
                "effective_instruction_set() must return Avx512 while the guard is active"
            );
            drop(guard);
        }
        Err(e) => {
            assert!(matches!(
                e,
                IsaOverrideError::UnsupportedIsa {
                    isa: InstructionSet::Avx512,
                    ..
                }
            ));
        }
    }
    assert_eq!(
        effective_instruction_set(),
        initial_isa,
        "effective_instruction_set() must revert to initial state after guard drops"
    );
}

#[test]
#[serial]
#[cfg(nam_avx512)]
fn test_force_avx512_guard_new_unchecked_injects_without_executing() {
    // Controlled fault-injection: new_unchecked() forces the override even
    // on a partial host, but setting the atomic alone does NOT execute any
    // instruction. The test only inspects the override state and restores
    // it — no kernel is dispatched while the override is active.
    let initial_isa = effective_instruction_set();
    {
        // SAFETY: no SIMD kernel is dispatched for the guard's lifetime —
        // the test only reads effective_instruction_set() (a lock-free
        // atomic read), which cannot SIGILL.
        let _guard = unsafe { ForceAvx512Guard::new_unchecked() };
        assert_eq!(
            effective_instruction_set(),
            InstructionSet::Avx512,
            "new_unchecked() must force the Avx512 override unconditionally"
        );
    }
    assert_eq!(
        effective_instruction_set(),
        initial_isa,
        "guard drop must restore the previous override"
    );
}

#[test]
#[serial]
fn test_clear_override_helper_restores_detection() {
    let _guard = ForceAvx2Guard::new();
    assert_eq!(effective_instruction_set(), InstructionSet::Avx2);
    let prev = clear_test_isa_override();
    assert_eq!(
        effective_instruction_set(),
        SIMD_MATH.instruction_set,
        "clear_test_isa_override() must fall back to the auto-detected ISA"
    );
    // Restore the environment as it was before the test.
    TEST_ISA_OVERRIDE.store(prev, Ordering::SeqCst);
}
