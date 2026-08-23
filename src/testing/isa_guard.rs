// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RAII Guards for forcing specific instruction set dispatch in testing & benchmarking.

use crate::math::common::dispatch::detect::TEST_ISA_OVERRIDE;
#[cfg(feature = "avx512")]
use crate::math::common::encode_isa_override;
use crate::math::common::{InstructionSet, IsaOverrideError, set_test_isa_override};
use std::sync::atomic::Ordering;

/// Off-RT RAII guard that forces SIMD instruction dispatch to operate
/// exclusively on [`InstructionSet::Avx2`] for the lifetime of the guard.
///
/// On [`Drop::drop`], the previous instruction set override setting is restored.
///
/// AVX2 is the mandatory `x86-64-v3` baseline, so this guard is unconditionally
/// safe to install on every supported host.
#[derive(Debug)]
pub struct ForceAvx2Guard {
    prev_override: u8,
}

impl ForceAvx2Guard {
    /// Creates a new guard, forcing [`InstructionSet::Avx2`] SIMD dispatch.
    pub fn new() -> Self {
        let prev = set_test_isa_override(InstructionSet::Avx2)
            .expect("AVX2 override must always be installable on the x86-64-v3 baseline");
        Self {
            prev_override: prev,
        }
    }
}

impl Default for ForceAvx2Guard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ForceAvx2Guard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

/// Off-RT RAII guard that forces SIMD instruction dispatch to operate
/// exclusively on [`InstructionSet::Avx512`] for the lifetime of the guard.
///
/// On [`Drop::drop`], the previous instruction set override setting is restored.
#[cfg(feature = "avx512")]
#[derive(Debug)]
pub struct ForceAvx512Guard {
    prev_override: u8,
}

#[cfg(feature = "avx512")]
impl ForceAvx512Guard {
    /// Fallibly creates a new guard, forcing [`InstructionSet::Avx512`] SIMD
    /// dispatch — but ONLY after validating that the host CPU supports the
    /// complete AVX-512 capability matrix (`F+VL+BW+DQ`) required by the
    /// reachable kernels (T2.2 / F-ROB-03).
    ///
    /// On a host with a partial AVX-512 subset (e.g. F+VL without BW/DQ —
    /// common under VMs/hypervisors) the override is NOT installed and a
    /// structured [`IsaOverrideError::UnsupportedIsa`] is returned. This is
    /// the fail-closed path: safe Rust can never force dispatch towards
    /// instructions that would raise `SIGILL`.
    ///
    /// Callers in tests/harnesses should map the error to a typed skip
    /// (e.g. `SKIP_CAPABILITY`) instead of panicking.
    pub fn try_new() -> Result<Self, IsaOverrideError> {
        set_test_isa_override(InstructionSet::Avx512).map(|prev| Self {
            prev_override: prev,
        })
    }

    /// Unchecked constructor for emulation and controlled fault-injection
    /// harnesses (e.g. `remote-simd-gate.sh` under Intel SDE, or tests that
    /// exercise the override machinery without executing kernels).
    ///
    /// # Safety
    /// The caller must guarantee that dispatching [`InstructionSet::Avx512`]
    /// kernels is safe for the lifetime of the guard: either the host CPU
    /// supports the full `F+VL+BW+DQ` matrix, the code runs under an
    /// instruction emulator (SDE), or no SIMD kernel is actually dispatched
    /// while the override is active. Violating this precondition causes
    /// `SIGILL` (illegal instruction).
    pub unsafe fn new_unchecked() -> Self {
        // SAFETY: guaranteed by the caller per the # Safety contract above.
        let prev = TEST_ISA_OVERRIDE.swap(
            encode_isa_override(InstructionSet::Avx512),
            Ordering::SeqCst,
        );
        Self {
            prev_override: prev,
        }
    }

    /// Legacy infallible constructor.
    ///
    /// Prefer [`Self::try_new`]: this constructor panics (never installs the
    /// override) when the host lacks the full `F+VL+BW+DQ` capability set —
    /// the fail-closed safe behavior.
    #[deprecated(
        note = "use try_new() which returns IsaOverrideError instead of panicking when the host lacks full AVX-512 (F+VL+BW+DQ)"
    )]
    pub fn new() -> Self {
        Self::try_new()
            .expect("AVX-512 dispatch override requires the full host capability set (F+VL+BW+DQ)")
    }
}

#[cfg(feature = "avx512")]
impl Default for ForceAvx512Guard {
    fn default() -> Self {
        Self::try_new()
            .expect("AVX-512 dispatch override requires the full host capability set (F+VL+BW+DQ)")
    }
}

#[cfg(feature = "avx512")]
impl Drop for ForceAvx512Guard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

/// Generic RAII guard that forces SIMD instruction dispatch to operate
/// on the specified [`InstructionSet`] for the lifetime of the guard.
#[derive(Debug)]
pub struct IsaGuard {
    prev_override: u8,
}

impl IsaGuard {
    /// Fallibly creates a new guard, forcing the specified ISA for SIMD
    /// dispatch after validating host CPU capability (T2.3).
    ///
    /// Returns [`IsaOverrideError`] when the requested ISA cannot be executed
    /// on the current host (e.g. AVX-512 on a CPU with a partial subset) so
    /// safe Rust never installs an override that would `SIGILL`.
    pub fn try_set(isa: InstructionSet) -> Result<Self, IsaOverrideError> {
        set_test_isa_override(isa).map(|prev| Self {
            prev_override: prev,
        })
    }

    /// Legacy infallible constructor — panics (fail-closed, never `SIGILL`)
    /// when the requested ISA is not executable on the current host.
    #[deprecated(note = "use try_set() which returns IsaOverrideError on capability mismatch")]
    pub fn set(isa: InstructionSet) -> Self {
        Self::try_set(isa).expect("ISA dispatch override requires matching host capabilities")
    }
}

impl Drop for IsaGuard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
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
    #[cfg(feature = "avx512")]
    fn test_force_avx512_guard_try_new_requires_full_capability() {
        // T2.2 acceptance: try_new() on a host without the full F+VL+BW+DQ
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
    #[cfg(feature = "avx512")]
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
    #[cfg(feature = "avx512")]
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
}
