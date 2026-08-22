// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RAII Guards for forcing specific instruction set dispatch in testing & benchmarking.

use crate::math::common::{InstructionSet, TEST_ISA_OVERRIDE, encode_isa_override};
use std::sync::atomic::Ordering;

/// Off-RT RAII guard that forces SIMD instruction dispatch to operate
/// exclusively on [`InstructionSet::Avx2`] for the lifetime of the guard.
///
/// On [`Drop::drop`], the previous instruction set override setting is restored.
#[derive(Debug)]
pub struct ForceAvx2Guard {
    prev_override: u8,
}

impl ForceAvx2Guard {
    /// Creates a new guard, forcing [`InstructionSet::Avx2`] SIMD dispatch.
    pub fn new() -> Self {
        let prev =
            TEST_ISA_OVERRIDE.swap(encode_isa_override(InstructionSet::Avx2), Ordering::SeqCst);
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
    /// Creates a new guard, forcing [`InstructionSet::Avx512`] SIMD dispatch.
    pub fn new() -> Self {
        let prev = TEST_ISA_OVERRIDE.swap(
            encode_isa_override(InstructionSet::Avx512),
            Ordering::SeqCst,
        );
        Self {
            prev_override: prev,
        }
    }
}

#[cfg(feature = "avx512")]
impl Default for ForceAvx512Guard {
    fn default() -> Self {
        Self::new()
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
    /// Creates a new guard, forcing the specified ISA for SIMD dispatch.
    pub fn set(isa: InstructionSet) -> Self {
        let prev = TEST_ISA_OVERRIDE.swap(encode_isa_override(isa), Ordering::SeqCst);
        Self {
            prev_override: prev,
        }
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
    use crate::math::common::effective_instruction_set;
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
            let _guard = IsaGuard::set(InstructionSet::Avx2);
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
    fn test_force_avx512_guard_lifecycle() {
        let initial_isa = effective_instruction_set();
        {
            let _guard = ForceAvx512Guard::new();
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx512,
                "effective_instruction_set() must return InstructionSet::Avx512 while guard is active"
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
    fn test_generic_isa_guard_avx512_lifecycle() {
        let initial_isa = effective_instruction_set();
        {
            let _guard = IsaGuard::set(InstructionSet::Avx512);
            assert_eq!(
                effective_instruction_set(),
                InstructionSet::Avx512,
                "effective_instruction_set() must return InstructionSet::Avx512 while guard is active"
            );
        }
        assert_eq!(
            effective_instruction_set(),
            initial_isa,
            "effective_instruction_set() must revert to initial state after guard drops"
        );
    }
}
