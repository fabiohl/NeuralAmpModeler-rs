// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RAII Guard for forcing AVX2 instruction set dispatch in testing & benchmarking.

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::common::effective_instruction_set;

    #[test]
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
}
