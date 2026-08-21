// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::detect::{SIMD_MATH, TEST_ISA_OVERRIDE, decode_isa_override};
use super::instruction_set::InstructionSet;
use std::sync::atomic::Ordering;

/// Static SIMD configuration — descriptive data only, no v-table.
///
/// All dispatch is handled via the `dispatch_simd!` macro, which matches
/// on `instruction_set` and monomorphizes calls via the `SimdMath` trait
/// (e.g. `<Avx2Math>::apply_gain(...)`). Function pointers have been removed;
/// all consumers use trait-based static dispatch.
#[derive(Clone, Copy)]
pub struct SimdMathConfig {
    /// Active instruction set.
    pub instruction_set: InstructionSet,
    /// Friendly backend name.
    pub name: &'static str,
    /// Whether the backend is AVX-512.
    pub is_avx512: bool,
}

impl SimdMathConfig {
    /// Returns the active global SIMD configuration, respecting any
    /// `TEST_ISA_OVERRIDE` that may be set by integration tests.
    #[expect(deprecated)]
    pub fn current() -> Self {
        let raw = TEST_ISA_OVERRIDE.load(Ordering::Relaxed);
        if let Some(isa) = decode_isa_override(raw) {
            Self {
                instruction_set: isa,
                name: match isa {
                    InstructionSet::Avx2 => "AVX2 (overridden)",
                    InstructionSet::Avx512 => "AVX-512 (overridden)",
                    InstructionSet::Avx512VnniBf16 => "AVX-512 VNNI+BF16 (overridden)",
                },
                is_avx512: matches!(isa, InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16),
            }
        } else {
            *SIMD_MATH
        }
    }

    /// Alias for current().
    pub fn get() -> Self {
        Self::current()
    }
}
