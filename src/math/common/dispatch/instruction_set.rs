// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Enumerates the supported instruction sets.
///
/// Note: There is no scalar `Fallback` variant in this enum. The project targets
/// x86-64-v3 (AVX2+FMA) as mandatory. If AVX2 is not detected,
/// `detect_best_simd()` panics at boot (fail-fast).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum InstructionSet {
    /// AVX2 + FMA (x86-64-v3).
    Avx2,
    /// AVX-512 Foundation (Skylake-X+, Zen 4+).
    Avx512,
    /// AVX-512 VNNI + BF16.
    Avx512VnniBf16,
}
