// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// Enumerates the supported instruction sets in the public API.
///
/// Note: There is no scalar `Fallback` variant in this enum. The crate strictly
/// enforces `x86-64-v3` (AVX2 + FMA + BMI1/2) as its mandatory compile-time
/// baseline (`compile_error!` in `src/lib.rs` and `.cargo/config.toml`).
///
/// In standard production builds (without the opt-in `avx512` feature enabled),
/// [`InstructionSet::Avx2`] is the sole backend compiled, detected, and dispatched.
/// The [`InstructionSet::Avx512`] and [`InstructionSet::Avx512VnniBf16`] variants are
/// maintained in the stable public API for consumer neutrality, but are only selected
/// when the crate is explicitly compiled with the `avx512` research/evaluation feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
pub enum InstructionSet {
    /// AVX2 + FMA (x86-64-v3 baseline, standard production backend).
    Avx2,
    /// AVX-512 Foundation + Vector Length Extensions (Skylake-X+, Zen 4+).
    ///
    /// Only detected and dispatched when the `avx512` feature is explicitly enabled.
    Avx512,
    /// AVX-512 VNNI + BF16 (legacy / evaluation-only; runtime detection uses standard AVX-512).
    #[deprecated(
        note = "Avx512VnniBf16 is legacy/evaluation-only. Production uses Avx512 f32 math."
    )]
    Avx512VnniBf16,
}
