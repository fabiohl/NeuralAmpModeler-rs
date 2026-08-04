// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic dispatch system for SIMD kernels.

/// Static `SimdMathConfig` struct and its methods.
pub mod config;
/// CPU feature detection and global `SIMD_MATH` initialization.
pub mod detect;
/// `InstructionSet` enum for supported x86-64 ISA variants.
pub mod instruction_set;

pub use config::SimdMathConfig;
pub use detect::SIMD_MATH;
pub use detect::TEST_ISA_OVERRIDE;
pub use detect::{decode_isa_override, effective_instruction_set, encode_isa_override};
pub use instruction_set::InstructionSet;

/// Generates a `SimdMathConfig` with static descriptive fields only.
///
/// # Parameters
/// - `$isa`: The `InstructionSet` variant.
/// - `$name`: Friendly backend name.
/// - `$avx512`: Whether the backend is AVX-512 (`true`/`false`).
#[macro_export]
macro_rules! config_table {
    ($isa:expr, $name:expr, $avx512:expr) => {
        $crate::math::common::dispatch::config::SimdMathConfig {
            instruction_set: $isa,
            name: $name,
            is_avx512: $avx512,
        }
    };
}
