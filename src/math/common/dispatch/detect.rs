// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::config::SimdMathConfig;
use super::instruction_set::InstructionSet;
use crate::config_table;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU8, Ordering};

/// Global SIMD configuration instance, detected at system boot.
///
/// Using `LazyLock` ensures the user's CPU is inspected only once,
/// at the moment the first DSP mathematical operation is invoked. After that, the
/// corresponding SIMD configuration struct (with instruction set and name)
/// is cached in memory and accessed immediately (no real-time checking cost).
pub static SIMD_MATH: LazyLock<SimdMathConfig> = LazyLock::new(detect_best_simd);

/// Test-only ISA override.
///
/// When set to a value other than `u8::MAX`, the `dispatch_simd!` macro uses
/// this field instead of `SIMD_MATH.instruction_set` to select the SIMD path.
/// This allows integration tests to force a specific ISA (e.g. AVX2 vs AVX-512)
/// and measure cross-ISA determinism / error floor.
///
/// Encoding: 0 = AVX2, 1 = AVX-512, 2 = AVX-512 VNNI+BF16, u8::MAX = disabled.
///
/// # Safety
/// This is a `#[doc(hidden)]` test-only facility. Setting an ISA that the host
/// CPU does not support will cause `SIGILL` (illegal instruction).
#[doc(hidden)]
pub static TEST_ISA_OVERRIDE: AtomicU8 = AtomicU8::new(u8::MAX);

/// Encodes an [`InstructionSet`] to its `TEST_ISA_OVERRIDE` byte value.
#[doc(hidden)]
#[inline]
pub fn encode_isa_override(isa: InstructionSet) -> u8 {
    match isa {
        InstructionSet::Avx2 => 0,
        InstructionSet::Avx512 => 1,
        InstructionSet::Avx512VnniBf16 => 2,
    }
}

/// Decodes a `TEST_ISA_OVERRIDE` byte into an [`InstructionSet`], or `None`
/// when the override is disabled.
#[doc(hidden)]
#[inline]
pub fn decode_isa_override(raw: u8) -> Option<InstructionSet> {
    match raw {
        0 => Some(InstructionSet::Avx2),
        1 => Some(InstructionSet::Avx512),
        2 => Some(InstructionSet::Avx512VnniBf16),
        _ => None,
    }
}

/// Returns the effective [`InstructionSet`] to use for dispatch, respecting
/// the `TEST_ISA_OVERRIDE` if set.
#[doc(hidden)]
#[inline]
pub fn effective_instruction_set() -> InstructionSet {
    let raw = TEST_ISA_OVERRIDE.load(Ordering::Relaxed);
    if let Some(isa) = decode_isa_override(raw) {
        isa
    } else {
        SIMD_MATH.instruction_set
    }
}

/// Inspects the CPU hardware capabilities at runtime and returns the best
/// compatible SIMD configuration.
///
/// Detection checks supported hardware features using the compiler macro `is_x86_feature_detected!`.
/// We follow a descending priority order, choosing the most advanced instruction set
/// available on the processor where the software is running.
fn detect_best_simd() -> SimdMathConfig {
    if is_x86_feature_detected!("avx512bf16") && is_x86_feature_detected!("avx512vnni") {
        return config_table!(InstructionSet::Avx512VnniBf16, "AVX-512 (VNNI+BF16)", true);
    }
    if is_x86_feature_detected!("avx512f") {
        return config_table!(InstructionSet::Avx512, "AVX-512", true);
    }
    config_table!(InstructionSet::Avx2, "AVX2", false)
}
