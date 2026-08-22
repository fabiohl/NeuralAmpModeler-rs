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

/// Encodes an `InstructionSet` enum into the corresponding test override raw byte.
#[inline]
#[cfg_attr(feature = "avx512", expect(deprecated))]
pub const fn encode_isa_override(isa: InstructionSet) -> u8 {
    match isa {
        InstructionSet::Avx2 => 0,
        #[cfg(feature = "avx512")]
        InstructionSet::Avx512 => 1,
        #[cfg(feature = "avx512")]
        InstructionSet::Avx512VnniBf16 => 2,
        #[cfg(not(feature = "avx512"))]
        _ => 0,
    }
}

/// Decodes a raw test override byte into an `InstructionSet` enum.
///
/// Returns `None` if the byte does not correspond to a valid ISA enum variant.
#[inline]
#[cfg_attr(feature = "avx512", expect(deprecated))]
pub const fn decode_isa_override(raw: u8) -> Option<InstructionSet> {
    match raw {
        0 => Some(InstructionSet::Avx2),
        #[cfg(feature = "avx512")]
        1 => Some(InstructionSet::Avx512),
        #[cfg(feature = "avx512")]
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
/// AVX-512 is selected only when both `avx512f` and `avx512vl` are supported by the processor and the `avx512` Cargo feature is enabled.
/// Production runtime detection never emits `InstructionSet::Avx512VnniBf16`.
fn detect_best_simd() -> SimdMathConfig {
    #[cfg(feature = "avx512")]
    if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512vl") {
        return config_table!(InstructionSet::Avx512, "AVX-512", true);
    }
    config_table!(InstructionSet::Avx2, "AVX2", false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_instruction_set_default_avx2() {
        #[cfg(not(feature = "avx512"))]
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
    fn test_override_ignored_without_avx512_feature() {
        #[cfg(not(feature = "avx512"))]
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
    #[cfg(feature = "avx512")]
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
}
