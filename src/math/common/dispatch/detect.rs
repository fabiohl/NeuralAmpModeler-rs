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
/// This is a `#[doc(hidden)]` test-only facility. Writing an ISA that the host
/// CPU does not support will cause `SIGILL` (illegal instruction) as soon as a
/// kernel is dispatched. All *install* operations MUST go through the validated
/// Helpers [`set_test_isa_override`] / [`clear_test_isa_override`]; the
/// static is `pub(crate)` so external crates cannot bypass capability checks.
/// Restores of a previously validated raw byte (from the `Ok` value of
/// [`set_test_isa_override`]) are the only sanctioned raw stores.
#[doc(hidden)]
pub(crate) static TEST_ISA_OVERRIDE: AtomicU8 = AtomicU8::new(u8::MAX);

/// Encodes an `InstructionSet` enum into the corresponding test override raw byte.
#[inline]
#[cfg_attr(nam_avx512, expect(deprecated))]
pub const fn encode_isa_override(isa: InstructionSet) -> u8 {
    match isa {
        InstructionSet::Avx2 => 0,
        #[cfg(nam_avx512)]
        InstructionSet::Avx512 => 1,
        #[cfg(nam_avx512)]
        InstructionSet::Avx512VnniBf16 => 2,
        #[cfg(not(nam_avx512))]
        _ => 0,
    }
}

/// Decodes a raw test override byte into an `InstructionSet` enum.
///
/// Returns `None` if the byte does not correspond to a valid ISA enum variant.
#[inline]
#[cfg_attr(nam_avx512, expect(deprecated))]
pub const fn decode_isa_override(raw: u8) -> Option<InstructionSet> {
    match raw {
        0 => Some(InstructionSet::Avx2),
        #[cfg(nam_avx512)]
        1 => Some(InstructionSet::Avx512),
        #[cfg(nam_avx512)]
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

/// Error returned when a test ISA override cannot be installed because the
/// host CPU (or the active feature set) lacks the required capability set.
///
/// Produced by `set_test_isa_override` and the fallible test guards
/// (`testing::isa_guard::ForceAvx512Guard::try_new`,
/// `testing::isa_guard::IsaGuard::try_set`). Safe Rust must never force
/// dispatch towards instructions the host cannot execute (F-ROB-03).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IsaOverrideError {
    /// The requested ISA requires CPU sub-features that are absent on the
    /// current host, or the `avx512` Cargo feature is not enabled.
    #[error("cannot force ISA dispatch to {isa:?}: host/feature set is missing: {missing}")]
    UnsupportedIsa {
        /// The ISA that was requested for the override.
        isa: InstructionSet,
        /// Human-readable, comma-joined list of missing capabilities
        /// (e.g. `avx512bw, avx512dq`) or `avx512 feature not enabled`.
        missing: String,
    },
}

/// `true` when the complete AVX-512 capability matrix (`F + VL + BW + DQ`)
/// is present. This is the pure decision function behind runtime detection
/// And the validated overrides; unit-testable without hardware.
///
/// The reachable AVX-512 kernels require the full set: `avx512f` (foundation),
/// `avx512vl` (VL256/xmm-ymm EVEX lowering), `avx512bw` (byte/word) and
/// `avx512dq` (doubleword/quadword / 32×8 inserts-extracts). Selecting
/// `InstructionSet::Avx512` on a CPU with a partial subset risks `SIGILL`.
pub const fn avx512_capability_complete(f: bool, vl: bool, bw: bool, dq: bool) -> bool {
    f && vl && bw && dq
}

/// Returns `true` when the current host CPU supports the full AVX-512
/// capability set required by the reachable kernels (F+VL+BW+DQ).
///
/// # Production Usage Notice
///
/// **AVX-512 is discouraged in production.** See `docs/audio_fidelity_map.md` and `docs/architecture.md`.
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub fn has_full_avx512() -> bool {
    avx512_capability_complete(
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512vl"),
        is_x86_feature_detected!("avx512bw"),
        is_x86_feature_detected!("avx512dq"),
    )
}

/// Names the AVX-512 sub-features missing on the current host (F/VL/BW/DQ).
///
/// Returns an empty slice when the full capability set is present. Used to
/// build the structured error of [`IsaOverrideError::UnsupportedIsa`].
///
/// # Production Usage Notice
///
/// **AVX-512 is discouraged in production.** See `docs/audio_fidelity_map.md` and `docs/architecture.md`.
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
pub fn missing_avx512_features() -> Vec<&'static str> {
    let mut missing = Vec::with_capacity(4);
    for (name, ok) in [
        ("avx512f", is_x86_feature_detected!("avx512f")),
        ("avx512vl", is_x86_feature_detected!("avx512vl")),
        ("avx512bw", is_x86_feature_detected!("avx512bw")),
        ("avx512dq", is_x86_feature_detected!("avx512dq")),
    ] {
        if !ok {
            missing.push(name);
        }
    }
    missing
}

/// Installs a validated test ISA override.
///
/// This is the single sanctioned *install* path for `TEST_ISA_OVERRIDE`
/// The requested ISA is validated against the host CPU capabilities
/// (and the active Cargo features) BEFORE the atomic is written, so safe Rust
/// can never force dispatch towards instructions that would `SIGILL`.
///
/// - [`InstructionSet::Avx2`] is always accepted (x86-64-v3 baseline).
/// - [`InstructionSet::Avx512`] requires the full `F+VL+BW+DQ` matrix and the
///   `avx512` Cargo feature.
/// - [`InstructionSet::Avx512VnniBf16`] additionally requires `avx512bf16`
///   and `avx512vnni`.
///
/// Returns the previous raw override byte (for restore semantics); callers
/// may restore it via [`clear_test_isa_override`] or a raw store of the
/// returned byte.
#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
#[expect(deprecated)]
pub fn set_test_isa_override(isa: InstructionSet) -> Result<u8, IsaOverrideError> {
    match isa {
        InstructionSet::Avx2 => {}
        #[cfg(nam_avx512)]
        InstructionSet::Avx512 => {
            if !has_full_avx512() {
                return Err(IsaOverrideError::UnsupportedIsa {
                    isa,
                    missing: missing_avx512_features().join(", "),
                });
            }
        }
        #[cfg(nam_avx512)]
        InstructionSet::Avx512VnniBf16 => {
            let mut missing = missing_avx512_features();
            if !is_x86_feature_detected!("avx512bf16") {
                missing.push("avx512bf16");
            }
            if !is_x86_feature_detected!("avx512vnni") {
                missing.push("avx512vnni");
            }
            if !missing.is_empty() {
                return Err(IsaOverrideError::UnsupportedIsa {
                    isa,
                    missing: missing.join(", "),
                });
            }
        }
        #[cfg(not(nam_avx512))]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            return Err(IsaOverrideError::UnsupportedIsa {
                isa,
                missing:
                    "avx512 feature not enabled (encode/decode would silently fall back to AVX2)"
                        .to_string(),
            });
        }
    }
    let prev = TEST_ISA_OVERRIDE.swap(encode_isa_override(isa), Ordering::SeqCst);
    Ok(prev)
}

/// Clears the test ISA override (returns to the auto-detected `SIMD_MATH`).
///
/// Returns the previous raw override byte (usually the one produced by a
/// prior [`set_test_isa_override`] call).
#[cfg(any(test, feature = "testing"))]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
#[inline]
pub fn clear_test_isa_override() -> u8 {
    TEST_ISA_OVERRIDE.swap(u8::MAX, Ordering::SeqCst)
}

/// Inspects the CPU hardware capabilities at runtime and returns the best
/// compatible SIMD configuration.
///
/// Detection checks supported hardware features using the compiler macro `is_x86_feature_detected!`.
/// AVX-512 is selected only when the full `F+VL+BW+DQ` capability matrix
/// (`avx512f` + `avx512vl` + `avx512bw` + `avx512dq`) is supported by the
/// Processor and the `avx512` Cargo feature is enabled. A partial
/// subset (e.g. F+VL without BW/DQ) deterministically falls back to
/// [`InstructionSet::Avx2`] — the reachable kernels require byte/word and
/// doubleword/quadword instructions that a partial subset cannot execute.
/// Production runtime detection never emits `InstructionSet::Avx512VnniBf16`.
fn detect_best_simd() -> SimdMathConfig {
    #[cfg(nam_avx512)]
    if has_full_avx512() {
        return config_table!(InstructionSet::Avx512, "AVX-512", true);
    }
    #[cfg(target_arch = "aarch64")]
    return config_table!(InstructionSet::Avx2, "NEON (AVX2-compatible layer)", false);
    #[cfg(not(target_arch = "aarch64"))]
    config_table!(InstructionSet::Avx2, "AVX2", false)
}

#[cfg(test)]
#[path = "detect_test.rs"]
mod tests;
