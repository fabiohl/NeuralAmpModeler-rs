// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#![warn(clippy::undocumented_unsafe_blocks)]

//! Common foundation for mathematical operations and SIMD.
//!
//! This module contains the structural definitions that allow NeuralAmpModeler-rs to be
//! hardware-agnostic while maintaining native performance.
//!
//! # Components
//! - `traits`: The `SimdMath` trait defining the interface for all kernels.
//! - `dispatch`: Dynamic architecture selection mechanism (AVX2, AVX-512, etc.).
//! - `avx2_impl` / `avx512`: Concrete kernel implementations for x86-64.
//! - `scalar_ref`: Fallback implementations for compatibility and testing.
//! - `aligned`: Structures to guarantee memory alignment (RT-Safety).

/// 64-byte aligned memory primitives: `Aligned64` newtype and `AlignedVec`.
pub mod aligned;
/// AVX2 kernel implementations: activations, DSP, GEMV, reductions, BF16.
pub mod avx2_impl;
/// Half-precision (f16/bf16) conversion and helper types.
pub mod half;
/// Huge-page allocation support: `HugePageVec` for 2 MB TLB-friendly buffers.
pub mod huge_alloc;
pub use huge_alloc::HugePageVec;
/// AVX-512 kernel implementations: activations, DSP/VNNI-BF16, reductions, BF16.
#[cfg(feature = "avx512")]
pub mod avx512;
/// Unit tests for the common math infrastructure.
#[cfg(test)]
pub mod common_test;
/// Instruction set dispatch: runtime CPU feature detection and ISA selection.
pub mod dispatch;
/// Kahan compensated summation: error-bounded floating-point accumulation kernels.
pub mod kahan;
/// Common mathematical operations and SIMD utility functions.
pub mod ops;
/// Scalar reference implementations: portable fallback kernels for testing and oracles.
pub mod scalar_ref;
/// SIMD traits: abstract interfaces for architecture-specific math kernels.
pub mod traits;
/// General-purpose SIMD utility functions (horizontal sums, shuffles, broadcasts).
pub mod utility;

pub use aligned::Aligned64;
pub use aligned::AlignedVec;
pub use avx2_impl::Avx2Math;
#[cfg(feature = "avx512")]
#[expect(deprecated)]
pub use avx512::{Avx512Math, Avx512VnniBf16Math};
pub use dispatch::{InstructionSet, SIMD_MATH, SimdMathConfig, TEST_ISA_OVERRIDE};
pub use dispatch::{decode_isa_override, effective_instruction_set, encode_isa_override};
/// Kahan compensated summation types and accumulator.
pub use kahan::{Kahan4F32, KahanF32, kahan_add};
pub use ops::*;
pub use scalar_ref::*;
pub use traits::SimdMath;
pub use utility::*;

/// Macro for dynamic SIMD dispatch based on global configuration.
///
/// Dispatches execution to the optimal SIMD math implementation ([`Avx2Math`] or
/// [`Avx512Math`]) based on the detected host CPU capabilities or test overrides.
///
/// # Baseline and Dynamic Dispatch Policy
/// - **Baseline (`x86-64-v3`):** The crate compiles unconditionally with AVX2 + FMA + BMI2
///   support. [`Avx2Math`] is the default baseline backend.
/// - **Upward-only Dispatch:** Runtime CPU feature detection (`SIMD_MATH` via `detect_best_simd`)
///   checks for AVX-512 (`avx512f` + `avx512vl`). If supported and the `avx512` Cargo feature is enabled,
///   [`InstructionSet::Avx512`] is active.
/// - **VNNI / BF16 is Not Production:** Production neural inference runs strictly in single-precision
///   `f32`. Runtime detection never returns [`InstructionSet::Avx512VnniBf16`]. Both [`InstructionSet::Avx512`]
///   and [`InstructionSet::Avx512VnniBf16`] fold into [`Avx512Math`] (f32) to eliminate redundant
///   monomorphizations and ensure numerical parity with C++ NAMCore.
///
/// # Modes
/// - **Mode 1 (Generic Method):** `dispatch_simd!(target, method, args...)`
///   Calls `$target.$method::<Avx512Math>(...)` (with feature) or `$target.$method::<Avx2Math>(...)`.
/// - **Mode 2 (Specific Method / Multi-branch):** `dispatch_simd!(@ target, m512bf16, m512, m256, args...)`
///   Calls `$target.$m512(...)` for AVX-512 arms (with feature), and `$target.$m256(...)` for AVX2.
/// - **Mode 3 (Static Associated Function):** `dispatch_simd!(method(args...))`
///   Calls `<Avx512Math as SimdMath>::$method(...)` (with feature) or `<Avx2Math as SimdMath>::$method(...)`.
///
/// # Test Overrides
/// Checks `TEST_ISA_OVERRIDE` via [`effective_instruction_set()`] before consulting
/// `SIMD_MATH.instruction_set`. This allows integration tests (such as `tests/parity/isa_parity.rs`)
/// to deterministically force an ISA path in serial execution.
#[cfg(feature = "avx512")]
#[macro_export]
macro_rules! dispatch_simd {
    // Mode 2: Dispatch to specific methods of an object (e.g.: lstm.rs)
    (@ $target:expr, $m512bf16:ident, $m512:ident, $m256:ident $(, $arg:expr)*) => {
        {
            use $crate::math::common::InstructionSet;
            let __isa = $crate::math::common::effective_instruction_set();
            #[expect(deprecated)]
            match __isa {
                InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => $target.$m512($($arg),*),
                InstructionSet::Avx2 => $target.$m256($($arg),*),
            }
        }
    };

    // Mode 1: Dispatch to generic method of an object (e.g.: wavenet.rs)
    ($target:expr, $method:ident $(, $arg:expr)*) => {
        {
            use $crate::math::common::InstructionSet;
            let __isa = $crate::math::common::effective_instruction_set();
            #[expect(deprecated)]
            match __isa {
                InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                    $target.$method::<$crate::math::common::Avx512Math>($($arg),*)
                }
                InstructionSet::Avx2 => {
                    $target.$method::<$crate::math::common::Avx2Math>($($arg),*)
                }
            }
        }
    };

    // Mode 3: Static dispatch to SimdMath trait associated functions
    ($method:ident ($($arg:expr),*)) => {
        {
            use $crate::math::common::InstructionSet;
            use $crate::math::common::traits::SimdMath;
            let __isa = $crate::math::common::effective_instruction_set();
            // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
            #[allow(clippy::macro_metavars_in_unsafe, clippy::allow_attributes, deprecated)]
            unsafe {
                match __isa {
                    InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                        $crate::math::common::Avx512Math::$method($($arg),*)
                    }
                    InstructionSet::Avx2 => {
                        $crate::math::common::Avx2Math::$method($($arg),*)
                    }
                }
            }
        }
    };
}

/// Macro for dynamic SIMD dispatch based on global configuration (AVX2 baseline).
///
/// In this default configuration (`feature = "avx512"` disabled), all invocations
/// directly resolve to [`Avx2Math`] with zero branching, zero function pointers,
/// and no compilation of AVX-512 symbols.
#[cfg(not(feature = "avx512"))]
#[macro_export]
macro_rules! dispatch_simd {
    // Mode 2: Dispatch to specific methods of an object (e.g.: lstm.rs)
    (@ $target:expr, $m512bf16:ident, $m512:ident, $m256:ident $(, $arg:expr)*) => {
        {
            $target.$m256($($arg),*)
        }
    };

    // Mode 1: Dispatch to generic method of an object (e.g.: wavenet.rs)
    ($target:expr, $method:ident $(, $arg:expr)*) => {
        {
            $target.$method::<$crate::math::common::Avx2Math>($($arg),*)
        }
    };

    // Mode 3: Static dispatch to SimdMath trait associated functions
    ($method:ident ($($arg:expr),*)) => {
        {
            use $crate::math::common::traits::SimdMath;
            // SAFETY: Inner safety guarantees are upheld by caller invariants or the execution environment.
            #[allow(clippy::macro_metavars_in_unsafe, clippy::allow_attributes)]
            unsafe {
                <$crate::math::common::Avx2Math as SimdMath>::$method($($arg),*)
            }
        }
    };
}

pub use dispatch_simd;
