// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#![warn(clippy::undocumented_unsafe_blocks)]

//! Common foundation for mathematical operations and SIMD.
//!
//! This module contains the structural definitions that allow NAM-rs to be
//! hardware-agnostic while maintaining native performance.
//!
//! # Components
//! - `traits`: The `SimdMath` trait defining the interface for all kernels.
//! - `dispatch`: Dynamic architecture selection mechanism (AVX2, AVX-512, etc.).
//! - `avx2_impl` / `avx512`: Concrete kernel implementations for x86-64.
//! - `scalar_ref`: Fallback implementations for compatibility and testing.
//! - `aligned`: Structures to guarantee memory alignment (RT-Safety).

pub mod aligned;
pub mod avx2_impl;
pub mod half;
pub mod huge_alloc;
pub use huge_alloc::HugePageVec;
pub mod avx512;
/// Unit tests for the common math infrastructure.
#[cfg(test)]
pub mod common_test;
pub mod dispatch;
pub mod kahan;
pub mod ops;
pub mod scalar_ref;
pub mod traits;
pub mod utility;

pub use aligned::Aligned64;
pub use aligned::AlignedVec;
pub use avx2_impl::Avx2Math;
pub use avx512::{Avx512Math, Avx512VnniBf16Math};
pub use dispatch::{InstructionSet, SIMD_MATH, SimdMathConfig, TEST_ISA_OVERRIDE};
pub use dispatch::{decode_isa_override, effective_instruction_set, encode_isa_override};
pub use kahan::{Kahan4F32, KahanF32, kahan_add};
pub use ops::*;
pub use scalar_ref::*;
pub use traits::SimdMath;
pub use utility::*;

/// Macro for dynamic SIMD dispatch based on global configuration.
///
/// Checks `TEST_ISA_OVERRIDE` (u8::MAX = disabled) before consulting
/// `SIMD_MATH.instruction_set`. This allows integration tests to force a
/// specific ISA path (AVX2, AVX-512, VNNI+BF16) and measure cross-ISA
/// determinism.
#[macro_export]
macro_rules! dispatch_simd {
    // Mode 2: Dispatch to specific methods of an object (e.g.: lstm.rs)
    (@ $target:expr, $m512bf16:ident, $m512:ident, $m256:ident $(, $arg:expr)*) => {
        {
            use $crate::math::common::InstructionSet;
            let __isa = $crate::math::common::effective_instruction_set();
            match __isa {
                InstructionSet::Avx512VnniBf16 => $target.$m512bf16($($arg),*),
                InstructionSet::Avx512 => $target.$m512($($arg),*),
                InstructionSet::Avx2 => $target.$m256($($arg),*),
            }
        }
    };

    // Mode 1: Dispatch to generic method of an object (e.g.: wavenet.rs)
    ($target:expr, $method:ident $(, $arg:expr)*) => {
        {
            use $crate::math::common::InstructionSet;
            let __isa = $crate::math::common::effective_instruction_set();
            match __isa {
                InstructionSet::Avx512VnniBf16 => {
                    $target.$method::<$crate::math::common::Avx512VnniBf16Math>($($arg),*)
                }
                InstructionSet::Avx512 => {
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
            #[allow(clippy::macro_metavars_in_unsafe, clippy::allow_attributes)]
            unsafe {
                match __isa {
                    InstructionSet::Avx512VnniBf16 => {
                        $crate::math::common::Avx512VnniBf16Math::$method($($arg),*)
                    }
                    InstructionSet::Avx512 => {
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

pub use dispatch_simd;
