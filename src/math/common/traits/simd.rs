// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD register, vector, and basic operation abstractions.

use super::super::InstructionSet;

/// Abstraction trait for SIMD register types (e.g., `__m256`, `__m512`).
pub trait SimdRegister: Copy {
    /// ISA classification associated with this register type.
    const ISA: InstructionSet;
}

/// Vector abstraction trait for SIMD math containers.
pub trait SimdVector {
    /// Element scalar type.
    type Element;
    /// Register representation type.
    type Register: SimdRegister;
}

/// Floating-point SIMD vector trait.
pub trait SimdFloat: SimdVector {
    /// Indicates whether this vector precision represents BF16.
    const IS_BF16: bool = false;
}

/// Basic SIMD operation abstractions.
pub trait SimdOps: SimdFloat {
    /// Store register contents as packed BF16 values to memory.
    ///
    /// # Safety
    /// `ptr` must point to allocated memory with sufficient capacity for the register.
    unsafe fn store_bf16(ptr: *mut u16, v: Self::Register);
}
