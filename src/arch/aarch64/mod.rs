// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! NEON implementation of the x86 AVX2/FMA/F16C intrinsic surface used by the
//! kernels.
//!
//! `__m256` is a pair of `float32x4_t` (lanes 0-3, lanes 4-7) and `__m128` is
//! `float32x4_t` itself, so every 256-bit operation becomes two 128-bit NEON
//! operations with the same per-lane semantics and order. Fused multiply-add
//! maps to `vfmaq_f32`/`vfmsq_f32` (single rounding, like x86 FMA), and
//! `max`/`min` reproduce the x86 operand-selection rule (`a > b ? a : b`)
//! instead of NEON's NaN-propagating `fmax`, so arithmetic kernels produce the
//! same per-lane results as the AVX2 path.
//!
//! Known numeric differences from x86 (documented at each function):
//! - `_mm256_rcp_ps`: NEON estimate + one Newton step (~2^-16 relative
//!   error) vs. x86 `rcpps` (~2^-12). Both are approximations; the bits differ.
//! - `_mm256_cvtps_epi32`: out-of-range inputs saturate instead of producing
//!   the x86 "integer indefinite" value `i32::MIN`.
//!
//! Safety split mirrors `core::arch::x86_64` (Rust 1.99): intrinsics that take
//! a raw pointer or reference are `unsafe fn`; all others are safe. Intrinsics
//! the kernels call with an immediate argument (`_mm256_cmp_ps(a, b, IMM)`)
//! take it as a plain `i32`; they are `#[inline(always)]`, so it folds to a
//! constant at each call site and the selection logic compiles away.

use core::arch::aarch64::*;

mod convert;
mod float;
mod int;
mod mem;

pub use convert::*;
pub use float::*;
pub use int::*;
pub use mem::*;

/// 256-bit vector of eight `f32`, as two NEON registers (lanes 0-3, 4-7).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct __m256(pub float32x4_t, pub float32x4_t);

/// 256-bit integer vector, as two NEON registers (lanes 0-3, 4-7 as `i32`).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct __m256i(pub int32x4_t, pub int32x4_t);

/// 128-bit vector of four `f32`.
#[expect(non_camel_case_types, reason = "mirrors core::arch::x86_64::__m128")]
pub type __m128 = float32x4_t;

/// 128-bit integer vector (bit container, viewed as four `i32`).
#[expect(non_camel_case_types, reason = "mirrors core::arch::x86_64::__m128i")]
pub type __m128i = int32x4_t;

// ── Comparison predicates (`_mm256_cmp_ps`), same values as x86 ─────────────
pub const _CMP_EQ_OQ: i32 = 0x00;
pub const _CMP_LT_OS: i32 = 0x01;
pub const _CMP_LE_OS: i32 = 0x02;
pub const _CMP_UNORD_Q: i32 = 0x03;
pub const _CMP_NEQ_UQ: i32 = 0x04;
pub const _CMP_NLT_US: i32 = 0x05;
pub const _CMP_NLE_US: i32 = 0x06;
pub const _CMP_ORD_Q: i32 = 0x07;
pub const _CMP_EQ_UQ: i32 = 0x08;
pub const _CMP_NGE_US: i32 = 0x09;
pub const _CMP_NGT_US: i32 = 0x0a;
pub const _CMP_FALSE_OQ: i32 = 0x0b;
pub const _CMP_NEQ_OQ: i32 = 0x0c;
pub const _CMP_GE_OS: i32 = 0x0d;
pub const _CMP_GT_OS: i32 = 0x0e;
pub const _CMP_TRUE_UQ: i32 = 0x0f;
pub const _CMP_EQ_OS: i32 = 0x10;
pub const _CMP_LT_OQ: i32 = 0x11;
pub const _CMP_LE_OQ: i32 = 0x12;
pub const _CMP_UNORD_S: i32 = 0x13;
pub const _CMP_NEQ_US: i32 = 0x14;
pub const _CMP_NLT_UQ: i32 = 0x15;
pub const _CMP_NLE_UQ: i32 = 0x16;
pub const _CMP_ORD_S: i32 = 0x17;
pub const _CMP_EQ_US: i32 = 0x18;
pub const _CMP_NGE_UQ: i32 = 0x19;
pub const _CMP_NGT_UQ: i32 = 0x1a;
pub const _CMP_FALSE_OS: i32 = 0x1b;
pub const _CMP_NEQ_OS: i32 = 0x1c;
pub const _CMP_GE_OQ: i32 = 0x1d;
pub const _CMP_GT_OQ: i32 = 0x1e;
pub const _CMP_TRUE_US: i32 = 0x1f;

// ── Rounding control (`_mm256_round_ps`, `_mm_cvtps_ph`) ────────────────────
pub const _MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
pub const _MM_FROUND_TO_NEG_INF: i32 = 0x01;
pub const _MM_FROUND_TO_POS_INF: i32 = 0x02;
pub const _MM_FROUND_TO_ZERO: i32 = 0x03;
pub const _MM_FROUND_CUR_DIRECTION: i32 = 0x04;
pub const _MM_FROUND_NO_EXC: i32 = 0x08;

// ── Prefetch hints (`_mm_prefetch`) ─────────────────────────────────────────
pub const _MM_HINT_T0: i32 = 3;
pub const _MM_HINT_T1: i32 = 2;
pub const _MM_HINT_T2: i32 = 1;
pub const _MM_HINT_NTA: i32 = 0;

#[inline]
#[target_feature(enable = "neon")]
fn f32x4_to_array(v: float32x4_t) -> [f32; 4] {
    // SAFETY: `float32x4_t` and `[f32; 4]` have the same size and lane order.
    unsafe { core::mem::transmute(v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn f32x4_from_array(a: [f32; 4]) -> float32x4_t {
    // SAFETY: `a` is a live local array of 4 `f32`, valid for the 16-byte read.
    unsafe { vld1q_f32(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn i32x4_to_array(v: int32x4_t) -> [i32; 4] {
    // SAFETY: `int32x4_t` and `[i32; 4]` have the same size and lane order.
    unsafe { core::mem::transmute(v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn i32x4_from_array(a: [i32; 4]) -> int32x4_t {
    // SAFETY: `a` is a live local array of 4 `i32`, valid for the 16-byte read.
    unsafe { vld1q_s32(a.as_ptr()) }
}

#[cfg(test)]
#[path = "aarch64_test.rs"]
mod tests;
