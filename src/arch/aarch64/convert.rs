// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Half-precision (F16C) and float-to-integer conversions.

use super::{__m128, __m128i, __m256, __m256i};
use core::arch::aarch64::*;

/// Eight IEEE binary16 values in `a` → eight `f32` (exact).
#[inline(always)]
pub fn _mm256_cvtph_ps(a: __m128i) -> __m256 {
    unsafe {
        let h = vreinterpretq_u16_s32(a);
        __m256(
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))),
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))),
        )
    }
}

/// The low four binary16 values in `a` → four `f32` (exact).
#[inline(always)]
pub fn _mm_cvtph_ps(a: __m128i) -> __m128 {
    unsafe { vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(vreinterpretq_u16_s32(a)))) }
}

/// Four `f32` → four binary16 in the low 64 bits (high 64 bits zero).
///
/// Rounds with the current FPCR mode, round-to-nearest-even unless changed,
/// which matches `_MM_FROUND_TO_NEAREST_INT` and `_MM_FROUND_CUR_DIRECTION`
/// (the only modes the kernels use).
#[inline(always)]
pub fn _mm_cvtps_ph(a: __m128, rounding: i32) -> __m128i {
    debug_assert!(
        rounding & 0x3 == super::_MM_FROUND_TO_NEAREST_INT
            || rounding & super::_MM_FROUND_CUR_DIRECTION != 0,
        "_mm_cvtps_ph: only nearest/current rounding is implemented"
    );
    unsafe {
        let h = vreinterpret_u16_f16(vcvt_f16_f32(a));
        vreinterpretq_s32_u16(vcombine_u16(h, vdup_n_u16(0)))
    }
}

/// Rounds to nearest-even `i32`. Out-of-range inputs saturate (x86 returns
/// `i32::MIN`); NaN converts to 0 (x86 returns `i32::MIN`).
#[inline(always)]
pub fn _mm256_cvtps_epi32(a: __m256) -> __m256i {
    unsafe { __m256i(vcvtnq_s32_f32(a.0), vcvtnq_s32_f32(a.1)) }
}
