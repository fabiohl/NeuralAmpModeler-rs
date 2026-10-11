// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 32-bit integer and cast intrinsics (`__m256i` / `__m128i`).

use super::{__m128i, __m256, __m256i, i32x4_from_array};
use core::arch::aarch64::*;

#[inline(always)]
pub fn _mm256_set1_epi32(a: i32) -> __m256i {
    let v = unsafe { vdupq_n_s32(a) };
    __m256i(v, v)
}

/// Lanes in memory order: `e0` is lane 0.
#[inline(always)]
pub fn _mm256_setr_epi32(
    e0: i32,
    e1: i32,
    e2: i32,
    e3: i32,
    e4: i32,
    e5: i32,
    e6: i32,
    e7: i32,
) -> __m256i {
    __m256i(
        i32x4_from_array([e0, e1, e2, e3]),
        i32x4_from_array([e4, e5, e6, e7]),
    )
}

#[inline(always)]
pub fn _mm256_add_epi32(a: __m256i, b: __m256i) -> __m256i {
    unsafe { __m256i(vaddq_s32(a.0, b.0), vaddq_s32(a.1, b.1)) }
}

/// Logical left shift of each lane; counts above 31 produce zero.
#[inline(always)]
pub fn _mm256_slli_epi32(a: __m256i, imm8: i32) -> __m256i {
    if !(0..=31).contains(&imm8) {
        return _mm256_set1_epi32(0);
    }
    unsafe {
        let count = vdupq_n_s32(imm8);
        __m256i(vshlq_s32(a.0, count), vshlq_s32(a.1, count))
    }
}

/// Logical right shift of each lane; counts above 31 produce zero.
#[inline(always)]
pub fn _mm256_srli_epi32(a: __m256i, imm8: i32) -> __m256i {
    if !(0..=31).contains(&imm8) {
        return _mm256_set1_epi32(0);
    }
    unsafe {
        let count = vdupq_n_s32(-imm8);
        let shr = |v: int32x4_t| vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v), count));
        __m256i(shr(a.0), shr(a.1))
    }
}

/// Per 128-bit half: `[sat_u16(a0..a3), sat_u16(b0..b3)]` (signed i32 to
/// unsigned u16 with saturation).
#[inline(always)]
pub fn _mm256_packus_epi32(a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        let pack = |x: int32x4_t, y: int32x4_t| {
            vreinterpretq_s32_u16(vcombine_u16(vqmovun_s32(x), vqmovun_s32(y)))
        };
        __m256i(pack(a.0, b.0), pack(a.1, b.1))
    }
}

/// Selects 64-bit quarters: result quarter `i` = `a` quarter `imm8[2i+1:2i]`.
#[inline(always)]
pub fn _mm256_permute4x64_epi64(a: __m256i, imm8: i32) -> __m256i {
    let q: [u64; 4] = unsafe { core::mem::transmute(a) };
    let pick = |i: i32| q[((imm8 >> (2 * i)) & 0x3) as usize];
    let r = [pick(0), pick(1), pick(2), pick(3)];
    unsafe { core::mem::transmute::<[u64; 4], __m256i>(r) }
}

#[inline(always)]
pub fn _mm256_castsi256_ps(a: __m256i) -> __m256 {
    unsafe { __m256(vreinterpretq_f32_s32(a.0), vreinterpretq_f32_s32(a.1)) }
}

#[inline(always)]
pub fn _mm256_castps_si256(a: __m256) -> __m256i {
    unsafe { __m256i(vreinterpretq_s32_f32(a.0), vreinterpretq_s32_f32(a.1)) }
}

#[inline(always)]
pub fn _mm256_castsi256_si128(a: __m256i) -> __m128i {
    a.0
}

#[inline(always)]
pub fn _mm_cvtsi32_si128(a: i32) -> __m128i {
    i32x4_from_array([a, 0, 0, 0])
}

#[inline(always)]
pub fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
    unsafe { vgetq_lane_s32::<0>(a) }
}
