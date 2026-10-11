// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Floating-point arithmetic, comparison, bitwise and lane-movement
//! intrinsics (`__m256` / `__m128`).

use super::{__m128, __m256, f32x4_from_array, f32x4_to_array};
use core::arch::aarch64::*;

/// Applies a 128-bit NEON operation to both halves of a `__m256`.
macro_rules! halves {
    ($f:ident, $a:expr) => {
        __m256($f($a.0), $f($a.1))
    };
    ($f:ident, $a:expr, $b:expr) => {
        __m256($f($a.0, $b.0), $f($a.1, $b.1))
    };
    ($f:ident, $a:expr, $b:expr, $c:expr) => {
        __m256($f($a.0, $b.0, $c.0), $f($a.1, $b.1, $c.1))
    };
}

// ── Construction ─────────────────────────────────────────────────────────────

#[inline(always)]
pub fn _mm256_setzero_ps() -> __m256 {
    let z = unsafe { vdupq_n_f32(0.0) };
    __m256(z, z)
}

#[inline(always)]
pub fn _mm256_set1_ps(a: f32) -> __m256 {
    let v = unsafe { vdupq_n_f32(a) };
    __m256(v, v)
}

/// Lanes in memory order: `e0` is lane 0.
#[inline(always)]
pub fn _mm256_setr_ps(
    e0: f32,
    e1: f32,
    e2: f32,
    e3: f32,
    e4: f32,
    e5: f32,
    e6: f32,
    e7: f32,
) -> __m256 {
    __m256(
        f32x4_from_array([e0, e1, e2, e3]),
        f32x4_from_array([e4, e5, e6, e7]),
    )
}

/// Lanes in reverse order: the last argument (`e0`) is lane 0.
#[inline(always)]
pub fn _mm256_set_ps(
    e7: f32,
    e6: f32,
    e5: f32,
    e4: f32,
    e3: f32,
    e2: f32,
    e1: f32,
    e0: f32,
) -> __m256 {
    _mm256_setr_ps(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// `hi` becomes lanes 4-7, `lo` lanes 0-3.
#[inline(always)]
pub fn _mm256_set_m128(hi: __m128, lo: __m128) -> __m256 {
    __m256(lo, hi)
}

#[inline(always)]
pub fn _mm_setzero_ps() -> __m128 {
    unsafe { vdupq_n_f32(0.0) }
}

#[inline(always)]
pub fn _mm_set1_ps(a: f32) -> __m128 {
    unsafe { vdupq_n_f32(a) }
}

#[inline(always)]
pub fn _mm_setr_ps(e0: f32, e1: f32, e2: f32, e3: f32) -> __m128 {
    f32x4_from_array([e0, e1, e2, e3])
}

#[inline(always)]
pub fn _mm_set_ss(a: f32) -> __m128 {
    f32x4_from_array([a, 0.0, 0.0, 0.0])
}

// ── Arithmetic ───────────────────────────────────────────────────────────────

#[inline(always)]
fn add4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vaddq_f32(a, b) }
}

#[inline(always)]
fn sub4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vsubq_f32(a, b) }
}

#[inline(always)]
fn mul4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vmulq_f32(a, b) }
}

#[inline(always)]
fn div4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vdivq_f32(a, b) }
}

/// `a * b + c`, single rounding.
#[inline(always)]
fn fmadd4(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    unsafe { vfmaq_f32(c, a, b) }
}

/// `-(a * b) + c`, single rounding.
#[inline(always)]
fn fnmadd4(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    unsafe { vfmsq_f32(c, a, b) }
}

/// `a * b - c`, single rounding (negating `c` is exact).
#[inline(always)]
fn fmsub4(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    unsafe { vfmaq_f32(vnegq_f32(c), a, b) }
}

/// x86 `maxps`: `a > b ? a : b` (returns `b` when either is NaN or both are
/// zero), unlike NEON `fmax` which propagates NaN.
#[inline(always)]
fn max4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vbslq_f32(vcgtq_f32(a, b), a, b) }
}

/// x86 `minps`: `a < b ? a : b`.
#[inline(always)]
fn min4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe { vbslq_f32(vcltq_f32(a, b), a, b) }
}

/// Reciprocal estimate refined by one Newton-Raphson step (~2^-16 relative
/// error; x86 `rcpps` is ~2^-12). `rcp(±0) = ±inf`, `rcp(±inf) = ±0`.
#[inline(always)]
fn rcp4(a: float32x4_t) -> float32x4_t {
    unsafe {
        let e = vrecpeq_f32(a);
        vmulq_f32(e, vrecpsq_f32(a, e))
    }
}

#[inline(always)]
pub fn _mm256_add_ps(a: __m256, b: __m256) -> __m256 {
    halves!(add4, a, b)
}

#[inline(always)]
pub fn _mm256_sub_ps(a: __m256, b: __m256) -> __m256 {
    halves!(sub4, a, b)
}

#[inline(always)]
pub fn _mm256_mul_ps(a: __m256, b: __m256) -> __m256 {
    halves!(mul4, a, b)
}

#[inline(always)]
pub fn _mm256_div_ps(a: __m256, b: __m256) -> __m256 {
    halves!(div4, a, b)
}

#[inline(always)]
pub fn _mm256_fmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    halves!(fmadd4, a, b, c)
}

#[inline(always)]
pub fn _mm256_fnmadd_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    halves!(fnmadd4, a, b, c)
}

#[inline(always)]
pub fn _mm256_fmsub_ps(a: __m256, b: __m256, c: __m256) -> __m256 {
    halves!(fmsub4, a, b, c)
}

#[inline(always)]
pub fn _mm256_max_ps(a: __m256, b: __m256) -> __m256 {
    halves!(max4, a, b)
}

#[inline(always)]
pub fn _mm256_min_ps(a: __m256, b: __m256) -> __m256 {
    halves!(min4, a, b)
}

#[inline(always)]
pub fn _mm256_rcp_ps(a: __m256) -> __m256 {
    halves!(rcp4, a)
}

#[inline(always)]
pub fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
    add4(a, b)
}

#[inline(always)]
pub fn _mm_fmadd_ps(a: __m128, b: __m128, c: __m128) -> __m128 {
    fmadd4(a, b, c)
}

#[inline(always)]
pub fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
    max4(a, b)
}

/// Lane 0 = `a0 + b0`; lanes 1-3 from `a`.
#[inline(always)]
pub fn _mm_add_ss(a: __m128, b: __m128) -> __m128 {
    unsafe { vsetq_lane_f32::<0>(vgetq_lane_f32::<0>(a) + vgetq_lane_f32::<0>(b), a) }
}

/// `[a0 + a1, a2 + a3, b0 + b1, b2 + b3]`.
#[inline(always)]
pub fn _mm_hadd_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vpaddq_f32(a, b) }
}

#[inline(always)]
pub fn _mm_cvtss_f32(a: __m128) -> f32 {
    unsafe { vgetq_lane_f32::<0>(a) }
}

// ── Rounding ─────────────────────────────────────────────────────────────────

#[inline(always)]
fn round4(a: float32x4_t, rounding: i32) -> float32x4_t {
    unsafe {
        if rounding & super::_MM_FROUND_CUR_DIRECTION != 0 {
            return vrndiq_f32(a);
        }
        match rounding & 0x3 {
            super::_MM_FROUND_TO_NEAREST_INT => vrndnq_f32(a),
            super::_MM_FROUND_TO_NEG_INF => vrndmq_f32(a),
            super::_MM_FROUND_TO_POS_INF => vrndpq_f32(a),
            _ => vrndq_f32(a),
        }
    }
}

/// `rounding` is `_MM_FROUND_*` (the exception-suppression bit is ignored).
#[inline(always)]
pub fn _mm256_round_ps(a: __m256, rounding: i32) -> __m256 {
    __m256(round4(a.0, rounding), round4(a.1, rounding))
}

// ── Comparison ───────────────────────────────────────────────────────────────

/// All 32 x86 predicates. The low 4 bits select the relation; bit 4 only
/// toggles signaling vs. quiet NaN behavior, which has no effect on the result.
#[inline(always)]
fn cmp4(a: float32x4_t, b: float32x4_t, predicate: i32) -> uint32x4_t {
    unsafe {
        let ordered = vandq_u32(vceqq_f32(a, a), vceqq_f32(b, b));
        match predicate & 0xf {
            0x0 => vceqq_f32(a, b),
            0x1 => vcltq_f32(a, b),
            0x2 => vcleq_f32(a, b),
            0x3 => vmvnq_u32(ordered),
            0x4 => vmvnq_u32(vceqq_f32(a, b)),
            0x5 => vmvnq_u32(vcltq_f32(a, b)),
            0x6 => vmvnq_u32(vcleq_f32(a, b)),
            0x7 => ordered,
            0x8 => vorrq_u32(vceqq_f32(a, b), vmvnq_u32(ordered)),
            0x9 => vmvnq_u32(vcgeq_f32(a, b)),
            0xa => vmvnq_u32(vcgtq_f32(a, b)),
            0xb => vdupq_n_u32(0),
            0xc => vandq_u32(vmvnq_u32(vceqq_f32(a, b)), ordered),
            0xd => vcgeq_f32(a, b),
            0xe => vcgtq_f32(a, b),
            _ => vdupq_n_u32(u32::MAX),
        }
    }
}

/// Returns an all-ones / all-zeros mask per lane. `predicate` is `_CMP_*`.
#[inline(always)]
pub fn _mm256_cmp_ps(a: __m256, b: __m256, predicate: i32) -> __m256 {
    unsafe {
        __m256(
            vreinterpretq_f32_u32(cmp4(a.0, b.0, predicate)),
            vreinterpretq_f32_u32(cmp4(a.1, b.1, predicate)),
        )
    }
}

/// Bit `i` of the result is the sign bit of lane `i`.
#[inline(always)]
fn movemask4(a: float32x4_t) -> u32 {
    const LANE_SHIFT: [i32; 4] = [0, 1, 2, 3];
    unsafe {
        let sign = vshrq_n_u32::<31>(vreinterpretq_u32_f32(a));
        vaddvq_u32(vshlq_u32(sign, vld1q_s32(LANE_SHIFT.as_ptr())))
    }
}

#[inline(always)]
pub fn _mm256_movemask_ps(a: __m256) -> i32 {
    (movemask4(a.0) | (movemask4(a.1) << 4)) as i32
}

// ── Bitwise ──────────────────────────────────────────────────────────────────

#[inline(always)]
fn and4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe {
        vreinterpretq_f32_u32(vandq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
}

#[inline(always)]
fn or4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe {
        vreinterpretq_f32_u32(vorrq_u32(
            vreinterpretq_u32_f32(a),
            vreinterpretq_u32_f32(b),
        ))
    }
}

/// `!a & b`.
#[inline(always)]
fn andnot4(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    unsafe {
        vreinterpretq_f32_u32(vbicq_u32(
            vreinterpretq_u32_f32(b),
            vreinterpretq_u32_f32(a),
        ))
    }
}

#[inline(always)]
pub fn _mm256_and_ps(a: __m256, b: __m256) -> __m256 {
    halves!(and4, a, b)
}

#[inline(always)]
pub fn _mm256_or_ps(a: __m256, b: __m256) -> __m256 {
    halves!(or4, a, b)
}

#[inline(always)]
pub fn _mm256_andnot_ps(a: __m256, b: __m256) -> __m256 {
    halves!(andnot4, a, b)
}

// ── Blend / select ───────────────────────────────────────────────────────────

/// Lane `i` from `b` where the sign bit of `mask` lane `i` is set, else `a`.
#[inline(always)]
fn blendv4(a: float32x4_t, b: float32x4_t, mask: float32x4_t) -> float32x4_t {
    unsafe {
        let select = vreinterpretq_u32_s32(vshrq_n_s32::<31>(vreinterpretq_s32_f32(mask)));
        vbslq_f32(select, b, a)
    }
}

#[inline(always)]
pub fn _mm256_blendv_ps(a: __m256, b: __m256, mask: __m256) -> __m256 {
    halves!(blendv4, a, b, mask)
}

/// Selection mask for four lanes from the low four bits of `imm`.
#[inline(always)]
fn lane_mask4(imm: i32) -> uint32x4_t {
    let bit = |i: i32| if (imm >> i) & 1 != 0 { u32::MAX } else { 0 };
    let lanes = [bit(0), bit(1), bit(2), bit(3)];
    unsafe { vld1q_u32(lanes.as_ptr()) }
}

/// Lane `i` from `b` where bit `i` of `imm8` is set, else from `a`.
#[inline(always)]
pub fn _mm256_blend_ps(a: __m256, b: __m256, imm8: i32) -> __m256 {
    unsafe {
        __m256(
            vbslq_f32(lane_mask4(imm8), b.0, a.0),
            vbslq_f32(lane_mask4(imm8 >> 4), b.1, a.1),
        )
    }
}

// ── Lane movement ────────────────────────────────────────────────────────────

/// `[a[imm[1:0]], a[imm[3:2]], b[imm[5:4]], b[imm[7:6]]]`.
#[inline(always)]
fn shuffle4(a: float32x4_t, b: float32x4_t, imm8: i32) -> float32x4_t {
    let a = f32x4_to_array(a);
    let b = f32x4_to_array(b);
    let idx = |shift: i32| ((imm8 >> shift) & 0x3) as usize;
    f32x4_from_array([a[idx(0)], a[idx(2)], b[idx(4)], b[idx(6)]])
}

#[inline(always)]
pub fn _mm_shuffle_ps(a: __m128, b: __m128, imm8: i32) -> __m128 {
    shuffle4(a, b, imm8)
}

/// Same shuffle applied independently to each 128-bit half.
#[inline(always)]
pub fn _mm256_shuffle_ps(a: __m256, b: __m256, imm8: i32) -> __m256 {
    __m256(shuffle4(a.0, b.0, imm8), shuffle4(a.1, b.1, imm8))
}

/// One 128-bit half chosen by a 4-bit `permute2f128` control nibble.
#[inline(always)]
fn select_half(a: __m256, b: __m256, control: i32) -> float32x4_t {
    if control & 0x8 != 0 {
        return _mm_setzero_ps();
    }
    match control & 0x3 {
        0 => a.0,
        1 => a.1,
        2 => b.0,
        _ => b.1,
    }
}

#[inline(always)]
pub fn _mm256_permute2f128_ps(a: __m256, b: __m256, imm8: i32) -> __m256 {
    __m256(select_half(a, b, imm8), select_half(a, b, imm8 >> 4))
}

#[inline(always)]
pub fn _mm256_extractf128_ps(a: __m256, imm8: i32) -> __m128 {
    if imm8 & 1 == 0 { a.0 } else { a.1 }
}

#[inline(always)]
pub fn _mm256_insertf128_ps(a: __m256, b: __m128, imm8: i32) -> __m256 {
    if imm8 & 1 == 0 {
        __m256(b, a.1)
    } else {
        __m256(a.0, b)
    }
}

#[inline(always)]
pub fn _mm256_castps256_ps128(a: __m256) -> __m128 {
    a.0
}

/// Per 128-bit half: `[a0, b0, a1, b1]`.
#[inline(always)]
pub fn _mm256_unpacklo_ps(a: __m256, b: __m256) -> __m256 {
    unsafe { __m256(vzip1q_f32(a.0, b.0), vzip1q_f32(a.1, b.1)) }
}

/// Per 128-bit half: `[a2, b2, a3, b3]`.
#[inline(always)]
pub fn _mm256_unpackhi_ps(a: __m256, b: __m256) -> __m256 {
    unsafe { __m256(vzip2q_f32(a.0, b.0), vzip2q_f32(a.1, b.1)) }
}

/// `[b2, b3, a2, a3]`.
#[inline(always)]
pub fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128 {
    unsafe { vcombine_f32(vget_high_f32(b), vget_high_f32(a)) }
}

/// `[a1, a1, a3, a3]`.
#[inline(always)]
pub fn _mm_movehdup_ps(a: __m128) -> __m128 {
    unsafe { vtrn2q_f32(a, a) }
}
