// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loads, stores, masked access, gather and prefetch. Raw-pointer forms are
//! `unsafe` (callers uphold the x86 pointer-validity contract); the
//! reference-taking broadcasts and the non-dereferencing prefetch are safe, as
//! in `core::arch::x86_64`. NEON loads and stores have no alignment
//! requirement, so aligned and unaligned forms are identical.

use super::{__m128, __m128i, __m256, __m256i, f32x4_from_array, i32x4_to_array};
use core::arch::aarch64::*;

#[inline(always)]
pub unsafe fn _mm256_loadu_ps(mem_addr: *const f32) -> __m256 {
    unsafe { __m256(vld1q_f32(mem_addr), vld1q_f32(mem_addr.add(4))) }
}

#[inline(always)]
pub unsafe fn _mm256_load_ps(mem_addr: *const f32) -> __m256 {
    unsafe { _mm256_loadu_ps(mem_addr) }
}

#[inline(always)]
pub unsafe fn _mm256_storeu_ps(mem_addr: *mut f32, a: __m256) {
    unsafe {
        vst1q_f32(mem_addr, a.0);
        vst1q_f32(mem_addr.add(4), a.1);
    }
}

#[inline(always)]
pub unsafe fn _mm256_store_ps(mem_addr: *mut f32, a: __m256) {
    unsafe { _mm256_storeu_ps(mem_addr, a) }
}

#[inline(always)]
pub fn _mm256_broadcast_ss(f: &f32) -> __m256 {
    let v = unsafe { vld1q_dup_f32(f) };
    __m256(v, v)
}

#[inline(always)]
pub fn _mm256_broadcast_ps(a: &__m128) -> __m256 {
    __m256(*a, *a)
}

#[inline(always)]
pub unsafe fn _mm_loadu_ps(mem_addr: *const f32) -> __m128 {
    unsafe { vld1q_f32(mem_addr) }
}

#[inline(always)]
pub unsafe fn _mm_storeu_ps(mem_addr: *mut f32, a: __m128) {
    unsafe { vst1q_f32(mem_addr, a) }
}

#[inline(always)]
pub unsafe fn _mm_load1_ps(mem_addr: *const f32) -> __m128 {
    unsafe { vld1q_dup_f32(mem_addr) }
}

/// Stores lane 0 only.
#[inline(always)]
pub unsafe fn _mm_store_ss(mem_addr: *mut f32, a: __m128) {
    unsafe { vst1q_lane_f32::<0>(mem_addr, a) }
}

#[inline(always)]
pub unsafe fn _mm256_loadu_si256(mem_addr: *const __m256i) -> __m256i {
    let p = mem_addr as *const i32;
    unsafe { __m256i(vld1q_s32(p), vld1q_s32(p.add(4))) }
}

#[inline(always)]
pub unsafe fn _mm_loadu_si128(mem_addr: *const __m128i) -> __m128i {
    unsafe { vld1q_s32(mem_addr as *const i32) }
}

#[inline(always)]
pub unsafe fn _mm_storeu_si128(mem_addr: *mut __m128i, a: __m128i) {
    unsafe { vst1q_s32(mem_addr as *mut i32, a) }
}

/// Loads 64 bits into the low half; the high half is zero.
#[inline(always)]
pub unsafe fn _mm_loadu_si64(mem_addr: *const u8) -> __m128i {
    unsafe { vcombine_s32(vreinterpret_s32_u8(vld1_u8(mem_addr)), vdup_n_s32(0)) }
}

/// Loads lane `i` from `mem_addr + i` where the sign bit of `mask` lane `i` is
/// set, else 0. Masked-off lanes are never read, so the access may end before
/// the full 8-lane span (the x86 fault-suppression contract).
#[inline(always)]
pub unsafe fn _mm256_maskload_ps(mem_addr: *const f32, mask: __m256i) -> __m256 {
    let lo = i32x4_to_array(mask.0);
    let hi = i32x4_to_array(mask.1);
    let lane = |m: i32, i: usize| {
        if m < 0 {
            unsafe { mem_addr.add(i).read_unaligned() }
        } else {
            0.0
        }
    };
    __m256(
        f32x4_from_array([
            lane(lo[0], 0),
            lane(lo[1], 1),
            lane(lo[2], 2),
            lane(lo[3], 3),
        ]),
        f32x4_from_array([
            lane(hi[0], 4),
            lane(hi[1], 5),
            lane(hi[2], 6),
            lane(hi[3], 7),
        ]),
    )
}

/// Stores lane `i` to `mem_addr + i` where the sign bit of `mask` lane `i` is
/// set; masked-off addresses are never written.
#[inline(always)]
pub unsafe fn _mm256_maskstore_ps(mem_addr: *mut f32, mask: __m256i, a: __m256) {
    let m: [i32; 8] = unsafe { core::mem::transmute(mask) };
    let v: [f32; 8] = unsafe { core::mem::transmute(a) };
    for i in 0..8 {
        if m[i] < 0 {
            unsafe { mem_addr.add(i).write_unaligned(v[i]) };
        }
    }
}

/// Lane `i` = `*(base_addr as *const u8).offset(vindex[i] * scale)` as `f32`.
#[inline(always)]
pub unsafe fn _mm256_i32gather_ps(base_addr: *const f32, vindex: __m256i, scale: i32) -> __m256 {
    let idx: [i32; 8] = unsafe { core::mem::transmute(vindex) };
    let base = base_addr as *const u8;
    let lane = |i: usize| unsafe {
        (base.offset(idx[i] as isize * scale as isize) as *const f32).read_unaligned()
    };
    __m256(
        f32x4_from_array([lane(0), lane(1), lane(2), lane(3)]),
        f32x4_from_array([lane(4), lane(5), lane(6), lane(7)]),
    )
}

/// Data prefetch. `T0` → L1, `T1` → L2, `T2` → L3, `NTA` → L1 streaming.
/// A hint only: never dereferences `p` and never faults.
#[inline(always)]
pub fn _mm_prefetch<const STRATEGY: i32>(p: *const i8) {
    unsafe {
        match STRATEGY {
            super::_MM_HINT_T0 => core::arch::asm!(
                "prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)
            ),
            super::_MM_HINT_T1 => core::arch::asm!(
                "prfm pldl2keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)
            ),
            super::_MM_HINT_T2 => core::arch::asm!(
                "prfm pldl3keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)
            ),
            _ => core::arch::asm!(
                "prfm pldl1strm, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)
            ),
        }
    }
}
