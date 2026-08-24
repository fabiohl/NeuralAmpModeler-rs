// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// SIMD intrinsics are called directly inside `#[target_feature]` `unsafe fn`
// bodies; each function states its pointer/slice contract in a `# Safety`
// section below. The `unsafe_op_in_unsafe_fn` allow avoids double-wrapping
// intrinsic calls that are the documented body of these kernels.
#![allow(unsafe_op_in_unsafe_fn)]

//! SIMD utilities for reductions and horizontal operations.

use super::kahan::kahan_add;
use core::arch::x86_64::*;

/// Horizontal sum of an AVX2 (256-bit) register to scalar f32.
///
/// Performs the reduction via 128-bit lane extraction and successive additions.
/// Total instructions: ~7 (including store_ss).
///
/// # Safety
/// The caller must guarantee AVX2 is available (guaranteed by the
/// `#[target_feature]` attribute on this `unsafe fn`). No pointers are
/// dereferenced: the reduction and final store operate on the register argument
/// and a stack-local `f32`.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(s128);
    let sums = _mm_add_ps(s128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let r = _mm_add_ss(sums, shuf2);
    let mut out = 0.0f32;
    _mm_store_ss(&mut out, r);
    out
}

/// Horizontal sum of an AVX-512 (512-bit) register to scalar f32.
///
/// Uses the native AVX-512 Foundation reduction intrinsic.
///
/// # Safety
/// The caller must guarantee AVX-512F is available (enforced by the
/// `#[target_feature]` attribute). No pointers are dereferenced.
#[cfg(feature = "avx512")]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Horizontal sum of an f32 buffer via AVX2.
///
/// # Safety
/// `ptr` must be non-null, aligned to `align_of::<f32>()`, and valid for `len`
/// consecutive reads of initialized `f32` values. The caller must guarantee
/// AVX2 is available (`#[target_feature]`).
#[target_feature(enable = "avx2")]
pub unsafe fn horizontal_sum_avx2(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm256_setzero_ps();

    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        sum_v = _mm256_add_ps(sum_v, v);
        i += 8;
    }

    let mut total = hsum_avx2(sum_v);
    let mut compensation = 0.0f32;

    while i < len {
        (total, compensation) = kahan_add(total, compensation, *ptr.add(i));
        i += 1;
    }

    total
}

/// Horizontal sum of an f32 buffer via AVX-512.
///
/// # Safety
/// `ptr` must be non-null, aligned to `align_of::<f32>()`, and valid for `len`
/// consecutive reads of initialized `f32` values. The caller must guarantee
/// AVX-512F is available (`#[target_feature]`).
#[cfg(feature = "avx512")]
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        let v = _mm512_loadu_ps(ptr.add(i));
        sum_v = _mm512_add_ps(sum_v, v);
        i += 16;
    }

    let mut total = hsum_avx512(sum_v);
    let compensation = 0.0f32;

    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        let v = _mm512_maskz_loadu_ps(mask, ptr.add(i));
        (total, _) = kahan_add(total, compensation, hsum_avx512(v));
    }

    total
}
