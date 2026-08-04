// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Safe partial-YMM load/store helpers for specialized GEMV kernels.
//!
//! `f16_avx2_fused` and `f16_avx2_overwrite` depend on these helpers to
//! fix the UB of loading/storing YMM registers from/to partial slices
//! (bias/out_frame with < 8 elements).

use core::arch::x86_64::*;

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Loads up to `len` f32 values from `src` into a YMM register, zeroing excess lanes.
///
/// Uses direct xmm load + ymm insert when `len <= 4` to avoid a stack round-trip.
///
/// # Safety
/// `src` must have at least `len` elements.
#[inline(always)]
pub(crate) unsafe fn load_partial_ymm(src: &[f32], len: usize) -> __m256 {
    if len <= 4 {
        _mm256_insertf128_ps(_mm256_setzero_ps(), _mm_loadu_ps(src.as_ptr()), 0)
    } else {
        let mut tmp = [0.0f32; 8];
        for (i, item) in tmp.iter_mut().enumerate().take(len) {
            *item = *src.get_unchecked(i);
        }
        _mm256_loadu_ps(tmp.as_ptr())
    }
}

/// Stores the first `len` f32 lanes of a YMM register to `dst`.
///
/// Uses direct xmm store when `len <= 4` to avoid a stack round-trip.
///
/// # Safety
/// `dst` must have at least `len` elements.
#[inline(always)]
pub(crate) unsafe fn store_partial_ymm(ymm: __m256, dst: &mut [f32], len: usize) {
    if len <= 4 {
        _mm_storeu_ps(dst.as_mut_ptr(), _mm256_castps256_ps128(ymm));
    } else {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), ymm);
        for (i, &val) in tmp.iter().enumerate().take(len) {
            *dst.get_unchecked_mut(i) = val;
        }
    }
}
