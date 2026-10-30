// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD non-finite detection and sanitization for f32 audio buffers.
//!
//! Hosts gate hostile input (denormal/NaN/Inf storms from a misbehaving bus)
//! before it reaches the neural inference path. These helpers centralize the
//! scan so consumers do not reimplement per-sample `is_finite` loops.
//!
//! # Real-time contract
//!
//! Both functions are RT-safe: they perform no allocation, no locking and no
//! logging. Implemented with AVX2 intrinsics (`_mm256_*`) — no scalar
//! fallback: the `x86-64-v3` baseline guarantees AVX2 unconditionally.

use crate::arch::simd::*;

/// Returns `true` when every sample is finite (no NaN and no ±inf).
///
/// AVX2 implementation: `_mm256_cmp_ps` with `_CMP_ORD_Q` rejects NaN, and a
/// magnitude comparison against `f32::MAX` rejects ±inf, lane-reduced with
/// `_mm256_movemask_ps`.
///
/// # Safety
/// `samples` must be a valid slice. AVX2 must be available (guaranteed by the
/// crate's `x86-64-v3` build baseline).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn all_finite_f32_avx2(samples: &[f32]) -> bool {
    let len = samples.len();
    let mut i = 0;
    // SAFETY: the two `while` guards keep every 8-lane `loadu` within
    // `samples` (`i + 8 <= len`); the scalar tail covers the remainder.
    // `loadu` needs no alignment; AVX2 is enabled by `#[target_feature]`.
    unsafe {
        let max = _mm256_set1_ps(f32::MAX);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 8 <= len {
            let v = _mm256_loadu_ps(samples.as_ptr().add(i));
            // NaN check: ordered comparison is false for NaN lanes.
            let ord = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
            if _mm256_movemask_ps(ord) as u32 != 0xFF {
                return false;
            }
            // Inf check: |v| > MAX only for ±inf (finite magnitudes pass).
            let abs = _mm256_and_ps(v, abs_mask);
            let le = _mm256_cmp_ps(abs, max, _CMP_LE_OQ);
            if _mm256_movemask_ps(le) as u32 != 0xFF {
                return false;
            }
            i += 8;
        }
    }
    while i < len {
        if !samples[i].is_finite() {
            return false;
        }
        i += 1;
    }
    true
}

/// Replaces NaN and ±inf with zero in-place.
///
/// Branchless AVX2 implementation via `_mm256_blendv_ps`: lanes flagged
/// non-finite by the ordered + magnitude comparisons select `0.0`.
///
/// # Safety
/// `samples` must be a valid mutable slice. AVX2 must be available
/// (guaranteed by the crate's `x86-64-v3` build baseline).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn sanitize_nonfinite_f32_avx2(samples: &mut [f32]) {
    let len = samples.len();
    let mut i = 0;
    // SAFETY: the two `while` guards keep every 8-lane load/store within
    // `samples` (`i + 8 <= len`); the scalar tail covers the remainder.
    // `loadu`/`storeu` need no alignment; AVX2 is enabled by `#[target_feature]`.
    unsafe {
        let max = _mm256_set1_ps(f32::MAX);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let zero = _mm256_setzero_ps();
        while i + 8 <= len {
            let ptr = samples.as_mut_ptr().add(i);
            let v = _mm256_loadu_ps(ptr);
            let ord = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
            let abs = _mm256_and_ps(v, abs_mask);
            let le = _mm256_cmp_ps(abs, max, _CMP_LE_OQ);
            // finite = ord AND le; blend selects zero where not finite.
            let finite = _mm256_and_ps(ord, le);
            let clean = _mm256_blendv_ps(zero, v, finite);
            _mm256_storeu_ps(ptr, clean);
            i += 8;
        }
    }
    while i < len {
        if !samples[i].is_finite() {
            samples[i] = 0.0;
        }
        i += 1;
    }
}

/// Returns `true` when every sample is finite (no NaN and no ±inf).
///
/// Baseline dispatch: AVX2 is guaranteed by the `x86-64-v3` build target, so
/// this calls the AVX2 kernel unconditionally with no scalar fallback and no
/// runtime feature detection.
#[inline]
pub fn all_finite_f32(samples: &[f32]) -> bool {
    // SAFETY: `all_finite_f32_avx2` requires AVX2, guaranteed by the crate's
    // `x86-64-v3` build baseline (`-Ctarget-cpu=x86-64-v3`).
    unsafe { all_finite_f32_avx2(samples) }
}

/// Replaces NaN and ±inf with zero in-place.
///
/// Baseline dispatch: AVX2 is guaranteed by the `x86-64-v3` build target, so
/// this calls the AVX2 kernel unconditionally with no scalar fallback and no
/// runtime feature detection.
#[inline]
pub fn sanitize_nonfinite_f32(samples: &mut [f32]) {
    // SAFETY: `sanitize_nonfinite_f32_avx2` requires AVX2, guaranteed by the
    // crate's `x86-64-v3` build baseline (`-Ctarget-cpu=x86-64-v3`).
    unsafe { sanitize_nonfinite_f32_avx2(samples) }
}

/// Sanitizes non-finite values (replacing NaN and ±inf with `0.0`) while
/// copying from `src` to `dst`, and simultaneously accumulates the absolute peak
/// of the clean output via AVX2.
///
/// Returns the maximum absolute value `max(|clean_i|)`.
///
/// # Real-time contract
///
/// RT-safe: zero heap allocations, zero locking, zero logging.
///
/// # Safety
///
/// `src` and `dst` must be valid slices. AVX2 must be available (guaranteed by
/// the crate's `x86-64-v3` build baseline).
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn sanitize_copy_peak_avx2(src: &[f32], dst: &mut [f32]) -> f32 {
    let len = core::cmp::min(src.len(), dst.len());
    if len == 0 {
        return 0.0;
    }

    let mut i = 0;
    let mut peak = 0.0f32;

    // SAFETY: the loop bounds keep every 8-lane load/store within `src` and `dst`
    // (`i + 8 <= len`). AVX2 is enabled by `#[target_feature]`.
    unsafe {
        let max = _mm256_set1_ps(f32::MAX);
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let zero = _mm256_setzero_ps();
        let mut max_v = _mm256_setzero_ps();

        while i + 8 <= len {
            let src_ptr = src.as_ptr().add(i);
            let dst_ptr = dst.as_mut_ptr().add(i);
            let v = _mm256_loadu_ps(src_ptr);

            let ord = _mm256_cmp_ps(v, v, _CMP_ORD_Q);
            let abs = _mm256_and_ps(v, abs_mask);
            let le = _mm256_cmp_ps(abs, max, _CMP_LE_OQ);

            // finite = ord AND le; blend selects zero where not finite.
            let finite = _mm256_and_ps(ord, le);
            let clean = _mm256_blendv_ps(zero, v, finite);
            _mm256_storeu_ps(dst_ptr, clean);

            let abs_clean = _mm256_and_ps(clean, abs_mask);
            max_v = _mm256_max_ps(max_v, abs_clean);

            i += 8;
        }

        // Horizontal max reduction across 8 SIMD lanes
        let hi = _mm256_extractf128_ps(max_v, 1);
        let lo = _mm256_castps256_ps128(max_v);
        let m128 = _mm_max_ps(lo, hi);
        let shuf = _mm_shuffle_ps(m128, m128, 0xEE);
        let m64 = _mm_max_ps(m128, shuf);
        let shuf2 = _mm_shuffle_ps(m64, m64, 0x55);
        let m32 = _mm_max_ps(m64, shuf2);
        _mm_store_ss(&mut peak, m32);
    }

    while i < len {
        let s = src[i];
        let clean = if s.is_finite() { s } else { 0.0 };
        dst[i] = clean;
        let a = clean.abs();
        if a > peak {
            peak = a;
        }
        i += 1;
    }

    peak
}

/// Sanitizes non-finite values (replacing NaN and ±inf with `0.0`) while
/// copying from `src` to `dst`, and simultaneously accumulates the absolute peak
/// of the clean output.
///
/// Returns the maximum absolute value `max(|clean_i|)`.
///
/// Baseline dispatch: AVX2 is guaranteed by the `x86-64-v3` build target, so
/// this calls the AVX2 kernel unconditionally with no scalar fallback and no
/// runtime feature detection.
#[inline]
pub fn sanitize_copy_peak(src: &[f32], dst: &mut [f32]) -> f32 {
    // SAFETY: `sanitize_copy_peak_avx2` requires AVX2, guaranteed by the
    // crate's `x86-64-v3` build baseline (`-Ctarget-cpu=x86-64-v3`).
    unsafe { sanitize_copy_peak_avx2(src, dst) }
}

#[cfg(test)]
#[path = "nonfinite_test.rs"]
mod tests;
