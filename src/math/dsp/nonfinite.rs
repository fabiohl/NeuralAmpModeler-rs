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

use core::arch::x86_64::*;

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
#[target_feature(enable = "avx2")]
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
#[target_feature(enable = "avx2")]
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
#[target_feature(enable = "avx2")]
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
mod tests {
    use super::*;

    #[test]
    fn test_clean_slice_is_finite() {
        let v: Vec<f32> = (0..64).map(|i| i as f32 * 0.01 - 0.3).collect();
        assert!(all_finite_f32(&v));
        assert!(all_finite_f32(&[]));
    }

    #[test]
    fn test_nan_at_start_middle_end_detected() {
        for &pos in &[0usize, 17, 63] {
            let mut v = vec![0.25f32; 64];
            v[pos] = f32::NAN;
            assert!(!all_finite_f32(&v), "NaN at {pos} must be detected");
        }
    }

    #[test]
    fn test_infinities_detected() {
        for &bad in &[f32::INFINITY, f32::NEG_INFINITY] {
            let mut v = vec![0.25f32; 65];
            v[33] = bad;
            assert!(!all_finite_f32(&v), "{bad} must be detected");
            // Scalar-tail lane (index 64, beyond the 8-wide loop).
            let mut tail = vec![0.25f32; 65];
            tail[64] = bad;
            assert!(!all_finite_f32(&tail), "{bad} in tail must be detected");
        }
    }

    #[test]
    fn test_sanitize_zeroes_nonfinite_keeps_finite() {
        let mut v = vec![0.5f32; 40];
        v[0] = f32::NAN;
        v[13] = f32::INFINITY;
        v[27] = f32::NEG_INFINITY;
        v[39] = f32::NAN;
        sanitize_nonfinite_f32(&mut v);
        assert!(all_finite_f32(&v));
        for (i, &x) in v.iter().enumerate() {
            let expected = if [0, 13, 27, 39].contains(&i) {
                0.0
            } else {
                0.5
            };
            assert_eq!(x, expected, "mismatch at {i}");
        }
    }

    #[test]
    fn test_sanitize_matches_scalar_reference() {
        let mut v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        v[3] = f32::NAN;
        v[50] = f32::INFINITY;
        v[99] = f32::NEG_INFINITY;
        let mut reference = v.clone();
        for x in &mut reference {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        sanitize_nonfinite_f32(&mut v);
        assert_eq!(v, reference);
    }

    #[test]
    fn test_push_pop_is_zero_alloc() {
        use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

        let v = vec![0.25f32; 256];
        let mut m = v.clone();
        let _guard = TrackingGuard::new();
        assert!(all_finite_f32(&v));
        sanitize_nonfinite_f32(&mut m);
        assert_eq!(
            get_alloc_count(),
            0,
            "finite scan/sanitize must not allocate on the RT path"
        );
    }

    #[test]
    fn test_sanitize_copy_peak_clean() {
        for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 65, 128] {
            let src: Vec<f32> = (0..len).map(|i| (i as f32 * 0.1).sin() * 0.8).collect();
            let mut dst = vec![999.0f32; len];
            let peak = sanitize_copy_peak(&src, &mut dst);
            assert_eq!(src, dst, "clean data must copy identically for len={len}");
            let expected_peak = src.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            assert_eq!(
                peak.to_bits(),
                expected_peak.to_bits(),
                "peak mismatch for len={len}"
            );
        }
    }

    #[test]
    fn test_sanitize_copy_peak_nonfinite() {
        let mut src = vec![0.4f32; 70];
        src[0] = f32::NAN;
        src[7] = f32::INFINITY;
        src[8] = f32::NEG_INFINITY;
        src[31] = f32::NAN;
        src[63] = f32::INFINITY;
        src[64] = f32::NEG_INFINITY; // tail
        src[69] = f32::NAN; // tail
        src[12] = -0.9f32; // max abs peak

        let mut dst = vec![1.0f32; 70];
        let peak = sanitize_copy_peak(&src, &mut dst);

        assert_eq!(peak, 0.9f32);
        assert!(all_finite_f32(&dst));
        for i in [0, 7, 8, 31, 63, 64, 69] {
            assert_eq!(dst[i], 0.0f32, "bad index {i} must be sanitized to 0.0");
        }
        assert_eq!(dst[12], -0.9f32);
        assert_eq!(dst[1], 0.4f32);
    }

    #[test]
    fn test_sanitize_copy_peak_parity_with_separate_passes() {
        for len in [0, 1, 7, 8, 9, 32, 65, 128] {
            let mut src: Vec<f32> = (0..len).map(|i| (i as f32 - 16.0) * 0.05).collect();
            if len > 3 {
                src[1] = f32::NAN;
                src[len - 2] = f32::INFINITY;
            }
            let mut ref_dst = src.clone();
            sanitize_nonfinite_f32(&mut ref_dst);
            let ref_peak = ref_dst.iter().fold(0.0f32, |m, &x| m.max(x.abs()));

            let mut fused_dst = vec![0.0f32; len];
            let fused_peak = sanitize_copy_peak(&src, &mut fused_dst);

            assert_eq!(fused_dst, ref_dst, "dest buffer mismatch for len={len}");
            assert_eq!(
                fused_peak.to_bits(),
                ref_peak.to_bits(),
                "fused peak mismatch for len={len}"
            );
        }
    }

    #[test]
    fn test_sanitize_copy_peak_zero_alloc() {
        use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

        let src = vec![0.25f32; 256];
        let mut dst = vec![0.0f32; 256];
        let _guard = TrackingGuard::new();
        let _ = sanitize_copy_peak(&src, &mut dst);
        assert_eq!(
            get_alloc_count(),
            0,
            "sanitize_copy_peak must not allocate on the RT path"
        );
    }
}
