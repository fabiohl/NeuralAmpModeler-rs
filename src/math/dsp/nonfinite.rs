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
}
