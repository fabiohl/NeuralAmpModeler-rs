// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Safe partial-YMM load/store helpers for specialized GEMV kernels.
//!
//! `f16_avx2_fused` and `f16_avx2_overwrite` depend on these helpers to
//! fix the UB of loading/storing YMM registers from/to partial slices
//! (bias/out_frame with < 8 elements).

use core::arch::x86_64::*;

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Loads `len` f32 values from `src` into a YMM register, zeroing excess lanes.
///
/// Direct XMM load + YMM insert is used when `len <= 4` (in practice `len == 4`)
/// to avoid a stack round-trip. Because the fast path executes `_mm_loadu_ps`
/// (loading 4 f32s = 128 bits unconditionally), `src` must have at least 4 elements
/// even when `len < 4`. The actual precondition for this helper is `len ∈ 4..=8`.
///
/// # Preconditions
/// - `len` must be in the range `4..=8`.
///
/// # Safety
/// - `src` must have at least `len` elements (and thus at least 4).
#[inline(always)]
pub(crate) unsafe fn load_partial_ymm(src: &[f32], len: usize) -> __m256 {
    debug_assert!(
        (4..=8).contains(&len),
        "load_partial_ymm requires len in 4..=8, got {len}"
    );
    debug_assert!(
        src.len() >= len,
        "src slice too short: len={len}, src.len()={}",
        src.len()
    );
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
/// Direct XMM store is used when `len <= 4` (in practice `len == 4`) to avoid a
/// stack round-trip. Because the fast path executes `_mm_storeu_ps` (writing 4 f32s = 128 bits
/// unconditionally), `dst` must have at least 4 elements even when `len < 4`.
/// The actual precondition for this helper is `len ∈ 4..=8`.
///
/// # Preconditions
/// - `len` must be in the range `4..=8`.
///
/// # Safety
/// - `dst` must have at least `len` elements (and thus at least 4).
#[inline(always)]
pub(crate) unsafe fn store_partial_ymm(ymm: __m256, dst: &mut [f32], len: usize) {
    debug_assert!(
        (4..=8).contains(&len),
        "store_partial_ymm requires len in 4..=8, got {len}"
    );
    debug_assert!(
        dst.len() >= len,
        "dst slice too short: len={len}, dst.len()={}",
        dst.len()
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_partial_ymm_len_4() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut out = [0.0f32; 8];
        // SAFETY: src has 6 elements (>= 4), and out has 8 elements.
        unsafe {
            let ymm = load_partial_ymm(&src, 4);
            _mm256_storeu_ps(out.as_mut_ptr(), ymm);
        }
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_load_partial_ymm_len_5_to_8() {
        let src = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        for len in 5..=8 {
            let mut out = [0.0f32; 8];
            // SAFETY: src has 9 elements (>= len), and out has 8 elements.
            unsafe {
                let ymm = load_partial_ymm(&src, len);
                _mm256_storeu_ps(out.as_mut_ptr(), ymm);
            }
            assert_eq!(&out[..len], &src[..len]);
            for &val in &out[len..] {
                assert_eq!(val, 0.0f32);
            }
        }
    }

    #[test]
    fn test_store_partial_ymm_len_4() {
        // SAFETY: _mm256_set_ps loads scalar constants into a register.
        let ymm = unsafe { _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0) };
        let mut dst = [-1.0f32; 6];
        // SAFETY: dst has 6 elements (>= 4).
        unsafe {
            store_partial_ymm(ymm, &mut dst, 4);
        }
        assert_eq!(&dst[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&dst[4..], &[-1.0, -1.0]);
    }

    #[test]
    fn test_store_partial_ymm_len_5_to_8() {
        // SAFETY: _mm256_set_ps loads scalar constants into a register.
        let ymm = unsafe { _mm256_set_ps(80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0) };
        let expected = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        for len in 5..=8 {
            let mut dst = [-1.0f32; 10];
            // SAFETY: dst has 10 elements (>= len).
            unsafe {
                store_partial_ymm(ymm, &mut dst, len);
            }
            assert_eq!(&dst[..len], &expected[..len]);
            for &val in &dst[len..] {
                assert_eq!(val, -1.0f32);
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "load_partial_ymm requires len in 4..=8")]
    fn test_load_partial_ymm_underflow_panics() {
        let src = [1.0f32, 2.0, 3.0];
        // SAFETY: Testing precondition assertion panic with len < 4.
        unsafe {
            let _ = load_partial_ymm(&src, 3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "load_partial_ymm requires len in 4..=8")]
    fn test_load_partial_ymm_overflow_panics() {
        let src = [0.0f32; 10];
        // SAFETY: Testing precondition assertion panic with len > 8.
        unsafe {
            let _ = load_partial_ymm(&src, 9);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "store_partial_ymm requires len in 4..=8")]
    fn test_store_partial_ymm_underflow_panics() {
        let mut dst = [0.0f32; 4];
        // SAFETY: zeroing ymm register.
        let ymm = unsafe { _mm256_setzero_ps() };
        // SAFETY: Testing precondition assertion panic with len < 4.
        unsafe {
            store_partial_ymm(ymm, &mut dst, 3);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "store_partial_ymm requires len in 4..=8")]
    fn test_store_partial_ymm_overflow_panics() {
        let mut dst = [0.0f32; 10];
        // SAFETY: zeroing ymm register.
        let ymm = unsafe { _mm256_setzero_ps() };
        // SAFETY: Testing precondition assertion panic with len > 8.
        unsafe {
            store_partial_ymm(ymm, &mut dst, 9);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "src slice too short")]
    fn test_load_partial_ymm_short_slice_panics() {
        let src = [1.0f32, 2.0, 3.0];
        // SAFETY: Testing slice length assertion panic when src.len() < len.
        unsafe {
            let _ = load_partial_ymm(&src, 4);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "dst slice too short")]
    fn test_store_partial_ymm_short_slice_panics() {
        let mut dst = [1.0f32, 2.0, 3.0];
        // SAFETY: zeroing ymm register.
        let ymm = unsafe { _mm256_setzero_ps() };
        // SAFETY: Testing slice length assertion panic when dst.len() < len.
        unsafe {
            store_partial_ymm(ymm, &mut dst, 4);
        }
    }
}
