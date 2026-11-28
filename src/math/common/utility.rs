// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! SIMD utilities for reductions and horizontal operations.

use super::kahan::kahan_add;
use crate::arch::simd::*;

/// Horizontal sum of an AVX2 (256-bit) register to scalar f32.
///
/// Performs the reduction via 128-bit lane extraction and successive additions.
/// Total instructions: ~7 (including store_ss).
///
/// # Safety
/// The caller must ensure the CPU supports AVX2 (x86_64) or NEON (aarch64).
/// The x86 project build targets x86-64-v3 (AVX2+FMA), but that is a runtime
/// CPU requirement, not a runtime check. No pointers are dereferenced: the
/// reduction and final store operate on the register argument and a stack-local
/// `f32`.
#[inline]
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn hsum_avx2(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let s128 = _mm_add_ps(lo, hi);
    #[cfg(target_arch = "aarch64")]
    {
        let pairs = _mm_hadd_ps(s128, s128);
        _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let shuf = _mm_movehdup_ps(s128);
        let sums = _mm_add_ps(s128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let r = _mm_add_ss(sums, shuf2);
        let mut out = 0.0f32;
        // SAFETY: `out` is a live stack-local `f32`; `_mm_store_ss` writes one
        // f32 to a valid, properly aligned destination. ISA: SSE is x86-64
        // baseline; aarch64 maps this helper to its NEON baseline.
        unsafe { _mm_store_ss(&mut out, r) };
        out
    }
}

/// Horizontal sum of an AVX-512 (512-bit) register to scalar f32.
///
/// Uses the native AVX-512 Foundation reduction intrinsic.
///
/// # Safety
/// The caller must ensure the CPU supports AVX-512F. The dispatched AVX-512
/// path is selected only when `has_full_avx512()` (F+VL+BW+DQ) or a validated
/// test override permits it; `#[target_feature]` does not itself check the CPU.
/// No pointers are dereferenced.
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn hsum_avx512(v: __m512) -> f32 {
    _mm512_reduce_add_ps(v)
}

/// Horizontal sum of an f32 buffer via AVX2.
///
/// # Safety
/// `ptr` must be non-null, aligned to `align_of::<f32>()`, and valid for `len`
/// consecutive reads of initialized `f32` values. The caller must ensure the
/// CPU supports AVX2 (x86_64) or NEON (aarch64); `#[target_feature]` is not a
/// runtime check. x86 callers must provide an x86-64-v3-compatible CPU
/// (AVX2+FMA); NEON is baseline on aarch64.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2"))]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub unsafe fn horizontal_sum_avx2(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm256_setzero_ps();

    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` and the caller guarantees `ptr` is valid for
        // `len` reads, so the unaligned 8-f32 load is in bounds. ISA: caller
        // must provide x86-64-v3 (AVX2+FMA) or aarch64 NEON; no runtime check.
        let v = unsafe { _mm256_loadu_ps(ptr.add(i)) };
        sum_v = _mm256_add_ps(sum_v, v);
        i += 8;
    }

    // SAFETY: `hsum_avx2` never dereferences memory. ISA: caller must provide
    // x86-64-v3 (AVX2) or aarch64 NEON; no runtime ISA check at this call.
    let mut total = unsafe { hsum_avx2(sum_v) };
    let mut compensation = 0.0f32;

    while i < len {
        // SAFETY: `i < len` and the caller guarantees `ptr` is valid for `len`
        // initialized f32 reads.
        (total, compensation) = kahan_add(total, compensation, unsafe { *ptr.add(i) });
        i += 1;
    }

    total
}

/// Horizontal sum of an f32 buffer via AVX-512.
///
/// # Safety
/// `ptr` must be non-null, aligned to `align_of::<f32>()`, and valid for `len`
/// consecutive reads of initialized `f32` values. The caller must ensure the
/// CPU supports AVX-512F; reachable dispatch is gated by `has_full_avx512()`
/// (F+VL+BW+DQ) or a validated test override, not by `#[target_feature]`.
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[target_feature(enable = "avx512f")]
pub unsafe fn horizontal_sum_avx512(ptr: *const f32, len: usize) -> f32 {
    let mut i = 0;
    let mut sum_v = _mm512_setzero_ps();

    while i + 16 <= len {
        // SAFETY: `i + 16 <= len` and the caller guarantees `ptr` is valid for
        // `len` reads, so the unaligned 16-f32 load is in bounds. ISA:
        // AVX-512F, reachable only after `has_full_avx512()` / validated
        // override; `#[target_feature]` is not a runtime check.
        let v = unsafe { _mm512_loadu_ps(ptr.add(i)) };
        sum_v = _mm512_add_ps(sum_v, v);
        i += 16;
    }

    // SAFETY: `hsum_avx512` never dereferences memory. ISA: same AVX-512F
    // dispatch guarantee as above.
    let mut total = unsafe { hsum_avx512(sum_v) };
    let compensation = 0.0f32;

    if i < len {
        let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
        // SAFETY: `i < len`; the mask selects exactly the remaining `len - i`
        // elements and the caller guarantees `ptr` is valid for `len` reads.
        // ISA: AVX-512F, reachable only after full AVX-512 detection / validated
        // override; `#[target_feature]` does not check CPU support.
        let v = unsafe { _mm512_maskz_loadu_ps(mask, ptr.add(i)) };
        // SAFETY: same AVX-512F guarantee as above; reduction reads no memory.
        (total, _) = kahan_add(total, compensation, unsafe { hsum_avx512(v) });
    }

    total
}
