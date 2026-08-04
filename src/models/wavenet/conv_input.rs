// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use core::arch::x86_64::{_mm_loadu_ps, _mm_storeu_ps, _mm256_loadu_ps, _mm256_storeu_ps};

/// Stores 4 accumulator values back to the output buffer with
/// bounds‑check elision for the hot path. Extracted verbatim from single‑frame
/// convolution kernel.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path (`out_n` multiple of 4 or
/// `out_c + 3 < out_n`), `out_c + 3` must be < `out.len()`.
#[inline(always)]
pub(crate) unsafe fn store_4_accums(out: &mut [f32], out_c: usize, r: [f32; 4], out_n: usize) {
    if out_n.is_multiple_of(4) || out_c + 3 < out_n {
        let v = unsafe { _mm_loadu_ps(r.as_ptr()) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(out_c), v) };
    } else {
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        if out_c + 1 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 1) = r[1] };
        }
        if out_c + 2 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 2) = r[2] };
        }
        if out_c + 3 < out_n {
            unsafe { *out.get_unchecked_mut(out_c + 3) = r[3] };
        }
    }
}

/// Stores 8 accumulator values back to the output buffer.
///
/// # Safety
/// `out_c` must be < `out.len()`. On the fast path, `out_c + 7 < out_n`.
#[inline(always)]
#[expect(
    clippy::needless_range_loop,
    reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
)]
pub(crate) unsafe fn store_8_accums(out: &mut [f32], out_c: usize, r: [f32; 8], out_n: usize) {
    if out_n.is_multiple_of(8) || out_c + 7 < out_n {
        let v = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v) };
    } else {
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        for i in 1..8 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}

/// Stores 16 accumulator values back to the output buffer.
///
/// Three paths are used depending on the remaining output channels:
///
/// | Condition                       | Path            | Use case             |
/// |---------------------------------|-----------------|----------------------|
/// | `out_n % 16 == 0` or full block | 2×`vmovups ymm` | CH=16 (all lanes)    |
/// | `out_c + 11 < out_n`            | `ymm` + `xmm`  | CH=12 pad-to-16      |
/// | otherwise                       | scalar tail     | other partial widths |
///
/// The 8+4 path (ymm + xmm) completes the
/// pad-to-16 strategy: `select_interleave_width(12) = 16` routes CH=12 through
/// the 16-wide dot-product kernel, but the stores must also be SIMD to realize
/// the speedup. Lanes 12–15 (zero-padding) are never written.
///
/// # Safety
/// `out_c` must be `< out.len()`. On the fast path, `out_c + 15 < out.len()`.
/// On the 8+4 path, `out_c + 11 < out.len()`.
#[inline(always)]
#[expect(
    clippy::needless_range_loop,
    reason = "Range loop required for explicit SIMD lane indexing not expressible via iterator"
)]
pub(crate) unsafe fn store_16_accums(out: &mut [f32], out_c: usize, r: [f32; 16], out_n: usize) {
    if out_n.is_multiple_of(16) || out_c + 15 < out_n {
        // Full 2×ymm path: CH=16 or aligned block (all 16 lanes valid).
        let v0 = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        let v1 = unsafe { _mm256_loadu_ps(r.as_ptr().add(8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v0) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c + 8), v1) };
    } else if out_c + 11 < out_n {
        // 8+4 SIMD path: ymm (lanes 0-7) + xmm (lanes 8-11).
        // Activated for CH=12 (out_n=12, out_c=0): 2 SIMD stores instead of
        // 12 scalar writes. Lanes 12-15 (zero-padding) are not written — the
        // output buffer only has out_n=12 positions.
        //
        // Safety:
        //   - ymm write [out_c..out_c+8]: valid since out_c+11 < out_n ≤ out.len()
        //     implies out_c+7 < out.len().
        //   - xmm write [out_c+8..out_c+12]: valid since out_c+11 < out_n ≤ out.len()
        //     implies out_c+11 < out.len().
        let v0 = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        let v1 = unsafe { _mm_loadu_ps(r.as_ptr().add(8)) };
        unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(out_c), v0) };
        unsafe { _mm_storeu_ps(out.as_mut_ptr().add(out_c + 8), v1) };
    } else {
        // Scalar tail for any other partial width (e.g. CH < 12 in 16-wide layout).
        unsafe { *out.get_unchecked_mut(out_c) = r[0] };
        for i in 1..16 {
            if out_c + i < out_n {
                unsafe { *out.get_unchecked_mut(out_c + i) = r[i] };
            }
        }
    }
}
