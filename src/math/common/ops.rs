// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Basic mathematical operations and low-level SIMD utilities.

use crate::arch::simd::*;

/// Converts a vector of f32 numbers (normal) to bf16 (compact) via AVX-512.
/// The bf16 format takes up half the space but maintains the range of f32 numbers,
/// making it ideal for fast AI models.
///
/// # Safety
/// `src` and `dest` must be valid slices. `dest` must have at least `src.len()` elements.
/// AVX-512F+VL target feature must be available (enforced by CPU dispatch).
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[target_feature(enable = "avx512f,avx512vl")]
// SAFETY: Caller must ensure AVX-512F+VL is available and src/dest are valid
// non-overlapping slices. This `pub unsafe fn` is the trust boundary.
pub unsafe fn f32_to_bf16_avx512(src: &[f32], dest: &mut [u16]) {
    let n = core::cmp::min(src.len(), dest.len());
    let mut i = 0;
    // Process 16 conversions at once.
    while i + 16 <= n {
        // SAFETY: i+16 ≤ n (loop condition), so src[..i+16] and dest[..i+16]
        // are in bounds. _mm512_loadu_ps is an unaligned load — no alignment precondition.
        unsafe {
            let v = _mm512_loadu_ps(src.as_ptr().add(i)); // Load 16 f32 numbers.
            let v_i = _mm512_castps_si512(v); // Treat as integers for bit manipulation.
            let v_shifted = _mm512_srli_epi32(v_i, 16); // Discard the least important part (extra precision).
            let packed = _mm512_cvtepi32_epi16(v_shifted); // Compact to 16 bits each.
            _mm256_storeu_si256(dest.as_mut_ptr().add(i) as *mut __m256i, packed); // Save 16 bf16 numbers.
        }
        i += 16;
    }
    // Convert the remainder manually.
    while i < n {
        // SAFETY: i < n (loop condition), so both src.get_unchecked(i) and
        // dest.get_unchecked_mut(i) are within bounds.
        unsafe {
            *dest.get_unchecked_mut(i) = ((*src.get_unchecked(i)).to_bits() >> 16) as u16;
        }
        i += 1;
    }
}

/// Adaptive prefetch based on dilation (Causal Conv1D).
///
/// # Safety
/// The `ptr` pointer must be valid or within the safe margin of the buffer.
#[inline(always)]
// SAFETY: Caller must ensure ptr is valid and the CPU supports SSE (guaranteed on x86-64).
// _mm_prefetch is a cache hint that never dereferences.
pub unsafe fn adaptive_prefetch_f32(ptr: *const f32, dilation: usize) {
    if dilation <= 8 {
        // Hardware prefetcher dominates.
    } else if dilation <= 64 {
        // Fetch to L1 (imminent reuse).
        // SAFETY: _mm_prefetch is a cache hint — it never dereferences. ptr cast to
        // *const i8 is sound; the hardware cache line address is identical.
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(ptr as *const i8);
        }
    } else {
        // Massive dilations: fetch to L2 to spare L1 from aggressive eviction.
        // SAFETY: Same rationale as T0 — prefetch hint only, ptr is valid.
        unsafe {
            _mm_prefetch::<_MM_HINT_T1>(ptr as *const i8);
        }
    }
}

/// Adaptive 2-stage prefetch for extreme dilations (Causal Conv1D).
///
/// # Safety
/// The pointers must be valid or within the safe margin of the buffer.
#[inline(always)]
// SAFETY: Caller must ensure ptr_next and ptr_next_next are valid pointers.
// _mm_prefetch only hints the cache and never dereferences.
pub unsafe fn adaptive_prefetch_2stage_f32(
    ptr_next: *const f32,
    ptr_next_next: *const f32,
    dilation: usize,
) {
    if dilation >= 128 {
        // SAFETY: Both _mm_prefetch calls are cache hints that never dereference.
        // ptr_next and ptr_next_next are valid per caller precondition.
        unsafe {
            // Fetch the next tap to L1 (immediate use in the next k)
            _mm_prefetch::<_MM_HINT_T0>(ptr_next as *const i8);
            // Fetch the subsequent tap to L2 (prepares for k+2)
            _mm_prefetch::<_MM_HINT_T1>(ptr_next_next as *const i8);
        }
    } else {
        // Fallback to simple prefetch for smaller dilations
        // SAFETY: adaptive_prefetch_f32 requires valid ptr; ptr_next satisfies this
        // per caller precondition.
        unsafe {
            adaptive_prefetch_f32(ptr_next, dilation);
        }
    }
}

/// Speculative prefetch to L1 cache (T0 hint), avoiding pointer arithmetic UB.
///
/// On x86-64 `_mm_prefetch` is a non-faulting, memory-safe hint. On aarch64 it
/// is a documented no-op: the `prfm` emitted by `asm!` acted as a scheduling
/// barrier around NEON loads and was never measured as a win on Apple or
/// Neoverse, so the x86 policy is not transplanted to that ISA.
#[inline(always)]
pub fn prefetch_t0<T>(base: *const T, offset: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: prefetch does not dereference the pointer, so using wrapping_add is UB-free even if the address goes out of bounds.
        unsafe {
            crate::arch::simd::_mm_prefetch::<{ crate::arch::simd::_MM_HINT_T0 }>(
                base.wrapping_add(offset).cast(),
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (base, offset);
}

/// Speculative prefetch to L2 cache (T1 hint), avoiding pointer arithmetic UB.
///
/// On x86-64 `_mm_prefetch` is a non-faulting, memory-safe hint. On aarch64 it
/// is a documented no-op for the same reason as [`prefetch_t0`].
#[inline(always)]
pub fn prefetch_t1<T>(base: *const T, offset: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: prefetch does not dereference the pointer, so using wrapping_add is UB-free even if the address goes out of bounds.
        unsafe {
            crate::arch::simd::_mm_prefetch::<{ crate::arch::simd::_MM_HINT_T1 }>(
                base.wrapping_add(offset).cast(),
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = (base, offset);
}

/// Simple prefetch strategy for small/medium dilations.
///
/// Stride-guarded: the fixed +16 f32 (64-byte line) hint only pays off when
/// it lands on the next causal tap's row, i.e. when the tap stride
/// `step = dilation * in_ch` fits inside one cache line. For larger strides
/// the prefetched line is never read by any tap (misaligned junk work on a
/// load port slot), and for the short strides the hardware prefetcher already
/// streams the rows. Measured (conv1d_hotpath_bench, rustc 1.98.1, Zen 2 5700U):
/// removing prefetch with stride > 16 improves 8-wide kernels by ~9%
/// (d <= 8), is neutral on 4-wide (the hardware streamer dominates), and regresses
/// +3..5% if removed on aligned cases (16x16 d=1, 12x12 d=1) — which is why
/// the threshold applies to stride rather than pure dilation: large dilations with
/// small IN remain misaligned, whereas dilation 1 with IN 12/16 remains beneficial.
#[inline(always)]
pub fn prefetch_strategy_simple(
    base_ptr: *const f32,
    step: usize,
    _k: usize,
    _k_limit: usize,
    _dilation: usize,
) {
    if step <= 16 {
        // SAFETY: Speculative prefetch using wrapping_add is safe from out-of-bounds UB.
        prefetch_t0(base_ptr, 16);
    }
}

/// 2-stage prefetch strategy for extreme dilations.
#[inline(always)]
pub fn prefetch_strategy_2stage(
    base_ptr: *const f32,
    step: usize,
    k: usize,
    k_limit: usize,
    _dilation: usize,
) {
    if k + 1 < k_limit {
        // SAFETY: Speculative prefetch using wrapping_add is safe from out-of-bounds UB.
        prefetch_t0(base_ptr, step);

        if k + 2 < k_limit {
            prefetch_t1(base_ptr, 2 * step);
        }
    }
}

/// Sets the FTZ (bit 15) and DAZ (bit 6) control bits of the calling thread's
/// MXCSR control word (x86_64: `stmxcsr`/`ldmxcsr` on a stack-local `u32`).
///
/// # Safety
///
/// - The operation reads and writes **no process memory** (the MXCSR value is
///   staged in a stack local), performs no allocation, cannot unwind, and
///   terminates in constant time — callable from lock-free real-time contexts.
/// - The mutation is confined to the **calling thread**: MXCSR is per-thread
///   CPU state; other threads' floating-point environments are untouched.
/// - The operation is **idempotent**: it ORs `0x8040` into the current control
///   word, so repeated calls cannot accumulate state and a single call fully
///   re-asserts the policy.
/// - Numerical effect on the caller's thread: denormal inputs/outputs of
///   subsequent scalar/SSE floating point code on this thread are flushed to
///   zero in constant time instead of being handled as microcoded denormals.
///   The caller must intend exactly this — this is why the function must
///   remain `unsafe` rather than `safe`: calling it as safe from library code
///   would let one crate silently change the floating-point environment (and
///   therefore the numerical results of denormal-tail math) of unrelated code
///   scheduled later on the same thread.
/// - Preconditions: zero. SSE2 is guaranteed by the x86-64 baseline (the crate
///   targets x86-64-v3), so no runtime feature check is ever needed.
#[cfg(target_arch = "x86_64")]
pub unsafe fn set_daz_ftz() {
    // SAFETY: MXCSR manipulation via stmxcsr/ldmxcsr is always safe on x86-64
    // (SSE2 is guaranteed by the x86-64-v3 target). The asm! block uses a properly
    // aligned local variable; bits 0x8040 (DAZ | FTZ) are valid MXCSR control flags.
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{0}]", in(reg) &mut mxcsr);
        mxcsr |= 0x8040;
        core::arch::asm!("ldmxcsr [{0}]", in(reg) &mxcsr);
    }
}

/// Sets FPCR.FZ (bit 24) for the calling thread (aarch64: `mrs`/`msr fpcr`).
/// FZ flushes both denormal inputs and outputs of single- and
/// double-precision arithmetic, the equivalent of x86 DAZ + FTZ.
///
/// # Safety
///
/// Same contract as the x86_64 version: no process memory is touched, the
/// change is confined to the calling thread, the call is idempotent, and it
/// deliberately changes the thread's floating-point environment (denormals
/// flush to zero), which is why it stays `unsafe`. NEON/FP is part of the
/// aarch64 baseline, so there are no runtime preconditions.
#[cfg(target_arch = "aarch64")]
pub unsafe fn set_daz_ftz() {
    // SAFETY: reading and writing FPCR is unprivileged on aarch64 and only
    // affects the calling thread's floating-point control state; bit 24 (FZ)
    // is a valid control bit.
    unsafe {
        let mut fpcr: u64;
        core::arch::asm!("mrs {0}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
        fpcr |= 1 << 24;
        core::arch::asm!("msr fpcr, {0}", in(reg) fpcr, options(nomem, nostack, preserves_flags));
    }
}

/// Flushes a subnormal `f32` to signed zero, matching FTZ/DAZ semantics.
///
/// Scalar tails cannot rely on the thread's FP control register (the caller may
/// not have enabled FTZ/DAZ yet), so they apply the same policy explicitly.
/// Signed zero is preserved: `-1e-40` becomes `-0.0`, not `+0.0`.
#[inline]
pub(crate) fn flush_subnormal(x: f32) -> f32 {
    if x.abs() < f32::MIN_POSITIVE {
        0.0_f32.copysign(x)
    } else {
        x
    }
}
