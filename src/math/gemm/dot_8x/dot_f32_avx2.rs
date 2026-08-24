// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 8x f32 — AVX2/FMA kernel (f32 native weights).
//!
//! Processes `state[i] · weights[i]` for 8 interleaved output channels,
//! packing all 8 channels in a single `__m256` register.
//!
//! # Precision
//! Both the scalar reference (`mul_add`) and this kernel (`_mm256_fmadd_ps`) use
//! the same FMA3 fused multiply‑add instruction, producing mathematically
//! equivalent results. The 4‑way accumulator unrolling splits the summation
//! across independent chains (`acc0`..`acc3`) with a final horizontal reduction,
//! which may yield slightly different rounding (< 2 ULP) compared to the
//! strictly‑serial FMA chain of the scalar reference. No dequantization or
//! precision conversion is involved.

use crate::dot4x_simd4;
use core::arch::x86_64::*;

/// 8‑lane interleaved dot product (`weights: &[[f32; 8]]`, `state: &[f32]`) with
/// AVX2/FMA.
///
/// # Strategy
/// - 8 weights per row loaded into a single `__m256`.
/// - State scalar broadcast to all 8 lanes via `_mm256_set1_ps`.
/// - Main loop processes 4 input samples per iteration using 4 independent
///   `__m256` accumulators (`acc0`..`acc3`), interleaved by index `i`, to
///   break the FMA latency chain and allow OoO execution to overlap them.
/// - Tail (< 4 elements) falls back to a single‑accumulator loop.
/// - Final reduction: `acc0 + acc1 + acc2 + acc3`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_8x_f32_avx2(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
    let len = state.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    // SAFETY: `i < len = state.len()` (loop guard) and the caller guarantees
    // `weights.len() >= state.len()`, so `state.get_unchecked(i)` and the
    // 8-f32 `_mm256_loadu_ps` loads from `weights[i]` stay in bounds.
    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm256_set1_ps(*state.get_unchecked(i));
            acc0 = _mm256_fmadd_ps(w0, s0, acc0);

            let w1 = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm256_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm256_fmadd_ps(w1, s1, acc1);

            let w2 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm256_set1_ps(*state.get_unchecked(i + 2));
            acc2 = _mm256_fmadd_ps(w2, s2, acc2);

            let w3 = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s3 = _mm256_set1_ps(*state.get_unchecked(i + 3));
            acc3 = _mm256_fmadd_ps(w3, s3, acc3);
        });

        while i < len {
            let s = _mm256_set1_ps(*state.get_unchecked(i));
            let w = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm256_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Dual‑frame 8‑lane interleaved dot product (`weights: &[[f32; 8]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX2/FMA.
///
/// # Strategy
/// - 8 weights per row loaded into a single `__m256` once per iteration.
/// - State scalars broadcast separately for each frame via `_mm256_set1_ps`.
/// - Main loop processes 4 input samples per iteration using 4 independent
///   pairs of `__m256` accumulators for each frame (`acc_f0_0..3`,
///   `acc_f1_0..3`), totalling 8 accumulator registers to break the FMA
///   latency chain while computing both frames from a single weight load.
/// - Tail (< 4 elements) falls back to single‑accumulator‑pair loop.
/// - Final reduction: sum each frame’s 4 accumulators independently, store
///   as `([f32; 8], [f32; 8])`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_8x_f32_dual_avx2(
    weights: &[[f32; 8]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 8], [f32; 8]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0_0 = _mm256_setzero_ps();
    let mut acc_f0_1 = _mm256_setzero_ps();
    let mut acc_f0_2 = _mm256_setzero_ps();
    let mut acc_f0_3 = _mm256_setzero_ps();
    let mut acc_f1_0 = _mm256_setzero_ps();
    let mut acc_f1_1 = _mm256_setzero_ps();
    let mut acc_f1_2 = _mm256_setzero_ps();
    let mut acc_f1_3 = _mm256_setzero_ps();
    let mut i = 0;

    // SAFETY: `len` is the minimum of `weights.len()`, `state_f0.len()`, and
    // `state_f1.len()`; the `dot4x_simd4!` guard keeps `i < len`, so all
    // unchecked reads and 8-f32 loads stay in bounds.
    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm256_fmadd_ps(w0, s_f0_0, acc_f0_0);
            acc_f1_0 = _mm256_fmadd_ps(w0, s_f1_0, acc_f1_0);

            let w1 = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_1 = _mm256_fmadd_ps(w1, s_f0_1, acc_f0_1);
            acc_f1_1 = _mm256_fmadd_ps(w1, s_f1_1, acc_f1_1);

            let w2 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_2 = _mm256_fmadd_ps(w2, s_f0_2, acc_f0_2);
            acc_f1_2 = _mm256_fmadd_ps(w2, s_f1_2, acc_f1_2);

            let w3 = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_3 = _mm256_fmadd_ps(w3, s_f0_3, acc_f0_3);
            acc_f1_3 = _mm256_fmadd_ps(w3, s_f1_3, acc_f1_3);
        });

        while i < len {
            let w = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm256_fmadd_ps(w, s_f0, acc_f0_0);
            acc_f1_0 = _mm256_fmadd_ps(w, s_f1, acc_f1_0);
            i += 1;
        }

        acc_f0_0 = _mm256_add_ps(acc_f0_0, acc_f0_1);
        acc_f0_2 = _mm256_add_ps(acc_f0_2, acc_f0_3);
        acc_f0_0 = _mm256_add_ps(acc_f0_0, acc_f0_2);

        acc_f1_0 = _mm256_add_ps(acc_f1_0, acc_f1_1);
        acc_f1_2 = _mm256_add_ps(acc_f1_2, acc_f1_3);
        acc_f1_0 = _mm256_add_ps(acc_f1_0, acc_f1_2);

        let mut out_f0 = [0.0f32; 8];
        let mut out_f1 = [0.0f32; 8];
        _mm256_storeu_ps(out_f0.as_mut_ptr(), acc_f0_0);
        _mm256_storeu_ps(out_f1.as_mut_ptr(), acc_f1_0);
        (out_f0, out_f1)
    }
}

/// Fused accumulate 8‑lane interleaved dot product (`weights: &[[f32; 8]]`,
/// `state: &[f32]`, `init: &[f32; 8]`) with AVX2/FMA.
///
/// Fuses the `init` accumulator (bias + mixin) into `acc0`, avoiding an
/// extra pass over the output. Other unroll accumulators (`acc1..acc3`) are
/// zero‑initialized.
///
/// # Strategy
/// - Same loop structure as `dot_product_8x_f32_avx2`, but `acc0` starts
///   from `_mm256_loadu_ps(init.as_ptr())` instead of `_mm256_setzero_ps()`.
/// - Tail and reduction identical to the base kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory
/// regions are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_8x_f32_accumulate_avx2(
    weights: &[[f32; 8]],
    state: &[f32],
    init: &[f32; 8],
) -> [f32; 8] {
    let len = state.len();
    let mut acc0 = _mm256_loadu_ps(init.as_ptr());
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    // SAFETY: `i < len = state.len()` (loop guard) and the caller guarantees
    // `weights.len() >= state.len()`, so `state.get_unchecked(i)` and the
    // 8-f32 `_mm256_loadu_ps` loads from `weights[i]` stay in bounds.
    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm256_set1_ps(*state.get_unchecked(i));
            acc0 = _mm256_fmadd_ps(w0, s0, acc0);

            let w1 = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm256_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm256_fmadd_ps(w1, s1, acc1);

            let w2 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm256_set1_ps(*state.get_unchecked(i + 2));
            acc2 = _mm256_fmadd_ps(w2, s2, acc2);

            let w3 = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s3 = _mm256_set1_ps(*state.get_unchecked(i + 3));
            acc3 = _mm256_fmadd_ps(w3, s3, acc3);
        });

        while i < len {
            let s = _mm256_set1_ps(*state.get_unchecked(i));
            let w = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm256_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Fused accumulate dual‑frame 8‑lane interleaved dot product
/// (`weights: &[[f32; 8]]`, `state_f0: &[f32]`, `state_f1: &[f32]`,
/// `init_f0: &[f32; 8]`, `init_f1: &[f32; 8]`) with AVX2/FMA.
///
/// Fuses the `init_f0`/`init_f1` accumulators (bias + mixin) into
/// `acc_f0_0`/`acc_f1_0`, avoiding an extra pass over the outputs. Other
/// unroll accumulator pairs (`acc_f{0,1}_1..3`) are zero‑initialized.
///
/// # Strategy
/// - Same loop structure as `dot_product_8x_f32_dual_avx2`, but
///   `acc_f0_0`/`acc_f1_0` start from init loads instead of
///   `_mm256_setzero_ps()`.
/// - Tail and reduction identical to the base dual kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_8x_f32_dual_accumulate_avx2(
    weights: &[[f32; 8]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 8],
    init_f1: &[f32; 8],
) -> ([f32; 8], [f32; 8]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0_0 = _mm256_loadu_ps(init_f0.as_ptr());
    let mut acc_f0_1 = _mm256_setzero_ps();
    let mut acc_f0_2 = _mm256_setzero_ps();
    let mut acc_f0_3 = _mm256_setzero_ps();
    let mut acc_f1_0 = _mm256_loadu_ps(init_f1.as_ptr());
    let mut acc_f1_1 = _mm256_setzero_ps();
    let mut acc_f1_2 = _mm256_setzero_ps();
    let mut acc_f1_3 = _mm256_setzero_ps();
    let mut i = 0;

    // SAFETY: `len` is the minimum of `weights.len()`, `state_f0.len()`, and
    // `state_f1.len()`; the `dot4x_simd4!` guard keeps `i < len`, so all
    // unchecked reads and 8-f32 loads stay in bounds.
    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm256_fmadd_ps(w0, s_f0_0, acc_f0_0);
            acc_f1_0 = _mm256_fmadd_ps(w0, s_f1_0, acc_f1_0);

            let w1 = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_1 = _mm256_fmadd_ps(w1, s_f0_1, acc_f0_1);
            acc_f1_1 = _mm256_fmadd_ps(w1, s_f1_1, acc_f1_1);

            let w2 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_2 = _mm256_fmadd_ps(w2, s_f0_2, acc_f0_2);
            acc_f1_2 = _mm256_fmadd_ps(w2, s_f1_2, acc_f1_2);

            let w3 = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_3 = _mm256_fmadd_ps(w3, s_f0_3, acc_f0_3);
            acc_f1_3 = _mm256_fmadd_ps(w3, s_f1_3, acc_f1_3);
        });

        while i < len {
            let w = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm256_fmadd_ps(w, s_f0, acc_f0_0);
            acc_f1_0 = _mm256_fmadd_ps(w, s_f1, acc_f1_0);
            i += 1;
        }

        acc_f0_0 = _mm256_add_ps(acc_f0_0, acc_f0_1);
        acc_f0_2 = _mm256_add_ps(acc_f0_2, acc_f0_3);
        acc_f0_0 = _mm256_add_ps(acc_f0_0, acc_f0_2);

        acc_f1_0 = _mm256_add_ps(acc_f1_0, acc_f1_1);
        acc_f1_2 = _mm256_add_ps(acc_f1_2, acc_f1_3);
        acc_f1_0 = _mm256_add_ps(acc_f1_0, acc_f1_2);

        let mut out_f0 = [0.0f32; 8];
        let mut out_f1 = [0.0f32; 8];
        _mm256_storeu_ps(out_f0.as_mut_ptr(), acc_f0_0);
        _mm256_storeu_ps(out_f1.as_mut_ptr(), acc_f1_0);
        (out_f0, out_f1)
    }
}
