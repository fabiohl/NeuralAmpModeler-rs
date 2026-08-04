// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 4x f32 — AVX2/FMA kernel (f32 native weights).
//!
//! Processes `state[i] · weights[i]` for 4 interleaved output channels.
//!
//! # Precision
//! Both the scalar reference (`mul_add`) and this kernel (`_mm_fmadd_ps`) use
//! the same FMA3 fused multiply‑add instruction, producing mathematically
//! equivalent results. The 4‑way accumulator unrolling splits the summation
//! across independent chains (`acc0`..`acc3`) with a final horizontal reduction,
//! which may yield slightly different rounding (< 2 ULP) compared to the
//! strictly‑serial FMA chain of the scalar reference. No dequantization or
//! precision conversion is involved.

use crate::dot4x_simd4;
use core::arch::x86_64::*;

/// 4‑lane interleaved dot product (`weights: &[[f32; 4]]`, `state: &[f32]`) with
/// AVX2/FMA.
///
/// # Strategy
/// - Main loop processes 4 input samples per iteration using 4 independent
///   `__m128` accumulators (`acc0`..`acc3`), interleaved by index `i`, to
///   break the FMA latency chain and allow OoO execution to overlap them.
/// - Each iteration: broadcast `state[i+k]` → `_mm_set1_ps`, load
///   `weights[i+k]` → `_mm_loadu_ps`, accumulate into the corresponding
///   `_mm_fmadd_ps`.
/// - Tail (< 4 elements) falls back to a single‑accumulator loop.
/// - Final reduction: `acc0 + acc1 + acc2 + acc3`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory regions
/// are valid for unaligned load. Both slices must be accessible for reading.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_avx2(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            acc0 = _mm_fmadd_ps(w0, s0, acc0);

            let w1 = _mm_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm_fmadd_ps(w1, s1, acc1);

            let w2 = _mm_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            acc2 = _mm_fmadd_ps(w2, s2, acc2);

            let w3 = _mm_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            acc3 = _mm_fmadd_ps(w3, s3, acc3);
        });

        while i < len {
            let s = _mm_set1_ps(*state.get_unchecked(i));
            let w = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm_add_ps(acc0, acc1);
        acc2 = _mm_add_ps(acc2, acc3);
        acc0 = _mm_add_ps(acc0, acc2);

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Dual-frame 4‑lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX2/FMA.
///
/// # Strategy
/// - Main loop processes 4 input samples per iteration using 4 independent
///   `__m256` accumulators (`acc0`..`acc3`), interleaved by index `i`, to
///   break the FMA latency chain.
/// - Each iteration: load `weights[i+k]` into `__m128` → broadcast into
///   both halves of `__m256`; broadcast `state_f0[i+k]` and `state_f1[i+k]`
///   into `__m128` → blend into `__m256` via `_mm256_set_m128`.
/// - Single `_mm256_fmadd_ps` per sample accumulates both frames simultaneously
///   into the corresponding accumulator.
/// - Tail (< 4 elements) falls back to a single‑accumulator loop.
/// - Final reduction: `acc0 + acc1 + acc2 + acc3`, then split into per‑frame
///   `__m128` results via `_mm256_extractf128_ps`.
///
/// # Precision
/// Both the scalar reference (`mul_add`) and this kernel (`_mm_fmadd_ps`)
/// use FMA3 fused multiply‑add → mathematically equivalent result. Due to
/// 4‑way accumulator splitting, rounding may differ slightly (< 2 ULP) from
/// the strictly‑serial scalar chain.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_dual_avx2(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let w0_128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w0_256 = _mm256_broadcast_ps(&w0_128);
            let s_f0_0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend0 = _mm256_set_m128(s_f1_0, s_f0_0);
            acc0 = _mm256_fmadd_ps(w0_256, s_blend0, acc0);

            let w1_128 = _mm_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let w1_256 = _mm256_broadcast_ps(&w1_128);
            let s_f0_1 = _mm_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm_set1_ps(*state_f1.get_unchecked(i + 1));
            let s_blend1 = _mm256_set_m128(s_f1_1, s_f0_1);
            acc1 = _mm256_fmadd_ps(w1_256, s_blend1, acc1);

            let w2_128 = _mm_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let w2_256 = _mm256_broadcast_ps(&w2_128);
            let s_f0_2 = _mm_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm_set1_ps(*state_f1.get_unchecked(i + 2));
            let s_blend2 = _mm256_set_m128(s_f1_2, s_f0_2);
            acc2 = _mm256_fmadd_ps(w2_256, s_blend2, acc2);

            let w3_128 = _mm_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let w3_256 = _mm256_broadcast_ps(&w3_128);
            let s_f0_3 = _mm_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm_set1_ps(*state_f1.get_unchecked(i + 3));
            let s_blend3 = _mm256_set_m128(s_f1_3, s_f0_3);
            acc3 = _mm256_fmadd_ps(w3_256, s_blend3, acc3);
        });

        while i < len {
            let w128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w256 = _mm256_broadcast_ps(&w128);
            let s_f0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend = _mm256_set_m128(s_f1, s_f0);
            acc0 = _mm256_fmadd_ps(w256, s_blend, acc0);
            i += 1;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let acc_f0 = _mm256_extractf128_ps(acc0, 0);
        let acc_f1 = _mm256_extractf128_ps(acc0, 1);
        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), acc_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), acc_f1);
        (out_f0, out_f1)
    }
}

/// Fused accumulate 4‑lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state: &[f32]`, `init: &[f32; 4]`) with AVX2/FMA.
///
/// Fuses the `init` accumulator (bias + mixin) into `acc0`, avoiding an
/// extra pass over the output. Other unroll accumulators (`acc1..acc3`) are
/// zero‑initialized.
///
/// # Strategy
/// - Same loop structure as `dot_product_4x_f32_avx2`, but `acc0` starts
///   from `_mm_loadu_ps(init.as_ptr())` instead of `_mm_setzero_ps()`.
/// - Tail and reduction identical to the base kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory
/// regions are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_accumulate_avx2(
    weights: &[[f32; 4]],
    state: &[f32],
    init: &[f32; 4],
) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut acc0 = _mm_loadu_ps(init.as_ptr());
    let mut acc1 = _mm_setzero_ps();
    let mut acc2 = _mm_setzero_ps();
    let mut acc3 = _mm_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let w0 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            acc0 = _mm_fmadd_ps(w0, s0, acc0);

            let w1 = _mm_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm_fmadd_ps(w1, s1, acc1);

            let w2 = _mm_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            acc2 = _mm_fmadd_ps(w2, s2, acc2);

            let w3 = _mm_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            acc3 = _mm_fmadd_ps(w3, s3, acc3);
        });

        while i < len {
            let s = _mm_set1_ps(*state.get_unchecked(i));
            let w = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm_add_ps(acc0, acc1);
        acc2 = _mm_add_ps(acc2, acc3);
        acc0 = _mm_add_ps(acc0, acc2);

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Fused accumulate dual‑frame 4‑lane interleaved dot product
/// (`weights: &[[f32; 4]]`, `state_f0: &[f32]`, `state_f1: &[f32]`,
/// `init_f0: &[f32; 4]`, `init_f1: &[f32; 4]`) with AVX2/FMA.
///
/// Fuses the `init_f0`/`init_f1` accumulators (bias + mixin) into `acc0`,
/// avoiding an extra pass over the outputs. Other unroll accumulators
/// (`acc1..acc3`) are zero‑initialized.
///
/// # Strategy
/// - Same loop structure as `dot_product_4x_f32_dual_avx2`, but `acc0`
///   starts from `_mm256_set_m128(init_f1, init_f0)` instead of
///   `_mm256_setzero_ps()`.
/// - Tail and reduction identical to the base dual kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_4x_f32_dual_accumulate_avx2(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 4],
    init_f1: &[f32; 4],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let init_f0 = _mm_loadu_ps(init_f0.as_ptr());
    let init_f1 = _mm_loadu_ps(init_f1.as_ptr());
    let mut acc0 = _mm256_set_m128(init_f1, init_f0);
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let w0_128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w0_256 = _mm256_broadcast_ps(&w0_128);
            let s_f0_0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend0 = _mm256_set_m128(s_f1_0, s_f0_0);
            acc0 = _mm256_fmadd_ps(w0_256, s_blend0, acc0);

            let w1_128 = _mm_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let w1_256 = _mm256_broadcast_ps(&w1_128);
            let s_f0_1 = _mm_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm_set1_ps(*state_f1.get_unchecked(i + 1));
            let s_blend1 = _mm256_set_m128(s_f1_1, s_f0_1);
            acc1 = _mm256_fmadd_ps(w1_256, s_blend1, acc1);

            let w2_128 = _mm_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let w2_256 = _mm256_broadcast_ps(&w2_128);
            let s_f0_2 = _mm_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm_set1_ps(*state_f1.get_unchecked(i + 2));
            let s_blend2 = _mm256_set_m128(s_f1_2, s_f0_2);
            acc2 = _mm256_fmadd_ps(w2_256, s_blend2, acc2);

            let w3_128 = _mm_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let w3_256 = _mm256_broadcast_ps(&w3_128);
            let s_f0_3 = _mm_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm_set1_ps(*state_f1.get_unchecked(i + 3));
            let s_blend3 = _mm256_set_m128(s_f1_3, s_f0_3);
            acc3 = _mm256_fmadd_ps(w3_256, s_blend3, acc3);
        });

        while i < len {
            let w128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w256 = _mm256_broadcast_ps(&w128);
            let s_f0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend = _mm256_set_m128(s_f1, s_f0);
            acc0 = _mm256_fmadd_ps(w256, s_blend, acc0);
            i += 1;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc0 = _mm256_add_ps(acc0, acc2);

        let acc_f0 = _mm256_extractf128_ps(acc0, 0);
        let acc_f1 = _mm256_extractf128_ps(acc0, 1);
        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), acc_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), acc_f1);
        (out_f0, out_f1)
    }
}
