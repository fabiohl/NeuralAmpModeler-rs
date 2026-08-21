// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 4x f32 — AVX-512 VL256 kernel (256-bit EVEX, 32 YMM registers, zero spill).
//!
//! Processes `state[i] · weights[i]` for 4 interleaved output channels using
//! 256-bit EVEX vector instructions (`__m256`) and extended register file `ymm0`..`ymm31`.
//!
//! # Precision
//! Both the scalar reference (`mul_add`) and this kernel (`_mm256_fmadd_ps` / `_mm_fmadd_ps`)
//! use FMA3 fused multiply-add instructions. Summation order splits across 8 independent
//! accumulator chains (`acc0`..`acc7`) for latency hiding, which may yield minor rounding
//! differences (< 2 ULP, MSE < 1e-12) compared to the strictly serial scalar chain.

use core::arch::x86_64::*;

/// 4-lane interleaved dot product (`weights: &[[f32; 4]]`, `state: &[f32]`) with
/// AVX-512 VL256 (256-bit EVEX).
///
/// # Strategy
/// - Main loop processes 16 input samples per iteration using 8 independent `__m256`
///   accumulators (`acc0`..`acc7`). Each 256-bit accumulator processes 2 consecutive
///   samples (2 weight rows = 8 f32 values) in a single `_mm256_fmadd_ps`.
/// - 8 independent accumulator chains saturate the 2 FMA execution ports and break the
///   FMA dependency latency without spilling registers to the stack.
/// - Unrolls 8, 4, and 2 samples in the tail, with a single-sample fallback for the remainder.
/// - Final reduction: pairwise tree reduction `acc0 + ... + acc7`, followed by 256-to-128 bit
///   reduction `_mm_add_ps(low, high)`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions are valid
/// for unaligned load. Both slices must be accessible for reading.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_avx512vl(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        // 16-element main unroll (8 pairs)
        while i + 16 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            let w45 = _mm256_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let s4 = _mm_set1_ps(*state.get_unchecked(i + 4));
            let s5 = _mm_set1_ps(*state.get_unchecked(i + 5));
            let s45 = _mm256_set_m128(s5, s4);
            acc2 = _mm256_fmadd_ps(w45, s45, acc2);

            let w67 = _mm256_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let s6 = _mm_set1_ps(*state.get_unchecked(i + 6));
            let s7 = _mm_set1_ps(*state.get_unchecked(i + 7));
            let s67 = _mm256_set_m128(s7, s6);
            acc3 = _mm256_fmadd_ps(w67, s67, acc3);

            let w89 = _mm256_loadu_ps(weights.as_ptr().add(i + 8) as *const f32);
            let s8 = _mm_set1_ps(*state.get_unchecked(i + 8));
            let s9 = _mm_set1_ps(*state.get_unchecked(i + 9));
            let s89 = _mm256_set_m128(s9, s8);
            acc4 = _mm256_fmadd_ps(w89, s89, acc4);

            let w1011 = _mm256_loadu_ps(weights.as_ptr().add(i + 10) as *const f32);
            let s10 = _mm_set1_ps(*state.get_unchecked(i + 10));
            let s11 = _mm_set1_ps(*state.get_unchecked(i + 11));
            let s1011 = _mm256_set_m128(s11, s10);
            acc5 = _mm256_fmadd_ps(w1011, s1011, acc5);

            let w1213 = _mm256_loadu_ps(weights.as_ptr().add(i + 12) as *const f32);
            let s12 = _mm_set1_ps(*state.get_unchecked(i + 12));
            let s13 = _mm_set1_ps(*state.get_unchecked(i + 13));
            let s1213 = _mm256_set_m128(s13, s12);
            acc6 = _mm256_fmadd_ps(w1213, s1213, acc6);

            let w1415 = _mm256_loadu_ps(weights.as_ptr().add(i + 14) as *const f32);
            let s14 = _mm_set1_ps(*state.get_unchecked(i + 14));
            let s15 = _mm_set1_ps(*state.get_unchecked(i + 15));
            let s1415 = _mm256_set_m128(s15, s14);
            acc7 = _mm256_fmadd_ps(w1415, s1415, acc7);

            i += 16;
        }

        // 8-element unroll (4 pairs)
        while i + 8 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            let w45 = _mm256_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let s4 = _mm_set1_ps(*state.get_unchecked(i + 4));
            let s5 = _mm_set1_ps(*state.get_unchecked(i + 5));
            let s45 = _mm256_set_m128(s5, s4);
            acc2 = _mm256_fmadd_ps(w45, s45, acc2);

            let w67 = _mm256_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let s6 = _mm_set1_ps(*state.get_unchecked(i + 6));
            let s7 = _mm_set1_ps(*state.get_unchecked(i + 7));
            let s67 = _mm256_set_m128(s7, s6);
            acc3 = _mm256_fmadd_ps(w67, s67, acc3);

            i += 8;
        }

        // 4-element unroll (2 pairs)
        while i + 4 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            i += 4;
        }

        // 2-element unroll (1 pair)
        while i + 2 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            i += 2;
        }

        // Pairwise tree reduction across 8 accumulators
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);

        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);

        acc0 = _mm256_add_ps(acc0, acc4);

        // Fold 256-bit to 128-bit (lower 4 floats + upper 4 floats)
        let low128 = _mm256_castps256_ps128(acc0);
        let high128 = _mm256_extractf128_ps(acc0, 1);
        let mut res128 = _mm_add_ps(low128, high128);

        // Remaining 1 sample tail
        if i < len {
            let w = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s = _mm_set1_ps(*state.get_unchecked(i));
            res128 = _mm_fmadd_ps(w, s, res128);
        }

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), res128);
        out
    }
}

/// Dual-frame 4-lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX-512 VL256 (256-bit EVEX).
///
/// # Strategy
/// - Main loop processes 8 input samples per iteration using 8 independent `__m256`
///   accumulators (`acc0`..`acc7`).
/// - Each iteration: `weights[i+k]` loaded into `__m128` and broadcast to both 128-bit halves
///   of `__m256`. `state_f0[i+k]` and `state_f1[i+k]` are broadcast and blended via
///   `_mm256_set_m128`.
/// - Single `_mm256_fmadd_ps` per sample accumulates both frames simultaneously into the
///   dedicated accumulator register.
/// - Unrolls 4 and 1 samples in the tail.
/// - Final reduction: `acc0 + ... + acc7`, split into frame 0 and frame 1 via `_mm256_extractf128_ps`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_dual_avx512vl(
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
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        // 8-sample unroll across 8 accumulators
        while i + 8 <= len {
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

            let w4_128 = _mm_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let w4_256 = _mm256_broadcast_ps(&w4_128);
            let s_f0_4 = _mm_set1_ps(*state_f0.get_unchecked(i + 4));
            let s_f1_4 = _mm_set1_ps(*state_f1.get_unchecked(i + 4));
            let s_blend4 = _mm256_set_m128(s_f1_4, s_f0_4);
            acc4 = _mm256_fmadd_ps(w4_256, s_blend4, acc4);

            let w5_128 = _mm_loadu_ps(weights.as_ptr().add(i + 5) as *const f32);
            let w5_256 = _mm256_broadcast_ps(&w5_128);
            let s_f0_5 = _mm_set1_ps(*state_f0.get_unchecked(i + 5));
            let s_f1_5 = _mm_set1_ps(*state_f1.get_unchecked(i + 5));
            let s_blend5 = _mm256_set_m128(s_f1_5, s_f0_5);
            acc5 = _mm256_fmadd_ps(w5_256, s_blend5, acc5);

            let w6_128 = _mm_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let w6_256 = _mm256_broadcast_ps(&w6_128);
            let s_f0_6 = _mm_set1_ps(*state_f0.get_unchecked(i + 6));
            let s_f1_6 = _mm_set1_ps(*state_f1.get_unchecked(i + 6));
            let s_blend6 = _mm256_set_m128(s_f1_6, s_f0_6);
            acc6 = _mm256_fmadd_ps(w6_256, s_blend6, acc6);

            let w7_128 = _mm_loadu_ps(weights.as_ptr().add(i + 7) as *const f32);
            let w7_256 = _mm256_broadcast_ps(&w7_128);
            let s_f0_7 = _mm_set1_ps(*state_f0.get_unchecked(i + 7));
            let s_f1_7 = _mm_set1_ps(*state_f1.get_unchecked(i + 7));
            let s_blend7 = _mm256_set_m128(s_f1_7, s_f0_7);
            acc7 = _mm256_fmadd_ps(w7_256, s_blend7, acc7);

            i += 8;
        }

        // 4-sample unroll
        while i + 4 <= len {
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

            i += 4;
        }

        // Remaining sample-by-sample tail
        while i < len {
            let w128 = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w256 = _mm256_broadcast_ps(&w128);
            let s_f0 = _mm_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm_set1_ps(*state_f1.get_unchecked(i));
            let s_blend = _mm256_set_m128(s_f1, s_f0);
            acc0 = _mm256_fmadd_ps(w256, s_blend, acc0);
            i += 1;
        }

        // Pairwise tree reduction
        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);

        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);

        acc0 = _mm256_add_ps(acc0, acc4);

        let acc_f0 = _mm256_extractf128_ps(acc0, 0);
        let acc_f1 = _mm256_extractf128_ps(acc0, 1);
        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), acc_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), acc_f1);
        (out_f0, out_f1)
    }
}

/// Fused accumulate 4-lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state: &[f32]`, `init: &[f32; 4]`) with AVX-512 VL256 (256-bit EVEX).
///
/// Fuses the `init` accumulator (bias + mixin) into the reduction, avoiding an
/// extra vector pass over the output.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_accumulate_avx512vl(
    weights: &[[f32; 4]],
    state: &[f32],
    init: &[f32; 4],
) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 16 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            let w45 = _mm256_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let s4 = _mm_set1_ps(*state.get_unchecked(i + 4));
            let s5 = _mm_set1_ps(*state.get_unchecked(i + 5));
            let s45 = _mm256_set_m128(s5, s4);
            acc2 = _mm256_fmadd_ps(w45, s45, acc2);

            let w67 = _mm256_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let s6 = _mm_set1_ps(*state.get_unchecked(i + 6));
            let s7 = _mm_set1_ps(*state.get_unchecked(i + 7));
            let s67 = _mm256_set_m128(s7, s6);
            acc3 = _mm256_fmadd_ps(w67, s67, acc3);

            let w89 = _mm256_loadu_ps(weights.as_ptr().add(i + 8) as *const f32);
            let s8 = _mm_set1_ps(*state.get_unchecked(i + 8));
            let s9 = _mm_set1_ps(*state.get_unchecked(i + 9));
            let s89 = _mm256_set_m128(s9, s8);
            acc4 = _mm256_fmadd_ps(w89, s89, acc4);

            let w1011 = _mm256_loadu_ps(weights.as_ptr().add(i + 10) as *const f32);
            let s10 = _mm_set1_ps(*state.get_unchecked(i + 10));
            let s11 = _mm_set1_ps(*state.get_unchecked(i + 11));
            let s1011 = _mm256_set_m128(s11, s10);
            acc5 = _mm256_fmadd_ps(w1011, s1011, acc5);

            let w1213 = _mm256_loadu_ps(weights.as_ptr().add(i + 12) as *const f32);
            let s12 = _mm_set1_ps(*state.get_unchecked(i + 12));
            let s13 = _mm_set1_ps(*state.get_unchecked(i + 13));
            let s1213 = _mm256_set_m128(s13, s12);
            acc6 = _mm256_fmadd_ps(w1213, s1213, acc6);

            let w1415 = _mm256_loadu_ps(weights.as_ptr().add(i + 14) as *const f32);
            let s14 = _mm_set1_ps(*state.get_unchecked(i + 14));
            let s15 = _mm_set1_ps(*state.get_unchecked(i + 15));
            let s1415 = _mm256_set_m128(s15, s14);
            acc7 = _mm256_fmadd_ps(w1415, s1415, acc7);

            i += 16;
        }

        while i + 8 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            let w45 = _mm256_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let s4 = _mm_set1_ps(*state.get_unchecked(i + 4));
            let s5 = _mm_set1_ps(*state.get_unchecked(i + 5));
            let s45 = _mm256_set_m128(s5, s4);
            acc2 = _mm256_fmadd_ps(w45, s45, acc2);

            let w67 = _mm256_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let s6 = _mm_set1_ps(*state.get_unchecked(i + 6));
            let s7 = _mm_set1_ps(*state.get_unchecked(i + 7));
            let s67 = _mm256_set_m128(s7, s6);
            acc3 = _mm256_fmadd_ps(w67, s67, acc3);

            i += 8;
        }

        while i + 4 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            let w23 = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s2 = _mm_set1_ps(*state.get_unchecked(i + 2));
            let s3 = _mm_set1_ps(*state.get_unchecked(i + 3));
            let s23 = _mm256_set_m128(s3, s2);
            acc1 = _mm256_fmadd_ps(w23, s23, acc1);

            i += 4;
        }

        while i + 2 <= len {
            let w01 = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm_set1_ps(*state.get_unchecked(i));
            let s1 = _mm_set1_ps(*state.get_unchecked(i + 1));
            let s01 = _mm256_set_m128(s1, s0);
            acc0 = _mm256_fmadd_ps(w01, s01, acc0);

            i += 2;
        }

        acc0 = _mm256_add_ps(acc0, acc1);
        acc2 = _mm256_add_ps(acc2, acc3);
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);

        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);

        acc0 = _mm256_add_ps(acc0, acc4);

        let low128 = _mm256_castps256_ps128(acc0);
        let high128 = _mm256_extractf128_ps(acc0, 1);
        let init128 = _mm_loadu_ps(init.as_ptr());
        let sum128 = _mm_add_ps(low128, high128);
        let mut res128 = _mm_add_ps(init128, sum128);

        if i < len {
            let w = _mm_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s = _mm_set1_ps(*state.get_unchecked(i));
            res128 = _mm_fmadd_ps(w, s, res128);
        }

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), res128);
        out
    }
}

/// Fused accumulate dual-frame 4-lane interleaved dot product
/// (`weights: &[[f32; 4]]`, `state_f0: &[f32]`, `state_f1: &[f32]`,
/// `init_f0: &[f32; 4]`, `init_f1: &[f32; 4]`) with AVX-512 VL256 (256-bit EVEX).
///
/// Fuses the `init_f0`/`init_f1` accumulators (bias + mixin) into `acc0`,
/// avoiding an extra pass over the outputs.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_dual_accumulate_avx512vl(
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
    let init_f0_128 = _mm_loadu_ps(init_f0.as_ptr());
    let init_f1_128 = _mm_loadu_ps(init_f1.as_ptr());
    let mut acc0 = _mm256_set_m128(init_f1_128, init_f0_128);
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut acc4 = _mm256_setzero_ps();
    let mut acc5 = _mm256_setzero_ps();
    let mut acc6 = _mm256_setzero_ps();
    let mut acc7 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 8 <= len {
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

            let w4_128 = _mm_loadu_ps(weights.as_ptr().add(i + 4) as *const f32);
            let w4_256 = _mm256_broadcast_ps(&w4_128);
            let s_f0_4 = _mm_set1_ps(*state_f0.get_unchecked(i + 4));
            let s_f1_4 = _mm_set1_ps(*state_f1.get_unchecked(i + 4));
            let s_blend4 = _mm256_set_m128(s_f1_4, s_f0_4);
            acc4 = _mm256_fmadd_ps(w4_256, s_blend4, acc4);

            let w5_128 = _mm_loadu_ps(weights.as_ptr().add(i + 5) as *const f32);
            let w5_256 = _mm256_broadcast_ps(&w5_128);
            let s_f0_5 = _mm_set1_ps(*state_f0.get_unchecked(i + 5));
            let s_f1_5 = _mm_set1_ps(*state_f1.get_unchecked(i + 5));
            let s_blend5 = _mm256_set_m128(s_f1_5, s_f0_5);
            acc5 = _mm256_fmadd_ps(w5_256, s_blend5, acc5);

            let w6_128 = _mm_loadu_ps(weights.as_ptr().add(i + 6) as *const f32);
            let w6_256 = _mm256_broadcast_ps(&w6_128);
            let s_f0_6 = _mm_set1_ps(*state_f0.get_unchecked(i + 6));
            let s_f1_6 = _mm_set1_ps(*state_f1.get_unchecked(i + 6));
            let s_blend6 = _mm256_set_m128(s_f1_6, s_f0_6);
            acc6 = _mm256_fmadd_ps(w6_256, s_blend6, acc6);

            let w7_128 = _mm_loadu_ps(weights.as_ptr().add(i + 7) as *const f32);
            let w7_256 = _mm256_broadcast_ps(&w7_128);
            let s_f0_7 = _mm_set1_ps(*state_f0.get_unchecked(i + 7));
            let s_f1_7 = _mm_set1_ps(*state_f1.get_unchecked(i + 7));
            let s_blend7 = _mm256_set_m128(s_f1_7, s_f0_7);
            acc7 = _mm256_fmadd_ps(w7_256, s_blend7, acc7);

            i += 8;
        }

        while i + 4 <= len {
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

            i += 4;
        }

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
        acc4 = _mm256_add_ps(acc4, acc5);
        acc6 = _mm256_add_ps(acc6, acc7);

        acc0 = _mm256_add_ps(acc0, acc2);
        acc4 = _mm256_add_ps(acc4, acc6);

        acc0 = _mm256_add_ps(acc0, acc4);

        let acc_f0 = _mm256_extractf128_ps(acc0, 0);
        let acc_f1 = _mm256_extractf128_ps(acc0, 1);
        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), acc_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), acc_f1);
        (out_f0, out_f1)
    }
}
