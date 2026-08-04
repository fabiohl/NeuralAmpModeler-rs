// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::half::f16_bits_to_f32;
use crate::math::common::kahan_add;

/// Interleaved Dot Product (4x).
/// Instead of computing a single sum, this function computes 4 sums at the same time
/// using the same input data but different weights.
/// Useful when a sound (state) affects 4 different "channels" or "neurons".
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_4x_interleaved_fallback(weights: &[[u16; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut sum = [0.0f32; 4];
    let mut comp = [0.0f32; 4];

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            let s = *state.get_unchecked(i);
            let w = weights.get_unchecked(i);

            let w0 = f16_bits_to_f32(w[0]);
            let w1 = f16_bits_to_f32(w[1]);
            let w2 = f16_bits_to_f32(w[2]);
            let w3 = f16_bits_to_f32(w[3]);

            let (s0, c0) = kahan_add(sum[0], comp[0], w0 * s);
            sum[0] = s0;
            comp[0] = c0;
            let (s1, c1) = kahan_add(sum[1], comp[1], w1 * s);
            sum[1] = s1;
            comp[1] = c1;
            let (s2, c2) = kahan_add(sum[2], comp[2], w2 * s);
            sum[2] = s2;
            comp[2] = c2;
            let (s3, c3) = kahan_add(sum[3], comp[3], w3 * s);
            sum[3] = s3;
            comp[3] = c3;
        }
    }
    sum
}

/// F32-native 4-lane interleaved dot product (scalar reference for SIMD validation).
///
/// Uses `mul_add` (FMA3 fused multiply-add) to match the instruction set of the
/// AVX2/FMA kernel (`dot_product_4x_f32_avx2`). Both paths produce **mathematically
/// equivalent** results with FMA‑level precision. The SIMD kernel uses 4‑way
/// accumulator unrolling and a final horizontal reduction, which may yield slightly
/// different rounding (< 2 ULP) compared to this strictly‑serial FMA chain.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_scalar(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut r = [0.0f32; 4];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s = state.get_unchecked(i);
        r[0] = (*w.get_unchecked(0)).mul_add(*s, r[0]);
        r[1] = (*w.get_unchecked(1)).mul_add(*s, r[1]);
        r[2] = (*w.get_unchecked(2)).mul_add(*s, r[2]);
        r[3] = (*w.get_unchecked(3)).mul_add(*s, r[3]);
    }
    r
}

/// F32-native 4-lane interleaved dual-frame dot product (scalar reference).
///
/// Processes two state vectors against the same weight slice using `mul_add`
/// (FMA3) to match the rounding of the AVX2/FMA kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_dual_scalar(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut r0 = [0.0f32; 4];
    let mut r1 = [0.0f32; 4];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s0 = state_f0.get_unchecked(i);
        let s1 = state_f1.get_unchecked(i);
        r0[0] = (*w.get_unchecked(0)).mul_add(*s0, r0[0]);
        r0[1] = (*w.get_unchecked(1)).mul_add(*s0, r0[1]);
        r0[2] = (*w.get_unchecked(2)).mul_add(*s0, r0[2]);
        r0[3] = (*w.get_unchecked(3)).mul_add(*s0, r0[3]);
        r1[0] = (*w.get_unchecked(0)).mul_add(*s1, r1[0]);
        r1[1] = (*w.get_unchecked(1)).mul_add(*s1, r1[1]);
        r1[2] = (*w.get_unchecked(2)).mul_add(*s1, r1[2]);
        r1[3] = (*w.get_unchecked(3)).mul_add(*s1, r1[3]);
    }
    (r0, r1)
}

/// Interleaved Dot Product for "Dual Frame" (Two audio frames).
/// This function is even more hardworking: it computes 4 sums for the first
/// audio frame AND 4 sums for the second frame, all in a single loop.
/// This saves time because we read the weights from memory only once.
// SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
#[inline]
pub unsafe fn dot_product_4x_interleaved_dual_frame_fallback(
    weights: &[[u16; 4]],
    state_f0: &[f32], // First audio frame (e.g.: current sample)
    state_f1: &[f32], // Second frame (e.g.: previous or next sample)
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut sum_f0 = [0.0f32; 4];
    let mut sum_f1 = [0.0f32; 4];
    let mut comp_f0 = [0.0f32; 4];
    let mut comp_f1 = [0.0f32; 4];

    for i in 0..len {
        // SAFETY: Preconditions (alignment, bounds, size) are guaranteed by caller of this SIMD/unsafe function.
        unsafe {
            let s0 = *state_f0.get_unchecked(i);
            let s1 = *state_f1.get_unchecked(i);
            let w = weights.get_unchecked(i);

            // We unpack the 4 weights just once to use in both frames.
            let w0 = f16_bits_to_f32(w[0]);
            let w1 = f16_bits_to_f32(w[1]);
            let w2 = f16_bits_to_f32(w[2]);
            let w3 = f16_bits_to_f32(w[3]);

            // Kahan-compensated accumulations for frame 0.
            let (s, c) = kahan_add(sum_f0[0], comp_f0[0], w0 * s0);
            sum_f0[0] = s;
            comp_f0[0] = c;
            let (s, c) = kahan_add(sum_f0[1], comp_f0[1], w1 * s0);
            sum_f0[1] = s;
            comp_f0[1] = c;
            let (s, c) = kahan_add(sum_f0[2], comp_f0[2], w2 * s0);
            sum_f0[2] = s;
            comp_f0[2] = c;
            let (s, c) = kahan_add(sum_f0[3], comp_f0[3], w3 * s0);
            sum_f0[3] = s;
            comp_f0[3] = c;

            // Kahan-compensated accumulations for frame 1.
            let (s, c) = kahan_add(sum_f1[0], comp_f1[0], w0 * s1);
            sum_f1[0] = s;
            comp_f1[0] = c;
            let (s, c) = kahan_add(sum_f1[1], comp_f1[1], w1 * s1);
            sum_f1[1] = s;
            comp_f1[1] = c;
            let (s, c) = kahan_add(sum_f1[2], comp_f1[2], w2 * s1);
            sum_f1[2] = s;
            comp_f1[2] = c;
            let (s, c) = kahan_add(sum_f1[3], comp_f1[3], w3 * s1);
            sum_f1[3] = s;
            comp_f1[3] = c;
        }
    }
    (sum_f0, sum_f1)
}

/// F32-native 4-lane interleaved dot product with init accumulation
/// (scalar reference for SIMD validation).
///
/// Computes `dot_product_4x_f32(weights, state)` then adds `init` element-wise.
/// Uses `mul_add` (FMA3) to match the SIMD kernel rounding.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_accumulate_scalar(
    weights: &[[f32; 4]],
    state: &[f32],
    init: &[f32; 4],
) -> [f32; 4] {
    let r = dot_product_4x_f32_scalar(weights, state);
    [
        init[0] + r[0],
        init[1] + r[1],
        init[2] + r[2],
        init[3] + r[3],
    ]
}

/// F32-native 4-lane interleaved dual-frame dot product with init accumulation
/// (scalar reference for SIMD validation).
///
/// Computes `dot_product_4x_f32_dual(weights, state_f0, state_f1)` then adds
/// `init_f0` / `init_f1` element-wise.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_4x_f32_dual_accumulate_scalar(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 4],
    init_f1: &[f32; 4],
) -> ([f32; 4], [f32; 4]) {
    let (r0, r1) = dot_product_4x_f32_dual_scalar(weights, state_f0, state_f1);
    (
        [
            init_f0[0] + r0[0],
            init_f0[1] + r0[1],
            init_f0[2] + r0[2],
            init_f0[3] + r0[3],
        ],
        [
            init_f1[0] + r1[0],
            init_f1[1] + r1[1],
            init_f1[2] + r1[2],
            init_f1[3] + r1[3],
        ],
    )
}
