// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

/// F32-native 8-lane interleaved dot product (scalar reference for SIMD validation).
///
/// Uses `mul_add` (FMA3 fused multiply-add) to match the instruction set of the
/// AVX2/FMA kernel (`dot_product_8x_f32_avx2`). Both paths produce **mathematically
/// equivalent** results with FMA‑level precision.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_8x_f32_scalar(weights: &[[f32; 8]], state: &[f32]) -> [f32; 8] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut r = [0.0f32; 8];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s = state.get_unchecked(i);
        r[0] = (*w.get_unchecked(0)).mul_add(*s, r[0]);
        r[1] = (*w.get_unchecked(1)).mul_add(*s, r[1]);
        r[2] = (*w.get_unchecked(2)).mul_add(*s, r[2]);
        r[3] = (*w.get_unchecked(3)).mul_add(*s, r[3]);
        r[4] = (*w.get_unchecked(4)).mul_add(*s, r[4]);
        r[5] = (*w.get_unchecked(5)).mul_add(*s, r[5]);
        r[6] = (*w.get_unchecked(6)).mul_add(*s, r[6]);
        r[7] = (*w.get_unchecked(7)).mul_add(*s, r[7]);
    }
    r
}

/// F32-native 16-lane interleaved dot product (scalar oracle for SIMD validation).
///
/// Uses `mul_add` (FMA3 fused multiply-add) to match the instruction set of the
/// AVX‑512 kernel (`dot_product_16x_f32_avx512`) and the AVX2/FMA kernel
/// (`dot_product_16x_f32_avx2`). This function serves exclusively as the
/// correctness reference in test and benchmark suites. It MUST NOT be called
/// from any production code path (trait implementations, model inference loops).
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_16x_f32_scalar(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
    let len = core::cmp::min(weights.len(), state.len());
    let mut r = [0.0f32; 16];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s = state.get_unchecked(i);
        r[0] = (*w.get_unchecked(0)).mul_add(*s, r[0]);
        r[1] = (*w.get_unchecked(1)).mul_add(*s, r[1]);
        r[2] = (*w.get_unchecked(2)).mul_add(*s, r[2]);
        r[3] = (*w.get_unchecked(3)).mul_add(*s, r[3]);
        r[4] = (*w.get_unchecked(4)).mul_add(*s, r[4]);
        r[5] = (*w.get_unchecked(5)).mul_add(*s, r[5]);
        r[6] = (*w.get_unchecked(6)).mul_add(*s, r[6]);
        r[7] = (*w.get_unchecked(7)).mul_add(*s, r[7]);
        r[8] = (*w.get_unchecked(8)).mul_add(*s, r[8]);
        r[9] = (*w.get_unchecked(9)).mul_add(*s, r[9]);
        r[10] = (*w.get_unchecked(10)).mul_add(*s, r[10]);
        r[11] = (*w.get_unchecked(11)).mul_add(*s, r[11]);
        r[12] = (*w.get_unchecked(12)).mul_add(*s, r[12]);
        r[13] = (*w.get_unchecked(13)).mul_add(*s, r[13]);
        r[14] = (*w.get_unchecked(14)).mul_add(*s, r[14]);
        r[15] = (*w.get_unchecked(15)).mul_add(*s, r[15]);
    }
    r
}

/// F32-native 8-lane interleaved dual-frame dot product (scalar reference).
///
/// Processes two state vectors against the same weight slice using `mul_add`
/// (FMA3) to match the rounding of the AVX2/FMA kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_8x_f32_dual_scalar(
    weights: &[[f32; 8]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 8], [f32; 8]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut r0 = [0.0f32; 8];
    let mut r1 = [0.0f32; 8];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s0 = state_f0.get_unchecked(i);
        let s1 = state_f1.get_unchecked(i);
        r0[0] = (*w.get_unchecked(0)).mul_add(*s0, r0[0]);
        r0[1] = (*w.get_unchecked(1)).mul_add(*s0, r0[1]);
        r0[2] = (*w.get_unchecked(2)).mul_add(*s0, r0[2]);
        r0[3] = (*w.get_unchecked(3)).mul_add(*s0, r0[3]);
        r0[4] = (*w.get_unchecked(4)).mul_add(*s0, r0[4]);
        r0[5] = (*w.get_unchecked(5)).mul_add(*s0, r0[5]);
        r0[6] = (*w.get_unchecked(6)).mul_add(*s0, r0[6]);
        r0[7] = (*w.get_unchecked(7)).mul_add(*s0, r0[7]);
        r1[0] = (*w.get_unchecked(0)).mul_add(*s1, r1[0]);
        r1[1] = (*w.get_unchecked(1)).mul_add(*s1, r1[1]);
        r1[2] = (*w.get_unchecked(2)).mul_add(*s1, r1[2]);
        r1[3] = (*w.get_unchecked(3)).mul_add(*s1, r1[3]);
        r1[4] = (*w.get_unchecked(4)).mul_add(*s1, r1[4]);
        r1[5] = (*w.get_unchecked(5)).mul_add(*s1, r1[5]);
        r1[6] = (*w.get_unchecked(6)).mul_add(*s1, r1[6]);
        r1[7] = (*w.get_unchecked(7)).mul_add(*s1, r1[7]);
    }
    (r0, r1)
}

/// F32-native 16-lane interleaved dual-frame dot product (scalar reference).
///
/// Processes two state vectors against the same weight slice using `mul_add`
/// (FMA3) to match the rounding of the AVX2/FMA and AVX-512 kernels.
/// This function serves exclusively as the correctness reference in test and
/// benchmark suites. It MUST NOT be called from any production code path.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_16x_f32_dual_scalar(
    weights: &[[f32; 16]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 16], [f32; 16]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut r0 = [0.0f32; 16];
    let mut r1 = [0.0f32; 16];
    for i in 0..len {
        let w = weights.get_unchecked(i);
        let s0 = state_f0.get_unchecked(i);
        let s1 = state_f1.get_unchecked(i);
        r0[0] = (*w.get_unchecked(0)).mul_add(*s0, r0[0]);
        r0[1] = (*w.get_unchecked(1)).mul_add(*s0, r0[1]);
        r0[2] = (*w.get_unchecked(2)).mul_add(*s0, r0[2]);
        r0[3] = (*w.get_unchecked(3)).mul_add(*s0, r0[3]);
        r0[4] = (*w.get_unchecked(4)).mul_add(*s0, r0[4]);
        r0[5] = (*w.get_unchecked(5)).mul_add(*s0, r0[5]);
        r0[6] = (*w.get_unchecked(6)).mul_add(*s0, r0[6]);
        r0[7] = (*w.get_unchecked(7)).mul_add(*s0, r0[7]);
        r0[8] = (*w.get_unchecked(8)).mul_add(*s0, r0[8]);
        r0[9] = (*w.get_unchecked(9)).mul_add(*s0, r0[9]);
        r0[10] = (*w.get_unchecked(10)).mul_add(*s0, r0[10]);
        r0[11] = (*w.get_unchecked(11)).mul_add(*s0, r0[11]);
        r0[12] = (*w.get_unchecked(12)).mul_add(*s0, r0[12]);
        r0[13] = (*w.get_unchecked(13)).mul_add(*s0, r0[13]);
        r0[14] = (*w.get_unchecked(14)).mul_add(*s0, r0[14]);
        r0[15] = (*w.get_unchecked(15)).mul_add(*s0, r0[15]);
        r1[0] = (*w.get_unchecked(0)).mul_add(*s1, r1[0]);
        r1[1] = (*w.get_unchecked(1)).mul_add(*s1, r1[1]);
        r1[2] = (*w.get_unchecked(2)).mul_add(*s1, r1[2]);
        r1[3] = (*w.get_unchecked(3)).mul_add(*s1, r1[3]);
        r1[4] = (*w.get_unchecked(4)).mul_add(*s1, r1[4]);
        r1[5] = (*w.get_unchecked(5)).mul_add(*s1, r1[5]);
        r1[6] = (*w.get_unchecked(6)).mul_add(*s1, r1[6]);
        r1[7] = (*w.get_unchecked(7)).mul_add(*s1, r1[7]);
        r1[8] = (*w.get_unchecked(8)).mul_add(*s1, r1[8]);
        r1[9] = (*w.get_unchecked(9)).mul_add(*s1, r1[9]);
        r1[10] = (*w.get_unchecked(10)).mul_add(*s1, r1[10]);
        r1[11] = (*w.get_unchecked(11)).mul_add(*s1, r1[11]);
        r1[12] = (*w.get_unchecked(12)).mul_add(*s1, r1[12]);
        r1[13] = (*w.get_unchecked(13)).mul_add(*s1, r1[13]);
        r1[14] = (*w.get_unchecked(14)).mul_add(*s1, r1[14]);
        r1[15] = (*w.get_unchecked(15)).mul_add(*s1, r1[15]);
    }
    (r0, r1)
}

/// F32-native 8-lane interleaved dot product with init accumulation
/// (scalar reference for SIMD validation).
///
/// Computes `dot_product_8x_f32(weights, state)` then adds `init` element-wise.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_8x_f32_accumulate_scalar(
    weights: &[[f32; 8]],
    state: &[f32],
    init: &[f32; 8],
) -> [f32; 8] {
    let r = dot_product_8x_f32_scalar(weights, state);
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = init[i] + r[i];
    }
    out
}

/// F32-native 8-lane interleaved dual-frame dot product with init accumulation
/// (scalar reference for SIMD validation).
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_8x_f32_dual_accumulate_scalar(
    weights: &[[f32; 8]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 8],
    init_f1: &[f32; 8],
) -> ([f32; 8], [f32; 8]) {
    let (r0, r1) = dot_product_8x_f32_dual_scalar(weights, state_f0, state_f1);
    let mut out_f0 = [0.0f32; 8];
    let mut out_f1 = [0.0f32; 8];
    for i in 0..8 {
        out_f0[i] = init_f0[i] + r0[i];
        out_f1[i] = init_f1[i] + r1[i];
    }
    (out_f0, out_f1)
}

/// F32-native 16-lane interleaved dot product with init accumulation
/// (scalar oracle for SIMD validation).
///
/// This function serves exclusively as the correctness reference in test and
/// benchmark suites. It MUST NOT be called from any production code path.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()`.
#[inline]
pub unsafe fn dot_product_16x_f32_accumulate_scalar(
    weights: &[[f32; 16]],
    state: &[f32],
    init: &[f32; 16],
) -> [f32; 16] {
    let r = dot_product_16x_f32_scalar(weights, state);
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = init[i] + r[i];
    }
    out
}

/// F32-native 16-lane interleaved dual-frame dot product with init accumulation
/// (scalar oracle for SIMD validation).
///
/// This function serves exclusively as the correctness reference in test and
/// benchmark suites. It MUST NOT be called from any production code path.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[inline]
pub unsafe fn dot_product_16x_f32_dual_accumulate_scalar(
    weights: &[[f32; 16]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 16],
    init_f1: &[f32; 16],
) -> ([f32; 16], [f32; 16]) {
    let (r0, r1) = dot_product_16x_f32_dual_scalar(weights, state_f0, state_f1);
    let mut out_f0 = [0.0f32; 16];
    let mut out_f1 = [0.0f32; 16];
    for i in 0..16 {
        out_f0[i] = init_f0[i] + r0[i];
        out_f1[i] = init_f1[i] + r1[i];
    }
    (out_f0, out_f1)
}
