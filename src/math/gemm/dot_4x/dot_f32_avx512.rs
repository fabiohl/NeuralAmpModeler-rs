// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 4x f32 — AVX‑512 kernel (f32 native weights).
//!
//! Processes 4 interleaved weight rows (16 f32 values) per `__m512` iteration,
//! sharing the same FMA‑based rounding chain as the AVX2 kernel and the scalar
//! reference.
//!
//! # Precision
//! The AVX‑512 kernel (`_mm512_fmadd_ps`) and the scalar reference (`mul_add`)
//! both use FMA3 fused multiply‑add with identical summation order →
//! **bit‑identical** result on any x86 CPU with FMA support (x86‑64‑v3).
//! The AVX2 kernel uses 4‑way accumulator splitting for latency hiding, so
//! cross‑ISA comparisons may differ by < 2 ULP.

use core::arch::x86_64::*;

/// Build a `__m512` from 4 broadcast state scalars, one per 128‑bit lane.
///
/// Lane layout: lanes 0‑3 = `s0`, lanes 4‑7 = `s1`, lanes 8‑11 = `s2`,
/// lanes 12‑15 = `s3`.
#[inline(always)]
unsafe fn make_state_m512(s0: f32, s1: f32, s2: f32, s3: f32) -> __m512 {
    unsafe {
        let mut v = _mm512_setzero_ps();
        v = _mm512_insertf32x4::<0>(v, _mm_set1_ps(s0));
        v = _mm512_insertf32x4::<1>(v, _mm_set1_ps(s1));
        v = _mm512_insertf32x4::<2>(v, _mm_set1_ps(s2));
        v = _mm512_insertf32x4::<3>(v, _mm_set1_ps(s3));
        v
    }
}

/// 4‑lane interleaved dot product (`weights: &[[f32; 4]]`, `state: &[f32]`)
/// with AVX‑512/FMA.
///
/// # Strategy
/// - Four weight rows (`[f32;4]` × 4 = 16 f32) are loaded into `__m512`.
/// - Four state scalars are each broadcast across 4 lanes and packed into
///   `__m512` via `_mm512_insertf32x4_ps`.
/// - `_mm512_fmadd_ps(w512, s512, acc512)` accumulates 4 input samples per
///   iteration for all 4 output channels.
/// - Tail (< 4 elements) zero‑padded into a 4‑element buffer → same SIMD
///   path preserves the continuous FMA rounding chain.
/// - Horizontal reduction: the 4 128‑bit lanes are summed element‑wise.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions
/// are valid for unaligned load. Both slices must be accessible for reading.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_avx512(weights: &[[f32; 4]], state: &[f32]) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let mut acc512 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 4 <= len {
            let w512 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s512 = make_state_m512(
                *state.get_unchecked(i),
                *state.get_unchecked(i + 1),
                *state.get_unchecked(i + 2),
                *state.get_unchecked(i + 3),
            );
            acc512 = _mm512_fmadd_ps(w512, s512, acc512);
            i += 4;
        }

        if i < len {
            let rem = len - i;
            let mut s_buf = [0.0f32; 4];
            let mut w_buf = [[0.0f32; 4]; 4];
            for j in 0..rem {
                s_buf[j] = *state.get_unchecked(i + j);
                w_buf[j] = *weights.as_ptr().add(i + j);
            }
            let w512 = _mm512_loadu_ps(w_buf.as_ptr() as *const f32);
            let s512 = make_state_m512(s_buf[0], s_buf[1], s_buf[2], s_buf[3]);
            acc512 = _mm512_fmadd_ps(w512, s512, acc512);
        }

        let v0 = _mm512_extractf32x4_ps(acc512, 0);
        let v1 = _mm512_extractf32x4_ps(acc512, 1);
        let v2 = _mm512_extractf32x4_ps(acc512, 2);
        let v3 = _mm512_extractf32x4_ps(acc512, 3);

        let r01 = _mm_add_ps(v0, v1);
        let r23 = _mm_add_ps(v2, v3);
        let result = _mm_add_ps(r01, r23);

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }
}

/// Dual‑frame 4‑lane interleaved dot product (`weights: &[[f32; 4]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX‑512/FMA.
///
/// # Strategy
/// - Four weight rows (16 f32) are loaded into `__m512` once and reused
///   for both frames.
/// - Two `__m512` accumulators accumulate frame‑0 and frame‑1 in parallel.
/// - Same `_mm512_insertf32x4_ps` packing of 4 broadcast state scalars per
///   frame.
/// - Tail (< 4 elements) zero‑padded into a 4‑element buffer → continuous
///   FMA rounding chain preserved for both frames.
///
/// # Precision
/// Uses the same FMA3 instructions as the scalar reference → bit‑identical
/// result on any x86‑64‑v3 CPU. The AVX2 dual kernel uses accumulator
/// splitting and may differ by < 2 ULP.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_dual_avx512(
    weights: &[[f32; 4]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 4], [f32; 4]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0 = _mm512_setzero_ps();
    let mut acc_f1 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 4 <= len {
            let w512 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);

            let s_f0_512 = make_state_m512(
                *state_f0.get_unchecked(i),
                *state_f0.get_unchecked(i + 1),
                *state_f0.get_unchecked(i + 2),
                *state_f0.get_unchecked(i + 3),
            );
            let s_f1_512 = make_state_m512(
                *state_f1.get_unchecked(i),
                *state_f1.get_unchecked(i + 1),
                *state_f1.get_unchecked(i + 2),
                *state_f1.get_unchecked(i + 3),
            );

            acc_f0 = _mm512_fmadd_ps(w512, s_f0_512, acc_f0);
            acc_f1 = _mm512_fmadd_ps(w512, s_f1_512, acc_f1);
            i += 4;
        }

        if i < len {
            let rem = len - i;
            let mut s0_buf = [0.0f32; 4];
            let mut s1_buf = [0.0f32; 4];
            let mut w_buf = [[0.0f32; 4]; 4];
            for j in 0..rem {
                s0_buf[j] = *state_f0.get_unchecked(i + j);
                s1_buf[j] = *state_f1.get_unchecked(i + j);
                w_buf[j] = *weights.as_ptr().add(i + j);
            }
            let w512 = _mm512_loadu_ps(w_buf.as_ptr() as *const f32);
            let s_f0_512 = make_state_m512(s0_buf[0], s0_buf[1], s0_buf[2], s0_buf[3]);
            let s_f1_512 = make_state_m512(s1_buf[0], s1_buf[1], s1_buf[2], s1_buf[3]);

            acc_f0 = _mm512_fmadd_ps(w512, s_f0_512, acc_f0);
            acc_f1 = _mm512_fmadd_ps(w512, s_f1_512, acc_f1);
        }

        let v0_0 = _mm512_extractf32x4_ps(acc_f0, 0);
        let v0_1 = _mm512_extractf32x4_ps(acc_f0, 1);
        let v0_2 = _mm512_extractf32x4_ps(acc_f0, 2);
        let v0_3 = _mm512_extractf32x4_ps(acc_f0, 3);
        let r0_01 = _mm_add_ps(v0_0, v0_1);
        let r0_23 = _mm_add_ps(v0_2, v0_3);
        let result_f0 = _mm_add_ps(r0_01, r0_23);

        let v1_0 = _mm512_extractf32x4_ps(acc_f1, 0);
        let v1_1 = _mm512_extractf32x4_ps(acc_f1, 1);
        let v1_2 = _mm512_extractf32x4_ps(acc_f1, 2);
        let v1_3 = _mm512_extractf32x4_ps(acc_f1, 3);
        let r1_01 = _mm_add_ps(v1_0, v1_1);
        let r1_23 = _mm_add_ps(v1_2, v1_3);
        let result_f1 = _mm_add_ps(r1_01, r1_23);

        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), result_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), result_f1);
        (out_f0, out_f1)
    }
}

/// 4‑lane interleaved dot product with accumulator init
/// (`weights: &[[f32; 4]]`, `state: &[f32]`, `init: &[f32; 4]`)
/// with AVX‑512/FMA.
///
/// Fuses the `init` array (bias + mixin) into the accumulator at lane 0,
/// eliminating a separate vector‑add pass over the output array.
///
/// # Strategy
/// - `acc512` is initialized with `init` replicated into the lowest 128‑bit lane
///   (lane 0). Lanes 1‑3 start zero. After horizontal reduction across all 4
///   lanes the result is `init + dot_sum`.
/// - Main loop, tail, and reduction are identical to the base kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions
/// are valid for unaligned load. Both slices must be accessible for reading.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_accumulate_avx512(
    weights: &[[f32; 4]],
    state: &[f32],
    init: &[f32; 4],
) -> [f32; 4] {
    let len = state.len().min(weights.len());
    debug_assert!(weights.len() >= len);
    let init128 = _mm_loadu_ps(init.as_ptr());
    let mut acc512 = _mm512_setzero_ps();
    acc512 = _mm512_insertf32x4::<0>(acc512, init128);
    let mut i = 0;

    unsafe {
        while i + 4 <= len {
            let w512 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s512 = make_state_m512(
                *state.get_unchecked(i),
                *state.get_unchecked(i + 1),
                *state.get_unchecked(i + 2),
                *state.get_unchecked(i + 3),
            );
            acc512 = _mm512_fmadd_ps(w512, s512, acc512);
            i += 4;
        }

        if i < len {
            let rem = len - i;
            let mut s_buf = [0.0f32; 4];
            let mut w_buf = [[0.0f32; 4]; 4];
            for j in 0..rem {
                s_buf[j] = *state.get_unchecked(i + j);
                w_buf[j] = *weights.as_ptr().add(i + j);
            }
            let w512 = _mm512_loadu_ps(w_buf.as_ptr() as *const f32);
            let s512 = make_state_m512(s_buf[0], s_buf[1], s_buf[2], s_buf[3]);
            acc512 = _mm512_fmadd_ps(w512, s512, acc512);
        }

        let v0 = _mm512_extractf32x4_ps(acc512, 0);
        let v1 = _mm512_extractf32x4_ps(acc512, 1);
        let v2 = _mm512_extractf32x4_ps(acc512, 2);
        let v3 = _mm512_extractf32x4_ps(acc512, 3);

        let r01 = _mm_add_ps(v0, v1);
        let r23 = _mm_add_ps(v2, v3);
        let result = _mm_add_ps(r01, r23);

        let mut out = [0.0f32; 4];
        _mm_storeu_ps(out.as_mut_ptr(), result);
        out
    }
}

/// Dual‑frame 4‑lane interleaved dot product with accumulator init
/// (`weights: &[[f32; 4]]`, `state_f0/state_f1: &[f32]`,
/// `init_f0/init_f1: &[f32; 4]`) with AVX‑512/FMA.
///
/// Fuses each frame’s `init` array into the lowest 128‑bit lane of the
/// respective accumulator.
///
/// # Strategy
/// - `acc_f0` and `acc_f1` are initialized with `init_f0`/`init_f1` in lane 0.
///   Lanes 1‑3 start zero. After horizontal reduction: `init + dot_sum`.
/// - Main loop, tail, and reduction are identical to the base dual kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_4x_f32_dual_accumulate_avx512(
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
    let mut acc_f0 = _mm512_setzero_ps();
    let mut acc_f1 = _mm512_setzero_ps();
    acc_f0 = _mm512_insertf32x4::<0>(acc_f0, init_f0_128);
    acc_f1 = _mm512_insertf32x4::<0>(acc_f1, init_f1_128);
    let mut i = 0;

    unsafe {
        while i + 4 <= len {
            let w512 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);

            let s_f0_512 = make_state_m512(
                *state_f0.get_unchecked(i),
                *state_f0.get_unchecked(i + 1),
                *state_f0.get_unchecked(i + 2),
                *state_f0.get_unchecked(i + 3),
            );
            let s_f1_512 = make_state_m512(
                *state_f1.get_unchecked(i),
                *state_f1.get_unchecked(i + 1),
                *state_f1.get_unchecked(i + 2),
                *state_f1.get_unchecked(i + 3),
            );

            acc_f0 = _mm512_fmadd_ps(w512, s_f0_512, acc_f0);
            acc_f1 = _mm512_fmadd_ps(w512, s_f1_512, acc_f1);
            i += 4;
        }

        if i < len {
            let rem = len - i;
            let mut s0_buf = [0.0f32; 4];
            let mut s1_buf = [0.0f32; 4];
            let mut w_buf = [[0.0f32; 4]; 4];
            for j in 0..rem {
                s0_buf[j] = *state_f0.get_unchecked(i + j);
                s1_buf[j] = *state_f1.get_unchecked(i + j);
                w_buf[j] = *weights.as_ptr().add(i + j);
            }
            let w512 = _mm512_loadu_ps(w_buf.as_ptr() as *const f32);
            let s_f0_512 = make_state_m512(s0_buf[0], s0_buf[1], s0_buf[2], s0_buf[3]);
            let s_f1_512 = make_state_m512(s1_buf[0], s1_buf[1], s1_buf[2], s1_buf[3]);

            acc_f0 = _mm512_fmadd_ps(w512, s_f0_512, acc_f0);
            acc_f1 = _mm512_fmadd_ps(w512, s_f1_512, acc_f1);
        }

        let v0_0 = _mm512_extractf32x4_ps(acc_f0, 0);
        let v0_1 = _mm512_extractf32x4_ps(acc_f0, 1);
        let v0_2 = _mm512_extractf32x4_ps(acc_f0, 2);
        let v0_3 = _mm512_extractf32x4_ps(acc_f0, 3);
        let r0_01 = _mm_add_ps(v0_0, v0_1);
        let r0_23 = _mm_add_ps(v0_2, v0_3);
        let result_f0 = _mm_add_ps(r0_01, r0_23);

        let v1_0 = _mm512_extractf32x4_ps(acc_f1, 0);
        let v1_1 = _mm512_extractf32x4_ps(acc_f1, 1);
        let v1_2 = _mm512_extractf32x4_ps(acc_f1, 2);
        let v1_3 = _mm512_extractf32x4_ps(acc_f1, 3);
        let r1_01 = _mm_add_ps(v1_0, v1_1);
        let r1_23 = _mm_add_ps(v1_2, v1_3);
        let result_f1 = _mm_add_ps(r1_01, r1_23);

        let mut out_f0 = [0.0f32; 4];
        let mut out_f1 = [0.0f32; 4];
        _mm_storeu_ps(out_f0.as_mut_ptr(), result_f0);
        _mm_storeu_ps(out_f1.as_mut_ptr(), result_f1);
        (out_f0, out_f1)
    }
}
