// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 16x f32 — AVX‑512 kernel (f32 native weights).
//!
//! Processes 16 interleaved weight rows (16 f32 values) per `__m512` iteration,
//! sharing the same FMA‑based rounding chain as the scalar reference.
//!
//! # Precision
//! The AVX‑512 kernel (`_mm512_fmadd_ps`) and the scalar reference (`mul_add`)
//! both use FMA3 fused multiply‑add with identical summation order →
//! **bit‑identical** result on any x86 CPU with FMA support (x86‑64‑v3).

use core::arch::x86_64::*;

/// 16‑lane interleaved dot product (`weights: &[[f32; 16]]`, `state: &[f32]`)
/// with AVX‑512/FMA.
///
/// # Strategy
/// - 16 weights per row loaded into a single `__m512`.
/// - State scalar broadcast to all 16 lanes via `_mm512_set1_ps`.
/// - Main loop processes 2 input samples per iteration using 2 independent
///   `__m512` accumulators (`acc0`, `acc1`), interleaved by index `i`, to
///   break the FMA latency chain.
/// - Tail (< 2 elements) falls back to a single‑accumulator loop.
/// - Final reduction: `acc0 + acc1`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_16x_f32_avx512(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
    let len = state.len();
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 2 <= len {
            let w0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm512_set1_ps(*state.get_unchecked(i));
            acc0 = _mm512_fmadd_ps(w0, s0, acc0);

            let w1 = _mm512_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm512_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm512_fmadd_ps(w1, s1, acc1);

            i += 2;
        }

        while i < len {
            let s = _mm512_set1_ps(*state.get_unchecked(i));
            let w = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm512_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm512_add_ps(acc0, acc1);

        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Dual‑frame 16‑lane interleaved dot product (`weights: &[[f32; 16]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX‑512/FMA.
///
/// # Strategy
/// - 16 weights per row loaded into a single `__m512`.
/// - State scalars broadcast once per frame per iteration via `_mm512_set1_ps`.
/// - Main loop processes 2 input samples per iteration using 2 independent
///   pairs of `__m512` accumulators per frame (`acc_f0_0`, `acc_f0_1`,
///   `acc_f1_0`, `acc_f1_1`), totalling 4 accumulator registers to break the
///   FMA latency chain while computing both frames from a single weight load.
/// - Tail (< 2 elements) falls back to a single‑accumulator‑pair loop.
/// - Final reduction: sum each frame’s 2 accumulators independently, store
///   as `([f32; 16], [f32; 16])`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_16x_f32_dual_avx512(
    weights: &[[f32; 16]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 16], [f32; 16]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0_0 = _mm512_setzero_ps();
    let mut acc_f0_1 = _mm512_setzero_ps();
    let mut acc_f1_0 = _mm512_setzero_ps();
    let mut acc_f1_1 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 2 <= len {
            let w0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm512_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm512_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm512_fmadd_ps(w0, s_f0_0, acc_f0_0);
            acc_f1_0 = _mm512_fmadd_ps(w0, s_f1_0, acc_f1_0);

            let w1 = _mm512_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm512_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm512_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_1 = _mm512_fmadd_ps(w1, s_f0_1, acc_f0_1);
            acc_f1_1 = _mm512_fmadd_ps(w1, s_f1_1, acc_f1_1);

            i += 2;
        }

        while i < len {
            let w = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm512_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm512_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm512_fmadd_ps(w, s_f0, acc_f0_0);
            acc_f1_0 = _mm512_fmadd_ps(w, s_f1, acc_f1_0);
            i += 1;
        }

        acc_f0_0 = _mm512_add_ps(acc_f0_0, acc_f0_1);
        acc_f1_0 = _mm512_add_ps(acc_f1_0, acc_f1_1);

        let mut out_f0 = [0.0f32; 16];
        let mut out_f1 = [0.0f32; 16];
        _mm512_storeu_ps(out_f0.as_mut_ptr(), acc_f0_0);
        _mm512_storeu_ps(out_f1.as_mut_ptr(), acc_f1_0);
        (out_f0, out_f1)
    }
}

/// 16‑lane interleaved dot product with accumulator init
/// (`weights: &[[f32; 16]]`, `state: &[f32]`, `init: &[f32; 16]`)
/// with AVX‑512/FMA.
///
/// Fuses the `init` array (bias + mixin) into the first accumulator register,
/// eliminating a separate vector‑add pass over the output array.
///
/// # Strategy
/// - `acc0` is initialized from `init` via `_mm512_loadu_ps` instead of zero.
/// - `acc1` remains zero‑initialized for latency‑hiding interleaving.
/// - Main loop, tail loop, and final reduction are identical to the base kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_16x_f32_accumulate_avx512(
    weights: &[[f32; 16]],
    state: &[f32],
    init: &[f32; 16],
) -> [f32; 16] {
    let len = state.len();
    let mut acc0 = _mm512_loadu_ps(init.as_ptr());
    let mut acc1 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 2 <= len {
            let w0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s0 = _mm512_set1_ps(*state.get_unchecked(i));
            acc0 = _mm512_fmadd_ps(w0, s0, acc0);

            let w1 = _mm512_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s1 = _mm512_set1_ps(*state.get_unchecked(i + 1));
            acc1 = _mm512_fmadd_ps(w1, s1, acc1);

            i += 2;
        }

        while i < len {
            let s = _mm512_set1_ps(*state.get_unchecked(i));
            let w = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            acc0 = _mm512_fmadd_ps(w, s, acc0);
            i += 1;
        }

        acc0 = _mm512_add_ps(acc0, acc1);

        let mut out = [0.0f32; 16];
        _mm512_storeu_ps(out.as_mut_ptr(), acc0);
        out
    }
}

/// Dual‑frame 16‑lane interleaved dot product with accumulator init
/// (`weights: &[[f32; 16]]`, `state_f0/state_f1: &[f32]`,
/// `init_f0/init_f1: &[f32; 16]`) with AVX‑512/FMA.
///
/// Fuses each frame’s `init` array into the first accumulator register pair.
///
/// # Strategy
/// - `acc_f0_0` and `acc_f1_0` are initialized from `init_f0`/`init_f1` via
///   `_mm512_loadu_ps` instead of zero.
/// - `acc_f0_1` and `acc_f1_1` remain zero‑initialized for latency hiding.
/// - Main loop, tail loop, and reduction are identical to the base dual kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn dot_product_16x_f32_dual_accumulate_avx512(
    weights: &[[f32; 16]],
    state_f0: &[f32],
    state_f1: &[f32],
    init_f0: &[f32; 16],
    init_f1: &[f32; 16],
) -> ([f32; 16], [f32; 16]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0_0 = _mm512_loadu_ps(init_f0.as_ptr());
    let mut acc_f0_1 = _mm512_setzero_ps();
    let mut acc_f1_0 = _mm512_loadu_ps(init_f1.as_ptr());
    let mut acc_f1_1 = _mm512_setzero_ps();
    let mut i = 0;

    unsafe {
        while i + 2 <= len {
            let w0 = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm512_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm512_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm512_fmadd_ps(w0, s_f0_0, acc_f0_0);
            acc_f1_0 = _mm512_fmadd_ps(w0, s_f1_0, acc_f1_0);

            let w1 = _mm512_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm512_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm512_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_1 = _mm512_fmadd_ps(w1, s_f0_1, acc_f0_1);
            acc_f1_1 = _mm512_fmadd_ps(w1, s_f1_1, acc_f1_1);

            i += 2;
        }

        while i < len {
            let w = _mm512_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm512_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm512_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_0 = _mm512_fmadd_ps(w, s_f0, acc_f0_0);
            acc_f1_0 = _mm512_fmadd_ps(w, s_f1, acc_f1_0);
            i += 1;
        }

        acc_f0_0 = _mm512_add_ps(acc_f0_0, acc_f0_1);
        acc_f1_0 = _mm512_add_ps(acc_f1_0, acc_f1_1);

        let mut out_f0 = [0.0f32; 16];
        let mut out_f1 = [0.0f32; 16];
        _mm512_storeu_ps(out_f0.as_mut_ptr(), acc_f0_0);
        _mm512_storeu_ps(out_f1.as_mut_ptr(), acc_f1_0);
        (out_f0, out_f1)
    }
}
