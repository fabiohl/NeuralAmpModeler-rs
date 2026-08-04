// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! Dot Product 16x f32 — AVX2/FMA kernel (f32 native weights).
//!
//! Processes 16 interleaved weight rows using two `__m256` registers per row
//! (lo and hi halves of the 16-element array), paired with a single state
//! broadcast per iteration.
//!
//! # Precision
//! Both the scalar reference (`mul_add`) and this kernel (`_mm256_fmadd_ps`) use
//! the same FMA3 fused multiply‑add instruction, producing mathematically
//! equivalent results. The 4‑way accumulator unrolling splits the summation
//! across independent chains (`acc_lo0..3` and `acc_hi0..3`) with a final
//! horizontal reduction, which may yield slightly different rounding (< 2 ULP)
//! compared to the strictly‑serial FMA chain of the scalar reference. No
//! dequantization or precision conversion is involved.
//!
//! # Lane‑equivalence with 4x kernel (T2.S1.1)
//!
//! **Decision:** the 16x kernel is **bit‑exact per‑lane** to the 4x kernel
//! for any shared lane index `j ∈ [0..11]` (12‑channel Lite via pad‑to‑16),
//! assuming the same weight values and padding‑zero invariant on lanes 12‑15.
//!
//! ## Proof outline
//!
//! Both kernels implement `acc_lane += w_lane * x` with the same:
//! 1. **FMA instruction** (`_mm256_fmadd_ps` / `_mm_fmadd_ps`) — bit‑exact
//!    per‑lane semantics.
//! 2. **Iteration order** `i = 0, 1, 2, …, len‑1` over tap index, same
//!    weight column `w[i][j]` and same state broadcast `x[i]`.
//! 3. **4‑way accumulator split by `i % 4`:** iteration `i` goes to
//!    accumulator `acc_{i mod 4}`.
//! 4. **Reduction tree:** `(acc0 + acc1) + (acc2 + acc3)`.
//!
//! Each SIMD lane evolves **independently** — lane j of a `__m256` only
//! accumulates `w[i][j] * x[i]`. The presence of 8/16 lanes in the register
//! does not affect per‑lane arithmetic. Therefore:
//!
//! ```text
//! 4x kernel, block b, lane j'  ≡  16x kernel, lane (4b + j')
//! ```
//!
//! The two‑pass lo/hi structure (T2.E2.1) preserves equivalence: pass 1
//! covers lanes 0‑7, pass 2 covers lanes 8‑15 — each pass uses the same
//! 4‑way split and reduction as the 4x kernel.
//!
//! ## Padding‑zero invariant
//!
//! When `weights[i][12..16]` = 0 (as required by the pad‑to‑16 strategy for
//! CH=12 models), the 4 zero‑lanes contribute nothing to their FMA chains.
//! Lanes 0‑11 are completely independent of lanes 12‑15. The caller must
//! ensure the weight loader (`transpose_conv1d_interleaved`) pads with zeros
//! and the `select_interleave_width` dispatcher routes CH=12 to 16‑wide.

use crate::dot4x_simd4;
use core::arch::x86_64::*;

/// 16‑lane interleaved dot product (`weights: &[[f32; 16]]`, `state: &[f32]`)
/// with AVX2/FMA.
///
/// # Strategy
/// - 16 weights per row split into two `__m256` loads (lo `[0..8)`, hi `[8..16)`).
/// - State scalar broadcast to all lanes of both halves via `_mm256_set1_ps`.
/// - Main loop processes 4 input samples per iteration using 4 independent
///   pairs of `__m256` accumulators (`acc_lo0..3`, `acc_hi0..3`), interleaved
///   by index `i`, to break the FMA latency chain and allow OoO execution to
///   overlap them.
/// - Tail (< 4 elements) falls back to a single‑accumulator‑pair loop.
/// - Final reduction: tree‑sum both halves independently, then store as
///   contiguous `[f32; 16]`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory regions
/// are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_16x_f32_avx2(weights: &[[f32; 16]], state: &[f32]) -> [f32; 16] {
    let len = state.len();
    let mut acc_lo0 = _mm256_setzero_ps();
    let mut acc_lo1 = _mm256_setzero_ps();
    let mut acc_lo2 = _mm256_setzero_ps();
    let mut acc_lo3 = _mm256_setzero_ps();
    let mut acc_hi0 = _mm256_setzero_ps();
    let mut acc_hi1 = _mm256_setzero_ps();
    let mut acc_hi2 = _mm256_setzero_ps();
    let mut acc_hi3 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let s0 = _mm256_set1_ps(*state.get_unchecked(i));
            let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w0_lo, s0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w0_hi, s0, acc_hi0);

            let s1 = _mm256_set1_ps(*state.get_unchecked(i + 1));
            let w1_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let w1_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 1) as *const f32).add(8));
            acc_lo1 = _mm256_fmadd_ps(w1_lo, s1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(w1_hi, s1, acc_hi1);

            let s2 = _mm256_set1_ps(*state.get_unchecked(i + 2));
            let w2_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let w2_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 2) as *const f32).add(8));
            acc_lo2 = _mm256_fmadd_ps(w2_lo, s2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(w2_hi, s2, acc_hi2);

            let s3 = _mm256_set1_ps(*state.get_unchecked(i + 3));
            let w3_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let w3_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 3) as *const f32).add(8));
            acc_lo3 = _mm256_fmadd_ps(w3_lo, s3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(w3_hi, s3, acc_hi3);
        });

        while i < len {
            let s = _mm256_set1_ps(*state.get_unchecked(i));
            let w_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w_lo, s, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w_hi, s, acc_hi0);
            i += 1;
        }

        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo1);
        acc_lo2 = _mm256_add_ps(acc_lo2, acc_lo3);
        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo2);

        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi1);
        acc_hi2 = _mm256_add_ps(acc_hi2, acc_hi3);
        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi2);

        let mut out = [0.0f32; 16];
        _mm256_storeu_ps(out.as_mut_ptr(), acc_lo0);
        _mm256_storeu_ps(out.as_mut_ptr().add(8), acc_hi0);
        out
    }
}

/// Dual‑frame 16‑lane interleaved dot product (`weights: &[[f32; 16]]`,
/// `state_f0: &[f32]`, `state_f1: &[f32]`) with AVX2/FMA.
///
/// # Strategy — two‑pass lo/hi
/// The original kernel maintained all 16 accumulator registers
/// simultaneously, forcing LLVM to spill 4 accumulators to the stack on
/// x86‑64 (only 16 ymm regs available vs. 16 acc + weight loads +
/// broadcasts). This kernel splits the computation into two independent
/// passes:
///
/// 1. **Pass 1 (lo):** iterates over all taps loading only `w_lo`
///    (`weights[i][0..8]`). Only 8 accumulator registers are live
///    (`acc_f0_lo{0..3}`, `acc_f1_lo{0..3}`), leaving sufficient
///    register headroom to eliminate spills.
/// 2. **Pass 2 (hi):** iterates over all taps loading only `w_hi`
///    (`weights[i][8..16]`, offset +8 floats). Only the 8 hi accumulator
///    registers are live.
///
/// The state broadcasts `s_f0`/`s_f1` are re‑executed in pass 2 and the
/// weight rows are re‑read — an acceptable cost compared to the
/// eliminated register spills. The same FMA sequence per lane and the
/// same tree reduction `(acc0+acc1)+(acc2+acc3)` guarantee bit‑exact
/// results identical to the original single‑pass kernel.
///
/// Main loop processes 4 input samples per iteration using 4 independent
/// accumulator registers to break the FMA latency chain.
/// Tail (< 4 elements) falls back to a single‑accumulator loop.
/// Final reduction: tree‑sum each frame's lo/hi accumulators independently,
/// then store as contiguous `([f32; 16], [f32; 16])`.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_16x_f32_dual_avx2(
    weights: &[[f32; 16]],
    state_f0: &[f32],
    state_f1: &[f32],
) -> ([f32; 16], [f32; 16]) {
    let len = core::cmp::min(
        weights.len(),
        core::cmp::min(state_f0.len(), state_f1.len()),
    );
    let mut acc_f0_lo0 = _mm256_setzero_ps();
    let mut acc_f0_lo1 = _mm256_setzero_ps();
    let mut acc_f0_lo2 = _mm256_setzero_ps();
    let mut acc_f0_lo3 = _mm256_setzero_ps();
    let mut acc_f0_hi0 = _mm256_setzero_ps();
    let mut acc_f0_hi1 = _mm256_setzero_ps();
    let mut acc_f0_hi2 = _mm256_setzero_ps();
    let mut acc_f0_hi3 = _mm256_setzero_ps();
    let mut acc_f1_lo0 = _mm256_setzero_ps();
    let mut acc_f1_lo1 = _mm256_setzero_ps();
    let mut acc_f1_lo2 = _mm256_setzero_ps();
    let mut acc_f1_lo3 = _mm256_setzero_ps();
    let mut acc_f1_hi0 = _mm256_setzero_ps();
    let mut acc_f1_hi1 = _mm256_setzero_ps();
    let mut acc_f1_hi2 = _mm256_setzero_ps();
    let mut acc_f1_hi3 = _mm256_setzero_ps();

    unsafe {
        let mut i = 0;

        dot4x_simd4!(i, len, {
            let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_lo0 = _mm256_fmadd_ps(w0_lo, s_f0_0, acc_f0_lo0);
            acc_f1_lo0 = _mm256_fmadd_ps(w0_lo, s_f1_0, acc_f1_lo0);

            let w1_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_lo1 = _mm256_fmadd_ps(w1_lo, s_f0_1, acc_f0_lo1);
            acc_f1_lo1 = _mm256_fmadd_ps(w1_lo, s_f1_1, acc_f1_lo1);

            let w2_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_lo2 = _mm256_fmadd_ps(w2_lo, s_f0_2, acc_f0_lo2);
            acc_f1_lo2 = _mm256_fmadd_ps(w2_lo, s_f1_2, acc_f1_lo2);

            let w3_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_lo3 = _mm256_fmadd_ps(w3_lo, s_f0_3, acc_f0_lo3);
            acc_f1_lo3 = _mm256_fmadd_ps(w3_lo, s_f1_3, acc_f1_lo3);
        });

        while i < len {
            let w_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_lo0 = _mm256_fmadd_ps(w_lo, s_f0, acc_f0_lo0);
            acc_f1_lo0 = _mm256_fmadd_ps(w_lo, s_f1, acc_f1_lo0);
            i += 1;
        }

        acc_f0_lo0 = _mm256_add_ps(acc_f0_lo0, acc_f0_lo1);
        acc_f0_lo2 = _mm256_add_ps(acc_f0_lo2, acc_f0_lo3);
        acc_f0_lo0 = _mm256_add_ps(acc_f0_lo0, acc_f0_lo2);
        acc_f1_lo0 = _mm256_add_ps(acc_f1_lo0, acc_f1_lo1);
        acc_f1_lo2 = _mm256_add_ps(acc_f1_lo2, acc_f1_lo3);
        acc_f1_lo0 = _mm256_add_ps(acc_f1_lo0, acc_f1_lo2);

        i = 0;

        dot4x_simd4!(i, len, {
            let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_hi0 = _mm256_fmadd_ps(w0_hi, s_f0_0, acc_f0_hi0);
            acc_f1_hi0 = _mm256_fmadd_ps(w0_hi, s_f1_0, acc_f1_hi0);

            let w1_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 1) as *const f32).add(8));
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_hi1 = _mm256_fmadd_ps(w1_hi, s_f0_1, acc_f0_hi1);
            acc_f1_hi1 = _mm256_fmadd_ps(w1_hi, s_f1_1, acc_f1_hi1);

            let w2_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 2) as *const f32).add(8));
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_hi2 = _mm256_fmadd_ps(w2_hi, s_f0_2, acc_f0_hi2);
            acc_f1_hi2 = _mm256_fmadd_ps(w2_hi, s_f1_2, acc_f1_hi2);

            let w3_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 3) as *const f32).add(8));
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_hi3 = _mm256_fmadd_ps(w3_hi, s_f0_3, acc_f0_hi3);
            acc_f1_hi3 = _mm256_fmadd_ps(w3_hi, s_f1_3, acc_f1_hi3);
        });

        while i < len {
            let w_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_hi0 = _mm256_fmadd_ps(w_hi, s_f0, acc_f0_hi0);
            acc_f1_hi0 = _mm256_fmadd_ps(w_hi, s_f1, acc_f1_hi0);
            i += 1;
        }

        acc_f0_hi0 = _mm256_add_ps(acc_f0_hi0, acc_f0_hi1);
        acc_f0_hi2 = _mm256_add_ps(acc_f0_hi2, acc_f0_hi3);
        acc_f0_hi0 = _mm256_add_ps(acc_f0_hi0, acc_f0_hi2);
        acc_f1_hi0 = _mm256_add_ps(acc_f1_hi0, acc_f1_hi1);
        acc_f1_hi2 = _mm256_add_ps(acc_f1_hi2, acc_f1_hi3);
        acc_f1_hi0 = _mm256_add_ps(acc_f1_hi0, acc_f1_hi2);

        let mut out_f0 = [0.0f32; 16];
        let mut out_f1 = [0.0f32; 16];
        _mm256_storeu_ps(out_f0.as_mut_ptr(), acc_f0_lo0);
        _mm256_storeu_ps(out_f0.as_mut_ptr().add(8), acc_f0_hi0);
        _mm256_storeu_ps(out_f1.as_mut_ptr(), acc_f1_lo0);
        _mm256_storeu_ps(out_f1.as_mut_ptr().add(8), acc_f1_hi0);
        (out_f0, out_f1)
    }
}

/// Fused accumulate 16‑lane interleaved dot product (`weights: &[[f32; 16]]`,
/// `state: &[f32]`, `init: &[f32; 16]`) with AVX2/FMA.
///
/// Fuses the `init` accumulator (bias + mixin) into `acc_lo0`/`acc_hi0`,
/// avoiding an extra pass over the output. Other unroll accumulator pairs
/// (`acc_lo1..3`, `acc_hi1..3`) are zero‑initialized.
///
/// # Strategy
/// - Same loop structure as `dot_product_16x_f32_avx2`, but `acc_lo0` and
///   `acc_hi0` start from init loads instead of `_mm256_setzero_ps()`.
/// - Tail and reduction identical to the base kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state.len()` and that the memory
/// regions are valid for unaligned load.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_16x_f32_accumulate_avx2(
    weights: &[[f32; 16]],
    state: &[f32],
    init: &[f32; 16],
) -> [f32; 16] {
    let len = state.len();
    let mut acc_lo0 = _mm256_loadu_ps(init.as_ptr());
    let mut acc_lo1 = _mm256_setzero_ps();
    let mut acc_lo2 = _mm256_setzero_ps();
    let mut acc_lo3 = _mm256_setzero_ps();
    let mut acc_hi0 = _mm256_loadu_ps(init.as_ptr().add(8));
    let mut acc_hi1 = _mm256_setzero_ps();
    let mut acc_hi2 = _mm256_setzero_ps();
    let mut acc_hi3 = _mm256_setzero_ps();
    let mut i = 0;

    unsafe {
        dot4x_simd4!(i, len, {
            let s0 = _mm256_set1_ps(*state.get_unchecked(i));
            let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w0_lo, s0, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w0_hi, s0, acc_hi0);

            let s1 = _mm256_set1_ps(*state.get_unchecked(i + 1));
            let w1_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let w1_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 1) as *const f32).add(8));
            acc_lo1 = _mm256_fmadd_ps(w1_lo, s1, acc_lo1);
            acc_hi1 = _mm256_fmadd_ps(w1_hi, s1, acc_hi1);

            let s2 = _mm256_set1_ps(*state.get_unchecked(i + 2));
            let w2_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let w2_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 2) as *const f32).add(8));
            acc_lo2 = _mm256_fmadd_ps(w2_lo, s2, acc_lo2);
            acc_hi2 = _mm256_fmadd_ps(w2_hi, s2, acc_hi2);

            let s3 = _mm256_set1_ps(*state.get_unchecked(i + 3));
            let w3_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let w3_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 3) as *const f32).add(8));
            acc_lo3 = _mm256_fmadd_ps(w3_lo, s3, acc_lo3);
            acc_hi3 = _mm256_fmadd_ps(w3_hi, s3, acc_hi3);
        });

        while i < len {
            let s = _mm256_set1_ps(*state.get_unchecked(i));
            let w_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let w_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            acc_lo0 = _mm256_fmadd_ps(w_lo, s, acc_lo0);
            acc_hi0 = _mm256_fmadd_ps(w_hi, s, acc_hi0);
            i += 1;
        }

        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo1);
        acc_lo2 = _mm256_add_ps(acc_lo2, acc_lo3);
        acc_lo0 = _mm256_add_ps(acc_lo0, acc_lo2);

        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi1);
        acc_hi2 = _mm256_add_ps(acc_hi2, acc_hi3);
        acc_hi0 = _mm256_add_ps(acc_hi0, acc_hi2);

        let mut out = [0.0f32; 16];
        _mm256_storeu_ps(out.as_mut_ptr(), acc_lo0);
        _mm256_storeu_ps(out.as_mut_ptr().add(8), acc_hi0);
        out
    }
}

/// Fused accumulate dual‑frame 16‑lane interleaved dot product
/// (`weights: &[[f32; 16]]`, `state_f0: &[f32]`, `state_f1: &[f32]`,
/// `init_f0: &[f32; 16]`, `init_f1: &[f32; 16]`) with AVX2/FMA.
///
/// Fuses the `init_f0`/`init_f1` accumulators (bias + mixin) into
/// `acc_f0_lo0`/`acc_f0_hi0` and `acc_f1_lo0`/`acc_f1_hi0`, avoiding an
/// extra pass over the outputs. Other unroll accumulator pairs
/// (`acc_f{0,1}_lo1..3`, `acc_f{0,1}_hi1..3`) are zero‑initialized.
///
/// # Strategy — two‑pass lo/hi
/// The original kernel maintained all 16 accumulator registers
/// simultaneously, forcing LLVM to spill 4 accumulators to the stack on
/// x86‑64 (only 16 ymm regs available vs. 16 acc + weight loads +
/// broadcasts). This kernel splits the computation into two independent
/// passes:
///
/// 1. **Pass 1 (lo):** iterates over all taps loading only `w_lo`
///    (`weights[i][0..8]`). Only 8 accumulator registers are live
///    (`acc_f0_lo{0..3}`, `acc_f1_lo{0..3}`), leaving sufficient
///    register headroom to eliminate spills.
/// 2. **Pass 2 (hi):** iterates over all taps loading only `w_hi`
///    (`weights[i][8..16]`, offset +8 floats). Only the 8 hi accumulator
///    registers are live.
///
/// The state broadcasts `s_f0`/`s_f1` are re‑executed in pass 2 and the
/// weight rows are re‑read — an acceptable cost compared to the
/// eliminated register spills. The same FMA sequence per lane and the
/// same tree reduction `(acc0+acc1)+(acc2+acc3)` guarantee bit‑exact
/// results identical to the original single‑pass kernel.
///
/// # Safety
/// Caller must ensure `weights.len() >= state_f0.len()` and
/// `weights.len() >= state_f1.len()`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_product_16x_f32_dual_accumulate_avx2(
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
    let mut acc_f0_lo0 = _mm256_loadu_ps(init_f0.as_ptr());
    let mut acc_f0_lo1 = _mm256_setzero_ps();
    let mut acc_f0_lo2 = _mm256_setzero_ps();
    let mut acc_f0_lo3 = _mm256_setzero_ps();
    let mut acc_f0_hi0 = _mm256_loadu_ps(init_f0.as_ptr().add(8));
    let mut acc_f0_hi1 = _mm256_setzero_ps();
    let mut acc_f0_hi2 = _mm256_setzero_ps();
    let mut acc_f0_hi3 = _mm256_setzero_ps();
    let mut acc_f1_lo0 = _mm256_loadu_ps(init_f1.as_ptr());
    let mut acc_f1_lo1 = _mm256_setzero_ps();
    let mut acc_f1_lo2 = _mm256_setzero_ps();
    let mut acc_f1_lo3 = _mm256_setzero_ps();
    let mut acc_f1_hi0 = _mm256_loadu_ps(init_f1.as_ptr().add(8));
    let mut acc_f1_hi1 = _mm256_setzero_ps();
    let mut acc_f1_hi2 = _mm256_setzero_ps();
    let mut acc_f1_hi3 = _mm256_setzero_ps();

    unsafe {
        let mut i = 0;

        dot4x_simd4!(i, len, {
            let w0_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_lo0 = _mm256_fmadd_ps(w0_lo, s_f0_0, acc_f0_lo0);
            acc_f1_lo0 = _mm256_fmadd_ps(w0_lo, s_f1_0, acc_f1_lo0);

            let w1_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 1) as *const f32);
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_lo1 = _mm256_fmadd_ps(w1_lo, s_f0_1, acc_f0_lo1);
            acc_f1_lo1 = _mm256_fmadd_ps(w1_lo, s_f1_1, acc_f1_lo1);

            let w2_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 2) as *const f32);
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_lo2 = _mm256_fmadd_ps(w2_lo, s_f0_2, acc_f0_lo2);
            acc_f1_lo2 = _mm256_fmadd_ps(w2_lo, s_f1_2, acc_f1_lo2);

            let w3_lo = _mm256_loadu_ps(weights.as_ptr().add(i + 3) as *const f32);
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_lo3 = _mm256_fmadd_ps(w3_lo, s_f0_3, acc_f0_lo3);
            acc_f1_lo3 = _mm256_fmadd_ps(w3_lo, s_f1_3, acc_f1_lo3);
        });

        while i < len {
            let w_lo = _mm256_loadu_ps(weights.as_ptr().add(i) as *const f32);
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_lo0 = _mm256_fmadd_ps(w_lo, s_f0, acc_f0_lo0);
            acc_f1_lo0 = _mm256_fmadd_ps(w_lo, s_f1, acc_f1_lo0);
            i += 1;
        }

        acc_f0_lo0 = _mm256_add_ps(acc_f0_lo0, acc_f0_lo1);
        acc_f0_lo2 = _mm256_add_ps(acc_f0_lo2, acc_f0_lo3);
        acc_f0_lo0 = _mm256_add_ps(acc_f0_lo0, acc_f0_lo2);
        acc_f1_lo0 = _mm256_add_ps(acc_f1_lo0, acc_f1_lo1);
        acc_f1_lo2 = _mm256_add_ps(acc_f1_lo2, acc_f1_lo3);
        acc_f1_lo0 = _mm256_add_ps(acc_f1_lo0, acc_f1_lo2);

        i = 0;

        dot4x_simd4!(i, len, {
            let w0_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            let s_f0_0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1_0 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_hi0 = _mm256_fmadd_ps(w0_hi, s_f0_0, acc_f0_hi0);
            acc_f1_hi0 = _mm256_fmadd_ps(w0_hi, s_f1_0, acc_f1_hi0);

            let w1_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 1) as *const f32).add(8));
            let s_f0_1 = _mm256_set1_ps(*state_f0.get_unchecked(i + 1));
            let s_f1_1 = _mm256_set1_ps(*state_f1.get_unchecked(i + 1));
            acc_f0_hi1 = _mm256_fmadd_ps(w1_hi, s_f0_1, acc_f0_hi1);
            acc_f1_hi1 = _mm256_fmadd_ps(w1_hi, s_f1_1, acc_f1_hi1);

            let w2_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 2) as *const f32).add(8));
            let s_f0_2 = _mm256_set1_ps(*state_f0.get_unchecked(i + 2));
            let s_f1_2 = _mm256_set1_ps(*state_f1.get_unchecked(i + 2));
            acc_f0_hi2 = _mm256_fmadd_ps(w2_hi, s_f0_2, acc_f0_hi2);
            acc_f1_hi2 = _mm256_fmadd_ps(w2_hi, s_f1_2, acc_f1_hi2);

            let w3_hi = _mm256_loadu_ps((weights.as_ptr().add(i + 3) as *const f32).add(8));
            let s_f0_3 = _mm256_set1_ps(*state_f0.get_unchecked(i + 3));
            let s_f1_3 = _mm256_set1_ps(*state_f1.get_unchecked(i + 3));
            acc_f0_hi3 = _mm256_fmadd_ps(w3_hi, s_f0_3, acc_f0_hi3);
            acc_f1_hi3 = _mm256_fmadd_ps(w3_hi, s_f1_3, acc_f1_hi3);
        });

        while i < len {
            let w_hi = _mm256_loadu_ps((weights.as_ptr().add(i) as *const f32).add(8));
            let s_f0 = _mm256_set1_ps(*state_f0.get_unchecked(i));
            let s_f1 = _mm256_set1_ps(*state_f1.get_unchecked(i));
            acc_f0_hi0 = _mm256_fmadd_ps(w_hi, s_f0, acc_f0_hi0);
            acc_f1_hi0 = _mm256_fmadd_ps(w_hi, s_f1, acc_f1_hi0);
            i += 1;
        }

        acc_f0_hi0 = _mm256_add_ps(acc_f0_hi0, acc_f0_hi1);
        acc_f0_hi2 = _mm256_add_ps(acc_f0_hi2, acc_f0_hi3);
        acc_f0_hi0 = _mm256_add_ps(acc_f0_hi0, acc_f0_hi2);
        acc_f1_hi0 = _mm256_add_ps(acc_f1_hi0, acc_f1_hi1);
        acc_f1_hi2 = _mm256_add_ps(acc_f1_hi2, acc_f1_hi3);
        acc_f1_hi0 = _mm256_add_ps(acc_f1_hi0, acc_f1_hi2);

        let mut out_f0 = [0.0f32; 16];
        let mut out_f1 = [0.0f32; 16];
        _mm256_storeu_ps(out_f0.as_mut_ptr(), acc_f0_lo0);
        _mm256_storeu_ps(out_f0.as_mut_ptr().add(8), acc_f0_hi0);
        _mm256_storeu_ps(out_f1.as_mut_ptr(), acc_f1_lo0);
        _mm256_storeu_ps(out_f1.as_mut_ptr().add(8), acc_f1_hi0);
        (out_f0, out_f1)
    }
}
