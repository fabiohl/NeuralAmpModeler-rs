// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Polynomial Tanh activation kernels (f32 native weights + degree-6 Taylor exp), AVX2/FMA.
//!
//! Uses an exp-based formula with a degree-6 Taylor minimax polynomial
//! and integer range-reduction (`k = round(x·log₂e)`, `r = x − k·ln 2`).
//!
//! ```text
//! tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)    (exp-based, single-division, branchless)
//! ```
//!
//! - Max absolute error (tanh): ≤ 2.4e-7 vs `f32::tanh` on [-20, 20].
//! - Throughput (tanh): ~18 SIMD ops (1 exp + 1 div + 2 mul + 2 add/sub + clamp).
//! - Sigmoid polynomial kernels live in `crate::math::activations::sigmoid::high_fidelity`.
//!
//! Coefficients in `crate::math::constants` (`POLY_*`).

pub use super::high_fidelity_avx512::*;
use crate::math::activations::sigmoid::high_fidelity::simd_sigmoid_poly_avx2;
use crate::math::constants::*;
use core::arch::x86_64::*;

// ══════════════════════════════════════════════════════════════════════════════
// Internal — polynomial SIMD exp kernel (degree-6 Taylor, range reduction)
// ══════════════════════════════════════════════════════════════════════════════

/// Polynomial `exp(x)` for `__m256` — degree-6 Taylor polynomial with integer
/// range reduction `x = k·ln2 + r`.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.  Input clamped to
/// [-20, 20] to prevent overflow (`k ∈ [-29, 29]`, poses no int32 overflow).
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn simd_exp_poly_avx2(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(POLY_LOG2_E);
    let ln2 = _mm256_set1_ps(POLY_LN2);
    let c6 = _mm256_set1_ps(POLY_EXP_C6);
    let c5 = _mm256_set1_ps(POLY_EXP_C5);
    let c4 = _mm256_set1_ps(POLY_EXP_C4);
    let c3 = _mm256_set1_ps(POLY_EXP_C3);
    let c2 = _mm256_set1_ps(POLY_EXP_C2);
    let one = _mm256_set1_ps(1.0f32);
    let bias = _mm256_set1_epi32(127);

    // k = round(x * log₂e),  r = x − k·ln2  (∈ [-ln2/2, ln2/2])
    let k_f = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(k_f, ln2, x);

    // P(r) = (((((c6·r + c5)·r + c4)·r + c3)·r + c2)·r + 1)·r + 1
    let p = _mm256_fmadd_ps(c6, r, c5);
    let p = _mm256_fmadd_ps(p, r, c4);
    let p = _mm256_fmadd_ps(p, r, c3);
    let p = _mm256_fmadd_ps(p, r, c2);
    let p = _mm256_fmadd_ps(p, r, one); // c₁ = 1
    let p = _mm256_fmadd_ps(p, r, one); // c₀ = 1

    // Scale by 2ᵏ: reinterpret ((k+127) << 23) as f32 and multiply
    let k_i = _mm256_cvtps_epi32(k_f);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, bias), 23);
    let scale = _mm256_castsi256_ps(exp_bits);
    _mm256_mul_ps(p, scale)
}

// ══════════════════════════════════════════════════════════════════════════════
// Public — polynomial Tanh kernels
// ══════════════════════════════════════════════════════════════════════════════

/// Polynomial `tanh(x)` for `__m256` — exp-based, single-division (AVX2/FMA).
///
/// Formula: `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`.
/// Input clamped to [-20, 20] for overflow safety, output clamped to [-1, 1].
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_poly_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm256_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm256_set1_ps(1.0f32);
    let neg_one = _mm256_set1_ps(-1.0f32);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let exp_x = unsafe { simd_exp_poly_avx2(x) };
    let u2 = _mm256_mul_ps(exp_x, exp_x); // e²ˣ
    let num = _mm256_sub_ps(u2, one); // e²ˣ − 1
    let den = _mm256_add_ps(u2, one); // e²ˣ + 1
    let tanh_val = _mm256_div_ps(num, den);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, tanh_val))
}

/// Polynomial `tanh(x)` — dual 16-float path, single-division (AVX2/FMA).
///
/// Evaluates two independent `__m256` registers sharing constant broadcasts.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_poly_dual_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let clamp_lo = _mm256_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm256_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm256_set1_ps(1.0f32);
    let neg_one = _mm256_set1_ps(-1.0f32);

    let x1 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x1));
    let x2 = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x2));

    let exp1 = unsafe { simd_exp_poly_avx2(x1) };
    let exp2 = unsafe { simd_exp_poly_avx2(x2) };

    let u2_1 = _mm256_mul_ps(exp1, exp1);
    let u2_2 = _mm256_mul_ps(exp2, exp2);

    let num1 = _mm256_sub_ps(u2_1, one);
    let den1 = _mm256_add_ps(u2_1, one);
    let num2 = _mm256_sub_ps(u2_2, one);
    let den2 = _mm256_add_ps(u2_2, one);

    let res1 = _mm256_div_ps(num1, den1);
    let res2 = _mm256_div_ps(num2, den2);

    (
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res1)),
        _mm256_max_ps(neg_one, _mm256_min_ps(one, res2)),
    )
}

// ══════════════════════════════════════════════════════════════════════════════
// NR experimental Tanh variants — reciprocal + Newton-Raphson
// ══════════════════════════════════════════════════════════════════════════════
//
// TC3: Evaluate whether `vrcpps` + Newton-Raphson can replace the remaining
// `vdivps` in the polynomial tanh hot-path.  NR1 (~23-bit reciprocal) trades
// ~1 ulp precision for potential throughput gain; NR2 (~48-bit) saturates the
// f32 mantissa.
//
//   rcp_ps → y₀ ≈ 1/d          (~12-bit, ~3.6 decimal digits)
//   NR1    → y₁ = y₀·(2−d·y₀)  (~23-bit, ~7.2 decimal digits)
//   NR2    → y₂ = y₁·(2−d·y₁)  (~48-bit, saturates f32 mantissa)

/// Polynomial `tanh(x)` - NR1, single Newton-Raphson on reciprocal (AVX2/FMA).
///
/// Experimental variant (TC3). Replaces `_mm256_div_ps` with `_mm256_rcp_ps` +
/// 1×Newton-Raphson refinement (~23-bit reciprocal precision).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_poly_nr1_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm256_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm256_set1_ps(1.0f32);
    let neg_one = _mm256_set1_ps(-1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let exp_x = unsafe { simd_exp_poly_avx2(x) };
    let u2 = _mm256_mul_ps(exp_x, exp_x);
    let num = _mm256_sub_ps(u2, one);
    let den = _mm256_add_ps(u2, one);

    let mut r = _mm256_rcp_ps(den);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));

    let tanh_val = _mm256_mul_ps(num, r);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, tanh_val))
}

/// Polynomial `tanh(x)` - NR2, double Newton-Raphson on reciprocal (AVX2/FMA).
///
/// Experimental variant (TC3). Replaces `_mm256_div_ps` with `_mm256_rcp_ps` +
/// 2×Newton-Raphson refinement (saturates f32 mantissa).
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_poly_nr2_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm256_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm256_set1_ps(1.0f32);
    let neg_one = _mm256_set1_ps(-1.0f32);
    let two = _mm256_set1_ps(2.0f32);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));

    let exp_x = unsafe { simd_exp_poly_avx2(x) };
    let u2 = _mm256_mul_ps(exp_x, exp_x);
    let num = _mm256_sub_ps(u2, one);
    let den = _mm256_add_ps(u2, one);

    let mut r = _mm256_rcp_ps(den);
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));
    r = _mm256_mul_ps(r, _mm256_fnmadd_ps(den, r, two));

    let tanh_val = _mm256_mul_ps(num, r);
    _mm256_max_ps(neg_one, _mm256_min_ps(one, tanh_val))
}

// ══════════════════════════════════════════════════════════════════════════════
// Polynomial `(tanh(x1), sigmoid(x2))` — dual gate (AVX2/FMA)
// ══════════════════════════════════════════════════════════════════════════════

/// Polynomial `(tanh(x1), sigmoid(x2))` — dual gate (AVX2/FMA).
///
/// Used in WaveNet gated activation: `tanh(zf) * sigmoid(zg)`.
/// Delegates sigmoid to `sigmoid::high_fidelity::simd_sigmoid_poly_avx2`.
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_tanh_sigmoid_dual_poly_avx2(x1: __m256, x2: __m256) -> (__m256, __m256) {
    let t1 = unsafe { simd_tanh_poly_avx2(x1) };
    let s2 = unsafe { simd_sigmoid_poly_avx2(x2) };
    (t1, s2)
}

/// Applies polynomial Tanh activation to a slice of f32 using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn tanh_poly_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 16 <= len {
        unsafe {
            let x1 = _mm256_loadu_ps(slice.as_ptr().add(i));
            let x2 = _mm256_loadu_ps(slice.as_ptr().add(i + 8));
            let (y1, y2) = simd_tanh_poly_dual_avx2(x1, x2);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y1);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i + 8), y2);
        }
        i += 16;
    }

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_tanh_poly_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_tanh_poly(*item);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Scalar polynomial exp/tanh (degree-6 Taylor, range reduction)
// ══════════════════════════════════════════════════════════════════════════════

/// Scalar polynomial `exp(x)` — degree-6 Taylor polynomial with integer
/// range reduction `x = k·ln2 + r`.  Matches `simd_exp_poly_avx2` logic.
///
/// Input must be pre-clamped to [-POLY_ACTIVATION_CLAMP, POLY_ACTIVATION_CLAMP].
#[inline]
fn scalar_exp_poly_inner(x: f32) -> f32 {
    let k_f64 = (x as f64) * (POLY_LOG2_E as f64);
    let k = k_f64.round_ties_even() as f32;
    let r = (-k).mul_add(POLY_LN2, x);

    let p = POLY_EXP_C6.mul_add(r, POLY_EXP_C5);
    let p = p.mul_add(r, POLY_EXP_C4);
    let p = p.mul_add(r, POLY_EXP_C3);
    let p = p.mul_add(r, POLY_EXP_C2);
    let p = p.mul_add(r, 1.0);
    let p = p.mul_add(r, 1.0);

    let scale = f32::from_bits(((k as i32 + 127_i32) as u32) << 23);
    p * scale
}

/// Scalar polynomial `tanh(x)` — exp-based, using degree-6 Taylor.
/// Formula: `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`.
/// Max absolute error: ≤ 2.4e-7 vs `f32::tanh` on [-20, 20].
#[inline]
pub fn scalar_tanh_poly(x: f32) -> f32 {
    let x = x.clamp(-POLY_ACTIVATION_CLAMP, POLY_ACTIVATION_CLAMP);
    let exp_x = scalar_exp_poly_inner(x);
    let e2x = exp_x * exp_x;
    ((e2x - 1.0) / (e2x + 1.0)).clamp(-1.0, 1.0)
}

#[cfg(test)]
#[path = "high_fidelity_test.rs"]
mod high_fidelity_test;
