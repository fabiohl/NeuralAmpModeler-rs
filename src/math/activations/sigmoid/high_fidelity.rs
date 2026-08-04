// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Polynomial Sigmoid activation kernels (f32 native weights + degree-6 Taylor exp), AVX2/FMA.
//!
//! Uses an exp-based formula with a degree-6 Taylor minimax polynomial
//! and integer range-reduction (`k = round(x·log₂e)`, `r = x − k·ln 2`).
//!
//! ```text
//! σ(x) = 1 / (1 + e⁻ˣ)
//! ```
//!
//! - Max absolute error (sigmoid): ≤ 2.1e-7 vs `f32::exp` reference on [-20, 20].
//! - Throughput (sigmoid): ~17 SIMD ops (1 exp + 1 div + add + clamp).
//!
//! Coefficients in `crate::math::constants` (`POLY_*`).

pub use super::high_fidelity_avx512::*;
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

    let k_f = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm256_fnmadd_ps(k_f, ln2, x);

    let p = _mm256_fmadd_ps(c6, r, c5);
    let p = _mm256_fmadd_ps(p, r, c4);
    let p = _mm256_fmadd_ps(p, r, c3);
    let p = _mm256_fmadd_ps(p, r, c2);
    let p = _mm256_fmadd_ps(p, r, one);
    let p = _mm256_fmadd_ps(p, r, one);

    let k_i = _mm256_cvtps_epi32(k_f);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, bias), 23);
    let scale = _mm256_castsi256_ps(exp_bits);
    _mm256_mul_ps(p, scale)
}

// ══════════════════════════════════════════════════════════════════════════════
// Public — polynomial Sigmoid kernels
// ══════════════════════════════════════════════════════════════════════════════

/// Polynomial `sigmoid(x)` for `__m256` — exp-based, branchless (AVX2/FMA).
///
/// Formula: `σ(x) = 1 / (1 + e⁻ˣ)`.
/// Input clamped to [-20, 20] for overflow safety, output clamped to [0, 1].
///
/// # Safety
/// The caller must guarantee AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn simd_sigmoid_poly_avx2(x: __m256) -> __m256 {
    let clamp_lo = _mm256_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm256_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm256_set1_ps(1.0f32);
    let zero = _mm256_set1_ps(0.0f32);

    let x = _mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, x));
    let neg_x = _mm256_sub_ps(zero, x);
    let exp_neg_x = unsafe { simd_exp_poly_avx2(neg_x) };
    let den = _mm256_add_ps(one, exp_neg_x);
    let sig = _mm256_div_ps(one, den);
    _mm256_max_ps(zero, _mm256_min_ps(one, sig))
}

// ══════════════════════════════════════════════════════════════════════════════
// Slice-level functions
// ══════════════════════════════════════════════════════════════════════════════

/// Applies polynomial Sigmoid activation to a slice of f32 using AVX2.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_poly_slice_avx2(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    while i + 8 <= len {
        unsafe {
            let x = _mm256_loadu_ps(slice.as_ptr().add(i));
            let y = simd_sigmoid_poly_avx2(x);
            _mm256_storeu_ps(slice.as_mut_ptr().add(i), y);
        }
        i += 8;
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_sigmoid_poly(*item);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Scalar polynomial sigmoid (degree-6 Taylor, range reduction)
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

/// Scalar polynomial `sigmoid(x)` — exp-based, using degree-6 Taylor.
/// Formula: `σ(x) = 1 / (1 + e⁻ˣ)`.
/// Max absolute error: ≤ 2.1e-7 vs `f32::exp` reference on [-20, 20].
#[inline]
pub fn scalar_sigmoid_poly(x: f32) -> f32 {
    let x = x.clamp(-POLY_ACTIVATION_CLAMP, POLY_ACTIVATION_CLAMP);
    let exp_neg_x = scalar_exp_poly_inner(-x);
    (1.0 / (1.0 + exp_neg_x)).clamp(0.0, 1.0)
}

#[cfg(test)]
#[path = "high_fidelity_test.rs"]
mod high_fidelity_test;
