// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 polynomial exp/sigmoid kernels (degree-6 Taylor).
//!
//! Depends on scalar fallback functions from `super::high_fidelity`
//! and polynomial constants from `crate::math::constants`.

use super::high_fidelity::scalar_sigmoid_poly;
use crate::math::constants::*;
use core::arch::x86_64::*;

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 polynomial exp/sigmoid kernels
// ══════════════════════════════════════════════════════════════════════════════

/// Polynomial `exp(x)` for `__m512` — degree-6 Taylor polynomial with integer
/// range reduction `x = k·ln2 + r`.  AVX-512 counterpart of `simd_exp_poly_avx2`.
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.  Input clamped to
/// [-20, 20] to prevent overflow.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn simd_exp_poly_avx512(x: __m512) -> __m512 {
    let log2e = _mm512_set1_ps(POLY_LOG2_E);
    let ln2 = _mm512_set1_ps(POLY_LN2);
    let c6 = _mm512_set1_ps(POLY_EXP_C6);
    let c5 = _mm512_set1_ps(POLY_EXP_C5);
    let c4 = _mm512_set1_ps(POLY_EXP_C4);
    let c3 = _mm512_set1_ps(POLY_EXP_C3);
    let c2 = _mm512_set1_ps(POLY_EXP_C2);
    let one = _mm512_set1_ps(1.0f32);
    let bias = _mm512_set1_epi32(127);

    let k_f = _mm512_roundscale_ps(
        _mm512_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm512_fnmadd_ps(k_f, ln2, x);

    let p = _mm512_fmadd_ps(c6, r, c5);
    let p = _mm512_fmadd_ps(p, r, c4);
    let p = _mm512_fmadd_ps(p, r, c3);
    let p = _mm512_fmadd_ps(p, r, c2);
    let p = _mm512_fmadd_ps(p, r, one);
    let p = _mm512_fmadd_ps(p, r, one);

    let k_i = _mm512_cvtps_epi32(k_f);
    let exp_bits = _mm512_slli_epi32(_mm512_add_epi32(k_i, bias), 23);
    let scale = _mm512_castsi512_ps(exp_bits);
    _mm512_mul_ps(p, scale)
}

/// Polynomial `sigmoid(x)` for `__m512` — exp-based, branchless (AVX-512F/VL).
///
/// Formula: `σ(x) = 1 / (1 + e⁻ˣ)`.
/// Input clamped to [-20, 20] for overflow safety, output clamped to [0, 1].
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_sigmoid_poly_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm512_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm512_set1_ps(1.0f32);
    let zero = _mm512_set1_ps(0.0f32);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));
    let neg_x = _mm512_sub_ps(zero, x);
    let exp_neg_x = unsafe { simd_exp_poly_avx512(neg_x) };
    let den = _mm512_add_ps(one, exp_neg_x);
    let sig = _mm512_div_ps(one, den);
    _mm512_max_ps(zero, _mm512_min_ps(one, sig))
}

/// Applies polynomial Sigmoid activation to a slice of f32 using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn sigmoid_poly_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    unsafe {
        crate::activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_sigmoid_poly_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_sigmoid_poly(*item);
    }
}
