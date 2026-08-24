// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 polynomial exp/tanh kernels (degree-6 Taylor).
//!
//! Depends on scalar fallback functions from `super::high_fidelity`
//! and polynomial constants from `crate::math::constants`.
//! Sigmoid polynomial AVX-512 kernels live in
//! `crate::math::activations::sigmoid::high_fidelity_avx512`.

use super::high_fidelity::scalar_tanh_poly;
use crate::math::activations::sigmoid::high_fidelity_avx512::simd_sigmoid_poly_avx512;
use crate::math::constants::*;
use core::arch::x86_64::*;

// ══════════════════════════════════════════════════════════════════════════════
// AVX-512 polynomial exp/tanh/sigmoid kernels
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

    // k = round(x * log₂e),  r = x − k·ln2  (∈ [-ln2/2, ln2/2])
    let k_f = _mm512_roundscale_ps(
        _mm512_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );
    let r = _mm512_fnmadd_ps(k_f, ln2, x);

    // P(r) = (((((c6·r + c5)·r + c4)·r + c3)·r + c2)·r + 1)·r + 1
    let p = _mm512_fmadd_ps(c6, r, c5);
    let p = _mm512_fmadd_ps(p, r, c4);
    let p = _mm512_fmadd_ps(p, r, c3);
    let p = _mm512_fmadd_ps(p, r, c2);
    let p = _mm512_fmadd_ps(p, r, one);
    let p = _mm512_fmadd_ps(p, r, one);

    // Scale by 2ᵏ: reinterpret ((k+127) << 23) as f32 and multiply
    let k_i = _mm512_cvtps_epi32(k_f);
    let exp_bits = _mm512_slli_epi32(_mm512_add_epi32(k_i, bias), 23);
    let scale = _mm512_castsi512_ps(exp_bits);
    _mm512_mul_ps(p, scale)
}

/// Polynomial `tanh(x)` for `__m512` — exp-based, single-division (AVX-512F/VL).
///
/// Formula: `tanh(x) = (e²ˣ − 1) / (e²ˣ + 1)`.
/// Input clamped to [-20, 20] for overflow safety, output clamped to [-1, 1].
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_poly_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm512_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm512_set1_ps(1.0f32);
    let neg_one = _mm512_set1_ps(-1.0f32);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    // SAFETY: `simd_exp_poly_avx512` requires AVX-512F/VL, guaranteed by this
    // `unsafe fn`'s `#[target_feature(enable = "avx512f,avx512vl")]` and caller
    // contract.
    let exp_x = unsafe { simd_exp_poly_avx512(x) };
    let u2 = _mm512_mul_ps(exp_x, exp_x); // e²ˣ
    let num = _mm512_sub_ps(u2, one); // e²ˣ − 1
    let den = _mm512_add_ps(u2, one); // e²ˣ + 1
    let tanh_val = _mm512_div_ps(num, den);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, tanh_val))
}

/// Polynomial `(tanh(x1), sigmoid(x2))` — dual gate (AVX-512F/VL).
///
/// Used in WaveNet gated activation: `tanh(zf) * sigmoid(zg)`.
/// Delegates sigmoid to `sigmoid::high_fidelity_avx512::simd_sigmoid_poly_avx512`.
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_sigmoid_dual_poly_avx512(x1: __m512, x2: __m512) -> (__m512, __m512) {
    // SAFETY: `simd_tanh_poly_avx512` requires AVX-512F/VL, guaranteed by this
    // `unsafe fn`'s `#[target_feature(enable = "avx512f,avx512vl")]` and caller
    // contract.
    let t1 = unsafe { simd_tanh_poly_avx512(x1) };
    // SAFETY: `simd_sigmoid_poly_avx512` needs the same AVX-512F/VL support
    // guaranteed by this `unsafe fn`'s `#[target_feature]` and caller contract.
    let s2 = unsafe { simd_sigmoid_poly_avx512(x2) };
    (t1, s2)
}

/// Polynomial `tanh(x)` - NR1, single Newton-Raphson on reciprocal (AVX-512).
///
/// Experimental variant (TC3). Replaces `_mm512_div_ps` with `_mm512_rcp14_ps` +
/// 1×Newton-Raphson refinement (~23-bit reciprocal precision).
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_poly_nr1_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm512_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm512_set1_ps(1.0f32);
    let neg_one = _mm512_set1_ps(-1.0f32);
    let two = _mm512_set1_ps(2.0f32);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    // SAFETY: `simd_exp_poly_avx512` requires AVX-512F/VL, guaranteed by this
    // `unsafe fn`'s `#[target_feature(enable = "avx512f,avx512vl")]` and caller
    // contract.
    let exp_x = unsafe { simd_exp_poly_avx512(x) };
    let u2 = _mm512_mul_ps(exp_x, exp_x);
    let num = _mm512_sub_ps(u2, one);
    let den = _mm512_add_ps(u2, one);

    let mut r = _mm512_rcp14_ps(den);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));

    let tanh_val = _mm512_mul_ps(num, r);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, tanh_val))
}

/// Polynomial `tanh(x)` - NR2, double Newton-Raphson on reciprocal (AVX-512).
///
/// Experimental variant (TC3). Replaces `_mm512_div_ps` with `_mm512_rcp14_ps` +
/// 2×Newton-Raphson refinement (saturates f32 mantissa).
///
/// # Safety
/// The caller must guarantee AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn simd_tanh_poly_nr2_avx512(x: __m512) -> __m512 {
    let clamp_lo = _mm512_set1_ps(-POLY_ACTIVATION_CLAMP);
    let clamp_hi = _mm512_set1_ps(POLY_ACTIVATION_CLAMP);
    let one = _mm512_set1_ps(1.0f32);
    let neg_one = _mm512_set1_ps(-1.0f32);
    let two = _mm512_set1_ps(2.0f32);

    let x = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, x));

    // SAFETY: `simd_exp_poly_avx512` requires AVX-512F/VL, guaranteed by this
    // `unsafe fn`'s `#[target_feature(enable = "avx512f,avx512vl")]` and caller
    // contract.
    let exp_x = unsafe { simd_exp_poly_avx512(x) };
    let u2 = _mm512_mul_ps(exp_x, exp_x);
    let num = _mm512_sub_ps(u2, one);
    let den = _mm512_add_ps(u2, one);

    let mut r = _mm512_rcp14_ps(den);
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));
    r = _mm512_mul_ps(r, _mm512_fnmadd_ps(den, r, two));

    let tanh_val = _mm512_mul_ps(num, r);
    _mm512_max_ps(neg_one, _mm512_min_ps(one, tanh_val))
}

/// Applies polynomial Tanh activation to a slice of f32 using AVX-512.
///
/// # Safety
/// Requires AVX-512F and AVX-512VL support.
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn tanh_poly_slice_avx512(slice: &mut [f32]) {
    let mut i = 0;
    let len = slice.len();

    // SAFETY: `activation_simd_avx512!` runs its body while `i + 16 <= len` (it
    // loads/stores only at offset `i`), so the 16-lane access stays within
    // `slice`; `loadu`/`storeu` need no alignment.
    unsafe {
        crate::activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(slice.as_ptr().add(i));
            let y = simd_tanh_poly_avx512(x);
            _mm512_storeu_ps(slice.as_mut_ptr().add(i), y);
        });
    }

    for item in slice.iter_mut().skip(i) {
        *item = scalar_tanh_poly(*item);
    }
}
