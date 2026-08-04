// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized FastTanh (Padé rational approximation) activation kernels.

use crate::activation_simd_avx2;
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Applies FastTanh (Padé rational approximation) to a slice using AVX2+FMA.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
#[expect(
    clippy::excessive_precision,
    reason = "High-precision constants required for bit-exact numerical validation against reference"
)]
pub unsafe fn fast_tanh_slice_avx2(data: &mut [f32]) {
    let ca = _mm256_set1_ps(2.45550750702956_f32);
    let cb = _mm256_set1_ps(0.893229853513558_f32);
    let cc = _mm256_set1_ps(0.821226666969744_f32);
    let cd = _mm256_set1_ps(2.44506634652299_f32);
    let ce = _mm256_set1_ps(0.814642734961073_f32);
    let sign_mask = _mm256_set1_ps(-0.0_f32);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
                let ax1 = _mm256_andnot_ps(sign_mask, x1);
                let ax2 = _mm256_andnot_ps(sign_mask, x2);
                let x21 = _mm256_mul_ps(x1, x1);
                let x22 = _mm256_mul_ps(x2, x2);
                let num_inner1 = _mm256_fmadd_ps(cc, ax1, cb);
                let num_inner2 = _mm256_fmadd_ps(cc, ax2, cb);
                let num_poly1 = _mm256_fmadd_ps(num_inner1, x21, _mm256_fmadd_ps(ca, ax1, ca));
                let num_poly2 = _mm256_fmadd_ps(num_inner2, x22, _mm256_fmadd_ps(ca, ax2, ca));
                let num1 = _mm256_mul_ps(x1, num_poly1);
                let num2 = _mm256_mul_ps(x2, num_poly2);
                let xe1 = _mm256_mul_ps(ce, _mm256_mul_ps(x1, ax1));
                let xe2 = _mm256_mul_ps(ce, _mm256_mul_ps(x2, ax2));
                let xterm1 = _mm256_add_ps(x1, xe1);
                let xterm2 = _mm256_add_ps(x2, xe2);
                let abs_xterm1 = _mm256_andnot_ps(sign_mask, xterm1);
                let abs_xterm2 = _mm256_andnot_ps(sign_mask, xterm2);
                let den_inner1 = _mm256_add_ps(cd, x21);
                let den_inner2 = _mm256_add_ps(cd, x22);
                let den1 = _mm256_fmadd_ps(den_inner1, abs_xterm1, cd);
                let den2 = _mm256_fmadd_ps(den_inner2, abs_xterm2, cd);
                let y1 = _mm256_div_ps(num1, den1);
                let y2 = _mm256_div_ps(num2, den2);
                _mm256_storeu_ps(data.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(data.as_ptr().add(i));
                let ax = _mm256_andnot_ps(sign_mask, x);
                let x2 = _mm256_mul_ps(x, x);
                let num_inner = _mm256_fmadd_ps(cc, ax, cb);
                let num_poly = _mm256_fmadd_ps(num_inner, x2, _mm256_fmadd_ps(ca, ax, ca));
                let num = _mm256_mul_ps(x, num_poly);
                let xe = _mm256_mul_ps(ce, _mm256_mul_ps(x, ax));
                let xterm = _mm256_add_ps(x, xe);
                let abs_xterm = _mm256_andnot_ps(sign_mask, xterm);
                let den_inner = _mm256_add_ps(cd, x2);
                let den = _mm256_fmadd_ps(den_inner, abs_xterm, cd);
                let y = _mm256_div_ps(num, den);
                _mm256_storeu_ps(data.as_mut_ptr().add(i), y);
            }
        );
    }
    for x in data.iter_mut().skip(i) {
        *x = fast_tanh(*x);
    }
}

/// Applies FastTanh (Padé rational approximation) to a slice using AVX-512+FMA.
///
/// # Safety
/// Requires AVX-512F, AVX-512VL, AVX-512DQ, and FMA support.
#[target_feature(enable = "avx512f,avx512vl,avx512dq,fma")]
#[expect(
    clippy::excessive_precision,
    reason = "High-precision constants required for bit-exact numerical validation against reference"
)]
pub unsafe fn fast_tanh_slice_avx512(data: &mut [f32]) {
    let ca = _mm512_set1_ps(2.45550750702956_f32);
    let cb = _mm512_set1_ps(0.893229853513558_f32);
    let cc = _mm512_set1_ps(0.821226666969744_f32);
    let cd = _mm512_set1_ps(2.44506634652299_f32);
    let ce = _mm512_set1_ps(0.814642734961073_f32);
    let sign_mask = _mm512_set1_ps(-0.0_f32);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(data.as_ptr().add(i));
            let ax = _mm512_andnot_ps(sign_mask, x);
            let x2 = _mm512_mul_ps(x, x);
            let num_inner = _mm512_fmadd_ps(cc, ax, cb);
            let num_poly = _mm512_fmadd_ps(num_inner, x2, _mm512_fmadd_ps(ca, ax, ca));
            let num = _mm512_mul_ps(x, num_poly);
            let xe = _mm512_mul_ps(ce, _mm512_mul_ps(x, ax));
            let xterm = _mm512_add_ps(x, xe);
            let abs_xterm = _mm512_andnot_ps(sign_mask, xterm);
            let den_inner = _mm512_add_ps(cd, x2);
            let den = _mm512_fmadd_ps(den_inner, abs_xterm, cd);
            let y = _mm512_div_ps(num, den);
            _mm512_storeu_ps(data.as_mut_ptr().add(i), y);
        });
    }
    for x in data.iter_mut().skip(i) {
        *x = fast_tanh(*x);
    }
}

/// Fast rational Padé approximation for the tanh function.
#[inline(always)]
#[expect(
    clippy::excessive_precision,
    reason = "High-precision constants required for bit-exact numerical validation against reference"
)]
pub fn fast_tanh(x: f32) -> f32 {
    let ax = x.abs();
    let x2 = x * x;

    (x * (2.45550750702956
        + 2.45550750702956 * ax
        + (0.893229853513558 + 0.821226666969744 * ax) * x2))
        / (2.44506634652299 + (2.44506634652299 + x2) * (x + 0.814642734961073 * x * ax).abs())
}
