// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Optimized LeakyHardTanh activation kernels.

use crate::activation_simd_avx2;
use crate::activation_simd_avx512;
use core::arch::x86_64::*;

/// Applies LeakyHardTanh (branchless blend) to a slice using AVX2+FMA.
///
/// # Safety
/// Requires AVX2 and FMA support.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn leaky_hard_tanh_slice_avx2(
    data: &mut [f32],
    min_val: f32,
    max_val: f32,
    min_slope: f32,
    max_slope: f32,
) {
    let min_v = _mm256_set1_ps(min_val);
    let max_v = _mm256_set1_ps(max_val);
    let min_sl_v = _mm256_set1_ps(min_slope);
    let max_sl_v = _mm256_set1_ps(max_slope);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx2!(
            i,
            len,
            {
                let x1 = _mm256_loadu_ps(data.as_ptr().add(i));
                let x2 = _mm256_loadu_ps(data.as_ptr().add(i + 8));
                let lt_min1 = _mm256_cmp_ps(x1, min_v, _CMP_LT_OS);
                let lt_min2 = _mm256_cmp_ps(x2, min_v, _CMP_LT_OS);
                let lo_part1 = _mm256_fmadd_ps(_mm256_sub_ps(x1, min_v), min_sl_v, min_v);
                let lo_part2 = _mm256_fmadd_ps(_mm256_sub_ps(x2, min_v), min_sl_v, min_v);
                let gt_max1 = _mm256_cmp_ps(x1, max_v, _CMP_GT_OS);
                let gt_max2 = _mm256_cmp_ps(x2, max_v, _CMP_GT_OS);
                let hi_part1 = _mm256_fmadd_ps(_mm256_sub_ps(x1, max_v), max_sl_v, max_v);
                let hi_part2 = _mm256_fmadd_ps(_mm256_sub_ps(x2, max_v), max_sl_v, max_v);
                let y1 =
                    _mm256_blendv_ps(_mm256_blendv_ps(x1, lo_part1, lt_min1), hi_part1, gt_max1);
                let y2 =
                    _mm256_blendv_ps(_mm256_blendv_ps(x2, lo_part2, lt_min2), hi_part2, gt_max2);
                _mm256_storeu_ps(data.as_mut_ptr().add(i), y1);
                _mm256_storeu_ps(data.as_mut_ptr().add(i + 8), y2);
            },
            {
                let x = _mm256_loadu_ps(data.as_ptr().add(i));
                let lt_min = _mm256_cmp_ps(x, min_v, _CMP_LT_OS);
                let lo_part = _mm256_fmadd_ps(_mm256_sub_ps(x, min_v), min_sl_v, min_v);
                let gt_max = _mm256_cmp_ps(x, max_v, _CMP_GT_OS);
                let hi_part = _mm256_fmadd_ps(_mm256_sub_ps(x, max_v), max_sl_v, max_v);
                let y = _mm256_blendv_ps(_mm256_blendv_ps(x, lo_part, lt_min), hi_part, gt_max);
                _mm256_storeu_ps(data.as_mut_ptr().add(i), y);
            }
        );
    }
    for x in data.iter_mut().skip(i) {
        if *x < min_val {
            *x = (*x - min_val) * min_slope + min_val;
        } else if *x > max_val {
            *x = (*x - max_val) * max_slope + max_val;
        }
    }
}

/// Applies LeakyHardTanh (branchless blend) to a slice using AVX-512+FMA.
///
/// # Safety
/// Requires AVX-512F, AVX-512VL, and FMA support.
#[target_feature(enable = "avx512f,avx512vl,fma")]
pub unsafe fn leaky_hard_tanh_slice_avx512(
    data: &mut [f32],
    min_val: f32,
    max_val: f32,
    min_slope: f32,
    max_slope: f32,
) {
    let min_v = _mm512_set1_ps(min_val);
    let max_v = _mm512_set1_ps(max_val);
    let min_sl_v = _mm512_set1_ps(min_slope);
    let max_sl_v = _mm512_set1_ps(max_slope);
    let mut i = 0;
    let len = data.len();
    unsafe {
        activation_simd_avx512!(i, len, {
            let x = _mm512_loadu_ps(data.as_ptr().add(i));
            let lt_min = _mm512_cmp_ps_mask(x, min_v, _CMP_LT_OS);
            let lo_part = _mm512_fmadd_ps(_mm512_sub_ps(x, min_v), min_sl_v, min_v);
            let gt_max = _mm512_cmp_ps_mask(x, max_v, _CMP_GT_OS);
            let hi_part = _mm512_fmadd_ps(_mm512_sub_ps(x, max_v), max_sl_v, max_v);
            let temp = _mm512_mask_blend_ps(lt_min, x, lo_part);
            let y = _mm512_mask_blend_ps(gt_max, temp, hi_part);
            _mm512_storeu_ps(data.as_mut_ptr().add(i), y);
        });
    }
    for x in data.iter_mut().skip(i) {
        if *x < min_val {
            *x = (*x - min_val) * min_slope + min_val;
        } else if *x > max_val {
            *x = (*x - max_val) * max_slope + max_val;
        }
    }
}

/// Scalar LeakyHardTanh.
#[inline(always)]
pub fn leaky_hard_tanh(x: f32, min_val: f32, max_val: f32, min_slope: f32, max_slope: f32) -> f32 {
    if x < min_val {
        (x - min_val) * min_slope + min_val
    } else if x > max_val {
        (x - max_val) * max_slope + max_val
    } else {
        x
    }
}
