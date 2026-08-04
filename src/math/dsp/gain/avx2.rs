// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::gain_kernel_avx2;
use crate::gain_simd_avx2;
use core::arch::x86_64::*;

/// Applies constant gain to a mono buffer using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_avx2(data: &mut [f32], gain: f32) {
    let len = data.len();
    let vg = _mm256_set1_ps(gain);
    let mut i = 0;
    gain_kernel_avx2!(
        i,
        len,
        {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_mul_ps(v, vg));
        },
        {
            data[i] *= gain;
        }
    );
}

/// Applies gain and detects clipping in mono in a single pass using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_and_detect_clipping_mono_avx2(data: &mut [f32], gain: f32) -> bool {
    let len = data.len();
    let vg = _mm256_set1_ps(gain);
    let limit = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut any_clip = _mm256_setzero_ps();
    let mut i = 0;

    gain_simd_avx2!(i, len, {
        let p = data.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(p);
        let g = _mm256_mul_ps(v, vg);
        _mm256_storeu_ps(p, g);
        let abs = _mm256_andnot_ps(sign_mask, g);
        let cmp = _mm256_cmp_ps(abs, limit, _CMP_GT_OQ);
        any_clip = _mm256_or_ps(any_clip, cmp);
    });

    let mut clipped = _mm256_movemask_ps(any_clip) != 0;

    while i < len {
        let v = *data.get_unchecked(i) * gain;
        *data.get_unchecked_mut(i) = v;
        if !clipped && v.abs() > 1.0 {
            clipped = true;
        }
        i += 1;
    }
    clipped
}

/// Applies gain and detects clipping in stereo in a single pass using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_and_detect_clipping_stereo_avx2(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    let n = core::cmp::min(left.len(), right.len());
    let ymm_gain = _mm256_set1_ps(gain);
    let limit = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut any_clip = _mm256_setzero_ps();
    let mut i = 0;

    gain_simd_avx2!(i, n, {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);

        let vl = _mm256_loadu_ps(pl);
        let vr = _mm256_loadu_ps(pr);

        let gl = _mm256_mul_ps(vl, ymm_gain);
        let gr = _mm256_mul_ps(vr, ymm_gain);

        _mm256_storeu_ps(pl, gl);
        _mm256_storeu_ps(pr, gr);

        let abs_l = _mm256_andnot_ps(sign_mask, gl);
        let abs_r = _mm256_andnot_ps(sign_mask, gr);

        let cmp_l = _mm256_cmp_ps(abs_l, limit, _CMP_GT_OQ);
        let cmp_r = _mm256_cmp_ps(abs_r, limit, _CMP_GT_OQ);

        any_clip = _mm256_or_ps(any_clip, _mm256_or_ps(cmp_l, cmp_r));
    });

    let mut clipped = _mm256_movemask_ps(any_clip) != 0;

    while i < n {
        let vl = *left.get_unchecked(i) * gain;
        let vr = *right.get_unchecked(i) * gain;
        *left.get_unchecked_mut(i) = vl;
        *right.get_unchecked_mut(i) = vr;
        if !clipped && (vl.abs() > 1.0 || vr.abs() > 1.0) {
            clipped = true;
        }
        i += 1;
    }
    clipped
}

/// Applies constant gain in stereo via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_gain_stereo_avx2(left: &mut [f32], right: &mut [f32], gain: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let ymm_gain = _mm256_set1_ps(gain);
    let mut i = 0;
    gain_kernel_avx2!(
        i,
        n,
        {
            let pl = left.as_mut_ptr().add(i);
            let pr = right.as_mut_ptr().add(i);
            _mm256_storeu_ps(pl, _mm256_mul_ps(_mm256_loadu_ps(pl), ymm_gain));
            _mm256_storeu_ps(pr, _mm256_mul_ps(_mm256_loadu_ps(pr), ymm_gain));
        },
        {
            *left.get_unchecked_mut(i) *= gain;
            *right.get_unchecked_mut(i) *= gain;
        }
    );
}

/// Applies linear gain ramp in stereo via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_ramp_stereo_avx2(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let mut current_ramp = _mm256_set_ps(
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_8 = _mm256_set1_ps(8.0 * step);
    gain_simd_avx2!(i, n, {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm256_storeu_ps(pl, _mm256_mul_ps(_mm256_loadu_ps(pl), current_ramp));
        _mm256_storeu_ps(pr, _mm256_mul_ps(_mm256_loadu_ps(pr), current_ramp));
        current_ramp = _mm256_add_ps(current_ramp, v_step_8);
    });
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}

/// Fused gain + dither: `data[i] = data[i] * gain + offset` in a single pass using AVX2+FMA.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_gain_then_dither_avx2(data: &mut [f32], gain: f32, offset: f32) {
    let len = data.len();
    let vg = _mm256_set1_ps(gain);
    let vo = _mm256_set1_ps(offset);
    let mut i = 0;
    gain_kernel_avx2!(
        i,
        len,
        {
            let v = _mm256_loadu_ps(data.as_ptr().add(i));
            _mm256_storeu_ps(data.as_mut_ptr().add(i), _mm256_fmadd_ps(v, vg, vo));
        },
        {
            *data.get_unchecked_mut(i) = f32::mul_add(*data.get_unchecked(i), gain, offset);
        }
    );
}

/// Adds a broadcast constant to every element of a mono buffer using AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_dither_add_avx2(data: &mut [f32], offset: f32) {
    let len = data.len();
    let voffset = _mm256_set1_ps(offset);
    let mut i = 0;
    gain_kernel_avx2!(
        i,
        len,
        {
            let p = data.as_mut_ptr().add(i);
            _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), voffset));
        },
        {
            *data.get_unchecked_mut(i) += offset;
        }
    );
}

/// Crossfade blend (mono): `out[i] = fma(pending[i] - out[i], t, out[i])`.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn crossfade_blend_mono_avx2(out: &mut [f32], pending: &[f32], t: f32) {
    let n = core::cmp::min(out.len(), pending.len());
    let mut i = 0;
    let vt = _mm256_set1_ps(t);
    gain_simd_avx2!(i, n, {
        let v_out = _mm256_loadu_ps(out.as_ptr().add(i));
        let v_pending = _mm256_loadu_ps(pending.as_ptr().add(i));
        let v_diff = _mm256_sub_ps(v_pending, v_out);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(v_diff, vt, v_out));
    });
    let one_minus_t = 1.0 - t;
    while i < n {
        *out.get_unchecked_mut(i) =
            *out.get_unchecked(i) * one_minus_t + *pending.get_unchecked(i) * t;
        i += 1;
    }
}

/// Applies linear gain ramp to a mono buffer via AVX2.
#[target_feature(enable = "avx2")]
pub unsafe fn apply_ramp_avx2(buffer: &mut [f32], start: f32, step: f32) {
    let len = buffer.len();
    let mut i = 0;
    let mut current_ramp = _mm256_set_ps(
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_8 = _mm256_set1_ps(8.0 * step);
    gain_simd_avx2!(i, len, {
        let ptr = buffer.as_mut_ptr().add(i);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(_mm256_loadu_ps(ptr), current_ramp));
        current_ramp = _mm256_add_ps(current_ramp, v_step_8);
    });
    let mut m = start + (i as f32) * step;
    while i < len {
        *buffer.get_unchecked_mut(i) *= m;
        m += step;
        i += 1;
    }
}
