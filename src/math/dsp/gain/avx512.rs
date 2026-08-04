// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

use crate::gain_kernel_avx512_masked;
use crate::gain_kernel_avx512_scalar;
use crate::gain_simd_avx512;
use core::arch::x86_64::*;

/// Fused gain + dither: `data[i] = data[i] * gain + offset` in a single pass using AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_then_dither_avx512(data: &mut [f32], gain: f32, offset: f32) {
    let len = data.len();
    let vg = _mm512_set1_ps(gain);
    let vo = _mm512_set1_ps(offset);
    let mut i = 0;
    gain_kernel_avx512_masked!(
        i,
        len,
        {
            let v = _mm512_loadu_ps(data.as_ptr().add(i));
            _mm512_storeu_ps(data.as_mut_ptr().add(i), _mm512_fmadd_ps(v, vg, vo));
        },
        {
            let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
            let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
            _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, _mm512_fmadd_ps(v, vg, vo));
        }
    );
}

/// Adds a broadcast constant to every element of a mono buffer using AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn apply_dither_add_avx512(data: &mut [f32], offset: f32) {
    let len = data.len();
    let voffset = _mm512_set1_ps(offset);
    let mut i = 0;
    gain_kernel_avx512_masked!(
        i,
        len,
        {
            let p = data.as_mut_ptr().add(i);
            _mm512_storeu_ps(p, _mm512_add_ps(_mm512_loadu_ps(p), voffset));
        },
        {
            let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
            let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
            _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, _mm512_add_ps(v, voffset));
        }
    );
}

/// Applies constant gain to a mono buffer using AVX-512.
#[target_feature(enable = "avx512f")]
pub unsafe fn apply_gain_avx512(data: &mut [f32], gain: f32) {
    let len = data.len();
    let vg = _mm512_set1_ps(gain);
    let mut i = 0;
    gain_kernel_avx512_masked!(
        i,
        len,
        {
            let v = _mm512_loadu_ps(data.as_ptr().add(i));
            _mm512_storeu_ps(data.as_mut_ptr().add(i), _mm512_mul_ps(v, vg));
        },
        {
            let mask = _cvtu32_mask16((1u32 << (len - i)) - 1);
            let v = _mm512_maskz_loadu_ps(mask, data.as_ptr().add(i));
            _mm512_mask_storeu_ps(data.as_mut_ptr().add(i), mask, _mm512_mul_ps(v, vg));
        }
    );
}

/// Applies gain and detects clipping in mono in a single pass using AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_and_detect_clipping_mono_avx512(data: &mut [f32], gain: f32) -> bool {
    let len = data.len();
    let v_gain = _mm512_set1_ps(gain);
    let v_limit = _mm512_set1_ps(1.0);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut k_clip = 0u16;
    let mut i = 0;

    gain_simd_avx512!(i, len, {
        let p = data.as_mut_ptr().add(i);
        let v = _mm512_loadu_ps(p);
        let g = _mm512_mul_ps(v, v_gain);
        _mm512_storeu_ps(p, g);
        let abs = _mm512_andnot_ps(sign_mask, g);
        let k = _mm512_cmp_ps_mask(abs, v_limit, _CMP_GT_OQ);
        k_clip |= k;
    });

    let mut clipped = k_clip != 0;

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

/// Applies gain and detects clipping in stereo in a single pass using AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_and_detect_clipping_stereo_avx512(
    left: &mut [f32],
    right: &mut [f32],
    gain: f32,
) -> bool {
    let n = core::cmp::min(left.len(), right.len());
    let v_gain = _mm512_set1_ps(gain);
    let v_limit = _mm512_set1_ps(1.0);
    let sign_mask = _mm512_set1_ps(-0.0f32);
    let mut k_clip = 0u16;
    let mut i = 0;

    gain_simd_avx512!(i, n, {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);

        let vl = _mm512_loadu_ps(pl);
        let vr = _mm512_loadu_ps(pr);

        let gl = _mm512_mul_ps(vl, v_gain);
        let gr = _mm512_mul_ps(vr, v_gain);

        _mm512_storeu_ps(pl, gl);
        _mm512_storeu_ps(pr, gr);

        let abs_l = _mm512_andnot_ps(sign_mask, gl);
        let abs_r = _mm512_andnot_ps(sign_mask, gr);

        let k_l = _mm512_cmp_ps_mask(abs_l, v_limit, _CMP_GT_OQ);
        let k_r = _mm512_cmp_ps_mask(abs_r, v_limit, _CMP_GT_OQ);

        k_clip |= k_l | k_r;
    });

    let mut clipped = k_clip != 0;

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

/// Applies constant gain in stereo via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_gain_stereo_avx512(left: &mut [f32], right: &mut [f32], gain: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let zmm_gain = _mm512_set1_ps(gain);
    let mut i = 0;
    gain_kernel_avx512_scalar!(
        i,
        n,
        {
            let pl = left.as_mut_ptr().add(i);
            let pr = right.as_mut_ptr().add(i);
            _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), zmm_gain));
            _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), zmm_gain));
        },
        {
            *left.get_unchecked_mut(i) *= gain;
            *right.get_unchecked_mut(i) *= gain;
        }
    );
}

/// Applies linear gain ramp in stereo via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_stereo_avx512(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
    let n = core::cmp::min(left.len(), right.len());
    let mut i = 0;
    let mut current_ramp = _mm512_set_ps(
        start + 15.0 * step,
        start + 14.0 * step,
        start + 13.0 * step,
        start + 12.0 * step,
        start + 11.0 * step,
        start + 10.0 * step,
        start + 9.0 * step,
        start + 8.0 * step,
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_16 = _mm512_set1_ps(16.0 * step);
    gain_simd_avx512!(i, n, {
        let pl = left.as_mut_ptr().add(i);
        let pr = right.as_mut_ptr().add(i);
        _mm512_storeu_ps(pl, _mm512_mul_ps(_mm512_loadu_ps(pl), current_ramp));
        _mm512_storeu_ps(pr, _mm512_mul_ps(_mm512_loadu_ps(pr), current_ramp));
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
    });
    let mut g = start + (i as f32) * step;
    while i < n {
        *left.get_unchecked_mut(i) *= g;
        *right.get_unchecked_mut(i) *= g;
        g += step;
        i += 1;
    }
}

/// Applies linear gain ramp to a mono buffer via AVX-512.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn apply_ramp_avx512(buffer: &mut [f32], start: f32, step: f32) {
    let len = buffer.len();
    let mut i = 0;
    let mut current_ramp = _mm512_set_ps(
        start + 15.0 * step,
        start + 14.0 * step,
        start + 13.0 * step,
        start + 12.0 * step,
        start + 11.0 * step,
        start + 10.0 * step,
        start + 9.0 * step,
        start + 8.0 * step,
        start + 7.0 * step,
        start + 6.0 * step,
        start + 5.0 * step,
        start + 4.0 * step,
        start + 3.0 * step,
        start + 2.0 * step,
        start + 1.0 * step,
        start,
    );
    let v_step_16 = _mm512_set1_ps(16.0 * step);
    gain_simd_avx512!(i, len, {
        let ptr = buffer.as_mut_ptr().add(i);
        _mm512_storeu_ps(ptr, _mm512_mul_ps(_mm512_loadu_ps(ptr), current_ramp));
        current_ramp = _mm512_add_ps(current_ramp, v_step_16);
    });
    let mut m = start + (i as f32) * step;
    while i < len {
        *buffer.get_unchecked_mut(i) *= m;
        m += step;
        i += 1;
    }
}

/// Crossfade blend (mono): `out[i] = fma(pending[i] - out[i], t, out[i])`.
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn crossfade_blend_mono_avx512(out: &mut [f32], pending: &[f32], t: f32) {
    let n = core::cmp::min(out.len(), pending.len());
    let vt = _mm512_set1_ps(t);
    let mut i = 0;
    gain_kernel_avx512_masked!(
        i,
        n,
        {
            let v_out = _mm512_loadu_ps(out.as_ptr().add(i));
            let v_pending = _mm512_loadu_ps(pending.as_ptr().add(i));
            let v_diff = _mm512_sub_ps(v_pending, v_out);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), _mm512_fmadd_ps(v_diff, vt, v_out));
        },
        {
            let mask = _cvtu32_mask16((1u32 << (n - i)) - 1);
            let v_out = _mm512_maskz_loadu_ps(mask, out.as_ptr().add(i));
            let v_pending = _mm512_maskz_loadu_ps(mask, pending.as_ptr().add(i));
            let v_diff = _mm512_sub_ps(v_pending, v_out);
            _mm512_mask_storeu_ps(
                out.as_mut_ptr().add(i),
                mask,
                _mm512_fmadd_ps(v_diff, vt, v_out),
            );
        }
    );
}
