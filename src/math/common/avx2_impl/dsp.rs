// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 (x86-64-v3) dispatch macro for DSP kernels
//! (`convolve`, `apply_gain`, `apply_gain_ramp`, `butterfly`/`ibutterfly` (FFT),
//!  `batch_norm`, `crossfade_blend_mono`).
//!
//! Generates thin wrappers delegating to hand-optimized AVX2+FMA intrinsics
//! in [`crate::math::dsp`] and [`crate::math::dsp::stereo`]. Invoked by
//! [`dispatch_simd!`] when AVX2+FMA (x86-64-v3) is detected at runtime.

macro_rules! impl_avx2_dsp {
    () => {
        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length n; CPU supports AVX2+FMA
        // (x86-64-v3, verified by dispatch). No aliasing between slices.
        unsafe fn complex_mac_overwrite(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            out_re: &mut [f32],
            out_im: &mut [f32],
        ) {
            let n = h_re.len();
            let mut i = 0;
            while i + 8 <= n {
                // SAFETY: i is bounds-checked (i+8 <= n); unaligned loads/stores valid for f32 slices.
                unsafe {
                    let hr = _mm256_loadu_ps(h_re.as_ptr().add(i));
                    let hi = _mm256_loadu_ps(h_im.as_ptr().add(i));
                    let xr = _mm256_loadu_ps(x_re.as_ptr().add(i));
                    let xi = _mm256_loadu_ps(x_im.as_ptr().add(i));
                    let prod_re = _mm256_fmsub_ps(hr, xr, _mm256_mul_ps(hi, xi));
                    let prod_im = _mm256_fmadd_ps(hr, xi, _mm256_mul_ps(hi, xr));
                    _mm256_storeu_ps(out_re.as_mut_ptr().add(i), prod_re);
                    _mm256_storeu_ps(out_im.as_mut_ptr().add(i), prod_im);
                }
                i += 8;
            }
            for j in i..n {
                let hr = *h_re.get_unchecked(j);
                let hi = *h_im.get_unchecked(j);
                let xr = *x_re.get_unchecked(j);
                let xi = *x_im.get_unchecked(j);
                *out_re.get_unchecked_mut(j) = f32::mul_add(hr, xr, -hi * xi);
                *out_im.get_unchecked_mut(j) = f32::mul_add(hr, xi, hi * xr);
            }
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length n; CPU supports AVX2+FMA
        // (x86-64-v3, verified by dispatch). No aliasing between slices.
        unsafe fn complex_mac_accumulate(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            acc_re: &mut [f32],
            acc_im: &mut [f32],
        ) {
            let n = h_re.len();
            let mut i = 0;
            while i + 8 <= n {
                // SAFETY: i is bounds-checked (i+8 <= n); unaligned loads/stores valid for f32 slices.
                unsafe {
                    let hr = _mm256_loadu_ps(h_re.as_ptr().add(i));
                    let hi = _mm256_loadu_ps(h_im.as_ptr().add(i));
                    let xr = _mm256_loadu_ps(x_re.as_ptr().add(i));
                    let xi = _mm256_loadu_ps(x_im.as_ptr().add(i));
                    let prod_re = _mm256_fmsub_ps(hr, xr, _mm256_mul_ps(hi, xi));
                    let prod_im = _mm256_fmadd_ps(hr, xi, _mm256_mul_ps(hi, xr));
                    let cur_re = _mm256_loadu_ps(acc_re.as_ptr().add(i));
                    let cur_im = _mm256_loadu_ps(acc_im.as_ptr().add(i));
                    _mm256_storeu_ps(acc_re.as_mut_ptr().add(i), _mm256_add_ps(cur_re, prod_re));
                    _mm256_storeu_ps(acc_im.as_mut_ptr().add(i), _mm256_add_ps(cur_im, prod_im));
                }
                i += 8;
            }
            for j in i..n {
                let hr = *h_re.get_unchecked(j);
                let hi = *h_im.get_unchecked(j);
                let xr = *x_re.get_unchecked(j);
                let xi = *x_im.get_unchecked(j);
                *acc_re.get_unchecked_mut(j) += f32::mul_add(hr, xr, -hi * xi);
                *acc_im.get_unchecked_mut(j) += f32::mul_add(hr, xi, hi * xr);
            }
        }

        #[inline(always)]
        // SAFETY: re, im, tw_re, tw_im are valid for the described ranges;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        // In-place butterfly: re/im read from and written to the same array.
        unsafe fn fft_butterfly_stage(
            re: *mut f32,
            im: *mut f32,
            half: usize,
            tw_re: *const f32,
            tw_im: *const f32,
            group_start: usize,
            inverse: bool,
        ) {
            let top = group_start;
            let bot = group_start + half;
            let zero = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= half {
                // SAFETY: j+8 <= half, pointers are advanced by j and
                // top/bot indices are in bounds.
                unsafe {
                    let w_re = _mm256_loadu_ps(tw_re.add(j));
                    let w_im = if inverse {
                        _mm256_sub_ps(zero, _mm256_loadu_ps(tw_im.add(j)))
                    } else {
                        _mm256_loadu_ps(tw_im.add(j))
                    };

                    let re_top = _mm256_loadu_ps(re.add(top + j));
                    let im_top = _mm256_loadu_ps(im.add(top + j));
                    let re_bot = _mm256_loadu_ps(re.add(bot + j));
                    let im_bot = _mm256_loadu_ps(im.add(bot + j));

                    let t_re = _mm256_fmsub_ps(w_re, re_bot, _mm256_mul_ps(w_im, im_bot));
                    let t_im = _mm256_fmadd_ps(w_re, im_bot, _mm256_mul_ps(w_im, re_bot));

                    _mm256_storeu_ps(re.add(bot + j), _mm256_sub_ps(re_top, t_re));
                    _mm256_storeu_ps(im.add(bot + j), _mm256_sub_ps(im_top, t_im));
                    _mm256_storeu_ps(re.add(top + j), _mm256_add_ps(re_top, t_re));
                    _mm256_storeu_ps(im.add(top + j), _mm256_add_ps(im_top, t_im));
                }
                j += 8;
            }
            for j in j..half {
                // SAFETY: j < half, both top+j and bot+j are in bounds.
                unsafe {
                    let w_re = *tw_re.add(j);
                    let w_im = if inverse {
                        -(*tw_im.add(j))
                    } else {
                        *tw_im.add(j)
                    };

                    let re_idx1 = *re.add(top + j);
                    let im_idx1 = *im.add(top + j);
                    let re_idx2 = *re.add(bot + j);
                    let im_idx2 = *im.add(bot + j);

                    let t_re = f32::mul_add(w_re, re_idx2, -w_im * im_idx2);
                    let t_im = f32::mul_add(w_re, im_idx2, w_im * re_idx2);

                    *re.add(bot + j) = re_idx1 - t_re;
                    *im.add(bot + j) = im_idx1 - t_im;
                    *re.add(top + j) = re_idx1 + t_re;
                    *im.add(top + j) = im_idx1 + t_im;
                }
            }
        }
    };
}
pub(crate) use impl_avx2_dsp;
