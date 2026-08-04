// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 dispatch macro for DSP kernels (`impl_avx512_dsp!()`).
//!
//! Covers `convolve`, `apply_gain`, `apply_gain_ramp`, `butterfly`/`ibutterfly`
//! (FFT), `batch_norm`, and `crossfade_blend_mono`. Delegates to AVX-512
//! intrinsics kernels in [`crate::math::dsp`] and
//! [`crate::math::dsp::stereo`]. Invoked by [`dispatch_simd!`].

macro_rules! impl_avx512_dsp {
    () => {
        #[inline(always)]
        // SAFETY: coeffs is a 64-byte-aligned raw pointer to taps valid f32 elements;
        // input_l and input_r are valid raw pointers to taps valid f32 elements;
        // CPU supports AVX-512F (verified by dispatch). Coeffs use aligned 512-bit loads.
        unsafe fn convolve_stereo(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: coeffs, input_l, input_r satisfy function invariants (64-byte alignment
            // for coeffs, taps valid elements at each pointer).
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_avx512(coeffs, input_l, input_r, taps)
            }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 are 64-byte-aligned pointers to taps valid f32 elements;
        // input_l, input_r are valid pointers to taps valid f32 elements;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn convolve_stereo_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            // SAFETY: all raw pointers satisfy function invariants (aligned coeffs, taps elements).
            unsafe {
                crate::math::dsp::stereo::convolve_stereo_dual_avx512(
                    coeffs0, coeffs1, input_l, input_r, taps,
                )
            }
        }

        #[inline(always)]
        // SAFETY: coeffs is a 64-byte-aligned pointer to taps valid f32 elements;
        // input is a valid pointer to taps valid f32 elements; CPU supports AVX-512F.
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: coeffs and input satisfy function invariants.
            unsafe { crate::math::dsp::stereo::convolve_mono_avx512(coeffs, input, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 are 64-byte-aligned pointers to taps valid f32 elements;
        // input is a valid pointer to taps valid f32 elements; CPU supports AVX-512F.
        unsafe fn convolve_mono_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: all raw pointers satisfy function invariants.
            unsafe {
                crate::math::dsp::stereo::convolve_mono_dual_avx512(coeffs0, coeffs1, input, taps)
            }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512F (verified by dispatch). Kernel uses unaligned loads/stores.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            // SAFETY: data and gain satisfy function invariants.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_mono_avx512(data, gain)
            }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512F.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo_avx512(
                    left, right, gain,
                )
            }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512F.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_stereo_avx512(left, right, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512F (verified by dispatch).
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            // SAFETY: data and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_avx512(data, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; start and step are finite f32 values;
        // CPU supports AVX-512F.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            // SAFETY: data, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_avx512(data, start, step) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // start and step are finite f32 values; CPU supports AVX-512F.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            // SAFETY: left, right, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_stereo_avx512(left, right, start, step) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain and offset are finite f32 values;
        // CPU supports AVX-512F (verified by dispatch). Kernel uses unaligned 512-bit loads/stores.
        unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32) {
            // SAFETY: data, gain, and offset satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_then_dither_avx512(data, gain, offset) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; offset is a finite f32;
        // CPU supports AVX-512F.
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            // SAFETY: data and offset satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_dither_add_avx512(data, offset) }
        }

        #[inline(always)]
        // SAFETY: out and pending are valid f32 slices of equal length; t is a finite f32;
        // CPU supports AVX-512F.
        unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
            // SAFETY: out, pending, and t satisfy function invariants.
            unsafe { crate::math::dsp::gain::crossfade_blend_mono_avx512(out, pending, t) }
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length n; CPU supports AVX-512F
        // (verified by dispatch). No aliasing between slices.
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
            while i + 16 <= n {
                // SAFETY: i is bounds-checked (i+16 <= n); unaligned 512-bit loads/stores valid
                // for f32 slices.
                unsafe {
                    let hr = _mm512_loadu_ps(h_re.as_ptr().add(i));
                    let hi = _mm512_loadu_ps(h_im.as_ptr().add(i));
                    let xr = _mm512_loadu_ps(x_re.as_ptr().add(i));
                    let xi = _mm512_loadu_ps(x_im.as_ptr().add(i));
                    let prod_re = _mm512_fmsub_ps(hr, xr, _mm512_mul_ps(hi, xi));
                    let prod_im = _mm512_fmadd_ps(hr, xi, _mm512_mul_ps(hi, xr));
                    _mm512_storeu_ps(out_re.as_mut_ptr().add(i), prod_re);
                    _mm512_storeu_ps(out_im.as_mut_ptr().add(i), prod_im);
                }
                i += 16;
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
        // SAFETY: all 6 slices are valid f32 slices of equal length n; CPU supports AVX-512F
        // (verified by dispatch). No aliasing between slices.
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
            while i + 16 <= n {
                // SAFETY: i is bounds-checked (i+16 <= n); unaligned 512-bit loads/stores valid
                // for f32 slices.
                unsafe {
                    let hr = _mm512_loadu_ps(h_re.as_ptr().add(i));
                    let hi = _mm512_loadu_ps(h_im.as_ptr().add(i));
                    let xr = _mm512_loadu_ps(x_re.as_ptr().add(i));
                    let xi = _mm512_loadu_ps(x_im.as_ptr().add(i));
                    let prod_re = _mm512_fmsub_ps(hr, xr, _mm512_mul_ps(hi, xi));
                    let prod_im = _mm512_fmadd_ps(hr, xi, _mm512_mul_ps(hi, xr));
                    let cur_re = _mm512_loadu_ps(acc_re.as_ptr().add(i));
                    let cur_im = _mm512_loadu_ps(acc_im.as_ptr().add(i));
                    _mm512_storeu_ps(acc_re.as_mut_ptr().add(i), _mm512_add_ps(cur_re, prod_re));
                    _mm512_storeu_ps(acc_im.as_mut_ptr().add(i), _mm512_add_ps(cur_im, prod_im));
                }
                i += 16;
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
        // CPU supports AVX-512F (verified by dispatch).
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
            let zero = _mm512_setzero_ps();
            let mut j = 0;
            while j + 16 <= half {
                // SAFETY: j+16 <= half, pointers are advanced by j.
                unsafe {
                    let w_re = _mm512_loadu_ps(tw_re.add(j));
                    let w_im = if inverse {
                        _mm512_sub_ps(zero, _mm512_loadu_ps(tw_im.add(j)))
                    } else {
                        _mm512_loadu_ps(tw_im.add(j))
                    };

                    let re_top = _mm512_loadu_ps(re.add(top + j));
                    let im_top = _mm512_loadu_ps(im.add(top + j));
                    let re_bot = _mm512_loadu_ps(re.add(bot + j));
                    let im_bot = _mm512_loadu_ps(im.add(bot + j));

                    let t_re = _mm512_fmsub_ps(w_re, re_bot, _mm512_mul_ps(w_im, im_bot));
                    let t_im = _mm512_fmadd_ps(w_re, im_bot, _mm512_mul_ps(w_im, re_bot));

                    _mm512_storeu_ps(re.add(bot + j), _mm512_sub_ps(re_top, t_re));
                    _mm512_storeu_ps(im.add(bot + j), _mm512_sub_ps(im_top, t_im));
                    _mm512_storeu_ps(re.add(top + j), _mm512_add_ps(re_top, t_re));
                    _mm512_storeu_ps(im.add(top + j), _mm512_add_ps(im_top, t_im));
                }
                j += 16;
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

        #[inline(always)]
        // SAFETY: data, scale, offset are valid f32 slices; n_ch * num_frames == data.len();
        // CPU supports AVX-512F (verified by dispatch). Kernel uses unaligned 512-bit loads/stores.
        unsafe fn batch_norm_process(
            data: &mut [f32],
            scale: &[f32],
            offset: &[f32],
            n_ch: usize,
            num_frames: usize,
        ) {
            for f in 0..num_frames {
                let frame_start = f * n_ch;
                let mut c = 0;
                while c + 16 <= n_ch {
                    // SAFETY: c is bounds-checked (c+16 <= n_ch); frame_start+c is within data
                    // bounds (n_ch * num_frames). Unaligned 512-bit loads/stores valid for f32.
                    unsafe {
                        let x = _mm512_loadu_ps(data.as_ptr().add(frame_start + c));
                        let s = _mm512_loadu_ps(scale.as_ptr().add(c));
                        let o = _mm512_loadu_ps(offset.as_ptr().add(c));
                        let y = _mm512_fmadd_ps(x, s, o);
                        _mm512_storeu_ps(data.as_mut_ptr().add(frame_start + c), y);
                    }
                    c += 16;
                }
                for c in c..n_ch {
                    let idx = frame_start + c;
                    // SAFETY: c < n_ch ensures idx is within data bounds; scale/offset have
                    // at least n_ch elements (caller invariant).
                    unsafe {
                        *data.get_unchecked_mut(idx) = (*data.get_unchecked(idx))
                            .mul_add(*scale.get_unchecked(c), *offset.get_unchecked(c));
                    }
                }
            }
        }
    };
}
