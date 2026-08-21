// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 dispatch macro for DSP kernels (`impl_avx512_dsp!()`).
//!
//! Covers `convolve`, `apply_gain`, `apply_gain_ramp`, `butterfly`/`ibutterfly`
//! (FFT), `batch_norm`, and `crossfade_blend_mono`. Memory-bound kernels (gain,
//! dither, ramp, crossfade, cabsim FFT/MAC) delegate directly to `Avx2Math`
//! to eliminate low-ROI AVX-512 code duplication and instruction-cache overhead.
//! Invoked by [`dispatch_simd!`].

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
        // AVX-512 implies AVX2/x86-64-v3 baseline. Delegates to AVX2 kernel.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            // SAFETY: data and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_and_detect_clipping_mono_avx2(data, gain) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe {
                crate::math::dsp::gain::apply_gain_and_detect_clipping_stereo_avx2(
                    left, right, gain,
                )
            }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            // SAFETY: left, right, and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_stereo_avx2(left, right, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            // SAFETY: data and gain satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_avx2(data, gain) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; start and step are finite f32 values;
        // AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            // SAFETY: data, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_avx2(data, start, step) }
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // start and step are finite f32 values; AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            // SAFETY: left, right, start, and step satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_ramp_stereo_avx2(left, right, start, step) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain and offset are finite f32 values;
        // AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32) {
            // SAFETY: data, gain, and offset satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_gain_then_dither_avx2(data, gain, offset) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; offset is a finite f32;
        // AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            // SAFETY: data and offset satisfy function invariants.
            unsafe { crate::math::dsp::gain::apply_dither_add_avx2(data, offset) }
        }

        #[inline(always)]
        // SAFETY: out and pending are valid f32 slices of equal length; t is a finite f32;
        // AVX-512 implies AVX2. Delegates to AVX2 kernel.
        unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
            // SAFETY: out, pending, and t satisfy function invariants.
            unsafe { crate::math::dsp::gain::crossfade_blend_mono_avx2(out, pending, t) }
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length n; AVX-512 implies AVX2.
        // Delegates to Avx2Math complex MAC routine (Sprint 1.4 low-ROI deduplication).
        unsafe fn complex_mac_overwrite(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            out_re: &mut [f32],
            out_im: &mut [f32],
        ) {
            // SAFETY: argument invariants forwarded to Avx2Math.
            unsafe {
                crate::math::common::Avx2Math::complex_mac_overwrite(
                    h_re, h_im, x_re, x_im, out_re, out_im,
                )
            }
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length n; AVX-512 implies AVX2.
        // Delegates to Avx2Math complex MAC accumulate routine.
        unsafe fn complex_mac_accumulate(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            acc_re: &mut [f32],
            acc_im: &mut [f32],
        ) {
            // SAFETY: argument invariants forwarded to Avx2Math.
            unsafe {
                crate::math::common::Avx2Math::complex_mac_accumulate(
                    h_re, h_im, x_re, x_im, acc_re, acc_im,
                )
            }
        }

        #[inline(always)]
        // SAFETY: re, im, tw_re, tw_im are valid for the described ranges; AVX-512 implies AVX2.
        // Delegates to Avx2Math FFT butterfly routine.
        unsafe fn fft_butterfly_stage(
            re: *mut f32,
            im: *mut f32,
            half: usize,
            tw_re: *const f32,
            tw_im: *const f32,
            group_start: usize,
            inverse: bool,
        ) {
            // SAFETY: pointer invariants forwarded to Avx2Math.
            unsafe {
                crate::math::common::Avx2Math::fft_butterfly_stage(
                    re,
                    im,
                    half,
                    tw_re,
                    tw_im,
                    group_start,
                    inverse,
                )
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
