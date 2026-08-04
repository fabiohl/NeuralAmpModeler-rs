// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX-512 VNNI+BF16 dispatch macro for DSP kernels
//! (`impl_avx512vnni_bf16_dsp!()`).
//!
//! Currently delegates to the standard `Avx512Math` DSP implementations.
//! VNNI-BF16 acceleration for DSP operators is planned for a future release.
//! Invoked by [`dispatch_simd!`] when `IS_BF16=true`.

macro_rules! impl_avx512vnni_bf16_dsp {
    () => {
        #[inline(always)]
        // SAFETY: coeffs (64-byte-aligned), input_l, input_r are raw pointers to taps valid
        // f32 elements; CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn convolve_stereo(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: all pointers satisfy invariants per the base Avx512Math kernel.
            unsafe { Avx512Math::convolve_stereo(coeffs, input_l, input_r, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 (64-byte-aligned), input_l, input_r valid pointers;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn convolve_stereo_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            // SAFETY: all pointers satisfy invariants per the base kernel.
            unsafe { Avx512Math::convolve_stereo_dual(coeffs0, coeffs1, input_l, input_r, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs (64-byte-aligned) and input are valid pointers to taps valid f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: pointers satisfy invariants per the base kernel.
            unsafe { Avx512Math::convolve_mono(coeffs, input, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1 (64-byte-aligned), input valid pointers;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn convolve_mono_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: pointers satisfy invariants per the base kernel.
            unsafe { Avx512Math::convolve_mono_dual(coeffs0, coeffs1, input, taps) }
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            Avx512Math::apply_gain_and_detect_clipping_mono(data, gain)
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            Avx512Math::apply_gain_and_detect_clipping_stereo(left, right, gain)
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // gain is a finite f32; CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            Avx512Math::apply_gain_stereo(left, right, gain)
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain is a finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            crate::math::dsp::gain::apply_gain_avx512(data, gain)
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; start and step are finite f32 values;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            Avx512Math::apply_ramp(data, start, step)
        }

        #[inline(always)]
        // SAFETY: left and right are valid mutable f32 slices of equal length;
        // start and step are finite f32 values; CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            Avx512Math::apply_ramp_stereo(left, right, start, step)
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; gain and offset are finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32) {
            Avx512Math::apply_gain_then_dither(data, gain, offset)
        }

        #[inline(always)]
        // SAFETY: data is a valid mutable f32 slice; offset is a finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            Avx512Math::apply_dither_add(data, offset)
        }

        #[inline(always)]
        // SAFETY: out and pending are valid f32 slices of equal length; t is a finite f32;
        // CPU supports AVX-512 VNNI+BF16.
        unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
            Avx512Math::crossfade_blend_mono(out, pending, t)
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn complex_mac_overwrite(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            out_re: &mut [f32],
            out_im: &mut [f32],
        ) {
            Avx512Math::complex_mac_overwrite(h_re, h_im, x_re, x_im, out_re, out_im)
        }

        #[inline(always)]
        // SAFETY: all 6 slices are valid f32 slices of equal length;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn complex_mac_accumulate(
            h_re: &[f32],
            h_im: &[f32],
            x_re: &[f32],
            x_im: &[f32],
            acc_re: &mut [f32],
            acc_im: &mut [f32],
        ) {
            Avx512Math::complex_mac_accumulate(h_re, h_im, x_re, x_im, acc_re, acc_im)
        }

        #[inline(always)]
        // SAFETY: re, im, tw_re, tw_im are valid for the described ranges;
        // CPU supports AVX-512 VNNI+BF16 (verified by dispatch).
        unsafe fn fft_butterfly_stage(
            re: *mut f32,
            im: *mut f32,
            half: usize,
            tw_re: *const f32,
            tw_im: *const f32,
            group_start: usize,
            inverse: bool,
        ) {
            Avx512Math::fft_butterfly_stage(re, im, half, tw_re, tw_im, group_start, inverse)
        }

        #[inline(always)]
        // SAFETY: valid f32 slices; delegates to Avx512Math for f32 batch norm.
        unsafe fn batch_norm_process(
            data: &mut [f32],
            scale: &[f32],
            offset: &[f32],
            n_ch: usize,
            num_frames: usize,
        ) {
            Avx512Math::batch_norm_process(data, scale, offset, n_ch, num_frames)
        }
    };
}
