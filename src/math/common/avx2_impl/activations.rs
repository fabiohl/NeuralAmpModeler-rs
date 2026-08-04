// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! AVX2 (x86-64-v3) dispatch macro for activation functions
//! (`accumulate_head`, `tanh_and_accumulate_block`, `gated_activation_and_accumulate_block`,
//!  `prelu_slice`, `silu_slice`, `leaky_hard_tanh_slice`, `hard_swish_slice`,
//!  `softsign_slice`, `relu_slice`, `hard_tanh_slice`, `tanh_slice`, `sigmoid_slice`,
//!  `fast_tanh_slice`, `silu_slice_hf`).
//!
//! Generates thin wrappers that delegate to the AVX2 kernels in
//! [`crate::math::wavenet::accumulate`], [`crate::math::activations`], etc.
//! This macro is invoked by the top-level [`dispatch_simd!`] machinery in
//! [`crate::math::common::mod`] when AVX2+FMA is detected at runtime.

macro_rules! impl_avx2_activations {
    () => {
        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn accumulate_head(dest: &mut [f32], src: &[f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::wavenet::accumulate::accumulate_head_avx2(dest, src) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn tanh_and_accumulate_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::wavenet::accumulate::tanh_and_accumulate_block_avx2(head_input, block)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gated_activation_and_accumulate_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::wavenet::accumulate::gated_activation_and_accumulate_block_avx2(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn tanh_and_overwrite_block(head_input: &mut [f32], block: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::wavenet::accumulate::tanh_and_overwrite_block_avx2(head_input, block)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn tanh_and_accumulate_with_seed(
            head_input: &mut [f32],
            block: &mut [f32],
            seed: &[f32],
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::wavenet::accumulate::tanh_and_accumulate_with_seed_avx2(
                    head_input, block, seed,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn gated_activation_and_overwrite_block(
            head_input: &mut [f32],
            block: &mut [f32],
            ch: usize,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::wavenet::accumulate::gated_activation_and_overwrite_block_avx2(
                    head_input, block, ch,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn tanh_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::tanh_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn sigmoid_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::sigmoid_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn tanh_slice_hf(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::tanh::high_fidelity::tanh_poly_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn sigmoid_slice_hf(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::activations::sigmoid::high_fidelity::sigmoid_poly_slice_avx2(slice)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn relu_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::relu_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn prelu_slice(slice: &mut [f32], slopes: &[f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::prelu_slice_avx2(slice, slopes) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn softsign_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::softsign_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn silu_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::silu_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn silu_slice_hf(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::silu_poly_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn hard_tanh_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::hard_tanh_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn hard_swish_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::hard_swish_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fast_tanh_slice(slice: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::fast_tanh_slice_avx2(slice) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn leaky_hard_tanh_slice(
            slice: &mut [f32],
            min_val: f32,
            max_val: f32,
            min_slope: f32,
            max_slope: f32,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                crate::math::activations::leaky_hard_tanh_slice_avx2(
                    slice, min_val, max_val, min_slope, max_slope,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn activation_tanh_block(buf: &mut [f32]) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { crate::math::activations::tanh_slice_avx2(buf) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn fused_lstm_gates_dyn(
            gates: &mut [f32],
            cell_state: &mut [f32],
            cell_error: &mut [f32],
            hidden_state: &mut [f32],
            hidden_size: usize,
        ) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::lstm::fused_lstm_gates_dyn_avx2(
                    gates,
                    cell_state,
                    cell_error,
                    hidden_state,
                    hidden_size,
                )
            }
        }

        // Convolve Stereo: Applies filtering (equalization) to both channels (left and right) simultaneously.
        //
        // ## SIMD Throughput Optimization:
        // Since the impulse response (FIR filter coefficients) is identical for both channels in standard
        // stereo processing, we load the coefficients once into AVX2 registers and
        // multiply them concurrently by the left (input_l) and right (input_r) audio samples.
        // This doubles the computation efficiency compared to filtering each channel sequentially.
        #[inline(always)]
        // SAFETY: coeffs, input_l, and input_r are valid raw pointers to taps valid f32 elements;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch). Kernel may use aligned loads.
        unsafe fn convolve_stereo(
            coeffs: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: raw pointers satisfy the function's documented invariants.
            unsafe {
                super::super::dsp::stereo::convolve_stereo_avx2(coeffs, input_l, input_r, taps)
            }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1, input_l, and input_r are valid raw pointers to taps valid f32
        // elements; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn convolve_stereo_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input_l: *const f32,
            input_r: *const f32,
            taps: usize,
        ) -> ((f32, f32), (f32, f32)) {
            // SAFETY: raw pointers satisfy the function's documented invariants.
            unsafe {
                super::super::dsp::stereo::convolve_stereo_dual_avx2(
                    coeffs0, coeffs1, input_l, input_r, taps,
                )
            }
        }

        #[inline(always)]
        // SAFETY: coeffs and input are valid raw pointers to taps valid f32 elements;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn convolve_mono(coeffs: *const f32, input: *const f32, taps: usize) -> f32 {
            // SAFETY: raw pointers satisfy the function's documented invariants.
            unsafe { super::super::dsp::stereo::convolve_mono_avx2(coeffs, input, taps) }
        }

        #[inline(always)]
        // SAFETY: coeffs0, coeffs1, and input are valid raw pointers to taps valid f32 elements;
        // CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn convolve_mono_dual(
            coeffs0: *const f32,
            coeffs1: *const f32,
            input: *const f32,
            taps: usize,
        ) -> (f32, f32) {
            // SAFETY: raw pointers satisfy the function's documented invariants.
            unsafe {
                super::super::dsp::stereo::convolve_mono_dual_avx2(coeffs0, coeffs1, input, taps)
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_gain_and_detect_clipping_mono(data: &mut [f32], gain: f32) -> bool {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_gain_and_detect_clipping_mono_avx2(data, gain) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_gain_and_detect_clipping_stereo(
            left: &mut [f32],
            right: &mut [f32],
            gain: f32,
        ) -> bool {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe {
                super::super::dsp::gain::apply_gain_and_detect_clipping_stereo_avx2(
                    left, right, gain,
                )
            }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_gain_stereo(left: &mut [f32], right: &mut [f32], gain: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_gain_stereo_avx2(left, right, gain) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_gain(data: &mut [f32], gain: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_gain_avx2(data, gain) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_ramp(data: &mut [f32], start: f32, step: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_ramp_avx2(data, start, step) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_ramp_stereo(left: &mut [f32], right: &mut [f32], start: f32, step: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_ramp_stereo_avx2(left, right, start, step) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_gain_then_dither(data: &mut [f32], gain: f32, offset: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_gain_then_dither_avx2(data, gain, offset) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn apply_dither_add(data: &mut [f32], offset: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::apply_dither_add_avx2(data, offset) }
        }

        #[inline(always)]
        // SAFETY: slices are valid; CPU supports AVX2+FMA (x86-64-v3, verified by dispatch).
        unsafe fn crossfade_blend_mono(out: &mut [f32], pending: &[f32], t: f32) {
            // SAFETY: arguments satisfy the function's documented invariants.
            unsafe { super::super::dsp::gain::crossfade_blend_mono_avx2(out, pending, t) }
        }
    };
}
pub(crate) use impl_avx2_activations;
