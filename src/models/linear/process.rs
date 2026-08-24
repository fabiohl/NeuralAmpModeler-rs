// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Hot-path audio processing methods for the Linear model.
//!
//! Separated from the model definition to keep the core struct and
//! constructors in `linear.rs` while isolating the RT-critical
//! process/prewarm/reset logic.

use super::LinearMode;
use crate::math::common::SimdMath;

impl super::LinearModel {
    /// Processes a single audio sample using the Linear model.
    ///
    /// 1. Writes the sample into the ring buffer (`history`).
    /// 2. Advances the write pointer in the mirrored area.
    /// 3. Dispatches according to the active `mode`:
    ///    - **Direct**: dot product over the full receptive field + bias.
    ///    - **FFT**: dot product over the head (`P` taps) + bias + tail sample
    ///      from the pre-computed `tail_output_buf`. Every `P` samples, a new
    ///      tail block is computed via `LinearFftState::process_tail_block`.
    ///
    /// The convolution is monomorphized over `M: SimdMath` so the ISA is
    /// resolved once per block (by the `dispatch_simd!` hub in `process`),
    /// eliminating the per-sample atomic load and branch that the previous
    /// `convolve_mono` wrapper incurred.
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned (guaranteed by `AlignedVec`).
    #[inline(always)]
    pub(crate) unsafe fn process_sample<M: SimdMath>(&mut self, input: f32) -> f32 {
        self.history[self.write_pos] = input;

        self.write_pos += 1;
        if self.write_pos >= self.double_limit {
            self.write_pos -= self.history.size();
        }

        match &mut self.mode {
            LinearMode::Direct => {
                let start = self.write_pos - self.receptive_field;
                let window = &self.history[start..self.write_pos];
                // SAFETY: `window` is `history[start..write_pos]` with
                // `start = write_pos - receptive_field`, so it has exactly
                // `receptive_field` elements; `self.weights` also holds
                // `receptive_field` elements (set in `new` from `weights.len()`) and
                // is 64-byte aligned (AlignedVec); `M` matches the CPU ISA.
                let dot = unsafe {
                    M::convolve_mono(self.weights.as_ptr(), window.as_ptr(), self.receptive_field)
                };
                self.bias + dot
            }
            LinearMode::Fft(state) => {
                let p = state.p;

                // SAFETY: `self.weights` holds exactly `receptive_field` elements
                // (set in `new` from `weights.len()`), and
                // `receptive_field - p + p == receptive_field`, so `add(receptive_field - p)`
                // plus the subsequent `p`-element read stays within bounds; `self.weights`
                // is 64-byte aligned (AlignedVec).
                let head_weights_ptr =
                    unsafe { self.weights.as_ptr().add(self.receptive_field - p) };
                let head_start = self.write_pos - p;
                let head_window = &self.history[head_start..self.write_pos];
                // SAFETY: `head_window` is `history[head_start..write_pos]` with
                // `head_start = write_pos - p`, so it has exactly `p` elements;
                // `head_weights_ptr` is valid for `p` elements (see above);
                // `self.weights` is 64-byte aligned; `M` matches the CPU ISA.
                let head_dot =
                    unsafe { M::convolve_mono(head_weights_ptr, head_window.as_ptr(), p) };

                let y_tail = state.tail_output_buf[state.sample_counter];
                state.sample_counter += 1;

                if state.sample_counter >= p {
                    let block_start = self.write_pos - 2 * p;
                    let block_window = &self.history[block_start..self.write_pos];
                    state.process_tail_block(block_window);
                    state.sample_counter = 0;
                }

                self.bias + head_dot + y_tail
            }
        }
    }

    /// Processes a block of audio samples, monomorphized over `M: SimdMath`.
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let n = core::cmp::min(input.len(), output.len());
        for i in 0..n {
            // SAFETY: `process_sample::<M>` is an `unsafe fn`; its documented
            // precondition holds (`self.weights` is 64-byte aligned via AlignedVec),
            // `i < n <= input.len()` and `i < n <= output.len()`, and `M` matches
            // the CPU ISA (top-level `dispatch_simd!`).
            unsafe {
                output[i] = self.process_sample::<M>(input[i]);
            }
        }
    }

    /// Processes a block of audio samples (SIMD dispatch once per block).
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned.
    #[inline(always)]
    pub unsafe fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a
        // matching `#[target_feature]` backend; `input`/`output` are the same valid
        // slices passed to this `unsafe fn`, whose documented precondition
        // (64-byte-aligned weights) holds.
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output);
        }
    }

    /// Fills the history buffer with zeros, resets the write pointer, and
    /// reinitializes the FFT state (if active).
    #[cold]
    pub fn prewarm(&mut self, _num_samples: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
        if let LinearMode::Fft(ref mut state) = self.mode {
            state.reset();
        }
    }

    /// Resets internal state: zeroes the history buffer, write pointer,
    /// and FFT state (if active).
    #[cold]
    pub fn reset(&mut self, _sample_rate: u32, _max_buffer_size: usize) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
        if let LinearMode::Fft(ref mut state) = self.mode {
            state.reset();
        }
    }
}
