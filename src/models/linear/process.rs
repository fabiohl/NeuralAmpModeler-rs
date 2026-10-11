// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Hot-path audio processing methods for the Linear model.
//!
//! Separated from the model definition to keep the core struct and
//! constructors in `linear.rs` while isolating the RT-critical
//! process/prewarm/reset logic.

use super::LinearMode;
use crate::math::common::SimdMath;
use crate::models::plane_adapter;

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
    #[cfg(test)]
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
    /// Monomorphizes over `M: SimdMath` and hoists the `self.mode` branch outside
    /// the sample loop, so neither the ISA dispatch nor the convolution mode match
    /// is evaluated per sample.
    ///
    /// # Safety
    /// `self.weights` must be 64-byte aligned.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        let n = core::cmp::min(input.len(), output.len());
        match &mut self.mode {
            LinearMode::Direct => {
                let rf = self.receptive_field;
                let weights_ptr = self.weights.as_ptr();
                if rf < 8 {
                    for i in 0..n {
                        self.history[self.write_pos] = input[i];

                        self.write_pos += 1;
                        if self.write_pos >= self.double_limit {
                            self.write_pos -= self.history.size();
                        }

                        let start = self.write_pos - rf;
                        let window = &self.history[start..self.write_pos];
                        // Sequential scalar loop is bit-exact with `M::convolve_mono`
                        // when rf < 8, because `convolve_mono` initializes zero YMM/ZMM,
                        // skips vector chunks (< 8 taps), reduces 0.0, and executes this
                        // exact sequential loop. Hoisting/avoiding the SIMD function setup
                        // and horizontal zero-reduction saves cycles on small receptive fields.
                        let mut dot = 0.0f32;
                        let win_ptr = window.as_ptr();
                        for k in 0..rf {
                            // SAFETY: `self.weights` has length `receptive_field` (rf), and
                            // `window` has length `rf` from `history` bounds; `k < rf`.
                            dot += unsafe { *weights_ptr.add(k) * *win_ptr.add(k) };
                        }
                        output[i] = self.bias + dot;
                    }
                } else {
                    for i in 0..n {
                        self.history[self.write_pos] = input[i];

                        self.write_pos += 1;
                        if self.write_pos >= self.double_limit {
                            self.write_pos -= self.history.size();
                        }

                        let start = self.write_pos - rf;
                        let window = &self.history[start..self.write_pos];
                        // SAFETY: `window` is `history[start..write_pos]` with
                        // `start = write_pos - receptive_field`, so it has exactly
                        // `receptive_field` elements; `self.weights` also holds
                        // `receptive_field` elements (set in `new` from `weights.len()`) and
                        // is 64-byte aligned (AlignedVec); `M` matches the CPU ISA.
                        let dot = unsafe { M::convolve_mono(weights_ptr, window.as_ptr(), rf) };
                        output[i] = self.bias + dot;
                    }
                }
            }
            LinearMode::Fft(state) => {
                let p = state.p;
                for i in 0..n {
                    self.history[self.write_pos] = input[i];

                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.history.size();
                    }

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
                    // `head_weights_ptr` is valid for `p` elements;
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

                    output[i] = self.bias + head_dot + y_tail;
                }
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

    /// Processes a block of audio samples via raw multichannel pointers.
    ///
    /// For mono models (`in_channels == 1 && out_channels == 1`), delegates to mono `process`
    /// using `*input` and `*output`.
    /// For multichannel models, delegates to `self.multichannel`.
    ///
    /// # Safety
    /// - `input` must point to an array of at least `in_channels` valid non-null `*const f32` pointers,
    ///   each pointing to at least `num_frames` samples.
    /// - `output` must point to an array of at least `out_channels` valid non-null `*mut f32` pointers,
    ///   each pointing to at least `num_frames` writable samples.
    /// - In-place processing (`input == output` or aliasing between input and output channels) is
    ///   fully supported and guaranteed not to corrupt data.
    #[inline(always)]
    pub unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        if let Some(ref mut mc) = self.multichannel {
            // SAFETY: Caller guarantees input and output pointers point to valid channel buffers of length num_frames.
            unsafe { mc.process_raw(input, output, num_frames) };
        } else {
            // SAFETY: Caller guarantees the input/output pointer arrays hold
            // the mono geometry and each plane covers `num_frames`.
            let (in0, out0) = unsafe { (*input, *output) };
            if !std::ptr::eq(in0, out0 as *const f32) {
                // SAFETY: `in0`/`out0` each cover `num_frames` and, in this
                // branch, do not alias, so shared/mutable views are disjoint.
                let (in_slice, out_slice) = unsafe {
                    (
                        core::slice::from_raw_parts(in0, num_frames),
                        core::slice::from_raw_parts_mut(out0, num_frames),
                    )
                };
                // SAFETY: in_slice/out_slice are valid, distinct, length num_frames.
                unsafe { self.process(in_slice, out_slice) };
            } else {
                // Exact in-place alias: stage the input before writing.
                let outputs = [out0];
                // SAFETY: `in0` and the single output plane cover `num_frames`;
                // exact alias is the permitted case.
                unsafe {
                    plane_adapter::run_staged(in0, &outputs, num_frames, 1, |inp, outp| {
                        self.process(inp, outp)
                    });
                }
            }
        }
    }

    /// Fills the history buffer with zeros, resets the write pointer, and
    /// reinitializes the FFT state (if active).
    #[cold]
    pub fn prewarm(&mut self, num_samples: usize) {
        if let Some(ref mut mc) = self.multichannel {
            mc.prewarm(num_samples);
        }
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
        if let Some(ref mut mc) = self.multichannel {
            mc.reset();
        }
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
