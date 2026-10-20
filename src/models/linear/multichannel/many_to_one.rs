// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! N -> 1 multichannel Linear model (N inputs, 1 output channel, N independent kernels).

use super::mode::MultichannelMode;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::linear::{FFT_AUTO_THRESHOLD, select_partition_size};
use crate::models::linear_fft::LinearFftState;

/// N -> 1 multichannel Linear model (N inputs, 1 output channel, N independent kernels).
pub struct LinearManyToOne {
    /// Filter weights per input channel, each of length `receptive_field`, stored in reversed order.
    pub weights: Vec<AlignedVec<f32>>,
    /// Output bias scalar (bias[0]).
    pub bias: f32,
    /// Circular buffers of past input samples for each of the $N$ input channels.
    pub histories: Vec<MirroredBuffer<f32>>,
    /// Synchronized write pointer into all `histories`.
    pub write_pos: usize,
    /// Receptive field (filter length).
    pub receptive_field: usize,
    /// Precalculated limit * 2 to avoid overflow checks.
    double_limit: usize,
    /// Convolution mode (Direct or FFT).
    pub mode: MultichannelMode,
    /// Number of input channels.
    pub in_channels: usize,
}

impl LinearManyToOne {
    /// Constructs a new `LinearManyToOne` instance.
    pub fn new(topo: LinearTopology, weights: &[f32], biases: &[f32]) -> std::io::Result<Self> {
        let in_channels = topo.in_channels;
        let rf = topo.receptive_field;

        let use_fft = match topo.implementation {
            LinearImplementation::Direct => false,
            LinearImplementation::Auto => {
                rf >= FFT_AUTO_THRESHOLD && select_partition_size(rf) < rf
            }
            LinearImplementation::Fft => rf >= FFT_AUTO_THRESHOLD,
        };

        let mode = if use_fft {
            let p = select_partition_size(rf);
            let mut states = Vec::with_capacity(in_channels);
            for k in 0..in_channels {
                let kernel_slice = &weights[k * rf..(k + 1) * rf];
                let state = LinearFftState::new(p, kernel_slice).map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "Linear FFT state initialization failed",
                    )
                })?;
                states.push(state);
            }
            MultichannelMode::Fft(states)
        } else {
            MultichannelMode::Direct
        };

        let mut aligned_weights = Vec::with_capacity(in_channels);
        for k in 0..in_channels {
            let mut w = weights[k * rf..(k + 1) * rf].to_vec();
            w.reverse();
            let aligned = AlignedVec::from_vec(w).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "AlignedVec allocation failed",
                )
            })?;
            aligned_weights.push(aligned);
        }

        let mut histories = Vec::with_capacity(in_channels);
        for _ in 0..in_channels {
            histories.push(MirroredBuffer::<f32>::new(rf.max(1))?);
        }
        let limit = histories[0].size();
        let double_limit = limit.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Limit overflow")
        })?;

        Ok(Self {
            weights: aligned_weights,
            bias: biases.first().copied().unwrap_or(0.0),
            histories,
            write_pos: limit,
            receptive_field: rf,
            double_limit,
            mode,
            in_channels,
        })
    }

    /// Resets internal history and FFT tail state.
    pub fn reset(&mut self) {
        let size = self.histories[0].size();
        for h in &mut self.histories {
            for i in 0..(size * 2) {
                h[i] = 0.0;
            }
        }
        self.write_pos = size;
        if let MultichannelMode::Fft(ref mut states) = self.mode {
            for state in states {
                state.reset();
            }
        }
    }

    /// Monomorphized sample processing for $N \to 1$.
    ///
    /// Accumulates starting with `bias`, then sequentially sums channels $0, 1, \dots, N-1$.
    ///
    /// # Safety
    /// `input` and `output` must point to valid arrays of channel pointers.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        if num_frames == 0 {
            return;
        }
        // SAFETY: Caller guarantees output contains at least 1 valid channel buffer.
        let out_ch0 = unsafe { *output };
        let n = num_frames;
        let rf = self.receptive_field;
        let in_ch = self.in_channels;

        match &mut self.mode {
            MultichannelMode::Direct => {
                for i in 0..n {
                    // 1. Read all input channels into history ring buffers (in-place safe)
                    for ch in 0..in_ch {
                        // SAFETY: ch < in_channels and input points to valid array of channel pointers.
                        let in_ptr = unsafe { *input.add(ch) };
                        // SAFETY: in_ptr points to at least num_frames valid samples.
                        let s = unsafe { *in_ptr.add(i) };
                        self.histories[ch][self.write_pos] = s;
                    }
                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.histories[0].size();
                    }

                    // 2. Accumulate in strictly defined order: bias, then ch 0, 1, ..., N-1
                    let mut acc = self.bias;
                    let start = self.write_pos - rf;
                    for ch in 0..in_ch {
                        let win_ptr = self.histories[ch][start..self.write_pos].as_ptr();
                        let w_ptr = self.weights[ch].as_ptr();
                        let dot = if rf < 8 {
                            let mut sum = 0.0f32;
                            for k in 0..rf {
                                // SAFETY: rf <= receptive_field and win_ptr/w_ptr are valid for rf elements.
                                sum += unsafe { *w_ptr.add(k) * *win_ptr.add(k) };
                            }
                            sum
                        } else {
                            // SAFETY: w_ptr and win_ptr point to contiguous aligned buffers of at least rf elements.
                            unsafe { M::convolve_mono(w_ptr, win_ptr, rf) }
                        };
                        acc += dot;
                    }

                    // 3. Write to output channel 0
                    // SAFETY: out_ch0 points to at least num_frames samples and i < num_frames.
                    unsafe {
                        *out_ch0.add(i) = acc;
                    }
                }
            }
            MultichannelMode::Fft(states) => {
                let p = states[0].p;
                for i in 0..n {
                    for ch in 0..in_ch {
                        // SAFETY: ch < in_channels and input points to valid array of channel pointers.
                        let in_ptr = unsafe { *input.add(ch) };
                        // SAFETY: in_ptr points to at least num_frames samples and i < num_frames.
                        let s = unsafe { *in_ptr.add(i) };
                        self.histories[ch][self.write_pos] = s;
                    }
                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.histories[0].size();
                    }

                    let head_start = self.write_pos - p;
                    let mut acc = self.bias;
                    for (ch, state) in states.iter_mut().enumerate().take(in_ch) {
                        let win_ptr = self.histories[ch][head_start..self.write_pos].as_ptr();
                        // SAFETY: self.weights[ch] has length rf >= p, so rf - p is within bounds.
                        let head_weights_ptr = unsafe { self.weights[ch].as_ptr().add(rf - p) };
                        // SAFETY: head_weights_ptr and win_ptr point to contiguous memory of at least p elements.
                        let head_dot = unsafe { M::convolve_mono(head_weights_ptr, win_ptr, p) };
                        let y_tail = state.tail_output_buf[state.sample_counter];
                        state.sample_counter += 1;
                        acc += head_dot + y_tail;
                    }

                    // SAFETY: out_ch0 points to at least num_frames samples and i < num_frames.
                    unsafe {
                        *out_ch0.add(i) = acc;
                    }

                    if states[0].sample_counter >= p {
                        let block_start = self.write_pos - 2 * p;
                        for (ch, state) in states.iter_mut().enumerate().take(in_ch) {
                            let block_window = &self.histories[ch][block_start..self.write_pos];
                            state.process_tail_block(block_window);
                            state.sample_counter = 0;
                        }
                    }
                }
            }
        }
    }

    /// Dispatches SIMD processing for $N \to 1$.
    ///
    /// # Safety
    /// `input` and `output` must point to valid arrays of channel pointers with at least
    /// `num_frames` samples.
    #[inline(always)]
    pub unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        // SAFETY: Caller guarantees input and output valid for num_frames across channels.
        unsafe {
            crate::math::common::dispatch_simd!(self, process_internal, input, output, num_frames);
        }
    }
}
