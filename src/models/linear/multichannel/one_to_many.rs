// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 1 -> N multichannel Linear model (1 input, N output channels, N independent kernels).

use super::mode::MultichannelMode;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::linear::{FFT_AUTO_THRESHOLD, select_partition_size};
use crate::models::linear_fft::LinearFftState;

/// 1 -> N multichannel Linear model (1 input, N output channels, N independent kernels).
pub struct LinearOneToMany {
    /// Filter weights per output channel, each of length `receptive_field`, stored in reversed order.
    pub weights: Vec<AlignedVec<f32>>,
    /// Bias scalar per output channel.
    pub biases: Vec<f32>,
    /// Circular buffer of past input samples for the single input channel.
    pub history: MirroredBuffer<f32>,
    /// Write pointer into `history`.
    pub write_pos: usize,
    /// Receptive field (filter length).
    pub receptive_field: usize,
    /// Precalculated limit * 2 to avoid overflow checks.
    double_limit: usize,
    /// Convolution mode (Direct or FFT).
    pub mode: MultichannelMode,
    /// Number of output channels.
    pub out_channels: usize,
}

impl LinearOneToMany {
    /// Returns whether this variant can safely reach its unchecked kernels.
    pub fn is_sound(&self) -> bool {
        let rf = self.receptive_field;
        if !super::history_is_sound(&self.history, self.write_pos, rf, self.double_limit)
            || self.out_channels == 0
            || self.weights.len() < self.out_channels
            || self.biases.len() < self.out_channels
            || self
                .weights
                .iter()
                .take(self.out_channels)
                .any(|w| w.len() < rf)
        {
            return false;
        }
        match &self.mode {
            MultichannelMode::Direct => true,
            MultichannelMode::Fft(states) => {
                states.len() >= self.out_channels && states.iter().all(|state| state.is_sound(rf))
            }
        }
    }

    /// Constructs a new `LinearOneToMany` instance.
    pub fn new(topo: LinearTopology, weights: &[f32], biases: &[f32]) -> std::io::Result<Self> {
        let out_channels = topo.out_channels;
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
            let mut states = Vec::with_capacity(out_channels);
            for k in 0..out_channels {
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

        let mut aligned_weights = Vec::with_capacity(out_channels);
        for k in 0..out_channels {
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

        let history = MirroredBuffer::<f32>::new(rf.max(1))?;
        let limit = history.size();
        let double_limit = limit.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Limit overflow")
        })?;

        let mut channel_biases = vec![biases.first().copied().unwrap_or(0.0); out_channels];
        if biases.len() >= out_channels {
            channel_biases.copy_from_slice(&biases[..out_channels]);
        }

        Ok(Self {
            weights: aligned_weights,
            biases: channel_biases,
            history,
            write_pos: limit,
            receptive_field: rf,
            double_limit,
            mode,
            out_channels,
        })
    }

    /// Resets internal history and FFT tail state.
    pub fn reset(&mut self) {
        let size = self.history.size();
        for i in 0..(size * 2) {
            self.history[i] = 0.0;
        }
        self.write_pos = size;
        if let MultichannelMode::Fft(ref mut states) = self.mode {
            for state in states {
                state.reset();
            }
        }
    }

    /// Monomorphized sample processing for $1 \to N$.
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
        // SAFETY: Caller guarantees input points to valid array of channel pointers.
        let in_ch0 = unsafe { *input };
        let n = num_frames;
        let rf = self.receptive_field;
        let out_ch = self.out_channels;

        match &mut self.mode {
            MultichannelMode::Direct => {
                for i in 0..n {
                    // 1. Read input sample i and update history (in-place safe)
                    // SAFETY: in_ch0 contains at least num_frames elements and i < num_frames.
                    let s = unsafe { *in_ch0.add(i) };
                    self.history[self.write_pos] = s;
                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.history.size();
                    }

                    let start = self.write_pos - rf;
                    let window = &self.history[start..self.write_pos];
                    let win_ptr = window.as_ptr();

                    // 2. Compute convolution for each output channel
                    for ch in 0..out_ch {
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
                        // SAFETY: ch < out_channels, output array is valid, and i < num_frames.
                        unsafe {
                            *(*output.add(ch)).add(i) = self.biases[ch] + dot;
                        }
                    }
                }
            }
            MultichannelMode::Fft(states) => {
                let p = states[0].p;
                for i in 0..n {
                    // SAFETY: in_ch0 contains at least num_frames elements and i < num_frames.
                    let s = unsafe { *in_ch0.add(i) };
                    self.history[self.write_pos] = s;
                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.history.size();
                    }

                    let head_start = self.write_pos - p;
                    let head_window = &self.history[head_start..self.write_pos];
                    let win_ptr = head_window.as_ptr();

                    for (ch, state) in states.iter_mut().enumerate().take(out_ch) {
                        // SAFETY: self.weights[ch] has length rf >= p, so rf - p is within bounds.
                        let head_weights_ptr = unsafe { self.weights[ch].as_ptr().add(rf - p) };
                        // SAFETY: head_weights_ptr and win_ptr point to contiguous memory of at least p elements.
                        let head_dot = unsafe { M::convolve_mono(head_weights_ptr, win_ptr, p) };
                        let y_tail = state.tail_output_buf[state.sample_counter];
                        state.sample_counter += 1;
                        // SAFETY: ch < out_channels, output array is valid, and i < num_frames.
                        unsafe {
                            *(*output.add(ch)).add(i) = self.biases[ch] + head_dot + y_tail;
                        }
                    }

                    if states[0].sample_counter >= p {
                        let block_start = self.write_pos - 2 * p;
                        let block_window = &self.history[block_start..self.write_pos];
                        for state in states.iter_mut() {
                            // SAFETY: this model validated the whole state at the safe entry point; no
                            // public field can change under the current `&mut self` borrow,
                            // and `block_window` has exactly `2P` samples.
                            unsafe { state.process_tail_block_unchecked(block_window) };
                            state.sample_counter = 0;
                        }
                    }
                }
            }
        }
    }

    /// Dispatches SIMD processing for $1 \to N$.
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
