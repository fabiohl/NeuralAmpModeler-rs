// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! N -> N multichannel Linear model (N inputs, N outputs, 1 single shared kernel).

use super::mode::MultichannelMode;
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::linear::{FFT_AUTO_THRESHOLD, select_partition_size};
use crate::models::linear_fft::LinearFftState;

/// N -> N multichannel Linear model (N inputs, N outputs, 1 single shared kernel).
pub struct LinearManyToManyShared {
    /// Single shared filter weights of length `receptive_field`, stored in reversed order.
    pub weights: AlignedVec<f32>,
    /// Biases per channel (each replicated from bias[0]).
    pub biases: Vec<f32>,
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
    /// Number of channels ($N$).
    pub channels: usize,
}

impl LinearManyToManyShared {
    /// Constructs a new `LinearManyToManyShared` instance.
    pub fn new(topo: LinearTopology, weights: &[f32], biases: &[f32]) -> std::io::Result<Self> {
        let channels = topo.in_channels;
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
            let mut states = Vec::with_capacity(channels);
            for _ in 0..channels {
                let state = LinearFftState::new(p, &weights[..rf]).map_err(|_| {
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

        let mut w = weights[..rf].to_vec();
        w.reverse();
        let aligned = AlignedVec::from_vec(w).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "AlignedVec allocation failed",
            )
        })?;

        let mut histories = Vec::with_capacity(channels);
        for _ in 0..channels {
            histories.push(MirroredBuffer::<f32>::new(rf.max(1))?);
        }
        let limit = histories[0].size();
        let double_limit = limit.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "Limit overflow")
        })?;

        let mut channel_biases = vec![biases.first().copied().unwrap_or(0.0); channels];
        if biases.len() >= channels {
            channel_biases.copy_from_slice(&biases[..channels]);
        }

        Ok(Self {
            weights: aligned,
            biases: channel_biases,
            histories,
            write_pos: limit,
            receptive_field: rf,
            double_limit,
            mode,
            channels,
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

    /// Monomorphized sample processing for $N \to N$ with shared kernel.
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
        let n = num_frames;
        let rf = self.receptive_field;
        let chs = self.channels;

        match &mut self.mode {
            MultichannelMode::Direct => {
                let weights_ptr = self.weights.as_ptr();
                for i in 0..n {
                    // 1. Read all input channels into history (in-place safe)
                    for ch in 0..chs {
                        // SAFETY: ch < channels and input points to valid array of channel pointers.
                        let in_ptr = unsafe { *input.add(ch) };
                        // SAFETY: in_ptr points to at least num_frames samples and i < num_frames.
                        let s = unsafe { *in_ptr.add(i) };
                        self.histories[ch][self.write_pos] = s;
                    }
                    self.write_pos += 1;
                    if self.write_pos >= self.double_limit {
                        self.write_pos -= self.histories[0].size();
                    }

                    // 2. Convolve each channel with shared kernel
                    let start = self.write_pos - rf;
                    for ch in 0..chs {
                        let win_ptr = self.histories[ch][start..self.write_pos].as_ptr();
                        let dot = if rf < 8 {
                            let mut sum = 0.0f32;
                            for k in 0..rf {
                                // SAFETY: rf <= receptive_field and win_ptr/weights_ptr are valid for rf elements.
                                sum += unsafe { *weights_ptr.add(k) * *win_ptr.add(k) };
                            }
                            sum
                        } else {
                            // SAFETY: weights_ptr and win_ptr point to contiguous aligned buffers of at least rf elements.
                            unsafe { M::convolve_mono(weights_ptr, win_ptr, rf) }
                        };
                        // SAFETY: ch < channels, output array is valid, and i < num_frames.
                        unsafe {
                            *(*output.add(ch)).add(i) = self.biases[ch] + dot;
                        }
                    }
                }
            }
            MultichannelMode::Fft(states) => {
                let p = states[0].p;
                // SAFETY: self.weights has length rf >= p, so rf - p is within bounds.
                let head_weights_ptr = unsafe { self.weights.as_ptr().add(rf - p) };
                for i in 0..n {
                    for ch in 0..chs {
                        // SAFETY: ch < channels and input points to valid array of channel pointers.
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
                    for (ch, state) in states.iter_mut().enumerate().take(chs) {
                        let win_ptr = self.histories[ch][head_start..self.write_pos].as_ptr();
                        // SAFETY: head_weights_ptr and win_ptr point to contiguous memory of at least p elements.
                        let head_dot = unsafe { M::convolve_mono(head_weights_ptr, win_ptr, p) };
                        let y_tail = state.tail_output_buf[state.sample_counter];
                        state.sample_counter += 1;
                        // SAFETY: ch < channels, output array is valid, and i < num_frames.
                        unsafe {
                            *(*output.add(ch)).add(i) = self.biases[ch] + head_dot + y_tail;
                        }
                    }

                    if states[0].sample_counter >= p {
                        let block_start = self.write_pos - 2 * p;
                        for (ch, state) in states.iter_mut().enumerate().take(chs) {
                            let block_window = &self.histories[ch][block_start..self.write_pos];
                            state.process_tail_block(block_window);
                            state.sample_counter = 0;
                        }
                    }
                }
            }
        }
    }

    /// Dispatches SIMD processing for $N \to N$.
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
