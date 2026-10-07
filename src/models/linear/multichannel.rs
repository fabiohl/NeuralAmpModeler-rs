// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Specialized multichannel processing architectures for the Linear FIR/FFT model.
//!
//! Provides dedicated zero-allocation paths for non-mono Linear topologies:
//! - [`LinearOneToMany`]: 1 input channel convolved with $N$ independent kernels producing
//!   $N$ output channels ($1 \to N$).
//! - [`LinearManyToOne`]: $N$ input channels convolved with $N$ independent kernels and
//!   summed in strictly ascending channel index order into 1 output channel ($N \to 1$).
//! - [`LinearManyToManyShared`]: $N$ input channels convolved with a single shared kernel
//!   producing $N$ output channels ($N \to N$).
//!
//! # In-Place Processing Guarantee
//!
//! In digital audio workstations and low-latency hosts, input and output buffer pointers
//! frequently alias (`input == output`). All variants in this module guarantee complete
//! in-place safety without temporary heap allocations:
//!
//! For each sample frame $i \in 0..\text{num\_frames}$:
//! 1. All input samples `*input[ch].add(i)` are read and copied into their corresponding
//!    `MirroredBuffer` ring buffer(s) *prior* to calculating or writing sample $i$ to any
//!    output channel pointer.
//! 2. Future samples ($i+1, \dots$) remain untouched in host buffers, and past history is
//!    accessed solely through the internal `MirroredBuffer` memory mapping.
//! 3. This eliminates read-after-write corruption completely across all channel configurations.
//!
//! # Summation Order for $N \to 1$
//!
//! In $N \to 1$ topologies, channel outputs are accumulated into the single output channel.
//! Due to the non-associativity of floating-point arithmetic, accumulation order directly
//! impacts numerical output and ESR:
//!
//! `acc = bias[0] + (w_0 * x_0) + (w_1 * x_1) + ... + (w_{N-1} * x_{N-1})`
//!
//! The accumulator is initialized with `bias[0]`, then channels are accumulated in strictly
//! ascending order ($ch = 0, 1, \dots, N-1$). This matches C++ NAMCore `linear.cpp:328-342`
//! bit-identically.

use super::{FFT_AUTO_THRESHOLD, select_partition_size};
use crate::dsp::mirror_buf::MirroredBuffer;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::linear_fft::LinearFftState;

/// Multichannel convolution implementation mode.
#[derive(Debug)]
pub enum MultichannelMode {
    /// Time-domain direct convolution (dot product over full receptive field).
    Direct,
    /// Frequency-domain partitioned convolution with per-path `LinearFftState`.
    Fft(Vec<LinearFftState>),
}

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
                            state.process_tail_block(block_window);
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

/// Unified multichannel dispatcher enum for the Linear model.
pub enum LinearMultichannel {
    /// 1 -> N multichannel model.
    OneToMany(LinearOneToMany),
    /// N -> 1 multichannel model.
    ManyToOne(LinearManyToOne),
    /// N -> N multichannel model with shared kernel.
    ManyToManyShared(LinearManyToManyShared),
}

impl LinearMultichannel {
    /// Constructs a specialized multichannel Linear model according to `topo`.
    pub fn new(topo: LinearTopology, weights: &[f32], biases: &[f32]) -> std::io::Result<Self> {
        let in_ch = topo.in_channels;
        let out_ch = topo.out_channels;

        if in_ch == 1 && out_ch > 1 {
            Ok(Self::OneToMany(LinearOneToMany::new(
                topo, weights, biases,
            )?))
        } else if in_ch > 1 && out_ch == 1 {
            Ok(Self::ManyToOne(LinearManyToOne::new(
                topo, weights, biases,
            )?))
        } else if in_ch > 1 && in_ch == out_ch {
            Ok(Self::ManyToManyShared(LinearManyToManyShared::new(
                topo, weights, biases,
            )?))
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Unsupported Linear multichannel geometry",
            ))
        }
    }

    /// Resets internal history and FFT tail state.
    pub fn reset(&mut self) {
        match self {
            Self::OneToMany(m) => m.reset(),
            Self::ManyToOne(m) => m.reset(),
            Self::ManyToManyShared(m) => m.reset(),
        }
    }

    /// Prewarms internal state.
    pub fn prewarm(&mut self, _num_samples: usize) {
        self.reset();
    }

    /// Processes audio samples via raw channel pointers.
    ///
    /// # Safety
    /// `input` and `output` must point to valid arrays of channel pointers with at least
    /// `num_frames` valid samples per channel. In-place processing is fully supported.
    #[inline(always)]
    pub unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        match self {
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::OneToMany(m) => unsafe { m.process_raw(input, output, num_frames) },
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::ManyToOne(m) => unsafe { m.process_raw(input, output, num_frames) },
            // SAFETY: Caller guarantees input and output arrays and buffers are valid for num_frames.
            Self::ManyToManyShared(m) => unsafe { m.process_raw(input, output, num_frames) },
        }
    }
}
