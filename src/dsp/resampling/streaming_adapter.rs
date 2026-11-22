// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic bounded streaming resample buffer with strict output cardinality.
//!
//! [`StreamingResampleBuffer`] couples a fractional host sample rate to a
//! model sample rate (e.g. 44.1 kHz ↔ 48 kHz, 96 kHz ↔ 48 kHz) through a
//! caller-supplied model processor.
//!
//! # Output cardinality
//!
//! - [`pull_output`](StreamingResampleBuffer::pull_output) writes exactly
//!   `min(n, out_l.len(), out_r.len())` host samples per channel. Excess output
//!   is retained in the internal output FIFO for the next call.
//! - [`process`](StreamingResampleBuffer::process) writes exactly
//!   `n = min(in_l.len(), in_r.len(), out_l.len(), out_r.len())` host samples
//!   per channel. Output cardinality is **independent** of how much input was
//!   accepted in the same call.
//!
//! `real + padded == written` holds for every [`PullResult`] and
//! [`StreamingResult`] returned by a single call: `real` counts generated
//! samples and `padded` counts zeros (declared priming plus any deficit).
//!
//! # Input acceptance (`process` only)
//!
//! `process` never buffers more host input than the input FIFO can hold, so
//! `consumed <= n`; `consumed == n` is guaranteed only when the caller passes
//! at most `max_block()` samples, the input FIFO is empty, and the model
//! callback keeps making progress. In particular `consumed < n` occurs when the
//! caller exceeds `max_block()` (only the first `max_block()` samples are
//! accepted) or when queued input cannot be drained. `consumed` is never
//! greater than the input length `n`, and no relation between `consumed` and
//! `written` is a contract.
//!
//! # Priming and underflow
//!
//! The first [`latency_samples`](StreamingResampleBuffer::latency_samples)
//! host-rate samples delivered are zeros (the declared initial latency); the
//! resampler startup transient is discarded so the signal is delayed by exactly
//! that latency. Beyond the priming budget, any output deficit is still
//! zero-filled to preserve cardinality, is counted in
//! [`underflow_total`](StreamingResampleBuffer::underflow_total), and flags
//! [`underflow`](StreamingResult::underflow) on the result. Underflow has two
//! independent causes: not enough real host output was available, or the model
//! callback produced fewer model-rate samples than it received.
//!
//! # Backpressure
//!
//! All buffers are pre-allocated and bounded. [`push_input`](StreamingResampleBuffer::push_input)
//! accepts only what fits in the input FIFO; [`push_model_output`](StreamingResampleBuffer::push_model_output)
//! accepts what fits in the model FIFO and drains as much as the output FIFO
//! allows. The remainder stays queued in order and is resumed by the next
//! granular call or by [`pull_output`](StreamingResampleBuffer::pull_output) /
//! [`process`](StreamingResampleBuffer::process). Accepted samples are never
//! dropped or duplicated.
//!
//! # Callback semantics
//!
//! The model is injected as a plain
//! `FnMut(&[f32], &[f32], &mut [f32], &mut [f32]) -> usize` processor at model
//! rate (mono or stereo). It receives equal-length input and output slices and
//! returns the number of output samples it produced. It reports **production,
//! not consumption**: the adapter does not guess which inputs a partially
//! producing callback consumed and never fabricates non-zero model samples. If
//! `produced < drained`, the `produced` samples are kept, the call is flagged as
//! underflow, and a callback returning `0` stops the loop (remaining host input
//! stays queued) instead of spinning.
//!
//! # Granular API vs. one-shot call
//!
//! [`push_input`](StreamingResampleBuffer::push_input),
//! [`drain_model_samples`](StreamingResampleBuffer::drain_model_samples),
//! [`push_model_output`](StreamingResampleBuffer::push_model_output) and
//! [`pull_output`](StreamingResampleBuffer::pull_output) let the caller drive
//! each pipeline stage. Every push returns the number of samples actually
//! accepted: the caller must preserve only the **unaccepted suffix**, present it
//! again in order, and must never re-send input already accepted into the FIFO.
//! [`process`](StreamingResampleBuffer::process) performs the same stages in one
//! call, draining pre-existing FIFO state before the new block.
//!
//! # Conservation
//!
//! While the callback produces one model sample per input sample, every accepted
//! input is eventually drained, and padding does not exceed the declared latency,
//! the totals satisfy
//! `output_real_total() == input_total() - latency_samples()`.
//! The relation does not hold after a contract violation (oversized block,
//! stalled callback or underflow).
//!
//! # Real-time safety
//!
//! The API is RT-safe: zero heap allocation on the continuous processing path
//! (all storage is pre-allocated at construction). The abstraction is generic,
//! pure and host-agnostic (Apache-2.0): it does not reference any plugin-host
//! concept, so any consumer (plugin host, standalone audio host, offline
//! renderer) can drive it.

use log::error;

use crate::common::diagnostics::NamErrorCode;
use crate::dsp::resampler::NamResampler;
use crate::math::common::AlignedVec;

use super::fifo::SampleFifo;

/// Drains the stereo input FIFOs through the host→model resampler into
/// `out_l/out_r`. Returns model-rate samples written.
///
/// Free function over disjoint fields so both the public (external output
/// slices) and internal (`mid_in`) call sites share one implementation without
/// self-borrow conflicts.
fn drain_channels(
    in_fifo: &mut [SampleFifo; 2],
    in_scratch: &mut [AlignedVec<f32>; 2],
    resampler: &mut NamResampler,
    out_l: &mut [f32],
    out_r: &mut [f32],
) -> usize {
    let [fifo_l, fifo_r] = in_fifo;
    let [scratch_l, scratch_r] = in_scratch;

    if out_l.is_empty() || out_r.is_empty() {
        return 0;
    }
    let pending = fifo_l.len().min(fifo_r.len());
    if pending == 0 {
        return 0;
    }
    let pop = pending.min(scratch_l.cap()).min(scratch_r.cap());
    let popped = fifo_l.pop_into(&mut scratch_l[..pop]);
    let popped_r = fifo_r.pop_into(&mut scratch_r[..pop]);
    let n_in = popped.min(popped_r);

    let progress = resampler.process_input(&scratch_l[..n_in], &scratch_r[..n_in], out_l, out_r);

    let consumed = progress.samples_read.min(n_in);
    if consumed < n_in {
        fifo_l.unpop(n_in - consumed);
        fifo_r.unpop(n_in - consumed);
    }
    progress.samples_written
}

/// Maximum supported host block size per call (samples, per channel).
///
/// The internal FIFO capacities are sized deterministically for the worst case
/// up to this value (blocks from 1 sample to `MAX_STREAM_BLOCK`).
pub const MAX_STREAM_BLOCK: usize = 8192;

/// Upper bound for the model-rate buffer capacity (samples per channel).
///
/// Guards against extreme host↔model rate ratios that would otherwise scale
/// per-instance allocation to tens of MB (e.g. host 4 kHz → model 384 kHz).
/// Covers the supported matrix up to 44.1 kHz → 192 kHz with headroom.
pub const MAX_MODEL_CAP: usize = MAX_STREAM_BLOCK * 8;

/// Result of a single [`pull_output`](StreamingResampleBuffer::pull_output) call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PullResult {
    /// Total samples delivered: `min(requested, out_l.len(), out_r.len())`.
    pub written: usize,
    /// Real (non-primed) samples among `written`.
    pub real: usize,
    /// Zero samples among `written` (priming + any underflow deficit).
    ///
    /// `real + padded == written` always holds.
    pub padded: usize,
    /// `true` when real samples were fabricated beyond the declared priming
    /// budget — indicates a mid-stream underflow (contract violation).
    pub underflow: bool,
}

/// Result of a single [`process`](StreamingResampleBuffer::process) call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingResult {
    /// Host samples actually accepted into the input FIFO from the input
    /// slices. Bounded by the input length `n` and by the free input-FIFO
    /// space, so `consumed <= n`; `consumed == n` holds only when the caller's
    /// block fits in `max_block()`, the input FIFO is empty and the model
    /// callback makes progress. Inspect it to detect a partially consumed block.
    pub consumed: usize,
    /// Host samples delivered to the output slices. Always
    /// `min(in_l.len(), in_r.len(), out_l.len(), out_r.len())`, independent of
    /// `consumed` (hence `written == consumed` is not guaranteed).
    pub written: usize,
    /// Real (non-primed) samples among `written`.
    pub real: usize,
    /// Zero samples among `written` (declared priming and any output deficit).
    ///
    /// `real + padded == written` always holds.
    pub padded: usize,
    /// `true` when samples were fabricated beyond the declared priming budget,
    /// or when the model callback produced fewer samples than it received.
    pub underflow: bool,
}

/// Bounded, pre-allocated streaming adapter coupling host rate ↔ model rate.
///
/// # Pipeline
///
/// ```text
/// push_input ──► in_fifo ──► resampler_in ──► mid_in (model rate)
///                     model_out ──► model_fifo ──► resampler_out ──► out_fifo ──► pull_output
/// ```
///
/// The caller (or [`process`](StreamingResampleBuffer::process)) runs the model
/// between [`drain_model_samples`](StreamingResampleBuffer::drain_model_samples)
/// and [`push_model_output`](StreamingResampleBuffer::push_model_output).
/// All intermediate storage is pre-allocated at construction; the hot path is
/// zero-alloc.
pub struct StreamingResampleBuffer {
    /// Bidirectional resampler (host → model input, model → host output).
    /// `is_bypass()` when host rate == model rate.
    resampler: NamResampler,
    /// Host-rate input FIFOs (L/R) awaiting resampling.
    in_fifo: [SampleFifo; 2],
    /// Host-rate output FIFOs (L/R) ready to be pulled.
    out_fifo: [SampleFifo; 2],
    /// Model-rate FIFOs (L/R) holding model output awaiting resampling to host.
    model_fifo: [SampleFifo; 2],
    /// Scratch: popped host input fed to `resampler_in`.
    in_scratch: [AlignedVec<f32>; 2],
    /// Scratch: `resampler_in` output (model rate) / model input.
    mid_in: [AlignedVec<f32>; 2],
    /// Scratch: model output (model rate) before queuing to `model_fifo`.
    model_out: [AlignedVec<f32>; 2],
    /// Scratch: popped `model_fifo` content fed to `resampler_out`.
    mid_drain: [AlignedVec<f32>; 2],
    /// Scratch: `resampler_out` output (host rate) before queuing to `out_fifo`.
    out_scratch: [AlignedVec<f32>; 2],
    /// Worst-case host block size (samples per channel) used for sizing.
    max_block: usize,
    /// Declared initial latency (host-rate samples) zero-padded during warm-up.
    latency_host: u32,
    /// Remaining declared latency to prime (zero-pad) on future pulls.
    priming_remaining: u64,
    /// Remaining real resampler outputs to discard during warm-up (host-rate
    /// samples). Equals `latency_host` at construction/reset: the filter's
    /// startup transient is discarded so the delivered waveform is delayed by
    /// exactly `latency_host` (not `2 × latency_host`).
    discard_remaining: u64,
    /// Total host samples pushed (per channel).
    input_total: u64,
    /// Total real host samples pulled.
    output_real_total: u64,
    /// Total zero samples pulled (priming + any underflow fabrication).
    output_padded_total: u64,
    /// Total samples fabricated beyond the priming budget (must stay 0).
    underflow_total: u64,
    /// Capability flag for the linear-phase filter variant.
    linear_phase: bool,
}

impl StreamingResampleBuffer {
    /// Computes the maximum model-rate samples producible from `max_block`
    /// host samples at the given rates (deterministic, integer-only).
    #[inline]
    pub fn max_model_samples(max_block: usize, host_rate: u32, model_rate: u32) -> usize {
        let numer = (max_block as u64).saturating_mul(model_rate as u64);
        let denom = host_rate.max(1) as u64;
        let m = numer.div_ceil(denom);
        m.saturating_add(2).min(usize::MAX as u64) as usize
    }

    /// Computes the deterministic host-rate output FIFO capacity for the worst
    /// case: one full `max_block` production plus the retained priming (latency)
    /// plus phase-surplus margin.
    #[inline]
    pub fn output_capacity(max_block: usize, latency_host: u32) -> usize {
        max_block
            .saturating_add(latency_host as usize)
            .saturating_add(4)
    }

    /// Creates the streaming buffer (minimum-phase filter variant).
    ///
    /// All storage is allocated here — the processing path is zero-alloc.
    ///
    /// # Parameters
    /// - `host_rate`: host sample rate (e.g. 44100).
    /// - `model_rate`: model sample rate (e.g. 48000).
    /// - `max_block`: worst-case host block size per channel (1..=8192).
    ///
    /// # Errors
    /// Returns [`NamErrorCode::ResamplerBuildFailed`] if `host_rate` or
    /// `model_rate` is outside `4_000..=384_000` Hz, or
    /// [`NamErrorCode::OutOfMemory`] on allocation failure or when the derived
    /// model-rate capacity exceeds `MAX_MODEL_CAP`.
    #[cold]
    pub fn new(host_rate: u32, model_rate: u32, max_block: usize) -> Result<Self, NamErrorCode> {
        Self::build(host_rate, model_rate, max_block, false)
    }

    /// Creates the streaming buffer with the linear-phase filter variant.
    ///
    /// # Errors
    /// Same as [`new`](Self::new).
    #[cold]
    pub fn new_linear(
        host_rate: u32,
        model_rate: u32,
        max_block: usize,
    ) -> Result<Self, NamErrorCode> {
        Self::build(host_rate, model_rate, max_block, true)
    }

    #[cold]
    fn build(
        host_rate: u32,
        model_rate: u32,
        max_block: usize,
        linear_phase: bool,
    ) -> Result<Self, NamErrorCode> {
        let max_block = max_block.clamp(1, MAX_STREAM_BLOCK);
        let resampler = if linear_phase {
            NamResampler::new_linear_simple_typed(host_rate, model_rate)
        } else {
            NamResampler::new_simple_typed(host_rate, model_rate)
        }
        .inspect_err(|&code| {
            error!(
                "[StreamingResampleBuffer] Failed to build resampler (host_rate={}, model_rate={}, linear={}): {} ({:?})",
                host_rate,
                model_rate,
                linear_phase,
                code.code(),
                code
            );
        })?;

        let latency_host = resampler.latency_samples(host_rate);
        let model_cap = Self::max_model_samples(max_block, host_rate, model_rate);
        if model_cap > MAX_MODEL_CAP {
            // Extreme host↔model rate ratio: reject instead of allocating tens
            // of MB per instance. Realistic NAM usage (model ≤ 192 kHz) is far
            // below this bound.
            error!(
                "[StreamingResampleBuffer] Model FIFO capacity {} exceeds MAX_MODEL_CAP {} (host_rate={}, model_rate={}, max_block={}): {} ({:?})",
                model_cap,
                MAX_MODEL_CAP,
                host_rate,
                model_rate,
                max_block,
                NamErrorCode::OutOfMemory.code(),
                NamErrorCode::OutOfMemory
            );
            return Err(NamErrorCode::OutOfMemory);
        }
        let out_cap = Self::output_capacity(max_block, latency_host);

        let pair = |cap: usize| -> Result<[SampleFifo; 2], NamErrorCode> {
            Ok([SampleFifo::new(cap)?, SampleFifo::new(cap)?])
        };
        let vec_pair = |cap: usize| -> Result<[AlignedVec<f32>; 2], NamErrorCode> {
            Ok([AlignedVec::new(cap, 0.0f32)?, AlignedVec::new(cap, 0.0f32)?])
        };

        Ok(Self {
            resampler,
            in_fifo: pair(max_block)?,
            out_fifo: pair(out_cap)?,
            model_fifo: pair(model_cap)?,
            in_scratch: vec_pair(max_block)?,
            mid_in: vec_pair(model_cap)?,
            model_out: vec_pair(model_cap)?,
            mid_drain: vec_pair(model_cap)?,
            out_scratch: vec_pair(out_cap)?,
            max_block,
            latency_host,
            priming_remaining: latency_host as u64,
            discard_remaining: latency_host as u64,
            input_total: 0,
            output_real_total: 0,
            output_padded_total: 0,
            underflow_total: 0,
            linear_phase,
        })
    }

    /// `true` when `host_rate == model_rate` (resampler bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.resampler.is_bypass()
    }

    /// Host sample rate.
    #[inline]
    pub fn host_rate(&self) -> u32 {
        self.resampler.host_rate()
    }

    /// Model sample rate.
    #[inline]
    pub fn model_rate(&self) -> u32 {
        self.resampler.nam_rate()
    }

    /// Worst-case host block size configured at construction.
    #[inline]
    pub fn max_block(&self) -> usize {
        self.max_block
    }

    /// Declared initial latency in host-rate samples.
    ///
    /// This is the nominal group delay of the filter chain (input + output
    /// stages) and the exact number of zero samples primed during warm-up.
    #[inline]
    pub fn latency_samples(&self) -> u32 {
        self.latency_host
    }

    /// `true` when the linear-phase filter variant was selected.
    #[inline]
    pub fn is_linear_phase(&self) -> bool {
        self.linear_phase
    }

    /// Deterministic capacity of the host-rate input FIFO (samples).
    #[inline]
    pub fn input_capacity(&self) -> usize {
        self.in_fifo[0].capacity()
    }

    /// Deterministic capacity of the host-rate output FIFO (samples).
    #[inline]
    pub fn output_capacity_actual(&self) -> usize {
        self.out_fifo[0].capacity()
    }

    /// Deterministic capacity of the model-rate FIFO (samples).
    #[inline]
    pub fn model_capacity(&self) -> usize {
        self.model_fifo[0].capacity()
    }

    /// Samples currently stored in the host-rate input FIFO (per channel).
    #[inline]
    pub fn input_pending(&self) -> usize {
        self.in_fifo[0].len()
    }

    /// Samples currently stored in the host-rate output FIFO (per channel).
    #[inline]
    pub fn output_pending(&self) -> usize {
        self.out_fifo[0].len()
    }

    /// Samples currently stored in the model-rate FIFO (per channel).
    #[inline]
    pub fn model_pending(&self) -> usize {
        self.model_fifo[0].len()
    }

    /// Total host samples pushed since construction/last reset (per channel).
    #[inline]
    pub fn input_total(&self) -> u64 {
        self.input_total
    }

    /// Total real (non-primed) host samples pulled since construction/reset.
    #[inline]
    pub fn output_real_total(&self) -> u64 {
        self.output_real_total
    }

    /// Total zero samples pulled since construction/reset.
    #[inline]
    pub fn output_padded_total(&self) -> u64 {
        self.output_padded_total
    }

    /// Total samples fabricated beyond the declared priming budget
    /// (mid-stream underflow; must remain zero under correct operation).
    #[inline]
    pub fn underflow_total(&self) -> u64 {
        self.underflow_total
    }

    /// Resets the adapter to its post-construction state: clears all FIFOs,
    /// resets resampler phase/delay-line state, and re-arms the priming budget.
    ///
    /// RT-safe: zero allocations.
    #[inline]
    pub fn reset(&mut self) {
        for i in 0..2 {
            self.in_fifo[i].clear();
            self.out_fifo[i].clear();
            self.model_fifo[i].clear();
        }
        self.resampler.reset();
        self.priming_remaining = self.latency_host as u64;
        self.discard_remaining = self.latency_host as u64;
        self.input_total = 0;
        self.output_real_total = 0;
        self.output_padded_total = 0;
        self.underflow_total = 0;
    }

    /// Pushes a host-rate input block into the input FIFO.
    ///
    /// Returns the number of samples accepted per channel, bounded by the free
    /// input-FIFO space and by `min(in_l.len(), in_r.len())`; both channels stay
    /// synchronized. When the return value is smaller than the block length, the
    /// caller owns the **unaccepted suffix** and must present it again in order;
    /// already accepted samples must not be re-sent.
    #[inline]
    pub fn push_input(&mut self, in_l: &[f32], in_r: &[f32]) -> usize {
        let free = self.in_fifo[0].free().min(self.in_fifo[1].free());
        let n = in_l.len().min(in_r.len()).min(free);
        let a = self.in_fifo[0].push(&in_l[..n]);
        let b = self.in_fifo[1].push(&in_r[..n]);
        let accepted = a.min(b);
        debug_assert_eq!(accepted, n);
        self.input_total += accepted as u64;
        accepted
    }

    /// Drains pending host input through `resampler_in` into `out_l/out_r`
    /// (model rate). Returns the number of model-rate samples produced.
    ///
    /// Unconsumed host input (when `out` capacity is exhausted) stays in the
    /// input FIFO for a subsequent call. RT-safe: zero allocations.
    pub fn drain_model_samples(&mut self, out_l: &mut [f32], out_r: &mut [f32]) -> usize {
        drain_channels(
            &mut self.in_fifo,
            &mut self.in_scratch,
            &mut self.resampler,
            out_l,
            out_r,
        )
    }

    /// Internal: drains pending host input into `self.mid_in[..limit]` (model rate),
    /// bounded by the available free space in `model_fifo` so model outputs
    /// never exceed `model_fifo` capacity.
    fn drain_to_mid_bounded(&mut self, max_model_out: usize) -> usize {
        let [mid_l, mid_r] = &mut self.mid_in;
        let cap = max_model_out.min(mid_l.cap()).min(mid_r.cap());
        if cap == 0 {
            return 0;
        }
        drain_channels(
            &mut self.in_fifo,
            &mut self.in_scratch,
            &mut self.resampler,
            &mut mid_l[..cap],
            &mut mid_r[..cap],
        )
    }

    /// Queues a model-rate output block and drains the model FIFO through
    /// `resampler_out` into the host-rate output FIFO as far as free output
    /// capacity allows.
    ///
    /// Any model samples that cannot yet fit into `out_fifo` remain buffered in
    /// `model_fifo` and will resume draining on subsequent
    /// [`pull_output`](Self::pull_output) or `push_model_output` calls.
    ///
    /// Returns the exact number of model samples per channel accepted into the
    /// adapter (guaranteed never to be dropped or silently overwritten). When
    /// less than `min(in_l.len(), in_r.len())`, retain the unaccepted suffix and
    /// re-present it in order; never re-send accepted samples.
    /// RT-safe: zero allocations.
    pub fn push_model_output(&mut self, in_l: &[f32], in_r: &[f32]) -> usize {
        let n = in_l.len().min(in_r.len());
        let mut pushed_total = 0usize;
        loop {
            if pushed_total < n {
                let free = self.model_fifo[0].free().min(self.model_fifo[1].free());
                let chunk = (n - pushed_total).min(free);
                if chunk > 0 {
                    let a = self.model_fifo[0].push(&in_l[pushed_total..pushed_total + chunk]);
                    let b = self.model_fifo[1].push(&in_r[pushed_total..pushed_total + chunk]);
                    let pushed = a.min(b);
                    debug_assert_eq!(pushed, chunk);
                    pushed_total += pushed;
                }
            }
            self.drain_model_to_out();
            if pushed_total >= n {
                break;
            }
            let free_after = self.model_fifo[0].free().min(self.model_fifo[1].free());
            if free_after == 0 {
                break;
            }
        }
        pushed_total
    }

    /// Internal: drains the model FIFO through `resampler_out` into `out_fifo`,
    /// strictly bounded by `out_fifo` free capacity plus remaining warm-up
    /// discard budget. Unconsumed model samples stay in `model_fifo` without
    /// re-processing already consumed samples.
    fn drain_model_to_out(&mut self) {
        let scratch_cap = self.out_scratch[0].cap().min(self.out_scratch[1].cap());
        let mid_cap = self.mid_drain[0].cap().min(self.mid_drain[1].cap());
        loop {
            let pending = self.model_fifo[0].len().min(self.model_fifo[1].len());
            if pending == 0 {
                break;
            }
            let out_free = self.out_fifo[0].free().min(self.out_fifo[1].free());
            // Warm-up samples up to `discard_remaining` are dropped before
            // reaching `out_fifo`, so only outputs beyond that budget need
            // free slots in `out_fifo`.
            let max_out = (out_free as u64)
                .saturating_add(self.discard_remaining)
                .min(scratch_cap as u64) as usize;
            if max_out == 0 {
                break;
            }

            let pop = pending.min(mid_cap);
            let popped = self.model_fifo[0].pop_into(&mut self.mid_drain[0][..pop]);
            let popped_r = self.model_fifo[1].pop_into(&mut self.mid_drain[1][..pop]);
            let n_in = popped.min(popped_r);
            if n_in == 0 {
                break;
            }

            let progress = {
                let [scratch_l, scratch_r] = &mut self.out_scratch;
                self.resampler.process_output(
                    &self.mid_drain[0][..n_in],
                    &self.mid_drain[1][..n_in],
                    &mut scratch_l[..max_out],
                    &mut scratch_r[..max_out],
                )
            };

            let consumed = progress.samples_read.min(n_in);
            if consumed < n_in {
                self.model_fifo[0].unpop(n_in - consumed);
                self.model_fifo[1].unpop(n_in - consumed);
            }
            let mut written = progress.samples_written.min(max_out);
            let mut skip = 0usize;
            if self.discard_remaining > 0 {
                // Warm-up: discard the first `latency_host` real resampler
                // outputs (the filter startup transient) so the delivered
                // waveform aligns at exactly `latency_host` instead of 2× it.
                let d = (self.discard_remaining.min(written as u64)) as usize;
                self.discard_remaining -= d as u64;
                skip = d;
                written = written.saturating_sub(d);
            }
            if written > 0 {
                let a = self.out_fifo[0].push(&self.out_scratch[0][skip..skip + written]);
                let b = self.out_fifo[1].push(&self.out_scratch[1][skip..skip + written]);
                let pushed = a.min(b);
                // Guaranteed by construction: `written <= max_out - d <= out_free`.
                debug_assert_eq!(pushed, written);
            }
            if consumed == 0 && progress.samples_written == 0 {
                break;
            }
        }
    }

    /// Consumes leading priming zeros (up to `n - *pos`) into `out_l`/`out_r`,
    /// updating `*pos` and `*pad_used`.
    #[inline]
    fn fill_priming_zeros(
        &mut self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        pos: &mut usize,
        n: usize,
        pad_used: &mut usize,
    ) {
        if *pos >= n || self.priming_remaining == 0 {
            return;
        }
        let avail = n - *pos;
        let pad = (self.priming_remaining.min(avail as u64)) as usize;
        if pad > 0 {
            out_l[*pos..*pos + pad].fill(0.0);
            out_r[*pos..*pos + pad].fill(0.0);
            self.priming_remaining -= pad as u64;
            *pos += pad;
            *pad_used += pad;
        }
    }

    /// Drains `model_fifo -> out_fifo` and pops available real samples into
    /// `out_l[*pos..n]` / `out_r[*pos..n]`, resuming `drain_model_to_out` after
    /// each pop so backpressured model outputs keep flowing.
    fn pull_available_real(
        &mut self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        pos: &mut usize,
        n: usize,
        real_used: &mut usize,
    ) {
        self.drain_model_to_out();
        while *pos < n {
            let avail = self.out_fifo[0].len().min(self.out_fifo[1].len());
            if avail == 0 {
                break;
            }
            let take = (n - *pos).min(avail);
            let p0 = self.out_fifo[0].pop_into(&mut out_l[*pos..*pos + take]);
            let p1 = self.out_fifo[1].pop_into(&mut out_r[*pos..*pos + take]);
            let popped = p0.min(p1);
            debug_assert_eq!(popped, take);
            *pos += popped;
            *real_used += popped;
            self.drain_model_to_out();
        }
    }

    /// Pulls `min(n, out_l.len(), out_r.len())` host-rate samples, retaining any
    /// excess for the next call and resuming any backpressured model output.
    ///
    /// The returned [`PullResult::written`] is that clamped count and always
    /// satisfies `real + padded == written`. During warm-up, the formally
    /// declared latency
    /// ([`latency_samples`](StreamingResampleBuffer::latency_samples)) is
    /// zero-primed deterministically; afterwards every delivered sample is real.
    /// If real samples fall short beyond the priming budget (mid-stream
    /// underflow), the deficit is zero-filled to preserve cardinality and
    /// accounted in [`underflow_total`](StreamingResampleBuffer::underflow_total).
    ///
    /// RT-safe: zero allocations.
    pub fn pull_output(&mut self, out_l: &mut [f32], out_r: &mut [f32], n: usize) -> PullResult {
        let n = n.min(out_l.len()).min(out_r.len());
        if n == 0 {
            self.drain_model_to_out();
            return PullResult {
                written: 0,
                real: 0,
                padded: 0,
                underflow: false,
            };
        }

        let mut pos = 0usize;
        let mut pad_used = 0usize;
        let mut real_used = 0usize;

        // Priming: consume the declared latency as contiguous leading zeros
        // first, so the delivered waveform aligns at exactly `latency_samples()`
        // (the real resampler outputs are discarded during warm-up).
        self.fill_priming_zeros(out_l, out_r, &mut pos, n, &mut pad_used);
        self.pull_available_real(out_l, out_r, &mut pos, n, &mut real_used);

        let deficit = n - pos;
        if deficit > 0 {
            out_l[pos..n].fill(0.0);
            out_r[pos..n].fill(0.0);
            self.underflow_total += deficit as u64;
        }

        self.output_real_total += real_used as u64;
        self.output_padded_total += (pad_used + deficit) as u64;

        PullResult {
            written: n,
            real: real_used,
            padded: pad_used + deficit,
            underflow: deficit > 0,
        }
    }

    /// Internal: drains pending host input in `in_fifo` through `model`,
    /// bounded by `model_fifo` free space, pushing produced model outputs to
    /// `model_fifo`, draining to `out_fifo`, and pulling available real samples
    /// into `out_l[*pos..n]` / `out_r[*pos..n]` when space is still needed.
    ///
    /// Returns `(model_underflow, model_stalled)`.
    fn drain_input_through_model<F>(
        &mut self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        pos: &mut usize,
        n: usize,
        real_used: &mut usize,
        model: &mut F,
    ) -> (bool, bool)
    where
        F: FnMut(&[f32], &[f32], &mut [f32], &mut [f32]) -> usize,
    {
        let mut model_underflow = false;
        let mut model_stalled = false;
        loop {
            self.drain_model_to_out();
            if *pos < n {
                self.pull_available_real(out_l, out_r, pos, n, real_used);
            }
            let model_free = self.model_fifo[0].free().min(self.model_fifo[1].free());
            if model_free == 0 {
                break;
            }
            let drained = self.drain_to_mid_bounded(model_free);
            if drained == 0 {
                break;
            }
            let produced = {
                let in_mid_l = &self.mid_in[0][..drained];
                let in_mid_r = &self.mid_in[1][..drained];
                let [model_out_l, model_out_r] = &mut self.model_out;
                model(
                    in_mid_l,
                    in_mid_r,
                    &mut model_out_l[..drained],
                    &mut model_out_r[..drained],
                )
            };
            let produced = produced.min(drained);
            if produced < drained {
                model_underflow = true;
            }
            if produced > 0 {
                let a = self.model_fifo[0].push(&self.model_out[0][..produced]);
                let b = self.model_fifo[1].push(&self.model_out[1][..produced]);
                let pushed = a.min(b);
                debug_assert_eq!(pushed, produced);
                self.drain_model_to_out();
                if *pos < n {
                    self.pull_available_real(out_l, out_r, pos, n, real_used);
                }
            } else {
                // Pathological model: zero output produced for `drained`
                // inputs. Bail out without fabricating fake samples or guessing
                // unconsumed input counts; any remaining `in_fifo` stays queued.
                model_stalled = true;
                break;
            }
        }
        (model_underflow, model_stalled)
    }

    /// One-shot streaming pass: pushes the input block, runs the model on all
    /// pending model-rate samples, and pulls exactly `n` samples.
    ///
    /// # Contract
    /// - `in_l/in_r` and `out_l/out_r` carry host-rate samples; the effective
    ///   length is `n = min(in_l.len(), in_r.len(), out_l.len(), out_r.len())`.
    /// - Exactly `n` host samples are written per channel (`written == n`), so
    ///   `real + padded == written`. Input acceptance is separate:
    ///   `consumed <= n` because the block is bounded by the free input-FIFO
    ///   space. `consumed < n` means only the first `consumed` input samples
    ///   were accepted; re-present the suffix (in order) and never re-send the
    ///   accepted prefix. Passing more than `max_block()` samples makes partial
    ///   acceptance and underflow observable.
    /// - `model` is invoked with equal-length model-rate slices (`&mid[..drained]`
    ///   and `&mut model_out[..drained]`) and is expected to write `drained`
    ///   output samples (1:1 synchronous cardinality, returning the written count).
    ///   It reports production, not consumption; unconsumed inputs are never
    ///   inferred.
    /// - **Under-producing callback (`produced < drained`):** the callback
    ///   signature does not report input consumption separately from output
    ///   production, so the adapter never guesses unconsumed inputs and never
    ///   fabricates non-zero model-rate samples. It preserves the `produced`
    ///   real samples written, flags `StreamingResult::underflow = true`
    ///   immediately (stopping the loop on `produced == 0` without spinning),
    ///   and accounts any host-rate zero-fill deficit in `underflow_total`.
    /// - **Bypass + granular API compatibility:** when `is_bypass()` is `true`
    ///   and all internal FIFOs (`in_fifo`, `model_fifo`, `out_fifo`) are empty,
    ///   `process` takes a zero-copy direct fast path. If any FIFO holds pending
    ///   samples from prior granular calls (`push_input`, `push_model_output`,
    ///   partial `pull_output`), `process` routes through the FIFO pipeline so
    ///   pending samples are never reordered or abandoned.
    ///
    /// RT-safe: zero allocations on the processing path.
    pub fn process<F>(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
        mut model: F,
    ) -> StreamingResult
    where
        F: FnMut(&[f32], &[f32], &mut [f32], &mut [f32]) -> usize,
    {
        let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());

        // Bypass fast path (host rate == model rate) when no granular FIFO
        // state is pending: zero-copy, zero latency, exact cardinality.
        let fifos_empty =
            self.input_pending() == 0 && self.model_pending() == 0 && self.output_pending() == 0;
        if self.is_bypass() && fifos_empty {
            let produced = model(&in_l[..n], &in_r[..n], &mut out_l[..n], &mut out_r[..n]);
            let produced = produced.min(n);
            self.input_total += n as u64;
            self.output_real_total += produced as u64;
            let padded = n - produced;
            self.output_padded_total += padded as u64;
            if produced < n {
                out_l[produced..n].fill(0.0);
                out_r[produced..n].fill(0.0);
                self.underflow_total += padded as u64;
            }
            return StreamingResult {
                consumed: n,
                written: n,
                real: produced,
                padded,
                underflow: produced < n,
            };
        }

        let mut pos = 0usize;
        let mut pad_used = 0usize;
        let mut real_used = 0usize;
        let mut model_underflow = false;
        let mut model_stalled = false;

        // 1. Honor any remaining declared priming first, then pull pre-existing
        //    output samples so downstream FIFOs drain before new input arrives.
        self.fill_priming_zeros(out_l, out_r, &mut pos, n, &mut pad_used);
        self.pull_available_real(out_l, out_r, &mut pos, n, &mut real_used);

        // 2. Drain any pre-existing host input queued via `push_input` before
        //    pushing `in_l/in_r`, preserving strict chronological order and
        //    freeing `in_fifo` capacity.
        if self.input_pending() > 0 {
            let (uf, stalled) = self.drain_input_through_model(
                out_l,
                out_r,
                &mut pos,
                n,
                &mut real_used,
                &mut model,
            );
            model_underflow |= uf;
            model_stalled = stalled;
        }

        // 3. Push the new host-rate block (bounded by `in_fifo` free space).
        let consumed = self.push_input(&in_l[..n], &in_r[..n]);

        // 4. Drain → model → model_fifo → resampler_out → out_fifo → pull.
        if !model_stalled {
            let (uf, _) = self.drain_input_through_model(
                out_l,
                out_r,
                &mut pos,
                n,
                &mut real_used,
                &mut model,
            );
            model_underflow |= uf;
        }

        // 5. Complete `out_l/out_r[..n]` cardinality (pull any final real
        //    samples; zero-fill any residual deficit as underflow).
        self.pull_available_real(out_l, out_r, &mut pos, n, &mut real_used);
        let deficit = n - pos;
        if deficit > 0 {
            out_l[pos..n].fill(0.0);
            out_r[pos..n].fill(0.0);
            self.underflow_total += deficit as u64;
        }

        self.output_real_total += real_used as u64;
        self.output_padded_total += (pad_used + deficit) as u64;

        StreamingResult {
            consumed,
            written: n,
            real: real_used,
            padded: pad_used + deficit,
            underflow: deficit > 0 || model_underflow,
        }
    }
}

#[cfg(test)]
#[path = "streaming_adapter_test.rs"]
mod streaming_adapter_test;
