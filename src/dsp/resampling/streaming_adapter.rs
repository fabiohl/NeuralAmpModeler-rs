// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Host-agnostic bounded streaming resample buffer with strict cardinality.
//!
//! [`StreamingResampleBuffer`] couples a fractional host sample rate to a
//! model sample rate (e.g. 44.1 kHz ↔ 48 kHz, 96 kHz ↔ 48 kHz) through a
//! caller-supplied model processor, guaranteeing that:
//!
//! - Every [`pull_output`](StreamingResampleBuffer::pull_output) / [`process`](StreamingResampleBuffer::process)
//!   call delivers **exactly** the requested number of host samples
//!   (zero-padding only the formally declared initial latency during warm-up);
//! - Excess output is retained in an internal FIFO for the next call;
//! - No sample is ever dropped, duplicated, or fabricated beyond the declared
//!   latency — strict conservation holds:
//!   `total real output == total input consumed - declared_latency`;
//! - The API is RT-safe: zero heap allocation on the continuous processing path
//!   (all storage is pre-allocated at construction).
//!
//! The abstraction is generic, pure and host-agnostic (Apache-2.0): it does not
//! reference any CLAP/DAW concept. The neural model is injected as a plain
//! `FnMut(&[f32], &[f32], &mut [f32], &mut [f32]) -> usize` processor at model
//! rate (mono or stereo), so any consumer (CLAP plugin, PipeWire host, offline
//! renderer) can drive it.

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
    let pending = fifo_l.len();
    if pending == 0 {
        return 0;
    }
    let pop = pending.min(scratch_l.cap());
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
    /// Total samples delivered (`== requested`).
    pub written: usize,
    /// Real (non-primed) samples among `written`.
    pub real: usize,
    /// Zero samples among `written` (priming/deficit compensation).
    pub padded: usize,
    /// `true` when real samples were fabricated beyond the declared priming
    /// budget — indicates a mid-stream underflow (contract violation).
    pub underflow: bool,
}

/// Result of a single [`process`](StreamingResampleBuffer::process) call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamingResult {
    /// Host samples consumed (pushed) from the input slices.
    pub consumed: usize,
    /// Host samples delivered to the output slices (`== consumed`).
    pub written: usize,
    /// Real (non-primed) samples among `written`.
    pub real: usize,
    /// Zero samples among `written` (declared initial latency priming).
    pub padded: usize,
    /// `true` when samples were fabricated beyond the declared priming budget.
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
    /// Returns [`NamErrorCode::OutOfMemory`] on allocation failure.
    #[cold]
    pub fn new(host_rate: u32, model_rate: u32, max_block: usize) -> Result<Self, NamErrorCode> {
        Self::build(host_rate, model_rate, max_block, false)
    }

    /// Creates the streaming buffer with the linear-phase filter variant.
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
            NamResampler::new_linear_simple(host_rate, model_rate)
        } else {
            NamResampler::new_simple(host_rate, model_rate)
        }
        .map_err(|_| NamErrorCode::OutOfMemory)?;

        let latency_host = resampler.latency_samples(host_rate);
        let model_cap = Self::max_model_samples(max_block, host_rate, model_rate);
        if model_cap > MAX_MODEL_CAP {
            // Extreme host↔model rate ratio: reject instead of allocating tens
            // of MB per instance. Realistic NAM usage (model ≤ 192 kHz) is far
            // below this bound.
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
    /// Returns the number of samples accepted per channel (all of `in_l.len()`
    /// when within `max_block` of free space). Bounded: never overflows.
    #[inline]
    pub fn push_input(&mut self, in_l: &[f32], in_r: &[f32]) -> usize {
        let n = in_l.len().min(in_r.len());
        let a = self.in_fifo[0].push(&in_l[..n]);
        let b = self.in_fifo[1].push(&in_r[..n]);
        let accepted = a.min(b);
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

    /// Internal: drains pending host input into `self.mid_in` (model rate).
    fn drain_to_mid(&mut self) -> usize {
        let [mid_l, mid_r] = &mut self.mid_in;
        drain_channels(
            &mut self.in_fifo,
            &mut self.in_scratch,
            &mut self.resampler,
            mid_l,
            mid_r,
        )
    }

    /// Queues a model-rate output block and drains the model FIFO through
    /// `resampler_out` into the host-rate output FIFO as far as possible.
    ///
    /// Returns the number of model samples accepted. RT-safe: zero allocations.
    pub fn push_model_output(&mut self, in_l: &[f32], in_r: &[f32]) -> usize {
        let n = in_l.len().min(in_r.len());
        let a = self.model_fifo[0].push(&in_l[..n]);
        let b = self.model_fifo[1].push(&in_r[..n]);
        let accepted = a.min(b);
        self.drain_model_to_out();
        accepted
    }

    /// Internal: drains the model FIFO through `resampler_out` into `out_fifo`.
    fn drain_model_to_out(&mut self) {
        loop {
            let pending = self.model_fifo[0].len();
            if pending == 0 {
                break;
            }
            let pop = pending.min(self.mid_drain[0].cap());
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
                    scratch_l,
                    scratch_r,
                )
            };

            let consumed = progress.samples_read.min(n_in);
            if consumed < n_in {
                self.model_fifo[0].unpop(n_in - consumed);
                self.model_fifo[1].unpop(n_in - consumed);
            }
            let mut written = progress.samples_written;
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
                // Bounded by construction: `out_fifo` capacity absorbs the worst
                // case (max_block production + priming retention + margin).
                debug_assert_eq!(pushed, written);
            }
        }
    }

    /// Pulls **exactly** `n` host-rate samples from the output FIFO, retaining
    /// any excess for the next call.
    ///
    /// During warm-up, the formally declared latency
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
            return PullResult {
                written: 0,
                real: 0,
                padded: 0,
                underflow: false,
            };
        }

        // Priming: consume the declared latency as contiguous leading zeros
        // first, so the delivered waveform aligns at exactly `latency_samples()`
        // (the real resampler outputs are discarded during warm-up).
        let pad_target = if self.priming_remaining > 0 {
            (self.priming_remaining.min(n as u64)) as usize
        } else {
            0
        };
        let pad = pad_target.min(n);
        let real_target = n - pad;

        let avail = self.out_fifo[0].len().min(self.out_fifo[1].len());
        let real = real_target.min(avail);

        // Zero-fill the priming pad first (leading), then pop real samples
        // contiguously after it; any residual deficit is zero-filled last.
        if pad > 0 {
            out_l[..pad].fill(0.0);
            out_r[..pad].fill(0.0);
        }
        self.out_fifo[0].pop_into(&mut out_l[pad..pad + real]);
        self.out_fifo[1].pop_into(&mut out_r[pad..pad + real]);
        if pad + real < n {
            out_l[pad + real..n].fill(0.0);
            out_r[pad + real..n].fill(0.0);
        }

        self.priming_remaining -= pad as u64;
        self.output_real_total += real as u64;
        self.output_padded_total += (n - real) as u64;

        // Any fabrication beyond the nominal priming pad is a mid-stream
        // underflow (contract violation): the pipeline fell short of the real
        // samples requested. Under correct operation this never fires.
        let deficit = real_target.saturating_sub(real);
        if deficit > 0 {
            self.underflow_total += deficit as u64;
        }

        PullResult {
            written: n,
            real,
            padded: n - real,
            underflow: deficit > 0,
        }
    }

    /// One-shot streaming pass: pushes the input block, runs the model on all
    /// pending model-rate samples, and pulls exactly `out` samples.
    ///
    /// # Contract
    /// - `in_l/in_r` and `out_l/out_r` carry host-rate samples; equal lengths
    ///   are expected (the effective length is the minimum of all four).
    /// - `model` is invoked with model-rate slices; it must write at most as
    ///   many output samples as the input length (returns the written count).
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

        // Bypass fast path (host rate == model rate): no resampling and no
        // latency, so the model can be called directly on the caller's buffers
        // with exact cardinality — skipping the FIFO pipeline entirely.
        if self.is_bypass() {
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

        let consumed = self.push_input(&in_l[..n], &in_r[..n]);

        // Drain → model → feed output, until all pending host input is consumed.
        loop {
            let drained = self.drain_to_mid();
            if drained == 0 {
                break;
            }
            let produced = {
                let in_mid_l = &self.mid_in[0][..drained];
                let in_mid_r = &self.mid_in[1][..drained];
                let [model_out_l, model_out_r] = &mut self.model_out;
                model(in_mid_l, in_mid_r, model_out_l, model_out_r)
            };
            let produced = produced.min(self.model_out[0].cap());
            if produced > 0 {
                self.model_fifo[0].push(&self.model_out[0][..produced]);
                self.model_fifo[1].push(&self.model_out[1][..produced]);
                self.drain_model_to_out();
            } else {
                // Pathological model: no forward progress possible. Bail out;
                // the caller must replace/reset the model before continuing.
                break;
            }
        }

        let pull = self.pull_output(&mut out_l[..n], &mut out_r[..n], n);
        StreamingResult {
            consumed,
            written: pull.written,
            real: pull.real,
            padded: pull.padded,
            underflow: pull.underflow,
        }
    }
}

#[cfg(test)]
#[path = "streaming_adapter_test.rs"]
mod streaming_adapter_test;
