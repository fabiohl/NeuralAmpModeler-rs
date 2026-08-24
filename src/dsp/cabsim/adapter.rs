// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RT-safe variable-block adapter for
//! [`ConvEngine`](crate::dsp::cabsim::conv::ConvEngine).
//!
//! [`ConvEngine::process()`](crate::dsp::cabsim::conv::ConvEngine::process) requires exactly
//! `partition_size` samples per call.
//! Audio hosts fragment blocks into arbitrary-sized sub-blocks due to
//! sample-accurate parameter automation events. `CabSimAdapter` accumulates
//! sub-blocks until a full partition is available, processes it through
//! the UPOLS engine, and delivers output causally — matching the
//! engine's inherent `partition_size`-sample latency.
//!
//! ## Design
//!
//! *   **FIFO input accumulator** — sub-blocks are buffered until
//!     `partition_size` samples are collected.
//! *   **FIFO output queue** — [`ConvEngine::process()`](crate::dsp::cabsim::conv::ConvEngine::process)
//!     renders partitions directly into the output queue (`2 × partition_size`),
//!     with no intermediate scratch partition; slices are returned on subsequent
//!     [`process_variable()`](crate::dsp::cabsim::adapter::CabSimAdapter::process_variable) calls. The output
//!     buffer is `2 × partition_size` to avoid overwriting unconsumed
//!     output when a second partition completes before the first is fully
//!     drained.
//! *   **Zero-alloc hot path** — all buffers are pre-allocated in
//!     [`new()`](crate::dsp::cabsim::adapter::CabSimAdapter::new).
//! *   **In-place processing** — [`process_in_place()`](crate::dsp::cabsim::adapter::CabSimAdapter::process_in_place)
//!     reads and writes the same buffer, eliminating the destination copy
//!     on the audio callback.
//! *   **Causal output** — the adapter respects the engine's intrinsic
//!     latency. Until the first partition is fully accumulated, output is
//!     silence.

use super::conv::ConvEngine;
use crate::common::diagnostics::NamErrorCode;
use crate::common::spsc::{RT_STATUS_CABSIM_CONTRACT_VIOLATION, RtStatusFlags};
use crate::math::common::AlignedVec;

/// RT-safe adapter that bridges fixed-partition [`ConvEngine`] to
/// variable-size sub-blocks (as produced by host sample-accurate events).
pub struct CabSimAdapter {
    engine: Box<ConvEngine>,
    partition: usize,
    input_buf: AlignedVec<f32>,
    output_buf: AlignedVec<f32>,
    input_count: usize,
    output_read: usize,
    output_write: usize,
}

impl CabSimAdapter {
    /// Creates an adapter wrapping the given convolution engine.
    ///
    /// Pre-allocates all FIFO and scratch buffers to the engine's
    /// `partition_size`. The hot path in [`process_variable`](CabSimAdapter::process_variable) is zero-alloc.
    pub fn new(engine: Box<ConvEngine>) -> Result<Self, NamErrorCode> {
        let partition = engine.partition_size();
        Ok(Self {
            engine,
            partition,
            input_buf: AlignedVec::new(2 * partition, 0.0_f32)?,
            output_buf: AlignedVec::new(2 * partition, 0.0_f32)?,
            input_count: 0,
            output_read: 0,
            output_write: 0,
        })
    }

    /// Returns the engine's partition size in samples.
    #[inline(always)]
    pub fn partition_size(&self) -> usize {
        self.partition
    }

    /// Returns the algorithmic latency in samples (= `partition_size`).
    #[inline(always)]
    pub fn latency_samples(&self) -> usize {
        self.partition
    }

    /// Returns `true` if no IR is loaded (passthrough mode).
    #[inline(always)]
    pub fn is_passthrough(&self) -> bool {
        self.engine.is_passthrough()
    }

    /// Returns the number of IR partitions in the underlying engine.
    #[inline(always)]
    pub fn num_partitions(&self) -> usize {
        self.engine.num_partitions()
    }

    /// Returns a reference to the inner [`ConvEngine`].
    #[inline(always)]
    pub fn engine(&self) -> &ConvEngine {
        &self.engine
    }

    /// Returns a mutable reference to the inner [`ConvEngine`].
    #[inline(always)]
    pub fn engine_mut(&mut self) -> &mut ConvEngine {
        &mut self.engine
    }

    /// Returns `true` if the adapter has unprocessed input in its
    /// accumulator or unconsumed output in its queue.
    #[inline(always)]
    pub fn needs_flush(&self) -> bool {
        self.input_count > 0 || self.output_read < self.output_write
    }

    /// Returns the estimated number of non-silent output samples remaining
    /// before the IR tail is fully drained. Blocks the round number of
    /// partitions needed to flush the FDL plus the accumulated input.
    #[inline(always)]
    pub fn tail_samples(&self) -> usize {
        if self.engine.is_passthrough() {
            return 0;
        }
        self.engine.num_partitions().saturating_mul(self.partition) + self.partition // one extra block for adapter fifo accumulator
    }

    /// Processes a variable-size sub-block through the convolution engine.
    ///
    /// Accumulates samples until a full partition is ready, then runs
    /// [`ConvEngine::process`] and stages the output. Delivers available
    /// output samples into `output`, filling the remainder with silence.
    ///
    /// # Constraints
    ///
    /// *   `input.len() == output.len() <= partition_size`
    /// *   `input.len() > 0` (use empty slice for flush passes)
    /// *   `input` and `output` must not overlap — use
    ///     [`process_in_place`](CabSimAdapter::process_in_place) to process a
    ///     single buffer in place.
    ///
    /// # Host Contract Violations (F-03)
    ///
    /// During host quantum renegotiation a sub-block may exceed `partition_size`
    /// (or `input`/`output` lengths may disagree) before the new adapter arrives
    /// via the SPSC swap. Instead of panicking on the audio thread, the adapter
    /// clamps processing to `min(input.len(), partition, output.len())` and
    /// raises [`RT_STATUS_CABSIM_CONTRACT_VIOLATION`] on `rt_status` (lock-free,
    /// zero-alloc, no RT logging) so the control thread can log/remediate.
    /// The main thread observes the flag together with
    /// [`crate::common::spsc::RT_STATUS_NEEDS_CABSIM_REBUILD`].
    ///
    /// # RT-Safety
    ///
    /// Zero-alloc, lock-free, never panics.
    pub fn process_variable(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    ) {
        let sub_n = input.len().min(self.partition).min(output.len());
        debug_assert_eq!(output.len(), input.len());
        let contract_violation = input.len() > self.partition || output.len() != input.len();
        if contract_violation && let Some(rt) = rt_status {
            rt.set_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION);
        }

        if self.engine.is_passthrough() {
            output[..sub_n].copy_from_slice(&input[..sub_n]);
            output[sub_n..].fill(0.0);
            return;
        }

        self.accumulate(input, sub_n);
        self.run_partitions(rt_status);
        self.deliver(sub_n, output);
    }

    /// In-place variant of [`process_variable`](CabSimAdapter::process_variable)
    /// that reads and writes the same buffer, avoiding a separate destination
    /// slice (and its intermediate copy) on the audio callback.
    ///
    /// All input samples are consumed into the internal input FIFO before any
    /// output is written back into `input_output`, so aliasing the source and
    /// destination is safe.
    ///
    /// # Constraints
    ///
    /// *   `input_output.len() <= partition_size`
    /// *   `input_output.len() > 0` (use empty slice for flush passes)
    ///
    /// # RT-Safety
    ///
    /// Zero-alloc, lock-free, never panics.
    pub fn process_in_place(
        &mut self,
        input_output: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    ) {
        let sub_n = input_output.len().min(self.partition);
        let contract_violation = input_output.len() > self.partition;
        if contract_violation && let Some(rt) = rt_status {
            rt.set_flag(RT_STATUS_CABSIM_CONTRACT_VIOLATION);
        }

        if self.engine.is_passthrough() {
            input_output[sub_n..].fill(0.0);
            return;
        }

        self.accumulate(input_output, sub_n);
        self.run_partitions(rt_status);
        self.deliver(sub_n, input_output);
    }

    /// Consumes `sub_n` samples from `input` into the input FIFO.
    #[inline(always)]
    fn accumulate(&mut self, input: &[f32], sub_n: usize) {
        if sub_n > 0 {
            self.input_buf[self.input_count..self.input_count + sub_n]
                .copy_from_slice(&input[..sub_n]);
            self.input_count += sub_n;
        }
    }

    /// Renders every accumulated full partition directly into the free region
    /// of `self.output_buf`, keeping the output FIFO contiguous via the
    /// shift-to-front compaction.
    #[inline(always)]
    fn run_partitions(&mut self, rt_status: Option<&RtStatusFlags>) {
        while self.input_count >= self.partition {
            if self.output_read > 0 {
                let remaining = self.output_write - self.output_read;
                if remaining > 0 {
                    self.output_buf
                        .copy_within(self.output_read..self.output_write, 0);
                }
                self.output_write = remaining;
                self.output_read = 0;
            }

            // P-04 / T5.1: render straight into the output-FIFO partitions. The
            // compaction above guarantees at most `partition` samples remain
            // queued at the front, so the write region
            // `[output_write, output_write + partition)` never overflows the
            // `2 * partition` pre-allocated queue.
            debug_assert!(self.output_write + self.partition <= self.output_buf.len());
            self.engine.process(
                &self.input_buf[..self.partition],
                &mut self.output_buf[self.output_write..self.output_write + self.partition],
                rt_status,
            );
            self.output_write += self.partition;

            let remaining = self.input_count - self.partition;
            if remaining > 0 {
                self.input_buf
                    .copy_within(self.partition..self.input_count, 0);
            }
            self.input_count = remaining;
        }
    }

    /// Moves up to `sub_n` queued output samples into `output`, filling the
    /// remainder with silence.
    #[inline(always)]
    fn deliver(&mut self, sub_n: usize, output: &mut [f32]) {
        let available = self.output_write - self.output_read;
        let n = sub_n.min(available);
        if n > 0 {
            output[..n].copy_from_slice(&self.output_buf[self.output_read..self.output_read + n]);
            self.output_read += n;
        }
        output[n..].fill(0.0);

        if self.output_read >= self.output_write {
            self.output_read = 0;
            self.output_write = 0;
        }
    }
}

#[cfg(test)]
#[path = "adapter_test.rs"]
mod adapter_test;
