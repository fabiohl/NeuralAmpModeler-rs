// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unified dispatch layer: maps the [`NamModel`] trait interface onto every
//! concrete [`StaticModel`] variant via exhaustive `match` arms.
//!
//! # Concurrency model
//! All methods in this `impl` are called from the audio processing thread
//! (`SCHED_FIFO`) and must remain panic-free. The `process` method on
//! `Linear` wraps its call in `unsafe` because the Linear kernel uses
//! raw-pointer SIMD intrinsics that do not carry lifetime bounds through
//! the trait indirection; the pointer is derived from the `&[f32]`
//! arguments already borrowed by `process`.
//!
//! # Adding a new model variant
//! 1. Add the variant to [`StaticModel`].
//! 2. Thread it through every `match` arm in this file.
//! 3. Add the corresponding dispatch arm in
//!    [`crate::loader::dispatcher`] and [`crate::loader::nam_json::topology`].

use super::slimmable::SlimmableModel;
use super::{NamModel, StaticModel};

impl NamModel for StaticModel {
    /// Routes sample-by-sample neural inference to the active model variant.
    ///
    /// # Design Rationale: Static Enum Dispatch vs Dynamic Dispatch
    /// Instead of dynamic trait object dispatch (`Box<dyn NamModel>` or `&mut dyn NamModel`),
    /// this engine uses static enum dispatch via exhaustive `match` arms over [`StaticModel`]:
    ///
    /// - **Predictable Branch Target Buffer (BTB) Performance:** In an audio processing session,
    ///   the active model variant does not change dynamically on a sample-by-sample basis.
    ///   Consequently, after the first branch prediction warm-up, the CPU BTB achieves
    ///   near-100% prediction accuracy without pipeline flushes or indirect branch penalties.
    /// - **Devirtualization and Monomorphization:** Enum dispatch eliminates vtable pointer
    ///   indirection and indirect call overhead (`call *%rax`). The compiler is able to
    ///   devirtualize each call site, inline target kernels where profitable, and perform
    ///   per-architecture SIMD register allocation and instruction scheduling.
    /// - **Cache and Real-Time Safety:** Trait objects typically incur fat pointers (data ptr +
    ///   vtable ptr) and dynamic heap indirection. Static enum dispatch keeps state layouts
    ///   deterministic, cache-local, and avoids pointer chasing on the critical `SCHED_FIFO`
    ///   audio callback thread.
    ///
    /// # Safety and Wrapping
    /// All WaveNet-derived, LSTM, ConvNet, and Container variants delegate directly to their
    /// respective `process` methods. The `Linear` variant uses an `unsafe` block because the
    /// raw pointer passed to the GEMM kernel cannot carry a compile-time lifetime through the
    /// enum dispatch. The pointer validity is guaranteed by the `&[f32]` / `&mut [f32]`
    /// borrows already held by the caller.
    #[inline(always)]
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self {
            Self::WavenetStandard(m) => m.process(input, output),
            Self::WavenetLite(m) => m.process(input, output),
            Self::WavenetFeather(m) => m.process(input, output),
            Self::WavenetNano(m) => m.process(input, output),
            Self::WavenetA2Full(m) => m.process(input, output),
            Self::WavenetA2Lite(m) => m.process(input, output),
            Self::WavenetA2Dyn(m) => m.process(input, output),
            Self::WavenetA2Cascade(m) => m.process(input, output),
            Self::WavenetDyn(m) => m.process(input, output),
            Self::Container(m) => m.process(input, output),
            Self::Lstm1x3(m) => m.process(input, output),
            Self::Lstm1x8(m) => m.process(input, output),
            Self::Lstm1x12(m) => m.process(input, output),
            Self::Lstm1x16(m) => m.process(input, output),
            Self::Lstm1x24(m) => m.process(input, output),
            Self::Lstm2x8(m) => m.process(input, output),
            Self::Lstm2x12(m) => m.process(input, output),
            Self::Lstm2x16(m) => m.process(input, output),
            Self::Lstm1x40(m) => m.process(input, output),
            Self::Lstm2x24(m) => m.process(input, output),
            Self::LstmDyn(m) => m.process(input, output),
            // SAFETY: the `Linear` kernel takes a raw pointer that cannot carry a compile-time
            // lifetime through the enum dispatch; the pointer is derived from the `&[f32]` /
            // `&mut [f32]` borrows already held by this function (see module docs).
            Self::Linear(m) => unsafe { m.process(input, output) },
            Self::ConvNet(m) => m.process(input, output),
        }
    }

    /// Prewarms internal causal states (convolution delay buffers, LSTM hidden/cell states).
    ///
    /// # Audio Invariants and Click Prevention
    /// Recursive networks (LSTM) and causal dilated convolutions (WaveNet, ConvNet) rely on
    /// internal recurrent history or ring buffers. On cold startup or transport restart,
    /// uninitialized or zero-filled history buffers can produce transient discontinuities,
    /// DC offset step responses, or audible clicks/pops when audio playback begins.
    ///
    /// Calling `prewarm` pushes a sequence of silent samples (or initial baseline frames)
    /// through the network, allowing internal state variables to settle into their stable
    /// operating regime before live audio is routed through [`process`](Self::process).
    ///
    /// # Execution Context
    /// This method is marked `#[cold]` because it runs strictly off-RT during model instantiation,
    /// buffer re-allocation, or transport reset—never in the per-buffer hot audio path.
    #[cold]
    fn prewarm(&mut self, num_samples: usize) {
        match self {
            Self::WavenetStandard(m) => m.prewarm(),
            Self::WavenetLite(m) => m.prewarm(),
            Self::WavenetFeather(m) => m.prewarm(),
            Self::WavenetNano(m) => m.prewarm(),
            Self::WavenetA2Full(m) => m.prewarm(),
            Self::WavenetA2Lite(m) => m.prewarm(),
            Self::WavenetA2Dyn(m) => m.prewarm(),
            Self::WavenetA2Cascade(m) => m.prewarm(),
            Self::WavenetDyn(m) => m.prewarm(),
            Self::Container(m) => m.prewarm(num_samples),
            Self::Lstm1x3(m) => m.prewarm(num_samples),
            Self::Lstm1x8(m) => m.prewarm(num_samples),
            Self::Lstm1x12(m) => m.prewarm(num_samples),
            Self::Lstm1x16(m) => m.prewarm(num_samples),
            Self::Lstm1x24(m) => m.prewarm(num_samples),
            Self::Lstm2x8(m) => m.prewarm(num_samples),
            Self::Lstm2x12(m) => m.prewarm(num_samples),
            Self::Lstm2x16(m) => m.prewarm(num_samples),
            Self::Lstm1x40(m) => m.prewarm(num_samples),
            Self::Lstm2x24(m) => m.prewarm(num_samples),
            Self::LstmDyn(m) => m.prewarm(num_samples),
            Self::Linear(m) => m.prewarm(num_samples),
            Self::ConvNet(m) => m.prewarm(),
        }
    }

    /// Queries whether internal recurrent states should be prewarmed when the DSP pipeline resets.
    ///
    /// When `true`, subsequent invocations of [`reset`](Self::reset) will automatically flush
    /// internal delay buffers and prime recurrent states to prevent cold-start transients.
    fn prewarm_on_reset(&self) -> bool {
        match self {
            Self::WavenetStandard(m) => m.prewarm_on_reset(),
            Self::WavenetLite(m) => m.prewarm_on_reset(),
            Self::WavenetFeather(m) => m.prewarm_on_reset(),
            Self::WavenetNano(m) => m.prewarm_on_reset(),
            Self::WavenetA2Full(m) => m.prewarm_on_reset(),
            Self::WavenetA2Lite(m) => m.prewarm_on_reset(),
            Self::WavenetA2Dyn(m) => m.prewarm_on_reset(),
            Self::WavenetA2Cascade(m) => m.prewarm_on_reset(),
            Self::WavenetDyn(m) => m.prewarm_on_reset(),
            Self::Container(m) => m.prewarm_on_reset(),
            Self::Lstm1x3(m) => m.prewarm_on_reset(),
            Self::Lstm1x8(m) => m.prewarm_on_reset(),
            Self::Lstm1x12(m) => m.prewarm_on_reset(),
            Self::Lstm1x16(m) => m.prewarm_on_reset(),
            Self::Lstm1x24(m) => m.prewarm_on_reset(),
            Self::Lstm2x8(m) => m.prewarm_on_reset(),
            Self::Lstm2x12(m) => m.prewarm_on_reset(),
            Self::Lstm2x16(m) => m.prewarm_on_reset(),
            Self::Lstm1x40(m) => m.prewarm_on_reset(),
            Self::Lstm2x24(m) => m.prewarm_on_reset(),
            Self::LstmDyn(m) => m.prewarm_on_reset(),
            Self::Linear(m) => m.prewarm_on_reset(),
            Self::ConvNet(m) => m.prewarm_on_reset(),
        }
    }

    /// Configures whether internal recurrent states should be prewarmed on reset.
    ///
    /// Setting this to `true` ensures that any future transport reset or sample rate reconfiguration
    /// flushes recurrent states to steady-state before audio processing resumes.
    fn set_prewarm_on_reset(&mut self, val: bool) {
        match self {
            Self::WavenetStandard(m) => m.set_prewarm_on_reset(val),
            Self::WavenetLite(m) => m.set_prewarm_on_reset(val),
            Self::WavenetFeather(m) => m.set_prewarm_on_reset(val),
            Self::WavenetNano(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Full(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Lite(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Dyn(m) => m.set_prewarm_on_reset(val),
            Self::WavenetA2Cascade(m) => m.set_prewarm_on_reset(val),
            Self::WavenetDyn(m) => m.set_prewarm_on_reset(val),
            Self::Container(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x3(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x8(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x12(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x16(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x24(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x8(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x12(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x16(m) => m.set_prewarm_on_reset(val),
            Self::Lstm1x40(m) => m.set_prewarm_on_reset(val),
            Self::Lstm2x24(m) => m.set_prewarm_on_reset(val),
            Self::LstmDyn(m) => m.set_prewarm_on_reset(val),
            Self::Linear(m) => m.set_prewarm_on_reset(val),
            Self::ConvNet(m) => m.set_prewarm_on_reset(val),
        }
    }

    fn reset(&mut self, sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        match self {
            Self::WavenetStandard(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetLite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetFeather(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetNano(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Full(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Lite(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Dyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetA2Cascade(m) => m.reset(sample_rate, max_buffer_size),
            Self::WavenetDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::Container(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x3(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x8(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x12(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x16(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm1x40(m) => m.reset(sample_rate, max_buffer_size),
            Self::Lstm2x24(m) => m.reset(sample_rate, max_buffer_size),
            Self::LstmDyn(m) => m.reset(sample_rate, max_buffer_size),
            Self::Linear(m) => NamModel::reset(m.as_mut(), sample_rate, max_buffer_size),
            Self::ConvNet(m) => NamModel::reset(m.as_mut(), sample_rate, max_buffer_size),
        }
    }

    fn set_max_buffer_size(&mut self, max_buf: usize) -> anyhow::Result<()> {
        match self {
            Self::WavenetStandard(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetLite(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetFeather(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetNano(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Full(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Lite(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Dyn(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetA2Cascade(m) => m.set_max_buffer_size(max_buf),
            Self::WavenetDyn(m) => m.set_max_buffer_size(max_buf),
            Self::Container(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x3(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x24(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x8(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x12(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x16(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm1x40(m) => m.set_max_buffer_size(max_buf),
            Self::Lstm2x24(m) => m.set_max_buffer_size(max_buf),
            Self::LstmDyn(m) => m.set_max_buffer_size(max_buf),
            Self::Linear(m) => NamModel::set_max_buffer_size(m.as_mut(), max_buf),
            Self::ConvNet(m) => NamModel::set_max_buffer_size(m.as_mut(), max_buf),
        }
    }

    fn prewarm_samples(&self) -> usize {
        match self {
            Self::WavenetStandard(m) => m.prewarm_samples(),
            Self::WavenetLite(m) => m.prewarm_samples(),
            Self::WavenetFeather(m) => m.prewarm_samples(),
            Self::WavenetNano(m) => m.prewarm_samples(),
            Self::WavenetA2Full(m) => m.prewarm_samples(),
            Self::WavenetA2Lite(m) => m.prewarm_samples(),
            Self::WavenetA2Dyn(m) => m.prewarm_samples(),
            Self::WavenetA2Cascade(m) => m.prewarm_samples(),
            Self::WavenetDyn(m) => m.prewarm_samples(),
            Self::Container(m) => m.prewarm_samples(),
            Self::Lstm1x3(m) => m.prewarm_samples(),
            Self::Lstm1x8(m) => m.prewarm_samples(),
            Self::Lstm1x12(m) => m.prewarm_samples(),
            Self::Lstm1x16(m) => m.prewarm_samples(),
            Self::Lstm1x24(m) => m.prewarm_samples(),
            Self::Lstm2x8(m) => m.prewarm_samples(),
            Self::Lstm2x12(m) => m.prewarm_samples(),
            Self::Lstm2x16(m) => m.prewarm_samples(),
            Self::Lstm1x40(m) => m.prewarm_samples(),
            Self::Lstm2x24(m) => m.prewarm_samples(),
            Self::LstmDyn(m) => m.prewarm_samples(),
            Self::Linear(m) => m.prewarm_samples(),
            Self::ConvNet(m) => m.prewarm_samples(),
        }
    }

    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        match self {
            Self::Container(c) => SlimmableModel::slimmable_breakpoints(c.as_ref()),
            Self::WavenetDyn(m) => SlimmableModel::slimmable_breakpoints(m.as_ref()),
            _ => Box::new([]),
        }
    }
}
