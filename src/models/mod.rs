// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Neural Inference Architectures (Brain Engines) module for NAM-rs.
//!
//! This module contains the acoustic brains of the program: neural networks that have learned how,
//! for example, a real amplifier or pedal distorts and colors a guitar sound.

/// A2 architecture (v0.6+): FiLM, gating, head1x1, bottleneck, multi-array cascades.
pub mod a2;
/// Slimmable model container: multi-size bundles with quality-threshold based dispatch.
pub mod container;
/// ConvNet feed-forward architecture.
pub mod convnet;
/// Linear FIR model: dot product of input history with learned weights + bias.
pub mod linear;
/// Linear FFT model: frequency-domain overlap-save FIR convolution kernel.
pub mod linear_fft;
/// LSTM recurrent architecture: configurable layers × hidden units, gate-level SIMD acceleration.
pub mod lstm;
/// Slimmable channel-slicing dispatcher for WaveNet quality-tier transitions.
pub mod slimmable;
/// WaveNet dilated convolution architecture: Standard, Lite, Feather, Nano, Dynamic variants.
pub mod wavenet;

/// NamModel trait implementation for StaticModel (dispatch methods).
mod nam_model;
mod static_model;

// =============================================================================
// Sealed Pattern — Prevents external implementations of NamModel
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

// =============================================================================
// Trait NamModel — Public Contract
// =============================================================================

/// Interface for all neural network model architectures in `NeuralAmpModeler-rs`.
///
/// `NamModel` defines the operational contract for acoustic neural inference
/// engines (WaveNet A1/A2, LSTM, ConvNet, Linear FIR/FFT, and Slimmable containers).
///
/// # Lifecycle & Execution Flow
///
/// 1. **Off-RT Instantiation & Prewarming:**
///    Models are constructed outside the real-time audio thread via [`loader::load_and_build_model`](crate::loader::load_and_build_model)
///    or concrete architecture constructors. During instantiation, weights are packed into
///    64-byte aligned SIMD structures (`AlignedVec<f32>`), internal history state buffers are
///    allocated, and [`prewarm`](NamModel::prewarm) is executed to prime dilated convolution
///    buffers or recurrent states.
///
/// 2. **Real-Time Audio Hot-Path Processing:**
///    The DAW audio callback or standalone audio loop invokes [`process`](NamModel::process) on
///    each audio quantum (block of `f32` samples). Execution strictly guarantees:
///    - **Zero Heap Allocations:** No `Box`, `Vec`, `String`, or dynamic allocation occurs during `process`.
///    - **Zero Mutex Locks / Blocking I/O:** No locks, condition variables, file I/O, or logging.
///    - **Deterministic Real-Time Bounds:** SIMD inner loops (AVX2 / AVX-512) execute within
///      sub-millisecond deadlines.
///
/// 3. **State Resets & Buffer Reallocations:**
///    When sample rates or maximum buffer sizes change, the control thread invokes [`reset`](NamModel::reset)
///    or [`set_max_buffer_size`](NamModel::set_max_buffer_size). Re-allocations happen off-RT,
///    preserving zero-allocation guarantees during subsequent audio callbacks.
///
/// 4. **Swapping & GC Deallocation Cascade:**
///    When models or quality tiers are swapped dynamically, old model instances are transferred via an
///    SPSC channel to an off-RT Garbage Collector (`GcProducer`), ensuring deallocation drops
///    happen off the audio thread.
///
/// # Thread Safety & Trait Sealing
///
/// `NamModel` requires `Send + Sync`, enabling safe cross-thread transfer and multi-threaded host dispatch.
/// The trait is sealed via `sealed::Sealed` to restrict public implementations to this crate, enabling
/// static dispatch via [`StaticModel`].
pub trait NamModel: Send + Sync + sealed::Sealed {
    /// Invoked by the DSP audio thread to process an acoustic sample block.
    ///
    /// # Length Contract
    /// `output.len()` may be smaller than `input.len()`; every implementation
    /// clamps to `n = input.len().min(output.len())` and never indexes beyond
    /// `output[..n]`. Samples past `n` in `output` are left untouched and the
    /// excess input is not consumed, so `process` never panics on asymmetric
    /// buffer lengths. Hosts are expected to use equal-length buffers, but the
    /// engine degrades gracefully when they do not.
    ///
    /// # Real-Time Safety
    /// This method MUST NOT allocate on the heap, acquire locks, or perform blocking I/O.
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// Primes internal state buffers by processing `num_samples` of zeroed input off-RT.
    ///
    /// Stabilizes receptive fields in WaveNet or recurrent states in LSTM before live audio processing.
    fn prewarm(&mut self, num_samples: usize);

    /// Returns whether prewarm should be executed on [`reset`](NamModel::reset).
    ///
    /// Default: `true` (prewarm on every reset).
    fn prewarm_on_reset(&self) -> bool {
        true
    }

    /// Sets whether prewarm should be executed on [`reset`](NamModel::reset).
    ///
    /// Default: no-op (fixed-size models ignore this flag).
    fn set_prewarm_on_reset(&mut self, _val: bool) {}

    /// Resets internal model states with a new sample rate and maximum block size.
    ///
    /// Default implementation calls `prewarm(max_buffer_size)` if `prewarm_on_reset()` is `true`.
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset() {
            self.prewarm(max_buffer_size);
        }
        Ok(())
    }

    /// Reallocates internal scratch buffers to support up to `max_buf` samples.
    ///
    /// Default: no-op (suitable for static models and LSTM).
    fn set_max_buffer_size(&mut self, _max_buf: usize) -> anyhow::Result<()> {
        Ok(())
    }

    /// Returns the number of samples needed to fully stabilize internal states.
    ///
    /// Default: `0` (suitable for LSTM). WaveNet models return their total receptive field depth.
    fn prewarm_samples(&self) -> usize {
        0
    }

    /// Returns quality-tier breakpoints `[0.0, 1.0]` for slimmable model bundles.
    ///
    /// # Allocation note
    /// The returned `Box<[f64]>` is allocated off-RT during configuration. MUST NOT be called on hot-path.
    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        Box::new([])
    }
}

// ── API Return Type Policy ────────────────────────────────────────────────────
// Methods returning collections of model configuration (not audio samples) use:
//   • Box<[T]>  when the set is fixed-size and immutable after model load.
//   • Vec<T>    only when the set is dynamic and caller-growable (justify inline).
// All collection-returning methods are off-RT only; document this in their
// doc-comments with the "# Allocation note" section.

/// Wrapper enum for trained model variants.
/// Enables static dispatch of DSP calls to the concrete variant, avoiding vtable overhead.
///
/// Named `StaticModel` because all variants are compile-time-fixed geometries.
/// The legacy "Dynamic" mode (arbitrary geometry at runtime) has been retired.
pub enum StaticModel {
    /// WaveNet Standard (16 channels, kernel 3, dilation 8).
    WavenetStandard(Box<wavenet::WaveNetModel<16, 3, 8>>),
    /// WaveNet Lite (12 channels, kernel 3, dilation 6).
    WavenetLite(Box<wavenet::WaveNetModel<12, 3, 6>>),
    /// WaveNet Feather (8 channels, kernel 3, dilation 4).
    WavenetFeather(Box<wavenet::WaveNetModel<8, 3, 4>>),
    /// WaveNet Nano (4 channels, kernel 3, dilation 2).
    WavenetNano(Box<wavenet::WaveNetModel<4, 3, 2>>),
    /// WaveNet A2 Full (8 channels, real inference).
    WavenetA2Full(Box<a2::WaveNetA2<8>>),
    /// WaveNet A2 Lite (3 channels, real inference).
    WavenetA2Lite(Box<a2::WaveNetA2<3>>),
    /// WaveNet A2 Dynamic (runtime-dimensioned, full topology spectrum).
    WavenetA2Dyn(Box<a2::WaveNetA2Dyn>),
    /// WaveNet A2 Cascade (multi-array chain of Dynamic engines).
    WavenetA2Cascade(Box<a2::WaveNetA2Cascade>),
    /// WaveNet Dynamic (runtime-dimensioned, free geometry).
    WavenetDyn(Box<wavenet::WaveNetModelDyn>),
    /// LSTM 1 Layer × 3 hidden units.
    Lstm1x3(Box<lstm::Lstm1x3>),
    /// LSTM 1 Layer × 8 hidden units.
    Lstm1x8(Box<lstm::Lstm1x8>),
    /// LSTM 1 Layer × 12 hidden units.
    Lstm1x12(Box<lstm::Lstm1x12>),
    /// LSTM 1 Layer × 16 hidden units.
    Lstm1x16(Box<lstm::Lstm1x16>),
    /// LSTM 1 Layer × 24 hidden units.
    Lstm1x24(Box<lstm::Lstm1x24>),
    /// LSTM 2 Layers × 8 hidden units.
    Lstm2x8(Box<lstm::Lstm2x8>),
    /// LSTM 2 Layers × 12 hidden units.
    Lstm2x12(Box<lstm::Lstm2x12>),
    /// LSTM 2 Layers × 16 hidden units.
    Lstm2x16(Box<lstm::Lstm2x16>),
    /// LSTM 1 Layer × 40 hidden units.
    Lstm1x40(Box<lstm::Lstm1x40>),
    /// LSTM 2 Layers × 24 hidden units.
    Lstm2x24(Box<lstm::Lstm2x24>),
    /// LSTM Dynamic — runtime-dimensioned, free geometry (F7 fallback).
    LstmDyn(Box<lstm::LstmModelDyn>),
    /// SlimmableContainer — bundle of submodels selected by quality threshold.
    Container(Box<container::ContainerModel>),
    /// Linear — FIR-based model (dot product of input history with weights + bias).
    Linear(Box<linear::LinearModel>),
    /// ConvNet feed-forward model (F4).
    ConvNet(Box<convnet::ConvNetModel>),
}

impl sealed::Sealed for StaticModel {}

pub(crate) use static_model::clone_condition_dsp;
