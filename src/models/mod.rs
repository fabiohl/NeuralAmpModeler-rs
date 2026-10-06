// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Neural Inference Architectures (Brain Engines) module for NeuralAmpModeler-rs.
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
    /// # Block Size Contract
    /// Pre-condition: `input.len()` must not exceed the negotiated maximum
    /// block size (`max_buffer_size`, set off-RT via
    /// [`set_max_buffer_size`](NamModel::set_max_buffer_size)). Input beyond
    /// that limit is silently truncated in release builds (only the first
    /// `max_buffer_size` frames are processed); debug builds trap the
    /// violation with a symmetric `debug_assert!` in every engine.
    ///
    /// # Real-Time Safety
    /// This method MUST NOT allocate on the heap, acquire locks, or perform blocking I/O.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
    /// use neural_amp_modeler_rs::models::NamModel;
    /// use neural_amp_modeler_rs::SystemSnapshot;
    ///
    /// let sys = SystemSnapshot::capture();
    /// let mut pair = load_and_build_model(
    ///     Path::new("path/to/model.nam"),
    ///     &sys,
    ///     false, // dual_mono: left-channel only
    ///     LoadOptions::default(),
    /// )
    /// .expect("Failed to load model");
    /// let model = pair.model_l.as_mut().expect("mono load yields model_l");
    ///
    /// // Process a block of audio samples
    /// let input = [0.0_f32; 64];
    /// let mut output = [0.0_f32; 64];
    /// model.process(&input, &mut output);
    /// ```
    fn process(&mut self, input: &[f32], output: &mut [f32]);

    /// Primes internal state buffers by processing zeroed input off-RT.
    ///
    /// Stabilizes receptive fields in WaveNet/ConvNet or recurrent states in LSTM before live audio
    /// processing.
    ///
    /// # Per-family semantics of `num_samples`
    ///
    /// - **LSTM, Linear, and Container:** honor `num_samples` — the recurrent/FIR state is primed by
    ///   processing exactly `num_samples` zeroed samples. Use [`prewarm_samples`](NamModel::prewarm_samples)
    ///   for the recommended count (LSTM returns half the expected sample rate).
    /// - **WaveNet A1/A2 and ConvNet:** ignore `num_samples`; the implementation performs a fixed
    ///   one-shot prewarm (inherent method) that fills the full receptive field regardless of the
    ///   argument. The value passed is irrelevant to the outcome.
    fn prewarm(&mut self, num_samples: usize);

    /// Clears temporal state to the freshly-built condition, leaving the
    /// stabilization pass outstanding.
    ///
    /// This is the frontier splitting an integral [`prewarm`](Self::prewarm)
    /// pass in two phases for real-time consumers with hard per-block
    /// deadlines:
    ///
    /// 1. **State-clearing phase (this method):** resets every temporal
    ///    buffer to the condition a just-constructed-and-stabilized model
    ///    reaches — delay/ring/recurrent state, ring write cursors, and the
    ///    metric preamble of the integral pass — while performing *none* of
    ///    the zeroed-sample stabilization work. Bounded O(ring clearing).
    /// 2. **Stabilization phase ([`prewarm_step`](Self::prewarm_step)):**
    ///    consumes zeroed samples toward convergence until
    ///    [`prewarm_complete`](Self::prewarm_complete) reports `true`.
    ///
    /// After this call, [`prewarm_complete`](Self::prewarm_complete) reports
    /// `false` whenever the stabilization work is not reduced to zero, and
    /// progress is made **exclusively** via
    /// [`prewarm_step`](Self::prewarm_step) calls. The state *pending* is
    /// internal to the model; the integral paths
    /// ([`prewarm`](Self::prewarm) / [`reset`](Self::reset)) are not altered
    /// by or observed through this split state.
    ///
    /// # Real-Time Safety
    /// Zero-allocation, lock-free, no I/O, bounded by the model's ring sizes
    /// — safe on the audio thread. The caller controls the step budget via
    /// [`prewarm_step`](Self::prewarm_step).
    ///
    /// # Per-family semantics
    /// Every concrete family implements the frontier so that
    /// `prewarm_reset()` followed by `prewarm_step` calls until completion
    /// reproduces the *bit-exact* post-stabilization state of the integral
    /// flow, independently of how the zeroed samples are chunked (validated
    /// by the engine's split-vs-integral equivalence tests).
    fn prewarm_reset(&mut self) {}

    /// Advances the outstanding split-stabilization pass by at most
    /// `samples` zeroed samples.
    ///
    /// Returns the stabilization work still pending after the call (in the
    /// family's own zeroed-sample unit of account; `0` = converged). The
    /// caller bounds its per-clock budget by the `samples` argument; the
    /// family clamps its work to that budget.
    ///
    /// # Real-Time Safety
    /// This method MUST NOT allocate on the heap, acquire locks, or perform
    /// blocking I/O — it is the same contract as [`process`](Self::process).
    /// Implementations drive the stabilization exclusively through the
    /// family's own `process` path or backfill kernels, so the resulting
    /// state is bit-equal to the integral stabilization regardless of the
    /// chunking the caller chooses.
    ///
    /// Default: performs the integral stabilization pass over
    /// [`prewarm_samples`](Self::prewarm_samples) and reports convergence.
    /// This is the conservative fallback for families without a split
    /// implementation; overriding it makes the stabilization amortizable.
    fn prewarm_step(&mut self, samples: usize) -> usize {
        let _ = samples;
        core::hint::cold_path();
        self.prewarm(self.prewarm_samples());
        0
    }

    /// Reports whether the split-stabilization pass requested by
    /// [`prewarm_reset`](Self::prewarm_reset) has fully converged.
    ///
    /// Always `true` for a freshly built model — the pending state is armed
    /// exclusively by [`prewarm_reset`](Self::prewarm_reset) and cleared when
    /// the stabilization work reaches zero. The integral paths
    /// ([`prewarm`](Self::prewarm) / [`reset`](Self::reset)) do not consult
    /// or alter it: a family that stabilizes during those calls converges by
    /// construction.
    fn prewarm_complete(&self) -> bool {
        true
    }

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
    ///
    /// # Errors
    ///
    /// Implementations may return an error if internal state buffers cannot be
    /// (re)allocated for `max_buffer_size` (out of memory). The default
    /// implementation never fails.
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        if self.prewarm_on_reset() {
            self.prewarm(max_buffer_size);
        }
        Ok(())
    }

    /// Reallocates internal scratch buffers to support up to `max_buf` samples.
    ///
    /// Default: no-op (suitable for static models and LSTM).
    ///
    /// # Errors
    ///
    /// Implementations may return an error if the scratch buffer cannot be
    /// (re)allocated for `max_buf` (out of memory). The default implementation
    /// (no-op) never fails.
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
/// Contains optimized static variants with compile-time fixed geometries (enabling
/// auto-vectorization and loop unrolling), alongside zero-allocation dynamic fallback variants
/// (`WavenetDyn`, `WavenetA2Dyn`, `WavenetA2Cascade`, `LstmDyn`) for arbitrary architectures.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use neural_amp_modeler_rs::prelude::*;
///
/// let sys = SystemSnapshot::capture();
/// let pair = load_and_build_model(
///     Path::new("models/amp.nam"),
///     &sys,
///     false, // dual_mono: left-channel only
///     LoadOptions::default(),
/// ).expect("failed to load model");
///
/// if let Some(mut model) = pair.model_l {
///     let input = [0.0f32; 64];
///     let mut output = [0.0f32; 64];
///     model.process(&input, &mut output);
/// }
/// ```
#[non_exhaustive]
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
