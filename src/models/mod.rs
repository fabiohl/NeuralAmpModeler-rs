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
/// Sequential chain composition: serial pipeline of independent DSP stages
/// with validated channel links, resolved expected sample rate (DEC-01), and
/// single-pass chain-wide stabilization.
pub mod sequential;
/// Slimmable channel-slicing dispatcher for WaveNet quality-tier transitions.
pub mod slimmable;
/// WaveNet dilated convolution architecture: Standard, Lite, Feather, Nano, Dynamic variants.
pub mod wavenet;

/// NamModel trait implementation for StaticModel (dispatch methods).
mod nam_model;
/// Planar <-> interleaved boundary adapter (zero-allocation staging).
pub(crate) mod plane_adapter;
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

/// Number of channel pointers held on the stack per call by the safe
/// multichannel helpers.
///
/// This is **not** a model channel cap: it is only the threshold below which
/// the pointer tables are materialized on the stack (zero heap allocation).
/// Larger geometries fall back to a temporary heap table (`Vec`), so no valid
/// model geometry is rejected by this bound. The heap fallback runs off the
/// hot allocation budget: `process_multichannel`/`process_multichannel_in_place`
/// with `> 16` channels allocates two small pointer tables per call and must
/// not be used on the RT thread — prefer [`process_raw`](NamModel::process_raw)
/// there (T5.6 deviation record: the fallback is a documented host-side
/// convenience, not an RT path).
const STACK_CHANNEL_PTRS: usize = 16;

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

    /// Number of input channels required by [`process_multichannel`](Self::process_multichannel)
    /// and [`process_multichannel_in_place`](Self::process_multichannel_in_place).
    ///
    /// Mirrors the C++ NAMcore `DSP::NumInputChannels()` contract: the base class
    /// is constructed with an explicit input-channel count and every DSP exposes it.
    /// Default: `1` (mono models). Multichannel families override it with the
    /// geometry fixed at construction time.
    fn num_input_channels(&self) -> usize {
        1
    }

    /// Number of output channels produced by [`process_multichannel`](Self::process_multichannel)
    /// and [`process_multichannel_in_place`](Self::process_multichannel_in_place).
    ///
    /// Mirrors the C++ NAMcore `DSP::NumOutputChannels()` contract. Default:
    /// `1` (mono models). WaveNet A1/A2 with `head_size > 1`, Linear multi-output
    /// and Sequential chains override it with the geometry fixed at construction.
    fn num_output_channels(&self) -> usize {
        1
    }

    /// Processes a block of multichannel audio samples via raw channel pointers.
    ///
    /// # Safety
    /// - `input` must point to an array of at least `in_channels` valid non-null `*const f32` pointers,
    ///   each pointing to at least `num_frames` samples.
    /// - `output` must point to an array of at least `out_channels` valid non-null `*mut f32` pointers,
    ///   each pointing to at least `num_frames` writable samples.
    /// - In-place processing (`input == output` or aliasing between input and output channels) is
    ///   fully supported and guaranteed not to corrupt data.
    #[inline(always)]
    unsafe fn process_raw(
        &mut self,
        input: *const *const f32,
        output: *const *mut f32,
        num_frames: usize,
    ) {
        // SAFETY: Caller guarantees input/output pointer arrays hold the
        // declared geometry and each pointed-to plane covers `num_frames`.
        let (in0, out0) = unsafe { (*input, *output) };
        let in_ch = self.num_input_channels();
        let out_ch = self.num_output_channels();

        // Fast path: mono 1->1 model into distinct buffers — no alias, no staging.
        if in_ch == 1 && out_ch == 1 && !std::ptr::eq(in0, out0 as *const f32) {
            // SAFETY: `in0`/`out0` each cover `num_frames` and do not alias.
            let (in_slice, out_slice) = unsafe {
                (
                    core::slice::from_raw_parts(in0, num_frames),
                    core::slice::from_raw_parts_mut(out0, num_frames),
                )
            };
            self.process(in_slice, out_slice);
            return;
        }

        // SAFETY: Caller guarantees input pointer array holds in_ch valid pointers.
        let inputs = unsafe { core::slice::from_raw_parts(input, in_ch) };
        // SAFETY: Caller guarantees output pointer array holds out_ch valid pointers.
        let outputs = unsafe { core::slice::from_raw_parts(output, out_ch) };
        // SAFETY: `inputs[c]` and `outputs[c]` cover `num_frames` each.
        unsafe {
            plane_adapter::run_staged_io(
                inputs,
                outputs,
                num_frames,
                in_ch,
                out_ch,
                |inp, outp| self.process(inp, outp),
            );
        }
    }

    /// Safe multichannel processing helper taking slices of channel slices.
    ///
    /// The caller supplies one input plane per required input channel and one
    /// output plane per required output channel (as declared by
    /// [`num_input_channels`](Self::num_input_channels) /
    /// [`num_output_channels`](Self::num_output_channels)). Surplus planes are
    /// ignored, exactly like the C++ host contract; the frame count is the
    /// minimum length across the **required** planes only.
    ///
    /// # Errors
    /// Returns [`crate::common::diagnostics::NamErrorCode::HostFormatContractViolation`]
    /// (`E2304`) — without
    /// touching any pointer — when fewer input or output planes are supplied
    /// than the model's declared geometry. This is the fail-closed guard that
    /// removes the previous null-pointer dereference on under-sized plane sets.
    ///
    /// # Real-Time Safety Note
    /// Zero heap allocation when the required channel counts are `<= 16`. For
    /// higher channel counts a temporary pointer table is allocated (the
    /// engine applies no artificial channel cap); prefer
    /// [`process_raw`](Self::process_raw) there.
    fn process_multichannel(
        &mut self,
        inputs: &[&[f32]],
        outputs: &mut [&mut [f32]],
    ) -> Result<usize, crate::common::diagnostics::NamErrorCode> {
        let ni = self.num_input_channels();
        let no = self.num_output_channels();
        if inputs.len() < ni || outputs.len() < no {
            return Err(crate::common::diagnostics::NamErrorCode::HostFormatContractViolation);
        }
        let frames = inputs[..ni]
            .iter()
            .map(|p| p.len())
            .chain(outputs[..no].iter().map(|p| p.len()))
            .min()
            .unwrap_or(0);
        if frames == 0 {
            return Ok(0);
        }
        if ni <= STACK_CHANNEL_PTRS && no <= STACK_CHANNEL_PTRS {
            let mut in_ptrs = [core::ptr::null::<f32>(); STACK_CHANNEL_PTRS];
            let mut out_ptrs = [core::ptr::null_mut::<f32>(); STACK_CHANNEL_PTRS];
            for i in 0..ni {
                in_ptrs[i] = inputs[i].as_ptr();
            }
            for i in 0..no {
                out_ptrs[i] = outputs[i].as_mut_ptr();
            }
            // SAFETY: all `ni` input and `no` output slots are non-null (each
            // came from a valid slice) and each covers at least `frames`
            // samples by the `min` above; `process_raw` reads exactly the
            // declared geometry.
            unsafe {
                self.process_raw(in_ptrs.as_ptr(), out_ptrs.as_ptr(), frames);
            }
        } else {
            let in_ptrs: Vec<*const f32> = inputs[..ni].iter().map(|s| s.as_ptr()).collect();
            let out_ptrs: Vec<*mut f32> =
                outputs[..no].iter_mut().map(|s| s.as_mut_ptr()).collect();
            // SAFETY: same as the stack path — full non-null tables of the
            // declared geometry, each plane covering `frames` samples.
            unsafe {
                self.process_raw(in_ptrs.as_ptr(), out_ptrs.as_ptr(), frames);
            }
        }
        Ok(frames)
    }

    /// Safe multichannel in-place processing helper.
    ///
    /// The caller supplies a single plane set that is both read and written:
    /// the first [`num_input_channels`](Self::num_input_channels) planes are the
    /// inputs and the first [`num_output_channels`](Self::num_output_channels)
    /// planes are the outputs. At least `max(in, out)` planes must be present.
    ///
    /// # Errors
    /// Returns [`crate::common::diagnostics::NamErrorCode::HostFormatContractViolation`]
    /// (`E2304`) without
    /// touching any pointer when fewer than `max(in, out)` planes are supplied.
    ///
    /// # Real-Time Safety Note
    /// Zero heap allocation when `max(in, out) <= 16`; larger geometries
    /// allocate a temporary pointer table.
    fn process_multichannel_in_place(
        &mut self,
        planes: &mut [&mut [f32]],
    ) -> Result<usize, crate::common::diagnostics::NamErrorCode> {
        let ni = self.num_input_channels();
        let no = self.num_output_channels();
        let required = ni.max(no);
        if planes.len() < required {
            return Err(crate::common::diagnostics::NamErrorCode::HostFormatContractViolation);
        }
        let frames = planes[..required]
            .iter()
            .map(|p| p.len())
            .min()
            .unwrap_or(0);
        if frames == 0 {
            return Ok(0);
        }
        if required <= STACK_CHANNEL_PTRS {
            let mut in_ptrs = [core::ptr::null::<f32>(); STACK_CHANNEL_PTRS];
            let mut out_ptrs = [core::ptr::null_mut::<f32>(); STACK_CHANNEL_PTRS];
            for i in 0..ni {
                in_ptrs[i] = planes[i].as_ptr();
            }
            for i in 0..no {
                out_ptrs[i] = planes[i].as_mut_ptr();
            }
            // SAFETY: every required slot is non-null and covers at least
            // `frames` samples; the in-place kernels read their inputs before
            // writing their outputs.
            unsafe {
                self.process_raw(in_ptrs.as_ptr(), out_ptrs.as_ptr(), frames);
            }
        } else {
            let mut in_ptrs: Vec<*const f32> = Vec::with_capacity(ni);
            for plane in planes[..ni].iter() {
                in_ptrs.push(plane.as_ptr());
            }
            let mut out_ptrs: Vec<*mut f32> = Vec::with_capacity(no);
            for plane in planes[..no].iter_mut() {
                out_ptrs.push(plane.as_mut_ptr());
            }
            // SAFETY: see stack path.
            unsafe {
                self.process_raw(in_ptrs.as_ptr(), out_ptrs.as_ptr(), frames);
            }
        }
        Ok(frames)
    }

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

    /// Clears every temporal state buffer to the freshly-built (all-zero)
    /// condition, in place and without allocating.
    ///
    /// This is the state-clearing phase shared by [`reset`](Self::reset) and
    /// [`prewarm_reset`](Self::prewarm_reset): it restores delay-line/ring
    /// memory and cursors so a subsequent [`process`](Self::process) behaves
    /// exactly like a freshly constructed model, independently of any history.
    /// The C++ NAMcore `DSP::Reset` (`dsp.cpp:133-140`) always clears state via
    /// `SetMaxBufferSize` before the optional prewarm; this method is the Rust
    /// engine's equivalent for families whose state clearing is a distinct,
    /// allocation-free operation (the stabilization pass is separate).
    ///
    /// The default is a no-op for families that already clear their state
    /// through `reset` (LSTM recurrent state, Linear FIR history, A2 ring
    /// masks, composed Container/Sequential children).
    ///
    /// # Real-Time Safety
    /// Zero-allocation, lock-free, no I/O, bounded by the model's ring sizes.
    fn clear_state(&mut self) {}

    /// Resets internal model states with a new sample rate and maximum block size.
    ///
    /// Default implementation clears every temporal state buffer
    /// ([`clear_state`](Self::clear_state)) and then calls
    /// `prewarm(max_buffer_size)` if `prewarm_on_reset()` is `true`. This mirrors
    /// the C++ NAMcore `DSP::Reset` order: state is always cleared, and the
    /// stabilization pass is gated by the prewarm policy.
    ///
    /// # Errors
    ///
    /// Implementations may return an error if internal state buffers cannot be
    /// (re)allocated for `max_buffer_size` (out of memory). The default
    /// implementation never fails.
    fn reset(&mut self, _sample_rate: u32, max_buffer_size: usize) -> anyhow::Result<()> {
        self.clear_state();
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

    /// Measures this model's steady-state per-sample inference cost, in
    /// nanoseconds, on the calling thread, then restores the model to its
    /// zero/prewarm-equivalent (reset) condition.
    ///
    /// # Method
    /// After one untimed warm-up pass, executes five timed passes of a
    /// `frames`-sample block of deterministic pseudo-random
    /// excitation (fixed-seed `xorshift32`) and returns the **minimum**
    /// observed `ns / sample`. The minimum — not the mean — makes the estimate
    /// robust to OS preemption, frequency transitions, and other external
    /// jitter on the calibration host.
    ///
    /// Timing uses the calibrated TSC ([`crate::common::tsc::rdtsc_nanos`]);
    /// when the TSC is not calibrated, that helper's `Instant` fallback is
    /// used, so an uncalibrated host never panics here.
    ///
    /// # State restoration
    /// Measurement drives real audio through [`process`](Self::process), which
    /// mutates temporal state. Before returning, the model is restored with the
    /// engine's two-phase stabilization
    /// ([`prewarm_reset`](Self::prewarm_reset), then
    /// [`prewarm_step`](Self::prewarm_step) until
    /// [`prewarm_complete`](Self::prewarm_complete)) — the same split pass as
    /// [`reset`](Self::reset), which the engine proves bit-exact against the
    /// integral reset. The restored state is therefore the **zero/prewarm
    /// equivalent** state, which for families with `prewarm_samples() == 0`
    /// (e.g. `Linear`) still advances their canonical cursor by the single
    /// zeroed stabilization sample, and does not restore an in-flight
    /// `Container` crossfade transition. Callers that need the exact state of a
    /// separately constructed instance must not rely on a bit-for-bit match for
    /// those families. The excitation/output buffers are allocated here, off-RT.
    ///
    /// # Real-time safety
    /// This method MUST NOT be called on the audio thread: it allocates
    /// (`Vec`), runs several off-deadline inference passes, and performs the
    /// full stabilization pass. It is intended for the control thread during
    /// model load or reconfiguration. `frames` should not exceed the model's
    /// negotiated maximum block size.
    fn measure_cost_ns_per_sample(&mut self, frames: usize) -> f32
    where
        Self: Sized,
    {
        measure_cost_ns_per_sample_impl(self, frames)
    }

    /// Returns quality-tier breakpoints `[0.0, 1.0]` for slimmable model bundles.
    ///
    /// # Allocation note
    /// The returned `Box<[f64]>` is allocated off-RT during configuration. MUST NOT be called on hot-path.
    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        Box::new([])
    }
}

/// Core of [`NamModel::measure_cost_ns_per_sample`], generic over the concrete
/// model so the provided trait method can require `Self: Sized` (keeping the
/// trait object-safe) while still measuring every family through the shared
/// [`NamModel`] surface.
fn measure_cost_ns_per_sample_impl<M: NamModel>(model: &mut M, frames: usize) -> f32 {
    /// Number of timed passes; one extra untimed warm-up pass precedes them.
    const RUNS: usize = 5;

    let frames = frames.max(1);
    let in_ch = model.num_input_channels().max(1);
    let out_ch = model.num_output_channels().max(1);

    // Deterministic excitation (fixed-seed xorshift32): identical arithmetic
    // for every pass, so data-dependent branches never perturb the estimate.
    let mut state: u32 = 0x9E37_79B9;
    let mut excitation = vec![0.0f32; frames];
    for sample in excitation.iter_mut() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *sample = (state as f32) * (2.0 / u32::MAX as f32) - 1.0;
    }

    let mut outputs: Vec<Vec<f32>> = (0..out_ch).map(|_| vec![0.0f32; frames]).collect();

    // Minimum of RUNS timed passes, one per pass plus an initial warm-up.
    let mut best = u64::MAX;
    if in_ch == 1 && out_ch == 1 {
        for pass in 0..=RUNS {
            let start = crate::common::tsc::rdtsc_nanos();
            model.process(&excitation, &mut outputs[0]);
            if pass == 0 {
                continue;
            }
            best = best.min(crate::common::tsc::rdtsc_nanos().wrapping_sub(start));
        }
    } else {
        let in_planes: Vec<&[f32]> = (0..in_ch).map(|_| excitation.as_slice()).collect();
        let mut out_planes: Vec<&mut [f32]> = outputs
            .iter_mut()
            .map(|plane| plane.as_mut_slice())
            .collect();
        for pass in 0..=RUNS {
            let start = crate::common::tsc::rdtsc_nanos();
            let _ = model.process_multichannel(&in_planes, &mut out_planes);
            if pass == 0 {
                continue;
            }
            best = best.min(crate::common::tsc::rdtsc_nanos().wrapping_sub(start));
        }
    }

    // Restore the freshly-built, stabilized condition via the bit-exact split
    // pass shared with `reset`, so the caller observes no state drift.
    model.prewarm_reset();
    while !model.prewarm_complete() {
        if model.prewarm_step(usize::MAX) == 0 {
            break;
        }
    }

    if best == u64::MAX {
        return 0.0;
    }
    best as f32 / frames as f32
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
    /// Sequential — serial chain of independent DSP stages (validated
    /// channel links, resolved sample rate, single-pass chain stabilization).
    Sequential(Box<sequential::SequentialModel>),
}

impl sealed::Sealed for StaticModel {}

pub(crate) use static_model::clone_condition_dsp;

#[cfg(test)]
#[path = "nam_model_measure_test.rs"]
mod nam_model_measure_test;
