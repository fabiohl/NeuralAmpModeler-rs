// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Native Minimum-Phase Polyphase FIR Sinc Resampler with AVX2+FMA SIMD convolution.
//!
//! Implements `NamResampler`, an RT-safe sample rate conversion engine
//! that replaces the `rubato` crate with a custom Polyphase Sinc FIR filter.
//!
//! ## Advantages over rubato (linear phase)
//!
//! - **Pre-ringing elimination**: the minimum-phase transform via Real Cepstrum
//!   concentrates all filter energy into the shortest possible delay, removing 100%
//!   of the pre-echo artifacts on guitar transients.
//! - **Algorithmic latency reduction**: ~50% less latency than equivalent linear-phase.
//! - **Vectorized convolution**: AVX2+FMA inner product with coefficients aligned
//!   to 64 bytes, saturating the processor's FMA port throughput.
//!
//! ## Architecture: Polyphase Oversampled with Interpolation
//!
//! Instead of using discrete L/M phases (impractical for L=160 at ratio 44.1→48),
//! the resampler uses an overabundant bank of 256 phases with linear interpolation
//! between adjacent phases. This yields arbitrary conversion ratios with
//! measured passband ripple < 0.05 dB and stopband ≥ 100 dB.
//!
//! ## Quality Mode
//!
//! `NamResampler::new()` produces the production default: minimum-phase polyphase
//! with TAPS_PER_PHASE (currently 64). `NamResampler::new_linear()` produces a
//! linear-phase variant for offline/mixdown use. Both are RT-safe and zero-alloc
//! in the hot path after construction.
//!
//! ## Real-Time Guarantees
//!
//! All allocation happens in `NamResampler::new()` / `new_linear()`, outside the DSP thread.
//! In the RT callback, only `process_input()` / `process_output()` are invoked —
//! zero-alloc operations that manipulate pre-allocated ring buffers.

use anyhow::{Result, bail};
use log::info;
use std::ptr;

use crate::common::diagnostics::NamErrorCode;

/// Minimum sample rate to guard against catastrophic upsampling (4 kHz).
const MIN_RATE: u32 = 4_000;

/// Maximum sample rate for stability and reasonable mem usage (384 kHz).
const MAX_RATE: u32 = 384_000;

use super::sinc_kernel::{generate_polyphase_bank, generate_polyphase_bank_linear};

mod core;
mod delay_line;

use core::ResamplerCore;

pub use core::ResamplerProgress;

enum PhaseType {
    Minimum,
    Linear,
}

/// RT-safe wrapper for bidirectional Minimum-Phase Polyphase Sinc FIR resampling.
///
/// Encapsulates two independent pre-allocated engines (input + output).
/// In the DSP thread only `process_input()` / `process_output()` are called —
/// zero-alloc operations that work on pre-allocated delay lines.
///
/// When `host_rate == nam_rate`, both engines are bypassed (`None`)
/// and the hot path passes through with zero overhead.
///
/// # Examples
///
/// ```no_run
/// use neural_amp_modeler_rs::dsp::resampler::NamResampler;
///
/// let host_rate = 44_100;
/// let nam_rate = 48_000;
/// let resampler = NamResampler::new_simple(host_rate, nam_rate).expect("valid sample rates");
///
/// let latency = resampler.latency_samples(host_rate);
/// assert!(latency > 0);
/// ```
pub struct NamResampler {
    /// Input engine: `host_rate → nam_rate`. `None` = bypass.
    inner: Option<ResamplerCore>,
    /// Output engine: `nam_rate → host_rate`. `None` = bypass.
    outer: Option<ResamplerCore>,
    /// Host sample rate.
    host_rate: u32,
    /// Target NAM model rate.
    nam_rate: u32,
}

impl NamResampler {
    /// Validates the sample-rate window shared by the `anyhow` and typed
    /// constructors.
    ///
    /// # Errors
    ///
    /// Returns [`NamErrorCode::ResamplerBuildFailed`] (E2200) when either
    /// `host_rate` or `nam_rate` falls outside the supported
    /// `MIN_RATE..=MAX_RATE` window. In this constructor that code is unique
    /// to rate validation: allocation failures are reported separately as
    /// [`NamErrorCode::OutOfMemory`] (E5000).
    #[inline]
    fn validate_rates(host_rate: u32, nam_rate: u32) -> Result<(), NamErrorCode> {
        if (MIN_RATE..=MAX_RATE).contains(&host_rate) && (MIN_RATE..=MAX_RATE).contains(&nam_rate) {
            Ok(())
        } else {
            Err(NamErrorCode::ResamplerBuildFailed)
        }
    }

    #[cold]
    fn new_inner(host_rate: u32, nam_rate: u32, phase: PhaseType) -> Result<Self> {
        if Self::validate_rates(host_rate, nam_rate).is_err() {
            bail!(
                "NamResampler: sample rates must be in range {}-{}, got host={} nam={}",
                MIN_RATE,
                MAX_RATE,
                host_rate,
                nam_rate
            );
        }
        Self::build_inner(host_rate, nam_rate, phase).map_err(Into::into)
    }

    #[cold]
    fn new_typed_inner(
        host_rate: u32,
        nam_rate: u32,
        phase: PhaseType,
    ) -> Result<Self, NamErrorCode> {
        Self::validate_rates(host_rate, nam_rate)?;
        Self::build_inner(host_rate, nam_rate, phase)
    }

    /// Builds the resampler pair after rate validation has passed.
    ///
    /// Every fallible step here is already typed: `generate_polyphase_bank`
    /// and `ResamplerCore::new` report allocation failure as
    /// [`NamErrorCode::OutOfMemory`].
    #[cold]
    fn build_inner(host_rate: u32, nam_rate: u32, phase: PhaseType) -> Result<Self, NamErrorCode> {
        if host_rate == nam_rate {
            let label = match phase {
                PhaseType::Minimum => "Bypass",
                PhaseType::Linear => "Linear-phase bypass",
            };
            info!(
                "[Resampler] {label}: host_rate={}, nam_rate={} (match)",
                host_rate, nam_rate
            );
            return Ok(Self {
                inner: None,
                outer: None,
                host_rate,
                nam_rate,
            });
        }

        let gen_bank = match phase {
            PhaseType::Minimum => generate_polyphase_bank,
            PhaseType::Linear => generate_polyphase_bank_linear,
        };

        let inner = ResamplerCore::new(host_rate, nam_rate, gen_bank(host_rate, nam_rate)?)?;
        let outer = ResamplerCore::new(nam_rate, host_rate, gen_bank(nam_rate, host_rate)?)?;

        let label = match phase {
            PhaseType::Minimum => "Minimum-phase",
            PhaseType::Linear => "Linear-phase",
        };
        info!(
            "[Resampler] {label} resampler built: host_rate={}, nam_rate={}",
            host_rate, nam_rate
        );

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            host_rate,
            nam_rate,
        })
    }
    /// Creates the pair of resamplers (input+output), pre-allocating all buffers.
    ///
    /// Produces a **minimum-phase** polyphase resampler — the production default.
    /// Minimum-phase eliminates pre-ringing at the cost of non-linear phase response,
    /// ideal for live monitoring where transient fidelity dominates.
    ///
    /// If `host_rate == nam_rate`, full bypass with no overhead.
    ///
    /// # Parameters
    /// - `host_rate`: Host sample rate (e.g., 44100, 48000, 96000).
    /// - `nam_rate`: NAM model rate (e.g., 48000).
    /// - `_chunk_size`: kept for API compatibility (not used internally).
    ///
    /// # Errors
    ///
    /// Returns an error if `host_rate` or `nam_rate` is outside the supported
    /// range `4_000..=384_000` Hz, or if allocation of the polyphase filter
    /// banks or internal delay lines fails (out of memory).
    ///
    /// For a typed-error counterpart (no `anyhow`), see
    /// [`new_typed`](NamResampler::new_typed).
    #[cold]
    pub fn new(host_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        Self::new_inner(host_rate, nam_rate, PhaseType::Minimum)
    }

    /// Creates the pair of resamplers (input+output) without the unused chunk-size parameter.
    ///
    /// Equivalent to [`new`](NamResampler::new)`(host_rate, nam_rate, 0)`.
    /// See [`new`](NamResampler::new) for the full parameter and error documentation.
    #[cold]
    pub fn new_simple(host_rate: u32, nam_rate: u32) -> Result<Self> {
        Self::new(host_rate, nam_rate, 0)
    }

    /// Creates the pair of resamplers using **linear-phase** polyphase banks.
    ///
    /// Linear-phase preserves perfect phase linearity at the cost of pre-ringing —
    /// suitable for offline rendering and mixdown where latency is irrelevant and
    /// phase accuracy is paramount.
    ///
    /// If `host_rate == nam_rate`, full bypass with no overhead.
    ///
    /// # Errors
    ///
    /// Returns an error if `host_rate` or `nam_rate` is outside the supported
    /// range `4_000..=384_000` Hz, or if allocation of the polyphase filter
    /// banks or internal delay lines fails (out of memory).
    ///
    /// For a typed-error counterpart (no `anyhow`), see
    /// [`new_linear_typed`](NamResampler::new_linear_typed).
    #[cold]
    pub fn new_linear(host_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        Self::new_inner(host_rate, nam_rate, PhaseType::Linear)
    }

    /// Creates the linear-phase pair of resamplers without the unused chunk-size parameter.
    ///
    /// Equivalent to [`new_linear`](NamResampler::new_linear)`(host_rate, nam_rate, 0)`.
    /// See [`new_linear`](NamResampler::new_linear) for the full parameter and error documentation.
    #[cold]
    pub fn new_linear_simple(host_rate: u32, nam_rate: u32) -> Result<Self> {
        Self::new_linear(host_rate, nam_rate, 0)
    }

    /// Typed-error counterpart of [`new`](NamResampler::new) (minimum-phase).
    ///
    /// Identical construction and semantics, but failures are reported as a
    /// structured [`NamErrorCode`] instead of an opaque `anyhow` error, so a
    /// consumer can triage the failure without a downcast. The `anyhow`
    /// constructors remain available; deprecated/replacement is deferred to
    /// the breaking release (Épico E / E4).
    ///
    /// # Errors
    ///
    /// - [`NamErrorCode::ResamplerBuildFailed`] (E2200) when `host_rate` or
    ///   `nam_rate` is outside `4_000..=384_000` Hz.
    /// - [`NamErrorCode::OutOfMemory`] (E5000) when allocation of the
    ///   polyphase filter banks or internal delay lines fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    /// use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
    ///
    /// assert!(NamResampler::new_typed(44_100, 48_000, 0).is_ok());
    /// assert_eq!(
    ///     NamResampler::new_typed(1_000, 48_000, 0).err(),
    ///     Some(NamErrorCode::ResamplerBuildFailed),
    /// );
    /// ```
    #[cold]
    pub fn new_typed(
        host_rate: u32,
        nam_rate: u32,
        _chunk_size: usize,
    ) -> Result<Self, NamErrorCode> {
        Self::new_typed_inner(host_rate, nam_rate, PhaseType::Minimum)
    }

    /// Typed-error counterpart of [`new_simple`](NamResampler::new_simple).
    ///
    /// Equivalent to [`new_typed`](NamResampler::new_typed)`(host_rate, nam_rate, 0)`.
    #[cold]
    pub fn new_simple_typed(host_rate: u32, nam_rate: u32) -> Result<Self, NamErrorCode> {
        Self::new_typed(host_rate, nam_rate, 0)
    }

    /// Typed-error counterpart of [`new_linear`](NamResampler::new_linear)
    /// (linear-phase).
    ///
    /// # Errors
    ///
    /// Same as [`new_typed`](NamResampler::new_typed):
    /// [`NamErrorCode::ResamplerBuildFailed`] (E2200) for an out-of-range
    /// sample rate, [`NamErrorCode::OutOfMemory`] (E5000) on allocation
    /// failure.
    #[cold]
    pub fn new_linear_typed(
        host_rate: u32,
        nam_rate: u32,
        _chunk_size: usize,
    ) -> Result<Self, NamErrorCode> {
        Self::new_typed_inner(host_rate, nam_rate, PhaseType::Linear)
    }

    /// Typed-error counterpart of
    /// [`new_linear_simple`](NamResampler::new_linear_simple).
    ///
    /// Equivalent to
    /// [`new_linear_typed`](NamResampler::new_linear_typed)`(host_rate, nam_rate, 0)`.
    #[cold]
    pub fn new_linear_simple_typed(host_rate: u32, nam_rate: u32) -> Result<Self, NamErrorCode> {
        Self::new_linear_typed(host_rate, nam_rate, 0)
    }

    /// Returns `true` when `host_rate == nam_rate` (bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
    }

    /// Resets both resampler engines (input + output) to their post-construction
    /// state: phase accumulators and delay lines are cleared.
    ///
    /// RT-safe: zero allocations. Useful for stream resets and resource-swap
    /// protocols where pending filter state must be discarded.
    #[inline]
    pub fn reset(&mut self) {
        if let Some(ref mut core) = self.inner {
            core.reset_state();
        }
        if let Some(ref mut core) = self.outer {
            core.reset_state();
        }
    }

    /// Returns the host sample rate.
    #[inline]
    pub fn host_rate(&self) -> u32 {
        self.host_rate
    }

    /// Returns the NAM model rate.
    #[inline]
    pub fn nam_rate(&self) -> u32 {
        self.nam_rate
    }

    /// Computes the total latency (input + output) in host-rate samples.
    ///
    /// Uses the empirical group delay from the polyphase bank:
    /// - Linear-phase: exactly `TAPS_PER_PHASE / 2` per stage.
    /// - Minimum-phase: energy centroid of the prototype divided by NUM_PHASES.
    ///
    /// The output-stage delay is rate-converted to host-rate samples.
    ///
    /// # Parameters
    /// - `_host_rate`: Host sample rate (ignored in favor of the configured `self.host_rate`;
    ///   retained for backward compatibility and scheduled for removal in v0.8 / Sprint 5).
    ///
    /// # Returns
    /// Total latency in samples at `self.host_rate()`.
    pub fn latency_samples(&self, _host_rate: u32) -> u32 {
        if self.is_bypass() {
            return 0;
        }

        let delay_in = match self.inner {
            Some(ref core) => core.group_delay(),
            None => 0.0,
        };
        let delay_out = match self.outer {
            Some(ref core) => core.group_delay() * (self.host_rate as f64 / self.nam_rate as f64),
            None => 0.0,
        };
        (delay_in + delay_out).round() as u32
    }

    /// Computes the minimum guaranteed output buffer size that accommodates
    /// all output samples produced when consuming `input_samples` at the
    /// given sample rates, including worst-case fractional phase offset
    /// and tail-phase drain.
    ///
    /// Uses integer arithmetic only (`checked_mul`, `div_ceil`) with zero
    /// heap allocations. Overflow saturates at `usize::MAX`.
    ///
    /// Consumer-neutral capacity helper (host-agnostic).
    ///
    /// # Parameters
    /// - `input_samples`: number of input samples to be processed.
    /// - `in_rate`: input sample rate (Hz).
    /// - `out_rate`: output sample rate (Hz).
    #[inline]
    pub fn min_output_samples(input_samples: usize, in_rate: u32, out_rate: u32) -> usize {
        let numer = if let Some(v) = (input_samples as u64).checked_mul(out_rate as u64) {
            v
        } else {
            return usize::MAX;
        };
        let denom = in_rate as u64;
        let min = numer.div_ceil(denom);
        if min > usize::MAX as u64 {
            usize::MAX
        } else {
            min as usize
        }
    }

    /// Computes the maximum number of input samples that can be safely
    /// processed without exceeding `output_capacity` output buffer samples,
    /// given the sample rates.
    ///
    /// Uses integer arithmetic only (`checked_mul`, `div_ceil`) with zero
    /// heap allocations. Overflow saturates at `usize::MAX`.
    ///
    /// Consumer-neutral capacity helper (host-agnostic).
    ///
    /// # Parameters
    /// - `output_capacity`: available output buffer size (samples).
    /// - `in_rate`: input sample rate (Hz).
    /// - `out_rate`: output sample rate (Hz).
    #[inline]
    pub fn max_input_samples(output_capacity: usize, in_rate: u32, out_rate: u32) -> usize {
        let numer = if let Some(v) = (output_capacity as u64).checked_mul(in_rate as u64) {
            v
        } else {
            return usize::MAX;
        };
        let denom = out_rate as u64;
        let max = numer / denom;
        if max > usize::MAX as u64 {
            usize::MAX
        } else {
            max as usize
        }
    }

    /// **Input resampling** (input path): `host_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// **Non-overlap precondition (R-9 / A9):** `in_*` and `out_*` slices must
    /// not overlap. The bypass path uses `copy_nonoverlapping`, which requires
    /// disjoint source/destination regions. Safe Rust guarantees this between
    /// `&[f32]` inputs and distinct `&mut [f32]` outputs; callers that reach the
    /// buffers through raw pointers (FFI/host) must uphold it themselves.
    pub fn process_input(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            // SAFETY: `n` is the minimum length of the source and destination slices,
            // so each `copy_nonoverlapping` moves `n` initialized `f32` values within
            // bounds; in/out buffers are distinct (bypass path), so no overlap.
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return ResamplerProgress {
                samples_read: n,
                samples_written: n,
            };
        };
        core.process_static_stereo(in_l, in_r, out_l, out_r)
    }

    /// **Output resampling** (output path): `nam_rate → host_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// **Non-overlap precondition (R-9 / A9):** `in_*` and `out_*` slices must
    /// not overlap. The bypass path uses `copy_nonoverlapping`, which requires
    /// disjoint source/destination regions. Safe Rust guarantees this between
    /// `&[f32]` inputs and distinct `&mut [f32]` outputs; callers that reach the
    /// buffers through raw pointers (FFI/host) must uphold it themselves.
    pub fn process_output(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            // SAFETY: `n` is the minimum length of source and destination slices, so
            // each `copy_nonoverlapping` copies `n` `f32` values within bounds; the
            // in/out buffers are distinct on the bypass path (no overlap).
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return ResamplerProgress {
                samples_read: n,
                samples_written: n,
            };
        };
        core.process_static_stereo(in_l, in_r, out_l, out_r)
    }

    /// **Mono input resampling** (input path): `host_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// **Non-overlap precondition (R-9 / A9):** `in_l`, `out_l` and `out_r`
    /// must not overlap. The bypass path uses `copy_nonoverlapping`, which
    /// requires disjoint source/destination regions. Safe Rust guarantees this
    /// between a `&[f32]` input and distinct `&mut [f32]` outputs; callers that
    /// reach the buffers through raw pointers (FFI/host) must uphold it
    /// themselves.
    pub fn process_input_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            // SAFETY: `n` is the minimum of the source and both destination lengths,
            // so both copies stay in bounds; `out_l`/`out_r` are distinct buffers and
            // distinct from `in_l`, so the regions never overlap.
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return ResamplerProgress {
                samples_read: n,
                samples_written: n,
            };
        };
        core.process_static_mono(in_l, out_l, out_r)
    }

    /// **Mono output resampling** (output path): `nam_rate → host_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// **Non-overlap precondition (R-9 / A9):** `in_l`, `out_l` and `out_r`
    /// must not overlap. The bypass path uses `copy_nonoverlapping`, which
    /// requires disjoint source/destination regions. Safe Rust guarantees this
    /// between a `&[f32]` input and distinct `&mut [f32]` outputs; callers that
    /// reach the buffers through raw pointers (FFI/host) must uphold it
    /// themselves.
    pub fn process_output_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            // SAFETY: `n` is the minimum of the source and both destination lengths,
            // so both copies stay in bounds; `out_l`/`out_r` are distinct buffers and
            // distinct from `in_l`, so the regions never overlap.
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return ResamplerProgress {
                samples_read: n,
                samples_written: n,
            };
        };
        core.process_static_mono(in_l, out_l, out_r)
    }
}

#[cfg(test)]
#[path = "../resampler_test.rs"]
mod resampler_test;
