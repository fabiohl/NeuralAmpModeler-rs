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

/// Minimum sample rate to guard against catastrophic upsampling (4 kHz).
const MIN_RATE: u32 = 4_000;

/// Maximum sample rate for stability and reasonable mem usage (384 kHz).
const MAX_RATE: u32 = 384_000;

use super::sinc_kernel::{generate_polyphase_bank, generate_polyphase_bank_linear};

mod core;
mod delay_line;

use core::ResamplerCore;

/// RT-safe wrapper for bidirectional Minimum-Phase Polyphase Sinc FIR resampling.
///
/// Encapsulates two independent pre-allocated engines (input + output).
/// In the DSP thread only `process_input()` / `process_output()` are called —
/// zero-alloc operations that work on pre-allocated delay lines.
///
/// When `host_rate == nam_rate`, both engines are bypassed (`None`)
/// and the hot path passes through with zero overhead.
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
    #[cold]
    pub fn new(host_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        if !(MIN_RATE..=MAX_RATE).contains(&host_rate) || !(MIN_RATE..=MAX_RATE).contains(&nam_rate)
        {
            bail!(
                "NamResampler: sample rates must be in range {}-{}, got host={} nam={}",
                MIN_RATE,
                MAX_RATE,
                host_rate,
                nam_rate
            );
        }

        if host_rate == nam_rate {
            info!(
                "[Resampler] Bypass: host_rate={}, nam_rate={} (match)",
                host_rate, nam_rate
            );
            return Ok(Self {
                inner: None,
                outer: None,
                host_rate,
                nam_rate,
            });
        }

        let inner = ResamplerCore::new(
            host_rate,
            nam_rate,
            generate_polyphase_bank(host_rate, nam_rate).map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        let outer = ResamplerCore::new(
            nam_rate,
            host_rate,
            generate_polyphase_bank(nam_rate, host_rate).map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;

        info!(
            "[Resampler] Minimum-phase resampler built: host_rate={}, nam_rate={}",
            host_rate, nam_rate
        );

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            host_rate,
            nam_rate,
        })
    }

    /// Creates the pair of resamplers using **linear-phase** polyphase banks.
    ///
    /// Linear-phase preserves perfect phase linearity at the cost of pre-ringing —
    /// suitable for offline rendering and mixdown where latency is irrelevant and
    /// phase accuracy is paramount.
    ///
    /// If `host_rate == nam_rate`, full bypass with no overhead.
    #[cold]
    pub fn new_linear(host_rate: u32, nam_rate: u32, _chunk_size: usize) -> Result<Self> {
        if !(MIN_RATE..=MAX_RATE).contains(&host_rate) || !(MIN_RATE..=MAX_RATE).contains(&nam_rate)
        {
            bail!(
                "NamResampler: sample rates must be in range {}-{}, got host={} nam={}",
                MIN_RATE,
                MAX_RATE,
                host_rate,
                nam_rate
            );
        }

        if host_rate == nam_rate {
            info!(
                "[Resampler] Linear-phase bypass: host_rate={}, nam_rate={} (match)",
                host_rate, nam_rate
            );
            return Ok(Self {
                inner: None,
                outer: None,
                host_rate,
                nam_rate,
            });
        }

        let inner = ResamplerCore::new(
            host_rate,
            nam_rate,
            generate_polyphase_bank_linear(host_rate, nam_rate)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;
        let outer = ResamplerCore::new(
            nam_rate,
            host_rate,
            generate_polyphase_bank_linear(nam_rate, host_rate)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        )?;

        info!(
            "[Resampler] Linear-phase resampler built: host_rate={}, nam_rate={}",
            host_rate, nam_rate
        );

        Ok(Self {
            inner: Some(inner),
            outer: Some(outer),
            host_rate,
            nam_rate,
        })
    }

    /// Returns `true` when `host_rate == nam_rate` (bypass).
    #[inline]
    pub fn is_bypass(&self) -> bool {
        self.inner.is_none()
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
    /// - `host_rate`: host sample rate (e.g., 44100, 48000, 96000).
    ///
    /// # Returns
    /// Total latency in samples at `host_rate`.
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

    /// **Input resampling** (input path): `host_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_input(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        core.process_static_stereo(in_l, in_r, out_l, out_r)
    }

    /// **Output resampling** (output path): `nam_rate → host_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_output(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(in_r.len()).min(out_l.len()).min(out_r.len());
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_r.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        core.process_static_stereo(in_l, in_r, out_l, out_r)
    }

    /// **Mono input resampling** (input path): `host_rate → nam_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_input_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.inner else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        core.process_static_mono(in_l, out_l, out_r)
    }

    /// **Mono output resampling** (output path): `nam_rate → host_rate`.
    ///
    /// RT-safe: zero allocations. On bypass, copies directly.
    ///
    /// # Returns
    /// Number of samples written to `out_l` / `out_r`.
    pub fn process_output_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let Some(ref mut core) = self.outer else {
            let n = in_l.len().min(out_l.len()).min(out_r.len());
            unsafe {
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_l.as_mut_ptr(), n);
                ptr::copy_nonoverlapping(in_l.as_ptr(), out_r.as_mut_ptr(), n);
            }
            return n;
        };
        core.process_static_mono(in_l, out_l, out_r)
    }
}

#[cfg(test)]
#[path = "../resampler_test.rs"]
mod resampler_test;
