// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Resampling engine for one direction (input or output).
//!
//! Contains the polyphase bank, delay lines, and fractional phase state
//! for tracking sample-rate conversion in a single direction.

use crate::common::diagnostics::NamErrorCode;
#[cfg(feature = "avx512")]
use crate::math::common::Avx512Math;
use crate::math::common::{Avx2Math, InstructionSet, SimdMath, effective_instruction_set};

use super::super::sinc_kernel::{NUM_PHASES, PolyphaseBank};
use super::delay_line::DelayLine;

/// Progress returned by a single resampler processing call.
///
/// - `samples_read`: number of input samples consumed (in each of L/R).
/// - `samples_written`: number of output samples produced (in each of L/R).
///
/// When the output buffer fills before all input is consumed, `samples_read` may
/// be less than the input length. Unconsumed input is NOT pushed to the delay lines
/// and the resampler state (`phase_accum`, delay lines) remains exactly at the
/// point of suspension — ready for the next call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResamplerProgress {
    /// Number of input samples consumed (per L/R channel).
    pub samples_read: usize,
    /// Number of output samples produced (per L/R channel).
    pub samples_written: usize,
}

/// Resampling engine for one direction (input or output).
///
/// Contains the polyphase bank and the fractional state for tracking
/// the position between input and output samples.
pub(crate) struct ResamplerCore {
    /// Polyphase filter bank (32B-aligned coefficients).
    pub bank: PolyphaseBank,
    /// Left channel delay line.
    pub state_l: DelayLine,
    /// Right channel delay line.
    pub state_r: DelayLine,
    /// Fractional position in phase space (0.0 .. NUM_PHASES) represented in 24.40 fixed-point.
    ///
    /// Upper 24 bits index the polyphase filter bank phase (`0..NUM_PHASES`), while the lower 40 bits
    /// hold the sub-phase fractional offset used for linear/cubic phase interpolation.
    /// Advances by `phase_step` on each output sample.
    pub phase_accum: u64,
    /// Phase increment per output sample in 24.40 fixed-point (`from_rate / to_rate * NUM_PHASES << 40`).
    ///
    /// Guarantees phase step precision down to ~9e-13 phase units per sample without floating-point
    /// accumulation drift over arbitrarily long stream renders.
    pub phase_step: u64,
    /// Empirical group delay in output-rate samples (from source bank).
    pub group_delay: f64,
}

impl ResamplerCore {
    pub fn new(from_rate: u32, to_rate: u32, bank: PolyphaseBank) -> Result<Self, NamErrorCode> {
        let group_delay = bank.group_delay;

        let phase_step_f = (from_rate as f64 / to_rate as f64) * NUM_PHASES as f64;
        let phase_step = (phase_step_f * ((1u64 << 40) as f64)).round() as u64;

        Ok(Self {
            bank,
            state_l: DelayLine::new()?,
            state_r: DelayLine::new()?,
            phase_accum: (NUM_PHASES as u64) << 40,
            phase_step,
            group_delay,
        })
    }

    /// Returns the empirical group delay in output-rate samples.
    #[inline]
    pub fn group_delay(&self) -> f64 {
        self.group_delay
    }

    /// Processes a stereo block. RT-safe: zero allocations.
    ///
    /// Returns `ResamplerProgress` with samples read and written.
    /// When the output buffer fills, processing stops immediately
    /// without consuming remaining input or mutating state.
    fn process_internal<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let n_in = in_l.len().min(in_r.len());
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            // ── Step 1: Accumulate phase. When it rolls past NUM_PHASES, consume
            //    one input sample and push it into both delay lines. ──
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return ResamplerProgress {
                        samples_read: in_idx,
                        samples_written: out_idx,
                    };
                }
                // SAFETY: the `in_idx >= n_in` early return above guarantees
                // `in_idx < n_in = min(in_l.len(), in_r.len())`, so both
                // `get_unchecked` reads are in bounds.
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                    self.state_r.push(*in_r.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            // ── Step 2: Extract integer phase index and fractional sub-phase ──
            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            // Convert 40-bit fraction to f32 in [0, 1) range
            let frac = ((frac_bits >> 9) as i32 as f32) * (1.0 / (1u32 << 31) as f32);

            // ── Step 3: Determine next phase for linear interpolation ──
            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            // ── Step 4: Convolve with current and next polyphase banks, then
            //    linearly interpolate between the two results for smooth output ──
            // SAFETY: `phase_idx < NUM_PHASES` (upper 24 bits of `phase_accum`,
            // maintained < NUM_PHASES by the loop), so both `phase_ptr` calls are
            // valid for `taps_per_phase` coeffs; `window_ptr()` returns a pointer
            // to `taps_per_phase` contiguous samples in the double-buffer delay
            // line; `M` is selected by ISA dispatch matching its target features.
            let (y_l, y_r) = unsafe {
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let x_r = self.state_r.window_ptr();
                let taps = self.bank.taps_per_phase;

                let ((y0_l, y0_r), (y1_l, y1_r)) = M::convolve_stereo_dual(c0, c1, x_l, x_r, taps);
                // Linear interpolation: lerp(phase0, phase1, frac)
                (y0_l + frac * (y1_l - y0_l), y0_r + frac * (y1_r - y0_r))
            };

            // ── Step 5: Write interpolated output and advance phase ──
            // SAFETY: `out_idx < n_out_max = min(out_l.len(), out_r.len())`
            // (while condition above), so both writes are in bounds.
            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_r;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        ResamplerProgress {
            samples_read: in_idx,
            samples_written: out_idx,
        }
    }

    /// Processes a mono block. RT-safe: zero allocations.
    ///
    /// Returns `ResamplerProgress` with samples read and written.
    /// When the output buffer fills, processing stops immediately
    /// without consuming remaining input or mutating state.
    fn process_internal_mono<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        let n_in = in_l.len();
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            // ── Step 1: Accumulate phase; push mono input into left delay line when phase rolls ──
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return ResamplerProgress {
                        samples_read: in_idx,
                        samples_written: out_idx,
                    };
                }
                // SAFETY: the `in_idx >= n_in` early return above guarantees
                // `in_idx < n_in = in_l.len()` on the mono path.
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            // ── Step 2: Extract phase index and fractional sub-phase ──
            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            let frac = ((frac_bits >> 9) as i32 as f32) * (1.0 / (1u32 << 31) as f32);

            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            // ── Step 3: Dual-phase convolution + linear interpolation (mono) ──
            // SAFETY: `phase_idx < NUM_PHASES` (loop invariant), so both `phase_ptr`
            // calls are valid for `taps_per_phase` coeffs; `window_ptr()` points to
            // `taps_per_phase` contiguous samples; `M` is ISA-dispatched.
            let y_l = unsafe {
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let taps = self.bank.taps_per_phase;

                let (y0_l, y1_l) = M::convolve_mono_dual(c0, c1, x_l, taps);
                y0_l + frac * (y1_l - y0_l)
            };

            // ── Step 4: Write mono output to both channels and advance phase ──
            // SAFETY: `out_idx < n_out_max = min(out_l.len(), out_r.len())`
            // (while condition above), so both writes are in bounds.
            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_l;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        ResamplerProgress {
            samples_read: in_idx,
            samples_written: out_idx,
        }
    }

    /// Performs ISA-dispatched static stereo resampling.
    #[inline(always)]
    pub(crate) fn process_static_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        #[expect(deprecated)]
        match effective_instruction_set() {
            #[cfg(feature = "avx512")]
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                self.process_internal::<Avx512Math>(in_l, in_r, out_l, out_r)
            }
            #[cfg(not(feature = "avx512"))]
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                self.process_internal::<Avx2Math>(in_l, in_r, out_l, out_r)
            }
            InstructionSet::Avx2 => self.process_internal::<Avx2Math>(in_l, in_r, out_l, out_r),
        }
    }

    /// Performs ISA-dispatched static mono resampling.
    #[inline(always)]
    pub(crate) fn process_static_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> ResamplerProgress {
        #[expect(deprecated)]
        match effective_instruction_set() {
            #[cfg(feature = "avx512")]
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                self.process_internal_mono::<Avx512Math>(in_l, out_l, out_r)
            }
            #[cfg(not(feature = "avx512"))]
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                self.process_internal_mono::<Avx2Math>(in_l, out_l, out_r)
            }
            InstructionSet::Avx2 => self.process_internal_mono::<Avx2Math>(in_l, out_l, out_r),
        }
    }
}
