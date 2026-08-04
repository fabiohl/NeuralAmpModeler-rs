// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Resampling engine for one direction (input or output).
//!
//! Contains the polyphase bank, delay lines, and fractional phase state
//! for tracking sample-rate conversion in a single direction.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::{
    Avx2Math, Avx512Math, Avx512VnniBf16Math, InstructionSet, SimdMath, effective_instruction_set,
};

use super::super::sinc_kernel::{NUM_PHASES, PolyphaseBank};
use super::delay_line::DelayLine;

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
    /// Returns the number of samples written to `out_l` / `out_r`.
    fn process_internal<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let n_in = in_l.len().min(in_r.len());
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return out_idx;
                }
                // SAFETY: `in_idx < n_in` is checked immediately above
                // on L78. The while-loop condition ensures we only
                // enter the body when `phase_accum >= num_phases_fp`,
                // and `in_idx` is only incremented after push.
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                    self.state_r.push(*in_r.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            let frac = ((frac_bits >> 9) as i32 as f32) * (1.0 / (1u32 << 31) as f32);

            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            let (y_l, y_r) = unsafe {
                // SAFETY: `phase_idx` and `phase_next` are in [0, NUM_PHASES)
                // (guarded by the modulo on L95-99). `self.state_l.window_ptr()`
                // and `self.state_r.window_ptr()` return pointers to internal
                // delay-line buffers whose length ≥ `taps`, validated at
                // construction time. `phase_ptr` returns a pointer into the
                // pre-allocated polyphase coefficient bank (length
                // `NUM_PHASES * taps_per_phase`).
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let x_r = self.state_r.window_ptr();
                let taps = self.bank.taps_per_phase;

                let ((y0_l, y0_r), (y1_l, y1_r)) = M::convolve_stereo_dual(c0, c1, x_l, x_r, taps);
                (y0_l + frac * (y1_l - y0_l), y0_r + frac * (y1_r - y0_r))
            };

            // SAFETY: `out_idx < n_out_max` is checked by the outer
            // while-loop on L77. Both `out_l` and `out_r` have length
            // ≥ `n_out_max` (L71). No other mutable access to these
            // buffers exists during the loop.
            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_r;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        while self.phase_accum >= num_phases_fp && in_idx < n_in {
            // SAFETY: `in_idx < n_in` is checked in the while-loop
            // condition. The loop body executes at most `n_in - in_idx`
            // times, draining the remaining samples.
            unsafe {
                self.state_l.push(*in_l.get_unchecked(in_idx));
                self.state_r.push(*in_r.get_unchecked(in_idx));
            }
            self.phase_accum -= num_phases_fp;
            in_idx += 1;
        }

        out_idx
    }

    /// Processes a mono block. RT-safe: zero allocations.
    ///
    /// Returns the number of samples written to `out_l` / `out_r`.
    fn process_internal_mono<M: SimdMath>(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        let n_in = in_l.len();
        let n_out_max = out_l.len().min(out_r.len());
        let mut in_idx = 0usize;
        let mut out_idx = 0usize;

        let num_phases_fp = (NUM_PHASES as u64) << 40;

        while out_idx < n_out_max {
            while self.phase_accum >= num_phases_fp {
                if in_idx >= n_in {
                    return out_idx;
                }
                // SAFETY: `in_idx < n_in` is checked immediately above.
                // The loop body only executes after the while condition
                // is satisfied, and `in_idx` is incremented after each push.
                unsafe {
                    self.state_l.push(*in_l.get_unchecked(in_idx));
                }
                self.phase_accum -= num_phases_fp;
                in_idx += 1;
            }

            let phase_idx = (self.phase_accum >> 40) as usize;
            const FRAC_MASK: u64 = (1u64 << 40) - 1;
            let frac_bits = self.phase_accum & FRAC_MASK;
            let frac = ((frac_bits >> 9) as i32 as f32) * (1.0 / (1u32 << 31) as f32);

            let phase_next = if phase_idx + 1 >= NUM_PHASES {
                0
            } else {
                phase_idx + 1
            };

            let y_l = unsafe {
                // SAFETY: `phase_idx` and `phase_next` are in [0, NUM_PHASES).
                // `window_ptr` and `phase_ptr` return pointers to
                // pre-allocated, valid buffers whose length ≥ `taps`.
                let c0 = self.bank.phase_ptr(phase_idx);
                let c1 = self.bank.phase_ptr(phase_next);
                let x_l = self.state_l.window_ptr();
                let taps = self.bank.taps_per_phase;

                let (y0_l, y1_l) = M::convolve_mono_dual(c0, c1, x_l, taps);
                y0_l + frac * (y1_l - y0_l)
            };

            // SAFETY: `out_idx < n_out_max` is checked by the outer
            // while-loop on L149. Both `out_l` and `out_r` have length
            // ≥ `n_out_max` (L143). Mono path writes the same sample
            // to both channels.
            unsafe {
                *out_l.get_unchecked_mut(out_idx) = y_l;
                *out_r.get_unchecked_mut(out_idx) = y_l;
            }
            out_idx += 1;

            self.phase_accum += self.phase_step;
        }

        while self.phase_accum >= num_phases_fp && in_idx < n_in {
            // SAFETY: `in_idx < n_in` is checked in the while-loop
            // condition. Mono path only pushes to the left channel.
            unsafe {
                self.state_l.push(*in_l.get_unchecked(in_idx));
            }
            self.phase_accum -= num_phases_fp;
            in_idx += 1;
        }

        out_idx
    }

    /// Performs ISA-dispatched static stereo resampling.
    #[inline(always)]
    pub(crate) fn process_static_stereo(
        &mut self,
        in_l: &[f32],
        in_r: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        match effective_instruction_set() {
            InstructionSet::Avx2 => self.process_internal::<Avx2Math>(in_l, in_r, out_l, out_r),
            InstructionSet::Avx512 => self.process_internal::<Avx512Math>(in_l, in_r, out_l, out_r),
            InstructionSet::Avx512VnniBf16 => {
                self.process_internal::<Avx512VnniBf16Math>(in_l, in_r, out_l, out_r)
            }
        }
    }

    /// Performs ISA-dispatched static mono resampling.
    #[inline(always)]
    pub(crate) fn process_static_mono(
        &mut self,
        in_l: &[f32],
        out_l: &mut [f32],
        out_r: &mut [f32],
    ) -> usize {
        match effective_instruction_set() {
            InstructionSet::Avx2 => self.process_internal_mono::<Avx2Math>(in_l, out_l, out_r),
            InstructionSet::Avx512 => self.process_internal_mono::<Avx512Math>(in_l, out_l, out_r),
            InstructionSet::Avx512VnniBf16 => {
                self.process_internal_mono::<Avx512VnniBf16Math>(in_l, out_l, out_r)
            }
        }
    }
}
