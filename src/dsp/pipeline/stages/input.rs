// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 1: Gate, Input Gains, Mono Detection, and Silence Bypass.

#[cfg(feature = "testing")]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::common::spsc::RtStatusFlags;
use crate::dsp::gate::GateState;
use crate::math::common::SimdMath;

use super::super::bridge::DspBridgeWriter;
use super::super::context::DspPipelineContext;

#[cfg(feature = "testing")]
#[cfg_attr(docsrs, doc(cfg(feature = "testing")))]
/// Global control to disable the noise gate/silence bypass during profiling/benchmarks.
/// Only available in testing builds — production host plugin builds must not share
/// this state across instances.
pub static DISABLE_GATE: AtomicBool = AtomicBool::new(false);

/// Ultra-low DC offset injected at the input stage to prevent subnormal floats
/// from propagating through neural network activations during fade-out/silence.
/// At -220 dBFS, this bias is 76 dB below the 24-bit DAC noise floor and inaudible.
pub const DENORMAL_DITHER_OFFSET: f32 = 1.0e-11;

/// Silence Bypass: signals silence and zeros the bridge so that playback emits silence.
/// Delegates gate-flag reporting to the canonical `report_gate_flags`.
#[cold]
#[inline(never)]
pub fn handle_silence_bypass(bridge: Option<DspBridgeWriter>, rt_status: &RtStatusFlags) {
    crate::dsp::gate_flags::report_gate_flags(rt_status, crate::dsp::gate::GateState::Closed);

    if let Some(writer) = bridge {
        writer.write_silence();
    }
}

/// Stage 1: Gate, Input Gains, and Mono Detection.
#[inline(always)]
pub fn apply_input_stage(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    #[cfg(feature = "avx512")]
    use crate::math::common::Avx512Math;
    use crate::math::common::{Avx2Math, InstructionSet, effective_instruction_set};
    #[expect(deprecated)]
    match effective_instruction_set() {
        #[cfg(feature = "avx512")]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe { apply_input_stage_inner::<Avx512Math>(samples_l, samples_r, n_samples, ctx) }
        }
        #[cfg(not(feature = "avx512"))]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe { apply_input_stage_inner::<Avx2Math>(samples_l, samples_r, n_samples, ctx) }
        }
        InstructionSet::Avx2 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe { apply_input_stage_inner::<Avx2Math>(samples_l, samples_r, n_samples, ctx) }
        }
    }
}

/// Inner generic implementation of the input stage, monomorphized over SIMD backend.
///
/// # Safety
/// Caller must ensure valid buffer references and that the SIMD backend is
/// supported by the CPU.
#[inline(always)]
pub(crate) unsafe fn apply_input_stage_inner<M: SimdMath>(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
) -> GateState {
    #[cfg(feature = "stereo")]
    let energy_ms = {
        // SAFETY: both slices are valid references of identical length.
        unsafe { M::compute_energy_stereo(&samples_l[..n_samples], &samples_r[..n_samples]) }
    };
    #[cfg(not(feature = "stereo"))]
    let energy_ms = {
        // SAFETY: slice is valid.
        unsafe { M::compute_energy(&samples_l[..n_samples]) }
    };

    #[cfg(not(feature = "stereo"))]
    let _ = samples_r;

    // 1. UPDATE THE NOISE GATE
    ctx.silence_hysteresis.update(
        energy_ms,
        ctx.threshold_open_sq,
        ctx.threshold_close_sq,
        ctx.gate_params,
        n_samples,
    );

    if ctx.silence_hysteresis.state() == GateState::Closed {
        #[cfg(feature = "testing")]
        if DISABLE_GATE.load(Ordering::Relaxed) {
            return GateState::Open;
        }
        return GateState::Closed;
    }

    #[cfg(feature = "stereo")]
    {
        // 2. MONO SOUND DETECTION (SAME ON BOTH SIDES)
        // SAFETY: both slices are valid references of identical length.
        let max_diff =
            unsafe { M::compute_max_diff(&samples_l[..n_samples], &samples_r[..n_samples]) };

        ctx.mono_hysteresis.update(
            max_diff,
            ctx.gate_params.mono_epsilon,
            ctx.gate_params.mono_epsilon * 0.9,
            ctx.gate_params,
            n_samples,
        );

        *ctx.process_mono = ctx.mono_hysteresis.state() == GateState::Closed
            || ctx.mono_hysteresis.state() == GateState::FadingOut;
    }
    #[cfg(not(feature = "stereo"))]
    {
        *ctx.process_mono = true;
    }

    // 3. INPUT VOLUME ADJUSTMENT (GAIN) + DENORMAL SUPPRESSION (FUSED WHEN POSSIBLE)
    // SAFETY: slice is valid; gain and offset are finite f32.
    if (ctx.input_gain_mult - 1.0).abs() >= 1e-6 {
        unsafe {
            M::apply_gain_then_dither(
                &mut samples_l[..n_samples],
                ctx.input_gain_mult,
                DENORMAL_DITHER_OFFSET,
            )
        };

        #[cfg(feature = "stereo")]
        if !*ctx.process_mono {
            // SAFETY: slice is valid; gain and offset are finite f32.
            unsafe {
                M::apply_gain_then_dither(
                    &mut samples_r[..n_samples],
                    ctx.input_gain_mult,
                    DENORMAL_DITHER_OFFSET,
                )
            };
        }
    } else {
        // SAFETY: slice is valid and offset is a finite f32.
        unsafe { M::apply_dither_add(&mut samples_l[..n_samples], DENORMAL_DITHER_OFFSET) };

        #[cfg(feature = "stereo")]
        if !*ctx.process_mono {
            // SAFETY: slice is valid and offset is a finite f32.
            unsafe { M::apply_dither_add(&mut samples_r[..n_samples], DENORMAL_DITHER_OFFSET) };
        }
    }

    ctx.silence_hysteresis.state()
}
