// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Full capture DSP pipeline — aggregates all stages.

use crate::dsp::gate::GateState;
use crate::math::common::SimdMath;

use super::context::{DspBuffers, DspPipelineContext};
use super::stages::{
    apply_input_stage_inner, apply_output_stage_inner, run_inference, write_bridge,
};

/// Full DSP Pipeline (Aggregator).
///
/// Statically dispatches to a monomorphized inner implementation, eliminating
/// v-table overhead from all inner SIMD operations.
#[inline(always)]
pub fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: DspPipelineContext<'_>,
    bufs: DspBuffers<'_>,
    sample_rate: u32,
) {
    use crate::math::common::{
        Avx2Math, Avx512Math, Avx512VnniBf16Math, InstructionSet, effective_instruction_set,
    };
    match effective_instruction_set() {
        InstructionSet::Avx512VnniBf16 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe {
                capture_dsp_pipeline_inner::<Avx512VnniBf16Math>(
                    samples_l,
                    samples_r,
                    n_samples,
                    ctx,
                    bufs,
                    sample_rate,
                )
            }
        }
        InstructionSet::Avx512 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe {
                capture_dsp_pipeline_inner::<Avx512Math>(
                    samples_l,
                    samples_r,
                    n_samples,
                    ctx,
                    bufs,
                    sample_rate,
                )
            }
        }
        InstructionSet::Avx2 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe {
                capture_dsp_pipeline_inner::<Avx2Math>(
                    samples_l,
                    samples_r,
                    n_samples,
                    ctx,
                    bufs,
                    sample_rate,
                )
            }
        }
    }
}

/// Inner monomorphized implementation of the full DSP pipeline.
///
/// Receives a concrete `M: SimdMath` type resolved by the outer dispatch,
/// propagating it to all inner stages. This eliminates all v-table indirection
/// from the pipeline hot-path.
///
/// # Safety
/// Caller must ensure valid buffer references and that `M` corresponds to the
/// CPU features detected at initialization.
#[inline(always)]
unsafe fn capture_dsp_pipeline_inner<M: SimdMath>(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
    bufs: DspBuffers<'_>,
    sample_rate: u32,
) {
    if ctx.bridge_writer.is_none() {
        return;
    }
    // STAGE 1: INPUT AND CLEANUP
    let gate_state =
        // SAFETY: slices and context are valid; M corresponds to detected CPU features.
        unsafe { apply_input_stage_inner::<M>(samples_l, samples_r, n_samples, &mut ctx) };

    // STATE MANAGEMENT (SILENCE vs SOUND)
    crate::dsp::gate_flags::report_gate_flags(ctx.rt_status, gate_state);

    if gate_state == GateState::Closed {
        if let Some(writer) = ctx.bridge_writer {
            writer.write_silence();
        }
        return;
    }

    // STAGE 2: THE "BRAIN" (AMP/PEDAL SIMULATION)
    let n_pw = run_inference(
        samples_l,
        samples_r,
        n_samples,
        &mut ctx,
        bufs.resamp_mid_l,
        bufs.resamp_mid_r,
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        bufs.model_out_l,
        bufs.model_out_r,
        bufs.os_in_l,
        bufs.os_in_r,
        bufs.os_model_l,
        bufs.os_model_r,
    );

    // STAGE 3: CAB-SIM (OPTIONAL IR CONVOLUTION)
    if let Some(ref mut conv) = ctx.conv {
        conv.process_variable(&bufs.resamp_out_l[..n_pw], &mut bufs.model_out_l[..n_pw]);
        unsafe {
            core::ptr::copy_nonoverlapping(
                bufs.model_out_l.as_ptr(),
                bufs.resamp_out_l.as_mut_ptr(),
                n_pw,
            );
        }
        if !*ctx.process_mono {
            conv.process_variable(&bufs.resamp_out_r[..n_pw], &mut bufs.model_out_r[..n_pw]);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bufs.model_out_r.as_ptr(),
                    bufs.resamp_out_r.as_mut_ptr(),
                    n_pw,
                );
            }
        } else {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bufs.resamp_out_l.as_ptr(),
                    bufs.resamp_out_r.as_mut_ptr(),
                    n_pw,
                );
            }
        }
    }

    // STAGE 4: FINAL ADJUSTMENT AND PROTECTION
    // SAFETY: buffers and context are valid; M corresponds to detected CPU features.
    unsafe {
        apply_output_stage_inner::<M>(
            bufs.resamp_out_l,
            bufs.resamp_out_r,
            n_pw,
            ctx.output_gain_mult,
            ctx.silence_hysteresis,
            ctx.rt_status,
            *ctx.process_mono,
            ctx.adaptive,
            sample_rate,
        );
    }

    // STAGE 5: FINAL DELIVERY (THE BRIDGE)
    write_bridge(
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_pw,
        ctx.bridge_writer,
        *ctx.process_mono,
    );
}
