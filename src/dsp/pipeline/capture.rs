// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Full capture DSP pipeline — aggregates all stages.

use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;
use crate::dsp::gate::GateState;
use crate::math::common::SimdMath;
use crate::math::common::set_daz_ftz;

use super::context::{DspBuffers, DspPipelineContext};
use super::stages::{
    apply_input_stage_inner, apply_output_stage_inner, run_inference, write_bridge,
};

/// Full DSP Pipeline (Aggregator).
///
/// Statically dispatches to a monomorphized inner implementation, eliminating
/// v-table overhead from all inner SIMD operations.
///
/// Returns the number of output samples processed (`n_pw`). Returns 0 if `bridge_writer` is None or gate is closed.
///
/// # Host Contract Guard (F-12 / T2.4)
///
/// `n_samples` is defensively clamped to
/// `min(n_samples, samples_l.len(), samples_r.len(), MAX_RESAMP_BUF)` before
/// entering the pipeline. If the host supplied a divergent count, the
/// `RT_STATUS_HOST_CONTRACT_VIOLATION` flag is raised (lock-free, zero-alloc,
/// no RT logging) — the audio thread never panics on slice out-of-bounds.
///
/// # Denormal Protection (FTZ + DAZ) (F-04 / T3.2)
///
/// MXCSR is a per-thread register: a host that never configures it leaves the
/// audio thread exposed to denormal stalls (up to 100× per instruction). This
/// entry point therefore reasserts **Flush-To-Zero** and **Denormals-Are-Zero**
/// at the start of every processing call via
/// [`crate::math::common::set_daz_ftz`] — a fixed `stmxcsr`/`ldmxcsr` pair,
/// outside any sample loop, with no allocation, no lock, and no blocking I/O.
#[inline]
pub fn capture_dsp_pipeline(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: DspPipelineContext<'_>,
    bufs: DspBuffers<'_>,
    sample_rate: u32,
) -> usize {
    #[cfg(feature = "avx512")]
    use crate::math::common::Avx512Math;
    use crate::math::common::{Avx2Math, InstructionSet, effective_instruction_set};

    // F-04 / T3.2: reassert FTZ+DAZ (MXCSR bits 0x8040) on the audio thread
    // before any DSP runs. This is a fixed stmxcsr/ldmxcsr pair — zero-alloc,
    // lock-free, no RT logging — reasserted on every audio callback.
    // SAFETY: `set_daz_ftz` only manipulates the MXCSR register of the current
    // thread; SSE2 is implicit on x86-64 and the `asm!` uses properly aligned
    // locals with valid control-flag bits (0x8040).
    unsafe {
        set_daz_ftz();
    }

    let n = n_samples
        .min(samples_l.len())
        .min(samples_r.len())
        .min(super::bridge::MAX_RESAMP_BUF);
    if n != n_samples {
        ctx.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
    }

    #[expect(deprecated)]
    match effective_instruction_set() {
        #[cfg(feature = "avx512")]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe {
                capture_dsp_pipeline_inner::<Avx512Math>(
                    samples_l,
                    samples_r,
                    n,
                    ctx,
                    bufs,
                    sample_rate,
                )
            }
        }
        #[cfg(not(feature = "avx512"))]
        InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
            // SAFETY: inner invariants upheld by caller.
            unsafe {
                capture_dsp_pipeline_inner::<Avx2Math>(
                    samples_l,
                    samples_r,
                    n,
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
                    n,
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
) -> usize {
    if ctx.bridge_writer.is_none() {
        return 0;
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
        return 0;
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
        bufs.crossfade_scratch_l,
        bufs.crossfade_scratch_r,
    );

    // STAGE 3: CAB-SIM (OPTIONAL IR CONVOLUTION)
    //
    // P-03 / T5.1: process the resampled buffers in place — each adapter
    // consumes the sub-block into its input FIFO before writing back the
    // causal output, so source and destination may alias. This removes the
    // up-to-32 KiB copy-back per callback (and the destination scratch).
    //
    // T2.3 / F-RB-006: the stereo-decoupled pair path runs independent L/R
    // adapters so no convolucional state is shared between channels. The
    // shared-state single-adapter path is retained for mono-only consumers.
    let convolved = if let Some(ref mut pair) = ctx.conv_pair {
        pair.l
            .process_in_place(&mut bufs.resamp_out_l[..n_pw], Some(ctx.rt_status));
        if !*ctx.process_mono {
            pair.r
                .process_in_place(&mut bufs.resamp_out_r[..n_pw], Some(ctx.rt_status));
        }
        true
    } else if let Some(ref mut conv) = ctx.conv {
        conv.process_in_place(&mut bufs.resamp_out_l[..n_pw], Some(ctx.rt_status));
        if !*ctx.process_mono {
            conv.process_in_place(&mut bufs.resamp_out_r[..n_pw], Some(ctx.rt_status));
        }
        true
    } else {
        false
    };

    if convolved && *ctx.process_mono {
        // Mono: cab-sim runs on the left channel only; the right channel
        // mirrors the processed left signal.
        // SAFETY: `n_pw <= MAX_RESAMP_BUF` and both `resamp_out_l`/`resamp_out_r`
        // are at least `MAX_RESAMP_BUF` elements long, so the `n_pw`-element
        // source and destination ranges are in-bounds; the two buffers are
        // distinct allocations, hence non-overlapping.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bufs.resamp_out_l.as_ptr(),
                bufs.resamp_out_r.as_mut_ptr(),
                n_pw,
            );
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

    n_pw
}
