// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Full capture DSP pipeline — aggregates all stages.

use crate::common::spsc::RT_STATUS_HOST_CONTRACT_VIOLATION;
use crate::dsp::gate::{DynamicHysteresis, GateState};
use crate::dsp::resampling::StreamingResampleBuffer;
use crate::math::common::SimdMath;
use crate::math::common::set_daz_ftz;

use super::context::{DspBuffers, DspPipelineContext};
use super::stages::{
    apply_input_stage_inner, apply_output_stage_inner, run_inference, run_inference_streaming,
    write_bridge,
};

/// Full DSP Pipeline (Aggregator).
///
/// Statically dispatches to a monomorphized inner implementation, eliminating
/// v-table overhead from all inner SIMD operations.
///
/// Returns the number of output samples processed (`n_pw`), or 0 when the
/// gate is closed. `bridge_writer = None` runs the full pipeline normally
/// and only skips the bridge write (the bridge stage itself treats `None`
/// as "no listener").
///
/// # Host Contract Guard
///
/// `n_samples` is defensively clamped to
/// `min(n_samples, samples_l.len(), samples_r.len(), MAX_RESAMP_BUF)` before
/// entering the pipeline. If the host supplied a divergent count, the
/// `RT_STATUS_HOST_CONTRACT_VIOLATION` flag is raised (lock-free, zero-alloc,
/// no RT logging) — the audio thread never panics on slice out-of-bounds.
///
/// # Denormal Protection (FTZ + DAZ)
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
    use crate::math::common::Avx2Math;
    #[cfg(feature = "avx512")]
    use crate::math::common::{Avx512Math, InstructionSet, effective_instruction_set};

    // Reassert FTZ+DAZ (MXCSR bits 0x8040) on the audio thread
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

    #[cfg(feature = "avx512")]
    {
        #[expect(deprecated)]
        match effective_instruction_set() {
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
    #[cfg(not(feature = "avx512"))]
    {
        // SAFETY: inner invariants upheld by caller.
        unsafe {
            capture_dsp_pipeline_inner::<Avx2Math>(samples_l, samples_r, n, ctx, bufs, sample_rate)
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
    // `bridge_writer = None` is a first-class mode (headless consumers): the
    // pipeline below runs every stage and `write_bridge` simply skips the
    // delivery when there is no listener. There is deliberately no early
    // return on `bridge_writer.is_none()` — that would force every consumer
    // without a monitoring bridge to reimplement the whole stage
    // orchestration (F5).

    // STAGE 1: INPUT AND CLEANUP
    let gate_state =
        // SAFETY: slices and context are valid; M corresponds to detected CPU features.
        unsafe { apply_input_stage_inner::<M>(samples_l, samples_r, n_samples, &mut ctx) };

    // STATE MANAGEMENT (SILENCE vs SOUND)
    crate::dsp::gate_flags::report_gate_flags(ctx.rt_status, gate_state);

    if gate_state == GateState::Closed {
        // No IR tail drain on this non-streaming entry: the gate-capable
        // consumer path is the streaming entry
        // (`capture_dsp_pipeline_streaming`), which drains the cab-sim tail
        // on closure instead of cutting to silence instantly.
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
    // Process the resampled buffers in place — each adapter
    // consumes the sub-block into its input FIFO before writing back the
    // causal output, so source and destination may alias. This removes the
    // up-to-32 KiB copy-back per callback (and the destination scratch).
    //
    // Stereo decoupling: the stereo-decoupled pair path runs independent L/R
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

/// Full DSP Pipeline with Streaming Resampler Adapter (Strict Host Cardinality).
///
/// Drives the multi-stage DSP pipeline through [`run_inference_streaming`],
/// guaranteeing that **exactly** `n_samples` host samples are consumed and
/// produced per invocation regardless of fractional sample-rate ratios.
///
/// `bridge_writer = None` is a first-class mode: every stage runs normally
/// and only the bridge delivery is skipped — consumers without a monitoring
/// bridge get the full orchestrated pipeline instead of having to
/// reimplement the stage orchestration themselves (F5).
///
/// # Noise-Gate Closure and IR Tail Drain
///
/// When the noise gate closes, the cab-sim convolution does not cut to
/// silence instantly: while any attached cab-sim adapter still reports
/// pending ring-out
/// ([`remaining_tail_samples`](crate::dsp::cabsim::adapter::CabSimAdapter::remaining_tail_samples)),
/// closed-gate blocks feed zero input through the convolution stage and emit
/// the decaying IR response
/// ([`drain_tail`](crate::dsp::cabsim::adapter::CabSimAdapter::drain_tail)),
/// so the audible reverb tail completes naturally instead of being
/// truncated. The budget is re-armed on the active-audio path
/// ([`rearm_tail`](crate::dsp::cabsim::adapter::CabSimAdapter::rearm_tail))
/// and the drain applies the output stage through a fresh unity gate — the
/// ring-out is intentional signal, not noise floor, while the real gate FSM
/// (stage 1) keeps tracking the host input and reopens on the next loud
/// block. Once the budget is exhausted (or no cab-sim is attached), closed
/// blocks emit true silence again. The drain is RT-safe: zero allocations,
/// zero locks, bounded by the armed budget.
///
/// Returns the number of output samples produced for this block
/// (`n_pw == n_samples` while the gate is open or the IR tail is still
/// draining, `0` once the gate is closed and the tail is fully flushed).
#[inline]
pub fn capture_dsp_pipeline_streaming<'b>(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: DspPipelineContext<'_>,
    stream: &mut StreamingResampleBuffer,
    bufs: impl Into<DspBuffers<'b>>,
    sample_rate: u32,
) -> usize {
    let bufs = bufs.into();
    use crate::math::common::Avx2Math;
    #[cfg(feature = "avx512")]
    use crate::math::common::{Avx512Math, InstructionSet, effective_instruction_set};

    // SAFETY: Setting denormals-as-zero / flush-to-zero is safe on x86_64 targets.
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

    #[cfg(feature = "avx512")]
    {
        #[expect(deprecated)]
        match effective_instruction_set() {
            InstructionSet::Avx512 | InstructionSet::Avx512VnniBf16 => {
                // SAFETY: Target CPU is guaranteed to support AVX-512 when dynamic dispatch returns this branch.
                unsafe {
                    capture_dsp_pipeline_streaming_inner::<Avx512Math>(
                        samples_l,
                        samples_r,
                        n,
                        ctx,
                        stream,
                        bufs,
                        sample_rate,
                    )
                }
            }
            InstructionSet::Avx2 => {
                // SAFETY: Target architecture baseline guarantees AVX2 support.
                unsafe {
                    capture_dsp_pipeline_streaming_inner::<Avx2Math>(
                        samples_l,
                        samples_r,
                        n,
                        ctx,
                        stream,
                        bufs,
                        sample_rate,
                    )
                }
            }
        }
    }
    #[cfg(not(feature = "avx512"))]
    {
        // SAFETY: Fallback executes baseline AVX2 kernels supported by x86-64-v3.
        unsafe {
            capture_dsp_pipeline_streaming_inner::<Avx2Math>(
                samples_l,
                samples_r,
                n,
                ctx,
                stream,
                bufs,
                sample_rate,
            )
        }
    }
}

#[inline(always)]
unsafe fn capture_dsp_pipeline_streaming_inner<M: SimdMath>(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    mut ctx: DspPipelineContext<'_>,
    stream: &mut StreamingResampleBuffer,
    bufs: DspBuffers<'_>,
    sample_rate: u32,
) -> usize {
    // `bridge_writer = None` is a first-class mode (headless consumers): the
    // pipeline below runs every stage and `write_bridge` simply skips the
    // delivery when there is no listener. There is deliberately no early
    // return on `bridge_writer.is_none()` — that would force every consumer
    // without a monitoring bridge to reimplement the whole stage
    // orchestration (F5).

    // STAGE 1: INPUT AND CLEANUP
    // SAFETY: Caller guarantees valid pointers and aligned slices within buffer lengths.
    let gate_state =
        unsafe { apply_input_stage_inner::<M>(samples_l, samples_r, n_samples, &mut ctx) };

    // STATE MANAGEMENT (SILENCE vs SOUND)
    crate::dsp::gate_flags::report_gate_flags(ctx.rt_status, gate_state);

    if gate_state == GateState::Closed {
        stream.reset();

        // ── Cab-sim IR tail drain ────────────────────────────────────────
        // The gate closed because the *input* went silent, but the
        // convolution FDL still holds the ringing IR response. Feeding zero
        // input through the adapter renders that tail causally; cutting to
        // silence here would truncate the reverb audibly (F8).
        //
        // The drain budget is armed on the active-audio path (`rearm_tail`)
        // and consumed here (`drain_tail` never re-arms), so the drain is
        // strictly bounded and always terminates.
        let convolved = if let Some(ref mut pair) = ctx.conv_pair {
            // Both channels always advance together so their FDLs flush
            // independently (a mono mirror below overwrites R afterwards).
            let pending = pair.remaining_tail_samples();
            if pending > 0 {
                bufs.resamp_out_l[..n_samples].fill(0.0);
                bufs.resamp_out_r[..n_samples].fill(0.0);
                pair.drain_tail_stereo(
                    &mut bufs.resamp_out_l[..n_samples],
                    &mut bufs.resamp_out_r[..n_samples],
                    Some(ctx.rt_status),
                );
                true
            } else {
                false
            }
        } else if let Some(ref mut conv) = ctx.conv {
            let pending = conv.remaining_tail_samples();
            if pending > 0 {
                bufs.resamp_out_l[..n_samples].fill(0.0);
                conv.drain_tail(&mut bufs.resamp_out_l[..n_samples], Some(ctx.rt_status));
                if !*ctx.process_mono {
                    bufs.resamp_out_r[..n_samples].fill(0.0);
                    conv.drain_tail(&mut bufs.resamp_out_r[..n_samples], Some(ctx.rt_status));
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        if convolved {
            if *ctx.process_mono {
                // Mono mirror: cab-sim runs on the left channel only; the
                // right channel mirrors the drained left signal.
                // SAFETY: `n_samples <= MAX_RESAMP_BUF` and both `resamp_out_l`
                // / `resamp_out_r` are at least `MAX_RESAMP_BUF` elements long,
                // so the `n_samples`-element source and destination ranges are
                // in-bounds; the two buffers are distinct allocations, hence
                // non-overlapping.
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        bufs.resamp_out_l.as_ptr(),
                        bufs.resamp_out_r.as_mut_ptr(),
                        n_samples,
                    );
                }
            }

            // The ring-out is intentional signal, not noise floor — the
            // closed-gate multiplier (0.0) must not zero it. A fresh unity
            // gate (Open, multiplier 1.0, steady) yields exactly the wet
            // output gain + smoothing while the IR rings to completion; the
            // real gate FSM (stage 1) keeps tracking the host input
            // independently and reopens on the next loud block.
            let mut tail_gate = DynamicHysteresis::new();
            // SAFETY: buffers and context are valid; M corresponds to detected CPU features.
            unsafe {
                apply_output_stage_inner::<M>(
                    bufs.resamp_out_l,
                    bufs.resamp_out_r,
                    n_samples,
                    ctx.output_gain_mult,
                    &mut tail_gate,
                    ctx.rt_status,
                    *ctx.process_mono,
                    ctx.adaptive,
                    sample_rate,
                );
            }

            // STAGE 5: FINAL DELIVERY (THE BRIDGE) — `None` simply skips.
            write_bridge(
                bufs.resamp_out_l,
                bufs.resamp_out_r,
                n_samples,
                ctx.bridge_writer,
                *ctx.process_mono,
            );

            // Strict host cardinality: the drained block still consumed and
            // produced exactly `n_samples` host samples.
            return n_samples;
        }

        if let Some(writer) = ctx.bridge_writer {
            writer.write_silence();
        }
        return 0;
    }

    // STAGE 2: THE "BRAIN" (AMP/PEDAL SIMULATION WITH STREAMING RESAMPLER)
    let n_pw = run_inference_streaming(
        samples_l,
        samples_r,
        bufs.resamp_out_l,
        bufs.resamp_out_r,
        n_samples,
        &mut ctx,
        stream,
        bufs.os_in_l,
        bufs.os_in_r,
        bufs.os_model_l,
        bufs.os_model_r,
        bufs.crossfade_scratch_l,
        bufs.crossfade_scratch_r,
    );

    // STAGE 3: CAB-SIM (OPTIONAL IR CONVOLUTION)
    let convolved = if let Some(ref mut pair) = ctx.conv_pair {
        pair.l
            .process_in_place(&mut bufs.resamp_out_l[..n_pw], Some(ctx.rt_status));
        if !*ctx.process_mono {
            pair.r
                .process_in_place(&mut bufs.resamp_out_r[..n_pw], Some(ctx.rt_status));
        }
        if n_pw > 0 {
            // Fresh signal reached the convolution: re-arm the IR ring-out
            // budget so the next gate close flushes the complete tail. The
            // drain path never re-arms (no signal reaches the conv there),
            // which is what keeps the drain strictly bounded.
            pair.rearm_tail();
        }
        true
    } else if let Some(ref mut conv) = ctx.conv {
        conv.process_in_place(&mut bufs.resamp_out_l[..n_pw], Some(ctx.rt_status));
        if !*ctx.process_mono {
            conv.process_in_place(&mut bufs.resamp_out_r[..n_pw], Some(ctx.rt_status));
        }
        if n_pw > 0 {
            // See the pair path above: re-arm on the active-audio path only.
            conv.rearm_tail();
        }
        true
    } else {
        false
    };

    if convolved && *ctx.process_mono {
        // SAFETY: bufs.resamp_out_l and bufs.resamp_out_r are disjoint slices of length >= n_pw.
        unsafe {
            core::ptr::copy_nonoverlapping(
                bufs.resamp_out_l.as_ptr(),
                bufs.resamp_out_r.as_mut_ptr(),
                n_pw,
            );
        }
    }

    // STAGE 4: FINAL ADJUSTMENT AND PROTECTION
    // SAFETY: Slices bufs.resamp_out_l and bufs.resamp_out_r are guaranteed to have length >= n_pw.
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
