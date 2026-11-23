// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

use super::super::bridge::MAX_RESAMP_BUF;
use super::super::context::DspPipelineContext;
use super::adaptive_ctrl::{WAVENET_CROSSFADE_MAX_STATES, configure_adaptive_model};
use super::routing::{finish_downsample, model_process_stereo_with_os, passthru};
use crate::common::spsc::{RT_STATUS_HOST_CONTRACT_VIOLATION, RtStatusFlags};
use crate::dsp::oversample::OversampleEngine;
use crate::dsp::resampler::NamResampler;
use crate::dsp::resampling::StreamingResampleBuffer;
use crate::math::dsp::gain::crossfade_blend_ramp_mono_simd;
use crate::models::{NamModel, StaticModel};

/// Executes single-channel WaveNet adaptive crossfade (`old_eff` -> `new_eff`)
/// with continuous per-sample linear ramp and single-pass oversampling FIR
/// preservation (`upsample` once -> stack-chunked dual-pass + ramp blend ->
/// `downsample` once).
#[expect(
    clippy::too_many_arguments,
    reason = "DSP channel crossfade kernel taking disjoint model, oversampler, and buffer slices"
)]
#[inline(always)]
fn crossfade_wavenet_channel(
    model: &mut Option<Box<StaticModel>>,
    os: &mut OversampleEngine,
    model_in: &[f32],
    os_in: &mut [f32],
    os_model: &mut [f32],
    m_out: &mut [f32],
    scr: &mut [f32],
    old_eff: usize,
    new_eff: usize,
    t_start: f32,
    t_end: f32,
    rt_status: &RtStatusFlags,
) {
    let Some(m) = model.as_deref_mut() else {
        passthru(model_in, m_out);
        return;
    };

    if old_eff == new_eff {
        m.set_effective_layers(new_eff);
        if os.is_bypass() {
            m.process(model_in, m_out);
        } else {
            let n_os = os.upsample(model_in, os_in, Some(rt_status));
            m.process(&os_in[..n_os], &mut os_model[..n_os]);
            let written = os.downsample(&os_model[..n_os], m_out, Some(rt_status));
            finish_downsample(written, model_in.len(), m_out, Some(rt_status));
        }
        return;
    }

    let mut backup_starts = [0usize; WAVENET_CROSSFADE_MAX_STATES];
    let mut post_first_starts = [0usize; WAVENET_CROSSFADE_MAX_STATES];
    if os.is_bypass() {
        let mid_t = 0.5 * (t_start + t_end);
        let old_second = mid_t < 0.5;
        let (first_eff, second_eff) = if old_second {
            (new_eff, old_eff)
        } else {
            (old_eff, new_eff)
        };

        let mut offset = 0;
        m.backup_buffer_starts(&mut backup_starts, &mut offset);
        m.set_effective_layers(first_eff);
        if old_second {
            m.process(model_in, scr);
        } else {
            m.process(model_in, m_out);
        }

        if first_eff > second_eff {
            let mut post_off = 0;
            m.backup_buffer_starts(&mut post_first_starts, &mut post_off);
        }

        let mut offset_restore = 0;
        m.restore_buffer_starts(&backup_starts, &mut offset_restore);
        m.set_effective_layers(second_eff);
        if old_second {
            m.process(model_in, m_out);
        } else {
            m.process(model_in, scr);
        }

        if first_eff > second_eff {
            let mut post_restore = 0;
            m.restore_buffer_starts(&post_first_starts, &mut post_restore);
        }

        let n = model_in.len().min(m_out.len()).min(scr.len());
        let t_step = if n > 0 {
            (t_end - t_start) / n as f32
        } else {
            0.0
        };
        crossfade_blend_ramp_mono_simd(&mut m_out[..n], &scr[..n], t_start, t_step);
    } else {
        // Upsample once before chunked evaluation so the half-band FIR delay lines
        // advance continuously without paying double FIR cost.
        let n_os = os.upsample(model_in, os_in, Some(rt_status));
        let t_step_os = if n_os > 0 {
            (t_end - t_start) / n_os as f32
        } else {
            0.0
        };
        // Process in stack-buffered sub-chunks (512 OS frames = 2 KB stack, zero heap)
        // so common RT blocks at 2x/4x oversampling run in minimal dual-pass sweeps.
        const OS_XFADE_CHUNK: usize = 512;
        let mut new_chunk = [0.0f32; OS_XFADE_CHUNK];
        let mut pos = 0usize;
        while pos < n_os {
            let c = (n_os - pos).min(OS_XFADE_CHUNK);
            let in_chunk = &os_in[pos..pos + c];
            let out_chunk = &mut os_model[pos..pos + c];
            let chunk_t_start = t_start + (pos as f32) * t_step_os;
            let chunk_t_mid = chunk_t_start + 0.5 * (c as f32) * t_step_os;
            // Keep the higher-weighted crossfade branch's state authoritative in the
            // shared ring buffers (`old_eff` while `t < 0.5`, `new_eff` once `t >= 0.5`).
            let old_second = chunk_t_mid < 0.5;
            let (first_eff, second_eff) = if old_second {
                (new_eff, old_eff)
            } else {
                (old_eff, new_eff)
            };

            let mut offset = 0;
            m.backup_buffer_starts(&mut backup_starts, &mut offset);
            m.set_effective_layers(first_eff);
            if old_second {
                m.process(in_chunk, &mut new_chunk[..c]);
            } else {
                m.process(in_chunk, out_chunk);
            }

            if first_eff > second_eff {
                let mut post_off = 0;
                m.backup_buffer_starts(&mut post_first_starts, &mut post_off);
            }

            let mut offset_restore = 0;
            m.restore_buffer_starts(&backup_starts, &mut offset_restore);
            m.set_effective_layers(second_eff);
            if old_second {
                m.process(in_chunk, out_chunk);
            } else {
                m.process(in_chunk, &mut new_chunk[..c]);
            }

            // Keep layers `[second_eff..first_eff]` aligned with `[0..second_eff]` across
            // chunks and blocks whenever the first pass exercised more layers.
            if first_eff > second_eff {
                let mut post_restore = 0;
                m.restore_buffer_starts(&post_first_starts, &mut post_restore);
            }

            crossfade_blend_ramp_mono_simd(out_chunk, &new_chunk[..c], chunk_t_start, t_step_os);
            pos += c;
        }
        let written = os.downsample(&os_model[..n_os], m_out, Some(rt_status));
        finish_downsample(written, model_in.len(), m_out, Some(rt_status));
    }
}

/// Model processing core — dispatches between passthrough, crossfade, and normal paths.
///
/// `scratch_l` / `scratch_r` receive the second-pass output during non-oversampled
/// WaveNet crossfading, and must NOT alias the final output buffer when chunking is active.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "DSP kernel signature with internal scratch buffers required by chunking design"
)]
fn process_model_core(
    model_in_l: &[f32],
    model_in_r: &[f32],
    m_out_l: &mut [f32],
    m_out_r: &mut [f32],
    ctx: &mut DspPipelineContext<'_>,
    os_in_l: &mut [f32],
    os_in_r: &mut [f32],
    os_model_l: &mut [f32],
    os_model_r: &mut [f32],
    scratch_l: &mut [f32],
    scratch_r: &mut [f32],
    lstm_passthrough: bool,
    is_crossfading_wavenet: bool,
) {
    if lstm_passthrough {
        // Path 1: Adaptive Passthrough.
        // When dynamic compute control activates fallback mode (e.g. CPU overload protection
        // for heavy LSTM models), inference is bypassed cleanly by streaming input directly
        // to output. This maintains continuity and prevents thread overruns or audio glitches.
        //
        // SAFETY: each copy length is the minimum of the source and destination
        // lengths, so the copies of initialized `f32` values stay in bounds; input
        // and output buffers are distinct, so the regions cannot overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_l.as_ptr(),
                m_out_l.as_mut_ptr(),
                model_in_l.len().min(m_out_l.len()),
            );
            if *ctx.process_mono {
                if m_out_r.len() >= m_out_l.len() {
                    core::ptr::copy_nonoverlapping(
                        m_out_l.as_ptr(),
                        m_out_r.as_mut_ptr(),
                        m_out_l.len(),
                    );
                }
            } else {
                core::ptr::copy_nonoverlapping(
                    model_in_r.as_ptr(),
                    m_out_r.as_mut_ptr(),
                    model_in_r.len().min(m_out_r.len()),
                );
            }
        }
    } else if is_crossfading_wavenet {
        // Path 2: WaveNet Dynamic Adaptive Crossfading.
        // Dual-pass evaluation with per-sample SIMD ramp and preserved oversampling FIR state.
        let old_eff_l = ctx
            .active_model_l
            .as_ref()
            .map(|m| {
                ctx.adaptive
                    .wavenet_effective_layers_for_state(ctx.adaptive.prev_state(), m.layer_count())
            })
            .unwrap_or(0);
        let new_eff_l = ctx
            .active_model_l
            .as_ref()
            .map(|m| {
                ctx.adaptive
                    .wavenet_effective_layers_for_state(ctx.adaptive.state(), m.layer_count())
            })
            .unwrap_or(0);
        let old_eff_r = if !*ctx.process_mono {
            ctx.active_model_r
                .as_ref()
                .map(|m| {
                    ctx.adaptive.wavenet_effective_layers_for_state(
                        ctx.adaptive.prev_state(),
                        m.layer_count(),
                    )
                })
                .unwrap_or(0)
        } else {
            0
        };
        let new_eff_r = if !*ctx.process_mono {
            ctx.active_model_r
                .as_ref()
                .map(|m| {
                    ctx.adaptive
                        .wavenet_effective_layers_for_state(ctx.adaptive.state(), m.layer_count())
                })
                .unwrap_or(0)
        } else {
            0
        };

        if old_eff_l != new_eff_l || (!*ctx.process_mono && old_eff_r != new_eff_r) {
            let frame_advance = if ctx.resampler.is_bypass() {
                model_in_l.len()
            } else {
                let nam_rate = ctx.resampler.nam_rate().max(1) as u64;
                let host_rate = ctx.resampler.host_rate() as u64;
                ((model_in_l.len() as u64 * host_rate) / nam_rate).max(1) as usize
            };
            let (t_start, t_end) = ctx.adaptive.crossfade_ramp_bounds(frame_advance);

            let scr_l = &mut scratch_l[..model_in_l.len()];
            crossfade_wavenet_channel(
                ctx.active_model_l,
                ctx.os_l,
                model_in_l,
                os_in_l,
                os_model_l,
                m_out_l,
                scr_l,
                old_eff_l,
                new_eff_l,
                t_start,
                t_end,
                ctx.rt_status,
            );

            if *ctx.process_mono {
                if m_out_r.len() >= m_out_l.len() {
                    passthru(m_out_l, m_out_r);
                }
            } else {
                let scr_r = &mut scratch_r[..model_in_r.len()];
                crossfade_wavenet_channel(
                    ctx.active_model_r,
                    ctx.os_r,
                    model_in_r,
                    os_in_r,
                    os_model_r,
                    m_out_r,
                    scr_r,
                    old_eff_r,
                    new_eff_r,
                    t_start,
                    t_end,
                    ctx.rt_status,
                );
            }
            return;
        }
    }

    // Standard inference with optional oversampling (also used when adaptive
    // crossfade has not changed the active layer count).
    model_process_stereo_with_os(
        ctx.os_l,
        ctx.os_r,
        ctx.active_model_l,
        ctx.active_model_r,
        model_in_l,
        model_in_r,
        os_in_l,
        os_in_r,
        os_model_l,
        os_model_r,
        m_out_l,
        m_out_r,
        *ctx.process_mono,
        Some(ctx.rt_status),
    );
}

/// Stage 2: Neural Inference and Resampling.
///
/// PATH A (resampler bypass): model processes input directly. No chunking needed
/// as input/output sample counts are equal.
///
/// PATH B (active resampler): input is chunked to respect `MAX_RESAMP_BUF` output
/// capacity. For upsampling ratios (e.g. 44100→48000 Hz), a single pass would
/// overflow the intermediate resampling buffers. The function loops, processing
/// chunks bounded by `NamResampler::max_input_samples(MAX_RESAMP_BUF, …)`, and
/// accumulates the resampled output contiguously in `resamp_out_l/r`.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn run_inference(
    samples_l: &mut [f32],
    samples_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
    resamp_mid_l: &mut [f32],
    resamp_mid_r: &mut [f32],
    resamp_out_l: &mut [f32],
    resamp_out_r: &mut [f32],
    model_out_l: &mut [f32],
    model_out_r: &mut [f32],
    os_in_l: &mut [f32],
    os_in_r: &mut [f32],
    os_model_l: &mut [f32],
    os_model_r: &mut [f32],
    crossfade_scratch_l: &mut [f32],
    crossfade_scratch_r: &mut [f32],
) -> usize {
    let n = n_samples.min(MAX_RESAMP_BUF);
    if n == 0 {
        return 0;
    }
    if n != n_samples {
        ctx.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
    }

    // Defensive buffer capacity validation across all builds (R-16).
    // An insufficient slice must not panic, access OOB or execute partial DSP.
    let mut capacity_valid = samples_l.len() >= n && resamp_out_l.len() >= n;
    if !*ctx.process_mono {
        capacity_valid = capacity_valid && samples_r.len() >= n && resamp_out_r.len() >= n;
    }

    let is_resamp_bypass = ctx.resampler.is_bypass();
    let supports_skip = ctx
        .active_model_l
        .as_ref()
        .is_some_and(|m| m.supports_layer_skip());
    let is_crossfading_wavenet = supports_skip && ctx.adaptive.is_crossfading();

    if is_resamp_bypass {
        capacity_valid = capacity_valid && model_out_l.len() >= n;
        if !*ctx.process_mono {
            capacity_valid = capacity_valid && model_out_r.len() >= n;
        }
        let factor_l = if ctx.os_l.is_bypass() {
            1
        } else {
            ctx.os_l.factor().multiplier()
        };
        if factor_l > 1 {
            let need_os = n * factor_l;
            capacity_valid =
                capacity_valid && os_in_l.len() >= need_os && os_model_l.len() >= need_os;
        }
        if !*ctx.process_mono {
            let factor_r = if ctx.os_r.is_bypass() {
                1
            } else {
                ctx.os_r.factor().multiplier()
            };
            if factor_r > 1 {
                let need_os = n * factor_r;
                capacity_valid =
                    capacity_valid && os_in_r.len() >= need_os && os_model_r.len() >= need_os;
            }
        }
    } else {
        // Resampler active: intermediate rate conversion chunks require MAX_RESAMP_BUF
        capacity_valid = capacity_valid
            && resamp_mid_l.len() >= MAX_RESAMP_BUF
            && model_out_l.len() >= MAX_RESAMP_BUF;
        if !*ctx.process_mono {
            capacity_valid = capacity_valid
                && resamp_mid_r.len() >= MAX_RESAMP_BUF
                && model_out_r.len() >= MAX_RESAMP_BUF;
        }
        let factor_l = if ctx.os_l.is_bypass() {
            1
        } else {
            ctx.os_l.factor().multiplier()
        };
        if factor_l > 1 {
            let need_os = MAX_RESAMP_BUF * factor_l;
            capacity_valid =
                capacity_valid && os_in_l.len() >= need_os && os_model_l.len() >= need_os;
        }
        if !*ctx.process_mono {
            let factor_r = if ctx.os_r.is_bypass() {
                1
            } else {
                ctx.os_r.factor().multiplier()
            };
            if factor_r > 1 {
                let need_os = MAX_RESAMP_BUF * factor_r;
                capacity_valid =
                    capacity_valid && os_in_r.len() >= need_os && os_model_r.len() >= need_os;
            }
        }
        if is_crossfading_wavenet {
            capacity_valid = capacity_valid && crossfade_scratch_l.len() >= MAX_RESAMP_BUF;
            if !*ctx.process_mono {
                capacity_valid = capacity_valid && crossfade_scratch_r.len() >= MAX_RESAMP_BUF;
            }
        }
    }

    if !capacity_valid {
        ctx.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
        let safe_l = resamp_out_l.len().min(n);
        resamp_out_l[..safe_l].fill(0.0);
        let safe_r = resamp_out_r.len().min(n);
        resamp_out_r[..safe_r].fill(0.0);
        return 0;
    }

    let lstm_passthrough = configure_adaptive_model(
        ctx.active_model_l,
        ctx.active_model_r,
        ctx.adaptive,
        ctx.rt_status,
    );

    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &[][..]
        } else {
            &samples_r[..n]
        };
        let m_out_l = &mut resamp_out_l[..n];
        let m_out_r = if *ctx.process_mono {
            if resamp_out_r.len() >= n {
                &mut resamp_out_r[..n]
            } else {
                &mut [][..]
            }
        } else {
            &mut resamp_out_r[..n]
        };
        let scr_r = if *ctx.process_mono {
            &mut [][..]
        } else {
            &mut model_out_r[..n]
        };

        process_model_core(
            model_in_l,
            model_in_r,
            m_out_l,
            m_out_r,
            ctx,
            os_in_l,
            os_in_r,
            os_model_l,
            os_model_r,
            &mut model_out_l[..n],
            scr_r,
            lstm_passthrough,
            is_crossfading_wavenet,
        );

        n
    } else {
        let host_rate = ctx.resampler.host_rate();
        let nam_rate = ctx.resampler.nam_rate();
        let mono_mid_cap = 256usize;
        let use_mono_mid_scratch = *ctx.process_mono && resamp_mid_r.len() < MAX_RESAMP_BUF;
        let mid_out_cap = if use_mono_mid_scratch {
            mono_mid_cap
        } else {
            MAX_RESAMP_BUF
        };
        let max_input_per_pass =
            NamResampler::max_input_samples_checked(mid_out_cap, host_rate, nam_rate).unwrap_or(0);

        let mut mono_mid_r = [0.0f32; 256];
        let mut mono_out_r = [0.0f32; 256];
        let mut total_output: usize = 0;
        let mut input_offset: usize = 0;

        while input_offset < n {
            let remaining = n - input_offset;
            let chunk_input = if max_input_per_pass == 0 {
                remaining.min(mid_out_cap)
            } else {
                remaining.min(max_input_per_pass)
            };

            let progress = if *ctx.process_mono {
                if use_mono_mid_scratch {
                    ctx.resampler.process_input_mono(
                        &samples_l[input_offset..input_offset + chunk_input],
                        &mut resamp_mid_l[..mono_mid_cap],
                        &mut mono_mid_r,
                    )
                } else {
                    ctx.resampler.process_input_mono(
                        &samples_l[input_offset..input_offset + chunk_input],
                        &mut resamp_mid_l[..MAX_RESAMP_BUF],
                        &mut resamp_mid_r[..MAX_RESAMP_BUF],
                    )
                }
            } else {
                ctx.resampler.process_input(
                    &samples_l[input_offset..input_offset + chunk_input],
                    &samples_r[input_offset..input_offset + chunk_input],
                    &mut resamp_mid_l[..MAX_RESAMP_BUF],
                    &mut resamp_mid_r[..MAX_RESAMP_BUF],
                )
            };

            let consumed = progress.samples_read;
            let n_48k = progress.samples_written;

            if consumed == 0 {
                input_offset += chunk_input;
                continue;
            }
            input_offset += consumed;

            let model_in_l = &resamp_mid_l[..n_48k];
            let model_in_r = if *ctx.process_mono {
                &[][..]
            } else {
                &resamp_mid_r[..n_48k]
            };
            let m_out_l = &mut model_out_l[..n_48k];
            let m_out_r = if *ctx.process_mono {
                &mut [][..]
            } else {
                &mut model_out_r[..n_48k]
            };

            process_model_core(
                model_in_l,
                model_in_r,
                m_out_l,
                m_out_r,
                ctx,
                os_in_l,
                os_in_r,
                os_model_l,
                os_model_r,
                crossfade_scratch_l,
                crossfade_scratch_r,
                lstm_passthrough,
                is_crossfading_wavenet,
            );

            if *ctx.process_mono {
                if resamp_out_r.len() >= resamp_out_l.len() {
                    let out_progress = ctx.resampler.process_output_mono(
                        m_out_l,
                        &mut resamp_out_l[total_output..],
                        &mut resamp_out_r[total_output..],
                    );
                    total_output += out_progress.samples_written;
                } else {
                    let mut out_in_pos = 0usize;
                    while out_in_pos < m_out_l.len() && total_output < resamp_out_l.len() {
                        let out_avail = (resamp_out_l.len() - total_output).min(mono_out_r.len());
                        let out_progress = ctx.resampler.process_output_mono(
                            &m_out_l[out_in_pos..],
                            &mut resamp_out_l[total_output..total_output + out_avail],
                            &mut mono_out_r[..out_avail],
                        );
                        total_output += out_progress.samples_written;
                        if out_progress.samples_read == 0 && out_progress.samples_written == 0 {
                            break;
                        }
                        out_in_pos += out_progress.samples_read;
                    }
                }
            } else {
                let out_progress = ctx.resampler.process_output(
                    m_out_l,
                    m_out_r,
                    &mut resamp_out_l[total_output..],
                    &mut resamp_out_r[total_output..],
                );
                total_output += out_progress.samples_written;
            }
        }

        total_output
    }
}

/// Streaming inference pass with strict host cardinality (F-PERF-002).
///
/// Drives the host-agnostic [`StreamingResampleBuffer`] through the real model
/// processing chain, guaranteeing that **exactly** `n_samples` host samples are
/// consumed and produced per call — regardless of the sample-rate ratio, block
/// size, or phase history. Excess resampler output is retained internally;
/// the declared latency (see [`StreamingResampleBuffer::latency_samples`]) is
/// zero-primed during warm-up.
///
/// This is the additive counterpart of [`run_inference`]: it does not modify
/// the legacy path (whose gates/parity remain untouched). The caller owns the
/// `stream` adapter and the working slices; all allocations happen off-RT at
/// `StreamingResampleBuffer::new` — the processing path is zero-alloc.
///
/// # Returns
/// The number of host samples written to `out_l/out_r` — always `n_samples`
/// (truncated to the minimum slice length).
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn run_inference_streaming(
    in_l: &[f32],
    in_r: &[f32],
    out_l: &mut [f32],
    out_r: &mut [f32],
    n_samples: usize,
    ctx: &mut DspPipelineContext<'_>,
    stream: &mut StreamingResampleBuffer,
    os_in_l: &mut [f32],
    os_in_r: &mut [f32],
    os_model_l: &mut [f32],
    os_model_r: &mut [f32],
    crossfade_scratch_l: &mut [f32],
    crossfade_scratch_r: &mut [f32],
) -> usize {
    let n = if *ctx.process_mono {
        n_samples.min(in_l.len()).min(MAX_RESAMP_BUF)
    } else {
        n_samples
            .min(in_l.len())
            .min(in_r.len())
            .min(MAX_RESAMP_BUF)
    };
    if n == 0 {
        return 0;
    }
    if n != n_samples {
        ctx.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
    }

    let mut capacity_valid = in_l.len() >= n && out_l.len() >= n;
    if !*ctx.process_mono {
        capacity_valid = capacity_valid && in_r.len() >= n && out_r.len() >= n;
    }

    let factor_l = if ctx.os_l.is_bypass() {
        1
    } else {
        ctx.os_l.factor().multiplier()
    };
    let max_block = stream.max_block();
    let max_chunk = max_block.min(MAX_RESAMP_BUF);
    if factor_l > 1 {
        let need_os = max_chunk * factor_l;
        capacity_valid = capacity_valid && os_in_l.len() >= need_os && os_model_l.len() >= need_os;
    }
    if !*ctx.process_mono {
        let factor_r = if ctx.os_r.is_bypass() {
            1
        } else {
            ctx.os_r.factor().multiplier()
        };
        if factor_r > 1 {
            let need_os = max_chunk * factor_r;
            capacity_valid =
                capacity_valid && os_in_r.len() >= need_os && os_model_r.len() >= need_os;
        }
    }
    let supports_skip = ctx
        .active_model_l
        .as_ref()
        .is_some_and(|m| m.supports_layer_skip());
    let is_crossfading_wavenet = supports_skip && ctx.adaptive.is_crossfading();
    if is_crossfading_wavenet {
        capacity_valid = capacity_valid && crossfade_scratch_l.len() >= max_chunk;
        if !*ctx.process_mono {
            capacity_valid = capacity_valid && crossfade_scratch_r.len() >= max_chunk;
        }
    }
    if !capacity_valid {
        ctx.rt_status.set_flag(RT_STATUS_HOST_CONTRACT_VIOLATION);
        let safe_l = out_l.len().min(n);
        out_l[..safe_l].fill(0.0);
        let safe_r = out_r.len().min(n);
        out_r[..safe_r].fill(0.0);
        return 0;
    }

    let lstm_passthrough = configure_adaptive_model(
        ctx.active_model_l,
        ctx.active_model_r,
        ctx.adaptive,
        ctx.rt_status,
    );

    // The streaming adapter's FIFO capacities are sized for `max_block` per
    // call. Drive it in sub-blocks when the host block exceeds that bound so no
    // input is silently dropped and cardinality stays exact.
    let max_block = stream.max_block();
    let process_mono = *ctx.process_mono;
    let mono_needs_dummy_out_r = process_mono && out_r.len() < n;
    let mono_needs_dummy_in_r = process_mono && in_r.len() < n;
    let mut dummy_out_r = [0.0f32; 256];
    let mut offset = 0;
    while offset < n {
        let mut chunk = (n - offset).min(max_block);
        if mono_needs_dummy_out_r {
            chunk = chunk.min(dummy_out_r.len());
        }
        // Fresh closure per call: `stream.process` consumes the model by value,
        // so a new `FnMut` re-borrowing the pipeline context is created here.
        let model = |model_in_l: &[f32],
                     model_in_r: &[f32],
                     m_out_l: &mut [f32],
                     m_out_r: &mut [f32]|
         -> usize {
            let total = if process_mono {
                model_in_l.len().min(m_out_l.len())
            } else {
                model_in_l
                    .len()
                    .min(model_in_r.len())
                    .min(m_out_l.len())
                    .min(m_out_r.len())
            };
            // Split model-rate audio into `MAX_RESAMP_BUF` slices so internal scratch buffers never overflow.
            let mut inner = 0;
            while inner < total {
                let inner_chunk = (total - inner).min(MAX_RESAMP_BUF);
                // Omit right-channel slices in mono mode so downstream avoids redundant R processing.
                let m_in_r = if process_mono {
                    &[][..]
                } else {
                    &model_in_r[inner..inner + inner_chunk]
                };
                let m_out_r_slice = if process_mono && m_out_r.len() < inner + inner_chunk {
                    &mut [][..]
                } else {
                    &mut m_out_r[inner..inner + inner_chunk]
                };
                process_model_core(
                    &model_in_l[inner..inner + inner_chunk],
                    m_in_r,
                    &mut m_out_l[inner..inner + inner_chunk],
                    m_out_r_slice,
                    ctx,
                    os_in_l,
                    os_in_r,
                    os_model_l,
                    os_model_r,
                    crossfade_scratch_l,
                    crossfade_scratch_r,
                    lstm_passthrough,
                    is_crossfading_wavenet,
                );
                inner += inner_chunk;
            }
            total
        };
        let stream_in_r = if mono_needs_dummy_in_r {
            &in_l[offset..offset + chunk]
        } else {
            &in_r[offset..offset + chunk]
        };
        let stream_out_r = if mono_needs_dummy_out_r {
            &mut dummy_out_r[..chunk]
        } else {
            &mut out_r[offset..offset + chunk]
        };
        stream.process(
            &in_l[offset..offset + chunk],
            stream_in_r,
            &mut out_l[offset..offset + chunk],
            stream_out_r,
            model,
        );
        offset += chunk;
    }
    n
}

#[cfg(test)]
#[path = "inference_test.rs"]
mod inference_streaming_test;
