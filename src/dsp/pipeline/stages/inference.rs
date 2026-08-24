// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

use super::super::bridge::MAX_RESAMP_BUF;
use super::super::context::DspPipelineContext;
use super::adaptive_ctrl::{WAVENET_CROSSFADE_MAX_STATES, configure_adaptive_model};
use super::routing::{model_process_stereo_with_os, run_stereo_or_mono};
use crate::dsp::resampler::NamResampler;
use crate::math::dsp::gain::crossfade_blend_mono_simd;

/// Model processing core — dispatches between passthrough, crossfade, and normal paths.
///
/// `scratch_l` / `scratch_r` receive the second-pass output during WaveNet crossfading,
/// and must NOT alias the final output buffer when chunking is active.
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
        // SAFETY: each copy length is the minimum of the source and destination
        // lengths, so the copies of initialized `f32` values stay in bounds; input
        // and output buffers are distinct, so the regions cannot overlap.
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_l.as_ptr(),
                m_out_l.as_mut_ptr(),
                model_in_l.len().min(m_out_l.len()),
            );
            core::ptr::copy_nonoverlapping(
                model_in_r.as_ptr(),
                m_out_r.as_mut_ptr(),
                model_in_r.len().min(m_out_r.len()),
            );
        }
    } else if is_crossfading_wavenet {
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
            let mut backup_starts = [0usize; WAVENET_CROSSFADE_MAX_STATES];
            let mut offset = 0;

            if let Some(m) = ctx.active_model_l.as_ref() {
                m.backup_buffer_starts(&mut backup_starts, &mut offset);
            }
            if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_ref()) {
                m.backup_buffer_starts(&mut backup_starts, &mut offset);
            }

            if let Some(m) = ctx.active_model_l.as_mut() {
                m.set_effective_layers(old_eff_l);
            }
            if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                m.set_effective_layers(old_eff_r);
            }

            run_stereo_or_mono(
                ctx.active_model_l,
                ctx.active_model_r,
                model_in_l,
                model_in_r,
                m_out_l,
                m_out_r,
                *ctx.process_mono,
            );

            let mut offset_restore = 0;
            if let Some(m) = ctx.active_model_l.as_mut() {
                m.restore_buffer_starts(&backup_starts, &mut offset_restore);
            }
            if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                m.restore_buffer_starts(&backup_starts, &mut offset_restore);
            }

            if let Some(m) = ctx.active_model_l.as_mut() {
                m.set_effective_layers(new_eff_l);
            }
            if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                m.set_effective_layers(new_eff_r);
            }

            let scr_l = &mut scratch_l[..model_in_l.len()];
            let scr_r = &mut scratch_r[..model_in_l.len()];
            run_stereo_or_mono(
                ctx.active_model_l,
                ctx.active_model_r,
                model_in_l,
                model_in_r,
                scr_l,
                scr_r,
                *ctx.process_mono,
            );

            let t = ctx.adaptive.current_crossfade_multiplier();
            crossfade_blend_mono_simd(m_out_l, scr_l, t);
            if !*ctx.process_mono {
                crossfade_blend_mono_simd(m_out_r, scr_r, t);
            }
        } else {
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
    } else {
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
    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    let lstm_passthrough = configure_adaptive_model(
        ctx.active_model_l,
        ctx.active_model_r,
        ctx.adaptive,
        ctx.rt_status,
    );

    let supports_skip = ctx
        .active_model_l
        .as_ref()
        .is_some_and(|m| m.supports_layer_skip());
    let is_crossfading_wavenet = supports_skip && ctx.adaptive.is_crossfading();

    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let m_out_l = &mut resamp_out_l[..n];
        let m_out_r = &mut resamp_out_r[..n];

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
            &mut model_out_r[..n],
            lstm_passthrough,
            is_crossfading_wavenet,
        );

        n
    } else {
        let host_rate = ctx.resampler.host_rate();
        let nam_rate = ctx.resampler.nam_rate();
        let max_input_per_pass =
            NamResampler::max_input_samples(MAX_RESAMP_BUF, host_rate, nam_rate);

        let mut total_output: usize = 0;
        let mut input_offset: usize = 0;

        while input_offset < n {
            let remaining = n - input_offset;
            let chunk_input = if max_input_per_pass == 0 {
                remaining.min(MAX_RESAMP_BUF)
            } else {
                remaining.min(max_input_per_pass)
            };

            let progress = if *ctx.process_mono {
                ctx.resampler.process_input_mono(
                    &samples_l[input_offset..input_offset + chunk_input],
                    &mut resamp_mid_l[..MAX_RESAMP_BUF],
                    &mut resamp_mid_r[..MAX_RESAMP_BUF],
                )
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
            let model_in_r = &resamp_mid_r[..n_48k];
            let m_out_l = &mut model_out_l[..n_48k];
            let m_out_r = &mut model_out_r[..n_48k];

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

            let out_progress = if *ctx.process_mono {
                ctx.resampler.process_output_mono(
                    m_out_l,
                    &mut resamp_out_l[total_output..],
                    &mut resamp_out_r[total_output..],
                )
            } else {
                ctx.resampler.process_output(
                    m_out_l,
                    m_out_r,
                    &mut resamp_out_l[total_output..],
                    &mut resamp_out_r[total_output..],
                )
            };

            total_output += out_progress.samples_written;
        }

        total_output
    }
}
