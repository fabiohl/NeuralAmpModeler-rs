// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

use super::super::bridge::MAX_RESAMP_BUF;
use super::super::context::DspPipelineContext;
use super::adaptive_ctrl::{WAVENET_CROSSFADE_MAX_STATES, configure_adaptive_model};
use super::routing::{model_process_stereo_with_os, run_stereo_or_mono};

/// Stage 2: Neural Inference and Resampling.
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
) -> usize {
    use crate::math::dsp::gain::crossfade_blend_mono_simd;

    let is_resamp_bypass = ctx.resampler.is_bypass();
    let n = n_samples.min(MAX_RESAMP_BUF);

    // Soft-degrade: configure model layers based on CPU pressure
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

    // PATH A: Quality adjustment off (Resampler in Bypass).
    if is_resamp_bypass {
        let model_in_l = &samples_l[..n];
        let model_in_r = if *ctx.process_mono {
            &samples_l[..n]
        } else {
            &samples_r[..n]
        };
        let m_out_l = &mut resamp_out_l[..n];
        let m_out_r = &mut resamp_out_r[..n];

        if lstm_passthrough {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    model_in_l.as_ptr(),
                    m_out_l.as_mut_ptr(),
                    m_out_l.len(),
                );
                core::ptr::copy_nonoverlapping(
                    model_in_r.as_ptr(),
                    m_out_r.as_mut_ptr(),
                    m_out_r.len(),
                );
            }
        } else if is_crossfading_wavenet {
            let old_eff_l = ctx
                .active_model_l
                .as_ref()
                .map(|m| {
                    ctx.adaptive.wavenet_effective_layers_for_state(
                        ctx.adaptive.prev_state(),
                        m.layer_count(),
                    )
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
                        ctx.adaptive.wavenet_effective_layers_for_state(
                            ctx.adaptive.state(),
                            m.layer_count(),
                        )
                    })
                    .unwrap_or(0)
            } else {
                0
            };

            if old_eff_l != new_eff_l || (!*ctx.process_mono && old_eff_r != new_eff_r) {
                // WaveNet crossfading: double-pass and blend
                let mut backup_starts = [0usize; WAVENET_CROSSFADE_MAX_STATES];
                let mut offset = 0;

                // 1. Save buffer start pointers
                if let Some(m) = ctx.active_model_l.as_ref() {
                    m.backup_buffer_starts(&mut backup_starts, &mut offset);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_ref()) {
                    m.backup_buffer_starts(&mut backup_starts, &mut offset);
                }

                // 2. Configure model for first pass (OLD layer count)
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.set_effective_layers(old_eff_l);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.set_effective_layers(old_eff_r);
                }

                // 3. First pass: run model with OLD layer count (outputs to destination)
                run_stereo_or_mono(
                    ctx.active_model_l,
                    ctx.active_model_r,
                    model_in_l,
                    model_in_r,
                    m_out_l,
                    m_out_r,
                    *ctx.process_mono,
                );

                // 4. Restore buffer starts to pre-block state
                let mut offset_restore = 0;
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.restore_buffer_starts(&backup_starts, &mut offset_restore);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.restore_buffer_starts(&backup_starts, &mut offset_restore);
                }

                // 5. Configure model for second pass (NEW layer count)
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.set_effective_layers(new_eff_l);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.set_effective_layers(new_eff_r);
                }

                // 6. Second pass: run model with NEW layer count (outputs to scratch)
                let scratch_l = &mut model_out_l[..n];
                let scratch_r = &mut model_out_r[..n];
                run_stereo_or_mono(
                    ctx.active_model_l,
                    ctx.active_model_r,
                    model_in_l,
                    model_in_r,
                    scratch_l,
                    scratch_r,
                    *ctx.process_mono,
                );

                // 7. Blend: m_out (OLD) + t * (scratch (NEW) - m_out)
                let t = ctx.adaptive.current_crossfade_multiplier();
                crossfade_blend_mono_simd(m_out_l, scratch_l, t);
                if !*ctx.process_mono {
                    crossfade_blend_mono_simd(m_out_r, scratch_r, t);
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
            );
        }

        n
    } else {
        // PATH B: Quality adjustment on (Active Resampler).

        // 1. Translates the sound to the frequency the neural "Brain" understands (usually 48kHz).
        let n_48k = if *ctx.process_mono {
            ctx.resampler.process_input_mono(
                &samples_l[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_input(
                &samples_l[..n],
                &samples_r[..n],
                &mut resamp_mid_l[..MAX_RESAMP_BUF],
                &mut resamp_mid_r[..MAX_RESAMP_BUF],
            )
        };

        let model_in_l = &resamp_mid_l[..n_48k];
        let model_in_r = &resamp_mid_r[..n_48k];
        let m_out_l = &mut model_out_l[..n_48k];
        let m_out_r = &mut model_out_r[..n_48k];

        // 2. Applies the amplifier simulation (Neural Model).
        if lstm_passthrough {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    model_in_l.as_ptr(),
                    m_out_l.as_mut_ptr(),
                    m_out_l.len(),
                );
                core::ptr::copy_nonoverlapping(
                    model_in_r.as_ptr(),
                    m_out_r.as_mut_ptr(),
                    m_out_r.len(),
                );
            }
        } else if is_crossfading_wavenet {
            let old_eff_l = ctx
                .active_model_l
                .as_ref()
                .map(|m| {
                    ctx.adaptive.wavenet_effective_layers_for_state(
                        ctx.adaptive.prev_state(),
                        m.layer_count(),
                    )
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
                        ctx.adaptive.wavenet_effective_layers_for_state(
                            ctx.adaptive.state(),
                            m.layer_count(),
                        )
                    })
                    .unwrap_or(0)
            } else {
                0
            };

            if old_eff_l != new_eff_l || (!*ctx.process_mono && old_eff_r != new_eff_r) {
                // WaveNet crossfading: double-pass and blend
                let mut backup_starts = [0usize; WAVENET_CROSSFADE_MAX_STATES];
                let mut offset = 0;

                // 1. Save buffer start pointers
                if let Some(m) = ctx.active_model_l.as_ref() {
                    m.backup_buffer_starts(&mut backup_starts, &mut offset);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_ref()) {
                    m.backup_buffer_starts(&mut backup_starts, &mut offset);
                }

                // 2. Configure model for first pass (OLD layer count)
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.set_effective_layers(old_eff_l);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.set_effective_layers(old_eff_r);
                }

                // 3. First pass: run model with OLD layer count (outputs to destination)
                run_stereo_or_mono(
                    ctx.active_model_l,
                    ctx.active_model_r,
                    model_in_l,
                    model_in_r,
                    m_out_l,
                    m_out_r,
                    *ctx.process_mono,
                );

                // 4. Restore buffer starts to pre-block state
                let mut offset_restore = 0;
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.restore_buffer_starts(&backup_starts, &mut offset_restore);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.restore_buffer_starts(&backup_starts, &mut offset_restore);
                }

                // 5. Configure model for second pass (NEW layer count)
                if let Some(m) = ctx.active_model_l.as_mut() {
                    m.set_effective_layers(new_eff_l);
                }
                if let (false, Some(m)) = (*ctx.process_mono, ctx.active_model_r.as_mut()) {
                    m.set_effective_layers(new_eff_r);
                }

                // 6. Second pass: run model with NEW layer count (outputs to scratch)
                let scratch_l = &mut resamp_out_l[..n_48k];
                let scratch_r = &mut resamp_out_r[..n_48k];
                run_stereo_or_mono(
                    ctx.active_model_l,
                    ctx.active_model_r,
                    model_in_l,
                    model_in_r,
                    scratch_l,
                    scratch_r,
                    *ctx.process_mono,
                );

                // 7. Blend: m_out (OLD) + t * (scratch (NEW) - m_out)
                let t = ctx.adaptive.current_crossfade_multiplier();
                crossfade_blend_mono_simd(m_out_l, scratch_l, t);
                if !*ctx.process_mono {
                    crossfade_blend_mono_simd(m_out_r, scratch_r, t);
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
            );
        }

        // 3. Translates the sound back to the original frequency of your sound card.
        if *ctx.process_mono {
            ctx.resampler.process_output_mono(
                m_out_l,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        } else {
            ctx.resampler.process_output(
                m_out_l,
                m_out_r,
                &mut resamp_out_l[..MAX_RESAMP_BUF],
                &mut resamp_out_r[..MAX_RESAMP_BUF],
            )
        }
    }
}
