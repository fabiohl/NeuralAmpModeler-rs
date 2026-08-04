// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

use crate::dsp::adaptive::{AdaptiveCompute, AdaptiveState};
use crate::models::{NamModel, StaticModel};

use super::super::bridge::MAX_RESAMP_BUF;
use super::super::context::DspPipelineContext;

/// Maximum number of WaveNet layer states to backup during double-pass crossfade.
/// Worst-case: stereo (×2) × 8 arrays × 64 dilations = 1024 state entries.
const WAVENET_CROSSFADE_MAX_STATES: usize = 1024;

/// Evaluates adaptive compute FSM state and updates model effective layer count or slimmable size.
///
/// Returns `true` if model processing should be skipped for this block (e.g., minimal mode on LSTM).
#[inline(always)]
fn configure_adaptive_model(
    model_l: &mut Option<Box<crate::models::StaticModel>>,
    model_r: &mut Option<Box<crate::models::StaticModel>>,
    adaptive: &AdaptiveCompute,
    rt_status: &crate::common::spsc::RtStatusFlags,
) -> bool {
    if adaptive.mode() == crate::dsp::adaptive::AdaptiveComputeMode::Off
        && adaptive.slim_override() == crate::dsp::adaptive::SlimOverride::Auto
    {
        return false;
    }

    let hold_layers = adaptive.is_crossfading();

    match adaptive.state() {
        AdaptiveState::Full => {
            if let Some(m) = model_l {
                let layers = m.layer_count();
                if !hold_layers {
                    m.set_effective_layers(layers);
                }
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r {
                let layers = m.layer_count();
                if !hold_layers {
                    m.set_effective_layers(layers);
                }
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
        AdaptiveState::Reduced => {
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_l {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_r {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
        AdaptiveState::Minimal => {
            let lstm_skip = model_l.as_ref().is_some_and(|m| m.is_lstm());
            if lstm_skip {
                return true;
            }
            if let Some(m) = model_l.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_l {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            if let Some(m) = model_r.as_mut().filter(|m| m.is_wavenet()) {
                let layers = m.layer_count();
                let effective = adaptive.wavenet_effective_layers(layers);
                if !hold_layers {
                    m.set_effective_layers(effective);
                }
            }
            if let Some(m) = model_r {
                m.set_slimmable_size(adaptive.slimmable_size(), Some(rt_status));
            }
            false
        }
    }
}

/// Unified helper for mono/stereo processing of neural models.
///
/// Processes the L channel model (_always_) and decides whether the R channel is a mono copy
/// or independent processing via the active R model.
#[inline(always)]
fn run_stereo_or_mono(
    active_model_l: &mut Option<Box<StaticModel>>,
    active_model_r: &mut Option<Box<StaticModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    m_out_l: &mut [f32],
    m_out_r: &mut [f32],
    process_mono: bool,
) {
    if let Some(model_l) = active_model_l {
        model_l.process(model_in_l, m_out_l);
    } else {
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_l.as_ptr(),
                m_out_l.as_mut_ptr(),
                model_in_l.len().min(m_out_l.len()),
            );
        }
    }

    if process_mono {
        unsafe {
            core::ptr::copy_nonoverlapping(
                m_out_l.as_ptr(),
                m_out_r.as_mut_ptr(),
                m_out_l.len().min(m_out_r.len()),
            );
        }
    } else if let Some(model_r) = active_model_r {
        model_r.process(model_in_r, m_out_r);
    } else {
        unsafe {
            core::ptr::copy_nonoverlapping(
                model_in_r.as_ptr(),
                m_out_r.as_mut_ptr(),
                model_in_r.len().min(m_out_r.len()),
            );
        }
    }
}

/// Processes stereo/mono through neural models with optional half-band oversampling.
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
fn model_process_stereo_with_os(
    os_l: &mut crate::dsp::oversample::OversampleEngine,
    os_r: &mut crate::dsp::oversample::OversampleEngine,
    active_model_l: &mut Option<Box<StaticModel>>,
    active_model_r: &mut Option<Box<StaticModel>>,
    model_in_l: &[f32],
    model_in_r: &[f32],
    os_buf_l: &mut [f32],
    os_buf_r: &mut [f32],
    os_model_l: &mut [f32],
    os_model_r: &mut [f32],
    native_out_l: &mut [f32],
    native_out_r: &mut [f32],
    process_mono: bool,
) {
    // L channel
    if os_l.is_bypass() {
        if let Some(m) = active_model_l {
            m.process(model_in_l, native_out_l);
        } else {
            passthru(model_in_l, native_out_l);
        }
    } else {
        let n_os = os_l.upsample(model_in_l, os_buf_l);
        if let Some(m) = active_model_l {
            m.process(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        } else {
            passthru(&os_buf_l[..n_os], &mut os_model_l[..n_os]);
        }
        os_l.downsample(&os_model_l[..n_os], native_out_l);
    }

    // R channel (or mono copy)
    if process_mono {
        passthru(native_out_l, native_out_r);
    } else if os_r.is_bypass() {
        if let Some(m) = active_model_r {
            m.process(model_in_r, native_out_r);
        } else {
            passthru(model_in_r, native_out_r);
        }
    } else {
        let n_os = os_r.upsample(model_in_r, os_buf_r);
        if let Some(m) = active_model_r {
            m.process(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        } else {
            passthru(&os_buf_r[..n_os], &mut os_model_r[..n_os]);
        }
        os_r.downsample(&os_model_r[..n_os], native_out_r);
    }
}

/// Copies `model_in` through to `model_out` when no model is active.
#[inline(always)]
fn passthru(in_buf: &[f32], out_buf: &mut [f32]) {
    let n = in_buf.len().min(out_buf.len());
    unsafe {
        core::ptr::copy_nonoverlapping(in_buf.as_ptr(), out_buf.as_mut_ptr(), n);
    }
}

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
