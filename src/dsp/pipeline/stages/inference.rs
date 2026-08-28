// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Stage 2: Neural Inference and Resampling.

use super::super::bridge::MAX_RESAMP_BUF;
use super::super::context::DspPipelineContext;
use super::adaptive_ctrl::{WAVENET_CROSSFADE_MAX_STATES, configure_adaptive_model};
use super::routing::{model_process_stereo_with_os, run_stereo_or_mono};
use crate::dsp::resampler::NamResampler;
use crate::dsp::resampling::StreamingResampleBuffer;
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
    let n = n_samples
        .min(in_l.len())
        .min(in_r.len())
        .min(out_l.len())
        .min(out_r.len());

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

    // The streaming adapter's FIFO capacities are sized for `max_block` per
    // call. Drive it in sub-blocks when the host block exceeds that bound so no
    // input is silently dropped and cardinality stays exact.
    let max_block = stream.max_block();
    let mut offset = 0;
    while offset < n {
        let chunk = (n - offset).min(max_block);
        // Fresh closure per call: `stream.process` consumes the model by value,
        // so a new `FnMut` re-borrowing the pipeline context is created here.
        let model = |model_in_l: &[f32],
                     model_in_r: &[f32],
                     m_out_l: &mut [f32],
                     m_out_r: &mut [f32]|
         -> usize {
            let total = model_in_l
                .len()
                .min(model_in_r.len())
                .min(m_out_l.len())
                .min(m_out_r.len());
            let mut inner = 0;
            while inner < total {
                let chunk = (total - inner).min(MAX_RESAMP_BUF);
                process_model_core(
                    &model_in_l[inner..inner + chunk],
                    &model_in_r[inner..inner + chunk],
                    &mut m_out_l[inner..inner + chunk],
                    &mut m_out_r[inner..inner + chunk],
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
                inner += chunk;
            }
            total
        };
        stream.process(
            &in_l[offset..offset + chunk],
            &in_r[offset..offset + chunk],
            &mut out_l[offset..offset + chunk],
            &mut out_r[offset..offset + chunk],
            model,
        );
        offset += chunk;
    }
    n
}

#[cfg(test)]
mod inference_streaming_test {
    use super::*;
    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
    use crate::common::params::AdaptiveComputeMode;
    use crate::common::spsc::RtStatusFlags;
    use crate::dsp::adaptive::AdaptiveCompute;
    use crate::dsp::gate::{DynamicHysteresis, GateParams};
    use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use crate::models::{NamModel, StaticModel};
    use std::fs;
    use std::path::PathBuf;

    fn load_test_model(name: &str) -> Box<StaticModel> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("tests/fixtures/models");
        path.push(name);
        let json_data = fs::read_to_string(path).expect("Failed to read model file");
        let model_data = parse_nam_json(&json_data).expect("Failed to process model JSON");
        build_model(&model_data).expect("Failed to build model")
    }

    /// Runs `run_inference_streaming` with a real NAM model over irregular
    /// blocks, validating exact cardinality, conservation and zero allocation.
    fn check_streaming_inference(host: u32, model_rate: u32, n: usize, iterations: usize) {
        let mut model = load_test_model("BossWN-nano.nam");
        model.prewarm(2048);

        let mut stream =
            StreamingResampleBuffer::new(host, model_rate, 8192).expect("streaming buffer failed");
        let latency = stream.latency_samples() as u64;

        let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

        let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
        let (os_in_l, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
        let (os_in_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_l, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (scratch_l, scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

        let in_l = vec![0.05f32; n];
        let in_r = vec![0.02f32; n];
        let mut out_l = vec![f32::NAN; n];
        let mut out_r = vec![f32::NAN; n];

        let mut model_opt: Option<Box<StaticModel>> = Some(model);
        let mut model_r_opt: Option<Box<StaticModel>> = None;
        let mut resampler = NamResampler::new_simple(host, model_rate).unwrap();
        let mut ctx = DspPipelineContext::from_parts(
            &mut resampler,
            &mut os_engine_l,
            &mut os_engine_r,
            &mut model_opt,
            &mut model_r_opt,
            1.0,
            1.0,
            &gate_params,
            &mut silence_hysteresis,
            &mut mono_hysteresis,
            0.0,
            0.0,
            &mut process_mono,
            &rt_status,
            &mut adaptive,
        );

        let _guard = TrackingGuard::new();
        for _ in 0..iterations {
            let written = run_inference_streaming(
                &in_l,
                &in_r,
                &mut out_l,
                &mut out_r,
                n,
                &mut ctx,
                &mut stream,
                os_in_l,
                os_in_r,
                os_model_l,
                os_model_r,
                scratch_l,
                scratch_r,
            );
            assert_eq!(written, n, "host={host} model={model_rate} n={n}");
            assert!(
                out_l[..n].iter().all(|x| x.is_finite()),
                "non-finite L at host={host} model={model_rate} n={n}"
            );
            assert!(
                out_r[..n].iter().all(|x| x.is_finite()),
                "non-finite R at host={host} model={model_rate} n={n}"
            );
            assert_eq!(stream.input_pending(), 0);
            assert_eq!(stream.model_pending(), 0);
            assert!(stream.output_pending() <= stream.output_capacity_actual());
        }

        let allocs = get_alloc_count();
        drop(_guard);
        assert_eq!(allocs, 0, "streaming inference must be zero-alloc");

        assert_eq!(
            stream.output_real_total(),
            stream.input_total().saturating_sub(latency),
            "conservation violated: host={host} model={model_rate} n={n}"
        );
        assert_eq!(stream.underflow_total(), 0);
    }

    const TEST_HOST_RATES: &[u32] = &[44_100, 48_000, 96_000, 192_000];
    const IRREGULAR_BLOCKS: &[usize] = &[1, 7, 31, 63, 64, 65, 127, 256];

    #[test]
    fn test_run_inference_streaming_irregular_blocks() {
        for &host in TEST_HOST_RATES {
            for &n in IRREGULAR_BLOCKS {
                check_streaming_inference(host, 48_000, n, 16);
            }
        }
    }

    #[test]
    fn test_run_inference_streaming_max_block() {
        check_streaming_inference(44_100, 48_000, 8192, 8);
        check_streaming_inference(96_000, 48_000, 8192, 8);
        check_streaming_inference(48_000, 48_000, 8192, 8);
    }

    #[test]
    fn test_run_inference_streaming_oversized_block_subchunked() {
        // Host block larger than the adapter's max_block must be processed in
        // sub-blocks without dropping input or fabricating output.
        let host = 44_100u32;
        let model_rate = 48_000u32;
        let n = 700usize;
        let iterations = 16usize;
        let max_block = 256usize;

        let mut model = load_test_model("BossWN-nano.nam");
        model.prewarm(2048);
        let mut stream = StreamingResampleBuffer::new(host, model_rate, max_block)
            .expect("streaming buffer failed");
        let latency = stream.latency_samples() as u64;

        let mut os_engine_l = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let mut os_engine_r = OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).unwrap();
        let gate_params = GateParams::default();
        let mut silence_hysteresis = DynamicHysteresis::new();
        let mut mono_hysteresis = DynamicHysteresis::new();
        let mut process_mono = false;
        let rt_status = RtStatusFlags::default();
        let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
        let mut model_opt: Option<Box<StaticModel>> = Some(model);
        let mut model_r_opt: Option<Box<StaticModel>> = None;
        let mut resampler = NamResampler::new_simple(host, model_rate).unwrap();
        let mut ctx = DspPipelineContext::from_parts(
            &mut resampler,
            &mut os_engine_l,
            &mut os_engine_r,
            &mut model_opt,
            &mut model_r_opt,
            1.0,
            1.0,
            &gate_params,
            &mut silence_hysteresis,
            &mut mono_hysteresis,
            0.0,
            0.0,
            &mut process_mono,
            &rt_status,
            &mut adaptive,
        );

        let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
        let (os_in_l, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
        let (os_in_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_l, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (os_model_r, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
        let (scratch_l, scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

        let in_l = vec![0.05f32; n];
        let in_r = vec![0.02f32; n];
        let mut out_l = vec![f32::NAN; n];
        let mut out_r = vec![f32::NAN; n];

        let _guard = TrackingGuard::new();
        for _ in 0..iterations {
            let written = run_inference_streaming(
                &in_l,
                &in_r,
                &mut out_l,
                &mut out_r,
                n,
                &mut ctx,
                &mut stream,
                os_in_l,
                os_in_r,
                os_model_l,
                os_model_r,
                scratch_l,
                scratch_r,
            );
            assert_eq!(written, n, "oversized block must produce exactly n");
            assert!(
                out_l[..n].iter().all(|x| x.is_finite()),
                "non-finite output for oversized block"
            );
            assert_eq!(stream.input_pending(), 0);
            assert!(stream.output_pending() <= stream.output_capacity_actual());
        }
        let allocs = get_alloc_count();
        drop(_guard);
        assert_eq!(allocs, 0, "sub-chunked processing must be zero-alloc");

        assert_eq!(
            stream.output_real_total(),
            stream.input_total().saturating_sub(latency),
            "sub-chunked conservation violated"
        );
        assert_eq!(stream.underflow_total(), 0);
    }
}
