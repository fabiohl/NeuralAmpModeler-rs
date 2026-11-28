// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Observable mono contract for the working output buffers.
//!
//! In mono (`process_mono == true`) only the left channel is processed
//! independently. Both public delivery surfaces must therefore expose the
//! *final* left signal on the right channel as well:
//!
//! * `resamp_out_l` / `resamp_out_r` — the caller-owned working buffers that
//!   direct consumers (e.g. recording) read after the pipeline returns.
//! * the monitoring bridge back-buffer, which independently mirrors L into R.
//!
//! The mirror is applied at the end of the output stage, after gain, gating,
//! dither subtraction, clipping detection and non-finite sanitization.

use super::super::*;
use crate::common::params::AdaptiveComputeMode;
use crate::common::spsc::RtStatusFlags;
use crate::dsp::adaptive::AdaptiveCompute;
use crate::dsp::cabsim::adapter::{CabSimAdapter, CabSimPair};
use crate::dsp::cabsim::conv::ConvEngine;
use crate::dsp::gate::{DynamicHysteresis, GateParams};
use crate::dsp::oversample::{OversampleEngine, OversampleFactor};
use crate::dsp::resampler::NamResampler;
use crate::dsp::resampling::StreamingResampleBuffer;

const N: usize = 64;

struct BlockOutput {
    n_pw: usize,
    buffers_l: Vec<f32>,
    buffers_r: Vec<f32>,
    bridge_l: Vec<f32>,
    bridge_r: Vec<f32>,
}

struct MonoFixture {
    streaming: bool,
    output_gain: f32,
    resampler: NamResampler,
    stream: StreamingResampleBuffer,
    bridge: Box<DspBridge>,
    mid_l: Vec<f32>,
    mid_r: Vec<f32>,
    out_l: Vec<f32>,
    out_r: Vec<f32>,
    model_l: Vec<f32>,
    model_r: Vec<f32>,
    os_in_l: Vec<f32>,
    os_in_r: Vec<f32>,
    os_model_l: Vec<f32>,
    os_model_r: Vec<f32>,
    xf_l: Vec<f32>,
    xf_r: Vec<f32>,
    gate_params: GateParams,
    silence_hysteresis: DynamicHysteresis,
    mono_hysteresis: DynamicHysteresis,
    process_mono: bool,
    adaptive: AdaptiveCompute,
    os_l: OversampleEngine,
    os_r: OversampleEngine,
    rt_status: RtStatusFlags,
    conv_pair: Option<CabSimPair>,
}

impl MonoFixture {
    fn new(streaming: bool, output_gain: f32, with_cab: bool) -> Self {
        let buf = || vec![0.0f32; MAX_RESAMP_BUF];
        let conv_pair = with_cab.then(|| {
            let ir = [1.0f32];
            let make = || {
                CabSimAdapter::new(Box::new(
                    ConvEngine::new(&ir, N).expect("test convolution engine"),
                ))
                .expect("test cab adapter")
            };
            CabSimPair {
                l: Box::new(make()),
                r: Box::new(make()),
                sample_rate: 48_000,
            }
        });

        Self {
            streaming,
            output_gain,
            resampler: NamResampler::new_simple(48_000, 48_000).expect("resampler"),
            stream: StreamingResampleBuffer::new(48_000, 48_000, MAX_RESAMP_BUF)
                .expect("streaming adapter"),
            bridge: DspBridge::new_boxed(),
            mid_l: buf(),
            mid_r: buf(),
            out_l: buf(),
            out_r: buf(),
            model_l: buf(),
            model_r: buf(),
            os_in_l: buf(),
            os_in_r: buf(),
            os_model_l: buf(),
            os_model_r: buf(),
            xf_l: buf(),
            xf_r: buf(),
            // `hold_frames == 0` lets the idempotent mono detector engage on
            // the first block whose L == R instead of waiting for the default
            // 2048-frame hold window.
            gate_params: GateParams::new(-70.0, -80.0, 0, 256, 1e-4),
            silence_hysteresis: DynamicHysteresis::new(),
            mono_hysteresis: DynamicHysteresis::new(),
            process_mono: false,
            adaptive: AdaptiveCompute::new(AdaptiveComputeMode::Off),
            os_l: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF)
                .expect("oversampler"),
            os_r: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF)
                .expect("oversampler"),
            rt_status: RtStatusFlags::default(),
            conv_pair,
        }
    }

    fn run(&mut self, input_l: &[f32], input_r: &[f32]) -> BlockOutput {
        let mut samples_l = input_l.to_vec();
        let mut samples_r = input_r.to_vec();

        let ctx = DspPipelineContext {
            resampler: &mut self.resampler,
            os_l: &mut self.os_l,
            os_r: &mut self.os_r,
            active_model_l: &mut None,
            active_model_r: &mut None,
            input_gain_mult: 1.0,
            output_gain_mult: self.output_gain,
            gate_params: &self.gate_params,
            silence_hysteresis: &mut self.silence_hysteresis,
            mono_hysteresis: &mut self.mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut self.process_mono,
            rt_status: &self.rt_status,
            adaptive: &mut self.adaptive,
            // SAFETY: the boxed bridge outlives the context and is exclusively
            // owned by this fixture for the duration of the call.
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&raw mut *self.bridge)) },
            conv: None,
            conv_pair: self.conv_pair.as_mut(),
        };

        let bufs = DspBuffers {
            resamp_mid_l: &mut self.mid_l,
            resamp_mid_r: &mut self.mid_r,
            resamp_out_l: &mut self.out_l,
            resamp_out_r: &mut self.out_r,
            model_out_l: &mut self.model_l,
            model_out_r: &mut self.model_r,
            os_in_l: &mut self.os_in_l,
            os_in_r: &mut self.os_in_r,
            os_model_l: &mut self.os_model_l,
            os_model_r: &mut self.os_model_r,
            crossfade_scratch_l: &mut self.xf_l,
            crossfade_scratch_r: &mut self.xf_r,
        };

        let n_pw = if self.streaming {
            capture_dsp_pipeline_streaming(
                &mut samples_l,
                &mut samples_r,
                N,
                ctx,
                &mut self.stream,
                bufs,
                48_000,
            )
        } else {
            capture_dsp_pipeline(&mut samples_l, &mut samples_r, N, ctx, bufs, 48_000)
        };

        let read_idx = self
            .bridge
            .active_read_idx
            .load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: single-threaded read after the pipeline stopped writing.
        let bridge_buf = unsafe { &*self.bridge.buffers[read_idx].get() };
        let n_bridge = bridge_buf.n_samples as usize;
        let generation = bridge_buf.generation;
        let bridge_l = bridge_buf.buf_l[..n_bridge].to_vec();
        let bridge_r = bridge_buf.buf_r[..n_bridge].to_vec();
        // Acknowledge the publication exactly like `DspBridgeReader::read_block`
        // so the next callback is allowed to publish again.
        self.bridge
            .consumed_gen
            .store(generation, std::sync::atomic::Ordering::Release);

        BlockOutput {
            n_pw,
            buffers_l: self.out_l[..n_pw].to_vec(),
            buffers_r: self.out_r[..n_pw].to_vec(),
            bridge_l,
            bridge_r,
        }
    }
}

fn mono_input() -> Vec<f32> {
    (0..N).map(|i| (i as f32 + 1.0) * 0.01).collect()
}

#[test]
fn output_stage_mono_mirrors_final_left_after_gain_and_dither() {
    let mut out_l = mono_input();
    for sample in &mut out_l {
        *sample += DENORMAL_DITHER_OFFSET;
    }
    let mut out_r = vec![0.125f32; N];
    let mut hysteresis = DynamicHysteresis::new();
    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    apply_output_stage(
        &mut out_l,
        &mut out_r,
        N,
        2.0,
        &mut hysteresis,
        &rt_status,
        true,
        &mut adaptive,
        48_000,
    );

    assert_eq!(out_l, out_r, "mono output stage must mirror final L into R");
    let reference = mono_input();
    for (i, (&sample, &input)) in out_l.iter().zip(reference.iter()).enumerate() {
        assert!(
            (sample - 2.0 * input).abs() < 1e-6,
            "sample {i}: dither must be subtracted before the mono mirror, got {sample}"
        );
    }
}

#[test]
fn output_stage_mono_mirrors_zero_when_gate_is_closed() {
    let mut out_l = mono_input();
    let mut out_r = vec![0.25f32; N];
    let params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
    let mut hysteresis = DynamicHysteresis::new();
    hysteresis.update(0.0, 1.0, 1.0, &params, N);
    assert_eq!(hysteresis.multiplier(), 0.0, "gate must be fully closed");

    let rt_status = RtStatusFlags::default();
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);
    apply_output_stage(
        &mut out_l,
        &mut out_r,
        N,
        1.0,
        &mut hysteresis,
        &rt_status,
        true,
        &mut adaptive,
        48_000,
    );

    assert!(out_l.iter().all(|&s| s == 0.0));
    assert_eq!(
        out_l, out_r,
        "closed gate must silence R as well, never leave pre-gate audio"
    );
}

#[test]
fn output_stage_mono_sanitized_mirrors_after_sanitization() {
    let mut out_l = mono_input();
    out_l[7] = f32::NAN;
    let mut out_r = vec![0.5f32; N];
    let mut hysteresis = DynamicHysteresis::new();
    let rt_status = RtStatusFlags::default();
    rt_status.set_flag(crate::common::spsc::RT_STATUS_NON_FINITE_INPUT_DETECTED);
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    apply_output_stage_sanitized(
        &mut out_l,
        &mut out_r,
        N,
        1.0,
        &mut hysteresis,
        &rt_status,
        true,
        &mut adaptive,
        48_000,
    );

    assert_eq!(out_l[7], 0.0);
    assert!(out_l.iter().all(|s| s.is_finite()));
    assert_eq!(
        out_l, out_r,
        "R must mirror the sanitized final L, not a stale non-finite copy"
    );
}

#[test]
fn mono_pipeline_public_buffers_match_bridge_normal_and_streaming() {
    for streaming in [false, true] {
        for with_cab in [false, true] {
            let label = format!("streaming={streaming} cab={with_cab}");
            let mut fixture = MonoFixture::new(streaming, 2.0, with_cab);
            let input = mono_input();
            let out = fixture.run(&input, &input);

            assert_eq!(out.n_pw, N, "{label}: full block expected");
            assert!(
                fixture.process_mono,
                "{label}: identical L/R must engage mono"
            );
            assert_eq!(
                out.buffers_l, out.buffers_r,
                "{label}: public working buffers must expose final mono L in R"
            );
            assert_eq!(
                out.bridge_l, out.buffers_l,
                "{label}: bridge L must equal the delivered L"
            );
            assert_eq!(
                out.bridge_r, out.buffers_r,
                "{label}: bridge R must equal the delivered R"
            );
            if !with_cab {
                for (i, (&sample, &input)) in out.buffers_l.iter().zip(input.iter()).enumerate() {
                    assert!(
                        (sample - 2.0 * input).abs() < 1e-6,
                        "{label}: sample {i} must carry the output gain"
                    );
                }
            }
        }
    }
}

#[test]
fn stereo_to_mono_transition_keeps_buffers_and_bridge_coherent() {
    let mut fixture = MonoFixture::new(true, 1.0, false);
    let input = mono_input();

    let inverted: Vec<f32> = input.iter().map(|&s| -s).collect();
    let stereo = fixture.run(&input, &inverted);
    assert!(!fixture.process_mono, "distinct L/R must stay stereo");
    assert_ne!(
        stereo.buffers_l, stereo.buffers_r,
        "stereo block must keep independent channels"
    );

    let mono = fixture.run(&input, &input);
    assert!(fixture.process_mono, "identical L/R must engage mono");
    assert_eq!(mono.n_pw, N);
    assert_eq!(
        mono.buffers_l, mono.buffers_r,
        "after the stereo→mono transition R must mirror final L"
    );
    assert_eq!(mono.bridge_l, mono.buffers_l);
    assert_eq!(mono.bridge_r, mono.buffers_r);
}
