// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance regression gate — Criterion benches with adequate statistical power
//! for CI regression detection against persisted baselines.
//!
//! ## Running
//!
//! ```sh
//! # Save a baseline (first run, or after verified optimization):
//! taskset -c 0 cargo bench --bench regression_gate -- --save-baseline ci-baseline
//!
//! # CI run: compare against saved baseline, fail on regression:
//! taskset -c 0 cargo bench --bench regression_gate -- --baseline ci-baseline
//! ```
//!
//! ## Bench layout
//!
//! Each bench function loads/initializes the real model fixture or DSP component,
//! warms up, and measures per-block processing time (64 samples).
//! Criterion handles statistical comparison against the saved baseline.
//!
//! ## Model & DSP coverage
//!
//! ### Model Inference Core:
//! - WaveNet: Standard (CH16), Feather (CH8), Lite (CH12), Nano (CH4), Dyn Free
//! - A2: Full (CH8), Lite (CH3), Dyn Gated (CH8), Dyn Blended (CH3)
//! - LSTM: 1x16, 2x8, Dyn 1x7
//! - Linear (RF=2048), ConvNet
//!
//! ### DSP Infrastructure:
//! - Resampler: 44.1 kHz -> 48 kHz, 96 kHz -> 48 kHz (64-sample block)
//! - CabSim: Medium IR (2048 taps, 64-sample block)
//! - Pipeline: Canonical Base (No OS, 48 kHz, 64-sample block)
//! - Pipeline: HQ 4xOS (4x oversampling, 48 kHz, 64-sample block)

#[path = "common.rs"]
mod common;

use criterion::{Criterion, criterion_group};
use neural_amp_modeler_rs::common::params::AdaptiveComputeMode;
use neural_amp_modeler_rs::common::spsc::RtStatusFlags;
use neural_amp_modeler_rs::dsp::adaptive::AdaptiveCompute;
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::dsp::pipeline::{
    BridgeBuffer, DspBridge, DspBridgeWriter, DspBuffers, DspPipelineContext, MAX_BRIDGE_BUF,
    MAX_RESAMP_BUF, capture_dsp_pipeline,
};
use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::models::NamModel;

macro_rules! regression_bench {
    ($c:expr, $label:expr, $file:expr) => {
        let mut model = common::load_and_prewarm_required($file);
        let out_channels = match &model {
            neural_amp_modeler_rs::models::StaticModel::ConvNet(c) => c.out_channels(),
            _ => 1,
        };
        let input = common::generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64 * out_channels];
        $c.bench_function($label, |b| {
            b.iter(|| {
                model.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                )
            });
        });
    };
}

// ── Model Inference Core: Standard Catalog SKUs ──────────────────────────────

fn bench_wavenet_standard(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Std_CH16", "BossWN-standard.nam");
}

fn bench_wavenet_feather(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Feather_CH8", "BossWN-feather.nam");
}

fn bench_wavenet_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Lite_CH12", "BossWN-lite.nam");
}

fn bench_wavenet_nano(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Nano_CH4", "BossWN-nano.nam");
}

fn bench_a2_full(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Full_CH8", "wavenet_a2_full.nam");
}

fn bench_a2_lite(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Lite_CH3", "wavenet_a2_lite.nam");
}

fn bench_lstm_1x16(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_1x16", "BossLSTM-1x16.nam");
}

fn bench_lstm_2x8(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_2x8", "BossLSTM-2x8.nam");
}

fn bench_linear(c: &mut Criterion) {
    regression_bench!(c, "RT_Linear", "linear_test.nam");
}

fn bench_convnet(c: &mut Criterion) {
    regression_bench!(c, "RT_ConvNet", "convnet_test.nam");
}

// ── Model Inference Core: Dynamic Models ─────────────────────────────────────

fn bench_wavenet_dyn_free(c: &mut Criterion) {
    regression_bench!(c, "RT_WaveNet_Dyn_Free", "wavenet_dyn_free.nam");
}

fn bench_lstm_dyn_1x7(c: &mut Criterion) {
    regression_bench!(c, "RT_LSTM_Dyn_1x7", "lstm_dyn_test.nam");
}

fn bench_a2_dyn_gated(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Dyn_Gated_CH8", "a2_dynamic_gated_ch8.nam");
}

fn bench_a2_dyn_blended(c: &mut Criterion) {
    regression_bench!(c, "RT_A2_Dyn_Blended_CH3", "a2_dynamic_blended_ch3.nam");
}

// ── DSP Infrastructure ───────────────────────────────────────────────────────

// Sub-µs kernels need batched work per Criterion sample. A single 64-sample
// block is ~0.7–1.4 µs and trips the 2% noise wall from timer jitter alone.
// Batch size is fixed; Criterion params (sample_size/measurement/noise) stay
// canonical. Reported time is for the full batch (not per-block).
const DSP_MICRO_BATCH: usize = 64;

fn bench_dsp_resampler_44k1_to_48k(c: &mut Criterion) {
    let mut rs =
        NamResampler::new(44_100, 48_000, 64).expect("Failed to initialize 44.1k->48k resampler");
    let in_l = common::generate_sine_440hz(64);
    let in_r = common::generate_sine_440hz(64);
    let mut out_l = vec![0.0f32; 128];
    let mut out_r = vec![0.0f32; 128];

    // Prewarm resampler internal delay lines
    for _ in 0..32 {
        rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
    }

    c.bench_function("RT_DSP_Resampler_44k1_to_48k", |b| {
        b.iter(|| {
            for _ in 0..DSP_MICRO_BATCH {
                rs.process_input(
                    std::hint::black_box(&in_l),
                    std::hint::black_box(&in_r),
                    std::hint::black_box(&mut out_l),
                    std::hint::black_box(&mut out_r),
                );
            }
        });
    });
}

fn bench_dsp_resampler_96k_to_48k(c: &mut Criterion) {
    let mut rs =
        NamResampler::new(96_000, 48_000, 64).expect("Failed to initialize 96k->48k resampler");
    let in_l = common::generate_sine_440hz(64);
    let in_r = common::generate_sine_440hz(64);
    let mut out_l = vec![0.0f32; 128];
    let mut out_r = vec![0.0f32; 128];

    // Prewarm resampler internal delay lines
    for _ in 0..32 {
        rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
    }

    c.bench_function("RT_DSP_Resampler_96k_to_48k", |b| {
        b.iter(|| {
            for _ in 0..DSP_MICRO_BATCH {
                rs.process_input(
                    std::hint::black_box(&in_l),
                    std::hint::black_box(&in_r),
                    std::hint::black_box(&mut out_l),
                    std::hint::black_box(&mut out_r),
                );
            }
        });
    });
}

fn bench_dsp_cabsim_ir_medium(c: &mut Criterion) {
    let ir = common::synth_ir(2048, 440.0, 10.0);
    let mut engine = ConvEngine::new(&ir, 64).expect("CabSim ConvEngine allocation failed");

    let input = common::generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    // Prewarm partitions
    for _ in 0..engine.num_partitions().max(1) {
        engine.process(&input, &mut output, None);
    }

    c.bench_function("RT_DSP_CabSim_IR_Medium", |b| {
        b.iter(|| {
            for _ in 0..DSP_MICRO_BATCH {
                engine.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    None,
                );
            }
        });
    });
}

fn bench_dsp_pipeline_helper(c: &mut Criterion, label: &str, os_factor: OversampleFactor) {
    let block_size = 64;
    let model = common::load_and_prewarm_required("BossWN-standard.nam");
    let mut opt_model_l = Some(Box::new(model));
    let mut opt_model_r = None;

    let mut resampler = NamResampler::new(48000, 48000, block_size).expect("Resampler init failed");
    let mut os_engine_l =
        OversampleEngine::new(os_factor, MAX_RESAMP_BUF).expect("OS engine init failed");
    let mut os_engine_r =
        OversampleEngine::new(os_factor, MAX_RESAMP_BUF).expect("OS engine init failed");

    let rt_status = RtStatusFlags::default();
    let mut bridge = Box::new(DspBridge {
        buffers: [
            BridgeBuffer {
                buf_l: [0.0; MAX_BRIDGE_BUF],
                buf_r: [0.0; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
            BridgeBuffer {
                buf_l: [0.0; MAX_BRIDGE_BUF],
                buf_r: [0.0; MAX_BRIDGE_BUF],
                n_samples: 0,
            },
        ],
        active_read_idx: std::sync::atomic::AtomicUsize::new(0),
        generation: std::sync::atomic::AtomicU64::new(0),
        consumed_gen: std::sync::atomic::AtomicU64::new(0),
        dropped_frames: std::sync::atomic::AtomicU32::new(0),
    });

    let mut resamp_mid_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_mid_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut resamp_out_r = vec![0.0f32; MAX_RESAMP_BUF];
    let mut model_out_l = vec![0.0f32; MAX_RESAMP_BUF];
    let mut model_out_r = vec![0.0f32; MAX_RESAMP_BUF];

    let gate_params = GateParams::default();
    let mut silence_hysteresis = DynamicHysteresis::new();
    let mut mono_hysteresis = DynamicHysteresis::new();
    let mut process_mono = true;
    let mut adaptive = AdaptiveCompute::new(AdaptiveComputeMode::Off);

    let mut os_buf: [f32; MAX_RESAMP_BUF * 6] = [0.0f32; MAX_RESAMP_BUF * 6];
    let (os_in_l_slice, rest) = os_buf.split_at_mut(MAX_RESAMP_BUF);
    let (os_in_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_l_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (os_model_r_slice, rest) = rest.split_at_mut(MAX_RESAMP_BUF);
    let (crossfade_scratch_l, crossfade_scratch_r) = rest.split_at_mut(MAX_RESAMP_BUF);

    let template_in_l = common::generate_sine_440hz(block_size);
    let template_in_r = common::generate_sine_440hz(block_size);
    let mut samples_l = template_in_l.clone();
    let mut samples_r = template_in_r.clone();

    // Warm up pipeline
    for _ in 0..16 {
        samples_l.copy_from_slice(&template_in_l);
        samples_r.copy_from_slice(&template_in_r);
        let ctx = DspPipelineContext {
            resampler: &mut resampler,
            os_l: &mut os_engine_l,
            os_r: &mut os_engine_r,
            active_model_l: &mut opt_model_l,
            active_model_r: &mut opt_model_r,
            input_gain_mult: 1.0,
            output_gain_mult: 1.0,
            gate_params: &gate_params,
            silence_hysteresis: &mut silence_hysteresis,
            mono_hysteresis: &mut mono_hysteresis,
            threshold_open_sq: 0.0,
            threshold_close_sq: 0.0,
            process_mono: &mut process_mono,
            rt_status: &rt_status,
            adaptive: &mut adaptive,
            bridge_writer: unsafe { Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge)) },
            conv: None,
        };
        let bufs = DspBuffers {
            resamp_mid_l: &mut resamp_mid_l,
            resamp_mid_r: &mut resamp_mid_r,
            resamp_out_l: &mut resamp_out_l,
            resamp_out_r: &mut resamp_out_r,
            model_out_l: &mut model_out_l,
            model_out_r: &mut model_out_r,
            os_in_l: os_in_l_slice,
            os_in_r: os_in_r_slice,
            os_model_l: os_model_l_slice,
            os_model_r: os_model_r_slice,
            crossfade_scratch_l,
            crossfade_scratch_r,
        };
        capture_dsp_pipeline(&mut samples_l, &mut samples_r, block_size, ctx, bufs, 48000);
    }

    c.bench_function(label, |b| {
        b.iter(|| {
            samples_l.copy_from_slice(&template_in_l);
            samples_r.copy_from_slice(&template_in_r);
            let ctx = DspPipelineContext {
                resampler: &mut resampler,
                os_l: &mut os_engine_l,
                os_r: &mut os_engine_r,
                active_model_l: &mut opt_model_l,
                active_model_r: &mut opt_model_r,
                input_gain_mult: 1.0,
                output_gain_mult: 1.0,
                gate_params: &gate_params,
                silence_hysteresis: &mut silence_hysteresis,
                mono_hysteresis: &mut mono_hysteresis,
                threshold_open_sq: 0.0,
                threshold_close_sq: 0.0,
                process_mono: &mut process_mono,
                rt_status: &rt_status,
                adaptive: &mut adaptive,
                bridge_writer: unsafe {
                    Some(DspBridgeWriter::new(&mut *bridge as *mut DspBridge))
                },
                conv: None,
            };
            let bufs = DspBuffers {
                resamp_mid_l: &mut resamp_mid_l,
                resamp_mid_r: &mut resamp_mid_r,
                resamp_out_l: &mut resamp_out_l,
                resamp_out_r: &mut resamp_out_r,
                model_out_l: &mut model_out_l,
                model_out_r: &mut model_out_r,
                os_in_l: os_in_l_slice,
                os_in_r: os_in_r_slice,
                os_model_l: os_model_l_slice,
                os_model_r: os_model_r_slice,
                crossfade_scratch_l,
                crossfade_scratch_r,
            };
            capture_dsp_pipeline(
                std::hint::black_box(&mut samples_l),
                std::hint::black_box(&mut samples_r),
                block_size,
                ctx,
                bufs,
                48000,
            );
        });
    });
}

fn bench_dsp_pipeline_base_no_os(c: &mut Criterion) {
    bench_dsp_pipeline_helper(c, "RT_DSP_Pipeline_Base_NoOS", OversampleFactor::Off);
}

fn bench_dsp_pipeline_hq_4x_os(c: &mut Criterion) {
    bench_dsp_pipeline_helper(c, "RT_DSP_Pipeline_HQ_4xOS", OversampleFactor::X4);
}

criterion_group!(
    name = regression_gates;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(std::time::Duration::from_secs(5))
        .warm_up_time(std::time::Duration::from_secs(1))
        .noise_threshold(0.02);
    targets =
        bench_wavenet_standard,
        bench_wavenet_feather,
        bench_wavenet_lite,
        bench_wavenet_nano,
        bench_a2_full,
        bench_a2_lite,
        bench_lstm_1x16,
        bench_lstm_2x8,
        bench_linear,
        bench_convnet,
        bench_wavenet_dyn_free,
        bench_lstm_dyn_1x7,
        bench_a2_dyn_gated,
        bench_a2_dyn_blended,
        bench_dsp_resampler_44k1_to_48k,
        bench_dsp_resampler_96k_to_48k,
        bench_dsp_cabsim_ir_medium,
        bench_dsp_pipeline_base_no_os,
        bench_dsp_pipeline_hq_4x_os,
);

fn main() {
    let _guard = neural_amp_modeler_rs::testing::ForceAvx2Guard::new();
    regression_gates();
    Criterion::default().configure_from_args().final_summary();
}
