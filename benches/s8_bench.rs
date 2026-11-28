// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Sprint S8 measurement bench — isolated copies and calculation candidates.
//!
//! Covers the two S8 tasks:
//!
//! * **T8.1 (data traffic):** mono vs stereo pipeline, bridge write, cab-sim
//!   adapter FIFO/engine copies, IR sizes, block sizes and tail drain.
//! * **T8.2 (micro-optimisations):** `WaveNetLayerState::advance_frames`
//!   division cost, A2 output pre-fill, static WaveNet full chain and dynamic
//!   LSTM prefetch path.
//!
//! Recalibration protocol: `docs/benchmarks.md` §6.
//!
//! ```sh
//! taskset -c <isolated-cpu> cargo bench --bench s8_bench -- --save-baseline s8-before
//! # ...toggle one candidate only...
//! taskset -c <isolated-cpu> cargo bench --bench s8_bench -- <filter> --baseline s8-before
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::common::params::AdaptiveComputeMode;
use neural_amp_modeler_rs::common::spsc::RtStatusFlags;
use neural_amp_modeler_rs::dsp::adaptive::AdaptiveCompute;
use neural_amp_modeler_rs::dsp::cabsim::adapter::{CabSimAdapter, CabSimPair};
use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;
use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::dsp::pipeline::{
    DspBridge, DspBridgeWriter, DspBuffers, DspPipelineContext, MAX_RESAMP_BUF,
    capture_dsp_pipeline,
};
use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::dsp::smoother::ParamSmoother;
use neural_amp_modeler_rs::math::common::scalar_ref::dot_product_f32_native_kahan4;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::wavenet::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState};
use std::hint::black_box;
use std::sync::atomic::Ordering;

mod common;

// ─────────────────────────────────────────────────────────────────────────────
// T8.2 — advance_frames integer division
// ─────────────────────────────────────────────────────────────────────────────

fn bench_advance_frames(c: &mut Criterion) {
    let mut group = c.benchmark_group("s8_advance_frames");
    for &channels in &[4usize, 8, 12, 16] {
        let mut state =
            WaveNetLayerState::new(channels, 4096, 0).expect("WaveNetLayerState allocation");
        group.bench_with_input(
            BenchmarkId::new("advance_64frames", channels),
            &channels,
            |b, &_ch| {
                b.iter(|| {
                    state.advance_frames(black_box(WAVENET_MAX_NUM_FRAMES), black_box(channels));
                    black_box(state.buffer_start);
                });
            },
        );
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.2 — isolated cost of the remaining candidates
// ─────────────────────────────────────────────────────────────────────────────

fn bench_div_compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("s8_div_compare");
    let size = 50_000usize;
    let channels = 16usize;
    let frames = size / channels;
    group.bench_function("runtime_div", |b| {
        b.iter(|| black_box(black_box(size) / black_box(channels)));
    });
    group.bench_function("cached", |b| {
        b.iter(|| black_box(frames));
    });
    group.finish();
}

fn bench_smoother_tick64(c: &mut Criterion) {
    let mut smoother = ParamSmoother::new(0.0, 48_000.0, 20.0);
    smoother.set_target(1.0);
    c.bench_function("s8_smoother_tick64", |b| {
        b.iter(|| {
            let mut acc = 0.0f32;
            for _ in 0..64 {
                acc += smoother.tick();
            }
            black_box(acc);
        });
    });
}

fn bench_kahan_head(c: &mut Criterion) {
    let mut group = c.benchmark_group("s8_kahan_head");
    for &h in &[16usize, 32] {
        let weights = vec![0.01f32; 4 * h];
        let state = vec![0.5f32; h];
        group.bench_with_input(BenchmarkId::new("dot_kahan4", h), &h, |b, &_h| {
            b.iter(|| {
                let mut out = 0.0f32;
                for chunk in weights.chunks_exact(4 * h) {
                    out += dot_product_f32_native_kahan4(black_box(&state), black_box(chunk));
                }
                black_box(out);
            });
        });
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.2 — model paths (dirty output proves full overwrite)
// ─────────────────────────────────────────────────────────────────────────────

fn bench_model_process(c: &mut Criterion, label: &str, fixture: &str, out_channels: usize) {
    let mut model = common::load_and_prewarm_required(fixture);
    let input = common::generate_sine_440hz(64);
    let mut output = vec![0.5f32; 64 * out_channels];
    for _ in 0..8 {
        model.process(&input, &mut output);
    }
    c.bench_function(label, |b| {
        b.iter(|| {
            output.fill(0.5);
            model.process(black_box(&input), black_box(&mut output));
        });
    });
}

fn bench_wavenet_std(c: &mut Criterion) {
    bench_model_process(c, "s8_wavenet_std_ch16", "BossWN-standard.nam", 1);
}

fn bench_a2_full(c: &mut Criterion) {
    bench_model_process(c, "s8_a2_full_ch8", "wavenet_a2_full.nam", 1);
}

fn bench_lstm_dyn(c: &mut Criterion) {
    bench_model_process(c, "s8_lstm_dyn_1x7", "lstm_dyn_test.nam", 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.1 — bridge write mono vs stereo
// ─────────────────────────────────────────────────────────────────────────────

fn bench_bridge_write(c: &mut Criterion) {
    let mut group = c.benchmark_group("s8_bridge_write");
    for &block_size in &[64usize, 128, 256] {
        let input_l = vec![0.5f32; block_size];
        let input_r = vec![0.25f32; block_size];
        for mono in [true, false] {
            let mut bridge = DspBridge::new_boxed();
            let bridge_ptr = &raw mut *bridge;
            let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
            group.bench_with_input(
                BenchmarkId::new(if mono { "mono" } else { "stereo" }, block_size),
                &block_size,
                |b, &bs| {
                    b.iter(|| {
                        bridge
                            .consumed_gen
                            .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
                        writer.write_block(
                            black_box(&input_l),
                            black_box(&input_r),
                            black_box(bs),
                            black_box(mono),
                        );
                    });
                },
            );
        }
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.1 — cab-sim adapter FIFO vs raw engine
// ─────────────────────────────────────────────────────────────────────────────

fn make_adapter(ir_len: usize, partition: usize) -> CabSimAdapter {
    let ir = common::synth_ir(ir_len, 440.0, 10.0);
    let engine = ConvEngine::new(&ir, partition).expect("cabsim engine allocation");
    CabSimAdapter::new(Box::new(engine)).expect("cabsim adapter allocation")
}

fn bench_cabsim_adapter(c: &mut Criterion) {
    let mut group = c.benchmark_group("s8_cabsim_adapter");
    for &(ir_len, partition) in &[(64usize, 64usize), (2048, 64), (16384, 64), (2048, 256)] {
        let mut input = common::generate_sine_440hz(partition);
        let label = format!("ir{ir_len}_p{partition}");

        let mut engine = {
            let ir = common::synth_ir(ir_len, 440.0, 10.0);
            ConvEngine::new(&ir, partition).expect("cabsim engine allocation")
        };
        let engine_in = input.clone();
        let mut engine_out = vec![0.0f32; partition];
        for _ in 0..engine.num_partitions().max(1) {
            engine.process(&engine_in, &mut engine_out, None);
        }
        group.bench_with_input(BenchmarkId::new("raw_engine", &label), &label, |b, _| {
            b.iter(|| {
                engine.process(
                    black_box(&engine_in),
                    black_box(&mut engine_out),
                    black_box(None),
                );
            });
        });

        let mut adapter = make_adapter(ir_len, partition);
        for _ in 0..16 {
            adapter.process_block(&mut input, None);
        }
        group.bench_with_input(BenchmarkId::new("adapter", &label), &label, |b, _| {
            b.iter(|| {
                adapter.process_block(black_box(&mut input), black_box(None));
            });
        });

        adapter.rearm_tail();
        let mut tail = vec![0.0f32; 512];
        group.bench_with_input(
            BenchmarkId::new("adapter_tail512", &label),
            &label,
            |b, _| {
                b.iter(|| {
                    adapter.rearm_tail();
                    adapter.drain_tail(black_box(&mut tail), black_box(None));
                });
            },
        );
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// T8.1 — pipeline mono vs stereo, with and without cab-sim
// ─────────────────────────────────────────────────────────────────────────────

struct PipelineFixture {
    resampler: NamResampler,
    os_l: OversampleEngine,
    os_r: OversampleEngine,
    model_l: Option<Box<neural_amp_modeler_rs::models::StaticModel>>,
    model_r: Option<Box<neural_amp_modeler_rs::models::StaticModel>>,
    gate_params: GateParams,
    silence_hysteresis: DynamicHysteresis,
    mono_hysteresis: DynamicHysteresis,
    process_mono: bool,
    rt_status: RtStatusFlags,
    adaptive: AdaptiveCompute,
    resamp_mid_l: Vec<f32>,
    resamp_mid_r: Vec<f32>,
    resamp_out_l: Vec<f32>,
    resamp_out_r: Vec<f32>,
    model_out_l: Vec<f32>,
    model_out_r: Vec<f32>,
    os_in_l: Vec<f32>,
    os_in_r: Vec<f32>,
    os_model_l: Vec<f32>,
    os_model_r: Vec<f32>,
    crossfade_scratch_l: Vec<f32>,
    crossfade_scratch_r: Vec<f32>,
}

fn pipeline_fixture(mono: bool) -> PipelineFixture {
    let model_l = common::load_and_prewarm_required("BossWN-standard.nam");
    let model_r = if mono {
        None
    } else {
        Some(Box::new(common::load_and_prewarm_required(
            "BossWN-standard.nam",
        )))
    };
    PipelineFixture {
        resampler: NamResampler::new_simple(48000, 48000).expect("resampler"),
        os_l: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os l"),
        os_r: OversampleEngine::new(OversampleFactor::Off, MAX_RESAMP_BUF).expect("os r"),
        model_l: Some(Box::new(model_l)),
        model_r,
        gate_params: GateParams::default(),
        silence_hysteresis: DynamicHysteresis::new(),
        mono_hysteresis: DynamicHysteresis::new(),
        process_mono: mono,
        rt_status: RtStatusFlags::default(),
        adaptive: AdaptiveCompute::new(AdaptiveComputeMode::Off),
        resamp_mid_l: vec![0.0; MAX_RESAMP_BUF],
        resamp_mid_r: vec![0.0; MAX_RESAMP_BUF],
        resamp_out_l: vec![0.0; MAX_RESAMP_BUF],
        resamp_out_r: vec![0.0; MAX_RESAMP_BUF],
        model_out_l: vec![0.0; MAX_RESAMP_BUF],
        model_out_r: vec![0.0; MAX_RESAMP_BUF],
        os_in_l: vec![0.0; MAX_RESAMP_BUF],
        os_in_r: vec![0.0; MAX_RESAMP_BUF],
        os_model_l: vec![0.0; MAX_RESAMP_BUF],
        os_model_r: vec![0.0; MAX_RESAMP_BUF],
        crossfade_scratch_l: vec![0.0; MAX_RESAMP_BUF],
        crossfade_scratch_r: vec![0.0; MAX_RESAMP_BUF],
    }
}

fn bench_pipeline(c: &mut Criterion) {
    for mono in [true, false] {
        for cab in [false, true] {
            let mut f = pipeline_fixture(mono);
            let mut cab_mono = cab.then(|| make_adapter(2048, 64));
            let mut cab_pair = cab.then(|| CabSimPair {
                l: Box::new(make_adapter(2048, 64)),
                r: Box::new(make_adapter(2048, 64)),
                sample_rate: 48_000,
            });

            let in_l = common::generate_sine_440hz(64);
            let in_r: Vec<f32> = (0..64)
                .map(|i| (i as f64 * 2.0 * std::f64::consts::PI * 880.0 / 48_000.0).sin() as f32)
                .collect();
            let mut samples_l = in_l.clone();
            let mut samples_r = in_r.clone();

            let run = |f: &mut PipelineFixture,
                       samples_l: &mut [f32],
                       samples_r: &mut [f32],
                       cab_mono: &mut Option<CabSimAdapter>,
                       cab_pair: &mut Option<CabSimPair>| {
                let ctx = DspPipelineContext {
                    resampler: &mut f.resampler,
                    os_l: &mut f.os_l,
                    os_r: &mut f.os_r,
                    active_model_l: &mut f.model_l,
                    active_model_r: &mut f.model_r,
                    input_gain_mult: 1.0,
                    output_gain_mult: 1.0,
                    gate_params: &f.gate_params,
                    silence_hysteresis: &mut f.silence_hysteresis,
                    mono_hysteresis: &mut f.mono_hysteresis,
                    threshold_open_sq: 0.0,
                    threshold_close_sq: 0.0,
                    process_mono: &mut f.process_mono,
                    rt_status: &f.rt_status,
                    adaptive: &mut f.adaptive,
                    bridge_writer: None,
                    conv: cab_mono.as_mut(),
                    conv_pair: cab_pair.as_mut(),
                };
                let bufs = DspBuffers {
                    resamp_mid_l: &mut f.resamp_mid_l,
                    resamp_mid_r: &mut f.resamp_mid_r,
                    resamp_out_l: &mut f.resamp_out_l,
                    resamp_out_r: &mut f.resamp_out_r,
                    model_out_l: &mut f.model_out_l,
                    model_out_r: &mut f.model_out_r,
                    os_in_l: &mut f.os_in_l,
                    os_in_r: &mut f.os_in_r,
                    os_model_l: &mut f.os_model_l,
                    os_model_r: &mut f.os_model_r,
                    crossfade_scratch_l: &mut f.crossfade_scratch_l,
                    crossfade_scratch_r: &mut f.crossfade_scratch_r,
                };
                capture_dsp_pipeline(samples_l, samples_r, 64, ctx, bufs, 48000);
            };

            for _ in 0..16 {
                samples_l.copy_from_slice(&in_l);
                samples_r.copy_from_slice(&in_r);
                run(
                    &mut f,
                    &mut samples_l,
                    &mut samples_r,
                    &mut cab_mono,
                    &mut cab_pair,
                );
            }

            let label = format!(
                "s8_pipeline_{}_{}",
                if mono { "mono" } else { "stereo" },
                if cab { "cab" } else { "nocab" }
            );
            c.bench_function(&label, |b| {
                b.iter(|| {
                    samples_l.copy_from_slice(&in_l);
                    samples_r.copy_from_slice(&in_r);
                    run(
                        &mut f,
                        black_box(&mut samples_l),
                        black_box(&mut samples_r),
                        &mut cab_mono,
                        &mut cab_pair,
                    );
                });
            });
        }
    }
}

criterion_group! {
    name = s8_benches;
    config = Criterion::default().sample_size(50).noise_threshold(0.05);
    targets = bench_advance_frames,
    bench_div_compare,
    bench_smoother_tick64,
    bench_kahan_head,
    bench_wavenet_std,
    bench_a2_full,
    bench_lstm_dyn,
    bench_bridge_write,
    bench_cabsim_adapter,
    bench_pipeline
}

criterion_main!(s8_benches);
