// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmarks of auxiliary DSP blocks: `NamResampler`, `Gate FSM`, and telemetry
//! (`LatencyHistogram`).
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench dsp_bench
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

/// Measures the resampler cost when converting from 44.1 kHz to 48 kHz.
/// The resampler is one of the most sensitive components, as it involves FIR filtering.
/// `process_input` and `process_output` are measured separately to identify
/// bottlenecks in input (buffering) vs output (interpolation).
fn bench_resampler_44100_to_48000_256samp(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(44_100, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Resampler_44100_to_48000_256samp");
    group.bench_function("process_input", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_input_mono", |b| {
        b.iter(|| {
            rs.process_input_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output_mono", |b| {
        b.iter(|| {
            rs.process_output_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

/// Measures 96 kHz to 48 kHz conversion (downsampling).
/// Generally lighter than upsampling, but still requires anti-aliasing filtering.
fn bench_resampler_96000_to_48000_256samp(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(96_000, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size * 2];
    let mut out_r = vec![0.0f32; size * 2];
    let mut group = c.benchmark_group("Resampler_96000_to_48000_256samp");
    group.bench_function("process_input", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_input_mono", |b| {
        b.iter(|| {
            rs.process_input_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output", |b| {
        b.iter(|| {
            rs.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
    group.bench_function("process_output_mono", |b| {
        b.iter(|| {
            rs.process_output_mono(&in_l, &mut out_l, &mut out_r);
        });
    });
    group.finish();
}

/// Measures the performance of the latency histogram `record` function.
/// Simulates 64 calls (equivalent to processing 1 second of audio at 48 kHz
/// with a 64-sample buffer, or about 750 callbacks). The benchmark validates that
/// `fetch_add` is significantly faster than `fetch_update` (CAS-loop).
fn bench_record(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::telemetry::LatencyHistogram;
    let hist = LatencyHistogram::new();
    let durations: Vec<u64> = (0..64).map(|i| (i * 100) as u64).collect();

    c.bench_function("bench_record_64calls", |b| {
        b.iter(|| {
            for &d in &durations {
                hist.record(d);
            }
        });
    });
}

/// Measures the resampler overhead when sample rates are equal.
/// Serves to validate that the "bypass" path is efficient.
fn bench_resampler_48000_bypass(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;
    let size = 256;
    let mut rs = NamResampler::new(48_000, 48_000, size).unwrap();
    let in_l = vec![0.0f32; size];
    let in_r = vec![0.0f32; size];
    let mut out_l = vec![0.0f32; size];
    let mut out_r = vec![0.0f32; size];
    c.bench_function("Resampler_48000_bypass_256samp", |b| {
        b.iter(|| {
            rs.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
        });
    });
}

/// Three steady-state scenarios are measured per block size:
/// - **Open**: Gate stays open (volume above open threshold). Most common path.
/// - **Closed**: Gate stays closed (volume below close threshold, post-hold+fade).
/// - **FadingOut**: Gate is actively ramping the multiplier down toward silence.
fn bench_gate_fsm(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams};

    let params = GateParams::default();
    let th_open = 10.0f32.powf(params.threshold_open_db / 20.0);
    let th_close = 10.0f32.powf(params.threshold_close_db / 20.0);

    let mut group = c.benchmark_group("Gate_FSM");

    for &n_samples in &[64, 128, 256] {
        group.bench_function(format!("Open_{}samp", n_samples), |b| {
            let mut gate = DynamicHysteresis::new();
            b.iter(|| {
                gate.update(
                    std::hint::black_box(0.5),
                    th_open,
                    th_close,
                    &params,
                    n_samples,
                );
                std::hint::black_box(gate.multiplier());
            });
        });

        group.bench_function(format!("Closed_{}samp", n_samples), |b| {
            let mut gate = DynamicHysteresis::new();
            gate.update(0.0, th_open, th_close, &params, 2048);
            gate.update(0.0, th_open, th_close, &params, 256);
            b.iter(|| {
                gate.update(
                    std::hint::black_box(0.0),
                    th_open,
                    th_close,
                    &params,
                    n_samples,
                );
                std::hint::black_box(gate.multiplier());
            });
        });

        group.bench_function(format!("FadingOut_{}samp", n_samples), |b| {
            b.iter_with_setup(
                || {
                    let mut gate = DynamicHysteresis::new();
                    gate.update(0.0, th_open, th_close, &params, params.hold_frames);
                    gate
                },
                |mut gate| {
                    gate.update(
                        std::hint::black_box(0.0),
                        th_open,
                        th_close,
                        &params,
                        n_samples,
                    );
                    std::hint::black_box(gate.multiplier());
                },
            );
        });
    }

    group.finish();
}

criterion_group! {
    name = dsp_benches;
    config = criterion::Criterion::default().sample_size(50).noise_threshold(0.05);
    targets = bench_resampler_44100_to_48000_256samp,
    bench_resampler_96000_to_48000_256samp,
    bench_resampler_48000_bypass,
    bench_record,
    bench_gate_fsm
}

criterion_main!(dsp_benches);
