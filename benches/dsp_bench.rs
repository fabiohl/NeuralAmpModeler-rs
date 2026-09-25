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
    let mut rs = NamResampler::new_simple(44_100, 48_000).unwrap();
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
    let mut rs = NamResampler::new_simple(96_000, 48_000).unwrap();
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
    let mut rs = NamResampler::new_simple(48_000, 48_000).unwrap();
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

/// Measures the isolated cost of X2Stage upsampling and downsampling (F-PERF-15.1).
///
/// Benchmarks half-band FIR polyphase filtering at typical audio buffer sizes
/// (64, 128, 256 samples).
fn bench_x2stage_isolated(c: &mut Criterion) {
    use neural_amp_modeler_rs::dsp::stage::X2Stage;

    let mut group = c.benchmark_group("X2Stage_isolated");

    for &n_samples in &[64, 128, 256] {
        let in_up = vec![0.1f32; n_samples];
        let mut out_up = vec![0.0f32; n_samples * 2];

        let in_down = vec![0.1f32; n_samples * 2];
        let mut out_down = vec![0.0f32; n_samples];

        group.bench_function(format!("upsample_{}samp", n_samples), |b| {
            let mut stage = X2Stage::new().unwrap();
            b.iter(|| {
                std::hint::black_box(stage.upsample(
                    std::hint::black_box(&in_up),
                    std::hint::black_box(&mut out_up),
                ));
            });
        });

        group.bench_function(format!("downsample_{}samp", n_samples), |b| {
            let mut stage = X2Stage::new().unwrap();
            b.iter(|| {
                std::hint::black_box(stage.downsample(
                    std::hint::black_box(&in_down),
                    std::hint::black_box(&mut out_down),
                ));
            });
        });
    }

    group.finish();
}

/// Measures isolated stages of the real-to-complex FFT across the effective FFT
/// sizes of the production RFFT consumers (F-PERF-15.4, F-PERF-28).
///
/// Effective sizes resolved from the planners on this codebase:
/// - `linear_test.nam` (the `RT_Linear_Direct_RF4` bench fixture): receptive field 4 is below
///   `FFT_AUTO_THRESHOLD` (256) in `src/models/linear.rs`, so the model resolves to
///   `LinearMode::Direct` — **no RFFT runs on its hot path**.
/// - Linear FFT-hybrid path (`receptive_field >= 256`): `N = 2P` with
///   `P = select_partition_size(RF)` = largest power of two ≤ `RF/2` — an RF=2048
///   model uses N=2048 (`src/models/linear_fft.rs`).
/// - CabSim `ConvEngine::new(ir, 64)` (block 64): `N = (2 × 64).next_power_of_two() = 128`.
///
/// The group covers N ∈ {64, 128, 256, 512, 1024} so per-stage cost is measured
/// per size instead of a single arbitrary size (the previous group covered only
/// N=512, which is why a small-size RFFT regression went unmeasured). Breaks the
/// pipeline into its constituent micro-kernels:
/// - `pack_re_im`: Interleaving/packing real input into complex half-size scratch.
/// - `post_twiddle`: Post-processing complex half-size spectrum via Hermitian symmetry.
/// - `pre_twiddle`: Pre-processing complex spectrum for half-size inverse FFT.
/// - `unpack_re_im`: Unpacking complex half-size scratch into real output.
/// - Full forward and inverse transforms for context.
fn bench_rfft_stages_by_size(c: &mut Criterion) {
    use neural_amp_modeler_rs::math::dsp::fft::RfftPlanner;

    const SIZES: [usize; 5] = [64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("Rfft_stages_by_size");

    for n in SIZES {
        let n_half = n / 2;
        let mut rfft = RfftPlanner::<f32>::new(n);

        let input: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05).sin()).collect();
        let mut out_re = vec![0.0f32; n_half + 1];
        let mut out_im = vec![0.0f32; n_half + 1];
        let mut inv_out = vec![0.0f32; n];

        group.bench_function(format!("{n}/pack_re_im"), |b| {
            b.iter(|| {
                rfft.pack_re_im(std::hint::black_box(&input));
                std::hint::black_box(rfft.scratch_buffers_mut().0[0]);
            });
        });

        group.bench_function(format!("{n}/post_twiddle"), |b| {
            rfft.pack_re_im(&input);
            b.iter(|| {
                rfft.post_twiddle(
                    std::hint::black_box(&mut out_re),
                    std::hint::black_box(&mut out_im),
                );
                std::hint::black_box(out_re[0]);
            });
        });

        group.bench_function(format!("{n}/pre_twiddle"), |b| {
            rfft.pack_re_im(&input);
            rfft.post_twiddle(&mut out_re, &mut out_im);
            let base_re = out_re.clone();
            let base_im = out_im.clone();
            let mut work_re = base_re.clone();
            let mut work_im = base_im.clone();
            b.iter(|| {
                work_re.copy_from_slice(&base_re);
                work_im.copy_from_slice(&base_im);
                rfft.pre_twiddle(
                    std::hint::black_box(&mut work_re),
                    std::hint::black_box(&mut work_im),
                );
                std::hint::black_box(work_re[0]);
            });
        });

        group.bench_function(format!("{n}/unpack_re_im"), |b| {
            let in_re = vec![0.2f32; n_half];
            let in_im = vec![0.3f32; n_half];
            b.iter(|| {
                rfft.unpack_re_im(
                    std::hint::black_box(&in_re),
                    std::hint::black_box(&in_im),
                    std::hint::black_box(&mut inv_out),
                );
                std::hint::black_box(inv_out[0]);
            });
        });

        group.bench_function(format!("{n}/process_forward_full"), |b| {
            b.iter(|| {
                rfft.process_forward(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut out_re),
                    std::hint::black_box(&mut out_im),
                );
                std::hint::black_box(out_re[0]);
            });
        });

        group.bench_function(format!("{n}/process_inverse_full"), |b| {
            rfft.pack_re_im(&input);
            rfft.post_twiddle(&mut out_re, &mut out_im);
            let base_re = out_re.clone();
            let base_im = out_im.clone();
            let mut work_re = base_re.clone();
            let mut work_im = base_im.clone();
            b.iter(|| {
                work_re.copy_from_slice(&base_re);
                work_im.copy_from_slice(&base_im);
                rfft.process_inverse(
                    std::hint::black_box(&mut work_re),
                    std::hint::black_box(&mut work_im),
                    std::hint::black_box(&mut inv_out),
                );
                std::hint::black_box(inv_out[0]);
            });
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
    bench_gate_fsm,
    bench_x2stage_isolated,
    bench_rfft_stages_by_size
}

criterion_main!(dsp_benches);
