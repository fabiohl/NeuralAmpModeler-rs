// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance benchmarks for the Linear model convolution modes.
//!
//! Compares Direct (time-domain dot product) vs FFT (partitioned overlap-save)
//! across receptive field sizes from 128 to 8192 taps, at the standard 64-sample
//! DSP block size.
//!
//! ## Purpose
//!
//! These benchmarks validate the optimal auto-selection threshold
//! (`FFT_AUTO_THRESHOLD = 256`). The threshold is correct if:
//! - Direct is faster for RF < 256 (FFT overhead dominates)
//! - FFT is faster for RF ≥ 256 (crossing point)
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench linear
//! ```
//!
//! ## Interpreting results
//!
//! | Metric               | Meaning                                          |
//! |----------------------|--------------------------------------------------|
//! | `per block`          | Time to process 64 samples (1 DSP block)         |
//! | `per sample`         | Amortized time per individual sample             |
//! | Real-time deadline   | 1.33 ms at 48 kHz with 64-sample buffer          |

use criterion::{Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::loader::nam_json::{LinearImplementation, LinearTopology};
use neural_amp_modeler_rs::models::linear::{LinearMode, LinearModel};

mod common;

/// Receptive field sizes to benchmark.
const RF_SIZES: &[usize] = &[128, 256, 512, 1024, 2048, 4096, 8192];

/// DSP block size (64 samples at 48 kHz = 1.33 ms deadline).
const BLOCK_SIZE: usize = 64;

/// Benchmarks Direct vs FFT per-block processing time for each receptive
/// field size. Each RF size gets a comparison group with two entries:
/// `Direct` and `FFT`.
fn bench_direct_vs_fft_per_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Direct_vs_FFT_per_block");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let input = common::generate_sine_440hz(BLOCK_SIZE);
        let mut output_direct = vec![0.0f32; BLOCK_SIZE];
        let mut output_fft = vec![0.0f32; BLOCK_SIZE];

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_direct.process(&input, &mut output_direct);
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_fft.process(&input, &mut output_fft);
            });
        });
    }

    group.finish();
}

/// Benchmarks Direct vs FFT per-sample time by processing 1024 samples
/// in 1-sample blocks. This measures the per-sample dispatch overhead
/// including block boundary detection in the FFT path.
fn bench_direct_vs_fft_per_sample(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Direct_vs_FFT_per_sample");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let inputs: Vec<f32> = (0..1024)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / 48000.0).sin())
            .collect();

        let mut sample_idx = 0usize;

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| {
                let x = inputs[sample_idx % inputs.len()];
                sample_idx = sample_idx.wrapping_add(1);
                let mut out = 0.0f32;
                unsafe { model_direct.process(&[x], std::slice::from_mut(&mut out)) };
                out
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| {
                let x = inputs[sample_idx % inputs.len()];
                sample_idx = sample_idx.wrapping_add(1);
                let mut out = 0.0f32;
                unsafe { model_fft.process(&[x], std::slice::from_mut(&mut out)) };
                out
            });
        });
    }

    group.finish();
}

/// Benchmarks the FFT prewarm cost across RF sizes.
/// Prewarm zeroes history and resets the FFT state (allocation-free,
/// but involves large zero-fill operations).
fn bench_fft_prewarm(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_FFT_Prewarm");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        group.bench_function(format!("Prewarm_RF{rf}"), |b| {
            b.iter_with_setup(
                || LinearModel::new(ir.clone(), 0.1, LinearImplementation::Fft).unwrap(),
                |mut model| {
                    model.prewarm(std::hint::black_box(4096));
                },
            );
        });
    }

    group.finish();
}

/// Measures FFT tail block processing time in isolation.
///
/// This isolates the cost of `process_tail_block` (FFT + SIMD complex MAC +
/// IFFT) from the per-sample head convolution, showing how the FFT cost
/// scales with partition count.
fn bench_fft_tail_block(c: &mut Criterion) {
    let group_name = "Linear_FFT_TailBlock";
    let mut group = c.benchmark_group(group_name);

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);
        let mut model = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model.prewarm(4096);

        if let LinearMode::Fft(ref mut state) = model.mode {
            let p = state.p;
            let window: Vec<f32> = (0..(2 * p)).map(|i| (i as f32 * 0.1).sin()).collect();

            group.bench_function(format!("TailBlock_RF{rf}_P{p}"), |b| {
                b.iter(|| {
                    state.process_tail_block(&window);
                });
            });
        }
    }

    group.finish();
}

/// Measures the processing cost of a large block across RF sizes.
/// A 4096-sample block exercises block boundary crossings and cache
/// behavior, providing a stress test perspective.
fn bench_large_block_4096(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_LargeBlock_4096samp");

    for &rf in RF_SIZES {
        let ir = common::synth_ir(rf, 880.0, 8.0);

        let mut model_direct =
            LinearModel::new(ir.clone(), 0.1, LinearImplementation::Direct).unwrap();
        model_direct.prewarm(4096);

        let mut model_fft = LinearModel::new(ir, 0.1, LinearImplementation::Fft).unwrap();
        model_fft.prewarm(4096);

        let input = common::generate_sine_440hz(4096);
        let mut output_direct = vec![0.0f32; 4096];
        let mut output_fft = vec![0.0f32; 4096];

        group.bench_function(format!("Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_direct.process(&input, &mut output_direct);
            });
        });

        group.bench_function(format!("FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                model_fft.process(&input, &mut output_fft);
            });
        });
    }

    group.finish();
}

/// Helper to construct a synthetic Linear model for multichannel benchmarks.
fn make_multichannel_model(
    in_channels: usize,
    out_channels: usize,
    receptive_field: usize,
    implementation: LinearImplementation,
) -> LinearModel {
    let topo = LinearTopology {
        in_channels,
        out_channels,
        receptive_field,
        has_bias: true,
        implementation,
    };
    let num_kernels = topo.num_kernels();
    let num_biases = topo.num_biases();
    let weights: Vec<f32> = (0..num_kernels * receptive_field)
        .map(|i| ((i as f32 * 0.05).sin()) * 0.1)
        .collect();
    let biases = vec![0.05f32; num_biases];
    let mut model = LinearModel::new_with_topology(topo, weights, biases)
        .expect("LinearModel::new_with_topology should succeed");
    model.prewarm(4096);
    model
}

/// Benchmarks per-block processing across multichannel geometries (1->1, 1->2, 2->1, 2->2)
/// for both Direct and FFT implementations at the standard 64-sample block size.
fn bench_multichannel_geometries_per_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Multichannel_Geometries_64samp");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_millis(1500));
    group.sample_size(30);

    let mc_rf_sizes: &[usize] = &[128, 256, 512, 2048];

    for &rf in mc_rf_sizes {
        // --- 1 -> 1 Mono Legacy ---
        let mut mono_direct = make_multichannel_model(1, 1, rf, LinearImplementation::Direct);
        let mut mono_fft = make_multichannel_model(1, 1, rf, LinearImplementation::Fft);
        let mono_in = common::generate_sine_440hz(BLOCK_SIZE);
        let mut mono_out_direct = vec![0.0f32; BLOCK_SIZE];
        let mut mono_out_fft = vec![0.0f32; BLOCK_SIZE];

        group.bench_function(format!("Mono_1x1_Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mono_direct.process(&mono_in, &mut mono_out_direct);
            });
        });
        group.bench_function(format!("Mono_1x1_FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mono_fft.process(&mono_in, &mut mono_out_fft);
            });
        });

        // --- 1 -> 2 OneToMany ---
        let mut otm_direct = make_multichannel_model(1, 2, rf, LinearImplementation::Direct);
        let mut otm_fft = make_multichannel_model(1, 2, rf, LinearImplementation::Fft);
        let otm_in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut otm_out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let mut otm_out_ch1 = vec![0.0f32; BLOCK_SIZE];
        let otm_in_ptrs: [*const f32; 1] = [otm_in_ch0.as_ptr()];
        let otm_out_ptrs: [*mut f32; 2] = [otm_out_ch0.as_mut_ptr(), otm_out_ch1.as_mut_ptr()];

        group.bench_function(format!("OneToMany_1x2_Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                otm_direct.process_raw(otm_in_ptrs.as_ptr(), otm_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
        group.bench_function(format!("OneToMany_1x2_FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                otm_fft.process_raw(otm_in_ptrs.as_ptr(), otm_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });

        // --- 2 -> 1 ManyToOne ---
        let mut mto_direct = make_multichannel_model(2, 1, rf, LinearImplementation::Direct);
        let mut mto_fft = make_multichannel_model(2, 1, rf, LinearImplementation::Fft);
        let mto_in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let mto_in_ch1 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut mto_out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let mto_in_ptrs: [*const f32; 2] = [mto_in_ch0.as_ptr(), mto_in_ch1.as_ptr()];
        let mto_out_ptrs: [*mut f32; 1] = [mto_out_ch0.as_mut_ptr()];

        group.bench_function(format!("ManyToOne_2x1_Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mto_direct.process_raw(mto_in_ptrs.as_ptr(), mto_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
        group.bench_function(format!("ManyToOne_2x1_FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mto_fft.process_raw(mto_in_ptrs.as_ptr(), mto_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });

        // --- 2 -> 2 ManyToManyShared ---
        let mut mtm_direct = make_multichannel_model(2, 2, rf, LinearImplementation::Direct);
        let mut mtm_fft = make_multichannel_model(2, 2, rf, LinearImplementation::Fft);
        let mtm_in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let mtm_in_ch1 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut mtm_out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let mut mtm_out_ch1 = vec![0.0f32; BLOCK_SIZE];
        let mtm_in_ptrs: [*const f32; 2] = [mtm_in_ch0.as_ptr(), mtm_in_ch1.as_ptr()];
        let mtm_out_ptrs: [*mut f32; 2] = [mtm_out_ch0.as_mut_ptr(), mtm_out_ch1.as_mut_ptr()];

        group.bench_function(format!("ManyToManyShared_2x2_Direct_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mtm_direct.process_raw(mtm_in_ptrs.as_ptr(), mtm_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
        group.bench_function(format!("ManyToManyShared_2x2_FFT_RF{rf}"), |b| {
            b.iter(|| unsafe {
                mtm_fft.process_raw(mtm_in_ptrs.as_ptr(), mtm_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
    }

    group.finish();
}

/// Evaluates execution overhead between In-Place (`input == output`) and Out-of-Place
/// processing to verify the performance impact of early input ring-buffer absorption.
fn bench_multichannel_inplace_vs_outofplace(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Multichannel_InPlace_vs_OutOfPlace_64samp");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_millis(1500));
    group.sample_size(30);

    // 1) 2 -> 2 ManyToManyShared (Direct RF=128 and FFT RF=2048)
    for &(rf, impl_mode, tag) in &[
        (128, LinearImplementation::Direct, "Direct_RF128"),
        (2048, LinearImplementation::Fft, "FFT_RF2048"),
    ] {
        let mut model_oop = make_multichannel_model(2, 2, rf, impl_mode);
        let in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let in_ch1 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ch1 = vec![0.0f32; BLOCK_SIZE];
        let oop_in_ptrs: [*const f32; 2] = [in_ch0.as_ptr(), in_ch1.as_ptr()];
        let oop_out_ptrs: [*mut f32; 2] = [out_ch0.as_mut_ptr(), out_ch1.as_mut_ptr()];

        group.bench_function(format!("ManyToMany_2x2_{tag}_OutOfPlace"), |b| {
            b.iter(|| unsafe {
                model_oop.process_raw(oop_in_ptrs.as_ptr(), oop_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });

        let mut model_ip = make_multichannel_model(2, 2, rf, impl_mode);
        let mut ip_buf_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut ip_buf_ch1 = common::generate_sine_440hz(BLOCK_SIZE);
        let ip_in_ptrs: [*const f32; 2] = [ip_buf_ch0.as_ptr(), ip_buf_ch1.as_ptr()];
        let ip_out_ptrs: [*mut f32; 2] = [ip_buf_ch0.as_mut_ptr(), ip_buf_ch1.as_mut_ptr()];

        group.bench_function(format!("ManyToMany_2x2_{tag}_InPlace"), |b| {
            b.iter(|| unsafe {
                model_ip.process_raw(ip_in_ptrs.as_ptr(), ip_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
    }

    // 2) 1 -> 2 OneToMany (Ch0 In-Place vs Out-Of-Place)
    for &(rf, impl_mode, tag) in &[
        (128, LinearImplementation::Direct, "Direct_RF128"),
        (2048, LinearImplementation::Fft, "FFT_RF2048"),
    ] {
        let mut model_oop = make_multichannel_model(1, 2, rf, impl_mode);
        let in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let mut out_ch1 = vec![0.0f32; BLOCK_SIZE];
        let oop_in_ptrs: [*const f32; 1] = [in_ch0.as_ptr()];
        let oop_out_ptrs: [*mut f32; 2] = [out_ch0.as_mut_ptr(), out_ch1.as_mut_ptr()];

        group.bench_function(format!("OneToMany_1x2_{tag}_OutOfPlace"), |b| {
            b.iter(|| unsafe {
                model_oop.process_raw(oop_in_ptrs.as_ptr(), oop_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });

        let mut model_ip = make_multichannel_model(1, 2, rf, impl_mode);
        let mut shared_buf = common::generate_sine_440hz(BLOCK_SIZE);
        let mut out_ch1_ip = vec![0.0f32; BLOCK_SIZE];
        let ip_in_ptrs: [*const f32; 1] = [shared_buf.as_ptr()];
        let ip_out_ptrs: [*mut f32; 2] = [shared_buf.as_mut_ptr(), out_ch1_ip.as_mut_ptr()];

        group.bench_function(format!("OneToMany_1x2_{tag}_InPlace_Ch0"), |b| {
            b.iter(|| unsafe {
                model_ip.process_raw(ip_in_ptrs.as_ptr(), ip_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
    }

    // 3) 2 -> 1 ManyToOne (Ch0 In-Place vs Out-Of-Place)
    for &(rf, impl_mode, tag) in &[
        (128, LinearImplementation::Direct, "Direct_RF128"),
        (2048, LinearImplementation::Fft, "FFT_RF2048"),
    ] {
        let mut model_oop = make_multichannel_model(2, 1, rf, impl_mode);
        let in_ch0 = common::generate_sine_440hz(BLOCK_SIZE);
        let in_ch1 = common::generate_sine_440hz(BLOCK_SIZE);
        let mut out_ch0 = vec![0.0f32; BLOCK_SIZE];
        let oop_in_ptrs: [*const f32; 2] = [in_ch0.as_ptr(), in_ch1.as_ptr()];
        let oop_out_ptrs: [*mut f32; 1] = [out_ch0.as_mut_ptr()];

        group.bench_function(format!("ManyToOne_2x1_{tag}_OutOfPlace"), |b| {
            b.iter(|| unsafe {
                model_oop.process_raw(oop_in_ptrs.as_ptr(), oop_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });

        let mut model_ip = make_multichannel_model(2, 1, rf, impl_mode);
        let mut shared_buf = common::generate_sine_440hz(BLOCK_SIZE);
        let in_ch1_ip = common::generate_sine_440hz(BLOCK_SIZE);
        let ip_in_ptrs: [*const f32; 2] = [shared_buf.as_ptr(), in_ch1_ip.as_ptr()];
        let ip_out_ptrs: [*mut f32; 1] = [shared_buf.as_mut_ptr()];

        group.bench_function(format!("ManyToOne_2x1_{tag}_InPlace_Ch0"), |b| {
            b.iter(|| unsafe {
                model_ip.process_raw(ip_in_ptrs.as_ptr(), ip_out_ptrs.as_ptr(), BLOCK_SIZE);
            });
        });
    }

    group.finish();
}

/// Maps the Direct vs FFT crossing point across audio block sizes (64 to 2048 samples)
/// for multichannel configurations.
fn bench_multichannel_crossover_direct_vs_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear_Multichannel_Crossover_Direct_vs_FFT");
    group.warm_up_time(std::time::Duration::from_millis(500));
    group.measurement_time(std::time::Duration::from_millis(1500));
    group.sample_size(30);

    const BLOCK_SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];

    // Evaluate RF=256 (near the auto-selection boundary) across block sizes
    let rf = 256;
    for &block in BLOCK_SIZES {
        // 2 -> 2 ManyToManyShared
        let mut mtm_direct = make_multichannel_model(2, 2, rf, LinearImplementation::Direct);
        let mut mtm_fft = make_multichannel_model(2, 2, rf, LinearImplementation::Fft);
        let in_ch0 = common::generate_sine_440hz(block);
        let in_ch1 = common::generate_sine_440hz(block);
        let mut out_ch0_d = vec![0.0f32; block];
        let mut out_ch1_d = vec![0.0f32; block];
        let mut out_ch0_f = vec![0.0f32; block];
        let mut out_ch1_f = vec![0.0f32; block];

        let in_ptrs: [*const f32; 2] = [in_ch0.as_ptr(), in_ch1.as_ptr()];
        let out_ptrs_d: [*mut f32; 2] = [out_ch0_d.as_mut_ptr(), out_ch1_d.as_mut_ptr()];
        let out_ptrs_f: [*mut f32; 2] = [out_ch0_f.as_mut_ptr(), out_ch1_f.as_mut_ptr()];

        group.bench_function(format!("ManyToMany_2x2_Direct_RF256_B{block}"), |b| {
            b.iter(|| unsafe {
                mtm_direct.process_raw(in_ptrs.as_ptr(), out_ptrs_d.as_ptr(), block);
            });
        });
        group.bench_function(format!("ManyToMany_2x2_FFT_RF256_B{block}"), |b| {
            b.iter(|| unsafe {
                mtm_fft.process_raw(in_ptrs.as_ptr(), out_ptrs_f.as_ptr(), block);
            });
        });

        // 1 -> 2 OneToMany
        let mut otm_direct = make_multichannel_model(1, 2, rf, LinearImplementation::Direct);
        let mut otm_fft = make_multichannel_model(1, 2, rf, LinearImplementation::Fft);
        let mut otm_out_ch0_d = vec![0.0f32; block];
        let mut otm_out_ch1_d = vec![0.0f32; block];
        let mut otm_out_ch0_f = vec![0.0f32; block];
        let mut otm_out_ch1_f = vec![0.0f32; block];

        let otm_in_ptrs: [*const f32; 1] = [in_ch0.as_ptr()];
        let otm_out_ptrs_d: [*mut f32; 2] =
            [otm_out_ch0_d.as_mut_ptr(), otm_out_ch1_d.as_mut_ptr()];
        let otm_out_ptrs_f: [*mut f32; 2] =
            [otm_out_ch0_f.as_mut_ptr(), otm_out_ch1_f.as_mut_ptr()];

        group.bench_function(format!("OneToMany_1x2_Direct_RF256_B{block}"), |b| {
            b.iter(|| unsafe {
                otm_direct.process_raw(otm_in_ptrs.as_ptr(), otm_out_ptrs_d.as_ptr(), block);
            });
        });
        group.bench_function(format!("OneToMany_1x2_FFT_RF256_B{block}"), |b| {
            b.iter(|| unsafe {
                otm_fft.process_raw(otm_in_ptrs.as_ptr(), otm_out_ptrs_f.as_ptr(), block);
            });
        });
    }

    // Evaluate RF=2048 (long FIR) across block sizes to demonstrate FFT scaling
    let rf_long = 2048;
    for &block in &[64, 256, 1024, 2048] {
        let mut mtm_direct = make_multichannel_model(2, 2, rf_long, LinearImplementation::Direct);
        let mut mtm_fft = make_multichannel_model(2, 2, rf_long, LinearImplementation::Fft);
        let in_ch0 = common::generate_sine_440hz(block);
        let in_ch1 = common::generate_sine_440hz(block);
        let mut out_ch0_d = vec![0.0f32; block];
        let mut out_ch1_d = vec![0.0f32; block];
        let mut out_ch0_f = vec![0.0f32; block];
        let mut out_ch1_f = vec![0.0f32; block];

        let in_ptrs: [*const f32; 2] = [in_ch0.as_ptr(), in_ch1.as_ptr()];
        let out_ptrs_d: [*mut f32; 2] = [out_ch0_d.as_mut_ptr(), out_ch1_d.as_mut_ptr()];
        let out_ptrs_f: [*mut f32; 2] = [out_ch0_f.as_mut_ptr(), out_ch1_f.as_mut_ptr()];

        group.bench_function(format!("ManyToMany_2x2_Direct_RF2048_B{block}"), |b| {
            b.iter(|| unsafe {
                mtm_direct.process_raw(in_ptrs.as_ptr(), out_ptrs_d.as_ptr(), block);
            });
        });
        group.bench_function(format!("ManyToMany_2x2_FFT_RF2048_B{block}"), |b| {
            b.iter(|| unsafe {
                mtm_fft.process_raw(in_ptrs.as_ptr(), out_ptrs_f.as_ptr(), block);
            });
        });
    }

    group.finish();
}

criterion_group!(
    name = linear_benches;
    config = Criterion::default().noise_threshold(0.05);
    targets = bench_direct_vs_fft_per_block,
              bench_direct_vs_fft_per_sample,
              bench_fft_prewarm,
              bench_fft_tail_block,
              bench_large_block_4096,
              bench_multichannel_geometries_per_block,
              bench_multichannel_inplace_vs_outofplace,
              bench_multichannel_crossover_direct_vs_fft
);

criterion_main!(linear_benches);
