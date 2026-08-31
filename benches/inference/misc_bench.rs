// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Miscellaneous inference benchmarks: LinearModel dot product, ContainerModel
//! crossfade, dynamic fallbacks, non-distributable models, and ConvNet.

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::models::container::ContainerModel;
use neural_amp_modeler_rs::models::slimmable::SlimmableModel;

use super::common::{
    generate_sine_440hz, load_and_prewarm, load_model_data, make_lstm_data,
    make_wavenet_a2_dyn_data,
};

/// Benchmarks the LinearModel dot product kernel (AVX2/AVX-512 SIMD vs scalar).
/// With RF 256, the scalar path performs 16k FMAs per 64-sample block;
/// the SIMD path reduces this by 4-8×.
pub fn bench_linear_model_dot_product(c: &mut Criterion) {
    use neural_amp_modeler_rs::models::linear::LinearModel;

    let rf = 256;
    let weights: Vec<f32> = (0..rf).map(|i| (i as f32 * 0.01).sin()).collect();
    let bias = 0.1;
    let mut model = LinearModel::new(
        weights,
        bias,
        neural_amp_modeler_rs::loader::nam_json::LinearImplementation::default(),
    )
    .unwrap();
    model.prewarm(0);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LinearModel_RF256_64samp_SIMD", |b| {
        b.iter(|| unsafe {
            model.process(
                std::hint::black_box(&input),
                std::hint::black_box(&mut output),
            );
        });
    });
}

/// Measures the processing cost of a ContainerModel crossfade block
/// (dual inference + SIMD blend via FMA), the worst-case per-block cost
/// during slimmable submodel switching.
pub fn bench_container_crossfade_64samp(c: &mut Criterion) {
    let full_data = match load_model_data("wavenet_a2_full.nam") {
        Some(d) => d,
        None => return,
    };
    let full_model = build_model(&full_data).expect("Dispatcher failed for A2-Full");

    let lite_model = match load_and_prewarm("wavenet_a2_lite.nam") {
        Some(m) => m,
        None => return,
    };

    let sr = full_data.sample_rate.map(|s| s as u32).unwrap_or(48000);
    let mut container =
        ContainerModel::new(vec![(0.5, Box::new(lite_model)), (1.0, full_model)], sr)
            .expect("Failed to create ContainerModel benchmark");

    container.set_slimmable_size(0.25, None);
    assert!(container.is_crossfading());

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    let mut model = StaticModel::Container(Box::new(container));
    c.bench_function("Container_Crossfade_64samp", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a WaveNet A2 Dynamic model (CH=4, gated).
///
/// CH=4 is not in the A2 const-generic dispatch table ({3, 8}),
/// forcing routing to `WaveNetA2Dyn`. This covers the dynamic A2 hot-path
/// with gating active on the first layer, exercising the full dynamic engine.
pub fn bench_wavenet_a2_dyn_gated_process(c: &mut Criterion) {
    let data = make_wavenet_a2_dyn_data();
    let mut model = build_model(&data).expect("Dispatcher failed for WaveNet A2 Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_Gated_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of an LSTM Dynamic model (1 layer × 7 hidden).
///
/// H=7 is not in the const-generic dispatch table ({3,8,12,16,24,40}),
/// forcing routing to `LstmModelDyn`. This covers the dynamic LSTM hot-path.
pub fn bench_lstm_dynamic_process(c: &mut Criterion) {
    let data = make_lstm_data(1, 7);
    let mut model = build_model(&data).expect("Dispatcher failed for LSTM Dynamic benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("LSTM_Dynamic_1x7_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures inference latency for any present non-distributable models.
pub fn bench_nondist_models(c: &mut Criterion) {
    let nondist_path = match neural_amp_modeler_rs::testing::fixtures::nondist_models_dir() {
        Some(p) => p,
        None => return,
    };

    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&nondist_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext == "nam" || ext == "json")
            {
                models.push(path);
            }
        }
    }

    for model_path in models {
        let filename = model_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let json_data = match std::fs::read_to_string(&model_path) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let model_data = match parse_nam_json(&json_data) {
            Ok(data) => data,
            Err(_) => continue,
        };
        let mut model = match build_model(&model_data) {
            Ok(m) => m,
            Err(_) => continue,
        };
        model.prewarm(2048);

        let input = generate_sine_440hz(64);
        let mut output = vec![0.0f32; 64];

        c.bench_function(&format!("NonDist_Model_{}_64samp", filename), |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures the end-to-end inference cost of a full ConvNet model (2 blocks, CH=8→4, K=3).
///
/// Unlike the ConvNetBlock-level benches (now in separate benches), this loads the
/// `convnet_test.nam` fixture, exercises the full model pipeline (multi-block chaining
/// + head_scale), and profiles the dispatcher build_model path.
pub fn bench_convnet_model_process(c: &mut Criterion) {
    let mut model = match load_and_prewarm("convnet_test.nam") {
        Some(m) => m,
        None => return,
    };

    let num_out = match &model {
        StaticModel::ConvNet(c) => c.out_channels(),
        _ => 1,
    };
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64 * num_out];

    c.bench_function("ConvNet_Model_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}
