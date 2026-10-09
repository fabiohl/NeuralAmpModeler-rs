// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Miscellaneous inference benchmarks: LinearModel dot product, ContainerModel
//! crossfade, dynamic fallbacks, non-distributable models, ConvNet, and the
//! A2-Dynamic FiLM family (dense, grouped, and the A2-Max flagship cascade).

use criterion::Criterion;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::models::container::ContainerModel;
use neural_amp_modeler_rs::models::slimmable::SlimmableModel;

use super::common::{
    generate_sine_440hz, load_and_prewarm, load_model_data, make_lstm_data,
    make_wavenet_a2_dyn_cond_dsp_data, make_wavenet_a2_dyn_data,
    make_wavenet_a2_dyn_film_grouped_data,
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

/// Measures the processing time of a WaveNet A2 Dynamic control model
/// (CH=4, `condition_size=4`, Linear `condition_dsp`) without FiLM.
///
/// Control counterpart of [`bench_wavenet_a2_dyn_film_grouped_process`]:
/// identical topology and conditioning path minus the FiLM slots, so the
/// latency delta between `A2Dyn_CondDsp_CH4_64samp_48kHz` and
/// `A2Dyn_FiLM_Grouped_CH4_64samp_48kHz` isolates the grouped-FiLM hot-path
/// cost from the `condition_dsp` cost.
pub fn bench_wavenet_a2_dyn_cond_dsp_process(c: &mut Criterion) {
    let data = make_wavenet_a2_dyn_cond_dsp_data();
    let mut model =
        build_model(&data).expect("Dispatcher failed for A2 Dynamic condition_dsp benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_CondDsp_CH4_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of a WaveNet A2 Dynamic model (CH=4) with
/// grouped FiLM (all 8 insertion slots active, `groups=2`, `shift=true`)
/// and a Linear `condition_dsp`.
///
/// Exercises the grouped `cond_to_scale_shift` + global-row modulation path
/// on every layer — the FiLM layout production consumes for grouped A2
/// topologies. Compare against [`bench_wavenet_a2_dyn_cond_dsp_process`]
/// for the isolated FiLM cost and against the stage sweep in
/// `a2_dyn_stage_bench` for the per-slot group-count effect.
pub fn bench_wavenet_a2_dyn_film_grouped_process(c: &mut Criterion) {
    let data = make_wavenet_a2_dyn_film_grouped_data();
    let mut model =
        build_model(&data).expect("Dispatcher failed for A2 Dynamic grouped-FiLM benchmark");
    model.prewarm(2048);

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_FiLM_Grouped_CH4_64samp_48kHz", |b| {
        b.iter(|| {
            model.process(&input, &mut output);
        });
    });
}

/// Measures the processing time of dense-FiLM A2 Dynamic models on the
/// Full (CH=8) and Lite (CH=3) topologies (`groups=1`, `shift=true`,
/// 4 FiLM slots each).
///
/// These fixtures route to `WaveNetA2Dyn` because FiLM is active (the
/// const-generic A2 fast path rejects FiLM), so they quantify the FiLM
/// modulation cost on the exact full/lite dimensions used by the
/// non-FiLM `RT_A2_Full_CH8` / `RT_A2_Lite_CH3` regression-gate targets.
pub fn bench_wavenet_a2_dyn_film_dense_process(c: &mut Criterion) {
    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    for (fixture, id) in [
        (
            "wavenet_a2_film_full.nam",
            "A2Dyn_FiLM_Dense_CH8_64samp_48kHz",
        ),
        (
            "wavenet_a2_film_lite.nam",
            "A2Dyn_FiLM_Dense_CH3_64samp_48kHz",
        ),
    ] {
        let mut model = match load_and_prewarm(fixture) {
            Some(m) => m,
            None => continue,
        };
        c.bench_function(id, |b| {
            b.iter(|| {
                model.process(&input, &mut output);
            });
        });
    }
}

/// Measures the processing time of the flagship A2-Max topology
/// (`wavenet_a2_max.nam`): a main A2-Dynamic array with grouped FiLM
/// (`groups` 1/2/4/8 across the 8 slots, `condition_size=8`) driven by a
/// two-array condition_dsp cascade.
///
/// This is the only in-tree model combining grouped FiLM with a multi-array
/// cascade, so it is the bench that covers the last-layer residual of
/// non-final arrays (`skip_last_residual=false` in the cascade path) next to
/// the grouped-FiLM hot path at production dimensions.
pub fn bench_wavenet_a2_max_film_grouped_process(c: &mut Criterion) {
    let mut model = match load_and_prewarm("wavenet_a2_max.nam") {
        Some(m) => m,
        None => return,
    };

    let input = generate_sine_440hz(64);
    let mut output = vec![0.0f32; 64];

    c.bench_function("A2Dyn_FiLM_Grouped_A2Max_64samp_48kHz", |b| {
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
