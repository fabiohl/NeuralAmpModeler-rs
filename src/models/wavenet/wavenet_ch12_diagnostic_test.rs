// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// Minimal test: CH=12 dense layer with known weights, compare SIMD vs scalar
#[test]
fn test_dense_ch12_scalar_vs_simd() {
    use crate::math::common::AlignedVec;

    // Create a 12x12 DenseLayer with known weights
    let mut raw = vec![0.0f32; 12 * 12];

    // Fill with deterministic values
    for (i, item) in raw.iter_mut().enumerate() {
        *item = (i as f32 - 72.0) * 0.01; // range ~[-0.72, 0.72]
    }

    let bias = AlignedVec::from_vec(vec![0.1f32; 12])
        .expect("allocation should succeed for test-sized buffers");

    let dense = crate::models::wavenet::DenseLayer::<12, 12> {
        bias: bias.clone(),
        do_bias: true,
        weights: AlignedVec::from_vec(raw.clone())
            .expect("allocation should succeed for test-sized buffers"),
    };

    let input: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.2).collect();
    let mut simd_out = vec![0.0f32; 12];
    let mut scalar_out = [0.0f32; 12];

    // SIMD path
    unsafe {
        dense.process_block::<crate::math::common::Avx2Math>(&input, &mut simd_out, 1);
    }

    // Scalar reference (column-major f32 weights: f32_weights[ic * out + oc])
    for oc in 0..12 {
        let mut sum = bias[oc];
        for (ic, inval) in input.iter().enumerate() {
            let w = dense.weights[ic * 12 + oc];
            sum += inval * w;
        }
        scalar_out[oc] = sum;
    }

    for i in 0..12 {
        let diff = (simd_out[i] - scalar_out[i]).abs();
        assert!(
            diff < 1e-5,
            "Dense CH=12 SIMD mismatch at channel {}: {} vs {}, diff={}",
            i,
            simd_out[i],
            scalar_out[i],
            diff
        );
    }
}

#[test]
fn test_conv1d_ch12_scalar_vs_simd() {
    use crate::math::common::AlignedVec;
    use crate::models::wavenet::Conv1d;

    const CH: usize = 12;
    const K: usize = 3;
    let dil: usize = 1;

    // Create raw conv weights (row-major for scalar reference)
    let mut raw = vec![0.0f32; CH * K * CH];
    for (i, item) in raw.iter_mut().enumerate() {
        *item = (i as f32 - (CH * K * CH / 2) as f32) * 0.01;
    }

    use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
    use crate::loader::dispatcher::wavenet::layout::transpose_conv1d_interleaved_8wide;
    use crate::loader::dispatcher::wavenet::layout::transpose_conv1d_interleaved_16wide;

    let interleave_width = select_interleave_width(CH);
    let num_blocks = CH.div_ceil(interleave_width);
    let padded_len = num_blocks * K * CH * interleave_width;
    let mut weights_f32 = AlignedVec::new(padded_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    match interleave_width {
        16 => {
            transpose_conv1d_interleaved_16wide(&raw, &mut weights_f32, CH, CH, K);
        }
        8 => {
            transpose_conv1d_interleaved_8wide(&raw, &mut weights_f32, CH, CH, K);
        }
        _ => {
            crate::loader::dispatcher::wavenet::transpose_conv1d_interleaved_4wide(
                &raw,
                &mut weights_f32,
                CH,
                CH,
                K,
            );
        }
    }

    let bias = AlignedVec::from_vec(vec![0.01f32; CH])
        .expect("allocation should succeed for test-sized buffers");

    let conv = Conv1d::<CH, CH, K> {
        weights: weights_f32,
        bias: bias.clone(),
        do_bias: true,
        dilation: dil,
    };

    // Create state buffer (CH * 10 elements)
    let mut state = vec![0.0f32; CH * 10];
    for (i, item) in state.iter_mut().enumerate() {
        *item = (i as f32 - (CH * 5) as f32) * 0.1;
    }

    let frame_idx = 5;
    let mut simd_out = vec![0.0f32; CH];
    let mut scalar_out = [0.0f32; CH];

    // SIMD path
    unsafe {
        conv.process_single_frame::<crate::math::common::Avx2Math>(
            &state,
            &mut simd_out,
            frame_idx,
        );
    }

    // Scalar reference with interleaved weights
    for oc in 0..CH {
        let mut sum = bias[oc];
        for k in 0..K {
            let offset = (dil as isize) * ((k as isize) + 1 - (K as isize));
            let in_idx = ((frame_idx as isize) + offset) as usize * CH;
            for ic in 0..CH {
                let b = oc / interleave_width;
                let lane = oc % interleave_width;
                let w_idx = b * (K * CH * interleave_width)
                    + k * (CH * interleave_width)
                    + ic * interleave_width
                    + lane;
                let w = conv.weights[w_idx];
                sum += state[in_idx + ic] * w;
            }
        }
        scalar_out[oc] = sum;
    }

    let mut max_diff = 0.0f32;
    for i in 0..CH {
        let diff = (simd_out[i] - scalar_out[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff < 2e-5,
        "Conv1D CH=12 SIMD mismatch: max_diff={}",
        max_diff
    );
}

#[test]
#[ignore]
fn profile_wavenet_lite_kernels() {
    use crate::loader::dispatcher::build_model;
    use crate::loader::nam_json::parse_nam_json;
    use crate::models::NamModel;
    use crate::models::wavenet::layer::{
        ACC_CONV, ACC_MIXIN, ACC_ONE_BY_ONE, ACC_TANH, TELEMETRY_ACTIVE,
    };
    use std::time::{Duration, Instant};

    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/models/BossWN-lite.nam");

    assert!(path.exists(), "BossWN-lite.nam test fixture not found!");

    let json_data = std::fs::read_to_string(&path).expect("Failed to read model");
    let model_data = parse_nam_json(&json_data).expect("Failed to parse JSON");
    let mut model = build_model(&model_data).expect("Failed to build model");

    model.prewarm(2048);

    let input = vec![0.1f32; 64];
    let mut output = vec![0.0f32; 64];

    // Warmup the execution to trigger JIT / cache loads
    for _ in 0..1000 {
        model.process(&input, &mut output);
    }

    // Reset telemetry variables
    ACC_MIXIN.with(|v| *v.borrow_mut() = Duration::ZERO);
    ACC_CONV.with(|v| *v.borrow_mut() = Duration::ZERO);
    ACC_TANH.with(|v| *v.borrow_mut() = Duration::ZERO);
    ACC_ONE_BY_ONE.with(|v| *v.borrow_mut() = Duration::ZERO);

    TELEMETRY_ACTIVE.with(|v| *v.borrow_mut() = true);

    let total_start = Instant::now();
    let iterations = 30000;
    for _ in 0..iterations {
        model.process(&input, &mut output);
    }
    let total_duration = total_start.elapsed();

    TELEMETRY_ACTIVE.with(|v| *v.borrow_mut() = false);

    let mixin_dur = ACC_MIXIN.with(|v| *v.borrow());
    let conv_dur = ACC_CONV.with(|v| *v.borrow());
    let tanh_dur = ACC_TANH.with(|v| *v.borrow());
    let one_by_one_dur = ACC_ONE_BY_ONE.with(|v| *v.borrow());

    let sum_measured = mixin_dur + conv_dur + tanh_dur + one_by_one_dur;

    println!("\n========================================================");
    println!(" WaveNet Lite CH12 Internal Kernel Telemetry (30k iterations)");
    println!("========================================================");
    println!("Total execution time: {:?}", total_duration);
    println!("Sum of measured layer blocks: {:?}", sum_measured);
    println!("--------------------------------------------------------");
    println!(
        "1. Input Mixin (DenseLayer):   {:?} ({:.2}%)",
        mixin_dur,
        (mixin_dur.as_nanos() as f64 / sum_measured.as_nanos() as f64) * 100.0
    );
    println!(
        "2. Dilated Conv1D (SIMD 16x):  {:?} ({:.2}%)",
        conv_dur,
        (conv_dur.as_nanos() as f64 / sum_measured.as_nanos() as f64) * 100.0
    );
    println!(
        "3. Tanh Activation + Accum:   {:?} ({:.2}%)",
        tanh_dur,
        (tanh_dur.as_nanos() as f64 / sum_measured.as_nanos() as f64) * 100.0
    );
    println!(
        "4. OneByOne Residual (Dense):  {:?} ({:.2}%)",
        one_by_one_dur,
        (one_by_one_dur.as_nanos() as f64 / sum_measured.as_nanos() as f64) * 100.0
    );
    println!("========================================================\n");
}
