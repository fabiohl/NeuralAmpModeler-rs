// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::path::PathBuf;

use crate::loader::nam_json::model::NamModelData;

use super::*;

#[test]
fn test_oracle_a2_multichannel_dispersion() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let nam_path = manifest_dir.join("tests/fixtures/models/wavenet_a2_max.nam");
    if !nam_path.exists() {
        return;
    }
    let json_data = std::fs::read_to_string(&nam_path).expect("Failed to read A2 Max model");
    let model_data: NamModelData =
        serde_json::from_str(&json_data).expect("Failed to parse A2 Max JSON");
    let cond_json = model_data
        .config
        .condition_dsp
        .as_ref()
        .expect("condition_dsp must exist");
    let cond_model: NamModelData =
        serde_json::from_value(cond_json.clone()).expect("Failed to parse condition_dsp JSON");

    let num_frames = 256;
    let input: Vec<f64> = (0..num_frames).map(|i| (i as f64 * 0.05).sin()).collect();
    let config = PrecisionConfig::default();
    let out = oracle_condition_dsp_channels(&cond_model, &input, &config);

    let num_channels = 8;
    assert_eq!(
        out.len(),
        num_frames * num_channels,
        "Expected {} interleaved samples ({} frames × {} channels), got {}",
        num_frames * num_channels,
        num_frames,
        num_channels,
        out.len()
    );

    // Compute standard deviation across channels per frame after initial receptive field window (frames 64..256).
    let mut total_std = 0.0;
    let mut valid_frames = 0;
    for f in 64..num_frames {
        let frame_slice = &out[f * num_channels..(f + 1) * num_channels];
        let mean = frame_slice.iter().sum::<f64>() / num_channels as f64;
        let variance =
            frame_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / num_channels as f64;
        total_std += variance.sqrt();
        valid_frames += 1;
    }
    let avg_std = total_std / valid_frames as f64;
    println!("// Measured: average inter-channel std dev = {avg_std:.6e}");
    assert!(
        avg_std > 1e-4,
        "Expected inter-channel std dev > 1e-4, got {avg_std:.6e}"
    );
}
