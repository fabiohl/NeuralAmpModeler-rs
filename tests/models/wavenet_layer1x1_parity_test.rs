// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A1 `layer1x1` provenance parity.
//!
//! The C++ A1 engine defaults `layer1x1_active = true`, and the Rust A1 engine
//! always consumes the 1×1 block. Declaring `{"active": true, "groups": 1}`
//! explicitly therefore must not alter the built graph: the output must be
//! bit-exact against the same model without the key.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_snr_db};

use super::common;
use common::io_helpers::{cpp_render_mono, model_path, process_in_blocks, require_render_or_skip};

const BLOCK: usize = 64;

#[test]
fn wavenet_a1_layer1x1_active_groups1_is_bit_exact() {
    let path = model_path("BossWN-standard.nam");
    if !path.exists() {
        eprintln!("SKIP: BossWN-standard.nam fixture not found.");
        return;
    }
    let json = std::fs::read_to_string(&path).expect("read BossWN-standard.nam");

    let mut injected: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    let layers = injected["config"]["layers"]
        .as_array_mut()
        .expect("fixture must have config.layers");
    for layer in layers.iter_mut() {
        layer
            .as_object_mut()
            .expect("layer must be an object")
            .insert(
                "layer1x1".to_string(),
                serde_json::json!({"active": true, "groups": 1}),
            );
    }

    let base_data = parse_nam_json(&json).expect("parse base");
    let injected_data = parse_nam_json(&injected.to_string()).expect("parse injected");

    let mut base = build_model(&base_data).expect("build base");
    let mut inj = build_model(&injected_data).expect("build injected");

    let input: Vec<f32> = (0..BLOCK * 4)
        .map(|i| (i as f32 * 0.05).sin() * 0.5)
        .collect();
    let mut out_base = vec![0.0f32; input.len()];
    let mut out_inj = vec![0.0f32; input.len()];

    process_in_blocks(&mut base, &input, &mut out_base, BLOCK);
    process_in_blocks(&mut inj, &input, &mut out_inj, BLOCK);

    assert_eq!(
        out_base, out_inj,
        "explicit layer1x1 {{active:true, groups:1}} must be bit-exact vs the default"
    );
}

/// The A1 engine (base and explicitly-`layer1x1` model) must match the NAMcore
/// `render` tool on the untouched BossWN-standard fixture.
#[test]
fn wavenet_a1_layer1x1_matches_cpp_render() {
    let Some(render_bin) = require_render_or_skip("a1 layer1x1 c++ parity") else {
        return;
    };
    let path = model_path("BossWN-standard.nam");
    assert!(path.exists(), "BossWN-standard.nam is a required fixture");
    let json = std::fs::read_to_string(&path).expect("read BossWN-standard.nam");
    let val: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    let sample_rate = val
        .get("sample_rate")
        .and_then(|v| v.as_u64())
        .unwrap_or(48_000) as u32;

    let input: Vec<f32> = (0..BLOCK * 4)
        .map(|i| (i as f32 * 0.05).sin() * 0.5)
        .collect();
    let cpp = cpp_render_mono(&render_bin, &json, &input, sample_rate, "a1_l1x1");

    // Base model (no injected key) vs C++ render.
    let base_data = parse_nam_json(&json).expect("parse base");
    let mut base = build_model(&base_data).expect("build base");
    base.reset(sample_rate, BLOCK).expect("reset base");
    let mut out_base = vec![0.0f32; input.len()];
    process_in_blocks(&mut base, &input, &mut out_base, BLOCK);
    let esr = compute_esr(&cpp, &out_base);
    let snr = compute_snr_db(&cpp, &out_base);
    assert!(esr < 1e-4, "A1 base vs C++ render ESR too high: {esr}");
    assert!(snr > 80.0, "A1 base vs C++ render SNR too low: {snr} dB");

    // Explicit layer1x1 model must match the C++ render as well.
    let mut injected: serde_json::Value = serde_json::from_str(&json).expect("parse fixture");
    for layer in injected["config"]["layers"]
        .as_array_mut()
        .expect("config.layers")
    {
        layer.as_object_mut().expect("layer object").insert(
            "layer1x1".to_string(),
            serde_json::json!({"active": true, "groups": 1}),
        );
    }
    let inj_data = parse_nam_json(&injected.to_string()).expect("parse injected");
    let mut inj = build_model(&inj_data).expect("build injected");
    inj.reset(sample_rate, BLOCK).expect("reset injected");
    let mut out_inj = vec![0.0f32; input.len()];
    process_in_blocks(&mut inj, &input, &mut out_inj, BLOCK);
    let esr_inj = compute_esr(&cpp, &out_inj);
    assert!(
        esr_inj < 1e-4,
        "explicit layer1x1 vs C++ render ESR too high: {esr_inj}"
    );
}
