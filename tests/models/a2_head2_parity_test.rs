// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! WaveNet A2-Dynamic with a single array and `head_size > 1` must emit
//! interleaved output channels matching an exact f64 oracle, route correctly
//! through the planar boundary, and keep the canonical layer-array head
//! geometry consistent. The single-array dynamic head previously emitted
//! silence for `head_size > 1` while `StaticModel::num_output_channels`
//! reported `1`.
//!
//! The NAMcore `render` tool is **mono-only** (it allocates a
//! `num_frames` output buffer and saves a single channel), so it cannot serve
//! as a multi-output oracle; multi-output C++ parity belongs to the dedicated
//! `render_multichannel` harness in `tests/parity`. The C++ channel-0 evidence
//! in this file is therefore the canonical mono A2-Max regression.

use std::fs;
use std::process::Command;

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use neural_amp_modeler_rs::testing::perceptual::{compute_esr, compute_snr_db};
use neural_amp_modeler_rs::testing::wav::{read_wav_f32, write_wav_f32};

use super::common::io_helpers::require_render_or_skip;

/// Model-level head scale carried as the trailing weight (`!= 0.5` on purpose).
const HEAD_SCALE: f32 = 0.75;
/// Per-output-channel head weights (kernel = 1, one input channel).
const HEAD_W: [f32; 2] = [2.0, 4.0];
/// Three-channel head with distinct weights and distinct biases.
const HEAD3_W: [f32; 3] = [2.0, -3.0, 4.0];
const HEAD3_B: [f32; 3] = [0.1, -0.2, 0.3];

/// Minimal 1-layer A2-Dynamic single array with an arbitrary `head_size`.
///
/// Weight stream: rechannel(1) + conv_w(1) + conv_b(1) + mixin(1) + l1x1_w(1)
/// + l1x1_b(1) + head_rechannel_w(head_size) + head_rechannel_b(head_size)
/// + trailing head_scale(1).
fn build_head_model(head_size: usize, head_w: &[f32], head_b: &[f32], scale: f32) -> StaticModel {
    assert_eq!(head_w.len(), head_size);
    assert_eq!(head_b.len(), head_size);

    let mut weights = vec![1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0];
    weights.extend_from_slice(head_w);
    weights.extend_from_slice(head_b);
    weights.push(scale);

    let json = format!(
        r#"{{
            "version": "0.6.0",
            "architecture": "WaveNet",
            "config": {{
                "in_channels": 1,
                "head_scale": {scale},
                "head": null,
                "layers": [{{
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": {head_size},
                    "head_bias": true,
                    "channels": 1,
                    "bottleneck": 1,
                    "kernel_size": 1,
                    "dilations": [1],
                    "activation": {{"type": "ReLU"}},
                    "gating_mode": null
                }}]
            }},
            "weights": {weights:?}
        }}"#
    );
    let data = parse_nam_json(&json).expect("inline A2 head JSON parses");
    *build_model(&data).expect("A2 head model builds")
}

fn build_head2_model() -> StaticModel {
    build_head_model(2, &HEAD_W, &[0.0, 0.0], HEAD_SCALE)
}

fn positive_signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| 0.01 * (i as f32) + 0.05).collect()
}

fn sine_signal(n: usize, sr: u32) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * (i as f32) / sr as f32).sin())
        .collect()
}

#[test]
fn a2_head2_single_array_vs_interleaved_oracle() {
    let mut model = build_head2_model();
    assert_eq!(model.num_input_channels(), 1);
    assert_eq!(model.num_output_channels(), 2);

    let input = positive_signal(64);
    let mut output = vec![0.0f32; input.len() * 2];
    model.process(&input, &mut output);

    assert_eq!(output.len(), input.len() * 2, "2x frames interleaved");
    for f in 0..input.len() {
        // f64 oracle: ReLU(conv(x)) == x for non-negative x and unit weights.
        let x = input[f] as f64;
        let expected0 = x * HEAD_W[0] as f64 * HEAD_SCALE as f64;
        let expected1 = x * HEAD_W[1] as f64 * HEAD_SCALE as f64;
        let got0 = output[f * 2] as f64;
        let got1 = output[f * 2 + 1] as f64;
        assert!(
            (got0 - expected0).abs() < 1e-6,
            "frame {f} ch0: got {got0}, oracle {expected0}"
        );
        assert!(
            (got1 - expected1).abs() < 1e-6,
            "frame {f} ch1: got {got1}, oracle {expected1}"
        );
    }
}

#[test]
fn a2_cascade_head2_planes_have_all_frames() {
    // Boundary contract: the planar entry point must route the interleaved
    // mono `process()` output into one plane per channel, covering the whole
    // block. The multi-output A2 single-array head exercises the same
    // `StaticModel` staging path as a multi-output cascade array.
    let input = positive_signal(48);

    let mut interleaved_model = build_head2_model();
    let mut interleaved = vec![0.0f32; input.len() * 2];
    interleaved_model.process(&input, &mut interleaved);

    let mut planar_model = build_head2_model();
    let mut plane0 = vec![0.0f32; input.len()];
    let mut plane1 = vec![0.0f32; input.len()];
    let frames = planar_model
        .process_multichannel(&[&input], &mut [&mut plane0, &mut plane1])
        .expect("A2 head2 requires 1 input and 2 output planes");
    assert_eq!(frames, input.len(), "all frames must be processed");

    for f in 0..input.len() {
        assert_eq!(
            plane0[f],
            interleaved[f * 2],
            "plane 0 = even frame samples"
        );
        assert_eq!(
            plane1[f],
            interleaved[f * 2 + 1],
            "plane 1 = odd frame samples"
        );
    }
    // In-place alias path across the boundary must match too (first channel).
    let mut aliased = build_head2_model();
    let mut io0 = input.clone();
    let mut io1 = vec![0.0f32; input.len()];
    let frames_ip = aliased
        .process_multichannel_in_place(&mut [&mut io0, &mut io1])
        .expect("A2 head2 in-place needs max(1,2)=2 planes");
    assert_eq!(frames_ip, input.len());
    assert_eq!(io0, plane0, "in-place plane 0 must match out-of-place");
    assert_eq!(io1, plane1, "in-place plane 1 must match out-of-place");
}

#[test]
fn a2_head3_single_array_distinct_bias_channels() {
    // `head_size = 3` with distinct weights, distinct biases and a head scale
    // different from the historical 0.5 must expose three independently
    // computed channels. The kernel=1, unit-weight, ReLU topology makes the
    // whole model stateless, so the f64 oracle is exact.
    let mut model = build_head_model(3, &HEAD3_W, &HEAD3_B, HEAD_SCALE);
    assert_eq!(model.num_output_channels(), 3);
    assert_eq!(model.num_input_channels(), 1);

    let input = positive_signal(32);
    let mut output = vec![0.0f32; input.len() * 3];
    model.process(&input, &mut output);

    let mut ch_differ = false;
    for f in 0..input.len() {
        let x = input[f] as f64;
        let e = [
            (x * HEAD3_W[0] as f64 + HEAD3_B[0] as f64) * HEAD_SCALE as f64,
            (x * HEAD3_W[1] as f64 + HEAD3_B[1] as f64) * HEAD_SCALE as f64,
            (x * HEAD3_W[2] as f64 + HEAD3_B[2] as f64) * HEAD_SCALE as f64,
        ];
        for c in 0..3 {
            let got = output[f * 3 + c] as f64;
            assert!(
                (got - e[c]).abs() < 1e-6,
                "frame {f} ch{c}: got {got}, oracle {}",
                e[c]
            );
        }
        if (output[f * 3] - output[f * 3 + 1]).abs() > 1e-6 {
            ch_differ = true;
        }
    }
    assert!(
        ch_differ,
        "head_size=3 channels must not collapse to one value"
    );
}

/// Canonical layer-array A2 whose head is widened to `head_size = 3` must
/// expose three distinct output channels through the generic dynamic loader.
#[test]
fn a2_canonical_head3_multi_output_geometry() {
    const HEAD_SIZE: usize = 3;
    const HEAD_KERNEL: usize = 16;
    const BOTTLENECK: usize = 3;
    let lite = fs::read_to_string(neural_amp_modeler_rs::testing::fixtures::model_path(
        "wavenet_a2_lite.nam",
    ))
    .expect("read wavenet_a2_lite.nam");
    let mut val: serde_json::Value = serde_json::from_str(&lite).expect("parse lite fixture");
    let mut w: Vec<f32> = val["weights"]
        .as_array()
        .expect("weights array")
        .iter()
        .map(|v| v.as_f64().expect("f64 weight") as f32)
        .collect();
    let old_head = HEAD_KERNEL * BOTTLENECK + 1 + 1;
    assert!(w.len() > old_head, "lite fixture unexpectedly small");
    let orig_head_w: Vec<f32> =
        w[w.len() - old_head..w.len() - old_head + HEAD_KERNEL * BOTTLENECK].to_vec();
    w.truncate(w.len() - old_head);
    for (c, gain) in [1.0f32, 0.7, -0.5].into_iter().enumerate() {
        assert!(c < HEAD_SIZE);
        for &v in &orig_head_w {
            w.push(gain * v);
        }
    }
    w.extend_from_slice(&[0.1, -0.2, 0.3]);
    w.push(HEAD_SCALE);
    val["config"]["layers"][0]["head"]["out_channels"] = serde_json::json!(HEAD_SIZE);
    val["weights"] = serde_json::json!(w);
    let json = val.to_string();

    let data = parse_nam_json(&json).expect("parse canonical head3");
    let mut model = build_model(&data).expect("build canonical head3 in rust");
    assert_eq!(model.num_output_channels(), HEAD_SIZE);

    let input = sine_signal(512, 48000);
    let mut out = vec![0.0f32; input.len() * HEAD_SIZE];
    let mut pos = 0;
    while pos < input.len() {
        let n = (input.len() - pos).min(64);
        model.process(
            &input[pos..pos + n],
            &mut out[pos * HEAD_SIZE..(pos + n) * HEAD_SIZE],
        );
        pos += n;
    }
    assert!(
        out.iter().all(|v| v.is_finite()),
        "canonical head3 output must be finite"
    );
    let mut differ = false;
    for f in 0..input.len() {
        if (out[f * 3] - out[f * 3 + 1]).abs() > 1e-6
            || (out[f * 3 + 1] - out[f * 3 + 2]).abs() > 1e-6
        {
            differ = true;
        }
    }
    assert!(
        differ,
        "canonical head_size=3 channels must not collapse to one value"
    );
}

/// A2-Max regression: the canonical mono cascade fixture must stay matched to
/// the C++ renderer's (only) channel 0 after the single-array head changes.
#[test]
fn a2_max_regression_vs_cpp_channel0() {
    let Some(render_bin) = require_render_or_skip("a2-max c++ parity") else {
        return;
    };
    let nam_src = neural_amp_modeler_rs::testing::fixtures::model_path("wavenet_a2_max.nam");
    if !nam_src.exists() {
        panic!("required fixture wavenet_a2_max.nam is missing");
    }
    let json = fs::read_to_string(&nam_src).expect("read a2-max fixture");

    let num_samples = 2048;
    let sample_rate = 48000u32;
    let input = sine_signal(num_samples, sample_rate);

    let tmp = std::env::temp_dir();
    let in_wav = tmp.join("nam_rs_a2max_in.wav");
    let out_wav = tmp.join("nam_rs_a2max_out.wav");
    write_wav_f32(&in_wav, &input, sample_rate).expect("write input wav");

    let status = Command::new(&render_bin)
        .arg(&nam_src)
        .arg(&in_wav)
        .arg(&out_wav)
        .output()
        .expect("spawn namcore render");
    assert!(
        status.status.success(),
        "namcore render failed for a2-max: {}",
        String::from_utf8_lossy(&status.stderr)
    );

    let (cpp_out, cpp_sr) = read_wav_f32(&out_wav).expect("read cpp output");
    assert_eq!(cpp_sr, sample_rate);
    assert_eq!(cpp_out.len(), num_samples);

    let data = parse_nam_json(&json).expect("parse a2-max json");
    let mut model = build_model(&data).expect("build a2-max in rust");
    assert_eq!(model.num_output_channels(), 1, "A2-Max is mono");
    model.reset(sample_rate, 64).expect("reset");
    let mut rust_out = vec![0.0f32; num_samples];
    let mut pos = 0;
    while pos < num_samples {
        let n = (num_samples - pos).min(64);
        model.process(&input[pos..pos + n], &mut rust_out[pos..pos + n]);
        pos += n;
    }

    let esr = compute_esr(&cpp_out, &rust_out);
    let snr = compute_snr_db(&cpp_out, &rust_out);
    eprintln!("[A2-Max C++ parity] ESR={esr:.4e}, SNR={snr:.1} dB");
    assert!(esr < 1e-4, "A2-Max ESR too high: {esr}");
    assert!(snr > 80.0, "A2-Max SNR too low: {snr} dB");

    let _ = fs::remove_file(&in_wav);
    let _ = fs::remove_file(&out_wav);
}
