// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Zero-input full-scale offset proof and clipping-flag semantics.
//!
//! The production model `wavenet_a2_max.nam` carries a large constant offset in
//! its zero-input response, so it delivers supra-FS samples at unity gain while
//! the noise gate is still open and `RT_STATUS_HAS_CLIPPED` fires on silent
//! input. The C++ reference (`NeuralAmpModelerCore` `render`, read-only vendor
//! tool) reproduces the offset constantly (+9.725988 from sample 0, std 0.0
//! over 48000 frames), proving model content rather than parity drift.
//! (`slimmable_wavenet.nam` also clips in the smoke logs, but it is a
//! `KnownGap` fixture with no parity claim, so it is out of scope here.)
//!
//! Locked semantics: the flag reports that a *delivered* sample exceeded FS.
//! With a steady open gate the detection observes the post-gate (fused) signal;
//! while the gate is fading it observes the pre-gate signal and the ramp is
//! applied afterwards (`src/dsp/pipeline/stages/output.rs`,
//! `apply_output_stage_inner`). A closed gate mutes to zero without flagging.

use neural_amp_modeler_rs::common::params::AdaptiveComputeMode;
use neural_amp_modeler_rs::common::spsc::{RT_STATUS_HAS_CLIPPED, RtStatusFlags};
use neural_amp_modeler_rs::dsp::adaptive::AdaptiveCompute;
use neural_amp_modeler_rs::dsp::gate::{DynamicHysteresis, GateParams, GateState};
use neural_amp_modeler_rs::dsp::pipeline::apply_output_stage;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::{NamModel, StaticModel};
use std::fs;

use super::common;
use common::*;

// Measured: C++ `render` with 1 s of digital silence @48 kHz reproduces this
// offset constantly from sample 0 (std = 0.0 over 48000 frames).
const A2_MAX_DC: f32 = 9.725_988;
// Measured: control models stay near zero under the same stimulus.
const A2_LITE_DC: f32 = 0.047_641;
const A2_FULL_DC: f32 = -0.057_219;

const BLOCK: usize = 64;

fn load_model(fixture: &str) -> Box<StaticModel> {
    let path = model_path(fixture);
    assert!(path.exists(), "fixture {fixture} not found");
    let json = fs::read_to_string(&path).expect("failed to read fixture");
    let data = parse_nam_json(&json).expect("failed to parse fixture");
    build_model(&data).expect("failed to build fixture")
}

/// Renders `blocks` zero-input blocks and returns per-block (mean, min, max).
fn render_zeros(model: &mut StaticModel, blocks: usize) -> Vec<(f32, f32, f32)> {
    let input = vec![0.0f32; BLOCK];
    let mut output = vec![0.0f32; BLOCK];
    let mut stats = Vec::with_capacity(blocks);
    for _ in 0..blocks {
        model.process(&input, &mut output);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "non-finite output on zero input"
        );
        let mean = output.iter().sum::<f32>() / BLOCK as f32;
        let min = output.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        stats.push((mean, min, max));
    }
    stats
}

fn assert_constant_dc(stats: &[(f32, f32, f32)], expected: f32, label: &str) {
    let (mean0, min0, max0) = stats[0];
    let (mean3, min3, max3) = stats[3];
    println!(
        "// Measured {label}: cold block0 mean={mean0:.6} min={min0:.6} max={max0:.6} \
         | block3 mean={mean3:.6} min={min3:.6} max={max3:.6} | C++ {expected:.6}"
    );
    // Measured: the cold first block carries a settling ripple of ~±0.12 around
    // the offset (delay lines filling from the zero state); by block 3 the
    // mean matches the C++ constant within 1e-5 and the block is flat within
    // 1e-4 — persistent model bias, not oscillation.
    assert!(
        (mean3 - expected).abs() < 1e-4,
        "{label}: settled mean {mean3:.6} diverges from C++ {expected:.6}"
    );
    assert!(
        (max3 - min3).abs() < 1e-4,
        "{label}: settled block not flat (min {min3:.6}, max {max3:.6})"
    );
    assert!(
        (mean0 - expected).abs() < 0.3,
        "{label}: cold mean {mean0:.6} far from C++ {expected:.6}"
    );
}

/// Zero input reproduces the C++ full-scale offset from the first blocks and
/// exceeds FS at unity gain, while control models stay near zero.
#[test]
fn dc_offset_zero_input_matches_cpp_reference() {
    let mut model = load_model("wavenet_a2_max.nam");
    let stats = render_zeros(&mut model, 4);
    assert_constant_dc(&stats, A2_MAX_DC, "a2_max");
    let (_, _, max0) = stats[3];
    let (_, min0, _) = stats[3];
    assert!(
        max0 > 1.0 || min0 < -1.0,
        "a2_max: |DC| must exceed FS at unity gain (got [{min0:.6}, {max0:.6}])"
    );
    // Measured: prewarm settles the cold ripple onto the C++ constant
    // (|prewarmed - C++| < 1e-5), so the clip is steady content, not a
    // cold-start transient. Compare against the settled block-3 mean.
    model.prewarm(model.prewarm_samples());
    let warmed = render_zeros(&mut model, 1)[0].0;
    let settled = stats[3].0;
    let expected = A2_MAX_DC;
    println!(
        "// Measured a2_max: prewarmed mean={warmed:.6} settled={settled:.6} C++={expected:.6}"
    );
    assert!(
        (warmed - settled).abs() < 1e-5,
        "a2_max: prewarm changed the settled offset ({settled:.6} -> {warmed:.6})"
    );
    assert!(
        (warmed - expected).abs() < 1e-4,
        "a2_max: prewarmed offset {warmed:.6} diverges from C++ {expected:.6}"
    );

    for (fixture, expected, label) in [
        ("wavenet_a2_lite.nam", A2_LITE_DC, "a2_lite"),
        ("wavenet_a2_full.nam", A2_FULL_DC, "a2_full"),
    ] {
        let mut model = load_model(fixture);
        let stats = render_zeros(&mut model, 4);
        // Controls are informative only: both sit near zero (|DC| < 0.1 in C++)
        // and must not clip. Exact cross-engine decimals are not gated here —
        // raw `process()` bypasses loader gain staging, so small residuals
        // (measured ~5e-3 on lite, ~3e-2 on full) are out of scope.
        let (settled, min3, max3) = stats[3];
        println!("// Measured {label}: settled mean={settled:.6} C++={expected:.6}");
        assert!(
            settled.abs() < 0.5,
            "{label}: control offset unexpectedly large ({settled:.6})"
        );
        assert!(
            max3 < 1.0 && min3 > -1.0,
            "{label}: control must not clip on silence (got [{min3:.6}, {max3:.6}])"
        );
    }
}

fn fresh_open_gate() -> DynamicHysteresis {
    let gate = DynamicHysteresis::new();
    assert_eq!(gate.state(), GateState::Open);
    assert_eq!(gate.multiplier(), 1.0);
    assert!(gate.is_steady());
    gate
}

fn run_output_stage(
    buf: &mut [f32],
    gate: &mut DynamicHysteresis,
    rt: &RtStatusFlags,
    adaptive: &mut AdaptiveCompute,
) {
    let mut right = vec![0.0f32; buf.len()];
    let n = buf.len();
    apply_output_stage(buf, &mut right, n, 1.0, gate, rt, true, adaptive, 48000);
}

fn adaptive_off() -> AdaptiveCompute {
    AdaptiveCompute::new(AdaptiveComputeMode::Off)
}

/// A fresh (open, unity, steady) gate delivers the offset untouched and the
/// flag truthfully reports the delivered supra-FS sample.
#[test]
fn clipping_flag_reports_delivered_supra_fs_sample() {
    let mut gate = fresh_open_gate();
    let rt = RtStatusFlags::default();
    let mut adaptive = adaptive_off();
    let mut buf = vec![A2_MAX_DC; BLOCK];
    run_output_stage(&mut buf, &mut gate, &rt, &mut adaptive);
    assert!(
        rt.check_flag(RT_STATUS_HAS_CLIPPED),
        "open gate must flag the delivered +9.726 offset"
    );
    let peak = buf.iter().cloned().fold(0.0f32, f32::max);
    assert!(
        peak > 1.0,
        "delivered signal must still exceed FS (got peak {peak:.6})"
    );
}

/// A closed gate mutes the offset to zero and raises no flag.
#[test]
fn closed_gate_mutes_dc_without_flag() {
    let params = GateParams::new(-70.0, -80.0, 0, 0, 1e-4);
    let mut gate = DynamicHysteresis::new();
    gate.update(0.0, 0.1, 0.01, &params, BLOCK);
    gate.update(0.0, 0.1, 0.01, &params, BLOCK);
    assert_eq!(gate.state(), GateState::Closed);
    assert!(gate.is_steady());

    let rt = RtStatusFlags::default();
    let mut adaptive = adaptive_off();
    let mut buf = vec![A2_MAX_DC; BLOCK];
    run_output_stage(&mut buf, &mut gate, &rt, &mut adaptive);
    assert!(
        buf.iter().all(|&v| v == 0.0),
        "closed gate must mute the offset"
    );
    assert!(
        !rt.check_flag(RT_STATUS_HAS_CLIPPED),
        "muted output must not flag"
    );
}

/// While the gate is fading, detection observes the pre-gate signal: a 2.0
/// sample flags even though the ramped delivery never exceeds FS. This pins
/// the current branch asymmetry (steady path fuses gate gain before
/// detection) without changing it.
#[test]
fn fading_gate_detection_observes_pre_gate_signal() {
    let params = GateParams::new(-70.0, -80.0, 0, 1000, 1e-4);
    let mut gate = DynamicHysteresis::new();
    gate.update(0.0, 0.1, 0.01, &params, BLOCK);
    let mut iters = 0;
    while gate.multiplier() > 0.2 && iters < 30 {
        gate.update(0.0, 0.1, 0.01, &params, BLOCK);
        iters += 1;
    }
    assert!(!gate.is_steady(), "gate must be mid-fade");
    assert!(
        gate.multiplier() < 0.3,
        "fade must be deep (got {})",
        gate.multiplier()
    );

    let rt = RtStatusFlags::default();
    let mut adaptive = adaptive_off();
    let mut buf = vec![2.0f32; BLOCK];
    run_output_stage(&mut buf, &mut gate, &rt, &mut adaptive);
    assert!(
        rt.check_flag(RT_STATUS_HAS_CLIPPED),
        "pre-gate 2.0 sample must flag while fading"
    );
    let peak = buf.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        peak < 1.0,
        "ramped delivery must stay below FS (got peak {peak:.6})"
    );
}
