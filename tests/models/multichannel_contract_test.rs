// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Fail-closed contract for the safe multichannel helpers.
//!
//! `NamModel::process_multichannel` / `process_multichannel_in_place` must
//! reject host plane sets that do not satisfy the model's declared geometry
//! with [`NamErrorCode::HostFormatContractViolation`] (`E2304`) **before** any
//! raw pointer is dereferenced. Under-sized inputs previously produced
//! null-pointer reads inside the Linear kernels; these tests pin the
//! fail-closed behaviour, the surplus-plane ignoring rule and the
//! minimum-length frame accounting.

use neural_amp_modeler_rs::common::diagnostics::NamErrorCode;
use neural_amp_modeler_rs::loader::nam_json::{LinearImplementation, LinearTopology};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::linear::LinearModel;

/// Builds a Linear model with explicit deterministic weights/biases.
fn make_linear(in_ch: usize, out_ch: usize, weights: Vec<f32>, biases: Vec<f32>) -> LinearModel {
    let topo = LinearTopology {
        in_channels: in_ch,
        out_channels: out_ch,
        receptive_field: weights.len() / topo_kernels(in_ch, out_ch).max(1),
        has_bias: !biases.is_empty(),
        implementation: LinearImplementation::Direct,
    };
    LinearModel::new_with_topology(topo, weights, biases).expect("synthetic Linear model")
}

/// Number of FIR kernels required by a Linear geometry (mirrors the loader).
fn topo_kernels(in_ch: usize, out_ch: usize) -> usize {
    if in_ch == out_ch {
        1
    } else {
        in_ch.max(out_ch)
    }
}

/// Builds a Linear model with `rf` identical taps per kernel.
fn make_linear_uniform(in_ch: usize, out_ch: usize, rf: usize) -> LinearModel {
    let weights = vec![0.25f32; topo_kernels(in_ch, out_ch) * rf];
    let biases = vec![0.0f32; if in_ch == out_ch { 1 } else { out_ch }];
    let topo = LinearTopology {
        in_channels: in_ch,
        out_channels: out_ch,
        receptive_field: rf,
        has_bias: true,
        implementation: LinearImplementation::Direct,
    };
    LinearModel::new_with_topology(topo, weights, biases).expect("synthetic Linear model")
}

#[test]
fn linear_2to1_with_one_plane_is_rejected() {
    let mut model = make_linear_uniform(2, 1, 8);
    let input = [0.0f32; 32];
    let mut output = [0.0f32; 32];

    let result = model.process_multichannel(&[&input[..]], &mut [&mut output[..]]);
    assert_eq!(result, Err(NamErrorCode::HostFormatContractViolation));
}

#[test]
fn linear_1to2_in_place_with_one_plane_is_rejected() {
    let mut model = make_linear_uniform(1, 2, 8);
    let mut plane = [0.0f32; 32];

    let result = model.process_multichannel_in_place(&mut [&mut plane[..]]);
    assert_eq!(result, Err(NamErrorCode::HostFormatContractViolation));
}

#[test]
fn extra_planes_are_ignored_and_frames_use_required_only() {
    // 1 -> 1: only one input and one output plane are required. A shorter
    // surplus plane must not shrink the frame count.
    let mut model = make_linear_uniform(1, 1, 8);
    let long = [0.5f32; 64];
    let short = [0.1f32; 8];
    let mut out = [0.0f32; 64];

    let frames = model
        .process_multichannel(&[&long[..], &short[..]], &mut [&mut out[..]])
        .expect("required geometry present");
    assert_eq!(frames, 64, "surplus short plane must be ignored");
}

#[test]
fn mismatched_lengths_process_min_frames() {
    let mut model = make_linear_uniform(2, 1, 8);
    let a = [0.5f32; 64];
    let b = [0.25f32; 40];
    let mut out = [0.0f32; 100];

    let frames = model
        .process_multichannel(&[&a[..], &b[..]], &mut [&mut out[..]])
        .expect("required geometry present");
    assert_eq!(frames, 40, "frames must be the min across required planes");
}

#[test]
fn rejected_geometry_leaves_outputs_and_state_untouched() {
    // A rejected call must fail before any pointer is dereferenced: the output
    // planes stay untouched and the model's temporal state does not advance.
    let w = [1.0f32, 0.5, 0.25, 0.0];
    let bias = 0.125f32;
    let x0 = [0.5f32; 32];
    let x1 = [0.25f32; 32];

    // Seed the recurrent (delay-line) state of both models identically.
    let mut model = make_linear(2, 2, w.to_vec(), vec![bias]);
    let mut seed0 = [0.0f32; 16];
    let mut seed1 = [0.0f32; 16];
    model
        .process_multichannel(
            &[&x0[..16], &x1[..16]],
            &mut [&mut seed0[..], &mut seed1[..]],
        )
        .expect("warm-up geometry is valid");

    // A rejected call (only one output plane for a 2-output model) must not
    // write the sentinel-filled output plane.
    let mut sentinel = [7.5f32; 32];
    let result = model.process_multichannel(&[&x0[..], &x1[..]], &mut [&mut sentinel[..]]);
    assert_eq!(result, Err(NamErrorCode::HostFormatContractViolation));
    assert!(
        sentinel.iter().all(|&v| v == 7.5),
        "rejected call must not touch the output plane"
    );

    // State must be unchanged: the continuation must match a model that only
    // ever saw the same seed (both get one seed call, then one continuation).
    let mut reference = make_linear(2, 2, w.to_vec(), vec![bias]);
    let mut scratch0 = [0.0f32; 16];
    let mut scratch1 = [0.0f32; 16];
    reference
        .process_multichannel(
            &[&x0[..16], &x1[..16]],
            &mut [&mut scratch0[..], &mut scratch1[..]],
        )
        .expect("reference seed is valid");

    let mut r0 = [0.0f32; 16];
    let mut r1 = [0.0f32; 16];
    reference
        .process_multichannel(&[&x0[..16], &x1[..16]], &mut [&mut r0[..], &mut r1[..]])
        .expect("reference continuation is valid");

    let mut m0 = [0.0f32; 16];
    let mut m1 = [0.0f32; 16];
    model
        .process_multichannel(&[&x0[..16], &x1[..16]], &mut [&mut m0[..], &mut m1[..]])
        .expect("continuation after rejection is valid");

    assert_eq!(m0, r0, "rejected call must not advance channel 0 state");
    assert_eq!(m1, r1, "rejected call must not advance channel 1 state");
}

#[test]
fn linear_2to2_in_place_matches_f64_oracle() {
    // In-place 2 -> 2: each output channel maps onto its own input plane and
    // must reproduce the FIR oracle, proving reads happen before writes.
    let w = [1.0f32, 0.5, 0.25, 0.0];
    let bias = 0.125f32;
    let mut model = make_linear(2, 2, w.to_vec(), vec![bias]);

    let mut p0 = [0.0f32; 16];
    p0[0] = 1.0;
    let mut p1 = [0.0f32; 16];

    let frames = model
        .process_multichannel_in_place(&mut [&mut p0[..], &mut p1[..]])
        .expect("2 -> 2 in-place geometry is valid");
    assert_eq!(frames, 16);

    for i in 0..16 {
        let expected0 = bias as f64 + if i < w.len() { w[i] as f64 } else { 0.0 };
        let expected1 = bias as f64;
        assert!(
            (p0[i] as f64 - expected0).abs() < 1e-6,
            "in-place ch0[{i}] = {} vs f64 oracle {expected0}",
            p0[i]
        );
        assert!(
            (p1[i] as f64 - expected1).abs() < 1e-6,
            "in-place ch1[{i}] = {} vs f64 oracle {expected1}",
            p1[i]
        );
    }
}

#[test]
fn linear_2to2_matches_f64_fir_oracle() {
    // Shared forward-time kernel `w`; y_c[n] = bias + Σ_k w[k] * x_c[n-k].
    let w = [1.0f32, 0.5, 0.25, 0.0];
    let bias = 0.125f32;
    let mut model = make_linear(2, 2, w.to_vec(), vec![bias]);

    let mut x0 = [0.0f32; 16];
    x0[0] = 1.0;
    let x1 = [0.0f32; 16];
    let mut out0 = [0.0f32; 16];
    let mut out1 = [0.0f32; 16];

    let frames = model
        .process_multichannel(&[&x0[..], &x1[..]], &mut [&mut out0[..], &mut out1[..]])
        .expect("2 -> 2 geometry present");
    assert_eq!(frames, 16);

    for i in 0..16 {
        let expected0 = bias as f64 + if i < w.len() { w[i] as f64 } else { 0.0 };
        let expected1 = bias as f64;
        assert!(
            (out0[i] as f64 - expected0).abs() < 1e-6,
            "ch0[{i}] = {} vs f64 oracle {expected0}",
            out0[i]
        );
        assert!(
            (out1[i] as f64 - expected1).abs() < 1e-6,
            "ch1[{i}] = {} vs f64 oracle {expected1}",
            out1[i]
        );
    }
}
