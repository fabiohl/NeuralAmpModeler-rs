// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Boundary behaviour of the planar <-> interleaved adapter.
//!
//! `NamModel::process_raw` (and the Linear mono fast path) stage the input
//! before writing the output when the buffers alias exactly, which removes the
//! `&[f32]` / `&mut [f32]` simultaneous-borrow undefined behaviour while
//! preserving bit-exact results. Sequential chains bridge multi-channel
//! interiors through their boundary planes and must still deliver the whole
//! block.

use neural_amp_modeler_rs::loader::nam_json::{LinearImplementation, LinearTopology};
use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::models::linear::LinearModel;
use neural_amp_modeler_rs::models::sequential::SequentialModel;

const RF: usize = 16;

fn make_linear(in_ch: usize, out_ch: usize, seed: f32) -> LinearModel {
    let kernels = if in_ch == out_ch {
        1
    } else {
        in_ch.max(out_ch)
    };
    let weights: Vec<f32> = (0..kernels * RF)
        .map(|i| ((i as f32) * 0.37 + seed).sin() * 0.5)
        .collect();
    let biases = vec![seed * 0.1; if in_ch == out_ch { 1 } else { out_ch }];
    let topo = LinearTopology {
        in_channels: in_ch,
        out_channels: out_ch,
        receptive_field: RF,
        has_bias: true,
        implementation: LinearImplementation::Direct,
    };
    LinearModel::new_with_topology(topo, weights, biases).expect("synthetic Linear model")
}

fn signal(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.11).sin()).collect()
}

fn signal_b(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.17 + 0.5).cos()).collect()
}

#[test]
fn in_place_mono_linear_matches_out_of_place() {
    let input = signal(4096);

    let mut reference = make_linear(1, 1, 0.3);
    let mut out_of_place = vec![0.0f32; input.len()];
    NamModel::process(&mut reference, &input, &mut out_of_place);

    let mut aliased = make_linear(1, 1, 0.3);
    let mut in_place = input.clone();
    let in_ptr = in_place.as_ptr();
    let out_ptr = in_place.as_mut_ptr();
    // SAFETY: `in_ptr`/`out_ptr` alias exactly; the adapter stages the input
    // before writing, which is the supported exact-alias case.
    unsafe { aliased.process_raw(&in_ptr, &out_ptr, in_place.len()) };

    assert_eq!(
        in_place, out_of_place,
        "exact in-place alias must be bit-identical to out-of-place"
    );
}

#[test]
fn in_place_multichannel_exact_alias_matches_out_of_place() {
    // 2 -> 2 exact alias across both channels (input plane c == output plane c).
    // The block exceeds the staging chunk (STAGE_SAMPLES / out_ch = 512 frames)
    // so the boundary must iterate the deinterleave staging several times.
    let n = 2000;
    let in0 = signal(n);
    let in1 = signal_b(n);

    let mut reference = make_linear(2, 2, 0.4);
    let mut ref0 = vec![0.0f32; n];
    let mut ref1 = vec![0.0f32; n];
    reference
        .process_multichannel(&[&in0[..], &in1[..]], &mut [&mut ref0[..], &mut ref1[..]])
        .expect("2 -> 2 geometry is valid");

    let mut aliased = make_linear(2, 2, 0.4);
    let mut io0 = in0.clone();
    let mut io1 = in1.clone();
    let in_ptrs = [io0.as_ptr(), io1.as_ptr()];
    let mut out_ptrs = [io0.as_mut_ptr(), io1.as_mut_ptr()];
    // SAFETY: both input planes alias their matching output plane exactly; the
    // adapter stages every input frame before any output is written.
    unsafe {
        aliased.process_raw(in_ptrs.as_ptr(), out_ptrs.as_mut_ptr(), n);
    }

    assert_eq!(
        io0, ref0,
        "exact-alias channel 0 (n={n} > staging chunk) must match out-of-place"
    );
    assert_eq!(
        io1, ref1,
        "exact-alias channel 1 (n={n} > staging chunk) must match out-of-place"
    );
}

#[test]
fn sequential_1_2_1_full_length() {
    let input = signal(2048);
    let stages = vec![
        StaticModel::Linear(Box::new(make_linear(1, 2, 0.2))),
        StaticModel::Linear(Box::new(make_linear(2, 1, 0.7))),
    ];
    let mut chain = SequentialModel::new(stages, vec![None, None], Some(48_000.0))
        .expect("synthetic 1->2->1 chain");

    let mut mono_out = vec![0.0f32; input.len()];
    chain.process(&input, &mut mono_out);

    let mut chain2 = SequentialModel::new(
        vec![
            StaticModel::Linear(Box::new(make_linear(1, 2, 0.2))),
            StaticModel::Linear(Box::new(make_linear(2, 1, 0.7))),
        ],
        vec![None, None],
        Some(48_000.0),
    )
    .expect("synthetic 1->2->1 chain");
    let mut plane_out = vec![0.0f32; input.len()];
    let frames = chain2
        .process_multichannel(&[&input], &mut [&mut plane_out])
        .expect("chain is 1 -> 1");
    assert_eq!(
        frames,
        input.len(),
        "chain must report the full frame count"
    );
    assert_eq!(
        plane_out, mono_out,
        "planar chain entry must match the mono entry bit-for-bit over the full length"
    );
}
