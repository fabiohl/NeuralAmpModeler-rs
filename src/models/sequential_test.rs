// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Entity-level tests for the `SequentialModel` chain: DEC-01 sample-rate
//! resolution (`resolve_expected_sample_rate`), channel-link validation and
//! geometry reporting, prewarm lifecycle (child flag save/restore and
//! propagation, saturating sum, single chain-wide stabilization pass), and
//! process parity against a manual series of the same stages (C++
//! `test_sequential.cpp` conventions).

use super::{NamModel, SequentialModel, resolve_expected_sample_rate, saturating_prewarm_sum};
use crate::common::alloc_audit::{
    TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
};
use crate::common::diagnostics::NamErrorCode;
use crate::loader::nam_json::{LinearImplementation, LinearTopology};
use crate::models::StaticModel;
use crate::models::linear::LinearModel;
use crate::models::lstm::LstmModel1;

/// Default 48000 Hz standalone fallback (DEC-01 resolution tail).
const EXPECTED_FALLBACK: f32 = 48000.0;

/// Builds a Linear static stage. `kernel` holds the forward-time tap set
/// (identity semantics for `[1.0, 0.0]`-style taps); the kernel is repeated
/// across the fanned channel dimension (`max(in, out)` kernels).
fn linear_stage(in_ch: usize, out_ch: usize, bias: f32, kernel: &[f32]) -> StaticModel {
    let rf = kernel.len();
    let topo = LinearTopology {
        in_channels: in_ch,
        out_channels: out_ch,
        receptive_field: rf,
        has_bias: bias != 0.0,
        implementation: LinearImplementation::Direct,
    };
    let fan = in_ch.max(out_ch);
    let mut weights = Vec::with_capacity(fan * rf);
    for _ in 0..fan {
        weights.extend_from_slice(kernel);
    }
    let biases = match (in_ch, out_ch) {
        (1, _) => vec![bias; out_ch],
        (_, 1) => vec![bias],
        _ => panic!("N->M chains are out of scope for the test helper"),
    };
    StaticModel::Linear(Box::new(
        LinearModel::new_with_topology(topo, weights, biases).expect("Linear stage allocation"),
    ))
}

/// Builds a `[b1; b2]`-shaped two-stage mono chain through the typed
/// constructor.
fn chain_two_linears(first: &[f32], second: &[f32]) -> Result<Box<SequentialModel>, NamErrorCode> {
    let stages = vec![
        linear_stage(1, 1, 0.0, first),
        linear_stage(1, 1, 0.0, second),
    ];
    let rates = vec![None, None];
    SequentialModel::new(stages, rates, Some(EXPECTED_FALLBACK)).map(Box::new)
}

/// Builds a fresh zero-weight [`LstmModel1`] stage (1->1) as a static model.
// LstmModel1::new() is constructor-based (recurrent state zeroed by
// construction), not a field-complete Default; box_default fires on the
// explicit construction pattern anyway.
#[expect(clippy::box_default)]
fn lstm_stage() -> StaticModel {
    StaticModel::Lstm1x3(Box::new(LstmModel1::<3, 4, 12>::new()))
}
/// Deterministic pseudo-audio input block (stable across runs/platforms).
fn test_input(num_samples: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| (i as f32 * 0.017_3).sin() * 0.5)
        .collect()
}

/// Feeds `input` through the chain in the C++ `process_model` chunking
/// schedule (mirrors test_sequential.cpp:101-110: `requested` chunks wrap in
/// the schedule, and the tail truncates to the remaining input).
fn chunked_process(model: &mut impl NamModel, input: &[f32], chunk_sizes: &[usize]) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let mut offset = 0usize;
    let mut chunk_index = 0usize;
    while offset < input.len() {
        let requested = chunk_sizes[chunk_index % chunk_sizes.len()];
        let take = requested.min(input.len() - offset);
        model.process(
            &input[offset..offset + take],
            &mut output[offset..offset + take],
        );
        offset += take;
        chunk_index += 1;
    }
    output
}

/// Renders the manual two-stage mono series for the same chunk schedule (C++
/// `process_models_in_series`, test_sequential.cpp:113-131): stage A is
/// process per chunk into the intermediate, stage B consumes the
/// intermediate, each offset-aligned.
fn manual_series(
    first: &mut impl NamModel,
    second: &mut impl NamModel,
    input: &[f32],
    chunk_sizes: &[usize],
    max_chunk: usize,
) -> Vec<f32> {
    first.reset(48000, max_chunk).expect("first reset");
    second.reset(48000, max_chunk).expect("second reset");
    let mut intermediate = vec![0.0f32; input.len()];
    let mut output = vec![0.0f32; input.len()];
    let mut offset = 0usize;
    let mut chunk_index = 0usize;
    while offset < input.len() {
        let requested = chunk_sizes[chunk_index % chunk_sizes.len()];
        let take = requested.min(input.len() - offset);
        first.process(
            &input[offset..offset + take],
            &mut intermediate[offset..offset + take],
        );
        second.process(
            &intermediate[offset..offset + take],
            &mut output[offset..offset + take],
        );
        offset += take;
        chunk_index += 1;
    }
    output
}

// ── DEC-01 resolution table (case table of cpp_parity_map.md §5.1) ──────────

#[test]
fn test_dec01_case_table() {
    let unknown: Option<f32> = None;
    let known = Some(48000.0f32);
    let k44 = Some(44100.0f32);

    // ∅ | ∅ | ∅ → 48000 (all-unknown fallback; C++ keeps the -1 sentinel).
    let got = resolve_expected_sample_rate(&[unknown, unknown], unknown)
        .expect("all-unknown resolves via fallback");
    assert_eq!(got, EXPECTED_FALLBACK);

    // ∅ | 44100 | ∅ → 44100 (dictating child).
    let got = resolve_expected_sample_rate(&[k44, unknown], unknown).expect("child dictates");
    assert_eq!(got, 44100.0);

    // 44100 | 48000 | ∅ → REJECTED (root conflicts with the first child).
    match resolve_expected_sample_rate(&[k44, known], unknown) {
        Err(NamErrorCode::SequentialSampleRateMismatch) => {}
        other => panic!("cross-child conflict must reject, got: {other:?}"),
    }

    // ∅ | ∅ | 48000 → 48000 (top level dictates; unknown children conform).
    let got = resolve_expected_sample_rate(&[unknown, unknown], known).expect("top dictates");
    assert_eq!(got, 48000.0);

    // 44100 | ∅ | 48000 → REJECTED (top level conflicts with a known child).
    match resolve_expected_sample_rate(&[unknown, known], k44) {
        Err(NamErrorCode::SequentialSampleRateMismatch) => {}
        other => panic!("top-level conflict must reject, got: {other:?}"),
    }

    // Late conflicts in any position also reject.
    match resolve_expected_sample_rate(&[known, unknown, k44], unknown) {
        Err(NamErrorCode::SequentialSampleRateMismatch) => {}
        other => panic!("late child conflict must reject, got: {other:?}"),
    }

    // Homogeneous children under an unspecified root conform calmly.
    let got = resolve_expected_sample_rate(&[known, known], unknown).expect("homogeneous");
    assert_eq!(got, 48000.0);
}

#[test]
fn test_dec01_resolution_is_exact_f32_equality() {
    // Identical integral literals compare exactly (the C++ `double` compare
    // mirror); anything bit-different is a conflict regardless of magnitude.
    assert!(
        resolve_expected_sample_rate(&[Some(44100.0), Some(44100.0)], None).is_ok(),
        "identical integral rates accept"
    );
    let odd = Some(44101.0f32);
    assert_eq!(
        resolve_expected_sample_rate(&[Some(44100.0), odd], None)
            .err()
            .unwrap(),
        NamErrorCode::SequentialSampleRateMismatch,
        "any differing known rate rejects"
    );
}

// ── Channel geometry and link validation (C++ ctor + L82-93) ────────────────

#[test]
fn test_sequential_rejects_empty_stages() {
    // Mirrors test_sequential.cpp:244 test_sequential_rejects_empty_models.
    let err = SequentialModel::new(Vec::new(), Vec::new(), Some(EXPECTED_FALLBACK))
        .err()
        .expect("empty chains reject");
    assert_eq!(err, NamErrorCode::SequentialEmptyModels);
}

#[test]
fn test_sequential_rejects_channel_mismatch() {
    // Mirrors test_sequential.cpp:284 test_sequential_rejects_channel_mismatch:
    // stage 0 (1->2) feeding stage 1 (1->1) breaks `out(0) == in(1)`.
    let stages = vec![
        linear_stage(1, 2, 0.0, &[1.0, 0.0]),
        linear_stage(1, 1, 0.0, &[1.0]),
    ];
    let err = SequentialModel::new(stages, vec![None, None], None)
        .err()
        .expect("channel mismatch must reject");
    assert_eq!(err, NamErrorCode::SequentialChannelMismatch);

    // The mirrored geometry (1->1 feeding 2->1) fails the same link.
    let stages = vec![
        linear_stage(1, 1, 0.0, &[1.0]),
        linear_stage(2, 1, 0.0, &[1.0, 0.0]),
    ];
    let err = SequentialModel::new(stages, vec![None, None], None)
        .err()
        .expect("channel mismatch must reject");
    assert_eq!(err, NamErrorCode::SequentialChannelMismatch);
}

#[test]
fn test_sequential_builds_valid_geometry_and_reports_it() {
    // 1 -> 2 -> 1 with identity kernels: legal per the channel-link scan.
    let stages = vec![
        linear_stage(1, 2, 0.0, &[1.0, 0.0]),
        linear_stage(2, 1, 0.0, &[1.0, 0.0]),
    ];
    let mut chain =
        SequentialModel::new(stages, vec![None, None], None).expect("valid chain builds");
    assert_eq!(chain.num_stages(), 2);
    assert_eq!(chain.in_channels(), 1);
    assert_eq!(chain.out_channels(), 1);
    assert_eq!(chain.stage_output_channels(0), Some(2));
    assert_eq!(chain.stage_output_channels(1), Some(1));
    assert_eq!(chain.stage_output_channels(2), None);
    assert_eq!(
        chain.expected_sample_rate(),
        EXPECTED_FALLBACK,
        "all-unknown chain resolves to the 48 kHz engine default"
    );

    // The mono front/back chain accepts the mono process contract. Identity
    // taps on both stages make the interior 2-channel hop a summing mixer:
    // y = (x + x) = 2 * x, bit-exactly.
    let input = test_input(64);
    let mut output = vec![0.0f32; 64];
    chain.process(&input, &mut output);
    for (i, sample) in output.iter().enumerate() {
        assert!(sample.is_finite(), "identity chain must stay finite");
        let doubled = input[i] + input[i];
        assert_eq!(
            *sample, doubled,
            "identity kernel chain must pass 2x the signal at sample {i}"
        );
    }
}

#[test]
fn test_sequential_accepts_nested_chain_child() {
    // Mirrors test_sequential.cpp:233 test_sequential_accepts_nested_sequential_child:
    // an already-built chain embeds as a stage; channels propagate.
    let inner = SequentialModel::new(
        vec![
            linear_stage(1, 2, 0.0, &[1.0, 0.0]),
            linear_stage(2, 1, 0.0, &[1.0, 0.0]),
        ],
        vec![None, None],
        None,
    )
    .expect("inner chain");
    let stages: Vec<StaticModel> = vec![
        StaticModel::Sequential(Box::new(inner)),
        linear_stage(1, 1, 0.0, &[1.0]),
    ];
    let chain = SequentialModel::new(stages, vec![None, None], None).expect("nested chain builds");
    assert_eq!(chain.num_stages(), 2);
    assert_eq!(chain.in_channels(), 1);
    assert_eq!(chain.out_channels(), 1);
}

// ── Prewarm sum + lifecycle (C++ L152-199) ──────────────────────────────────

#[test]
fn test_saturating_prewarm_sum() {
    // GetPrewarmSamples saturates instead of overflowing (C++ L193-196);
    // the Rust engine saturates at usize::MAX.
    let max = usize::MAX;
    assert_eq!(saturating_prewarm_sum(std::iter::empty()), 0);
    assert_eq!(saturating_prewarm_sum([1usize, 2, 3].into_iter()), 6);
    assert_eq!(saturating_prewarm_sum([max, 5, 7].into_iter()), max);
    let mid = usize::MAX - 5;
    assert_eq!(
        saturating_prewarm_sum([mid, 5, 7].into_iter()),
        max,
        "a partial overflow saturates at usize::MAX"
    );
}

#[test]
fn test_sequential_prewarm_samples_is_child_sum() {
    // Linear children report 0; an LSTM child reports 0.5 * expected rate
    // (24000 at the LSTM's default 48000 Hz).
    let stages = vec![lstm_stage(), linear_stage(1, 1, 0.0, &[1.0, 0.0])];
    let chain = SequentialModel::new(stages, vec![None, None], None).expect("chain builds");
    assert_eq!(chain.prewarm_samples(), 24000usize);

    // Linear-only chains stabilize in zero samples.
    let chain = chain_two_linears(&[1.0, 0.5], &[-2.0]).expect("chain builds");
    assert_eq!(chain.prewarm_samples(), 0);
}

#[test]
fn test_sequential_set_prewarm_on_reset_propagates() {
    let mut chain = chain_two_linears(&[1.0, 0.5], &[-2.0]).expect("chain builds");
    chain.set_prewarm_on_reset(false);
    assert!(!chain.prewarm_on_reset());
    for model in chain.stages() {
        assert!(
            !model.prewarm_on_reset(),
            "child flag must follow the chain flag"
        );
    }
    chain.set_prewarm_on_reset(true);
    assert!(chain.prewarm_on_reset());
    for model in chain.stages() {
        assert!(
            model.prewarm_on_reset(),
            "child flag must follow the chain flag"
        );
    }
}

#[test]
fn test_sequential_reset_preserves_child_flags() {
    let mut chain = chain_two_linears(&[1.0, 0.5], &[-2.0]).expect("chain builds");
    // Asymmetric initial child states, like the C++ save/restore dance.
    let stages = chain.stages();
    stages[0].set_prewarm_on_reset(false);
    stages[1].set_prewarm_on_reset(true);

    chain.reset(48000, 64).expect("reset");

    let stages = chain.stages();
    assert!(
        !stages[0].prewarm_on_reset(),
        "disabled child flag must be restored exactly"
    );
    assert!(
        stages[1].prewarm_on_reset(),
        "enabled child flag must be restored exactly"
    );
    assert!(
        chain.prewarm_complete(),
        "the integral reset runs its own stabilization; the split state stays clear"
    );
}

#[test]
fn test_sequential_process_matches_manual_series() {
    // Mirrors test_sequential.cpp:172 test_sequential_process_matches_manual_series:
    // [0.25, 0.5] tap set (RF 2) feeding a [-0.75] tap set (RF 1), same
    // chunk schedule as the C++ test {1, 7, 32, 5, 64} over 257 samples.
    let mut chain = chain_two_linears(&[0.25, 0.5], &[-0.75]).expect("chain builds");
    let chunk_sizes = [1usize, 7, 32, 5, 64];
    let max_chunk = chunk_sizes.iter().copied().max().unwrap_or(64);
    chain.reset(48000, max_chunk).expect("chain reset");

    let input = test_input(257);
    let actual = chunked_process(chain.as_mut(), &input, &chunk_sizes);

    // Manual reference built through the same stage helpers as the chain.
    let mut first = linear_stage(1, 1, 0.0, &[0.25, 0.5]);
    let mut second = linear_stage(1, 1, 0.0, &[-0.75]);
    let expected = manual_series(&mut first, &mut second, &input, &chunk_sizes, max_chunk);

    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1.0e-7,
            "chain output must match the manual series at sample {i}: {a} vs {b}"
        );
    }
}

#[test]
fn test_sequential_multichain_process_matches_manual_series() {
    // 1 -> 2 -> 1 interior geometry inside the chain (mono front/back): the
    // chain's hot path must equal the manual per-stage series computed via
    // the multichannel helpers, including the interior 2-channel hop.
    let mut chain = SequentialModel::new(
        vec![
            linear_stage(1, 2, 0.25, &[1.0, 0.0]),
            linear_stage(2, 1, -1.5, &[1.0]),
        ],
        vec![None, None],
        None,
    )
    .expect("chain builds");
    let chunk_sizes = [3usize, 64, 1, 32];
    let max_chunk = chunk_sizes.iter().copied().max().unwrap_or(64);
    chain.reset(48000, max_chunk).expect("chain reset");

    let input = test_input(131);
    let actual = chunked_process(&mut chain, &input, &chunk_sizes);

    // Manual reference built through the same stage helpers as the chain.
    let mut first = linear_stage(1, 2, 0.25, &[1.0, 0.0]);
    let mut second = linear_stage(2, 1, -1.5, &[1.0]);
    first.reset(48000, max_chunk).expect("first reset");
    second.reset(48000, max_chunk).expect("second reset");

    let mut ch0 = vec![0.0f32; input.len()];
    let mut ch1 = vec![0.0f32; input.len()];
    let mut expected = vec![0.0f32; input.len()];
    let mut offset = 0usize;
    let mut chunk_index = 0usize;
    while offset < input.len() {
        let requested = chunk_sizes[chunk_index % chunk_sizes.len()];
        let take = requested.min(input.len() - offset);
        first.process_multichannel(
            &[&input[offset..offset + take]],
            &mut [
                &mut ch0[offset..offset + take],
                &mut ch1[offset..offset + take],
            ],
        );
        second.process_multichannel(
            &[&ch0[offset..offset + take], &ch1[offset..offset + take]],
            &mut [&mut expected[offset..offset + take]],
        );
        offset += take;
        chunk_index += 1;
    }

    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1.0e-6,
            "multichain output must match the manual series at sample {i}: {a} vs {b}"
        );
    }
}

/// Asserts the full zero-heap invariant (alloc/realloc/free) from the
/// unit-suite counters (the lib test build installs `CountingAllocator`
/// globally, so `TrackingGuard` observes real traffic).
fn assert_zero_heap(allocs: usize, reallocs: usize, deallocs: usize, label: &str) {
    assert_eq!(
        allocs, 0,
        "heap allocations detected on {label}! count={allocs}"
    );
    assert_eq!(
        reallocs, 0,
        "heap reallocations detected on {label}! count={reallocs}"
    );
    assert_eq!(
        deallocs, 0,
        "heap deallocations detected on {label}! count={deallocs}"
    );
}

#[test]
fn test_sequential_process_is_realtime_safe_after_warmup() {
    // Mirrors test_sequential.cpp:191: reset at the canonical 64-frame block,
    // one warm call first (the C++ matches Linear's lazy output-buffer path
    // convention), then a mixed-block audit window must count zero heap
    // traffic from the chain dispatch (boundary planes, pointer tables, and
    // the zero/sink feeds are all preallocated).
    let mut chain = chain_two_linears(&[0.5], &[-2.0]).expect("chain builds");
    chain.reset(48000, 64).expect("chain reset at 64 frames");

    let block_sizes = [1usize, 7, 32, 64];
    let inputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| test_input(bs)).collect();
    let mut outputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();
    for (bi, input) in inputs.iter().enumerate() {
        chain.process(input, &mut outputs[bi]);
    }

    let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
    let (allocs, reallocs, deallocs) = {
        let _guard = TrackingGuard::new();
        for _ in 0..iters {
            for (bi, input) in inputs.iter().enumerate() {
                chain.process(
                    std::hint::black_box(input),
                    std::hint::black_box(&mut outputs[bi]),
                );
            }
        }
        (get_alloc_count(), get_realloc_count(), get_dealloc_count())
    };
    assert_zero_heap(allocs, reallocs, deallocs, "Sequential mono process");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "exceeds the negotiated maximum")]
fn test_sequential_rejects_blocks_larger_than_reset_maximum() {
    // Mirrors test_sequential.cpp:211 (upstream C++ throws when num_frames
    // exceeds the reset-time maximum): the rejected-block contract on the
    // RT path is a debug trap — never silent state corruption in dev builds.
    let mut chain = chain_two_linears(&[0.5], &[-2.0]).expect("chain builds");
    chain.reset(48000, 4).expect("reset at maximum 4");
    let input = vec![0.25f32; 8];
    let mut output = vec![0.0f32; 8];
    chain.process(&input, &mut output);
}

#[test]
#[cfg(not(debug_assertions))]
fn test_sequential_truncates_blocks_larger_than_reset_maximum() {
    // Release half of the same upstream mirror: the RT path cannot unwind
    // (C++ throws), so the decided contract fails closed to controlled
    // truncation — first `max_buffer_size` frames processed, the caller's
    // output tail left byte-untouched, no panic.
    let mut chain = chain_two_linears(&[0.5], &[-2.0]).expect("chain builds");
    chain.reset(48000, 4).expect("reset at maximum 4");
    let input = vec![0.25f32; 8];
    let mut output = vec![7.7f32; 8];
    chain.process(&input, &mut output);

    // RF=1 stages with power-of-two taps: y = -2.0 * (0.5 * 0.25) = -0.25.
    for (i, k) in output.iter().enumerate().take(4) {
        assert_eq!(*k, -0.25f32, "first 4 frames are processed, sample {i}");
    }
    for i in 4..8 {
        assert_eq!(output[i], 7.7f32, "truncated tail stays untouched at {i}");
    }

    // The engine stays reusable after a truncated call — the next legal
    // block continues from the state left by the consumed frames.
    let next = [0.5f32];
    let mut output2 = vec![7.7f32; 1];
    chain.process(&next, &mut output2);
    assert_eq!(output2[0], -0.5f32, "y = -2.0 * (0.5 * 0.5)");
}

#[test]
fn test_split_vs_integral_bit_exact() {
    // A chain containing an LSTM child (24000 sample budget) plus a Linear
    // delay: the split flow (`prewarm_reset` + `prewarm_step` until
    // complete) must reproduce the integral reset stabilization bit-exactly
    // under arbitrary caller chunkings (frontier contract).
    let build = || -> Box<SequentialModel> {
        let stages = vec![lstm_stage(), linear_stage(1, 1, 0.0, &[1.0, 0.5])];
        Box::new(SequentialModel::new(stages, vec![None, None], None).expect("chain builds"))
    };

    let mut integral = build();
    integral.reset(48000, 256).expect("integral reset");
    let input = test_input(128);
    let mut got_integral = vec![0.0f32; input.len()];
    for (i, out) in got_integral.iter_mut().enumerate() {
        *out = sample_through(integral.as_mut(), input[i]);
    }

    let mut split = build();
    split.prewarm_reset();
    assert_eq!(split.prewarm_samples(), 24000usize);
    assert!(!split.prewarm_complete(), "stabilization armed");

    // Drain with a rotating step rhythm that exercises sub-chunking, larger
    // budgets, and single-sample steps.
    let chunk_cycle = [512usize, 4096, 1, 777, 8192, 2048];
    let mut chunk_index = 0usize;
    let mut guard = 0usize;
    while !split.prewarm_complete() {
        let _ = split.prewarm_step(chunk_cycle[chunk_index % chunk_cycle.len()]);
        chunk_index += 1;
        guard += 1;
        assert!(guard < 1000, "split stabilization must terminate");
    }
    assert!(split.prewarm_complete());

    let mut got_split = vec![0.0f32; input.len()];
    for (i, out) in got_split.iter_mut().enumerate() {
        *out = sample_through(split.as_mut(), input[i]);
    }
    for (i, (integral, split)) in got_integral.iter().zip(got_split.iter()).enumerate() {
        assert_eq!(
            integral, split,
            "split vs integral stabilization must be bit-identical at sample {i}"
        );
    }
}

/// Processes a single frame through the mono chain.
fn sample_through(model: &mut SequentialModel, frame: f32) -> f32 {
    let stream = [frame];
    let mut out = [0.0f32];
    model.process(&stream, &mut out);
    out[0]
}
