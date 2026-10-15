// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bit-identity suite for [`DelayLine::process_block`]: the block path must
//! reproduce the per-sample `push`/`pop` contract exactly (bit for bit,
//! including zero-priming, ring wraps, mid-stream retargets and resets)
//! across arbitrary capacities, delays and block sizes, and stay
//! allocation-free on the whole path.
//!
//! The oracle is the plain per-sample loop over the very same ring type —
//! any divergence between chunked bulk copies and interleaved index
//! arithmetic (out-of-order chunk writes, wrap miscounts, lost ring
//! history at retargets) fails loudly here.

use super::*;
use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
use proptest::prelude::*;

/// Deterministic pseudo-random signal in a non-degenerate float range
/// (xorshift over `u64`, never NaN/subnormal/commensurate with ring walk).
fn test_signal(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 33) as u32 as f32 / u32::MAX as f32 * 1.6 - 0.8
        })
        .collect()
}

/// Oracle: feeds `input` through `line` one sample at a time
/// (`push` then `pop`), returning the delayed output.
fn per_sample(line: &mut DelayLine<f32>, input: &[f32]) -> Vec<f32> {
    input
        .iter()
        .map(|&x| {
            line.push(x);
            line.pop()
        })
        .collect()
}

/// Feeds `input` through `line` in fixed-size chunks via `process_block`
/// (the last chunk keeps its remainder length).
fn by_block(line: &mut DelayLine<f32>, input: &[f32], block: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for start in (0..input.len()).step_by(block) {
        let end = (start + block).min(input.len());
        line.process_block(&input[start..end], &mut out[start..end]);
    }
    out
}

/// Asserts that two lines carry identical ring storage and cursor state
/// (white-box: this module is inside the `DelayLine` implementation file).
fn assert_identical_state(a: &DelayLine<f32>, b: &DelayLine<f32>) {
    assert_eq!(&*a.buf, &*b.buf, "ring storage diverged");
    assert_eq!(a.head, b.head, "write cursor diverged");
    assert_eq!(a.delay, b.delay, "delay state diverged");
}

/// Runs the same streams through two identical lines — one per-sample, one
/// block-processed — and asserts bit-exact outputs plus identical internal
/// state afterwards. `prefill` warms the delay through both lines before
/// the compared stream, so chunk reads also cover pre-block history.
fn assert_block_matches_samples(
    capacity: usize,
    delay: usize,
    prefill: usize,
    block: usize,
    stream_len: usize,
    seed: u64,
) {
    assert!(block >= 1 && stream_len >= 1, "degenerate sizes");
    let mut sampled = DelayLine::<f32>::with_capacity(capacity, delay);
    let mut blocked = DelayLine::<f32>::with_capacity(capacity, delay);

    // Shared warm-up history long enough to saturate both the applied
    // delay and at least one full ring wrap of the smallest ring shapes.
    let warm = test_signal(seed, prefill.max(delay.max(block) + capacity.min(64)));
    assert_eq!(
        per_sample(&mut sampled, &warm),
        by_block(&mut blocked, &warm, block),
        "cap={capacity} delay={delay} block={block}: warm-up diverged"
    );

    let stream = test_signal(seed ^ 0xDEADBEEF, stream_len);
    assert_eq!(
        per_sample(&mut sampled, &stream),
        by_block(&mut blocked, &stream, block),
        "cap={capacity} delay={delay} block={block}: block path diverged"
    );
    assert_identical_state(&sampled, &blocked);
}

#[test]
fn test_process_block_bit_identical_matrix() {
    // (capacity, delay) grid: passthrough, one-slot rings, a small ring
    // wrapping inside the delay class, maximum latency (`delay ==
    // capacity` forces one-sample chunks) and the long ring shape.
    let shapes: [(usize, usize); 10] = [
        (0, 0),
        (1, 0),
        (1, 1),
        (31, 3),
        (7, 4),
        (64, 64),
        (3, 1),
        (3200, 12),
        (3200, 512),
        (3200, 3200),
    ];
    for &(cap, delay) in &shapes {
        // Block sizes straddle ring wraps, chunk boundaries (`ring -
        // delay`) and exceed small rings entirely (`block > ring`).
        let block_set = [1usize, 2, 7, 64, 128, 257, 4096.min(cap * 4 + 7)];
        for &block in &block_set {
            for &prefill in &[0usize, 7usize, 333usize] {
                assert_block_matches_samples(cap, delay, prefill, block, 512, 0x5EED_0000);
            }
        }
    }
}

#[test]
fn test_process_block_clamps_n_to_shortest_slice() {
    // Both slices are bridged by their minimum, `n = min(in, out)`; the
    // excess of the longer slice is left untouched.
    let mut line = DelayLine::<f32>::with_capacity(16, 4);
    let input = [1.0f32; 32];
    let mut out = vec![7.5f32; 8];
    line.process_block(&input, &mut out);
    for (i, slot) in out.iter().enumerate() {
        let expected = if i >= 4 { 1.0 } else { 0.0 };
        assert_eq!(*slot, expected, "clamp to shortest slice at {i}");
    }

    let mut line = DelayLine::<f32>::with_capacity(16, 2);
    let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let mut out = vec![0.0f32; 32];
    line.process_block(&input, &mut out);
    for i in 0..8 {
        let expected = if i >= 2 { input[i - 2] } else { 0.0 };
        assert_eq!(out[i], expected, "shortest-bound shift at {i}");
    }
    assert!(
        out[8..].iter().all(|&s| s == 0.0),
        "output beyond `n` must be untouched"
    );
}

#[test]
fn test_process_block_delay_zero_is_passthrough_and_retains_ring_state() {
    // Passthrough blocks must copy verbatim while still absorbing the
    // newest samples, so a later `set_delay` reads them back exactly as a
    // per-sample line would (compare against the oracle over the same
    // history, including the samples pushed by the passthrough blocks).
    let cap = 16usize;
    let warm: Vec<f32> = (0..2 * cap).map(|i| (i % 13) as f32 * 0.25).collect();
    let tail = [9.5f32, 8.5, 7.5, 6.5];
    let next = [1.25f32, 2.25, 3.25, 4.25, 5.25];
    let steady: Vec<f32> = (0..8).map(|i| i as f32 * 0.125 - 0.5).collect();

    let mut line = DelayLine::<f32>::with_capacity(cap, 0);
    let _ = per_sample(&mut line, &warm);
    let mut out = vec![0.0f32; tail.len()];
    line.process_block(&tail, &mut out);
    assert_eq!(out, tail, "delay == 0 must copy the input verbatim");

    line.set_delay(3);
    let mut out = vec![0.0f32; next.len()];
    line.process_block(&next, &mut out);

    // Oracle replays the identical schedule per sample.
    let mut oracle = DelayLine::<f32>::with_capacity(cap, 0);
    let _ = per_sample(&mut oracle, &warm);
    let _ = per_sample(&mut oracle, &tail);
    oracle.set_delay(3);
    let expected = per_sample(&mut oracle, &next);
    assert_eq!(out, expected, "retarget after passthrough diverged");

    let expected = per_sample(&mut oracle, &steady);
    let mut out = vec![0.0f32; steady.len()];
    line.process_block(&steady, &mut out);
    assert_eq!(out, expected, "steady state after retarget diverged");
    assert_identical_state(&line, &oracle);
}

#[test]
fn test_process_block_survives_dynamic_retargets_and_resets() {
    // Mirrors the per-sample dynamic-retarget contract in 32-sample blocks,
    // including a clamped oversized delay, a mid-stream reset and retargets
    // grouped with the reset on the same block edge. Any chunked read that
    // crossed a retarget or a reset out of place diverges from the oracle.
    let cap = 64usize;
    let step = 32usize;
    let mut line = DelayLine::<f32>::with_capacity(cap, 4);
    // Retargets and the reset only on block edges (multiples of `step`).
    let changes = [(0usize, 16usize), (128, 63), (512, 0), (640, 500), (704, 7)];
    let reset_at = 256usize;
    let stream = test_signal(0xCAFE, 768);
    let mut out = vec![0.0f32; stream.len()];
    let mut next_change = 0;
    for start in (0..stream.len()).step_by(step) {
        if next_change < changes.len() && start >= changes[next_change].0 {
            line.set_delay(changes[next_change].1);
            next_change += 1;
        }
        if start == reset_at {
            line.reset();
        }
        line.process_block(&stream[start..start + step], &mut out[start..start + step]);
    }
    let mut oracle = DelayLine::<f32>::with_capacity(cap, 4);
    next_change = 0;
    let expected: Vec<f32> = stream
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if next_change < changes.len() && i >= changes[next_change].0 {
                oracle.set_delay(changes[next_change].1);
                next_change += 1;
            }
            if i == reset_at {
                oracle.reset();
            }
            oracle.push(x);
            oracle.pop()
        })
        .collect();
    assert_eq!(out, expected, "dynamic schedule diverged");
}

#[test]
fn test_copy_state_from_rebases_identical_ring_state() {
    // Cloning the state (storage + cursor + delay) from an advanced source
    // must make the target behave exactly like the source from then on,
    // including reads that reach back across ring wraps.
    for &(cap, delay) in &[(0usize, 0usize), (16, 8), (31, 3), (64, 63), (17, 17)] {
        let mut src = DelayLine::<f32>::with_capacity(cap, delay);
        let warm = test_signal(0xBEEF, 3 * (cap + 1) + 11);
        let _ = per_sample(&mut src, &warm);

        let mut dst = DelayLine::<f32>::with_capacity(cap, 0);
        dst.copy_state_from(&src);
        assert_identical_state(&src, &dst);

        let stream = test_signal(0xF00D, 129);
        assert_eq!(
            per_sample(&mut src, &stream),
            per_sample(&mut dst, &stream),
            "cap={cap} delay={delay}: cloned state must replay identically"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn prop_process_block_bit_identical(
        capacity in 0usize..96,
        delay in 0usize..96,
        block in 1usize..129,
        prefill in 0usize..64,
        seed in any::<u64>(),
    ) {
        assert_block_matches_samples(capacity, delay, prefill, block, 384, seed);
    }
}

#[test]
fn test_process_block_is_zero_alloc() {
    // Every fixture allocation happens before the guard: only
    // `process_block` itself must be allocation-free across the
    // passthrough shortcut, single-chunk and chunked (> ring) paths,
    // with mid-stream retargets.
    let mut line = DelayLine::<f32>::with_capacity(64, 4);
    line.push(1.0);
    let _ = line.pop();

    let input: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let mut out = vec![0.0f32; 1024];

    let _guard = TrackingGuard::new();
    for (delay, block) in [
        (0usize, 1usize),
        (0, 64),
        (4, 64),
        (63, 64),
        (10, 64),
        (63, 1024),
        (0, 1024),
    ] {
        line.set_delay(delay);
        for chunk in input.chunks(block) {
            let out_window = &mut out[..chunk.len()];
            out_window.fill(0.0);
            line.process_block(chunk, out_window);
        }
        assert_eq!(
            get_alloc_count(),
            0,
            "allocation detected in delay={delay} block={block}"
        );
    }
}

// ── Core DelayLine Push/Pop, Lifecycle and Property Tests ───────────────────

#[test]
fn test_delay_zero_is_passthrough() {
    let mut line = DelayLine::<f32>::new(0);
    assert_eq!(line.latency_samples(), 0);
    for i in 0..64 {
        let x = i as f32 * 0.01;
        line.push(x);
        assert_eq!(
            line.pop(),
            x,
            "delay=0 must return the just-pushed sample at {i}"
        );
    }
}

#[test]
fn test_delay_n_shifts_by_n_with_zero_priming() {
    for &delay in &[1usize, 12, 64, 256] {
        let mut line = DelayLine::<f32>::new(delay);
        assert_eq!(line.latency_samples(), delay);
        let n = delay + 64;
        let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 + 0.5).collect();
        for (i, &x) in input.iter().enumerate() {
            line.push(x);
            let expected = if i >= delay { input[i - delay] } else { 0.0 };
            assert_eq!(line.pop(), expected, "delay={delay} shift mismatch at {i}");
        }
    }
}

#[test]
fn test_push_pop_circular_wrap_exact_shift() {
    let delay = 128usize;
    let mut line = DelayLine::<f32>::new(delay);
    let n = 2048;
    let input: Vec<f32> = (0..n).map(|i| (i % 997) as f32 * 0.001).collect();
    for (i, &x) in input.iter().enumerate() {
        line.push(x);
        let expected = if i >= delay { input[i - delay] } else { 0.0 };
        assert_eq!(line.pop(), expected, "wrap shift mismatch at {i}");
    }
}

#[test]
fn test_resize_clears_and_updates_latency() {
    let mut line = DelayLine::<f32>::new(8);
    for i in 0..32 {
        line.push(i as f32);
        let _ = line.pop();
    }
    line.resize(32);
    assert_eq!(line.latency_samples(), 32);
    // Fresh history: first `delay` pops after resize are zero-primed.
    for i in 0..32 {
        line.push(0.5);
        assert_eq!(line.pop(), 0.0, "post-resize priming at {i}");
    }
    line.push(0.5);
    assert_eq!(
        line.pop(),
        0.5,
        "steady state after re-prime must shift correctly"
    );
    line.resize(0);
    assert_eq!(line.latency_samples(), 0);
    line.push(1.25);
    assert_eq!(line.pop(), 1.25);
}

#[test]
fn test_push_pop_is_zero_alloc() {
    let mut line = DelayLine::<f32>::new(64);
    // Warm once outside the guard.
    line.push(1.0);
    let _ = line.pop();
    let _guard = TrackingGuard::new();
    for i in 0..256 {
        line.push(i as f32 * 0.001);
        let _ = line.pop();
    }
    assert_eq!(
        get_alloc_count(),
        0,
        "push/pop must not allocate on the RT path"
    );
}

#[test]
fn test_with_capacity_preallocates_capacity_plus_one_and_clamps() {
    let line = DelayLine::<f32>::with_capacity(3200, 64);
    assert_eq!(
        line.capacity(),
        3200,
        "capacity accessor must match request"
    );
    assert_eq!(line.latency_samples(), 64);

    // Requested delay above capacity clamps to the maximum usable latency.
    let mut clamped = DelayLine::<f32>::with_capacity(8, 100);
    assert_eq!(clamped.capacity(), 8);
    assert_eq!(clamped.latency_samples(), 8);

    // The `capacity + 1` ring slots must sustain the maximum latency
    // (`delay == capacity`) with exact shifts across many wrap-arounds.
    let delay = clamped.capacity();
    let n = 4 * (delay + 1);
    let input: Vec<f32> = (0..n).map(|i| (i % 251) as f32 * 0.01).collect();
    for (i, &x) in input.iter().enumerate() {
        clamped.push(x);
        let expected = if i >= delay { input[i - delay] } else { 0.0 };
        assert_eq!(clamped.pop(), expected, "max-delay shift mismatch at {i}");
    }

    // Degenerate request still yields a usable passthrough line.
    let mut degenerate = DelayLine::<f32>::with_capacity(0, 0);
    assert_eq!(degenerate.capacity(), 0);
    degenerate.push(1.25);
    assert_eq!(degenerate.pop(), 1.25);
}

#[test]
fn test_set_delay_dynamic_transition_preserves_circular_integrity() {
    let mut line = DelayLine::<f32>::with_capacity(64, 4);
    // Mid-stream retargets, including delay == capacity (max latency).
    let changes = [(0usize, 4usize), (10, 31), (60, 0), (100, 63), (160, 7)];
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i % 173) as f32 * 0.01 + 0.25).collect();

    let mut next_change = 0;
    let mut delay = 0usize;
    for (i, &x) in input.iter().enumerate() {
        if next_change < changes.len() && changes[next_change].0 == i {
            delay = changes[next_change].1;
            line.set_delay(delay);
            assert_eq!(line.latency_samples(), delay, "retarget at {i}");
            next_change += 1;
        }
        line.push(x);
        // History is continuous across retargets: `pop` must return the
        // sample pushed exactly `delay` steps ago (zero-primed before).
        let expected = if i >= delay { input[i - delay] } else { 0.0 };
        assert_eq!(
            line.pop(),
            expected,
            "dynamic delay mismatch at i={i} (delay={delay})"
        );
    }

    // Retargets above capacity clamp without corrupting the ring.
    line.set_delay(10_000);
    assert_eq!(line.latency_samples(), 64);
    line.set_delay(7);
    assert_eq!(line.latency_samples(), 7);
    line.push(-2.5);
    assert_eq!(line.pop(), input[n - 7], "post-clamp shift mismatch");
}

#[test]
fn test_reset_clears_history_and_re_primes() {
    let mut line = DelayLine::<f32>::with_capacity(32, 8);
    for i in 0..64 {
        line.push(i as f32 * 0.5);
        let _ = line.pop();
    }

    line.reset();
    assert_eq!(line.capacity(), 32, "reset must preserve capacity");
    assert_eq!(line.latency_samples(), 8, "reset must preserve delay");

    // Full wipe: pops are zero-primed again, then shift exactly.
    for i in 0..24 {
        line.push(0.75);
        let expected = if i >= 8 { 0.75 } else { 0.0 };
        assert_eq!(line.pop(), expected, "post-reset priming at {i}");
    }

    // A second reset rewinds the head from an arbitrary wrap position:
    // the ring behaves exactly like a freshly constructed line.
    line.reset();
    let input: Vec<f32> = (0..96).map(|i| (i % 71) as f32 * 0.02).collect();
    for (i, &x) in input.iter().enumerate() {
        line.push(x);
        let expected = if i >= 8 { input[i - 8] } else { 0.0 };
        assert_eq!(line.pop(), expected, "post-reset shift mismatch at {i}");
    }
}

#[test]
fn test_set_delay_and_reset_are_zero_alloc() {
    let mut line = DelayLine::<f32>::with_capacity(3200, 16);
    // Warm once outside the guard.
    line.push(1.0);
    let _ = line.pop();
    let _guard = TrackingGuard::new();
    for i in 0..512 {
        line.set_delay(i % 3201);
        line.push(i as f32 * 0.001);
        let _ = line.pop();
        if i % 128 == 0 {
            line.reset();
        }
    }
    assert_eq!(
        line.capacity(),
        3200,
        "RT-path retargets must not reallocate"
    );
    assert_eq!(
        get_alloc_count(),
        0,
        "set_delay/reset/push/pop must not allocate on the RT path"
    );
}
