// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn new_bridge() -> Box<DspBridge> {
    DspBridge::new_boxed()
}

#[test]
fn test_bridge_reader_interleaved_writer_race() {
    let mut bridge = new_bridge();

    let bridge_ptr = &raw mut *bridge;
    // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
    // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };

    // Publish block 1 (generation 1)
    writer.write_block(&[1.0, 1.0], &[1.0, 1.0], 2, false);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 1);

    let mut last_gen = 0u64;

    // Simulate reader loading generation=1
    let current_gen = bridge.generation.load(Ordering::Acquire);
    assert_eq!(current_gen, 1);

    // Before reader loads active_read_idx, consumer/writer publishes block 2 (generation 2).
    // Since reader has not updated consumed_gen (still 0), writer skips block 2 if consumed_gen < current_gen.
    // To test writer publication race, update consumed_gen to 1 so writer can publish block 2:
    bridge.consumed_gen.store(1, Ordering::Release);
    writer.write_block(&[2.0, 2.0], &[2.0, 2.0], 2, false);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 2);

    // Now reader reads active_read_idx and post_gen
    let _read_idx = bridge.active_read_idx.load(Ordering::Acquire);
    let post_gen = bridge.generation.load(Ordering::Acquire);
    assert_ne!(current_gen, post_gen);

    // Reader's post_gen check fails because post_gen (2) != current_gen (1).
    // Calling read_block from scratch now:
    let res = reader.read_block(&mut last_gen, |l, _| l[0]);
    assert_eq!(res, Some(2.0));
    assert_eq!(last_gen, 2);
}

/// A8 / R-6: `reset_to_silence` is gated behind `&mut` (teardown-only). After a
/// block is published and consumed, an exclusive reset must leave the bridge in
/// a silent state: buffer generations are zeroed and a fresh reader (whose
/// `last_bridge_gen` starts at 0) observes no block — never stale audio.
#[test]
fn test_bridge_reset_to_silence_teardown_only() {
    let mut bridge = new_bridge();

    let bridge_ptr = &raw mut *bridge;
    // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };

    // Publish and consume block 1 so `generation`/`consumed_gen` are both 1.
    writer.write_block(&[1.0, 1.0], &[1.0, 1.0], 2, false);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 1);
    bridge.consumed_gen.store(1, Ordering::Release);

    // Exclusive (teardown) reset — the gated `&mut self` API.
    bridge.reset_to_silence();

    // Buffers are silent and generations re-synchronized for the next session.
    // SAFETY: single-threaded teardown access — no producer/consumer is live, so
    // the buffers are read race-free through `UnsafeCell::get`.
    let (b0, b1) = unsafe { (&*bridge.buffers[0].get(), &*bridge.buffers[1].get()) };
    assert_eq!(b0.n_samples, 0);
    assert_eq!(b1.n_samples, 0);
    assert_eq!(b0.generation, 0);
    assert_eq!(b1.generation, 0);
    let generation = bridge.generation.load(Ordering::Relaxed);
    let consumed = bridge.consumed_gen.load(Ordering::Relaxed);
    assert_eq!(generation, 2);
    assert_eq!(
        consumed, generation,
        "consumed_gen must track generation after reset"
    );

    // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };
    let mut last_bridge_gen = 0u64;
    let res = reader.read_block(&mut last_bridge_gen, |l, _| l[0]);
    // A fresh reader sees silence (no stale block 1): the front buffer's
    // generation (0) no longer matches the bridge generation after reset.
    assert_eq!(
        res, None,
        "reset must leave the bridge silent, not replaying stale audio"
    );
    assert_eq!(
        last_bridge_gen, 0,
        "reader must not advance its generation on silence"
    );
}

/// F-PR-11 / S6-T4: off-RT `prime_silence` pre-rolls one real silence block so
/// a fresh reader's first `read_block` returns through the normal wet path
/// (never the starvation path) and consumes the primed generation. The write is
/// skipped (no drop) when a previous block is still unconsumed.
#[test]
fn test_bridge_prime_silence_pre_rolls_first_block() {
    let mut bridge = new_bridge();

    let bridge_ptr = &raw mut *bridge;
    // SAFETY: Pointer is valid and points to an allocated DspBridge instance.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };

    // Off-RT pre-roll (exclusive `&mut`).
    bridge.prime_silence(64);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 1);

    let mut last_bridge_gen = 0u64;
    let seen = reader.read_block(&mut last_bridge_gen, |l, r| (l.len(), l[0], r[0]));
    let (len, l0, r0) = seen.expect("pre-rolled silence must surface as a block, not starvation");
    assert_eq!(
        len, 64,
        "pre-roll must publish the requested quantum length"
    );
    assert_eq!(
        (l0, r0),
        (0.0, 0.0),
        "pre-roll block must be analytical silence"
    );
    assert_eq!(
        last_bridge_gen, 1,
        "reader must consume the primed generation"
    );
    assert_eq!(
        bridge.consumed_gen.load(Ordering::Relaxed),
        1,
        "primed block must be marked consumed"
    );
    assert_eq!(
        bridge.dropped_frames.load(Ordering::Relaxed),
        0,
        "pre-roll must never account a drop"
    );

    // A second pre-roll with the primed block still... now consumed: it advances.
    bridge.prime_silence(64);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 2);

    // Unconsumed primed block: a further pre-roll must be skipped (no drop, no
    // generation advance) to preserve the skip-on-overflow invariant.
    bridge.prime_silence(64);
    assert_eq!(
        bridge.generation.load(Ordering::Relaxed),
        2,
        "pre-roll over an unconsumed block must be skipped"
    );
    assert_eq!(bridge.dropped_frames.load(Ordering::Relaxed), 0);
}

#[test]
fn test_dsp_bridge_alignment_and_cache_isolation() {
    use core::mem::{align_of, offset_of, size_of};
    // Overall struct alignment and sizing
    assert_eq!(
        align_of::<DspBridge>(),
        128,
        "DspBridge must be 128-byte aligned"
    );
    assert_eq!(
        size_of::<DspBridge>() % 128,
        0,
        "DspBridge size must be a multiple of 128 bytes"
    );

    // Buffers offset
    let off_buf = offset_of!(DspBridge, buffers);
    assert_eq!(off_buf, 0, "buffers must start at offset 0");

    // Writer cache line offsets. 128-byte granularity: Apple P-cores fetch
    // whole 128-byte lines, so a 64-byte split between writer and reader still
    // produces false sharing on macOS/aarch64.
    let off_active_read = offset_of!(DspBridge, active_read_idx);
    let off_gen = offset_of!(DspBridge, generation);
    let off_dropped = offset_of!(DspBridge, dropped_frames);

    assert_eq!(
        off_active_read % 128,
        0,
        "active_read_idx must be on a 128-byte cache line boundary"
    );
    assert_eq!(
        off_active_read / 128,
        off_gen / 128,
        "generation must share the writer cache line with active_read_idx"
    );
    assert_eq!(
        off_active_read / 128,
        off_dropped / 128,
        "dropped_frames must share the writer cache line with active_read_idx"
    );

    // Reader cache line offset
    let off_consumed = offset_of!(DspBridge, consumed_gen);
    assert_eq!(
        off_consumed % 128,
        0,
        "consumed_gen must be on a distinct 128-byte cache line boundary"
    );
    assert_ne!(
        off_active_read / 128,
        off_consumed / 128,
        "writer atomics and reader atomics must reside on distinct 128-byte lines (false sharing prevention)"
    );

    // Spacing between writer and reader cache lines must be at least 128 bytes
    assert!(
        off_consumed >= off_active_read + 128,
        "reader cache line must follow writer cache line by at least 128 bytes"
    );
}

/// S1-T01 / R-15 regression suite: verifies that `DspBridgeWriter::write_block` validates source slice
/// lengths defensively across all builds (debug and release), preventing out-of-bounds reads or panics
/// on 0, 1, n-1, n, short-origin and max-capacity buffer lengths in both mono and stereo.
#[test]
fn test_bridge_write_block_source_length_bounds_safeguard() {
    let mut bridge = new_bridge();
    let bridge_ptr = &raw mut *bridge;
    // SAFETY: `bridge_ptr` is derived from a live `Box<DspBridge>` allocated by `new_bridge()`
    // which outlives both writer and reader in this test.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
    // SAFETY: same valid pointer into live heap allocation.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };
    let mut last_gen = 0u64;

    // 1. Length 0: n_pw = 0
    writer.write_block(&[], &[], 0, false);
    let seen = reader.read_block(&mut last_gen, |l, _| l.len());
    assert_eq!(seen, None, "Length 0 write publishes silence (None)");

    // 2. Length 1: n_pw = 1, valid buffer
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&[42.0], &[84.0], 1, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(seen, Some((1, 42.0, 84.0)));

    // 3. Length n-1: n_pw = 64, but slice has only 63 elements (short origin)
    let src63_l = vec![1.5f32; 63];
    let src63_r = vec![2.5f32; 63];
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&src63_l, &src63_r, 64, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(
        seen,
        Some((63, 1.5, 2.5)),
        "Short slice n-1 clamped safely without panic or OOB"
    );

    // 4. Length n: n_pw = 64, slice has exactly 64 elements
    let src64_l = vec![3.0f32; 64];
    let src64_r = vec![4.0f32; 64];
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&src64_l, &src64_r, 64, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(seen, Some((64, 3.0, 4.0)), "Exact length n preserved");

    // 5. Short origin extreme (reproducing R-15 without crash): n_pw = 64, slice has 1 element
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&[9.9], &[8.8], 64, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(
        seen,
        Some((1, 9.9, 8.8)),
        "Extreme 1-element slice clamped safely in debug and release"
    );

    // 6. Mono mode with empty right slice: mono only copies left channel to both L and R
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&[7.7; 32], &[], 32, true);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(
        seen,
        Some((32, 7.7, 7.7)),
        "Mono mode uses only L and copies to both channels"
    );

    // 7. Stereo mode with short right slice: clamps to available min(L, R)
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&[1.0; 64], &[2.0; 10], 64, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(seen, Some((10, 1.0, 2.0)), "Stereo clamps to min(L, R)");

    // 8. Max capacity ceiling: n_pw > MAX_BRIDGE_BUF
    let huge_l = vec![5.0f32; MAX_BRIDGE_BUF + 100];
    let huge_r = vec![6.0f32; MAX_BRIDGE_BUF + 100];
    bridge
        .consumed_gen
        .store(bridge.generation.load(Ordering::Relaxed), Ordering::Release);
    writer.write_block(&huge_l, &huge_r, MAX_BRIDGE_BUF + 100, false);
    let seen = reader.read_block(&mut last_gen, |l, r| (l.len(), l[0], r[0]));
    assert_eq!(
        seen,
        Some((MAX_BRIDGE_BUF, 5.0, 6.0)),
        "Clamped to MAX_BRIDGE_BUF"
    );
}

fn run_spsc_interleaved_tear_free(rounds: usize) -> usize {
    use core::sync::atomic::AtomicU32;
    use std::sync::Barrier;

    const BLOCK_LEN: usize = 64;
    assert!(
        rounds > 0 && (rounds as u64) * 2 + 2 <= (1u64 << 24),
        "generations must remain exact in IEEE-754 f32 (<= 2^24)"
    );

    let mut bridge = new_bridge();
    let bridge_ptr = &raw mut *bridge;
    // SAFETY: `bridge_ptr` points to the live `Box<DspBridge>` above, which outlives
    // the scoped producer and reader threads. Narrowing to `dropped_frames` avoids
    // creating a whole-struct `&DspBridge` concurrently with buffer writes.
    let dropped_frames: &AtomicU32 = unsafe { &(*bridge_ptr).dropped_frames };
    // SAFETY: `bridge_ptr` is valid, non-null, and outlives the scoped threads.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
    // SAFETY: `bridge_ptr` is valid, non-null, and outlives the scoped threads.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };

    // Four barriers enforce the deterministic interleaving per round:
    // (a) `b_pub_g`: producer published generation `g`
    // (b) `b_ack_g`: reader acknowledged `g` (advancing `consumed_gen = g`) while deferring copy
    // (c) `b_attempt_g2`: producer published `g + 1` in the complementary slot and attempted `g + 2` (dropped)
    // (d) `b_round_done`: reader copied `g` intact and consumed `g + 1` before the next round begins
    let b_pub_g = Barrier::new(2);
    let b_ack_g = Barrier::new(2);
    let b_attempt_g2 = Barrier::new(2);
    let b_round_done = Barrier::new(2);

    let torn_reads = std::thread::scope(|s| {
        {
            let b_pub_g = &b_pub_g;
            let b_ack_g = &b_ack_g;
            let b_attempt_g2 = &b_attempt_g2;
            let b_round_done = &b_round_done;
            s.spawn(move || {
                let mut src_l = [0.0f32; BLOCK_LEN];
                let mut src_r = [0.0f32; BLOCK_LEN];

                for round in 0..rounds {
                    let g = (round as u64) * 2 + 1;

                    // (a) Publish generation `g`.
                    let g_val = g as f32;
                    src_l.fill(g_val);
                    src_r.fill(g_val);
                    writer.write_block(&src_l, &src_r, BLOCK_LEN, false);
                    b_pub_g.wait();

                    // Wait for (b) reader to acknowledge `g`.
                    b_ack_g.wait();

                    // Publish `g + 1` into the complementary back-buffer.
                    let g1_val = (g + 1) as f32;
                    src_l.fill(g1_val);
                    src_r.fill(g1_val);
                    writer.write_block(&src_l, &src_r, BLOCK_LEN, false);

                    // (c) Attempt to publish `g + 2` before `g + 1` is consumed: must drop and increment `dropped_frames`.
                    let drops_before = dropped_frames.load(Ordering::Relaxed);
                    let g2_val = (g + 2) as f32;
                    src_l.fill(g2_val);
                    src_r.fill(g2_val);
                    writer.write_block(&src_l, &src_r, BLOCK_LEN, false);
                    let drops_after = dropped_frames.load(Ordering::Relaxed);
                    assert_eq!(
                        drops_after,
                        drops_before + 1,
                        "blocked write for g+2 must increment dropped_frames at round {round}"
                    );
                    b_attempt_g2.wait();

                    b_round_done.wait();
                }
            });
        }

        let reader_handle = {
            let b_pub_g = &b_pub_g;
            let b_ack_g = &b_ack_g;
            let b_attempt_g2 = &b_attempt_g2;
            let b_round_done = &b_round_done;
            s.spawn(move || {
                let mut last_gen = 0u64;
                let mut tears = 0usize;
                let mut copy_l = [0.0f32; BLOCK_LEN];
                let mut copy_r = [0.0f32; BLOCK_LEN];

                for round in 0..rounds {
                    let g = (round as u64) * 2 + 1;

                    // Wait for (a) producer to publish `g`.
                    b_pub_g.wait();

                    // (b) Acknowledge `g` and capture raw front-buffer addresses, mirroring
                    // the host playback path where the payload copy happens after `read_block` returns.
                    let captured = reader.read_block(&mut last_gen, |l, r| {
                        assert_eq!(l.len(), BLOCK_LEN, "unexpected L length for gen {g}");
                        assert_eq!(r.len(), BLOCK_LEN, "unexpected R length for gen {g}");
                        let expected = g as f32;
                        for (&sl, &sr) in l.iter().zip(r.iter()) {
                            if sl != expected || sr != expected {
                                tears += 1;
                            }
                        }
                        (l.as_ptr() as usize, l.len(), r.as_ptr() as usize, r.len())
                    });
                    let (src_l_addr, n_l, src_r_addr, n_r) = captured
                        .unwrap_or_else(|| panic!("expected generation {g} to be readable"));
                    assert_eq!(last_gen, g, "reader generation mismatch at round {round}");
                    assert_eq!(n_l, BLOCK_LEN);
                    assert_eq!(n_r, BLOCK_LEN);
                    b_ack_g.wait();

                    // Wait for (c) producer to publish `g + 1` and attempt `g + 2` (blocked).
                    b_attempt_g2.wait();

                    // Copy `g` after `g + 1` publication and blocked `g + 2` attempt: `g`'s slot
                    // must remain intact and homogeneous while `consumed_gen == g`.
                    // SAFETY: `src_l_addr`/`src_r_addr` point into the front-buffer selected for `g`.
                    // Because `consumed_gen == g` and `generation == g + 1`, the writer's skip-on-overflow
                    // guard prevented any overwrite of `g`'s buffer until `g + 1` is consumed below.
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            src_l_addr as *const f32,
                            copy_l.as_mut_ptr(),
                            BLOCK_LEN,
                        );
                        core::ptr::copy_nonoverlapping(
                            src_r_addr as *const f32,
                            copy_r.as_mut_ptr(),
                            BLOCK_LEN,
                        );
                    }
                    let expected_g = g as f32;
                    for i in 0..BLOCK_LEN {
                        if copy_l[i] != expected_g || copy_r[i] != expected_g {
                            tears += 1;
                        }
                        assert_eq!(
                            copy_l[i], expected_g,
                            "torn/overwritten L sample for gen {g} at index {i}"
                        );
                        assert_eq!(
                            copy_r[i], expected_g,
                            "torn/overwritten R sample for gen {g} at index {i}"
                        );
                    }

                    // Consume `g + 1` and verify uniform payload and coherent generation.
                    let next_ok = reader.read_block(&mut last_gen, |l, r| {
                        assert_eq!(l.len(), BLOCK_LEN, "unexpected L length for gen {}", g + 1);
                        assert_eq!(r.len(), BLOCK_LEN, "unexpected R length for gen {}", g + 1);
                        let expected_g1 = (g + 1) as f32;
                        for (i, (&sl, &sr)) in l.iter().zip(r.iter()).enumerate() {
                            if sl != expected_g1 || sr != expected_g1 {
                                tears += 1;
                            }
                            assert_eq!(
                                sl,
                                expected_g1,
                                "torn L sample for gen {} at index {i}",
                                g + 1
                            );
                            assert_eq!(
                                sr,
                                expected_g1,
                                "torn R sample for gen {} at index {i}",
                                g + 1
                            );
                        }
                    });
                    assert!(
                        next_ok.is_some(),
                        "expected generation {} to be readable",
                        g + 1
                    );
                    assert_eq!(
                        last_gen,
                        g + 1,
                        "reader generation mismatch for g+1 at round {round}"
                    );

                    b_round_done.wait();
                }

                tears
            })
        };

        reader_handle.join().expect("reader thread panicked")
    });

    assert_eq!(
        bridge.dropped_frames.load(Ordering::Relaxed),
        rounds as u32,
        "exactly one blocked write per round must be recorded in dropped_frames"
    );
    assert_eq!(torn_reads, 0, "zero torn reads allowed");
    torn_reads
}

#[test]
fn test_bridge_spsc_interleaved_tear_free_quick() {
    // Measured: N=256 rounds (512 consumed blocks, 256 dropped), torn_reads=0, elapsed=<0.01s
    let torn_reads = run_spsc_interleaved_tear_free(256);
    assert_eq!(torn_reads, 0);
}

#[test]
#[ignore = "long: deterministic SPSC interleaving soak across many buffer flips"]
fn test_bridge_spsc_interleaved_tear_free_soak() {
    // Measured: N=50_000 rounds (100_000 consumed blocks, 50_000 dropped), torn_reads=0, elapsed=1.36s
    let torn_reads = run_spsc_interleaved_tear_free(50_000);
    assert_eq!(torn_reads, 0);
}

/// T5.2 / F8: the capture-side timestamp travels with the published
/// generation, so the playback thread can compute capture→playback latency for
/// the *delivered* generation (never the last global capture timestamp).
#[test]
fn test_bridge_stamped_publication_preserves_generation_timestamp() {
    let mut bridge = new_bridge();
    let bridge_ptr = &raw mut *bridge;
    // SAFETY: Pointer is valid and points to an allocated DspBridge instance.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };
    // SAFETY: Pointer is valid and points to an allocated DspBridge instance.
    let reader = unsafe { DspBridgeReader::new(bridge_ptr) };

    writer.write_block_stamped(&[1.0, 2.0], &[3.0, 4.0], 2, false, 1_234_567_890);
    let mut last_gen = 0u64;
    let read = reader.read_block_stamped(&mut last_gen, |l, r| (l[0], r[0]));
    assert_eq!(
        read,
        Some((1, 1_234_567_890, (1.0f32, 3.0f32))),
        "generation and capture timestamp must be delivered together"
    );

    // The plain API stays source-compatible and still sees the block.
    writer.write_block(&[5.0, 6.0], &[7.0, 8.0], 2, false);
    let plain = reader.read_block(&mut last_gen, |l, _| l[0]);
    assert_eq!(plain, Some(5.0));

    // A silence publication never carries a stale capture timestamp: the
    // zero-length block is consumed (generation advances) but not surfaced.
    let before_silence_gen = last_gen;
    writer.write_silence();
    let read = reader.read_block_stamped(&mut last_gen, |l, _| l.len());
    assert_eq!(
        read, None,
        "zero-length silence blocks are not surfaced by read_block"
    );
    assert_eq!(
        last_gen,
        before_silence_gen + 1,
        "the silence generation is consumed even though no payload is surfaced"
    );
}

#[test]
fn test_split_spsc_and_from_ref() {
    // Null BridgeRef returns None for split_spsc and both from_ref constructors.
    let null_ref = BridgeRef::null();
    // SAFETY: `null_ref` is null and produces `None` without dereference.
    assert!(unsafe { DspBridgeWriter::split_spsc(null_ref) }.is_none());
    // SAFETY: `null_ref` is null and produces `None` without dereference.
    assert!(unsafe { DspBridgeWriter::from_ref(null_ref) }.is_none());
    // SAFETY: `null_ref` is null and produces `None` without dereference.
    assert!(unsafe { DspBridgeReader::from_ref(null_ref) }.is_none());

    let mut bridge = new_bridge();
    // SAFETY: `bridge` outlives both endpoints and this test is the single
    // SPSC producer and consumer.
    let bridge_ref = unsafe { BridgeRef::new(&raw mut *bridge) };
    // SAFETY: single SPSC pair derived from `bridge_ref`.
    let (writer, reader) = unsafe { DspBridgeWriter::split_spsc(bridge_ref) }.expect("non-null");
    writer.write_block(&[0.25, -0.5], &[0.5, -0.25], 2, false);
    let mut last_gen = 0u64;
    let read = reader.read_block(&mut last_gen, |l, r| (l.to_vec(), r.to_vec()));
    assert_eq!(read, Some((vec![0.25, -0.5], vec![0.5, -0.25])));
}
