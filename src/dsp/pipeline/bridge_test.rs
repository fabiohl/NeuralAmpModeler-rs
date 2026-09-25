// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn new_bridge() -> Box<DspBridge> {
    DspBridge::new_boxed()
}

#[test]
fn test_bridge_reader_interleaved_writer_race() {
    let mut bridge = new_bridge();

    let bridge_ptr = &mut *bridge as *mut DspBridge;
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

    let bridge_ptr = &mut *bridge as *mut DspBridge;
    // SAFETY: Pointer is valid and points to a allocated DspBridge instance.
    let writer = unsafe { DspBridgeWriter::new(bridge_ptr) };

    // Publish and consume block 1 so `generation`/`consumed_gen` are both 1.
    writer.write_block(&[1.0, 1.0], &[1.0, 1.0], 2, false);
    assert_eq!(bridge.generation.load(Ordering::Relaxed), 1);
    bridge.consumed_gen.store(1, Ordering::Release);

    // Exclusive (teardown) reset — the gated `&mut self` API.
    bridge.reset_to_silence();

    // Buffers are silent and generations re-synchronized for the next session.
    assert_eq!(bridge.buffers[0].n_samples, 0);
    assert_eq!(bridge.buffers[1].n_samples, 0);
    assert_eq!(bridge.buffers[0].generation, 0);
    assert_eq!(bridge.buffers[1].generation, 0);
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

    // Writer cache line offsets
    let off_active_read = offset_of!(DspBridge, active_read_idx);
    let off_gen = offset_of!(DspBridge, generation);
    let off_dropped = offset_of!(DspBridge, dropped_frames);

    assert_eq!(
        off_active_read % 64,
        0,
        "active_read_idx must be on a 64-byte cache line boundary"
    );
    assert_eq!(
        off_active_read / 64,
        off_gen / 64,
        "generation must share the writer cache line with active_read_idx"
    );
    assert_eq!(
        off_active_read / 64,
        off_dropped / 64,
        "dropped_frames must share the writer cache line with active_read_idx"
    );

    // Reader cache line offset
    let off_consumed = offset_of!(DspBridge, consumed_gen);
    assert_eq!(
        off_consumed % 64,
        0,
        "consumed_gen must be on a distinct 64-byte cache line boundary"
    );
    assert_ne!(
        off_active_read / 64,
        off_consumed / 64,
        "writer atomics and reader atomics must reside on distinct cache lines (false sharing prevention)"
    );

    // Spacing between writer and reader cache lines must be at least 64 bytes
    assert!(
        off_consumed >= off_active_read + 64,
        "reader cache line must follow writer cache line by at least 64 bytes"
    );
}
