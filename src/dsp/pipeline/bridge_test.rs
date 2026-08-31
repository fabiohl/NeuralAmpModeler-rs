// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_bridge_reader_interleaved_writer_race() {
    let mut bridge = Box::new(DspBridge {
        buffers: [
            BridgeBuffer {
                buf_l: [0.0; MAX_BRIDGE_BUF],
                buf_r: [0.0; MAX_BRIDGE_BUF],
                n_samples: 0,
                generation: 0,
            },
            BridgeBuffer {
                buf_l: [0.0; MAX_BRIDGE_BUF],
                buf_r: [0.0; MAX_BRIDGE_BUF],
                n_samples: 0,
                generation: 0,
            },
        ],
        active_read_idx: AtomicUsize::new(0),
        generation: AtomicU64::new(0),
        consumed_gen: AtomicU64::new(0),
        dropped_frames: AtomicU32::new(0),
    });

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
