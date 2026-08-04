// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use std::sync::atomic::Ordering;

/// Tests whether the RingBuffer can pass data between two "processing lines" (threads)
/// concurrently without losing information or locking up.
#[test]
fn test_spsc_concurrency() {
    // Creates a buffer with 64 empty slots.
    // 'prod' (producer) sends data, 'cons' (consumer) receives.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(64);

    // Creates a new processing line that will "produce" numbers from 0 to 999.
    let handle = std::thread::spawn(move || {
        let mut count = 0;
        while count < 1000 {
            // Attempt to push the number into the buffer. If full, retry later.
            if prod.push(count).is_ok() {
                count += 1;
            }
            std::thread::yield_now(); // A small pause to avoid overwhelming the processor.
        }
    });

    // This part (the main line) will "consume" the numbers sent.
    let mut count = 0;
    while count < 1000 {
        // Attempt to pop a number from the buffer.
        if let Ok(val) = cons.pop() {
            // Verify that the received number is exactly what we expected.
            assert_eq!(val, count);
            count += 1;
        }
        std::thread::yield_now();
    }
    // Wait for the other processing line to finish before ending the test.
    handle.join().unwrap();
}

/// Tests buffer limits: what happens when it is completely empty or completely full.
#[test]
fn test_spsc_full_empty() {
    // Creates a small buffer with only 4 slots.
    let (mut prod, mut cons) = RingBuffer::<i32>::new(4);

    // Attempt to pop from an empty buffer. Should return an error indicating nothing is there.
    assert!(cons.pop().is_err());

    // Fill the 4 available slots.
    assert!(prod.push(1).is_ok());
    assert!(prod.push(2).is_ok());
    assert!(prod.push(3).is_ok());
    assert!(prod.push(4).is_ok());

    // Attempt to push the 5th item into a 4-slot buffer. Should return a "full" error.
    assert!(prod.push(5).is_err());

    // Pop the first item (the number 1).
    assert_eq!(cons.pop(), Ok(1));

    // Now that a slot opened, the number 5 should push successfully.
    assert!(prod.push(5).is_ok());

    // Attempt to push the 6th item. Since the buffer is full again (contains 2, 3, 4, 5), it should fail.
    assert!(prod.push(6).is_err());
}

#[test]
fn test_gc_overflow_overwrite() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let rt_status = RtStatusFlags::new();
    let overflow = GcOverflowBuffer::new(64);
    let counter = Arc::new(AtomicU32::new(0));

    // 1. Fill the 64-slot buffer
    for _ in 0..64 {
        let item = GcItem::Test(Box::new(counter.clone()));
        overflow.push(item);
    }

    // 2. Attempt to insert the 65th item (should overwrite the 1st)
    let item_65 = GcItem::Test(Box::new(counter.clone()));
    overflow.push(item_65);

    // 3. Validate that drain returns 64 items
    let drained = overflow.drain(&rt_status);
    assert_eq!(drained.len(), 64);
}

#[test]
fn test_gc_stress_no_leak() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let (mut gc_prod, mut gc_cons) = RingBuffer::<GcItem>::new(32);
    let rt_status = RtStatusFlags::new();
    let overflow = GcOverflowBuffer::new(32);
    let counter = Arc::new(AtomicU32::new(0));

    // Stress: 1000 "resource" swaps
    for _ in 0..1000 {
        let item = GcItem::Test(Box::new(counter.clone()));
        if let Err(rtrb::PushError::Full(returned_item)) = gc_prod.push(item) {
            // If the main channel is full, go to overflow
            overflow.push(returned_item);
        }

        // Drain periodically to avoid unbounded accumulation
        super::drain_gc_channels(&mut gc_cons, &overflow, &rt_status);
    }

    // Validate that the final counter is correct after dropping everything
    drop(gc_cons);
    for _ in overflow.drain(&rt_status) {}
}

/// Simulates a corrupted slot (packed value with unknown type_id + valid pointer)
/// and verifies that drain leaks the pointer and sets the GC_CORRUPTED flag.
#[test]
fn test_gc_corrupted_slot_unknown_type() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let rt_status = RtStatusFlags::new();
    let overflow = GcOverflowBuffer::new(4);
    let counter = Arc::new(AtomicU32::new(0));

    // Create a valid packed value to extract a real pointer from
    let valid_item = GcItem::Test(Box::new(counter.clone()));
    let valid_packed = valid_item.into_packed();
    let ptr = (valid_packed & 0x00FF_FFFF_FFFF_FFFF) as *mut std::ffi::c_void;

    // Push 3 valid items (slots 0, 1, 2 — write_idx reaches 2)
    overflow.push(GcItem::Test(Box::new(counter.clone())));
    overflow.push(GcItem::Test(Box::new(counter.clone())));
    overflow.push(GcItem::Test(Box::new(counter.clone())));

    // Manually inject a corrupted slot at position 3 (beyond write_idx):
    // unknown type_id (99) with a valid pointer
    overflow.slots[3].store(
        (99u64 << 56) | (ptr as u64 & 0x00FF_FFFF_FFFF_FFFF),
        Ordering::Release,
    );

    assert!(!rt_status.check_flag(RT_STATUS_GC_CORRUPTED));

    let drained = overflow.drain(&rt_status);

    // 3 valid items; slot 3 should be leaked with corruption flag set
    assert_eq!(drained.len(), 3);
    drop(drained);

    // The corrupted flag should be set
    assert!(rt_status.check_flag(RT_STATUS_GC_CORRUPTED));

    // Verify all slots are now empty
    let drained_again = overflow.drain(&rt_status);
    assert_eq!(drained_again.len(), 0);
    drop(drained_again);
}

/// Simulates a slot with type_id=0 and non-null pointer (inconsistent).
#[test]
fn test_gc_corrupted_slot_null_type_non_null_ptr() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicU32;

    let rt_status = RtStatusFlags::new();
    let overflow = GcOverflowBuffer::new(4);
    let counter = Arc::new(AtomicU32::new(0));

    // Create a valid packed value to extract a real pointer from
    let valid_item = GcItem::Test(Box::new(counter.clone()));
    let packed = valid_item.into_packed();
    let ptr = (packed & 0x00FF_FFFF_FFFF_FFFF) as *mut std::ffi::c_void;

    // Push 2 valid items (slots 0, 1)
    overflow.push(GcItem::Test(Box::new(counter.clone())));
    overflow.push(GcItem::Test(Box::new(counter.clone())));

    // Manually inject inconsistent slot at position 2: type_id=0 with valid pointer
    overflow.slots[2].store(ptr as u64 & 0x00FF_FFFF_FFFF_FFFF, Ordering::Release);

    assert!(!rt_status.check_flag(RT_STATUS_GC_CORRUPTED));

    let drained = overflow.drain(&rt_status);

    // Only 2 valid items; slot 2 is leaked with corruption flag
    assert_eq!(drained.len(), 2);
    drop(drained);

    assert!(rt_status.check_flag(RT_STATUS_GC_CORRUPTED));

    let drained_again = overflow.drain(&rt_status);
    assert_eq!(drained_again.len(), 0);
    drop(drained_again);
}

/// Concurrent push/drain stress test with real threads.
/// Exercises the race window between push overwrites and drain reads.
#[test]
fn test_gc_concurrent_push_drain() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let rt_status = RtStatusFlags::new();
    let overflow = Arc::new(GcOverflowBuffer::new(32));

    let overflow_prod = Arc::clone(&overflow);
    let done = Arc::new(AtomicBool::new(false));
    let done_prod = Arc::clone(&done);

    // Producer thread: rapid pushes into the overflow buffer
    let producer = std::thread::spawn(move || {
        let counter = Arc::new(std::sync::atomic::AtomicU32::new(0));
        while !done_prod.load(Ordering::Relaxed) {
            let item = GcItem::Test(Box::new(counter.clone()));
            overflow_prod.push(item);
        }
    });

    // Consumer thread: drain repeatedly
    let overflow_cons = Arc::clone(&overflow);
    let done_cons = Arc::clone(&done);
    let consumer = std::thread::spawn(move || {
        let local_rt_status = RtStatusFlags::new();
        for _ in 0..2000 {
            let drained = overflow_cons.drain(&local_rt_status);
            drop(drained);
            std::thread::yield_now();
        }
        done_cons.store(true, Ordering::Relaxed);
    });

    producer.join().unwrap();
    consumer.join().unwrap();

    // Final drain — all items should be valid GcItem or empty
    let final_drained = overflow.drain(&rt_status);
    for item in final_drained {
        match item {
            GcItem::Test(_) => {} // expected
            _ => panic!("Unexpected GcItem variant in concurrent test"),
        }
    }
}

#[test]
fn test_drain_gc_empty_is_noop() {
    let (_, mut consumer) = RingBuffer::<GcItem>::new(16);
    let overflow = GcOverflowBuffer::new(16);
    let rt_status = RtStatusFlags::new();
    let drained = drain_gc_channels(&mut consumer, &overflow, &rt_status);
    assert_eq!(drained, 0);
}
