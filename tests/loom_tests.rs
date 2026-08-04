// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loom concurrency test suite for `NeuralAmpModeler-rs`.
//!
//! Validates atomic memory ordering invariants, lock-free SPSC GC overflow buffer operations,
//! atomic handshake protocols, and double-buffering DSP bridge concurrency model using the
//! `loom` permutation engine.

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use loom::thread;

// =============================================================================
// Atomic Handshake Protocol Verification
// =============================================================================

#[test]
fn test_handshake_correct_passes() {
    loom::model(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let cell = Arc::new(UnsafeCell::new(0));

        let flag_clone = flag.clone();
        let cell_clone = cell.clone();

        let t1 = thread::spawn(move || {
            cell_clone.with_mut(|ptr| unsafe {
                *ptr = 42;
            });
            flag_clone.store(true, Ordering::Release);
        });

        let t2 = thread::spawn(move || {
            if flag.load(Ordering::Acquire) {
                let val = cell.with(|ptr| unsafe { *ptr });
                assert_eq!(val, 42);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

#[test]
fn test_handshake_relaxed_fails() {
    let result = std::panic::catch_unwind(|| {
        loom::model(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let cell = Arc::new(UnsafeCell::new(0));

            let flag_clone = flag.clone();
            let cell_clone = cell.clone();

            let t1 = thread::spawn(move || {
                cell_clone.with_mut(|ptr| unsafe {
                    *ptr = 42;
                });
                flag_clone.store(true, Ordering::Relaxed);
            });

            let t2 = thread::spawn(move || {
                if flag.load(Ordering::Relaxed) {
                    let val = cell.with(|ptr| unsafe { *ptr });
                    assert_eq!(val, 42);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    });
    assert!(
        result.is_err(),
        "Expected loom to catch a data race under Relaxed ordering"
    );
}

// =============================================================================
// Lock-free GC Overflow Buffer Concurrency Verification
// =============================================================================

/// Mock item wrapper containing an `UnsafeCell` for race detection.
struct MockItem {
    cell: UnsafeCell<u32>,
}

/// Loom test buffer simulating lock-free SPSC GC overflow queue slot management.
struct LoomGcOverflowBuffer {
    slots: Vec<AtomicUsize>,
    write_idx: AtomicUsize,
}

impl LoomGcOverflowBuffer {
    /// Creates a new overflow buffer with the specified capacity.
    fn new(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            slots.push(AtomicUsize::new(0));
        }
        Self {
            slots,
            write_idx: AtomicUsize::new(0),
        }
    }

    /// Pushes an item index into the ring slot using AcqRel swap.
    fn push(&self, item: usize) -> bool {
        let len = self.slots.len();
        let idx = self.write_idx.fetch_add(1, Ordering::Relaxed) % len;
        let old = self.slots[idx].swap(item, Ordering::AcqRel);
        old != 0
    }

    /// Drains all slots in the buffer using AcqRel atomic swaps.
    fn drain(&self) -> Vec<usize> {
        let mut items = Vec::new();
        for i in 0..self.slots.len() {
            let item = self.slots[i].swap(0, Ordering::AcqRel);
            if item != 0 {
                items.push(item);
            }
        }
        items
    }
}

#[test]
fn test_gc_overflow_concurrency() {
    loom::model(|| {
        let buffer = Arc::new(LoomGcOverflowBuffer::new(2));
        let pool = Arc::new(vec![
            MockItem {
                cell: UnsafeCell::new(0),
            },
            MockItem {
                cell: UnsafeCell::new(0),
            },
            MockItem {
                cell: UnsafeCell::new(0),
            },
            MockItem {
                cell: UnsafeCell::new(0),
            },
        ]);

        let buffer_clone = buffer.clone();
        let pool_clone = pool.clone();

        let t1 = thread::spawn(move || {
            for item_id in 1..=3 {
                let idx = item_id - 1;
                pool_clone[idx].cell.with_mut(|ptr| unsafe {
                    *ptr = item_id as u32 * 10;
                });
                buffer_clone.push(item_id);
            }
        });

        let t2 = thread::spawn(move || {
            let drained = buffer.drain();
            for item_id in drained {
                let idx = item_id - 1;
                let val = pool[idx].cell.with(|ptr| unsafe { *ptr });
                assert_eq!(val, item_id as u32 * 10);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

// =============================================================================
// Double-Buffering DspBridge Concurrency Verification
// =============================================================================

/// Single buffer entry holding a data cell.
struct LoomBridgeBuffer {
    data: UnsafeCell<u32>,
}

/// Double-buffered DSP bridge supporting lock-free concurrent producer/consumer access.
struct LoomDspBridge {
    buffers: [LoomBridgeBuffer; 2],
    active_read_idx: AtomicUsize,
    generation: AtomicU64,
    consumed_gen: AtomicU64,
}

impl LoomDspBridge {
    /// Creates a new double-buffered DSP bridge instance.
    fn new() -> Self {
        Self {
            buffers: [
                LoomBridgeBuffer {
                    data: UnsafeCell::new(0),
                },
                LoomBridgeBuffer {
                    data: UnsafeCell::new(0),
                },
            ],
            active_read_idx: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
            consumed_gen: AtomicU64::new(0),
        }
    }

    /// Writes a data block into the back buffer if the consumer has caught up.
    fn write_block(&self, val: u32) -> bool {
        let current_gen = self.generation.load(Ordering::Relaxed);
        let consumed_gen = self.consumed_gen.load(Ordering::Acquire);

        if current_gen > consumed_gen {
            return false; // Skip write (overflow/dropped frame)
        }

        let back_idx = 1 - self.active_read_idx.load(Ordering::Relaxed);

        self.buffers[back_idx].data.with_mut(|ptr| unsafe {
            *ptr = val;
        });

        self.active_read_idx.store(back_idx, Ordering::Release);
        self.generation.store(current_gen + 1, Ordering::Release);
        true
    }

    /// Reads a data block from the active read buffer if a new generation is available.
    fn read_block(&self, last_bridge_gen: &mut u64) -> Option<u32> {
        let current_gen = self.generation.load(Ordering::Acquire);
        if current_gen == *last_bridge_gen {
            return None;
        }
        *last_bridge_gen = current_gen;
        self.consumed_gen.store(current_gen, Ordering::Release);

        let read_idx = self.active_read_idx.load(Ordering::Acquire);
        let val = self.buffers[read_idx].data.with(|ptr| unsafe { *ptr });
        Some(val)
    }
}

#[test]
fn test_dsp_bridge_concurrency() {
    loom::model(|| {
        let bridge = Arc::new(LoomDspBridge::new());
        let bridge_clone = bridge.clone();

        let t1 = thread::spawn(move || {
            let mut val = 1;
            for _ in 0..3 {
                if bridge_clone.write_block(val) {
                    val += 1;
                }
            }
        });

        let t2 = thread::spawn(move || {
            let mut last_gen = 0;
            let mut read_values = Vec::new();
            for _ in 0..3 {
                if let Some(val) = bridge.read_block(&mut last_gen) {
                    read_values.push(val);
                }
            }
            for &val in &read_values {
                assert!(val > 0);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}
