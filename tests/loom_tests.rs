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

// =============================================================================
// Composite Engine Swap via SPSC → GC Cascade (T6.4)
// =============================================================================
//
// Models the composite hot-swap protocol used for `CabSimAdapter` and
// `OversampleEngine` (cabsim/os SPSC channels + `gc_cascade`):
//
//   1. Main thread builds a new engine payload and publishes its id through
//      the SPSC channel with Release ordering.
//   2. RT thread acquires the id (Acquire), verifies the payload, swaps out
//      the active engine, and routes the obsolete one through the GC cascade
//      (ring → parking lot → overflow buffer).
//   3. Main thread drains the cascade; every delivered engine must appear
//      exactly once (no leak, no double-drop) with intact payload data.

/// An engine payload cell — loom instruments access to detect data races.
struct LoomEngineSlot {
    data: UnsafeCell<u64>,
}

/// Tiered GC cascade and SPSC publish channel, modeled with loom atomics.
struct LoomSwapMesh {
    /// Main → RT SPSC channel slots (sweep-published / sweep-drained).
    spsc: Vec<AtomicUsize>,
    /// GC Tier 1: SPSC drop-delegation slot (RT-only producer).
    gc_ring: Vec<AtomicUsize>,
    /// GC Tier 2: parking lot slot (RT-only producer).
    parking: Vec<AtomicUsize>,
    /// GC Tier 3: overflow slot (RT-only producer, overwrite would lose items).
    overflow: Vec<AtomicUsize>,
}

impl LoomSwapMesh {
    fn new() -> Self {
        Self {
            spsc: (0..2).map(|_| AtomicUsize::new(0)).collect(),
            gc_ring: (0..1).map(|_| AtomicUsize::new(0)).collect(),
            parking: (0..1).map(|_| AtomicUsize::new(0)).collect(),
            overflow: (0..1).map(|_| AtomicUsize::new(0)).collect(),
        }
    }

    /// Publishes an engine id into a free SPSC slot (Release on success).
    /// Returns `true` if the item was enqueued (channel full → `false`).
    fn publish(&self, id: usize) -> bool {
        for slot in &self.spsc {
            if slot.load(Ordering::Relaxed) == 0 && slot.swap(id, Ordering::AcqRel) == 0 {
                return true;
            }
        }
        false
    }

    /// RT-side poll: sweeps the SPSC channel and returns one pending id.
    fn poll_spsc(&self) -> Option<usize> {
        for slot in &self.spsc {
            let id = slot.swap(0, Ordering::AcqRel);
            if id != 0 {
                return Some(id);
            }
        }
        None
    }

    /// RT-side GC cascade push (Tier 1 → Tier 2 → Tier 3), mirroring
    /// `gc_cascade` in `src/common/spsc/gc.rs`. Each tier is a single slot;
    /// the slot sweep by the control thread is the drain counterpart.
    fn cascade_push(&self, id: usize) {
        if self.gc_ring[0].load(Ordering::Relaxed) == 0
            && self.gc_ring[0].swap(id, Ordering::AcqRel) == 0
        {
            return;
        }
        if self.parking[0].load(Ordering::Relaxed) == 0
            && self.parking[0].swap(id, Ordering::AcqRel) == 0
        {
            return;
        }
        self.overflow[0].swap(id, Ordering::AcqRel);
    }

    /// Drains all GC tiers. Returns the drained engine ids.
    fn drain_gc(&self) -> Vec<usize> {
        let mut items = Vec::new();
        for slot in self
            .gc_ring
            .iter()
            .chain(self.parking.iter())
            .chain(self.overflow.iter())
        {
            let id = slot.swap(0, Ordering::AcqRel);
            if id != 0 {
                items.push(id);
            }
        }
        items
    }

    /// Drains any ids still queued in the SPSC channel (never delivered).
    fn drain_spsc(&self) -> Vec<usize> {
        let mut items = Vec::new();
        for slot in &self.spsc {
            let id = slot.swap(0, Ordering::AcqRel);
            if id != 0 {
                items.push(id);
            }
        }
        items
    }
}

#[test]
fn test_composite_engine_swap_spsc_gc_cascade() {
    // Bounded preemption exploration: the composite protocol has a large
    // atomic-op count; a 3-preemption bound keeps the state space tractable
    // while still exercising the publish/acquire and cascade handoffs under
    // every bounded scheduling.
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        let mesh = Arc::new(LoomSwapMesh::new());
        let pool: Arc<Vec<LoomEngineSlot>> = Arc::new(
            (0..4)
                .map(|_| LoomEngineSlot {
                    data: UnsafeCell::new(0),
                })
                .collect(),
        );

        let mesh_main = mesh.clone();
        let pool_main = pool.clone();
        let t_main = thread::spawn(move || {
            let mut published_ids = Vec::new();
            let mut drained = Vec::new();
            for id in 1..=3usize {
                pool_main[id].data.with_mut(|ptr| unsafe {
                    *ptr = id as u64 * 100;
                });
                if mesh_main.publish(id) {
                    published_ids.push(id);
                }
                // Control-plane housekeeping: drain GC concurrently with
                // RT swaps (mirrors `drain_gc_channels`).
                drained.extend(mesh_main.drain_gc());
            }
            (published_ids, drained)
        });

        let mesh_rt = mesh.clone();
        let pool_rt = pool.clone();
        let t_rt = thread::spawn(move || {
            let mut active = 0usize;
            for _ in 0..3 {
                if let Some(id) = mesh_rt.poll_spsc() {
                    let val = pool_rt[id].data.with(|ptr| unsafe { *ptr });
                    assert_eq!(
                        val,
                        id as u64 * 100,
                        "payload integrity violated on RT swap"
                    );
                    if active != 0 {
                        mesh_rt.cascade_push(active);
                    }
                    active = id;
                }
            }
            if active != 0 {
                mesh_rt.cascade_push(active);
            }
        });

        let (published, mut drained) = t_main.join().unwrap();
        t_rt.join().unwrap();

        // Final drain after both threads stopped.
        drained.extend(mesh.drain_gc());
        let leftover = mesh.drain_spsc();

        // Every engine published is either still queued (never delivered) or
        // cascaded and drained exactly once.
        let mut delivered: Vec<usize> = published
            .iter()
            .copied()
            .filter(|id| !leftover.contains(id))
            .collect();
        let mut drained_sorted = drained.clone();
        drained_sorted.sort_unstable();
        delivered.sort_unstable();
        assert_eq!(
            drained_sorted, delivered,
            "every delivered engine must be cascaded and drained exactly once \
             (no leak, no double-drop) — leftover: {leftover:?}, drained: {drained:?}"
        );

        for id in &drained {
            let val = pool[*id].data.with(|ptr| unsafe { *ptr });
            assert_eq!(val, (*id as u64) * 100, "drained payload corrupted");
        }
    });
}

// =============================================================================
// Multi-Field RtStatusFlags Synchronization (T6.4)
// =============================================================================
//
// Models the RT→Main handshake used by `RtStatusFlags`: the RT thread writes
// a data field (`requested_host_rate`) and then publishes the associated
// status flag with Release; the control thread reads the flag with Acquire
// and must observe the matching data value (happens-before edge).

const RT_FLAG_NEEDS_RESAMPLER_REBUILD: u64 = 1 << 0;

/// Simplified multi-field status block: one data cell + one flag word.
struct LoomRtStatus {
    requested_host_rate: UnsafeCell<u32>,
    status_bits: AtomicU64,
}

impl LoomRtStatus {
    fn new() -> Self {
        Self {
            requested_host_rate: UnsafeCell::new(0),
            status_bits: AtomicU64::new(0),
        }
    }

    /// RT-side: publishes data + flag with Release ordering.
    fn publish_request(&self, rate: u32) {
        self.requested_host_rate.with_mut(|ptr| unsafe {
            *ptr = rate;
        });
        self.status_bits
            .fetch_or(RT_FLAG_NEEDS_RESAMPLER_REBUILD, Ordering::Release);
    }

    /// Main-side: acquires the flag and reads the gated data.
    fn poll_request(&self) -> Option<u32> {
        if (self.status_bits.load(Ordering::Acquire) & RT_FLAG_NEEDS_RESAMPLER_REBUILD) != 0 {
            let rate = self.requested_host_rate.with(|ptr| unsafe { *ptr });
            self.status_bits
                .fetch_and(!RT_FLAG_NEEDS_RESAMPLER_REBUILD, Ordering::Relaxed);
            Some(rate)
        } else {
            None
        }
    }
}

#[test]
fn test_rt_status_multifield_handshake() {
    loom::model(|| {
        let status = Arc::new(LoomRtStatus::new());
        let status_rt = status.clone();
        let status_main = status.clone();

        let t_rt = thread::spawn(move || {
            status_rt.publish_request(48_000);
        });

        let t_main = thread::spawn(move || {
            let mut observed = Vec::new();
            for _ in 0..4 {
                if let Some(rate) = status_main.poll_request() {
                    observed.push(rate);
                }
            }
            for rate in observed {
                assert_eq!(
                    rate, 48_000,
                    "multi-field handshake violated: flag observed without matching data"
                );
            }
        });

        t_rt.join().unwrap();
        t_main.join().unwrap();
    });
}

#[test]
fn test_rt_status_multifield_relaxed_fails() {
    let result = std::panic::catch_unwind(|| {
        loom::model(|| {
            let status = Arc::new(LoomRtStatus::new());
            let status_rt = status.clone();
            let status_main = status.clone();

            let t_rt = thread::spawn(move || {
                status_rt.requested_host_rate.with_mut(|ptr| unsafe {
                    *ptr = 48_000;
                });
                // Missing Release: the flag does not order the data write.
                status_rt
                    .status_bits
                    .fetch_or(RT_FLAG_NEEDS_RESAMPLER_REBUILD, Ordering::Relaxed);
            });

            let t_main = thread::spawn(move || {
                let mut observed = Vec::new();
                for _ in 0..4 {
                    if (status_main.status_bits.load(Ordering::Acquire)
                        & RT_FLAG_NEEDS_RESAMPLER_REBUILD)
                        != 0
                    {
                        observed.push(status_main.requested_host_rate.with(|ptr| unsafe { *ptr }));
                        status_main
                            .status_bits
                            .fetch_and(!RT_FLAG_NEEDS_RESAMPLER_REBUILD, Ordering::Relaxed);
                    }
                }
                for rate in observed {
                    assert_eq!(rate, 48_000);
                }
            });

            t_rt.join().unwrap();
            t_main.join().unwrap();
        });
    });
    assert!(
        result.is_err(),
        "Expected loom to catch a data race in the multi-field handshake without Release ordering"
    );
}
