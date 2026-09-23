// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Benchmark suite for the RT structural-swap scheduler (`spsc::swap`)
//! and 3-tier GC cascade (`spsc::gc`).
//!
//! Measures:
//! - Phase 0–2 drain under quiescent (empty), low-contention (single),
//!   high-contention coalescing (burst of 16 same-key payloads), and budget-exceeded
//!   deferral scenarios.
//! - 3-tier GC cascade: Tier 1 (SPSC ring), Tier 2 (16-slot RT parking lot),
//!   and Tier 3 (atomic 64-bit overflow buffer).
//! - Off-RT housekeeping sweeping all 3 tiers via `drain_gc_channels`.
//!
//! Strictly zero `log::*` calls in hot-paths and zero heap allocations/drops
//! during the real-time measurement windows.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use neural_amp_modeler_rs::common::spsc::{
    CabSimSwapPayload, GcItem, GcOverflowBuffer, GcSink, RtStatusFlags, RtSwapDrain, RtSwapHandler,
    SwapBudget, SwapTunables, drain_gc_channels, gc_cascade,
};
use rtrb::{Consumer, Producer, RingBuffer};
use std::hint::black_box;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug)]
struct BenchPayload {
    _id: u64,
    is_structural: bool,
    key: u64,
}

impl BenchPayload {
    fn structural(id: u64, key: u64) -> Box<Self> {
        Box::new(Self {
            _id: id,
            is_structural: true,
            key,
        })
    }
}

struct BenchHandler {
    #[expect(
        clippy::vec_box,
        reason = "Audio-thread safety: retains Box without deallocation"
    )]
    retained_payloads: Vec<Box<BenchPayload>>,
    gc_pool: Vec<GcItem>,
}

impl BenchHandler {
    fn new(capacity: usize) -> Self {
        let mut gc_pool = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            gc_pool.push(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                generation: 0,
                pair: None,
            })));
        }
        Self {
            retained_payloads: Vec::with_capacity(capacity),
            gc_pool,
        }
    }

    fn retire_one(&mut self, gc: &mut GcSink<'_>) {
        if let Some(item) = self.gc_pool.pop() {
            gc.retire(item);
        }
    }
}

impl RtSwapHandler for BenchHandler {
    type Payload = BenchPayload;

    #[inline(always)]
    fn is_structural(&self, payload: &Self::Payload) -> bool {
        payload.is_structural
    }

    #[inline(always)]
    fn coalesce_key(&self, payload: &Self::Payload) -> Option<u64> {
        if payload.is_structural {
            Some(payload.key)
        } else {
            None
        }
    }

    #[inline(always)]
    fn install(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>) {
        self.retained_payloads.push(payload);
        self.retire_one(gc);
    }

    #[inline(always)]
    fn discard(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>) {
        self.retained_payloads.push(payload);
        self.retire_one(gc);
    }
}

const DEFAULT_PARKING_LOT: [Option<GcItem>; 16] = [
    None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
];

struct SwapHarness {
    drain: RtSwapDrain<Consumer<Box<BenchPayload>>>,
    producer: Producer<Box<BenchPayload>>,
    handler: BenchHandler,
    gc_producer: Producer<GcItem>,
    _gc_consumer: Consumer<GcItem>,
    parking_lot: [Option<GcItem>; 16],
    overflow: GcOverflowBuffer,
    rt_status: RtStatusFlags,
    dirty: AtomicBool,
}

impl SwapHarness {
    fn new(ring_cap: usize, pool_cap: usize) -> Self {
        let (producer, consumer) = RingBuffer::new(ring_cap);
        let (gc_producer, gc_consumer) = RingBuffer::new(64);
        let tunables = SwapTunables {
            pops_per_callback: 32,
            swaps_per_callback: 1,
            backlog_flag: false,
        };
        Self {
            drain: RtSwapDrain::new(consumer, tunables),
            producer,
            handler: BenchHandler::new(pool_cap),
            gc_producer,
            _gc_consumer: gc_consumer,
            parking_lot: DEFAULT_PARKING_LOT,
            overflow: GcOverflowBuffer::new(32),
            rt_status: RtStatusFlags::new(),
            dirty: AtomicBool::new(false),
        }
    }

    #[inline(always)]
    fn drain_once(&mut self, swaps_allowed: usize) {
        let mut budget = SwapBudget::new(swaps_allowed);
        let mut gc_sink = GcSink {
            producer: &mut self.gc_producer,
            parking_lot: &mut self.parking_lot,
            overflow: &self.overflow,
            rt_status: &self.rt_status,
            parking_lot_dirty: Some(&self.dirty),
        };
        self.drain
            .drain(&mut self.handler, &mut budget, &mut gc_sink);
    }
}

fn bench_swap_drain_quiescent(c: &mut Criterion) {
    let mut harness = SwapHarness::new(16, 16);
    c.bench_function("Swap_Drain_Quiescent", |b| {
        b.iter(|| {
            harness.drain_once(1);
            black_box(&harness.drain);
        });
    });
}

fn bench_swap_drain_low_contention_single(c: &mut Criterion) {
    c.bench_function("Swap_Drain_LowContention_Single", |b| {
        b.iter_batched(
            || {
                let mut harness = SwapHarness::new(16, 16);
                let _ = harness.producer.push(BenchPayload::structural(1, 100));
                harness
            },
            |mut harness| {
                harness.drain_once(1);
                black_box(harness)
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_swap_drain_high_contention_coalescing(c: &mut Criterion) {
    c.bench_function("Swap_Drain_HighContention_Coalescing", |b| {
        b.iter_batched(
            || {
                let mut harness = SwapHarness::new(32, 32);
                for id in 1..=16 {
                    let _ = harness.producer.push(BenchPayload::structural(id, 42));
                }
                harness
            },
            |mut harness| {
                harness.drain_once(1);
                black_box(harness)
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_swap_drain_budget_exceeded_deferred(c: &mut Criterion) {
    c.bench_function("Swap_Drain_Budget_Exceeded_Deferred", |b| {
        b.iter_batched(
            || {
                let mut harness = SwapHarness::new(16, 16);
                // Push 2 structural payloads with different keys so they do not coalesce.
                let _ = harness.producer.push(BenchPayload::structural(1, 101));
                let _ = harness.producer.push(BenchPayload::structural(2, 102));
                harness
            },
            |mut harness| {
                // Callback 1: installs payload 1, budget (1) exhausted, parks payload 2 in deferred slot.
                harness.drain_once(1);
                // Callback 2: resolves deferred payload 2 in Phase 0.
                harness.drain_once(1);
                black_box(harness)
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gc_cascade_tier1_spsc(c: &mut Criterion) {
    c.bench_function("Gc_Cascade_Tier1_Spsc", |b| {
        b.iter_batched(
            || {
                let (producer, consumer) = RingBuffer::<GcItem>::new(64);
                let parking_lot = DEFAULT_PARKING_LOT;
                let overflow = GcOverflowBuffer::new(32);
                let rt_status = RtStatusFlags::new();
                let item = GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                    generation: 0,
                    pair: None,
                }));
                (producer, consumer, parking_lot, overflow, rt_status, item)
            },
            |(mut producer, _consumer, mut parking_lot, overflow, rt_status, item)| {
                gc_cascade(
                    Some(item),
                    &mut producer,
                    &mut parking_lot,
                    &overflow,
                    &rt_status,
                );
                black_box((producer, parking_lot))
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gc_cascade_tier2_parking_lot(c: &mut Criterion) {
    c.bench_function("Gc_Cascade_Tier2_ParkingLot", |b| {
        b.iter_batched(
            || {
                let (mut producer, consumer) = RingBuffer::<GcItem>::new(4);
                // Fill the SPSC ring completely so Tier 1 fails
                for _ in 0..4 {
                    let _ = producer.push(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    })));
                }
                let parking_lot = DEFAULT_PARKING_LOT;
                let overflow = GcOverflowBuffer::new(32);
                let rt_status = RtStatusFlags::new();
                let item = GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                    generation: 0,
                    pair: None,
                }));
                (producer, consumer, parking_lot, overflow, rt_status, item)
            },
            |(mut producer, _consumer, mut parking_lot, overflow, rt_status, item)| {
                gc_cascade(
                    Some(item),
                    &mut producer,
                    &mut parking_lot,
                    &overflow,
                    &rt_status,
                );
                black_box(parking_lot)
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gc_cascade_tier3_overflow(c: &mut Criterion) {
    c.bench_function("Gc_Cascade_Tier3_Overflow", |b| {
        b.iter_batched(
            || {
                let (mut producer, consumer) = RingBuffer::<GcItem>::new(4);
                // Fill the SPSC ring
                for _ in 0..4 {
                    let _ = producer.push(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    })));
                }
                // Fill all 16 slots of parking lot
                let mut parking_lot = DEFAULT_PARKING_LOT;
                for slot in &mut parking_lot {
                    *slot = Some(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    })));
                }
                let overflow = GcOverflowBuffer::new(32);
                let rt_status = RtStatusFlags::new();
                let item = GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                    generation: 0,
                    pair: None,
                }));
                (producer, consumer, parking_lot, overflow, rt_status, item)
            },
            |(mut producer, _consumer, mut parking_lot, overflow, rt_status, item)| {
                gc_cascade(
                    Some(item),
                    &mut producer,
                    &mut parking_lot,
                    &overflow,
                    &rt_status,
                );
                black_box((rt_status, overflow))
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_gc_drain_housekeeping_all_tiers(c: &mut Criterion) {
    c.bench_function("Gc_Drain_Housekeeping_AllTiers", |b| {
        b.iter_batched(
            || {
                let (mut producer, consumer) = RingBuffer::<GcItem>::new(16);
                for _ in 0..8 {
                    let _ = producer.push(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    })));
                }
                let mut parking_lot = DEFAULT_PARKING_LOT;
                for slot in parking_lot.iter_mut().take(4) {
                    *slot = Some(GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    })));
                }
                let overflow = GcOverflowBuffer::new(16);
                for _ in 0..4 {
                    let item = GcItem::CabSimSwap(Box::new(CabSimSwapPayload {
                        generation: 0,
                        pair: None,
                    }));
                    overflow.push(item);
                }
                let rt_status = RtStatusFlags::new();
                (consumer, parking_lot, overflow, rt_status)
            },
            |(mut consumer, mut parking_lot, overflow, rt_status)| {
                let dropped =
                    drain_gc_channels(&mut consumer, &overflow, &mut parking_lot, &rt_status);
                black_box(dropped)
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group! {
    name = spsc_swap_benches;
    config = Criterion::default()
        .sample_size(50)
        .noise_threshold(0.05);
    targets =
        bench_swap_drain_quiescent,
        bench_swap_drain_low_contention_single,
        bench_swap_drain_high_contention_coalescing,
        bench_swap_drain_budget_exceeded_deferred,
        bench_gc_cascade_tier1_spsc,
        bench_gc_cascade_tier2_parking_lot,
        bench_gc_cascade_tier3_overflow,
        bench_gc_drain_housekeeping_all_tiers,
}

criterion_main!(spsc_swap_benches);
