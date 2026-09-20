// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Test plan for the generic RT structural-swap scheduler: a fake payload
//! family with a counting handler and the real GC-cascade machinery — no
//! audio, and a heap-audit lane proving the drain path allocates nothing.

use super::{GcSink, RtSwapDrain, RtSwapHandler, SwapBudget, SwapRing, SwapTunables};
use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};
use crate::common::spsc::{
    GcItem, GcOverflowBuffer, RT_STATUS_PARAM_QUEUE_BACKLOG, RT_STATUS_SPSC_DRAIN_TRUNCATED,
    RT_STATUS_STRUCTURAL_DEFERRED, RT_STATUS_STRUCTURAL_SUPERSEDED, RtStatusFlags,
};
use proptest::prelude::*;
use rtrb::{Consumer, Producer, RingBuffer};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Fake payload: scalar (light), structural (coalescible), or restore-style
/// (non-coalescible, `coalesce_key = None`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Scalar,
    Structural,
    Restore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TestPayload {
    id: u64,
    kind: Kind,
    key: u64,
    generation: u64,
}

impl TestPayload {
    fn scalar(id: u64) -> Box<Self> {
        Box::new(Self {
            id,
            kind: Kind::Scalar,
            key: 0,
            generation: 0,
        })
    }

    fn structural(id: u64, key: u64) -> Box<Self> {
        Box::new(Self {
            id,
            kind: Kind::Structural,
            key,
            generation: 0,
        })
    }

    fn restore(id: u64) -> Box<Self> {
        Box::new(Self {
            id,
            kind: Kind::Restore,
            key: 0,
            generation: 0,
        })
    }

    fn stale(id: u64, key: u64, generation: u64) -> Box<Self> {
        Box::new(Self {
            id,
            kind: Kind::Structural,
            key,
            generation,
        })
    }
}

/// Counting handler: records installs/discards and retires pre-allocated GC
/// items through the real cascade (models a real handler whose swapped
/// resources go to the GC, never dropped on the RT thread). All bookkeeping
/// vectors are pre-reserved so the drain path stays allocation-free.
#[expect(
    clippy::redundant_allocation,
    reason = "GcItem::Test wraps Box<Arc<AtomicU32>>; the pool must pre-allocate exactly that shape so retiring during the drain moves, never allocates"
)]
struct TestHandler {
    current_gen: u64,
    installed: Vec<u64>,
    discarded: Vec<u64>,
    scalars_installed: usize,
    retired: usize,
    gc_pool: VecDeque<Box<Arc<AtomicU32>>>,
}

impl TestHandler {
    fn new(pool_size: usize) -> Self {
        Self {
            current_gen: 0,
            installed: Vec::with_capacity(pool_size),
            discarded: Vec::with_capacity(pool_size),
            scalars_installed: 0,
            retired: 0,
            gc_pool: (0..pool_size)
                .map(|_| Box::new(Arc::new(AtomicU32::new(0))))
                .collect(),
        }
    }

    fn retire_one(&mut self, gc: &mut GcSink<'_>) {
        if let Some(arc) = self.gc_pool.pop_front() {
            gc.retire(GcItem::Test(arc));
            self.retired += 1;
        }
    }
}

impl RtSwapHandler for TestHandler {
    type Payload = TestPayload;

    fn is_structural(&self, payload: &Self::Payload) -> bool {
        payload.kind != Kind::Scalar
    }

    fn coalesce_key(&self, payload: &Self::Payload) -> Option<u64> {
        match payload.kind {
            Kind::Structural => Some(payload.key),
            Kind::Scalar | Kind::Restore => None,
        }
    }

    fn current_generation(&self) -> Option<u64> {
        Some(self.current_gen)
    }

    fn generation_of(&self, payload: &Self::Payload) -> Option<u64> {
        Some(payload.generation)
    }

    fn install(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>) {
        if payload.kind == Kind::Scalar {
            self.scalars_installed += 1;
        } else {
            self.installed.push(payload.id);
        }
        self.retire_one(gc);
    }

    fn discard(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>) {
        self.discarded.push(payload.id);
        self.retire_one(gc);
    }
}

/// Per-test state: the RT-side drain, the producer handle, and the GC scope.
struct Harness {
    drain: RtSwapDrain<Consumer<Box<TestPayload>>>,
    tunables: SwapTunables,
    producer: Producer<Box<TestPayload>>,
    gc_prod: Producer<GcItem>,
    gc_cons: Consumer<GcItem>,
    parking_lot: [Option<GcItem>; 16],
    overflow: Arc<GcOverflowBuffer>,
    rt_status: Arc<RtStatusFlags>,
    dirty: Arc<AtomicBool>,
}

impl Harness {
    /// `ring_capacity`, pop cap, structural swaps per callback, backlog flag.
    fn new(ring_capacity: usize, pops: usize, swaps: usize, backlog: bool) -> Self {
        let (prod, cons) = RingBuffer::new(ring_capacity);
        let (gc_prod, gc_cons) = RingBuffer::new(ring_capacity.max(64));
        let tunables = SwapTunables {
            pops_per_callback: pops,
            swaps_per_callback: swaps,
            backlog_flag: backlog,
        };
        Self {
            drain: RtSwapDrain::new(cons, tunables),
            tunables,
            producer: prod,
            gc_prod,
            gc_cons,
            parking_lot: std::array::from_fn(|_| None),
            overflow: Arc::new(GcOverflowBuffer::new(64)),
            rt_status: Arc::new(RtStatusFlags::new()),
            dirty: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Runs one callback drain with a fresh budget, as a consumer would.
    fn callback(&mut self, handler: &mut TestHandler) {
        let mut budget = SwapBudget::new(self.tunables.swaps_per_callback);
        self.run_drain(handler, &mut budget);
    }

    fn run_drain(&mut self, handler: &mut TestHandler, budget: &mut SwapBudget) {
        let mut sink = GcSink {
            producer: &mut self.gc_prod,
            parking_lot: &mut self.parking_lot,
            overflow: &self.overflow,
            rt_status: &self.rt_status,
            parking_lot_dirty: Some(&self.dirty),
        };
        self.drain.drain(handler, budget, &mut sink);
    }

    /// Drains the off-RT GC side and returns the number of retired items
    /// actually dropped outside the RT thread.
    fn drain_gc(&mut self) -> usize {
        let mut dropped = 0;
        while let Ok(item) = self.gc_cons.pop() {
            drop(item);
            dropped += 1;
        }
        dropped
    }
}

/// Test 1 — with `swaps_per_callback = 1` and N queued different-key
/// structural payloads, exactly one install happens per callback, the next
/// payload parks, and the deferral flag/counter fire.
#[test]
fn budget_one_apply_per_callback() {
    let mut h = Harness::new(8, 8, 1, false);
    for id in 0..3u64 {
        h.producer
            .push(TestPayload::structural(id, 100 + id))
            .unwrap();
    }
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);

    assert_eq!(handler.installed, vec![0], "exactly one structural install");
    assert!(h.drain.has_deferred(), "the next payload must park");
    assert!(
        h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_DEFERRED)
    );
    assert_eq!(
        h.rt_status
            .structural_deferred_total
            .load(Ordering::Relaxed),
        2,
        "park + budget-stop both defer"
    );

    // Next callbacks with a fresh budget resolve the parked payload first.
    h.callback(&mut handler);
    assert_eq!(handler.installed, vec![0, 1]);
    h.callback(&mut handler);
    assert_eq!(handler.installed, vec![0, 1, 2]);
    assert!(!h.drain.has_deferred());
    assert!(h.drain.is_empty());
    assert_eq!(h.drain_gc(), 3, "every replaced resource reached the GC");
}

/// Test 2 — three same-key payloads in one window collapse to the last one;
/// the two intermediates are discarded to the GC cascade and counted as
/// superseded.
#[test]
fn latest_wins_coalescing_discards_to_gc() {
    let mut h = Harness::new(8, 8, 1, false);
    for id in 0..3u64 {
        h.producer.push(TestPayload::structural(id, 7)).unwrap();
    }
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);

    assert_eq!(handler.installed, vec![2], "only the latest installs");
    assert_eq!(handler.discarded, vec![0, 1], "intermediates retire to GC");
    assert!(
        h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_SUPERSEDED)
    );
    assert_eq!(
        h.rt_status
            .structural_superseded_total
            .load(Ordering::Relaxed),
        2
    );
    assert!(
        h.dirty.load(Ordering::Acquire),
        "retire raised the dirty latch"
    );
    // 2 discard-retires (intermediates) + 1 install-retire (replaced active).
    assert_eq!(
        h.drain_gc(),
        3,
        "retired resources reached the off-RT drain"
    );
}

/// Test 3 — a non-coalescible payload (restore transaction) is never
/// superseded: it applies in FIFO order before the newer different-key
/// structural that arrived behind it.
#[test]
fn non_coalescible_never_superseded() {
    let mut h = Harness::new(8, 8, 1, false);
    h.producer.push(TestPayload::restore(0)).unwrap();
    h.producer.push(TestPayload::structural(1, 5)).unwrap();
    let mut handler = TestHandler::new(16);

    h.callback(&mut handler);
    // The restore was drained first (FIFO); the model became the candidate
    // and parked under the exhausted budget.
    assert_eq!(
        handler.installed,
        vec![0],
        "restore applies before the model"
    );
    assert!(
        h.drain.has_deferred(),
        "the newer model parks, never supersedes"
    );
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_SUPERSEDED)
    );

    h.callback(&mut handler);
    assert_eq!(handler.installed, vec![0, 1], "both applied in FIFO order");
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_SUPERSEDED)
    );
}

/// Test 4 — payloads stamped with an old generation are discarded without
/// install: in flight (Phase 1) and while parked (Phase 0), without raising
/// the superseded flag.
#[test]
fn stale_generation_discarded_without_install() {
    // In-flight staleness: a stale envelope is discarded during the drain.
    let mut h = Harness::new(8, 8, 1, false);
    h.producer.push(TestPayload::stale(0, 3, 99)).unwrap();
    h.producer.push(TestPayload::structural(1, 3)).unwrap();
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);

    assert_eq!(handler.discarded, vec![0], "stale envelope discarded");
    assert_eq!(
        handler.installed,
        vec![1],
        "current-generation build installs"
    );
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_SUPERSEDED)
    );

    // Stale while parked: a parked payload whose generation went stale is
    // discarded at Phase 0 of the next callback, also without supersession.
    let mut h = Harness::new(8, 8, 1, false);
    h.producer.push(TestPayload::structural(0, 9)).unwrap();
    h.producer.push(TestPayload::structural(1, 10)).unwrap();
    let mut handler = TestHandler::new(16);
    // Payload 1 flush-installs payload 0 and parks under the exhausted budget.
    h.callback(&mut handler);
    assert!(h.drain.has_deferred(), "payload 1 parked by Phase 2");

    handler.current_gen += 1;
    h.callback(&mut handler);
    assert_eq!(handler.discarded, vec![1], "stale parked payload discarded");
    assert_eq!(handler.installed, vec![0]);
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_STRUCTURAL_SUPERSEDED)
    );
}

/// Test 5 — a parked payload resolves at the start of the next callback,
/// causally before newer payloads still in the ring.
#[test]
fn deferred_resolves_next_callback_before_newer_ring_payloads() {
    let mut h = Harness::new(8, 8, 1, false);
    h.producer.push(TestPayload::structural(0, 1)).unwrap();
    h.producer.push(TestPayload::structural(1, 2)).unwrap();
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);
    assert_eq!(handler.installed, vec![0]);
    assert!(h.drain.has_deferred(), "payload 1 parked");

    // A newer structural (different key) arrives while payload 1 is parked.
    h.producer.push(TestPayload::structural(2, 3)).unwrap();

    h.callback(&mut handler);
    assert_eq!(
        handler.installed,
        vec![0, 1],
        "the parked payload applies before the newer ring payload"
    );
    // Payload 2 became the candidate and parked under the exhausted budget.
    h.callback(&mut handler);
    assert_eq!(handler.installed, vec![0, 1, 2]);
}

/// Sequence-tracking ring wrapper recording the acknowledgment hooks.
#[derive(Debug, PartialEq, Eq)]
enum SeqEvent {
    Pop,
    Rollback,
    Advance,
}

struct SeqRing {
    inner: Consumer<Box<TestPayload>>,
    processed: u64,
    log: Rc<RefCell<Vec<SeqEvent>>>,
}

impl SwapRing for SeqRing {
    type Payload = TestPayload;

    fn pop(&mut self) -> Option<Box<Self::Payload>> {
        let payload = rtrb::Consumer::pop(&mut self.inner).ok();
        if payload.is_some() {
            self.processed = self.processed.wrapping_add(1);
            self.log.borrow_mut().push(SeqEvent::Pop);
        }
        payload
    }

    fn peek(&self) -> Option<&Self::Payload> {
        rtrb::Consumer::peek(&self.inner).ok().map(|b| &**b)
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn occupied(&self) -> usize {
        rtrb::Consumer::slots(&self.inner)
    }

    fn advance_resolved(&mut self) {
        self.processed = self.processed.wrapping_add(1);
        self.log.borrow_mut().push(SeqEvent::Advance);
    }

    fn rollback_last_pop(&mut self) {
        self.processed = self.processed.wrapping_sub(1);
        self.log.borrow_mut().push(SeqEvent::Rollback);
    }
}

/// Test 6 — sequence acknowledgment stays gapless across a park: the pop of
/// the parked candidate is rolled back and the deferred slot consumes its
/// sequence slot exactly once on resolution.
#[test]
fn park_rollback_and_resolve_keeps_ack_gapless() {
    let (mut prod, cons) = RingBuffer::new(8);
    let log: Rc<RefCell<Vec<SeqEvent>>> = Rc::new(RefCell::new(Vec::new()));
    let ring = SeqRing {
        inner: cons,
        processed: 0,
        log: Rc::clone(&log),
    };
    let mut drain = RtSwapDrain::new(
        ring,
        SwapTunables {
            pops_per_callback: 8,
            swaps_per_callback: 1,
            backlog_flag: false,
        },
    );
    prod.push(TestPayload::structural(0, 1)).unwrap();
    prod.push(TestPayload::structural(1, 2)).unwrap();

    let mut parking_lot: [Option<GcItem>; 16] = std::array::from_fn(|_| None);
    let overflow = GcOverflowBuffer::new(64);
    let rt_status = RtStatusFlags::new();
    let mut handler = TestHandler::new(16);
    let (mut gc_prod, gc_cons) = RingBuffer::new(64);

    // Callback 1: both payloads popped; the first flush-installs, the second
    // (candidate) parks — its pop must be rolled back so the ack published at
    // the end of the callback covers exactly the applied command.
    {
        let mut budget = SwapBudget::new(1);
        let mut sink = GcSink {
            producer: &mut gc_prod,
            parking_lot: &mut parking_lot,
            overflow: &overflow,
            rt_status: &rt_status,
            parking_lot_dirty: None,
        };
        drain.drain(&mut handler, &mut budget, &mut sink);
    }
    assert_eq!(
        *log.borrow(),
        vec![SeqEvent::Pop, SeqEvent::Pop, SeqEvent::Rollback],
        "rollback exactly on park"
    );
    assert!(drain.has_deferred(), "candidate parked");
    assert_eq!(ring_net_seq(&log), 1, "ack covers only the applied command");

    // Callback 2: the parked payload resolves via advance (its sequence slot
    // is consumed exactly once, gapless with the earlier pop).
    {
        let mut budget = SwapBudget::new(1);
        let mut sink = GcSink {
            producer: &mut gc_prod,
            parking_lot: &mut parking_lot,
            overflow: &overflow,
            rt_status: &rt_status,
            parking_lot_dirty: None,
        };
        drain.drain(&mut handler, &mut budget, &mut sink);
    }
    assert_eq!(
        *log.borrow(),
        vec![
            SeqEvent::Pop,
            SeqEvent::Pop,
            SeqEvent::Rollback,
            SeqEvent::Advance
        ],
        "advance exactly on resolve"
    );
    assert_eq!(
        ring_net_seq(&log),
        2,
        "both sequence slots consumed, gapless"
    );
    assert_eq!(handler.installed, vec![0, 1]);
    drop(gc_cons);
}

/// Net sequence counter implied by the recorded hook events.
fn ring_net_seq(log: &Rc<RefCell<Vec<SeqEvent>>>) -> u64 {
    log.borrow().iter().fold(0u64, |seq, event| match event {
        SeqEvent::Pop | SeqEvent::Advance => seq.wrapping_add(1),
        SeqEvent::Rollback => seq.wrapping_sub(1),
    })
}

/// Test 7 — reaching the pop cap with a non-empty ring flags truncation
/// (structural drains) or backlog (scalar-parameter drains).
#[test]
fn pop_cap_truncates_and_flags() {
    // Structural drain (backlog off): truncation flag on cap + non-empty.
    let mut h = Harness::new(8, 2, 4, false);
    for id in 0..4u64 {
        h.producer
            .push(TestPayload::structural(id, 100 + id))
            .unwrap();
    }
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);
    assert!(
        h.rt_status
            .check_and_clear_flag(RT_STATUS_SPSC_DRAIN_TRUNCATED)
    );
    assert_eq!(
        handler.installed.len(),
        2,
        "two flush-installs inside the cap"
    );

    // Scalar-parameter drain (backlog on): backlog flag instead.
    let mut h = Harness::new(8, 2, 4, true);
    for id in 0..4u64 {
        h.producer.push(TestPayload::scalar(id)).unwrap();
    }
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);
    assert!(
        h.rt_status
            .check_and_clear_flag(RT_STATUS_PARAM_QUEUE_BACKLOG)
    );
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_SPSC_DRAIN_TRUNCATED)
    );
    assert_eq!(handler.scalars_installed, 2, "scalar budget = pop cap");

    // Cap reached exactly as the ring empties: no truncation flag.
    let mut h = Harness::new(8, 2, 4, false);
    for id in 0..2u64 {
        h.producer
            .push(TestPayload::structural(id, 100 + id))
            .unwrap();
    }
    let mut handler = TestHandler::new(16);
    h.callback(&mut handler);
    assert!(
        !h.rt_status
            .check_and_clear_flag(RT_STATUS_SPSC_DRAIN_TRUNCATED)
    );
    assert_eq!(handler.installed, vec![0, 1]);
    assert!(handler.discarded.is_empty());
}

/// Test 8 — a budget shared between two drains: the first drain consumes the
/// allowance, the second defers and applies on the next callback.
#[test]
fn shared_budget_across_drains() {
    let (mut prod_a, cons_a) = RingBuffer::new(8);
    let (mut prod_b, cons_b) = RingBuffer::new(8);
    let tunables = SwapTunables {
        pops_per_callback: 8,
        swaps_per_callback: 1,
        backlog_flag: false,
    };
    let mut drain_a = RtSwapDrain::new(cons_a, tunables);
    let mut drain_b = RtSwapDrain::new(cons_b, tunables);
    prod_a.push(TestPayload::structural(0, 1)).unwrap();
    prod_b.push(TestPayload::structural(1, 2)).unwrap();

    let overflow = GcOverflowBuffer::new(64);
    let rt_status = RtStatusFlags::new();
    let mut parking_lot: [Option<GcItem>; 16] = std::array::from_fn(|_| None);
    let (mut gc_prod, gc_cons) = RingBuffer::new(64);
    let mut handler = TestHandler::new(16);

    // One shared budget for both drains of the same callback.
    let mut budget = SwapBudget::new(tunables.swaps_per_callback);
    let mut sink = GcSink {
        producer: &mut gc_prod,
        parking_lot: &mut parking_lot,
        overflow: &overflow,
        rt_status: &rt_status,
        parking_lot_dirty: None,
    };
    drain_a.drain(&mut handler, &mut budget, &mut sink);
    drain_b.drain(&mut handler, &mut budget, &mut sink);

    assert_eq!(handler.installed, vec![0], "first drain wins the allowance");
    assert!(
        !drain_b.has_deferred(),
        "drain B left its payload queued (in-ring deferral)"
    );
    assert!(rt_status.check_and_clear_flag(RT_STATUS_STRUCTURAL_DEFERRED));

    // Next callback, fresh shared budget: drain B applies.
    let mut budget = SwapBudget::new(tunables.swaps_per_callback);
    let mut sink = GcSink {
        producer: &mut gc_prod,
        parking_lot: &mut parking_lot,
        overflow: &overflow,
        rt_status: &rt_status,
        parking_lot_dirty: None,
    };
    drain_b.drain(&mut handler, &mut budget, &mut sink);
    assert_eq!(handler.installed, vec![0, 1]);
    drop(gc_cons);
}

/// Test 9 — heap-audit lane: a full saturation soak (coalescing windows,
/// flush-installs, parks, GC cascades through ring + parking-lot tiers)
/// performs zero allocations on the drain path.
#[test]
fn zero_alloc_retirement_under_saturation() {
    let mut h = Harness::new(64, 8, 1, false);
    // Setup allocations happen before the watchdog starts.
    for id in 0..32u64 {
        let payload = if id % 4 == 3 {
            TestPayload::scalar(id)
        } else {
            TestPayload::structural(id, id % 3)
        };
        h.producer.push(payload).unwrap();
    }
    let mut handler = TestHandler::new(64);

    let _guard = TrackingGuard::new();
    let mut rounds = 0;
    while (!h.drain.is_empty() || h.drain.has_deferred()) && rounds < 64 {
        h.callback(&mut handler);
        rounds += 1;
    }
    let allocs = get_alloc_count();
    drop(_guard);

    assert_eq!(allocs, 0, "the drain path must be allocation-free");
    assert!(
        h.drain.is_empty() && !h.drain.has_deferred(),
        "fully drained"
    );
    assert_eq!(handler.retired, 32, "one retire per accounted payload");
    assert_eq!(h.drain_gc(), 32, "all retired resources reached the GC");
}

// Test 10 — randomized push/callback interleavings: no payload is ever
// dropped on the RT thread, and the final installed state equals the latest
// delivered payload per coalesce key (restores apply in FIFO order).
proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn randomized_interleavings_never_drop_or_lose_payloads(
        ops in prop::collection::vec(op_strategy(), 1..=32),
    ) {
        let mut h = Harness::new(64, 8, 1, false);
        let mut handler = TestHandler::new(64);
        // Delivered payloads, split by kind: structural (id, key), restore
        // (ids in push order), and an inline-scalar counter.
        let mut structural_delivered: Vec<(u64, u64)> = Vec::new();
        let mut restore_ids: Vec<u64> = Vec::new();
        let mut scalars_delivered = 0usize;
        let mut next_id = 0u64;

        for op in ops {
            match op {
                Op::PushStructural(key) => {
                    let id = next_id;
                    next_id += 1;
                    if h.producer.push(TestPayload::structural(id, key)).is_ok() {
                        structural_delivered.push((id, key));
                    }
                }
                Op::PushScalar => {
                    let id = next_id;
                    next_id += 1;
                    if h.producer.push(TestPayload::scalar(id)).is_ok() {
                        scalars_delivered += 1;
                    }
                }
                Op::PushRestore => {
                    let id = next_id;
                    next_id += 1;
                    if h.producer.push(TestPayload::restore(id)).is_ok() {
                        restore_ids.push(id);
                    }
                }
                Op::Callback => h.callback(&mut handler),
            }
        }

        // Drain to quiescence with fresh budgets; every callback with pending
        // work makes progress (Phase 0 resolves the deferred slot, Phase 1
        // pops), so the loop terminates within the hard bound.
        let mut rounds = 0;
        while (!h.drain.is_empty() || h.drain.has_deferred())
            && rounds < 2 * structural_delivered.len() + 8
        {
            h.callback(&mut handler);
            rounds += 1;
        }
        prop_assert!(
            h.drain.is_empty() && !h.drain.has_deferred(),
            "drain must reach quiescence"
        );

        // (a) Every delivered structural/restore payload is accounted exactly
        // once: installed or discarded to GC — never dropped on the RT
        // thread. Scalars install inline and are counted separately.
        let mut accounted: Vec<u64> = handler
            .installed
            .iter()
            .chain(handler.discarded.iter())
            .copied()
            .collect();
        accounted.sort_unstable();
        accounted.dedup();
        prop_assert_eq!(
            accounted.len(),
            structural_delivered.len() + restore_ids.len(),
            "no loss, no duplicates"
        );
        prop_assert_eq!(handler.scalars_installed, scalars_delivered);

        // (b) Latest-wins per coalesce key: the last delivered structural for
        // a key is the last installed payload for that key.
        for key in 0..4u64 {
            let last_delivered = structural_delivered
                .iter()
                .rev()
                .find(|(_, k)| *k == key)
                .map(|(id, _)| *id);
            let last_installed_for_key = handler
                .installed
                .iter()
                .rev()
                .find(|installed| {
                    structural_delivered
                        .iter()
                        .rev()
                        .any(|(d, k)| *d == **installed && *k == key)
                })
                .copied();
            prop_assert_eq!(last_installed_for_key, last_delivered);
        }

        // (c) Non-coalescible restores apply in FIFO order.
        let installed_restores: Vec<u64> = handler
            .installed
            .iter()
            .copied()
            .filter(|id| restore_ids.contains(id))
            .collect();
        prop_assert_eq!(installed_restores, restore_ids, "restore FIFO order");

        // (e) All retired resources reached the off-RT GC drain.
        prop_assert_eq!(h.drain_gc(), handler.retired);
    }
}

#[derive(Debug, Clone)]
enum Op {
    PushStructural(u64),
    PushScalar,
    PushRestore,
    Callback,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        6 => (0u64..4).prop_map(Op::PushStructural),
        2 => Just(Op::PushScalar),
        2 => Just(Op::PushRestore),
        4 => Just(Op::Callback),
    ]
}
