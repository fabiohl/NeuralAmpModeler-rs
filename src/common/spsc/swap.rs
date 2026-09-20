// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Generic RT structural-swap scheduler: the canonical 3-phase drain protocol
//! shared by every consumer of off-RT resource swaps (model, IR, resampler,
//! oversampling engines, state restore).
//!
//! # The protocol
//!
//! Structural swaps follow three phases per audio callback:
//!
//! 1. **Phase 0 — deferred resolution.** A payload parked by a previous
//!    callback is resolved first: it is causally before everything still in
//!    the ring, so it applies before any newer command. It is discarded
//!    instead when stale (its generation no longer matches the current
//!    request) or superseded (the ring head carries the same coalesce key).
//! 2. **Phase 1 — bounded drain with latest-wins coalescing.** Payloads are
//!    popped under a per-callback pop cap; same-key structural payloads
//!    collapse to the latest one, light scalar payloads apply inline, and
//!    retired resources go through the GC cascade — never dropped on the RT
//!    thread.
//! 3. **Phase 2 — budgeted apply or park.** At most
//!    [`SwapTunables::swaps_per_callback`] structural applies per callback
//!    (shareable across all drains of the callback via [`SwapBudget`]); the
//!    excess parks in a single deferred slot owned by the drain.
//!
//! # Semantics preserved per consumer
//!
//! Every knob that historically diverged between integrations is an explicit
//! configuration value — nothing is averaged or silently unified:
//!
//! - Pop caps (e.g. 8 per dedicated drain vs 64 for a mixed command ring) —
//!   [`SwapTunables::pops_per_callback`].
//! - Scalar budgets (16 with a backlog flag vs none) —
//!   [`SwapTunables::backlog_flag`] + [`SwapTunables::pops_per_callback`].
//! - Shared vs exclusive budget scope — hand the same `&mut SwapBudget` to
//!   several drains (shared) or to one (exclusive, provably identical to a
//!   private budget).
//! - Generation stamping — optional [`RtSwapHandler::current_generation`] /
//!   [`RtSwapHandler::generation_of`] staleness guard.
//! - Sequence acknowledgment — optional [`SwapRing::advance_resolved`] /
//!   [`SwapRing::rollback_last_pop`] hooks keep the ack gapless across
//!   deferrals; plain rings use the no-op defaults.
//!
//! # Invariants
//!
//! - **Zero heap drop on the RT thread.** Every retired, superseded, or stale
//!   resource reaches the GC cascade via [`GcSink`]; a payload parked in the
//!   deferred slot stays owned by the slot.
//! - **Latest-wins per coalesce key, never across keys.** Non-coalescible
//!   payloads ([`RtSwapHandler::coalesce_key`] = `None`, e.g. atomic restore
//!   transactions) are never superseded and apply in FIFO order.
//! - **At most `swaps_per_callback` structural applies per callback**, shared
//!   across all drains that receive the same `&mut SwapBudget`.
//! - **An ack never covers a parked command** (sequence rings: the pop of a
//!   parked payload is rolled back; the deferred slot consumes its sequence
//!   slot exactly once on resolution).
//! - **No allocation, no lock, no `log::*`, no `unwrap`/`expect`** on the RT
//!   path; flags and counters use the existing [`RtStatusFlags`] ordering
//!   helpers.
//! - **No dynamic dispatch on the hot path**: all generic parameters
//!   monomorphize; the drain inlines and handler hooks are `#[cold]`.
//!
//! # Host agnosticism
//!
//! This module is part of the engine's public RT-safety fabric. It knows
//! nothing about any host technology: a consumer composes one [`RtSwapDrain`]
//! per command channel, implements [`RtSwapHandler`] for its payload family,
//! and composes the shared [`SwapBudget`] across the drains of one audio
//! callback.

use crate::common::spsc::{GcItem, GcOverflowBuffer, RtStatusFlags, gc_cascade};
use crate::common::spsc::{
    RT_STATUS_PARAM_QUEUE_BACKLOG, RT_STATUS_SPSC_DRAIN_TRUNCATED, RT_STATUS_STRUCTURAL_DEFERRED,
    RT_STATUS_STRUCTURAL_SUPERSEDED,
};
use core::sync::atomic::Ordering;
use rtrb::Consumer;
use std::sync::atomic::AtomicBool;

/// Per-drain configuration. Set once at construction; never mutated on the
/// RT path.
///
/// The values must reproduce each consumer's historical tuning exactly —
/// pop caps and budgets are never averaged or unified across consumers.
#[derive(Clone, Copy, Debug)]
pub struct SwapTunables {
    /// Maximum payloads popped from the ring per callback (coalescing
    /// window). Should be ≥ ring capacity so the window covers a full queue
    /// plus producer refills.
    pub pops_per_callback: usize,
    /// Maximum structural applies per callback for the drain(s) sharing the
    /// budget. The allowance lives in a [`SwapBudget`] so several drains can
    /// share one per-callback budget.
    pub swaps_per_callback: usize,
    /// Raise `RT_STATUS_PARAM_QUEUE_BACKLOG` when the ring is still non-empty
    /// after the drain (scalar-parameter-style telemetry). Structural-only
    /// drains typically set `false` and flag `RT_STATUS_SPSC_DRAIN_TRUNCATED`
    /// instead when the pop cap is reached with a non-empty ring.
    pub backlog_flag: bool,
}

/// Shared per-callback structural budget.
///
/// One instance per audio callback; every drain of that callback receives
/// `&mut SwapBudget`, reproducing the shared-budget semantics (an exclusive
/// budget is the single-participant case). Created at callback start from the
/// drain's [`SwapTunables`]; only the RT thread touches it, so plain fields
/// need no synchronization.
#[derive(Clone, Copy, Debug)]
pub struct SwapBudget {
    applied: usize,
    limit: usize,
}

impl SwapBudget {
    /// Creates a fresh callback budget allowing `swaps_per_callback`
    /// structural applies.
    #[inline(always)]
    pub fn new(swaps_per_callback: usize) -> Self {
        Self {
            applied: 0,
            limit: swaps_per_callback,
        }
    }

    /// Whether another structural apply fits in this callback's budget.
    #[inline(always)]
    pub fn can_apply(&self) -> bool {
        self.applied < self.limit
    }

    /// Number of structural applies already consumed in this callback window.
    ///
    /// Telemetry/accounting surface (e.g. harness `structural_applied`).
    #[inline(always)]
    pub const fn used(&self) -> usize {
        self.applied
    }

    /// Consumes one structural apply from the budget.
    ///
    /// Must only be called after [`can_apply`](Self::can_apply) returned
    /// `true`; the drain never over-consumes.
    #[inline(always)]
    pub fn consume(&mut self) {
        self.applied += 1;
    }

    /// Number of structural applies already consumed this callback.
    #[inline(always)]
    pub fn applied(&self) -> usize {
        self.applied
    }

    /// Total structural applies allowed this callback.
    #[inline(always)]
    pub fn limit(&self) -> usize {
        self.limit
    }
}

/// Bundled GC-cascade dependencies for RT-safe retirement of heap resources.
///
/// Wraps exactly the state [`gc_cascade`] already consumes (SPSC producer,
/// 16-slot parking lot, overflow buffer, status flags) and adds the optional
/// parking-lot dirty latch: stored with `Release` before each cascade so
/// off-RT housekeeping can skip an empty parking-lot sweep. Adds zero
/// synchronization of its own.
pub struct GcSink<'a> {
    /// GC producer: retired items are dropped by the off-RT drain.
    pub producer: &'a mut rtrb::Producer<GcItem>,
    /// 16-slot RT parking lot shared across all GC producers of the callback.
    pub parking_lot: &'a mut [Option<GcItem>; 16],
    /// Last-resort overwrite buffer for GC items.
    pub overflow: &'a GcOverflowBuffer,
    /// Status flags receiving `RT_STATUS_GC_TIER3` / `RT_STATUS_GC_OVERFLOW`.
    pub rt_status: &'a RtStatusFlags,
    /// Optional dirty latch, stored with `Release` before each cascade.
    /// `None` keeps the unconditional off-RT parking-lot sweep.
    pub parking_lot_dirty: Option<&'a AtomicBool>,
}

impl GcSink<'_> {
    /// Retires one heap resource through the 3-tier GC cascade
    /// (SPSC ring → parking lot → overflow buffer). Never drops on the RT
    /// thread.
    #[inline(always)]
    pub fn retire(&mut self, item: GcItem) {
        if let Some(dirty) = self.parking_lot_dirty {
            dirty.store(true, Ordering::Release);
        }
        gc_cascade(
            Some(item),
            self.producer,
            self.parking_lot,
            self.overflow,
            self.rt_status,
        );
    }
}

/// Ring-consumer surface the drain operates on.
///
/// Implemented for [`rtrb::Consumer<Box<P>>`] for any payload `P`.
/// Integrations whose command ring carries sequence acknowledgment wrap the
/// consumer and implement the two bookkeeping hooks to keep the ack gapless
/// across deferrals; plain rings use the no-op defaults.
pub trait SwapRing {
    /// Payload type carried by the ring (heap-boxed on the RT side).
    type Payload;

    /// Pops the next payload, taking ownership (RT side).
    fn pop(&mut self) -> Option<Box<Self::Payload>>;

    /// Borrows the ring head without consuming it (classification probe).
    fn peek(&self) -> Option<&Self::Payload>;

    /// Whether the ring currently holds no payloads.
    fn is_empty(&self) -> bool;

    /// Occupancy of the ring (number of queued payloads).
    ///
    /// Observability surface for harnesses and telemetry drain accounting —
    /// lock-free and allocation-free, but never called from the hot path.
    fn occupied(&self) -> usize;

    /// A payload parked in the deferred slot was resolved (applied or
    /// discarded) without a ring pop; consumes its sequence slot. Default:
    /// no-op (plain rings carry no acknowledgment).
    fn advance_resolved(&mut self) {}

    /// The most recent pop was parked unapplied; its slot is re-resolved by
    /// the next callback's Phase 0. Rewinds the sequence so the ack never
    /// covers a parked command. Default: no-op.
    fn rollback_last_pop(&mut self) {}
}

impl<P> SwapRing for Consumer<Box<P>> {
    type Payload = P;

    #[inline(always)]
    fn pop(&mut self) -> Option<Box<P>> {
        rtrb::Consumer::pop(self).ok()
    }

    #[inline(always)]
    fn peek(&self) -> Option<&P> {
        rtrb::Consumer::peek(self).ok().map(|boxed| &**boxed)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        rtrb::Consumer::is_empty(self)
    }

    #[inline(always)]
    fn occupied(&self) -> usize {
        rtrb::Consumer::slots(self)
    }
}

/// Policy and hooks for one structural payload family.
///
/// Every method runs on the RT thread inside the drain; `install`/`discard`
/// are rare structural events and must be `#[cold]` in implementations.
/// Implementations must never allocate, lock, or panic.
pub trait RtSwapHandler {
    /// Payload type matched with the drained ring.
    type Payload;

    /// Whether the payload consumes the structural budget. `false` (light
    /// scalar payloads) ⇒ applied inline during the drain, never parked,
    /// never coalesced by the scheduler.
    fn is_structural(&self, payload: &Self::Payload) -> bool;

    /// Latest-wins identity. Payloads sharing a key supersede each other;
    /// `None` ⇒ never superseded and never supersedes (atomic restore
    /// transactions, all-or-nothing by contract).
    fn coalesce_key(&self, payload: &Self::Payload) -> Option<u64>;

    /// Generation the RT side currently expects, for the staleness guard.
    /// `None` ⇒ no filtering (mixed command rings that rely on FIFO + kind
    /// coalescing).
    fn current_generation(&self) -> Option<u64> {
        None
    }

    /// Generation stamp carried by `payload`. `None` ⇒ no filtering for that
    /// payload.
    fn generation_of(&self, payload: &Self::Payload) -> Option<u64> {
        let _ = payload;
        None
    }

    /// Applies the payload; replaced resources go to `gc`. `#[cold]` in
    /// implementations.
    fn install(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);

    /// Decomposes a never-applied payload (superseded or stale) into GC items
    /// through `gc`. `#[cold]` in implementations.
    fn discard(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);

    /// End-of-drain flush (e.g. apply consumer-side pending scalar locals,
    /// latest-wins). Default: no-op.
    fn after_drain(&mut self, gc: &mut GcSink<'_>) {
        let _ = gc;
    }
}

/// The generic RT structural-swap drain (Phases 0–2).
///
/// Owns the ring consumer end and the single deferred slot. A consumer
/// instantiates one drain per command channel, implements one handler per
/// payload family, and composes all drains of a callback under a shared
/// [`SwapBudget`] and a [`GcSink`] scope.
pub struct RtSwapDrain<R: SwapRing> {
    ring: R,
    deferred: Option<Box<R::Payload>>,
    tunables: SwapTunables,
}

impl<R: SwapRing> RtSwapDrain<R> {
    /// Creates a drain owning `ring` under `tunables`.
    #[inline(always)]
    pub fn new(ring: R, tunables: SwapTunables) -> Self {
        Self {
            ring,
            deferred: None,
            tunables,
        }
    }

    /// Whether a payload is parked in the deferred slot.
    #[inline(always)]
    pub fn has_deferred(&self) -> bool {
        self.deferred.is_some()
    }

    /// Whether the ring currently holds no payloads.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Occupancy of the owned ring (number of queued payloads).
    ///
    /// Harness/telemetry drain accounting — not part of the hot path.
    #[inline(always)]
    pub fn ring_occupied(&self) -> usize {
        self.ring.occupied()
    }

    /// Runs one callback drain (Phases 0–2 + end-of-drain flush).
    ///
    /// Sets `RT_STATUS_STRUCTURAL_DEFERRED` / `RT_STATUS_STRUCTURAL_SUPERSEDED`
    /// (+ their monotonic counters), honors `backlog_flag`, and stops draining
    /// on the pop cap (flagging `RT_STATUS_SPSC_DRAIN_TRUNCATED` when the cap
    /// is reached with a non-empty ring and the backlog flag is off).
    #[inline(always)]
    pub fn drain<H>(&mut self, handler: &mut H, budget: &mut SwapBudget, gc: &mut GcSink<'_>)
    where
        H: RtSwapHandler<Payload = R::Payload>,
    {
        // Phase 0 — resolve a payload deferred by a previous callback.
        self.resolve_deferred(handler, budget, gc);
        // Phase 1 — bounded drain with latest-wins coalescing.
        let (pops, candidate) = self.bounded_drain(handler, budget, gc);
        self.flag_ring_backlog(pops, gc);
        // Phase 2 — budgeted apply or park of the window candidate.
        self.resolve_candidate(handler, budget, gc, candidate);
        handler.after_drain(gc);
    }

    /// Phase 0 — resolve a payload parked by a previous callback.
    ///
    /// The parked payload is causally before everything still in the ring, so
    /// it resolves first: discarded when stale, discarded when a newer
    /// same-key payload sits at the head (latest-wins), otherwise installed
    /// under the budget or re-parked when the budget was already consumed by
    /// an earlier drain of this callback.
    #[inline(always)]
    fn resolve_deferred<H>(&mut self, handler: &mut H, budget: &mut SwapBudget, gc: &mut GcSink<'_>)
    where
        H: RtSwapHandler<Payload = R::Payload>,
    {
        let Some(p) = self.deferred.take() else {
            return;
        };
        if is_stale(handler, &p) {
            // Stale while parked: discard without supersession — the build
            // simply no longer matches the current request.
            handler.discard(p, gc);
            self.ring.advance_resolved();
            return;
        }
        if is_superseded_by_head(handler, &p, &self.ring) {
            handler.discard(p, gc);
            self.ring.advance_resolved();
            flag_superseded(gc.rt_status);
            return;
        }
        if budget.can_apply() {
            handler.install(p, gc);
            budget.consume();
            self.ring.advance_resolved();
        } else {
            // Budget exhausted and nothing newer queued: keep parked for the
            // next callback (causally first there as well).
            self.deferred = Some(p);
            flag_deferred(gc.rt_status);
        }
    }

    /// Phase 1 — bounded drain with latest-wins coalescing.
    ///
    /// Structural payloads collapse per coalesce key into a single candidate;
    /// a different-key structural flush-installs the pending candidate first
    /// (FIFO across keys); light scalars install inline. When the structural
    /// budget is exhausted, the structural head stays owned by the ring and
    /// the drain stops — everything behind it is causally after it, so order
    /// is preserved without detaching the payload (which also avoids any
    /// deferred-slot contention under shared budgets).
    #[inline(always)]
    fn bounded_drain<H>(
        &mut self,
        handler: &mut H,
        budget: &mut SwapBudget,
        gc: &mut GcSink<'_>,
    ) -> (usize, Option<Box<R::Payload>>)
    where
        H: RtSwapHandler<Payload = R::Payload>,
    {
        let mut candidate: Option<Box<R::Payload>> = None;
        let mut pops = 0usize;
        while pops < self.tunables.pops_per_callback {
            // Classify the head without detaching it from the ring: an
            // unresolvable structural payload remains queued (zero-loss,
            // FIFO) instead of being detached into an already-occupied
            // deferred slot.
            let Some(head) = self.ring.peek() else {
                break;
            };
            if is_stale(handler, head) {
                // Superseded rebuild: the producer moved on while this
                // envelope was in flight. Discard without installing.
                if let Some(p) = self.ring.pop() {
                    handler.discard(p, gc);
                }
                pops += 1;
                continue;
            }
            if !handler.is_structural(head) {
                // Light scalar: applies inline, never parked, never coalesced
                // by the scheduler (handler-side coalescing owns that policy).
                if let Some(p) = self.ring.pop() {
                    handler.install(p, gc);
                }
                pops += 1;
                continue;
            }
            let head_key = handler.coalesce_key(head);
            let same_key_pending = head_key.is_some()
                && candidate
                    .as_deref()
                    .is_some_and(|c| handler.coalesce_key(c) == head_key);
            if same_key_pending && let Some(p) = self.ring.pop() {
                // Same key as the pending candidate: free coalescing (no
                // budget, no slot) — the older candidate is obsolete.
                if let Some(older) = candidate.replace(p) {
                    handler.discard(older, gc);
                    flag_superseded(gc.rt_status);
                }
                pops += 1;
                continue;
            }
            if !budget.can_apply() {
                // Budget exhausted with a newer different-key structural at
                // the head: it is deferred (stays queued, FIFO intact) and
                // the drain stops.
                flag_deferred(gc.rt_status);
                break;
            }
            // Budget available: pop and make this payload the single
            // candidate, flush-installing the pending different-key candidate
            // so at most one structural candidate is ever held.
            if let Some(p) = self.ring.pop()
                && let Some(older) = candidate.replace(p)
            {
                handler.install(older, gc);
                budget.consume();
            }
            pops += 1;
        }
        (pops, candidate)
    }

    /// Phase 2 — resolve the window candidate under the budget; a
    /// budget-exhausted candidate parks in the deferred slot for the next
    /// callback (latest-wins against anything queued behind it).
    #[inline(always)]
    fn resolve_candidate<H>(
        &mut self,
        handler: &mut H,
        budget: &mut SwapBudget,
        gc: &mut GcSink<'_>,
        candidate: Option<Box<R::Payload>>,
    ) where
        H: RtSwapHandler<Payload = R::Payload>,
    {
        let Some(p) = candidate else {
            return;
        };
        if budget.can_apply() {
            handler.install(p, gc);
            budget.consume();
        } else {
            // The slot is free here by construction: a candidate exists only
            // if a pop was budget-permitted this callback, which excludes the
            // Phase 0 re-park path (the only occupier).
            debug_assert!(
                self.deferred.is_none(),
                "deferred slot must be free when parking a window candidate"
            );
            self.deferred = Some(p);
            // The candidate's pop must not be covered by the ack: rewind the
            // sequence so Phase 0 consumes the slot exactly once on resolve.
            self.ring.rollback_last_pop();
            flag_deferred(gc.rt_status);
        }
    }

    /// End-of-drain telemetry: backlog flag when the ring still holds
    /// payloads (scalar-parameter style), or truncation flag when the pop cap
    /// was hit with payloads remaining (structural style).
    #[inline(always)]
    fn flag_ring_backlog(&self, pops: usize, gc: &GcSink<'_>) {
        if self.ring.is_empty() {
            return;
        }
        if self.tunables.backlog_flag {
            gc.rt_status.set_flag(RT_STATUS_PARAM_QUEUE_BACKLOG);
        } else if pops >= self.tunables.pops_per_callback {
            gc.rt_status.set_flag(RT_STATUS_SPSC_DRAIN_TRUNCATED);
        }
    }
}

/// Whether `payload` is stale: its generation no longer matches the current
/// request. Payloads without generation stamping are never stale.
#[inline(always)]
fn is_stale<H: RtSwapHandler>(handler: &H, payload: &H::Payload) -> bool {
    match (handler.current_generation(), handler.generation_of(payload)) {
        (Some(current), Some(stamped)) => stamped != current,
        _ => false,
    }
}

/// Whether a newer same-key payload sits at the ring head (latest-wins
/// supersession probe). Payloads without a coalesce key are never superseded
/// and never supersede.
#[inline(always)]
fn is_superseded_by_head<H, R>(handler: &H, payload: &H::Payload, ring: &R) -> bool
where
    H: RtSwapHandler,
    R: SwapRing<Payload = H::Payload>,
{
    let Some(key) = handler.coalesce_key(payload) else {
        return false;
    };
    ring.peek()
        .is_some_and(|head| handler.coalesce_key(head) == Some(key))
}

/// Raises `RT_STATUS_STRUCTURAL_DEFERRED` (+ monotonic counter, `Relaxed`).
#[inline(always)]
fn flag_deferred(rt_status: &RtStatusFlags) {
    rt_status.set_flag(RT_STATUS_STRUCTURAL_DEFERRED);
    rt_status
        .structural_deferred_total
        .fetch_add(1, Ordering::Relaxed);
}

/// Raises `RT_STATUS_STRUCTURAL_SUPERSEDED` (+ monotonic counter, `Relaxed`).
#[inline(always)]
fn flag_superseded(rt_status: &RtStatusFlags) {
    rt_status.set_flag(RT_STATUS_STRUCTURAL_SUPERSEDED);
    rt_status
        .structural_superseded_total
        .fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
#[path = "swap_test.rs"]
mod swap_test;
