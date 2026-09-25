<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Design Spike: Generic RT Structural-Swap Scheduler & Partitioned-Convolution Driver

> **Status:** design proposal (investigation spike). **No production code was changed** by this
> document. Implementation is gated on an explicit project-owner decision.
> **Origin:** cross-consumer audit finding — both first-party integrations independently
> reimplement (~600 lines total) the same real-time structural-swap protocol and have already
> diverged in tunings; the cab-sim algorithmic latency is currently pinned to the host block size
> in both, although the adapter contract already permits sub-partition driving.
> **Companion documents:** [architecture.md](architecture.md) (RT-safety policy, SPSC/GC mesh),
> [testing.md](testing.md) (oracles, tiers, gates), [functional-tests.md](functional-tests.md).

---

## 0. Scope and problem statement

### 0.1 The structural-swap protocol (today)

Off-RT resource swaps (model, resampler, streaming adapter, cab-sim IR, oversampling engines,
state restore) follow one protocol on every consumer:

1. **Phase 0 — deferred resolution.** A payload parked by the previous callback is resolved at
   the top of the current one: applied first (it is causally before everything still in the
   ring), or superseded/discarded if a newer equivalent command is queued or its generation went
   stale.
2. **Phase 1 — bounded drain with latest-wins coalescing.** The RT thread pops payloads under a
   per-callback pop cap; equivalent payloads collapse to the latest one, intermediate resources
   are retired through the GC cascade, never dropped on the audio thread.
3. **Phase 2 — budgeted apply or park.** At most one structural apply per callback (shared
   across all drains of that callback); the excess parks in a single deferred slot.

All consumers build this on the same engine primitives (`common::spsc` rings, `gc_cascade`,
`RT_STATUS_*` flags/counters), yet each reimplements ~300 lines of drain logic, and the
implementations have diverged:

| Aspect | Plugin-style integration (single mixed command ring, sequence-acked) | Standalone-host-style integration (dedicated per-resource rings, generation-stamped) |
| --- | --- | --- |
| Channel topology | one mixed ring carrying scalars + all structural kinds; one dedicated slimmable ring | four dedicated single-kind rings (resampler, cab-sim, slimmable, oversample) + one mixed scalar/structural ring |
| Structural budget | 1 per callback, exclusive to the mixed ring | 1 per callback, **shared** across all five drains |
| Pop cap | 64 payloads per callback (drain truncation flag) | 8 pops per dedicated drain; 16 for the scalar budget with a backlog flag |
| Coalescing key | payload kind (kinds classified coalescible vs not, e.g. restore transactions never supersede) | implicit (single-kind ring ⇒ latest wins) + generation stamping |
| Acknowledgment | monotonic sequence numbers with rollback/advance hooks (an ack must never cover a parked command) | none |
| Scalar coalescing | producer-side coalescing buffer (fused snapshot payload) | consumer-side pending locals (latest-wins inside the budget) |

The observable guarantees are identical (≤ 1 structural apply per callback, latest-wins, zero
heap drop on the RT thread), but every semantic knob lives in duplicated consumer code. Measured
cost on both sides is tiny (drain p99 ≈ 0.94 µs under full simultaneous saturation across five
drains, 0.28 % of a 333 µs deadline at quantum 16) — this spike is therefore **not** about
speeding up the drains; it is about deleting ~600 duplicated lines of RT-critical logic that has
already drifted and must be maintained under heap-audit/RT-deadline contracts in two places.

### 0.2 The partitioned-convolution gap (today)

`CabSimAdapter` enforces `input.len() == output.len() <= partition_size` per call and buffers
internally, so multiple calls per host block are legal — but no consumer drives it that way: both
build the adapter with `partition_size ==` host maximum block size, which pins the algorithmic
latency (= `partition_size`, see `CabSimAdapter::latency_samples`) to the host quantum and forces
a cab-sim rebuild on every quantum renegotiation. Running `partition_size < host_block` trades IR
latency against CPU (more, smaller UPOLS partitions), but a naive N-calls-per-block loop
multiplies the internal `accumulate`/`deliver` copies and per-call overhead by N. A driver that
batches those copies internally removes both blockers.

---

## Part B — Design: generic structural-swap scheduler (engine)

### B.1 Goals

- **G1** One canonical drain implementation in the engine, parametrized by payload type, built on
  the existing primitives (`common::spsc` rings, `gc_cascade`, `GcOverflowBuffer`, `RtStatusFlags`)
  — no new synchronization, no allocation, no locks on the RT path.
- **G2** Every existing tuning stays an explicit configuration value. No constant is averaged or
  silently unified: pop caps (8 vs 64), scalar budgets (16 vs none), shared vs exclusive budget
  scope, generation stamping, sequence acknowledgment.
- **G3** Bit-for-bit identical observable behavior per consumer after migration, verified by the
  existing test suites (command-budget tests, heap-audit, RT-deadline) in each consumer repo.
- **G4** Incremental migration: engine first (dead code until adopted), then one consumer drain at
  a time. The engine never references any host technology or consumer.

### B.2 Non-goals

- No change to producer-side protocols (request flags, generation bumping, coalescing buffers on
  the main thread). The scheduler covers the **RT drain side** only.
- No new GC item variants, no new status bits (existing `RT_STATUS_STRUCTURAL_DEFERRED`,
  `RT_STATUS_STRUCTURAL_SUPERSEDED`, `RT_STATUS_PARAM_QUEUE_BACKLOG`,
  `RT_STATUS_SPSC_DRAIN_TRUNCATED` and their counters are reused as-is).
- No dynamic dispatch on the hot path: all generic parameters are compile-time (monomorphization);
  the drain is `#[inline(always)]` with `#[cold]` install/discard handlers, exactly like the
  implementations it replaces.

### B.3 API sketch

Proposed module: `src/common/spsc/swap.rs` (re-exported from `common::spsc`).

```rust
/// Per-channel drain configuration. Set once at construction; never mutated on the RT path.
#[derive(Clone, Copy, Debug)]
pub struct SwapTunables {
    /// Maximum payloads popped from the ring per callback (coalescing window).
    /// Should be >= ring capacity so the window covers a full queue plus producer refills.
    pub pops_per_callback: usize,
    /// Maximum structural applies per callback for the drain(s) sharing the budget.
    pub swaps_per_callback: usize,
    /// Raise `RT_STATUS_PARAM_QUEUE_BACKLOG` when the ring is still non-empty after the drain
    /// (scalar-parameter-style telemetry). Structural-only drains typically set `false`.
    pub backlog_flag: bool,
}

/// Shared per-callback structural budget. One instance per audio callback; every drain of
/// that callback receives `&mut SwapBudget`, reproducing the shared-budget semantics.
pub struct SwapBudget { /* applied: usize, limit: usize */ }

impl SwapBudget {
    /// `new(swaps_per_callback)` — cloned from the drain's tunables at callback start.
    #[inline(always)] pub fn can_apply(&self) -> bool;
    #[inline(always)] pub fn consume(&mut self);
}

/// Bundled GC-cascade dependencies for RT-safe retirement. Wraps exactly the state
/// `gc_cascade` already consumes; adds zero synchronization.
pub struct GcSink<'a> {
    pub producer: &'a mut Producer<GcItem>,
    pub parking_lot: &'a mut [Option<GcItem>; 16],
    pub overflow: &'a GcOverflowBuffer,
    pub rt_status: &'a RtStatusFlags,
    /// Optional dirty latch, stored with `Release` before each cascade so off-RT
    /// housekeeping can skip an empty parking-lot sweep.
    pub parking_lot_dirty: Option<&'a AtomicBool>,
}

impl GcSink<'_> {
    /// Retires one heap resource through the 3-tier cascade (ring → parking lot → overflow).
    /// Never drops on the RT thread.
    #[inline(always)] pub fn retire(&mut self, item: GcItem);
}

/// Ring-consumer surface the drain operates on. Implemented for `rtrb::Consumer<Box<P>>`.
/// Integrations whose command ring carries sequence acknowledgment implement the two
/// bookkeeping hooks to keep the ack gapless across deferrals; plain rings use the no-op
/// defaults.
pub trait SwapRing {
    type Payload;
    fn pop(&mut self) -> Option<Box<Self::Payload>>;
    fn peek(&self) -> Option<&Self::Payload>;
    fn is_empty(&self) -> bool;
    /// A popped payload was resolved (applied or discarded). Default: no-op.
    fn advance_resolved(&mut self) {}
    /// The most recent pop was parked unapplied; its slot is re-resolved next callback.
    /// Default: no-op.
    fn rollback_last_pop(&mut self) {}
}

/// Policy and hooks for one structural payload family.
pub trait RtSwapHandler {
    type Payload;
    /// Whether the payload consumes the structural budget. `false` (light scalar payloads)
    /// ⇒ applied inline during the drain, never parked, never coalesced by the scheduler.
    fn is_structural(&self, payload: &Self::Payload) -> bool;
    /// Latest-wins identity. `None` ⇒ never superseded and never supersedes (atomic
    /// restore transactions).
    fn coalesce_key(&self, payload: &Self::Payload) -> Option<u64>;
    /// Generation-stamped staleness guard. `None` ⇒ no filtering (mixed command rings
    /// that rely on FIFO + kind coalescing).
    fn current_generation(&self) -> Option<u64> { None }
    fn generation_of(&self, payload: &Self::Payload) -> Option<u64> { None }
    /// Applies the payload; replaced resources go to `gc`. `#[cold]` in implementations.
    fn install(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);
    /// Decomposes a never-applied payload (superseded/stale) into GC items. `#[cold]`.
    fn discard(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);
    /// End-of-drain flush (e.g. apply consumer-side pending scalar locals, latest-wins).
    /// Default: no-op.
    fn after_drain(&mut self, gc: &mut GcSink<'_>) { let _ = gc; }
}

/// The generic RT drain (phases 0–2). Owns the ring end and the single deferred slot.
pub struct RtSwapDrain<R: SwapRing> {
    ring: R,
    deferred: Option<Box<R::Payload>>,
    tunables: SwapTunables,
}

impl<R: SwapRing> RtSwapDrain<R> {
    pub fn new(ring: R, tunables: SwapTunables) -> Self;
    /// Runs one callback drain. Sets `RT_STATUS_STRUCTURAL_DEFERRED` / `_SUPERSEDED`
    /// (+ monotonic counters), honors `backlog_flag`, and stops draining on pop cap
    /// (flagging `RT_STATUS_SPSC_DRAIN_TRUNCATED` when the cap is reached with a
    /// non-empty ring and the backlog flag is off).
    pub fn drain<H>(&mut self, handler: &mut H, budget: &mut SwapBudget, gc: &mut GcSink<'_>)
    where
        H: RtSwapHandler<Payload = R::Payload>;
}
```

### B.4 Canonical algorithm

```
Phase 0 — resolve the deferred slot (if occupied):
  p = deferred.take()
  if stale(p)                                  -> handler.discard(p); ring.advance_resolved()
  else if head.peek() has same coalesce_key    -> handler.discard(p); ring.advance_resolved()
                                                  flag + count(SUPERSEDED)
  else if budget.can_apply()                   -> handler.install(p); budget.consume()
                                                  ring.advance_resolved()
  else                                         -> re-park; flag + count(DEFERRED)

Phase 1 — bounded drain (pops < tunables.pops_per_callback):
  candidate: Option<Box<P>> = None
  loop:
    p = ring.pop() or break
    if stale(p)                     -> handler.discard(p); continue          // superseded rebuild
    if !handler.is_structural(p)    -> handler.install(p); continue          // light scalar: inline
    if budget.cannot_apply()        -> park(p); ring.rollback_last_pop();
                                       flag + count(DEFERRED); break          // FIFO preserved
    if key(p) is Some and candidate has same key:
        handler.discard(candidate.take()); flag + count(SUPERSEDED)
    candidate = Some(p)
  after loop: if pop cap reached and !ring.is_empty():
      backlog_flag ? flag(PARAM_QUEUE_BACKLOG) : flag(SPSC_DRAIN_TRUNCATED)

Phase 2 — resolve the window candidate:
  if let Some(p) = candidate:
    if budget.can_apply()           -> handler.install(p); budget.consume()
    else if deferred slot free      -> park(p); flag + count(DEFERRED)
    else                            -> handler.discard(parked older);
                                       flag + count(SUPERSEDED); park(p);
                                       flag + count(DEFERRED)

handler.after_drain(gc)   // flush pending scalar locals (latest-wins), etc.
```

Invariants (all inherited from the existing implementations — none is new):

- **Zero heap drop on the RT thread.** Every retired, superseded, or stale resource reaches
  `gc_cascade` via `GcSink`; a payload parked in the deferred slot stays owned by the slot.
- **Latest-wins per coalesce key**, never across keys; non-coalescible payloads (restore
  transactions) are applied in FIFO order and are never superseded.
- **At most `swaps_per_callback` structural applies per callback**, shared across all drains that
  receive the same `&mut SwapBudget`.
- **An ack never covers a parked command** (sequence rings: `rollback_last_pop` on park,
  `advance_resolved` on apply/discard; Phase 0 resolution of a parked slot calls
  `advance_resolved` exactly once).
- **No allocation, no lock, no `log::*`, no `unwrap/expect`** on the RT path (F-10 policy);
  flags/counters use the existing `RtStatusFlags` ordering helpers.

### B.5 Mapping of the two reference integrations (no semantic change)

| Drain | `SwapTunables` | Handler notes |
| --- | --- | --- |
| Standalone: dedicated resource rings | `{ pops: 8, swaps: 1 (shared), backlog: false }` | `coalesce_key = Some(constant)`; generation-stamped stale guard via `current_generation`/`generation_of` |
| Standalone: mixed scalar/structural ring | `{ pops: 16, swaps: 1 (shared), backlog: true }` | scalars are light (`is_structural = false`), pending locals + LUT work in the handler, flushed in `after_drain` |
| Plugin: mixed command ring | `{ pops: 64, swaps: 1, backlog: false }` (truncation flag on cap) | kind-based `coalesce_key`; restore transactions return `None`; seq-ack hooks implemented by the ring wrapper |
| Plugin: slimmable ring | `{ pops: >= 8, swaps: 1 (shared) }` | migration **normalizes** this drain to the budgeted, latest-wins protocol (today it is unbounded — see B.7) |

### B.6 Minimal test plan (engine, `src/common/spsc/swap_test.rs`)

All tests use a fake payload (`Box<u64>`) with a counting `GcSink` — no audio, no allocation:

1. `budget_one_apply_per_callback` — N queued structural payloads, `swaps_per_callback = 1`:
   exactly one install, one park, flags/counter set.
2. `latest_wins_coalescing_discards_to_gc` — three same-key payloads in one window: only the
   last installs; two discarded; `SUPERSEDED` counter = 2.
3. `non_coalescible_never_superseded` — restore-style payload (`coalesce_key = None`) survives a
   newer different-key payload and applies in FIFO order.
4. `stale_generation_discarded_without_install` — payloads stamped with an old generation are
   discarded (no flag on Phase 0 discard; `SUPERSEDED` counted in-window).
5. `deferred_resolves_next_callback_before_newer_ring_payloads` — causality of Phase 0.
6. `park_rollback_and_resolve_keeps_ack_gapless` — seq-hook ring records: rollback on park,
   advance on resolve; ack sequence monotonic without gaps.
7. `pop_cap_truncates_and_flags` — cap reached with non-empty ring: truncation/backlog flag per
   tunables.
8. `shared_budget_across_drains` — two drains, one `SwapBudget`, limit 1: second drain parks.
9. `zero_alloc_retirement` — under the heap-audit feature, a full saturation soak performs zero
   RT-thread allocations (counts through the counting allocator).
10. Property test (quick lane): randomized producer/consumer interleavings — no payload is ever
    dropped on the RT thread; final installed state equals the latest delivered per key.

### B.7 Migration plan

| Step | Scope | Validation |
| --- | --- | --- |
| M1 | Land `swap.rs` + tests in the engine (dead code; no consumer touched) | `utils/tests-quick.sh`; new unit tests; no baseline impact (no hot-path change) |
| M2 | Migrate the standalone-host dedicated drains one at a time (resampler → cab-sim → slimmable → oversample → mixed ring) | per-drain consumer test suite; heap-audit; RT-deadline; tunables preserved exactly |
| M3 | Migrate the plugin mixed ring + slimmable ring | consumer budget/burst tests (p99 contract); heap-audit; RT-deadline; ack-gapless test |
| M4 | Delete the superseded consumer drain code (~600 lines across consumers) | full consumer `cargo test`; release performance certification gate |

Normalizations agreed during the spike (behavior-preserving or strictly closer to the protocol):

- The plugin-style slimmable drain (currently unbounded, installs every current-generation
  rebuild in one callback) becomes budgeted and latest-wins like its standalone counterpart.
  Rebuild deliveries are rare one-at-a-time events; `pops_per_callback >= 8` keeps practical
  equivalence. **PO-visible behavior change: none** (at most one rebuild lands per callback —
  the excess resolves at the start of the next one).
- The standalone-style mixed-ring structural apply keeps its shared budget; the plugin-style
  mixed ring joins the shared-budget model with a single participant, which is provably the same
  semantics as today's exclusive budget.

### B.8 Risks

| Risk | Mitigation |
| --- | --- |
| Divergent phase-1 semantics (window-coalesce vs apply-until-budget) mask an ordering bug | Canonical algorithm is the **order-preserving** one (apply inline, park + stop on budget); window coalescing only collapses same-key payloads. Covered by tests 3, 5, 6. |
| Ack desynchronization during deferral | Sequence hooks are part of the trait contract; test 6 models deactivate/activate cycles; the consumer's existing ack-gapless test re-runs under M3. |
| Shared-budget starvation of a low-priority drain | Budget sharing is opt-in per callback composition (M2 keeps the standalone ordering: mixed ring first, dedicated drains after). Telemetry counters make any deferral visible. |
| Generic abstraction costs (monomorphization bloat, inlining misses) | Drain is generic over one payload type per instantiation — same monomorphization count as today's per-type functions. `#[inline(always)]` on the phase code, `#[cold]` on handler hooks; verified by objdump inspection in M1 per project convention. |

---

## Part C — Design: generic partitioned-convolution driver (engine)

### C.1 Motivation

- Cab-sim algorithmic latency equals `partition_size`; both reference integrations build the
  adapter with `partition_size ==` host maximum block, so IR latency tracks the host quantum
  (e.g. 256 samples ≈ 5.3 ms @ 48 kHz) and every quantum renegotiation triggers a full cab-sim
  rebuild.
- The adapter contract already permits multiple calls per block (`len <= partition_size`), but a
  consumer-side chunking loop would multiply the internal `accumulate`/`deliver` copies and
  per-call overhead by the number of sub-blocks.
- Goal: a **driver inside the engine** that accepts a host block of any size against a fixed
  partition, batching the FIFO mechanics internally, so consumers expose a simple
  *IR-latency vs CPU* knob without duplicating chunking, tail, or rearm logic.

### C.2 API sketch

Proposed additions to `src/dsp/cabsim/` (adapter-level; the pair wrapper mirrors it):

```rust
impl CabSimAdapter {
    /// Processes a host block of arbitrary size through the fixed-partition engine,
    /// internally batching the accumulate/deliver passes.
    ///
    /// # Constraints
    /// * `input_output.len() > 0` (empty slices remain the domain of `drain_tail`)
    /// * No partition cap on `input_output.len()` — the driver chunks internally.
    ///
    /// # Semantics (unchanged from the per-sub-block contract)
    /// * Output is causal; the first `partition_size - 1` samples of a fresh adapter
    ///   remain silent.
    /// * Bit-identical to driving `process_in_place` over `partition_size` sub-blocks:
    ///   same accumulate order, same partition MAC order, same delivery order.
    /// * `rearm_tail`/`drain_tail` semantics are unchanged; `rearm_tail` stays per host
    ///   block, never per internal chunk.
    ///
    /// # RT-Safety: zero-alloc, lock-free, never panics.
    pub fn process_block(&mut self, input_output: &mut [f32], rt_status: Option<&RtStatusFlags>);
}
```

### C.3 Batching mechanics

The block is consumed in a single FIFO sweep, not by re-entering `process_in_place` per chunk:

```
remaining = input_output.len()
while remaining >= partition:
    accumulate(partition samples)      // one copy per partition, same as today
    run_partitions()                   // UPOLS: 1 forward RFFT + P·N_p MACs + 1 inverse RFFT
    deliver(partition samples)         // one copy per partition, same as today
    remaining -= partition
tail: accumulate(rest); run_partitions(); deliver(rest)   // FIFO tail, identical to process_in_place
```

Cost profile versus the naive consumer-side loop over `process_in_place` sub-slices: identical
number of `accumulate`/`deliver` memcpys (they are per-partition in both), but the driver removes
the per-call contract checks, counter arithmetic, and call overhead (×N), and — more importantly
— it is the single place where chunking policy lives. The fixed-overhead saving per partition is
on the order of tens of nanoseconds; the point of the driver is **API ownership of the chunking
policy**, not a measurable kernel win. (The project is in the extreme-optimization stage: claims
here are deliberately conservative and bench-gated, see `benchmarks.md`.)

The consumer-visible consequences:

- `partition_size` becomes an **installation-time policy** (chosen when the IR is built), decoupled
  from the host quantum. Latency published via the existing `latency_samples()` channel semantics.
- Quantum renegotiations no longer force a cab-sim rebuild when only the block size changed
  (`RT_STATUS_NEEDS_CABSIM_REBUILD` remains for rate/IR changes and partition policy changes).
- The sub-block contract-violation path (`RT_STATUS_CABSIM_CONTRACT_VIOLATION`) is retained for
  direct adapter use; the driver never raises it (it chunks by construction).

### C.4 Latency vs CPU trade-off (selection guidance)

Per partition, UPOLS performs one forward RFFT of size `2P`, `P`-bin complex accumulation against
`N_p = IR_len / P` FDL partitions, and one inverse RFFT of size `2P`. Halving `P` halves the
algorithmic latency and doubles the number of partition events per second while keeping total MAC
work per second constant (`P × N_p / P = IR_len`) and reducing per-RFFT size. The remaining
trade-off is fixed per-event overhead (twiddle setup, FDL pointer walk, FIFO compaction) versus
latency — the classic UPOLS curve. The existing `cabsim_bench` suite (partition sweep) is the
sentinel for regressions; no new benchmark is required by this design beyond wiring the driver
into the existing group.

### C.5 Test plan (against the parity oracles)

1. **Bit-exact blocking sweep (stronger than ESR):** for one fixed IR, drive the adapter through
   the driver with `partition ∈ {32, 64, 128, 256}` × host block `∈ {16, 32, 64, 256, 333}` and
   assert **exact equality** with the current single-partition path output (the FIFO makes
   blocking-invariant output an exact identity, not a tolerance).
2. **End-to-end ESR gate** (acceptance): cab-sim stage with chunking 32→64 and 16→64
   (sub-block → partition), output within the established ESR threshold of the current path.
3. **Parity oracles unchanged:** `cpp_parity quick_parity` PASS and f64 oracle PASS — the driver
   must not alter arithmetic; it only reorders *when* FIFO copies happen, never the MAC order.
4. **Tail semantics:** silence → `drain_tail` ring-out produces identical samples under any
   block/partition combination; `rearm_tail` once per host block.
5. **Contract:** driver never raises `RT_STATUS_CABSIM_CONTRACT_VIOLATION`; direct adapter misuse
   still does (existing tests untouched).
6. **Heap-audit lane:** zero allocations on the RT thread across the sweep.

---

## Open decisions (gate before implementation)

1. **Approve the scheduler API surface (Part B)** — in particular the `SwapRing`/`RtSwapHandler`
   trait split and the shared `SwapBudget` model as the canonical semantics.
2. **Approve the slimmable normalization** (B.7): unbounded → budgeted/latest-wins (arguably a
   conformance fix; flagged here because it is the only observable behavior delta in the plan).
3. **Approve the driver API (Part C)** and confirm the default partition policy stays
   `partition_size ==` host maximum block until a consumer opts into sub-partition latency.
4. **Sequencing:** engine M1 may proceed immediately after approval; M2/M3 stay bound to the
   integration phases that already own consumer migrations.

---

## Summary

Both designs reuse the engine's existing RT-safety fabric (SPSC rings, GC cascade, status flags,
`#[cold]` applies) and remove ~600 lines of duplicated, already-divergent RT-critical logic from
consumers, at the cost of two new parametric API surfaces in the engine. Measured drain costs are
microseconds against millisecond deadlines; the risk is semantic, not performance — which is why
the migration plan is incremental (engine dead-code first, one drain per consumer at a time) and
gated on heap-audit/RT-deadline validation in each consumer.
