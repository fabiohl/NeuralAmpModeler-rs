<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Real-Time Structural-Swap Scheduler & Partitioned-Convolution Driver

This document describes two core architectural components of the `NeuralAmpModeler-rs` engine:

1. **Generic RT Structural-Swap Scheduler** ([`src/common/spsc/swap.rs`](../src/common/spsc/swap.rs)): coordinates safe, lock-free, zero-allocation installation of heavy heap resources on the audio thread.
2. **Partitioned-Convolution Driver** ([`src/dsp/cabsim/adapter.rs`](../src/dsp/cabsim/adapter.rs)): decouples the algorithmic impulse-response (IR) latency from the host block size, using a single FIFO sweep per block.

---

## 1. Overview & Core Problems

### 1.1 What is a Structural Swap?

In an audio engine, parameters change at different computational costs:

- **Scalar parameters** (e.g., input gain, output volume): Updated by changing a few numbers directly in memory.
- **Structural resources** (e.g., neural models, cab-sim impulse responses, resamplers, oversampling engines, complete preset state): Require pre-allocated heap objects (tens of kilobytes to several megabytes).

Replacing a structural resource during audio playback is a **structural swap**. In real-time audio it carries two strict requirements:

1. **Zero allocation and zero deallocation on the audio thread**: freeing memory (`drop`) invokes the system heap allocator, which can block on locks or kernel calls and cause audible glitches ("xruns"). Retired resources must be handed to an off-real-time garbage-collection (GC) thread.
2. **Deterministic execution time**: installing several large resources in one audio callback could exceed the deadline (for example, about 333 µs for a 16-sample buffer at 48 kHz). The engine must budget how many structural changes are allowed per callback.

The **RT Structural-Swap Scheduler** provides a single, canonical protocol for all resource swaps, independent of the host architecture.

### 1.2 What is the Partitioned-Convolution Problem?

Cabinet simulation uses **Uniformly Partitioned Overlap-Save (UPOLS)** convolution. In UPOLS, the impulse response is divided into small partitions of size $P$:

- Algorithmic latency is determined by the partition size ($P$ samples).
- If an adapter requires the host block size to equal the partition size, the latency is pinned to the host buffer size: a large host buffer (for example, 512 samples) forces high latency, and changing the host buffer size requires rebuilding the convolution engine.

The **Partitioned-Convolution Driver** removes that coupling by chunking each host block internally in a single FIFO sweep, so a fixed, low-latency partition works with any host buffer size.

---

## 2. Generic RT Structural-Swap Scheduler (`common::spsc::swap`)

### 2.1 The 3-Phase Drain Protocol

Each audio callback processes incoming commands in three sequential phases:

```text
[Start of Audio Callback]
           │
           ▼
┌──────────────────────────────────────┐
│ Phase 0: Deferred Resolution         │ ◄── Handles any payload parked from
│ - Check if older payload was parked  │     the previous callback
│ - Discard if superseded or stale     │
│ - Apply if budget permits            │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Phase 1: Bounded Drain & Coalescing  │ ◄── Examines up to pops_per_callback
│ - Classify the ring head (peek)      │     payloads from the SPSC ring
│ - Apply light scalar values inline   │
│ - Coalesce same-key structural items │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Phase 2: Budgeted Apply or Park      │ ◄── Applies at most swaps_per_callback
│ - Apply winning structural payload   │     heavy structural changes
│ - Park excess in deferred slot       │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ After Drain                          │ ◄── RtSwapHandler::after_drain
│ - Flush pending handler-local state  │
└──────────────────────────────────────┘
```

#### Phase 0: Deferred Resolution

If a previous callback parked a structural payload (because its swap budget was exhausted), that payload is causally before everything still in the ring and is resolved first:

- **Stale check**: if the handler's generation check reports that the payload no longer matches the current request, it is discarded through the handler (to the GC sink) without being counted as superseded.
- **Coalescing check**: if the head of the ring carries the same coalesce key, the parked payload is superseded and discarded, and `RT_STATUS_STRUCTURAL_SUPERSEDED` is raised.
- **Budget apply**: if budget is available, the payload is installed and the budget is consumed. Otherwise it stays parked for another callback and `RT_STATUS_STRUCTURAL_DEFERRED` is raised.

#### Phase 1: Bounded Drain & Coalescing

The audio thread examines up to `SwapTunables::pops_per_callback` payloads from the lock-free ring, classifying each head before detaching it:

- **Stale payloads**: popped and discarded through the handler (to the GC sink).
- **Scalar payloads**: applied immediately inline, never parked and never coalesced by the scheduler.
- **Structural payloads with the same coalesce key as the pending candidate**: the older candidate is discarded (`latest-wins`, `RT_STATUS_STRUCTURAL_SUPERSEDED`) at no budget cost.
- **Structural payloads with a different key**: the pending candidate is installed first and consumes budget, so at most one candidate is ever held and FIFO order across keys is preserved.
- **Budget stop**: if the structural budget is exhausted, the structural head stays queued in the ring (FIFO intact), `RT_STATUS_STRUCTURAL_DEFERRED` is raised, and the drain stops.

At the end of the phase, the drain raises `RT_STATUS_PARAM_QUEUE_BACKLOG` when the ring is still non-empty and `SwapTunables::backlog_flag` is set; otherwise it raises `RT_STATUS_SPSC_DRAIN_TRUNCATED` when the pop cap was reached with a non-empty ring.

#### Phase 2: Budgeted Apply or Park

If Phase 1 left a structural candidate:

- If `budget.can_apply()` is true, the payload is installed (the handler retires the replaced resource to the GC sink) and the budget is consumed.
- If the budget is exhausted, the payload is stored in the drain's single `deferred` slot, the ring's last pop is rolled back so that a sequence acknowledgment never covers it, and Phase 0 of the next callback resolves it.

After the three phases, `RtSwapHandler::after_drain` runs once, letting the handler flush pending local state (for example, batched scalar values).

---

### 2.2 Core Types and API

The scheduler is defined in [`src/common/spsc/swap.rs`](../src/common/spsc/swap.rs):

| Type                | Role                                                                                                                                                                                                                                   |
|:------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`SwapTunables`**  | Configuration struct (`pops_per_callback`, `swaps_per_callback`, `backlog_flag`). Set once at construction; never mutated on the RT path.                                                                                              |
| **`SwapBudget`**    | Per-callback counter tracking allowed structural swaps (`can_apply`, `consume`, `used`, `applied`, `limit`). Can be shared across multiple drains within the same audio callback.                                                      |
| **`GcSink`**        | Bundle of the 3-tier GC cascade (SPSC channel $\to$ 16-slot parking lot $\to$ overflow buffer) plus an optional parking-lot dirty latch. Never drops memory on the audio thread.                                                       |
| **`SwapRing`**      | Trait abstracting the lock-free consumer ring (`pop`, `peek`, `is_empty`, `occupied`), implemented for `rtrb::Consumer<Box<P>>`. Includes sequence acknowledgment hooks (`advance_resolved`, `rollback_last_pop`) with no-op defaults. |
| **`RtSwapHandler`** | Trait defining payload behavior: structural classification, coalescing key, optional generation checking, `install`, `discard`, and the `after_drain` hook.                                                                            |
| **`RtSwapDrain`**   | The coordinator struct owning the ring consumer and the single deferred slot (`drain`, `into_parts`, `has_deferred`, `ring_occupied`). Executes Phases 0–2.                                                                            |

```rust
// Simplified signatures of the scheduler API in common::spsc::swap
// (documentation comments omitted; see the source for the full contracts)

pub struct SwapTunables {
    pub pops_per_callback: usize,
    pub swaps_per_callback: usize,
    pub backlog_flag: bool,
}

pub struct SwapBudget {
    applied: usize,
    limit: usize,
}

impl SwapBudget {
    pub fn new(swaps_per_callback: usize) -> Self;
    pub fn can_apply(&self) -> bool;
    pub const fn used(&self) -> usize;
    pub fn consume(&mut self);
    pub fn applied(&self) -> usize;
    pub fn limit(&self) -> usize;
}

pub trait SwapRing {
    type Payload;
    fn pop(&mut self) -> Option<Box<Self::Payload>>;
    fn peek(&self) -> Option<&Self::Payload>;
    fn is_empty(&self) -> bool;
    fn occupied(&self) -> usize;
    fn advance_resolved(&mut self) {}
    fn rollback_last_pop(&mut self) {}
}

pub trait RtSwapHandler {
    type Payload;
    fn is_structural(&self, payload: &Self::Payload) -> bool;
    fn coalesce_key(&self, payload: &Self::Payload) -> Option<u64>;
    fn current_generation(&self) -> Option<u64> { None }
    fn generation_of(&self, payload: &Self::Payload) -> Option<u64> { None }
    fn install(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);
    fn discard(&mut self, payload: Box<Self::Payload>, gc: &mut GcSink<'_>);
    fn after_drain(&mut self, gc: &mut GcSink<'_>) { let _ = gc; }
}

pub struct RtSwapDrain<R: SwapRing> {
    ring: R,
    deferred: Option<Box<R::Payload>>,
    tunables: SwapTunables,
}

impl<R: SwapRing> RtSwapDrain<R> {
    pub fn new(ring: R, tunables: SwapTunables) -> Self;
    pub fn has_deferred(&self) -> bool;
    pub fn is_empty(&self) -> bool;
    pub fn ring_occupied(&self) -> usize;
    pub fn into_parts(self) -> (R, Option<Box<R::Payload>>);
    pub fn drain<H: RtSwapHandler<Payload = R::Payload>>(
        &mut self,
        handler: &mut H,
        budget: &mut SwapBudget,
        gc: &mut GcSink<'_>,
    );
}
```

---

### 2.3 Operational Invariants & Guarantees

1. **Zero Real-Time Allocation or Deallocation**:
   All buffers are pre-allocated. Replaced, stale, or superseded objects are moved into `GcSink`, where they are routed off the audio thread for reclamation. A payload parked in the deferred slot stays owned by the slot.
2. **Latest-Wins Coalescing**:
   Commands sharing the same `coalesce_key` collapse so only the latest configuration is applied, never across different keys. Commands with `coalesce_key == None` (such as atomic preset-restore sequences) are never superseded and keep FIFO order.
3. **Budget Enforcement**:
   At most `swaps_per_callback` structural applies occur per audio callback across all drains sharing the same `&mut SwapBudget`, preventing deadline overruns.
4. **Gapless Monotonic Acknowledgment**:
   With sequence-acknowledged rings, parking an unapplied payload calls `rollback_last_pop()`. The command is acknowledged via `advance_resolved()` only when it is actually installed or discarded in Phase 0. An acknowledgment never covers a parked command.
5. **Lock-Free and Non-Blocking**:
   The hot path contains no mutexes, system calls, logging macros (`log::*`), `unwrap`/`expect`, or dynamic allocations, and no dynamic dispatch: every generic parameter monomorphizes.
6. **Telemetry Through `RtStatusFlags`**:
   Deferrals and supersessions raise `RT_STATUS_STRUCTURAL_DEFERRED` and `RT_STATUS_STRUCTURAL_SUPERSEDED` and increment monotonic counters (`Relaxed`); backlog and truncation are reported as described in Phase 1.

---

### 2.4 Typical Integration Topologies

The scheduler supports different host architecture patterns through configuration:

| Topology Pattern        | Typical Use Case                                                      | Configuration (`SwapTunables`)                                  | Handler Strategy                                                      |
|:----------------------- |:--------------------------------------------------------------------- |:--------------------------------------------------------------- |:--------------------------------------------------------------------- |
| **Dedicated Rings**     | Hosts with per-resource channels (model, cab-sim, resampler)          | Small pop cap (e.g., 8); shared `SwapBudget` across all drains. | Generation-stamped checks (`current_generation`).                     |
| **Single Mixed Ring**   | Hosts with one command stream (scalars + models)                      | Larger pop cap (e.g., 64); scalar-style `backlog_flag`.         | Kind-based `coalesce_key`; light scalars applied inline.              |
| **Sequence-Acked Ring** | Hosts requiring explicit completion tracking for parameter automation | Standard pop cap; ring implements sequence hooks.               | `advance_resolved` on apply/discard; `rollback_last_pop` on deferral. |

Every knob is an explicit configuration value: pop caps and budgets are never averaged or unified across consumers.

---

## 3. Partitioned-Convolution Driver (`dsp::cabsim::adapter`)

### 3.1 Decoupling Latency from Block Size

Traditional convolution adapters required the audio block size to match the partition size ($N = P$). This created significant drawbacks:

- When a host negotiated a 512-sample buffer, algorithmic latency jumped to 512 samples (about 10.7 ms at 48 kHz).
- Whenever the host changed quantum sizes, the cab-sim engine had to be re-initialized.

The **Partitioned-Convolution Driver** decouples the engine's partition size from the host block size. The partition size is an installation-time policy chosen when the impulse response is built, based on latency requirements (for example, 64 or 128 samples). The algorithmic latency is exactly one partition (`latency_samples()`), whatever the block sizes. The host can pass blocks of any size ($N < P$, $N = P$, or $N > P$) to `process_block`.

```text
Host Audio Block (e.g., 512 samples)
[═══════════════════════════════════════════════════════════════]
                               │
               Chunked internally by driver
                               ▼
┌──────────────┬──────────────┬──────────────┬──────────────────┐
│ Partition 1  │ Partition 2  │ Partition 3  │ Partition 4      │
│ (128 samples)│ (128 samples)│ (128 samples)│ (128 samples)    │
└──────────────┴──────────────┴──────────────┴──────────────────┘
```

---

### 3.2 Processing Methods

The driver is implemented in [`src/dsp/cabsim/adapter.rs`](../src/dsp/cabsim/adapter.rs):

- **`CabSimAdapter::process_block`**:
  Processes a block of any size in place for a single channel. It splits the block into exact partition chunks, runs the UPOLS convolution, and carries the partial remainder in the internal FIFOs. It has no partition cap and never raises `RT_STATUS_CABSIM_CONTRACT_VIOLATION`; its output is bit-identical to feeding `process_in_place` with partition-capped windows.
- **`CabSimAdapter::process_in_place`**:
  In-place processing for sub-blocks bounded by the partition size (`samples.len() <= partition_size`). A longer slice is clamped to the partition size and raises `RT_STATUS_CABSIM_CONTRACT_VIOLATION` instead of panicking.
- **`CabSimAdapter::process_variable`**:
  Sub-block processing from an input slice to a distinct, non-overlapping output slice, with the same partition cap and contract-violation handling as `process_in_place`.
- **`CabSimPair::process_block_stereo`**:
  Runs two independent `CabSimAdapter` instances for Left and Right channels across arbitrary buffer sizes. Crucially, each channel maintains separate FIFO buffers and frequency-delay-line (FDL) states to prevent stereo crosstalk.
- **`CabSimPair::process_in_place_stereo`**:
  Stereo in-place processing for buffers bounded by partition size.
- **`CabSimAdapter::drain_tail` / `CabSimPair::drain_tail_stereo`**:
  When input audio stops (for example, noise-gate closure or silence), the impulse response decays ("rings out"). The `drain_tail` methods feed zero-input windows to flush the remaining energy without manual padding by the host. They accept an output slice of any size and chunk it internally.
- **Tail Budget Management**:
  `tail_samples()`, `rearm_tail()`, and `remaining_tail_samples()` track the sample budget required to flush the active FDL state. The total tail is the IR length in whole partitions plus one extra partition for the adapter's input accumulator. `rearm_tail` is called per host block, never per internal chunk.

```rust
// Key signatures in dsp::cabsim::adapter

impl CabSimAdapter {
    /// In-place block processing for arbitrary buffer sizes (no partition cap).
    pub fn process_block(&mut self, input_output: &mut [f32], rt_status: Option<&RtStatusFlags>);

    /// In-place processing capped at partition size (clamps and flags a violation).
    pub fn process_in_place(&mut self, samples: &mut [f32], rt_status: Option<&RtStatusFlags>);

    /// Variable sub-block processing between separate input and output slices.
    pub fn process_variable(&mut self, input: &[f32], output: &mut [f32], rt_status: Option<&RtStatusFlags>);

    /// Flushes decaying impulse response tail during silence.
    pub fn drain_tail(&mut self, output: &mut [f32], rt_status: Option<&RtStatusFlags>);

    /// Total samples required to flush the current IR tail.
    pub fn tail_samples(&self) -> usize;

    /// Re-arms the ring-out budget after active audio signal.
    pub fn rearm_tail(&mut self);

    /// Remaining non-silent samples before tail is fully flushed.
    pub fn remaining_tail_samples(&self) -> usize;
}

impl CabSimPair {
    /// Stereo in-place block processing for arbitrary buffer sizes.
    pub fn process_block_stereo(
        &mut self,
        samples_l: &mut [f32],
        samples_r: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    );

    /// Stereo in-place processing capped at partition size.
    pub fn process_in_place_stereo(
        &mut self,
        samples_l: &mut [f32],
        samples_r: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    );

    /// Flushes decaying impulse response tail on both channels.
    pub fn drain_tail_stereo(
        &mut self,
        out_l: &mut [f32],
        out_r: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    );

    /// Maximum remaining tail budget across both channels.
    pub fn remaining_tail_samples(&self) -> usize;

    /// Re-arms the tail budget across both channels.
    pub fn rearm_tail(&mut self);
}
```

---

### 3.3 FIFO Batching Mechanics

The internal execution flow of `process_block` operates as follows:

1. **Full Partitions Loop**:
   Using `chunks_exact_mut(partition_size)`, the driver iterates over complete partitions:
   - Accumulates input samples into the input buffer.
   - Executes UPOLS forward FFT, complex multiply-accumulate across partitions, and inverse FFT.
   - Delivers processed samples back into the slice.
2. **Remainder Processing**:
   Any trailing samples ($< P$) are accumulated into the input FIFO. When the accumulated samples reach the partition size, UPOLS executes; otherwise they stay buffered until the next callback.
3. **Shift-to-Front Compaction**:
   Both the input and output FIFOs are pre-allocated to $2 \times P$ samples. Before each partition is rendered, unconsumed samples are compacted to the front with `copy_within`, so the write region never overflows the queue. The engine renders directly into the output FIFO, with no intermediate scratch partition.
4. **Constant Latency**:
   The output queue starts with one partition of silence, so the delay is exactly $P$ samples for any sequence of block sizes.

---

### 3.4 Latency vs. CPU Trade-Off

Choosing partition size $P$ represents a trade-off between algorithmic latency and CPU efficiency:

| Profile                | Recommended Partition ($P$) | Latency @ 48 kHz   | Typical Application                         |
|:---------------------- |:--------------------------- |:------------------ |:------------------------------------------- |
| **Ultra-Low Latency**  | 32 or 64 samples            | 0.67 ms – 1.33 ms  | Live tracking, real-time guitar monitoring  |
| **Balanced (Default)** | 128 samples                 | 2.67 ms            | Standard studio recording, live performance |
| **High Efficiency**    | 256 or 512 samples          | 5.33 ms – 10.67 ms | Mixing, mastering, high track counts        |

For the mathematical formulation and measured CPU cost per partition size, see [`audio_fidelity_map.md`](audio_fidelity_map.md) §9.

---

## 4. Verification and Invariant Testing

Both components are verified by dedicated unit and integration test suites:

- **Structural-Swap Scheduler Verification** ([`src/common/spsc/swap_test.rs`](../src/common/spsc/swap_test.rs)):
  - **Budget adherence**: Verifies that with a budget of one, exactly one structural payload is applied per callback and the rest are deferred.
  - **Coalescing**: Confirms that superseded commands are routed to `GcSink` without leak.
  - **FIFO restoration**: Confirms non-coalescible payloads retain causal ordering.
  - **Staleness**: Confirms that stale-generation payloads are discarded without being installed.
  - **Pop cap**: Confirms that the pop cap truncates the drain and raises the corresponding flag.
  - **Zero RT allocations**: Validated by a heap-audit lane (feature `heap-audit`) proving that the drain path allocates nothing.
  - **Sequence consistency**: Validates that rollback and advance hooks keep sequence counters gapless.
- **Partitioned-Convolution Driver Verification** ([`src/dsp/cabsim/adapter_test.rs`](../src/dsp/cabsim/adapter_test.rs)):
  - **Blocking invariance**: Verifies that processing audio in arbitrary chunks (for example, 16, 64, or 333 samples) produces output bit-identical to fixed-partition processing, and that the impulse delay is constant for any blocking.
  - **Contract handling**: Verifies that `process_block` never raises the contract-violation flag.
  - **Tail flushing**: Validates that the ring-out terminates, matches the fixed-partition path, and is block-agnostic, without contract-violation flags.
- **Integration & RT Hardening Verification**:
  - **Heap audit**: [`tests/rt_constraints/cabsim_heap_audit.rs`](../tests/rt_constraints/cabsim_heap_audit.rs) validates zero dynamic heap allocations during steady-state convolution.
  - **C++ Reference Parity**: [`tests/parity/cabsim_cpp_parity.rs`](../tests/parity/cabsim_cpp_parity.rs) compares the cab-sim output against the C++ reference implementation across short, medium, and long impulse responses, gated by an ESR threshold documented in the test header.
  - **Golden Vectors**: [`tests/models/cabsim_golden.rs`](../tests/models/cabsim_golden.rs) verifies regression immunity against locked golden vectors.
  - **Stereo Invariance**: [`tests/cabsim_stereo.rs`](../tests/cabsim_stereo.rs) validates channel isolation, zero crosstalk, and multi-rate host quantum support.

---

## 5. Companion Documents

- [architecture.md](architecture.md): High-level system architecture, GC cascade (§2.1), and structural swap scheduler (§2.2).
- [audio_fidelity_map.md](audio_fidelity_map.md): Mathematical formulations and UPOLS trade-off analysis (§9).
- [testing.md](testing.md): Verification tiers, test execution rules, and parity oracles.
- [functional-tests.md](functional-tests.md): Functional test specifications and verification procedures.
