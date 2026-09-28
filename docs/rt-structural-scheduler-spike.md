<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Real-Time Structural-Swap Scheduler & Partitioned-Convolution Driver

This document describes two core architectural components of the `NeuralAmpModeler-rs` engine:

1. **Generic RT Structural-Swap Scheduler** ([`src/common/spsc/swap.rs`](../src/common/spsc/swap.rs)): Coordinates safe, lock-free, zero-allocation updates of heavy heap resources on the audio thread.
2. **Partitioned-Convolution Driver** ([`src/dsp/cabsim/adapter.rs`](../src/dsp/cabsim/adapter.rs)): Decouples algorithmic impulse response (IR) latency from audio host block sizes without multiplying memory copies.

---

## 1. Overview & Core Problems

### 1.1 What is a Structural Swap?

In an audio engine, parameters change at different computational costs:

- **Scalar parameters** (e.g., input gain, output volume): Updated by changing a few numbers directly in memory.
- **Structural resources** (e.g., neural models, cab-sim impulse responses, resamplers, oversampling engines, complete preset state): Require pre-allocated heap objects (tens of kilobytes to several megabytes).

Updating a structural resource during audio playback is a **structural swap**. In real-time audio, this presents two strict requirements:

1. **Zero allocation and zero deallocation on the audio thread**: Freeing memory (`drop`) invokes system heap allocators, which can block on locks or kernel calls and cause audible glitches ("xruns"). Retired resources must be deferred to an off-real-time garbage collection (GC) thread.
2. **Deterministic execution time**: Replacing multiple large resources in a single audio callback could exceed the audio deadline (e.g., ~333 µs at 16 samples buffer size). The audio engine must budget how many structural changes are allowed per callback.

The **RT Structural-Swap Scheduler** provides a single, canonical protocol for all resource swaps across any host architecture.

### 1.2 What is the Partitioned-Convolution Problem?

Cabinet simulation uses **Uniformly Partitioned Overlap-Save (UPOLS)** convolution. In UPOLS, the impulse response is divided into small partitions of size $P$:

- Algorithmic latency is determined by partition size ($P$ samples).
- If an adapter requires the block size to equal the partition size, the latency is pinned to the host's buffer size. A large host buffer (e.g., 512 samples) forces high latency, and changing host buffer sizes requires rebuilding the convolution engine.

The **Partitioned-Convolution Driver** solves this by internally batching sub-partitions in a single FIFO sweep, allowing fixed, low-latency partitions regardless of host buffer size.

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
│ Phase 1: Bounded Drain & Coalescing  │ ◄── Reads up to pops_per_callback
│ - Pop payloads from ring buffer      │     commands from SPSC ring
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
│ After Drain                          │ ◄── Flushes pending local state
│ - Flush pending scalar locals        │
└──────────────────────────────────────┘
```

#### Phase 0: Deferred Resolution

If the previous callback parked a structural command (because its swap budget was exhausted), that parked command is resolved first:

- **Stale check**: If the generation counter indicates the command was superseded while waiting, it is discarded to the GC sink.
- **Coalescing check**: If the head of the command ring has the same coalesce key, the older parked command is superseded and discarded.
- **Budget apply**: If budget is available, the parked command is installed immediately. If budget is still unavailable, it remains parked for another callback.

#### Phase 1: Bounded Drain & Coalescing

The audio thread pops up to `SwapTunables::pops_per_callback` commands from the lock-free ring:

- **Scalar commands**: Applied immediately inline.
- **Stale commands**: Discarded directly to the GC sink.
- **Structural commands**: If multiple commands target the same resource (e.g., rapid preset changes), intermediate commands are discarded (`latest-wins`), and only the most recent candidate is kept.
- **Budget stop**: If the structural budget is already exhausted, further structural commands are parked, preserving FIFO order.

#### Phase 2: Budgeted Apply or Park

If a structural candidate was selected in Phase 1:

- If `budget.can_apply()` is true, the payload is installed, the old resource is retired to the GC sink, and the budget is consumed.
- If the budget is exhausted, the payload is stored in the drain's `deferred` slot to be resolved in Phase 0 of the next callback.

---

### 2.2 Core Types and API

The scheduler is defined in [`src/common/spsc/swap.rs`](../src/common/spsc/swap.rs):

| Type                | Role                                                                                                                                      |
|:------------------- |:----------------------------------------------------------------------------------------------------------------------------------------- |
| **`SwapTunables`**  | Configuration struct (pop caps, structural apply limits, backlog reporting flags). Set once at initialization.                            |
| **`SwapBudget`**    | Per-callback counter tracking allowed structural swaps. Can be shared across multiple drains within the same audio callback.              |
| **`GcSink`**        | Safe wrapper around the 3-tier GC cascade (SPSC channel $\to$ parking lot $\to$ overflow buffer). Never drops memory on the audio thread. |
| **`SwapRing`**      | Trait abstracting the lock-free consumer ring. Includes sequence acknowledgment hooks (`advance_resolved`, `rollback_last_pop`).          |
| **`RtSwapHandler`** | Trait defining payload behavior: structural classification, coalescing key, generation checking, `install`, and `discard`.                |
| **`RtSwapDrain`**   | The coordinator struct owning the ring consumer and the single deferred slot. Executes Phases 0–2.                                        |

```rust
// Simplified structure of the scheduler API in common::spsc::swap

pub struct SwapTunables {
    pub pops_per_callback: usize,
    pub swaps_per_callback: usize,
    pub backlog_flag: bool,
}

pub struct SwapBudget {
    applied: usize,
    limit: usize,
}

pub trait SwapRing {
    type Payload;
    fn pop(&mut self) -> Option<Box<Self::Payload>>;
    fn peek(&self) -> Option<&Self::Payload>;
    fn is_empty(&self) -> bool;
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
```

---

### 2.3 Operational Invariants & Guarantees

1. **Zero Real-Time Allocation or Deallocation**:
   All buffers are pre-allocated. Replaced or discarded objects are moved into `GcSink`, where they are routed off the audio thread for safe reclamation.
2. **Latest-Wins Coalescing**:
   Commands sharing the same `coalesce_key` collapse so only the latest configuration is applied. Commands with `coalesce_key == None` (such as atomic preset restore sequences) are strictly preserved in FIFO order.
3. **Budget Enforcement**:
   At most `swaps_per_callback` structural operations occur in an audio callback, preventing audio deadline overruns.
4. **Gapless Monotonic Acknowledgment**:
   When sequence numbers are used, parking an unapplied payload calls `rollback_last_pop()`. The command is only acknowledged via `advance_resolved()` when actually installed or discarded in Phase 0.
5. **Lock-Free and Non-Blocking**:
   The hot path contains no mutexes, system calls, logging macros (`log::*`), or dynamic allocations.

---

### 2.4 Typical Integration Topologies

The scheduler supports different host architecture patterns through configuration:

| Topology Pattern        | Typical Use Case                                                        | Configuration (`SwapTunables`)                                  | Handler Strategy                                                      |
|:----------------------- |:----------------------------------------------------------------------- |:--------------------------------------------------------------- |:--------------------------------------------------------------------- |
| **Dedicated Rings**     | Standalone hosts with per-resource channels (model, cab-sim, resampler) | Small pop cap (e.g., 8); shared `SwapBudget` across all drains. | Generation-stamped checks (`current_generation`).                     |
| **Single Mixed Ring**   | Audio plugin hosts with one command stream (scalars + models)           | Larger pop cap (e.g., 64); `backlog_flag = false`.              | Kind-based `coalesce_key`; light scalars applied inline.              |
| **Sequence-Acked Ring** | Hosts requiring explicit completion tracking for parameter automation   | Standard pop cap; ring implements sequence hooks.               | `advance_resolved` on apply/discard; `rollback_last_pop` on deferral. |

---

## 3. Partitioned-Convolution Driver (`dsp::cabsim::adapter`)

### 3.1 Decoupling Latency from Block Size

Traditional convolution adapters required the audio block size to match the partition size ($N = P$). This created significant drawbacks:

- When a host negotiated a 512-sample buffer, algorithmic latency jumped to 512 samples (~10.7 ms at 48 kHz).
- Whenever the host changed quantum sizes, the cab-sim engine had to be re-initialized.

The **Partitioned-Convolution Driver** decouples the engine's internal partition size from the host's block size. The partition size is chosen when building the impulse response based on latency requirements (e.g., 64 or 128 samples). The host can then pass blocks of any size ($N < P$, $N = P$, or $N > P$).

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
  Processes an arbitrary-sized block of audio in place for a single channel. It splits the block into exact partition chunks, runs the UPOLS convolution, and handles partial sample remainders via internal FIFOs.
- **`CabSimPair::process_block_stereo`**:
  Runs two independent `CabSimAdapter` instances for Left and Right channels. Crucially, each channel maintains separate FIFO buffers and frequency-delay-line (FDL) states to prevent stereo crosstalk.
- **`CabSimAdapter::drain_tail` / `CabSimPair::drain_tail_stereo`**:
  When input audio stops (e.g., noise gate closure or silence), the impulse response naturally decays ("rings out"). The `drain_tail` methods feed zero-input blocks to flush the remaining reverberant energy without requiring manual padding by the host.

```rust
// Key signatures in dsp::cabsim::adapter

impl CabSimAdapter {
    /// In-place block processing for arbitrary buffer sizes.
    pub fn process_block(&mut self, input_output: &mut [f32], rt_status: Option<&RtStatusFlags>);

    /// Flushes decaying impulse response tail during silence.
    pub fn drain_tail(&mut self, output: &mut [f32], rt_status: Option<&RtStatusFlags>);
}

impl CabSimPair {
    /// Stereo in-place block processing for arbitrary buffer sizes.
    pub fn process_block_stereo(
        &mut self,
        samples_l: &mut [f32],
        samples_r: &mut [f32],
        rt_status: Option<&RtStatusFlags>,
    );
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
   Any trailing samples ($< P$) are accumulated into the FIFO. If accumulated samples reach partition size, UPOLS executes; otherwise, they remain buffered until the next audio callback.
3. **Shift-to-Front Compaction**:
   Output FIFOs are pre-allocated to $2 \times P$ samples. When unconsumed samples remain, they are compacted to the front using efficient SIMD memory moves (`copy_within`), guaranteeing that memory never overflows.

---

### 3.4 Latency vs. CPU Trade-Off

Choosing partition size $P$ represents a trade-off between algorithmic latency and CPU efficiency:

| Profile                | Recommended Partition ($P$) | Latency @ 48 kHz   | Typical Application                         |
|:---------------------- |:--------------------------- |:------------------ |:------------------------------------------- |
| **Ultra-Low Latency**  | 32 or 64 samples            | 0.67 ms – 1.33 ms  | Live tracking, real-time guitar monitoring  |
| **Balanced (Default)** | 128 samples                 | 2.67 ms            | Standard studio recording, live performance |
| **High Efficiency**    | 256 or 512 samples          | 5.33 ms – 10.67 ms | Mixing, mastering, high track counts        |

For mathematical details and performance curves, refer to [`docs/audio_fidelity_map.md`](audio_fidelity_map.md) (§9).

---

## 4. Verification and Invariant Testing

Both components are verified by dedicated test suites:

- **Structural-Swap Scheduler Verification** ([`src/common/spsc/swap_test.rs`](../src/common/spsc/swap_test.rs)):
  - **Budget adherence**: Verifies that when multiple structural commands arrive, exactly one is executed and the rest are deferred.
  - **Coalescing**: Confirms that superseded commands are routed to `GcSink` without leak.
  - **FIFO restoration**: Confirms non-coalescible payloads retain causal ordering.
  - **Zero RT allocations**: Validated via `#[cfg(feature = "heap-audit")]` tracking zero allocations on the audio path.
  - **Sequence consistency**: Validates that rollback and advance hooks keep sequence counters gapless.
- **Partitioned-Convolution Driver Verification** ([`src/dsp/cabsim/adapter_test.rs`](../src/dsp/cabsim/adapter_test.rs)):
  - **Blocking invariance**: Verifies that processing audio in arbitrary chunks (e.g., 16, 64, 333 samples) produces output bit-identical to fixed partition processing.
  - **Parity verification**: Output matches both the C++ NAMCore reference and 64-bit float mathematical oracles.
  - **Tail flushing**: Validates smooth decay to true zero without audible discontinuities or contract violation flags.

---

## 5. Companion Documents

- [architecture.md](architecture.md): High-level system architecture and memory management (§2.1 & §2.2).
- [audio_fidelity_map.md](audio_fidelity_map.md): Mathematical formulations and UPOLS trade-off analysis (§9).
- [testing.md](testing.md): Verification tiers, test execution rules, and parity oracles.
- [functional-tests.md](functional-tests.md): Functional test specifications and verification procedures.
