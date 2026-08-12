// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! `SlimmableModel` trait — models that can dynamically scale quality/complexity
//! at runtime without reallocation.
//!
//! This is the official NAM architecture for runtime quality scaling.
//!
//! ## Architectural divergence from C++ NAM — `SlimmableWavenet` staging
//!
//! The C++ reference implementation (`NeuralAmpModelerCore`) uses
//! `std::atomic<shared_ptr<WaveNet>>` for model staging: the DSP thread
//! atomically swaps the `shared_ptr` and the old model's reference count is
//! decremented inline. If the count reaches zero, the destructor runs on the
//! audio thread — potentially triggering `WaveNet`'s heap deallocation
//! (hundreds of KB to MB) inside the real-time callback.
//!
//! NAM-rs **intentionally diverges** from this design to guarantee strict
//! zero-heap-deallocation on the DSP hot-path. The old model is never dropped
//! on the audio thread. Instead, it is packed into a `GcItem::Model(…)` and
//! routed through the **SPSC GC pipeline**:
//!
//! - **RT thread (producer)**: `gc_cascade(item, &mut producer, &mut parking_lot)`
//!   attempts to push the `GcItem` into a lock-free `rtrb` SPSC ring buffer.
//!   If full, the item cascades into a small fixed-size `parking_lot` overflow
//!   array (16 slots, also lock-free). If the parking lot is exhausted, the
//!   item is safely dropped on the RT thread as a last-resort fallback (this
//!   should never occur under normal operation).
//!
//! - **Main thread (consumer)**: Pumps the GC channel periodically via
//!   `drain_gc_channels(&mut consumer, &overflow, &rt_status)`, pulling all
//!   pending `GcItem`s and dropping them outside the real-time callback —
//!   typically during host housekeeping cycles or plugin idle
//!   callbacks.
//!
//! This architecture removes all `std::sync::atomic`, `std::shared_ptr`,
//! and reference-counting contention from the audio path. The only
//! synchronization between the two threads is the SPSC ring buffer's atomic
//! head/tail indices — a single `fetch_add` on the producer side, no CAS loops,
//! no lock contention, and zero heap deallocation on the RT thread.
//!
//! ### Lifecycle of a slimmable swap
//!
//! 1. Async/main thread calls `model.slice_channels(target_ch)` to build a
//!    new `WaveNetModelDyn` with reduced channel count (allocating weights and
//!    states). Prewarm stabilizes the convolutional state.
//!
//! 2. The new model replaces the old via `Option<Box<StaticModel>>::replace()`
//!    — a simple pointer swap, no atomics needed on the stack-local `Option`.
//!
//! 3. The old `Box<StaticModel>` is handed to `gc_cascade` for non-RT disposal.
//!
//! ## Multi-size NAMCore parity disclaimer
//!
//! NAMCore (`NeuralAmpModelerCore`) has no channel-slicing API. It loads and
//! renders the full WaveNet topology but cannot produce output at intermediate
//! channel counts (`allowed_channels` breakpoints). Consequently, C++-adjudicated
//! golden and live multi-size parity is architecturally infeasible.
//!
//! This crate is **inference-only** for `SlimmableWavenet`. Load, breakpoint,
//! and slice-inference tests (`test_loader_gap_slimmable_wavenet`,
//! `test_slimmable_wavenet_inference_and_breakpoints`) remain active and green.
//! No claim of multi-size NAMCore parity is made; zero parity-related marketing
//! language appears in user-facing surfaces (README, parity-map §6, §7.4 P6).

/// Trait for models that can dynamically scale quality/complexity at runtime
/// without reallocation.
///
/// The value `0.0` represents the minimum quality/cheapest model,
/// and `1.0` represents the maximum quality/full model.
///
/// Implementors:
/// - `ContainerModel`: selects between pre-built submodels by threshold.
/// - `SlimmableWavenet` (planned): channel-slices a single network.
pub trait SlimmableModel {
    /// Sets the slimmable quality/size level.
    ///
    /// `val` is in `[0.0, 1.0]` where `0.0` = minimum quality and `1.0` = full quality.
    ///
    /// `rt_status` — when called from the RT hot-path, pass `Some(&rt_status)` so
    /// that errors from reset can be signaled atomically without I/O.
    fn set_slimmable_size(
        &mut self,
        val: f32,
        rt_status: Option<&crate::common::spsc::RtStatusFlags>,
    );

    /// Returns the breakpoints at which slimmable quality transitions occur.
    ///
    /// Each breakpoint represents a normalized value in `[0.0, 1.0]` where the
    /// model switches to a different submodel or internal quality tier. Hosts
    /// and plugins can use these to map and snap discrete quality parameters.
    ///
    /// Defaults to an empty slice for models without discrete breakpoints.
    fn slimmable_breakpoints(&self) -> Box<[f64]> {
        Box::new([])
    }
}

#[path = "slicing.rs"]
pub mod slicing;
pub use slicing::*;

#[cfg(test)]
#[path = "slimmable_test.rs"]
mod tests;
