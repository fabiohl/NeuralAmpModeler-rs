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
//! NeuralAmpModeler-rs **intentionally diverges** from this design to guarantee strict
//! zero-heap-deallocation on the DSP hot-path. The old model is never dropped
//! on the audio thread. Instead, it is packed into a `GcItem::Model(…)` and
//! routed through the **SPSC GC pipeline**:
//!
//! - **RT thread (producer)**: `gc_cascade(item, &mut producer, &mut parking_lot, &overflow, &rt_status)`
//!   attempts to push the `GcItem` into a lock-free `rtrb` SPSC ring buffer.
//!   Before parking anything, it flushes items already parked in the `parking_lot`
//!   back to the SPSC channel whenever there is free capacity, so the lot tends
//!   to empty every cycle. If the SPSC is full, the item cascades into a small
//!   fixed-size `parking_lot` overflow array (16 slots, also lock-free). If the
//!   parking lot is exhausted, the item cascades into the `GcOverflowBuffer`
//!   (an overwrite ring buffer) — it is **never** dropped on the RT thread.
//!
//! - **Main thread (consumer)**: Pumps the GC pipeline periodically via
//!   `drain_gc_channels(&mut consumer, &overflow, &mut parking_lot, &rt_status)`,
//!   draining the SPSC channel, the overflow buffer, and the parking lot, and
//!   dropping every `GcItem` outside the real-time callback — typically during
//!   host housekeeping cycles or plugin idle callbacks, and once more during
//!   orderly teardown after the real-time producers have stopped.
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
//! ## Heterogeneous Channel Slicing & Parity
//!
//! Slimmable WaveNet models allow dynamic channel reduction via `allowed_channels`
//! lists defined per array. When ratio-based scaling is requested, `slimmable_breakpoints`
//! computes the sorted union of fractional breakpoints (`i / n`) across all layer arrays.
//!
//! Upstream `NeuralAmpModelerCore` provides channel slicing via the CLI flag
//! `render --slim <ratio>`, which is verified in cross-validation parity tests alongside
//! single-precision sliced-vs-prefix-masked equivalence oracles (an independent C++
//! cross-check; the in-tree oracle is not a double-precision reference).
//!
//! Slicing strictly **reduces** channel width down from the full loaded model ("só reduz,
//! nunca amplia"). Re-expanding beyond the currently sliced channel width requires reloading
//! or cloning from the retained full-capacity model (see `clone_wavenet_for_slimmable_storage`).

/// Trait for models that can dynamically scale quality/complexity at runtime
/// without reallocation.
///
/// The value `0.0` represents the minimum quality/cheapest model,
/// and `1.0` represents the maximum quality/full model.
///
/// Implementors:
/// - `ContainerModel`: selects between pre-built submodels by threshold.
/// - `WaveNetModelDyn`: channel-slices a single network across layer arrays.
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

/// Maps a normalized quality `ratio` in `[0.0, 1.0]` to the channel-tier index
/// of a model with `num_tiers` allowed channel counts.
///
/// The mapping is `idx = min(floor(ratio * num_tiers), num_tiers - 1)`: each
/// tier owns an equal `1 / num_tiers` share of the quality axis, so tier `t`
/// is selected for `ratio ∈ [t / num_tiers, (t + 1) / num_tiers)`, and
/// `ratio == 1.0` clamps to the top tier.
///
/// `ratio` is taken in `f64` to mirror the C++ `ratio_to_channels`
/// (`slimmable.cpp:102-106`, `double ratio`) exactly. Passing an `f32`-rounded
/// ratio (e.g. `0.7f32`) into a wide `num_tiers` mapping can otherwise fall one
/// tier below the C++ result: with `num_tiers = 10`, `0.7f64 * 10 = 7.0` floors
/// to tier 7, whereas `0.7f32 as f64 * 10 = 6.999…` floors to tier 6. Callers
/// that need C++ parity must therefore forward the ratio as `f64`.
#[inline]
pub fn slimmable_tier_index(ratio: f64, num_tiers: usize) -> usize {
    if num_tiers == 0 {
        return 0;
    }
    let r = ratio.clamp(0.0, 1.0);
    ((r * num_tiers as f64).floor() as usize).min(num_tiers - 1)
}

/// Normalized breakpoints of a `num_tiers` channel-tier model: `i / num_tiers`
/// for `i in 1..num_tiers`, strictly increasing and deduplicated (a single
/// tier owns an equal slice of the quality axis).
///
/// These are the quality ratios at which [`slimmable_tier_index`] moves to the
/// next tier. Returns an empty slice for fewer than two tiers.
#[inline]
pub fn slimmable_channel_breakpoints(num_tiers: usize) -> Box<[f64]> {
    if num_tiers < 2 {
        return Box::new([]);
    }
    (1..num_tiers)
        .map(|i| i as f64 / num_tiers as f64)
        .collect()
}

/// Union of per-array slimmable breakpoints (C++
/// `SlimmableWavenet::GetSlimmableSizeBreakpoints` /
/// `get_ratio_breakpoints`).
///
/// For each non-empty per-array `allowed` list with `n` tiers, contributes
/// `i / n` for `i in 1..n`; the union is sorted and deduplicated. Empty
/// (non-slimmable) arrays contribute nothing.
#[inline]
pub fn slimmable_per_array_breakpoints(per_array: &[Vec<usize>]) -> Box<[f64]> {
    let mut bps: Vec<f64> = Vec::new();
    for allowed in per_array {
        let n = allowed.len();
        if n < 2 {
            continue;
        }
        for i in 1..n {
            bps.push(i as f64 / n as f64);
        }
    }
    bps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    bps.dedup();
    bps.into_boxed_slice()
}

#[path = "slicing.rs"]
pub mod slicing;
pub use slicing::*;

#[cfg(test)]
#[path = "slimmable_test.rs"]
mod tests;
