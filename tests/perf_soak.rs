// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance and soak test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates concurrency stress tolerance and high-load SPSC ring pipelines.
//! The extended soak battery lives in `tests/soak_pinned.rs`, compiled only
//! with the `pinned` feature.

mod common;

use common::alloc_audit::CountingAllocator;

// Global allocator fixture installed unconditionally so allocation counting
// (TrackingGuard thread-local counters inside `CountingAllocator`) is live in
// every feature configuration. The previous `cfg_attr(not(feature =
// "heap-audit"), ...)` gating silently stripped the attribute exactly when
// `--features heap-audit` audits ran, leaving the System allocator in charge
// and every soak zero-alloc phase green-but-vacuous (a probe inside the guard
// counted zero allocations). Mirrors the fix in `tests/rt_constraints.rs` /
// `tests/models.rs` (EC-09).
#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Affirmative regression guard for the soak heap-audit family: proves the
/// counting allocator is actually installed for `--test perf_soak` and the
/// TrackingGuard counters observe allocations in every feature configuration.
/// The zero-allocation checks in `perf_soak/soak_test.rs` are only meaningful
/// while this smoke check passes (a vacuous harness reports `count == 0`
/// forever).
#[test]
fn audit_harness_counts_allocations() {
    let count = {
        let _guard = common::alloc_audit::TrackingGuard::new();
        let probe: Vec<Box<usize>> = (0..8).map(Box::new).collect();
        std::hint::black_box(&probe);
        common::alloc_audit::get_alloc_count()
    };
    assert!(
        count >= 8,
        "audit harness must count allocations (got {count}) — \
         counting allocator is not installed; all zero-alloc audits would be vacuous"
    );
}

// ── Performance & Stress Submodules ──────────────────────────────────────────
#[path = "perf_soak/concurrency_stress.rs"]
mod concurrency_stress;
#[path = "perf_soak/pipeline_soak.rs"]
mod pipeline_soak;
#[path = "perf_soak/spsc_pipeline.rs"]
mod spsc_pipeline;
