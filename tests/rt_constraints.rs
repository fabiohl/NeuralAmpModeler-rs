// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time constraint test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates real-time audio thread guarantees, including zero-heap allocation audits
//! (`#[cfg(feature = "heap-audit")]`), processing deadline compliance, and timing jitter bounds.

mod common;

// Global allocator fixture unconditionally installed so allocation counting
// (TrackingGuard thread-local counters inside `CountingAllocator`) is live in
// every feature configuration. The previous `cfg_attr(not(feature =
// "heap-audit"), ...)` gating silently stripped the attribute exactly when
// `--features heap-audit` audits ran, leaving the System allocator in charge
// and every phase heap-audit green-but-vacuous (a probe inside the guard
// counted zero allocations).
use common::alloc_audit::CountingAllocator;

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

// ── Sequential-chain Heap-Audit Submodule ────────────────────────────────────
#[path = "rt_constraints/sequential_heap_audit.rs"]
mod sequential_heap_audit;

/// Affirmative regression guard for the whole heap-audit family: proves the
/// counting allocator is actually installed for `--test rt_constraints` and
/// the TrackingGuard counters observe allocations in every feature
/// configuration. Zero-allocation audits are only meaningful while this smoke
/// check passes (a vacuous harness reports `count == 0` forever).
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

// ── Shared RT budget (single source of truth) ─────────────────────────
#[path = "rt_constraints/budget.rs"]
mod budget;

// ── Zero-Heap Allocation Audit Submodules ────────────────────────────────────
#[path = "rt_constraints/a2_heap_audit.rs"]
mod a2_heap_audit;
#[path = "rt_constraints/cabsim_heap_audit.rs"]
mod cabsim_heap_audit;
#[path = "rt_constraints/resampler_heap_audit.rs"]
mod resampler_heap_audit;

// ── Real-Time Timing & Determinism Submodules ───────────────────────────────
#[path = "rt_constraints/rt_deadline.rs"]
mod rt_deadline;
#[path = "rt_constraints/rt_jitter.rs"]
mod rt_jitter;

// ── Budget coherence: both suites must observe the same RT budget ────────────
#[cfg(test)]
mod budget_coherence {
    use super::budget;

    #[test]
    fn rt_deadline_and_jitter_share_one_budget() {
        assert_eq!(budget::RT_DEADLINE_US, super::rt_deadline::RT_DEADLINE_US);
        assert_eq!(budget::BLOCK_SIZE, super::rt_deadline::BLOCK_SIZE);
        assert_eq!(budget::RT_DEADLINE_US, super::rt_jitter::RT_DEADLINE_US);
        assert_eq!(budget::BLOCK_SIZE, super::rt_jitter::BLOCK_SIZE);
        assert_eq!(
            budget::RT_DEADLINE_US,
            budget::budget_us(budget::BLOCK_SIZE, budget::BUDGET_SAMPLE_RATE_HZ) - 3,
            "gate keeps its 3 µs conservative margin below the exact budget"
        );
    }
}
