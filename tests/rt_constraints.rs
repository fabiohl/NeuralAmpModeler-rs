// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time constraint test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates real-time audio thread guarantees, including zero-heap allocation audits
//! (`#[cfg(feature = "heap-audit")]`). Deadline and jitter suites live in
//! `tests/rt_pinned.rs`, which is compiled only with the `pinned` feature.

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
// The budget unit tests stay in this target so the unpinned catch-all keeps
// exercising the frozen values; `tests/rt_pinned.rs` reuses the constants only.
#[path = "rt_constraints/budget.rs"]
mod budget;

#[cfg(test)]
mod budget_tests {
    use super::budget::*;

    #[test]
    fn budget_values_are_frozen() {
        assert_eq!(RT_DEADLINE_US, 1330);
        assert_eq!(BLOCK_SIZE, 64);
        assert_eq!(BUDGET_SAMPLE_RATE_HZ, 48_000);
    }

    #[test]
    fn canonical_budget_matches_formula_with_conservative_margin() {
        // Exact budget at the canonical point truncates to 1333 µs; the
        // enforced gate keeps a 3 µs conservative margin below it.
        assert_eq!(budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ), 1333);
        assert_eq!(budget_us(64, 48_000), 1333);
        assert!(
            RT_DEADLINE_US <= budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ),
            "gate {} must not exceed exact budget {}",
            RT_DEADLINE_US,
            budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ)
        );
        assert_eq!(
            budget_us(BLOCK_SIZE, BUDGET_SAMPLE_RATE_HZ) - RT_DEADLINE_US,
            3
        );
    }
}

// ── Zero-Heap Allocation Audit Submodules ────────────────────────────────────
#[path = "rt_constraints/a2_heap_audit.rs"]
mod a2_heap_audit;
#[path = "rt_constraints/cabsim_heap_audit.rs"]
mod cabsim_heap_audit;
#[path = "rt_constraints/resampler_heap_audit.rs"]
mod resampler_heap_audit;
