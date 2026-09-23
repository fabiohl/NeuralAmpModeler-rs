// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Real-time constraint test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates real-time audio thread guarantees, including zero-heap allocation audits
//! (`#[cfg(feature = "heap-audit")]`), processing deadline compliance, and timing jitter bounds.

mod common;

use common::alloc_audit::CountingAllocator;

#[cfg_attr(not(feature = "heap-audit"), global_allocator)]
// Global allocator fixture retained across all feature configurations.
#[allow(dead_code, clippy::allow_attributes)]
static GLOBAL: CountingAllocator = CountingAllocator;

// ── Shared RT budget (S1-T1: single source of truth) ─────────────────────────
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

// ── S1-T1 coherence: both suites must observe the same RT budget ────────────
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
