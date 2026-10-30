// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pinned real-time test suite entry point for `NeuralAmpModeler-rs`.
//!
//! This target is compiled only when the `pinned` feature is active, so the
//! default catch-all (`cargo test --features testing`) never runs these tests
//! without the isolation and scheduling setup provided by `utils/tests-long.sh`
//! phases `rt-deadline` and `rt-jitter`. The `pinned` feature is also enabled
//! by `utils/tests-quick.sh` so the non-ignored RT tests keep running in the
//! quick discipline.
//!
//! The `#[path]` declarations preserve the historical file layout while making
//! `super::` resolve to this target root, exactly as it did under
//! `tests/rt_constraints.rs`.

mod common;

// ── Shared RT budget (single source of truth) ─────────────────────────
#[path = "rt_constraints/budget.rs"]
mod budget;

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
