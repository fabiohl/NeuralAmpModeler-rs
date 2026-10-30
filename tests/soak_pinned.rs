// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Pinned soak test suite entry point for `NeuralAmpModeler-rs`.
//!
//! The long-running endurance tests are compiled only when the `pinned`
//! feature is active, so the default catch-all never executes the multi-minute
//! soak battery. `utils/tests-long.sh` runs this target in the `soak` phase
//! with `--include-ignored --test-threads=1`.
//!
//! The `CountingAllocator` is installed here exactly as in `tests/perf_soak.rs`
//! so the zero-allocation assertions inside `soak_test` remain meaningful
//! instead of observing the system allocator vacuously.

mod common;

use common::alloc_audit::CountingAllocator;

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

#[path = "perf_soak/soak_test.rs"]
mod soak_test;
