// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Performance and soak test suite entry point for `NeuralAmpModeler-rs`.
//!
//! Validates concurrency stress tolerance, high-load SPSC ring pipelines, and extended
//! continuous processing stability (soak tests) under heavy buffer contention.

mod common;

use common::alloc_audit::CountingAllocator;

#[cfg_attr(not(feature = "heap-audit"), global_allocator)]
#[allow(dead_code, clippy::allow_attributes)]
static GLOBAL: CountingAllocator = CountingAllocator;

// ── Performance & Stress Submodules ──────────────────────────────────────────
#[path = "perf_soak/concurrency_stress.rs"]
mod concurrency_stress;
#[path = "perf_soak/pipeline_soak.rs"]
mod pipeline_soak;
#[path = "perf_soak/soak_test.rs"]
mod soak_test;
#[path = "perf_soak/spsc_pipeline.rs"]
mod spsc_pipeline;
