// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Canonical bench constants shared across the benchmark binaries and the
//! quality dashboard.
//!
//! Single source of truth: `utils/quality-dashboard.sh` never hard-codes these
//! values — `build.rs` extracts them from this file into
//! `target/bench_constants.env`, which the dashboard sources at parse time
//! (`parse_benchmarks`). `target/` is gitignored, so the env file is a
//! regenerated build artifact, not tracked state.

/// Blocks processed per Criterion sample in the sub-µs DSP micro-benches
/// (`RT_DSP_Resampler_*`, `RT_DSP_CabSim_IR_Medium`).
///
/// A single 64-sample block is ~0.7–1.4 µs and trips the noise wall from
/// timer jitter alone; a fixed batch keeps timer noise below the threshold.
/// Criterion params (sample_size/measurement/noise) stay canonical. Reported
/// time is for the full batch (not per-block) — the quality dashboard divides
/// by this factor before comparing to `docs/quality-contract.json`
/// (`batch_factor`, per-block contract units).
pub const DSP_MICRO_BATCH: usize = 64;
