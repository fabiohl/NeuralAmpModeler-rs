<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Changelog

All notable changes to `NeuralAmpModeler-rs` are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Release notes are prepared here as changes land and are published with the
> version bump (see the ecosystem sprint plan, task `T8.2`). Nothing in the
> `Unreleased` section below has been published.

## [Unreleased]

### Added

- **High-Resolution Latency Telemetry (Sprint 4 / B4, B5)** (`feat(rt): migrate RT constraint suites to high-resolution LatencyHistogram`):
  Integrated linear `LatencyHistogram` with 0.5 µs resolution (0–4096 µs) into real-time constraint suites (`tests/rt_constraints/rt_deadline.rs`, `tests/rt_constraints/rt_jitter.rs`), replacing coarse log2 power-of-two buckets. Enables exact sub-microsecond percentile estimation ($p50, p90, p99, p99.9$) with bounded relative resolution $\le 12.5\%$.
- **Performance Regression Phase in Long Audit (Sprint 2 / B3)** (`fix(receipt): aggregate phase gaps in overall receipt and integrate perf regression verdict`):
  `utils/tests-performance-regression.sh` produces machine-readable `target/logs/regression_phase_receipt.jsonl` parsed fail-closed into the long-audit receipt (`nam_long_receipt`).
- **Cross-ISA Remote Gate & SDE Emulation Documentation (Sprint 8 / B8)** (`docs: convert ignored doctests to compilable examples and document cross-ISA skips`):
  Added operational guidance and syntax for `utils/remote-simd-gate.sh` (`--sde` for Intel SDE, SSH remote for physical AVX-512 nodes) and clarified architectural boundaries in `docs/testing.md`.
- `utils/miri-audit.sh`: opt-in, out-of-band Miri concurrency & aliasing audit covering `common::spsc`, `dsp::pipeline::bridge`, `dsp::mirror_buf`, `dsp::smoother` and `dsp::utils::delay_line`.
- `rt_hardening` (Linux, `rt-hardening` feature) exposes generic host-bring-up helpers (`pin_current_thread`, `build_cpu_affinity_mask`, `build_housekeeping_affinity_mask`, `apply_housekeeping_affinity`, `set_cpu_affinity_with`).

### Changed

- **BREAKING (`DspBridge`)** — `DspBridge::buffers` is now
  `[core::cell::UnsafeCell<BridgeBuffer>; 2]` (previously `[BridgeBuffer; 2]`) to
  provide interior mutability. The capture/playback lock-free protocol now
  reaches the front/back buffers through raw pointers (`UnsafeCell::get`) and no
  longer materializes a whole-struct `&DspBridge` / `&mut DspBridge` on the hot
  path, removing the aliasing UB that Miri flagged under the Stacked Borrows /
  Tree Borrows models. Struct layout, 128-byte alignment, cache-line isolation
  and every field offset are unchanged (`UnsafeCell` is `#[repr(transparent)]`
  and zero-cost); `DspBridge` retains `Sync` via a documented manual impl.
- **Migration for downstream readers of the field** — a direct
  `bridge.buffers[i].field` access becomes
  `unsafe { (*bridge.buffers[i].get()).field }`; the `DspBridgeWriter` /
  `DspBridgeReader` faces are unaffected and remain the supported API.
- **Activation Precision Table Partitioning (Sprint 6 / B2)** (`docs(lstm): triage oracle divergence, partition activation tables, and characterize cold transients`):
  Partitioned QA activation reports in `src/testing/qa/render.rs` into two independent tables: Warm NAMCore goldens vs. Cold-Start f64 reference oracles. Eliminates mixed-sample aggregate distortion.
- **Compilation Profile Segregation (Sprint 5 / B7)** (`test(loom): silenciar panicos loom, segregar CARGO_TARGET_DIR por perfil e alinhar logs de fases`):
  `utils/tests-long.sh` segregates `CARGO_TARGET_DIR` by profile (`release-default`, `release-heap-audit`, `loom`), preventing cross-phase compilation churn and cache invalidation.
- **Doctest Compilation (Sprint 8 / B8)** (`docs: convert ignored doctests to compilable examples and document cross-ISA skips`):
  Converted `OversampleEngine` usage and `testing::diagnostics` harness doctests into compilable/runnable examples; `cargo test --doc --all-features` now reports 0 ignored tests.

### Fixed

- **Fidelity & Perceptual Metrics Zero-Comparisons (Sprint 1 / B1)** (`fix(testing): correct zero-comparisons in error metrics and reject inconsistent contract records`):
  `compute_snr_db`, `compute_esr`, `esr_to_db`, and integration test validation helpers now check `== 0.0` rather than `<= f64::EPSILON`. Eliminates false `SNR = ∞` when error signal has non-zero sub-epsilon power (e.g. `convnet_silu` SNR correctly computed at $\approx 124.9\text{ dB}$).
- **Receipt Gap Aggregation & Discrepancies (Sprint 2 / B3)** (`fix(receipt): aggregate phase gaps in overall receipt and integrate perf regression verdict`):
  `nam_long_receipt summary` collects and reports gaps across all individual phases into `overall.gaps`. Reconciled architecture catalog fixture IDs.
- **Loom Model-Checker Panics (Sprint 5 / B6)** (`test(loom): silenciar panicos loom, segregar CARGO_TARGET_DIR por perfil e alinhar logs de fases`):
  Silenced expected unwinding panics during Loom lock-free branch exploration, ensuring clean test output.
- `tests/perf_soak.rs`: counting `#[global_allocator]` installed unconditionally with `audit_harness_counts_allocations` non-vacuity guard.

### Documentation & Investigation

- **Oracle Divergence & Kahan Accumulation Analysis (Sprint 6 / B2)** (`docs(lstm): triage oracle divergence, partition activation tables, and characterize cold transients`):
  Documented Kahan compensated accumulation in Rust LSTM cells vs naive float32 accumulation in upstream C++ NAMCore; proved Rust LSTM closer to f64 oracle ($9.08 \times 10^{-13}$) than C++ ($8.51 \times 10^{-12}$). Documented upstream bug reproduction for `known_gap:condition_lstm_cpp_crash`.
- **LSTM Cold-Start Transient Characterization (Sprint 6)** (`docs(lstm): triage oracle divergence, partition activation tables, and characterize cold transients`):
  Characterized LSTM cold-start transient under `docs/perceptual_validation.md#decomposition-cold-start` (§5.2); verified that prewarm of 24,000 samples reduces ESR from 0.0506 to $\le 1.72 \times 10^{-12}$.
- **Fidelity Drift Bisection (Sprint 3 / A2, A3)** (`docs: record fidelity drift bisection and reference noise floor`):
  Bisection results and noise floors recorded across `dyn-gated`, `nobn`, `nano`, `EVH`, `rf4096/8192`, `film-lite` topologies.
- **Cross-ISA `SKIP_CAPABILITY` & `FIDELITY: OK` Scope Boundary (Sprint 8 / B8)** (`docs: convert ignored doctests to compilable examples and document cross-ISA skips`):
  Documented in `docs/testing.md` (§8.3, §10.2) that `FIDELITY: OK` is a local single-architecture certification and does not assert cross-ISA parity, and updated the artifact inventory in `docs/functional-tests.md` (§4.3).
