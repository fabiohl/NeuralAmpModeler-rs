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

### Added

- `utils/miri-audit.sh`: opt-in, out-of-band Miri concurrency & aliasing audit
  covering `common::spsc`, `dsp::pipeline::bridge`, `dsp::mirror_buf`,
  `dsp::smoother` and `dsp::utils::delay_line`. It complements `loom` (which
  proves the absence of races) and is intentionally excluded from the stable
  `lints.sh` / `tests-quick.sh` / `tests-long.sh` gates.
- `rt_hardening` (Linux, `rt-hardening` feature) now exposes the previously
  downstream-only generic host-bring-up helpers so consumers stop duplicating
  them: `pin_current_thread` / `PinThreadError`, `build_cpu_affinity_mask`,
  `build_housekeeping_affinity_mask`, `apply_housekeeping_affinity` and the
  injectable `set_cpu_affinity_with` (renamed from the crate-private
  `set_cpu_affinity_one`). The affinity selection/topology API remains the
  single canonical implementation. Pure additive API; no behavior change for
  existing callers.

### Fixed

- `tests/perf_soak.rs`: the counting `#[global_allocator]` is now installed
  unconditionally. The previous `cfg_attr(not(feature = "heap-audit"), ...)`
  gating stripped the attribute exactly when `--features heap-audit` audits ran,
  leaving the soak zero-allocation checks green-but-vacuous. Added the
  affirmative `audit_harness_counts_allocations` non-vacuity guard.
