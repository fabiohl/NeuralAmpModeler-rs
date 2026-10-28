<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — 1.1.0

### Changed

- **QA packaging surface:** the maintainer-only QA binaries (`gen_stress`,
  `wav_to_golden`, `nam_golden_catalog`, `nam_freshness`, `nam_long_receipt`,
  `nam_quality`, `nam_remote_simd_receipt`, and the removed `capability_receipt`)
  are no longer shipped in the published crate. The `testing` library modules
  remain available because downstream suites use them as a dev-dependency, and
  the externally documented `nam_perf_gate` and `simd_probe` binaries stay in the
  package. The `include` list is now complete (`/src/arch/` was missing and made
  every verified tarball fail to compile); this ships in 1.1.0.

## [1.0.1] - 2026-10-19

### API Compatibility

- **100% Backward-Compatible (SemVer Patch Release)**.
- **Zero breaking changes**: All public structs, traits (`NamModel`), enums (`StaticModel`), functions (`load_and_build_model`, `load_and_build_wav_ir`), and error codes (`NamErrorCode` E1xxx–E5xxx) remain completely unchanged. Downstream crates (`NAM-Plug`, `NAM-Audio-Pipe`) require zero source code changes to upgrade.

### Added

- **OOM Containment & Memory Protection:**
  - Added [`utils/mem-guard.sh`](utils/mem-guard.sh), a transparent systemd user scope wrapper (`MemoryHigh=5G`, `MemoryMax=6G`, `oom_score_adj=500`) protecting host systems and IDEs against OOM killer termination during heavy release builds and test audits.
- **Hierarchical Benchmark Governance:**
  - Added support for nested, hierarchical Criterion benchmark groups (e.g., `PIPE_Host/Full_Load_64_samp`) in [`src/bin/nam_perf_gate.rs`](src/bin/nam_perf_gate.rs) and [`src/testing/qa/baseline_store.rs`](src/testing/qa/baseline_store.rs).
  - Added fail-closed validation and structured diagnostics when baselines are absent.
- **Baseline Fingerprint Reproducibility:**
  - Extended baseline environment fingerprint schema in [`src/testing/qa/fingerprint.rs`](src/testing/qa/fingerprint.rs) with `git_dirty` and `cargo_lock_sha256` flags.
- **Automated Regression Guard Scenarios:**
  - Added Scenario 5 (deterministic CPU core selection with mock sysfs) and Scenario 6 (benchmark binary selection prioritizing `nam_benches` and intruder fail-fast detection) to [`tests/scripts/test_regression_guard.sh`](tests/scripts/test_regression_guard.sh).
  - Integrated fail-fast Criterion smoke test (`--list`) before thermal cooldown in [`utils/tests-performance-regression.sh`](utils/tests-performance-regression.sh).
- **Technical Documentation:**
  - Published [`docs/postmortem-libm-symbol-interposition.md`](docs/postmortem-libm-symbol-interposition.md) detailing compiler-rt symbol collision root cause and linker script defenses.
  - Published [`docs/rt-structural-scheduler-spike.md`](docs/rt-structural-scheduler-spike.md) detailing lock-free 3-phase structural swap scheduling and partitioned UPOLS driver mechanics.

### Changed

- **Internal Core Modularization:**
  - Refactored `src/models/linear/multichannel.rs` into modular sub-units ([`src/models/linear/multichannel/`](src/models/linear/multichannel/): `many_to_many.rs`, `many_to_one.rs`, `one_to_many.rs`, `mode.rs`) with identical re-exports.
  - Refactored `src/loader/nam_json/topology/wavenet.rs` into modular sub-units ([`src/loader/nam_json/topology/wavenet/`](src/loader/nam_json/topology/wavenet/): `a2_validation.rs`, `budget.rs`, `guardrail.rs`, `sku.rs`, `slimmable.rs`).
  - Refactored `src/testing/receipt.rs` into modular sub-units ([`src/testing/receipt/`](src/testing/receipt/): `capability.rs`, `log_parser.rs`, `long_audit.rs`, `long_phase.rs`).
- **Documentation & Verification Synchronization:**
  - Resolved rustdoc intra-doc links and completed documentation across internal modules.
  - Documented Scenarios 5 & 6 of `test_regression_guard.sh` across [`docs/testing.md`](docs/testing.md), [`docs/benchmarks.md`](docs/benchmarks.md), and [`docs/functional-tests.md`](docs/functional-tests.md).
  - Synchronized automation tools and utility scripts in [`README.md`](README.md).
  - Sincronized feature permutations checklist in [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
- **Dependency Update:**
  - Bumped `libc` dependency from `0.2.169` to `0.2.191`.

### Fixed

- **Deterministic Benchmark Binary Extraction:**
  - Fixed benchmark binary resolution in [`utils/_lib_core.sh`](utils/_lib_core.sh) to deterministically select the primary benchmark harness via `jq`, preventing trailing binary mismatches.
- **Quality Dashboard Log Diagnostics:**
  - Preserved failure diagnostics in `target/logs/regression_gate.log` and handled missing Criterion estimate objects gracefully in [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh).
- **Golden Regeneration Diagnostics:**
  - Conditioned golden regeneration notices on missing required fixtures, eliminating false-alarm warnings when only optional mirrors are unpopulated.
- **A2 Dynamic Max Real-Time Heap Audit:**
  - Activated [`test_a2_dyn_max_heap_audit`](tests/rt_constraints/a2_heap_audit.rs) in the Phase 4 real-time heap audit battery (unignoring test execution) and streamlined [`utils/tests-long.sh`](utils/tests-long.sh) to verify zero heap allocations across all A2 dynamic model topologies in a single pass.

---

## [1.0.0] - 2026-10-15

- Initial stable release of **NeuralAmpModeler-rs**:
  - High-performance, real-time safe neural inference engine written in pure Rust.
  - Support for WaveNet (A1/A2, Standard, Lite, Feather, Nano, FiLM, Slimmable), LSTM (1x16, 2x8, multi-layer), ConvNet, Linear FIR/FFT, Container, and Sequential models.
  - Mandatory `x86-64-v3` vectorization baseline (AVX2, FMA, BMI2) with experimental opt-in AVX-512 backend.
  - Real-time safety guarantees: zero heap allocations, zero mutexes, and zero blocking syscalls on the audio path.
  - Dual-oracle numerical parity validated against C++ NAMCore and double-precision f64 reference.
  - UPOLS partitioned cabinet impulse response convolution driver (`CabSimAdapter`).
  - Polyphase anti-aliasing oversampling engine (`OversampleEngine`) and multi-rate sinc resampler (`NamResampler`).
  - 3-tier lock-free SPSC parameter and model hot-swap mechanism with off-RT garbage collection cascade.
  - Diagnostic error reporting catalogue (`NamErrorCode` E1xxx–E5xxx).

[1.0.1]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/releases/tag/v1.0.0
