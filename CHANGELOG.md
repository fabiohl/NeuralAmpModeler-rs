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

### Added

- **`NeonMath` alias:** `pub type NeonMath = Avx2Math` on aarch64. Additive; no existing name changes.
- **aarch64/NEON backend:** native NEON implementation of the AVX2/FMA/F16C
  intrinsic surface in `src/arch/`, with the same per-lane operation order and
  single-rounding FMA as the x86_64 path, so golden-catalog parity against the
  C++ reference is identical on both architectures.
- **macOS (Apple Silicon) support:** unlinked POSIX shared-memory mirror buffer
  (`shm_open` + `shm_unlink`), `FPCR.FZ` denormal control, generic-timer cycle
  counter (`CNTVCT_EL0`), and page-size-aware delay rings. Intel
  (`x86_64-apple-darwin`) macOS is not supported or validated.
- **`avx512` feature is x86_64-only:** the opt-in research backend now emits a
  `compile_error!` on other architectures instead of failing deep in the module
  tree.
- **macOS real-time hardening (`rt-hardening`, `target_os = "macos"`):**
  `src/rt_hardening/macos.rs` and `timebase.rs` add interactive QoS, time
  constraint policy, memory locking, `FPCR.FZ`, and an RAII Audio Workgroup
  guard. The design is recorded in [`docs/rt-macos-adr.md`](docs/rt-macos-adr.md)
  (Accepted). The Linux module and its API are unchanged. No jitter gain is
  claimed until measured on Apple hardware.

### Changed

- **f32 GEMV tail and `out_len == 1`:** the partial output-channel tail no
  longer uses masked AVX2 loads/stores (or their NEON predicate emulation).
  It accumulates four 128-bit lanes plus scalar lanes while keeping the exact
  per-lane 8-accumulator summation order. The single-output path uses two
  4-frame tiles per 8-frame block, removing the register spill from the
  8-frame transpose without changing the block partition or the trailing
  per-frame fallback. Measured x86-64 improvements are recorded in
  `docs/benchmarks.md` §10.1.
- **x86-64-v3 guard:** the runtime AVX2/FMA check in the model dispatcher is
  removed (dead code behind the `compile_error!` in `src/lib.rs`). A pre-v3 host
  fails with SIGILL, as documented. `E5001` (`UNSUPPORTED_CPU_ARCHITECTURE`) is
  reserved and discontinued; its number is not reused. `simd_probe` prints the
  build-time guarantee instead of a runtime compliance verdict.
- **docs.rs targets:** `x86_64-unknown-linux-gnu` (landing), `aarch64-apple-darwin`
  and `aarch64-unknown-linux-gnu`. The `avx512` modules compile only for x86_64
  through the `nam_avx512` cfg emitted by `build.rs`, so aarch64 documentation
  builds with `all-features` no longer fail. `utils/docsrs-check.sh` checks both
  the host and `aarch64-unknown-linux-gnu`.
- **Consumer onboarding:** the x86-64-v3 build-script error names the dependency
  case and points to the application's `.cargo/config.toml` and the README.
- **Packaging preflight:** `tests-long.sh` `preflight-package` runs
  `cargo package --locked` (verify) and a `consumer_smoke` check: a consumer
  outside the repository must fail on x86_64 without the baseline flag, and
  compile with it and on both aarch64 targets.
- **QA binaries isolated:** `gen_stress`, `wav_to_golden`, `nam_golden_catalog`,
  `nam_freshness`, `nam_long_receipt`, `nam_quality` and `nam_remote_simd_receipt`
  moved to the `nam-qa` workspace member (`tools/qa`, `publish = false`). The
  published manifest now declares only `nam_perf_gate` and `simd_probe`. Repo
  tooling calls them with `-p nam-qa`.
- **libm guard scope:** documented that the version script protects only this
  crate's own link targets; downstream binaries need their own.
- **A2 grouped and depthwise convolution:** depthwise weights are repacked into a
  tap-major SoA tail, removing the gather from the x86 depthwise kernel; aarch64
  gets explicit NEON kernels. Output bits are unchanged (`to_bits` tests against
  the previous layout). aarch64 speed is not yet measured.
- **Nano and gated residual paths:** the `fused_residual_batch` 4x4 case uses a
  four-accumulator SSE block; gated CH=4/12 pairs two frames per YMM and removes
  the `vmaskmovps` path on x86. Measured in `docs/benchmarks.md` §10.3.
- **Low-latency frame tails:** 2–3 frame pair tiles for the 4x4, 8x8 and 12x12
  residual kernels on x86 (bit-identical per frame). Measured in
  `docs/benchmarks.md` §10.4.
- **aarch64 prefetch:** `_mm_prefetch` is a documented no-op on aarch64, so no
  `prfm` instructions are emitted there. x86 prefetch policy is unchanged.
- **NEON reductions and tanh clamp:** `hsum` uses the same reduction tree as x86,
  and the tanh clamp uses `fmin`/`fmax` with the x86 NaN ordering. Pending
  hardware measurement.
- **LSTM and DspBridge:** LSTM activation precision is hoisted per block and a
  redundant prefetch is removed; the DspBridge reader and writer are padded to
  separate 128-byte lines. The Apple Silicon effect is not measured.
- **Numerical policy documented:** `NamModel::process` documents the FTZ/DAZ
  requirement with a compiling doc-test, and the scalar tanh tail flushes
  subnormals with the same policy.
- **Build script:** `target/bench_constants.env` is written only by the primary
  in-repo build, never inside a dependency checkout.
- **Performance claims:** the measured gains are x86-64 only. No aarch64 gain is
  declared until hardware receipts exist.

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

- **Quality Dashboard Test Counting:**
  - `nam_quality` now counts libtest `test result: ok. N passed` summaries (ANSI-stripped) across every command of a phase instead of per-test `test … ... ok` lines, so `--nocapture` runs no longer undercount green phases. A phase without a `test result` line fails closed as `no_test_result`; the unreachable `lstm_activation_precision` minimum now matches its two non-ignored tests.
- **Quality Provenance Verdict:**
  - `verify_contract` classifies `freshness` as `PROVENANCE` instead of a fidelity violation, emitting `PROVENANCE: FAIL` with `FIDELITY: NOT_VERIFIED` (D-06) when fixtures are stale; `--save` refuses failed provenance separately from failed fidelity.
- **Bounded Dashboard Commands:**
  - Dashboard phases now enforce their wall-clock deadline by killing the whole child process group and draining captured pipes, so a hung `cargo test` cannot leave orphaned test binaries behind.
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

[Unreleased]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/releases/tag/v1.0.0
