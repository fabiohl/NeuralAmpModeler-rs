<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed [Unreleased]

- **`--all-targets --all-features` on aarch64:** AVX-512 tests and benchmarks now gate on the `nam_avx512` cfg emitted by `build.rs` instead of the raw `avx512` feature, so `cargo check --all-targets --all-features` succeeds on non-`x86_64` targets. The obsolete `compile_error!` that rejected `avx512` on non-`x86_64` was removed; the `build.rs` gate still keeps the AVX-512 modules out of those builds.

- **QA summary isolation:** `utils/_lib.sh` is now side-effect free when sourced, and `emit` no longer appends to `target/logs/summary.txt` when a child process runs with `lib_quiet=1`. A failed summary append records `EMIT_RC` and forces `FAIL (fails=emit)` instead of silently reporting `PASS`.

- **Deterministic log initialization in CI:** The `msrv` and `cross-platform-parity` jobs now call `lib_init` before their first `phase`/`emit`, so `target/logs` is created on a cold cache and `summary.txt` is truncated per run instead of publishing stale lines from a warm cache.

## [2.0.0] - 2026-11-03

The engine now runs natively on **`aarch64`**, both Apple Silicon macOS and Linux. Its NEON backend matches the x86_64 path numerically. This release also makes several public entry points fail closed and trims the published package down to production binaries. Read **Breaking Changes** before you upgrade from 1.x.

### Highlights [2.0.0]

- **Native aarch64 and Apple Silicon:** A NEON implementation of the AVX2/FMA/F16C intrinsic surface (`src/arch/aarch64/`) keeps the same per-lane operation order and single-rounding FMA as x86_64. The result is identical SNR against C++ NAMCore across the golden catalog.
- **Compile-time SIMD baselines:** The crate requires `x86-64-v3` on x86_64 and NEON on aarch64, and checks both with `compile_error!`. There is no runtime dispatch and no scalar fallback.
- **Fail-closed public contracts:** Invalid topologies, out-of-range channel counts, corrupt snapshots and invalidated public state are rejected up front or rendered as silence. They no longer reach the unchecked kernels.
- **macOS real-time hardening:** The opt-in `rt-hardening` feature now covers macOS: interactive QoS, Mach time-constraint policy, per-buffer memory locking and Audio Workgroups.
- **Faster small-block kernels:** Maskless GEMV tails, tap-major depthwise convolutions and frame-pair tiling reduce overhead at low buffer sizes.

### Breaking Changes and Migration [2.0.0]

- **x86_64 build flags:** As in 1.x, the crate does not compile without the `x86-64-v3` baseline. When the flag is missing, `build.rs` now prints the exact `.cargo/config.toml` snippet you need. Scope it to `cfg(target_arch = "x86_64")` so aarch64 builds of the same project are unaffected.
- **`E5001` retired:** The loader no longer checks the CPU at runtime, so `NamErrorCode::UnsupportedCpuArchitecture` (`E5001`) is never emitted. The variant and number stay reserved. Remove any handling that depends on it.
- **`LinearModel::new_with_topology`:** Weight and bias counts that cannot satisfy the declared receptive field or bias geometry now return `Err`. Previously, missing biases were broadcast from the first value or defaulted to `0.0`.
- **`NamModel::process_multichannel` / `process_multichannel_in_place`:** When a declared channel count exceeds the planar adapter's 1024-channel staging capacity, these calls return `NamErrorCode::HostFormatContractViolation` (`E2304`) before any audio plane is touched. Check the channel count first, or call `process_raw` for wider layouts.
- **`restore_buffer_starts`:** Restores are now all-or-nothing. A snapshot containing any invalid start leaves the model unchanged instead of partially rewinding it.
- **Invalidated public state:** If public fields are mutated into an inconsistent state, `process` and `prewarm` produce silence (or do nothing) instead of computing from corrupt data. After any manual reconfiguration, call `validate_state()` on the WaveNet and Linear models, or `is_sound()` on layer arrays and multichannel linear models.
- **`testing::reference_oracle::oracle_forward`:** Topologies outside `oracle_domain` now panic with a descriptive `OracleError` instead of returning a zero vector. Use `try_oracle_forward` when the topology is discovered at runtime.
- **Installed binaries:** The package now installs only `simd_probe` and `nam_perf_gate`. Note that `nam_perf_gate` requires `--features testing`. Maintainer tools such as `gen_stress`, `wav_to_golden`, `nam_quality` and the receipt generators moved to the unpublished `tools/qa` workspace member, along with the maintainer-only QA helpers that used to live under the `testing` feature.

### Added [2.0.0]

- `aarch64` platform layer: `FPCR.FZ` denormal control, a generic-timer cycle clock (`CNTVCT_EL0`, calibrated from `CNTFRQ_EL0`), page-size-aware delay rings, and mirrored buffers backed by unlinked POSIX shared memory on macOS.
- `rt_hardening` on macOS: `set_user_interactive_qos`, `promote_time_constraint`, `compute_thread_timing`, `mlock_region`, `join_audio_workgroup` (RAII `AudioWorkgroupJoin`) and `set_ftz_daz`.
- `NamResampler::min_output_samples_checked` and `max_input_samples_checked` return `None` for a zero sample rate instead of panicking.
- `NeonMath` type alias on aarch64.
- `testing::vector_protocol`: a versioned, endian-explicit, bit-exact parity-vector format for comparing outputs across platforms.
- docs.rs builds for `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin` and `aarch64-unknown-linux-gnu`.

### Changed [2.0.0]

- **SIMD kernels:** GEMV partial output-channel tails accumulate four 128-bit lanes without masked operations. Single-output paths use two 4-frame tiles per 8-frame block to avoid transpose spills. WaveNet A2 grouped and depthwise convolutions repack weights tap-major (SoA) to remove gathers. The 4×4, 8×8 and 12×12 residual kernels use frame-pair tiling.
- **Noise gate:** `DynamicHysteresis::update` derives ramp slopes from `GateParams::fade_frames` on every block and clamps its counters, so changing parameters mid-fade can no longer push gain above unity.
- **Denormals:** FTZ/DAZ requirements for audio threads are now specified and covered by doctests. Scalar tails flush subnormals explicitly.
- **`avx512`:** The feature is restricted to x86_64 at compile time.

### Fixed [2.0.0]

- **WAV IR decoding:** Sample buffers are reserved fallibly and bounds-checked. Oversized inputs return `WavError::TooLong` instead of aborting.
- **`CabSimAdapter`:** The output queue is primed, so latency stays constant across host block sizes.
- **IR resampling:** Resample chunks are capped to the input buffer while downsampling, which prevents overruns.
- **FFT:** Bit-reversal reordering no longer creates aliased mutable references.

## [1.0.1] - 2026-10-19

Patch release, fully backward-compatible: public types, functions and `NamErrorCode` values are unchanged.

### Changed [1.0.1]

- Internal modularization of the multichannel linear models, the WaveNet topology loader and the testing receipt module, with identical re-exports.
- `libc` raised to `0.2.191`.
- New technical write-ups: [libm symbol-interposition postmortem](docs/postmortem-libm-symbol-interposition.md) and [RT structural scheduler spike](docs/rt-structural-scheduler-spike.md).

### Fixed [1.0.1]

- Quality-dashboard verdicts: test counts are taken from libtest summaries, provenance failures are reported apart from fidelity failures, and hung phases are killed with their whole process group.
- The real-time heap audit now covers every A2 dynamic topology.

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

[Unreleased]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v2.0.0...HEAD

[2.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.1...v2.0.0

[1.0.1]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.0...v1.0.1

[1.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/releases/tag/v1.0.0
