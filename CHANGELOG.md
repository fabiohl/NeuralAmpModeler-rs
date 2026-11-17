<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-11-17

This major release resolves physical-aliasing soundness in `MirroredBuffer<T>`, enforces strict backpressure and FIFO ordering in `StreamingResampleBuffer`, converges extensible WAV validation, adds experimental verifiable Windows 11 (`x86_64` and `aarch64`) support, and hardens cross-platform diagnostics and QA gates. Read **Breaking Changes** before upgrading from `2.0.0`.

### Breaking Changes and Migration [3.0.0]

- **`MirroredBuffer<T>` `DerefMut` removed (`src/dsp/mirror_buf.rs`):** Removed `impl DerefMut for MirroredBuffer<T>`, which previously exposed a `2N` mutable slice and allowed safe code (e.g. `split_at_mut(size())`) to obtain two aliasing `&mut [T]` references backed by the same physical memory. `Deref<Target = [T]>` (`2N` shared reads) is preserved, and safe bounded mutable access is now provided via `Index<I: SliceIndex<[T]>>`, `IndexMut<usize>`, `window_mut(offset, len)`, `try_window_mut(offset, len)` (enforcing `len <= size()`), `write_slice_at`, `fill`, and allocation-free `copy_within`.
- **WaveNet post-stack head and A1 geometry structs:**
  - `PostStackHead` (`src/models/wavenet/post_stack_head.rs`) replaced public `conv`, `activation`, and `state` fields and `process_block_internal` with canonical multi-stage head fields (`stages`, `canonical`).
  - `HeadConfig` (`src/loader/nam_json/model.rs`) added `kernel_sizes`, and `FreeWavenetGeometry` (`src/loader/nam_json/topology/wavenet/mod.rs`) added `layer_kernel_sizes` to support per-layer kernel sizes in WaveNet A1 geometry.
- **Internal allocator metadata (`src/math/common/huge_alloc.rs`):** Marked `AllocInfo` and `deallocate_huge` as `#[doc(hidden)]` (paired exclusively with `pub(crate) allocate_huge_pages`).

### Added [3.0.0]

- **Experimental Verifiable Windows 11 support (`x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`):**
  - Win32 `MirroredBuffer<T>` backend (`src/dsp/mirror_buf/windows.rs`) using `VirtualAlloc2` + `MapViewOfFile3` placeholders, 6-stage `S0`–`S5` transactional rollback, immediate `CloseHandle`, and dual `UnmapViewOfFile` on `Drop`, preserving `size_of::<MirroredBuffer<T>>() == 16`.
  - ARM64 timer probe (`src/common/tsc.rs`) protected by a Vectored Exception Handler (`AddVectoredExceptionHandler`) with automatic fallback to `std::time::Instant`.
  - Portable PII/Windows/UNC path redaction and crash directory resolution (`src/common/diagnostics/format.rs`, `src/common/panic_hook.rs`, `src/loader/loaded_model_pair.rs`).
  - Internal `nam-preflight` diagnostic crate (`tools/preflight`), `.gitattributes` (`eol=lf`), and native CI jobs (`windows-2025`, `windows-11-arm`) integrated into `cross-platform-parity` (`0 ULP` bitwise parity across all 7 canonical models).
- **WaveNet A1 per-layer `kernel_sizes` and canonical multi-stage `PostStackHead`** (`src/models/wavenet/`, `src/loader/nam_json/`), plus `LinearModel` prewarm sample parity matching NAMCore PR #352.
- **Fail-closed QA reconcilers (`tools/qa`):** Added typed `nam_simd_parity` (reconciling the 10 mandatory AVX2↔AVX-512 parity cases via `NAM_SIMD_PARITY_JSONL`) and `nam_skips` (fail-closed `SkipRecord` JSONL consumer in `tests-quick.sh` and `tests-long.sh`), replacing log-text `grep` checks.

### Changed [3.0.0]

- **`StreamingResampleBuffer` backpressure and FIFO ordering (`src/dsp/resampling/streaming_adapter.rs`):** Bounded `drain_model_to_out` to `out_fifo.free() + discard_remaining` with `unpop` preservation and zero-progress termination; resumed draining during `pull_output`; restricted `process` bypass fast-path to empty FIFOs; and signaled `StreamingResult::underflow` immediately when `produced < drained` without fabricating output samples.
- **Typed resampler construction errors (`src/dsp/resampling/streaming_adapter.rs`):** Switched `StreamingResampleBuffer` to `NamResampler::new_simple_typed` and `new_linear_simple_typed` so invalid sample rates report `NamErrorCode::ResamplerBuildFailed` (`E2200`) instead of `OutOfMemory`.
- **Cross-platform `tsc`, `huge_alloc`, `mirror_buf`, and diagnostics cleanup:**
  - Removed unreachable `QueryPerformanceCounter` dead code and `"Win32_System_Performance"` from root `Cargo.toml` in Windows ARM64 `read_counter()` hot-path (`src/common/tsc.rs`).
  - Moved POSIX `memfd_create`/`shm_open` backing helpers from `huge_alloc.rs` to `src/dsp/mirror_buf/posix.rs`, restricted Linux `mmap` huge-page paths to `target_os = "linux"`, and routed macOS allocations through the global allocator (`AllocInfo::Heap` / `HugePageStatus::Heap`).
  - Parameterized `baseline_isa_label()` for `x86_64` vs `aarch64`, added `uname`/`RtlGetVersion` kernel version detection in `system_info.rs`, scoped `emit_irq_advisory` to Linux, and implemented `sysctlbyname` CPU/OS reporting on macOS in `tools/preflight`.
- **Documentation, examples, and `docs.rs` surface:** Aligned `examples/offline_render.rs` and `README.md` (separating `crates.io` consumers from repo checkouts and fixing `nam_parity_vector` CLI syntax), updated `utils/consumer-smoke.sh` to validate the extracted `.crate` tarball, refined `doc(cfg)` attributes on `rt_hardening`, and extended `utils/docsrs-check.sh` across all 5 `[package.metadata.docs.rs]` targets.

### Fixed [3.0.0]

- **Strict `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) validation and parser convergence (`src/loader/wav.rs`, `src/dsp/cabsim/ir_parse.rs`):** Converged `fmt` chunk parsing via `parse_fmt_chunk`, enforcing `fmt_size >= 40`, `cbSize >= 22`, exact `EXTENSIBLE_GUID_TAIL`, `valid_bits == bits_per_sample`, `block_align`, and `byte_rate`, while preserving `cabsim` odd-sized EOF `data` chunks without padding.
- **`nam_quality dashboard --save` contract persistence (`tools/qa/src/bin/nam_quality.rs`):** Fixed `--save` so it refreshes `generated_at`, `provenance`, `fidelity`, `performance`, and `worst_esr_parity` in `docs/quality-contract.json` from the verified report before atomic write, failing closed if `regression_gate` did not pass (unless `--fidelity-only`).
- **AVX-512 `CH=8` `film.active_mask() == 0` fast-path (`src/models/a2/conv1d_ch8/simd.rs`):** Removed the early return in `layer_forward_ch8_block_simdmath` that skipped the dilated convolution loop, restoring `x86_64-linux` vs. `x86_64-avx512` parity on `wavenet_a2_full` (`SNR = 129.14 dB`, `ESR = 1.22e-13`).
- **Portable temporary paths in loader/CabSim tests (`src/dsp/cabsim/loader_test.rs`, `src/loader/loader_malformed_test.rs`):** Replaced hardcoded POSIX `/tmp/*.wav` paths with `std::env::temp_dir()` and RAII cleanup guards.
- **`--all-targets --all-features` on aarch64:** AVX-512 tests and benchmarks now gate on the `nam_avx512` cfg emitted by `build.rs` instead of the raw `avx512` feature, so `cargo check --all-targets --all-features` succeeds on non-`x86_64` targets. The obsolete `compile_error!` that rejected `avx512` on non-`x86_64` was removed; the `build.rs` gate still keeps the AVX-512 modules out of those builds.
- **QA summary isolation, script hardening, and CI provenance (`utils/*.sh`, `tests/parity/cpp_parity.rs`, `ci.yml`, `long.yml`):** Isolated `lib_quiet=1` child invocations from `summary.txt`, enforced fail-closed carrier/receipt writes and minimum test counts (`assert_ran_tests 2000`), unified `CARGO_TARGET_DIR: target/long/release` and provenance recording in `long.yml`, and fixed `LogBuffer` snapshot timing in `streaming_adapter_test`.

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
