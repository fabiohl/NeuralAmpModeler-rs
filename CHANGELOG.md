<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [4.1.0] - Unreleased

Minor feature release adding allocation-free runtime weight updates, host-precomputed FiLM modulation, real-time incremental `ContainerModel` transitions, stricter `.namb` layout/CRC validation, and pipeline/API hygiene fixes.

### Added [4.1.0]

- **Allocation-free runtime A2 weight updates ([`docs/a2-runtime-weight-updates.md`](docs/a2-runtime-weight-updates.md)):** Added in-place `update_weights` across `WaveNetA2::<3>`, `WaveNetA2::<8>`, `WaveNetA2Dyn`, and `WaveNetA2Cascade` (plus `runtime_weight_count` and typed `#[non_exhaustive]` `A2WeightUpdateError`), validating count and finiteness atomically while preserving causal streaming state and ring cursors.
- **Host-precomputed FiLM modulation ([`docs/precomputed-film.md`](docs/precomputed-film.md)):** Added allocation-free block-rate FiLM override APIs on `FiLMLayer` (`set_precomputed_modulation`, `has_precomputed_modulation`, `use_conditioned_modulation`) and atomic model-level helpers on `WaveNetA2<CH>`, `WaveNetA2Dyn`, and `WaveNetA2Cascade` (`activation_modulation_size`, `set_activation_modulation`, `resume_activation_conditioning`, and `#[non_exhaustive]` `FiLMModulationError`).
- **Bounded SPSC parking-lot flush:** Added `common::spsc::flush_gc_parking_lot` for latch-gated, bounded flushing of parked GC items back into the SPSC channel without real-time drops.
- **`ContainerModel` RT preparation accessors:** Added `preparing_index`, `requested_index`, and `is_preparing` on `ContainerModel` to inspect in-flight split-prewarm and queued transitions.
- **`.namb` layout validation helpers & typed errors:** Added `NambHeader::try_get_layout`, `layout_name`, `v1_crc_is_missing`, and `NambError` variants `UnsupportedLayout`, `LayoutArchitectureMismatch`, and `LayoutGeometryMismatch`.

### Changed [4.1.0]

- **Zero-allocation RT `ContainerModel::set_slimmable_size`:** Real-time calls (`rt_status = Some(...)`) now perform incremental split-prewarm (`Preparing -> Crossfading`) with per-sample linear crossfade ramps and replacement queuing instead of synchronous child `reset()`, leaving `RT_STATUS_SLIMMABLE_RESET_FAILED` reserved; off-RT calls (`rt_status = None`) retain synchronous reset semantics.
- **Fail-closed `.namb` parsing & encoding:** `parse_namb_typed` validates CRC32 before parsing JSON metadata and rejects unknown or architecture/geometry-incompatible `WeightsLayout` headers; `encode_namb` rejects unsupported versions, non-`Original` v1 layouts, v1 `crc32 == 0` sentinel collisions, and non-`Original` re-transpositions, and enforces little-endian targets at compile time.
- **A2 dynamic prewarm geometry bound:** `WaveNetA2Dyn::new` explicitly rejects geometries with `input_channels > 512` or `head_size > 512` (the fixed real-time prewarm scratch bound) instead of skipping stabilization.
- **`testing` feature isolation:** `panic_hook::crash_directory()` no longer redirects production crash directories when `feature = "testing"` is enabled (redirection is scoped to `#[cfg(test)]`).
- **Bounded split-prewarm defaults:** `NamModel::prewarm_step` default fallback now behaves all-or-nothing when the caller's budget is smaller than `prewarm_samples()`, and `ContainerModel::prewarm_step` enforces an aggregate per-callback budget across incomplete submodels.
- Updated `anyhow` minimum version to `1.0.105` and refreshed patch dependency resolutions in `Cargo.lock`.

### Fixed [4.1.0]

- **Mono output-stage mirroring:** `apply_output_stage_inner` and `apply_output_stage_sanitized` now mirror the final processed/sanitized left channel into the right channel at the end of the output stage rather than mid-pipeline.
- **Preserved A2 JSON metadata on `.namb` round-trip:** Implemented custom `Serialize` on `NamLayerConfig` preserving raw JSON keys (`layer_raw`) overlaid by typed fields so A2 topology keys survive `.namb` re-encoding.
- **Interleaved-4 WaveNet head transposition:** Aligned `transpose_wavenet_interleaved4` with the decoder (`head_size` default `1`, Conv1D transposition for `head_kernel_size > 1` or `head_dilation > 1`, and explicit rejection of A2 topologies).
- **Prewarm & container reset accounting:** `SequentialModel::prewarm_step` debits only actually processed frames bounded by `stage_frames` and avoids fictitious convergence at zero capacity; `ContainerModel::reset` cancels in-flight transitions before clearing inactive submodels.
- **Unified loader error code diagnostics:** Aligned the logged `NamErrorCode` with `LoadError::code()` in `parse_model_bytes` while preserving v4.0.0 `LoadError` variants and `Display` output.
- **Adaptive crossfade callback stack usage:** Moved WaveNet double-pass crossfade snapshot scratch arrays from the audio callback stack into `AdaptiveCompute`.

## [4.0.0] - 2026-11-23

This major release directly supersedes `2.0.0` (`3.0.0` was previously published and yanked on crates.io and was never tagged from this series). It resolves `MirroredBuffer<T>` physical-aliasing soundness, eliminates audio-thread `drop_glue`, enforces fail-closed contracts across `A2Layer`, `.nam` JSON loading, WAV/CabSim RIFF validation, and `StreamingResampleBuffer`, adds Experimental Verifiable Windows 11 (`x86_64` and `aarch64`) support, introduces reference-compatible input calibration (`OutputMode`) and `MusicalNoiseGate`, expands WaveNet A1/A2 parity, and refreshes verified latency baselines. Read **Breaking Changes and Migration** before upgrading from `2.0.0`.

### Highlights [4.0.0]

- **`MirroredBuffer<T>` soundness and fail-closed `A2Layer`:** Removed `DerefMut` from `MirroredBuffer<T>` in favor of safe bounded mutable windows (`window_mut`, `try_window_mut`, `write_slice_at`, `IndexMut<usize>`), and hardened `A2Layer::process_single_frame` alongside the new fallible `A2Layer::try_process_single_frame` (`A2LayerError`).
- **Experimental Verifiable Windows 11 support (`x86_64` and `aarch64`):** Added native Win32 placeholder virtual-memory mirroring, VEH-protected ARM64 timer probing with `std::time::Instant` fallback, portable path redaction, `tools/preflight`, and native CI parity achieving `0 ULP` bitwise identity across Linux, macOS, and Windows.
- **Input calibration & musical noise gate:** Added reference-compatible `OutputMode` dBu calibration (`src/dsp/calibration.rs`) and the branchless, zero-`libm` quadratic downward expander `MusicalNoiseGate` (`src/dsp/musical_gate.rs`), re-exported through `prelude`.
- **Zero-`drop_glue` real-time hardening:** Eliminated RT-thread `drop_glue` via `park_without_drop` in the SPSC GC cascade and swap drain, `Arc::ptr_eq` no-op `inject_rt_status_rt`, allocation-free scalar WaveNet adaptive targeting, and bounded SPSC swap coalescing.
- **Expanded WaveNet A1/A2 parity & faster baselines:** Added canonical multi-stage `PostStackHead`, per-layer `kernel_sizes` in WaveNet A1, `LinearModel` prewarm parity with NAMCore PR #352, stricter `.nam`/WAV validation, and refreshed SLA baselines (`WaveNet Standard` `41.26 µs`, `WaveNet Feather` `14.63 µs`, `LSTM 1×16` `6.44 µs`).

### Breaking Changes and Migration [4.0.0]

- **`MirroredBuffer<T>` `DerefMut` removed (`src/dsp/mirror_buf.rs`):** Removed `impl DerefMut for MirroredBuffer<T>`, which previously exposed a `2N` mutable slice and allowed safe code (e.g. `split_at_mut(size())`) to create aliasing `&mut [T]` references backed by the same physical memory. `Deref<Target = [T]>` (`2N` shared reads) is preserved; migrate mutable callers to `IndexMut<usize>`, `window_mut(offset, len)`, `try_window_mut(offset, len)` (enforcing `len <= size()`), `write_slice_at`, `fill`, or allocation-free `copy_within`.
- **`A2Layer::process_single_frame` preconditions (`src/models/a2/layer.rs`):** Invalid inputs — including active `head1x1` projections (`HeadProjectionActive`), grouped input mixins (`mixin_groups != 1`), grouped `layer1x1` residuals (`l1x1_groups != 1`), mixin or residual widths differing from output channels, buffer/weight/bias/grouped-layout bounds violations, integer overflows, and dilations exceeding `isize::MAX` (for every kernel size) — now trigger `debug_assert!` in debug builds and avoid buffer mutation in release builds. Valid inputs produce bit-for-bit identical outputs. Migrate to `A2Layer::try_process_single_frame` (`A2LayerError`, `#[non_exhaustive]`) for typed precondition errors.
- **WaveNet post-stack head and A1 geometry structs:**
  - `PostStackHead` (`src/models/wavenet/post_stack_head.rs`) replaced public `conv`, `activation`, and `state` fields and `process_block_internal` with canonical multi-stage `stages` (`Vec<HeadStage>`) and `canonical` (`bool`) fields.
  - `HeadConfig` (`src/loader/nam_json/model.rs`) added `kernel_sizes`, and `FreeWavenetGeometry` (`src/loader/nam_json/topology/wavenet/mod.rs`) added `layer_kernel_sizes` to represent per-layer kernel sizes in WaveNet A1 geometry.
- **Stricter `.nam` JSON loader and WAV/CabSim RIFF validation:**
  - `metadata.training` shares a single `1,048,576`-unit cost budget across the tree with a container nesting limit of 16 (depth 17 is rejected); `num_layers` and layer activation arrays are validated before replication (`num_layers > 64`, PReLU vectors `> 512` slopes, and aggregate PReLU cost `> 64 × 512` are rejected; `0` remains accepted); WaveNet gating, PReLU, FiLM, and `condition_dsp` channel parsing fail closed on mismatched geometries.
  - Converged WAV and CabSim RIFF parsing (`src/loader/wav.rs`, `src/dsp/cabsim/ir_parse.rs`) rejects `riff_size < 4` or exceeding the buffer as well as chunks crossing the container end, while ignoring trailing bytes after the container end and tolerating odd-sized terminal `data` chunks only when their payload ends exactly at the container end. `WAVE_FORMAT_EXTENSIBLE` (`0xFFFE`) enforces `fmt_size >= 40`, `cbSize >= 22`, exact `EXTENSIBLE_GUID_TAIL`, `valid_bits == bits_per_sample`, `block_align`, and `byte_rate`.
- **Internal allocator metadata (`src/math/common/huge_alloc.rs`):** Marked `AllocInfo` and `deallocate_huge` as `#[doc(hidden)]` (paired exclusively with `pub(crate) allocate_huge_pages`).

### Added [4.0.0]

- **Experimental Verifiable Windows 11 support (`x86_64-pc-windows-msvc` and `aarch64-pc-windows-msvc`):**
  - Win32 `MirroredBuffer<T>` backend (`src/dsp/mirror_buf/windows.rs`) using `VirtualAlloc2` + `MapViewOfFile3` placeholders, 6-stage `S0`–`S5` transactional rollback, immediate `CloseHandle`, and dual `UnmapViewOfFile` on `Drop`, preserving `size_of::<MirroredBuffer<T>>() == 16`.
  - ARM64 timer probe (`src/common/tsc.rs`) protected by a Vectored Exception Handler (`AddVectoredExceptionHandler`) with automatic fallback to `std::time::Instant`.
  - Portable PII, Windows drive-path, and UNC path redaction (`src/common/diagnostics/format.rs`, `src/common/panic_hook.rs`, `src/loader/loaded_model_pair.rs`).
  - Workspace `tools/preflight` (`nam-preflight`) diagnostic binary, `.gitattributes` (`eol=lf`), and native CI jobs (`windows-2025`, `windows-11-arm`) integrated into `cross-platform-parity` (`0 ULP` bitwise identity across all 7 canonical models).
- **Input calibration, musical noise gate, and CabSim helpers:**
  - `src/dsp/calibration.rs`: `OutputMode` (`Raw`, `Normalized`, `Calibrated`), `CalibrationGains`, `ModelCalibrationMeta`, `compute_calibration_gains`, and `sanitize_calibration_level_dbu`, re-exported in `prelude`.
  - `src/dsp/musical_gate.rs`: reference-compatible quadratic downward expander `MusicalNoiseGate` with off-RT `prepare` and branchless zero-`libm` hot-path `log10`/`10^x` approximations, re-exported in `prelude`.
  - `src/dsp/cabsim/mod.rs`: `validate_single_channel_model` and `compute_nam_tail_budget`.
- **Fallible A2 single-frame path (`src/models/a2/layer.rs`):** Added `A2Layer::try_process_single_frame` and `#[non_exhaustive]` `A2LayerError`, verifying the requested instruction set against the host before kernel dispatch.
- **Real-time zero-drop & pipeline telemetry APIs:**
  - `StaticModel::inject_rt_status_rt` and `WaveNetA2::inject_rt_status_rt` with `Arc::ptr_eq` no-op semantics to prevent `Arc::drop_slow` on the audio thread (`src/models/static_model.rs`, `src/models/a2/model/static/mod.rs`).
  - Allocation-free scalar WaveNet adaptive channel targeting (`wavenet_mono_rt`, `wavenet_rt_target_ch`, `clear_wavenet_registration_rt` in `src/dsp/adaptive.rs`).
  - `DspBridge` raw-pointer atomic readers (`written_gen_raw`, `consumed_gen_raw`, `drain_dropped_frames_raw` in `src/dsp/pipeline/bridge.rs`) and `write_bridge_stamped` (`src/dsp/pipeline/stages/bridge.rs`) for capture-timestamped output.
- **WaveNet A1 & Linear parity:** Canonical multi-stage `PostStackHead` (`src/models/wavenet/post_stack_head.rs`), per-layer `kernel_sizes` (`src/loader/nam_json/topology/wavenet/mod.rs`), and `LinearModel` prewarm sample parity matching NAMCore PR #352.
- **Expanded `NamErrorCode` catalog (`src/common/diagnostics/error_codes.rs`):** Added `E2004` (`OsScratchCapacity`), `E2111`–`E2115` (PM-QoS and sink probes), `E2120` (`PageSizeQueryFailed`), `E2130`–`E2131` (quiescence barriers), `E2303` (`RtGetSchedFailed`), `E2305`–`E2308` (`NonFiniteInputContained`, `CpuOutOfBounds`, `RateMismatch`, `RtSchedFailed`), and `E3104`–`E3105` (`SlimmableSliceFailed`, `SlimmableResetFailed`).
- **Examples, docs, and QA tooling:** Added `examples/streaming_partial.rs`, a seam doc-test on `MirroredBuffer<T>::try_window_mut`, `rust-toolchain.toml` pinning the stable channel, and fail-closed `tools/qa` tools (`nam_simd_parity` for AVX2 ↔ AVX-512 JSONL reconciliation, `nam_skips` for JSONL skip enforcement, and third-party mirror & contract verification commands).

### Changed [4.0.0]

- **Real-time SPSC hardening (`src/common/spsc/`):** Added `park_without_drop` (`gc.rs`, `swap.rs`) so parking payloads in empty slots emits no `drop_glue` for overwritten `None` values on the audio thread, and bounded swap coalescing (`swap.rs`) so a stale head no longer suppresses a valid deferred candidate.
- **`StreamingResampleBuffer` backpressure, FIFO ordering, and typed errors (`src/dsp/resampling/streaming_adapter.rs`):** Bounded `drain_model_to_out` to `out_fifo.free() + discard_remaining` with `unpop` preservation and zero-progress termination; resumed draining during `pull_output`; restricted `process` bypass fast-path to empty FIFOs; signaled `StreamingResult::underflow` immediately when `produced < drained` without fabricating output samples; documented underflow and consumption contracts; and switched construction to `NamResampler::new_simple_typed` / `new_linear_simple_typed` so invalid rates return `NamErrorCode::ResamplerBuildFailed` (`E2200`) instead of `OutOfMemory`.
- **Cross-platform `tsc`, `huge_alloc`, `mirror_buf`, and diagnostics cleanup:**
  - Removed unreachable `QueryPerformanceCounter` dead code and `"Win32_System_Performance"` from `Cargo.toml` in Windows ARM64 `read_counter()` (`src/common/tsc.rs`).
  - Moved POSIX `memfd_create`/`shm_open` backing helpers from `huge_alloc.rs` to `src/dsp/mirror_buf/posix.rs`, restricted Linux `mmap` huge-page paths to `target_os = "linux"`, routed macOS allocations through the global allocator, and documented the Linux ≥ 6.3 (`MFD_NOEXEC_SEAL`) requirement (#6).
  - Parameterized `baseline_isa_label()` for `x86_64` vs `aarch64`, added `uname`/`RtlGetVersion` kernel version detection in `src/common/diagnostics/system_info.rs`, scoped `emit_irq_advisory` to Linux, and implemented `sysctlbyname` CPU/OS reporting on macOS in `tools/preflight`.
- **Refreshed Quality Contract baselines (`docs/quality-contract.json`):** Updated verified latency envelopes (`WaveNet Standard` `41.26 µs`, `WaveNet Lite` `51.86 µs`, `WaveNet Feather` `14.63 µs`, `WaveNet Nano` `11.97 µs`, `WaveNet A2 Full` `18.68 µs`, `WaveNet A2 Lite` `13.76 µs`, `LSTM 1×16` `6.44 µs`, `LSTM 2×8` `6.91 µs`, `ConvNet` `9.30 µs`, `Full Pipeline Base` `41.81 µs`, `Full Pipeline HQ 4× OS` `170.23 µs`).
- **Examples, modularization, and `docs.rs` checks:** Propagated errors with `?` preserving `NamErrorCode` in `examples/cabsim.rs` and `examples/offline_render.rs`; modularized `src/bin/nam_perf_gate/` and `src/testing/catalog/`; refined `doc(cfg)` on `rt_hardening`; extended `package.metadata.docs.rs` and `utils/docsrs-check.sh` across all 5 targets; and updated `utils/consumer-smoke.sh` to validate extracted `.crate` tarballs.

### Fixed [4.0.0]

- **AVX-512 `CH=8` `film.active_mask() == 0` fast-path (`src/models/a2/conv1d_ch8/simd.rs`):** Removed the premature early return that skipped the dilated convolution loop, restoring `x86_64-linux` vs `x86_64-avx512` parity on `wavenet_a2_full` (`SNR = 129.14 dB`, `ESR = 1.22e-13`).
- **`testing::wav::write_wav_f32` (`src/testing/wav.rs`):** Fixed the emitted RIFF container size (`36 + data`) to match canonical WAV headers.
- **`--all-targets --all-features` on `aarch64`:** Gated AVX-512 tests and benches on `cfg(nam_avx512)` emitted by `build.rs` instead of the raw `avx512` feature, and removed the obsolete `compile_error!` on non-`x86_64` targets.
- **Cross-platform test isolation and temp paths:** Replaced hardcoded `/tmp/*.wav` paths in loader/CabSim tests with `std::env::temp_dir()` and RAII cleanup guards, isolated global `panic_hook` suppression tests from parallel library panics, and fixed `LogBuffer` snapshot timing in `streaming_adapter_test`.
- **QA tooling, scripts, and CI provenance:** Fixed `nam_quality dashboard --save` to refresh contract provenance, fidelity, performance, and `worst_esr_parity` atomically and fail closed if `regression_gate` did not pass (unless `--fidelity-only`); captured real compiler provenance in parity vectors; stubbed IDE-only `source_group` when configuring NAMCore with CMake 4.5+; and hardened `utils/` QA scripts (`-h`/invalid usage emit one `RESULT:` without overwriting previous carriers, heap audit runs with `--no-fail-fast`, capability `GAP` requires explicit authorization, `lib_quiet=1` isolates child logs from `summary.txt`, and `tests-quick.sh` enforces `assert_ran_tests 2000`).

---

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

[4.1.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v4.0.0...v4.1.0

[4.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v2.0.0...v4.0.0

[2.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.1...v2.0.0

[1.0.1]: https://github.com/fabiohl/NeuralAmpModeler-rs/compare/v1.0.0...v1.0.1

[1.0.0]: https://github.com/fabiohl/NeuralAmpModeler-rs/releases/tag/v1.0.0
