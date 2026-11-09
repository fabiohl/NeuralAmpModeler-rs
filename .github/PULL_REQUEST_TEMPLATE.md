<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Description of Changes
<!-- Provide a clear, concise summary of what was changed and the technical rationale behind it. -->

## Linked Issues
<!-- e.g. Fixes #123, Closes #456, or Relates to #789 -->
Fixes #

## Subsystem & Scope
<!-- Check all that apply. For documentation, packaging, or off-RT tooling changes, technical checklists below may be marked N/A. -->
- [ ] **Neural Network Backbones** (`src/models/wavenet/`, `src/models/lstm/`, `src/models/convnet/`, `src/models/linear/`, `src/models/a2/`, `src/models/sequential.rs`, `src/models/container.rs`)
- [ ] **SIMD & Math Kernels** (`src/math/gemm/`, `src/math/activations/`, `src/math/dsp/`, `src/math/lstm/`, `src/math/wavenet/`, `src/math/conv/`)
- [ ] **DSP Pipeline & Stages** (`src/dsp/pipeline/`, `src/dsp/cabsim/`, `src/dsp/resampler/`, `src/dsp/adaptive.rs`, `src/dsp/oversample.rs`, `src/dsp/gate.rs`)
- [ ] **Model Loading & Formats** (`src/loader/`, `.nam` JSON parser, `.namb` v1/v2 binary formats, WAV IR loader, dynamic dispatcher)
- [ ] **Real-Time Host Hardening & OS Thread Isolation** (`src/rt_hardening/`, CPU affinity, PM QoS, `SCHED_FIFO`, `mlockall`, THP disable)
- [ ] **Concurrency & Telemetry** (`src/common/spsc/`, `src/common/diagnostics/`, `src/common/tsc.rs`, `src/dsp/telemetry.rs` / HdrHistogram)
- [ ] **Benchmarks & Quality Gates** (`benches/`, `src/testing/`, `tests/`, `docs/quality-contract.json`, `docs/benchmarks.md`)
- [ ] **Documentation & Examples** (`docs/`, `examples/`, `README.md`)

---

## Real-Time Audio Safety Checklist (RT-Safe)
<!-- Applicable if modifying audio processing paths. Skip or mark N/A for non-RT, documentation, or tooling PRs. -->

- [ ] **N/A**: This PR does not touch the real-time audio hot-path.
- [ ] **Zero Dynamic Heap Allocations in Audio Path**: The audio path (`process()`, inner DSP stages, inference loops) makes zero calls to the heap allocator (`malloc`, `Box`, `Vec`, `String`, `Arc::new()`, `format!()`). Any resource swap happens off-RT via SPSC GC rings.
- [ ] **Lock-Free & Non-Blocking**: No mutexes, RwLocks, synchronization primitives, or thread-parking mechanisms exist in the audio thread.
- [ ] **Zero Blocking I/O & File Operations**: No filesystem access, network I/O, or synchronous standard output (`println!`, `eprintln!`) on the audio thread.
- [ ] **Zero `log::*` Invocations in Audio Hot-Path**: No `log::*` macros are executed inside `process()` or inner sample loops (anomalies/events are signaled via atomic bitmasks `RtStatusFlags`).
- [ ] **Static Bounds-Check Elimination**: Loops in DSP hot-paths are structured using `.chunks_exact()`, `.chunks_exact_mut()`, and `zip` to guarantee compile-time bounds-check elimination, with no `unwrap()` or `expect()`.
- [ ] **Multichannel API Contract & Buffer Safety**: Multichannel routing (`NamModel::process_multichannel_in_place`) guarantees zero allocation for up to 16 channels via stack pointer tables; asymmetric buffer slices clamp safely (`output.len().min(input.len())`) with zero panics.
- [ ] **Cache Isolation & False Sharing Prevention**: Cross-thread shared structures (RT ↔ Main) enforce 128-byte cache line alignment (`#[repr(align(128))]`).
- [ ] **Memory Ordering Determinism**: RT thread strictly uses `Ordering::Relaxed` for metrics/counters and `Ordering::Acquire`/`Release` for SPSC rings; never `SeqCst` on the audio thread.
- [ ] **Denormal / Subnormal Protection**: Subnormal float handling is protected via FTZ/DAZ or deterministic dither offset (`-220 dBFS`).

---

## SIMD & Mathematical Vectorization Checklist (target: x86-64-v3)
<!-- Applicable if modifying math kernels, activations, or neural inference. Skip or mark N/A otherwise. -->

- [ ] **N/A**: This PR does not touch SIMD kernels, math approximations, or vectorization.
- [ ] **Unconditional x86-64-v3 Baseline**: No runtime `is_x86_feature_detected!("avx2")` checks or scalar fallback branches on the primary execution path.
- [ ] **Data Alignment**: 64-byte memory alignment is preserved for SIMD buffers and weight matrices (`AlignedVec`).
- [ ] **Instruction-Level Parallelism (ILP)**: Inner products utilize multiple accumulators to saturate FMA execution ports.
- [ ] **FastMath Error Budget**: If activation functions or approximations are modified, accuracy and error budgets comply with `docs/fastmath-approximations.md` and `docs/audio_fidelity_map.md`.

---

## Numerical Parity, Oracles & Testing Checklist
<!-- Both the C++ NAMCore Parity oracle and the f64 Numerical Oracle are validated automatically in GitHub CI. -->

- [ ] **N/A**: This PR does not touch DSP models, weights, or loaders.
- [ ] **C++ NAMCore Parity**: Parity tests (`tests/parity/cpp_parity.rs`) pass within calibrated SNR/ESR/MSE/MR-STFT thresholds (validated locally or by GitHub CI `cpp-parity` job).
- [ ] **f64 Numerical Oracle**: Ideal numerical behavior is maintained against the reference implementation (`tests/parity/reference_oracle_f64.rs`) without unexpected float drift.
- [ ] **Block-Invariance & Meta-Coherence**: Block slicing equivalence (`tests/models/block_invariance_test.rs`) and metadata coherence suites (`tests/models/meta_coherence.rs`) pass.
- [ ] **Quality Contract Compliance**: If quality thresholds are affected, measurement comments are provided and `docs/quality-contract.json` is validated via `utils/quality-dashboard.sh --check docs/quality-contract.json`.
- [ ] **Parser Fuzzing & Property Tests**: If loaders or deserializers were modified, proptest fuzzing suites (`tests/models/proptest_parsers.rs`) pass.

---

## Verification & Quality Gates (Local & GitHub CI)
<!--
Responsible Verification Model:
- Fast Local Feedback: Run cargo check and targeted tests for quick local iteration.
- GitHub CI Automation: The full cross-platform matrix (x86_64, Linux ARM64, macOS ARM64, MSRV 1.99.0, C++ parity oracle, and consumer smoke tests) runs automatically in CI on every push and PR.
- Friendly Tip for Contributors: If you don't have C++ compilers, CMake, or ARM hardware locally, open a DRAFT PR and let GitHub Actions run the full verification matrix for you!
-->

### 1. Recommended Local Checks (Fast Feedback)

```bash
# Fast check during local development:
cargo check --locked --all-targets

# Targeted test for changed components (example):
cargo test --lib <module>
```

### 2. Optional Full Local Suite (If environment allows)

```bash
# Static analysis (8 phases, fmt, clippy -D warnings, policies, budget):
utils/lints.sh

# Agile QA test suite (debug mode, non-ignored targets):
utils/tests-quick.sh
```

- [ ] **Targeted Local Checks Passed**: Code compiles cleanly without warnings (`cargo check --all-targets`) and unit tests for modified components pass.
- [ ] **Full Local Suite Executed (Optional)**: Successfully ran `utils/lints.sh` and/or `utils/tests-quick.sh` (or deferring full matrix to GitHub CI via Draft PR).
- [ ] **Automated GitHub CI**: All checks pass on GitHub Actions (x86_64 Linux lints/quick, native ARM64 Linux/macOS, MSRV 1.99.0, C++ parity oracle, consumer smoke tests).
- [ ] **License & SPDX Headers**: All new and modified files include the Apache-2.0 SPDX header and copyright notice:

  ```text
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
  ```

- [ ] **Subproject Self-Containment**: No references or links escape the repository root, and the crate remains purely host-agnostic.
- [ ] **Undocumented Clippy Allows**: Any `#[allow(clippy::...)]` attribute includes an explanatory justification comment on the preceding line.
