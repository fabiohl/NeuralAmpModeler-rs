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
<!-- Check all that apply -->
- [ ] **Neural Network Backbones** (`src/models/wavenet/`, `src/models/lstm/`, `src/models/convnet/`, `src/models/linear/`, `src/models/a2/`)
- [ ] **SIMD & Math Kernels** (`src/math/gemm/`, `src/math/activations/`, `src/math/dsp/`, `src/math/lstm/`, `src/math/wavenet/`)
- [ ] **DSP Pipeline & Stages** (`src/dsp/pipeline/`, `src/dsp/cabsim/`, `src/dsp/resampler/`, `src/dsp/gate.rs`)
- [ ] **Model Loading & Formats** (`src/loader/`, `.nam` JSON parser, `.namb` binary loader)
- [ ] **Concurrency & Telemetry** (`src/common/spsc/`, `src/common/diagnostics/`)
- [ ] **Benchmarks & Quality Gates** (`benches/`, `src/testing/`, `docs/quality-contract.json`)
- [ ] **Documentation & Examples** (`docs/`, `examples/`, `README.md`)

---

## Real-Time Audio Safety Checklist (RT-Safe)
<!-- The audio thread runs at high priority with strict sub-millisecond deadlines. RT safety is non-negotiable. -->

- [ ] **Zero Dynamic Heap Allocations in Audio Path**: The audio path (`process()`, inner DSP stages, inference loops) makes zero calls to the heap allocator (`malloc`, `Box`, `Vec`, `String`, `Arc::new()`, `format!()`). Any resource swap happens off-RT via SPSC GC rings.
- [ ] **Lock-Free & Non-Blocking**: No mutexes, RwLocks, synchronization primitives, or thread-parking mechanisms exist in the audio thread.
- [ ] **Zero Blocking I/O & File Operations**: No filesystem access, network I/O, or synchronous standard output (`println!`, `eprintln!`) on the audio thread.
- [ ] **Zero `log::*` Invocations in Audio Hot-Path**: No `log::*` macros are executed inside `process()` or inner sample loops (anomalies/events are signaled via atomic bitmasks `RtStatusFlags`).
- [ ] **Static Bounds-Check Elimination**: Loops in DSP hot-paths are structured using `.chunks_exact()`, `.chunks_exact_mut()`, and `zip` to guarantee compile-time bounds-check elimination, with no `unwrap()` or `expect()`.
- [ ] **Denormal / Subnormal Protection**: Subnormal float handling is protected via FTZ/DAZ or deterministic dither offset (`-220 dBFS`).

---

## SIMD & Mathematical Vectorization Checklist (target: x86-64-v3)
<!-- The engine enforces an x86-64-v3 (AVX2, FMA, BMI2) mandatory minimum baseline in .cargo/config.toml. -->

- [ ] **Unconditional x86-64-v3 Baseline**: No runtime `is_x86_feature_detected!("avx2")` checks or scalar fallback branches on the primary execution path.
- [ ] **Data Alignment**: 64-byte memory alignment is preserved for SIMD buffers and weight matrices (`AlignedVec`).
- [ ] **Instruction-Level Parallelism (ILP)**: Inner products utilize multiple accumulators to saturate FMA execution ports.
- [ ] **FastMath Error Budget**: If activation functions or approximations are modified, accuracy and error budgets comply with `docs/fastmath-approximations.md` and `docs/audio_fidelity_map.md`.

---

## Numerical Parity, Oracles & Testing Checklist
<!-- Both the C++ NAMCore Parity oracle and the f64 Numerical Oracle must be satisfied. -->

- [ ] **C++ NAMCore Parity**: Parity tests (`tests/parity/cpp_parity.rs`) pass within calibrated SNR/ESR/MSE/MR-STFT thresholds.
- [ ] **f64 Numerical Oracle**: Ideal numerical behavior is maintained against the reference implementation without unexpected float drift.
- [ ] **Quality Contract Compliance**: If quality thresholds are affected, measurement comments are provided and `docs/quality-contract.json` is validated.
- [ ] **Parser Fuzzing & Property Tests**: If loaders or deserializers were modified, proptest fuzzing suites (`tests/models/proptest_parsers.rs`) pass.

---

## Pre-Submission Verification Suite (Mandatory)
<!-- Run these verification scripts from the crate root before opening or marking PR ready for review: -->

```bash
utils/lints.sh        # Static analysis, fmt, clippy (-D warnings), doc-tests, SPDX
utils/tests-quick.sh  # Agile testing, golden vectors, C++ parity gates, parser fuzzing
```

- [ ] **`utils/lints.sh` Passed**: 100% clean across all feature permutations (`--all-features`, `--no-default-features`, `dynamic-engine`, `stereo`, `testing`, `heap-audit`).
- [ ] **`utils/tests-quick.sh` Passed**: Executed once as final validation; all golden vectors, C++ parity gates, and proptest parser fuzzers pass without regressions.
- [ ] **License & SPDX Headers**: All new and modified files include the Apache-2.0 SPDX header and copyright notice:

  ```text
  SPDX-License-Identifier: Apache-2.0
  Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
  ```

- [ ] **Subproject Self-Containment**: No references or links escape the repository root, and the crate remains purely host-agnostic.
- [ ] **Undocumented Clippy Allows**: Any `#[allow(clippy::...)]` attribute includes an explanatory justification comment on the preceding line.
