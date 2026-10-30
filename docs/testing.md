<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Test Architecture

This document defines the organization, execution tiers, and validation principles of the test and static defense suite for `NeuralAmpModeler-rs`. The executable code (`tests/`, `src/**/*_test.rs`) and typed receipts are the primary source of truth.

> [!NOTE]
> **Document Scope & Hierarchy:**
>
> - **Static Analysis & Quality Defense:** Static validation across all feature axes, Clippy, doc-tests, and policy checks via [utils/lints.sh](../utils/lints.sh) (§2).
> - **Functional & Correctness Architecture:** Automated test architecture, feature gating, execution phases, and oracle validation via [utils/tests-quick.sh](../utils/tests-quick.sh) and [utils/tests-long.sh](../utils/tests-long.sh) (§3–§6).
> - **Manual Testing & Human Release Certification:** Step-by-step developer functional verification (Tiers 1–3) and the 5-receipt human release certification protocol reside in [functional-tests.md](functional-tests.md).
> - **Performance Regression Gates:** The statistical Criterion benchmarking wall and baseline governance reside in [benchmarks.md](benchmarks.md).
> - **Fixtures & Golden Vectors:** Ground-truth vector generation, model resolution order, and hash manifests reside in [fixtures.md](fixtures.md).
> - **Acoustic & Perceptual Validation:** Metric formulations (ESR, SNR, MR-STFT, THD+N) and perceptual calibration thresholds reside in [perceptual_validation.md](perceptual_validation.md).

---

## 1. Crate Features Taxonomy

The `NeuralAmpModeler-rs` crate defines several features in [Cargo.toml](../Cargo.toml) to configure runtime targets, test capabilities, and diagnostic facilities:

| Feature Name             | Default | Description                          | Active Scope & Modules                                                                                                   |
|:------------------------ |:-------:|:------------------------------------ |:------------------------------------------------------------------------------------------------------------------------ |
| **`dual-mono`**          | Yes     | Independent dual-channel inference   | [src/dsp/pipeline/stages/input.rs](../src/dsp/pipeline/stages/input.rs) (Stereo/dual-mono DSP variants)                  |
| **`testing`**            | No      | Test utilities, signals & generators | [src/testing/](../src/testing/), CLI binaries (`nam_golden_catalog`, `nam_freshness`, `nam_long_receipt`, etc.)          |
| **`heap-audit`**         | No      | Memory allocation interceptor        | [src/common/alloc_audit.rs](../src/common/alloc_audit.rs), [tests/rt_constraints/](../tests/rt_constraints/) heap checks |
| **`pinned`**             | No      | Harness for pinned long-test targets | [tests/rt_pinned.rs](../tests/rt_pinned.rs), [tests/soak_pinned.rs](../tests/soak_pinned.rs); enabled only by `utils/tests-long.sh` and `utils/tests-quick.sh` |
| **`fft-radix4-planner`** | No      | Radix-4 FFT benchmark planner        | [benches/fft_radix4_bench.rs](../benches/fft_radix4_bench.rs) (Radix-4 DIT FFT planning routines)                        |
| **`avx512`**             | No      | Research/measurement AVX-512 kernels | [src/math/activations/](../src/math/activations/), SIMD dispatch tables                                                  |
| **`rt-hardening`**       | No      | Real-time OS host tuning (Linux, macOS) | [src/rt_hardening/](../src/rt_hardening/) (Linux: THP, mlockall, SCHED_FIFO, DAZ/FTZ, IRQ affinity; macOS: QoS, time constraint, mlock_region, workgroups) |
| **`on-demand`**          | No      | Internal developer diagnostics       | Internal-only fixture generation and profiling harnesses (never enabled by CI or published test instructions)             |

---

## 2. Static Analysis & Quality Defense (`utils/lints.sh`)

[`utils/lints.sh`](../utils/lints.sh) is the **static first line of defense** of the project. It runs quickly (~45 seconds) and acts as an uncompromising quality gate that developers should execute regularly during active coding, before staging any commit, and as Receipt #1 in formal release certification.

Unlike test runners, `lints.sh` does not evaluate numerical audio outputs. Instead, it statically proves compilation cleanliness, zero-warning compliance, documentation hygiene, and policy conformance across the entire feature matrix.

### 2.1 The Eight Sequential Verification Phases

1. **Formatting:** Runs `cargo fmt --all`, or `cargo fmt --all -- --check` when `CI=true`.
2. **Compilation:** Runs `cargo check --locked --workspace --all-targets` with four configurations: defaults; `--no-default-features`; `--all-features`; and `--no-default-features --features testing,heap-audit,fft-radix4-planner,rt-hardening`.
3. **Clippy:** Runs the same four configurations with `-D warnings`.
4. **Cross-target Clippy:** Requires `aarch64-unknown-linux-gnu` and runs `cargo clippy --locked --lib --target aarch64-unknown-linux-gnu -- -D warnings`.
5. **Documentation:** Runs `cargo doc --locked --no-deps --all-features` with rustdoc warnings denied. Doc-tests belong to the test runner, not this phase.
6. **SPDX:** Checks tracked source files for an accepted SPDX identifier.
7. **Policies:** Checks `tests/common/`, lint-suppression comments, `doc(cfg)` features, and README version and MSRV.
8. **Script budget:** Checks declared line budgets for shell scripts in `utils/` and `tests/fixtures/`.

### 2.2 Developer Usage & Flags

```bash
# Standard interactive execution
./utils/lints.sh
```

### 2.3 docs.rs Parity Check (Opt-in, Nightly)

The crate page published by docs.rs is rendered by **nightly** rustdoc with `--cfg docsrs`. Documentation diagnostics that only nightly reports — broken intra-doc links, redundant explicit link targets, and missing `doc(cfg(feature = "..."))` seals — are invisible to the stable `lints.sh` gate (§2.1, phase 5). [utils/docsrs-check.sh](../utils/docsrs-check.sh) mirrors the docs.rs configuration to catch those regressions before publishing.

```bash
# Opt-in: requires an isolated nightly toolchain (never part of the stable gates)
./utils/docsrs-check.sh
```

The script exports `DOCS_RS=1` and `RUSTDOCFLAGS="--cfg docsrs -D warnings"`, derives the feature scope from `package.metadata.docs.rs` in [Cargo.toml](../Cargo.toml), then runs `cargo +nightly doc --locked --lib --no-deps` with that scope (`--lib` mirrors the docs.rs build), once on the host (x86_64) and once with `--target aarch64-unknown-linux-gnu`. `aarch64-apple-darwin`, also declared in the metadata, is not checked here (no macOS host; covered by the `G-C` cross-check). The `nam_avx512` cfg is emitted by `build.rs` only for x86_64 targets, so the AVX-512 modules are absent from the aarch64 pass. Exit `0` means zero rustdoc warnings on both passes; exit `2` is a typed gap (nightly, its aarch64 target, or metadata tooling unavailable); any rustdoc warning fails with exit `1`.

### 2.4 SemVer Baseline Check (Opt-in, Pre-release)

The published SemVer baseline is the **last release available on crates.io** (`0.8.0`), not the local `Cargo.toml` version. Before publishing, compare the current API surface against that baseline with [`cargo-semver-checks`](https://github.com/obi1kenobi/cargo-semver-checks):

```bash
# Release gate: the local version (1.0.0) is a major bump, so every intended
# break is already covered; a clean surface prints "no semver update required".
cargo semver-checks check-release --baseline-rev v0.8.0

# Enumeration aid (evidence only): force a non-major release type to list every
# breaking change relative to 0.8.0, so each one can be marked as intentional.
cargo semver-checks check-release --baseline-rev v0.8.0 --release-type minor
```

`--baseline-rev` resolves the baseline from the local `vX.Y.Z` git tag, avoiding a crates.io round-trip; `--baseline-version 0.8.0` is the registry equivalent. The tool requires an isolated nightly rustdoc toolchain and is an **opt-in, out-of-band** check: it is never part of the stable `utils/lints.sh` gate. Exit `0` from the release-gate form means the declared version bump covers every detected change. The `--release-type` form only selects which checks run; its exit code still reflects the declared version delta, so read the printed `Summary` line (not the exit code) when enumerating breaks. Reports are archived as PR/packaging evidence and are not committed as tool artifacts.

---

## 3. Test Execution Phase Architecture — Two-Axis Model

Test selection and execution are governed by **two orthogonal axes**:

- **Axis A — Rigor (`#[ignore]`):**
  - *Non-ignored:* First line of defense for daily local iteration. Fast, deterministic.
  - *`#[ignore]`*: Exhaustive audits, numerical soak, full proptest sweeps, and long parity matrices executed in nightly or pre-release runs.
- **Axis B — Codegen Path (Debug vs. `--release`):**
  - *Debug (`cargo test`):* Structural tests (logic, parsers, FSM transitions, bitwise determinism) execute with `debug-assertions` active. Float codegen optimizations are irrelevant here.
  - *Release (`cargo test --release`):* Measurement oracles (evaluating floating-point outputs against reference models) must execute under release optimization (`-O3`, FMA contraction, vectorization). Measuring floats in debug checks an unoptimized codegen path that does not reflect production runtime behavior.

```mermaid
graph TD
    L0["Static Defense: utils/lints.sh (8 phases)"] --> Q1["Quick: freshness (nam_freshness artifacts-hard)"]
    Q1 --> Q2["Quick: cargo test --features testing,pinned (debug, non-ignored)"]
    Q2 --> L1["Long: 5 preflights + 8 phases (release, --include-ignored)"]
```

### 3.1 Quick QA Suite (`utils/tests-quick.sh`)

The quick suite executes in ~16–87 seconds (measured: 16–22 s with a warm incremental cache, 64–87 s after a manual `git clean -fdx`; budget ≤ 5 minutes) and runs two sequential phases:

1. **Freshness (`nam_freshness artifacts-hard`):**
   - **Scope:** Verifies golden artifact freshness and integrity via [tools/qa/src/bin/nam_freshness.rs](../tools/qa/src/bin/nam_freshness.rs). Exit code 3 is recorded as the `generator-drift` gap; exit code 0 passes; other nonzero exit codes fail the phase. Generator drift is informational in this mode: the quick suite reports the gap and does not block (`artifacts-hard`), whereas the long `preflight-freshness` gate runs `hard-fail` (below).
2. **Tests (`cargo test --locked --features testing,pinned --no-fail-fast`):**
   - **Scope:** Runs every non-ignored test target in the **debug** profile with the `testing` and `pinned` features. This includes structural/logic/parser/FSM tests, non-ignored numerical and parity checks, the four non-ignored RT/soak tests from the pinned targets, and the `libm_export_guard` ELF symbol gate. Float oracles here are smoke checks only; release-profile measurements belong to the long suite.
   - **Exclusions:** `#[ignore]` tests (soak, RT deadline/jitter, exhaustive audits, and the ignored proptest sweeps) are not invoked by the quick suite.

**Logs and gaps:** the phases write `target/logs/quick-freshness.log` and `target/logs/quick-tests.log`; tests that skip an unavailable capability (C++ oracle, fixtures, non-distributable or community models) append a typed record to `target/logs/skips.jsonl`, which the quick script counts as a `skip-capability:<n>` warning gap (the quick suite has no strict mode; the long suite distinguishes optional records as non-blocking under `--strict`). The quick suite never emits a receipt; it ends with the canonical `RESULT:` line. Every failure path (freshness, tests/compilation, zero tests executed, or an unexpected error) emits the phase `FAIL`, exactly one `RESULT: FAIL`, and the current `target/logs/tests-quick-status.tsv` carrier. The stale carrier is removed at the start of the run, and a carrier write failure is itself reported as a failure instead of preserving an earlier verdict. `utils/tests-long.sh --strict` (which exports `NAM_REQUIRE_CPP_ORACLE=1`) is the fail-closed counterpart for a missing C++ oracle.

### 3.2 Golden Vector Supply Chain & Freshness

Golden vectors validate inference determinism against pre-rendered reference outputs:

- **Freshness Manifest:** [tests/fixtures/golden_gen_build.sh](../tests/fixtures/golden_gen_build.sh) maintains the versioned `.golden_manifest.sha256` digest manifest. The quick `freshness` phase and the long `preflight-freshness` gate verify this integrity using `src/testing/freshness.rs` (`nam_freshness`). Any model modification without golden regeneration triggers a hard failure.
- **Model Resolution Order:** Models are resolved hierarchically: (1) `$NAM_MODELS_DIR`, (2) `third-party/community_models/`, (3) `tests/fixtures/models-nondist/`, (4) `tests/fixtures/models/`. Details are documented in [fixtures.md](fixtures.md).

### 3.3 Binary Defense & Compile-Time Segregation

- **Libm Export Guard:** [tests/libm_export_guard.rs](../tests/libm_export_guard.rs) is a fail-closed ELF symbol gate executed on the linked binary artifact in the `tests-quick.sh` tests phase and the `tests-long.sh` `catch-all` phase. It ensures internal `libm` mathematical symbols are hidden and cannot interpose over system C library symbols. `nm`/`llvm-nm` always run with `LC_ALL=C`/`LANG=C` in the child process, so localized messages never change the verdict. The dynamic table (`nm -D`) is the canonical surface for interposition and must contain a `__libc_start_main` import as proof that `.dynsym` was read; the static table is scanned only when present (a stripped release binary legitimately has none). Missing tools, a failing `nm`, or an absent dynamic witness fail closed.
- **AVX-512 Segregation Contract:** AVX-512 code is strictly segregated at compile time via `#[cfg(feature = "avx512")]`. Default builds monomorphize only `Avx2Math`. The evidence for the absence of EVEX/ZMM code in default release artifacts is a compile-time guarantee (the AVX-512 modules are not compiled without the feature), not binary inspection of `.text`. Runtime hardware capabilities and effective dispatch status are inspected using [`src/bin/simd_probe.rs`](../src/bin/simd_probe.rs) (`cargo run --quiet --bin simd_probe`). Contractual compliance is enforced at compile time and runtime by `tests/isa_contract.rs` (`contract_default_build_never_dispatches_avx512`). **GAP `binary_inspection_evex_zmm_absent`:** no fail-closed `.text` inspection (`nm`/`objdump`) runs on the default release artifact; the historical tool (`utils/verify_no_avx512_release.sh`) was retired in commit `75b0f50ff0`. Closing this gap requires a new fail-closed scanner, which is not part of the current contract.

### 3.4 Continuous Integration Workflow (`.github/workflows/ci.yml`)

Every push and pull request targeting `main` or `dev` triggers the multi-platform CI matrix:

1. **`x86_64-linux` (`ubuntu-26.04`):** Installs `aarch64-unknown-linux-gnu` for the cross-target Clippy gate, executes `./utils/lints.sh` (with `CI=true`) and `./utils/tests-quick.sh`, compiles `nam-qa` in release mode, and generates release parity vectors for canonical reference models.
2. **`aarch64` matrix (`macos-latest` Apple Silicon and `ubuntu-26.04-arm`):** Runs `cargo clippy`, executes `./utils/tests-quick.sh`, compiles `nam-qa` in release mode, and generates release parity vectors on native ARM hardware.
3. **`msrv` (`ubuntu-26.04`):** Resolves `rust-version` from `Cargo.toml` (1.99.0) and verifies `cargo check --locked --all-targets --features testing` on the exact MSRV toolchain. The step enables `pipefail`, preserves the real `cargo`/`tee` exit status, and publishes `target/logs/msrv.log`, `summary.txt` and `msrv-status.tsv` as a diagnostic artifact even on failure; a `cargo check` or log-write failure fails the job.
4. **`cpp-parity` (`ubuntu-26.04`, `needs: x86_64-linux`):** Provisions the upstream C++ reference mirror (`./utils/setup-third-party.sh`), compiles the C++ `namcore_render` binary (`./utils/ensure_namcore_render.sh`), and executes `NAM_REQUIRE_CPP_ORACLE=1 ./utils/tests-quick.sh`. This job exercises only the non-ignored C++ oracle cases of the quick discipline: the 12 multichannel-linear and 3 WAV-IR tests in [tests/parity/cpp_parity.rs](../tests/parity/cpp_parity.rs). The extensive `quick_parity_*`, `live_cross_validation_*`, sequential and cabsim matrices remain `#[ignore]` and run only in the long discipline. The job fails closed if the oracle is unavailable; it is not a certificate of full C++ parity coverage.
5. **`consumer-smoke` (`ubuntu-26.04` and `ubuntu-26.04-arm`):** Installs cross targets and executes `./utils/consumer-smoke.sh` to validate the external-consumer contract.
6. **`cross-platform-parity` (`ubuntu-26.04`, `needs: [x86_64-linux, aarch64]`):** Downloads parity vector artifacts generated in the release profile across all three native runners (`x86_64-linux`, `linux-arm64`, and `macos-arm64`) for the exact same commit and run attempt (`NAM_RUN_ID`). Executes pairwise comparisons via `nam_parity_vector compare`: strict bitwise identity (`0 ULP`) for intra-ISA runs (`linux-arm64` vs `macos-arm64`), and calibrated numerical contract validation (`--max-esr 1e-6 --min-snr-db 100.0`, zero NaNs, zero subnormals) for cross-ISA runs (`x86_64` vs `aarch64`). Missing runner data triggers a fail-closed `GAP`. AVX-512 opt-in execution status is formally declared as `GAP` due to the absence of physical AVX-512 hardware on standard CI runners.

### 3.5 Cross-Platform Parity Vector Protocol (`NAMPARV1` & `nam_parity_vector`)

To ensure bit-level determinism and numerical parity across differing CPU architectures (x86-64-v3, aarch64 NEON, and Apple Silicon), NeuralAmpModeler-rs establishes a binary parity vector protocol (R-07, R-13, S5-T01, S5-T02):

- **Binary Specification:** Vectors begin with the 8-byte magic `NAMPARV1` followed by a 32-bit LE version, length-prefixed JSON provenance header, 32-byte SHA-256 payload checksum, and contiguous IEEE-754 `f32` samples in Little-Endian format.
- **Bitwise Precision:** Parity evaluation compares raw IEEE-754 bit representations (`f32::to_bits()`). Signed zeros (`+0.0` vs `-0.0`) have different bit encodings (`0x00000000` vs `0x80000000`) and are strictly flagged as mismatches.
- **FP Anomaly Policy:** NaNs and subnormals are never normalized or flushed silently during vector export or comparison; raw bit patterns are preserved and reported.
- **Compatibility Invariants:** Mismatches in sample rate, block quantum, model SHA-256, signal SHA-256, channels, or initial state are treated as incompatible vectors and rejected with exit code 2.
- **CLI Utility (`tools/qa/src/bin/nam_parity_vector.rs`):**

  ```bash
  # Generate release inference parity vector:
  cargo run --release -p nam-qa --bin nam_parity_vector -- generate \
    --model tests/fixtures/models/BossWN-standard.nam \
    --signal v1 --out target/parity-vectors/host.bin

  # Compare two vectors bit-by-bit:
  cargo run --release -p nam-qa --bin nam_parity_vector -- compare \
    target/parity-vectors/ref.bin target/parity-vectors/test.bin

  # Inspect metadata and signal statistics:
  cargo run --release -p nam-qa --bin nam_parity_vector -- inspect target/parity-vectors/host.bin
  ```

### 3.6 Hardware Classification Taxonomy & Baseline Comparability (R-12, R-23, S6-T03)

To ensure that latency regression benchmarks and performance baselines are anchored in real, comparable physical hardware environments (preventing the Ryzen 7 5700U baseline from being erroneously applied across differing architectures), the engine establishes a typed `HardwareClass` taxonomy ([`src/testing/qa/env.rs`](../src/testing/qa/env.rs)):

| Hardware Class Identifier | Target Platform & Architecture | Description |
|:--------------------------|:-------------------------------|:------------|
| `x86_zen3_laptop`         | Linux x86-64-v3 (Ryzen 7 5700U)| Reference development host baseline recorded in `docs/quality-contract.json`. |
| `x86_amd_zen`             | Linux x86-64-v3 (Zen 2/3/4/5)  | AMD desktop/server workstation host class. |
| `x86_intel_client`        | Linux x86-64-v3 (Core i-series)| Intel client architecture with AVX2/FMA/F16C/BMI2. |
| `x86_avx512_server`       | Linux x86-64 (Xeon/EPYC Genoa) | Server-grade host with full AVX-512 Foundation and Vector Length extensions. |
| `x86_generic_v3`          | Linux x86-64-v3                | Generic compliant x86-64 microarchitecture. |
| `aarch64_apple_silicon`   | macOS ARM64 (M-series)         | Apple Silicon M1/M2/M3/M4 with ARM64 NEON SIMD. |
| `aarch64_linux_server`    | Linux AArch64 (Graviton/Neoverse)| Cloud server ARM64 architecture (e.g. AWS Graviton 3/4). |
| `aarch64_generic`         | Linux AArch64                  | Standard AArch64 Linux host with NEON. |
| `unknown`                 | Unclassified / VM              | Unverified host architecture (fails closed). |

**Comparability Invariant:** Both the performance gate ([`src/bin/nam_perf_gate.rs`](../src/bin/nam_perf_gate.rs)) and the contract verification engine ([`src/testing/qa/verify.rs`](../src/testing/qa/verify.rs)) enforce strict hardware class matching. When a benchmark run or verify report belongs to a different hardware class than the baseline provenance, the performance domain evaluates to `NOT_VERIFIED` (`PerformanceVerdict::NotVerified` / `RegressionOutcome::NotVerified`), preventing false performance rejections or unverified passes.

### 3.7 Continuous Perceptual Metrics vs. Strict Bitwise Parity (R-22, S5, S6)

The engine clearly delineates between continuous numerical metrics and raw bitwise determinism:

- **Perceptual Error Metrics (ESR, SNR, MSE, MR-STFT):** Continuous metrics used to quantify fidelity against analog reference signals or floating-point models. Cross-ISA budgets (e.g. `WN_ESR_BUDGET: 1e-3`, `LSTM_ESR_BUDGET: 1e-2`, `A2_ESR_BUDGET: 1e-3` in `tests/parity/isa_parity.rs`) are conservative regression safety nets designed to prevent numerical drift in recurrent loops or transcendental approximations.
- **Strict Bitwise Parity (`f32::to_bits()`):** Ground-truth cross-runner evaluation performed by `nam_parity_vector compare` and `compare_samples_bitwise`. Requires exact IEEE-754 bit-pattern equality (`mismatch_count == 0` over all $N$ processed samples). Signed zeros (`+0.0` vs `-0.0`) and non-canonical subnormals are flagged as distinct bit mismatches.

### 3.8 Container Oracle Provenance & Explicit Historical Gaps (R-08, R-21, S7-T01)

- **SlimmableContainer A2 Example Provenance (R-21):** As defined in `src/testing/qa/ids.rs:318-326`, the f64 oracle label `SlimmableContainer A2 Example (CH=3→6)` is evaluated against the `wavenet_a2_lite.nam` fixture (identical baseline `1.82e-14`), because the golden catalog's `a2_example.nam` (CH=8) is an uncatalogued container fixture that the f64 oracle phase does not measure independently.
- **Matriz v2 §3.3 Explicit GAP:** The historical "matriz v2 §3.3" referenced in legacy notes is formally documented as an explicit gap (`GAP`). The original matrix document is absent from the repository; in accordance with fail-closed governance, no maturity claims are inferred from missing artifacts.

### 3.9 Operational Roles & Separation of Concerns (R-08, R-20)

To guarantee measurement integrity and host stability:

- **Automated AI Assistant Role:** Limited to static analysis (`utils/lints.sh`), fast unit and regression verification (`utils/tests-quick.sh`, maximum one run per task), and non-mutating quality checks (`utils/quality-dashboard.sh --check docs/quality-contract.json`). All resource-intensive compilation tasks must run under cgroup memory protection.
- **Human Operator Role:** Exclusive responsibility for long-running audit suites (`utils/tests-long.sh`), nightly ceremonies, baseline recording (`--save`), and physical latency benchmarks on calibrated, unconstrained bare-metal hardware without cgroups, CPU throttling, or concurrent builds.

---

## 4. Test Placement Rules

Test location is strictly governed by scope and size:

| Axis                                                          | Rule                                       | Execution Target                                                                                         |
|:------------------------------------------------------------- |:------------------------------------------ |:-------------------------------------------------------------------------------------------------------- |
| Structural / logic / parsers / FSM / bitwise                  | Non-ignored, **Debug**                     | `tests-quick.sh` tests phase (all non-ignored targets)                                                   |
| Non-ignored numerical / parity checks                         | Non-ignored, **Debug** (quick smoke)       | `tests-quick.sh` tests phase; release-grade measurement in `tests-long.sh` `catch-all`                    |
| Exhaustive matrix / soak / heap-audit / RT / Loom model check | `#[ignore]` or feature-gated               | `tests-long.sh` `catch-all`, `soak`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`                     |
| Performance regression gate                                   | Contract-driven benchmark check            | `tests-long.sh` `perf` (and the standalone `nam_perf_gate`)                                              |

Any test marked `#[ignore]` without an invocation in `tests-long.sh` or an explicit on-demand reason comment (`// on-demand:`) is considered an orphan.

### 4.1 Unit Test Placement & The 300-Line Rule

Unit tests adhere to strict source organization rules:

- **Files < 300 source lines:** Tests are declared inline within an embedded module:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;
      // ...
  }
  ```

- **Files ≥ 300 source lines:** Tests are extracted to a dedicated sibling file `<module>_test.rs` and included via:

  ```rust
  #[cfg(test)]
  #[path = "<module>_test.rs"]
  mod tests;
  ```

- **Off-RT Boundary:** Test utilities, golden generators, and off-RT validation helpers reside in `src/testing/` — never in `src/dsp/` or hot-path audio processing modules.

### 4.2 Modular Integration Entry Points

Integration tests are partitioned into cohesive entry points under `tests/`:

- [tests/models.rs](../tests/models.rs): Model instantiation, architecture validation, and golden vector comparisons.
- [tests/parity.rs](../tests/parity.rs): Reference oracle comparisons (f64, C++ NAMCore live parity, cross-ISA parity).
- [tests/perf_soak.rs](../tests/perf_soak.rs): Concurrency stress and high-load SPSC ring pipelines.
- [tests/soak_pinned.rs](../tests/soak_pinned.rs): Pinned continuous-processing soak battery (`required-features = ["testing", "pinned"]`).
- [tests/rt_constraints.rs](../tests/rt_constraints.rs): Real-time constraints and zero heap allocation audits.
- [tests/rt_pinned.rs](../tests/rt_pinned.rs): Pinned deadline and jitter telemetry (`required-features = ["testing", "pinned"]`).
- [tests/dsp_core.rs](../tests/dsp_core.rs): Resamplers, noise gate FSM, circular buffers, and DSP pipeline stages.
- Standalone harnesses: `cabsim_stereo.rs`, `libm_export_guard.rs`, `loom_tests.rs`.

---

## 5. Decoupled Long QA Audits (`utils/tests-long.sh`)

The long-duration stress and audit suite executes comprehensive numerical validation, full-count proptests, live C++ parity, memory audits, and real-time deadline tests (~10–13 minutes, 625–749 s measured, depending on hardware).

### 5.1 Blocking Preflight Gates

Before entering timed test execution, five preflight steps run:

1. **`preflight-render`:** Validates or builds the upstream C++ reference `render` CLI via `utils/ensure_namcore_render.sh`. The build cache key includes the NAMCore and plugin git commits, compiler identity/version, CMake version, C++ standard, `NAM_ENABLE_A2_FAST` and the compile flags, so a pin bump forces a rebuild. The phase then runs the binary as a smoke test (non-empty output, usage exit code `1`) and records `oracle_core_commit`/`oracle_compiler` in the receipt. Exit code `3` (mirror absent) becomes a `SKIPPED` phase with an `optional-dependency-missing` gap unless `--strict` is set.
2. **`preflight-catalog`:** Runs `cargo test --release --features testing --test models catalog_preflight` against the fixtures and V1/V2 goldens registered in [src/testing/catalog.rs](../src/testing/catalog.rs). The test enforces the strict counting invariant: `present + missing_required.len() + missing_optional.len() + known_gap_files == fixtures_checked`.
3. **`preflight-package`:** Verifies the published tarball by running `cargo package --locked --allow-dirty` (verify: the packaged crate must compile), then `utils/consumer-smoke.sh` (timeout 600 s): an external crate in an isolated temporary directory with a `path` dependency on the repository must fail on `x86_64` without `-Ctarget-cpu=x86-64-v3` with the actionable consumer message, compile with `-Ctarget-cpu=x86-64-v3`, and compile for `aarch64-unknown-linux-gnu` and `aarch64-apple-darwin` without flags. Only `nam_perf_gate` and `simd_probe` ship from `src/bin/`.
4. **`preflight-freshness`:** Runs `nam_freshness` in hard-fail mode against `tests/fixtures/.golden_manifest.sha256`. In this mode generator provenance drift is blocking: `nam_freshness` exits `5` (typed cause, not text), the phase is recorded `FAILED` with `reason: GENERATOR_DRIFT`, and the long suite fails closed. Other failures keep exit `1`. This intentionally differs from the quick `artifacts-hard` gate, where the same drift is an informational `generator-drift` gap (D-07).
5. **`preflight-simd-probe` (Diagnostic):** Runs `cargo run --quiet --bin simd_probe` to record host SIMD feature bits, OS AVX-512 state, and dispatched engine. It does not require tests, but a nonzero exit still marks the phase `FAILED`.

Each preflight writes `target/logs/long-<phase>.log` (for example, `target/logs/long-preflight-catalog.log`). Phase execution redirects stdout/stderr to the phase log while `xtrace_side_channel` (`utils/_lib.sh`) sends shell xtrace to `target/logs/long-xtrace.log`, so phase logs contain command output only.

### 5.2 Timeouts and Missing Oracle or Baseline

Every external command in a phase runs under `timeout --kill-after=30 <seconds>` (`run_cmd` helper); exit codes `124`/`137` become `FAIL rc=124 (timeout)`. `NAM_PHASE_TIMEOUT_S` overrides the timeout across all commands.

| Missing Resource | Without `--strict` (default) | With `--strict` |
| :-- | :-- | :-- |
| C++ Oracle (`preflight-render`, exit 3) | `GAP optional-dependency-missing`; `catch-all` omits `cpp_parity::live_cross_validation` (D-02) while `d02_live_cross_validation_oracle_probe` reports the skip, yielding `GAP skip-capability:<n>` | `FAIL`; full coverage required |
| Performance baseline (`.performance-baselines/baseline-fingerprint.json`) | `GAP perf-baseline-missing`; phase `perf` does not execute (`nam_perf_gate run --check` exits `3` before pre-compiling the benchmark) | `FAIL` (`MISSING_BASELINE`; `nam_perf_gate run --check --strict` exits `4` before pre-compiling; renewal is human-only via `--bootstrap-baseline`) |

`tests-long.sh` truncates `target/logs/skips.jsonl` at the start of the run and classifies its records: after `catch-all`, blocking records (capability, oracle, or missing distributable fixtures reported through `report_skip`) produce `GAP skip-capability:<n>`, while optional non-distributable model skips (`models_nondist_absent`, `models_nondist_empty`, `community_model_absent:<file>`) are recorded with `"optional":true` and do **not** create a blocking gap. The count comes only from the typed file — there is no `grep` over `cargo test --list` and no enumerated count of skipped test names. The D-02 exception is still declared once in the runner: in non-strict mode, `catch-all` passes `--skip cpp_parity::live_cross_validation`, and the `d02_live_cross_validation_oracle_probe` test records the omission in `skips.jsonl` (the probe panics under `NAM_REQUIRE_CPP_ORACLE=1`). The phase is recorded in the receipt as `SKIP_CAPABILITY`, so the gap reaches both `RESULT:` and `gaps`.

A `GAP` is never recorded as `PASS`: it is written to the receipt as `SKIPPED` (or as `SKIP_CAPABILITY` for the typed skip gap) and surfaced in `RESULT: PASS_WITH_GAPS`. The operator selects `--strict` explicitly; an explicit `NAM_BENCH_CORE` does not by itself prove isolation.

The performance gate decides baseline presence in Rust, before the benchmark pre-compilation (A2): non-strict missing baseline exits `3` (`GAP perf-baseline-missing`), strict missing baseline exits `4` (`FAILED` with `reason: MISSING_BASELINE`). `tests-long.sh` maps both codes and no longer inspects `.performance-baselines/` itself. The summary therefore reports `PERF: NOT_VERIFIED` and `PERF_REGRESSION: NOT_RUN` in a clone without a baseline, instead of a false regression.

### 5.3 Sequential Execution Phases

1. **`catch-all` (`phase1`):** `cargo test --release --features testing --no-fail-fast -- --include-ignored`. Pinned targets are not compiled because `pinned` is not enabled, so no manual `SKIP` list exists. The single runtime exception is the documented D-02 path: when the optional C++ oracle is absent in non-strict mode, only `cpp_parity::live_cross_validation` is omitted, `d02_live_cross_validation_oracle_probe` reports the typed skip, and the run records `GAP skip-capability:<n>`. This is the broad numerical matrix: live C++ parity, multi-sample-rate V2 goldens, cross-ISA checks, defense/invariant tests, the `libm_export_guard`, and the proptest sweeps.
2. **`soak` (`phase2`):** `cargo test --release --features testing,pinned --test soak_pinned -- --include-ignored --test-threads=1`: 10M+ frames continuous endurance testing ([tests/soak_pinned.rs](../tests/soak_pinned.rs)).
3. **`isa-codegen` (`phase3`):** Runs `cargo run --quiet --bin simd_probe` and then `cargo test --release --features testing --test isa_contract -- --nocapture`.
4. **`heap-audit` (`phase4`, filtered, `--include-ignored heap_audit`, profile `target/long/heap-audit`):** Strict verification of zero heap allocations on the audio processing hot path (`CountingAllocator` via `--features heap-audit`, including resampler, cabsim, A2 standard, A2-Max dynamic `test_a2_dyn_max_heap_audit`, and sequential chains).
5. **`rt-deadline` (`phase5`):** Resolves a core with `nam_perf_gate pick-core` (exporting `NAM_BENCH_CORE`, with thread pinning performed in Rust via `sched_setaffinity`, or declaring `GAP no-bench-core` if unresolved in non-strict mode) and runs `cargo test --release --features testing,pinned --test rt_pinned rt_deadline:: -- --include-ignored --nocapture`. Processing budget enforcement ($p99 < 1.33\text{ ms}$ for 64-sample blocks at 48 kHz across all SKUs, including A2-Max `test_rt_deadline_a2_max`), measured with `LatencyHistogram` (256 log-linear sub-bins, relative error bound $\le 12.5\%$) paired with `ExactLatencyReservoir` for exact $p99$ arbitration. Under `--strict` (`NAM_RT_STRICT=1`), any environmental preflight failure or missing core fails fail-closed.
6. **`rt-jitter` (`phase6`):** Inherits or resolves `NAM_BENCH_CORE` via `pick-core` and runs `cargo test --release --features testing,pinned --test rt_pinned rt_jitter:: -- --include-ignored --nocapture`: measures processing latency distribution and context switch telemetry (`cs_vol`, `cs_invol` via `getrusage(RUSAGE_THREAD)`) under CPU contention (`stress-1`, `stress-2`, `saturate-N`) across $\ge 16384$ blocks with initial 512-block cache/branch warmup. Linux is the only platform that reports `cs_vol`/`cs_invol`; on non-Linux hosts the pinned target still compiles and the receipt prints `cs_vol=N/A cs_invol=N/A` rather than inventing zeros. When `NAM_BENCH_CORE` is set, the DSP thread runs pinned to the isolated core while stress workers saturate the remaining cores, proving zero deadline violations under full memory/L3 load.
7. **`loom` (`phase7`):** `cargo test --release --test loom_tests --config "target.'cfg(target_arch = \"x86_64\")'.rustflags=['--cfg','loom']" -- --nocapture` (profile `target/long/loom`; `--config` appends `--cfg loom` to the flags in `.cargo/config.toml` instead of replacing them, and any inherited `RUSTFLAGS` is ignored): Loom model verification for lock-free queues and atomic bitmasks.
8. **`perf` (`phase8`):** `nam_perf_gate run --check docs/quality-contract.json` (with `--bench-core` when `pick-core` succeeds and `--strict` when requested): automated benchmark regression check against the baseline contracts.

#### 5.3.1 Numeric Telemetry (`fidelity-metrics.jsonl`)

The long suite writes one fixed telemetry stream, `target/logs/fidelity-metrics.jsonl`, without introducing a new environment variable (D-05 = (b)). The stream is truncated before `phase1` and is append-only afterwards:

- **Fidelity rows** (`kind: fidelity`): ESR, ESR dB, SNR, MR-STFT and MSE for every model exercised by the fidelity helpers, emitted by `tests/common/validation.rs` through the existing metric sink. `NAM_METRICS_JSONL`, when set, still overrides the fixed path for isolated consumers.
- **Heap rows** (`kind: heap_alloc`): `scenario` and `allocations` totals measured under `TrackingGuard` with `get_alloc_count()` in `phase4`. These rows are telemetry only; the zero-allocation assertions remain the gate.

The file is a diagnostic enrichment, never a gate: a missing sink is ignored, and no phase is promoted or demoted from its presence.

### 5.4 Structured Audit Receipt (`long-audit-receipt.jsonl`)

Each phase appends a structured record to `target/logs/long-audit-receipt.jsonl` using the Rust binary `nam_long_receipt`:

```json
{
  "phase_id": "phase1",
  "name": "catch-all",
  "status": "PASSED",
  "duration_ms": 42000,
  "tests_executed": 26,
  "gaps": [],
  "timestamp": "2026-08-14T03:00:00Z",
  "run_id": "20260814T030000Z-1",
  "git_sha": "e2d090c…",
  "git_dirty": false
}
```

- **Receipt Statuses:** `PASSED`, `FAILED`, `SKIPPED`, `INCONCLUSIVE`, `SKIP_CAPABILITY`, `NOT_RUN`, `SIMULATED`. The suite line uses `PASSED`, `FAILED`, `COMPLETED_WITH_GAPS`, or `SIMULATED`.

- **Optional Gaps:** a phase line may carry `optional_gaps` (additive, omitted when empty). These are typed skips of optional non-distributable fixtures reported for traceability and counted in the receipt, but they never enter `gaps`, never downgrade `OVERALL` to `COMPLETED_WITH_GAPS`, and never fail `validate --strict`. Missing mandatory fixtures remain blocking gaps or failures.

- **Provenance and Typed Reason:** every line carries `run_id` (from `NAM_RUN_ID`), `git_sha` and `git_dirty` (captured by `nam_long_receipt` itself, never by the shell). A `FAILED` line also carries a typed `reason` — `GENERATOR_DRIFT`, `MISSING_BASELINE`, `TEST_FAILURE`, `TIMEOUT`, or `PREFLIGHT_FAILURE` — so root causes are machine-readable. All four fields are additive on read (`serde(default)` / `skip_serializing_if`), so historical receipts keep parsing. Under `validate --strict`, every phase line must carry all three provenance fields, must not be from a dirty tree, and all lines must share the same `run_id`/`git_sha`.

- **Verdict Lines:** the human summary separates `PROVENANCE` (from `preflight-freshness`), `FIDELITY` (only `preflight-catalog`, `phase1`, `phase3`, `phase4`, `phase7`), `PERF`, `PERF_REGRESSION`, `RT_DEADLINE`, and `RT_JITTER`. When `PROVENANCE` is `FAIL`, `FIDELITY` reports `NOT_VERIFIED` (D-06): stale fixtures cannot back either a green or a red fidelity assertion.

- **RT Verdict Mapping:** `RT_DEADLINE` and `RT_JITTER` share one exhaustive mapping over `LongPhaseStatus` (no wildcard arm): `FAILED → FAIL`, `SKIPPED`/`INCONCLUSIVE → INCONCLUSIVE`, `SKIP_CAPABILITY → SKIP_CAPABILITY`, `NOT_RUN`/`SIMULATED → NOT_RUN`, `COMPLETED_WITH_GAPS → INCONCLUSIVE`, and `PASSED → PASS` only when the phase has no declared gap. A phase selected but not measured can never report `PASS`.

- **Aggregated Overall Gaps:** The terminal `overall` record aggregates the deduplicated union of gaps from all phases prefixed by `phase_id` (`<phase_id>:<gap_id>`), guaranteeing full traceability and exact match with `overall.gaps.len()`.

- **Performance Regression Integration:** `PERF` derives from the `phase8` line: `PASS` only when the phase passed without a `perf-baseline-missing` gap; `NOT_VERIFIED` for a missing baseline (`reason: MISSING_BASELINE`) or a skipped phase; `FAIL` for any other failure. `PERF_REGRESSION` is the comparison verdict and only has meaning when `PERF` is `PASS`: it is evaluated dynamically from `target/logs/regression_phase_receipt.jsonl` (within 24 hours / matching run ID), emits `NOT_RUN` when `PERF` is `NOT_VERIFIED`/`NOT_RUN`, and emits `FAIL` when `PERF` is `FAIL`. A `PERF_REGRESSION: FAIL` promotes `OVERALL` to `FAILED`, emits `ERROR: perf_regression — FAILED`, and causes `utils/tests-long.sh` to fail closed. A missing baseline never produces `PERF_REGRESSION: FAIL`.

- **Simulated Receipts:** Pre-registered receipts with status `SIMULATED` carry `tests_executed: 0` without executing any test. The derived suite line is `SIMULATED` and every verdict line reads `NOT_RUN` (`PROVENANCE`, `FIDELITY`, `PERF`, `RT_DEADLINE`, `RT_JITTER`, `PERF_REGRESSION`); a simulated receipt never prints `OK`/`PASS`, and strict verification (`--strict`) always rejects it.

- **Typed Markers:** Discrepancies or skips must emit recognized markers parsed into typed gaps:

  - `[STATUS] SKIP_CAPABILITY reason="<detail>"`: Hardware or ISA absence.
  - `[STATUS] SKIP_OPTIONAL reason="<detail>"`: Missing non-distributable optional model.
  - Optional model skips must use `neural_amp_modeler_rs::testing::report_skip(test, reason)` (reasons `models_nondist_absent`, `models_nondist_empty`, `community_model_absent:<file>`), never a bare `println!`/`eprintln!`: only `report_skip` writes `target/logs/skips.jsonl`. The three optional reasons are emitted as `[STATUS] SKIP_OPTIONAL`, recorded in `optional_gaps`, and are non-blocking under `--strict`; every other `report_skip` reason remains a blocking `[STATUS] SKIP_CAPABILITY` gap (F-R2-07).
  - `[STATUS] KNOWN_GAP id="<id>" reason="<detail>"`: Upstream gap under active tracking.
  - `[STATUS] INCONCLUSIVE reason="<detail>"`: Measurement bypass in non-calibrated environments.

- **Fail-Closed Invariant:** `overall: PASSED` is emitted only when all phases complete with `gaps: []` and no timed phase executed zero tests. `nam_long_receipt validate` is a **semantic** verifier — never a substring search: it enforces the canonical roster (`CANONICAL_ROSTER`: 5 preflights + `phase1`..`phase8` + terminal `overall`), rejecting empty/partial receipts, missing/duplicate/unknown IDs, a timed phase that claims `PASSED` with zero executed tests, and a declared `OVERALL:` verdict that disagrees with the verdict derived from the phase/gap evidence (a forged/injected `OVERALL: PASSED` line — including one pasted into a comment — is rejected at parse time). In strict mode (`--strict`) it additionally rejects any declared blocking gap or failure, any `FAILED` line without a typed `reason`, and any phase line with missing/dirty/mixed provenance; `optional_gaps` are admitted and counted, never rejected. A structurally valid receipt whose `overall` verdict is `FAILED` is reported by `validate` with exit code `4` (the audit is rejected but the receipt is not corrupt; exit `1` remains reserved for a corrupt or invalid receipt), and `utils/tests-long.sh` does not duplicate that failure as a derived `receipt-validation` entry. Verification is enforced via:

  ```bash
  cargo run --locked -p nam-qa --bin nam_long_receipt -- validate --strict --out target/logs/long-audit-receipt.jsonl
  ```

  The roster is a parameter (`LongAuditReceipt::semantic_verdict_with_roster` /
  `campaign_strict_verdict_with_roster`) so downstream producers (hosts,
  standalone applications, plugins) reuse the exact schema and semantics with
  their own phase matrix.

  When running with `--only <name>[,<name>]`, the runner selects phases by their display name (`preflight-*`, `catch-all`, `soak`, `isa-codegen`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`, `perf`; `tests-long.sh --help` prints the table); unselected phases are recorded as `SKIPPED` under their canonical `phase_id` (`phase1`..`phase8`, `preflight-*`). The receipt tools also accept the `phase_id` in `--only`. Both `nam_long_receipt summary --only <phases>` and `nam_long_receipt validate --only <phases>` natively recognize `--only`, ensuring unselected phases do not cause strict validation failures or spurious regression flags.

- **Referenced-Artifact Provenance:** each JSONL line may carry the optional
  SHA-256 of the artifacts that backed its verdict — `oracle_sha256`,
  `fixture_sha256`, `artifact_sha256` (producer-neutral names; no host-specific
  field is ever encoded). The fields are omitted when absent
  (`skip_serializing_if`) and default when absent on read (`#[serde(default)]`),
  so historical goldens remain byte-compatible. `nam_long_receipt append`
  accepts `--oracle-sha256` / `--fixture-sha256` / `--artifact-sha256`.

### 5.5 Build Target Segregation (`CARGO_TARGET_DIR`)

`utils/tests-long.sh` segregates build outputs by profile under `target/long/`, so that switching `--features heap-audit` or `RUSTFLAGS="--cfg loom"` does not invalidate the cache of the other phases:

- `target/long/release`: preflights and phases 1, 2, 3, 5, 6 and 8.
- `target/long/heap-audit`: phase 4 only.
- `target/long/loom`: phase 7 only.

`target/long` is removed at the start only when `--only` is not used; `--only` keeps the caches for iterative reruns. The phase 4 filter is `--test rt_constraints --test cabsim_stereo --test models -- --include-ignored heap_audit`: the heap-audit tests are compiled only with `--features heap-audit`, so phase 1 never runs them, and the filter also covers the non-ignored ones.

### 5.6 Long Suite Workflow on GitHub Actions (`.github/workflows/long.yml`)

The long QA audit can be dispatched manually by maintainers at the end of an epic or before a release via GitHub Actions:

- **Trigger:** Manual (`workflow_dispatch`), through the GitHub Actions web UI (`Actions → Long QA Suite → Run workflow`) or the GitHub CLI:

  ```bash
  gh workflow run long.yml --ref dev -f strict=true
  ```

- **Operator Runbook (Branch & Provenance Check):** When dispatching `Long QA Suite` from the GitHub Actions web UI ("Use workflow from"), **explicitly select the target branch** (e.g. `dev`) rather than leaving the dropdown on `main` when certifying an unmerged integration branch. Always verify the summary header (`Long QA Suite (<branch> @ <sha>)`) in `$GITHUB_STEP_SUMMARY` to confirm that the workflow executed against the expected commit SHA.
- **Inputs:**
  - `phases`: Comma-separated phase IDs to execute (default: `'preflight-render,preflight-catalog,preflight-package,preflight-freshness,phase1,phase2,phase3,phase4,phase7'`).
  - `strict`: Boolean (default: `true`), enforcing fail-closed zero-gap execution.
- **Quality Dashboard (mandatory, always):** Every run executes `./utils/quality-dashboard.sh --check docs/quality-contract.json --fidelity-only` after the long audit phases, including when the suite fails (`if: always()`). CI checks audio fidelity and parity only; latency/performance envelopes are not evaluated because cloud runners are not comparable to the calibrated reference machine. Provenance is self-contained: the dashboard deletes its own `fidelity_metrics.jsonl` and `dashboard_phase_receipt.jsonl` at start and only reuses phases whose `run_id` matches `NAM_RUN_ID`, so it cannot certify fidelity from another run's file. Absence or refusal of evidence remains a gap, never a pass.
- **Default Exclusion Policy:** The default phase list excludes `phase5` (RT deadline), `phase6` (RT jitter), and `phase8` (perf gate). Shared cloud virtual machines exhibit significant scheduling jitter and noisy-neighbor virtualization overhead, making them unsuitable as calibrated real-time performance arbiters.
- **Artifacts:** On completion, the workflow publishes the execution log (`target/logs/**`) and receipts (`target/long/**/*.jsonl`) as workflow artifacts with 14-day retention, alongside step summary metrics in `$GITHUB_STEP_SUMMARY`.

---

## 6. Ignored-Test Policy

Tests annotated with `#[ignore]` represent rigorous or long-duration validations, not deprecated code. Every ignored test must belong to one of two categories:

1. Invoked by `utils/tests-long.sh` during nightly or pre-release auditing.
2. Accompanied by an explicit on-demand reason in the source comments (`// on-demand: <reason>`), explaining when and how it should be run manually.

---

## 7. Execution Policies & Operator Delegation

The test framework applies two distinct execution paradigms depending on the operational goal:

### 7.1 Fail-Fast Policy (`utils/tests-quick.sh`)

- **Goal:** Minimize feedback latency during local development and pre-commit checks.
- **Behavior:** Halts immediately on the first failure (`set -e`, standard bash error traps).

### 7.2 Complete View Policy (`utils/tests-long.sh`)

- **Goal:** Provide a comprehensive status report across all functional domains for pre-release and nightly certification.
- **Behavior:** Continues through all phases despite individual target failures, aggregating logs and compiling the final audit receipt.

### 7.3 Operator Delegation

The long audit suite (`utils/tests-long.sh`), baseline renewal (`nam_perf_gate run --bootstrap-baseline`), and contract snapshots (`utils/quality-dashboard.sh --save`) are executed **exclusively by human operators** on calibrated, isolated machines. The complete manual testing protocol and release certification checklist reside in [functional-tests.md](functional-tests.md).

The performance gate is **fail-closed**: a missing baseline is `MISSING_BASELINE`, and a fingerprint drift (CPU model, ISA, toolchain, governor, pinned core) is `INCOMPARABLE_ENVIRONMENT` — neither silently re-creates or ignores a baseline; renewal is an explicit operator operation under a calibrated environment. The baseline fingerprint schema also records audit and provenance metadata (`git_commit`, optional `git_dirty`, and optional `cargo_lock_sha256`). Divergence in these audit fields emits informational diagnostic warnings to `stderr` without failing the gate, preserving strict backward compatibility with legacy baseline files. The gate binary `nam_perf_gate` is reusable by downstream suites (installed via `cargo install NeuralAmpModeler-rs --features testing --bin nam_perf_gate`, or `NAM_PERF_GATE`); `coverage`/`expected-count`/`verdict` accept an external `--manifest` of expected ids (trailing-`*` group patterns) so any Criterion suite is verified without engine-internal names. See [benchmarks.md §4](benchmarks.md) for the full contract.

---

## 8. Measurement & Perceptual Validation Framework

Audio fidelity assessment combines physical error metrics with perceptual domain envelopes, documented in [perceptual_validation.md](perceptual_validation.md).

### 8.1 Integration with Test Targets

| Test Target                 | Metrics Evaluated                                 | Purpose                                                    |
|:--------------------------- |:------------------------------------------------- |:---------------------------------------------------------- |
| **`cpp_parity`**            | ESR, SNR, PSNR, MSE, MAE, Anchor SNR              | Interoperability with upstream C++ reference               |
| **`golden_vectors`**        | Per-model ESR thresholds, MSE, SNR                | Long-term regression gate against frozen binary outputs    |
| **`isa_parity`**            | ESR cross-ISA budgets, self-consistency (MSE = 0) | Architectural determinism across AVX2, AVX-512, and scalar |
| **`spectral_fidelity`**     | ASR, Farina FR+THD, THD+N (AES17), IMD (SMPTE)    | Aliasing and non-linear distortion characterization        |
| **`reference_oracle_f64`**  | ESR (f64 vs. f32), decomposed error sources       | Numerical ideality and activation precision budgeting      |
| **`threshold_calibration`** | Per-model ESR/SNR baselines, Fidelity Margins     | Calibration and noise-envelope definition                  |

### 8.2 Core Principles

- **Dual Reference Systems:** Parity (C++ NAMCore f32) measures implementation alignment; absolute fidelity (f64 Oracle) measures intrinsic precision loss from single-precision approximations. Discrepancies between both references mandate review; neither oracle overrides the other unconditionally.
- **Energy-to-Signal Ratio (ESR) & Signal-to-Noise Ratio (SNR):** Exact zero comparisons (`noise_power == 0.0`, `signal_power == 0.0`) govern edge conditions. Bit-identical parity yields `SNR = +∞` (and `ESR = 0.0` / `ESR_dB = -∞`). Any non-zero error energy, even at the sub-epsilon numerical noise floor ($\sim 10^{-16}$, e.g. 5–6 ULPs across samples), yields finite logarithmic values, preventing false infinities during high-fidelity model validation.
- **Off-RT Execution:** All perceptual and spectral metrics allocate memory and run strictly off-RT. Hot-path audio threads use sample-peak detection only.

### 8.3 Cross-ISA Verification & Declared Skips (`SKIP_CAPABILITY`)

The test suite enforces a strict boundary between **single-architecture determinism** and **cross-ISA numerical parity**:

1. **Declared Skips (`SKIP_CAPABILITY`):**

   - The engine targets `x86-64-v3` (AVX2, FMA, BMI2) on x86_64 and NEON on aarch64 as its strict compile-time baselines. On host machines without AVX-512 hardware or when compiling without `--features avx512`, AVX-512 execution paths cannot physically run.

   - Tests in [tests/parity/isa_parity.rs](../tests/parity/isa_parity.rs) detect missing hardware capabilities or disabled Cargo features via `skip_if_unsupported!` and emit machine-parseable markers:

     ```text
     [STATUS] SKIP_CAPABILITY reason="<reason>:<test_name>"
     ```

   - In local runs and the QA dashboard ([utils/quality-dashboard.sh](../utils/quality-dashboard.sh)), the AVX2-vs-AVX2 self-consistency phase (`isa_self_consistency`) executes the 10 empirical model checks (7 standard and 3 exact-grade HF models) under `NAM_METRICS_JSONL`, recording empirical per-model MSE. The inter-ISA matrix (`isa_parity_cross_isa`) is declared as an explicit gap (`cross_isa_matrix_requires_avx512_remote`), avoiding false failures while preventing unvetted assumptions. This gap is `GAP` in the matrix: no AVX2 vs AVX-512 comparison is counted as executed until the operator runs the x86-64-v4 recipe on a host with AVX-512 (e.g. AWS EC2 `c8id.16xlarge`, operator-only); a `SKIP_CAPABILITY` is never counted as a comparison.

2. **`FIDELITY: OK` Scope Boundary:**

   - A local test run reporting `FIDELITY: OK` validates that all active single-architecture mathematical reference oracles (`reference_oracle_f64`), committed golden vectors (`golden_vectors`), and C++ reference parity (`cpp_parity`) passed within contractual error budgets on the local host architecture.
   - **`FIDELITY: OK` does NOT cover cross-ISA parity.** Local self-consistency (AVX2 vs. AVX2) guarantees local model determinism across execution runs, not numerical identity across differing SIMD instruction sets. Bitwise identity requires explicit per-sample representation verification (`f32::to_bits()`).

3. **Remote SIMD Verification (`utils/remote-simd-gate.sh`):**

   - Cross-ISA parity (AVX2 vs. AVX-512) runs on AVX-512 hardware or under emulation with [utils/remote-simd-gate.sh](../utils/remote-simd-gate.sh). Comparative SIMD micro-benchmarks run only on native hardware:

     ```bash
     # Execute under Intel SDE emulator:
     ./utils/remote-simd-gate.sh --sde

     # Execute on native AVX-512 hardware:
     ./utils/remote-simd-gate.sh

     # Specify custom destination for the receipt JSONL:
     ./utils/remote-simd-gate.sh --out target/logs/remote-simd-receipt.jsonl
     ```

   - The script compiles with `--features avx512 --include-ignored` and executes the `isa_parity` test battery. Native runs also generate a Criterion receipt at `target/logs/remote-simd-receipt.jsonl`; SDE and `--skip-bench` runs do not. The script currently generates the receipt without `--check`, so a successful script exit does not enforce the receipt's statistical verdict.

---

## 9. Test Value Hierarchy

Tests are categorized into three hierarchical tiers based on the guarantees they provide:

| Tier    | Category                                                     | Primary Guarantee                                | Execution Target                 |
|:-------:|:------------------------------------------------------------ |:------------------------------------------------ |:-------------------------------- |
| **1🔴** | Upstream parity (`golden_vectors`, `cpp_parity`)             | Compatibility with the NAM ecosystem             | Quick tests + Long `catch-all`   |
| **1🔴** | Real-time safety (`heap-audit`, zero-alloc hot-path)         | Zero dynamic allocation on audio thread          | Long `heap-audit`                |
| **1🔴** | Parser security & robustness (`.nam` / `.namb` fuzzing, CRC) | Memory safety and format integrity under attacks | Quick tests + Long `catch-all`   |
| **2🟠** | Spectral fidelity (ASR, Farina FR+THD, THD+N AES17)          | Clean frequency response and low aliasing        | Quick tests + Long `catch-all`   |
| **2🟠** | Mathematical correctness (vs. `f32::tanh` / `f64::tanh`)     | Activation approximations within tolerance       | Quick tests + Long `catch-all`   |
| **2🟠** | f64 Oracle, Cross-ISA determinism, RT deadline budget        | Absolute precision, portability, and latency     | Quick tests (f64/self-consistency) + Long `catch-all`/`isa-codegen`/`rt-deadline` |
| **3🟡** | Kernel regression locators (`avx2_vs_scalar`, GEMV, conv)    | Pinpoint localized SIMD regression sites         | Quick tests + Long `catch-all`   |
| **3🟡** | Approximation consistency checks (`nr2_vs_nr1`, Padé)        | Relative numerical stability across variants     | Long `catch-all`                 |
| **3🟡** | Proptest mathematical invariant sweeps                       | Stochastic exploration of numerical edge cases   | Quick tests (non-ignored) + Long `catch-all` (including ignored) |

### 9.1 Stochastic Proptest Tolerance Scaling

When executing property-based tests across randomized input vectors ($N \le 1024$ in $[-1.0, 1.0]$), accumulated single-precision rounding error scales with vector length and magnitude. Tolerances must be scaled by the $L_1$ norm of the product terms to prevent cancellation false positives:

$$\text{threshold} = 10^{-6} \times \max\left(1.0, \sum_{i=1}^N |x_i \cdot y_i|\right)$$

### 9.2 Virtualized & Cloud Execution Caveats

When running audits inside virtualized environments (e.g., cloud VMs):

- **Timing & Jitter (Phases 5 & 6):** Hypervisors cannot guarantee hard-real-time clock stability or dedicated CPU governors. The suite detects virtualized environments and classifies jitter deviations as `INCONCLUSIVE`, preventing false failures on hard functional gates.
- **SIMD Micro-benchmarking:** Gating scripts (`utils/remote-simd-gate.sh`) use Criterion statistical confidence intervals ($p < 0.05$) to ensure decisive comparative evaluations across ISAs.

---

## 10. Quality Contract

The Quality Contract establishes an immutable baseline that freezes audio fidelity and inference latency to prevent silent regressions.

### 10.1 Architecture & Governance

The contract is governed by [utils/quality-dashboard.sh](../utils/quality-dashboard.sh) against the canonical schema file [docs/quality-contract.json](quality-contract.json):

```bash
# Check current build against baseline contract
./utils/quality-dashboard.sh --check docs/quality-contract.json

# Freeze current results as the baseline contract (Human operator only)
./utils/quality-dashboard.sh --save docs/quality-contract.json
```

The dashboard also accepts `--fidelity-only` to check fidelity and parity without the performance baseline. Full `--check` (including the `regression_gate` performance domain) remains receipt 5 for release certification. The fidelity-only check re-executes its suites and was measured at ~134–164 s on 2026-10-30. `.github/workflows/long.yml` runs it mandatorily in `--fidelity-only` mode, since cloud runners are not comparable to the calibrated reference machine.

The dashboard counts executed tests from the canonical libtest `test result: ok. N passed` summary (after stripping ANSI escapes), summing every command of a phase. A phase with no `test result` line fails closed with the typed reason `no_test_result`; per-test `test … ... ok` lines are never counted, so `--nocapture` output cannot undercount a green phase. `lstm_activation_precision` certifies its two non-ignored tests (`structural`, `gain`); the ignored `gain_stress_v2` sweep remains part of the long suite.

`freshness` is a provenance phase, not a fidelity phase. The dashboard reports it separately as `PROVENANCE: OK|FAIL|NOT_RUN`; when provenance is `FAIL`, `FIDELITY` becomes `NOT_VERIFIED` (D-06) rather than `FAIL`, because stale fixtures cannot back either a green or a red fidelity assertion. The `--save` gate refuses a failed fidelity phase and a failed provenance phase, while performance `NOT_VERIFIED` never blocks (PERF-006).

The `third_party` phase verifies provenance, not mere presence: `NeuralAmpModelerCore` and `NeuralAmpModelerPlugin` must be git checkouts at the commits pinned in `variables.env` (`NAM_CORE_COMMIT`, `NAM_PLUGIN_COMMIT`), with a clean `git status --porcelain`, and the core submodules (`Dependencies/eigen`, `Dependencies/AudioDSPTools`) initialized at their recorded commits. A dirty tree, wrong commit, or uninitialized core submodule is recorded as `FAIL` with a typed reason (`third_party_core_commit_mismatch:<head>`, `third_party_core_dirty`, `third_party_core_submodule:<line>`, …), which makes the fidelity domain fail closed. A missing core mirror stays `SKIP_CAPABILITY` (`third_party_absent`); a missing plugin mirror is `SKIP_CAPABILITY` (`third_party_plugin_absent`) so environments that only need the core render keep working.

The contract schema is enforced via typed serde structures in `src/testing/qa/`. During the `reference_oracle_f64` measurement phase, `nam_quality` executes both the non-ignored `reference_oracle_f64::` suite and the ignored `reference_oracle_f64::test_summary_table` target (`--include-ignored --exact`), ensuring `ESR_F64` metrics are emitted for the summary models (`Linear`, `ConvNet`, `A2Full`, `A2Lite`) as well as the canonicalWaveNet/LSTM tiers.

### 10.2 Tolerance Envelopes

The `--check` mode distinguishes measurement noise from true regressions:

| Metric Domain                  | Failure Criterion                                                                            | Rationale                                                |
|:------------------------------ |:-------------------------------------------------------------------------------------------- |:-------------------------------------------------------- |
| **Fidelity — ESR**             | Dynamic noise envelope: $\max(\text{baseline} \times 3, \text{baseline} + 5\times 10^{-14})$ | Detects subtle 2×–5× regressions on ultra-precise models |
| **Fidelity — SNR**             | $\text{SNR}_{\text{new}} < \text{SNR}_{\text{contract}} - 6.0\text{ dB}$                     | Absorbs minor quantization and scheduling variance       |
| **Fidelity — MR-STFT**         | Envelope ceiling vs. contract                                                                | Limits multi-resolution spectral drift                   |
| **Oracle Divergence**          | `REVIEW_REQUIRED` if NAMCore and f64 diverge in opposing directions                          | Enforces human review when oracles disagree              |
| **Performance — Latency (µs)** | $\text{Latency}_{\text{new}} > \text{Latency}_{\text{contract}} \times 1.10$                 | 10% tolerance absorbs background OS scheduling noise     |

> [!IMPORTANT]
> **Decoupling Fidelity from Performance:**
> Latency deviations do **not** constitute audio degradation. The verification pipeline reports these domains independently:
>
> - `FIDELITY: OK` / `PERFORMANCE: NOT_VERIFIED`: All mathematical and audio fidelity checks passed; only benchmark timings varied. Sonic output is intact. Note: as detailed in §8.3, `FIDELITY: OK` reflects local single-architecture validation and does not cover cross-ISA parity between AVX2 and AVX-512.
> - `PROVENANCE: FAIL` / `FIDELITY: NOT_VERIFIED`: The golden generator is stale or the fixtures are derived/absent. No fidelity claim — green or red — can be certified until provenance is restored; this is not an engine regression.
> - `FIDELITY: FAIL`: Numerical or spectral degradation detected. Requires immediate DSP investigation.
>
> Primary statistical authority for performance regressions belongs to `nam_perf_gate run --check` (`src/bin/nam_perf_gate.rs`) via Criterion confidence intervals; the quality contract serves as an integrated secondary check.

---

## 11. Utility Scripts Inventory

The `utils/` directory provides deterministic defense tools, inspection utilities, and test runners:

| Script                                                                                | Responsibility                       | Guarantees & Operation                                                                                                                                                                                                                                                                                                                               |
|:------------------------------------------------------------------------------------- |:------------------------------------ |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[`utils/lints.sh`](../utils/lints.sh)**                                             | Static analysis & quality defense    | Eight phases: `fmt`, compilation and Clippy across four feature configurations, cross-target `clippy-aarch64`, `cargo doc`, SPDX, policy checks, and `script-budget` (§2).                                                                                                                                                                     |
| **`utils/tests-quick.sh`**                                                            | Agile first-line test suite          | Two-phase gate: `nam_freshness artifacts-hard` followed by `cargo test --locked --features testing,pinned --no-fail-fast` (debug, non-ignored targets). Runs in ~16–87s (16–22s warm incremental; 64–87s cold; budget ≤ 5 min) (§3). |
| **`utils/tests-long.sh`**                                                             | Nightly & pre-release audit suite    | 5 preflights + 8 phases (`catch-all`, `soak`, `isa-codegen`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`, `perf`; ~10–13 min measured, 625–749 s). Emits structured JSONL receipts (§5). Nightly & human operator execution.                                                                                                                                                                                           |
| **`utils/_lib.sh`**                                                                   | Shared shell library                 | Dynamic project path resolution, log formatting, QA-binary resolution and helper primitives.                                                                                                                                                                                                                                                         |
| **`utils/quality-dashboard.sh`**                                                      | Quality contract manager             | Executes fidelity and latency matrices; verifies or updates `docs/quality-contract.json` (§10). Finalizes with a canonical `RESULT:` and defaults to plain rendering when stdout is not a TTY. Periodic / on-demand evaluation.                                                                                                                         |
| **`src/bin/nam_perf_gate.rs`** | Performance regression wall | Baseline-gated Criterion evaluation (`nam_perf_gate run --check`); statistical confidence interval verification against ±5% noise bands. `pick-core --require-isolated` accepts only an online CPU listed in `/sys/devices/system/cpu/isolated` (an explicit `NAM_BENCH_CORE` must itself be isolated) and exits `3` with `GAP` when no such CPU exists, distinct from the `1` of `--strict`. On-demand execution. |
| **`utils/setup-third-party.sh`**                                                      | Upstream vendor mirror manager       | Clones or updates pinned commits of `NeuralAmpModelerCore` and `NeuralAmpModelerPlugin`.                                                                                                                                                                                                                                                             |
| **`utils/ensure_namcore_render.sh`**                                                  | C++ Reference Render Builder         | Idempotently builds the C++ NAMCore render binary for parity oracles.                                                                                                                                                                                                                                                                                |
| **[`utils/consumer-smoke.sh`](../utils/consumer-smoke.sh)**                           | External consumer contract smoke     | Validates that a standalone external crate in an isolated temporary directory fails on `x86_64` without `-Ctarget-cpu=x86-64-v3`, compiles with it, and compiles without extra flags on `aarch64-unknown-linux-gnu` and `aarch64-apple-darwin`.                                                                                                     |
| **`utils/check-model.sh`**                                                            | Model inspection CLI wrapper         | Diagnostic / spot-check CLI: inspects `.nam` (JSON) and `.namb` (binary) models, reporting topology, metadata, and tensors.                                                                                                                                                                                                                          |
| **`utils/remote-simd-gate.sh`**                                                       | AVX-512 SIMD gate                    | Targeted SIMD gate: executes automated cross-ISA parity and comparative benchmarking on AVX-512 hardware or under Intel SDE emulation.                                                                                                                                                                                                               |
| **`utils/miri-audit.sh`**                                                             | Miri UB/aliasing audit (opt-in)      | Out-of-band audit: provenance and aliasing verification under the Miri MIR interpreter (Stacked/Tree Borrows) on lock-free targets. Requires isolated nightly; **never** run in stable gates (§13).                                                                                                                                                  |
| **[`utils/docsrs-check.sh`](../utils/docsrs-check.sh)**                               | Docs.rs parity check (opt-in)        | Out-of-band check: simulates the docs.rs build environment under nightly rustdoc (`--cfg docsrs -D warnings`) to validate intra-doc links and `doc(cfg)` attributes (§2.3).                                                                                                                                                                          |
| **[`utils/mod-update.sh`](../utils/mod-update.sh)**                                   | Supply chain update pipeline         | Updates Rust toolchain (`rustup update`), `Cargo.toml` specifications (`cargo upgrade`), and `Cargo.lock` (`cargo update`); missing `rustup` or `cargo upgrade` is reported as a typed gap.                                                                                                                                                                       |
| **[`utils/render-a2-dumps.sh`](../utils/render-a2-dumps.sh)**                         | Forensic A2-Max tensor dumps         | Offline / forensic utility: regenerates intermediate activation dumps for A2-Max (`tests/fixtures/dumps_a2_max/`) using the C++ instrumentation in `utils/namcore-a2-dumps.patch`.                                                                                                                                                                   |
| **[`utils/namcore-a2-dumps.patch`](../utils/namcore-a2-dumps.patch)**                 | C++ A2 activation patch              | Tracked Git patch for `NeuralAmpModelerCore` v0.6.0 instrumenting per-layer intermediate activations in the A2-Max model.                                                                                                                                                                                                                            |
| **[`utils/debug/repro_oversample_hang.sh`](../utils/debug/repro_oversample_hang.sh)** | Debug safety wrapper for hangs       | Diagnostic safety harness for manual thread-hang reproduction under systemd scopes (`RuntimeMaxSec`, cgroup v2, resource isolation) with guaranteed teardown of descendant processes (§15).                                                                                                                                                          |

---

## 12. Code Coverage (Optional Development Aid)

Code coverage is not an official quality gate. Developers wishing to analyze coverage locally may install `cargo-llvm-cov` and execute:

```bash
cargo llvm-cov --features testing
```

Coverage reports serve as an informative diagnostic tool and are excluded from official verification scripts.

---

## 13. Out-of-Band Miri Concurrency & Aliasing Audit (`utils/miri-audit.sh`)

Standard release, CI, and long audit gates remain strictly **stable-only** ([§2](#2-static-analysis--quality-defense-utilslintssh), [§5](#5-decoupled-long-qa-audits-utilstests-longsh)). The `nightly` toolchain is never used by automated gates; it is used strictly on demand for isolated, out-of-band developer audits (`utils/miri-audit.sh` and `utils/docsrs-check.sh`).

To verify formal pointer provenance and
compliance with the Stacked Borrows / Tree Borrows aliasing models on the unsafe lock-free
primitives, an **opt-in, out-of-band** audit tool is provided (requires an isolated nightly
toolchain plus the `miri` component):

```bash
# Optional manual audit (NOT part of lints.sh / tests-quick.sh / tests-long.sh):
./utils/miri-audit.sh
```

**Relationship to Loom.** Miri *complements* `loom` (`utils/tests-long.sh` phase 7) and does not
replace it: `loom` exhaustively explores atomic interleavings to prove the **absence of data
races**, whereas Miri interprets concrete executions to prove **pointer provenance / aliasing
validity** (UB detection) that `loom` cannot see.

**Target matrix** (each target fails closed unless it executes a real minimum number of tests;
full log at `target/logs/miri-audit.log`):

| Target (`--lib` filter)  | Min tests | Notes                                                                                                                                      |
|:------------------------ |:---------:|:------------------------------------------------------------------------------------------------------------------------------------------ |
| `common::spsc`           | 28        | SPSC ring, GC cascade/overflow, status flags.                                                                                              |
| `dsp::pipeline::bridge`  | 4         | `DspBridge` double-buffer lock-free protocol (interior mutability via `UnsafeCell`).                                                       |
| `dsp::mirror_buf`        | 3         | Mirrored-buffer state machine; the `memfd_create`/`mmap` allocation tests are `#[cfg_attr(miri, ignore)]` (Miri cannot interpret the FFI). |
| `dsp::smoother`          | 5         | One-pole smoother primitives.                                                                                                              |
| `dsp::utils::delay_line` | 16        | `DelayLine` ring arithmetic.                                                                                                               |

**Flags.** `-Zmiri-disable-isolation` (host clock / pseudo-sleeps in timing paths) and
`-Zmiri-ignore-leaks`. The latter is required because the GC overflow tests deliberately leak the
displaced items (the `GcOverflowBuffer` controlled-leak contract) and the corrupted-slot tests leak
on purpose; without it, Miri reports those *intentional* leaks and drowns the provenance/aliasing
signal being audited.

**Accepted provenance limitation (GC overflow packing).** The `GcOverflowBuffer`
packs a heap pointer and a `type_id` into a single `AtomicU64`
(`GcItem::into_packed` / `from_packed`) — the only audited path that must
reconstruct a pointer from an address. The packing side exposes provenance
explicitly through the stable `expose_provenance` API and the recovery side uses
the stable `with_exposed_provenance_mut`; no raw integer-to-pointer `as` cast
remains. Because that recovery is still an address-to-pointer operation, Miri
emits a single (deduplicated) *"integer-to-pointer cast"* warning at that site.
This is **inherent to the 56-bit packing scheme**, not a soundness defect, and is
an accepted, documented limitation. Consequently the `common::spsc` target
validates the SPSC protocol and the GC cascade/overflow dispatch, but **not** the
strict provenance of the overflow packing itself. To study that site under
strict provenance, run with `MIRIFLAGS="-Zmiri-strict-provenance ..."` — it flags
the packing by design and is an out-of-band experiment, never a stable gate.

**Fail-closed gap.** When the nightly toolchain or the `miri` component is unavailable, the script
emits a typed `GAP:` line and exits non-zero (code `2`) — never a silent success. A target that
executes zero (or too few) tests is a hard failure (code `1`). The audit therefore never turns a
missing toolchain into a green gate.

---

## 14. Script Contract & Automation Disciplines

All automation scripts in `utils/` adhere to a lean, predictable execution contract designed for deterministic automation and transparent developer workflows.

### 14.1 Script Classes & Error Policies

Each script belongs to one of four strict architectural classes:

| Class | Typical Scripts | Error Handling Policy | Architectural Rationale |
| :----- | :--------------- | :--------------------- | :----------------------- |
| **Class A — Interactive** | `lints.sh`, `tests-quick.sh` | `set -euo pipefail`; aborts on first failure; `ERR` trap emits `RESULT: FAIL` | Real-time developer loops. Fast failure saves cycles and surfaces errors immediately. |
| **Class B — Decoupled Audit** | `tests-long.sh` | `set -o pipefail` (no `-e`, no `-u`); each phase records status independently; collects all failures; exit `1` at end if any phase failed | Nightly and pre-release audits. Must not abort prematurely so all failures are diagnosed in a single run. |
| **Class C — Build** | `build-release.sh` | `set -euo pipefail`; aborts on first failure | Build pipelines where intermediate failure invalidates downstream artifacts. |
| **Class D — Manual Tool** | `setup-third-party.sh`, `ensure_namcore_render.sh`, `consumer-smoke.sh`, `miri-audit.sh`, `docsrs-check.sh`, `check-model.sh`, `mod-update.sh`, `remote-simd-gate.sh`, `render-a2-dumps.sh` | `set -euo pipefail` | Targeted utilities executed under direct operator supervision or dedicated CI steps. |

### 14.2 Output Protocol & Exit Codes

Script output follows a structured stream suitable for automated monitoring:

1. **Phase Announcement:**

   ```text
   PHASE: <phase_name>
   ```

2. **Phase Completion:**

   ```text
   PHASE: <phase_name> PASS log=<path>
   PHASE: <phase_name> FAIL rc=<exit_code> log=<path>
   PHASE: <phase_name> GAP <technical_reason>
   ```

3. **Unified Final Result Line:**
   Every script terminates by printing exactly one final summary line:

   ```text
   RESULT: PASS|PASS_WITH_GAPS|FAIL (<seconds>s) fails=<list|none> gaps=<list|none>
   ```

4. **Exit Codes:**
   - `0`: Complete success (`PASS`), or success with documented gaps (`PASS_WITH_GAPS`) when not in strict mode.
   - `1`: Execution failure (`FAIL`), or unfulfilled `GAP` when invoked with `--strict`.
   - `2`: Syntax error, unknown option, or missing precondition.

Detailed execution logs are written to `target/logs/<script>-<phase>.log` via `tee`. The canonical `PHASE:`/`RESULT:` lines are also appended to `target/logs/summary.txt`; CI publishes that file through the local `.github/actions/qa-summary` composite action instead of re-grepping logs per job. On a failed long phase, `utils/tests-long.sh` additionally writes `target/logs/long-<phase>-failures.txt` with the libtest diagnostic headers and the line following each `panicked at`; the summary prints that file and falls back to the last 120 log lines only when it is absent. The integral log is never truncated and remains in the artifact.

Each `finish` from `utils/_lib.sh` also writes the typed carrier `target/logs/<script>-status.tsv` (discipline, verdict, comma-separated gaps, comma-separated fails). The summary stays a human-readable record, not a decision source.

### 14.3 Core Automation Principles

- **No Tests of Test Tooling:** Test suites validate DSP kernels, neural inference, parsers, and public APIs. Meta-tests that verify auxiliary scripts, receipt JSON schemas, or internal QA tools are forbidden. Tooling integrity is guaranteed by strict typing, minimal surface area, and routine execution.
- **Quick Never Provisions Infrastructure:** `utils/tests-quick.sh` operates entirely on already-available fixtures and pre-built tooling. It never clones vendor repositories, invokes CMake, or triggers out-of-band builds. Missing optional tooling or fixtures gracefully emits a typed `GAP`.
- **Long Collects All Failures:** `utils/tests-long.sh` executes all configured preflights and stress phases to completion, aggregating passes, gaps, and failures into a final diagnostic summary without early termination.
- **Convention Over Parameterization:** All scripts default to host capabilities (all online cores, zero artificial thermal cooldown, unpinned execution). Environment variables only exist to override defaults when explicitly required.

### 14.4 Canonical Command-Line Flags

Only the following canonical flags are supported across scripts (no legacy aliases):

| Script | Supported Flags | Notes |
| :------ | :--------------- | :----- |
| **All scripts** | `-h`, `--help` | Displays concise usage and exits 0. |
| `lints.sh` | None | Strict zero-flag policy; formatting is applied in-place during local development and verified via `CI=true` on CI. |
| `tests-quick.sh` | None | Fast first-line discipline; no allow-gaps switch and no simulation mode. |
| `tests-long.sh` | `--strict`, `--only <name>[,<name>]` | `--strict` promotes any `GAP` to `FAIL` (exit 1); `--only` runs specific phases for triage by display name (`preflight-*`, `catch-all`, `soak`, `isa-codegen`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`, `perf`; see §5.3). |
| `quality-dashboard.sh` | `--check [<contract.json>]`, `--fidelity-only`, `--save` | `--check` validates against baseline (the contract defaults to `docs/quality-contract.json` inside `nam_quality`); `--fidelity-only` skips the performance domain for a fidelity-only check; `--save` is reserved for human operators. `nam_quality` reads `target/bench_constants.env` when it exists; `build.rs` (`is_primary_build`) writes that file only when `CARGO_PRIMARY_PACKAGE` is set or, as a fallback on Cargo 1.99.1 (which does not pass `CARGO_PRIMARY_PACKAGE` to build scripts), when `OUT_DIR` is rooted under `<manifest_dir>/target`. An external consumer using its own `target/` directory never triggers the write; the only residual edge case is a consumer that explicitly points `CARGO_TARGET_DIR` inside the crate checkout's `target/` directory. |
| `consumer-smoke.sh` | None | Executes the external-consumer `x86-64-v3` and `aarch64` compilation contract in an isolated temporary directory (`${TMPDIR:-/tmp}/nam-consumer-XXXXXX`). |
| `remote-simd-gate.sh` | `--sde`, `--check-only`, `--skip-bench`, `--out <file>` | Runs AVX-512 parity and benchmarks under SDE emulation or native hardware. |

### 14.5 Canonical Environment Variables

#### Interface Variables (User & CI Control)

| Variable | Default | Scope | Purpose & Semantics |
| :-------- | :------- | :----- | :------------------- |
| `CARGO_BUILD_JOBS` | Unset (all cores) | Host / Cargo | Set in the host shell profile (`~/.bashrc`) to limit compiler parallelism on memory-constrained systems. Scripts never override or hardcode this. |
| `CI` | Unset | CI Runner | When `true`, instructs `lints.sh` to execute `cargo fmt --check` instead of modifying files in-place. |
| `NAM_THERMAL_COOLDOWN_S` | `0` | RT / Benchmark | Non-negative integer seconds for thermal settling between benchmark runs. |
| `NAM_BENCH_CORE` | `auto` | Benchmark | Core pinning: explicit CPU ID, isolated core if online, first allowed CPU, or unpinned fallback (`GAP`). |
| `NAM_PHASE_TIMEOUT_S` | Per-phase default | `tests-long.sh` | Overrides timeout duration across all audit phases for emergency debugging. |
| `NAM_RENDER_JOBS` | `nproc` | C++ Reference | Thread count for CMake compilation in `ensure_namcore_render.sh` (integer $\ge 1$). |
| `NAM_COMMUNITY_MODELS_SRC` | Unset | Setup | Optional source directory for external community models. |

#### Harness Variables (Internal Script → Rust Communication)

| Variable | Configured By | Read By | Purpose |
| :-------- | :------------- | :------- | :------- |
| `NAM_RUN_ID` | `utils/_lib.sh` | Test harnesses | Unique timestamped run identifier for logging and receipt isolation. |
| `NAM_REQUIRE_CPP_ORACLE` | `tests-long.sh --strict` | Parity suites | Enforces hard failure if the C++ reference render binary is unavailable. |
| `NAM_RT_STRICT` | `tests-long.sh --strict` | RT deadline tests | Enforces hard failure on uncontrolled environment (preflight failure) or latency violations fail-closed. |
| `NAM_CORE_RENDER_BIN` | `_lib.sh` / preflights | C++ Parity tests | Absolute path to the compiled upstream C++ `render` CLI binary. |
| `NAM_FIXTURES_DIR` | `_lib.sh` / preflights | Model loaders | Explicit path to test model fixtures directory. |
| `NAM_MODELS_DIR` | User / harness | Model loaders | Root directory for resolving model files hierarchically. |

### 14.6 Measured Durations & Budgets

Execution times measured on standard development hardware (16-core, SSD):

| Discipline / Script | Measured Duration | Budget Ceiling | Notes |
| :------------------- | :----------------- | :-------------- | :----- |
| `utils/lints.sh` | ~3s (warm cache, 2026-10-27) to ~75s (clean build) | $\le 2$ min | Covers fmt, check and clippy (four configurations), cross-target clippy, doc, SPDX, policies, and script budget. |
| `utils/tests-quick.sh` | ~16–87s (16–22s warm incremental; 64–87s cold, 2026-10-27…30) | $\le 5$ min | Covers golden-artifact freshness plus the full non-ignored test suite in the debug profile. |
| `utils/tests-long.sh` | ~10–13 min (625–749 s measured, 2026-10-27…30) | $\le 15$ min | Exhaustive catch-all (including ignored tests), soak, ISA/codegen, heap audit, RT deadline/jitter, Loom, and the perf gate. |
| `utils/quality-dashboard.sh` | ~134–164 s (fidelity-only; 2026-10-30) | $\le 3$ min | Evaluates full audio fidelity (ESR, SNR, MR-STFT) and latency envelope contracts. |

#### QA Footprint (2026-10-27)

Method: `wc -l` over the shell scripts of the `utils/` directory (shell), the Python
and awk support files of that directory (support glue) and the QA Rust surface
(engine `tools/qa/src/bin/*` (`nam-qa`, `publish = false`) + `src/bin/*` + `src/testing/qa` + `src/testing/receipt`; Plug/Pipe `src/bin/*`).
This is deliberately broader than the v2 estimate (≈20k), which excluded benchmark bins.

| Repository / Component | Shell | Py/awk | Rust QA | Total |
| :-- | --: | --: | --: | --: |
| NeuralAmpModeler-rs (Engine) | 1.170 | 0 | 12.959 (bins 5.856 + testing 7.103) | 14.129 |
| Plugin consumer reference | 1.257 | 670 | 3.024 | 4.951 |
| Standalone host consumer reference | 1.502 | 362 | 6.024 | 7.888 |
| **Total** | **3.929** | **1.032** | **22.007** | **26.968** |

Compared with v2: shell ≈4.7k → **3.9k** (−16%); the broad Rust QA count is
22.0k because it includes all bins (the v2 ≈20k excluded bench bins). The
published engine QA surface fell from 13.036 to **9.413 lines** (−27.8%): only
`nam_perf_gate` and `simd_probe` remain packaged, while `src/testing/*` stays
because the `testing` feature is a downstream dev-dependency.

---

## 15. Debugging Safety Harness & Hang Reproduction (`utils/debug/repro_oversample_hang.sh`)

[`utils/debug/repro_oversample_hang.sh`](../utils/debug/repro_oversample_hang.sh) is a developer safety wrapper designed for manual debugging and reproduction of suspected thread/process hangs under bounded host resources:

### 15.1 Operational Guarantees

1. **Cgroup v2 Resource Isolation:** Executes commands under `systemd-run --user --scope` with strict resource controls (`MemoryMax=1G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=64`), preventing runaway processes from exhausting memory, inducing OOM lockups, or freezing the graphical session.
2. **Reliable Whole-Cgroup Teardown:** Unlike shell `timeout` (which only signals the direct `cargo` parent while leaking spawned test binaries at 100% CPU), the harness sets `-p RuntimeMaxSec=<timeout_seconds>` on the systemd scope. Systemd signals the entire process tree (`SIGTERM` followed by `SIGKILL`), ensuring zero residual or orphaned processes survive completion.
3. **Durable Artifacts:** Captures timestamped output directly to `target/debug-logs/<label>.log` and stores the raw exit code in `target/debug-logs/<label>.exit`, followed by an automatic `dmesg` tail to capture kernel-level events.
4. **False-Negative Detection:** Inspects exit code 0 logs to verify the presence of the `test result:` summary banner. Logs that end abruptly without summary markers are flagged as suspicious and treated as inconclusive rather than false passes.

### 15.2 Usage Pattern

```bash
# General syntax:
./utils/debug/repro_oversample_hang.sh <timeout_seconds> <label> -- <cargo full command...>

# Example reproducing a single test under a 15-second cgroup deadline:
./utils/debug/repro_oversample_hang.sh 15 oversample-test -- \
  cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
    --ignored --nocapture --test-threads=1
```

For the historical post-mortem and architectural root cause analysis that motivated this harness, see [docs/postmortem-libm-symbol-interposition.md](postmortem-libm-symbol-interposition.md).

---

## 16. Host Memory Guards (OOM Containment)

The reference host has 16 GiB and no swap. Release test compilation
(`utils/tests-quick.sh`, `utils/tests-long.sh`) and Miri can exhaust RAM
alongside `rust-analyzer` and the IDE, triggering a **global OOM**; because
Electron's main process carries `oom_score_adj=300`, the kernel kills the IDE
instead of the compiler. Guards:

- **Parallelism:** Cargo defaults to utilizing all host cores; configure
  `CARGO_BUILD_JOBS` in the host shell profile (`~/.bashrc`) on resource-constrained systems.
- **Host Containment:** Heavy out-of-band builds can be wrapped with host-level
  cgroup containment if needed. Repository scripts do not embed automatic re-execs,
  preserving clean separation and zero timing interference.
- **IDE protection:** Lower the running IDE's (Antigravity / VS Code family)
  `oom_score_adj` (default `-500`; negative values need `sudo`) so a runaway build is targeted first.
- **Benchmarks:** RT measurements and benchmarks are strictly not memory-capped
  as cgroup reclaim could add jitter. `[profile.bench]` uses thin LTO; `codegen-units = 1` is
  retained for benchmark fidelity.

### 16.1 Test Profile Optimization Evaluation (`opt-level = 0` vs `opt-level = 2`)

Empirical benchmarking was conducted to assess whether setting `[profile.test] opt-level = 2` or `[profile.test.package.NeuralAmpModeler-rs] opt-level = 2` provided a net speedup in development and CI cycles:

- **Compilation Overhead:** Cold build time jumped from 15.8s (`opt-level = 0`) to 43.2s (`opt-level = 2`, +173%), while user CPU time increased 13.6× (from 30.8s to 419.3s). Peak memory usage (RSS) grew from 769 MB to 1.02 GB.
- **Incremental Cycle:** In iterative edit-compile-test cycles touching engine sources, compiler optimization passes added ~3.8s to compilation overhead. While runtime for `rt_constraints` dropped from 3.33s to 0.22s, the net iteration loop duration is negative or negligible.
- **Safety & Allocator Auditing:** Retaining baseline `opt-level = 0` guarantees that compiler dead-allocation or inlining passes do not subtly elide allocations monitored by `TrackingGuard` and heap audit tests.
- **Decision:** Retain baseline (`[profile.dev] opt-level = 0`, inherited by `[profile.test]`). Test suites are already engineered to execute under ~25s in debug mode.
