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
| **`fft-radix4-planner`** | No      | Radix-4 FFT benchmark planner        | [benches/fft_radix4_bench.rs](../benches/fft_radix4_bench.rs) (Radix-4 DIT FFT planning routines)                        |
| **`avx512`**             | No      | Research/measurement AVX-512 kernels | [src/math/activations/](../src/math/activations/), SIMD dispatch tables                                                  |
| **`rt-hardening`**       | No      | Real-time OS host tuning (Linux)     | [src/rt_hardening/](../src/rt_hardening/) (THP, mlockall, SCHED_FIFO, DAZ/FTZ, IRQ affinity)                             |
| **`on-demand`**          | No      | Developer diagnostics & harnesses    | Developer-only fixture generation and profiling harnesses (never enabled by CI)                                          |

---

## 2. Static Analysis & Quality Defense (`utils/lints.sh`)

[`utils/lints.sh`](../utils/lints.sh) is the **static first line of defense** of the project. It runs quickly (~45 seconds) and acts as an uncompromising quality gate that developers should execute regularly during active coding, before staging any commit, and as Receipt #1 in formal release certification.

Unlike test runners, `lints.sh` does not evaluate numerical audio outputs. Instead, it statically proves compilation cleanliness, zero-warning compliance, documentation hygiene, and policy conformance across the entire feature matrix.

### 2.1 The 9 Sequential Verification Gates

```mermaid
graph LR
    L1["1. cargo fmt"] --> L2["2. cargo check (7 axes)"]
    L2 --> L3["3. clippy (-D warnings)"]
    L3 --> L4["4. cargo doc + doctests"]
    L4 --> L5["5. SPDX Headers"]
    L5 --> L6["6. Anti-pattern tests/common"]
    L6 --> L7["7. Undocumented #[allow]"]
    L7 --> L8["8. doc(cfg) Validation"]
    L8 --> L9["9. README / MSRV Check"]
```

1. **`[1/9] In-Place Formatting (cargo fmt):`**
   Runs `cargo fmt --all` to enforce standard formatting across library code, binaries, examples, benches, and tests.
2. **`[2/9] Dynamic Compilation Matrix (cargo check):`**
   Verifies that the codebase compiles cleanly across 7 distinct feature axes, preventing hidden breakages behind conditional compilation:
   - Catch-all: `--all-targets --all-features`
   - Pure minimal core: `--lib --no-default-features`
   - All targets without defaults: `--all-targets --no-default-features`
   - Individual feature permutations: `fft-radix4-planner`, `dual-mono`, `testing`, `heap-audit`, and `rt-hardening`.
3. **`[3/9] Strict Static Analysis (cargo clippy):`**
   Executes Clippy across the same 7 feature permutations with `-D warnings`. Fails on any compiler or Clippy warning, dead code, or unidiomatic construct.
4. **`[4/9] Documentation & Doc-Tests (cargo doc + cargo test --doc):`**
   Builds full rustdoc documentation (`cargo doc --no-deps`) with zero warnings under `--all-features`, and executes every documentation test example in public APIs and README code blocks.
5. **`[5/9] SPDX License Notice Compliance:`**
   Scans every source code file to ensure the presence of the proper SPDX identifier comment and copyright notice (`Apache-2.0` for `NeuralAmpModeler-rs`).
6. **`[6/9] Anti-Pattern Enforcement in Tests:`**
   Ensures that helper files under `tests/common/` contain zero `#[test]` annotations, preventing duplicate or accidental executions outside official test entry points.
7. **`[7/9] Undocumented #[allow(...)] Policy Check:`**
   Enforces the project's zero-silent-suppression rule: every `#[allow(...)]` or `#![allow(...)]` compiler attribute must have an immediately preceding comment justifying why the lint suppression is necessary.
8. **`[8/9] doc(cfg) Annotation Consistency:`**
   Cross-references all `#[doc(cfg(feature = "..."))]` annotations against the declared features in [Cargo.toml](../Cargo.toml) to prevent orphan or mistyped documentation tags.
9. **`[9/9] Version & MSRV Synchronization:`**
   Asserts that the version and Minimum Supported Rust Version (MSRV `1.99.0`) stated in `README.md` exactly match [Cargo.toml](../Cargo.toml).

### 2.2 Developer Usage & Flags

```bash
# Standard interactive execution
./utils/lints.sh
```

### 2.3 docs.rs Parity Check (Opt-in, Nightly)

The crate page published by docs.rs is rendered by **nightly** rustdoc with `--cfg docsrs`. Documentation diagnostics that only nightly reports — broken intra-doc links, redundant explicit link targets, and missing `doc(cfg(feature = "..."))` seals — are invisible to the stable `lints.sh` gate (§2.1, phase 4). [utils/docsrs-check.sh](../utils/docsrs-check.sh) mirrors the docs.rs configuration to catch those regressions before publishing.

```bash
# Opt-in: requires an isolated nightly toolchain (never part of the stable gates)
./utils/docsrs-check.sh
```

The script exports `DOCS_RS=1` and `RUSTDOCFLAGS="--cfg docsrs -D warnings"`, derives the feature scope from `package.metadata.docs.rs` in [Cargo.toml](../Cargo.toml), then runs `cargo +nightly doc --locked --lib --no-deps` with that scope (`--lib` mirrors the docs.rs build). Exit `0` means zero rustdoc warnings; exit `2` is a typed gap (nightly or metadata tooling unavailable); any rustdoc warning fails with exit `1`.

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
    L0["Static Defense: utils/lints.sh (9 gates)"] --> Q1["Quick: freshness (nam_freshness artifacts-hard)"]
    Q1 --> Q2["Quick: cargo test --features testing (debug, non-ignored)"]
    Q2 --> L1["Long: 5 preflights + 8 phases (release, --include-ignored)"]
```

### 3.1 Quick QA Suite (`utils/tests-quick.sh`)

The quick suite executes in ~16–46 seconds (measured warm cache; budget ≤ 5 minutes) and runs two sequential phases:

1. **Freshness (`nam_freshness artifacts-hard`):**
   - **Scope:** Verifies golden artifact freshness and integrity via [src/bin/nam_freshness.rs](../src/bin/nam_freshness.rs). A `GENERATOR CHANGED` line is recorded as the `generator-drift` gap; a nonzero exit fails the phase.
2. **Tests (`cargo test --locked --features testing --no-fail-fast`):**
   - **Scope:** Runs every non-ignored test target in the **debug** profile with the `testing` feature. This includes structural/logic/parser/FSM tests, non-ignored numerical and parity checks, and the `libm_export_guard` ELF symbol gate. Float oracles here are smoke checks only; release-profile measurements belong to the long suite.
   - **Exclusions:** `#[ignore]` tests (soak, RT deadline/jitter, exhaustive audits, and the ignored proptest sweeps) are not invoked by the quick suite.

**Logs and gaps:** the phases write `target/logs/quick-freshness.log` and `target/logs/quick-tests.log`; tests that skip an unavailable optional C++ oracle append to `target/logs/skips.jsonl`, which the script turns into a `cpp-oracle-absent:<n>` gap. The quick suite never emits a receipt; it ends with the canonical `RESULT:` line. `utils/tests-long.sh --strict` (which exports `NAM_REQUIRE_CPP_ORACLE=1`) is the fail-closed counterpart for a missing C++ oracle.

### 3.2 Golden Vector Supply Chain & Freshness

Golden vectors validate inference determinism against pre-rendered reference outputs:

- **Freshness Manifest:** [tests/fixtures/golden_gen_build.sh](../tests/fixtures/golden_gen_build.sh) maintains the versioned `.golden_manifest.sha256` digest manifest. The quick `freshness` phase and the long `preflight-freshness` gate verify this integrity using `src/testing/freshness.rs` (`nam_freshness`). Any model modification without golden regeneration triggers a hard failure.
- **Model Resolution Order:** Models are resolved hierarchically: (1) `$NAM_MODELS_DIR`, (2) `third-party/community_models/`, (3) `tests/fixtures/models-nondist/`, (4) `tests/fixtures/models/`. Details are documented in [fixtures.md](fixtures.md).

### 3.3 Binary Defense & Compile-Time Segregation

- **Libm Export Guard:** [tests/libm_export_guard.rs](../tests/libm_export_guard.rs) is a fail-closed ELF symbol gate executed on the linked binary artifact in the `tests-quick.sh` tests phase and the `tests-long.sh` `catch-all` phase. It ensures internal `libm` mathematical symbols are hidden and cannot interpose over system C library symbols.
- **AVX-512 Segregation Contract:** AVX-512 code is strictly segregated at compile time via `#[cfg(feature = "avx512")]`. Default builds monomorphize only `Avx2Math`; no EVEX or ZMM opcodes are generated in default release artifacts. Runtime hardware capabilities and effective dispatch status are inspected using [`src/bin/simd_probe.rs`](../src/bin/simd_probe.rs) (`cargo run --quiet --bin simd_probe`).

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
- [tests/perf_soak.rs](../tests/perf_soak.rs): Numerical soak, thread contention, and long-running endurance checks.
- [tests/rt_constraints.rs](../tests/rt_constraints.rs): Real-time constraints, zero heap allocation audits, deadline, and jitter telemetry.
- [tests/dsp_core.rs](../tests/dsp_core.rs): Resamplers, noise gate FSM, circular buffers, and DSP pipeline stages.
- Standalone harnesses: `cabsim_stereo.rs`, `libm_export_guard.rs`, `loom_tests.rs`.

---

## 5. Decoupled Long QA Audits (`utils/tests-long.sh`)

The long-duration stress and audit suite executes comprehensive numerical validation, full-count proptests, live C++ parity, memory audits, and real-time deadline tests (~8–12 minutes measured, depending on hardware).

### 5.1 Blocking Preflight Gates

Before entering timed test execution, five preflight steps run:

1. **`preflight-render`:** Validates or builds the upstream C++ reference `render` CLI via `utils/ensure_namcore_render.sh`. Exit code `3` (mirror absent) becomes a `SKIPPED` phase with an `optional-dependency-missing` gap unless `--strict` is set.
2. **`preflight-catalog`:** Runs `cargo test --release --features testing --test models catalog_preflight` against the fixtures and V1/V2 goldens registered in [src/testing/catalog.rs](../src/testing/catalog.rs). The test enforces the strict counting invariant: `present + missing_required.len() + missing_optional.len() + known_gap_files == fixtures_checked`.
3. **`preflight-package`:** Checks crate packaging hygiene via `cargo package --list --allow-dirty`.
4. **`preflight-freshness`:** Runs the `check_freshness hard-fail` helper, i.e. `nam_freshness` in hard-fail mode, against `tests/fixtures/.golden_manifest.sha256`.
5. **`preflight-simd-probe` (Diagnostic):** Runs `cargo run --quiet --bin simd_probe` to record host SIMD feature bits, OS AVX-512 state, and dispatched engine. It does not require tests, but a nonzero exit still marks the phase `FAILED`.

Each preflight writes `target/logs/long-<phase>.log` (for example, `target/logs/long-preflight-catalog.log`).

### 5.4 Timeouts e ausência de oráculo ou baseline

Cada comando externo de uma fase roda sob `timeout --kill-after=30 <tempo>` (função `run_cmd`); exit `124`/`137` vira `FAIL rc=124 (timeout)`. `NAM_PHASE_TIMEOUT_S` sobrescreve o tempo de todos os comandos.

| Ausência | Sem `--strict` (padrão) | Com `--strict` (noturno com CPU isolada e oráculo) |
|:--|:--|:--|
| Oráculo C++ (`preflight-render`, exit 3) | `GAP optional-dependency-missing`; `catch-all` pula `cpp_parity::live_cross_validation` e registra `GAP cpp-parity-skipped:<n>` | `FAIL`; cobertura integral |
| Baseline de performance (`.performance-baselines/baseline-fingerprint.json`) | `GAP perf-baseline-missing`; a fase `perf` não executa | `FAIL` (`MISSING_BASELINE`; renovação só por operação humana, `--bootstrap-baseline`) |

Um GAP nunca é `PASS`: fica no recibo como `SKIPPED` e em `RESULT: PASS_WITH_GAPS`.

### 5.2 Sequential Execution Phases

1. **`catch-all` (`phase1`):** `cargo test --release --features testing --no-fail-fast -- --include-ignored`, skipping `rt_deadline::`, `rt_jitter::`, and `soak_test::` (the script's fixed `SKIP` list). This is the broad numerical matrix: live C++ parity, multi-sample-rate V2 goldens, cross-ISA checks, defense/invariant tests, the `libm_export_guard`, and the proptest sweeps.
2. **`soak` (`phase2`):** `cargo test --release --features testing --test perf_soak soak_test:: -- --ignored --test-threads=1`: 10M+ frames continuous endurance testing and SPSC queue contention sweeps ([tests/perf_soak.rs](../tests/perf_soak.rs)).
3. **`isa-codegen` (`phase3`):** Runs `cargo run --quiet --bin simd_probe` and then `cargo test --release --features testing --test isa_contract -- --nocapture`.
4. **`heap-audit` (`phase4`, filtered, `--include-ignored heap_audit`, profile `target/long/heap-audit`):** Strict verification of zero heap allocations on the audio processing hot path (`CountingAllocator` via `--features heap-audit`, including resampler, cabsim, A2 standard, A2-Max dynamic `test_a2_dyn_max_heap_audit`, and sequential chains).
5. **`rt-deadline` (`phase5`):** Resolves a core with `nam_perf_gate pick-core` (exporting `NAM_BENCH_CORE` unless the result is `GAP`) and runs `cargo test --release --features testing --test rt_constraints rt_deadline:: -- --ignored --nocapture`. Processing budget enforcement ($p99 < 1.33\text{ ms}$ for 64-sample blocks at 48 kHz across all SKUs, including A2-Max `test_rt_deadline_a2_max`), measured with `LatencyHistogram` (256 log-linear sub-bins, relative error bound $\le 12.5\%$) paired with `ExactLatencyReservoir` for exact $p99$ arbitration.
6. **`rt-jitter` (`phase6`):** Runs `cargo test --release --features testing --test rt_constraints rt_jitter:: -- --ignored --nocapture`: measures processing latency distribution under simulated CPU contention (`stress-1`, `stress-2`, `saturate-N`) across $\ge 16384$ blocks with initial 512-block cache/branch warmup to eliminate cold-start transients.
7. **`loom` (`phase7`):** `cargo test --release --test loom_tests --config "target.'cfg(target_arch = \"x86_64\")'.rustflags=['--cfg','loom']" -- --nocapture` (profile `target/long/loom`; `--config` soma `--cfg loom` às flags de `.cargo/config.toml` em vez de substituí-las, e `RUSTFLAGS` herdado é ignorado): Loom model verification for lock-free queues and atomic bitmasks.
8. **`perf` (`phase8`):** `nam_perf_gate run --check docs/quality-contract.json` (with `--bench-core` when `pick-core` succeeds and `--strict` when requested): automated benchmark regression check against the baseline contracts.

### 5.3 Structured Audit Receipt (`long-audit-receipt.jsonl`)

Each phase appends a structured record to `target/logs/long-audit-receipt.jsonl` using the Rust binary `nam_long_receipt`:

```json
{
  "phase_id": "phase1",
  "name": "catch-all",
  "status": "PASSED",
  "duration_ms": 42000,
  "tests_executed": 26,
  "gaps": [],
  "timestamp": "2026-08-14T03:00:00Z"
}
```

- **Receipt Statuses:** `PASSED`, `FAILED`, `SKIPPED`, `INCONCLUSIVE`, `SKIP_CAPABILITY`, `NOT_RUN`, `SIMULATED`. The suite line uses `PASSED`, `FAILED`, `COMPLETED_WITH_GAPS`, or `SIMULATED`.

- **Aggregated Overall Gaps:** The terminal `overall` record aggregates the deduplicated union of gaps from all phases prefixed by `phase_id` (`<phase_id>:<gap_id>`), guaranteeing full traceability and exact match with `overall.gaps.len()`.

- **Performance Regression Integration:** The `PERF_REGRESSION` human summary line is evaluated dynamically from `target/logs/regression_phase_receipt.jsonl` (within 24 hours / matching run ID). If absent, skipped via `--only`, or simulated, it emits `NOT_RUN`. Any failed regression evaluates to `FAIL`, promotes `OVERALL` to `FAILED`, emits `ERROR: perf_regression — FAILED`, and causes `utils/tests-long.sh` to fail closed.

- **Simulated Receipts:** Pre-registered receipts with status `SIMULATED` carry `tests_executed: 0` without executing any test. The derived suite line is `SIMULATED` and every verdict line reads `NOT_RUN` (`FIDELITY`, `RT_DEADLINE`, `RT_JITTER`, `PERF_REGRESSION`) — a simulated receipt never prints `OK`/`PASS`, and strict verification (`--strict`) always rejects it.

- **Typed Markers:** Discrepancies or skips must emit recognized markers parsed into typed gaps:

  - `[STATUS] SKIP_CAPABILITY reason="<detail>"`: Hardware or ISA absence.
  - `[STATUS] SKIP_OPTIONAL reason="<detail>"`: Missing non-distributable optional model.
  - `[STATUS] KNOWN_GAP id="<id>" reason="<detail>"`: Upstream gap under active tracking.
  - `[STATUS] INCONCLUSIVE reason="<detail>"`: Measurement bypass in non-calibrated environments.

- **Fail-Closed Invariant:** `overall: PASSED` is emitted only when all phases complete with `gaps: []` and no timed phase executed zero tests. `nam_long_receipt validate` is a **semantic** verifier — never a substring search: it enforces the canonical roster (`CANONICAL_ROSTER`: 5 preflights + `phase1`..`phase8` + terminal `overall`), rejecting empty/partial receipts, missing/duplicate/unknown IDs, a timed phase that claims `PASSED` with zero executed tests, and a declared `OVERALL:` verdict that disagrees with the verdict derived from the phase/gap evidence (a forged/injected `OVERALL: PASSED` line — including one pasted into a comment — is rejected at parse time). In strict mode (`--strict`) it additionally rejects any declared gap or failure. Verification is enforced via:

  ```bash
  cargo run --locked --features testing --bin nam_long_receipt -- validate --strict --out target/logs/long-audit-receipt.jsonl
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

### 5.4 Build Target Segregation (`CARGO_TARGET_DIR`)

`utils/tests-long.sh` segregates build outputs by profile under `target/long/`, so that switching `--features heap-audit` or `RUSTFLAGS="--cfg loom"` does not invalidate the cache of the other phases:

- `target/long/release`: preflights and phases 1, 2, 3, 5, 6 and 8.
- `target/long/heap-audit`: phase 4 only.
- `target/long/loom`: phase 7 only.

`target/long` is removed at the start only when `--only` is not used; `--only` keeps the caches for iterative reruns. The phase 4 filter is `--test rt_constraints --test cabsim_stereo --test models -- --include-ignored heap_audit`: the heap-audit tests are compiled only with `--features heap-audit`, so phase 1 never runs them, and the filter also covers the non-ignored ones.

### 5.5 Long Suite Workflow on GitHub Actions (`.github/workflows/long.yml`)

The long QA audit can be dispatched manually by maintainers at the end of an epic or before a release via GitHub Actions:

- **Trigger:** Manual (`workflow_dispatch`), through the GitHub Actions web UI (`Actions → Long QA Suite → Run workflow`) or the GitHub CLI:
  ```bash
  gh workflow run long.yml -f strict=true
  ```
- **Inputs:**
  - `phases`: Comma-separated phase IDs to execute (default: `'preflight-render,preflight-catalog,preflight-freshness,phase1,phase2,phase3,phase4,phase7'`).
  - `strict`: Boolean (default: `true`), enforcing fail-closed zero-gap execution.
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

   - In local runs and the QA dashboard ([utils/quality-dashboard.sh](../utils/quality-dashboard.sh)), only the AVX2-vs-AVX2 self-consistency check executes (`isa_self_consistency`). The inter-ISA matrix (`isa_parity_cross_isa`) is declared as an explicit gap (`cross_isa_matrix_requires_avx512_remote`), avoiding false failures while preventing unvetted assumptions.

2. **`FIDELITY: OK` Scope Boundary:**

   - A local test run reporting `FIDELITY: OK` validates that all active single-architecture mathematical reference oracles (`reference_oracle_f64`), committed golden vectors (`golden_vectors`), and C++ reference parity (`cpp_parity`) passed within contractual error budgets on the local host architecture.
   - **`FIDELITY: OK` does NOT cover cross-ISA parity.** Local self-consistency (AVX2 vs. AVX2) guarantees local determinism (an MSE = 0 tautology), not mathematical or numerical parity across differing SIMD instruction sets.

3. **Remote SIMD Verification (`utils/remote-simd-gate.sh`):**

   - Full cross-ISA parity gating (AVX2 vs. AVX-512) and comparative SIMD micro-benchmarks must be executed on AVX-512 hardware or under emulation using [utils/remote-simd-gate.sh](../utils/remote-simd-gate.sh):

     ```bash
     # Execute under Intel SDE emulator:
     ./utils/remote-simd-gate.sh --sde

     # Execute on native AVX-512 hardware:
     ./utils/remote-simd-gate.sh

     # Specify custom destination for the receipt JSONL:
     ./utils/remote-simd-gate.sh --out target/logs/remote-simd-receipt.jsonl
     ```

   - The script compiles with `--features avx512 --include-ignored`, executes the full `isa_parity` test battery across all topologies, and validates the resulting receipt `target/logs/remote-simd-receipt.jsonl` using Criterion statistical confidence intervals ($p < 0.05$).

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

The contract schema is enforced via typed serde structures in `src/testing/qa/`.

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
> - `FIDELITY: FAIL`: Numerical or spectral degradation detected. Requires immediate DSP investigation.
>
> Primary statistical authority for performance regressions belongs to `nam_perf_gate run --check` (`src/bin/nam_perf_gate.rs`) via Criterion confidence intervals; the quality contract serves as an integrated secondary check.

---

## 11. Utility Scripts Inventory

The `utils/` directory provides deterministic defense tools, inspection utilities, and test runners:

| Script                                                                                | Responsibility                       | Guarantees & Operation                                                                                                                                                                                                                                                                                                                               |
|:------------------------------------------------------------------------------------- |:------------------------------------ |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[`utils/lints.sh`](../utils/lints.sh)**                                             | Static analysis & quality defense    | 9 automated gates: in-place `fmt`, compilation & Clippy across 7 feature axes, `cargo doc`+doctests, SPDX, and policy checks (§2).                                                                                                                                                                                                                   |
| **`utils/tests-quick.sh`**                                                            | Agile first-line test suite          | Two-phase gate: `nam_freshness artifacts-hard` followed by `cargo test --locked --features testing --no-fail-fast` (debug, non-ignored targets). Runs in ~16–46s (measured warm; budget ≤ 5 min) (§3). |
| **`utils/tests-long.sh`**                                                             | Nightly & pre-release audit suite    | 5 preflights + 8 phases (`catch-all`, `soak`, `isa-codegen`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`, `perf`; ~8–12 min measured). Emits structured JSONL receipts (§5). Nightly & human operator execution.                                                                                                                                                                                           |
| **`utils/_lib.sh`**                                                                   | Shared shell library                 | Dynamic project path resolution, log formatting, QA-binary resolution and helper primitives.                                                                                                                                                                                                                                                         |
| **`utils/quality-dashboard.sh`**                                                      | Quality contract manager             | Executes fidelity and latency matrices; verifies or updates `docs/quality-contract.json` (§10). Periodic / on-demand evaluation.                                                                                                                                                                                                                     |
| **`src/bin/nam_perf_gate.rs`** | Performance regression wall | Baseline-gated Criterion evaluation (`nam_perf_gate run --check`); statistical confidence interval verification against ±5% noise bands. On-demand execution. |
| **`utils/setup-third-party.sh`**                                                      | Upstream vendor mirror manager       | Clones or updates pinned commits of `NeuralAmpModelerCore` and `NeuralAmpModelerPlugin`.                                                                                                                                                                                                                                                             |
| **`utils/ensure_namcore_render.sh`**                                                  | C++ Reference Render Builder         | Idempotently builds the C++ NAMCore render binary for parity oracles.                                                                                                                                                                                                                                                                                |
| **`utils/check-model.sh`**                                                            | Model inspection CLI wrapper         | Diagnostic / spot-check CLI: inspects `.nam` (JSON) and `.namb` (binary) models, reporting topology, metadata, and tensors.                                                                                                                                                                                                                          |
| **`utils/remote-simd-gate.sh`**                                                       | AVX-512 SIMD gate                    | Targeted SIMD gate: executes automated cross-ISA parity and comparative benchmarking on AVX-512 hardware or under Intel SDE emulation.                                                                                                                                                                                                               |
| **`utils/miri-audit.sh`**                                                             | Miri UB/aliasing audit (opt-in)      | Out-of-band audit: provenance and aliasing verification under the Miri MIR interpreter (Stacked/Tree Borrows) on lock-free targets. Requires isolated nightly; **never** run in stable gates (§13).                                                                                                                                                  |
| **[`utils/docsrs-check.sh`](../utils/docsrs-check.sh)**                               | Docs.rs parity check (opt-in)        | Out-of-band check: simulates the docs.rs build environment under nightly rustdoc (`--cfg docsrs -D warnings`) to validate intra-doc links and `doc(cfg)` attributes (§2.3).                                                                                                                                                                          |
| **[`utils/mod-update.sh`](../utils/mod-update.sh)**                                   | Supply chain update pipeline         | Supply chain update pipeline: updates Rust toolchain (`rustup update`), updates `Cargo.toml` specifications (`cargo upgrade`), and synchronizes `Cargo.lock` (`cargo update`).                                                                                                                                                                       |
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
|:----- |:--------------- |:--------------------- |:----------------------- |
| **Class A — Interactive** | `lints.sh`, `tests-quick.sh` | `set -euo pipefail`; aborts on first failure; `ERR` trap emits `RESULT: FAIL` | Real-time developer loops. Fast failure saves cycles and surfaces errors immediately. |
| **Class B — Decoupled Audit** | `tests-long.sh` | `set -o pipefail` (no `-e`, no `-u`); each phase records status independently; collects all failures; exit `1` at end if any phase failed | Nightly and pre-release audits. Must not abort prematurely so all failures are diagnosed in a single run. |
| **Class C — Build** | `build-release.sh` | `set -euo pipefail`; aborts on first failure | Build pipelines where intermediate failure invalidates downstream artifacts. |
| **Class D — Manual Tool** | `setup-third-party.sh`, `ensure_namcore_render.sh`, `miri-audit.sh`, `docsrs-check.sh`, `check-model.sh`, `mod-update.sh`, `remote-simd-gate.sh`, `render-a2-dumps.sh` | `set -euo pipefail` | Targeted utilities executed under direct operator supervision. |

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

Detailed execution logs are written to `target/logs/<script>-<phase>.log` via `tee`. The canonical `PHASE:`/`RESULT:` lines are also appended to `target/logs/summary.txt`; CI publishes that file through the local `.github/actions/qa-summary` composite action instead of re-grepping logs per job.

### 14.3 Core Automation Principles

- **No Tests of Test Tooling:** Test suites validate DSP kernels, neural inference, parsers, and public APIs. Meta-tests that verify auxiliary scripts, receipt JSON schemas, or internal QA tools are forbidden. Tooling integrity is guaranteed by strict typing, minimal surface area, and routine execution.
- **Quick Never Provisions Infrastructure:** `utils/tests-quick.sh` operates entirely on already-available fixtures and pre-built tooling. It never clones vendor repositories, invokes CMake, or triggers out-of-band builds. Missing optional tooling or fixtures gracefully emits a typed `GAP`.
- **Long Collects All Failures:** `utils/tests-long.sh` executes all configured preflights and stress phases to completion, aggregating passes, gaps, and failures into a final diagnostic summary without early termination.
- **Convention Over Parameterization:** All scripts default to host capabilities (all online cores, zero artificial thermal cooldown, unpinned execution). Environment variables only exist to override defaults when explicitly required.

### 14.4 Canonical Command-Line Flags

Only the following canonical flags are supported across scripts (no legacy aliases):

| Script | Supported Flags | Notes |
|:------ |:--------------- |:----- |
| **All scripts** | `-h`, `--help` | Displays concise usage and exits 0. |
| `lints.sh` | None | Strict zero-flag policy; formatting is applied in-place during local development and verified via `CI=true` on CI. |
| `tests-quick.sh` | None | Fast first-line discipline; no allow-gaps switch and no simulation mode. |
| `tests-long.sh` | `--strict`, `--only <name>[,<name>]` | `--strict` promotes any `GAP` to `FAIL` (exit 1); `--only` runs specific phases for triage by display name (`preflight-*`, `catch-all`, `soak`, `isa-codegen`, `heap-audit`, `rt-deadline`, `rt-jitter`, `loom`, `perf`; see §5.2). |
| `quality-dashboard.sh` | `--check [<contract.json>]`, `--save` | `--check` validates against baseline (the contract defaults to `docs/quality-contract.json` inside `nam_quality`); `--save` is reserved for human operators. |
| `remote-simd-gate.sh` | `--sde`, `--check-only`, `--skip-bench`, `--out <file>` | Runs AVX-512 parity and benchmarks under SDE emulation or native hardware. |

### 14.5 Canonical Environment Variables

#### Interface Variables (User & CI Control)

| Variable | Default | Scope | Purpose & Semantics |
|:-------- |:------- |:----- |:------------------- |
| `CARGO_BUILD_JOBS` | Unset (all cores) | Host / Cargo | Set in the host shell profile (`~/.bashrc`) to limit compiler parallelism on memory-constrained systems. Scripts never override or hardcode this. |
| `CI` | Unset | CI Runner | When `true`, instructs `lints.sh` to execute `cargo fmt --check` instead of modifying files in-place. |
| `NAM_THERMAL_COOLDOWN_S` | `0` | RT / Benchmark | Non-negative integer seconds for thermal settling between benchmark runs. |
| `NAM_BENCH_CORE` | `auto` | Benchmark | Core pinning: explicit CPU ID, isolated core if online, first allowed CPU, or unpinned fallback (`GAP`). |
| `NAM_PHASE_TIMEOUT_S` | Per-phase default | `tests-long.sh` | Overrides timeout duration across all audit phases for emergency debugging. |
| `NAM_RENDER_JOBS` | `nproc` | C++ Reference | Thread count for CMake compilation in `ensure_namcore_render.sh` (integer $\ge 1$). |
| `NAM_COMMUNITY_MODELS_SRC` | Unset | Setup | Optional source directory for external community models. |

#### Harness Variables (Internal Script → Rust Communication)

| Variable | Configured By | Read By | Purpose |
|:-------- |:------------- |:------- |:------- |
| `NAM_RUN_ID` | `utils/_lib.sh` | Test harnesses | Unique timestamped run identifier for logging and receipt isolation. |
| `NAM_REQUIRE_CPP_ORACLE` | `tests-long.sh --strict` | Parity suites | Enforces hard failure if the C++ reference render binary is unavailable. |
| `NAM_RT_STRICT` | `tests-long.sh --strict` | RT deadline tests | Rejects latency outliers and deadline budget misses fail-closed. |
| `NAM_CORE_RENDER_BIN` | `_lib.sh` / preflights | C++ Parity tests | Absolute path to the compiled upstream C++ `render` CLI binary. |
| `NAM_FIXTURES_DIR` | `_lib.sh` / preflights | Model loaders | Explicit path to test model fixtures directory. |
| `NAM_MODELS_DIR` | User / harness | Model loaders | Root directory for resolving model files hierarchically. |

### 14.6 Measured Durations & Budgets

Execution times measured on standard development hardware (16-core, SSD):

| Discipline / Script | Measured Duration | Budget Ceiling | Notes |
|:------------------- |:----------------- |:-------------- |:----- |
| `utils/lints.sh` | ~3s (warm cache, 2026-10-27) to ~75s (clean build) | $\le 2$ min | Covers fmt, check (7 axes), clippy (`-D warnings`), doc+doctests, SPDX, and policies. |
| `utils/tests-quick.sh` | ~15s (warm cache, 2026-10-27; `PASS_WITH_GAPS` for the absent C++ oracle) to ~46s | $\le 5$ min | Covers golden-artifact freshness plus the full non-ignored test suite in the debug profile. |
| `utils/tests-long.sh` | ~8–12 min (full suite) | $\le 15$ min | Exhaustive catch-all (including ignored tests), soak, ISA/codegen, heap audit, RT deadline/jitter, Loom, and the perf gate. |
| `utils/quality-dashboard.sh` | ~60s–90s | $\le 3$ min | Evaluates full audio fidelity (ESR, SNR, MR-STFT) and latency envelope contracts. |

#### QA footprint (métrica F-2, 2026-10-27)

Method: `wc -l` over the shell scripts of the `utils/` directory (shell), the Python
and awk support files of that directory (support glue) and the QA Rust surface
(engine `src/bin/*` + `src/testing/qa` + `src/testing/receipt`; Plug/Pipe `src/bin/*`).
This is deliberately broader than the v2 estimate (≈20k), which excluded benchmark bins.

| Repository | Shell | Py/awk | Rust QA | Total |
|:--|--:|--:|--:|--:|
| NeuralAmpModeler-rs | 1.170 | 0 | 12.959 (bins 5.856 + testing 7.103) | 14.129 |
| NAM-Plug | 1.257 | 670 | 3.024 | 4.951 |
| NAM-Audio-Pipe | 1.502 | 362 | 6.024 | 7.888 |
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

