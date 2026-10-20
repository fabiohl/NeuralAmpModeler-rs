<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Functional Testing & Human Certification Guide

**Audience:** Developers, QA Engineers, and Human Release Operators working with the `NeuralAmpModeler-rs` DSP engine crate.

---

## 1. Executive Summary & Scope

This guide defines the manual functional verification procedures and the formal human certification protocol for `NeuralAmpModeler-rs`. It establishes the operational protocols for:

1. **Manual Functional Tests (Tiers 1–3):** Targeted verification during development, subsystem integration, and endurance sweeps.
2. **Automated Runner Execution & Receipt Auditing:** Operating [utils/tests-quick.sh](../utils/tests-quick.sh) and [utils/tests-long.sh](../utils/tests-long.sh).
3. **Formal Release Certification:** Producing and verifying the 5 mandatory receipts required to sign off on a release commit.

> [!NOTE]
> For test suite architecture, placement rules, and mathematical oracles, see [testing.md](testing.md). For Criterion performance regression gating and baseline governance, see [benchmarks.md](benchmarks.md). For fixture management and golden vectors, see [fixtures.md](fixtures.md).

### Verification Hierarchy & Cadence

| **Tier / Runner**       | **Scope & Purpose**                                                                      | **Target Duration**          | **Execution Cadence**                              |
|:----------------------- |:---------------------------------------------------------------------------------------- |:---------------------------- |:-------------------------------------------------- |
| **Static Quality Gate** | 🔍 **Static Analysis & Lints:** Formatting, 7-axis check, Clippy, doc-tests, SPDX        | ~45s                         | Routine developer loop, pre-commit & Receipt #1    |
| **Tier 1 (Manual)**     | ⚡ **Smoke Test:** Sanity checks on model loading, dispatch, and basic inference         | ~2 min                       | After modifications to core DSP or loader modules  |
| **Tier 2 (Manual)**     | 🎯 **Feature Verification:** Block-invariance, reset idempotency, DSP stage transitions  | ~10–15 min                   | Milestone completion or major feature integration  |
| **Tier 3 (Manual)**     | 🛡️ **Robustness & Stress:** Continuous soak, SPSC burst contention, rate modulation      | ~20–30 min                   | Pre-release audits or major engine refactorings    |
| **Agile Quick Runner**  | 🚀 **Agile First-Line QA:** Structural debug tests, float/C++ parity, parser fuzzing     | ~2 min (hardware-dependent)  | Pre-commit check or local iterative validation     |
| **Long Audit Runner**   | 🔬 **Exhaustive Pre-Release Audit:** Soak, QA defenses, full matrix, heap audit, RT      | ~10 min (hardware-dependent) | Nightly builds and pre-release human certification |
| **Regression Guard**    | 🛡️ **Fail-Closed Gate Defense:** Missing baseline gate, contract rejection, sanitization | ~5s                          | Tooling validation or pre-release verification     |

---

## 2. Host & Environmental Prerequisites

Before executing manual stress scenarios, micro-benchmarks, or pre-release runner certifications, configure the host environment:

1. **CPU Frequency Scaling Governor:**
   Set the scaling governor to `performance` across physical cores to eliminate dynamic throttling, core migration, and timer jitter during real-time deadline tests.

   ```bash
   # Verify governor status across all cores
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
   # Expected output: performance
   ```

   The test harness probes the governor of the effective benchmark core (resolved via `pick_bench_core`: explicit `NAM_BENCH_CORE`, online isolated core from `/sys/devices/system/cpu/isolated`, or fallback `nproc / 2`).

2. **System Load & Thermal Stability:**
   Close resource-intensive background processes, IDE indexers, browsers, and background compilers to prevent scheduling interference.

3. **Process Priority & CPU Affinity (`taskset`, `nice`/`ionice`):**

   - The test runners ([`tests-quick.sh`](../utils/tests-quick.sh) and [`lints.sh`](../utils/lints.sh)) lower process priority automatically via `nice -n 19 ionice -c 3` unless `NAM_NO_LOW_PRIORITY=1` is set.
   - For timing-sensitive phases, [`tests-long.sh`](../utils/tests-long.sh) and [`tests-performance-regression.sh`](../utils/tests-performance-regression.sh) pin execution to a dedicated core via `taskset -c $BENCH_CORE`.

4. **Upstream Vendor Mirrors:**
   Ensure `third-party/NeuralAmpModelerCore/` is populated and matches the commit pinned in [`variables.env`](../variables.env):

   ```bash
   ./utils/setup-third-party.sh
   ```

5. **Toolchain & Dependency Supply Chain Maintenance:**
   Keep the active Rust toolchain and dependencies synchronized using [`utils/mod-update.sh`](../utils/mod-update.sh):

   ```bash
   ./utils/mod-update.sh
   ```

---

## 3. Manual Functional Verification Matrix (Tiers 1–3)

### 3.1 Tier 1: ⚡ Smoke Tests (High-Yield Verification)

- [ ] **1.1 Model Loading & Inference:** Load a `.nam` (JSON) and a `.namb` (binary) model using `load_and_build_model()`. Call `process()` with a 64-sample buffer of silence. *Expected:* Execution completes without panics; output contains finite floats.
- [ ] **1.2 Static Dispatch Coverage:** Load one model from each supported architecture family (WaveNet A1, WaveNet A2, LSTM, ConvNet, Linear). Process a single block per model. *Expected:* All five architectures instantiate and process successfully.
- [ ] **1.3 Dynamic Model Fallback:** Load a model with non-standard geometry (e.g., custom LSTM dimensions or WaveNet channel counts). *Expected:* Engine dispatches to the appropriate `Dyn` variant and processes valid output.
- [ ] **1.4 Malformed Model Rejection:** Attempt to load a truncated, malformed, or 0-byte file. *Expected:* Returns a strongly typed `Err(LoadError)` with a descriptive `NamErrorCode` without panicking.
- [ ] **1.5 Sample Rate Adaptation:** Load a 48 kHz model. Call `model.reset(44100, 64)`, process a block, then call `model.reset(96000, 64)` and process again. *Expected:* Clean state transition without memory leaks or audio discontinuities.

---

### 3.2 Tier 2: 🎯 Feature & Subsystem Verification

#### Domain 2A: Model & Pipeline Architecture

- [ ] **2A.1 Block-Size Invariance:** Processing identical audio split into `[32 + 32]` samples vs. a single `[64]` sample block yields equivalent output ($< 10^{-7}$ maximum difference). *Verifies:* Accurate receptive-field buffer tracking across block boundaries.
- [ ] **2A.2 Prewarm Determinism:** After `prewarm(n)`, the initial output block is deterministic and free of initialization transients exceeding noise gate thresholds.
- [ ] **2A.3 Reset Idempotency:** Executing `reset()` → `process()` → Output A, followed by `reset()` → `process()` → Output B on the same input produces $A = B$.
- [ ] **2A.4 Container Crossfade:** In a `SlimmableContainer`, trigger a `Full → Lite` profile swap during active audio processing. *Expected:* Smooth 32 ms equal-power crossfade without audible clicks.
- [ ] **2A.5 Lock-Free SPSC Model Hot-Swap:** Push a new model via the lock-free SPSC channel while the audio thread runs. *Expected:* Model swap occurs without priority inversion or audio dropouts; the old model is dropped off-RT by the GC thread.

#### Domain 2B: DSP Pipeline Stages (Gate, Resampler, CabSim)

- [ ] **2B.1 Noise Gate FSM:** Input digital silence → gate triggers fade-out and clamps output to zero. Input signal above threshold → gate smoothly ramps open without overshoot.
- [ ] **2B.2 Oversampling Anti-Aliasing:** Pass a full-scale high-frequency sine sweep through a non-linear WaveNet model at $2\times$ and $4\times$ oversampling. *Expected:* Aliasing suppression meets thresholds in [audio_fidelity_map.md](audio_fidelity_map.md).
- [ ] **2B.3 Native Rate Passthrough:** When host sample rate matches model native rate (e.g., 48 kHz $\to$ 48 kHz), `NamResampler` enters zero-overhead passthrough.
- [ ] **2B.4 Multi-Rate Resampling:** Verify operation across 44.1, 48, 88.2, 96, and 192 kHz. *Expected:* High SNR preservation and clean phase response.
- [ ] **2B.5 CabSim Impulse Response Engine:** Load a standard `.wav` IR into `ConvEngine`. Process audio and verify convolution. Clear the IR and verify clean bypass.

#### Domain 2C: RT-Safety & Allocation Watchdog

- [ ] **2C.1 Zero-Allocation Hot-Path:** Build with `--features heap-audit`. Process 10,000 blocks on the audio thread. *Expected:* `CountingAllocator` reports exactly zero heap allocations in `process()`.
- [ ] **2C.2 Denormal Protection (FTZ/DAZ):** Process low-level decaying signals (below $-120\text{ dBFS}$). *Expected:* FTZ/DAZ flags prevent denormal performance penalties.
- [ ] **2C.3 Robustness Against Extreme Floats:** Feed out-of-range ($\pm 100.0$), NaN, or $\pm\infty$ float buffers to `process()`. *Expected:* Soft-clipping/sanitization occurs; zero panics or undefined behavior.
- [ ] **2C.4 Memory Leak Audit:** Repeatedly load and unload 100 models in a loop off-RT. *Expected:* Process RSS stabilizes; zero leaked file descriptors or memory handles.

---

### 3.3 Tier 3: 🛡️ Robustness & Stress Scenarios

- [ ] **3.1 Soak Endurance (10M+ Frames):** Execute continuous inference over $>10$ million frames with randomized block sizes ($16$ to $256$ samples). *Expected:* Zero panics, zero memory drift, strictly finite float output.
- [ ] **3.2 SPSC Command Contention:** Submit 1,000 rapid model-load commands via SPSC while the audio thread processes minimal blocks. *Expected:* Zero dropped commands, GC drains cleanly, audio deadline is maintained.
- [ ] **3.3 Dynamic Sample Rate Modulation:** Dynamically change sample rate every 100 blocks (44.1 $\leftrightarrow$ 48 $\leftrightarrow$ 96 kHz). *Expected:* Resampler ring buffers reinitialize cleanly without memory corruption or output artifacts.
- [ ] **3.4 Extended CabSim Partitioning:** Load extreme-length impulse responses ($2^{20}$ samples). *Expected:* Frequency-domain delay-line (FDL) partitioning allocates only during off-RT setup, retaining zero allocations during processing.

---

### 3.4 Interactive Test Harness Reference

The following pattern illustrates standard manual functional verification:

```rust
use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::models::NamModel;
use std::path::Path;

fn verify_model_invariance(model_path: &Path) -> anyhow::Result<()> {
    let sys = SystemSnapshot::capture();
    let mut mp = load_and_build_model(model_path, &sys, false, LoadOptions::default())?;
    let model = mp.model_l.as_mut().expect("Model instance expected");

    let input = vec![0.1f32; 128];
    let sr = 48000;

    // 1. Process as two 64-sample blocks
    let mut out_split = vec![0.0f32; 128];
    model.reset(sr, 128)?;
    model.process(&input[..64], &mut out_split[..64]);
    model.process(&input[64..], &mut out_split[64..]);

    // 2. Process as one contiguous 128-sample block
    let mut out_full = vec![0.0f32; 128];
    model.reset(sr, 128)?;
    model.process(&input, &mut out_full);

    // 3. Verify block-size invariance
    let max_diff = out_full.iter().zip(&out_split)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_diff < 1e-7, "Block-size invariance violation: max diff = {max_diff}");
    Ok(())
}
```

---

## 4. Automated Runner Protocols & Artifacts Inventory

> [!IMPORTANT]
> **Operator Delegation Policy:**
> Automated execution of `utils/tests-long.sh`, `utils/tests-performance-regression.sh --bootstrap-baseline`, and `utils/quality-dashboard.sh --save` is restricted to human operators. These scripts must run on calibrated, isolated machines.

### 4.1 Agile Quick Suite (`utils/tests-quick.sh`)

The quick suite is the primary automated gate for local development (~2 minutes).

```bash
# Standard local verification (skips missing optional vendor fixtures gracefully)
./utils/tests-quick.sh

# Strict release-gate mode (promotes any missing fixture to a hard failure)
NAM_QUICK_STRICT=1 ./utils/tests-quick.sh
```

- **Phase 1 (Structural & Logic, Debug):** Unit tests, DSP math logic, parsers, FSM transitions, and SPSC channels.
- **Phase 2 (Measurement Oracles & Parity, Release):** Evaluates production float codegen across `golden_vectors`, `reference_oracle_f64`, `spectral_fidelity`, `linear_fft_test`, and C++ parity (`quick_parity`).
- **Phase 3 (Parser Fuzzing, Release `--ignored`):** Capped `proptest` sweeps on `.nam` and `.namb` format inputs.

**Expected Outcome:**

- Local runs: `FIDELITY: OK` or `FIDELITY: INCOMPLETE` (with documented `GAP:` entries if vendor mirrors are absent) and `OVERALL: PASSED` (exit status 0).
- Release certification: Must run with `NAM_QUICK_STRICT=1` and yield `FIDELITY: OK` with `OVERALL: PASSED` (zero gaps, exit status 0).

---

### 4.2 Long Audit Suite (`utils/tests-long.sh`)

The long audit suite provides exhaustive pre-release validation (~10 minutes).

```bash
# Standard nightly audit (tolerates gaps for missing optional fixtures; exit status 0)
./utils/tests-long.sh

# Strict pre-release mode (FAIL-CLOSED: converts any gap or skipped test into exit status 1)
./utils/tests-long.sh --strict-pre-release
```

- **Preflight Gates:** `preflight-render`, `preflight-catalog`, `preflight-package`, `preflight-freshness`, `preflight-meta`, and diagnostic `preflight-simd-probe`.

- **7-Phase Battery:** (1) Soak & Concurrency, (2) Defense Scripts & Linking Guards, (3) Full Live Parity & Proptests, (4) Heap Audit, (5) RT Deadline Gate, (6) RT Jitter Telemetry, (7) Loom Model Checking.

- **Receipt Validation:**

  ```bash
  cargo run --locked --features testing --bin nam_long_receipt -- validate --strict --out target/logs/long-audit-receipt.jsonl
  ```

---

### 4.3 Automated Regression Guard Suite (`tests/scripts/test_regression_guard.sh`)

[`tests/scripts/test_regression_guard.sh`](../tests/scripts/test_regression_guard.sh) is an automated regression test suite designed to verify the fail-closed defenses and diagnostic invariants of the performance gate and quality contract verification tooling:

```bash
# Execute the regression guard test suite locally:
bash tests/scripts/test_regression_guard.sh
```

- **Scenario 1 (Absence of baseline in `.performance-baselines/`):** Shelves active baselines and asserts that `utils/tests-performance-regression.sh --check` fails closed (`exit 1`) with `MISSING_BASELINE`.
- **Scenario 2 (Contract format rejection & domain separation):**
  - **Scenario 2a:** Rejects legacy `.txt` contracts (`docs/quality-contract.txt`) with `ERROR` and `exit 2` prior to test execution.
  - **Scenario 2b:** Verifies `nam_quality verify` with `docs/quality-contract.json` against `tests/fixtures/qa/report_gate_not_verified.jsonl`, asserting `FIDELITY: OK`, `PERFORMANCE: NOT_VERIFIED`, and `CONTRACT VIOLATED` (`exit 1`), proving PERF-006 domain separation.
- **Scenario 3 (Stale log truncation):** Writes a unique dummy marker to `target/logs/regression-check.log` and asserts truncation upon starting `check_regression()`.
- **Scenario 4 (Baseline store sanitization):** Synthesizes nested baselines (`RT_Dummy/ci-baseline/ci-baseline/...`) and asserts `nam_perf_gate restore-baseline` strips nesting and preserves a single flat baseline layer.
- **Scenario 5 (CPU Core Pinning & Mock Sysfs Tests):** Validates deterministic CPU core selection logic via mock sysfs trees (`/sys/devices/system/cpu/isolated`, `online`, `offline`), explicit `NAM_BENCH_CORE` overrides, and error paths.
- **Scenario 6 (Benchmark Binary Selection & Fail-Fast Smoke Test):** Exercises binary resolution prioritizing `nam_benches` while supporting legacy `nam_benchmarks` fallback, fail-fast invocation of `--list`, and execution error propagation.

---

### 4.4 Disk Logs & Diagnostic Artifacts Inventory

All test runners persist execution logs under `target/logs/` (and `target/debug-logs/` for debug harnesses):

| File Path                                        | Generating Component / Phase           | Diagnostic Contents                                                                                                                          |
|:------------------------------------------------ |:-------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------- |
| **`target/logs/quick-receipt.txt`**              | `tests-quick.sh` (Final)               | Summary receipt with `FIDELITY:`, `GAP:`, and `OVERALL:` status.                                                                             |
| **`target/logs/quick-phase1.log`**               | `tests-quick.sh` (Phase 1)             | Structural unit tests, DSP logic, and channel checks (Debug profile).                                                                        |
| **`target/logs/quick-phase2.log`**               | `tests-quick.sh` (Phase 2)             | Float measurement oracles, golden vectors, and C++ `quick_parity` (Release profile).                                                         |
| **`target/logs/quick-phase3.log`**               | `tests-quick.sh` (Phase 3)             | Capped proptest parser fuzzing logs.                                                                                                         |
| **`target/logs/long-audit-receipt.jsonl`**       | `tests-long.sh` (Final)                | Machine-readable structured audit receipt (per-phase JSON entries).                                                                          |
| **`target/logs/catalog_preflight.log`**          | `tests-long.sh` (Preflight 2)          | Fixture catalog discovery, SHA-256 manifest checks, and missing fixture diagnostics.                                                         |
| **`target/logs/meta_coherence.log`**             | `tests-long.sh` (Preflight 4)          | Catalog definition to test registration cross-validation.                                                                                    |
| **`target/logs/simd_probe.log`**                 | `tests-long.sh` (Preflight diagnostic) | CPU feature bits, OS AVX-512 state, Cargo feature status, and dispatched backend.                                                            |
| **`target/logs/package-list.err`**               | `tests-long.sh` (Preflight 3)          | Diagnostics from `cargo package --list` crate packaging validations.                                                                         |
| **`target/logs/cmake-configure.log`**            | `utils/ensure_namcore_render.sh`       | CMake configuration output when compiling C++ `render` CLI.                                                                                  |
| **`target/logs/cmake-build.log`**                | `utils/ensure_namcore_render.sh`       | Compilation logs for the C++ reference render binary.                                                                                        |
| **`target/logs/phase1-soak.log`**                | `tests-long.sh` (Phase 1)              | Numerical soak logs, SPSC queue sweeps, and endurance metrics.                                                                               |
| **`target/logs/phase2-defense-scripts.log`**     | `tests-long.sh` (Phase 2)              | Structural invariant checks, QA defenses (`qa_defense.rs`), and symbol export guards (`libm_export_guard.rs`).                               |
| **`target/logs/phase3-proptests-parity.log`**    | `tests-long.sh` (Phase 3)              | Full C++ parity comparisons, multi-SR goldens, and 100k-case proptests.                                                                      |
| **`target/logs/subphase-isa-parity.log`**        | `tests-long.sh` (Phase 3 Subphase)     | Cross-ISA determinism validation logs (`isa_parity`).                                                                                        |
| **`target/logs/phase4-heap-audit.log`**          | `tests-long.sh` (Phase 4)              | `CountingAllocator` memory interceptor reports verifying zero allocations on RT thread.                                                      |
| **`target/logs/phase5-rt-deadline.log`**         | `tests-long.sh` (Phase 5)              | Latency histograms and deadline statistics ($p99 < 1.33\text{ ms}$). High-resolution linear bins (0.5 µs resolution via `LatencyHistogram`). |
| **`target/logs/phase6-rt-jitter.log`**           | `tests-long.sh` (Phase 6)              | Real-time jitter telemetry and thread contention profiles.                                                                                   |
| **`target/logs/phase7-loom.log`**                | `tests-long.sh` (Phase 7)              | Loom model-checking state-space exploration logs.                                                                                            |
| **`target/logs/regression_phase_receipt.jsonl`** | `tests-performance-regression.sh` / S2 | Machine-readable performance regression phase receipts integrated fail-closed into audit summary.                                            |
| **`target/logs/perf-regression.log`**            | `tests-performance-regression.sh`      | Criterion evaluation logs and statistical delta metrics across benchmark SKUs.                                                               |
| **`target/logs/regression-check.log`**           | `tests-performance-regression.sh`      | Detailed Criterion output and statistical confidence interval results during regression checks (truncated on invocation start).              |
| **`target/logs/remote-simd-receipt.jsonl`**      | `utils/remote-simd-gate.sh`            | Structured multi-ISA verification receipt emitted from remote or emulated AVX-512 runs.                                                      |
| **`target/logs/remote-simd-parity.log`**         | `utils/remote-simd-gate.sh`            | Detailed parity output comparing AVX2 and AVX-512 execution streams.                                                                         |
| **`target/logs/dashboard/`**                     | `utils/quality-dashboard.sh`           | Structured JSON outputs for individual model fidelity and latency benchmarks.                                                                |
| **`target/debug-logs/<label>.log`**              | `utils/debug/repro_oversample_hang.sh` | Timestamped execution log and command output under systemd cgroup v2 resource limits.                                                        |
| **`target/debug-logs/<label>.exit`**             | `utils/debug/repro_oversample_hang.sh` | Raw exit status code emitted by the wrapped command or systemd scope timeout (`143` on SIGTERM, `137` on SIGKILL).                           |
| **`~/.cache/neural-amp-modeler-rs/crash-*.txt`** | Runtime Panic Hook                     | Diagnostic crash reports rendered without heap allocations.                                                                                  |

---

### 4.5 Developer Diagnostic Harness & Hang Reproduction (`utils/debug/repro_oversample_hang.sh`)

When investigating suspected thread/process hangs or regressions under high contention, developers can isolate execution inside a systemd user scope:

```bash
# General syntax:
./utils/debug/repro_oversample_hang.sh <timeout_seconds> <label> -- <command...>

# Example:
./utils/debug/repro_oversample_hang.sh 15 oversample-test -- \
  cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
    --ignored --nocapture --test-threads=1
```

- Enforces strict cgroup v2 limits (`MemoryMax=1G`, `CPUQuota=100%`, `TasksMax=64`).
- Kills all descendant processes via `RuntimeMaxSec` on the scope itself, preventing orphaned test processes.
- Records logs to `target/debug-logs/<label>.log` and raw exit codes to `target/debug-logs/<label>.exit`.
- Validates the presence of `test result:` in the log to detect truncated or false-negative results.

---

### 4.6 Known Benign Signals (False-Alarm Catalog)

> [!NOTE]
> **Compilation Profile Segregation:**
> To eliminate build cache thrashing across phases with incompatible Cargo feature sets, `utils/tests-long.sh` automatically segregates `CARGO_TARGET_DIR` per profile (e.g. `target/profile_default`, `target/profile_heap_audit`, and `target/profile_loom`). Independent logs in `target/logs/` are preserved across runs.

Signals that routinely appear in runner logs but are **expected**; they must not be escalated as defects or regressions. Each row cites the artifact that emits it (paths under `target/logs/` are runtime artifacts, not versioned files).

| Signal (log excerpt)                                                                         | Emitting artifact                                                                                                      | Why it is benign                                                                                                                                                                                                  | Correct action                                                                                                   |
|:-------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------- |
| `isa_parity_cross_isa: SKIP_CAPABILITY`; `remote-simd-receipt.jsonl` skips                   | `target/logs/subphase-isa-parity.log`, `target/logs/remote-simd-receipt.jsonl`, `target/logs/long-audit-receipt.jsonl` | The local CPU lacks AVX-512; cross-ISA parity is a **separate** remote/emulated gate, not a local-coverage gap.                                                                                                   | Confirm `utils/remote-simd-gate.sh` ran separately; do not read the local skip as a failure.                     |
| `avx512_opt_in_not_run` (receipt gap)                                                        | `target/logs/simd_probe.log`, `target/logs/long-audit-receipt.jsonl`                                                   | AVX-512 is an opt-in Cargo feature (`--features avx512`); the default runner never compiles it.                                                                                                                   | Expected on a default run; only a scheduled `--features avx512` run covers it.                                   |
| `[RECEIPT] status=CHARACTERIZED label=…-saturate-N … violations=137/16384`                   | `target/logs/phase6-rt-jitter.log`                                                                                     | Saturation stress is a **CHARACTERIZATION** ("environmental characterization, NOT a deadline gate"); `status=CHARACTERIZED` means the harness completed, not that jitter passed (S5-T2 relabeled the old `PASS`). | Read the `violations=` counter; never treat the receipt token as a deadline assertion.                           |
| `… has been running for over 60 seconds`                                                     | `target/logs/quick-phase1.log`, `target/logs/phase1-soak.log`                                                          | Long debug-profile tests (soak, block-invariance, deterministic-energy); the default test runner prints this after 60 s.                                                                                          | Informational only; not a hang — wait for completion.                                                            |
| `Found N outliers among 100 measurements (x%)`                                               | `target/logs/regression-check.log`, `target/logs/perf-regression.log`                                                  | Criterion reports high-severity sampling outliers; the regression verdict uses the median delta and p-value, not the outlier banner.                                                                              | Read Δ median and p; ignore the outlier count.                                                                   |
| `[BOOTSTRAP] ⚠ This operation must be performed by a human operator.`                        | `utils/tests-performance-regression.sh --bootstrap-baseline` output (`target/logs/regression-check.log`)               | Baseline bootstrap is operator-only by governance (§5.2); AI runs are forbidden from it.                                                                                                                          | Do not run `--bootstrap-baseline`; report and wait for the operator.                                             |
| `SIMULATED` (e.g. `overall SIMULATED (0 tests, 0 ms, 13 gaps)`)                              | `utils/tests-long.sh --simulate` (`target/logs/long-audit-receipt.jsonl`)                                              | Dry-run scaffolding executed before the real audit; simulated receipts carry `gaps: [simulated]`.                                                                                                                 | Not evidence — ignore until the real (non-`--simulate`) run.                                                     |
| `RT_WaveNet_Lite_CH12` `+8.56%` (`p=0.04`) — `Change within noise threshold.`                | `target/logs/regression-check.log`                                                                                     | Sub-5 % variance band under ambient/thermal noise (F-10/F-12); the effective `noise_threshold` classifies it as non-significant.                                                                                  | Re-measure on an isolated pinned core before treating as a regression; never bootstrap a baseline to silence it. |
| `[JIT_PINS] pinned=false` / `pinned=false` in phase6 receipts; P50 ≈ 98 µs vs phase5 ≈ 45 µs | `target/logs/phase6-rt-jitter.log`                                                                                     | Jitter runs in the generic multi-core environment unless the operator pins with `taskset -c $NAM_BENCH_CORE`; the delta is environmental, never a regression.                                                     | Re-run phase6 under `taskset -c $NAM_BENCH_CORE` for a phase5-comparable P50; read `pinned=` before comparing.   |
| `[AVISO] --check pós---bootstrap valida só repetibilidade`                                   | `bootstrap-repo.sh` output                                                                                             | `--check` immediately after `--bootstrap-baseline` in the same session proves self-consistency only (F-12); the REGRESSION GATE verdict needs a baseline from another session/commit (`noise_threshold=0.05`).    | Do not read the paired green as absence of regression.                                                           |
| `known_gap:condition_lstm_rejected_by_both`                                                  | `target/logs/phase3-proptests-parity.log`                                                                              | The WaveNet model whose `condition_dsp` is an LSTM is refused by **both** oracles by design (Rust fail-closed policy; C++ `condition_size` mismatch) — see [`cpp_parity_map.md`](cpp_parity_map.md) §7.5.         | A declared gap, not a pass and not a defect; keep it declared in receipts.                                       |

---

## 5. Release Certification Protocol (5 Mandatory Receipts)

A build is certified for release only when **all five mandatory receipts** are produced and verified against the **same clean git commit hash**.

### 5.1 The Five Mandatory Receipts

| #   | Receipt Artifact        | Generating Command                                              | Execution Flags / Env                    | Blocking Criteria                                                                                                                                 |
|:---:|:----------------------- |:--------------------------------------------------------------- |:---------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `lints.log`             | `utils/lints.sh`                                                | `--all-targets --all-features` (script)  | Exit 0; zero compiler/clippy/rustdoc warnings; SPDX and anti-pattern checks green; AVX-512 compile-time segregation confirmed.                    |
| 2   | `quick-strict.log`      | `utils/tests-quick.sh`                                          | `NAM_QUICK_STRICT=1`                     | Exit 0; `FIDELITY: OK`; `OVERALL: PASSED`; zero `GAP:` entries.                                                                                   |
| 3   | `long-strict.log`       | `utils/tests-long.sh --strict-pre-release`                      | `--strict-pre-release`                   | Exit 0; `OVERALL: PASSED`; `gaps: []`; receipt validated via `nam_long_receipt validate --strict`.                                                |
| 4   | `perf-check.log`        | `utils/tests-performance-regression.sh --check`                 | Pre-approved baseline (see §5.2)         | Exit 0; machine regression verdict green (`nam_perf_gate verdict` over Criterion `change/estimates.json`) against a pre-approved baseline commit. |
| 5   | `quality-contract.json` | `utils/quality-dashboard.sh --check docs/quality-contract.json` | `docs/quality-contract.json` (committed) | Exit 0; all audio fidelity envelopes (ESR/SNR/MR-STFT) and f64-oracle checks satisfied; `regression_gate` is not `NOT_VERIFIED`.                  |

> [!IMPORTANT]
> **Same-Commit Binding:** All five receipts must be produced on the **exact same clean git commit**. The release record (§6) binds them together with their SHA-256 digests. Receipts produced on dirty working trees or across different revisions are invalid.

### 5.2 Baseline Bootstrap Governance

`utils/tests-performance-regression.sh --bootstrap-baseline` and `utils/quality-dashboard.sh --save` are **independent ceremonies** executed exclusively by a human operator:

- **Distinct Producer Commit:** The performance baseline must be generated and approved on an earlier reference commit. Bootstrapping a baseline on the same commit being certified only demonstrates self-consistency, not the absence of regressions.
- **Controlled Hardware:** Baselines must be generated on isolated machines with the CPU governor set to `performance` and background contention eliminated.
- **Intentional Updates Only:** Baselines and contracts are updated only when performance shifts or algorithmic changes are intentional, verified, and documented.

### 5.3 Human Certification Checklist

- [ ] Working tree is clean on the target commit (`git status --porcelain` is empty).
- [ ] **Receipt 1 (`lints.log`):** `utils/lints.sh` passes with zero warnings.
- [ ] **Receipt 2 (`quick-strict.log`):** `NAM_QUICK_STRICT=1 utils/tests-quick.sh` passes with `FIDELITY: OK`, `OVERALL: PASSED`, and zero gaps.
- [ ] **Receipt 3 (`long-strict.log`):** `utils/tests-long.sh --strict-pre-release` passes with `OVERALL: PASSED` and `gaps: []`.
- [ ] **Receipt 4 (`perf-check.log`):** `utils/tests-performance-regression.sh --check` passes against an approved baseline from a distinct reference commit.
- [ ] **Receipt 5 (`quality-contract.json`):** `utils/quality-dashboard.sh --check docs/quality-contract.json` validates all fidelity and latency envelopes.
- [ ] SHA-256 digests and run identifiers are archived in the release record.

---

## 6. Human Pre-Release Certification Record Template

When certifying a release candidate, complete and archive the following record:

```markdown
### Release / Audit Certification Record

- **Date (UTC):** YYYY-MM-DD HH:MM:SS
- **Operator Name:** Fábio Henrique de Lima Silva
- **Git Commit:** <commit-sha> (clean working tree required)
- **Rustc Version:** rustc X.Y.Z (Edition 2024)
- **CPU Model:** <lscpu summary>
- **CPU Governor:** performance
- **Baseline Producer Commit:** <commit-sha of approved performance baseline>

#### The Five Mandatory Receipts (same clean commit):
- [ ] **Receipt 1 — lints:** `target/logs/lints.log` (exit 0, zero warnings). Digest: <sha256>
- [ ] **Receipt 2 — quick-strict:** `target/logs/quick-receipt.txt` (`FIDELITY: OK`, `OVERALL: PASSED`, zero gaps). Digest: <sha256>
- [ ] **Receipt 3 — long-strict:** `target/logs/long-audit-receipt.jsonl` validated via `nam_long_receipt validate --strict` (`OVERALL: PASSED`, `gaps: []`). Digest: <sha256>
- [ ] **Receipt 4 — perf-check:** `target/logs/regression_phase_receipt.jsonl` (exit 0 against pre-approved baseline). Digest: <sha256>
- [ ] **Receipt 5 — quality-contract:** `docs/quality-contract.json` verified via `utils/quality-dashboard.sh --check` (exit 0, all fidelity envelopes passed). Digest: <sha256>

#### Final Release Verdict:
[ APPROVED FOR RELEASE / REJECTED ]
```
