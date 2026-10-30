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
| **Static Quality Gate** | 🔍 **Static Analysis & Lints:** Formatting, four-configuration check and Clippy, documentation, SPDX        | ~45s                         | Routine developer loop, pre-commit & Receipt #1    |
| **Tier 1 (Manual)**     | ⚡ **Smoke Test:** Sanity checks on model loading, dispatch, and basic inference         | ~2 min                       | After modifications to core DSP or loader modules  |
| **Tier 2 (Manual)**     | 🎯 **Feature Verification:** Block-invariance, reset idempotency, DSP stage transitions  | ~10–15 min                   | Milestone completion or major feature integration  |
| **Tier 3 (Manual)**     | 🛡️ **Robustness & Stress:** Continuous soak, SPSC burst contention, rate modulation      | ~20–30 min                   | Pre-release audits or major engine refactorings    |
| **Agile Quick Runner**  | 🚀 **Agile First-Line QA:** Structural debug tests, float/C++ parity, parser fuzzing     | ~16–87s (16–22s warm; 64–87s cold; budget ≤ 5 min) | Pre-commit check or local iterative validation     |
| **Long Audit Runner**   | 🔬 **Exhaustive Pre-Release Audit:** Soak, full parity matrix, heap audit, RT deadline   | ~10–13 min (625–749s measured) | Nightly builds and pre-release human certification |

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

   The test harness probes the governor of the effective benchmark core (resolved via `nam_perf_gate pick-core`: explicit `NAM_BENCH_CORE`, online isolated core from `/sys/devices/system/cpu/isolated`; when neither resolves, `pick-core` returns `GAP` and no core is pinned).

2. **System Load & Thermal Stability:**
   Close resource-intensive background processes, IDE indexers, browsers, and background compilers to prevent scheduling interference.

3. **CPU Affinity (`NAM_BENCH_CORE`):**

   - For timing-sensitive phases, [`tests-long.sh`](../utils/tests-long.sh) and `nam_perf_gate run` pin execution to the core resolved by the `NAM_BENCH_CORE` precedence (see `benchmarks.md` §4).

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
> Automated execution of `utils/tests-long.sh`, `nam_perf_gate run --bootstrap-baseline`, and `utils/quality-dashboard.sh --save` is restricted to human operators. These scripts must run on calibrated, isolated machines.

### 4.1 Agile Quick Suite (`utils/tests-quick.sh`)

The quick suite is the primary automated gate for local development (~16–87s measured: 16–22s warm incremental, 64–87s cold; budget ≤ 5 min).

```bash
# Standard local verification (skips missing optional vendor fixtures gracefully)
./utils/tests-quick.sh

# Capacidades e fixtures obrigatórias ausentes ficam em target/logs/skips.jsonl
# como GAP skip-capability:<n>; skips opcionais (models_nondist*/community_model_absent*)
# são tipados como opcionais e não reprovam --strict. O modo estrito é do long:
# ./utils/tests-long.sh --strict
```

- **Freshness (`nam_freshness artifacts-hard`):** Verifies golden-artifact freshness and integrity; exit code 3 becomes a `generator-drift` gap, exit code 0 passes, and other nonzero exits fail the run.
- **Tests (`cargo test --locked --features testing,pinned --no-fail-fast`, Debug):** Runs every non-ignored test target, including structural/logic/parser/FSM checks, non-ignored numerical and parity checks, and the `libm_export_guard` symbol gate. Ignored tests are not invoked.

**Expected Outcome:**

- Local runs: `RESULT: PASS` or `RESULT: PASS_WITH_GAPS` with exit status 0 (a `skip-capability:<n>` gap is derived only from blocking records of `target/logs/skips.jsonl`; optional non-distributable model skips are recorded with `"optional":true` and do not create a blocking gap); a failed phase yields `RESULT: FAIL` and exit status 1. Freshness, test/compilation, zero-test and unexpected-error paths all terminate through `finish`, producing exactly one `RESULT`, the phase `FAIL` with its log path, and the current `target/logs/tests-quick-status.tsv` carrier (a stale carrier is removed at start, and a carrier write failure fails the run).
- The quick suite emits no receipt. Release certification requires `utils/tests-long.sh --strict` with zero gaps and exit status 0 (the quick suite has no strict flag).

---

### 4.2 Long Audit Suite (`utils/tests-long.sh`)

The long audit suite provides exhaustive pre-release validation (~10–13 minutes, 625–749 s measured).

```bash
# Standard nightly audit (tolerates gaps for missing optional fixtures; exit status 0)
./utils/tests-long.sh

# Strict pre-release mode (FAIL-CLOSED: converts any gap or skipped test into exit status 1)
./utils/tests-long.sh --strict
```

- **Preflight Gates:** `preflight-render` (build fingerprint by oracle commit/compiler, wrapper `render_multichannel` cache key, binary smoke, receipt provenance), `preflight-catalog`, `preflight-package`, `preflight-freshness`, and diagnostic `preflight-simd-probe`.

- **8-Phase Battery:** (1) `catch-all` (`phase1`), (2) `soak` (`phase2`), (3) `isa-codegen` (`phase3`), (4) `heap-audit` (`phase4`), (5) `rt-deadline` (`phase5`), (6) `rt-jitter` (`phase6`), (7) `loom` (`phase7`), (8) `perf` (`phase8`). Each phase writes `target/logs/long-<phase>.log` with shell xtrace routed to `target/logs/long-xtrace.log` by `xtrace_side_channel` (`utils/_lib.sh`), so the phase log contains command output only.

- **Numeric telemetry:** `target/logs/fidelity-metrics.jsonl` carries ESR/SNR/MR-STFT/MSE rows from the fidelity helpers and `heap_alloc` totals (`scenario`, `allocations`) from `phase4`; it is truncated before `phase1` and never gates a phase.

- **Quality verdict:** `utils/quality-dashboard.sh --check docs/quality-contract.json --fidelity-only` checks fidelity without the performance baseline. The full `--check` remains receipt 5 for release certification.

- **Typed discipline verdicts:** each discipline that uses `_lib.sh` writes `target/logs/<script>-status.tsv` (discipline, verdict, comma-separated gaps, comma-separated fails).

- **Receipt Validation:**

  ```bash
  cargo run --locked -p nam-qa --bin nam_long_receipt -- validate --strict --out target/logs/long-audit-receipt.jsonl
  ```

---

### 4.3 Disk Logs & Diagnostic Artifacts Inventory

All test runners persist execution logs under `target/logs/` (and `target/debug-logs/` for debug harnesses):

| File Path                                        | Generating Component / Phase           | Diagnostic Contents                                                                                                                          |
|:------------------------------------------------ |:-------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------- |
| **`target/logs/quick-freshness.log`**            | `tests-quick.sh` (freshness)           | `nam_freshness artifacts-hard` output and generator-drift diagnostics.                                                                       |
| **`target/logs/quick-tests.log`**                | `tests-quick.sh` (tests)               | Full non-ignored `cargo test --features testing,pinned` output (Debug profile).                                                                     |
| **`target/logs/skips.jsonl`**                    | `tests-quick.sh` / test harness        | Typed `report_skip` records (missing C++ oracle, distributable fixtures, non-distributable/community models); blocking records aggregate into `skip-capability:<n>` gaps, optional ones are flagged `optional:true`. |
| **`target/logs/<script>-status.tsv`**            | `finish` em `utils/_lib.sh`            | Typed discipline verdict, gaps and failures, separated by commas.                                                                                                 |
| **`target/logs/long-audit-receipt.jsonl`**       | `tests-long.sh` (Final)                | Machine-readable structured audit receipt (per-phase JSON entries).                                                                          |
| **`target/logs/long-preflight-render.log`**      | `tests-long.sh` (`preflight-render`)   | C++ reference render build/validation diagnostics.                                                                                           |
| **`target/logs/long-preflight-catalog.log`**     | `tests-long.sh` (`preflight-catalog`)  | Fixture catalog discovery, SHA-256 manifest checks, and missing fixture diagnostics.                                                         |
| **`target/logs/long-preflight-package.log`**     | `tests-long.sh` (`preflight-package`)  | Output of `cargo package --locked` (packaging and verify build).                                                                         |
| **`target/logs/long-preflight-freshness.log`**   | `tests-long.sh` (`preflight-freshness`)| `nam_freshness` hard-fail freshness verification output.                                                                                     |
| **`target/logs/long-preflight-simd-probe.log`**  | `tests-long.sh` (`preflight-simd-probe`)| CPU feature bits, OS AVX-512 state, Cargo feature status, and dispatched backend.                                                           |
| **`target/logs/cmake-configure.log`**            | `utils/ensure_namcore_render.sh`       | CMake configuration output when compiling C++ `render` CLI.                                                                                  |
| **`target/logs/cmake-build.log`**                | `utils/ensure_namcore_render.sh`       | Compilation logs for the C++ reference render binary.                                                                                        |
| **`target/logs/long-catch-all.log`**             | `tests-long.sh` (`catch-all`)          | Full release suite including ignored tests: parity, multi-SR goldens, proptests, defense/invariants, and the libm export guard.              |
| **`target/logs/long-<phase>-failures.txt`**      | `tests-long.sh` (failed phase)         | Diagnostic extract with libtest failure headers, `panicked at` and its message, `failures:`, and `test result: FAILED`; the full log is preserved. |
| **`target/logs/long-soak.log`**                  | `tests-long.sh` (`soak`)               | Numerical soak logs, SPSC queue sweeps, and endurance metrics.                                                                               |
| **`target/logs/long-isa-codegen.log`**           | `tests-long.sh` (`isa-codegen`)        | `simd_probe` output and `isa_contract` checks.                                                                                               |
| **`target/logs/long-heap-audit.log`**            | `tests-long.sh` (`heap-audit`)         | `CountingAllocator` memory interceptor reports verifying zero allocations on RT thread.                                                      |
| **`target/logs/fidelity-metrics.jsonl`**         | `tests-long.sh` (`catch-all`/`heap-audit`)| Numeric telemetry: fidelity rows (ESR/SNR/MR-STFT/MSE) and `heap_alloc` totals per scenario; truncated before `phase1`, never a gate.        |
| **`target/logs/quality_report.jsonl`**           | `nam_quality dashboard` (quality phase)| Typed per-phase quality report (`freshness`, `third_party`, fidelity phases, performance). `observed_records` counts executed test cases for test-backed phases and measured metrics for measurement phases. |
| **`target/logs/fidelity_metrics.jsonl`**         | `nam_quality dashboard` (quality phase)| Quality fidelity metrics (ESR/SNR/MR-STFT/MSE) consumed by `nam_quality ingest`; the underscore distinguishes it from the long-suite `fidelity-metrics.jsonl` (hyphen). |
| **`target/logs/long-xtrace.log`**                | `tests-long.sh` (manual runs)          | Shell xtrace side-channel for long phases.                  |
| **`target/logs/long-rt-deadline.log`**           | `tests-long.sh` (`rt-deadline`)        | Latency histograms and deadline statistics ($p99 < 1.33\text{ ms}$). High-resolution linear bins (0.5 µs resolution via `LatencyHistogram`). |
| **`target/logs/long-rt-jitter.log`**             | `tests-long.sh` (`rt-jitter`)          | Real-time jitter telemetry and thread contention profiles.                                                                                   |
| **`target/logs/long-loom.log`**                  | `tests-long.sh` (`loom`)               | Loom model-checking state-space exploration logs.                                                                                            |
| **`target/logs/long-perf.log`**                  | `tests-long.sh` (`perf`)               | `nam_perf_gate run --check docs/quality-contract.json` output.                                                                               |
| **`target/logs/regression_phase_receipt.jsonl`** | `nam_quality` / S2 | Machine-readable performance regression phase receipts integrated fail-closed into audit summary (enforces `hardware_class` comparability).  |
| **`target/logs/regression-check.log`**           | `nam_perf_gate run --check`            | Detailed Criterion output and statistical confidence interval results during regression checks (truncated on invocation start).              |
| **`target/logs/remote-simd-receipt.jsonl`**      | `utils/remote-simd-gate.sh`            | Structured multi-ISA verification receipt emitted from remote or emulated AVX-512 runs.                                                      |
| **`target/logs/remote-simd-parity.log`**         | `utils/remote-simd-gate.sh`            | Detailed parity output comparing AVX2 and AVX-512 execution streams.                                                                         |
| **`target/logs/`**                              | `utils/quality-dashboard.sh`           | Structured JSON outputs (receipt, metrics, report) for individual model fidelity and latency benchmarks.                                      |
| **`target/parity-vectors/<target>/*.bin`**       | `nam_parity_vector generate` / CI      | `NAMPARV1` binary parity vectors containing exact IEEE-754 output samples and provenance headers.                                              |
| **`target/logs/parity/report-*.json`**           | `nam_parity_vector compare` / CI       | Detailed forensic JSON reports from pairwise cross-platform bitwise parity comparisons.                                                       |
| **`target/debug-logs/<label>.log`**              | `utils/debug/repro_oversample_hang.sh` | Timestamped execution log and command output under systemd cgroup v2 resource limits.                                                        |
| **`target/debug-logs/<label>.exit`**             | `utils/debug/repro_oversample_hang.sh` | Raw exit status code emitted by the wrapped command or systemd scope timeout (`143` on SIGTERM, `137` on SIGKILL).                           |
| **`~/.cache/neural-amp-modeler-rs/crash-*.txt`** | Runtime Panic Hook                     | Diagnostic crash reports rendered without heap allocations.                                                                                  |

---

### 4.4 Developer Diagnostic Harness & Hang Reproduction (`utils/debug/repro_oversample_hang.sh`)

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

### 4.5 Known Benign Signals (False-Alarm Catalog)

> [!NOTE]
> **Compilation Profile Segregation:**
> To eliminate build cache thrashing across phases with incompatible Cargo feature sets, `utils/tests-long.sh` automatically segregates `CARGO_TARGET_DIR` per profile (`target/long/release`, `target/long/heap-audit`, and `target/long/loom`). Independent logs in `target/logs/` are preserved across runs.

Signals that routinely appear in runner logs but are **expected**; they must not be escalated as defects or regressions. Each row cites the artifact that emits it (paths under `target/logs/` are runtime artifacts, not versioned files).

| Signal (log excerpt)                                                                         | Emitting artifact                                                                                                      | Why it is benign                                                                                                                                                                                                  | Correct action                                                                                                   |
|:-------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------- |
| `isa_parity_cross_isa: SKIP_CAPABILITY`; `remote-simd-receipt.jsonl` skips                   | `target/logs/long-catch-all.log`, `target/logs/remote-simd-receipt.jsonl`, `target/logs/long-audit-receipt.jsonl` | The local CPU lacks AVX-512; cross-ISA parity is a **separate** remote/emulated gate, not a local-coverage gap.                                                                                                   | Confirm `utils/remote-simd-gate.sh` ran separately; do not read the local skip as a failure.                     |
| `avx512_opt_in_not_run` (receipt gap)                                                        | `target/logs/long-preflight-simd-probe.log`, `target/logs/long-audit-receipt.jsonl`                                    | AVX-512 is an opt-in Cargo feature (`--features avx512`); the default runner never compiles it.                                                                                                                   | Expected on a default run; only a scheduled `--features avx512` run covers it.                                   |
| `[RECEIPT] status=CHARACTERIZED label=…-saturate-N … violations=… cs_invol=…`                   | `target/logs/long-rt-jitter.log`                                                                                       | Saturation stress is a **CHARACTERIZATION** ("environmental characterization, NOT a deadline gate"); `status=CHARACTERIZED` means the harness completed, not that jitter passed. Under unpinned saturation (`pinned=false`), thread preemption by the Linux scheduler causes high `cs_invol` and tail latency spikes (~2.8–3.2 ms); under pinned saturation (`pinned=true`, `NAM_BENCH_CORE=N`), `exact_max` stays ~195 µs with 0 violations. | Read `violations=` and `cs_invol`; never treat the receipt token as a deadline assertion under unpinned execution. |
| `… has been running for over 60 seconds`                                                     | `target/logs/quick-tests.log`, `target/logs/long-catch-all.log`, `target/logs/long-soak.log`                           | Long-running tests (soak, block-invariance, deterministic-energy); the default test runner prints this after 60 s.                                                                                                | Informational only; not a hang — wait for completion.                                                            |
| `Found N outliers among 100 measurements (x%)`                                               | `target/logs/regression-check.log`                                                                                     | Criterion reports high-severity sampling outliers; the regression verdict uses the median delta and p-value, not the outlier banner.                                                                              | Read Δ median and p; ignore the outlier count.                                                                   |
| `[BOOTSTRAP] ⚠ This operation must be performed by a human operator.`                        | `nam_perf_gate run --bootstrap-baseline` output (`target/logs/regression-check.log`)               | Baseline bootstrap is operator-only by governance (§5.2); AI runs are forbidden from it.                                                                                                                          | Do not run `--bootstrap-baseline`; report and wait for the operator.                                             |
| `RT_WaveNet_Lite_CH12` `+8.56%` (`p=0.04`) — `Change within noise threshold.`                | `target/logs/regression-check.log`                                                                                     | Sub-5 % variance band under ambient/thermal noise (F-10/F-12); the effective `noise_threshold` classifies it as non-significant.                                                                                  | Re-measure on an isolated pinned core before treating as a regression; never bootstrap a baseline to silence it. |
| `[JIT_PINS] pinned=false` / `pinned=false` in rt-jitter receipts; P50 ≈ 98 µs vs rt-deadline ≈ 45 µs | `target/logs/long-rt-jitter.log`                                                                                 | Jitter runs in the generic multi-core environment unless `NAM_BENCH_CORE` is exported to an isolated CPU; under unpinned saturation, Linux CFS preemption generates high involuntary context switches (`cs_invol > 0`), which is purely environmental and never an engine regression. | Ensure `NAM_BENCH_CORE` is set or run via `tests-long.sh` for pinned isolation where the DSP thread runs on the bench core and workers on the remaining cores.   |
| `[AVISO] --check pós---bootstrap valida só repetibilidade`                                   | `nam_perf_gate run --check` output                                                                                             | `--check` immediately after `--bootstrap-baseline` in the same session proves self-consistency only (F-12); the REGRESSION GATE verdict needs a baseline from another session/commit (`noise_threshold=0.05`).    | Do not read the paired green as absence of regression.                                                           |
| `known_gap:condition_lstm_rejected_by_both`                                                  | `target/logs/long-catch-all.log`                                                                                       | The WaveNet model whose `condition_dsp` is an LSTM is refused by **both** oracles by design (Rust fail-closed policy; C++ `condition_size` mismatch) — see [`cpp_parity_map.md`](cpp_parity_map.md) §7.5.         | A declared gap, not a pass and not a defect; keep it declared in receipts.                                       |

---

### 4.6 Remote SIMD Gate & ISA Parity Protocol (`utils/remote-simd-gate.sh`)

The AVX-512 backend (`src/dsp/simd/avx512/`) provides opt-in mathematical acceleration (`--features avx512`). Because production execution defaults to single-precision `f32` on the AVX2 (`x86-64-v3`) architecture, AVX-512 is maintained as an opt-in research path. Cross-ISA parity ensures mathematical equivalence between AVX-512 kernels and the AVX2 golden reference.

#### 1. Execution Modes and Topology Invariants

- **Native AVX-512 Hardware:**
  On machines equipped with native AVX-512 hardware (AMD Zen 4/5, Intel Xeon / Sapphire Rapids), invoking `utils/remote-simd-gate.sh` executes the full suite:
  ```bash
  utils/remote-simd-gate.sh [--out target/logs/remote-simd-receipt.jsonl]
  ```
  1. `preflight`: Verifies CPU capability via `simd_probe` (`avx512f` and `avx512vl`).
  2. `parity`: Compiles with `--features avx512` and executes `isa_parity` across all neural topologies with forced ISA overrides, asserting mathematical output parity against AVX2.
  3. `bench`: Executes Criterion comparison benchmarks (`isa_compare`) and evaluates whether SKUs achieve the contractual $\ge 12\%$ ROI threshold ($p < 0.05$) via `nam_remote_simd_receipt`.

- **Intel SDE Emulation Mode:**
  On development machines lacking AVX-512 hardware, the full cross-ISA mathematical parity matrix can be validated under the Intel Software Development Emulator:
  ```bash
  utils/remote-simd-gate.sh --sde
  ```
  This automatically configures the runner (`sde64 -spr --`), executes the complete mathematical parity verification, and skips Criterion benchmarks to eliminate emulation timing artifacts.

- **Probe & Telemetry Mode (`--check-only`):**
  Used during automated preflight or developer sanity checks:
  ```bash
  utils/remote-simd-gate.sh --check-only
  ```
  If AVX-512 is absent and SDE is not configured, the script deterministically emits `PHASE: preflight GAP avx512-not-detected` and exits with code 0 (`RESULT: PASS_WITH_GAPS`), ensuring zero false negatives on standard `x86-64-v3` workstations.

#### 2. Fixture Catalog Classification & Hygiene

To maintain strict hygiene across public checkouts, the fixture catalog (`src/testing/catalog.rs`) partitions models and goldens:
- **`RequiredLocal`:** Standard models committed to repository git history (`tests/fixtures/models/*.nam`). Their absence hard-fails the preflight gate.
- **`OptionalExternal`:** Proprietary community models (`EVH-5150-Lite.nam`, `APP-EVH-Stealth100-Dialled-xSTD.nam`, `Boss BD-2 H2O Mod T-12_00 G-12_00.nam`, `SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam`). When absent in clean developer clones, `catalog_preflight` gracefully classifies them as `MISSING-OPTIONAL` / `SKIP-CAPABILITY`, allowing `catalog_preflight` to pass cleanly with exit code 0 (`PASS`).
- **Typed skip evidence:** tests that need an absent optional model (`nondist_validation`, `nam_infer_test::test_community_models_inference`) call `neural_amp_modeler_rs::testing::report_skip` with reasons `models_nondist_absent`, `models_nondist_empty`, or `community_model_absent:<file>`. The record lands in `target/logs/skips.jsonl`; blocking records become a `skip-capability:<n>` gap, while the three documented optional reasons are non-blocking under the long `--strict` and remain counted in the receipt. A missing model is never a silent `PASS`.

---

## 5. Release Certification Protocol (5 Mandatory Receipts)

A build is certified for release only when **all five mandatory receipts** are produced and verified against the **same clean git commit hash**.

### 5.1 The Five Mandatory Receipts

| #   | Receipt Artifact        | Generating Command                                              | Execution Flags / Env                    | Blocking Criteria                                                                                                                                 |
|:---:|:----------------------- |:--------------------------------------------------------------- |:---------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `lints.log`             | `utils/lints.sh`                                                | `--all-targets --all-features` (script)  | Exit 0; zero compiler/clippy/rustdoc warnings; SPDX and anti-pattern checks green; AVX-512 compile-time segregation confirmed.                    |
| 2   | `quick-strict.log`      | `utils/tests-quick.sh`                                          | Canonical execution                      | Exit 0; `RESULT: PASS`; zero gaps.                                                                                                                |
| 3   | `long-strict.log`       | `utils/tests-long.sh --strict`                                  | `--strict`                               | Exit 0; `RESULT: PASS`; `gaps: []`; receipt validated via `nam_long_receipt validate --strict`.                                                |
| 4   | `perf-check.log`        | `nam_perf_gate run --check`                 | Pre-approved baseline (see §5.2)         | Exit 0; machine regression verdict green (`nam_perf_gate verdict` over Criterion `change/estimates.json`) against a pre-approved baseline commit. |
| 5   | `quality-contract.json` | `utils/quality-dashboard.sh --check docs/quality-contract.json` | `docs/quality-contract.json` (committed) | Exit 0; all audio fidelity envelopes (ESR/SNR/MR-STFT) and f64-oracle checks satisfied; `regression_gate` is not `NOT_VERIFIED`.                  |

> [!IMPORTANT]
> **Same-Commit Binding:** All five receipts must be produced on the **exact same clean git commit**. The release record (§6) binds them together with their SHA-256 digests. Receipts produced on dirty working trees or across different revisions are invalid.

### 5.2 Baseline Bootstrap Governance

`nam_perf_gate run --bootstrap-baseline` and `utils/quality-dashboard.sh --save` are **independent ceremonies** executed exclusively by a human operator:

- **Distinct Producer Commit:** The performance baseline must be generated and approved on an earlier reference commit. Bootstrapping a baseline on the same commit being certified only demonstrates self-consistency, not the absence of regressions.
- **Controlled Hardware:** Baselines must be generated on isolated machines with the CPU governor set to `performance` and background contention eliminated.
- **Intentional Updates Only:** Baselines and contracts are updated only when performance shifts or algorithmic changes are intentional, verified, and documented.

### 5.3 Human Certification Checklist

- [ ] Working tree is clean on the target commit (`git status --porcelain` is empty).
- [ ] **Receipt 1 (`lints.log`):** `utils/lints.sh` passes with zero warnings.
- [ ] **Receipt 2 (`quick-strict.log`):** `utils/tests-quick.sh` passes with `FIDELITY: OK`, `OVERALL: PASSED`, and zero gaps.
- [ ] **Receipt 3 (`long-strict.log`):** `utils/tests-long.sh --strict` passes with `OVERALL: PASSED` and `gaps: []`.
- [ ] **Receipt 4 (`perf-check.log`):** `nam_perf_gate run --check` passes against an approved baseline from a distinct reference commit.
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
