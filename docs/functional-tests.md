<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Functional Testing & Human Certification Guide

**Audience:** Developers, QA Engineers, and Human Release Operators working with the `NeuralAmpModeler-rs` DSP engine.

---

## 1. Executive Summary & Scope

This guide establishes the comprehensive manual functional verification procedures and the official human certification protocol for `NeuralAmpModeler-rs`. It bridges targeted developer debugging workflows, multi-tiered manual functional tests, and the formal execution/receipt-audit protocols for the automated test runners: **Agile Quick Suite** ([`utils/tests-quick.sh`](../utils/tests-quick.sh)) and **Nightly / Pre-Release Long Suite** ([`utils/tests-long.sh`](../utils/tests-long.sh)).

Automated test architecture, gate taxonomy, and mathematical oracle hierarchies are documented in [testing.md](testing.md), while performance regression gates are detailed in [benchmarks.md](benchmarks.md).

### Verification Hierarchy & Cadence

| Tier / Runner          | Scope & Purpose                                                                                            | Target Duration                           | When to Run                                        |
|:---------------------- |:---------------------------------------------------------------------------------------------------------- |:----------------------------------------- |:-------------------------------------------------- |
| **Tier 1 (Manual)**    | ⚡ **Smoke Test:** High-yield sanity checks on loading, fallback, and basic inference                      | ~2 min                                    | After code changes to core DSP/loader modules      |
| **Tier 2 (Manual)**    | 🎯 **Feature Verification:** Determinism, block-invariance, stage transitions, RT-safety                   | ~10–15 min                                | Milestone completion or major feature integration  |
| **Tier 3 (Manual)**    | 🛡️ **Robustness & Stress:** Extended endurance, rapid SPSC storms, rate modulation                         | ~20–30 min                                | Pre-release audits or major refactorings           |
| **Agile Quick Runner** | 🚀 **Agile 1st Line QA:** Structural debug tests, float/C++ parity oracles, parser fuzzing                 | Approximately 2 min (hardware-dependent)  | Pre-commit check or local iterative validation     |
| **Long Audit Runner**  | 🔬 **Exhaustive Pre-Release Audit:** Soak, QA defenses, full matrix, heap audit, RT deadline, jitter, Loom | Approximately 10 min (hardware-dependent) | Nightly builds and pre-release human certification |

---

## 2. Host & Environmental Prerequisites

Before executing manual stress scenarios, micro-benchmarking, or pre-release runner certifications, verify that the host machine satisfies the following baseline requirements:

1. **CPU Frequency Scaling Governor:**
   Must be configured to `performance` across all physical CPU cores to eliminate frequency throttling, core migration jitter, and timer noise during real-time deadline tests.

   ```bash
   # Check active governor status across all cores
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort -u
   # Expected output: performance
   ```

2. **Low System Load & Thermal Stability:**
   Close all resource-intensive background applications, IDE indexers, browsers, and background compilers to prevent scheduling contention.

3. **Process Priority & CPU Affinity (`taskset`, `nice`/`ionice`):**

   - The test runners ([`tests-quick.sh`](../utils/tests-quick.sh) and [`lints.sh`](../utils/lints.sh)) automatically lower priority via `nice -n 19 ionice -c 3` unless `NAM_NO_LOW_PRIORITY=1` is set.
   - For performance and jitter-sensitive phases, [`tests-long.sh`](../utils/tests-long.sh) and [`tests-performance-regression.sh`](../utils/tests-performance-regression.sh) pin execution to a dedicated core (default: `nproc / 2`, configurable via `NAM_BENCH_CORE`).

4. **Upstream Vendor Mirrors & Fixtures:**
   Ensure `third-party/NeuralAmpModelerCore/` is properly populated and matches the pinned commit specified in [`variables.env`](../variables.env).

   ```bash
   ./utils/setup-third-party.sh
   ```

---

## 3. Manual Functional Verification Matrix (Tiers 1–3)

### 3.1 Tier 1: ⚡ Smoke Test (High-Yield Verification)

- [ ] **T1.1 Model Loading & Basic Inference:** Load a `.nam` (JSON) and a `.namb` (binary container) model via `load_and_build_model()`. Call `process()` with a 64-sample block of silence. *Expected:* Function returns without panic; output buffer contains finite, non-NaN values.
- [ ] **T1.2 Static Dispatch Coverage:** Load one model from each supported architecture family (WaveNet A1, WaveNet A2, LSTM, ConvNet, Linear). Run a single block of inference per model. *Expected:* All five architectures instantiate and process successfully.
- [ ] **T1.3 Dynamic Model Fallback:** Load a model whose geometry does not match compile-time static profiles (e.g., custom LSTM 3×5 or WaveNet CH=7). *Expected:* Dispatches to the appropriate `Dyn` variant and produces valid output.
- [ ] **T1.4 Corrupted Model Rejection:** Attempt to load a truncated, malformed, or 0-byte `.nam`/`.namb` file. *Expected:* Returns a strongly typed `Err(LoadError)` (see `src/loader/error.rs`) and emits structured diagnostics with a descriptive `NamErrorCode` without panicking.
- [ ] **T1.5 Sample Rate Adaptation:** Load a 48 kHz model. Execute `model.reset(44100, 64)`, process a block, execute `model.reset(96000, 64)`, and process again. *Expected:* Clean state transition without allocation leaks or audio discontinuities.

---

### 3.2 Tier 2: 🎯 Feature & Subsystem Verification

#### Domain 2A: Model & Pipeline Architecture

- [ ] **2A.1 Block-Size Invariance:** Processing the same audio input split into `[32 + 32]` samples vs. a single `[64]` sample block must yield bit-identical (or $10^{-7}$ float-equivalent) output. *Verifies:* Receptive-field buffer tracking and zero state leakage across block boundaries.
- [ ] **2A.2 Prewarm Correctness:** After `prewarm(n)`, the first output block must be deterministic and free of initialization transients exceeding baseline noise gate thresholds.
- [ ] **2A.3 Reset Idempotency:** Invoking `reset()` → `process()` → Output A, followed by `reset()` → `process()` → Output B with identical input must yield $A = B$.
- [ ] **2A.4 Container Submodel Crossfade:** Load a `SlimmableContainer` and trigger a `Full → Lite` profile swap during active audio processing. *Expected:* 32 ms equal-power crossfade executes seamlessly without audible clicks; output stays within ESR tolerance.
- [ ] **2A.5 Lock-Free SPSC Model Hot-Swap:** Push a `LoadModel` command via the lock-free SPSC channel while the audio thread runs continuously. *Expected:* Model swap executes without priority inversion or audio dropouts; the retired model is reclaimed safely by the background GC cascade.

#### Domain 2B: DSP Pipeline Stages (Gate, Resampler, CabSim)

- [ ] **2B.1 Noise Gate FSM:** Feed digital silence → gate triggers fade-out and clamps output to zero. Feed signal above threshold → gate smoothly ramps open without overshoot. *Verifies:* Hysteresis envelope and smooth gain transitions.
- [ ] **2B.2 Oversampling Anti-Aliasing:** Run a full-scale high-frequency sine sweep through a non-linear WaveNet model at $2\times$ and $4\times$ oversampling. *Expected:* Aliasing suppression conforms to thresholds in [audio_fidelity_map.md](audio_fidelity_map.md).
- [ ] **2B.3 Native Rate Resampler Bypass:** When host sample rate matches model native rate (e.g., 48 kHz $\to$ 48 kHz), `NamResampler` engages zero-copy passthrough with zero kernel computational overhead.
- [ ] **2B.4 Multi-Rate Resampling:** Verify processing across 44.1, 48, 88.2, 96, and 192 kHz. *Expected:* High SNR preservation, no ring-buffer overflow, and clean phase response.
- [ ] **2B.5 CabSim Impulse Response Engine:** Load a standard `.wav` IR into `ConvEngine`. Process audio and verify convolution. Clear the IR and verify instantaneous fallback to clean bypass.

#### Domain 2C: RT-Safety & Allocation Watchdog

- [ ] **2C.1 Zero-Allocation Audio Hot-Path:** Build with `--features heap-audit`. Process 10,000 continuous audio blocks on the audio thread. *Expected:* `CountingAllocator` reports exactly zero heap allocations in `process()`.
- [ ] **2C.2 Denormal Protection (FTZ/DAZ):** Process low-level decaying signals (below $-120\text{ dBFS}$). *Expected:* FTZ/DAZ hardware flags prevent denormal performance degradation.
- [ ] **2C.3 Robustness Against Extreme Float Inputs:** Feed out-of-range ($\pm 100.0$), NaN, or $\pm\infty$ float buffers to `process()`. *Expected:* Soft-clipping/sanitization occurs; zero hot-path panics or undefined behavior.
- [ ] **2C.4 Memory Footprint & Resource Leak Audit:** Repeatedly load and unload 100 models in a loop. *Expected:* Process RSS memory stabilizes; zero leaked file descriptors or memory handles.

---

### 3.3 Tier 3: 🛡️ Robustness & Stress Scenarios

- [ ] **3.1 Soak Test (10M+ Frames):** Execute continuous inference over $>10$ million frames with randomized buffer block sizes ($16$ to $256$ samples). *Expected:* Zero panics, zero memory drift, strictly finite float output.
- [ ] **3.2 SPSC Command Burst Contention:** Submit 1,000 rapid model-load commands via SPSC while the RT thread operates at minimal block size. *Expected:* Zero dropped commands, GC drains smoothly, audio latency does not exceed deadline budget.
- [ ] **3.3 Adaptive Compute FSM Stress:** Artificially saturate CPU to trigger adaptive downgrade (`Full` $\to$ `Reduced` $\to$ `Minimal`). When contention clears, verify hysteresis recovery back to `Full`.
- [ ] **3.4 Dynamic Sample Rate Modulation:** Dynamically change sample rate every 100 blocks (44.1 $\leftrightarrow$ 48 $\leftrightarrow$ 96 kHz). *Expected:* Resampler ring buffers reinitialize cleanly without memory corruption or output artifacts.
- [ ] **3.5 Extended CabSim Partitioning:** Load extreme-length impulse responses ($2^{20}$ samples). *Expected:* Frequency-domain delay-line (FDL) partition scaling allocates only during off-RT setup, retaining zero-allocation processing.

---

### 3.4 Interactive Test Harness Reference

The following Rust pattern illustrates standard manual functional verification:

```rust
use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::models::NamModel;
use std::path::Path;

// 1. Sanity Smoke Test: Load and process single block
let sys = SystemSnapshot::capture();
let mut mp = load_and_build_model(Path::new("model.nam"), &sys, false, LoadOptions::default())?;
let model = mp.model_l.as_mut().expect("Model instance expected");

let input = vec![0.0f32; 128];
let mut output = vec![0.0f32; 128];
model.process(&input[..64], &mut output[..64]);

// 2. Block-Size Invariance Verification ([32+32] vs [64])
let sr = 48000;
let mut out_full = vec![0.0f32; 128];
let mut out_split = vec![0.0f32; 128];

model.reset(sr, 128)?;
model.process(&input[..64], &mut out_split[..64]);
model.process(&input[64..], &mut out_split[64..]);

model.reset(sr, 128)?;
model.process(&input, &mut out_full);

let max_diff = out_full.iter().zip(&out_split)
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
assert!(max_diff < 1e-7, "Block-size invariance violation: max diff = {max_diff}");
```

---

## 4. Automated Runner Protocols & Receipt Auditing

> [!IMPORTANT]
> **AI Agent Execution Restrictions:**
> AI agents are strictly prohibited from executing `utils/tests-long.sh`, `utils/tests-performance-regression.sh --bootstrap-baseline`, or `utils/quality-dashboard.sh --save`. These operations must be executed exclusively by human operators under calibrated, isolated hardware environments.

### 4.1 Agile Quick Suite Protocol (`utils/tests-quick.sh`)

The quick suite serves as the fast, agile first line of defense (approximately 2 minutes, depending on the hardware).

#### Execution Modes

```bash
# Standard interactive / agile verification (skips missing vendor fixtures gracefully)
./utils/tests-quick.sh

# Strict release-gate mode (promotes any missing fixture or toolchain gap to a hard failure)
NAM_QUICK_STRICT=1 ./utils/tests-quick.sh
```

#### Phase Breakdown

- **Phase 1 (Structural & Logic, Debug):** Unit tests, DSP math logic, format parsers, FSM transitions, and lock-free SPSC channels across `models`, `perf_soak`, `parity`, `dsp_core`, `cabsim_stereo`, `target_features_compliance_test`, `libm_export_guard`, and `avx512_guard`.
- **Phase 2 (Measurement Oracles & Parity, Release):** Validates release float codegen across `golden_vectors`, `reference_oracle_f64`, `spectral_fidelity`, `linear_fft_test`, and canonical C++ parity (`quick_parity`).
- **Phase 3 (Parser Fuzzing, Release `--ignored`):** Capped `proptest` sweeps on `.nam` and `.namb` format inputs.

#### Receipt & Log Verification

Every execution produces structured logs and a summary receipt in `target/logs/`:

- Receipt file: `target/logs/quick-receipt.txt`
- Phase logs: `target/logs/quick-phase1.log`, `target/logs/quick-phase2.log`, `target/logs/quick-phase3.log`

**Expected Outcome:**

- **Agile / Local Iteration:** `FIDELITY: OK` or `FIDELITY: INCOMPLETE` (with documented `GAP:` entries if vendor mirrors are absent) and `OVERALL: PASSED` (exit status 0).
- **Release Certification:** Must be executed with `NAM_QUICK_STRICT=1` and yield `FIDELITY: OK` with `OVERALL: PASSED` (zero gaps, exit status 0).

---

### 4.2 Long Audit Suite Protocol (`utils/tests-long.sh`)

The long audit suite provides exhaustive, multi-phase pre-release validation (approximately 10 minutes, depending on the hardware).

#### Preflight Defense Gates

Ahead of any timed test phase, the runner executes blocking preflight validations:

1. `preflight-render`: Builds or validates the C++ `render` binary via `utils/ensure_namcore_render.sh`.
2. `preflight-catalog`: Validates fixture presence against `src/testing/catalog.rs`.
3. `preflight-package`: Validates crate packaging hygiene via `cargo package --list`.
4. `preflight-freshness`: Enforces SHA-256 integrity against `tests/fixtures/.golden_manifest.sha256`.
5. `preflight-meta`: Asserts catalog↔test metadata coherence via `meta_coherence.rs`.

#### Execution: Modes

```bash
# Full nightly / standard audit (tolerates declared gaps for missing optional community models; exit status 0)
./utils/tests-long.sh

# Strict pre-release mode (FAIL-CLOSED: converts any missing optional fixture, skipped test, or gap into exit status 1)
./utils/tests-long.sh --strict-pre-release
```

| Mode                   | Invocation                                   | Behavior on Declared Gaps / Missing Optional Fixtures                                            | Exit Code     | Purpose                                                                                    |
|:---------------------- |:-------------------------------------------- |:------------------------------------------------------------------------------------------------ |:-------------:|:------------------------------------------------------------------------------------------ |
| **Standard / Nightly** | `./utils/tests-long.sh`                      | Emits `OVERALL: COMPLETED_WITH_GAPS` if optional community models or vendor mirrors are omitted. | `0` (Success) | Unattended nightly runs and environments without proprietary test fixtures.                |
| **Strict Pre-Release** | `./utils/tests-long.sh --strict-pre-release` | Emits `OVERALL: FAILED` and aborts if any phase was skipped, inconclusive, or incomplete.        | `1` (Failure) | Official milestone release gates; mandates 100% test completion and full fixture presence. |

#### 7-Phase Audit Breakdown

1. **Phase 1 — Soak & Concurrency:** 10M+ frames continuous endurance, lock-free SPSC contention sweeps.
2. **Phase 2 — Defense Scripts & Invariant Tests:** Rust defense harness (`tests/qa_defense.rs`), ELF symbol export guard (`libm_export_guard`).
3. **Phase 3 — Exhaustive Matrix & Parity:** Full C++ live parity matrix, v2 multi-SR goldens, cross-ISA validation, spectral baselines, and uncapped proptest sweeps.
4. **Phase 4 — Heap Audit:** Strict verification of zero heap allocations on the audio processing hot path (`CountingAllocator`).
5. **Phase 5 — RT Deadline & Constraints:** Validates $p99 < 1.33\text{ ms}$ processing budget per 64-sample block at 48 kHz.
6. **Phase 6 — RT Jitter Telemetry:** Measures processing latency distribution under simulated CPU contention.
7. **Phase 7 — Concurrency Model Checking:** Loom model verification for lock-free queues and atomic bitmasks.

#### Structured Audit Receipt (`long-audit-receipt.jsonl`)

The suite records a structured JSONL receipt (`target/logs/long-audit-receipt.jsonl`), where each phase logs structured metrics:

```json
{"phase_id":"phase1","name":"Soak Tests (Numerical Stability)","status":"PASSED","duration_ms":42000,"tests_executed":26,"gaps":[],"timestamp":"2026-08-14T03:00:00Z"}
```

Final verification must be asserted using the typed receipt validator:

```bash
# Verify receipt integrity and summary verdict
cargo run --locked --features testing --bin nam_long_receipt -- validate --out target/logs/long-audit-receipt.jsonl
# Expected output: VALID: <n> receipt line(s) ... | PASSED: <n> | preflight: <m>
```

---

### 4.3 Disk Logs & Diagnostic Artifacts Inventory

All test runners, compilation helpers, and preflight steps persist detailed execution logs on disk under the target directory (`target/logs/`):

| File Path                                     | Generating Component / Phase       | Contents & Diagnostic Value                                                                        |
|:--------------------------------------------- |:---------------------------------- |:-------------------------------------------------------------------------------------------------- |
| **`target/logs/quick-receipt.txt`**           | `tests-quick.sh` (Final)           | Summary receipt containing `FIDELITY:`, `GAPS:`, and `OVERALL:` status.                            |
| **`target/logs/quick-phase1.log`**            | `tests-quick.sh` (Phase 1)         | Stdout/stderr of structural unit tests, DSP logic, and channel checks (Debug profile).             |
| **`target/logs/quick-phase2.log`**            | `tests-quick.sh` (Phase 2)         | Measurement oracles output, float golden vectors, and `quick_parity` C++ checks (Release profile). |
| **`target/logs/quick-phase3.log`**            | `tests-quick.sh` (Phase 3)         | Fuzzing logs from capped proptest parser sweeps.                                                   |
| **`target/logs/long-audit-receipt.jsonl`**    | `tests-long.sh` (Final)            | Single Source of Truth structured audit record (machine-readable per-phase JSON entries).          |
| **`target/logs/catalog_preflight.log`**       | `tests-long.sh` (Preflight 2)      | Fixture catalog discovery, SHA-256 manifest checks, and missing fixture diagnostics.               |
| **`target/logs/meta_coherence.log`**          | `tests-long.sh` (Preflight 4)      | Cross-validation between catalog definitions and test module registrations.                        |
| **`target/logs/package-list.err`**            | `tests-long.sh` (Preflight 3)      | Diagnostics from `cargo package --list` crate packaging validations.                               |
| **`target/logs/cmake-configure.log`**         | `utils/ensure_namcore_render.sh`   | CMake build configuration output when compiling C++ `tools/render`.                                |
| **`target/logs/cmake-build.log`**             | `utils/ensure_namcore_render.sh`   | CMake compilation logs for the C++ reference render binary.                                        |
| **`target/logs/phase1-soak.log`**             | `tests-long.sh` (Phase 1)          | Continuous numerical soak logs, SPSC buffer sweeps, and endurance metrics.                         |
| **`target/logs/phase-defense-scripts.log`**   | `tests-long.sh` (Phase 2)          | Structural invariant checks, QA defenses (`tests/qa_defense.rs`).                                  |
| **`target/logs/phase-libm-exports.log`**      | `tests-long.sh` (Phase 2)          | Dynamic linker symbol export audits (`libm_export_guard`).                                         |
| **`target/logs/phase2-proptests-parity.log`** | `tests-long.sh` (Phase 3)          | Full live C++ parity comparisons, multi-SR goldens, and 100k-case proptests.                       |
| **`target/logs/subphase-isa-parity.log`**     | `tests-long.sh` (Phase 3 Subphase) | Cross-ISA determinism validation logs (`isa_parity`).                                              |
| **`target/logs/phase3-heap-audit.log`**       | `tests-long.sh` (Phase 4)          | Memory interceptor allocation reports (`CountingAllocator`).                                       |
| **`target/logs/phase4-rt-deadline.log`**      | `tests-long.sh` (Phase 5)          | Latency histograms, deadline overshoot statistics ($p99 < 1.33\text{ ms}$).                        |
| **`target/logs/phase5-rt-jitter.log`**        | `tests-long.sh` (Phase 6)          | Real-time jitter telemetry and thread contention profiles.                                         |
| **`target/logs/phase6-loom.log`**             | `tests-long.sh` (Phase 7)          | Concurrency model checker state-space exploration logs.                                            |
| **`~/.cache/nam-rs/crash-*.txt`**             | Runtime Panic Hook (DSP/Plugin)    | Stack-safe diagnostic crash reports rendered without heap allocations.                             |

---

## 5. Release Certification Protocol (5 Receipts)

A build is certified for release only when **all five mandatory receipts** are
generated, archived, and verified against the **same clean git commit hash**.
A partial runner (e.g. a quick suite executed without strictness, a long suite
without `--strict-pre-release`, or a performance comparison against a freshly
bootstrapped baseline) is *evidence of execution*, never a release certificate.

### 5.1. The Five Mandatory Receipts

| #   | Receipt (record name)   | Command (in `NeuralAmpModeler-rs/`)                             | Environment / Flags                      | Generated Artifact(s)                                                               | Blocking Criteria                                                                                                                                                    |
|:---:|:----------------------- |:--------------------------------------------------------------- |:---------------------------------------- |:----------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `lints.log`             | `utils/lints.sh`                                                | `--all-targets --all-features` (script)  | console output — operator must archive `target/logs/lints.log` (`tee`)              | exit 0; zero compiler/clippy warnings; zero rustdoc warnings; SPDX, anti-pattern and `#[allow]` policies green; binary AVX-512 absence certification green           |
| 2   | `quick-strict.log`      | `utils/tests-quick.sh`                                          | `NAM_QUICK_STRICT=1`                     | `target/logs/quick-receipt.txt` (+ `quick-phase{1,2,3}.log`)                        | `FIDELITY: OK` and `OVERALL: PASSED`; zero `GAP:` entries; exit 0                                                                                                    |
| 3   | `long-strict.log`       | `utils/tests-long.sh --strict-pre-release`                      | `--strict-pre-release`                   | `target/logs/long-audit-receipt.jsonl`                                              | `OVERALL: PASSED` with `gaps: []` (validated via `nam_long_receipt validate --strict`); exit 0                                                                       |
| 4   | `perf-check.log`        | `utils/tests-performance-regression.sh --check`                 | pre-approved baseline (see §5.2)         | `target/logs/regression_phase_receipt.jsonl` (+ `target/logs/regression-check.log`) | statistically valid comparison (two-sample t-test, p < 0.05) against a baseline **produced by an earlier, explicitly approved commit**; exit 0                       |
| 5   | `quality-contract.json` | `utils/quality-dashboard.sh --check docs/quality-contract.json` | `docs/quality-contract.json` (committed) | `target/logs/dashboard/report.jsonl` + `target/logs/dashboard/phase_receipt.jsonl`  | all fidelity envelopes (ESR/SNR/MR-STFT) and f64-oracle checks validated; `regression_gate` must not be `NOT_VERIFIED` (perf receipt #4 must be green first); exit 0 |

> [!IMPORTANT]
> **Same-commit rule (F-EVID-06).** All five receipts must refer to the *same*
> commit hash with a **clean working tree**. The release record (§6) must bind
> them together: the operator archives the five artifacts under the canonical
> names above plus the `git_commit`, `run_id`, timestamps and SHA-256 digests.
> A receipt produced on any other commit, dirty tree, or stale artifact does
> not count. `git_commit` in fingerprints is provenance only — the archived
> receipts are the authority.

### 5.2. Baseline Bootstrap Is a Separate Ceremony

`utils/tests-performance-regression.sh --bootstrap-baseline` (Criterion
baseline) and `utils/quality-dashboard.sh --save` (contract snapshot) are
**never** run in the same command as their `--check`, and never by an AI agent.

- The bootstrap/`--save` is a **separate ceremony** performed by a human
  operator on an isolated, calibrated machine (governor `performance`, low
  load, optional `NAM_BENCH_CORE`).
- The produced baseline is bound to its **producer commit**: the release record
  must record which commit generated the baseline and evidence that this commit
  was explicitly approved as the performance/fidelity reference for the release.
  Re-using a baseline bootstrapped in the same session as the check proves
  repeatability only — it does not prove absence of regression (F-EVID-06).
- A baseline may be renewed only when the delta is intentional and
  reproducible; the renewal is a new approved ceremony with its own record
  entry, never folded silently into the certification run.

### 5.3. Human Certification Checklist (all five receipts, same clean commit)

- [ ] **Zero Panics or Memory Leaks:** Across all manual tiers and automated test suites.
- [ ] **Strict RT-Safety:** Zero heap allocations on the audio thread hot-path (Tier 2C.1 and Long Phase 4).
- [ ] **Determinism & Invariance:** Block-size invariance verified ($< 10^{-7}$ difference) and reset idempotency confirmed.
- [ ] **Architecture Coverage:** All five model families (WaveNet A1/A2, LSTM, ConvNet, Linear) load and process cleanly.
- [ ] **Receipt 1 — `lints.log`:** `utils/lints.sh` exit 0, zero warnings (all-features `--all-targets`), formatting/SPDX/anti-pattern/`#[allow]` policies green, binary AVX-512 absence certification green.
- [ ] **Receipt 2 — `quick-strict.log`:** `NAM_QUICK_STRICT=1 utils/tests-quick.sh` with `FIDELITY: OK` and `OVERALL: PASSED` (zero gaps, exit 0).
- [ ] **Receipt 3 — `long-strict.log`:** `utils/tests-long.sh --strict-pre-release` with `OVERALL: PASSED` and receipt validated via `nam_long_receipt validate --strict` (`gaps: []`, exit 0).
- [ ] **Receipt 4 — `perf-check.log`:** `utils/tests-performance-regression.sh --check` passed against a pre-approved baseline from an earlier commit (exit 0).
- [ ] **Receipt 5 — `quality-contract.json`:** `utils/quality-dashboard.sh --check docs/quality-contract.json` validated all audio fidelity and f64-oracle contracts (exit 0, `regression_gate` green).
- [ ] **Same-commit binding:** all five receipts reference the same clean commit hash; digests and `run_id`s recorded in the certification record (§6).

---

## 6. Human Pre-Release Certification Record Template

When certifying a release candidate or milestone tag, the human operator completes and archives the following audit record. **All five receipts must reference the same clean commit hash** (§5.1); the baseline producer commit (§5.2) is recorded separately.

```markdown
### Release / Audit Certification Record

- **Date (UTC):** YYYY-MM-DD HH:MM:SS
- **Operator Name:** Fábio Henrique de Lima Silva
- **Git Commit:** <commit-sha> (clean working tree required)
- **Rustc Version:** rustc X.Y.Z (Edition 2024)
- **CPU Architecture & Model:** <lscpu output summary>
- **CPU Governor:** performance
- **Run IDs:** <run_id> per receipt below
- **Baseline Producer Commit (perf):** <commit-sha of the approved --bootstrap-baseline ceremony, §5.2>

#### The Five Mandatory Receipts (same clean commit):
- [ ] **Receipt 1 — lints:** `target/logs/lints.log` (archived), exit 0, zero warnings, binary AVX-512 absence certification green. Digest: <sha256>
- [ ] **Receipt 2 — quick-strict:** `target/logs/quick-receipt.txt` with `FIDELITY: OK` / `OVERALL: PASSED` (zero gaps). Digest: <sha256>
- [ ] **Receipt 3 — long-strict:** `target/logs/long-audit-receipt.jsonl` validated via `nam_long_receipt validate --strict` — `OVERALL: PASSED`, `gaps: []`. Digest: <sha256>
- [ ] **Receipt 4 — perf-check:** `target/logs/regression_phase_receipt.jsonl` (exit 0) against baseline produced by `<baseline producer commit>` — distinct from the certified commit. Digest: <sha256>
- [ ] **Receipt 5 — quality-contract:** `docs/quality-contract.json` verified via `utils/quality-dashboard.sh --check` (exit 0, all fidelity + f64-oracle envelopes green, `regression_gate` not `NOT_VERIFIED`). Digest: <sha256>

#### Verification Checklist:
- [ ] All five receipts were generated on the **same clean commit** (no dirty-tree or stale-artifact receipts).
- [ ] `utils/lints.sh` executed cleanly (zero warnings, formatting intact, SPDX headers verified, binary AVX-512 absence certified).
- [ ] `NAM_QUICK_STRICT=1 utils/tests-quick.sh` passed with `FIDELITY: OK` and `OVERALL: PASSED` (zero gaps).
- [ ] `utils/tests-long.sh --strict-pre-release` executed with `OVERALL: PASSED` (receipt validated via `nam_long_receipt validate --strict`, `gaps: []`).
- [ ] `utils/tests-performance-regression.sh --check` passed without regression against the pre-approved baseline (producer commit recorded, distinct from the certified commit).
- [ ] `utils/quality-dashboard.sh --check docs/quality-contract.json` satisfied all fidelity and latency envelopes.

#### Certification Verdict:
[ APPROVED FOR RELEASE / REJECTED ]
```
