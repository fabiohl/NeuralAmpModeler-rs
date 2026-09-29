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

| Tier / Runner          | Scope & Purpose                                                                         | Target Duration              | Execution Cadence                                  |
|:---------------------- |:--------------------------------------------------------------------------------------- |:---------------------------- |:-------------------------------------------------- |
| **Static Quality Gate**| 🔍 **Static Analysis & Lints:** Formatting, 7-axis check, Clippy, doc-tests, SPDX       | ~45s                         | Routine developer loop, pre-commit & Receipt #1    |
| **Tier 1 (Manual)**    | ⚡ **Smoke Test:** Sanity checks on model loading, dispatch, and basic inference        | ~2 min                       | After modifications to core DSP or loader modules  |
| **Tier 2 (Manual)**    | 🎯 **Feature Verification:** Block-invariance, reset idempotency, DSP stage transitions | ~10–15 min                   | Milestone completion or major feature integration  |
| **Tier 3 (Manual)**    | 🛡️ **Robustness & Stress:** Continuous soak, SPSC burst contention, rate modulation     | ~20–30 min                   | Pre-release audits or major engine refactorings    |
| **Agile Quick Runner** | 🚀 **Agile First-Line QA:** Structural debug tests, float/C++ parity, parser fuzzing    | ~2 min (hardware-dependent)  | Pre-commit check or local iterative validation     |
| **Long Audit Runner**  | 🔬 **Exhaustive Pre-Release Audit:** Soak, QA defenses, full matrix, heap audit, RT     | ~10 min (hardware-dependent) | Nightly builds and pre-release human certification |

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

   The test harness probes the governor of the effective benchmark core (`NAM_BENCH_CORE`, default: `nproc / 2`).

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

### 4.3 Disk Logs & Diagnostic Artifacts Inventory

All test runners persist execution logs under `target/logs/`:

| File Path                                        | Generating Component / Phase           | Diagnostic Contents                                                                                                 |
|:------------------------------------------------ |:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------- |
| **`target/logs/quick-receipt.txt`**              | `tests-quick.sh` (Final)               | Summary receipt with `FIDELITY:`, `GAP:`, and `OVERALL:` status.                                                    |
| **`target/logs/quick-phase1.log`**               | `tests-quick.sh` (Phase 1)             | Structural unit tests, DSP logic, and channel checks (Debug profile).                                               |
| **`target/logs/quick-phase2.log`**               | `tests-quick.sh` (Phase 2)             | Float measurement oracles, golden vectors, and C++ `quick_parity` (Release profile).                                |
| **`target/logs/quick-phase3.log`**               | `tests-quick.sh` (Phase 3)             | Capped proptest parser fuzzing logs.                                                                                |
| **`target/logs/long-audit-receipt.jsonl`**       | `tests-long.sh` (Final)                | Machine-readable structured audit receipt (per-phase JSON entries).                                                 |
| **`target/logs/catalog_preflight.log`**          | `tests-long.sh` (Preflight 2)          | Fixture catalog discovery, SHA-256 manifest checks, and missing fixture diagnostics.                                |
| **`target/logs/meta_coherence.log`**             | `tests-long.sh` (Preflight 4)          | Catalog definition to test registration cross-validation.                                                           |
| **`target/logs/simd_probe.log`**                 | `tests-long.sh` (Preflight diagnostic) | CPU feature bits, OS AVX-512 state, Cargo feature status, and dispatched backend.                                   |
| **`target/logs/package-list.err`**               | `tests-long.sh` (Preflight 3)          | Diagnostics from `cargo package --list` crate packaging validations.                                                |
| **`target/logs/cmake-configure.log`**            | `utils/ensure_namcore_render.sh`       | CMake configuration output when compiling C++ `render` CLI.                                                         |
| **`target/logs/cmake-build.log`**                | `utils/ensure_namcore_render.sh`       | Compilation logs for the C++ reference render binary.                                                               |
| **`target/logs/phase1-soak.log`**                | `tests-long.sh` (Phase 1)              | Numerical soak logs, SPSC queue sweeps, and endurance metrics.                                                      |
| **`target/logs/phase-defense-scripts.log`**      | `tests-long.sh` (Phase 2)              | Structural invariant checks, QA defenses (`qa_defense.rs`), and symbol export guards (`libm_export_guard.rs`).      |
| **`target/logs/phase2-proptests-parity.log`**    | `tests-long.sh` (Phase 3)              | Full C++ parity comparisons, multi-SR goldens, and 100k-case proptests.                                             |
| **`target/logs/subphase-isa-parity.log`**        | `tests-long.sh` (Phase 3 Subphase)     | Cross-ISA determinism validation logs (`isa_parity`).                                                               |
| **`target/logs/phase3-heap-audit.log`**          | `tests-long.sh` (Phase 4)              | `CountingAllocator` memory interceptor reports verifying zero allocations on RT thread.                             |
| **`target/logs/phase4-rt-deadline.log`**         | `tests-long.sh` (Phase 5)              | Latency histograms and deadline statistics ($p99 < 1.33\text{ ms}$). Note: percentiles are log2 bucket upper edges. |
| **`target/logs/phase5-rt-jitter.log`**           | `tests-long.sh` (Phase 6)              | Real-time jitter telemetry and thread contention profiles.                                                          |
| **`target/logs/phase6-loom.log`**                | `tests-long.sh` (Phase 7)              | Loom model-checking state-space exploration logs.                                                                   |
| **`~/.cache/neural-amp-modeler-rs/crash-*.txt`** | Runtime Panic Hook                     | Diagnostic crash reports rendered without heap allocations.                                                         |

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
