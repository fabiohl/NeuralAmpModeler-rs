<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Performance Benchmarks (Criterion)

The `nam-rs` project uses **Criterion.rs** as its official performance benchmarking suite. Given the latency-sensitive nature of a real-time audio engine (DSP), conducting measurements with statistical rigor is essential to avoid being misled by operating system variations (noise, context switches, clock fluctuations).

> [!NOTE]
> **Document scope.** This is the authoritative reference for Criterion benchmarking in
> `nam-rs`: how to run/interpret benches, and the full rationale, workflow, and
> troubleshooting for the performance regression gate ([`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh)).
> The functional/correctness `cargo test` suites ([`utils/tests-quick.sh`](../utils/tests-quick.sh), [`utils/tests-long.sh`](../utils/tests-long.sh))
> and their feature/phase architecture are documented separately in [`testing.md`](testing.md);
> that document only cross-references benchmarks, it does not duplicate this one.

## How to Run the Benchmarks

To execute the performance suite:

```bash
# Core inference benchmark suite
cargo bench --bench inference_bench

# Performance regression gate suite
cargo bench --bench regression_gate

# Specialized DSP, Math & Kernel benchmark suites
cargo bench --bench cabsim_bench
cargo bench --bench dsp_bench
cargo bench --bench math_bench
cargo bench --bench gemv_bench
cargo bench --bench head_gemv_bench
cargo bench --bench linear
cargo bench --bench dot_4x_bench
cargo bench --bench fft_radix4_bench
cargo bench --bench kahan_conv1d_bench
```

### Long-Duration Benchmarks (Soak Bench)

To evaluate performance under constant pressure and identify jitter caused by cache misses or TLB misses in large blocks, the project offers a long-duration benchmarking suite (30s+ per function):

```bash
cargo bench --features long_bench --bench long_inference_bench
```

Or via the recommended manual trigger script:

```bash
bash utils/tests-long.sh
```

These benchmarks use blocks of **4096 samples** (~85ms), reducing the relative weight of invocation overhead and focusing purely on the DSP engine's throughput.

## How to Interpret Criterion Output

When you run a benchmark, Criterion reports output similar to this:

```text
WaveNet_Standard_CH16_64samp_48kHz
                        time:   [107.03 µs 107.32 µs 107.61 µs]
                        change: [−9.3273% −6.2506% −3.5233%] (p = 0.00 < 0.05)
                        Performance has improved.
                        Found 5 outliers among 50 measurements (10.00%)
  ...
```

### Understanding the Metrics

1. **`time: [A B C]` (Confidence Interval)**
   Shows the execution time per iteration, expressed through a **95% confidence interval**.
   * The central number (`B`, e.g., `107.32 µs`) is the best point estimate of the mean time.
   * The outer numbers (`A` and `C`) define the lower and upper bounds, statistically guaranteeing (with 95% certainty) that the true performance lies within this margin.
2. **`change: [...]` and `(p = X < 0.05)` (Statistical Significance)**
   * `change` displays the percentage difference compared to the last run on the same machine (negative values indicate faster code).
   * The **p-value** (`p`) indicates the probability that this variation occurred by chance. If `p < 0.05` (5% significance level), Criterion certifies that the observed variation is real and not just operating system noise.
3. **Textual Conclusions**
   Based on mathematical calculations, the software summarizes the conclusions:
   * **Performance has improved / regressed**: The p-value confirmed that the source code change caused a measurable statistical difference (positive or negative).
   * **Change within noise threshold**: The p-value is high, the error margins overlap, or the variation is negligible. The detected change is noise.
4. **Outliers (Jitter)**
   Samples are run hundreds of times, and anomalies are reported. In a critical real-time system like `nam-rs`, occurrences of `high severe` are usually linked to *jitter* (processing glitches, audio thread preemption by the OS kernel, cache misses, etc.). Running benchmarks in shielded environments (SCHED_FIFO and CPU affinity enabled) mitigates outliers.

## Temporal History (Baselines)

You do not need to compare times mentally. **Criterion automatically saves the baseline of your last run**.

All historical tracking metrics are recorded in local files within your project under: `target/criterion/`

> [!IMPORTANT]
> **Current vs. historical numbers.** The authoritative *current* per-model latency
> figures come from a fresh `regression_gate` run — most conveniently via
> [`utils/quality-dashboard.sh`](../utils/quality-dashboard.sh) (PERFORMANCE section, median per 64-sample block).
> Reference snapshot (Ryzen 7 5700U, AVX2 @ 64 samples / 48 kHz): WaveNet Std CH16 ≈ 36.9 µs
> (2.8%, 2404 µs/MMAC), Feather CH8 ≈ 19.4 µs (1.5%, 5031 µs/MMAC), Lite CH12 ≈ 52.6 µs (3.9%, 6039 µs/MMAC),
> Nano CH4 ≈ 17.4 µs (1.3%, 17969 µs/MMAC outlier due to layer overhead),
> A2-Full CH8 ≈ 27.6 µs (2.1%), A2-Lite CH3 ≈ 18.4 µs (1.4%), LSTM 1×16 ≈ 7.5 µs (0.6%),
> LSTM 2×8 ≈ 7.6 µs (0.6%), ConvNet ≈ 10.2 µs (0.8%), Linear RF=2048 ≈ 0.3 µs (0.02%),
> DSP Resampler (44.1k→48k) ≈ 1.3 µs, DSP CabSim IR Medium (512) ≈ 1.3 µs,
> Full DSP Pipeline Base (No OS) ≈ 37.2 µs (2.8%), Full DSP Pipeline HQ (4× OS) ≈ 150.6 µs (11.3%).
> All ≤ 3.9% of the 1333 µs RT budget for single-model inference. The "Experiment Report"
> sections further down are **historical point-in-time studies** documenting engineering
> decisions; their absolute numbers (e.g. WaveNet Std ≈ 92.6 µs) predate later optimizations
> and are retained only to justify the decisions, not as current performance claims.

*(Note: `nam-rs` intentionally disables HTML report generation with temporal charts in `Cargo.toml` (`default-features = false`) to omit downloading extensive visual dependencies, limiting evaluation to the console).*

## Regression Gate — Catching Latency Degradation Before It Ships

[`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) is the **canonical home of benchmark-based
performance defense** in `nam-rs`: the one script whose entire job is to stand as a
statistical wall against DSP hot-path decay. It acts as a CI guard — it compares the
current build against a persisted statistical baseline and fails the pipeline if a
slowdown is detected. This is your primary tool to ensure that no commit silently pushes
latency toward the 1.33 ms real-time deadline. It is deliberately narrow in scope (unlike
[`utils/tests-quick.sh`](../utils/tests-quick.sh) and [`utils/tests-long.sh`](../utils/tests-long.sh), which cover functional/correctness
regressions): its only mandate is baseline-gated performance.

### How It Works

1. **Core pinning** — The script uses `taskset -c <core>` (dynamically defaulting to `nproc / 2` to avoid OS/IRQ noise; configurable via `NAM_BENCH_CORE`) to lock the benchmark to a single CPU core, eliminating scheduler noise and cache-line bouncing between cores.
2. **Statistical rigor** — The `regression_gate` bench suite runs **19** targets (10 static models + 4 dynamic models + 5 DSP infrastructure benches) with `sample_size=100, measurement_time=5s, warm_up_time=1s, noise_threshold=0.05`. Dispatch is forced to `InstructionSet::Avx2` via `ForceAvx2Guard` so hosts with AVX-512 still measure the x86-64-v3 contract path.
3. **Baseline comparison** — Criterion performs a two-sample t-test between the current run and the stored baseline. If it detects a statistically significant regression (p < 0.05 **and** outside the 5% noise band), the script exits with code 1.
4. **Baseline storage** — Baselines are persisted under **`.performance-baselines/`** (repo-local, gitignored). `target/criterion/` is only a **transient** Criterion working area restored from `.performance-baselines/` before each run. Persist/restore use **replace-copy of top-level** `…/<bench>/ci-baseline/` only; nested `ci-baseline/ci-baseline/…` paths (historical `cp -a` into an existing dest) are sanitized and never re-copied. An environment fingerprint (`.performance-baselines/baseline-fingerprint.json`) records CPU model, full x86-64-v3 ISA label (`AVX2/FMA/F16C/BMI`, including LZCNT via `lzcnt` or Linux `abm`), rustc, target triple, governor, bench core, and producing commit.
5. **Immutability** — `--check` is strictly read-only. It never auto-creates a baseline. If no baseline exists, it fails with `MISSING_BASELINE` and exit code 1, directing the operator to run `--bootstrap-baseline` manually.
6. **Sub-µs DSP micro-benches** — `RT_DSP_Resampler_*` and `RT_DSP_CabSim_IR_Medium` process a fixed batch of **64 blocks** per Criterion sample so timer noise stays under the 5% wall. The quality dashboard divides those medians by 64 and reports **per-block** latency (contract units unchanged). Changing the batch size requires a human baseline renewal.

### Daily Workflow

```sh
# 1. Before starting work: confirm the current baseline is clean.
utils/tests-performance-regression.sh --check

# 2. Develop your changes. Run lints and quick tests frequently.
utils/lints.sh && utils/tests-quick.sh

# 3. Before committing: re-run the regression gate.
utils/tests-performance-regression.sh --check

# 4. GREEN  → safe to commit/push.
#    RED    → investigate the regression before proceeding.

# 5. Only update the baseline when you intentionally changed performance
#    (e.g., adding a feature with a measured, understood, acceptable cost)
#    and all other tests pass. This MUST be performed by a human operator —
#    automated/bootstrap execution is prohibited.
utils/tests-performance-regression.sh --bootstrap-baseline
```

### First-Time Setup and Post-Optimization Renewal (Human-Only)

Agents and CI **must not** run `--bootstrap-baseline` or `quality-dashboard.sh --save`.
Both are deliberate human operations on a machine with governor `performance`,
low background load, and preferably a single pinned core (`NAM_BENCH_CORE`).

**Canonical sequence after intentional DSP/perf changes (or first clone):**

```sh
# 0. Optional: commit shell/docs-only fixes first so the tree is clean.
#    Prefer a clean tree for provenance; dirty state is recorded but harder to audit.

# 1. Create / replace the Criterion baseline under .performance-baselines/
utils/tests-performance-regression.sh --bootstrap-baseline

# 2. Prove the new baseline is readable (standalone gate must be green)
utils/tests-performance-regression.sh --check
# Expected: "No performance regression detected." exit 0

# 3. Only then freeze the integrated fidelity+perf contract
utils/quality-dashboard.sh --save docs/quality-contract.json
# Requires ALL dashboard phases PASS (including regression_gate).
# Fail-closed: if regression_gate != PASS, the file is NOT written.

# 4. Close the loop on the same revision
utils/quality-dashboard.sh --check docs/quality-contract.json
# Expected: FIDELITY OK + PERFORMANCE OK + contract satisfied
```

**Why step 2 can PASS and step 4 still fail:** the dashboard runs ~2 minutes of
fidelity work (goldens, f64 oracle, quick_parity, …) **before** invoking the
regression gate. That raises thermals/OS noise versus a cold standalone
`--check`. Micro-benches near the noise floor (e.g. `RT_LSTM_2x8` ~7.5 µs,
`RT_Linear` ~340 ns) may then report a Criterion "regressed" of a few percent
even with **no code change**. This is environmental, not a fidelity bug.

**Operator response to a flaky dashboard `--check` after a green standalone gate:**

1. Inspect `target/logs/regression-check.log` for the `"Performance has regressed"` line.
2. Re-run **only** `utils/tests-performance-regression.sh --check` with load ≪ 1.
3. If standalone is green again: re-run `utils/quality-dashboard.sh --check …`
   after a short cool-down; do **not** bootstrap solely to silence noise.
4. Bootstrap again only when the delta is intentional (real code change) or
   reproducible under isolation across multiple cold runs.

`--check` never auto-creates a baseline after `cargo clean` or a fresh clone:
missing `.performance-baselines/` → `MISSING_BASELINE`, exit 1.

### Pre-Flight Checklist (Performance Gate)

Performance measurement is only meaningful under a controlled environment.
Run this checklist **before** `--check` or `--bootstrap-baseline`:

* [ ] **Baseline present.** `.performance-baselines/baseline-fingerprint.json` exists
      (and the `ci-baseline` series under `.performance-baselines/`).
      Absent → the standalone gate fails `MISSING_BASELINE`; the dashboard
      displays performance as `NOT_VERIFIED`.
* [ ] **CPU governor = `performance`.** Verify with
      `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`.
      A different governor (`powersave`, `schedutil`, amd_pstate default) makes
      measurements incomparable → `INCOMPARABLE_ENVIRONMENT`.
* [ ] **Low background load.** Close browsers/heavy services; thermals and
      co-resident load dominate the micro-bench noise floor (see the flaky
      `--check` note above).
* [ ] **Core pinning available.** `taskset` present (defaults to `nproc / 2`;
      override with `NAM_BENCH_CORE`). Same physical core count as the
      bootstrap machine — it is part of the environment fingerprint.
* [ ] **Fresh benchmark set.** Every `regression_gate` benchmark must have a
      saved baseline series: a new/renamed benchmark fails `--check` with
      `BASELINE_COVERAGE_GAP` until a human re-bootstraps the baseline.

> [!IMPORTANT]
> **`NOT_VERIFIED` has exactly one semantic.** `MISSING_BASELINE` and
> `INCOMPARABLE_ENVIRONMENT` both mean "performance could not be verified
> against the saved baseline". The standalone gate fails typed on both;
> `quality-dashboard.sh` renders a single unambiguous `NOT_VERIFIED` state
> (never green, never counted as PASS) and its `--check` mode fails on it.
> **Baseline bootstrap is always a human operation** — agents and CI are
> prohibited from `--bootstrap-baseline` and `quality-dashboard.sh --save`.

### Script Modes

| Mode                | Command                                                      | Purpose                                                                                                                                |
|:------------------- |:------------------------------------------------------------ |:-------------------------------------------------------------------------------------------------------------------------------------- |
| **Check** (default) | `utils/tests-performance-regression.sh` or `--check`         | Compare against baseline; fail on statistically significant regression (p < 0.05). Strictly read-only — never auto-creates a baseline. |
| **Bootstrap**       | `utils/tests-performance-regression.sh --bootstrap-baseline` | Create a new baseline and environment fingerprint. Human-only operation.                                                               |

### Environment Variables

| Variable            | Default       | Purpose                                                 |
|:------------------- |:------------- |:------------------------------------------------------- |
| `NAM_BENCH_CORE`    | `nproc / 2`   | CPU core number to pin benchmarks to via `taskset`.     |
| `NAM_BASELINE_NAME` | `ci-baseline` | Criterion baseline name (allows per-machine baselines). |

### Relationship to Other QA Tools

| Tool                                                                                | Role                                                                                                                                                                                                   |
|:----------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `tests/rt_constraints/rt_deadline.rs`                                               | **Absolute hard gate** — `assert!(p99 < 1330 μs)` for all SKUs. This is the pass/fail ceiling.                                                                                                         |
| [`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) | **Relative guard, baseline-gated** — the canonical home for perf-regression benchmarking. Catches degradations *within* the safe zone (e.g., 100 μs → 150 μs, still under 1.33 ms but 50% worse).      |
| [`utils/remote-simd-gate.sh`](../utils/remote-simd-gate.sh)                         | **Remote SIMD Gating Suite** — Automated harness for executing cross-ISA parity validation and Criterion benchmarks on AVX-512 remote hardware, emitting `target/logs/remote-simd-receipt.json`.       |
| [`utils/tests-long.sh`](../utils/tests-long.sh)                                     | **Nightly Audit Suite** — Focuses on heavy functional, soak, parity, and RT-safety tests; benchmarks are omitted from the nightly runner to optimize execution time.                                   |
| [`utils/tests-quick.sh`](../utils/tests-quick.sh)                                   | Fast path (approximately 2 minutes, depending on the hardware) — does **not** include benchmarks (would exceed the time budget). Use `utils/tests-performance-regression.sh` directly for perf checks. |

> [!IMPORTANT]
> **Always run `--check` before pushing.** A passing `utils/tests-quick.sh` and `utils/tests-long.sh` does
> **not** guarantee the absence of performance regression — only the regression gate provides
> a statistical comparison against the known-good baseline.

### Interpreting a Failed Gate

If the script exits with `❌ PERFORMANCE REGRESSION DETECTED`:

1. Open `target/logs/regression-check.log` and locate the `"regressed"` entry.
2. Look at the reported confidence interval for the regressed benchmark(s) — how many μs and what percentage?
3. Re-run `cargo bench --bench regression_gate -- --baseline ci-baseline` to confirm the result is reproducible (not noise from a transient system load spike).
4. If the regression is real and unintentional: bisect your recent changes to find the cause.
5. If the regression is intentional (e.g., a new feature with a measured, accepted overhead): re-save the baseline with `--bootstrap-baseline` **and document the change and its measured cost** in your commit message.

## Quality Contract — Performance Lens

The **Quality Contract** ([`quality-contract.json`](quality-contract.json)) extends the
regression defense with a dashboard-integrated second line of defense that freezes
both fidelity and performance metrics into a versioned, machine-readable baseline.

### How It Fits with the Regression Gate

| Tool                                                                                | Statistical Rigor                    | Speed    | Scope                                                                                     |
|:----------------------------------------------------------------------------------- |:------------------------------------ |:-------- |:----------------------------------------------------------------------------------------- |
| [`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) | Criterion two-sample t-test (p<0.05) | ~5-8 min | **Primary authority** — catches slow regressions within the safe zone (e.g., 100→150 µs). |
| [`utils/quality-dashboard.sh`](../utils/quality-dashboard.sh) `--check`             | Conservative relative margin         | ~3-5 min | **Second line** — integrated with fidelity checks; +10% latency tolerance.                |

The two tools serve complementary roles:

* **`utils/tests-performance-regression.sh`** is the strict, narrow statistical gate —
  the definitive answer to "did latency increase with p < 0.05 confidence?"
* **`utils/quality-dashboard.sh --check`** is the broad, integrated check — it answers
  "do fidelity *and* performance both pass, in one command?" with conservative
  margins designed to absorb OS scheduling noise without false positives.

### Performance Tolerance in the Contract

The contract applies a **10% margin** on median latency:

```text
nova_lat > contrato_lat × 1.10  →  VIOLAÇÃO
```

This is intentionally more conservative than the regression gate's statistical
test — a 10% margin absorbs transient scheduling noise while still catching
degradations large enough to matter (e.g., 56 µs → 62 µs is within margin;
56 µs → 95 µs is a clear violation).

> [!NOTE]
> The contract's performance section is filled from the same `regression_gate`
> Criterion run that `utils/tests-performance-regression.sh` drives. The dashboard
> copies latencies **only** when `regression_gate=PASS` for the current `run_id`
> (stale `target/logs/regression-check.log` is never reused — PERF-002).
>
> **Two independent artifacts (both human-renewed when perf intentionally changes):**
>
> | Artifact           | Path                                           | Role                                                    |
> | ------------------ | ---------------------------------------------- | ------------------------------------------------------- |
> | Criterion baseline | `.performance-baselines/` (+ fingerprint JSON) | Statistical relative gate (t-test, p&lt;0.05)           |
> | Quality contract   | `docs/quality-contract.json`                   | Frozen fidelity + median latency snapshot for `--check` |
>
> Updating one does **not** update the other. Order is always:
> bootstrap Criterion → standalone `--check` green → dashboard `--save` → dashboard `--check`.

### Baselines and Renewal

The official performance baseline lives in [`quality-contract.json`](quality-contract.json) alongside
fidelity metrics. The **full renewal procedure** — including prerequisites, the
`--bootstrap-baseline` / `--check` cycle, and the mandatory commit-message justification — is
documented in [`testing.md`](testing.md#95-baseline-renewal-procedure-human-only).

> [!CAUTION]
> The Criterion baseline (`.performance-baselines/`, managed by
> `utils/tests-performance-regression.sh --bootstrap-baseline`) and the Quality
> Contract (`docs/quality-contract.json`) are **independent**. Updating one does
> not update the other. Criterion data is local/gitignored; the contract file is
> committed. Both must be renewed (human-only) when latency or fidelity
> characteristics intentionally change — always Criterion first, then `--save`.

#### Operational Protocol & Toolchain Variance Defense (F-SIMD-05)

To maintain absolute reproducibility across development sessions, CI runs, and AI pair-programming:

1. **Toolchain & Thermal Decoupling:** Code correctness and mathematical integrity are evaluated via the fidelity and unit test suites (`utils/tests-quick.sh`), decoupling code verification from thermal fluctuations or compiler toolchain variations (`rustc 1.98` vs `1.97.1`).
2. **Strict AI Prohibition:** AI agents are **strictly prohibited** from executing `quality-dashboard.sh --save` or `utils/tests-performance-regression.sh --bootstrap-baseline`.
3. **Operator Renewal Protocol:** Baseline updates to `docs/quality-contract.json` are the exclusive prerogative of the human operator / PO, performed on a cold, isolated machine (`governor=performance`, pinned core, background load < 0.1).

---

## Comparative Results: Scalar LSTM vs. SIMD (Fused Gates)

Optimizations introduced gate fusion and SIMD activations (AVX2/AVX-512) into the recurrent networks' hot-path. Below are the measured gains on an x86-64-v3 (AVX2/FMA) architecture for 64-sample blocks:

| Topology      | Implementation    | Latency (Average) | Speedup    |
|:------------- |:----------------- |:----------------- |:---------- |
| **LSTM 1x8**  | Scalar (Baseline) | ~45.12 µs         | -          |
| **LSTM 1x8**  | **SIMD Fused**    | **~2.27 µs**      | **19.84x** |
| **LSTM 2x16** | Scalar (Baseline) | ~45.19 µs         | -          |
| **LSTM 2x16** | **SIMD Fused**    | **~10.86 µs**     | **4.16x**  |

### Technical Conclusion

The performance gain exceeding **4x** on complex models (2x16) and nearly **20x** on simple models (1x8) validates the kernel fusion strategy. By processing the 4 LSTM gates simultaneously via SIMD vectors and keeping data in registers between the Sigmoid and Tanh activations, we drastically reduce CPU cycles wasted on redundant loads/stores and memory latency.

---

## Cycle Budget (WaveNet Hot-Path)

To guide future optimizations, granular instrumentation of the WaveNet hot-path (`WaveNetLayer::process_block_internal`) was performed using hardware cycle counters (**RDTSC**). This measurement identifies where the CPU spends most of its time during audio block processing.

### Cycle Distribution per Stage (Per Layer)

Below is the average percentage distribution of cycles on an x86-64-v3 (AVX2) architecture for a Standard model (CH=16):

| Operational Stage          | Operations Involved                 | Budget (%) | Technical Justification                                              |
|:-------------------------- |:----------------------------------- |:---------- |:-------------------------------------------------------------------- |
| **Conv1D (SIMD GEMV)**     | Causal convolution, MACs, dilation  | **~45%**   | Most computationally intensive phase (matrix-vector multiplication). |
| **1x1 & Residual (Fused)** | Dense projection, residual addition | **~25%**   | High memory pressure (read-modify-write) and channel projection.     |
| **Mixin (Conditioning)**   | Timbre metadata injection           | **~15%**   | Dense operation applied to the input of each layer.                  |
| **Act & Head (Fused)**     | Tanh/Sigmoid, Skip-Connections      | **~15%**   | Cost of transcendental functions (approximated via SIMD).            |

### Data Flow Analysis (Array Level)

At the `WaveNetLayerArray` level, the layer cascade dominates processing (**>90% of total time**). Interface stages (input **Rechannel** and output **Head Rechannel**) represent a negligible fixed overhead as the number of layers increases, validating the scalability of the `nam-rs` architecture for complex models.

> [!TIP]
> Fusing **Tanh** with **Head Accumulation** was the most impactful optimization, reducing the activation stage budget from ~30% to ~15% by eliminating redundant passes through L1 Cache memory.

---

## Experiment Report: Temporal Tiling (Dual-Frame) on Conv1D

In the hot-path optimization, a **Temporal Tiling** variant ("Dual-Frame" processing) was designed and tested for `Conv1D` kernels, aiming to maximize L1 Cache weight reuse by processing two frames simultaneously in WaveNet inference.

### Measurement Results (64 samples, 48kHz, CH=16, AVX2)

* **Single-Frame (Baseline):** ~92.6 µs
* **Dual-Frame Tiling:** ~110 µs (Regression of ~19%)

### Analysis and Architectural Decision

Although theory suggested that loading weights from memory half as often would save bandwidth (L1 cache), in practice the x86-64 architecture (AVX2/FMA) proved to be limited by **Register Pressure**.
To process two frames in parallel:

1. The number of required SIMD accumulators doubled (from 4 YMM to 8 YMM per channel).
2. Instruction overhead in the frontend (e.g., broadcasts and blends) outweighed the savings on loads.
3. The compiler was forced to use register spilling or hit execution port bottlenecks for blend/shuffle instructions (Port 5).

**Conclusion:** The primary bottleneck of `Conv1D` in `nam-rs` is not tied to L1 Cache bandwidth, but rather to computational throughput and register contention in the backend (FMA). Because of this, while the kernel implementation has been kept in the `SimdMath` trait for portability and testing on architectures with more registers (e.g., AVX-512 or ARM NEON), the main loop in `WaveNetLayer` continues to use **Single-Frame processing** to ensure the lowest latency and highest real-time stability.

---

## Experiment Report: Stereo Fusion in the Output Stage

The goal was to eliminate redundant memory passes in the final output stage by fusing the gain (Hysteresis/Gate) operations of the L and R channels into a single stereo SIMD call.

### Measurement Results (64 samples, 48kHz, AVX2)

| Topology        | Before Fusion | After Fusion | Gain (%)  |
|:--------------- |:------------- |:------------ |:--------- |
| **WaveNet Std** | ~98.0 µs      | ~92.6 µs     | **~5.5%** |
| **LSTM 2x16**   | ~11.4 µs      | ~10.9 µs     | **~4.5%** |

### Conclusion

Stereo fusion reduces memory traffic in the L1 Cache by reading the L and R channels simultaneously and applying the gain/ramp weights in a single loop. The gain is more pronounced in smaller blocks (e.g., 32 samples, where a **~8.5%** improvement was measured), where dispatch overhead and partial cache misses have a higher relative weight.

---

## Criterion A2 Architecture

The A2 architecture introduces per-layer conditioning (FiLM + Gating) and a configurable channel count (CH=3 Lite, CH=8 Full). The implementation focused on a SIMD-heavy hot-path for CH=8 (`A2Conv1dCh8`) with col-major-per-tap weight layout, enabling AVX2 T=4 broadcast-FMA convolution.

### A2-Full (CH=8) — Optimized SIMD Path

A2-Full uses the `A2Conv1dCh8` fast path with f32 weights in col-major layout (`w[k * 64 + in * 8 + out]`), where 8 output-channel weights are contiguous per `(tap, input)` pair. This layout feeds directly into AVX2 broadcast-FMA without transposition.

| Block Size   | Latency (µs) | Per-Sample (ns) | CPU % at 48kHz |
|:------------ |:------------ |:--------------- |:-------------- |
| **64 samp**  | **~30.7 µs** | ~480            | ~2.3%          |
| **128 samp** | ~30.5 µs     | ~238            | ~1.1%          |
| **256 samp** | ~30.6 µs     | ~120            | ~0.6%          |

### A2-Lite (CH=3) — f32 Native GEMV Path

A2-Lite uses the dedicated `A2Conv1dCh3` fast path ([`src/models/a2/conv1d_ch3/mod.rs`](../src/models/a2/conv1d_ch3/mod.rs)), mirroring the CH=8 kernel design: f32 native weights in col-major-per-tap layout (one `_mm_loadu_ps` load, one `_mm_fmadd_ps` FMA per input channel — no f16 decode). The kernel is a fully unrolled GEMV (18 FMAs for K=6, 45 FMAs for K=15), with post-conv operations (Mixin, LeakyReLU, head, l1x1) batched via AVX2.

| Block Size   | Latency (µs) | Per-Sample (ns) | CPU % at 48kHz |
|:------------ |:------------ |:--------------- |:-------------- |
| **64 samp**  | **~16.3 µs** | ~255            | ~1.2%          |
| **128 samp** | ~16.3 µs     | ~127            | ~0.6%          |
| **256 samp** | ~16.3 µs     | ~64             | ~0.3%          |

### Comparative Analysis

| Variant | Weights | Channels | Conv Path                   | 64-samp Latency |
|:------- |:------- |:-------- |:--------------------------- |:--------------- |
| A2-Full | 12,146  | 8        | f32 col-major SIMD          | **~30.7 µs**    |
| A2-Lite | 1,871   | 3        | f32 col-major unrolled GEMV | **~16.3 µs**    |

---

## Gate FSM (Dynamic Hysteresis)

The gate FSM (`DynamicHysteresis`) runs in the DSP hot-path on every audio callback to decide whether to open or close the noise gate based on detected volume. The benchmark measures `update()` (state machine tick) + `multiplier()` (current gain read) across three steady-state scenarios at realistic DSP block sizes.

### Results (64, 128, 256 samples — x86-64-v3 AVX2/FMA)

| Scenario             | 64 samp  | 128 samp | 256 samp | Steady Path                                  |
|:-------------------- |:-------- |:-------- |:-------- |:-------------------------------------------- |
| **Open**             | ~2.11 ns | ~2.16 ns | ~2.17 ns | Volume above open threshold, gate stays open |
| **Closed**           | ~1.64 ns | ~1.73 ns | ~1.73 ns | Gate already closed, volume stays below      |
| **FadingOut (ramp)** | ~1.21 µs | ~1.14 µs | ~1.09 µs | Gate actively ramping multiplier toward zero |

### Running Gate_FSM bench

```sh
cargo bench --bench dsp_bench -- "Gate_FSM"
```

---

## IR Cabsim Convolution

The cabsim engine uses UPOLS (Uniform-Partitioned Overlap-Save) frequency-domain convolution. All FFTs of the kernel partitions are pre-computed at construction time; the `ConvEngine::process()` hot-path performs zero allocations and operates on pre-allocated buffers exclusively.

### Benchmarks (64-sample blocks at 48 kHz)

| Benchmark                 | IR Samples | Partitions | Latency (µs) | CPU % at 48kHz |
|:------------------------- |:---------- |:---------- |:------------ |:-------------- |
| ShortIR_64samp            | 64         | 1          | ~1.39        | ~0.1%          |
| MediumIR_2048_64          | 2,048      | 32         | ~8.15        | ~0.6%          |
| LongIR_16384_64           | 16,384     | 256        | ~58.34       | ~4.4%          |
| MediumIR_2048_256samp     | 2,048      | 8          | ~12.58       | ~0.2%          |
| Engine_Construction_2048  | 2,048      | 32         | ~19.65       | — (load-time)  |
| Engine_Construction_16384 | 16,384     | 256        | ~133.27      | — (load-time)  |

### RT-Safety Validation

* **Heap-audit tests** ([`tests/rt_constraints.rs`](../tests/rt_constraints.rs)) confirm zero allocations on the `ConvEngine::process()` hot-path.
* **Golden convolution tests** ([`tests/models/cabsim_golden.rs`](../tests/models/cabsim_golden.rs)) verify UPOLS output against direct convolution reference using deterministic synthetic IRs.

---

## Kahan Per-Tap Cost in Conv1d (Removed)

### Context

The static implementation of conv1d ([`src/models/wavenet/conv1d.rs`](../src/models/wavenet/conv1d.rs) and [`src/models/wavenet/conv1d_dual.rs`](../src/models/wavenet/conv1d_dual.rs)) previously executed Kahan compensated summation inside the per-tap loop. For K ≤ 3 (all A1 WaveNet models), simple summation error is O(3·ε) — negligible for audio — making per-tap Kahan unnecessary.

Benchmark file: [`benches/kahan_conv1d_bench.rs`](../benches/kahan_conv1d_bench.rs).

### Decision & Impact

Kahan compensated summation was removed from the static hot-path:

* [`src/models/wavenet/conv1d.rs`](../src/models/wavenet/conv1d.rs): `kahan_add` → `+=`
* [`src/models/wavenet/conv1d_dual.rs`](../src/models/wavenet/conv1d_dual.rs): `kahan_add` → `+=`
* [`src/models/wavenet/conv_input.rs`](../src/models/wavenet/conv_input.rs): `store_kahan_4_accums` → `store_4_accums`

---

## WaveNet Lite CH12: Profiling, Memory Stride & Architectural Efficiency

The WaveNet Lite variant operates with an internal channel dimension of $CH=12$. While it has 25% fewer channels than WaveNet Standard ($CH=16$), its initial latency benchmark reported **64.5 µs**, which was nearly **1.75× slower** than the larger Standard model (~36.6 µs).

### 1. Implemented Optimizations & Weight Padding

To resolve the initial bottlenecks, structural changes were implemented:

* **SIMD 8+4 Store Path:** In `store_16_accums` ([`src/models/wavenet/conv_input.rs`](../src/models/wavenet/conv_input.rs)), scalar stores were replaced with a fused 256-bit YMM store (lanes 0..7) and a 128-bit XMM store (lanes 8..11).
* **Dedicated 12x12 GEMM Kernel & Weight Padding:** In [`src/math/gemm/gemm_batch/fused_residual_batch/mod.rs`](../src/math/gemm/gemm_batch/fused_residual_batch/mod.rs) and the model loader, residual convolution weights were padded to stride 16.

These optimizations reduced the global median latency of WaveNet Lite CH12 from **68.5 µs to 52.2 µs** (a **−19.7%** improvement).

### 2. Structural ASM Analysis

Assembly comparison and stride analysis revealed that fixed setup overhead (prologue, dispatch, bounds checks) accounts for 54% of instructions in Lite CH12 versus 34.5% in Standard CH16. On AVX2, the 8+4 channel split operates with 128-bit XMM instructions for the upper 4 lanes, yielding higher instruction counts per layer than standard 16-channel YMM operations.

**Final Decision:** The WaveNet Lite CH12 SKU operates cleanly at **~52.7 µs** (96.1% headroom from the 1333 µs RT deadline), with 1e-7 parity tolerance restored and zero unneeded technical debt.

---

## WaveNet A2 Dynamic AVX2+FMA Vectorization

### Context (WaveNet A2 Dynamic AVX2+FMA Vectorization)

The WaveNet A2 Dynamic model (`WaveNetA2Dyn`) is the runtime-dimensioned fallback engine for non-catalog A2 geometries (gating, blending, head1x1, heterogeneous activations, FiLM). In earlier baseline implementations, its hot-path was predominantly scalar with compiler auto-vectorization failing on the double-nested GEMV loops (mixin, head1x1, L1x1 residual). Assembly profiling confirmed:

* **Dilated Conv**: Unrolled scalar (optimal for depthwise), with `prefetcht0` L1 prefetch
* **Mixin GEMV**: Fully scalar with register spills to stack — primary optimization target
* **Head 1×1 Projection**: Fully scalar — secondary optimization target
* **L1×1 Residual**: Fully scalar — tertiary optimization target
* **Activation/Gating**: Vectorized via `SimdMath` trait

### Implemented Vectorization Architecture

**Head 1×1 & L1×1 Residual Vectorization** ([`src/models/a2/model/dynamic/process.rs`](../src/models/a2/model/dynamic/process.rs)):

* **Head 1×1**: 8-wide `_mm256_fmadd_ps` over the `h1_in` dimension per output channel. Lane extraction preserves exact left-to-right accumulation order for bit-identical golden vector output.
* **L1×1 Dense**: 8-wide `_mm256_set1_ps` (broadcast) + `_mm256_fmadd_ps` over contiguous col-major weight rows (`bottleneck × channels`).
* **L1×1 Grouped**: 8-wide SIMD dot product for `in_pg ≥ 8`, scalar fallback otherwise.
* **Accumulation loops** (`head_accum += scratch`, `layer_in += scratch`): `_mm256_add_ps` with scalar tail.

**Mixin GEMV with Off-RT Weight Transposition** ([`src/models/a2/model/dynamic/build.rs`](../src/models/a2/model/dynamic/build.rs), [`process.rs`](../src/models/a2/model/dynamic/process.rs)):

* **Builder**: One-time per-group transposition from row-major `[out_per_g][in_pg]` to col-major `[in_pg][out_per_g]` during `set_weights`. No transposition in the hot path.
* **Hot Path**: `_mm256_set1_ps` (broadcast condition) + `_mm256_loadu_ps` (8 contiguous weights) + `_mm256_fmadd_ps` per input channel. Unified flat (groups=1) and grouped (groups>1) paths.

### Performance Results (Regression Gate, `--baseline ci-baseline`)

| Benchmark               | Scalar Baseline | Vectorized (AVX2+FMA)                   | Change                               | Notes                                              |
| ----------------------- | --------------- | --------------------------------------- | ------------------------------------ | -------------------------------------------------- |
| `RT_A2_Dyn_Gated_CH8`   | ~259 µs         | **170.9 µs** (~12.8% of 1.33 ms budget) | **≈ −34%**                           | Primary win; CH=8 fully uses 8-wide FMA paths      |
| `RT_A2_Dyn_Blended_CH3` | ~133 µs         | **135.9 µs** (~10.2% of budget)         | ≈ +2–3%                              | CH=3 stays on scalar fallbacks; accepted trade-off |
| `RT_LSTM_Dyn_1x7`       | ~15.8 µs        | **~7.9 µs**                             | **≈ −50%** vs pre-vectorization tail | Dedicated AVX2 H&lt;8 gates                        |

The **Gated CH=8** path is the design target for the 8-wide kernels. **Blended CH=3** pays a small branch/code-size tax because every SIMD width check falls to the scalar tail; the dynamic engine still serves all geometries from one code path. The Criterion baseline and `docs/quality-contract.json` reflect these post-optimization medians.

### Fidelity & Invariants

* **Golden Vectors**: `a2_dynamic_gated_ch8` and `a2_dynamic_blended_ch3` both pass with bit-identical output (MSE=0.0 vs reference).
* **f64 Oracle**: `test_oracle_vs_python_anchor_a2_gated` and `_blended` pass with exact match.
* **Block Invariance**: All A2 catalog models pass block-size invariance tests.
* **Zero-Alloc**: Heap audit confirms zero allocations on the hot path (only YMM registers + stack `[f32; 8]` lane buffers).
* **`utils/tests-quick.sh`**: Full FIDELITY: OK — structural + measurement oracles + parser fuzzing all green.
* **No regressions in non-A2Dyn models**: Changes are scoped exclusively to `WaveNetA2Dyn` (dynamic path); static A2 Full/Lite and other model families use separate compilation units.

### Assembly Confirmation

Post-optimization assembly (`cargo rustc --release --bench regression_gate -- --emit=asm`) confirms:

* **12,594 FMA instructions** in the release binary (vs. 24,342 packed SIMD overall), with `vfmadd231ps` present in the mixin, head1x1, and L1x1 inner loops.
* **No register spills** in the SIMD paths: accumulators remain in YMM registers throughout the inner loops.
* **Scalar tail code** retains exact arithmetic order (sequential lane extraction from YMM → `[f32; 8]` on stack), preserving golden vector parity.

---

## Cloud AVX-512 Benchmarking & Remote SIMD Gating Protocol

AVX-512 is **not** a production backend (2026-08 receipt failed the ≥12% `process()` N=64 gate; see [`architecture.md`](architecture.md) §1.2). This section is the **re-measurement** protocol only: same-VM AVX2 vs AVX-512 parity and latency, so a future geometry or SKU can be re-evaluated without guessing. It is not a claim that distribution binaries should dispatch AVX-512.

### 1. The Same-VM Measurement Rule

Comparing cloud latency measurements directly against local workstation baselines (such as AMD Ryzen Zen 2) is mathematically invalid due to cross-architecture IPC differences, vCPU virtualization overhead, hypervisor scheduling, and clock frequencies.

Therefore, **AVX-512 speedup gates must be evaluated by comparing AVX-512 vs AVX2 on the SAME cloud virtual machine**:

1. **AVX2 Baseline on Cloud:** Run Criterion benchmarks forcing AVX2 execution path on the cloud VM.
2. **AVX-512 arm on Cloud:** Force the AVX-512 `SimdMath` monomorph (`ForceAvx512Guard` / `TEST_ISA_OVERRIDE`) on the same binary (`-Ctarget-cpu=x86-64-v3`). Do not compare against a `-Ctarget-cpu=native` build.
3. **Speedup Gate:** Promote only if $\ge 12\%$ vs AVX2 on the canonical N=64 SKUs (WaveNet Standard CH16, A2-Full CH8, A2-Lite CH3, LSTM 1×16, LSTM 2×16) with Welch's t-test ($p < 0.05$). The 2026-08 receipt failed this gate.

* **Parity Baseline (`-Ctarget-cpu=x86-64-v3`):** What a distribution binary is compiled as. **Policy:** production runs the AVX2 `Avx2Math` path. In default builds, `detect_best_simd()` resolves to `Avx2` and `dispatch_simd!` monomorphizes only `Avx2Math`. AVX-512 research kernels (VL256 for WaveNet `dot_4x`/`accumulate` and LSTM 4-gate GEMV, ZMM for LSTM fused gates and `dot_16x`) are compiled only with `--features avx512`.
* **Native Ceiling (`-Ctarget-cpu=native`):** Secondary benchmark build compiled with full compiler auto-vectorization (`-Ctarget-cpu=native`) across the entire crate to determine the upper architectural ceiling, kept strictly distinct from the baseline regression gate.

### 3. Recommended Cloud Instances & Target Hardware

For statistically reliable and reproducible SIMD gating, virtual machines should provide dedicated vCPUs with consistent CPU clock pinning and full AVX-512 feature exposure:

| Cloud Provider           | Recommended Instance / SKU                                                  | Microarchitecture                             | SIMD Capabilities Exposed                                              | Notes                                                                  |
|:------------------------ |:--------------------------------------------------------------------------- |:--------------------------------------------- |:---------------------------------------------------------------------- |:---------------------------------------------------------------------- |
| **AWS EC2**              | `c7i.large` / `c7i.xlarge`                                                  | Intel Xeon Scalable (4th Gen Sapphire Rapids) | AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, AVX-512CD, AVX-512VNNI, AMX | **Primary Reference Platform**. Predictable turbo and dedicated vCPUs. |
| **AWS EC2**              | `c7a.large` / `c7a.xlarge`                                                  | AMD EPYC 9004 (Zen 4 "Genoa")                 | AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, AVX-512CD, AVX-512VNNI      | Full AVX-512 throughput with zero frequency downclocking.              |
| **AWS EC2**              | `c6i.large`                                                                 | Intel Xeon Scalable (3rd Gen Ice Lake)        | AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ, AVX-512CD, AVX-512VNNI      | Secondary Intel verification platform.                                 |
| **GCP**                  | `c3-highcpu-4` / `c3d-highcpu-4`                                            | Intel Sapphire Rapids / AMD Genoa             | AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ                              | Ensure dedicated core pinning is configured in VM template.            |
| **Azure**                | `Standard_F4s_v5` / `Standard_F4as_v6`                                      | Intel Ice Lake / AMD Genoa                    | AVX-512F, AVX-512VL, AVX-512BW, AVX-512DQ                              | Compute-optimized tier recommended.                                    |
| **On-Prem / Bare-Metal** | AMD Ryzen 7000 / 8000 / 9000 series, Intel Core 11th–14th Gen, Intel Xeon W | AMD Zen 4/5, Intel Golden Cove / Raptor Cove  | Full native AVX-512 execution                                          | Ideal for baseline calibration with zero virtualization jitter.        |

### 4. Step-by-Step Operator & DevOps Runbook

Follow this standard operating procedure when executing the remote SIMD gate on a fresh cloud instance or dedicated runner:

```bash
# ---------------------------------------------------------------------------
# 1. Install System Dependencies & Build Tools (Ubuntu 22.04 / 24.04 LTS)
# ---------------------------------------------------------------------------
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    pkg-config \
    linux-tools-common \
    linux-tools-generic

# ---------------------------------------------------------------------------
# 2. Install Stable Rust Toolchain
# ---------------------------------------------------------------------------
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
source "$HOME/.cargo/env"

# ---------------------------------------------------------------------------
# 3. CPU Governor Configuration (Minimize Frequency Throttling / DVFS Jitter)
# ---------------------------------------------------------------------------
sudo cpupower frequency-set -g performance 2>/dev/null || \
    echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Clone Repository and Switch to Development Branch
# ---------------------------------------------------------------------------
git clone https://github.com/fabiohl/NeuralAmpModeler-rs.git
cd NeuralAmpModeler-rs
git checkout dev

# ---------------------------------------------------------------------------
# 5. Populate Pinned Third-Party Vendor Mirrors
# ---------------------------------------------------------------------------
./utils/setup-third-party.sh

# ---------------------------------------------------------------------------
# 6. Thermal Stabilization (Hardware Cooldown)
# ---------------------------------------------------------------------------
sleep 180

# ---------------------------------------------------------------------------
# 7. Execute the Automated Remote SIMD Gating Suite
# ---------------------------------------------------------------------------
./utils/remote-simd-gate.sh
```

### 5. Automated Harness Options (`utils/remote-simd-gate.sh`)

The gating harness [`utils/remote-simd-gate.sh`](../utils/remote-simd-gate.sh) supports flexible CLI flags for automated pipelines, remote servers, and local emulation:

```bash
Usage: utils/remote-simd-gate.sh [OPTIONS]

Options:
  --sde                Run in Intel SDE emulation mode (auto-configures runner, skips bench/cooldown)
  --check-only         Run Phase 0 (Preflight) only and exit (0 on AVX-512, 2 on missing ISA)
  --skip-cooldown      Skip 180s thermal cooldown intervals (useful for fast validation passes)
  --cooldown <SECS>    Specify custom thermal cooldown in seconds (default: 180)
  --skip-parity        Skip Phase 1 (mathematical parity test)
  --skip-bench         Skip Phase 2 (Criterion ISA comparison bench)
  --out <FILE>         Destination path for receipt JSON (default: target/logs/remote-simd-receipt.json)
  --help, -h           Show this usage summary
```

### 6. Local SIMD Emulation via Intel SDE (`sde64`)

For developers working on baseline `x86-64-v3` workstations (e.g. AMD Zen 2/Zen 3 or Intel 10th/11th gen) without native AVX-512 hardware, the full cross-ISA mathematical parity suite can be executed locally via the **Intel Software Development Emulator (SDE)**:

1. **Install Intel SDE:**
   Download the Linux tarball and add to `PATH`:

   ```bash
   export PATH="/path/to/sde-external-...-lin:$PATH"
   ```

2. **Execute Full Mathematical Gating Suite via SDE:**

   ```bash
   ./utils/remote-simd-gate.sh --sde
   ```

   * *Phase 0:* Automatically detects the SDE runner and acknowledges emulated `avx512f` + `avx512vl` + `avx512bw` + `avx512dq`.
   * *Phase 1:* Executes all 12+ cross-ISA mathematical parity test cases (WaveNet Standard/Feather/Nano, A2 Full/Lite, and LSTM 1x16 / 2x8) with `--include-ignored`.
   * *Phases 2 & 3:* Skips hardware Criterion microbenchmarks and thermal cooldowns, as software JIT emulation does not measure physical silicon clock cycles.

3. **Direct Cargo Target Runner Integration:**

   ```bash
   CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -spr --" cargo test --release --test parity isa_parity -- --ignored --nocapture
   ```

### 7. Audit Receipt Extraction & PR Attachment Procedure

Upon successful execution on real hardware, the harness invokes `nam_remote_simd_receipt` to evaluate statistical significance and emit the audit receipt:

1. **Receipt File Location:** `NeuralAmpModeler-rs/target/logs/remote-simd-receipt.json`.

2. **Display Formatted Summary Table:**

   ```bash
   cargo run --features testing --bin nam_remote_simd_receipt -- --table
   ```

3. **Pull Request & Audit Artifact Inclusion:**

   * Attach `target/logs/remote-simd-receipt.json` as an artifact on the release ticket or CI build.
   * Copy the formatted Markdown summary table into the PR description.
   * The receipt contains full cryptographic and environmental provenance: host CPU model, core counts, core frequency, Linux kernel version, rustc release, git commit hash, sample timing distributions, and two-tailed Welch's t-test p-values.

### 8. Statistical Decision Gate & Exit Codes

The gating suite evaluates performance using rigorous statistical thresholds:

* **Phase 0 (Hardware Preflight):** Verifies presence of the full AVX-512 capability matrix `avx512f` + `avx512vl` + `avx512bw` + `avx512dq` (or Intel SDE emulation) — the reachable kernels require all four sub-features (T2.1/F-ROB-03). If absent, exits cleanly with **code 2** (`Clean skip`).
* **Phase 1 (Mathematical Parity):** All monomorphized kernels must maintain exact mathematical parity against the baseline and f64 reference oracle (`isa_parity.rs`).
* **Phase 2 (Criterion Latency Sweeps):** Executes multi-sample inference benchmarks for block sizes $N=1, 8, 64$.
* **Phase 3 (Welch's t-test Gating):**
  * **Canonical 64-sample batch ($N=64$):** Must achieve $\ge 12.0\%$ speedup with $p < 0.05$ (two-tailed Welch's t-test) $\rightarrow$ **PASS (Exit Code 0)**.
  * **Small geometries ($N=1, 8$):** Must achieve $\ge 0.0\%$ speedup (smoke test: zero regression permitted) $\rightarrow$ **PASS (Exit Code 0)**.
  * **Deficit / Regression:** If speedup $< 12.0\%$ on $N=64$ or any regression is detected on $N=1, 8$, the script exits with **code 1** (`Gate violation`), triggering the post-measurement architectural decision tree.

---

## SIMD Multiversioning ROI & Dispatch Matrix

The `NeuralAmpModeler-rs` engine enforces a strict, empirically verified Return on Investment (ROI) policy for SIMD specialization. Duplicating mathematical routines for higher instruction set extensions (e.g. AVX-512 VL256 vs. baseline AVX2) introduces maintenance complexity and binary footprint; therefore, specialized kernels are merged into production dispatch only when empirical measurements on real hardware justify the investment.

### 1. Empirical Decision Boundaries (The 3-Tier Rule)

All specialization candidates are evaluated on end-to-end model execution (`NamModel::process()`) using 64-sample audio blocks @ 48 kHz:

| Speedup ($\Delta\%$) vs. AVX2 Baseline | Statistical Gate | Decision                       | Rationale                                                                                      |
|:-------------------------------------- |:---------------- |:------------------------------ |:---------------------------------------------------------------------------------------------- |
| **$\ge 12\%$**                         | $p < 0.05$       | **KEEP (Production Dispatch)** | Statistically significant throughput win that reduces RT audio CPU load meaningfully.          |
| **$< 5\%$**                            | Any              | **DROP (No Specialization)**   | Memory-bandwidth bound or already compiler-saturated; duplication overhead exceeds gain.       |
| **$5\% \le \Delta\% < 12\%$**          | $p < 0.05$       | **DROP / CONDITIONAL**         | Dropped unless significant reduction in real-time latency variance / N=1 jitter is documented. |

### 2. Domain & Model Topology ROI Matrix (Empirical Hardware Receipt)

> [!NOTE]
> **Empirical Hardware Measurement (2026-08-22 Remote Audit Receipt):** `NamModel::process()` on **Intel Xeon Platinum 8488C** (Sapphire Rapids, AWS EC2 `c7i.xlarge` 2c/4t, rustc 1.98.0, git `75ceac1` dirty). Cross-ISA f32 parity on the gate harness passed. Numbers below are from `target/logs-remoto/remote-simd-receipt.json` (also summarized in-tree as the operator copy). **Policy = Scenario 2 DROP** (do not promote). In default builds, `detect()` and `dispatch_simd!` unconditionally execute `Avx2Math`, with AVX-512 kernels cfg-gated out of default `.text`. A2-Full N=1 was +50% AVX-512; that does not pass the N=64 gate.

| Kernel / Domain                  | Target Model Family                                     | AVX2 Baseline (v3)     | AVX-512 Measured Latency       | Speedup $\Delta\%$   | Statistical Gate ($p$) | Verdict (policy) vs code status                                                             |
|:-------------------------------- |:------------------------------------------------------- |:---------------------- |:------------------------------ |:-------------------- |:---------------------- |:------------------------------------------------------------------------------------------- |
| **LSTM 2x16**                    | `LSTM_2x16_64samp_48kHz`                                | 14.77 µs (YMM FMA)     | 18.04 µs (VL256+ZMM mix)       | **−22.10%**          | $p < 0.0001$           | **DROP.** Default builds dispatch AVX2; AVX-512 arm is cfg-gated opt-in.                    |
| **LSTM 1x16**                    | `LSTM_1x16_64samp_48kHz`                                | 7.31 µs (YMM FMA)      | 10.40 µs (VL256+ZMM mix)       | **−42.20%**          | $p < 0.0001$           | **DROP.** Default builds dispatch AVX2; AVX-512 arm is cfg-gated opt-in. Small $H=16$ GEMV. |
| **A2-Full (CH=8)**               | `A2Full_CH8_64samp_48kHz`                               | 22.58 µs (YMM FMA)     | 29.74 µs (VL256)               | **−31.74%**          | $p < 0.0001$           | **DROP.** Default builds dispatch AVX2; AVX-512 arm is cfg-gated opt-in.                    |
| **A2-Lite (CH=3)**               | `A2Lite_CH3_64samp_48kHz`                               | 20.37 µs (YMM FMA)     | 21.43 µs (VL256)               | **−5.20%**           | $p < 0.0001$           | **DROP** (below 12%; at the $<5\%$ line). Default builds dispatch AVX2.                     |
| **WaveNet Standard (CH=16)**     | `WaveNet_Standard_CH16_64samp_48kHz`                    | 43.36 µs (YMM FMA)     | 44.22 µs (VL256+`dot_16x` ZMM) | **−1.98%**           | $p = 0.0121$           | **DROP.** Default builds dispatch AVX2; AVX-512 arm is cfg-gated opt-in.                    |
| **WaveNet `dot_8x`**             | WaveNet CH=8 Layers                                     | 10 YMM registers       | wraps AVX2                     | **< 3%**             | —                      | **AVX2 reuse in `Avx512Math` (true).**                                                      |
| **WaveNet `dot_16x`**            | WaveNet CH=16 Layers                                    | 2× YMM                 | dedicated `__m512`             | not separately gated | —                      | **Dedicated ZMM kernel** (`dot_product_16x_f32_avx512` under `--features avx512`).          |
| **Linear / Gain / Dither / Pan** | DSP Pipeline Stages                                     | Streaming FMA          | wraps AVX2                     | **< 2%**             | —                      | **AVX2 reuse (true)** for gain/dither/ramp.                                                 |
| **CabSim UPOLS FFT**             | CabSim IR Convolver                                     | Radix-4 / Radix-2 AVX2 | wraps AVX2                     | **< 3%**             | —                      | **AVX2 reuse (true)** for MAC/FFT butterflies. Stereo FIR has `convolve_*_avx512` opt-in.   |
| **Dynamic Topologies**           | `LstmModelDyn` `WaveNetModelDyn` `WaveNetA2Dyn` ConvNet | AVX2 only              | not selected                   | —                    | —                      | **AVX2 in default builds.** All dynamic and ConvNet paths dispatch `Avx2Math`.              |
| **Non-DSP Off-RT Paths**         | Loaders (`.namb`, `serde_json`, IR WAV, Alloc)          | Standard Rust / libc   | none                           | **< 1%**             | —                      | **NO DUPLICATION.** Off-RT, I/O-bound.                                                      |

### 3. Why ZMM 512-bit Loses in Small Geometries

In low-latency neural audio, network dimensions are compact ($C=3, 4, 8, 12, 16$). Utilizing full 512-bit ZMM registers (`__m512`) causes:

1. **Register Underutilization:** Padding 3 or 8 channels to 16 lanes introduces zero-masking overhead and false dependency tracking.
2. **Frequency Downclocking (License Throttling):** On Intel Skylake-SP and Ice Lake architectures, executing 512-bit ZMM instructions drops core turbo frequencies across all threads sharing the core.
3. **Register File Advantage in VL256:** AVX-512 VL256 provides access to all 32 vector registers (`YMM0`..`YMM31`) in 256-bit width, entirely eliminating stack register spilling in 4-gate GEMV and Conv1D without triggering frequency penalties.

---

## L1i Instruction Cache Budget & Code Size Analysis

Real-time audio callbacks execute within strict sub-millisecond windows (e.g. 1.33 ms for 64 samples @ 48 kHz). To prevent catastrophic latency spikes (*jitter*) caused by instruction cache misses (*i-cache thrashing*), the hot-path working set must fit comfortably within the Level 1 Instruction Cache (L1i).

### 1. Modern Microarchitecture L1i Budget

Across modern x86-64 processor microarchitectures, the Level 1 Instruction Cache is strictly bounded:

* **AMD Zen 3 / Zen 4 / Zen 5:** 32 KB per core (8-way associative, 64-byte lines).
* **Intel Golden Cove / Raptor Cove / Sapphire Rapids:** 32 KB per core (8-way associative, 64-byte lines).

### 2. Hot-Path Code Size Measurements (`.text` Section)

Static monomorphization via `dispatch_simd!` generates dedicated machine code per active ISA variant. The table below was produced by static `.text`-size analysis (`llvm-objdump` + binary symbol size):

> [!WARNING]
> **Not a measured performance claim (2026-08 audit).** These `.text` sizes
> are static analysis of one build; the **~10.22 KB combined working set and
> the <32% L1i headroom claims were never measured on AVX-512 hardware** —
> no receipt exists. Retained as design intent for the L1i budget defense,
> not as verified numbers.

| Function / Component                     | Monomorphized Instances | Compiled `.text` Size (AVX2) | Compiled `.text` Size (AVX-512) | Combined Working Set |
|:---------------------------------------- |:----------------------- |:---------------------------- |:------------------------------- |:-------------------- |
| `NamModel::process()` (Dispatch Table)   | 1                       | 0.42 KB                      | 0.42 KB                         | 0.42 KB              |
| `gemv_4gate_avx512vl` (LSTM)             | 1                       | 1.84 KB                      | 1.92 KB                         | 1.92 KB              |
| `dot_4x` (WaveNet A2 Conv1D)             | 1                       | 2.10 KB                      | 2.24 KB                         | 2.24 KB              |
| `accumulate_avx512` (WaveNet)            | 1                       | 1.45 KB                      | 1.58 KB                         | 1.58 KB              |
| `simd_tanh` / `simd_sigmoid` (Padé)      | 1                       | 0.88 KB                      | 0.94 KB                         | 0.94 KB              |
| DSP Pipeline (Input, Gate, Output)       | 1                       | 3.12 KB                      | 3.12 KB                         | 3.12 KB              |
| **Total Active Audio Callback Hot-Path** | —                       | **~9.81 KB**                 | **~10.22 KB**                   | **~10.22 KB**        |

### 3. Architectural Defenses Against L1i Cache Thrashing

1. **Collapsing `Avx512VnniBf16`:** Unifying the deprecated VNNI/BF16 dispatch branch into `Avx512Math` eliminated an entire 3rd monomorphized variant across all 23 static models, saving **~4.8 KB** of redundant code footprint in the `.text` segment.
2. **Selective Inlining (`#[inline(always)]` vs. `#[inline]`):** Only inner vector reduction and FMA step functions are aggressively inlined. Model loader setup, validation, and diagnostic error formatters are tagged `#[cold]` and `#[inline(never)]`, placing them in separate cold code pages.
3. **Headroom Invariant (unverified):** The "~10.22 KB / <32% of the 32 KB L1i" figures are static-analysis intent. The 2026-08 remote receipt measures `process()` latency, **not** L1i occupancy. Do not cite the KB numbers as measured.

---

## Non-Duplication Policy for Memory-Bound and Auto-Vectorized DSP

To prevent maintenance divergence and unnecessary binary growth, mathematical sub-routines that are memory-bandwidth bound or where LLVM already achieves peak efficiency are explicitly excluded from AVX-512 kernel duplication:

1. **Gain, Dither, Pan & Stereo Mixing:** Slices are streamed through L1/L2 cache; arithmetic density is $\le 1$ FLOP per 4 bytes loaded. AVX2 FMA instructions already saturate memory bus throughput; AVX-512 provides zero measurable speedup. Exclusively uses the unified `x86-64-v3` baseline.
2. **CabSim UPOLS Frequency Delay Line:** Complex multiplication and accumulation (`complex_mac_accumulate`) and FFT stages are memory-access dominated. The convolution engine relies exclusively on the unified `x86-64-v3` AVX2 baseline without specialized AVX-512 duplication.
3. **Dynamic Topology Handlers:** Rare or non-standard geometries are processed through unified dynamic loops with vectorized vector chunks and scalar tails, avoiding explosive combinatorial monomorphization and reusing the `x86-64-v3` baseline.
4. **Non-DSP Off-RT Operations (Loaders, Parsers, CRC32, Allocation):** File loading (`.nam`/`.namb`), JSON parsing (`serde_json`), CRC32 calculation, and buffer allocation occur exclusively off the real-time audio thread. They are bounded by disk I/O and memory throughput; manual SIMD specialization yields $< 1\%$ end-to-end impact and is explicitly rejected (*"Nenhum candidato"*).
