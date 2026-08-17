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
> Reference snapshot (Ryzen 7 5700U, AVX2): WaveNet Std CH16 ≈ 36.9 µs
> (2.8%, 2404 µs/MMAC), Feather CH8 ≈ 19.3 µs (1.4%, 5031 µs/MMAC), Lite CH12 ≈ 52.2 µs (3.9%, 6039 µs/MMAC),
> Nano CH4 ≈ 17.2 µs (1.3%, 17969 µs/MMAC outlier due to layer overhead),
> A2-Full CH8 ≈ 27.3 µs (2.0%), A2-Lite CH3 ≈ 18.4 µs (1.4%), LSTM 1×16 ≈ 7.5 µs (0.6%),
> LSTM 2×8 ≈ 7.5 µs (0.6%), ConvNet ≈ 10.2 µs (0.8%), Linear RF=2048 ≈ 0.3 µs (0.0%).
> All ≤ 3.9% of the 1333 µs RT budget (64 samples @ 48 kHz). The "Experiment Report"
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
2. **Statistical rigor** — The `regression_gate` bench suite runs **19** targets (10 static models + 4 dynamic models + 5 DSP infrastructure benches) with `sample_size=100, measurement_time=5s, warm_up_time=1s, noise_threshold=0.02`. Dispatch is forced to `InstructionSet::Avx2` via `ForceAvx2Guard` so hosts with AVX-512 still measure the x86-64-v3 contract path.
3. **Baseline comparison** — Criterion performs a two-sample t-test between the current run and the stored baseline. If it detects a statistically significant regression (p < 0.05 **and** outside the 2% noise band), the script exits with code 1.
4. **Baseline storage** — Baselines are persisted under **`.performance-baselines/`** (repo-local, gitignored). `target/criterion/` is only a **transient** Criterion working area restored from `.performance-baselines/` before each run. Persist/restore use **replace-copy of top-level** `…/<bench>/ci-baseline/` only; nested `ci-baseline/ci-baseline/…` paths (historical `cp -a` into an existing dest) are sanitized and never re-copied. An environment fingerprint (`.performance-baselines/baseline-fingerprint.json`) records CPU model, full x86-64-v3 ISA label (`AVX2/FMA/F16C/BMI`, including LZCNT via `lzcnt` or Linux `abm`), rustc, target triple, governor, bench core, and producing commit.
5. **Immutability** — `--check` is strictly read-only. It never auto-creates a baseline. If no baseline exists, it fails with `MISSING_BASELINE` and exit code 1, directing the operator to run `--bootstrap-baseline` manually.
6. **Sub-µs DSP micro-benches** — `RT_DSP_Resampler_*` and `RT_DSP_CabSim_IR_Medium` process a fixed batch of **64 blocks** per Criterion sample so timer noise stays under the 2% wall. The quality dashboard divides those medians by 64 and reports **per-block** latency (contract units unchanged). Changing the batch size requires a human baseline renewal.

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

| Tool                                                                                | Role                                                                                                                                                                                              |
|:----------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/rt_constraints/rt_deadline.rs`                                               | **Absolute hard gate** — `assert!(p99 < 1330 μs)` for all SKUs. This is the pass/fail ceiling.                                                                                                    |
| [`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) | **Relative guard, baseline-gated** — the canonical home for perf-regression benchmarking. Catches degradations *within* the safe zone (e.g., 100 μs → 150 μs, still under 1.33 ms but 50% worse). |
| [`utils/tests-long.sh`](../utils/tests-long.sh)                                     | **Nightly Audit Suite** — Focuses on heavy functional, soak, parity, and RT-safety tests; benchmarks are omitted from the nightly runner to optimize execution time.                              |
| [`utils/tests-quick.sh`](../utils/tests-quick.sh)                                   | Fast path (~3 min) — does **not** include benchmarks (would exceed the time budget). Use `utils/tests-performance-regression.sh` directly for perf checks.                                        |

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
