<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Performance Benchmarks (Criterion)

NeuralAmpModeler-rs uses **Criterion.rs** as its official statistical performance benchmarking suite. In a real-time digital signal processing (DSP) engine, execution latency and timing determinism must be measured with statistical rigor to isolate algorithmic throughput from operating system noise, scheduling jitter, and dynamic frequency scaling.

> [!NOTE]
> **Document Scope.** This document is the authoritative reference for performance benchmarking in NeuralAmpModeler-rs: how to run and interpret benchmarks, the performance regression gate ([`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh)), contract performance baselines ([`docs/quality-contract.json`](quality-contract.json)), and micro-architectural invariants.
> Functional correctness, mathematical oracle hierarchies, and test suites are documented separately in [`testing.md`](testing.md).

---

## 1. Running the Benchmarks

All benchmark commands must be executed within the `NeuralAmpModeler-rs/` subproject root:

```bash
# Core inference benchmark suite
cargo bench --bench inference_bench

# Performance regression gate suite (20 canonical targets)
cargo bench --bench regression_gate

# Specialized DSP, Math & Kernel benchmark suites
cargo bench --bench cabsim_bench
cargo bench --bench dsp_bench
cargo bench --bench dsp_bridge_bench
cargo bench --bench math_bench
cargo bench --bench spsc_swap_bench
cargo bench --bench gemv_bench
cargo bench --bench head_gemv_bench
cargo bench --bench linear
cargo bench --bench dot_4x_bench
cargo bench --bench fft_radix4_bench
cargo bench --bench kahan_conv1d_bench
cargo bench --bench conv1d_hotpath_bench
```

### Audio Block Budgets & Soak Benchmarks

* **Standard Audio Block:** Default benchmarks operate on blocks of **64 samples at 48 kHz** ($\approx 1.333\text{ ms}$ / $1333\text{ µs}$). This is the primary real-time processing deadline.
* **Block-Size Scaling IDs (`*_{N}samp_48kHz`):** The `inference_bench` scaling targets process **$N$ frames per Criterion iteration** (one block). The A2 scaling benches call `set_max_buffer_size($N$)` before timing, because A2's `process` clamps the frame count to the negotiated buffer size (constructor default = 64); without it, `128/256/512` would silently measure only 64 frames and the curve would appear flat. WaveNet/LSTM scaling benches have no such clamp and process the full buffer directly.
* **Long-Duration Soak Benchmarks:** To measure sustained memory throughput and evaluate cache/TLB stability under continuous load, long-duration benchmarks use blocks of **4096 samples** (~85 ms):

```bash
cargo bench --features testing --bench long_inference_bench
```

---

## 2. Interpreting Criterion Output

When executing a benchmark, Criterion reports output in this standard format:

```text
WaveNet_Standard_CH16_64samp_48kHz
                        time:   [43.15 µs 43.29 µs 43.45 µs]
                        change: [−1.85% −0.92% +0.12%] (p = 0.11 > 0.05)
                        Change within noise threshold.
                        Found 3 outliers among 100 measurements (3.00%)
```

### Metrics & Statistical Criteria

1. **Confidence Interval (`time: [A B C]`):**
   * Expresses execution time per iteration across a **95% confidence interval**.
   * The central value (`B`) is the bootstrapped point estimate of the mean.
   * Bounds (`A` and `C`) define the margin within which true mean performance lies with 95% certainty.
2. **Relative Change & Statistical Significance (`change: [...] (p = X)`):**
   * Shows the percentage delta against the saved baseline on the same machine (negative values represent speedup).
   * The **p-value** ($p$) measures the probability that the observed change was accidental. If $p < 0.05$, Criterion considers the variation statistically significant rather than background noise.
3. **Outliers & Jitter:**
   * High-severity outliers in real-time DSP often indicate CPU frequency throttling, OS thread preemption, or cache-line invalidation. Core pinning and setting the CPU governor to `performance` mitigate these anomalies.
4. **Baselines Storage:**
   * Criterion stores transient working data in `target/criterion/`.
   * The regression gate persists and restores authoritative baselines to `.performance-baselines/`.

---

## 3. Reference Latency Snapshot (Quality Contract)

The single canonical source of truth for committed performance baselines is [`docs/quality-contract.json`](quality-contract.json).

Below is the committed reference baseline measured on release builds under an isolated x86-64-v3 environment (AMD Ryzen 7 5700U, AVX2/FMA, 64-sample blocks @ 48 kHz):

| Target ID                      | Description                            | Median Latency    | % of 1.33 ms RT Budget |
|:------------------------------ |:-------------------------------------- |:----------------- |:---------------------- |
| `RT_WaveNet_Std_CH16`          | WaveNet Standard (CH=16)               | 41.77 µs          | 3.13%                  |
| `RT_WaveNet_Feather_CH8`       | WaveNet Feather (CH=8)                 | 20.11 µs          | 1.51%                  |
| `RT_WaveNet_Lite_CH12`         | WaveNet Lite (CH=12, padded-16)        | 57.16 µs          | 4.29%                  |
| `RT_WaveNet_Nano_CH4`          | WaveNet Nano (CH=4)                    | 18.29 µs          | 1.37%                  |
| `RT_A2_Full_CH8`               | A2-Full (CH=8, col-major SIMD)         | 19.75 µs          | 1.48%                  |
| `RT_A2_Lite_CH3`               | A2-Lite (CH=3, unrolled GEMV)          | 14.00 µs          | 1.05%                  |
| `RT_LSTM_1x16`                 | LSTM 1×16 (fused SIMD gates)           | 6.77 µs           | 0.51%                  |
| `RT_LSTM_2x8`                  | LSTM 2×8 (fused SIMD gates)            | 7.58 µs           | 0.57%                  |
| `RT_Linear_Direct_RF4`         | Linear Direct FIR (RF=4, time-domain)  | 0.13 µs           | 0.01%                  |
| `RT_Linear_Fft_RF2048`         | Linear Partitioned FFT (RF=2048)       | 4.93 µs           | 0.37%                  |
| `RT_ConvNet`                   | ConvNet (CH=8, 6 blocks)               | 9.48 µs           | 0.71%                  |
| `RT_WaveNet_Dyn_Free`          | WaveNet Dynamic Free-Shape             | 21.57 µs          | 1.62%                  |
| `RT_LSTM_Dyn_1x7`              | LSTM Dynamic 1×7                       | 8.24 µs           | 0.62%                  |
| `RT_A2_Dyn_Gated_CH8`          | A2 Dynamic Gated (CH=8)                | 177.12 µs         | 13.28%                 |
| `RT_A2_Dyn_Blended_CH3`        | A2 Dynamic Blended (CH=3)              | 131.09 µs         | 9.83%                  |
| `RT_A2_Max_CH4`                | A2 Max (CH=4, cond=8, FiLM)            | 190.06 µs         | 14.25%                 |
| `RT_DSP_Resampler_44k1_to_48k` | Polyphase Resampler 44.1k $\to$ 48k    | 1.32 µs / block   | 0.10%                  |
| `RT_DSP_Resampler_96k_to_48k`  | Polyphase Resampler 96k $\to$ 48k      | 0.66 µs / block   | 0.05%                  |
| `RT_DSP_CabSim_IR_Medium`      | CabSim UPOLS (2048 taps)               | 1.29 µs / block   | 0.10%                  |
| `RT_DSP_Pipeline_Base_NoOS`    | End-to-end Pipeline (Standard, 1×)     | 42.80 µs / block  | 3.21%                  |
| `RT_DSP_Pipeline_HQ_4xOS`      | End-to-end Pipeline (Standard, 4× OS)  | 174.36 µs / block | 13.08%                 |
| `RT_DSP_Pipeline_AdaptiveOn`   | End-to-end Pipeline (Adaptive Dynamic) | 64.29 µs / block  | 4.82%                  |

*Unit note: every row is **µs per 64-sample block** (the 1.333 ms RT deadline). The `RT_DSP_Resampler_*` and `RT_DSP_CabSim_IR_Medium` Criterion ids batch `DSP_MICRO_BATCH = 64` blocks per iteration and are divided by 64 before reporting (see §4.5); the `RT_DSP_Pipeline_*` ids already process a single 64-sample block per iteration and are reported as measured. The quality dashboard renders the same convention.*

*Note: All single-model inferences consume $\le 4.3\%$ of the real-time block budget; even dynamic A2 topologies and 4× oversampled pipelines maintain $> 85\%$ real-time headroom.*

---

## 4. Regression Gate — Automated Performance Defense

[`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) is the canonical benchmark-based performance defense tool. It compares current timings against a persisted statistical baseline and fails closed if latency degrades beyond the allowed threshold.

### Core Mechanisms

1. **CPU Core Pinning:** Pins benchmark execution to a dedicated core via `taskset -c <core>`. The core is resolved automatically via `pick_bench_core` using the 3-step precedence: (1) `NAM_BENCH_CORE` if explicitly set; (2) first online isolated CPU from `/sys/devices/system/cpu/isolated` (preferring core 8 when 8-9 are isolated, with cpu9 offline intentional); (3) fallback `nproc / 2` (integer division). Tested synthetically with mock sysfs via [`utils/test-pick-bench-core.sh`](../utils/test-pick-bench-core.sh) and drift-pinned by [`utils/test-lib-drift.sh`](../utils/test-lib-drift.sh). This prevents OS scheduler migrations and cache thrashing.
2. **Controlled Statistical Rigor:** Executes the `regression_gate` suite with `sample_size=100, measurement_time=5s, warm_up_time=1s, noise_threshold=0.05`. Enforces `InstructionSet::Avx2` via `ForceAvx2Guard` so that AVX-512 hosts evaluate the exact `x86-64-v3` baseline.
3. **Machine Verdict (`nam_perf_gate verdict`):** Evaluates Criterion's `target/criterion/<id>/change/estimates.json`. If the bootstrapped mean-change confidence interval lies entirely above the `+5%` noise band, the script exits with `REGRESSION_DETECTED`. If comparison artifacts are missing or unreadable, it fails closed with `REGRESSION_BLIND`.
4. **Baseline Storage & Fingerprinting:** Authoritative baselines reside in `.performance-baselines/` (gitignored). A machine fingerprint file (`baseline-fingerprint.json`) captures the CPU model, ISA extension flags, compiler version, target triple, governor, pinned core, and producing git commit. Managed by [`src/testing/qa/baseline_store.rs`](../src/testing/qa/baseline_store.rs) (`nam_perf_gate restore-baseline`) with recursive sanitization and replace-copy.
5. **Sub-Microsecond Batching:** Ultra-fast DSP routines (`RT_DSP_Resampler_*` and `RT_DSP_CabSim_IR_Medium`) process batches of **64 blocks** per Criterion sample to maintain timer resolution above the noise floor. Reported values are divided by 64 to represent per-block latency.

### Script Modes & Environment Variables

| Mode                | Command                                                      | Behavior                                                                                                                  |
|:------------------- |:------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------- |
| **Check** (default) | `utils/tests-performance-regression.sh --check`              | Read-only. Compares against `.performance-baselines/`. Exits non-zero if a regression is detected or baseline is missing. |
| **Bootstrap**       | `utils/tests-performance-regression.sh --bootstrap-baseline` | Re-generates `.performance-baselines/` and writes `baseline-fingerprint.json`. **Human-only operation.**                  |

| Variable                 | Default                                     | Purpose                                                             |
|:------------------------ |:------------------------------------------- |:------------------------------------------------------------------- |
| `NAM_BENCH_CORE`         | `pick_bench_core` (isolated or `nproc / 2`) | Dedicated CPU core to pin benchmarks via `taskset`.                 |
| `NAM_BASELINE_NAME`      | `ci-baseline`                               | Name of the Criterion baseline series.                              |
| `NAM_BENCH_SUITE`        | `regression_gate`                           | Benchmark binary to drive (e.g. `spsc_swap_bench`).                 |
| `NAM_BENCH_MANIFEST`     | *(empty)*                                   | Expected-ID manifest for a non-`regression_gate` suite (see below). |
| `NAM_THERMAL_COOLDOWN_S` | `180`                                       | Cooldown period before benchmarking to stabilize clock frequencies. |

### Downstream Adoption (External Suites)

`nam_perf_gate` is the only component that interprets performance-governance JSON:
fingerprint, coverage, machine verdict, baseline persistence and receipts are all
serde — the shell wrapper only orchestrates `taskset`/`cargo bench`. A different
repository (a host, a plugin, any Criterion suite) reuses the **same binary** and
never duplicates the logic, and the crate is not coupled to any consumer name.

1. **Install (route a — recommended).** The crate publishes `/src/bin/`, so the
   gate binary installs from crates.io pinned to the engine version:

   ```bash
   cargo install NeuralAmpModeler-rs --features testing --bin nam_perf_gate --version =1.0.0
   ```

   Alternatively, point `NAM_PERF_GATE` at any pre-built `nam_perf_gate`
   executable (an operator-provided binary skips the freshness build and is
   used as-is). The downstream wrapper never reaches into the engine source
   tree.

2. **Horizontal coverage (any suite).** `coverage`, `expected-count` and
   `verdict` accept `--suite <name> --manifest <file>`: the manifest is a plain
   text file with one expected Criterion benchmark id per line (blank lines and
   `#` comments ignored). A trailing `Group/*` entry matches any executed id with
   that prefix, so a suite can declare whole Criterion groups without enumerating
   the parametric ids; every other entry is an exact id. A `Group/*` entry
   certifies only that the group **ran** (at least one matching id executed), not
   that every parametric id inside it did — use exact ids whenever exhaustive
   per-id coverage is the requirement. Independently of the manifest, any
   executed id that lacks a saved baseline series still fails closed. This lets an
   **arbitrary** suite — not just the engine's `regression_gate` — be verified for
   coverage and evaluated by the machine verdict, without referencing any
   engine-internal benchmark name:

   ```bash
   nam_perf_gate coverage --log bench.log --suite custom_bench --manifest ids.txt
   nam_perf_gate expected-count --suite custom_bench --manifest ids.txt
   nam_perf_gate verdict  --log bench.log --suite custom_bench --manifest ids.txt
   ```

   Omitting `--manifest` for a non-`regression_gate` suite fails closed (exit 1):
   a suite with no declared expectation can never certify coverage.

3. **Stable exit codes.** Every run-time failure is exit `1` with a typed token on
   stdout, so a downstream wrapper can branch without parsing prose:

   | Token                      | Meaning                                                        |
   |:-------------------------- |:-------------------------------------------------------------- |
   | `MISSING_BASELINE`         | No persisted fingerprint/baseline for the current environment. |
   | `INCOMPARABLE_ENVIRONMENT` | Fingerprint drifted from the persisted baseline.               |
   | `BASELINE_COVERAGE_GAP`    | Executed/expected benchmark without a saved baseline series.   |
   | `REGRESSION_DETECTED`      | Mean-change CI entirely above the noise band.                  |
   | `REGRESSION_BLIND`         | Missing comparison artifact or unexecuted expected id.         |

   Usage errors (unknown subcommand/flag, malformed argument) are exit `2`.

### Regression Guard Automated Suite (`tests/scripts/test_regression_guard.sh`)

The integrity and fail-closed defense mechanisms of the performance regression wall and baseline storage engine are continuously validated by [`tests/scripts/test_regression_guard.sh`](../tests/scripts/test_regression_guard.sh):

```bash
# Execute performance regression guard suite:
bash tests/scripts/test_regression_guard.sh
```

The suite exercises four critical failure modes:

1. **Scenario 1 — Missing Baseline Gate:** Moves `.performance-baselines/` aside and runs `utils/tests-performance-regression.sh --check`. Asserts that the runner fails closed (`exit 1`) and emits `MISSING_BASELINE`.
2. **Scenario 2 — Contract Format Rejection & Domain Separation:**
   * **2a:** Rejects legacy ASCII `.txt` contracts (`docs/quality-contract.txt`) with `ERROR` and `exit 2`.
   * **2b:** Uses the frozen report fixture `tests/fixtures/qa/report_gate_not_verified.jsonl` with `nam_quality verify` to assert that when benchmark regression data is `NOT_VERIFIED`, the system reports `FIDELITY: OK`, `PERFORMANCE: NOT_VERIFIED`, and `CONTRACT VIOLATED` (`exit 1`), proving PERF-006 domain separation.
3. **Scenario 3 — Stale Log Truncation:** Asserts that `target/logs/regression-check.log` is truncated immediately at the start of `check_regression()`, preventing historical entries from masquerading as current results.
4. **Scenario 4 — Baseline Store Sanitization & Replace-Copy:** Synthesizes deeply nested baseline directories (`RT_Dummy/ci-baseline/ci-baseline/...`) and exercises `nam_perf_gate restore-baseline` ([`src/testing/qa/baseline_store.rs`](../src/testing/qa/baseline_store.rs)). Verifies that nested paths are sanitized and replace-copied into a single flat baseline layer without recursive nesting.

### Environmental Noise, False Alarms & Hardware Isolation

Because the DSP engine operates in a state of extreme micro-architectural optimization, latency measurements are highly sensitive to thermal throttling, DVFS transitions, and background operating system jitter.

> [!WARNING]
> **Risk of False Alarms:** In sub-microsecond and microsecond-scale benchmarks (e.g. `RT_Linear_Direct_RF4` at ~340 ns or `RT_LSTM_2x8` at ~7.1 µs), small environmental variations, CPU thermal drift, or concurrent OS tasks can trigger false-positive regression alarms ($+2–4\%$ variation). Exercise caution and do not hastily bootstrap new baselines or revert code without verifying that the result is reproducible under isolated conditions.

#### Recommended Mitigation Measures

1. **Kernel-Level Core Isolation (`isolcpus`, `nohz_full`, `rcu_nocbs`):**
   To shield benchmark cores from kernel scheduling ticks, timers, interrupts, and non-bound user space threads, configure the Linux kernel boot line (in `/etc/default/grub` via `GRUB_CMDLINE_LINUX_DEFAULT`) with maximum isolation parameters (example for dedicated core 8 and sibling 9 on a 16-thread host):

   ```text
   isolcpus=managed_irq,domain,8,9 nohz_full=8,9 rcu_nocbs=8,9 irqaffinity=0-7,10-15 nmi_watchdog=0 skew_tick=1
   ```

   * `isolcpus=managed_irq,domain,8,9`: Excludes cores 8 and 9 from OS load-balancing and managed IRQ affinity.
   * `nohz_full=8,9`: Suspends the scheduler tick while a single benchmark task runs.
   * `rcu_nocbs=8,9`: Offloads RCU callbacks to housekeeping cores (0-7, 10-15).
   * `irqaffinity=0-7,10-15`: Directs all hardware device interrupts to housekeeping cores.
   * `nmi_watchdog=0` & `skew_tick=1`: Eliminates NMI watchdog timer jitter and offsets tick alignment.

2. **Physical Core Exclusivity (Disable SMT Sibling):**
   In SMT (HyperThreading) topologies, taking the sibling logical thread offline gives the benchmark core exclusive access to L1/L2 caches and execution pipelines:

   ```bash
   echo 0 | sudo tee /sys/devices/system/cpu/cpu9/online
   ```

3. **CPU Frequency Scaling & Boost Locking:**
   Lock the CPU scaling governor to `performance` across all cores and disable boost/turbo to eliminate DVFS transition delays and prevent thermal clock throttling from creating false-positive variance:

   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
   ```

4. **Hardware IRQ Masking & PMU Paranoid:**
   Route hardware IRQs away from cores 8 and 9 and enable PMU sampling:

   ```bash
   echo fcff | sudo tee /proc/irq/default_smp_affinity
   for irq in /proc/irq/[0-9]*/smp_affinity_list; do
       [ -f "$irq" ] && echo 0-7,10-15 | sudo tee "$irq" >/dev/null 2>&1 || true
   done
   echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid
   ```

5. **Benchmark Execution under SCHED_FIFO Pinning:**
   Execute the benchmark runner pinned strictly to the isolated core under realtime FIFO scheduling:

   ```bash
   NAM_BENCH_CORE=8 chrt -f 80 taskset -c 8 utils/tests-performance-regression.sh --check
   ```

6. **Thermal Stabilization & Hardware Cooldown:**
   Allow the CPU to cool down before executing baseline benchmarks (`NAM_THERMAL_COOLDOWN_S=180` by default). The regression gate sleeps between heavy phases to prevent thermal throttling from skewing Criterion iteration times.

7. **Low Background Load & Host Quiescence:**
   Ensure workstation background load is minimal ($\text{load average} \ll 1.0$), with browsers, container engines, and heavy background services closed.

#### Reference Noise Floor & Run-to-Run Variance (Empirical Baseline Audit)

During the empirical contract bisection audit, consecutive back-to-back executions of the canonical `regression_gate` under core pinning (`taskset -c 8`) and thermal cooldown (`NAM_THERMAL_COOLDOWN_S=180`) established the empirical reference noise floor:

| Benchmark Target               | Run 1 Median | Run 2 Median | Observed Variance (Δ) | Typical Latency Scale            |
|:------------------------------ |:------------:|:------------:|:---------------------:|:--------------------------------:|
| `RT_Linear_Direct_RF4`         | 0.137 µs     | 0.132 µs     | **−3.8%**             | Sub-microsecond (~130 ns)        |
| `RT_A2_Dyn_Blended_CH3`        | 132.9 µs     | 136.0 µs     | **+2.3%**             | Large complex model (~136 µs)    |
| `RT_ConvNet`                   | 9.40 µs      | 9.37 µs      | **−0.3%**             | Medium network (~9.4 µs)         |
| `RT_WaveNet_Std_CH16`          | 41.60 µs     | 41.46 µs     | **−0.3%**             | Standard WaveNet (~41.5 µs)      |
| `RT_DSP_Resampler_44k1_to_48k` | 1.30 µs      | 1.31 µs      | **+0.8%**             | Ultra-fast DSP routine (~1.3 µs) |

**Key Invariants & Implications for Quality Contracts:**

1. **Empirical Noise Band:** Even under host core pinning and performance governors, environmental noise spans approximately **−3.8% to +2.3%** on complex pipelines, widening up to $\pm 5\%$ on sub-microsecond primitives (`RT_Linear_Direct_RF4`).
2. **Minimum Safety Margin:** To prevent false-positive regression alarms in CI and automated gates, contract tolerance envelopes must maintain a minimum margin of **$\ge 5\%$** (`latency_mult \ge 1.05` to `1.10` and `latency_floor_us \ge 0.05`). Any tolerance band tighter than $\pm 4\%$ will trigger sporadic false alarms caused purely by OS timer granularity and thermal jitter.

### Pre-Flight Checklist

Run this verification before executing `--check` or `--bootstrap-baseline`:

* [ ] **Baseline present:** `.performance-baselines/baseline-fingerprint.json` exists.
* [ ] **Core isolation active:** Preferred cores isolated via kernel command line (`isolcpus=8,9 nohz_full=8,9`) when available.
* [ ] **CPU governor locked:** Pinned core shows `performance` in `/sys/devices/system/cpu/cpu<N>/cpufreq/scaling_governor`.
* [ ] **Low background load:** Workstation load average $\ll 1.0$; heavy background processes and browsers closed.
* [ ] **Coverage completeness:** Every benchmark target in `regression_gate` has an existing baseline entry.

### First-Time Setup and Post-Optimization Renewal (Human-Only)

> [!CAUTION]
> AI agents and automated CI runners are **strictly prohibited** from executing `--bootstrap-baseline` or `quality-dashboard.sh --save`. Baseline renewal is exclusively a human operation performed under verified environmental conditions.

When an intentional algorithmic optimization or architectural change legitimately shifts latency:

```bash
# 1. Regenerate the local Criterion baseline
utils/tests-performance-regression.sh --bootstrap-baseline

# 2. Verify that standalone check passes cleanly
utils/tests-performance-regression.sh --check

# 3. Update the committed quality contract (requires all tests & gates to PASS)
utils/quality-dashboard.sh --save docs/quality-contract.json

# 4. Validate the loop on the same git revision
utils/quality-dashboard.sh --check docs/quality-contract.json
```

> [!WARNING]
> **S5-T3 (F-12) — `--check` after `--bootstrap-baseline` proves only
> self-consistency.** Step 2 above compares the tree against a baseline just
> written from that same tree in step 1: a green result means "repeatable",
> not "no regression". Reserve the "REGRESSION GATE" verdict for a `--check`
> against a baseline produced in another session/commit. Effective noise band:
> `noise_threshold = 0.05` (whole 95% CI above +5% is a machine regression;
> see `src/testing/qa/verdict.rs`). `bootstrap-repo.sh` prints this warning
> whenever it runs the bootstrap→check pair.

**Flaky dashboard `--check` after a green standalone gate:**
The quality dashboard runs several minutes of functional fidelity checks prior to executing Criterion. This pre-test workload increases CPU die temperatures and OS noise. Very fast micro-benchmarks near the noise floor (`RT_Linear_Direct_RF4` ~340 ns, `RT_LSTM_2x8` ~7.1 µs) can report a spurious $+2–4\%$ variation. If standalone `--check` passes on a cool machine, do not bootstrap a new baseline solely to silence transient thermal drift.

---

## 5. Quality Contract — Performance Integration

The **Quality Contract** ([`quality-contract.json`](quality-contract.json)) integrates fidelity and performance into a single machine-readable specification:

* **Regression Gate (`utils/tests-performance-regression.sh`):** Strict relative statistical wall evaluating whether mean latency regressed above the $+5\%$ noise band.
* **Dashboard Check (`utils/quality-dashboard.sh --check`):** Broad integration check applying a conservative **10% margin** on median latency:

$$\text{measured\_latency} > \text{contract\_latency} \times 1.10 \implies \text{VIOLATION}$$

This 10% envelope absorbs transient thermal fluctuations while strictly catching meaningful degradations.

### Bench-Label Mapping

Benchmark identifiers in `quality-contract.json` correspond directly to Criterion bench function labels in [`benches/regression_gate.rs`](../benches/regression_gate.rs). The mapping is enforced as an identity projection via `RT_BENCH_TABLE` in [`src/testing/qa/ids.rs`](../src/testing/qa/ids.rs) and validated by the `rt_table_contract_ids_match_committed_contract` regression test.

---

## 6. Micro-Architectural Decisions & Invariants

This section records empirical performance findings and micro-architectural invariants to prevent regressions caused by well-intentioned but counterproductive refactorings.

### Temporal Tiling (Dual-Frame) on Conv1D

Dilated 1D convolution (`Conv1D`) accounts for ~45% of WaveNet inference time. The relationship between temporal tiling (processing two audio frames simultaneously) and latency differs fundamentally between static and dynamic implementations:

1. **Static WaveNet (`WaveNetLayer`): Single-Frame Strictly Preserved**
   * *Finding:* Dual-frame tiling requires doubling SIMD accumulators (from 4 YMM to 8 YMM per channel). On x86-64-v3, this creates severe register pressure, leading to stack spilling and port contention on shuffle/blend units (Port 5), causing a **~19% latency regression** (~92.6 µs $\to$ ~110 µs for CH=16).
   * *Invariant:* The static processing loop in `src/models/wavenet/layer.rs::process_block_internal` uses **Single-Frame processing exclusively**.
2. **Dynamic WaveNet (`Conv1dDyn`): Dual-Frame Tiling Retained**
   * *Finding:* `Conv1dDyn::process_block` uses runtime-dimensioned channel counts and dynamic tap pointers. Dual-frame tiling amortizes per-frame fixed overhead (tap-pointer array setup and prefetch dispatch) across frame pairs, yielding a **5% to 13% speedup** across CH=4, 8, and 16.
   * *Invariant:* Dual-frame processing is retained as the primary loop in dynamic convolution (`chunks_exact_mut(2 * out_ch)`), falling back to single-frame only for odd frame remainders.

### WaveNet Hot-Path Cycle Budget & Kernel Fusion

Hardware cycle counter (RDTSC) analysis of `WaveNetLayer::process_block_internal` on AVX2 reveals the hot-path execution breakdown:

| Stage              | Operations                            | Cycle Share | Architectural Characteristic                      |
|:------------------ |:------------------------------------- |:----------- |:------------------------------------------------- |
| **Conv1D (GEMV)**  | Causal convolution, MACs, dilation    | **~45%**    | Primary compute bottleneck (FMA bound).           |
| **1×1 & Residual** | Channel projection, residual addition | **~25%**    | Memory pressure and read-modify-write.            |
| **Mixin**          | Conditioning metadata projection      | **~15%**    | Broadcast and dense channel addition.             |
| **Act & Head**     | Tanh / Sigmoid, skip accumulation     | **~15%**    | Padé transcendental approximation + accumulation. |

* **Activation & Skip Fusion:** Fusing the Tanh activation with Head skip-connection accumulation halved the activation stage budget (from ~30% to ~15%) by keeping intermediate state in YMM registers, avoiding redundant L1 cache round-trips.
* **Stereo Output Fusion:** Fusing L/R channel gain and noise gate hysteresis in the output stage reduces memory bandwidth and yields an end-to-end latency reduction of **~4.5% to 5.5%**.

### RFFT Staged Architecture: Scalar Interleaving vs. SIMD

The Real Fast Fourier Transform (`RfftPlanner` in `src/math/dsp/rfft.rs`) handles frequency-domain convolution in partitioned linear models and CabSim UPOLS:

* **Staged Architecture:** Transforms are divided into `pack_re_im`, twiddles, and `unpack_re_im`.
* **Scalar Packing Retained:** Although isolated AVX2 shuffle kernels (`pack_re_im_f32_avx2`) achieve speedups in standalone micro-benchmarks at small $N$, pack/unpack represents only 2–5% of total transform execution. In end-to-end convolution (`ConvEngine` in CabSim), scalar packing performs identically or slightly faster ($-2.5\%$, $p = 0.006$) while avoiding code-layout disruption and SIMD register pressure.
* **Separation of Concerns:** `RT_Linear_Direct_RF4` benchmarks time-domain FIR convolution (RF=4, `LinearMode::Direct`) and never touches RFFT. Partitioned FFT convolution is benchmarked independently via `RT_Linear_Fft_RF2048`.

### Linear Multichannel Architectures: Throughput, In-Place Safety, and Direct vs. FFT Crossover

The Linear model family supports multichannel FIR convolution across $1 \to N$ (`LinearOneToMany`), $N \to 1$ (`LinearManyToOne`), and $N \to N$ shared (`LinearManyToManyShared`) geometries ([`benches/linear.rs`](../benches/linear.rs)).

1. **Strict Zero-Regression on Legacy Mono ($1 \to 1$):**

   * Legacy mono models build with `multichannel = None`, preserving the original branch-free, time-domain and partitioned FFT inner loops.
   * Under regression gate auditing (`regression_gate`), `RT_Linear_Direct_RF4` executes in **~134 ns** (well beneath the 330 ns contract threshold), and `RT_Linear_Fft_RF2048` executes in **~4.85 µs** (statistically identical to the 4.84 µs baseline, delta = 0.0%).

2. **Multichannel Allocation Isolation & Throughput Scaling:**

   * Audio callback routines invoke `process_raw(*const *const f32, *const *mut f32, usize)` or `process_multichannel` using stack-allocated pointer arrays (`[*const f32; 16]`, `[*mut f32; 16]`), guaranteeing **zero heap allocations** and zero GC jitter.

   * At 64 samples @ 48 kHz (1.33 ms deadline), latency scales linearly with active convolution operations:

     | Topology Geometry              | Receptive Field (Taps) | Direct Latency (64 samp) | FFT Latency (64 samp) | % of 1.33 ms RT Budget (FFT) |
     |:------------------------------ |:---------------------- |:------------------------ |:--------------------- |:---------------------------- |
     | **Mono $1 \to 1$**             | RF = 2048              | 8.86 µs                  | 4.85 µs               | 0.36%                        |
     | **OneToMany $1 \to 2$**        | RF = 2048              | 18.15 µs                 | 10.01 µs              | 0.75%                        |
     | **ManyToOne $2 \to 1$**        | RF = 2048              | 18.73 µs                 | 10.21 µs              | 0.77%                        |
     | **ManyToManyShared $2 \to 2$** | RF = 2048              | 18.65 µs                 | 10.21 µs              | 0.77%                        |

3. **In-Place (`input == output`) vs. Out-of-Place Overhead:**

   * In DAW and low-latency host environments (plugin hosts, audio servers), channel buffers frequently alias. All multichannel variants employ early circular buffer absorption (`MirroredBuffer`) where input frames are absorbed into the ring buffer *before* computing output samples.
   * Benchmarks comparing in-place aliasing against separate out-of-place buffers show **virtually 0.0% overhead** (within ±1–2% environmental noise). In partitioned FFT mode, buffer address reuse slightly improves L1/L2 cache hit rate (e.g. 10.17 µs in-place vs 10.34 µs out-of-place for $2 \to 2$).

4. **Direct vs. FFT Crossover Across Block Sizes:**

   * For short impulse responses (RF $\le 128$), time-domain direct convolution outperforms FFT due to zero transform latency and unrolled SIMD dot-products.

   * For longer impulse responses (RF = 2048), partitioned FFT convolution delivers a sustained **~1.8× speedup** over Direct across all block sizes:

     | Block Size ($B$)            | ManyToMany $2 \to 2$ Direct (RF=2048) | ManyToMany $2 \to 2$ FFT (RF=2048) | Measured Speedup |
     |:--------------------------- |:------------------------------------- |:---------------------------------- |:---------------- |
     | **64 samples** (1.33 ms)    | 18.80 µs                              | 10.61 µs                           | **1.77×**        |
     | **256 samples** (5.33 ms)   | 76.80 µs                              | 41.40 µs                           | **1.85×**        |
     | **1024 samples** (21.33 ms) | 300.65 µs                             | 163.82 µs                          | **1.84×**        |
     | **2048 samples** (42.67 ms) | 600.21 µs                             | 326.41 µs                          | **1.84×**        |

### Recurrent Networks: SIMD Fused Gates (AVX2)

In LSTM models, fusing all four gates ($i, f, c, o$) into a unified matrix-vector multiplication with vectorized activations (AVX2/FMA) delivers substantial throughput gains over scalar execution:

| Topology      | Scalar Baseline | SIMD Fused (AVX2) | Measured Speedup |
|:------------- |:--------------- |:----------------- |:---------------- |
| **LSTM 1×8**  | ~45.1 µs        | **~2.27 µs**      | **19.8×**        |
| **LSTM 2×16** | ~45.2 µs        | **~10.86 µs**     | **4.2×**         |

Simultaneous gate computation eliminates redundant loads/stores and keeps intermediate vectors in YMM registers across the Sigmoid and Tanh activations.

### WaveNet Lite (CH=12) Memory Stride & Constant Folding

WaveNet Lite uses an internal dimension of 12 channels. On 256-bit SIMD (8 lanes), 12 channels do not align naturally to YMM boundaries:

1. **Padded-16 Route:** Residual convolution weights are padded to stride 16 in the model loader. The kernel executes `dot_product_16x_f32_accumulate` with zero-padded lanes 12..15. This route outperforms an 8+4 composite kernel (18–23 ns vs. 22–24 ns per tap) due to lower broadcast overhead and unrolled FMA pipelining.
2. **Compile-Time Monomorphization:** In `WaveNetLayer<IN, OUT, K>`, `OUT` is a const generic and `select_interleave_width` is a `const fn`. Assembly inspection confirms that the compiler monomorphizes exactly one dispatch arm per SKU without runtime branch overhead.

### Software Prefetch Policy for Causal Conv1D

The causal convolution tap loop uses a guarded software prefetch policy (`prefetch_strategy_simple` in `src/math/common/ops.rs`):

* **Guard Invariant:** `_mm_prefetch` with hint `_MM_HINT_T0` is executed **only when** the tap byte stride satisfies:

$$\text{step} = \text{dilation} \times \text{in\_ch} \le 16\text{ floats (64 bytes / 1 cache line)}$$

* **Rationale:** When $\text{step} \le 16$, the prefetched line overlaps the next tap's required memory, yielding a **3% to 5% improvement**. For larger strides ($\text{step} > 16$), prefetching targets memory that will not be read by subsequent taps, causing load-port contention and an **8% to 9% regression** on dilations $d \ge 4$.

### Conv1D Hotpath Bench — Calibration Protocol

[`benches/conv1d_hotpath_bench.rs`](../benches/conv1d_hotpath_bench.rs) is an off-line calibration tool used to verify kernel-level convolution prefetching (`prefetch_guard`) and lane routing (`ch12_lane_route`).

* **Execution:**

  ```bash
  taskset -c 8 cargo bench --bench conv1d_hotpath_bench -- --save-baseline cal-01
  # (apply code modifications)
  taskset -c 8 cargo bench --bench conv1d_hotpath_bench -- --baseline cal-01
  ```

* **Decision Rules:**

  * A regression $> 5\%$ in the $16\times 16$ ($d=1$) kernel warrants an immediate revert.
  * An improvement $> 5\%$ in the $8\times 8$ ($d \ge 4$) kernel warrants adjusting prefetch lookahead.
  * Always cross-validate kernel changes at the model level via `cargo bench --bench regression_gate -- RT_WaveNet`.

### A2 Architecture Vectorization Paths

The second-generation architecture (A2) features per-layer FiLM conditioning and flexible channel counts:

* **A2-Full (CH=8):** Uses `A2Conv1dCh8` with f32 weights stored in column-major-per-tap layout (`w[k * 64 + in * 8 + out]`). Eight contiguous output channels load directly into YMM registers without in-flight transposition (~25.6 µs per block).
* **A2-Lite (CH=3):** Uses `A2Conv1dCh3` with a fully unrolled GEMV kernel (18 FMAs for $K=6$, 45 FMAs for $K=15$) and AVX2-batched post-convolution stages (~22.6 µs per block).
* **A2 Dynamic (`WaveNetA2Dyn`):** Employs AVX2+FMA vectorization for runtime-dimensioned models:
  * Mixin GEMV weights are transposed from row-major to column-major during model loading (`set_weights`), enabling 8-wide broadcast-FMA in the hot-path.
  * Head 1×1 and L1×1 residual projections execute via 8-wide `_mm256_fmadd_ps` with exact sequential lane reduction to maintain bit-identical golden vector parity.

### A2 Pre-Warm Strategy & Settling Cost

A2 stabilizes its causal history with a **full forward pass**, unlike the
replicate-to-history backfill used by the A1 WaveNet family:

* **WaveNet A1** (`wavenet/layer_array.rs::prewarm_internal`): feeds **one**
  frame with `PREWARM = true`; each layer backfills its entire receptive-field
  window via `copy_within` (stationary replication, `layer_array.rs:103-124`).
  Cost is bounded by the per-layer `RF × CH` memcpy plus one frame of compute —
  constant and independent of the requested sample count
  (`Prewarm_WaveNet_Standard` ≈ 165 µs).
* **A2** (static and dynamic, `a2/model/*/prewarm.rs`): `prewarm_reset` zeroes
  every ring and arms a zero-feed of `receptive_field_size` frames; the feed
  then runs the **full 23-layer + dilated-head forward pass** in ≤ 64-frame
  chunks, discarding the output. The A2 receptive field is
  `Σ (kᵢ−1)·dᵢ + (16−1) = 6346` frames, so prewarm cost ≈ `RF × per-frame
  steady state` (≈ 0.30 µs/frame for CH=8, ≈ 0.24 µs/frame for CH=3):
  `Prewarm_A2Full_CH8` ≈ 2.07 ms, `Prewarm_A2Lite_CH3` ≈ 1.51 ms.

The `num_samples` argument to `NamModel::prewarm` is **ignored** by both A1 and
A2 (`fn prewarm(&mut self, _num_samples: usize)`): the workload is fixed by the
receptive field, not by the host block size. The `…_2048samp` suffix in the
bench IDs labels the `prewarm(2048)` call, **not** the number of processed
frames — A2 always feeds `receptive_field_size ≈ 6346` frames. Dividing the
2.07 ms figure by 2048 (≈ 1.0 µs/sample) overstates the per-frame cost by
~3.4×; the steady-state per-frame cost is ≈ 0.30 µs (19.75 µs / 64 frames).

Because the integral prewarm can exceed one 64-sample RT callback (1.33 ms), a
host amortizes it across callbacks: the split primitives
(`prewarm_reset` + `prewarm_step(k)`) make the zero-feed resumable without
allocation, and `prewarm_step(k)` performs exactly `⌈k / 64⌉` full `process`
calls. There is no smaller-chunk penalty to reclaim — A2 already feeds the
largest internal block (64 frames, `WAVENET_MAX_NUM_FRAMES`).

### IR CabSim Frequency-Domain Convolution (UPOLS)

The cabinet simulator uses Uniform-Partitioned Overlap-Save (UPOLS) convolution:

* Partition FFTs are pre-computed during initialization.
* `ConvEngine::process()` executes with **zero heap allocations**, operating entirely on pre-allocated aligned buffers.
* Median processing latency for a 2048-tap impulse response (32 partitions of 64 samples) is **~1.22 µs per block** (~0.09% of RT budget).

### Lock-Free SPSC & Multi-Tier GC Cascade (`spsc_swap_bench`)

Real-time resource updates (model swaps, IR reloading, oversampling mode changes) utilize a non-blocking 3-phase drain protocol on the audio thread combined with a 3-tier Garbage Collection (GC) cascade:

| Operation                              | Domain        | Invariant / Path                             | Latency (AVX2) |
|:-------------------------------------- |:------------- |:-------------------------------------------- |:-------------- |
| `Swap_Drain_Quiescent`                 | Audio Thread  | Phase 1 empty check under steady state       | **~2.7 ns**    |
| `Swap_Drain_LowContention_Single`      | Audio Thread  | Single pending payload drained and installed | **~448 ns**    |
| `Swap_Drain_HighContention_Coalescing` | Audio Thread  | 16-burst queue coalesced (latest-wins)       | **~888 ns**    |
| `Gc_Cascade_Tier1_Spsc`                | Audio Thread  | Primary lock-free SPSC channel enqueue       | **~212 ns**    |
| `Gc_Cascade_Tier2_ParkingLot`          | Audio Thread  | Fixed 16-slot parking lot fallback           | **~370 ns**    |
| `Gc_Cascade_Tier3_Overflow`            | Audio Thread  | Lock-free linked list overflow queue         | **~1.79 µs**   |
| `Gc_Drain_Housekeeping_AllTiers`       | Off-RT Thread | Sweeps and deallocates payloads              | **~1.53 µs**   |

*Real-time guarantees:* All queue operations on the audio thread complete in under 2 µs even under extreme saturation, with zero heap allocations or system calls.

### Real-Time Latency Telemetry: `LatencyHistogram` vs. `AtomicHistogram`

High-frequency latency monitoring in the audio callback requires zero locks, zero memory allocation, and negligible CPU overhead. Two complementary histogram structures serve distinct operational domains:

1. **`AtomicHistogram` (Runtime Telemetry in `RtStatusFlags`):**

   * Structure: 32 octave buckets (power-of-two boundaries: $[0, 1), [1, 2), \dots, [2^{30}, 2^{31})\text{ ns}$).
   * Memory Footprint: 384 bytes (`#[repr(align(128))]`, 6 cache lines).
   * Trade-off: Compact cache footprint on `RtStatusFlags`, but relative quantization error spans up to $100\%$ ($2\times$) between adjacent power-of-two bucket edges.

2. **`LatencyHistogram` (High-Resolution Verification Telemetry):**

   * Structure: 256 log-linear sub-bins (8 sub-bins per power-of-two octave), covering dynamic range up to $1.07\text{ s}$ ($2^{30}\text{ ns}$).
   * Relative Error Bound: $\le 12.5\%$ maximum relative error across all percentiles ($p50, p90, p99, p99.9$).
   * Micro-Benchmark Cost: Measured at **~15.9 ns per `record()` call** (64 calls in $1.019\ \mu\text{s}$ via `dsp_bench`). Zero dynamic allocations, lock-free relaxed atomic increments.
   * Testing Deployment: Adopted in real-time constraint verification suites ([`tests/rt_constraints/rt_deadline.rs`](../tests/rt_constraints/rt_deadline.rs) and [`tests/rt_constraints/rt_jitter.rs`](../tests/rt_constraints/rt_jitter.rs)), eliminating power-of-two quantization artifacts and providing fine-grained tail latency visibility without bloating `RtStatusFlags` in production host pipelines.

---

## 7. SIMD Multiversioning ROI & Dispatch Policy

The minimum target architecture for NeuralAmpModeler-rs is **`x86-64-v3`** (AVX2, FMA, BMI2). Code generation unconditionally relies on AVX2 instructions throughout all processing modules.

### The 3-Tier Return on Investment (ROI) Rule

To prevent codebase fragmentation and binary bloat, specialized instruction set implementations (such as AVX-512) are evaluated against a strict 3-tier promotion threshold:

| Speedup vs. AVX2 Baseline         | Statistical Significance | Action                         | Rationale                                                               |
|:--------------------------------- |:------------------------ |:------------------------------ |:----------------------------------------------------------------------- |
| **$\ge 12.0\%$**                  | $p < 0.05$               | **KEEP (Production Dispatch)** | Delivers meaningful CPU headroom in real-time callbacks.                |
| **$< 5.0\%$**                     | Any                      | **DROP (No Specialization)**   | Duplication overhead outweighs performance difference.                  |
| **$5.0\% \le \Delta\% < 12.0\%$** | $p < 0.05$               | **DROP / CONDITIONAL**         | Dropped unless significant reduction in tail latency ($p99$) is proven. |

### AVX-512 Evaluation on Physical Silicon

In canonical benchmark audits on native AVX-512 hardware (Intel Xeon Platinum 8488C Sapphire Rapids, 64-sample blocks @ 48 kHz), specialized AVX-512 kernels failed the promotion gate across all primary topologies:

* `LSTM_2x16_64samp_48kHz`: AVX2 = 13.72 µs vs. AVX-512 = 15.49 µs (**−12.87% deficit**, $p < 0.0001$)
* `LSTM_1x16_64samp_48kHz`: AVX2 = 6.55 µs vs. AVX-512 = 7.67 µs (**−17.18% deficit**, $p < 0.0001$)
* `A2Full_CH8_64samp_48kHz`: AVX2 = 23.47 µs vs. AVX-512 = 31.29 µs (**−33.31% deficit**, $p < 0.0001$)
* `A2Lite_CH3_64samp_48kHz`: AVX2 = 20.28 µs vs. AVX-512 = 21.30 µs (**−5.05% deficit**, $p < 0.0001$)
* `WaveNet_Std_CH16_64samp_48kHz`: AVX2 = 42.05 µs vs. AVX-512 = 42.98 µs (**−2.21% deficit**, $p = 0.0009$)

#### Why 512-bit ZMM Regresses in Compact Neural Audio

1. **Register Underutilization:** Audio networks use compact channel geometries ($C = 3, 4, 8, 12, 16$). Fitting these into 512-bit (16-lane) ZMM vectors requires zero-masking overhead and dummy computations.
2. **Frequency Throttling:** On earlier Intel architectures, executing 512-bit heavy instructions triggers core frequency downclocking.
3. **AVX2 Cache Density:** AVX2 (256-bit YMM) instructions pack more densely into the Level 1 Instruction Cache (L1i), avoiding pipeline fetch stalls.

### Production Policy & Role of `--features avx512`

* **Production Builds:** Default builds unconditionally dispatch `Avx2Math`. AVX-512 kernels are excluded from the default compiled binary.
* **The `avx512` Feature Flag:** Retained as an opt-in research target and multiversioning reference. Enabling this flag in production releases (e.g. DAW plugins or standalone hosts) is **actively discouraged**.

### Remote SIMD Gating Suite (`utils/remote-simd-gate.sh`)

To evaluate prospective SIMD extensions or verify cross-ISA parity on remote machines:

1. **Same-VM Measurement Rule:** Never compare cloud VM numbers against bare-metal desktop baselines. AVX-512 vs. AVX2 comparisons must occur on the **same virtual machine instance** using identical compiler configurations.

2. **Local Intel SDE Emulation:** Developers on AVX2 hardware can run mathematical parity checks via the Intel Software Development Emulator:

   ```bash
   ./utils/remote-simd-gate.sh --sde
   ```

3. **Audit Receipt:** Running the suite generates `target/logs/remote-simd-receipt.json`, recording machine provenance, sample distributions, and two-tailed Welch t-test results.

---

## 8. L1i Instruction Cache Budget & Code Size Analysis

Real-time audio callbacks must avoid instruction cache misses (*i-cache thrashing*). On modern x86-64 processors (AMD Zen and Intel Golden Cove / Raptor Cove), the Level 1 Instruction Cache (L1i) is limited to **32 KB per core**.

### Empirical Hardware Telemetry (Calibrated Hardware Audit)

To determine whether monomorphized dispatch functions induce instruction cache thrashing, hardware performance counter telemetry (`perf stat`) was captured during 64-sample block inference:

| Benchmark Target        | Topology               | Block Latency | IPC      | L1i Miss Rate | L1i MPKI   | Verdict               |
|:----------------------- |:---------------------- |:------------- |:-------- |:------------- |:---------- |:--------------------- |
| `RT_A2_Dyn_Gated_CH8`   | Dynamic Gated (CH=8)   | 180.9 µs      | **3.05** | **0.419%**    | **0.0129** | Negligible contention |
| `RT_A2_Dyn_Blended_CH3` | Dynamic Blended (CH=3) | 137.2 µs      | **2.85** | **0.046%**    | **0.0097** | Negligible contention |
| `RT_A2_Full_CH8`        | Static Baseline (CH=8) | 25.9 µs       | **2.74** | **0.236%**    | **0.0564** | Baseline              |

### Architectural Conclusions

1. **Static Footprint Does Not Dictate Dynamic Thrashing:**
   Even though dynamic cascade functions may exceed 32 KB on disk, their dynamic Misses Per Kilo-Instructions (MPKI) remains below **0.013** — two orders of magnitude below the thrashing threshold ($> 1.0\text{ MPKI}$). Tight inner loops execute repeatedly from the CPU Op-Cache.
2. **High IPC Efficiency:**
   An IPC of **2.74 to 3.05** confirms that the instruction pipeline operates with zero instruction-fetch stalls.
3. **Rejection of Artificial Function Splitting:**
   Proposals to partition large monomorphized functions simply to reduce `.text` symbol size are **rejected**. Splitting introduces function call overhead, increases register pressure, and disrupts compiler instruction scheduling without improving cache hit rates.

### Code Size Observability

Developers can inspect symbol code footprint using `cargo bloat`:

```bash
# Top 50 largest functions in the binary
cargo bloat --release --example synthetic_model -n 50 -w

# Crate-level breakdown of the .text section
cargo bloat --release --example synthetic_model --crates
```

---

## 9. Optimization Boundaries & Downstream Integration

### Kernel Specialization vs. Naive DRY

In performance-critical SIMD, naive deduplication ("Don't Repeat Yourself") must not be applied across divergent operational domains.

* **Case Study (FiLM vs. GEMM Dot Product):**
  * `film::dot_product_avx2` operates on very short vectors ($1..=8$) using 2 YMM accumulators; for `len < 8` it takes a plain left-to-right scalar fast path (bit-identical to summing the zero AVX2 accumulators first, since `0.0 + x == x`). It is `#[inline(always)]` so the branch folds into each FiLM row.
  * `gemm::dot_basic::dot_product_avx2` operates on large matrices using 4 YMM accumulators and Kahan compensated summation in the tail.
* **Floating-Point Non-Associativity:** IEEE-754 floating-point addition is non-associative: $(a + b) + c \ne a + (b + c)$. Forcing short FiLM vectors through the 4-accumulator GEMM kernel alters rounding by up to 2 ULPs and adds unnecessary branch overhead.
* **Rule:** Kernels with distinct structural profiles remain specialized. Numerical consistency is guarded by automated parity tests (`test_dot_product_avx2_identity_with_gemm`).

### FiLM Per-Frame Application Overhead (A2-Dynamic grouped topology)

The A2-Dynamic engine applies FiLM **per frame per slot** (`process_frame_dyn`), so a
topology with `L` layers and `S` active FiLM slots performs `L × S × 64` bundled
`FiLMLayer::process` calls per 64-sample block (`23 × 8 × 64 = 11 776` for the
canonical grouped fixture). The dominant cost is therefore call/overhead, not the
short-vector arithmetic:

* `FiLMLayer::process` is `#[inline]` (LLVM-decided), **not** `#[inline(always)]`:
  force-inlining its full body at all 8 call sites inside `process_frame_dyn`
  pushes the per-frame kernel past the µop cache; relaxing the hint improved the
  end-to-end A2-Dynamic FiLM benches by ~28–33 % (e.g. `A2Dyn_FiLM_Grouped_CH4`
  1602 → 1076 µs on the dev host), bit-exactly.
* The isolated stage cost is unaffected (`A2Dyn_Stage_FiLM_C4_C8_G2` ≈ 68 ns/call),
  confirming the gain is overhead relocation, not arithmetic change.

### Block Prologue Amortization

Operations that execute strictly **once per audio buffer** (in the prologue before the sample processing loop) have negligible performance impact.

* *Example:* An integer division instruction (`div`) in the cascade prologue calculating buffer bounds executes once per block (~375 times/sec at 48 kHz / 128 samples). At ~15–20 cycles (~5 ns), this accounts for only **0.0057%** of the block budget.
* Attempting strength reduction via conditional branches introduces branch misprediction risks that exceed the cost of the division itself. Prologue arithmetic that amortizes to negligible CPU impact remains unaltered.

### Upstream Engine vs. Downstream Application Boundaries

Profile-Guided Optimization (PGO) and Post-Link Optimization (such as LLVM BOLT) provide latency advantages, but the boundaries between `NeuralAmpModeler-rs` and downstream consumers are strictly defined:

1. **Upstream Engine Responsibilities:**
   * Exposes a host-agnostic, deterministic profiling catalog (`reference_architectures()` in `src/testing/catalog.rs`).
   * Provides headless profiling harnesses that can be driven without GUI or audio-server dependencies.
2. **Downstream Application Responsibilities:**
   * Downstream packaging pipelines (e.g. standalone hosts or DAW plugins) drive `-Cprofile-generate` and `-Cprofile-use` using their own compiler flags and target environments.
   * Binary post-optimization (BOLT reordering and `__bolt_hugify`) operates strictly on final linked ELF binaries or shared libraries, never on intermediate `.rlib` static libraries.
3. **No Distribution of Pre-Compiled Profiles:**
   Upstream does not distribute canned `.profdata` profiles because LLVM profile formats are tightly coupled to specific compiler revisions, CFG hash matching is fragile across dependency updates, and training on a single topology biases PGO branch weights, causing latency spikes when switching models.
