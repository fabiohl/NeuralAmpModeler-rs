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

| Target ID                      | Description                           | Median Latency  | % of 1.33 ms RT Budget |
|:------------------------------ |:------------------------------------- |:--------------- |:---------------------- |
| `RT_WaveNet_Std_CH16`          | WaveNet Standard (CH=16)              | 43.29 µs        | 3.25%                  |
| `RT_WaveNet_Feather_CH8`       | WaveNet Feather (CH=8)                | 19.85 µs        | 1.49%                  |
| `RT_WaveNet_Lite_CH12`         | WaveNet Lite (CH=12, padded-16)       | 56.40 µs        | 4.23%                  |
| `RT_WaveNet_Nano_CH4`          | WaveNet Nano (CH=4)                   | 17.83 µs        | 1.34%                  |
| `RT_A2_Full_CH8`               | A2-Full (CH=8, col-major SIMD)        | 25.64 µs        | 1.92%                  |
| `RT_A2_Lite_CH3`               | A2-Lite (CH=3, unrolled GEMV)         | 22.55 µs        | 1.69%                  |
| `RT_LSTM_1x16`                 | LSTM 1×16 (fused SIMD gates)          | 6.83 µs         | 0.51%                  |
| `RT_LSTM_2x8`                  | LSTM 2×8 (fused SIMD gates)           | 7.13 µs         | 0.53%                  |
| `RT_Linear_Direct_RF4`         | Linear Direct FIR (RF=4, time-domain) | 0.33 µs         | 0.02%                  |
| `RT_Linear_Fft_RF2048`         | Linear Partitioned FFT (RF=2048)      | 4.84 µs         | 0.36%                  |
| `RT_ConvNet`                   | ConvNet (CH=8, 6 blocks)              | 8.66 µs         | 0.65%                  |
| `RT_WaveNet_Dyn_Free`          | WaveNet Dynamic Free-Shape            | 21.69 µs        | 1.63%                  |
| `RT_LSTM_Dyn_1x7`              | LSTM Dynamic 1×7                      | 8.81 µs         | 0.66%                  |
| `RT_A2_Dyn_Gated_CH8`          | A2 Dynamic Gated (CH=8)               | 188.35 µs       | 14.13%                 |
| `RT_A2_Dyn_Blended_CH3`        | A2 Dynamic Blended (CH=3)             | 147.50 µs       | 11.06%                 |
| `RT_DSP_Resampler_44k1_to_48k` | Polyphase Resampler 44.1k $\to$ 48k   | 1.22 µs / block | 0.09%                  |
| `RT_DSP_Resampler_96k_to_48k`  | Polyphase Resampler 96k $\to$ 48k     | 0.62 µs / block | 0.05%                  |
| `RT_DSP_CabSim_IR_Medium`      | CabSim UPOLS (2048 taps)              | 1.22 µs / block | 0.09%                  |
| `RT_DSP_Pipeline_Base_NoOS`    | End-to-end Pipeline (Standard, 1×)    | 43.57 µs        | 3.27%                  |
| `RT_DSP_Pipeline_HQ_4xOS`      | End-to-end Pipeline (Standard, 4× OS) | 177.39 µs       | 13.31%                 |

*Note: All single-model inferences consume $\le 4.3\%$ of the real-time block budget; even dynamic A2 topologies and 4× oversampled pipelines maintain $> 85\%$ real-time headroom.*

---

## 4. Regression Gate — Automated Performance Defense

[`utils/tests-performance-regression.sh`](../utils/tests-performance-regression.sh) is the canonical benchmark-based performance defense tool. It compares current timings against a persisted statistical baseline and fails closed if latency degrades beyond the allowed threshold.

### Core Mechanisms

1. **CPU Core Pinning:** Pins benchmark execution to a dedicated core via `taskset -c <core>`. The core is resolved automatically via `pick_bench_core` using the 3-step precedence: (1) `NAM_BENCH_CORE` if explicitly set; (2) first online isolated CPU from `/sys/devices/system/cpu/isolated` (preferring core 8 when 8-9 are isolated, with cpu9 offline intentional); (3) fallback `nproc / 2` (integer division). This prevents OS scheduler migrations and cache thrashing.
2. **Controlled Statistical Rigor:** Executes the `regression_gate` suite with `sample_size=100, measurement_time=5s, warm_up_time=1s, noise_threshold=0.05`. Enforces `InstructionSet::Avx2` via `ForceAvx2Guard` so that AVX-512 hosts evaluate the exact `x86-64-v3` baseline.
3. **Machine Verdict (`nam_perf_gate verdict`):** Evaluates Criterion's `target/criterion/<id>/change/estimates.json`. If the bootstrapped mean-change confidence interval lies entirely above the `+5%` noise band, the script exits with `REGRESSION_DETECTED`. If comparison artifacts are missing or unreadable, it fails closed with `REGRESSION_BLIND`.
4. **Baseline Storage & Fingerprinting:** Authoritative baselines reside in `.performance-baselines/` (gitignored). A machine fingerprint file (`baseline-fingerprint.json`) captures the CPU model, ISA extension flags, compiler version, target triple, governor, pinned core, and producing git commit.
5. **Sub-Microsecond Batching:** Ultra-fast DSP routines (`RT_DSP_Resampler_*` and `RT_DSP_CabSim_IR_Medium`) process batches of **64 blocks** per Criterion sample to maintain timer resolution above the noise floor. Reported values are divided by 64 to represent per-block latency.

### Script Modes & Environment Variables

| Mode                | Command                                                      | Behavior                                                                                                                  |
|:------------------- |:------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------- |
| **Check** (default) | `utils/tests-performance-regression.sh --check`              | Read-only. Compares against `.performance-baselines/`. Exits non-zero if a regression is detected or baseline is missing. |
| **Bootstrap**       | `utils/tests-performance-regression.sh --bootstrap-baseline` | Re-generates `.performance-baselines/` and writes `baseline-fingerprint.json`. **Human-only operation.**                  |

| Variable                 | Default                                     | Purpose                                                             |
|:------------------------ |:------------------------------------------- |:------------------------------------------------------------------- |
| `NAM_BENCH_CORE`         | `pick_bench_core` (isolated or `nproc / 2`)  | Dedicated CPU core to pin benchmarks via `taskset`.                 |
| `NAM_BASELINE_NAME`      | `ci-baseline`                               | Name of the Criterion baseline series.                              |
| `NAM_BENCH_SUITE`        | `regression_gate` | Benchmark binary to drive (e.g. `spsc_swap_bench`).                 |
| `NAM_THERMAL_COOLDOWN_S` | `180`             | Cooldown period before benchmarking to stabilize clock frequencies. |

### Environmental Noise, False Alarms & Hardware Isolation

Because the DSP engine operates in a state of extreme micro-architectural optimization, latency measurements are highly sensitive to thermal throttling, DVFS transitions, and background operating system jitter.

> [!WARNING]
> **Risk of False Alarms:** In sub-microsecond and microsecond-scale benchmarks (e.g. `RT_Linear_Direct_RF4` at ~340 ns or `RT_LSTM_2x8` at ~7.1 µs), small environmental variations, CPU thermal drift, or concurrent OS tasks can trigger false-positive regression alarms ($+2–4\%$ variation). Exercise caution and do not hastily bootstrap new baselines or revert code without verifying that the result is reproducible under isolated conditions.

#### Recommended Mitigation Measures

1. **Kernel-Level Core Isolation (`isolcpus` & `nohz_full`):**
   To shield benchmark cores from kernel scheduling ticks, timers, and non-bound user space threads, boot the Linux kernel with CPU isolation parameters (e.g. for cores 8 and 9):

   ```text
   isolcpus=8,9 nohz_full=8,9
   ```

   Combined with pinning (`NAM_BENCH_CORE=8 taskset -c 8 ...`), this eliminates OS preemption and thread migration.
2. **CPU Frequency Scaling (`performance` Governor):**
   Lock the CPU scaling governor to `performance` across all cores to prevent Dynamic Voltage and Frequency Scaling (DVFS) transition delays:

   ```bash
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

3. **Thermal Stabilization & Hardware Cooldown:**
   Allow the CPU to cool down before executing baseline benchmarks (`NAM_THERMAL_COOLDOWN_S=180` by default). The regression gate sleeps between heavy phases to prevent thermal throttling from skewing Criterion iteration times.
4. **Low Background Load & Host Quiescence:**
   Ensure workstation background load is minimal ($\text{load average} \ll 1.0$), with browsers, container engines, and heavy background services closed.

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

     | Topology Geometry | Receptive Field (Taps) | Direct Latency (64 samp) | FFT Latency (64 samp) | % of 1.33 ms RT Budget (FFT) |
     | :--- | :--- | :--- | :--- | :--- |
     | **Mono $1 \to 1$** | RF = 2048 | 8.86 µs | 4.85 µs | 0.36% |
     | **OneToMany $1 \to 2$** | RF = 2048 | 18.15 µs | 10.01 µs | 0.75% |
     | **ManyToOne $2 \to 1$** | RF = 2048 | 18.73 µs | 10.21 µs | 0.77% |
     | **ManyToManyShared $2 \to 2$** | RF = 2048 | 18.65 µs | 10.21 µs | 0.77% |

3. **In-Place (`input == output`) vs. Out-of-Place Overhead:**
   * In DAW and low-latency host environments (CLAP, PipeWire, JACK), channel buffers frequently alias. All multichannel variants employ early circular buffer absorption (`MirroredBuffer`) where input frames are absorbed into the ring buffer *before* computing output samples.
   * Benchmarks comparing in-place aliasing against separate out-of-place buffers show **virtually 0.0% overhead** (within ±1–2% environmental noise). In partitioned FFT mode, buffer address reuse slightly improves L1/L2 cache hit rate (e.g. 10.17 µs in-place vs 10.34 µs out-of-place for $2 \to 2$).

4. **Direct vs. FFT Crossover Across Block Sizes:**
   * For short impulse responses (RF $\le 128$), time-domain direct convolution outperforms FFT due to zero transform latency and unrolled SIMD dot-products.
   * For longer impulse responses (RF = 2048), partitioned FFT convolution delivers a sustained **~1.8× speedup** over Direct across all block sizes:

     | Block Size ($B$) | ManyToMany $2 \to 2$ Direct (RF=2048) | ManyToMany $2 \to 2$ FFT (RF=2048) | Measured Speedup |
     | :--- | :--- | :--- | :--- |
     | **64 samples** (1.33 ms) | 18.80 µs | 10.61 µs | **1.77×** |
     | **256 samples** (5.33 ms) | 76.80 µs | 41.40 µs | **1.85×** |
     | **1024 samples** (21.33 ms) | 300.65 µs | 163.82 µs | **1.84×** |
     | **2048 samples** (42.67 ms) | 600.21 µs | 326.41 µs | **1.84×** |

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

### Kahan Summation Policy

Kahan compensated summation is deliberately **omitted from the static Conv1D tap loop**. For filter kernels with $K \le 3$, standard single-precision accumulation error is bounded by $O(3\varepsilon) < 10^{-7}$, which is inaudible and within golden test tolerances. Eliminating per-tap Kahan compensation removed redundant arithmetic operations without degrading audio fidelity. Compensated summation is retained only in dense GEMM reduction tails where accumulation lengths span hundreds of elements.

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
  * `film::dot_product_avx2` operates on very short vectors ($1..=8$) using 2 YMM accumulators and an aggressive `#[inline(always)]` annotation.
  * `gemm::dot_basic::dot_product_avx2` operates on large matrices using 4 YMM accumulators and Kahan compensated summation in the tail.
* **Floating-Point Non-Associativity:** IEEE-754 floating-point addition is non-associative: $(a + b) + c \ne a + (b + c)$. Forcing short FiLM vectors through the 4-accumulator GEMM kernel alters rounding by up to 2 ULPs and adds unnecessary branch overhead.
* **Rule:** Kernels with distinct structural profiles remain specialized. Numerical consistency is guarded by automated parity tests (`test_dot_product_avx2_identity_with_gemm`).

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
