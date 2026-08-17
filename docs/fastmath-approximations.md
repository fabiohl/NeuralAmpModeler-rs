<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# FastMath Approximations & Activation Precision Modes

Architectural decisions, performance benchmarks, and normative guidelines for transcendental activation functions (`tanh`, `sigmoid`) and precision modes in the NAM-rs DSP hot-path.

> [!IMPORTANT]
> This document records **definitive decisions** validated by benchmarks. Do not alter production choices without running `cargo bench` and confirming there is no statistically significant regression ($p < 0.05$).

---

## 1. Activation Precision Architecture

NAM-rs provides a runtime-selectable activation precision switch via the `ActivationPrecision` enum in [`src/math/activations/mod.rs`](../src/math/activations/mod.rs). The mode is configured per thread via Thread-Local Storage (`ACTIVE_MODEL_PRECISION` TLS), accessed via `set_activation_tls()`, `clear_activation_tls()`, and `activation_precision()` (which defaults to `Standard` if unset). The legacy process-wide atomic flag was removed to ensure complete thread safety and isolation across concurrent audio streams.

| Precision Mode | Tanh Strategy                      | Sigmoid Strategy                   | Max Error (vs `f32` ref)      | Throughput (256 elem, AVX2) | Default Status           |
|:-------------- |:---------------------------------- |:---------------------------------- |:----------------------------- |:--------------------------- |:------------------------ |
| **`Standard`** | Degree-6 Taylor minimax ($e^{2x}$) | Degree-6 Taylor minimax ($e^{-x}$) | $\le 2.4 \times 10^{-7}$      | ~110 ns                     | **Universal Default**    |
| **`Fast`**     | Padé [5,4] rational approx.        | Degree-17 Lawson minimax           | $\approx 2.32 \times 10^{-3}$ | **~54 ns**                  | Opt-in (CPU-constrained) |

### 1.1 Standard Mode (`ActivationPrecision::Standard`, Production Default)

Uses polynomial $\exp$-based kernels with degree-6 Taylor minimax and integer range reduction ($k = \text{round}(x \cdot \log_2 e)$, $r = x - k \cdot \ln 2$). Implemented in [`src/math/activations/tanh/high_fidelity.rs`](../src/math/activations/tanh/high_fidelity.rs) and [`src/math/activations/sigmoid/high_fidelity.rs`](../src/math/activations/sigmoid/high_fidelity.rs):

- **Tanh formula:** $\text{tanh}(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$
- **Sigmoid formula:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **Precision:** Precision is ~10,000× higher than `Fast` mode. Hardware division (`_mm256_div_ps`) incurs a throughput cost (~110 ns for 256 elements vs ~54 ns in `Fast` mode), but guarantees exact-grade outputs across all model topologies.

### 1.2 Fast Mode (`ActivationPrecision::Fast`, Performance Opt-in)

Designed for ultra-low latency or CPU-constrained setups:

- Uses Padé [5,4] rational approximation for `tanh` (~54 ns for 256 elements, AVX2).
- Uses direct minimax degree-17 polynomial for `sigmoid`.

> [!WARNING]
> **Calibration Limits under Fast Mode:** `Fast` mode approximations are optimized over compact domains: `tanh` on $[-4, 4]$ (max absolute error $\approx 2.32 \times 10^{-3}$) and `sigmoid` on $[-8, 8]$ (max absolute error $\approx 4.09 \times 10^{-4}$). In recurrent architectures (LSTM) with large hidden states where gate inputs $|g| > 4$, approximation errors accumulate over time, creating recurrent state drift. Standard mode avoids this drift and is recommended for recurrent models.

#### Activation Precision Impact Across Topologies (Measured in `quality-contract.json`)

| Model Topology          | Fast Mode SNR (Padé) | Standard Mode SNR (Exact) | Δ SNR Gain  |
|:----------------------- |:-------------------- |:------------------------- |:----------- |
| **LSTM 1×16**           | 15.9 dB              | 103.2 dB                  | **+87.3 dB**|
| **LSTM 2×8**            | 24.1 dB              | 114.0 dB                  | **+89.9 dB**|
| **Official lstm (H=3)** | 29.3 dB              | 120.5 dB                  | **+91.2 dB**|

*Average SNR gain with `Standard` (exact-grade): **+89.5 dB** across tested LSTM architectures.*

### 1.3 Interaction with Oversampling & Full Topology Coverage

- **Oversampling Interaction:** In HQ mode (4× oversampling, see [`docs/architecture.md`](architecture.md)), half-band filtering eliminates high-frequency aliasing. Residual distortion is then bounded by activation precision, where `Standard` mode achieves SNR $> 120\text{ dB}$.
- **Full Model Coverage:** Activation precision dispatch is supported across all model families (WaveNet A1/A2, LSTM 1×N / 2×N, ConvNet, and Dynamic models), including fused 4-gate LSTM GEMV kernels ([`src/math/lstm/gates.rs`](../src/math/lstm/gates.rs)).

---

## 2. Production FastMath Approximations (`Fast` Mode)

### 2.1 Tanh — Padé [5,4] with Hardware Division

#### Approximating Function

$$\text{tanh}(x) \approx \frac{x \cdot (x^2 + 105) \cdot (x^2 + 945)}{(15x^2 + 420) \cdot x^2 + 945}$$

Implemented in [`src/math/activations/tanh/production.rs`](../src/math/activations/tanh/production.rs):

- `simd_tanh_avx2(x: __m256)` — 8 floats, AVX2 + FMA.
- `simd_tanh_dual_avx2(x1, x2: __m256)` — 16 floats, broadcast coefficients shared once.
- `simd_tanh_avx512(x: __m512)` — 16 floats, AVX-512.
- `scalar_pade_tanh(x: f32)` — Scalar fallback with `mul_add`.

#### Solution Characteristics

| Property                                 | Value                                                                |
|:---------------------------------------- |:-------------------------------------------------------------------- |
| Maximum absolute error in $[-4, 4]$      | $\approx 2.32 \times 10^{-3}$                                        |
| SIMD operations (AVX2, 8 elem)           | ~9 ops                                                               |
| Throughput `tanh_slice` (256 elem, AVX2) | **~54 ns**                                                           |
| Coefficients                             | `PADE_TANH_*` in [`src/math/constants.rs`](../src/math/constants.rs) |

#### Rationale for Hardware Division (`_mm256_div_ps`) vs Newton-Raphson

Empirical evaluation (10M samples in $[-4, 4]$):

| Variant                        | Max Abs Error             | RMS Error     | Throughput (256 elem) |
|:------------------------------ |:------------------------- |:------------- |:--------------------- |
| 7-Segment Piecewise            | $4.90 \times 10^{-3}$     | —             | ~163 ns               |
| Padé NR2 (`rcp` + 2× Newton)   | $2.32 \times 10^{-3}$     | $\approx$ Div | ~104 ns               |
| **Padé Div (`_mm256_div_ps`)** | **$2.32 \times 10^{-3}$** | **Minimum**   | **~63 ns**            |

Double Newton-Raphson iteration (NR2) fully saturates the 24-bit `f32` mantissa, yielding an error ratio of 1.000× relative to hardware division. On modern x86 microarchitectures, `_mm256_div_ps` has low latency (10–14 cycles) and high throughput, making hardware division simpler, faster (~63 ns vs ~104 ns), and more accurate than manual NR pipelines.

---

### 2.2 Sigmoid — Direct Minimax (Degree 17)

Instead of propagating `tanh` error via $\sigma(x) = 0.5 + 0.5 \cdot \text{tanh}(x/2)$, `Fast` mode uses a direct odd polynomial of degree 17 (9 terms) for $[-8, 8]$, generated via Lawson's weighted minimax algorithm.

Implemented in [`src/math/activations/sigmoid/production.rs`](../src/math/activations/sigmoid/production.rs):

| Metric             | Tanh Identity Baseline       | Direct Minimax (Degree 17)                       |
|:------------------ |:---------------------------- |:------------------------------------------------ |
| Max Absolute Error | $\approx 6.8 \times 10^{-4}$ | **$\approx 4.09 \times 10^{-4}$** (1.67× better) |
| SIMD Operations    | 16 ops                       | **15 ops**                                       |

---

## 3. Micro-Architectural Experiments & Findings

### 3.1 Failed Experiment: Piecewise 7-Segment Tanh

Replacing Padé [5,4] with 7 polynomials of degree 5 blended branchlessly via `_mm256_blendv_ps` was evaluated and rejected:

| Metric                 | Padé [5,4] (Baseline) | 7-Segment Piecewise               |
|:---------------------- |:--------------------- |:--------------------------------- |
| SIMD Operations        | ~9                    | **~28** (7 polys + 6 blends)      |
| Max Error in $[-4, 4]$ | $2.32 \times 10^{-3}$ | **$4.90 \times 10^{-3}$** (worse) |
| Throughput (256 elem)  | 63 ns                 | **163 ns** (+159% latency)        |

**Root Cause of Failure:**

1. Branchless blending evaluates all 7 polynomials unconditionally.
2. Cascaded `blendv_ps` instructions bottleneck Port 5 (shuffle unit).
3. The implementation was removed from production code.

### 3.2 Single-Mode `f32` WaveNet & Model Fidelity

WaveNet A1 models operate exclusively with native `f32` weights and buffers (no `u16` BF16/F16 paths in the hot-path).

- **Weights:** Native `f32` arrays.
- **Activations & Interop:** Standard mode achieves SNR $> 129\text{ dB}$ (ESR $\sim 10^{-13}$ to $10^{-14}$) vs C++ NAMCore reference across standard models (Standard: 136.4 dB SNR, Feather: 133.2 dB SNR, Nano: 131.9 dB SNR).
- **Lite Array Alignment:** Historical divergence in WaveNet Lite (CH=12) was resolved by aligning `MirroredBuffer` delay line boundaries (`MirroredBuffer::new_aligned`), guaranteeing channel stride divisibility.

---

## 4. Real-Time Audio Policies (Silence & Subnormals)

### 4.1 WaveNet Non-Zero Silence Policy

Under silent input, WaveNet models produce a residual output of $\approx 3.58 \times 10^{-5}$ ($-89\text{ dBFS}$).

- **Root Cause:** Accumulation of Conv1D bias terms ($0.001$ per layer across 12 layers) through dense $1\times 1$ projections and `head_scale` ($0.1$).
- **Policy:** Faithful to C++ NAMCore (`NAM/dsp.h`). The inference hot-path does **not** force zero output, preserving authentic model characteristics (e.g. noise floor / saturation). Gating is handled by the dedicated noise gate layer ([`src/dsp/gate.rs`](../src/dsp/gate.rs)).

### 4.2 Anti-Subnormal Prevention with DC Dither

To prevent CPU soft-emulation penalties when processing near-zero values during quiet signals:

- Constant `DENORMAL_DITHER_OFFSET = 1.0e-11` ($-220\text{ dBFS}$) is added during input processing ([`src/dsp/pipeline/stages/input.rs`](../src/dsp/pipeline/stages/input.rs)) and subtracted during output processing ([`src/dsp/pipeline/stages/output.rs`](../src/dsp/pipeline/stages/output.rs)).
- Completely inaudible ($76\text{ dB}$ below 24-bit DAC floor) with zero runtime performance cost.

### 4.3 DAZ / FTZ Enforcement

Denormals-Are-Zero (DAZ) and Flush-To-Zero (FTZ) flags are active at the audio processing entry point:

- Helper function `set_daz_ftz()` in [`src/math/common/ops.rs`](../src/math/common/ops.rs).
- Reasserted at the start of every audio processing call (`capture_dsp_pipeline` in [`src/dsp/pipeline/capture.rs`](../src/dsp/pipeline/capture.rs)) via `set_daz_ftz()` — a fixed `stmxcsr`/`ldmxcsr` pair outside any sample loop.

---

## 5. Summary of Normative Guidelines & Checklist

When modifying or adding activation functions in [`src/math/activations/`](../src/math/activations/):

- [ ] **Benchmark LSTM Prewarm:** Run `cargo bench` to verify LSTM prewarm (`Prewarm_LSTM_2x16_2048samp`). Regressions $> 5\%$ are unacceptable.
- [ ] **Validate Vector Alignment:** Ensure AVX2 and AVX-512 kernels process aligned chunks and handle remainders cleanly.
- [ ] **Check Function Symmetry:** Verify odd symmetry for `tanh` ($f(-x) == -f(x)$).
- [ ] **Maintain Single/Dual Lanes:** Keep shared broadcast structure in `simd_tanh_dual_avx2` to amortize coefficient loading cost.
- [ ] **Hardware Division Preference:** Prefer `_mm256_div_ps` over manual Newton-Raphson reciprocal chains when target precision is `f32`.
- [ ] **Validate Parity & Lints:** Run `utils/lints.sh` and `cargo test` to ensure zero broken assertions across standard and fast precision modes.

---

## References

- Muller, J.-M. *Elementary Functions: Algorithms and Implementation*. 3rd ed. Birkhäuser, 2016. (Padé approximants)
- Intel® Intrinsics Guide — `_mm256_div_ps` instruction latency and throughput specifications.
- [Sollya](https://www.sollya.org/) — Software tool for computing optimal `fpminimax` polynomial coefficients.
- [docs/architecture.md](architecture.md) — System Architecture & Quality Modes.
- [docs/audio_fidelity_map.md](audio_fidelity_map.md) — Audio Fidelity and Parity Map.
- [docs/quality-contract.json](quality-contract.json) — Automated Quality Dashboard Baseline (JSON).
