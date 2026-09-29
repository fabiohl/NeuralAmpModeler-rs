<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# FastMath Approximations & Activation Precision Modes

Technical specification, numerical error bounds, and hardware implementation guidelines for transcendental activation functions (`tanh`, `sigmoid`, `silu`) and precision modes in the NeuralAmpModeler-rs DSP hot-path.

> [!IMPORTANT]
> The numerical kernels documented here govern production inference. Any modifications require validation via Criterion benchmarks (`cargo bench --bench math_bench`), parity verification against the C++ NAMcore reference, and confirmation against the numerical envelopes in [`quality-contract.json`](quality-contract.json).

---

## 1. Activation Precision Architecture

NeuralAmpModeler-rs provides runtime-selectable activation precision via the [`ActivationPrecision`](../src/math/activations/mod.rs) enum (`Standard` vs `Fast`). Precision selection is isolated per thread via Thread-Local Storage (`ACTIVE_MODEL_PRECISION`), preventing cross-stream state pollution in multi-instance DAW environments.

Thread-local state is managed through:

- [`set_activation_tls(mode)`](../src/math/activations/mod.rs): Sets precision for the calling thread.
- [`clear_activation_tls()`](../src/math/activations/mod.rs): Clears the override, reverting to `Standard`.
- [`set_thread_local_activation_precision(mode)`](../src/math/activations/mod.rs): RAII helper returning an [`ActivationPrecisionGuard`](../src/math/activations/mod.rs) that restores previous state on drop.
- [`activation_precision()`](../src/math/activations/mod.rs): Queries current precision, defaulting to `Standard` when unset.

### Precision Modes Summary

| Precision Mode | Tanh Kernel                            | Sigmoid Kernel                         | SiLU Kernel ($x \cdot \sigma(x)$)                                                | Max Absolute Error (vs `f32` ref)                                                | Throughput (256 elem, AVX2)       | Status                                      |
|:-------------- |:-------------------------------------- |:-------------------------------------- |:-------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------:|:---------------------------------:|:------------------------------------------- |
| **`Standard`** | Degree-6 Taylor minimax exp ($e^{2x}$) | Degree-6 Taylor minimax exp ($e^{-x}$) | Polynomial exp sigmoid ([`silu_slice_hf`](../src/math/activations/silu.rs))      | $\le 2.4 \times 10^{-7}$ ($\tanh$) $\le 2.1 \times 10^{-7}$ ($\sigma$)           | ~110 ns                           | **Universal Default** across all models     |
| **`Fast`**     | Padé [5,4] rational approximant        | Degree-17 Lawson minimax polynomial    | Minimax sigmoid multiplication ([`silu_slice`](../src/math/activations/silu.rs)) | $\approx 2.32 \times 10^{-3}$ ($\tanh$) $\approx 4.09 \times 10^{-4}$ ($\sigma$) | **~54 ns** (dual) ~63 ns (single) | Opt-in via `--activation fast` / host param |

```text
                              ┌──────────────────────────────────┐
                              │     activation_precision()       │
                              └─────────────────┬────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [Precision::Standard]                              [Precision::Fast]
              (Universal Default)                               (Opt-in Low CPU)
         ┌───────────────────────────┐                    ┌───────────────────────────┐
         │ • Degree-6 Taylor minimax │                    │ • Padé [5,4] rational     │
         │   range-reduced exp       │                    │   tanh (hardware div)     │
         │ • Exact-grade fidelity    │                    │ • Minimax deg-17 sigmoid  │
         │ • Error ≤ 2.4e-7          │                    │ • Error ≈ 2.32e-3         │
         │ • Full recurrent stability│                    │ • ~2× higher throughput   │
         └───────────────────────────┘                    └───────────────────────────┘
```

### 1.1 Standard Mode (`ActivationPrecision::Standard`, Production Default)

Standard mode uses range-reduced polynomial exponential kernels with integer exponent reconstruction. It provides near-machine-epsilon precision ($\le 2.4 \times 10^{-7}$), eliminating clamp discontinuities and numerical drift:

- **Tanh:** $\text{tanh}(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$, with single hardware division `_mm256_div_ps`.
- **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$, with single hardware division `_mm256_div_ps`.
- **SiLU:** $\text{silu}(x) = x \cdot \sigma(x)$.
- **Implementation:** [`src/math/activations/tanh/high_fidelity.rs`](../src/math/activations/tanh/high_fidelity.rs), [`src/math/activations/sigmoid/high_fidelity.rs`](../src/math/activations/sigmoid/high_fidelity.rs), and [`src/math/activations/silu.rs`](../src/math/activations/silu.rs).

### 1.2 Fast Mode (`ActivationPrecision::Fast`, Performance Opt-in)

Fast mode targets severely CPU-constrained setups or ultra-low buffer sizes:

- **Tanh:** Padé [5,4] rational approximant evaluated on $[-4, 4]$ with hardware division.
- **Sigmoid:** Degree-17 odd polynomial generated via Lawson's weighted minimax on $[-8, 8]$.
- **SiLU:** Direct multiplication of input by degree-17 minimax sigmoid.
- **Throughput:** ~54 ns for 256 elements on AVX2 (dual-register lane amortization).

> [!WARNING]
> **Recurrent State Drift in LSTMs under Fast Mode:** Fast mode approximations are bounded over compact intervals ($[-4, 4]$ for $\tanh$, $[-8, 8]$ for $\sigma$) with residual errors around $10^{-3}$. In recurrent architectures (LSTM), activation errors accumulate feedback-wise in the cell state $c_t$ and hidden state $h_t$ over thousands of consecutive samples. This causes audible tone degradation (Standard mode achieves an average **+89.5 dB SNR gain** over Fast mode in LSTM models). Fast mode is recommended primarily for feed-forward architectures (WaveNet, ConvNet); LSTM models should remain on Standard mode. Detailed topology-specific SNR measurements are catalogued in [`audio_fidelity_map.md §2`](audio_fidelity_map.md#2-activation-precision--standard-exact-grade-vs-fast-pad).

### 1.3 Interaction with Oversampling & Topology Dispatch

- **Oversampling Synergy:** In HQ mode (4× oversampling, see [`architecture.md`](architecture.md)), half-band filtering eliminates high-frequency aliasing. Residual harmonic distortion is governed by activation kernel accuracy, where Standard mode delivers $\text{SNR} > 120\text{ dB}$.
- **Full Model Coverage:** Runtime precision switching applies across all supported architectures: WaveNet (A1/A2), LSTM (1×N, 2×N, dynamic), ConvNet, and Linear models. This includes fused 4-gate LSTM GEMV kernels ([`src/math/lstm/gates.rs`](../src/math/lstm/gates.rs)) monomorphized over SIMD traits ([`src/math/common/traits/mod.rs`](../src/math/common/traits/mod.rs)).
- **Padé [5,4] vs C++ NAMcore `fast_tanh` Distinction:** `ActivationPrecision::Fast` governs the general-purpose approximations for `tanh` and `sigmoid`. In contrast, model files that explicitly specify `"FastTanh"` in their JSON topology metadata dispatch to [`src/math/activations/fast_tanh.rs`](../src/math/activations/fast_tanh.rs), reproducing upstream C++ NAMcore's Atkinson rational formula bit-for-bit (see §2.3).

---

## 2. Mathematical Formulations & Implementations

### 2.1 Standard Mode: Range-Reduced Polynomial Exp Kernels

#### Range Reduction & Polynomial Formulation

To evaluate $e^y$ accurately for single-precision floats over $[-20, 20]$:

$$k = \text{round}(y \cdot \log_2 e), \quad r = y - k \cdot \ln 2 \quad \text{where } r \in \left[-\frac{\ln 2}{2}, \frac{\ln 2}{2}\right]$$

The reduced argument $r$ is evaluated via a degree-6 Taylor minimax polynomial:

$$P(r) = \left(\left(\left(\left((c_6 \cdot r + c_5) \cdot r + c_4\right) \cdot r + c_3\right) \cdot r + c_2\right) \cdot r + 1\right) \cdot r + 1$$

The scale factor $2^k$ is reconstructed branchlessly by shifting the integer exponent directly into the IEEE 754 floating-point exponent field:

$$\text{scale} = 2^k = \text{reinterpret\_as\_f32}\left((k + 127) \ll 23\right)$$

$$e^y \approx P(r) \cdot \text{scale}$$

Coefficients (`POLY_EXP_C*`, `POLY_LOG2_E`, `POLY_LN2`) are defined in [`src/math/constants.rs`](../src/math/constants.rs).

#### Kernel Implementations

1. **Tanh ([`src/math/activations/tanh/high_fidelity.rs`](../src/math/activations/tanh/high_fidelity.rs)):**
   Sets $y = 2x$ clamped to $[-20, 20]$, evaluates $e^{2x}$, and executes a single division:
   $$\text{tanh}(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$$
   Max absolute error $\le 2.4 \times 10^{-7}$ across $[-20, 20]$.

2. **Sigmoid ([`src/math/activations/sigmoid/high_fidelity.rs`](../src/math/activations/sigmoid/high_fidelity.rs)):**
   Sets $y = -x$ clamped to $[-20, 20]$, evaluates $e^{-x}$, and executes a single division:
   $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
   Max absolute error $\le 2.1 \times 10^{-7}$ across $[-20, 20]$.

---

### 2.2 Fast Mode: Tanh (Padé [5,4] Rational Approximant)

#### Approximating Function

$$\text{tanh}(x) \approx \frac{x \cdot ((x^2 + 105) \cdot x^2 + 945)}{(15x^2 + 420) \cdot x^2 + 945} = \frac{x^5 + 105x^3 + 945x}{15x^4 + 420x^2 + 945}$$

Evaluated using Horner's scheme on $t = x^2$:

- Numerator: $N(x) = x \cdot ((t + 105) \cdot t + 945)$
- Denominator: $D(x) = (15t + 420) \cdot t + 945$

The input is clamped to $[-4, 4]$ and the output is clamped to $[-1, 1]$.

#### Implementation & Architecture

Implemented in [`src/math/activations/tanh/production.rs`](../src/math/activations/tanh/production.rs):

- `simd_tanh_avx2(x: __m256) -> __m256`: 8 floats, AVX2 + FMA.
- `simd_tanh_dual_avx2(x1, x2: __m256) -> (__m256, __m256)`: 16 floats simultaneously. Constants are broadcast once and shared between lanes, reducing register spills and improving memory throughput.
- `simd_tanh_avx512(x: __m512) -> __m512`: 16 floats, AVX-512 (behind `#[cfg(feature = "avx512")]`).
- `scalar_pade_tanh(x: f32) -> f32`: Remainder processing using `mul_add`.

#### Hardware Division (`_mm256_div_ps`) vs Newton-Raphson

On modern x86-64 microarchitectures, hardware vector division provides superior performance and accuracy compared to software reciprocal iteration:

| Kernel Variant                 | Max Abs Error ($[-4, 4]$) | Reciprocal Error          | Throughput (256 elem, AVX2)         | Micro-architectural Trade-off                          |
|:------------------------------ |:-------------------------:|:-------------------------:|:-----------------------------------:|:------------------------------------------------------ |
| **Padé Div (`_mm256_div_ps`)** | **$2.32 \times 10^{-3}$** | **Exact IEEE 754**        | **~54 ns (dual) / ~63 ns (single)** | Lowest latency (10–14 cycles), minimal code size       |
| Padé NR2 (`rcp` + 2× Newton)   | $2.32 \times 10^{-3}$     | Saturates 24-bit mantissa | ~104 ns                             | High instruction count, register pressure              |
| Padé NR1 (`rcp` + 1× Newton)   | $2.32 \times 10^{-3}$     | ~23-bit mantissa          | ~110 ns                             | Slower than hardware division on Zen 3+ and Intel Core |
| Piecewise 7-Segment Polynomial | $4.90 \times 10^{-3}$     | N/A (no div)              | ~163 ns                             | Evaluates all branches; Port 5 shuffle bottleneck      |

Double Newton-Raphson (NR2) reaches exact parity with hardware division but incurs higher instruction latency. Hardware division is simpler, faster, and exact.

---

### 2.3 Fast Mode: Sigmoid (Lawson Minimax Degree-17 Polynomial)

Rather than evaluating $\sigma(x) = 0.5 + 0.5 \cdot \text{tanh}(x/2)$ (which halves argument dynamic range and amplifies rational approximation errors), Fast mode uses a direct minimax odd polynomial optimized on $[-8, 8]$:

$$\sigma(x) \approx 0.5 + x \cdot \left(c_0 + c_1 x^2 + c_2 x^4 + c_3 x^6 + c_4 x^8 + c_5 x^{10} + c_6 x^{12} + c_7 x^{14} + c_8 x^{16}\right)$$

Input is clamped to $[-8, 8]$, and the output is clamped to $[0, 1]$.

Implemented in [`src/math/activations/sigmoid/production.rs`](../src/math/activations/sigmoid/production.rs):

| Metric                 | Tanh Identity Baseline        | Direct Minimax Polynomial (Degree 17)                 |
|:---------------------- |:-----------------------------:|:-----------------------------------------------------:|
| **Max Absolute Error** | $\approx 6.80 \times 10^{-4}$ | **$\approx 4.09 \times 10^{-4}$** (1.67× lower error) |
| **SIMD Operations**    | 16 ops (including division)   | **15 ops** (pure Horner FMA, zero division)           |
| **Output Bounds**      | $[0, 1]$                      | $[0, 1]$                                              |

---

### 2.4 Upstream C++ NAMcore Fast Tanh (`fast_tanh`, Atkinson Formula)

Distinct from `ActivationPrecision::Fast`, models explicitly specifying `"FastTanh"` topology activation in their model definition (such as specific WaveNet A2 or ConvNet layers) route to [`src/math/activations/fast_tanh.rs`](../src/math/activations/fast_tanh.rs). This matches C++ NAMcore (`NAM/activations.h`, `fast_tanh`) line-for-line using Atkinson's rational formula:

$$ax = |x|, \quad x^2 = x \cdot x$$

$$\text{fast\_tanh}(x) = \frac{x \cdot \left(c_a + c_a \cdot ax + (c_b + c_c \cdot ax) \cdot x^2\right)}{c_d + (c_d + x^2) \cdot |x + c_e \cdot x \cdot ax|}$$

Coefficients:

- $c_a = 2.45550750702956$
- $c_b = 0.893229853513558$
- $c_c = 0.821226666969744$
- $c_d = 2.44506634652299$
- $c_e = 0.814642734961073$

Implemented via `fast_tanh_slice_avx2`, `fast_tanh_slice_avx512`, and scalar `fast_tanh(x)`. This strict separation guarantees bit-exact output parity for models trained specifically with NAM's original `FastTanh` layer.

---

## 3. Micro-Architectural Decisions & Invariants

### 3.1 Rejection of Piecewise Polynomial Approximations

Evaluating 7 polynomials of degree 5 blended branchlessly via `_mm256_blendv_ps` was rejected:

1. **Unconditional Evaluation:** Branchless blending requires evaluating all 7 polynomials simultaneously across the vector register.
2. **Execution Port Saturation:** Cascaded `_mm256_blendv_ps` instructions saturate execution Port 5 (shuffle/blend unit) on x86 microarchitectures.
3. **Throughput Deficit:** Piecewise blending required ~163 ns for 256 elements (+159% latency vs Padé hardware division at ~63 ns) while yielding a worse error ($4.90 \times 10^{-3}$ vs $2.32 \times 10^{-3}$).

### 3.2 Single-Mode Native `f32` Weights

All neural network layers operate natively on single-precision `f32` arrays:

- **No Hot-Path Compression:** Half-precision (`f16c` / `bf16`) weight compression was removed from inference paths. The decompression latency, EVEX prefix overhead, and recurrent error accumulation ($0.1\%\text{--}0.5\%$ per layer) outweighed memory bandwidth savings (see [`audio_fidelity_map.md §1`](audio_fidelity_map.md#1-native-f32-weight-representation--numerical-error-budgets)).
- **Interoperability Parity:** Standard mode native `f32` inference achieves $\text{SNR} > 130\text{ dB}$ (ESR $\sim 10^{-13}$ to $10^{-14}$) compared to upstream C++ NAMcore (Standard: 136.4 dB SNR, Feather: 133.2 dB SNR, Nano: 131.9 dB SNR).
- **Buffer Alignment:** Delay line history circular buffers use [`MirroredBuffer::new_aligned`](../src/dsp/mirror_buf/alloc.rs) to ensure that the allocated element count is strictly divisible by the channel count (e.g. WaveNet Lite CH=12), eliminating stride misalignments across virtual memory mirror seams.

---

## 4. Real-Time Audio Policies (Silence, Subnormals, and IEEE Compliance)

### 4.1 WaveNet Non-Zero Silence Policy

Under continuous zero input, WaveNet models produce a static DC residual output of approximately $3.58 \times 10^{-5}$ ($-89\text{ dBFS}$).

- **Root Cause:** Direct mathematical consequence of accumulating non-zero Conv1D bias terms ($0.001$ per layer across 12 layers) through $1 \times 1$ projections and `head_scale` ($0.1$).
- **Engine Policy:** Consistent with C++ NAMcore (`NAM/dsp.h`). The inference hot-path does **not** clamp small outputs to zero, preserving authentic analog noise floors and saturation characteristics. Gating is managed strictly by the DSP noise gate stage ([`src/dsp/gate.rs`](../src/dsp/gate.rs)).

### 4.2 Subnormal Prevention via DC Dither

To prevent CPU soft-emulation penalties when handling subnormal (denormal) floating-point numbers:

- A constant offset `DENORMAL_DITHER_OFFSET = 1.0e-11` ($-220\text{ dBFS}$) is added during input conditioning ([`src/dsp/pipeline/stages/input.rs`](../src/dsp/pipeline/stages/input.rs)).
- The exact offset is subtracted in the output stage ([`src/dsp/pipeline/stages/output.rs`](../src/dsp/pipeline/stages/output.rs)) within the fused gain, clipping, and gating SIMD loop (`apply_gain_with_dither_and_detect_clipping_*`).
- The offset lies $76\text{ dB}$ below the 24-bit DAC noise floor, making it inaudible and introducing zero extra memory passes.

### 4.3 Prohibition of Unspecified Algebraic Float Operations (`algebraic_*`)

Rust 1.98 stabilized algebraic floating-point methods (`f32::algebraic_*` / `f64::algebraic_*`). These methods permit compiler reassociation, reciprocal substitutions, and non-deterministic optimizations analogous to `-ffast-math`.

**Strictly Prohibited:** `algebraic_*` methods must **never** be used in any module subject to oracle validation, bit-exact golden tests, or real-time DSP pipelines:

- Math primitives under [`src/math/`](../src/math/) (GEMM, activations, LSTM gates, dot products);
- Model inference under [`src/models/`](../src/models/);
- DSP processing under [`src/dsp/`](../src/dsp/) (resampling, filters, cabsim, noise gate);
- Kahan summation routines ([`kahan_add`](../src/math/common/scalar_ref/dot.rs)), which rely strictly on non-associative IEEE 754 addition.

Hot-path multiply-accumulate operations must use explicit IEEE 754 FMA (`mul_add` or `_mm256_fmadd_ps`).

### 4.4 Hardware DAZ / FTZ Enforcement

Denormals-Are-Zero (DAZ) and Flush-To-Zero (FTZ) flags are enforced at the DSP boundary:

- Helper function `set_daz_ftz()` in [`src/math/common/ops.rs`](../src/math/common/ops.rs) sets bits 6 (DAZ) and 15 (FTZ) in the SSE `MXCSR` register.
- Reasserted at the entry of audio processing in [`src/dsp/pipeline/capture.rs`](../src/dsp/pipeline/capture.rs) (`capture_dsp_pipeline`) via a fixed `stmxcsr`/`ldmxcsr` pair outside sample loops.

---

## 5. Normative Developer Checklist

When modifying or introducing activation kernels:

- [ ] **Odd Symmetry:** Verify $f(-x) == -f(x)$ for all odd functions ($\tanh$, `fast_tanh`) to prevent artificial DC bias.
- [ ] **Dual-Register Lane Amortization:** Maintain paired SIMD evaluations (such as `simd_tanh_dual_avx2` and `simd_sigmoid_dual_avx2`) to amortize coefficient broadcast and load instructions across 16 floats.
- [ ] **Hardware Division:** Use `_mm256_div_ps` for rational kernels; do not replace it with Newton-Raphson reciprocal chains on `f32` paths.
- [ ] **IEEE 754 Determinism:** Do not use `f32::algebraic_*` or `f64::algebraic_*` methods (see §4.3).
- [ ] **Performance Validation:** Run `cargo bench --bench math_bench` to confirm that throughput remains within expected envelope baselines (~110 ns for Standard, ~54 ns for Fast per 256 elements).
- [ ] **Fidelity & Regression Gate:** Run `utils/tests-quick.sh` and verify that quality metrics stay within baseline contracts in [`quality-contract.json`](quality-contract.json).

---

## References

- Muller, J.-M. *Elementary Functions: Algorithms and Implementation*. 3rd ed. Birkhäuser, 2016. (Padé approximation theory and range reduction).
- [Sollya](https://www.sollya.org/) — Software tool for computing certified minimax polynomial approximations.
- Intel® 64 and IA-32 Architectures Optimization Reference Manual — Instruction latencies and port distribution.
- [`architecture.md`](architecture.md) — System Architecture, Quality Modes, and Memory Layout.
- [`audio_fidelity_map.md`](audio_fidelity_map.md) — Canonical Audio Fidelity, Parity, and Distortion Trade-offs.
- [`benchmarks.md`](benchmarks.md) — Criterion Performance Benchmarking Guide.
- [`testing.md`](testing.md) — Verification Suite Structure and Quality Gates.
- [`quality-contract.json`](quality-contract.json) — Automated Quality Dashboard Baseline Envelopes.
