<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Audio Fidelity Map — Off-Spec DSP Design Decisions

Every engineering decision in `nam-rs` that lies **outside the strict NAM model specification** and
influences audio fidelity, real-time (RT) safety, or user experience is catalogued here. For each factor:
what it is, whether it is mandatory or optional, the sonic and performance impact, and where to
find the implementation.

The NAM specification defines only: topology (WaveNet, LSTM, A2, ConvNet), stored weight values,
and the mathematical forward pass. Everything below is a `nam-rs` implementation choice — not part
of the `.nam` / `.namb` file format contract.

---

## Quick Reference

| #   | Factor                                              | Spec? | Mandatory?                         | User-Controllable?       | Quality Impact                                                                          | Status      |
|:---:|:--------------------------------------------------- |:-----:|:----------------------------------:|:------------------------:|:--------------------------------------------------------------------------------------- |:-----------:|
| 1   | **Native f32 weights (Weight compression removed)** | ❌    | Was under review — removed         | ❌ No                    | Matches NAMCore native f32 representation; eliminates L1 decompression penalty          | ✅ Active   |
| 2   | **Activation precision (Standard vs Fast Padé)**    | ❌    | ✅ Default (Standard); Fast opt-in | ✅ Host / CLI            | Standard (exact-grade): ~110–144 dB SNR; Fast: −53 dB error (WaveNet) / degraded (LSTM) | ✅ Active   |
| 3   | **LSTM recurrent state precision**                  | ❌    | Partial (model-dependent)          | ✅ HF gates + Kahan head | Interop ESR ≈ 7.86e-13 to 1.00e-11 (Standard); vs f64 ideal: 5.68e-13 to 2.71e-12       | ✅ Resolved |
| 4   | **Host sample rate polyphase resampler**            | ❌    | ✅ When host ≠ 48 kHz              | ❌ No†                   | Passband ripple < 0.05 dB; stopband filter design ≥ 105 dB                              | ✅ Active   |
| 5   | **Neural stage oversampling (HQ Mode)**             | ❌    | ❌ Off by default                  | ✅ Host / CLI            | Suppresses non-linear aliasing; adds 12/24 samples latency                              | ✅ Active   |
| 6   | **Denormal prevention (Dither + FTZ/DAZ)**          | ❌    | ✅ Yes                             | ❌ No                    | Zero audible impact (−220 dBFS offset); prevents CPU microcode slowdown                 | ✅ Active   |
| 7   | **Adaptive Compute (quality fallback FSM)**         | ❌    | ✅ Default                         | 🔶 `--slim` flag         | Prevents xruns under CPU spikes via tier fallback                                       | ✅ Active   |

† Resampler quality (32-tap vs 64-tap) was evaluated and fixed to 64-tap HQ — see §4. Activation precision is exposed via host parameters and CLI (`--activation fast|standard`). Standard (exact-grade) is the universal production default across all model families.

---

## 1. Native f32 Weight Representation

**What it is.** All neural model weight matrices in `nam-rs` are stored and processed natively as `f32` vectors, matching the reference NAMCore engine (`Eigen::MatrixXf`/`VectorXf`).

**Rationale for removing weight compression.** An earlier optimization using half-precision (`f16c`/`bf16`) weight compression was evaluated and removed. Quantitative profiling confirmed that `f16c` decompression overhead on L1 cache outweighed any bandwidth benefits for LSTM models, while introducing artificial interop drift in recurrent models. Weights are loaded directly as `f32` across all model families.

**Implementation.** Weight loading and vector storage in [src/models/](../src/models/) (`set_weights.rs` and `model.rs` per architecture). Utility modules in [src/math/common/half.rs](../src/math/common/half.rs) are retained exclusively for benchmark and test reference.

---

## 2. Activation Precision — Standard (exact-grade) vs Fast (Padé)

**What it is.** Neural models rely on non-linear activations (`tanh`, `sigmoid`). `nam-rs` provides two approximation modes controlled via Thread-Local Storage (`ACTIVE_MODEL_PRECISION` TLS) in [`src/math/activations/mod.rs`](../src/math/activations/mod.rs):

| Mode                   | Activation Kernel                            | Max Absolute Error      | Approx Error (dBFS) | Compute Impact  |
|:---------------------- |:-------------------------------------------- |:-----------------------:|:-------------------:|:---------------:|
| **Fast** (opt-in)      | Padé [5,4] `tanh`, clamped \|x\| ≤ 4         | 2.32e-3                 | ≈ −53 dB            | Baseline        |
| **Fast** (opt-in)      | Minimax degree-17 sigmoid                    | 4.09e-4                 | ≈ −68 dB            | Baseline        |
| **Standard** (default) | Taylor-based `exp` kernels, degree-6 minimax | ~2.4e-7 (10,000× lower) | ≈ −133 dB           | +10–15% compute |

The Padé approximation clamp at `|x| > 4` introduces a derivative discontinuity that generates weak spectral aliasing at extreme gain settings. The minimax sigmoid has no clamp discontinuity, and the **Standard** kernels eliminate clamp discontinuities entirely.

### 2.1 Fast mode — opt-in for CPU-constrained setups

Exact scalar `f32::tanh` costs ~150 ns per 256-element AVX2 vector on x86-64. The Padé kernel runs in ~10 ns. For large WaveNet models processing hundreds of thousands of activations per 1.3 ms RT window, Fast mode (`--activation fast`) provides an explicit opt-in performance fallback for low-power CPUs.

**Deprecation Advisory for LSTM models:** In recurrent architectures (LSTM), activation errors accumulate feedback-wise in the hidden state vector $h_t$ over time. Under Fast (Padé) mode, LSTM gate errors compound to ~15.9–29.3 dB SNR vs reference (quality degradation), whereas Standard mode achieves 103.2–120.5 dB SNR:

| Model Topology          | Fast Mode SNR (Padé) | Standard Mode SNR (Exact) | Δ SNR Gain   |
|:----------------------- |:-------------------- |:------------------------- |:------------ |
| **LSTM 1×16**           | 15.9 dB              | 103.2 dB                  | **+87.3 dB** |
| **LSTM 2×8**            | 24.1 dB              | 114.0 dB                  | **+89.9 dB** |
| **Official lstm (H=3)** | 29.3 dB              | 120.5 dB                  | **+91.2 dB** |

Because LSTM execution is dominated by GEMV operations rather than activation math, the 10–15% compute saving is negligible compared to the fidelity loss (+89.5 dB average SNR gain with Standard). The CLI emits a warning when `--activation fast` is combined with an LSTM architecture.

### 2.2 Standard mode — universal default

`ActivationPrecision::Standard` uses Taylor-based polynomial `exp` kernels for `tanh` and `sigmoid` (max error ≈ 2.4e-7, ~10,000× lower than Fast mode). It is active by default from engine startup across all model families (WaveNet A1/A2, ConvNet, Linear, LSTM).

Runtime switching is supported without audio thread allocation via host parameters or CLI (`--activation standard|fast`), with offline-render mode enforcing `Standard` automatically.

### 2.3 Interaction with Oversampling

Standard mode is most effective when paired with 4× neural stage oversampling (§5): oversampling suppresses non-linear folded harmonics via half-band filtering, while Standard exact-grade activations eliminate high-order polynomial folding residual errors.

**Implementation.** [`src/math/activations/mod.rs`](../src/math/activations/mod.rs), [`src/math/activations/tanh/production.rs`](../src/math/activations/tanh/production.rs) (Fast mode), [`src/math/activations/tanh/high_fidelity.rs`](../src/math/activations/tanh/high_fidelity.rs) (Standard mode), [`src/math/activations/sigmoid/production.rs`](../src/math/activations/sigmoid/production.rs) (Fast mode), [`src/math/activations/sigmoid/high_fidelity.rs`](../src/math/activations/sigmoid/high_fidelity.rs) (Standard mode). Full mathematical analysis in [`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 3. LSTM Recurrent State Precision & Interop Parity

**Measured Interop Parity.** Under `ActivationPrecision::Standard`, recurrent state drift between `nam-rs` and reference NAMCore is eliminated across all LSTM model variants:

| Model                   | ESR vs NAMCore (Standard) | SNR vs NAMCore | ESR vs Ideal (f64 Oracle) | Status                   |
|:----------------------- |:-------------------------:|:--------------:|:-------------------------:|:------------------------:|
| **BossLSTM-1×16**       | **8.50e-12**              | 110.7 dB       | **8.90e-13**              | ✅ Bit-identical interop |
| **BossLSTM-2×8**        | **1.00e-11**              | 110.0 dB       | **5.68e-13**              | ✅ Bit-identical interop |
| **Official lstm (H=3)** | **7.86e-13**              | 121.0 dB       | **2.71e-12**              | ✅ Bit-identical interop |

*Note: All values measured with 24,000-sample warm-up prewarm in canonical live mode ([`docs/quality-contract.txt`](quality-contract.txt)).*

### 3.1 Steady-State Prewarm vs Cold-Start Decomposition

A critical distinction must be drawn between steady-state fidelity and cold-start unit testing:

1. **Canonical Steady-State (Prewarmed):** Measured after a 24,000-sample warmup period. In steady-state regime, `nam-rs` matches NAMCore to float32 precision limits (ESR ≈ 1e-11 to 1e-13, SNR 110–121 dB) and tracks the mathematical `f64` oracle to ESR ≈ 5.7e-13 to 2.7e-12.
2. **Cold-Start Decomposition (256 samples without prewarm):** Short-window tests (`test_decomposition_*` in `tests/parity/reference_oracle_f64.rs`) measure initial buffer-filling transients for architectures whose receptive field or recurrent memory exceeds 256 samples. These transient numbers reflect cold state initialization, not the steady-state precision floor. Consult [`docs/perceptual_validation.md`](perceptual_validation.md) §Decomposition Cold-Start for methodological details.

#### Empirical Cold-Start Error Decomposition (`quality-contract.txt`)

| Model                | Total ESR (f32 vs f64) | ΔESR F16C Weights  | ΔESR BF16 Weights  | ΔESR Padé Activation | ΔESR F32 Accumulation |
|:-------------------- |:---------------------- |:------------------ |:------------------ |:-------------------- |:--------------------- |
| **LSTM-H3**          | 2.59e-3 (-25.9 dB)     | 3.64e-5 (-44.4 dB) | 9.97e-5 (-40.0 dB) | 2.58e-3 (-25.9 dB)   | 6.43e-13 (-121.9 dB)  |
| **BossLSTM-2×8**     | 1.73e-3 (-27.6 dB)     | 4.91e-7 (-63.1 dB) | 5.83e-4 (-32.3 dB) | 1.74e-3 (-27.6 dB)   | 2.00e-13 (-127.0 dB)  |
| **BossLSTM-1×16**    | 5.06e-2 (-13.0 dB)     | 3.16e-6 (-55.0 dB) | 3.39e-3 (-24.7 dB) | 4.81e-2 (-13.2 dB)   | 2.79e-12 (-115.5 dB)  |
| **A2-Lite**          | 2.22e-14 (-136.5 dB)   | 2.81e-7 (-65.5 dB) | 8.86e-5 (-40.5 dB) | 0.00e0 (-inf dB)     | 8.02e-14 (-131.0 dB)  |
| **WaveNet-official** | 1.82e-12 (-117.4 dB)   | 3.71e-6 (-54.3 dB) | 5.41e-4 (-32.7 dB) | 3.66e-14 (-134.4 dB) | 1.03e-12 (-119.9 dB)  |
| **ConvNet-test**     | 3.57e-15 (-144.5 dB)   | 6.28e-8 (-72.0 dB) | 5.26e-7 (-62.8 dB) | 4.74e-33 (-323.2 dB) | 3.56e-15 (-144.5 dB)  |

### 3.2 Key Recurrent Mitigations

Three core mechanisms maintain high recurrent precision in LSTMs:

- **Exact-Grade Gate Activations:** Exp-based polynomial kernels (~2.4e-7 error) across scalar, AVX2, and AVX-512 LSTM gate paths prevent error propagation in state vector $h_t$.
- **Kahan-Compensated Head Projection:** Head projection accumulation ($H \to 1$) uses Kahan compensated summation, yielding ~2 dB SNR gain in deep heads.
- **Fixed Delay Constraint (Oversampling Exclusion):** External oversampling is **not recommended** for LSTM models. Unlike feedforward WaveNet models, LSTM recurrent feedback delays are fixed in absolute sample counts — oversampling by 2×/4× shortens the physical time-constant window, altering timbre drastically.

**Implementation.** [`src/models/lstm/layer_kernels.rs`](../src/models/lstm/layer_kernels.rs).

---

## 4. Host Sample Rate Adaptation (Polyphase Sinc Resampler)

**What it is.** NAM models are trained at 48 kHz. When host DAW software operates at a different sample rate (e.g., 44.1 kHz, 88.2 kHz, 96 kHz, 192 kHz), `nam-rs` performs rate conversion using a native minimum-phase polyphase FIR sinc resampler ([`src/dsp/resampler/mod.rs`](../src/dsp/resampler/mod.rs)).

**Configuration.** 256 phases × 64 taps, Kaiser window ($\beta = 12$), minimum-phase filter by default. A linear-phase variant is available internally for offline processing.

**Bypass path.** When `host_rate == 48 kHz`, a zero-cost bypass path forwards audio buffers directly without filter evaluation or memory copying.

**Performance & Quality Profile:**

| Metric                 | Value / Benchmark Result                                                              |
|:---------------------- |:------------------------------------------------------------------------------------- |
| Passband ripple        | < 0.05 dB (0 to 0.45 × Nyquist)                                                       |
| Stopband attenuation   | Filter design ≥ 105 dB; end-to-end multitone SNR ~31 dB (minimum-phase, gate ≥ 25 dB) |
| High-frequency rolloff | < 0.05 dB at 20 kHz (44.1 kHz host rate)                                              |
| Group delay            | Asymmetric minimum-phase response with minimal high-frequency dispersion              |

**Rejection of 32-tap mode.** A 32-tap resampler variant was benchmarked and discarded. While saving only ~40 ns per 64-sample block (< 0.1% total pipeline execution time), 32 taps degraded passband SNR from $\ge 100\text{ dB}$ down to $\sim 24\text{ dB}$. The 64-tap configuration is the permanent production standard.

**Implementation.** [`src/dsp/resampler/mod.rs`](../src/dsp/resampler/mod.rs), [`src/dsp/sinc_kernel.rs`](../src/dsp/sinc_kernel.rs).

---

## 5. Neural Stage Oversampling (HQ Mode)

**What it is.** Optional 2× or 4× oversampling surrounding neural model inference to suppress spectral aliasing generated by non-linear activations (`tanh`, `sigmoid`, `ReLU`). Based on Kahles, Esqueda & Välimäki (JAES 2019).

**Architecture.** Multi-stage half-band filtering using 25-tap Kaiser FIR filters ($\beta = 12$, >100 dB stopband attenuation). The half-band property zeros alternate coefficients, halving multiplication requirements.

Pipeline: `Upsample FIR stage(s) → Model Inference (at 2×/4× rate) → Downsample FIR stage(s)`.

| Mode          | Stages | Added Latency                                | Relative CPU Cost |
|:------------- |:------:|:--------------------------------------------:|:-----------------:|
| Off (default) | 0      | 0 samples                                    | 1.0×              |
| 2×            | 1      | 12 samples @ native rate (~0.25 ms @ 48 kHz) | ~2.0× model cost  |
| 4×            | 2      | 24 samples @ native rate (~0.50 ms @ 48 kHz) | ~4.0× model cost  |

Latency is reported dynamically to the host via `OversampleEngine::latency_samples()`. Host applications integrate latency announcements and dynamic updates via the SPSC rebuild cascade.

**User Control:**

- CLI: `--oversample off|2x|4x` (alias `--os`)
- Host parameter: Oversampling parameter (stepped enum, state-persisted)
- Mode changes trigger lock-free SPSC garbage-collected engine rebuilds off the real-time thread.

**ADAA Rejection Rationale.** Antiderivative Anti-Aliasing (ADAA) requires analytical antiderivatives per activation function, conflicting with `nam-rs`'s generic SIMD dispatch macro (`dispatch_simd!`) and multi-architecture model dispatcher. Half-band FIR oversampling is activation-agnostic and universally compatible across all topologies.

**Implementation.** [`src/dsp/oversample.rs`](../src/dsp/oversample.rs) (`OversampleEngine`), [`src/dsp/pipeline/stages/inference.rs`](../src/dsp/pipeline/stages/inference.rs).

---

## 6. Denormal Prevention — Dither + FTZ/DAZ

**What it is.** Two complementary mechanisms prevent subnormal (denormal) floating-point numbers from entering neural network state buffers. Denormals cause microcode exceptions on x86-64 processors, causing 10–100× CPU execution spikes that break real-time guarantees.

### 6.1 Deterministic Dither Offset

A fixed offset `DENORMAL_DITHER_OFFSET = 1.0e-11` (−220 dBFS) is injected into input samples prior to inference and subtracted from output samples after inference ([`src/dsp/pipeline/stages/input.rs`](../src/dsp/pipeline/stages/input.rs), [`src/dsp/pipeline/stages/output.rs`](../src/dsp/pipeline/stages/output.rs)).

Because the exact same constant is added and subtracted symmetrically, cancellation is bit-exact with zero residual DC drift or noise floor elevation.

### 6.2 Hardware FTZ/DAZ (MXCSR Register)

The helper `set_daz_ftz()` in [`src/math/common/ops.rs`](../src/math/common/ops.rs) configures SSE2 MXCSR control register flags:

- **FTZ (Flush-To-Zero):** Output subnormals flush to positive zero.
- **DAZ (Denormals-Are-Zero):** Input subnormals are read as zero.

Applied on the first audio block of `process()` and refreshed every 1024 blocks to guard against host thread state resets. Active unconditionally with zero audible impact.

---

## 7. Adaptive Compute (Quality Fallback FSM)

**What it is.** When audio thread p99 block processing latency exceeds real-time safety thresholds (1.33 ms at 48 kHz / 64 samples), the Adaptive Compute finite state machine (FSM) downgrades model quality tiers (Full → Reduced → Minimal) to guarantee uninterrupted audio rendering.

**Transition Mechanics:**

- **WaveNet A1 Models:** Utilize double-pass inference during quality tier changes to crossfade smoothly between sub-models without click artifacts.
- **WaveNet A2 Models (A2-Full, A2-Lite, A2-Dyn):** Do not support layer-skip mechanisms required for double-pass inference. A2 architectures execute single-pass direct state transitions to preserve recurrent history integrity.

**User Control:**

- CLI: `--slim auto|full|lite`
- Host parameter: Dedicated adaptive compute control. Setting `--slim full` disables dynamic fallback.

**Implementation.** [`src/dsp/adaptive.rs`](../src/dsp/adaptive.rs), [`src/models/static_model.rs`](../src/models/static_model.rs) (`supports_layer_skip()`).

---

## 8. Governance & Quality Contract Verification

All fidelity, SNR, and performance claims in this document are governed by the automated testing supply chain specified in [`tests/fixtures/README.md`](../tests/fixtures/README.md):

| Governance Layer                | Verification Mechanism                                                                   | Enforcement Gate                 |
|:------------------------------- |:---------------------------------------------------------------------------------------- |:-------------------------------- |
| **Layer 0 — Golden Generation** | `tests/fixtures/golden_gen_build.sh` + pinned reference commit (`1f42f88`, tag `v0.5.4`) | Operational contract             |
| **Layer 1 — Pre-committed**     | `tests/models/golden_vectors.rs` — verifies Rust output against committed binary goldens | `utils/tests-quick.sh` (Phase 2) |
| **Layer 2 — Live Parity**       | `tests/parity/cpp_parity.rs` — cross-engine C++ parity execution                         | `utils/tests-long.sh`            |

**Freshness Manifest:** `tests/fixtures/.golden_manifest.sha256` is enforced as a hard gate by `utils/tests-quick.sh`.

Live dashboard measurements are updated via `utils/quality-dashboard.sh` and recorded in [`docs/quality-contract.txt`](quality-contract.txt).

---

## 9. Architectural Rationale Archive

Key technical trade-offs validated during `nam-rs` development:

- **Native f32 Weights:** Removal of f16c weight quantization eliminated L1 cache decompression overhead and restored bit-identical LSTM interop parity with NAMCore.
- **64-Tap Polyphase Resampler:** Benchmark analysis demonstrated that 32-tap filtering saved < 0.1% CPU (~40 ns/block) while causing catastrophic passband SNR degradation (~24 dB vs ≥100 dB).
- **Exact-Grade Activation Default:** Standard mode Taylor/minimax exp kernels cost +10–15% activation compute while delivering +89.5 dB average SNR improvement across LSTM models.
- **Half-Band FIR Oversampling:** Selected over Antiderivative Anti-Aliasing (ADAA) to maintain universal compatibility with polymorphically dispatched SIMD neural kernels.

---

## See Also

- [`docs/fastmath-approximations.md`](fastmath-approximations.md) — Detailed Padé/minimax numerical bounds and error profiles
- [`docs/perceptual_validation.md`](perceptual_validation.md) — Measurement methodology, ESR thresholds, and cold-start analysis
- [`docs/architecture.md`](architecture.md) — Architectural overview, pipeline flow, and memory layouts
- [`docs/research-references.md`](research-references.md) — Scientific references (Kahles 2019, Sato & Smith 2025, etc.)
- [`docs/quality-contract.txt`](quality-contract.txt) — Automated Quality Dashboard Baseline
- [`tests/fixtures/README.md`](../tests/fixtures/README.md) — Golden vector test supply chain contract
