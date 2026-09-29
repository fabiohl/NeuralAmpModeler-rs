<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Audio Fidelity Map — Off-Spec DSP Design Decisions

This document catalogues all architectural and DSP engineering decisions in `NeuralAmpModeler-rs` that operate **outside the minimal NAM model specification** and influence audio fidelity, real-time (RT) safety, computational overhead, or latency.

The upstream `.nam` and `.namb` specifications define only: network topology (WaveNet, LSTM, ConvNet, Linear), stored floating-point weights, and the mathematical forward pass. All buffering, activation kernel approximations, rate conversions, oversampling schemes, denormal prevention, and convolution partitioning choices described below are implementation choices of NeuralAmpModeler-rs.

---

## Quick Reference Summary

| #     | Design Factor               | Spec? | Mandatory?                       | User Control       | Audio Fidelity & Latency Impact                                                                                             | Operational Status |
|:-----:|:--------------------------- |:-----:|:--------------------------------:|:------------------:|:--------------------------------------------------------------------------------------------------------------------------- |:------------------:|
| **1** | **Native f32 Weights**      | ❌    | ✅ Yes                           | ❌ No              | Unquantized FP32 matching NAMCore; eliminates quantization drift and L1 cache decompression penalty.                        | ✅ Active          |
| **2** | **Activation Precision**    | ❌    | ✅ Default (Standard)            | ✅ Host / CLI      | Standard (exact polynomial): ~103–150 dB SNR; Fast (Padé): −53 dB error (WaveNet) / recurrent drift (LSTM).                 | ✅ Active          |
| **3** | **LSTM State Precision**    | ❌    | ✅ Yes                           | ❌ No              | Interop ESR $\sim 10^{-11} \text{ to } 10^{-13}$ vs NAMCore; tracks f64 oracle to $\sim 10^{-12} \text{ to } 10^{-13}$.     | ✅ Active          |
| **4** | **Polyphase Resampler**     | ❌    | ✅ When $f_s \neq 48\text{ kHz}$ | ❌ No              | 64-tap minimum-phase sinc FIR; passband ripple $< 0.05\text{ dB}$; stopband attenuation $\ge 105\text{ dB}$. 48 kHz bypass. | ✅ Active          |
| **5** | **Neural Oversampling**     | ❌    | ❌ Off by default                | ✅ Host / CLI      | Multi-stage Kaiser half-band FIR; suppresses aliasing; adds 12/24 samples latency; recurrent time-constant shift in LSTMs.  | ✅ Active          |
| **6** | **Denormal Prevention**     | ❌    | ✅ Yes                           | ❌ No              | Symmetrical $\pm 10^{-11}$ dither offset ($-220\text{ dBFS}$) + hardware MXCSR FTZ/DAZ; zero CPU microcode stalls.          | ✅ Active          |
| **7** | **Adaptive Compute**        | ❌    | ✅ Default (Auto)                | ✅ `--slim` / Host | Graceful fallback FSM (Full → Reduced → Minimal) under CPU spikes to prevent xruns.                                         | ✅ Active          |
| **8** | **CabSim UPOLS Partitions** | ❌    | ✅ Host policy                   | ✅ Host / CLI      | Uniform-partitioned overlap-save FIR; zero frequency distortion; trades latency ($P$ samples) vs FFT event rate ($f_s/P$).  | ✅ Active          |

---

## 1. Native f32 Weight Representation & Precision Architecture

**Architecture:** All neural model weights in NeuralAmpModeler-rs are stored and evaluated in native single-precision floating-point (`f32`) vectors with 64-byte alignment (`AlignedVec<f32>`), matching upstream `NeuralAmpModelerCore` (`Eigen::MatrixXf`/`VectorXf`).

### Rejected Alternative: Half-Precision Weight Quantization (f16c / bfloat16)

Half-precision weight storage was evaluated and explicitly rejected for production inference:

1. **Significand Bit Width & Machine Epsilon:** Standard `f32` provides a 24-bit significand (23 stored + 1 hidden bit) with machine epsilon $\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$ ($\text{SNR} > 140\text{ dB}$). Conversely, `bfloat16` allocates only 8 bits to its significand (7 stored + 1 hidden bit), yielding $\epsilon_{\text{mach}} \approx 3.91 \times 10^{-3}$ and $1\text{ ULP} \approx 0.781\%$ relative precision ($\text{SNR} \approx 45\text{ dB}$).
2. **Multiplicative Error Compounding:** In recurrent topologies (LSTM) and deep convolutional cascades (WaveNet 23-layer), truncation errors compound multiplicatively across layers, degrading audio clarity and causing acoustic drift.
3. **Cache & SIMD Throughput Penalties:** Profiling on x86-64-v3 architectures demonstrates that on-the-fly decompression overhead in L1 cache and instruction prefix decoding out-tax any memory bandwidth savings.
4. **Upstream Interoperability:** NAMCore operates natively in `f32`. Quantizing weights introduces an irreducible interop error floor of $\sim -45\text{ to } -65\text{ dB}$.

**Implementation:** Weight loading in [`src/loader/`](../src/loader/) and model storage in [`src/models/`](../src/models/). Utility routines in [`src/math/common/half.rs`](../src/math/common/half.rs) are retained strictly for offline error decomposition benchmarks.

---

## 2. Activation Precision: Standard (Exact-Grade) vs. Fast (Padé)

**Architecture:** Non-linear neural activations (`tanh`, `sigmoid`, `silu`) run in two runtime-selectable modes managed via Thread-Local Storage (`ACTIVE_MODEL_PRECISION` TLS) in [`src/math/activations/mod.rs`](../src/math/activations/mod.rs):

| Precision Mode           | Tanh Kernel                                | Sigmoid Kernel               | Max Absolute Error            | Compute Overhead   | Primary Scope                       |
|:------------------------ |:------------------------------------------ |:---------------------------- |:-----------------------------:|:------------------:|:----------------------------------- |
| **`Standard`** (Default) | Degree-6 Taylor minimax exp                | Degree-6 Taylor minimax exp  | $\le 2.4 \times 10^{-7}$      | $+10\text{--}15\%$ | Universal default across all models |
| **`Fast`** (Opt-in)      | Padé [5,4] rational, clamped $\|x\| \le 4$ | Degree-17 minimax polynomial | $\approx 2.32 \times 10^{-3}$ | Baseline           | Low-power CPU fallback              |

The Padé approximation clamp at $|x| > 4$ introduces a derivative discontinuity that generates weak spectral artifacts at extreme gain. Standard kernels eliminate clamp discontinuities entirely.

### 2.1 Impact on Recurrent Architectures (LSTM)

In recurrent networks, hidden state vectors ($h_t$) accumulate activation approximation errors recursively step by step. Under Fast (Padé) mode, gate approximation errors compound, resulting in substantial audio degradation:

| Model Topology          | Fast Mode SNR (Padé) | Standard Mode SNR (Exact) | Δ SNR Gain with Standard |
|:----------------------- |:--------------------:|:-------------------------:|:------------------------:|
| **LSTM 1×16**           | $15.9\text{ dB}$     | $103.2\text{ dB}$         | **$+87.3\text{ dB}$**    |
| **LSTM 2×8**            | $24.1\text{ dB}$     | $114.0\text{ dB}$         | **$+89.9\text{ dB}$**    |
| **Official lstm (H=3)** | $29.3\text{ dB}$     | $120.5\text{ dB}$         | **$+91.2\text{ dB}$**    |

Because LSTM execution time is dominated by GEMV matrix multiplications rather than activation math, the $10\text{--}15\%$ activation compute saving under Fast mode is negligible compared to the fidelity loss ($+89.5\text{ dB}$ average SNR gain under Standard). The CLI emits an explicit warning when `--activation fast` is applied to an LSTM model.

### 2.2 Interaction with Oversampling

Standard exact activations and neural stage oversampling operate synergistically: oversampling strips folded non-linear harmonics via half-band decimation filtering, while Standard mode eliminates high-order polynomial approximation errors.

**Implementation:** [`src/math/activations/mod.rs`](../src/math/activations/mod.rs), [`src/math/activations/tanh/`](../src/math/activations/tanh/), and [`src/math/activations/sigmoid/`](../src/math/activations/sigmoid/). Mathematical formulations reside in [`docs/fastmath-approximations.md`](fastmath-approximations.md).

---

## 3. LSTM Recurrent State Precision & Interop Parity

**Measured Interop Parity:** Under `ActivationPrecision::Standard`, recurrent state drift between NeuralAmpModeler-rs and reference NAMCore is eliminated across all catalogued LSTM models:

| Model                   | ESR vs NAMCore (Standard)  | SNR vs NAMCore    | ESR vs Ideal (f64 Oracle)  | Status                   |
|:----------------------- |:--------------------------:|:-----------------:|:--------------------------:|:------------------------ |
| **BossLSTM-1×16**       | **$8.50 \times 10^{-12}$** | $110.7\text{ dB}$ | **$8.90 \times 10^{-13}$** | ✅ Bit-identical interop |
| **BossLSTM-2×8**        | **$1.00 \times 10^{-11}$** | $110.0\text{ dB}$ | **$5.68 \times 10^{-13}$** | ✅ Bit-identical interop |
| **Official lstm (H=3)** | **$7.86 \times 10^{-13}$** | $121.0\text{ dB}$ | **$2.71 \times 10^{-12}$** | ✅ Bit-identical interop |

*Measurements taken after 24,000-sample warmup prewarm in canonical live mode ([`docs/quality-contract.json`](quality-contract.json)).*

### 3.1 Steady-State Prewarm vs. Cold-Start Decomposition

- **Steady-State (Prewarmed):** Measured after a 24,000-sample warmup period. In this regime, NeuralAmpModeler-rs matches NAMCore to float32 numerical limits (ESR $\sim 10^{-11} \text{ to } 10^{-13}$) and tracks the double-precision f64 oracle to ESR $\sim 10^{-12} \text{ to } 10^{-13}$.
- **Cold-Start Transients (256 samples without prewarm):** Short-window unit tests (`test_decomposition_*`) measure initial state buffer filling for models whose receptive field or memory exceeds 256 samples. These transient figures reflect initial condition convergence rather than steady-state precision.

### 3.2 Key Recurrent Invariants & Mitigations

- **Exact-Grade Gate Activations:** Exp-based polynomial kernels ($\le 2.4 \times 10^{-7}$ error) across SIMD dispatch paths prevent error propagation in hidden state $h_t$.
- **Kahan-Compensated Head Projection:** Head projection accumulation ($H \to 1$) uses Kahan compensated summation, yielding $\sim 2\text{ dB}$ higher SNR in deep projection heads.
- **Oversampling Interaction:** In feedforward architectures (WaveNet, ConvNet), oversampling is acoustically transparent. In recurrent architectures (LSTM), discrete updates step at $\Delta t = 1/f_s$. Running at $2\times$ or $4\times$ causes the recurrence to step at $\Delta t / 2$ or $\Delta t / 4$, compressing physical decay time and modifying frequency response (typical $\text{ESR} \approx -15 \text{ to } -25\text{ dB}$ vs native rate). For clone fidelity matching analog hardware captures, LSTMs must be run at native sample rate (`Oversample::Off`).

### 3.3 Dynamic Path & Container Regression Fixtures

The following dynamic models and container architectures serve as permanent regression fixtures:

| Model                     | Topology                                  | ESR vs NAMCore             | SNR vs NAMCore    | Protection Suite                                                                          |
|:------------------------- |:----------------------------------------- |:--------------------------:|:-----------------:|:----------------------------------------------------------------------------------------- |
| **wavenet_official**      | WaveNetDyn (CH=3, free geom, 2 arrays)    | **$9.03 \times 10^{-14}$** | $130.4\text{ dB}$ | [`tests/models/wavenet_clone_exact_test.rs`](../tests/models/wavenet_clone_exact_test.rs) |
| **wavenet_condition_dsp** | WaveNetDyn (CH=3, cond=3, FiLM)           | **$1.11 \times 10^{-14}$** | $139.6\text{ dB}$ | [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs)                     |
| **slimmable_container**   | SlimmableContainer (LSTM+WaveNetDyn+Nano) | **$7.28 \times 10^{-14}$** | $131.4\text{ dB}$ | [`tests/models/container_slimmable.rs`](../tests/models/container_slimmable.rs)           |

---

## 4. Host Sample Rate Adaptation (Polyphase Sinc Resampler)

**Architecture:** NAM models are trained at 48 kHz. When host audio environments operate at a different sample rate (44.1, 88.2, 96, or 192 kHz), NeuralAmpModeler-rs converts sample rates using a native minimum-phase polyphase FIR sinc resampler ([`src/dsp/resampler/mod.rs`](../src/dsp/resampler/mod.rs)).

- **Configuration:** 256 phases × 64 taps, Kaiser window ($\beta = 12$), minimum-phase filter by default (linear-phase variant available for offline renderers).
- **Bypass:** When host rate equals 48 kHz, a zero-cost bypass path forwards audio buffers directly with zero latency and zero copying.
- **Passband Ripple:** $< 0.05\text{ dB}$ (from 0 to $0.45 \times \text{Nyquist}$).
- **Stopband Attenuation:** Filter design attenuation $\ge 105\text{ dB}$; end-to-end multitone SNR $\sim 31\text{ dB}$ (minimum-phase, gate $\ge 25\text{ dB}$).
- **High-Frequency Rolloff:** $< 0.05\text{ dB}$ at 20 kHz (under 44.1 kHz host rate).

### Rejected Alternative: 32-Tap Resampling Mode

A 32-tap resampler variant was evaluated and rejected: while saving $\sim 40\text{ ns}$ per 64-sample block ($< 0.1\%$ total pipeline execution time), 32 taps caused passband SNR to collapse from $\ge 100\text{ dB}$ down to $\sim 24\text{ dB}$. The 64-tap configuration is a permanent invariant.

**Implementation:** [`src/dsp/resampler/mod.rs`](../src/dsp/resampler/mod.rs), [`src/dsp/sinc_kernel.rs`](../src/dsp/sinc_kernel.rs).

---

## 5. Architectural Fidelity Invariants Matrix

Every layer in the DSP pipeline is covered by structural invariance tests validating mathematical boundary conditions:

| Domain               | Invariant                                         | Verified By                                                                             | Failure Mode Prevented                |
|:-------------------- |:------------------------------------------------- |:--------------------------------------------------------------------------------------- |:------------------------------------- |
| **Buffer Tracking**  | Block-size invariance (32+32 vs 64 bit-identical) | [`src/dsp/pipeline/pipeline_block_test.rs`](../src/dsp/pipeline/pipeline_block_test.rs) | Receptive-field phase drift           |
| **State Reset**      | Reset idempotency ($A = B$ on identical input)    | [`tests/models.rs`](../tests/models.rs)                                                 | Historical state contamination        |
| **SPSC Hot-Swap**    | Seamless model swap during active audio           | [`tests/perf_soak.rs`](../tests/perf_soak.rs)                                           | RT audio clicks / priority inversions |
| **Denormal Armor**   | Zero subnormal execution penalty                  | [`src/math/common/ops.rs`](../src/math/common/ops.rs)                                   | Microcode exception CPU stalls        |
| **Allocation Guard** | Zero heap allocations on audio callback           | [`tests/rt_constraints.rs`](../tests/rt_constraints.rs)                                 | OS allocator lock RT deadline breach  |

---

## 6. Neural Stage Oversampling (HQ Mode)

**Architecture:** Optional $2\times$ or $4\times$ oversampling surrounding neural inference suppresses spectral aliasing produced by non-linear activations (`tanh`, `sigmoid`, `ReLU`). Based on Kahles, Esqueda & Välimäki (JAES 2019).

- **Filter Design:** Multi-stage half-band Kaiser FIR filters (25 taps, $\beta = 12$, $>100\text{ dB}$ stopband attenuation). The half-band property zeros alternate coefficients, halving multiplication requirements.
- **Pipeline:** `Upsample FIR stage(s) → Model Inference (at 2×/4× rate) → Downsample FIR stage(s)`.

| Mode              | Stages | Added Latency                                | Relative CPU Cost           | Architectural Behavior                                              |
|:----------------- |:------:|:--------------------------------------------:|:---------------------------:|:------------------------------------------------------------------- |
| **Off** (Default) | 0      | 0 samples                                    | $1.0\times$                 | Native reference (all topologies)                                   |
| **2×**            | 1      | 12 samples @ native rate (~0.25 ms @ 48 kHz) | $\sim 2.0\times$ model cost | Transparent anti-aliasing (WaveNet/ConvNet/A2); Timbre shift (LSTM) |
| **4×**            | 2      | 24 samples @ native rate (~0.50 ms @ 48 kHz) | $\sim 4.0\times$ model cost | Transparent anti-aliasing (WaveNet/ConvNet/A2); Timbre shift (LSTM) |

Latency is reported dynamically to the host via `OversampleEngine::latency_samples()`.

### Rejected Alternative: Antiderivative Anti-Aliasing (ADAA)

ADAA requires analytical antiderivatives per activation function, conflicting with generic polymorphic SIMD vectorization across arbitrary activation graphs. Half-band FIR oversampling is activation-agnostic and universally compatible across all neural topologies.

**Implementation:** [`src/dsp/oversample.rs`](../src/dsp/oversample.rs), [`src/dsp/pipeline/stages/inference.rs`](../src/dsp/pipeline/stages/inference.rs).

---

## 7. Denormal Prevention: Dither + Hardware FTZ/DAZ

**Architecture:** Two complementary defenses prevent subnormal (denormal) floating-point numbers from entering neural network state buffers, avoiding 10–100× microcode execution stalls on x86 processors:

1. **Deterministic Symmetrical Dither:** A constant offset `DENORMAL_DITHER_OFFSET = 1.0e-11` ($-220\text{ dBFS}$) is injected into input samples before inference and subtracted after inference ([`src/dsp/pipeline/stages/input.rs`](../src/dsp/pipeline/stages/input.rs), [`src/dsp/pipeline/stages/output.rs`](../src/dsp/pipeline/stages/output.rs)). Symmetrical addition and subtraction provide bit-exact cancellation with zero noise floor elevation.
2. **Hardware FTZ/DAZ (MXCSR Register):** Configures SSE2 MXCSR control register flags:
   - **FTZ (Flush-To-Zero):** Output subnormals flush to positive zero.
   - **DAZ (Denormals-Are-Zero):** Input subnormals are treated as zero.
     Reasserted at the entry of every audio buffer callback to defend against host environments that fail to configure or reset MXCSR.

**Implementation:** [`src/math/common/ops.rs`](../src/math/common/ops.rs) (`set_daz_ftz`).

---

## 8. Adaptive Compute (Quality Fallback FSM)

**Architecture:** When real-time audio thread P99 block processing latency exceeds safety budgets ($1.33\text{ ms}$ at 48 kHz / 64 samples), the Adaptive Compute finite state machine (FSM) downgrades model quality tiers (Full → Reduced → Minimal) to avoid buffer underruns (xruns).

- **WaveNet A1 Models:** Use double-pass inference during quality tier transitions to crossfade between sub-models smoothly without click artifacts.
- **WaveNet A2 Models (A2-Full, A2-Lite, A2-Dyn):** Do not support layer-skip mechanisms. A2 models execute single-pass direct state transitions to preserve recurrent history integrity.
- **Control:** CLI `--slim auto|full|lite`; host parameter exposes adaptive compute mode.

**Implementation:** [`src/dsp/adaptive.rs`](../src/dsp/adaptive.rs), [`src/models/static_model.rs`](../src/models/static_model.rs) (`supports_layer_skip`).

---

## 9. Cabinet Simulation: Uniform-Partitioned Overlap-Save (UPOLS)

**Architecture:** Cabinet impulse response (IR) simulation uses Uniform-Partitioned Overlap-Save (UPOLS) frequency-domain convolution ([`src/dsp/cabsim/conv.rs`](../src/dsp/cabsim/conv.rs)). Unlike direct FIR convolution ($O(L_{\text{IR}})$ per sample) or unpartitioned FFT convolution (which adds $L_{\text{IR}}$ samples of latency), UPOLS segments an impulse response of length $L_{\text{IR}}$ into $N_p = \lceil L_{\text{IR}} / P \rceil$ equal partitions of length $P$.

### 9.1 Latency vs. FFT Event Rate Trade-Off

Algorithmic latency is strictly bounded by partition size $P$:
$$\text{Latency} = P \text{ samples} \quad \left(\tau = \frac{P}{f_s} \text{ seconds}\right)$$

At each partition step, the engine executes one forward FFT of size $2P$, $N_p$ complex multiply-accumulates across the Frequency Delay Line (FDL), and one inverse FFT of size $2P$. This cycle occurs at event frequency:
$$f_{\text{event}} = \frac{f_s}{P}$$

While total MAC throughput per second remains asymptotically constant ($\approx L_{\text{IR}} \times f_s$), reducing $P$ doubles the event rate $f_{\text{event}}$, increasing twiddle-factor setup and circular pointer overhead.

### 9.2 Empirical Latency and CPU Profile (`benches/cabsim_bench.rs`)

Empirical measurements on x86-64-v3 (AVX2/FMA) for standard guitar cabinet IRs ($L_{\text{IR}} = 2048$ samples @ 48 kHz):

| Partition ($P$) | Latency @ 48 kHz | $N_p$ (2048 taps) | FFT Size ($2P$) | Event Rate ($f_s/P$) | Per-Block Time (µs) | CPU Cost @ 48 kHz | Primary Use Case                   |
|:---------------:|:----------------:|:-----------------:|:---------------:|:--------------------:|:-------------------:|:-----------------:|:---------------------------------- |
| **32**          | **0.67 ms**      | 64                | 64              | 1500 Hz              | ~1.20 µs            | ~1.2%             | Ultra-low latency monitoring (IEM) |
| **64**          | **1.33 ms**      | 32                | 128             | 750 Hz               | ~1.22 µs            | ~0.6%             | Live tracking standard             |
| **128**         | **2.67 ms**      | 16                | 256             | 375 Hz               | ~3.50 µs            | ~0.35%            | **Universal production default**   |
| **256**         | **5.33 ms**      | 8                 | 512             | 187.5 Hz             | ~12.58 µs           | ~0.2%             | Complex DAW multi-track mixing     |
| **512**         | **10.67 ms**     | 4                 | 1024            | 93.75 Hz             | ~26.00 µs           | ~0.1%             | Offline export / mastering         |

Partition construction and FFT pre-computation occur strictly off-RT during loader initialization (~19.6 µs for 2048 taps).

### 9.3 Audio Fidelity & Bit-Exact Invariance

Partition sizing in UPOLS has **zero impact on audio fidelity**:

- **Bit-Exact Frequency Response:** Output is mathematically identical across all partition sizes ($\text{ESR} < 10^{-11}$ vs direct convolution, bounded only by floating-point MAC summation order).
- **Linear-Phase FIR Reconstruction:** Zero spectral coloration, zero truncation, zero frequency warping.
- **Tail Ring-Out Continuity:** When audio input drops to zero, the impulse response tail renders to completion via the block-agnostic tail drain.

### 9.4 Block-Agnostic Engine Driver

[`CabSimAdapter::process_block`](../src/dsp/cabsim/adapter.rs) and [`CabSimPair::process_block_stereo`](../src/dsp/cabsim/adapter.rs) decouple host audio buffer sizes from partition policy $P$:

- Accepts arbitrary host block sizes (e.g. 16, 64, 128, 256, 333, 512 samples) against fixed partition $P$. The driver chunks blocks internally in a single FIFO pass without reallocation or contract violation flags.
- Host buffer size changes at constant sample rate reuse the active instance in-place with zero memory allocation and zero audio glitches.

**Implementation:** [`src/dsp/cabsim/conv.rs`](../src/dsp/cabsim/conv.rs), [`src/dsp/cabsim/adapter.rs`](../src/dsp/cabsim/adapter.rs), [`benches/cabsim_bench.rs`](../benches/cabsim_bench.rs).

---

## 10. Automated Governance & Quality Contract Verification

All fidelity and performance thresholds are governed by the automated verification pipeline:

| Governance Layer                    | Verification Mechanism                                                                   | Test Suite & Gate                |
|:----------------------------------- |:---------------------------------------------------------------------------------------- |:-------------------------------- |
| **Layer 0 — Golden Generation**     | `tests/fixtures/golden_gen_build.sh` + pinned reference commit (`1f42f88`, tag `v0.5.4`) | Contract generation              |
| **Layer 1 — Pre-committed Goldens** | `tests/models/golden_vectors.rs` — validates Rust output vs binary goldens               | `utils/tests-quick.sh` (Phase 2) |
| **Layer 2 — Live Parity**           | `tests/parity/cpp_parity.rs` — cross-engine C++ execution                                | `utils/tests-long.sh`            |

The golden freshness manifest `tests/fixtures/.golden_manifest.sha256` is enforced as a hard gate by `utils/tests-quick.sh`. Regression baselines are recorded in [`docs/quality-contract.json`](quality-contract.json).

---

## 11. Architectural Design Invariants

1. **Unquantized FP32 Math:** Weight matrices and neural activations evaluate in full 32-bit floating point (`f32`) with 64-byte alignment (`AlignedVec<f32>`). Lossy half-precision weight quantization is rejected.
2. **Standard Activation Default:** Taylor/minimax polynomial exp kernels deliver $+89.5\text{ dB}$ average SNR improvement across LSTM models over Padé approximations.
3. **64-Tap Resampler Standard:** 32-tap mode is rejected due to passband SNR collapse to $\sim 24\text{ dB}$.
4. **Half-Band FIR Oversampling:** Selected over ADAA to preserve generic SIMD dispatch across heterogeneous non-linear neural topologies.
5. **Uniform-Partitioned Convolution (UPOLS):** Decouples algorithmic latency ($P$ samples) from total impulse response length ($2048\text{--}16384$ samples) with zero frequency-domain distortion.

---

## See Also

- [`docs/fastmath-approximations.md`](fastmath-approximations.md) — Numerical error bounds and polynomial activation kernel specifications.
- [`docs/perceptual_validation.md`](perceptual_validation.md) — Measurement methodology, ESR thresholds, and oracle calibration governance.
- [`docs/architecture.md`](architecture.md) — Architectural overview, pipeline lifecycle, and memory layouts.
- [`docs/quality-contract.json`](quality-contract.json) — Automated Quality Dashboard Baseline (JSON).
- [`docs/fixtures.md`](fixtures.md) — Golden vector test supply chain contract.
