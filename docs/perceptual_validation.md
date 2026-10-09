<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Perceptual Validation & Measurement Framework

This document specifies the measurement methodology, acoustic metrics, gate hierarchy, and calibration governance used to validate inference fidelity in `NeuralAmpModeler-rs`. The executable implementation resides in [`src/testing/`](../src/testing/) and [`tests/common/validation.rs`](../tests/common/validation.rs).

---

## 1. Measurement Philosophy & Reference Axes

NeuralAmpModeler-rs evaluates audio inference quality along two orthogonal reference axes:

```text
                     ┌───────────────────────────────────────────────┐
                     │            Model Audio Verification           │
                     └───────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         [Parity Reference: NAMCore]                     [Absolute Reference: f64 Oracle]
         • C++ NeuralAmpModelerCore (f32)                • Pure f64 double-precision arithmetic
         • Market compatibility arbiter                  • Mathematical ideality arbiter
         • Shared f32 numerical noise floor              • Isolates quantization, activation & FMA drift
         • Enforced via golden vectors & live parity     • Ground-truth anchored against NumPy f64
```

1. **Parity Reference — C++ `NeuralAmpModelerCore` (f32):** Measures implementation agreement against the upstream reference engine. Both engines run f32 arithmetic and exact-grade activations in their production defaults (`ActivationPrecision::Standard` in NeuralAmpModeler-rs; `using_fast_tanh = false` / libm `tanhf` in NAMCore). Parity ESR targets sit orders of magnitude below human auditory perception ($10^{-5}$ to $10^{-14}$), guaranteeing seamless interoperability with the market ecosystem of `.nam` models. See [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs) and [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs).

2. **Absolute Reference — f64 Oracle:** Measures the absolute precision floor of the production f32 pipeline against double-precision math with exact transcendental activations (`f64::tanh`, `f64::exp`) and compensated Kahan/Neumaier accumulation. It isolates and quantifies error budgets introduced by each approximation layer (weight representation, activation kernels, accumulation). See [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs).

**Off-RT Execution:** All measurement routines run strictly off the real-time audio thread. They perform heap allocations and extensive mathematical evaluations that are strictly prohibited in audio callback hot paths.

---

## 2. Core Numerical & Perceptual Metrics

### 2.1 ESR — Error-to-Signal Ratio (Primary Parity Metric)

**Files:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) | [`tests/common/metrics.rs`](../tests/common/metrics.rs) | **f64 variant:** [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs)

$$\text{ESR} = \frac{\sum_{i=0}^{N-1} (r_i - t_i)^2}{\sum_{i=0}^{N-1} r_i^2}, \quad \text{ESR}_{\text{dB}} = 10 \cdot \log_{10}(\text{ESR})$$

Where $r_i$ is the reference sample and $t_i$ is the test sample.

- **Scale-Robustness:** Absolute Mean Squared Error (MSE) is sensitive to arbitrary signal scaling (e.g., a gain change yields large MSE even when correlation is high). ESR normalizes squared error by reference energy, making it invariant to absolute signal level.
- **DC Offset Masking & Diagnostic DC-Free ESR:** When a golden vector contains a pronounced DC offset (notably `golden_wavenet_a2_max.bin` with mean $\mu \approx 8.19$ and standard deviation $\sigma \approx 2.45$), over $90\%$ of total signal energy ($P_{\text{sig}} \approx 73$) resides in the DC constant. In such cases, standard raw ESR is dominated by the constant term and can mask divergence in the dynamic AC components. To maintain diagnostic rigor, the test harness evaluates both raw ESR and DC-free ESR ($\text{ESR}_{\text{dc-free}}$ after mean removal from reference and test vectors). For `wavenet_a2_max`, raw ESR is $2.57 \times 10^{-14}$ while DC-free ESR is $3.08 \times 10^{-13}$ (~12× ratio, still $\approx -125\text{ dB}$, confirming that both AC and DC components remain at the float32 numerical noise floor). The DC-free metric serves as an informational diagnostic and does not replace the raw ESR parity gate.
- **Limitation:** ESR is a global time-domain metric. It cannot separate harmonic generation from inharmonic aliasing artifacts (Sato & Smith, DAFx 2025) and correlates non-linearly with human loudness perception (Wright & Välimäki, ICASSP 2020). For this reason, NeuralAmpModeler-rs supplements ESR with spectral metrics (MR-STFT, ASR).

| ESR (Linear)                | ESR (dB)                       | Practical Interpretation                                          |
|:--------------------------- |:------------------------------ |:----------------------------------------------------------------- |
| $0.0$                       | $-\infty\text{ dB}$            | Bit-identical output.                                             |
| $< 10^{-11}$                | $< -110\text{ dB}$             | Float32 numerical noise floor (near-bit-exact across ISAs).       |
| $\sim 10^{-7}$ to $10^{-5}$ | $-70\text{ to } -50\text{ dB}$ | High implementation agreement (floating-point summation noise).   |
| $\sim 3.3 \times 10^{-3}$   | $-24.8\text{ dB}$              | Tone3000 median error for WaveNet A2-Full vs analog hardware.     |
| $\sim 6.2 \times 10^{-3}$   | $-22.1\text{ dB}$              | Tone3000 median error for WaveNet A1-Standard vs analog hardware. |
| $\ge 1.0$                   | $\ge 0\text{ dB}$              | Complete signal divergence (placebo boundary).                    |

---

### 2.2 MR-STFT — Multi-Resolution STFT Loss (Spectral Regression Gate)

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)

$$\mathcal{L}_{\text{MR-STFT}} = \sum_{w \in W} \gamma_w \cdot \frac{1}{M_w} \sum_{m=1}^{M_w} \left( \mathcal{L}_{\text{sc}}^{(w,m)} + \mathcal{L}_{\text{mag}}^{(w,m)} \right)$$

Where:

- Analysis windows $W = [256, 1024, 4096]$ samples with Hann windowing, hop size $H_w = w / 4$.
- Weights $\gamma = [0.1, 0.3, 0.5]$ calibrated against the Tone3000 MUSHRA dataset.
- $\mathcal{L}_{\text{sc}}$ (Spectral Convergence): $\frac{\| |X_{\text{ref}}| - |X_{\text{test}}| \|_F}{\| |X_{\text{ref}}| \|_F}$ over frequency bins.
- $\mathcal{L}_{\text{mag}}$ (Log-Magnitude Loss): $\frac{1}{F} \sum_{f} \left| \ln |X_{\text{ref}}[f]| - \ln |X_{\text{test}}[f]| \right|$.

**Dual Gate Enforcement:**

1. **Hard Gate (Native 44.1/48 kHz):** When a model has a calibrated entry in `get_calibrated_threshold()`, MR-STFT must remain strictly below `mrstft_max` (typically $0.05$ to $0.45$). Violations trigger an immediate test assertion failure.
2. **Soft Gate (`MRSTFT_SOFT_THRESHOLD = 0.50`):** Enforced across non-standard rates (88.2, 96, 192 kHz) and uncalibrated models. Set at the anti-placebo ceiling ($0.50$), leaving a $0.05$ margin above the highest non-degenerated hard gate ($0.45$ for `wavenet_official`). Violations emit warning telemetry rather than aborting test execution.

**Sensitivity Caveat on Spectrally Sparse Signals:**
When processing signals with extended near-silent sections or narrow-band harmonics, bins near the noise floor can exhibit small absolute differences ($\sim 10^{-7}$) that produce inflated log-magnitude ratios because $\ln(\epsilon_{\text{ref}}) - \ln(\epsilon_{\text{test}})$ diverges as $\epsilon \to 0$. In such cases, time-domain metrics (ESR $\sim 10^{-14}$) confirm fidelity while MR-STFT shows an artifactual elevation.

---

### 2.3 ASR — Aliasing-to-Signal Ratio (DAFx 2025)

**File:** [`src/testing/aliasing.rs`](../src/testing/aliasing.rs)

$$\text{ASR} = \frac{\sum E_{\text{aliased}}}{\sum E_{\text{harmonic}}}, \quad \text{ASR}_{\text{dB}} = 10 \cdot \log_{10}(\text{ASR})$$

Measures folded non-harmonic spectral components generated when a non-linear network is excited by a pure tone:

1. Pure sine excitation at $f_0 = 2017\text{ Hz}$ driven at $+12\text{ dBFS}$ to force saturation. (The incommensurate $2017\text{ Hz}$ frequency prevents aliased foldover from landing exactly on harmonic bins at 48 kHz).
2. 4-term Blackman-Harris windowing ($a_0=0.35875, a_1=0.48829, a_2=0.14128, a_3=0.01168$).
3. Peak detection via `RfftPlanner<f64>`: identifies local maxima above dynamic noise floor ($\max(\text{median} \times 6.0, \text{peak} \times 10^{-4})$).
4. Peak classification: components within $1.5$ bins of $k \cdot f_0$ are classified as harmonic; all other peaks are classified as aliasing.

---

### 2.4 Farina Exponential Sine Sweep (FR + Harmonic Distortion)

**File:** [`src/testing/spectral/farina.rs`](../src/testing/spectral/farina.rs)

Simultaneous extraction of linear impulse response (frequency response magnitude and phase) and individual harmonic distortion orders ($2^{\text{nd}}$ through $N^{\text{th}}$) using exponential swept-sine excitation and deconvolution (Farina, AES 2000):

$$x(t) = \sin\left[ \frac{\omega_1 \cdot T}{\ln(\omega_2 / \omega_1)} \cdot \left( \exp\left( \frac{t}{T} \ln\frac{\omega_2}{\omega_1} \right) - 1 \right) \right]$$

The inverse filter compensates for the $-3\text{ dB/octave}$ pink spectrum of the sweep:
$$F[k] = \frac{S^*[k]}{|S[k]|^2 + \epsilon}$$

Results populate `FarinaResult`: impulse response `ir_linear`, `fr_magnitude_db`, `fr_phase_rad`, `thd_by_order`, and `thd_total_percent`.

---

### 2.5 Standardized Audio Distortion Metrics

- **THD+N (AES17):** [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs). Measures Total Harmonic Distortion + Noise using a $997\text{ Hz}$ tone. A second-order biquad notch filter removes the fundamental; THD+N is computed as the RMS ratio of notched output to total output.
- **IMD (SMPTE RP 120):** [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs). Measures Intermodulation Distortion using a dual tone: $60\text{ Hz}$ and $7\text{ kHz}$ at a $4:1$ amplitude ratio. Analyzes modulation sidebands around $7\text{ kHz}$ ($\pm 60\text{ Hz}, \pm 120\text{ Hz}, \dots$).

---

### 2.6 Broadcast Loudness & Peak Metrics

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)

- **Integrated Loudness (ITU-R BS.1770-4):** Two-pass K-weighted filtering with absolute gating ($-70\text{ LUFS}$) and relative gating ($-10\text{ LU}$). Used in the plausibility sanity gate (`LUFS_PLAUSIBLE_MIN = -50.0`, `LUFS_PLAUSIBLE_MAX = 10.0`). Signals shorter than $400\text{ ms}$ bypass the gate (non-finite LUFS); fixtures with legitimately out-of-window loudness bypass it explicitly through `report_dsp_fidelity_no_lufs`, whose skip message reports the measured LUFS, the window side, and the margin — above the $+10$ ceiling for high-gain IR-convolution goldens, below the $-50$ floor for inherently low-loudness model output (dynamic/free-shape, large-hidden LSTM, ReLU-without-BatchNorm).
- **Loudness Range (EBU Tech 3342):** Macro-dynamic loudness distribution between the $10^{\text{th}}$ and $95^{\text{th}}$ percentiles of gated loudness.
- **True-Peak (ITU-R BS.1770-4 Annex 2):** $4\times$ oversampled polyphase FIR ($48$ taps) measuring inter-sample peaks. **Strictly off-RT only:** hot-path DSP uses sample-peak detection to avoid thread deadline misses.

---

## 3. The 3-Tier Gate Hierarchy

Validation thresholds dynamically adapt across model architectures, sequence lengths, and sample rates:

```text
 ┌────────────────────────────────────────────────────────┐
 │  Tier 1: Per-Model Calibrated Base Thresholds          │  tests/common/validation.rs
 │  (Measured at 48 kHz native rate, 2048-sample v1)      │  (get_calibrated_threshold)
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Tier 2: Stress Signal × Sample-Rate Relaxation        │  tests/parity/cpp_parity.rs
 │  (Compensates for 5.0s v2 drift & elevated rates)      │  (sr_ratio & sequence scaling)
 └──────────────────────────┬─────────────────────────────┘
                            │
                            ▼
 ┌────────────────────────────────────────────────────────┐
 │  Tier 3: Topology-Specific Absolute Sentinels          │  tests/parity/cpp_parity.rs
 │  (Hard ceilings preventing runaway relaxation)         │  (ABSOLUTE_ESR_CAP_*_HF)
 └────────────────────────────────────────────────────────┘
```

### 3.1 Tier 1 — Per-Model Calibrated Thresholds

Defined in [`tests/common/validation.rs`](../tests/common/validation.rs) (`get_calibrated_threshold`). Measured with 2048-sample v1 stress signal at 48 kHz.

| Model / Architecture                         | Min SNR (dB) | Max ESR               | Max MR-STFT          | Verification Scope & Notes                         |
|:-------------------------------------------- |:------------:|:---------------------:|:--------------------:|:-------------------------------------------------- |
| **WaveNet Standard (CH=16)**                 | 105.0        | $3.0 \times 10^{-11}$ | 0.05                 | Near-bit-exact ($>134\text{ dB}$ measured)         |
| **WaveNet Feather (CH=8)**                   | 100.0        | $1.0 \times 10^{-10}$ | 0.05                 | Near-bit-exact ($>133\text{ dB}$ measured)         |
| **WaveNet Nano (CH=4)**                      | 95.0         | $3.0 \times 10^{-10}$ | 0.05                 | Near-bit-exact ($>132\text{ dB}$ measured)         |
| **WaveNet Lite / EVH-5150-Lite (CH=12)**     | 105.0        | $3.5 \times 10^{-11}$ | 0.05                 | Near-bit-exact ($>122\text{ dB}$ measured)         |
| **WaveNet A1 Standard (Official)**           | 85.0         | $3.0 \times 10^{-9}$  | 0.05                 | Live parity fixture                                |
| **WaveNet Official (CH=3 dynamic path)**     | 14.0         | $3.5 \times 10^{-2}$  | 0.45                 | Free-geometry dynamic path ($130.4\text{ dB}$ SNR) |
| **WaveNet Condition DSP (CH=3)**             | 100.0        | $1.0 \times 10^{-10}$ | 0.35                 | Sub-model dynamic path ($139.5\text{ dB}$ SNR)     |
| **WaveNet Dyn Free-Shape (CH=7→4)**          | 90.0         | $1.0 \times 10^{-11}$ | 0.18                 | Low head_scale ($\sim -65\text{ LUFS}$)            |
| **WaveNet A2-Full (CH=8)**                   | 105.0        | $3.0 \times 10^{-11}$ | 0.05                 | Native f32 weights ($129.5\text{ dB}$ measured)    |
| **WaveNet A2-Lite (CH=3)**                   | 105.0        | $3.5 \times 10^{-11}$ | 0.05                 | Native f32 weights ($132.2\text{ dB}$ measured)    |
| **WaveNet A2-FiLM-Lite (CH=3)**              | 114.0        | $1.0 \times 10^{-11}$ | $1.0 \times 10^{-4}$ | Native FiLM active                                 |
| **WaveNet A2-FiLM-Full (CH=8)**              | 120.0        | $1.0 \times 10^{-11}$ | $1.0 \times 10^{-4}$ | Native FiLM active ($138.8\text{ dB}$ measured)    |
| **WaveNet A2-FiLM Chaos Stress**             | 120.0        | $1.0 \times 10^{-12}$ | $5.0 \times 10^{-5}$ | Chaos fixture ($139.0\text{ dB}$ measured)         |
| **WaveNet A2-FiLM InputMixinPre**            | 120.0        | $1.0 \times 10^{-11}$ | $1.0 \times 10^{-4}$ | Single-slot FiLM ($134.4\text{ dB}$ measured)      |
| **WaveNet A2 Dynamic Gated CH=8**            | 85.0         | $1.0 \times 10^{-9}$  | 0.05                 | Dynamic Gating + LeakyReLU                         |
| **WaveNet A2 Dynamic Blended CH=3**          | 110.0        | $1.0 \times 10^{-12}$ | 0.05                 | Dynamic Blending + Tanh gate                       |
| **WaveNet A2 Max (CH=4, cond=8)**            | 120.0        | $1.0 \times 10^{-11}$ | $1.0 \times 10^{-4}$ | Generic WaveNet + FiLM ($135.9\text{ dB}$ measured) |
| **SlimmableContainer A2 Example**            | 120.0        | $3.5 \times 10^{-12}$ | 0.08                 | Multi-submodel container                           |
| **LSTM 1×16**                                | 93.0         | $1.5 \times 10^{-9}$  | 0.20                 | Standard exact activations ($108.5\text{ dB}$)     |
| **LSTM 2×8**                                 | 93.0         | $1.7 \times 10^{-9}$  | 0.12                 | Standard exact activations ($107.8\text{ dB}$)     |
| **LSTM Official (H=3)**                      | 105.0        | $9.0 \times 10^{-11}$ | 0.22                 | Standard exact activations ($120.8\text{ dB}$)     |
| **LSTM-Dyn 1×7**                             | 80.0         | $3.5 \times 10^{-9}$  | 0.10                 | Dynamic path topology ($144.3\text{ dB}$)          |
| **LSTM Synthetic (1×10, 2×24, 3×8)**         | 110.0        | $5.0 \times 10^{-12}$ | $5.0 \times 10^{-4}$ | Uncatalogued geometries                            |
| **ConvNet Test**                             | 120.0        | $1.0 \times 10^{-12}$ | $1.0 \times 10^{-4}$ | C++ render parity ($143.8\text{ dB}$ measured)     |
| **ConvNet Variants (nobn, relu, silu)**      | 115.0        | $1.0 \times 10^{-11}$ | $5.0 \times 10^{-4}$ | Activation & normalization variants                |
| **Linear FFT (RF=320..8192)**                | 125.0        | $1.0 \times 10^{-10}$ | 0.12                 | Partitioned FFT FIR ($>137\text{ dB}$)             |
| **Linear No Bias**                           | 125.0        | $1.0 \times 10^{-10}$ | 0.12                 | Zero-bias FIR ($144.1\text{ dB}$)                  |
| **Nondist Models (APP-EVH, BD-2, Marshall)** | 100.0        | $1.0 \times 10^{-10}$ | 0.05                 | External production captures                       |

**Uncalibrated Fallback Formulas (`topology_thresholds` in `tests/common/validation.rs`):**

- WaveNet: computed via channel-dependent tables ($16\text{ CH} \implies \text{SNR}=105\text{ dB}$).
- LSTM: $\text{SNR} = \text{clamp}(30.0 - \text{complexity} \times 0.65, 12.0, 30.0)$, $\text{ESR} = 2.0 \times 10^{-\text{SNR}/10}$.
- Linear: $\text{SNR} = 135.0\text{ dB}, \text{ESR} = 1.0 \times 10^{-10}, \text{MR-STFT} = 0.12$.
- ConvNet: $\text{SNR} = 140.0\text{ dB}, \text{ESR} = 1.0 \times 10^{-10}, \text{MR-STFT} = 0.05$.

---

### 3.2 Tier 2 — Stress Signal × Sample-Rate Relaxation

Applied exclusively during multi-sample-rate v2 stress tests ([`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs)) to compensate for error accumulation across 5.0-second signals ($100\times$ longer than v1) and higher sample rates:

$$\text{sr\_ratio} = \frac{f_s}{48000}$$

1. **Recurrent Architectures (LSTM):**
   $$\Delta_{\text{SNR}} = \min(3.5 \times \text{sr\_ratio}, 10.0\text{ dB})$$
   $$\text{min\_snr\_db} = \max(\text{min\_snr\_db} - \Delta_{\text{SNR}}, 7.0\text{ dB})$$
   $$\text{mse\_limit} \times= 10^{\Delta_{\text{SNR}}/10}, \quad \text{max\_esr} \times= 10^{\Delta_{\text{SNR}}/10}, \quad \text{mrstft\_max} \times= 10^{\Delta_{\text{SNR}}/10}$$

2. **Feedforward Architectures (WaveNet, ConvNet, Linear):**
   $$\Delta_{\text{SNR}} = \min(1.5 \times \text{sr\_ratio}, 4.0\text{ dB})$$
   $$\text{min\_snr\_db} -= \Delta_{\text{SNR}}, \quad \text{mse\_limit} \times= 10^{\Delta_{\text{SNR}}/10}, \quad \text{max\_esr} \times= 10^{\Delta_{\text{SNR}}/10}, \quad \text{mrstft\_max} \times= 10^{\Delta_{\text{SNR}}/10}$$

3. **Sample Rate Conversion Mismatch ($f_{s,\text{actual}} \neq f_{s,\text{model}}$):**
   $$\text{min\_snr\_db} -= 1.5\text{ dB}, \quad \text{mse\_limit} \times= 1.5, \quad \text{max\_esr} \times= 1.5, \quad \text{mrstft\_max} \times= 3.0$$

---

### 3.3 Tier 3 — Topology-Specific Absolute Sentinels

After Tier 2 relaxation, hard sentinel ceilings prevent runaway threshold degradation ([`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs)):

```rust
const ABSOLUTE_ESR_CAP_WAVENET_HF: f64     = 1.0e-10;
const ABSOLUTE_ESR_CAP_LSTM_NATIVE_HF: f64 = 1.0e-5;  // <= 96 kHz
const ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF: f64 = 1.0e-4;  // > 96 kHz
const ABSOLUTE_ESR_CAP_CONVNET_HF: f64     = 1.0e-10;
const ABSOLUTE_ESR_CAP_FILM_HF: f64        = 0.15;
const ABSOLUTE_SNR_FLOOR: f64              = 5.0;
const ABSOLUTE_MRSTFT_CAP: f64             = 0.95;
const ABSOLUTE_MRSTFT_CAP_FILM: f64        = 1.20;
```

If `max_esr > esr_cap`, `max_esr` is clamped to `esr_cap` and `mse_limit` is proportionally tightened. `min_snr_db` is bounded below by `ABSOLUTE_SNR_FLOOR` ($5.0\text{ dB}$).

---

### 3.4 Quality Dashboard Envelopes & Dual-Oracle Governance

Build-to-build regression monitoring via `utils/quality-dashboard.sh --check docs/quality-contract.json` applies noise envelopes:

- **Noise Limit:** $\text{noise\_limit} = \max(\text{baseline} \times 3.0, \text{baseline} + 5.0 \times 10^{-14})$. Anchored by an absolute floor of $5 \times 10^{-14}$ to prevent false alarms on machine epsilon noise.
- **Safety Ceiling:** $\text{safety\_limit} = \max(\text{baseline} \times 10.0, 1.0 \times 10^{-12})$.
- **Dual-Oracle Governance (`REVIEW_REQUIRED`):**
  - **Threshold Disagreement:** NAMCore parity passes while f64 oracle fails, or vice versa.
  - **Directional Divergence:** One oracle ratio improves ($R < 0.85$) while the other degrades ($R > 1.15$).
  - **Policy:** Neither oracle automatically prevails. Any disagreement blocks automated baseline renewal, requiring human inspection before updating [`docs/quality-contract.json`](quality-contract.json).

---

## 4. Oversampling Characterization: Anti-Aliasing vs. Recurrent Timbre Shift

**File:** [`tests/models/oversampling_characterization.rs`](../tests/models/oversampling_characterization.rs)

While multi-stage Kaiser half-band FIR oversampling ($2\times / 4\times$) can wrap any model, its acoustic impact differs fundamentally across architectures:

```text
Feedforward (WaveNet / ConvNet / A2):
  Audio ──► [Upsample 2×/4×] ──► [Static Receptive Field] ──► [Downsample] ──► Audio
            Transparent Anti-Aliasing (ΔASR < 0 dB, zero time-constant alteration)

Recurrent (LSTM):
  Audio ──► [Upsample 2×/4×] ──► [State Recurrence: Δt = 1/fs] ──► [Downsample] ──► Audio
            State steps at Δt/2 or Δt/4 → Compresses decay times → Audible Timbre Shift
```

1. **Feedforward Models:** Convolution taps and memoryless activations operate on a fixed sample window. Upsampling broadens Nyquist bandwidth before non-linear saturation, allowing the decimation filter to remove folded harmonics cleanly. The operation is **transparent anti-aliasing**.
2. **Recurrent Models (LSTM):** Discrete state equations ($c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$, $h_t = o_t \odot \tanh(c_t)$) step at discrete sample intervals ($\Delta t = 1/f_s$). Upsampling causes the recurrence to step at $\Delta t / 2$ or $\Delta t / 4$, compressing the physical decay time and frequency envelope.

### Empirical Characterization Data

| Model         | ASR Off           | ASR 4×            | Anti-Aliasing (ΔASR)         | ESR (4× vs Off)                           | MR-STFT (4× vs Off) | Acoustic Status           |
|:------------- |:-----------------:|:-----------------:|:----------------------------:|:-----------------------------------------:|:-------------------:|:------------------------- |
| **Boss BD-2** | $-32.4\text{ dB}$ | $-61.8\text{ dB}$ | **$-29.4\text{ dB}$ (Pass)** | $-18.2\text{ dB}$ ($1.51 \times 10^{-2}$) | 0.0842              | Measurable acoustic shift |
| **LSTM 1×16** | $-28.7\text{ dB}$ | $-58.4\text{ dB}$ | **$-29.7\text{ dB}$ (Pass)** | $-16.9\text{ dB}$ ($2.04 \times 10^{-2}$) | 0.0915              | Measurable acoustic shift |
| **LSTM 2×8**  | $-31.0\text{ dB}$ | $-60.1\text{ dB}$ | **$-29.1\text{ dB}$ (Pass)** | $-17.5\text{ dB}$ ($1.78 \times 10^{-2}$) | 0.0880              | Measurable acoustic shift |

> **Operational Policy:** Run LSTM models at native sample rate (`Oversample::Off`) for archival hardware capture reproduction. Treat oversampling on LSTMs as a creative tonal shaping option (tighter transient response, reduced aliasing) rather than transparent anti-aliasing.

---

## 5. The f64 Reference Oracle & Numerical Decomposition

**Module:** [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs)

The f64 reference oracle computes double-precision forward passes for WaveNet, LSTM, ConvNet, and A2 topologies using exact transcendental functions (`f64::tanh`, `f64::exp`) and Kahan/Neumaier compensated accumulation.

### 5.1 Five-Axis Error Decomposition Pipeline

`run_decomposition()` evaluates the model across 5 configurations to isolate individual error sources:

| Field              | Error Source Isolated                                                       |
|:------------------ |:--------------------------------------------------------------------------- |
| `esr_f32_vs_f64`   | Total error: production f32 pipeline vs ideal f64 reference.                |
| `esr_quant_f16c`   | Error introduced if weights were truncated to f16c.                         |
| `esr_quant_bf16`   | Error introduced if weights were truncated to bfloat16.                     |
| `esr_activation`   | Error from Padé/minimax activation approximations vs exact transcendentals. |
| `esr_accumulation` | Error from f32 FMA summation vs compensated double-precision accumulation.  |

### 5.2 Steady-State Prewarm vs. Cold-Start Transients

- **Canonical Steady-State (Prewarmed):** Measured after a 24,000-sample warmup period. WaveNet and LSTM models track NAMCore to ESR $\sim 10^{-11} \text{ to } 10^{-14}$ and track the mathematical f64 oracle to ESR $\sim 10^{-12} \text{ to } 10^{-14}$.
- **Cold-Start Transients (256 samples without prewarm):** In short sweeps, models with receptive fields or recurrent memories exceeding 256 samples exhibit initial buffer-fill transients (e.g., cold ESR of $5.06 \times 10^{-2}$ for LSTM 1×16). These reflect cold-buffer initialization rather than steady-state precision. Calibrated precision floors require paired-prewarm sweeps.

### 5.3 NumPy Anchor Ground-Truth Floor

Cross-checks between the Rust f64 oracle and Python NumPy f64 reference (`test_oracle_vs_python_anchor_*`) establish a consistent residual error floor:

| Architecture                | Anchor ESR                  | Equivalent dB    |
|:--------------------------- |:---------------------------:|:----------------:|
| **WaveNet (all SKUs)**      | $\sim 5.00 \times 10^{-16}$ | $-153\text{ dB}$ |
| **ConvNet**                 | $\sim 5.00 \times 10^{-16}$ | $-153\text{ dB}$ |
| **WaveNet A2 / FiLM / Dyn** | $\sim 5.00 \times 10^{-16}$ | $-153\text{ dB}$ |
| **LSTM (all SKUs)**         | $\sim 3.49 \times 10^{-30}$ | $-295\text{ dB}$ |

---

## 6. LSTM Recurrent State Drift & Activation Modes

Because LSTM cell states update recurrently:
$$c_t = f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)$$
small activation approximation errors in $f_t, i_t, g_t, o_t$ accumulate in $c_t$ over time.

### Empirical Activation Precision Comparison

Under `ActivationPrecision::Standard` (exact-grade polynomial exp activations, universal production default), recurrent state drift is eliminated:

| Model Topology          | Fast Mode SNR (Padé) | Standard Mode SNR (Exact) | Δ SNR Gain            |
|:----------------------- |:--------------------:|:-------------------------:|:---------------------:|
| **LSTM 1×16**           | $15.9\text{ dB}$     | $103.2\text{ dB}$         | **$+87.3\text{ dB}$** |
| **LSTM 2×8**            | $24.1\text{ dB}$     | $114.0\text{ dB}$         | **$+89.9\text{ dB}$** |
| **LSTM Official (H=3)** | $29.3\text{ dB}$     | $120.5\text{ dB}$         | **$+91.2\text{ dB}$** |

*Average SNR gain with `Standard` activations across LSTM models: **$+89.5\text{ dB}$**.*

---

## 7. Single-Pass Multi-Metric Fidelity Report

**File:** [`tests/common/validation.rs`](../tests/common/validation.rs) (`report_dsp_fidelity`)

Computes all verification metrics simultaneously in a single pass under an atomic thread lock (`REPORT_LOCK`):

| Metric              | Mathematical Basis                                      | Threshold / Gate Target                                    |
|:------------------- |:------------------------------------------------------- |:---------------------------------------------------------- |
| **MSE**             | $\frac{1}{N} \sum (r_i - t_i)^2$                        | $< \text{mse\_limit}$ (Tiers 1–3 relaxed)                  |
| **MAE**             | $\max                                                   | r_i - t_i                                                  |
| **SNR**             | $10 \log_{10}(\sum r_i^2 / \sum (r_i - t_i)^2)$         | $\ge \text{min\_snr\_db}$ (Tiers 1–3 relaxed)              |
| **PSNR**            | $10 \log_{10}(\text{peak}_{\text{ref}}^2 / \text{MSE})$ | Informational                                              |
| **Equivalent Bits** | $-0.5 \log_2(\text{MSE} / P_{\text{sig}})$              | Informational                                              |
| **ESR**             | $\sum (r_i - t_i)^2 / \sum r_i^2$                       | $< \text{max\_esr}$ (Primary gate, Tiers 1–3)              |
| **MR-STFT**         | Multi-resolution spectral loss                          | $< \text{mrstft\_max}$ (Hard at 44.1/48k; soft 0.50 above) |
| **LUFS (Ref)**      | ITU-R BS.1770-4 2-pass                                  | $[-50.0, +10.0]\text{ LUFS}$ sanity check                  |
| **dBTP (Ref)**      | ITU-R BS.1770-4 Annex 2                                 | Informational                                              |
| **Anchor SNR**      | SNR vs $3.5\text{ kHz}$ 1-pole low-pass                 | Baseline degradation check                                 |
| **Fidelity Margin** | $\text{SNR} - \text{SNR}_{\text{anchor}}$               | $> 8.0\text{ dB}$ target                                   |

When `NAM_METRICS_JSONL` is set, metrics are appended as structured JSONL lines for dashboard consumption.

---

## 8. Gate Calibration Governance Policy

All thresholds in [`tests/models/threshold_calibration.rs`](../tests/models/threshold_calibration.rs) and [`tests/common/validation.rs`](../tests/common/validation.rs) adhere to seven strict rules:

1. **Rule 1 — Independent Reference Derivation:** Thresholds must derive from an independently validated reference (NumPy f64 oracle or canonical C++ NAMCore). Self-referential baselines ("it passes current code") are prohibited.
2. **Rule 2 — Anti-Placebo Boundary:** Gates must not exceed the placebo boundary: **$\text{ESR} < 1.0$, $\text{MR-STFT} < 0.50$**. Gates above these limits cannot detect signal divergence.
3. **Rule 3 — Mandatory Measurement Comment:** Every calibrated match arm in `get_calibrated_threshold()` must carry a `// Measured:` comment documenting sample rate, signal length, prewarm condition, measured value, and margin.
4. **Rule 4 — Linked Relaxation:** Loosening any threshold requires linking to an independent measurement justifying the change.
5. **Rule 5 — Error Budget Sanity Check:** Summed modeled error sources must match total measured error within a $10\times$ window ($\sum \Delta\text{ESR}(\text{sources}) \approx \text{ESR}_{\text{total}}$). Receptive field buffer-filling transients under cold start emit uncolored notices.
6. **Rule 6 — Non-Circular Independence:** Reference oracles must execute on separate code paths. Modifying an oracle requires re-verifying independence against production paths.
7. **Rule 7 — Fix Code, Never Scope:** When an tightened gate fails, fix the underlying DSP code or document the physical limitation. Never silently drop failing inputs.

---

## 9. Quick Reference File Map

| Metric / Tool         | Core Implementation                                                             | Verification Tests                                                                                                                                     |
|:--------------------- |:------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **ESR (f32)**         | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`tests/common/metrics.rs`](../tests/common/metrics.rs)                                                                                                |
| **ESR (f64)**         | [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs) | [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs)                                                                      |
| **MR-STFT**           | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`tests/common/validation.rs`](../tests/common/validation.rs), [`tests/parity/parity_primitives.rs`](../tests/parity/parity_primitives.rs)             |
| **ASR**               | [`src/testing/aliasing.rs`](../src/testing/aliasing.rs)                         | [`src/testing/aliasing_test.rs`](../src/testing/aliasing_test.rs), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)         |
| **Farina FR+THD**     | [`src/testing/spectral/farina.rs`](../src/testing/spectral/farina.rs)           | [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs)                                                                                      |
| **THD+N / IMD**       | [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)                 | [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs)                                                                                      |
| **LUFS / LRA / dBTP** | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs), [`tests/models/ebu_lufs_compliance.rs`](../tests/models/ebu_lufs_compliance.rs) |
| **f64 Oracle**        | [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs) | [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs)                                                                      |
| **Fidelity Report**   | [`tests/common/validation.rs`](../tests/common/validation.rs)                   | [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs), [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs)                   |
| **ISA Parity**        | [`src/math/common/dispatch/detect.rs`](../src/math/common/dispatch/detect.rs)   | [`tests/parity/isa_parity.rs`](../tests/parity/isa_parity.rs)                                                                                          |
| **Stress Signals**    | [`src/testing/stress.rs`](../src/testing/stress.rs)                             | [`src/testing/stress_test.rs`](../src/testing/stress_test.rs)                                                                                          |

Fixture hashes and model version pins are catalogued in [`docs/fixtures.md`](fixtures.md).

---

## References

- **Sato & Smith (DAFx 2025):** *Aliasing-to-Signal Ratio (ASR) for Non-linear Audio Systems*.
- **Yamamoto, Song & Kim (ICASSP 2020):** *Parallel WaveGAN: A fast waveform generation model based on multi-resolution spectrogram discriminator*.
- **Farina (AES Convention 108, 2000):** *Simultaneous measurement of impulse response and distortion with a swept-sine technique*.
- **ITU-R BS.1770-4:** *Algorithms to measure audio programme loudness and true-peak audio level*.
- **EBU Tech 3342:** *Loudness Range (LRA) — An objective measure of loudness dynamics in audio*.
- **AES17:** *AES standard method for digital audio engineering — Measurement of digital audio equipment*.
- **SMPTE RP 120:** *Measurement of Intermodulation Distortion in Audio Equipment*.
- **t3k-mushra (Tone3000):** Empirical MUSHRA listening tests on neural amp models (<https://github.com/tone-3000/t3k-mushra>).
