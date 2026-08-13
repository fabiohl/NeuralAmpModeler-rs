<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Perceptual Validation & Measurement Framework

This document describes the complete measurement and perceptual validation infrastructure
for cross-validating `nam-rs` inference quality against precision references (C++
`NeuralAmpModelerCore`, f64 oracle) and for standalone audio fidelity assessment.

## Measurement Philosophy

`nam-rs` validates inference quality through two independent references:

1. **Parity Reference** — C++ `NeuralAmpModelerCore` (f32): Measures implementation agreement against
   the upstream reference. Since both share the same numerical approximations (f32 +
   Padé tanh + minimax sigmoid + FMA accumulation), ESR targets are orders of magnitude
   lower (1e-5 to 3e-7) than modeling error baselines. See [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs) and
   [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs).

2. **Absolute Correction** — f64 Oracle: Measures the absolute error floor of the
   production f32 path against an ideal double-precision computation with exact
   activation functions (`f64::tanh`, `f64::exp`) and compensated accumulation.
   Quantifies intrinsic quality loss from each approximation layer
   (quantization, activation, accumulation). See [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs).

All measurement routines run off the real-time audio thread. They are pure analytics
with zero heap allocation in hot loops (aside from FFT planner construction).

---

## ESR — Error-to-Signal Ratio (Primary Scale-Robust Metric)

**Files:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) | [`tests/common/metrics.rs`](../tests/common/metrics.rs) | **f64 variant:** [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs)

```text
ESR = Σ(rᵢ − tᵢ)² / Σ rᵢ²
```

Linear scale. Converted to dB via `10 · log₁₀(ESR)`.

### Why ESR?

Absolute MSE is not robust to scale mismatch — a signal 270× larger in amplitude
produces MSE ~10² even when SNR is 51–57 dB. ESR normalizes error by reference
energy, making it invariant to linear scaling. This is the primary threshold for
all parity gates.

**Limitation.** ESR is a global time-domain error metric — it is insensitive to
aliasing artifacts (Sato & Smith, DAFx 2025) and does not correlate linearly with
human auditory perception (Wright & Välimäki, ICASSP 2020). This is why `nam-rs`
supplements ESR with spectral metrics (MR-STFT, ASR) that capture frequency-domain
and aliasing-specific degradation modes that ESR alone cannot detect.

### Interpretation

| ESR range      | Meaning                                         |
| -------------- | ----------------------------------------------- |
| 0              | Perfect match                                   |
| 1              | Test signal has zero correlation with reference |
| < 1e-9         | Bit-identical (linear models, same-ISA)         |
| < 1e-7         | Bit-identical precision (typical same-ISA)      |
| ~1e-5          | Typical implementation parity (numerical noise) |
| ~6e-3 (median) | A1-Standard modeling error vs analog hardware   |
| > 1.0          | Complete divergence                             |

---

## 3-Tier Gate Hierarchy

`nam-rs` validation uses a three-tier gate system that governs how thresholds evolve from tight per-model
values through sample-rate and stress-signal relaxation, ultimately bounded by absolute sentinels.
See [`tests/common/validation.rs`](../tests/common/validation.rs) (`get_calibrated_threshold`) and [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs).

### Tier 1 — Per-Model Calibrated Thresholds

Defined in [`tests/common/validation.rs`](../tests/common/validation.rs) (`get_calibrated_threshold`). Each model entry stores
empirically measured `(mse_limit, min_snr_db, max_esr, mrstft_max)` at 48 kHz with 2048-sample v1
stress signal. Source measurements are documented in code comments.

| Model                                | SNR dB | ESR max | MR-STFT max | Notes                                                       |
| ------------------------------------ | ------ | ------- | ----------- | ----------------------------------------------------------- |
| WaveNet Standard (CH=16)             | 105    | 3.0e-11 | 0.05        | Golden only                                                 |
| WaveNet A1 Standard / Official CH=16 | 85     | 3.0e-9  | 0.05        | Live parity                                                 |
| WaveNet Feather (CH=8)               | 100    | 1.0e-10 | 0.05        | Golden only                                                 |
| WaveNet Nano (CH=4)                  | 95     | 3.0e-10 | 0.05        | Golden only                                                 |
| WaveNet Lite / EVH-5150-Lite (CH=12) | 105    | 3.5e-11 | 0.05        | Golden only                                                 |
| WaveNet Official (CH=3)              | 14     | 3.5e-2  | 0.45        | Live parity, dynamic path (free-geometry)                   |
| WaveNet Condition DSP                | 100    | 1.0e-10 | 0.35        | cond=3, dynamic sub-path                                    |
| WaveNet Dyn Free-Shape               | 90     | 1.0e-11 | 0.18        | CH=7→4, head_scale=0.02                                     |
| Nondist Models (3×)                  | 100    | 1.0e-10 | 0.05        | APP-EVH, Boss BD-2, Slammin Marshall                        |
| WaveNet A2-Full (CH=8)               | 105    | 3.0e-11 | 0.05        | Near-bit-exact                                              |
| WaveNet A2-Lite (CH=3)               | 105    | 3.5e-11 | 0.05        | Near-bit-exact                                              |
| WaveNet A2-FiLM-Lite (CH=3)          | 114    | 1.0e-11 | 1.0e-4      | Native FiLM active                                          |
| WaveNet A2-FiLM-Full (CH=8)          | 120    | 1.0e-11 | 1.0e-4      | Native FiLM active                                          |
| WaveNet A2-FiLM Chaos                | 120    | 1.0e-12 | 5.0e-5      | Chaos stress model                                          |
| WaveNet A2-FiLM InputMixinPre        | 120    | 1.0e-11 | 1.0e-4      | Single-slot FiLM                                            |
| WaveNet A2 Max                       | 90     | 1.0e-9  | 0.05        | Fail-closed TR1.1 guard (KB-A2-MAX known bug)               |
| WaveNet A2 Dyn Gated CH=8            | 85     | 1.0e-9  | 0.05        | Gating+LeakyReLU                                            |
| WaveNet A2 Dyn Blended CH=3          | 110    | 1.0e-12 | 0.05        | Blend+Tanh gate                                             |
| A2 Example (Slimmable)               | 120    | 3.5e-12 | 0.08        | SlimmableContainer                                          |
| ConvNet Test                         | 120    | 1.0e-12 | 1.0e-4      | C++ flat format render parity (prewarm fix TASK-CONVNET-01) |
| ConvNet Variants (nobn, relu, silu)  | 115    | 1.0e-11 | 5.0e-4      | ConvNet activation & batch-norm variants                    |
| LSTM 1×16                            | 93     | 1.5e-9  | 0.20        | Standard precision default (exact polynomial activations)   |
| LSTM 2×8                             | 93     | 1.7e-9  | 0.12        | Standard precision default                                  |
| LSTM Official (H=3)                  | 105    | 9.0e-11 | 0.22        | Standard precision default                                  |
| LSTM-Dyn 1×7                         | 80     | 3.5e-9  | 0.10        | Non-catalog geometry, 48 kHz only                           |
| LSTM Synthetic (1×10, 2×24, 3×8)     | 110    | 5.0e-12 | 5.0e-4      | Uncatalogued synthetic LSTM topologies                      |
| Linear (RF=320..8192)                | 125    | 1.0e-10 | 0.12        | Partitioned FFT FIR convolution (RF-independent precision)  |
| Linear No Bias                       | 125    | 1.0e-10 | 0.12        | FFT FIR convolution without bias (near-bit-exact)           |

Fallback formulas (when a model has no calibrated entry):

- **Golden vectors** (`topology_thresholds`, `tests/common/validation.rs`): LSTM `snr = (30 - complexity×0.65)`
- **Live parity** (`live_parity_thresholds`, `tests/common/validation.rs`): LSTM `snr = (85 - complexity×1.0)`
  Both are stricter than calibrated values and gated by Tier 3.

### Tier 2 — Stress Signal × Sample-Rate Relaxation

Applied only in v2 multi-SR tests ([`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs)). Compensates for numerical accumulation
over the 100× longer stress signal (5s vs 42.7ms) and for higher sample rates:

```text
sr_ratio = sample_rate / 48000

LSTM:     snr_relaxation = (3.5 × sr_ratio).min(10.0)   // capped at 10 dB
WaveNet:  snr_relaxation = (1.5 × sr_ratio).min(4.0)    // capped at 4 dB
Resample: snr -= 1.5; mse ×= 1.5; esr ×= 1.5            // only when actual_sr ≠ model_sr
```

- LSTM gets a steeper relaxation because recurrent state drift is proportional to step count.
  Full formula: `min_snr -= snr_relaxation; mse ×= 10^(snr_relax/10); esr ×= 10^(snr_relax/10)`
- At 96 kHz: LSTM relaxes 7.0 dB; at 192 kHz: 10.0 dB (capped).
- The relaxation is **deliberate** — it exists to distinguish "expected format limitation"
  from "unexpected engine regression." Tier 3 is the backstop.

### Tier 3 — ABSOLUTE_ESR_CAP / ABSOLUTE_SNR_FLOOR Sentinel

After all Tier 2 relaxation, absolute sentinels ([`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs)) clamp the result:

```text
ABSOLUTE_ESR_CAP  = A2ESR_A1_STANDARD_MEDIAN = 6.23e-3   // baseline "good" from t3k-mushra
ABSOLUTE_SNR_FLOOR = 5.0 dB                              // absolute minimum SNR meaning
```

If `max_esr > ABSOLUTE_ESR_CAP` after relaxation, it is scaled back to 6.23e-3 and `mse_limit`
is proportionally tightened. `min_snr_db` is clamped to at least 5.0 dB.

### Quality Dashboard Contract Envelopes & Oracle Governance

For build-to-build contract verification (`utils/quality-dashboard.sh --check`), ESR limits are governed by model-calibrated noise envelopes rather than permissive global multipliers (PERF-009):

- **Noise Envelope:** `noise_limit = max(baseline * 3.0, baseline + 5.0e-14)`. Tight $3\times$ ceiling anchored by an absolute floor of $5 \times 10^{-14}$ for models with extreme parity ($< 10^{-12}$). Prevents silent $2\times$–$5\times$ deterministic regressions while avoiding false positives from machine-level floating point noise.
- **Safety Ceiling:** `safety_limit = max(baseline * 10.0, 1.0e-12)`. Hard upper ceiling for contract verification.
- **Dual-Oracle Governance (`REVIEW_REQUIRED`):**
  - **Threshold Disagreement:** Triggered when NAMCore parity passes but the f64 oracle fails (or vice versa).
  - **Directional Divergence:** Triggered when one oracle ratio improves ($R < 0.85$) while the other degrades ($R > 1.15$).
  - **Policy:** Neither oracle automatically wins. Any oracle divergence blocks automatic contract approval, producing a `REVIEW_REQUIRED` failure state that mandates human inspection before baseline renewal.

---

## MR-STFT — Multi-Resolution STFT Loss (Hard + Soft Spectral Gate)

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)

```text
MR-STFT = Σ_w weight[w] · mean_frame( L1_sc + L2_sc )
```

Where:

- `w` ∈ [256, 1024, 4096] with Hann window, hop = w/4
- L1_sc = (1/F) Σ_f |ln|X_ref[f]| − ln|X_test[f]||  (spectral convergence)
- L2_sc = √( (1/F) Σ_f (ln|X_ref[f]| − ln|X_test[f]|)² )  (log-magnitude loss)
- F = w/2+1 frequency bins (unique non-redundant FFT bins)
- Weights = [0.1, 0.3, 0.5] from t3k-mushra golden calibration

### Why MR-STFT?

Single-window STFT loss is biased toward the chosen time-frequency resolution
trade-off. MR-STFT combines three window sizes to capture narrow-band and
transient errors simultaneously. As a regression-detection gate, this metric
captures frequency-domain degradation modes that ESR alone misses — making it
a stronger indicator of spectral fidelity changes than time-domain error alone.
Note: MR-STFT is used here as a spectral regression gate, not as a direct proxy
for subjective perceptual quality (which requires human listening tests).

### Dual Gate System

MR-STFT uses a per-model calibrated threshold with a dual enforcement strategy
([`tests/common/validation.rs`](../tests/common/validation.rs)):

**Hard gate — calibrated models at 44.1/48 kHz:** MR-STFT < `mrstft_max` from the
calibrated threshold table (§3-Tier Gate Hierarchy, Tier 1). The hard gate is armed
only when **both** conditions hold: the model has a calibrated `mrstft_max`
(`Some(...)`) **and** the sample rate is native (44.1/48 kHz). Failures
**assert-panic** the test. Per-model `mrstft_max` values range from
0.05 (WaveNet SKU, near bit-exact) to 0.45 (WaveNet Official, free-geometry dynamic path).

**Soft gate — everything else:** `MRSTFT_SOFT_THRESHOLD = 0.15`
([`tests/common/validation.rs`](../tests/common/validation.rs)). Informational only — not a hard assertion. It
applies at elevated rates (88.2/96/192 kHz, where LSTM recurrent artifacts
accumulate) **and also at native rates for models without a calibrated
`mrstft_max`** — notably the entire Linear architecture family, whose topology
fallback deliberately sets `mrstft_max = None`. The 0.15 constant is a
hardcoded heuristic, not a calibrated value, and is known to produce routine
false-positive warnings on narrow-band test signals.

FFT via `crate::math::dsp::fft::FftPlanner` (native, SoA, zero-alloc). Purely
scalar (non-RT), suitable for test validation.

Golden cross-check: `tests/fixtures/mrstft_golden.bin` generated by
`tests/fixtures/scripts/gen_mrstft_golden.py` (Python reference). When the
golden file is present, `test_mr_stft_parity_with_python` in
[`tests/parity/parity_primitives.rs`](../tests/parity/parity_primitives.rs) validates bit-parity within 1e-6 absolute tolerance.

### MR-STFT Sensitivity Caveat — Spectrally Sparse Signals

Models with spectrally sparse output — many frequency bins near zero across
consecutive frames — can yield artificially elevated MR-STFT values even when
time-domain fidelity is virtually perfect (ESR ≈ 1e-14, SNR > 140 dB). This is
not signal degradation — it is a known limitation of the log-magnitude
computation in bins with energy near the noise floor.

**Mechanism.** MR-STFT computes `|ln|X_ref[f]| − ln|X_test[f]||` (spectral
convergence) and the L2 analogue (log-magnitude loss) per bin. When a frequency
bin is near zero in both reference and test signals, small absolute differences
of ~1e-7 produce large log-ratios because `ln(ε_ref) − ln(ε_test)` diverges as
`ε → 0`:

```text
ln(1e-15) − ln(2e-15) = ln(0.5) ≈ −0.693   (large relative difference)
|X_ref − X_test| = 1e-15                    (negligible absolute difference)
```

The per-bin log-ratio contribution inflates the frame-level mean even when the
absolute sample error is below machine f32 epsilon. This effect is negligible
in spectrally dense signals (most bins well above the noise floor) but dominates
the MR-STFT score in models with extended near-silent regions.

---

## ASR — Aliasing-to-Signal Ratio (DAFx 2025)

**File:** [`src/testing/aliasing.rs`](../src/testing/aliasing.rs)

```text
ASR = Σ E_aliased / Σ E_harmonic
ASR_dB = 10 · log₁₀(ASR_linear)
```

Measures the energy ratio of aliased (non-harmonic) components to harmonic
components when a pure sine wave is processed through a nonlinear system.

### Algorithm

1. Apply 4-term Blackman-Harris window (a₀=0.35875, a₁=0.48829, a₂=0.14128, a₃=0.01168)
2. Real FFT via `RfftPlanner<f64>`
3. Noise floor = median of all bin magnitudes
4. Peak threshold = max(noise_floor × 6.0, max_mag × 1e-4)
5. Detect local maxima in magnitude spectrum (skip DC)
6. Classify peaks as harmonic (k·f₀ within 1.5 bins) or aliased
7. ASR = sum of aliased energies / sum of harmonic energies

### Why ASR?

Neural amp models contain nonlinearities (tanh, sigmoid) that generate harmonics
beyond Nyquist — these fold back as aliasing. ASR quantifies anti-aliasing quality
inherent to the model architecture and any built-in oversampling. It fingerprints
aliasing behavior and gates regressions.

### Interpretation ASR

- ASR < −60 dB or linear < 1e-6: effectively alias-free (linear system)
- ASR > −30 dB: significant aliasing (hard-clip, severe nonlinearity)
- No hard CI threshold — informational/diagnostic gate used to fingerprint model behavior

Tests: [`src/testing/aliasing_test.rs`](../src/testing/aliasing_test.rs) (unit), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)
(integration + model fingerprints).

---

## Farina Exponential Sine Sweep — FR + THD

**File:** [`src/testing/spectral/farina.rs`](../src/testing/spectral/farina.rs)

Simultaneous measurement of impulse response (frequency response magnitude/phase)
and THD per harmonic order via deconvolution, following Farina (AES Convention 108, 2000).

### Sweep Generation

```text
x(t) = sin[φ(t)],  φ(t) = ω₁·T / ln(ω₂/ω₁) · (exp(t·ln(ω₂/ω₁)/T) − 1)
```

Instantaneous phase computed per sample via `omega1 * duration_s / ln_ratio * ((t_norm * ln_ratio).exp_m1())`.

### Inverse Filter

```text
F[k] = conj(S[k]) / (|S[k]|² + ε)   (frequency-domain matched filter)
```

Normalized to peak amplitude 0.95. Compensates the −3 dB/octave spectral envelope
of the exponential sweep.

### Result Struct

`FarinaResult` ([`src/testing/spectral/farina.rs`](../src/testing/spectral/farina.rs)): `sample_rate`, `f1`, `f2`,
`duration_s`, `ir_linear`, `fr_magnitude_db`, `fr_phase_rad`, `freq_axis`,
`thd_by_order: Vec<(u32, f64)>`, `thd_total_percent`.

Tests: [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs) (unit), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)
(model measurements, `#[ignore]`).

---

## THD+N — AES17

**File:** [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)

Total Harmonic Distortion + Noise per AES17 standard, using a 997 Hz pure tone.

### Algorithm THD+N

1. Generate pure tone at f₀ Hz (default 997 Hz per AES17)
2. Process through `process_fn` closure
3. Biquad notch-filter the fundamental (Q ≈ 5, second-order design)
4. Discard 2000 samples for biquad settling
5. THD+N = 100% · RMS(notched) / RMS(total)
6. THD+N_dB = 20 · log₁₀(thdn_percent / 100)

`ThdnResult` ([`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)): `f0`, `sample_rate`, `thdn_percent`,
`thdn_db`, `rms_notched`, `rms_total`.

Tests: [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs) (unit), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)
(model measurements, `#[ignore]`).

---

## IMD — SMPTE/DIN

**File:** [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)

Intermodulation Distortion per SMPTE standard: 60 Hz + 7 kHz two-tone, 4:1 amplitude ratio.

`SmpteImdResult` ([`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)): `f_low`, `f_high`, `ratio`,
`sample_rate`, `imd_percent`, `imd_db`, `sideband_percents`.

Tests: [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs) (unit), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)
(model measurements, `#[ignore]`).

---

## LUFS — ITU-R BS.1770-4 Integrated Loudness (Full 2-Pass Gating)

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) (`compute_integrated_lufs`)

Full implementation of ITU-R BS.1770-4 integrated loudness with absolute and
relative gating. Single-channel mono computation.

### Plausibility Gate

`LUFS_PLAUSIBLE_MIN = −50.0`, `LUFS_PLAUSIBLE_MAX = +10.0`
([`tests/common/validation.rs`](../tests/common/validation.rs)).

The gate enforces that any reference signal outside [−50, +10] LUFS
triggers a "GOLDEN DEFECT" warning.

**Short-signal tolerance:** Signals shorter than 400ms (below one BS.1770-4
integration block) bypass the LUFS gate — the measurement produces non-finite
values. This is automatic in `report_dsp_fidelity` ([`tests/common/validation.rs`](../tests/common/validation.rs)).

**Opt-out:** `report_dsp_fidelity_no_lufs` ([`tests/common/validation.rs`](../tests/common/validation.rs)) skips the gate
for IR convolution goldens, where input is an impulse and high amplitude is
legitimate.

---

## LRA — EBU Tech 3342 Loudness Range

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) (`compute_lra`)

Quantifies the macro-dynamic range of a program — the statistical distribution
of loudness over time, not peak-to-average ratio.

Tests: [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs).

---

## True-Peak — ITU-R BS.1770-4 Annex 2 (dBTP)

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) (`compute_true_peak_db`)

Measures the inter-sample peak — the true analog peak after D/A reconstruction —
which can exceed 0 dBFS even when all digital samples are ≤ 0 dBFS (Gibbs phenomenon).

### RT-Safety Note

True-peak with 48-tap FIR × 4× oversampling (~48 MAC/sample) is **not used in
the RT hot-path**. The DSP output stage uses sample-peak only for clipping
detection. True-peak is off-RT QA/telemetry only.

Tests: [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs).

---

## Combined Loudness Measurement

**File:** [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs) (`measure_loudness`)

Computes LUFS + LRA + dBTP in a single pass, sharing the K-weighting filter
between LUFS and LRA.

Tests: [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs).

---

## f64 Reference Oracle — Absolute Error Floor

**Module:** [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs) (plus submodules `wavenet.rs`, `lstm.rs`, `a2/mod.rs`, `convnet.rs`)

Computes the ideal forward pass of WaveNet, LSTM, and A2 topologies using f64
arithmetic, exact activation functions (`f64::tanh`, `f64::exp`), and Kahan/Neumaier
compensated accumulation.

### Why an Oracle?

The production path (f32 + Padé tanh + minimax sigmoid + FMA accumulation) shares
the same limitations as C++ `NeuralAmpModelerCore`. The oracle provides an **independent**
high-precision reference that:

1. Measures the **absolute error floor** of the f32 production path
2. Permits **source decomposition** — isolating the contribution of each
   approximation (weight quantization, activation, accumulation) to total error

### Decomposition Pipeline

`run_decomposition()` ([`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs)) runs the oracle
under 5 configurations and returns a `DecompositionResult`:

| Field              | What it isolates                                      |
| ------------------ | ----------------------------------------------------- |
| `esr_f32_vs_f64`   | Full production f32 vs ideal f64 oracle (total error) |
| `esr_quant_f16c`   | f16c weight quantization error only                   |
| `esr_quant_bf16`   | bf16 weight quantization error only                   |
| `esr_activation`   | Padé tanh + minimax sigmoid vs exact f64 activations  |
| `esr_accumulation` | f32 accumulation vs Kahan/Neumaier f64 accumulation   |

> [!NOTE]
> **Cold-Start Windowing Note.** Un-prewarmed 256-sample cold-start decomposition sweeps reflect initial buffer filling. For architectures with receptive field > 256 samples (WaveNet, A2, ConvNet), buffer fill transients dominate total ESR and trigger Rule 5 ($\Sigma\,\text{sources} \approx \text{total}$) warnings. This is expected under cold start; calibrated steady-state precision floors require paired-prewarm sweeps (24k prewarm samples).

### Two References — Parity vs Absolute

| Reference         | Type     | What it measures                      | Typical target         |
| ----------------- | -------- | ------------------------------------- | ---------------------- |
| C++ NAMCore (f32) | Parity   | Implementation agreement (shared f32) | < mse_limit (Tier 1–3) |
| f64 Oracle        | Absolute | Intrinsic quality loss from f32 path  | Varies by architecture |

The parity reference answers "Is our f32 code compatible with upstream?" The
absolute reference answers "How much quality did we lose by using f32?"

For WaveNet, ESR(vs NAMCore) ≈ 1e-13 and ESR(vs f64 oracle, prewarm-paired) =
6.13e-14 (−132 dB) — virtually indistinguishable from the numerical floor.

For LSTM, when running in `Standard` (exact-grade, universal production default) mode:

| Model          | ESR vs NAMCore (24k prewarm) | SNR vs NAMCore | ESR vs f64 oracle (paired) | Status / Steady-State Note     |
| -------------- | ---------------------------- | -------------- | -------------------------- | ------------------------------ |
| BossLSTM-1x16  | 8.50e-12                     | 110.7 dB       | 8.90e-13 (−120.5 dB)       | Bit-identical interop & oracle |
| BossLSTM-2x8   | 1.00e-11                     | 110.0 dB       | 5.68e-13 (−122.5 dB)       | Bit-identical interop & oracle |
| lstm.nam (H=3) | 7.86e-13                     | 121.0 dB       | 2.71e-12 (−115.7 dB)       | Bit-identical interop & oracle |

#### *(Note: In un-prewarmed cold-start decomposition sweeps of 256 samples, buffer fill transients yield ESR of 5.06e-2 for 1×16, 1.73e-3 for 2×8, and 2.59e-3 for H=3, which reflect cold-start initialization rather than steady-state precision.)*

Tests: [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs) (oracle, anchors, decomposition), [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs) (v2 golden vectors).

### NumPy Anchor f64 Residual Floor

The NumPy anchor tests (`test_oracle_vs_python_anchor_*`) measure the ESR
between the Rust f64 oracle and the Python NumPy f64 reference across a short
validation sweep (256 acoustic samples, prewarm-paired).

The measured ESR converges to a uniform floor across feed-forward architectures:

| Architecture          | Anchor ESR    | ESR (dB) |
| --------------------- | ------------- | -------- |
| WaveNet (all SKUs)    | ~5.00 × 10⁻¹⁶ | −153     |
| ConvNet               | ~5.00 × 10⁻¹⁶ | −153     |
| A2 / A2-FiLM / A2-Dyn | ~5.00 × 10⁻¹⁶ | −153     |
| LSTM (all SKUs)       | ~3.49 × 10⁻³⁰ | −295     |

---

## LSTM Recurrent State Drift & Precision Modes

LSTM models (`BossLSTM-1×16`, `BossLSTM-2×8`, `LSTM Official H=3`) exhibit
recurrent state drift when executed with accelerated math.

### Mechanism

The LSTM cell state update accumulates activation approximation error step by step:

```text
cₜ = fₜ · cₜ₋₁ + iₜ · gₜ
hₜ = oₜ · tanh(cₜ)
```

In `Fast` mode (Padé[5,4] for tanh and minimax degree-17 for sigmoid), gate values
exceeding the calibration bounds inject approximation error into `cₜ`, which accumulates over time.

### Resolution via Standard Precision Mode

When `nam-rs` runs in `Standard` mode (exact-grade polynomial activations, universal production default)
and C++ NAMCore runs in its default mode (`using_fast_tanh = false`), the interop gap collapses
to near-zero (~1e-11 to ~1e-12). This confirms that interop divergence in `Fast` mode is caused by differing
approximation algorithms across runtimes, rather than structural engine bugs.

| Scope                               | BossLSTM-1×16           | BossLSTM-2×8            | lstm.nam (H=3)          |
| ----------------------------------- | ----------------------- | ----------------------- | ----------------------- |
| vs NAMCore (interop, 24k prewarm)   | `Standard` → 8.50e-12 ✓ | `Standard` → 1.00e-11 ✓ | `Standard` → 7.86e-13 ✓ |
| vs f64 ideal (paired, 24k prewarm)  | `Standard` → 8.90e-13 ✓ | `Standard` → 5.68e-13 ✓ | `Standard` → 2.71e-12 ✓ |
| vs f64 ideal (cold-start, 256 samp) | 5.06e-2 (−13.0 dB)      | 1.73e-3 (−27.6 dB)      | 2.59e-3 (−25.9 dB)      |

`Standard` is the universal default in both Live and HQ/Offline modes. `Fast` (Padé) mode remains an explicit opt-in for CPU-constrained targets.

### Empirical Activation Precision Gains

The [`utils/quality-dashboard.sh`](../utils/quality-dashboard.sh) suite measures the impact of precision modes across LSTM models:

| Model             | Fast (Padé) | Standard (Exact) | Δ SNR Gain   |
| ----------------- | ----------- | ---------------- | ------------ |
| **LSTM 1×16**     | 15.9 dB     | 103.2 dB         | **+87.3 dB** |
| **LSTM 2×8**      | 24.1 dB     | 114.0 dB         | **+89.9 dB** |
| **LSTM Official** | 29.3 dB     | 120.5 dB         | **+91.2 dB** |

*Average SNR gain with `Standard` (exact) polynomial activations: **+89.5 dB** across LSTM models.*

---

## Fidelity Report — Multi-Metric Pass

**File:** [`tests/common/validation.rs`](../tests/common/validation.rs) (`report_dsp_fidelity`)

Single-pass multi-metric report for golden vector and parity validation:

| Metric           | Computation                                     | Gate / Target                                        |
| ---------------- | ----------------------------------------------- | ---------------------------------------------------- |
| MSE              | noise_power / n                                 | < mse_limit (Tier 1–3 relaxed)                       |
| MAE              | max absolute difference                         | informational                                        |
| SNR              | 10 · log₁₀(signal_power / noise_power)          | > min_snr_db (Tier 1–3 relaxed)                      |
| PSNR             | 10 · log₁₀(peak_ref² / mse)                     | informational                                        |
| Equivalent Bits  | −0.5 · log₂(mse / signal_avg_power)             | informational                                        |
| ESR              | noise_power / signal_power                      | < max_esr (primary gate, Tier 1–3)                   |
| MR-STFT          | multi-resolution spectral loss                  | < mrstft_max (hard @ 44.1/48k, soft @ higher rates)  |
| LUFS (reference) | integrated loudness (BS.1770-4 2-pass)          | [−50, +10] plausibility (skipped for <400ms signals) |
| dBTP (reference) | true-peak (BS.1770-4 Annex 2, 4× polyphase)     | informational                                        |
| Anchor SNR       | SNR of test against 3.5 kHz 1-pole LP reference | degradation baseline                                 |
| Fidelity Margin  | SNR − anchor_SNR                                | > 8.0 dB target                                      |

---

## ISA Parity & Performance Gates

**File:** [`tests/parity/isa_parity.rs`](../tests/parity/isa_parity.rs)

End-to-end cross-ISA determinism validation. Runs golden vectors through each
supported SIMD ISA path and asserts output parity.

### ISA Override Infrastructure

`TEST_ISA_OVERRIDE: AtomicU8` ([`src/math/common/dispatch/detect.rs`](../src/math/common/dispatch/detect.rs)) allows
forcing a specific ISA path for test verification.

---

## RT Telemetry & Diagnostic Metrics

**File:** [`src/dsp/telemetry.rs`](../src/dsp/telemetry.rs) (`LatencyHistogram`)

32-bin exponential histogram (2⁵ ns to 2³⁶ ns), lock-free atomic bins.
Used for profiling inference latency in the production hot path.

Polled by the host application's RT telemetry collector every 100 cycles.

---

## Stress Signal Generators

**File:** [`src/testing/stress.rs`](../src/testing/stress.rs)

| Generator | Duration | Components                                                                                                            |
| --------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| `v1`      | 42.7 ms  | Low-E harmonics + chirp 220→3520 Hz + impulse at 25%                                                                  |
| `v2`      | 5.0 s    | 6 sections: single note w/bend, power chord, palm mute, pinch harmonic + saw sweep, bass amp low-A, chord decay C-E-G |

Default sample rates: [44100, 48000, 88200, 96000, 192000].

---

## Published Baselines (t3k-mushra / A2Esr.tsx)

Empirical ESR measurements from the Tone3000 dataset (NAM models trained on real gear):

| Model           | Q1 ESR  | **Median ESR** | Q3 ESR  | Median dB    | Interpretation   |
| --------------- | ------- | -------------- | ------- | ------------ | ---------------- |
| NAM A1-Standard | 0.00218 | **0.00623**    | 0.01571 | **−22.1 dB** | Baseline "good"  |
| NAM A2-Full     | 0.00114 | **0.00334**    | 0.00913 | **−24.8 dB** | State-of-the-art |

---

## Gate Calibration Policy

This policy governs how every threshold and gate in the project is derived, maintained,
and reviewed. All gates in [`tests/models/threshold_calibration.rs`](../tests/models/threshold_calibration.rs), [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs),
[`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs), and [`tests/common/validation.rs`](../tests/common/validation.rs) must comply.

### Rule 1 — Derivation from Validated Reference

Every threshold must derive from a metric whose validity has been independently
established (e.g., NumPy f64 anchor for the f64 oracle; NAMCore C++ for
interop parity). No gate may be derived from a metric whose only evidence is
"the current output passes."

### Rule 2 — Never Exceed the Placebo Line

Fidelity gates must never exceed the project's placebo boundary: **ESR < 1.0,
MR-STFT < 0.5**. A gate at or above these bounds is a placebo — it cannot catch
regressions because it sits above the point of total signal divergence. Enforced by
[`tests/models/threshold_calibration.rs`](../tests/models/threshold_calibration.rs).

### Rule 3 — Mandatory Measurement Provenance Comment

Every calibrated threshold entry must carry a `// Measured:` comment documenting:
the measured value, the conditions (sample rate, signal duration, prewarm),
and the margin applied to derive the limit.

### Rule 4 — Relaxation Requires Link to Independent Measurement

Any loosening of a gate requires linking to an independent measurement that
justifies it.

### Rule 5 — Mandatory Sanity-Check on Metric Meaning

Before declaring any gate calibrated, the sum of modeled error sources
must be consistent with the total measured error within a 10× bound (`Σ ΔESR(sources) ≈ ESR(total)`).

### Rule 6 — Independence Must Not Be Circular

A reference oracle is only independent if validated against a separate code path. Any change to oracle implementations requires re-verifying independence against the production engine.

### Rule 7 — Fix the Code, Not the Test's Scope

When a tightened gate fails, fix the underlying cause or record a documented limitation. Never silently drop failing inputs from a test suite.

---

## Quick Reference — File/Line Map

| Metric            | Implementation                                                                  | Tests                                                                                                                                                  |
| ----------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ESR (f32)         | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`tests/common/metrics.rs`](../tests/common/metrics.rs)                                                                                                |
| ESR (f64)         | [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs) | [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs)                                                                      |
| MR-STFT           | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`tests/common/validation.rs`](../tests/common/validation.rs)                                                                                          |
| ASR               | [`src/testing/aliasing.rs`](../src/testing/aliasing.rs)                         | [`src/testing/aliasing_test.rs`](../src/testing/aliasing_test.rs), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)         |
| Farina FR+THD     | [`src/testing/spectral/farina.rs`](../src/testing/spectral/farina.rs)           | [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)         |
| THD+N AES17       | [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)                 | [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)         |
| IMD SMPTE         | [`src/testing/spectral/thd.rs`](../src/testing/spectral/thd.rs)                 | [`src/testing/spectral_test.rs`](../src/testing/spectral_test.rs), [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs)         |
| LUFS              | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs), [`tests/models/ebu_lufs_compliance.rs`](../tests/models/ebu_lufs_compliance.rs) |
| LRA               | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs)                                                                                  |
| True-Peak dBTP    | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs)                                                                                  |
| Combined Loudness | [`src/testing/perceptual/mod.rs`](../src/testing/perceptual/mod.rs)             | [`src/testing/perceptual_test.rs`](../src/testing/perceptual_test.rs)                                                                                  |
| f64 Oracle        | [`src/testing/reference_oracle/mod.rs`](../src/testing/reference_oracle/mod.rs) | [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs)                                                                      |
| Fidelity Report   | [`tests/common/validation.rs`](../tests/common/validation.rs)                   | [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs), [`tests/models/golden_vectors.rs`](../tests/models/golden_vectors.rs)                   |
| ISA Parity        | [`tests/parity/isa_parity.rs`](../tests/parity/isa_parity.rs)                   | [`tests/parity/isa_parity.rs`](../tests/parity/isa_parity.rs)                                                                                          |
| RT Telemetry      | [`src/dsp/telemetry.rs`](../src/dsp/telemetry.rs)                               | [`src/dsp/telemetry.rs`](../src/dsp/telemetry.rs) (unit tests)                                                                                         |
| Stress Signals    | [`src/testing/stress.rs`](../src/testing/stress.rs)                             | [`src/testing/stress_test.rs`](../src/testing/stress_test.rs)                                                                                          |

**Fixture governance:** every golden `.bin`, model fixture, and per-model threshold used
by these tests is catalogued and version-pinned in
[`tests/fixtures/README.md`](../tests/fixtures/README.md).

---

## References

- t3k-mushra: <https://github.com/tone-3000/t3k-mushra> (MIT license)
- NAM A2 Technical Report (Atkinson 2023)
- ITU-R BS.1770-4: Algorithms to measure audio programme loudness
- EBU Tech 3342: Loudness Range
- AES Convention 108 (2000): Farina — Simultaneous measurement of impulse response and distortion with a swept-sine technique
- AES17: Measurement of digital audio equipment
- SMPTE RP 120: Intermodulation distortion measurements
- DAFx 2025: Sato & Smith — Aliasing-to-Signal Ratio (ASR)
- ICASSP 2020: Yamamoto, Song & Kim — Multi-Resolution STFT Loss (Parallel WaveGAN)
