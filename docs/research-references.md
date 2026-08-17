<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Research References — Annotated Technical Bibliography

Annotated catalog of the scientific and normative literature that underpins the architectural,
implementation, and quality-assurance decisions of the nam-rs project.

---

## 1. Aliasing Reduction & Anti-Aliasing

### R1 · Sato & Smith III (2025)

**Sato, R.; Smith III, J. O.**
*"Aliasing Reduction in Neural Amp Modeling by Smoothing Activations."*
DAFx 2025. arXiv:2505.04082.

**Why relevant to nam-rs.** Demonstrates that neural amp models (WaveNet/TCN) generate significant
aliasing from nonlinear activation functions, especially on high fundamentals and high gain. Introduces
the **ASR (Aliasing-to-Signal Ratio)** metric — the key missing measurement in nam-rs's QA arsenal.
Shows that smoother activations (tanh, Snake) reduce ASR without significantly increasing ESR, directly
informing the activation precision analysis (P-5) and the trade-off between Padé tanh (Fast mode approximation) and exact-grade tanh (Standard mode default).

| Traceability | Reference                                                                                |
|:------------ |:---------------------------------------------------------------------------------------- |
| Finding      | P-1 (non-linear activation aliasing)                                                     |
| Files        | `src/math/activations/`, `src/dsp/pipeline/stages/inference.rs`, `src/dsp/oversample.rs` |

---

### R2 · Carson, Wright & Bilbao (2025)

**Carson, A.; Wright, A.; Bilbao, S.**
*"Anti-aliasing of neural distortion effects via model fine tuning."*
DAFx 2025. arXiv:2505.11375.

**Why relevant to nam-rs.** Shows that fine-tuning for anti-aliasing can outperform 2× oversampling,
offering an alternative (or complementary) path to reducing aliasing. Motivates the multi-pronged
approach: oversampling as the primary mechanism (user-controllable), with fine-tuning left as a
potential future enhancement for shipped `.nam` models.

| Traceability | Reference                            |
|:------------ |:------------------------------------ |
| Finding      | P-1 (non-linear activation aliasing) |
| Files        | `src/dsp/oversample.rs`              |

---

### R3 · Kahles, Esqueda & Välimäki (2019)

**Kahles, J.; Esqueda, F.; Välimäki, V.**
*"Oversampling for Nonlinear Waveshaping: Choosing the Right Filters."*
Journal of the Audio Engineering Society, 67(6):440–449, 2019.

**Why relevant to nam-rs.** Provides the theoretical foundation and practical filter design
guidelines for oversampling around nonlinear stages. The half-band Kaiser-window polyphase filter
adopted (β=12, 25 taps, >100 dB stopband) follows the design methodology established in this
paper. Also referenced in the resampler redesign for the HQ polyphase bank.

| Traceability | Reference                                                                          |
|:------------ |:---------------------------------------------------------------------------------- |
| Finding      | P-1 (filtros de oversampling)                                                      |
| Files        | `src/dsp/oversample.rs`, `src/dsp/sinc_kernel.rs`, `docs/audio_fidelity_map.md` §5 |

---

### R4 · Parker, Zavalishin & Le Bivic (DAFx-16)

**Parker, J. D.; Zavalishin, V.; Le Bivic, E.**
*"Reducing the Aliasing of Nonlinear Waveshaping Using Continuous-Time Convolution."*
Proceedings of the 19th International Conference on Digital Audio Effects (DAFx-16), Brno, 2016.

**Why relevant to nam-rs.** Introduces **ADAA (Antiderivative Antialiasing)** of 1st order — a
low-cost alternative to oversampling for memoryless nonlinearities. Relevant as the theoretical
baseline against which the oversampling approach was evaluated. ADAA was considered but
ultimately **not adopted** because it would require per-model modification of the activation dispatch
(polymorphic dispatch conflict); the decision and rationale are documented in
`docs/audio_fidelity_map.md` §5 and `docs/architecture.md` §3.3.

| Traceability | Reference                                                    |
|:------------ |:------------------------------------------------------------ |
| Finding      | P-1 (ADAA alternativa de baixo custo)                        |
| Files        | `docs/audio_fidelity_map.md` §5, `docs/architecture.md` §3.3 |

---

### R5 · Bilbao, Esqueda, Parker & Välimäki (2017)

**Bilbao, S.; Esqueda, F.; Parker, J. D.; Välimäki, V.**
*"Antiderivative Antialiasing for Memoryless Nonlinearities."*
IEEE Signal Processing Letters, 24(7):1049–1053, 2017.

**Why relevant to nam-rs.** Extends ADAA to a rigorous theoretical framework for memoryless
nonlinearities (tanh, sigmoid, ReLU). This paper provides the antiderivative formulas applicable
to the activation functions used in nam-rs's WaveNet and LSTM models. Like R4, ADAA was
architecturally evaluated and deferred in favor of oversampling; the formulas and feasibility
analysis are preserved for potential future modes (e.g., embedded/high-performance targets).

| Traceability | Reference                                                    |
|:------------ |:------------------------------------------------------------ |
| Finding      | P-1 (ADAA para memoryless — tanh, ReLU)                      |
| Files        | `src/math/activations/tanh/`, `src/models/a2/activations.rs` |

---

### R6 · Holters (2019)

**Holters, M.**
*"Antiderivative Antialiasing for Stateful Systems."*
Proceedings of the 22nd International Conference on Digital Audio Effects (DAFx-19), Birmingham, 2019.

**Why relevant to nam-rs.** Extends ADAA to systems with internal state — the theoretical bridge to
applying antialiasing to the LSTM cell itself (stateful recurrent nonlinearity). This paper is
particularly pertinent because the LSTM head is the primary source of recurrent state quantization
drift (previously documented in finding F-2; the f16c root cause has been removed). While ADAA for stateful systems
was not implemented, this reference anchors future work should oversampling prove insufficient
for LSTM-family models at high sample rates.

| Traceability | Reference                                  |
|:------------ |:------------------------------------------ |
| Finding      | P-1 (ADAA para sistemas com estado — LSTM) |
| Files        | `src/models/lstm/`                         |

---

### R7 — Carson, Wright, Chowdhury, Välimäki & Bilbao (2024)

**Carson, A.; Wright, A.; Chowdhury, J.; Välimäki, V.; Bilbao, S.**
*"Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing."*
Proceedings of the 27th International Conference on Digital Audio Effects (DAFx-24),
Guildford, UK, 2024.

**Why relevant to nam-rs.** Addresses the core problem of sample-rate dependency in recurrent
neural networks — the very mechanism that causes LSTM timbre to change drastically under
oversampling (documented in `docs/perceptual_validation.md` §3, evaluated across LSTM recurrent state precision). Proposes architectural
modifications to make RNNs sample-rate independent, directly relevant to the observation that
LSTM feedback delay is fixed in absolute samples. This paper anchors the theoretical
understanding of why LSTM oversampling changes timbre (β3.1) and provides a potential path
for future sample-rate-agnostic LSTM models.

| Traceability | Reference                                       |
|:------------ |:----------------------------------------------- |
| Finding      | I5 (oversampling timbre change, β3.1)           |
| Files        | `tests/models/oversampling_characterization.rs` |

---

### R7b — Mikkonen & Werner (2025)

**Mikkonen, O.; Werner, K. J.**
*"Antiderivative Antialiasing for Recurrent Neural Networks."*
Proceedings of the 28th International Conference on Digital Audio Effects (DAFx-25),
Ancona, Italy, 2025.

**Why relevant to nam-rs.** Extends ADAA (Antiderivative Antialiasing) to explicit, computable
RNNs — specifically GRU and LSTM cells. Evaluated on pre-trained guitar amplifier models,
showing that ADAA reduces aliasing considerably across all sample rates while only moderately
affecting tonality, **without requiring high oversampling factors**. This is the theoretical
bridge between R4/R5 (ADAA for memoryless nonlinearities) and the LSTM cell — the stateful
recurrent nonlinearity that nam-rs's oversampling characterization (β3.1) identified as
problematic for timbre preservation. While ADAA for LSTM was architecturally evaluated and
deferred in favor of oversampling + Standard (exact-grade) activations (I6), this paper validates the
approach and anchors future work should the current mitigation strategy prove insufficient
for LSTM-family models at high sample rates.

| Traceability | Reference                                           |
|:------------ |:--------------------------------------------------- |
| Finding      | I5 (LSTM oversampling trade-off, β3.1); future ADAA |
| Files        | `docs/audio_fidelity_map.md` §3                     |

---

## 2. Perceptual Metrics & Loss Functions

### R8 · Wright & Välimäki (2020)

**Wright, A.; Välimäki, V.**
*"Perceptual Loss Function for Neural Modelling of Audio Systems."*
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, 2020.
arXiv:1911.08922.

**Why relevant to nam-rs.** Introduces perceptual weighting of the error signal (pre-emphasis
A-weighting curve) in the loss function used to train NAM models. This directly informs the
perceptual dimension of the activation precision analysis (P-5): the frequency-dependent
relevance of approximation error — errors in the presence region (~2–5 kHz) are audibly more
significant than low-frequency errors. The paper also justifies A-weighted ESR as a perceptual
complement to the standard (flat) ESR metric already implemented in `src/testing/perceptual/mod.rs`.

| Traceability | Reference                                                                                           |
|:------------ |:--------------------------------------------------------------------------------------------------- |
| Finding      | P-5 (pré-ênfase A-weighting; ESR perceptual)                                                        |
| Files        | `src/testing/perceptual/mod.rs`, `docs/perceptual_validation.md`, `docs/fastmath-approximations.md` |

---

### R9 · Wright et al. (2020)

**Wright, A.; Damskägg, E.-P.; Juvela, L.; Välimäki, V.**
*"Real-Time Guitar Amplifier Emulation with Deep Learning."*
Applied Sciences, 10(3):766, 2020.

**Why relevant to nam-rs.** The seminal paper that defines the **ESR (Error-to-Signal Ratio)**
metric — the primary scale-invariant fidelity gate adopted by the NAM ecosystem and by
nam-rs's cross-validation framework. Introduces the WaveNet-based architecture that became the
NAM standard. Every parity test in `tests/parity/cpp_parity.rs`, `tests/models/golden_vectors.rs`, and
`tests/common/validation.rs` uses ESR as the primary hard gate. Also documents
the real-time feasibility of neural amp modeling, directly validating nam-rs's low-latency
live-path design.

| Traceability | Reference                                                                                   |
|:------------ |:------------------------------------------------------------------------------------------- |
| Finding      | P-5 (origem da métrica ESR); F-2 (ESR como gate primário)                                   |
| Files        | `tests/parity/cpp_parity.rs`, `tests/common/validation.rs`, `docs/perceptual_validation.md` |

---

### R13 · Yamamoto, Song & Kim (2020)

**Yamamoto, R.; Song, E.; Kim, J.-M.**
*"Parallel WaveGAN: A fast waveform generation model based on generative adversarial
networks with multi-resolution spectrogram."*
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, 2020.
arXiv:1910.11480.

**Why relevant to nam-rs.** Introduces the **Multi-Resolution STFT (MR-STFT) loss** — the composite
spectral metric combining spectral convergence (L1 of log-magnitude difference) and log-magnitude
loss (L2) across multiple STFT resolutions. This is the exact technique adopted as the spectral
regression-detection gate in `src/testing/perceptual/mod.rs`, with per-model calibrated thresholds
(Tier 1) and a dual hard/soft enforcement strategy. The multi-resolution approach (window sizes
[256, 1024, 4096] in nam-rs) captures both narrow-band spectral artifacts and broadband transient
errors that single-window ESR cannot detect.

| Traceability | Reference                                                                                                                       |
|:------------ |:------------------------------------------------------------------------------------------------------------------------------- |
| Finding      | F-2 (MR-STFT como gate espectral); P-5 (complemento a ESR)                                                                      |
| Files        | `src/testing/perceptual/esr.rs`, `src/testing/perceptual/mod.rs`, `tests/common/validation.rs`, `docs/perceptual_validation.md` |

---

## 3. Measurement & Instrumentation

### R10 · Farina (2000)

**Farina, A.**
*"Simultaneous measurement of impulse response and distortion with a swept-sine technique."*
Audio Engineering Society Convention 108, Paris, 2000. Preprint 5093.

**Why relevant to nam-rs.** The foundational technique for measuring frequency response and
harmonic distortion (per order) from a single exponential sine sweep. This method is the
theoretical backbone of the spectral fidelity test suite (`tests/models/spectral_fidelity.rs`, P-3)
that measures THD, THD+N, IMD, and frequency response per model SKU. Enables versioned
spectral fingerprinting — the mechanism that converts "does it sound right?" into automated,
reproducible CI gates.

| Traceability | Reference                                                                                                    |
|:------------ |:------------------------------------------------------------------------------------------------------------ |
| Finding      | P-3 (suíte de fidelidade espectral — FR, THD por ordem)                                                      |
| Files        | `tests/models/spectral_fidelity.rs`, `src/testing/spectral/farina.rs`, `src/testing/`, `src/math/dsp/fft.rs` |

---

### R11 · AES17

**AES17-2015 (r2020).**
*"AES standard method for digital audio engineering — Measurement of digital audio equipment."*
Audio Engineering Society, 2015 (revised 2020).

**Why relevant to nam-rs.** Defines the standardized methodology for **THD+N** measurement:
sine tone at 997 Hz, notch filter with Q ∈ [1, 5], total energy of residual as THD+N. This
standard provides the normative reference for nam-rs's THD+N reporting (P-3), ensuring that
spectral quality metrics are comparable with published amplifier and audio interface
specifications.

| Traceability | Reference                                                                          |
|:------------ |:---------------------------------------------------------------------------------- |
| Finding      | P-3 (THD+N padronizado)                                                            |
| Files        | `tests/models/spectral_fidelity.rs`, `src/testing/spectral/thd.rs`, `src/testing/` |

---

### R12 · ITU-R BS.1770-4 / EBU R128 / EBU Tech 3342

**ITU-R BS.1770-4 (2015).**
*"Algorithms to measure audio programme loudness and true-peak audio level."*
International Telecommunication Union, 2015.

**EBU R128 (2020).**
*"Loudness normalisation and permitted maximum level of audio signals."*
European Broadcasting Union, 2020.

**EBU Tech 3342 (2016).**
*"Loudness Range: A measure to supplement EBU R128 loudness normalisation."*
European Broadcasting Union, 2016.

**Why relevant to nam-rs.** The normative trilogy for loudness and true-peak measurement:

- **ITU-R BS.1770-4 Annex 2** defines **true-peak (dBTP)** via 4× oversampled polyphase FIR peak detection.
  In accordance with real-time safety constraints, sample-peak detection is maintained in the RT hot-path
  (`RT_STATUS_HAS_CLIPPED` in `src/dsp/pipeline/stages/output.rs`, `src/common/spsc/status.rs`), while true-peak
  is exposed off-RT for telemetry and QA in `src/testing/perceptual/true_peak.rs`.
- **BS.1770-4 main body + EBU R128** define the full two-pass integrated **LUFS** algorithm (absolute gate −70 LUFS
  → relative gate −10 LU), implemented in `src/testing/perceptual/lufs.rs` and validated against EBU compliance test vectors
  in `tests/models/ebu_lufs_compliance.rs`.
- **EBU Tech 3342** defines **LRA (Loudness Range)** — implemented in `src/testing/perceptual/lra.rs` to complete
  the perceptual loudness measurement framework (P-6).

| Traceability | Reference                                                                                                                                                                      |
|:------------ |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Finding      | P-6 (LUFS BS.1770-4 pleno, LRA); P-3 (true-peak dBTP)                                                                                                                          |
| Files        | `src/testing/perceptual/lufs.rs`, `src/testing/perceptual/true_peak.rs`, `src/testing/perceptual/lra.rs`, `src/dsp/pipeline/stages/output.rs`, `docs/perceptual_validation.md` |

---

## Reference Index by Finding

| Finding                                      | References                                             |
|:-------------------------------------------- |:------------------------------------------------------ |
| P-1 (non-linear activation aliasing)         | R1, R2, R3, R4, R5, R6, R7b                            |
| P-2 (fidelidade do resampler)                | R3                                                     |
| P-3 (suíte espectral — THD/IMD/FR/true-peak) | R10, R11, R12                                          |
| P-4 (oráculo f64)                            | — (norma computacional, não referenciada externamente) |
| P-5 (erro de ativação — precisão)            | R1, R8, R9, R13                                        |
| P-6 (LUFS/LRA/true-peak)                     | R12                                                    |
| P-7 (gates de perf/deadline)                 | — (engenharia interna)                                 |
| P-8 (matriz cross-ISA)                       | — (engenharia interna)                                 |
| F-2 (ponto cego de fidelidade)               | R9, R13                                                |
| I5 (oversampling LSTM — timbre, β3.1)        | R7, R7b                                                |
