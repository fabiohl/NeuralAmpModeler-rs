<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Research References: Annotated Technical Bibliography

Annotated catalog of the scientific and normative literature that underpins the
architectural, implementation, and quality-assurance decisions of the
NeuralAmpModeler-rs project. Each entry states why the work matters, how the
project uses it, and where in the repository that use is anchored. Entries are
grouped by theme and indexed by topic at the end.

---

## 1. Aliasing Reduction and Anti-Aliasing

### R1 · Sato & Smith III (2025)

**Sato, R.; Smith III, J. O.**
*"Aliasing Reduction in Neural Amp Modeling by Smoothing Activations."*
28th International Conference on Digital Audio Effects (DAFx-25), Ancona, 2025.
arXiv:2505.04082.

**Why relevant.** Shows that neural amp models (WaveNet/TCN) generate significant
aliasing through their nonlinear activation functions, especially for high
fundamentals and high gain. Introduces the **ASR (Aliasing-to-Signal Ratio)**
metric, which the project implements in `src/testing/aliasing.rs` and uses in the
spectral-fidelity and oversampling characterization suites. Finds that smoother
activations reduce ASR without a significant increase in ESR, which informs the
activation-precision analysis and the trade-off between the Padé `tanh`
approximation (`Fast` mode) and the exact-grade `tanh` (`Standard` mode,
default).

| Traceability | Reference                                                                                                                        |
|:------------ |:-------------------------------------------------------------------------------------------------------------------------------- |
| Topic        | Aliasing from nonlinear activations                                                                                              |
| Files        | `src/testing/aliasing.rs`, `src/math/activations/`, `src/dsp/pipeline/stages/inference.rs`, `docs/perceptual_validation.md` §2.3 |

---

### R2 · Carson, Wright & Bilbao (2025)

**Carson, A.; Wright, A.; Bilbao, S.**
*"Anti-aliasing of neural distortion effects via model fine tuning."*
DAFx-25, Ancona, 2025. arXiv:2505.11375.

**Why relevant.** Proposes a teacher-student fine-tuning of pre-trained models on
an aliasing-free dataset and reports that, in most case studies, it outperforms
2× oversampling for both LSTM and TCN models, at the cost of some change to the
intended harmonic content. It is an alternative or complement to run-time
oversampling. The project adopts oversampling as the primary, user-controllable
mechanism; fine-tuning is a training-side measure outside the engine and applies
to the `.nam` models themselves.

| Traceability | Reference                           |
|:------------ |:----------------------------------- |
| Topic        | Aliasing from nonlinear activations |
| Files        | `src/dsp/oversample.rs`             |

---

### R3 · Kahles, Esqueda & Välimäki (2019)

**Kahles, J.; Esqueda, F.; Välimäki, V.**
*"Oversampling for Nonlinear Waveshaping: Choosing the Right Filters."*
Journal of the Audio Engineering Society, 67(6):440–449, 2019.

**Why relevant.** Provides the theoretical foundation and practical filter-design
guidelines for oversampling around nonlinear stages. The half-band Kaiser-window
filter used by the oversampling engine (β = 12, 25 taps, > 100 dB stop-band
rejection) follows the design methodology of this paper.

| Traceability | Reference                                                |
|:------------ |:-------------------------------------------------------- |
| Topic        | Oversampling filter design                               |
| Files        | `src/dsp/oversample.rs`, `docs/audio_fidelity_map.md` §6 |

---

### R4 · Parker, Zavalishin & Le Bivic (2016)

**Parker, J. D.; Zavalishin, V.; Le Bivic, E.**
*"Reducing the Aliasing of Nonlinear Waveshaping Using Continuous-Time Convolution."*
Proceedings of the 19th International Conference on Digital Audio Effects
(DAFx-16), Brno, 2016.

**Why relevant.** Introduces first-order **ADAA (Antiderivative Antialiasing)**, a
low-cost alternative to oversampling for memoryless nonlinearities. It is the
theoretical baseline against which oversampling was evaluated. ADAA was
considered and **not adopted**: it needs an analytical antiderivative per
activation function, which conflicts with generic, polymorphic SIMD dispatch
across arbitrary activation graphs. The decision is recorded in
`docs/audio_fidelity_map.md` §6; the oversampling engine that replaces it is
described in `docs/architecture.md` §4.3.

| Traceability | Reference                                                    |
|:------------ |:------------------------------------------------------------ |
| Topic        | ADAA, low-cost alternative to oversampling                   |
| Files        | `docs/audio_fidelity_map.md` §6, `docs/architecture.md` §4.3 |

---

### R5 · Bilbao, Esqueda, Parker & Välimäki (2017)

**Bilbao, S.; Esqueda, F.; Parker, J. D.; Välimäki, V.**
*"Antiderivative Antialiasing for Memoryless Nonlinearities."*
IEEE Signal Processing Letters, 24(7):1049–1053, 2017.

**Why relevant.** Places ADAA in a rigorous framework for memoryless
nonlinearities (such as `tanh`, sigmoid, and ReLU) and supplies the antiderivative
formulas that would apply to the activations used by the WaveNet and LSTM
models. As with R4, ADAA was evaluated and deferred in favor of oversampling; the
formulas remain the starting point for possible future modes (for example,
embedded or constrained targets).

| Traceability | Reference                                                    |
|:------------ |:------------------------------------------------------------ |
| Topic        | ADAA for memoryless nonlinearities (`tanh`, ReLU)            |
| Files        | `src/math/activations/tanh/`, `src/models/a2/activations.rs` |

---

### R6 · Holters (2019)

**Holters, M.**
*"Antiderivative Antialiasing for Stateful Systems."*
Proceedings of the 22nd International Conference on Digital Audio Effects
(DAFx-19), Birmingham, 2019.

**Why relevant.** Extends ADAA to systems with internal state, the theoretical
bridge to antialiasing the LSTM cell, whose recurrent state accumulates
activation approximation error step by step (see
`docs/audio_fidelity_map.md` §2.1 and §3). ADAA for stateful systems is not
implemented; this reference anchors future work should oversampling prove
insufficient for LSTM-family models at high sample rates.

| Traceability | Reference                        |
|:------------ |:-------------------------------- |
| Topic        | ADAA for stateful systems (LSTM) |
| Files        | `src/models/lstm/`               |

---

### R7 · Carson, Wright, Chowdhury, Välimäki & Bilbao (2024)

**Carson, A.; Wright, A.; Chowdhury, J.; Välimäki, V.; Bilbao, S.**
*"Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing."*
Proceedings of the 27th International Conference on Digital Audio Effects
(DAFx-24), Guildford, UK, 2024.

**Why relevant.** Addresses sample-rate dependency in recurrent networks, the
mechanism behind the LSTM timbre change observed under oversampling (see
[`perceptual_validation.md`](perceptual_validation.md) §4, "Oversampling
Characterization: Anti-Aliasing vs. Recurrent Timbre Shift"). An LSTM's feedback
delay is fixed in samples, so running it at a multiple of its training rate
compresses its decay times. The paper proposes architectural changes that make
RNNs sample-rate independent and is a possible path toward sample-rate-agnostic
LSTM models.

| Traceability | Reference                                       |
|:------------ |:----------------------------------------------- |
| Topic        | LSTM timbre change under oversampling           |
| Files        | `tests/models/oversampling_characterization.rs` |

---

### R8 · Mikkonen & Werner (2025)

**Mikkonen, O.; Werner, K. J.**
*"Antiderivative Antialiasing for Recurrent Neural Networks."*
Proceedings of the 28th International Conference on Digital Audio Effects
(DAFx-25), Ancona, 2025.

**Why relevant.** Extends ADAA to explicit, computable RNNs, specifically GRU and
LSTM cells, and evaluates it on pre-trained guitar-amplifier models. It is the
link between R4/R5 (memoryless ADAA) and the LSTM cell, the stateful recurrent
nonlinearity whose oversampling behavior the project characterizes (R7). ADAA for
LSTM was evaluated and deferred in favor of oversampling combined with `Standard`
(exact-grade) activations; this paper supports the approach and anchors future
work should the current mitigation prove insufficient.

| Traceability | Reference                                |
|:------------ |:---------------------------------------- |
| Topic        | LSTM oversampling trade-off; future ADAA |
| Files        | `docs/audio_fidelity_map.md` §3.2, §6    |

---

## 2. Perceptual Metrics and Loss Functions

### R9 · Wright & Välimäki (2020)

**Wright, A.; Välimäki, V.**
*"Perceptual Loss Function for Neural Modelling of Audio Systems."*
IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), Barcelona, 2020. arXiv:1911.08922.

**Why relevant.** Introduces perceptual weighting of the error signal
(pre-emphasis, including an A-weighting curve) in the loss used to train neural
audio models. It motivates the perceptual dimension of the activation-precision
analysis: approximation error in the presence region (about 2–5 kHz) matters
more audibly than low-frequency error. It also supports an A-weighted ESR as a
perceptual complement to the flat ESR used by the test framework.

| Traceability | Reference                                                                                                                                   |
|:------------ |:------------------------------------------------------------------------------------------------------------------------------------------- |
| Topic        | Activation precision; perceptual weighting of ESR                                                                                           |
| Files        | `src/testing/perceptual/mod.rs`, `tests/models/activation_precision.rs`, `docs/perceptual_validation.md`, `docs/fastmath-approximations.md` |

---

### R10 · Wright, Damskägg, Juvela & Välimäki (2020)

**Wright, A.; Damskägg, E.-P.; Juvela, L.; Välimäki, V.**
*"Real-Time Guitar Amplifier Emulation with Deep Learning."*
Applied Sciences, 10(3):766, 2020.

**Why relevant.** The reference for **ESR (Error-to-Signal Ratio)**, the
scale-invariant fidelity metric used throughout the NAM ecosystem and adopted as
the primary hard gate of the project's parity and cross-validation framework. It
also describes the WaveNet-based real-time architecture from which NAM's WaveNet
derives, and demonstrates the real-time feasibility of neural amplifier
modeling. Every parity test in `tests/parity/cpp_parity.rs`,
`tests/models/golden_vectors.rs`, and `tests/common/validation.rs` uses ESR as
its primary gate.

| Traceability | Reference                                                                                   |
|:------------ |:------------------------------------------------------------------------------------------- |
| Topic        | ESR as the primary fidelity gate; activation precision                                      |
| Files        | `tests/parity/cpp_parity.rs`, `tests/common/validation.rs`, `docs/perceptual_validation.md` |

---

### R11 · Yamamoto, Song & Kim (2020)

**Yamamoto, R.; Song, E.; Kim, J.-M.**
*"Parallel WaveGAN: A fast waveform generation model based on generative
adversarial networks with multi-resolution spectrogram."*
IEEE International Conference on Acoustics, Speech and Signal Processing
(ICASSP), Barcelona, 2020. arXiv:1910.11480.

**Why relevant.** Introduces the **Multi-Resolution STFT (MR-STFT) loss**, a
composite spectral metric of spectral convergence and log-magnitude distance
across several STFT resolutions. The project adopts it as its spectral
regression gate in `src/testing/perceptual/esr.rs`, with per-model calibrated
thresholds (Tier 1) and a dual hard/soft enforcement strategy (see
`docs/perceptual_validation.md` §2.2). The implementation uses analysis windows
of 256, 1024, and 4096 samples, which capture both narrow-band spectral
artifacts and broadband transient errors that a single-window ESR cannot detect.

| Traceability | Reference                                                                                                                       |
|:------------ |:------------------------------------------------------------------------------------------------------------------------------- |
| Topic        | MR-STFT as a spectral regression gate, complementing ESR                                                                        |
| Files        | `src/testing/perceptual/esr.rs`, `src/testing/perceptual/mod.rs`, `tests/common/validation.rs`, `docs/perceptual_validation.md` |

---

## 3. Measurement and Instrumentation

### R12 · Farina (2000)

**Farina, A.**
*"Simultaneous measurement of impulse response and distortion with a swept-sine
technique."*
Audio Engineering Society Convention 108, Paris, 2000. Preprint 5093.

**Why relevant.** The foundational technique for measuring frequency response and
per-order harmonic distortion from a single exponential sine sweep. It is the
backbone of the spectral-fidelity suite
(`tests/models/spectral_fidelity.rs`), which measures THD, THD+N, IMD, and
frequency response per model and enables versioned spectral fingerprints, turning
"does it sound right?" into automated, reproducible CI gates.

| Traceability | Reference                                                                                                                   |
|:------------ |:--------------------------------------------------------------------------------------------------------------------------- |
| Topic        | Spectral-fidelity suite (frequency response, THD by order)                                                                  |
| Files        | `tests/models/spectral_fidelity.rs`, `src/testing/spectral/farina.rs`, `src/testing/spectral/mod.rs`, `src/math/dsp/fft.rs` |

---

### R13 · AES17

**AES17-2020.**
*"AES standard method for digital audio engineering — Measurement of digital
audio equipment."*
Audio Engineering Society, 2020 (revision of AES17-2015).

**Why relevant.** Defines the standardized methodology for **THD+N**: a sine tone
at 997 Hz, a notch filter with Q between 1 and 5, and the total residual energy
as THD+N. The THD+N implementation in `src/testing/spectral/thd.rs` follows this
method (997 Hz tone, notch Q ≈ 5), keeping the reported figures comparable with
published amplifier and audio-interface specifications.

| Traceability | Reference                                                                                         |
|:------------ |:------------------------------------------------------------------------------------------------- |
| Topic        | Standardized THD+N                                                                                |
| Files        | `tests/models/spectral_fidelity.rs`, `src/testing/spectral/thd.rs`, `src/testing/spectral/mod.rs` |

---

### R14 · ITU-R BS.1770-4 / EBU R 128 / EBU Tech 3342

**ITU-R BS.1770-4 (2015).**
*"Algorithms to measure audio programme loudness and true-peak audio level."*
International Telecommunication Union, 2015.

**EBU R 128 (2020 / v5.0 2023).**
*"Loudness normalisation and permitted maximum level of audio signals."*
European Broadcasting Union, 2020 (v5.0 published 2023).

**EBU Tech 3342 (2016).**
*"Loudness Range: A measure to supplement EBU R 128 loudness normalisation."*
European Broadcasting Union, 2016.

**Why relevant.** The normative trilogy for loudness and true-peak measurement:

- **ITU-R BS.1770-4 Annex 2** defines **true-peak (dBTP)** through 4× oversampled
  polyphase FIR peak detection. To respect real-time constraints, the RT hot path
  keeps sample-peak detection (`RT_STATUS_HAS_CLIPPED`, in
  `src/dsp/pipeline/stages/output.rs` and `src/common/spsc/status.rs`), while
  true-peak is computed off-RT for telemetry and QA in
  `src/testing/perceptual/true_peak.rs`.
- **BS.1770-4 (main body) and EBU R128** define the two-pass integrated **LUFS**
  algorithm (absolute gate at −70 LUFS, then a relative gate), implemented in
  `src/testing/perceptual/lufs.rs` and validated against EBU compliance test
  vectors in `tests/models/ebu_lufs_compliance.rs`.
- **EBU Tech 3342** defines **LRA (Loudness Range)**, implemented in
  `src/testing/perceptual/lra.rs`.

| Traceability | Reference                                                                                                                                                                      |
|:------------ |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Topic        | Integrated loudness (LUFS), loudness range (LRA), true-peak (dBTP)                                                                                                             |
| Files        | `src/testing/perceptual/lufs.rs`, `src/testing/perceptual/true_peak.rs`, `src/testing/perceptual/lra.rs`, `src/dsp/pipeline/stages/output.rs`, `docs/perceptual_validation.md` |

---

### R15 · SMPTE RP 120

**SMPTE RP 120-1994 (r2000).**
*"Measurement of Intermodulation Distortion in Audio Equipment."*
Society of Motion Picture and Television Engineers, 1994 (reaffirmed 2000).

**Why relevant.** Standardizes **IMD (Intermodulation Distortion)** measurement
with the two-tone SMPTE/DIN test signal: a 60 Hz low-frequency tone and a 7 kHz
carrier at a 4:1 amplitude ratio (12 dB difference). The measurement is
implemented in `src/testing/spectral/mod.rs` and evaluated per model in
`tests/models/spectral_fidelity.rs`, quantifying nonlinear modulation products
around the carrier.

| Traceability | Reference                                                                                           |
|:------------ |:--------------------------------------------------------------------------------------------------- |
| Topic        | IMD (SMPTE/DIN two-tone)                                                                            |
| Files        | `src/testing/spectral/mod.rs`, `tests/models/spectral_fidelity.rs`, `docs/perceptual_validation.md` |

---

## Reference Index by Topic

| Topic                                         | References     |
|:--------------------------------------------- |:-------------- |
| Aliasing from nonlinear activations           | R1, R2         |
| Oversampling filter design                    | R3             |
| ADAA (memoryless, stateful, recurrent)        | R4, R5, R6, R8 |
| LSTM behavior under oversampling              | R7, R8         |
| Activation precision and perceptual weighting | R1, R9, R10    |
| ESR as the primary fidelity gate              | R10            |
| MR-STFT spectral regression gate              | R11            |
| Spectral-fidelity suite (FR, THD, THD+N, IMD) | R12, R13, R15  |
| Loudness, loudness range, true-peak           | R14            |
