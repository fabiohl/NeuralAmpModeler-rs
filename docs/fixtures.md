<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Fixture Governance & Golden Vector Reference — NeuralAmpModeler-rs

This document establishes the fixture supply chain, catalog governance, and ground-truth references for the `NeuralAmpModeler-rs` engine. It details how test vectors, model weights, impulse responses, and numerical anchors are generated, resolved, validated, and kept synchronized with upstream implementations.

---

## 1. Architectural Scope & Canonical Source of Truth

The original documentation and behavioral baseline of any neural architecture is its executable code. `NeuralAmpModeler-rs` anchors its numerical validation to upstream reference engines:

- **Primary C++ Inference Reference:** [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore) (Steven Atkinson). Canonical engine that trains and exports `.nam` models. Its CLI `render` tool generates the baseline golden vector outputs.
- **Upstream Impulse Response Reference:** [NeuralAmpModelerPlugin](https://github.com/sdatkinson/NeuralAmpModelerPlugin). Provides `dsp::ImpulseResponse` (from the `AudioDSPTools` submodule) for cabsim cross-validation.
- **Pinned Upstream Versions:** Tracked in [`variables.env`](../variables.env) (`NAM_CORE_COMMIT`, `NAM_CORE_TAG`, `NAM_PLUGIN_COMMIT`, `NAM_PLUGIN_TAG`). Sourced by all setup and generation scripts.
- **Repo-Local Vendor Mirrors:** Pinned mirrors live under the gitignored `third-party/` directory at the subproject root:
  - `third-party/NeuralAmpModelerCore/` (~143 MB): C++ render engine and reference models.
  - `third-party/NeuralAmpModelerPlugin/` (~164 MB): C++ cabsim impulse response engine.
  - `third-party/community_models/`: Optional symlink/directory to private non-distributable community models.
  - `build/namcore_render/` (~6 MB): Local CMake build artifacts for the C++ render tools.
- **Environment Overrides:** Paths default to subproject-local roots and can be overridden via `NAM_THIRD_PARTY_DIR`, `NAM_CORE_DIR`, `NAM_PLUGIN_DIR`, and `NAM_COMMUNITY_MODELS_SRC`.

Populate or refresh the vendor mirrors via:

```bash
./utils/setup-third-party.sh
```

---

## 2. Directory Layout & Model Resolution Protocol

### Directory Layout

```text
NeuralAmpModeler-rs/
├── tests/fixtures/
│   ├── models/                    # Version-controlled models (.nam, mocks)
│   ├── models-nondist/            # Local non-distributable models (gitignored)
│   ├── f64_anchors/               # Ground-truth f64 reference vectors
│   ├── scripts/                   # Specialized Python/NumPy reference generators
│   ├── .golden_manifest.sha256    # SHA-256 freshness integrity manifest
│   ├── golden_*.bin               # Pre-committed golden vectors (v1 and v2)
│   └── stress_signal*.wav         # Deterministic multi-component stimulus signals
└── third-party/
    └── community_models/          # Optional symlink to private community model store
```

### Model Resolution Order

Model resolution is strictly symmetric between the Rust test harness ([`src/testing/fixtures.rs::model_path()`](../src/testing/fixtures.rs)) and the shell generation pipeline (`resolve_nam_model()` in [`tests/fixtures/golden_gen_build.sh`](../tests/fixtures/golden_gen_build.sh)):

| Step  | Location                                   | Environment Variable  | Distribution Status           |
|:----- |:------------------------------------------ |:--------------------- |:----------------------------- |
| **1** | `$NAM_MODELS_DIR/<filename>`               | `NAM_MODELS_DIR`      | Explicit override             |
| **2** | `third-party/community_models/<filename>`  | `NAM_THIRD_PARTY_DIR` | Private archive (symlink)     |
| **3** | `tests/fixtures/models-nondist/<filename>` | —                     | Local non-distributable       |
| **4** | `tests/fixtures/models/<filename>`         | —                     | Version-controlled repository |

### Skip Semantics & Honest Gating

- **Optional Models:** Non-distributable captures (e.g. `EVH-5150-Lite.nam`) are skipped gracefully (`[STATUS] SKIP_CAPABILITY`) if and only if the file is absent in **all** four search paths.
- **Fail-Closed Execution:** If a model file is located in any search path, validation runs fail-closed. Tests must pass calibrated thresholds or fail hard; silent skipping of present fixtures is prohibited.
- **Golden Artifacts:** Golden `.bin` files contain computed float32 activations, not model weights, and are committed to git to enable standalone `cargo test` execution without C++ build dependencies.

### Model Inspection & Manifest Tooling

The CLI tool [`utils/check-model.sh`](../utils/check-model.sh) (backed by [`examples/inspect_model.rs`](../examples/inspect_model.rs)) inspects model data, validates architectural topology, inspects input/output levels (dBu, loudness dB), and generates machine-readable catalogs:

```bash
# Inspect a model interactively:
./utils/check-model.sh tests/fixtures/models/BossWN-standard.nam

# Generate or update manifest.json for a collection of models:
./utils/check-model.sh --manifest tests/fixtures/models/*.nam > tests/fixtures/models/manifest.json
```

---

## 3. Model Inventory & Provenance Registry

The single source of truth for model identities and architectural categorization is [`src/testing/catalog.rs::MODEL_CATALOG`](../src/testing/catalog.rs) (51 unique SHA-256 identities) and `reference_architectures()` ([`ArchitectureFamily`](../src/testing/catalog.rs)).

### Model Provenance & Classification

| Model File                              | Nature         | Architecture & Topology              | License & Provenance                     | Purpose in Engine Validation                                            |
|:--------------------------------------- |:-------------- |:------------------------------------ |:---------------------------------------- |:----------------------------------------------------------------------- |
| `wavenet_a1_standard.nam`               | Official Real  | WaveNet A1 (CH=16, K=3, HEAD=8, 20L) | CC0 / S. Atkinson                        | V1 and V2 multi-SR baseline; canonical A1 architecture.                 |
| `wavenet_official.nam`                  | Official Real  | WaveNet (CH=3, K=3, free geom)       | CC0 / S. Atkinson                        | V1/V2 golden tests; dynamic path clone determinism.                     |
| `lstm.nam`                              | Official Real  | LSTM (1 layer, H=3)                  | CC0 / S. Atkinson                        | Official LSTM sample; V1/V2 golden and live C++ parity.                 |
| `wavenet_condition_dsp.nam`             | Official Real  | WaveNet (CH=3, cond=3 FiLM + DSP)    | CC0 / S. Atkinson                        | FiLM dynamic path with post-FiLM DSP conditioning.                      |
| `BossWN-standard.nam`                   | Community Real | WaveNet (CH=16, K=3, 20L)            | Permissive / Boss Waza TAE               | Full-scale WaveNet; V1/V2 golden and zero-allocation gates.             |
| `BossWN-feather.nam`                    | Community Real | WaveNet (CH=8, K=3, 20L)             | Permissive / Boss Waza TAE               | Medium-scale WaveNet; V1/V2 golden multi-SR validation.                 |
| `BossWN-nano.nam`                       | Community Real | WaveNet (CH=4, K=3, 20L)             | Permissive / Boss Waza TAE               | Compact WaveNet; V1/V2 golden multi-SR validation.                      |
| `BossLSTM-1x16.nam`                     | Community Real | LSTM (1 layer, H=16)                 | Permissive / Boss Waza TAE               | Real amp weights; V1/V2 multi-SR recurrent validation.                  |
| `BossLSTM-2x8.nam`                      | Community Real | LSTM (2 layers, H=8)                 | Permissive / Boss Waza TAE               | Multi-layer LSTM; V1/V2 multi-SR weight layout tests.                   |
| `EVH-5150-Lite.nam`                     | Community Real | WaveNet Lite (CH=12, K=3, 20L)       | Non-distributable community              | Real WaveNet Lite capture; replaces obsolete synthetic fixture.         |
| `wavenet_a2_full.nam`                   | Synthetic      | WaveNet A2 (CH=8, K=6/15, 23L)       | Apache-2.0 / `generate_a2_fixtures.py`   | Fast-path A2 parity; calibrated audio regime (LUFS ≈ −22.6).            |
| `wavenet_a2_lite.nam`                   | Synthetic      | WaveNet A2 (CH=3, K=6/15, 23L)       | Apache-2.0 / `generate_a2_fixtures.py`   | Fast-path A2 parity; calibrated audio regime (LUFS ≈ −20.0).            |
| `wavenet_a2_film_full.nam`              | Synthetic      | WaveNet A2 (CH=8, 4 FiLM slots)      | Apache-2.0 / `generate_a2_fixtures.py`   | FiLM dynamic path parity against generic WaveNet reference.             |
| `wavenet_a2_film_lite.nam`              | Synthetic      | WaveNet A2 (CH=3, 4 FiLM slots)      | Apache-2.0 / `generate_a2_fixtures.py`   | FiLM dynamic path parity against generic WaveNet reference.             |
| `wavenet_a2_film_chaos_stress.nam`      | Synthetic      | WaveNet A2 (CH=3, non-identity FiLM) | Apache-2.0 / `generate_a2_fixtures.py`   | High-variance FiLM routing and dimension stress probe.                  |
| `wavenet_a2_film_input_mixin_pre.nam`   | Synthetic      | WaveNet A2 (CH=3, slot 2 FiLM only)  | Apache-2.0 / `generate_a2_fixtures.py`   | Regression fixture for Bug C1 (input_mixin_pre channel dimensions).     |
| `a2_example.nam`                        | Synthetic      | SlimmableContainer (A2 submodels)    | Apache-2.0 / `generate_a2_fixtures.py`   | Container routing and submodel swapping for A2 architectures.           |
| `a2_dynamic_gated_ch8.nam`              | Synthetic      | WaveNet A2 (CH=8, dynamic gating)    | Apache-2.0 / `generate_a2_fixtures.py`   | Dynamic gating inference engine validation.                             |
| `a2_dynamic_blended_ch3.nam`            | Synthetic      | WaveNet A2 (CH=3, dynamic blending)  | Apache-2.0 / `generate_a2_fixtures.py`   | Dynamic blending inference engine validation.                           |
| `convnet_test.nam`                      | Synthetic      | ConvNet (CH=8, 6 blocks, Tanh)       | Apache-2.0 / `generate_b1_2_fixtures.py` | ConvNet baseline architecture parity.                                   |
| `convnet_nobn.nam`                      | Synthetic      | ConvNet (CH=8, no BatchNorm)         | Apache-2.0 / `generate_fixtures.py`      | ConvNet batchnorm bypass topology validation.                           |
| `convnet_relu.nam`                      | Synthetic      | ConvNet (CH=8, ReLU activation)      | Apache-2.0 / `generate_fixtures.py`      | ConvNet ReLU activation path validation.                                |
| `convnet_silu.nam`                      | Synthetic      | ConvNet (CH=8, SiLU activation)      | Apache-2.0 / `generate_fixtures.py`      | ConvNet SiLU activation path validation.                                |
| `linear_test.nam`                       | Synthetic      | Linear (RF=4, direct FIR)            | Apache-2.0 / deterministic               | Direct FIR time-domain dot-product parity.                              |
| `linear_nobias.nam`                     | Synthetic      | Linear (RF=4, bias=0.0)              | Apache-2.0 / `generate_fixtures.py`      | Zero-bias linear inference validation.                                  |
| `linear_fft_rf{320,2048,4096,8192}.nam` | Synthetic      | Linear FFT (RF 320 to 8192)          | Apache-2.0 / deterministic               | Partitioned frequency-domain convolution cross-validation.              |
| `lstm_dyn_test.nam`                     | Synthetic      | LSTM Dynamic (1 layer, H=7)          | Apache-2.0 / `generate_b1_2_fixtures.py` | Uncatalogued hidden size routing to dynamic LSTM path.                  |
| `lstm_1x10.nam`                         | Synthetic      | LSTM (1 layer, H=10)                 | Apache-2.0 / `generate_fixtures.py`      | LSTM hidden dimension boundary validation.                              |
| `lstm_2x24.nam`                         | Synthetic      | LSTM (2 layers, H=24)                | Apache-2.0 / `generate_fixtures.py`      | Multi-layer LSTM scaling validation.                                    |
| `lstm_3x8.nam`                          | Synthetic      | LSTM (3 layers, H=8)                 | Apache-2.0 / `generate_fixtures.py`      | 3-layer recurrent topology validation.                                  |
| `wavenet_dyn_free.nam`                  | Synthetic      | WaveNetDyn (CH=7/4, free geom)       | Apache-2.0 / `generate_b1_2_fixtures.py` | Dynamic WaveNet path with non-power-of-two channel geometries.          |
| `keras_unsupported.json`                | Mock           | Legacy Keras / H5 dictionary         | Apache-2.0 / clean structure             | Verifies graceful rejection of unsupported legacy formats (F13).        |
| `mock_a2.nam`                           | Mock           | Zero weights / ReLU config           | Apache-2.0 / clean structure             | Verifies audio thread error transition (`RT_STATUS_MODEL_LOAD_FAILED`). |
| `wavenet_a1_secondary_act.nam`          | Mock           | WaveNet A1 (non-null secondary)      | Apache-2.0 / `generate_fixtures.py`      | Verifies rejection of unsupported secondary activations (F1/F5).        |
| `slimmable_container.nam`               | Mock           | Container (LSTM + WaveNet + Nano)    | Apache-2.0 / clean structure             | Validates heterogeneous topology routing across submodels.              |

### Tracked Known Gaps & Specific Guardrails

1. **`wavenet_a2_max.nam` (KB-A2-MAX retired — Fase 3, 2026-10-08):**
    - Official flagship model; production×C++ parity verified (V1 SNR 135.90 dB, V2 SNR 135.97 dB). Former fail-closed guard TR1.1 retired; active CI gate. The standalone f64 oracle still diverges (H0 Case C, oracle-only) and is non-gating. Detailed analysis in [`docs/cpp_parity_map.md`](cpp_parity_map.md) §4.3.
2. **`wavenet_condition_lstm.nam` (Upstream C++ Channel Mismatch):**
   - WaveNet outer model with embedded LSTM `condition_dsp`.
   - The upstream C++ `render` tool encounters an input channel mismatch (`input_size=1` vs `hidden_size=3`), blocking golden generation. Flagged with `skip_reason` in [`src/testing/catalog.rs`](../src/testing/catalog.rs); tests skip gracefully when the golden is absent.
3. **`slimmable_wavenet.nam` (Inference-Only Support):**
   - Supported for standalone inference. C++ multi-size parity is architecturally unfeasible because upstream `NeuralAmpModelerCore` lacks a channel-slicing API.
4. **`BossWN-lite.nam` (Obsolete):**
   - Legacy synthetic WaveNet Lite fixture (0.9 dB SNR). Superseded in all active gates by the real community capture `EVH-5150-Lite.nam` (≥ 105 dB SNR).

---

## 4. Golden Vector Matrix (V1 & V2 Multi-Sample-Rate)

Golden vectors capture the deterministic output of `NeuralAmpModelerCore`'s `render` tool when driven by standardized stress stimuli.

### Binary Layout (`.golden.bin`)

```text
[u32 num_samples]               4 bytes, Little-Endian
[f32 × N input samples]         N × 4 bytes, Little-Endian (Stimulus signal)
[f32 × N expected output]       N × 4 bytes, Little-Endian (C++ reference output)
```

### Stimulus Signals

- **Stress Signal v1 (`stress_signal.wav`):** 2048 samples @ 48 kHz (~42.7 ms). Contains a 220 Hz → 3520 Hz chirp sweep, guitar low-E harmonics (82/165/330/659 Hz), an isolated transient impulse (+0.9) at 25%, an attack-sustain-release envelope, and a fade tail for testing denormal/FTZ handling. Generated by `src/bin/gen_stress.rs --version v1`.
- **Stress Signal v2 (`stress_signal_v2_{sr}.wav`):** 5-second multi-rate stimulus (44.1k, 48k, 88.2k, 96k, 192k) concatenating 6 standardized segments:
  - `GA-1` (0.0–1.0s): Guitar Amp single-note Low-E with bend and vibrato.
  - `FRG-1` (1.0–2.0s): Full Rig Guitar power chord with ADSR envelope.
  - `P-1` / `P-2` (2.0–3.5s): Palm mutes (16 hits @ 120 BPM) and pinch harmonic train.
  - `BA-1` (3.5–4.5s): Bass amp Low-A (55 Hz) with 5 harmonics.
  - `PA-1` (4.5–5.0s): Post-amp chord ringing decay.

### V2 Multi-Sample-Rate Scopes

Governed by [`src/testing/catalog.rs::GOLDEN_GEN_CATALOG`](../src/testing/catalog.rs) (39 total entries: 24 active in V2, 15 V1-only):

| Scope               | Sample Rates                 | Target Models                                                                                                                                         | Architectural Rationale                                                                                                                                                                       |
|:------------------- |:---------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`AllRates`**      | 44.1k, 48k, 88.2k, 96k, 192k | `wavenet_feather`, `wavenet_nano`, `wavenet_lite`, `wavenet_a1_standard`                                                                              | Models without fixed `sample_rate` JSON field; validate multi-rate resampling and inference across all supported host rates.                                                                  |
| **`Exclude192k`**   | 44.1k, 48k, 88.2k, 96k       | `lstm_1x16`, `lstm_2x8`                                                                                                                               | At 192 kHz (960,000 samples), recursive LSTM hidden-state rounding causes cumulative drift exceeding 18 dB SNR. Capped at 96 kHz.                                                             |
| **`Sr48kOnly`**     | 48 kHz only                  | `wavenet_standard`, `lstm_official`, `wavenet_a2_full`, `wavenet_a2_lite`, `convnet_test`, `wavenet_official`, dynamic and non-distributable captures | Models declaring `"sample_rate": 48000` in their NAM JSON; C++ render tool enforces single-rate rendering for these files.                                                                    |
| **`v2_scope=none`** | None (V1 only)               | FiLM models (`wavenet_a2_film_*`), Linear FFT (`linear_fft_rf*`), container models                                                                    | C++ `a2_fast` rejects FiLM, falling back to generic WaveNet; dynamic FiLM paths are validated via live C++ cross-validation in [`tests/parity/cpp_parity.rs`](../tests/parity/cpp_parity.rs). |

---

## 5. Non-Neural & Ground-Truth Reference Fixtures

### CabSim Convolution Fixtures

The impulse response cabsim engine ([`src/dsp/cabsim/conv.rs`](../src/dsp/cabsim/conv.rs)) uses a dual-validation strategy:

1. **Inline Mathematical Oracle (Direct Convolution):**
   - UPOLS (Uniform Partitioned Overlap-Save) convolution is validated inline against an exact direct time-domain convolution ($O(N^2)$) reference in [`tests/models/cabsim_golden.rs`](../tests/models/cabsim_golden.rs).
   - Evaluates short (64), medium (512), long (8192), and stress (32768/65536) synthetic IRs generated with deterministic PCG random noise. Eliminates circular dependencies.
2. **C++ Cross-Validation Goldens:**
   - Generated from `dsp::ImpulseResponse` in `AudioDSPTools` using [`tests/fixtures/render_ir.cpp`](../tests/fixtures/render_ir.cpp).
   - Committed vectors: `golden_cabsim_cpp_short.bin` (64), `golden_cabsim_cpp_medium.bin` (512), `golden_cabsim_cpp_long.bin` (8192).
   - *Engine Constraint:* C++ `dsp::ImpulseResponse` hard-caps IR length at 8192 samples (`mMaxLength`). Tests exceeding 8192 samples are evaluated exclusively against the inline direct convolution oracle.

### A2-Max Intermediate C++ Anchors (Retired — Reproducible Forensic Record, Épico 1 Sprint 1.3)

> The four raw little-endian f32 tensors (`dump_array0.bin`, `dump_rechannel.bin`,
> `dump_condition_dsp.bin`, `dump_film.bin`, 712 KB total) were removed from git
> after KB-A2-MAX retirement: their value is forensic/historical, not gating —
> production f32 × C++ golden verifies at SNR 135.90 dB (V1) / 135.97 dB (V2)
> (see `docs/cpp_parity_map.md` §4.3), and the Axis-B structural guard
> (`test_structural_tests_contain_no_bin_references`) forbids `.bin` references
> in Phase 1 modules. Removal also resolves that guard violation by deletion.
>
> Reproduce at any time with `bash utils/render-a2-dumps.sh` (tracked
> [`utils/namcore-a2-dumps.patch`](../utils/namcore-a2-dumps.patch) instruments
> the pinned C++ v0.6.0 `NAM/dsp.cpp` and split WaveNet implementation
> `NAM/wavenet/model.cpp`; CMake `NAM_A2_DUMPS` gate defaults OFF; runtime capture
> needs `NAM_A2_DUMP_DIR`; offline-only). The script regenerates
> `tests/fixtures/dumps_a2_max/` (gitignored `*.bin`, tracked `manifest.json` +
> `capture_meta.txt`) and verifies the render reproduces the V1 golden output
> byte-identically.
>
> Last captured identities (SHA-256, 2026-10-07) for forensic verification of
> regenerated dumps:
>
> | File | SHA-256 |
> |:-----|:--------|
> | `dump_array0.bin` | `8764236e76c9f2f5ff68daff1a720aea35f7f1e840eece929cb72d62860acf4a` |
> | `dump_rechannel.bin` | `0552afcaebbc1fd45526b6dbcfb8c072d0193a2a91355227338806f107b41ec0` |
> | `dump_condition_dsp.bin` | `1227723e1466f390b60601f285bd744c204dfe21bf09cf75ffedd2e7eef1dceb` |
> | `dump_film.bin` | `b30a14a1aa1cd6d91d6331f54013f9cdc0fd0b4b518abfbf107329546a4d1587` |
>
> Shapes: `[2048, 3]` (24576 B), `[2048, 4]` (32768 B), `[2048, 8]` (65536 B),
> `[2048, 72]` (589824 B); frame-major f32 little-endian; stimulus is the
> 2048-frame, 48 kHz V1 input; frame zero is the first real input sample after
> the normal C++ reset/prewarm. FiLM slots ordered by layer, then `conv_pre`,
> `conv_post`, `input_mixin_pre`, `input_mixin_post`, `activation_pre`,
> `activation_post`, `layer1x1_post`, `head1x1_post` (slots 2 and 10 have
> 8 channels; other 14 have 4; exact per-frame offsets in regenerated
> `manifest.json`).
>
> Phase-1 input identities (SHA-256, 2026-10-07):
>
> | Asset | SHA-256 |
> |:------|:--------|
> | `models/wavenet_a2_max.nam` | `12384c6640e1126907b366584024c4abb129ac5920b3dc2d31b29e39315e820d` |
> | `stress_signal.wav` | `3d7d24609f8c004023b9560f7842994800bec96e4d7d35e63ed09080d8b57434` |
> | `golden_wavenet_a2_max.bin` | `7248380f90391c14b75269310b7f5b0fc6146f69a4264ffea4a5e13d483c182b` |

### f64 Reference Anchors (`tests/fixtures/f64_anchors/`)

- Generated by [`tests/fixtures/scripts/validate_oracle_f64.py`](../tests/fixtures/scripts/validate_oracle_f64.py) using an independent 64-bit floating-point NumPy implementation.
- Evaluates 256-sample chirp inputs across WaveNet, A2, A2+FiLM, ConvNet, and LSTM architectures.
- Consumed by [`tests/parity/reference_oracle_f64.rs`](../tests/parity/reference_oracle_f64.rs) to verify that the Rust f64 oracle achieves ESR < 1e-12 against ground truth before decomposing f32 numerical error budgets.

### Resampler Reference Vectors

- Generated by [`tests/fixtures/generate_resampler_reference.py`](../tests/fixtures/generate_resampler_reference.py) using `libsoxr` via ffmpeg (33-bit precision, Chebyshev passband).
- Files: `resampler_input_{rate}.f32` (10 log-spaced tones) and `resampler_ref_{from}_to_{to}.f32`.
- Covers conversions across 44100 Hz, 48000 Hz, and 96000 Hz. Verified by Goertzel tone analysis in [`src/dsp/resampler_test.rs`](../src/dsp/resampler_test.rs).

### EBU Tech 3341 / R 128 Compliance Sequences

- Deterministic mono 48 kHz float32 WAV signals generated by [`tests/fixtures/generate_ebu_sequences.py`](../tests/fixtures/generate_ebu_sequences.py):
  - `ebu_3341_1_sine_m23.wav`: 1 kHz sine target at −23.0 LUFS (± 0.1 LU).
  - `ebu_3341_7_sine_m33.wav`: 1 kHz sine target at −33.0 LUFS (± 0.1 LU).
  - `ebu_3341_sine_m18.wav`: 1 kHz sine target at −18.0 LUFS (± 0.1 LU).
  - `ebu_3341_dyn_alternating.wav`: Alternating −20/−46 dBFS signal testing BS.1770-4 two-pass gating.
- Verified exclusively by [`tests/models/ebu_lufs_compliance.rs`](../tests/models/ebu_lufs_compliance.rs).

### Multi-Resolution STFT & Spectral Fidelity Baseline

- **`mrstft_golden.bin`:** Binary reference generated by [`tests/fixtures/scripts/gen_mrstft_golden.py`](../tests/fixtures/scripts/gen_mrstft_golden.py) to validate multi-resolution spectral loss computation in [`tests/parity/parity_primitives.rs`](../tests/parity/parity_primitives.rs).
- **`spectral_fidelity_baseline.json`:** Multi-model baseline storing per-model spectral divergence limits for [`tests/models/spectral_fidelity.rs`](../tests/models/spectral_fidelity.rs).

---

## 6. Generation Pipeline & Maintenance Workflow

### Generator Scripts Catalog

| Script                                           | Primary Outputs                                                                    | Prerequisites                                               |
|:------------------------------------------------ |:---------------------------------------------------------------------------------- |:----------------------------------------------------------- |
| `tests/fixtures/golden_gen_build.sh`             | Orchestrates vendor checkout, C++ build, stimulus generation, and golden rendering | `cmake`, C++20 (`g++`/`clang++`), `cargo`, `python3`, `git` |
| `tests/fixtures/generate_a2_fixtures.py`         | Synthetic A2, FiLM, container, and dynamic gating `.nam` models                    | `python3`                                                   |
| `tests/fixtures/generate_b1_2_fixtures.py`       | Dynamic ConvNet, WaveNetDyn, and LstmDyn `.nam` models                             | `python3`                                                   |
| `tests/fixtures/generate_fixtures.py`            | Hidden-size LSTMs, ConvNet activation variants, linear `.nam` models               | `python3`                                                   |
| `tests/fixtures/generate_ebu_sequences.py`       | EBU R 128 / Tech 3341 compliance WAV signals                                       | `python3`                                                   |
| `tests/fixtures/generate_resampler_reference.py` | Polyphase multitone input and reference output buffers (`.f32`)                    | `python3`, `ffmpeg` (with `libsoxr`)                        |
| `tests/fixtures/scripts/gen_mrstft_golden.py`    | `mrstft_golden.bin`                                                                | `python3`, `numpy`                                          |
| `tests/fixtures/scripts/validate_oracle_f64.py`  | `f64_anchors/*.bin`                                                                | `python3`, `numpy`, C++20                                   |
| `src/bin/gen_stress.rs`                          | `stress_signal.wav`, `stress_signal_v2_{sr}.wav`                                   | `cargo`                                                     |
| `src/bin/wav_to_golden.rs`                       | Converts rendered WAVs and stimuli into `.golden.bin`                              | `cargo`                                                     |

### Full Regeneration Walkthrough

To perform a clean, reproducible regeneration of all golden vectors from scratch:

```bash
# 1. Ensure vendor mirrors match variables.env pins:
./utils/setup-third-party.sh

# 2. Run the 13-phase orchestrator (builds C++ renderers, generates models & goldens):
./tests/fixtures/golden_gen_build.sh
```

### Freshness Manifest Enforcement

The freshness manifest [`tests/fixtures/.golden_manifest.sha256`](../tests/fixtures/.golden_manifest.sha256) tracks the exact SHA-256 hashes of every `.nam` model and its corresponding `.golden.bin` file:

```text
<sha256_model> <sha256_golden> <model_filename> <golden_filename>
```

- **Automated Gate:** Verified fail-closed in Phase 2 of [`utils/tests-quick.sh`](../utils/tests-quick.sh) and by [`src/testing/freshness.rs`](../src/testing/freshness.rs).
- **Anti-Staleness Guarantee:** Modifying any model file without regenerating its golden vector fails the build, preventing silent model drift.

### Gate Calibration & Anti-Placebo Rules

Numerical acceptance gates are defined exclusively in code:

- **Single Source of Truth:** [`tests/common/validation.rs::get_calibrated_threshold()`](../tests/common/validation.rs).
- **Principle: "Every Golden Must Be Able to Fail":**
  1. No self-goldens (comparing output against itself).
  2. No neutralized thresholds (SNR ≤ 0 dB, ESR ≥ 1.0, or unbound MSE without rigid SNR/ESR compensation).
  3. No uncalibrated heuristic fallbacks.
- **Continuous Enforcement:** Meta-tests in [`tests/models/threshold_calibration.rs`](../tests/models/threshold_calibration.rs) (`test_all_golden_models_have_calibrated_thresholds`, `test_all_calibrated_entries_have_measurement_comments`, `test_all_thresholds_anti_placebo`) verify at test time that every golden entry has an active, documented measurement comment (`// Measured: SNR=..., ESR=...`) and non-placebo threshold boundaries.
