<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# NeuralAmpModeler-rs 0.1.0

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/x86__64-lightgrey.svg)

**NeuralAmpModeler-rs** is a high-performance, real-time neural inference DSP library written in pure Rust. It provides the core engine for loading, building, and executing [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) models — WaveNet (A1/A2), LSTM, ConvNet, and Linear — as well as impulse response (.wav) cabinet convolution. Designed for embedding in audio hosts, offline renderers, analysis tools, and embedded DSP pipelines, it guarantees zero heap allocations, zero locks, and zero blocking I/O on the real-time audio processing thread.

NeuralAmpModeler-rs is an independent public library for the wider audio and Rust communities. No particular application, audio backend, plugin format, or downstream consumer owns its architecture. Public APIs and policies are expected to remain host-agnostic and generally reusable; integration-specific behavior belongs in downstream crates.

> **❤️‍🔥 NeuralAmpModeler-rs is in beta stage.** Feedbacks, bug reports, suggestions, and testing are very welcome (and needed 😉)!

---

## ⚡ Key Strengths

* **Pure Rust & Zero-Allocation RT Safety:** Engineered from the ground up for absolute real-time determinism: zero heap allocations, zero locks, zero blocking I/O on the hot path. Lock-free SPSC channels handle all control-plane communication.
* **Extremely Fast SIMD Inference:** Hand-crafted AVX2 (`x86-64-v3`) baseline vectorization and optional AVX-512 kernels are protected by statistical performance gates and real-time deadline tests.
* **Uncompromising Audio Parity:** Validated against three independent test oracles (canonical C++ NAMCore f32, double-precision f64 reference oracle, and cross-ISA parity). In the 2026-08-02 quality run, BossWN Standard measured `2.31e-14` ESR against NAMCore and BossLSTM 1x16 measured `8.50e-12`; paired f64-oracle ESR was `9.05e-15` and `8.90e-13`, respectively.
* **Const-Generic Optimization:** Static WaveNet and LSTM variants leverage Rust const generics so kernel sizes and channel counts are known at compile time, enabling aggressive LLVM compiler optimization such as SIMD, instruction reordering, and loop unrolling.
* **Flexible Host Integration:** Exposes a clean, host-agnostic API for model loading, inference, resampling, gating, and cab simulation. Designed to be embedded as an `rlib` dependency in any Rust audio application.

---

## 🧠 Supported Architectures

| Architecture            | Static Profiles                                           | Dynamic Fallback  |
|:----------------------- |:--------------------------------------------------------- |:----------------- |
| **WaveNet A1**          | Standard (CH=16), Lite (12), Feather (8), Nano (4)        | `WaveNetModelDyn` |
| **WaveNet A2**          | Full (CH=8), Lite (CH=3), Cascade                         | `WaveNetA2Dyn`    |
| **LSTM**                | 10 profiles: 1-layer (hidden 3–40), 2-layer (hidden 8–24) | `LstmModelDyn`    |
| **ConvNet**             | Feed-forward causal conv1d + BatchNorm1D + activation     | —                 |
| **Linear**              | Direct FIR or Partitioned FFT convolution                 | —                 |
| **Slimmable Container** | Multi-submodel bundles with runtime quality transitions   | —                 |

---

## 🚀 Quick Start — Installation & Usage

### Add to Your `Cargo.toml`

```toml
[dependencies]
NeuralAmpModeler-rs = "0.1.0"
```

For off-RT testing utilities and audio signal generators:

```toml
[dev-dependencies]
NeuralAmpModeler-rs = { version = "0.1.0", features = ["testing"] }
```

### Feature Flags

| Feature          | Description                                                              |
|:---------------- |:------------------------------------------------------------------------ |
| `stereo`         | Enables multi-channel / stereo dual-model loader support                 |
| `testing`        | Exposes off-RT test utilities, signal generators, and perceptual metrics |
| `heap-audit`     | Enables heap-allocation auditing infrastructure                          |
| `long_bench`     | Enables long-form inference benchmarks                                   |
| `pgo`            | Build with PGO (Profile-Guided Optimization) support                     |
| `dynamic-engine` | Enables scalar fallback for non-standard A2 convolution geometries       |

### Minimal Example

```rust
use std::path::Path;
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::SystemSnapshot;

// Capture system capabilities (SIMD feature set, CPU topology)
let sys = SystemSnapshot::capture();

// Load a .nam or .namb model file
let model_pair = load_and_build_model(
    Path::new("path/to/model.nam"),
    &sys,
    false, // mono
    LoadOptions::default(),
).expect("Failed to load model");

// Process audio in blocks
let mut input = vec![0.0_f32; 64];
let mut output = vec![0.0_f32; 64];
if let Some(ref mut model) = model_pair.model_l {
    model.process(&input, &mut output);
}
```

### Rustdoc Module Map

| Module     | Purpose                                                      |
|:---------- |:------------------------------------------------------------ |
| [`loader`] | Model deserialization & construction (`.nam`, `.namb`)       |
| [`math`]   | SIMD math primitives, activation approximations, DSP kernels |
| [`models`] | Neural network architectures & `StaticModel` dispatch        |
| [`dsp`]    | DSP engine: resampling, gating, oversampling, pipeline       |
| [`common`] | Diagnostics, atomic bitmasks, lock-free SPSC queues          |
| `testing`  | Off-RT test utilities & perceptual metrics (feature-gated)   |

Full API documentation: `cargo doc --open`

---

## 🏆 Quality & Performance

* **Numerical Fidelity:** The quality contract tracks NAMCore parity, independent f64-oracle error, SNR, and MR-STFT instead of relying on a single metric.
* **Measured CPU Headroom:** On the logged AMD Ryzen 7 5700U run, WaveNet Standard CH16 processed a 64-sample block in **37.5 us** (**2.8%** of the 1.33 ms deadline), while LSTM 1x16 used **7.4 us** (**0.6%**).
* **Stress Coverage:** Soak, concurrency, heap-audit, deadline, and model-checking suites exercise long-running and real-time invariants; skipped coverage and failed audit phases must be reviewed separately from passing checks.
* **SIMD Acceleration:** AVX2 baseline (`x86-64-v3`), AVX-512 multiversioning with BF16/VNNI on supported hardware (Intel Sapphire Rapids+, AMD Zen 4+). FastMath activations (tanh, sigmoid) via Padé/minimax polynomial approximations.

---

## 📚 Documentation

The following engineering documents are maintained in the source repository. The public Rust API is documented on [docs.rs](https://docs.rs/NeuralAmpModeler-rs).

* [docs/architecture.md](docs/architecture.md) — Architecture: engine pipeline, SIMD microarchitecture, mixed precision, NAMB format
* [docs/audio_fidelity_map.md](docs/audio_fidelity_map.md) — DSP decision quality trade-off map
* [docs/fastmath-approximations.md](docs/fastmath-approximations.md) — Activation approximation benchmarks and math details
* [docs/namb-spec.md](docs/namb-spec.md) — Binary `.namb` container spec and CRC32 layout
* [docs/testing.md](docs/testing.md) — Test suite layout, execution phases, and verification rules
* [docs/perceptual_validation.md](docs/perceptual_validation.md) — Perceptual measurement framework (ESR, MR-STFT, ASR, LUFS)
* [docs/cpp_parity_map.md](docs/cpp_parity_map.md) — Parity audit against NeuralAmpModelerCore
* [docs/benchmarks.md](docs/benchmarks.md) — Criterion benchmark methodology and regression gates
* [docs/research-references.md](docs/research-references.md) — Scientific literature and DSP reference bibliography
* [docs/functional-tests.md](docs/functional-tests.md) — Engine functional test checklist
* [docs/postmortem-libm-symbol-interposition.md](docs/postmortem-libm-symbol-interposition.md) — libm symbol interposition resolution analysis
* [docs/quality-contract.txt](docs/quality-contract.txt) — Quality contract: benchmark and audio fidelity regression baseline gates
* [tests/fixtures/README.md](tests/fixtures/README.md) — Golden vector formats and stress signal generation

---

## 🧪 Tests & Quality Assurance

Over **1,000 automated unit and integration checks**, plus long-running audit phases, are backed by three independent verification oracles:

| Oracle                | Purpose                                    | Test Suite                             |
|:--------------------- |:------------------------------------------ |:-------------------------------------- |
| **C++ NAMCore** (f32) | Parity with canonical C++ implementation   | `tests/parity/cpp_parity.rs`           |
| **f64 Reference**     | Absolute mathematical accuracy             | `tests/parity/reference_oracle_f64.rs` |
| **ISA Parity**        | Vectorization consistency across platforms | `tests/parity/isa_parity.rs`           |

### Automated Test Scripts

These maintainer commands require a source repository checkout; they are not runtime requirements for downstream users.

```bash
# Formatting, SPDX checks, and Clippy lints
utils/lints.sh

# Fast test suite (unit, integration, measurement oracles)
utils/tests-quick.sh

# Quality Dashboard (benchmarks & audio fidelity contract check)
utils/quality-dashboard.sh --check docs/quality-contract.txt

# Model diagnostic check (validates .nam file metadata)
python3 utils/check-model.py path/to/model.nam
```

---

## 🤝 Contributing & Feedback

* **Test Models:** Try your favorite `.nam` models and IR files — share your experience.
* **Report Issues:** Submit detailed bug reports or feature suggestions on GitHub.
* **Code & Docs:** Pull requests for optimizations, bug fixes, or documentation enhancements are welcome.

---

## 🙏 Credits & Acknowledgments

* **Steven Atkinson** — Creator of [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler) for pioneering deep learning amplifier modeling and sharing the ecosystem with the community.
* **Mike Oliphant** — Author of [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio), whose codebase provided invaluable insight into WaveNet inference in the early stages.

---

## ⚖️ License & AI Transparency

### AI Transparency Note

The architecture, core engineering decisions, mathematical verification framework, and project orchestration are the intellectual work of the maintainer. The implementation was accelerated through pair programming with Artificial Intelligence (*Vibe Coding*) using models like Gemini, Claude, and DeepSeek within Google Antigravity IDE and Kilo Code.

### License

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for details.
