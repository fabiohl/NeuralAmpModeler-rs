<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# NeuralAmpModeler-rs

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/x86__64-lightgrey.svg) [![Crates.io](https://img.shields.io/crates/v/NeuralAmpModeler-rs.svg)](https://crates.io/crates/NeuralAmpModeler-rs) [![docs.rs](https://docs.rs/NeuralAmpModeler-rs/badge.svg)](https://docs.rs/crate/NeuralAmpModeler-rs) ![RT-Safe](https://img.shields.io/badge/RT--Safe-Zero--Alloc-brightgreen.svg) ![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--512-blueviolet.svg) ![Models](https://img.shields.io/badge/Models-WaveNet%20A1%20A2%20%7C%20LSTM%20%7C%20ConvNet-success.svg)

**NeuralAmpModeler-rs** is a high-performance, real-time neural inference DSP engine written in pure Rust. It provides the core DSP library for loading, building, and executing [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) models — WaveNet (A1/A2), LSTM, ConvNet, and Linear FIR/FFT — as well as impulse response (.wav) cabinet convolution.

Designed for embedding in audio hosts, CLAP plugins, standalone audio hosts, offline renderers, and embedded DSP pipelines, it guarantees **zero heap allocations**, **zero locks**, and **zero blocking system calls** on the real-time audio processing thread.

NeuralAmpModeler-rs is an independent public library for the wider audio and Rust communities. Public APIs and policies remain strictly host-agnostic and generally reusable; integration-specific logic belongs in downstream crates (such as standalone audio hosts, CLAP plugins, and real-time processing applications).

> **❤️‍🔥 NeuralAmpModeler-rs is in beta stage.** Feedback, bug reports, performance metrics, and patch contributions are very welcome!

---

## ⚡ Key Strengths & Architectural Highlights

* **Pure Rust & Zero-Allocation RT Safety:** Engineered from the ground up for absolute real-time audio determinism — zero heap allocations, zero mutex locks, and zero blocking I/O on the hot path. Parameter updates and control-plane commands pass through lock-free SPSC channels.
* **Extremely Fast SIMD Inference:** Hand-crafted AVX2 (`x86-64-v3`) baseline vectorization and optional AVX-512 multiversioning (BF16/VNNI) protected by statistical performance gates and real-time deadline tests.
* **Uncompromising Audio Parity:** Validated against three independent test oracles (canonical C++ NAMCore f32, double-precision f64 reference oracle, and cross-ISA parity). In quality audits, BossWN Standard measured `2.31e-14` ESR against NAMCore and BossLSTM 1x16 measured `8.50e-12`; paired f64-oracle ESR measured `9.05e-15` and `8.90e-13`, respectively.
* **Const-Generic Optimization:** Static WaveNet and LSTM profiles leverage Rust const generics so kernel sizes, receptive fields, and channel counts are known at compile time, enabling aggressive LLVM compiler optimization, register allocation, and SIMD loop unrolling.
* **Cabinet IR & DSP Pipeline:** Integrated partitioned FFT and direct FIR convolution engine for speaker cabinet impulse responses (.wav), paired with polyphase half-band anti-aliasing oversampling and Padé FastMath activations.
* **Flexible Host Integration:** Exposes a clean, host-agnostic API for model loading, DSP pipeline construction, dynamic quality switching (`.namb` bundles), and diagnostics. Designed to be embedded as an `rlib` dependency in any Rust application.

---

## 🥊 Feature Showcase ("Roofshoot")

| Feature / Attribute              | Technical Implementation                                                                 | Benefit & Impact                                                                     |
|:-------------------------------- |:---------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------ |
| **Inference Engine**             | Core WaveNet (A1/A2), LSTM (1-layer & 2-layer), ConvNet, and Linear topologies           | Complete model ecosystem compatibility with native Rust DSP speed                    |
| **RT Safety Determinism**        | Strict Zero Heap Drop, Zero Mutex Locks, Zero Hot-Path Logging                           | Guaranteed audio stability without dropouts/xruns under sub-millisecond deadlines    |
| **SIMD Acceleration**            | Mandatory `x86-64-v3` (AVX2/FMA) baseline + optional AVX-512 (BF16/VNNI) multiversioning | Ultra-low CPU usage (< 38 µs per 64-sample block on AMD Ryzen 7)                     |
| **Const-Generic Profiles**       | Rust const generics for channel counts (`CH=16`, `12`, `8`, `4`) and layer depths        | Enables compile-time LLVM loop unrolling and register allocation optimization        |
| **Numerical Parity Oracles**     | Verified against canonical C++ NAMCore f32, double-precision f64, and cross-ISA oracles  | Bit/float-exact accuracy matching C++ reference models (`< 1e-11` ESR / `2.31e-14`)  |
| **Cabinet IR Convolution**       | Partitioned FFT & Direct FIR convolution engine (.wav IRs)                               | Zero-latency, low-overhead speaker cabinet simulation                                |
| **Oversampling & Anti-Aliasing** | Half-band polyphase FIR filters (`off`, `2x`, `4x`)                                      | Attenuates non-linear high-frequency foldover/aliasing in high-gain amp models       |
| **Activation Math Modes**        | `Standard` (exact precision) vs `Fast` (Padé polynomial minimax approximations)          | User-selectable trade-off between floating-point precision and latency               |
| **Adaptive Compute Container**   | Multi-profile `.namb` bundle support with runtime fallback switching                     | Prevents audio dropouts by dynamically adjusting compute complexity under CPU spikes |
| **Comprehensive QA Suite**       | 1,000+ unit/integration tests, heap audit, soak, proptest, and Criterion benchmarks      | Enterprise-grade software stability and strict protection against regressions        |

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

## 🛠️ System Prerequisites

| Dependency                | Minimum Version                               | Package / Command     |
|:------------------------- |:--------------------------------------------- |:--------------------- |
| **CPU Architecture**      | `x86_64` with AVX2/FMA (`x86-64-v3` baseline) | `lscpu`               |
| **Rust Toolchain**        | ≥ 1.94.0 (Edition 2024)                       | `rustc --version`     |
| **Development Libraries** | `build-essential`, `pkg-config`, `cmake`      | See apt command below |

### Installation of System Build Dependencies (Debian / Ubuntu / Pop!_OS)

```bash
sudo apt update && sudo apt install -y build-essential pkg-config cmake
```

---

## 🚀 Quick Start — Installation & Usage

### Add to Your `Cargo.toml`

```toml
[dependencies]
NeuralAmpModeler-rs = "x.y.z"
```

For off-RT testing utilities and audio signal generators:

```toml
[dependencies]
NeuralAmpModeler-rs = { version = "x.y.z", features = ["testing"] }
```

### Feature Flags

| Feature          | Description                                                              |
|:---------------- |:------------------------------------------------------------------------ |
| `stereo`         | Enables multi-channel / stereo dual-model loader support                 |
| `testing`        | Exposes off-RT test utilities, signal generators, and perceptual metrics |
| `heap-audit`     | Enables heap-allocation auditing infrastructure                          |
| `long_bench`     | Enables long-form inference benchmarks                                   |
| `dynamic-engine` | Enables scalar fallback for non-standard A2 convolution geometries       |

---

### Code Examples

#### 1. Minimal Model Loading & Audio Processing

```rust
use std::path::Path;
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::models::NamModel; // trait providing `process()`
use neural_amp_modeler_rs::SystemSnapshot;

fn main() {
    // 1. Capture system hardware capabilities (SIMD features, CPU topology)
    let sys = SystemSnapshot::capture();

    // 2. Load a .nam or .namb neural model file
    let mut model_pair = load_and_build_model(
        Path::new("models/BossWN-standard.nam"),
        &sys,
        false, // mono processing
        LoadOptions::default(),
    ).expect("Failed to load NAM model");

    // 3. Process audio in block quanta (e.g. 64 samples)
    let input_buffer = vec![0.0_f32; 64];
    let mut output_buffer = vec![0.0_f32; 64];

    if let Some(ref mut model) = model_pair.model_l {
        model.process(&input_buffer, &mut output_buffer);
    }
}
```

#### 2. Full DSP Engine Pipeline (Model + Cabinet IR + Polyphase Oversampling)

```rust
use std::path::Path;
use neural_amp_modeler_rs::dsp::oversample::{OversampleEngine, OversampleFactor};
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::models::NamModel; // trait providing `process()`
use neural_amp_modeler_rs::SystemSnapshot;

const BLOCK_SIZE: usize = 128;

fn main() {
    let sys = SystemSnapshot::capture();

    // Load neural model
    let mut model_pair = load_and_build_model(
        Path::new("models/BossWN-standard.nam"),
        &sys,
        false,
        LoadOptions::default(),
    ).unwrap();
    let model = model_pair
        .model_l
        .as_mut()
        .expect("mono load (stereo=false) always yields model_l");

    // Create a 4x half-band polyphase oversampling engine (HQ mode)
    let os_factor = OversampleFactor::X4;
    let mut os_engine = OversampleEngine::new(os_factor, BLOCK_SIZE)
        .expect("Failed to create oversampling engine");
    let multiplier = os_factor.multiplier();

    // Process a real-time audio block through the oversampled pipeline
    let input = vec![0.1_f32; BLOCK_SIZE];
    let mut output = vec![0.0_f32; BLOCK_SIZE];
    let mut os_up_buf = vec![0.0_f32; BLOCK_SIZE * multiplier];
    let mut os_model_buf = vec![0.0_f32; BLOCK_SIZE * multiplier];

    let n_os = os_engine.upsample(&input, &mut os_up_buf, None);
    model.process(&os_up_buf[..n_os], &mut os_model_buf[..n_os]);
    os_engine.downsample(&os_model_buf[..n_os], &mut output, None);
}
```

#### 3. Executable Examples

`NeuralAmpModeler-rs` includes 6 runnable examples in `examples/` demonstrating key features:

| Example                                            | Description                                                                     | Run Command                                                 |
|:-------------------------------------------------- |:------------------------------------------------------------------------------- |:----------------------------------------------------------- |
| [`load_model`](examples/load_model.rs)             | Off-RT `.nam`/`.namb` model file loading & SIMD prewarming                      | `cargo run --example load_model -- <path/to/model.nam>`     |
| [`inspect_model`](examples/inspect_model.rs)       | Detailed inspection & metadata report of `.nam`/`.namb` files (Text/JSON/Batch) | `cargo run --example inspect_model -- <path/to/model.nam>`  |
| [`offline_render`](examples/offline_render.rs)     | Offline audio rendering with 4× polyphase oversampling (HQ mode)                | `cargo run --example offline_render -- <path/to/model.nam>` |
| [`cabsim`](examples/cabsim.rs)                     | Standalone cabinet impulse response (IR) convolution & resampling               | `cargo run --example cabsim -- <path/to/ir.wav>`            |
| [`diagnostics`](examples/diagnostics.rs)           | Circular log buffer (`LogBuffer`) & support bundle (`DiagnosticBundle`) export  | `cargo run --example diagnostics`                           |
| [`math_activations`](examples/math_activations.rs) | Performance and accuracy comparison of SIMD activations (`Standard` vs `Fast`)  | `cargo run --example math_activations`                      |

---

### Rustdoc Module Map

| Module     | Purpose                                                      |
|:---------- |:------------------------------------------------------------ |
| [`loader`] | Model deserialization & construction (`.nam`, `.namb`)       |
| [`math`]   | SIMD math primitives, activation approximations, DSP kernels |
| [`models`] | Neural network architectures & `StaticModel` dispatch        |
| [`dsp`]    | DSP engine: resampling, gating, oversampling, pipeline       |
| [`common`] | Diagnostics, atomic bitmasks, lock-free SPSC queues          |
| `testing`  | Off-RT test utilities & perceptual metrics (feature-gated)   |

Full API documentation:

* **Local Generation:** `cargo doc --open`
* **Docs.rs Environment Simulation:** `DOCS_RS=1 cargo doc --features "stereo,testing" --no-deps`
* **Online Documentation:** [docs.rs/NeuralAmpModeler-rs](https://docs.rs/NeuralAmpModeler-rs)

> **Note on `docs.rs` Builds:** `NeuralAmpModeler-rs/build.rs` detects `DOCS_RS=1` to early-return before `avx2+fma` CPU target feature assertions. This allows `docs.rs` builders (running on baseline x86-64 without AVX2) to document the API successfully. For new `crates.io` releases or manual doc rebuild requests, use the `docs.rs` re-trigger queue at `https://docs.rs/crate/NeuralAmpModeler-rs/latest/builds`.

---

## 🏆 Quality & Performance

* **Numerical Fidelity:** The quality contract tracks NAMCore parity, independent f64-oracle error, SNR, and MR-STFT instead of relying on a single metric.
* **Measured CPU Headroom:** On the logged AMD Ryzen 7 5700U run, WaveNet Standard CH16 processed a 64-sample block in **37.5 µs** (**2.8%** of the 1.33 ms deadline), while LSTM 1x16 used **7.4 µs** (**0.6%**).
* **Stress Coverage:** Soak, concurrency, heap-audit, deadline, and model-checking suites exercise long-running and real-time invariants; skipped coverage and failed audit phases must be reviewed separately from passing checks.
* **SIMD Acceleration:** AVX2 baseline (`x86-64-v3`), AVX-512 multiversioning with BF16/VNNI on supported hardware (Intel Sapphire Rapids+, AMD Zen 4+). FastMath activations (tanh, sigmoid) via Padé/minimax polynomial approximations.

---

## 🧰 Local Development Environment (engine maintainers)

Crate **consumers** only need a Rust toolchain and the published crate — no vendor trees.

Developers working **on NeuralAmpModeler-rs itself** (parity, golden regeneration, cabsim C++
cross-validation, optional community-model tests) should prepare a local `third-party/` tree
after clone. That directory is **gitignored** and is never part of the published package:

```bash
# From the NeuralAmpModeler-rs repository root:
./utils/setup-third-party.sh

# Optional: link a private non-distributable model archive
NAM_COMMUNITY_MODELS_SRC=/path/to/your/nam_models ./utils/setup-third-party.sh
# or: ln -s /path/to/your/nam_models third-party/community_models
```

| Path | Role |
| :----- | :----- |
| `third-party/NeuralAmpModelerCore/` | Pinned C++ NAMCore mirror (render / parity) |
| `third-party/NeuralAmpModelerPlugin/` | Pinned C++ plugin mirror (IR / cabsim xref) |
| `third-party/community_models/` | Optional symlink to local community models (not redistributable) |
| `variables.env` | Version-controlled pin file (tags/commits/URLs) |

Tests and scripts **skip gracefully with a declared gap** when these artifacts are missing:
the quick suite records `GAP:` lines in `target/logs/quick-receipt.txt`, prints
`FIDELITY: INCOMPLETE` / `OVERALL: PASSED_WITH_GAPS` (exit 0), and **never emits a green
fidelity seal for skipped oracles** — `NAM_QUICK_STRICT=1` promotes those gaps to FAIL
(exit 1). The long suite requires the NAMcore mirror outright (hard abort if absent).
Override locations with `NAM_THIRD_PARTY_DIR`, `NAM_CORE_DIR`, `NAM_PLUGIN_DIR`, or
`NAM_MODELS_DIR` if needed.
See [`tests/fixtures/README.md`](tests/fixtures/README.md) for the full model search order.

Rust dependency supply-chain updates remain separate: `./utils/mod-update.sh`.

---

## 🧪 CI & QA Automation Suite (`./utils/`)

The `./utils/` directory contains maintainer tools and standard scripts for code quality, numerical verification, and continuous integration:

| Script                                                           | Purpose & Execution Scope                                                                                                                                                                                                                                                          |
|:---------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`utils/setup-third-party.sh`](utils/setup-third-party.sh)       | **Local env bootstrap:** Clones/syncs pinned NAMCore + Plugin mirrors into `third-party/` and optionally links `community_models`. Required for full parity/golden work; not needed by crate consumers.                                                                            |
| [`utils/mod-update.sh`](utils/mod-update.sh)                     | **Rust supply chain:** Updates rustup toolchain, `cargo upgrade`, and `Cargo.lock` (does **not** manage vendor mirrors).                                                                                                                                                           |
| [`utils/lints.sh`](utils/lints.sh)                               | **Static Analysis Gate:** Runs `cargo fmt`, strict `cargo clippy`, compilation checks (`cargo check`), zero-warning doc-tests, and verifies SPDX license headers across all repository source files.                                                                               |
| [`utils/tests-quick.sh`](utils/tests-quick.sh)                   | **Agile 1st Line QA:** 3 phases — structural tests (debug), measurement oracles + C++ parity `quick_parity` (release), capped parser fuzzing (`NAM_QUICK_PROPTEST_CASES`). Oracle skips are fail-closed: missing fixtures/toolchain print `FIDELITY: INCOMPLETE` + `OVERALL: PASSED_WITH_GAPS` (exit 0) and write the receipt `target/logs/quick-receipt.txt`; `NAM_QUICK_STRICT=1` promotes gaps to FAIL (exit 1). Re-executes itself at low CPU/IO priority unless `NAM_NO_LOW_PRIORITY=1`.                                                                                                                               |
| [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh)       | **Regression & Quality Gate:** Executes Criterion benchmarks and verifies audio fidelity against `docs/quality-contract.txt`.                                                                                                                                                                                                                |
| [`utils/check-model.sh`](utils/check-model.sh)                   | **Model Inspector Wrapper:** Canonical tool backed by `examples/inspect_model.rs`. Inspects `.nam` & `.namb` files, outputting detailed human-readable reports, JSON (`--json`), or batch arrays (`--manifest`).                                                                                                                               |
| [`utils/tests-long.sh`](utils/tests-long.sh)                     | **Nightly / Pre-Release Suite:** Rust-gated pre-flight (`catalog_preflight` V1/V2 golden catalogs fail-closed + `check_freshness` manifest; no bash golden lists), soak, full proptest/fuzz, full C++ parity matrix, cross-ISA, RT-safety and heap-audits. Exits `OVERALL: FAILED` (1) / `COMPLETED_WITH_GAPS` (0) / `PASSED` (0); `--strict-pre-release` turns declared gaps into failure. *(AI agents must not run this script directly due to runtime length; ask the human operator.)* |

Exact QA commands:

```bash
# 1. Static analysis (fmt, SPDX, check, clippy)
./utils/lints.sh

# 2. Agile first line (AI tasks: at most once, as final validation)
./utils/tests-quick.sh

#    Release-gate mode: skipped oracles (missing fixtures/C++ toolchain) become FAIL
NAM_QUICK_STRICT=1 ./utils/tests-quick.sh

#    Receipt + per-phase logs of the last run
cat target/logs/quick-receipt.txt   # plus target/logs/quick-phase{1,2,3}.log
# 3. Nightly / pre-release audit — HUMAN OPERATOR ONLY (AI agents must never run it)

./utils/tests-long.sh
./utils/tests-long.sh --strict-pre-release

#    Human certification protocol (checklist + evidence record for both runners)

#    docs/runners-human-certification.md
```

---

## 📚 Architecture & Engineering Documentation

The following technical documents are maintained in the source repository. The public Rust API is documented on [docs.rs](https://docs.rs/NeuralAmpModeler-rs).

| Document                                                                                       | Primary Focus & Topic Coverage                                                               |
|:---------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------------------------------- |
| [`docs/architecture.md`](docs/architecture.md)                                                 | Engine architecture, SIMD microarchitecture, mixed precision math, and `.namb` format design |
| [`docs/audio_fidelity_map.md`](docs/audio_fidelity_map.md)                                     | DSP decision quality trade-off matrix and frequency response analysis                        |
| [`docs/fastmath-approximations.md`](docs/fastmath-approximations.md)                           | Activation function approximations (Padé / minimax polynomials) and error bound benchmarks   |
| [`docs/namb-spec.md`](docs/namb-spec.md)                                                       | Binary `.namb` multi-profile container specification, metadata schema, and CRC32 layout      |
| [`docs/testing.md`](docs/testing.md)                                                           | Test suite layout, verification phases, oracle hierarchy, and testing policies               |
| [`docs/perceptual_validation.md`](docs/perceptual_validation.md)                               | Perceptual measurement framework (ESR, MR-STFT, ASR, LUFS) and auditory distance metrics     |
| [`docs/cpp_parity_map.md`](docs/cpp_parity_map.md)                                             | Bit-exact and float-exact parity audit against canonical C++ NeuralAmpModelerCore            |
| [`docs/benchmarks.md`](docs/benchmarks.md)                                                     | Criterion benchmark methodology, throughput profiles, and performance regression gates       |
| [`docs/research-references.md`](docs/research-references.md)                                   | Scientific literature, DSP reference bibliography, and deep learning modeling research       |
| [`docs/functional-tests.md`](docs/functional-tests.md)                                         | Engine functional test checklist and verification matrices                                   |
| [`docs/runners-human-certification.md`](docs/runners-human-certification.md)                   | Human-only certification protocol for `tests-quick.sh` / `tests-long.sh` (checklist + evidence record) |
| [`docs/postmortem-libm-symbol-interposition.md`](docs/postmortem-libm-symbol-interposition.md) | Technical postmortem on libm symbol interposition resolution on Linux dynamic linkers        |
| [`docs/quality-contract.txt`](docs/quality-contract.txt)                                       | Quality contract: benchmark and audio fidelity regression baseline thresholds                |
| [`tests/fixtures/README.md`](tests/fixtures/README.md)                                         | Golden vector formats, stress signal generation, and non-distributable test model fixtures   |

---

## 🤝 Contributing & Feedback

* **Test Models:** Try your favorite `.nam` models and IR files — share your feedback and performance metrics.
* **Report Issues:** Submit detailed bug reports or feature suggestions on GitHub.
* **Code & Docs:** Pull requests for SIMD optimizations, bug fixes, or documentation enhancements are very welcome.

---

## 🙏 Credits & Acknowledgments

* **Steven Atkinson** — Creator of [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler) for pioneering deep learning amplifier modeling and sharing the ecosystem with the community.
* **Mike Oliphant** — Author of [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio), whose codebase provided invaluable insight into WaveNet inference in the early stages.

---

## ⚖️ License & AI Transparency

### AI Transparency Note

The system architecture, core DSP engineering decisions, mathematical verification framework, and project orchestration are the intellectual work of the maintainer (**Fábio Henrique de Lima Silva**). The implementation was accelerated through pair programming (*Vibe Coding*) using artificial intelligence models (Gemini, Claude, DeepSeek) within Google Antigravity IDE and Kilo Code.

### License

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for details.
