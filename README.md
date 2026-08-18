<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# NeuralAmpModeler-rs

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/x86__64-lightgrey.svg) [![Crates.io](https://img.shields.io/crates/v/NeuralAmpModeler-rs.svg)](https://crates.io/crates/NeuralAmpModeler-rs) [![docs.rs](https://docs.rs/NeuralAmpModeler-rs/badge.svg)](https://docs.rs/crate/NeuralAmpModeler-rs) ![RT-Safe](https://img.shields.io/badge/RT--Safe-Zero--Alloc-brightgreen.svg) ![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--512-blueviolet.svg) ![Models](https://img.shields.io/badge/Models-WaveNet%20A1%20A2%20%7C%20LSTM%20%7C%20ConvNet-success.svg)

> **Series note:** the current release series is `0.x`; the `3.x` versions on crates.io/docs.rs are yanked leftovers of an earlier monolithic packaging.

**NeuralAmpModeler-rs** is a high-performance, real-time neural inference DSP engine written in pure Rust. It provides the core DSP library for loading, building, and executing [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) models — WaveNet (A1/A2), LSTM, ConvNet, and Linear FIR/FFT — as well as impulse response (.wav) cabinet convolution.

Designed for embedding in audio hosts, CLAP plugins, standalone audio hosts, offline renderers, and embedded DSP pipelines, it guarantees **zero heap allocations**, **zero locks**, and **zero blocking system calls** on the real-time audio processing thread.

NeuralAmpModeler-rs is an independent public library for the wider audio and Rust communities. Public APIs and policies remain strictly host-agnostic and generally reusable; integration-specific logic belongs in downstream crates (such as standalone audio hosts, CLAP plugins, and real-time processing applications).

> **❤️‍🔥 NeuralAmpModeler-rs is in beta stage.** Feedback, bug reports, performance metrics, and patch contributions are very welcome!

---

## ⚡ Key Strengths & Architectural Highlights

* **Pure Rust & Strict Zero-Allocation RT Safety:** Engineered from the ground up for absolute real-time audio determinism — zero heap allocations, zero mutex locks, and zero blocking syscalls on the audio processing thread (verified via `CountingAllocator` in heap audit suites). Parameter updates and model swaps pass through lock-free SPSC channels, while a 3-tier GC cascade (*SPSC queue → 16-slot thread-local parking lot → overwrite ring*) guarantees safe off-RT resource disposal without audio glitches.
* **Extremely Fast SIMD Inference & Zero-Vtable Dispatch:** Mandatory `x86-64-v3` (AVX2/FMA/BMI2) baseline vectorization and optional AVX-512 multiversioning (BF16/VNNI) resolved at compile time via the `dispatch_simd!` macro with zero function pointers or vtables. Multi-accumulator ILP (`sum0..sum3` in AVX2, `acc0..acc7` in AVX-512) and tap-major frame-tiling saturate hardware execution ports.
* **Uncompromising Dual-Oracle Audio Parity:** Validated against two independent co-equal test oracles: canonical C++ NAMCore f32 (market interop) and double-precision f64 reference oracle (mathematical ideality). Grounded in `docs/quality-contract.json`, BossWN Standard measures `2.31e-14` ESR against NAMCore and `9.05e-15` against the f64 oracle (SNR `136.4 dB`, MR-STFT `6.46e-6`), with ConvNet reaching `4.23e-15` ESR (SNR `143.7 dB`).
* **Const-Generic Optimization & Dynamic Topology Fallback:** 23 static model profiles leverage Rust const generics so kernel sizes, receptive fields, and channel counts (`CH=16, 12, 8, 4, 3`) are known at compile time, enabling aggressive LLVM loop unrolling and register allocation. Non-standard topologies gracefully fallback to zero-allocation dynamic engines (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`, `WaveNetA2Cascade`).
* **Complete Native DSP Stack (Zero External Audio Crates):** Integrated native Minimum-Phase Polyphase FIR Sinc Resampler (256 phases × 64 taps, Kaiser $\beta=12$, >105 dB stopband, zero pre-ringing cepstrum, 0.7–1.3 µs latency, replacing external libraries like `rubato`), UPOLS Partitioned FFT CabSim IR convolution (1.3 µs), and multi-stage Half-Band FIR Anti-Aliasing Oversampling (2×/4×, >100 dB attenuation, Kahles et al. JAES 2019).
* **Adaptive Compute FSM & Pre-Transposed `.namb` v2 Container:** Dynamic CPU load monitoring with hysteresis FSM that gracefully degrades model complexity (Full → Reduced → Minimal) with 32 ms click-free linear crossfades to prevent audio xruns. Binary `.namb` v2 container (Gate-Major LSTM, Interleaved-4 WaveNet) reduces model hot-swap latency from ~50 ms to <1 ms with mandatory IEEE 802.3 CRC32 integrity validation.
* **Denormal & Subnormal Armor:** Injected symmetric `−220 dBFS` deterministic dither (`1.0e-11`) + SSE2 MXCSR FTZ/DAZ reassertion on every processing call, eliminating 10–100× CPU microcode penalties on digital silence with zero net DC drift.

---

## 🥊 Feature Showcase ("Roofshoot")

| Feature / Attribute              | Technical Implementation                                                                      | Benefit & Impact                                                                      |
|:-------------------------------- |:--------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------- |
| **Inference Engine**             | Core WaveNet (A1/A2), LSTM (1-layer & 2-layer), ConvNet, and Linear topologies                | Complete model ecosystem compatibility with native Rust DSP speed                     |
| **RT Safety Determinism**        | Strict Zero Heap Drop, Zero Mutex Locks, 3-Tier Lock-Free GC Cascade                          | Guaranteed audio stability without dropouts/xruns under sub-millisecond deadlines     |
| **SIMD Acceleration**            | Mandatory `x86-64-v3` (AVX2/FMA) baseline + optional AVX-512 (BF16/VNNI) multiversioning      | Ultra-low CPU usage (< 37 µs for WaveNet Std, 7.5 µs for LSTM on AMD Ryzen 7)         |
| **Const-Generic Profiles**       | 23 static variants with Rust const generics for channel counts (`CH=16`, `12`, `8`, `4`, `3`) | Enables compile-time LLVM loop unrolling and register allocation optimization         |
| **Numerical Parity Oracles**     | Verified against canonical C++ NAMCore f32, double-precision f64, and cross-ISA oracles       | Bit/float-exact accuracy matching C++ reference models (`2.31e-14` to `4.23e-15` ESR) |
| **Native Polyphase Resampler**   | 256 phases × 64 taps Kaiser sinc resampler (minimum-phase cepstrum, 0 pre-ringing)            | Pristine multi-rate conversion (>105 dB stopband, < 0.05 dB ripple) in 0.7–1.3 µs     |
| **Cabinet IR Convolution**       | Uniform-Partitioned Overlap-Save (UPOLS) FFT convolution engine (.wav IRs)                    | Ultra-low overhead speaker cabinet simulation (1.3 µs for 512-sample IRs)             |
| **Oversampling & Anti-Aliasing** | Half-band polyphase FIR filters (`off`, `2x`, `4x`, >100 dB stopband)                         | Attenuates non-linear high-frequency foldover/aliasing in high-gain amp models        |
| **Activation Math Modes**        | `Standard` (exact precision Taylor minimax) vs `Fast` (Padé polynomial minimax)               | User-selectable trade-off between floating-point precision (+89.5 dB SNR) & latency   |
| **Adaptive Compute Container**   | Multi-profile `.namb` bundle support with runtime fallback switching                          | Prevents audio dropouts by dynamically adjusting compute complexity under CPU spikes  |
| **Binary `.namb` v2 Format**     | Pre-transposed memory layout (Gate-Major LSTM, Interleaved-4 WaveNet) with CRC32              | Reduces model loading / hot-swap time from ~50 ms to < 1 ms                           |
| **Denormal Armor**               | Symmetric `−220 dBFS` dither injection + hardware MXCSR FTZ/DAZ                               | Prevents 10–100× CPU microcode stalls on silence with zero DC drift                   |
| **Comprehensive QA Suite**       | 1,000+ unit/integration tests, heap audit, soak, proptest, and Criterion benchmarks           | Enterprise-grade software stability and strict protection against regressions         |

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

| Feature          | Description                                                                                            |
|:---------------- |:------------------------------------------------------------------------------------------------------ |
| `stereo`         | Enables multi-channel / stereo dual-model loader support                                               |
| `testing`        | Exposes off-RT test utilities, signal generators, and perceptual metrics                               |
| `heap-audit`     | Enables heap-allocation auditing infrastructure                                                        |
| `long_bench`     | Enables long-form inference benchmarks                                                                 |
| `dynamic-engine` | Enables generic dynamic-dimension fallback execution paths for arbitrary non-standard model topologies |

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

#### 2. Full DSP Engine Pipeline (Model + Cab IR + Polyphase Oversampling)

For the complete pipeline — model, cabinet IR, and 4× polyphase oversampling — see the
[`offline_render`](examples/offline_render.rs) example (`cargo run --example offline_render -- <path/to/model.nam>`).
The API surface used there is documented in the [crate docs](https://docs.rs/NeuralAmpModeler-rs).

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

| Module                  | Purpose                                                      |
|:----------------------- |:------------------------------------------------------------ |
| [`loader`](src/loader/) | Model deserialization & construction (`.nam`, `.namb`)       |
| [`math`](src/math/)     | SIMD math primitives, activation approximations, DSP kernels |
| [`models`](src/models/) | Neural network architectures & `StaticModel` dispatch        |
| [`dsp`](src/dsp/)       | DSP engine: resampling, gating, oversampling, pipeline       |
| [`common`](src/common/) | Diagnostics, atomic bitmasks, lock-free SPSC queues          |
| `testing`               | Off-RT test utilities & perceptual metrics (feature-gated)   |

Full API documentation:

* **Local Generation:** `cargo doc --open`
* **Docs.rs Environment Simulation:** `DOCS_RS=1 cargo doc --features "stereo,testing" --no-deps`
* **Online Documentation:** [docs.rs/NeuralAmpModeler-rs](https://docs.rs/NeuralAmpModeler-rs)

> **Note on `docs.rs` Builds:** `NeuralAmpModeler-rs/build.rs` detects `DOCS_RS=1` to early-return before `avx2+fma` CPU target feature assertions. This allows `docs.rs` builders (running on baseline x86-64 without AVX2) to document the API successfully. For new `crates.io` releases or manual doc rebuild requests, use the `docs.rs` re-trigger queue at `https://docs.rs/crate/NeuralAmpModeler-rs/latest/builds`.

---

## 🏆 Quality & Performance

* **Numerical Fidelity:** The quality contract (`docs/quality-contract.json`) tracks NAMCore parity, independent f64-oracle error, SNR (110–144 dB), and MR-STFT (< 1e-5) across 584 lines of per-model baseline envelopes.
* **Measured CPU Headroom (AMD Ryzen 7 5700U, AVX2 @ 64 samples / 48 kHz):**
  * **WaveNet Standard CH16:** **36.9 µs** (**2.8%** of the 1.33 ms deadline)
  * **WaveNet Feather CH8:** **19.4 µs** (**1.5%**)
  * **WaveNet Lite CH12:** **52.6 µs** (**3.9%**)
  * **WaveNet Nano CH4:** **17.4 µs** (**1.3%**)
  * **WaveNet A2 Full CH8:** **27.6 µs** (**2.1%**)
  * **WaveNet A2 Lite CH3:** **18.4 µs** (**1.4%**)
  * **LSTM 1×16:** **7.5 µs** (**0.6%**)
  * **LSTM 2×8:** **7.6 µs** (**0.6%**)
  * **ConvNet:** **10.2 µs** (**0.8%**)
  * **Linear RF=2048:** **0.3 µs** (**0.02%**)
  * **Full DSP Pipeline Base (No OS):** **37.2 µs** (**2.8%**)
  * **Full DSP Pipeline HQ (4× OS):** **150.6 µs** (**11.3%**)
  * **DSP Resampler (44.1k→48k):** **1.3 µs** | **CabSim IR Medium (512):** **1.3 µs**
* **Stress Coverage:** Soak, concurrency, heap-audit, deadline, and model-checking suites exercise long-running and real-time invariants; skipped coverage and failed audit phases must be reviewed separately from passing checks.
* **SIMD Acceleration:** AVX2 baseline (`x86-64-v3`), AVX-512 multiversioning with BF16/VNNI on supported hardware (Intel Sapphire Rapids+, AMD Zen 4+). FastMath activations (tanh, sigmoid) via Padé/minimax polynomial approximations, with exact-grade `Standard` mode as default.

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

| Path                                  | Role                                                             |
|:------------------------------------- |:---------------------------------------------------------------- |
| `third-party/NeuralAmpModelerCore/`   | Pinned C++ NAMCore mirror (render / parity)                      |
| `third-party/NeuralAmpModelerPlugin/` | Pinned C++ plugin mirror (IR / cabsim xref)                      |
| `third-party/community_models/`       | Optional symlink to local community models (not redistributable) |
| `variables.env`                       | Version-controlled pin file (tags/commits/URLs)                  |

Tests and scripts **skip gracefully with a declared gap** when these artifacts are missing:
the quick suite records `GAP:` lines in `target/logs/quick-receipt.txt`, prints
`FIDELITY: INCOMPLETE` / `OVERALL: PASSED_WITH_GAPS` (exit 0), and **never emits a green
fidelity seal for skipped oracles** — `NAM_QUICK_STRICT=1` promotes those gaps to FAIL
(exit 1). The long suite requires the NAMcore mirror outright (hard abort if absent).
Override locations with `NAM_THIRD_PARTY_DIR`, `NAM_CORE_DIR`, `NAM_PLUGIN_DIR`, or
`NAM_MODELS_DIR` if needed.
See [`docs/fixtures.md`](docs/fixtures.md) for the full model search order.

Rust dependency supply-chain updates remain separate: `./utils/mod-update.sh`.

---

## 🧪 CI & QA Automation Suite (`./utils/`)

The `./utils/` directory contains maintainer tools and standard scripts for code quality, numerical verification, and continuous integration:

| Script                                                     | Purpose & Execution Scope                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|:---------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`utils/setup-third-party.sh`](utils/setup-third-party.sh) | **Local env bootstrap:** Clones/syncs pinned NAMCore + Plugin mirrors into `third-party/` and optionally links `community_models`. Required for full parity/golden work; not needed by crate consumers.                                                                                                                                                                                                                                                                                       |
| [`utils/mod-update.sh`](utils/mod-update.sh)               | **Rust supply chain:** Updates rustup toolchain, `cargo upgrade`, and `Cargo.lock` (does **not** manage vendor mirrors).                                                                                                                                                                                                                                                                                                                                                                      |
| [`utils/lints.sh`](utils/lints.sh)                         | **Static Analysis Gate:** Runs `cargo fmt`, strict `cargo clippy`, compilation checks (`cargo check`), zero-warning doc-tests, and verifies SPDX license headers across all repository source files.                                                                                                                                                                                                                                                                                          |
| [`utils/tests-quick.sh`](utils/tests-quick.sh)             | **Agile 1st Line QA:** 3 phases — structural tests (debug), measurement oracles + C++ parity `quick_parity` (release), capped parser fuzzing (`NAM_QUICK_PROPTEST_CASES`). Oracle skips are fail-closed: missing fixtures/toolchain print `FIDELITY: INCOMPLETE` + `OVERALL: PASSED_WITH_GAPS` (exit 0) and write the receipt `target/logs/quick-receipt.txt`; `NAM_QUICK_STRICT=1` promotes gaps to FAIL (exit 1). Re-executes itself at low CPU/IO priority unless `NAM_NO_LOW_PRIORITY=1`. |
| [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh) | **Regression & Quality Gate:** Executes Criterion benchmarks and verifies audio fidelity against `docs/quality-contract.json`.                                                                                                                                                                                                                                                                                                                                                                |
| [`utils/check-model.sh`](utils/check-model.sh)             | **Model Inspector Wrapper:** Canonical tool backed by `examples/inspect_model.rs`. Inspects `.nam` & `.namb` files, outputting detailed human-readable reports, JSON (`--json`), or batch arrays (`--manifest`).                                                                                                                                                                                                                                                                              |
| [`utils/tests-long.sh`](utils/tests-long.sh)               | **Nightly / Pre-Release Suite:** Rust-gated pre-flight (`catalog_preflight` V1/V2 golden catalogs fail-closed + `check_freshness` manifest; no bash golden lists), soak, full proptest/fuzz, full C++ parity matrix, cross-ISA, RT-safety and heap-audits. Exits `OVERALL: FAILED` (1) / `COMPLETED_WITH_GAPS` (0) / `PASSED` (0); `--strict-pre-release` turns declared gaps into failure. *(AI agents must not run this script directly due to runtime length; ask the human operator.)*    |

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
#    docs/functional-tests.md
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
| [`docs/functional-tests.md`](docs/functional-tests.md)                                         | Engine functional test matrix, runner execution protocols, and human certification record    |
| [`docs/postmortem-libm-symbol-interposition.md`](docs/postmortem-libm-symbol-interposition.md) | Technical postmortem on libm symbol interposition resolution on Linux dynamic linkers        |
| [`docs/quality-contract.json`](docs/quality-contract.json)                                     | Quality contract: benchmark and audio fidelity regression baseline thresholds (JSON)         |
| [`docs/fixtures.md`](docs/fixtures.md)                                                         | Golden vector formats, stress signal generation, and non-distributable test model fixtures   |

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

The system architecture, core DSP engineering decisions, mathematical verification framework, and project orchestration are intellectual work (and love) of the maintainer (**Fábio Lima**). The implementation was accelerated through pair programming (*Vibe Coding*) using artificial intelligence models (Gemini, Claude, Grok, DeepSeek and others) within Google Antigravity IDE and Kilo Code. IA is just a tool that make wonder in wise hands.

### License

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for details.
