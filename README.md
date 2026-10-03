<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# NeuralAmpModeler-rs

![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg) ![Rust](https://img.shields.io/badge/Rust-orange.svg) ![Platform](https://img.shields.io/badge/x86__64-lightgrey.svg) [![Crates.io](https://img.shields.io/crates/v/NeuralAmpModeler-rs.svg)](https://crates.io/crates/NeuralAmpModeler-rs) [![docs.rs](https://docs.rs/NeuralAmpModeler-rs/badge.svg)](https://docs.rs/crate/NeuralAmpModeler-rs) ![RT-Safe](https://img.shields.io/badge/RT--Safe-Zero--Alloc-brightgreen.svg) ![SIMD](https://img.shields.io/badge/SIMD-AVX2%20x86--64--v3-blueviolet.svg) ![Models](https://img.shields.io/badge/Models-WaveNet%20A1%20A2%20%7C%20LSTM%20%7C%20ConvNet-success.svg) ![MSRV](https://img.shields.io/badge/MSRV-1.99.0-informational?logo=rust)

**NeuralAmpModeler-rs** is a very high-performance and low-latency real-time neural inference DSP engine written in pure Rust. It provides a production-grade DSP library for loading, building, and executing [Neural Amp Modeler (NAM)](https://www.neuralampmodeler.com/) models — WaveNet (A1/A2), LSTM, ConvNet, and Linear FIR/FFT — alongside speaker cabinet impulse response (.wav) convolution, multi-rate sinc resampling, and polyphase anti-aliasing oversampling.

Engineered for seamless embedding into audio plugins (CLAP, VST3, AU), standalone real-time hosts, offline renderers, and embedded audio pipelines, it guarantees **strict zero heap allocations**, **zero mutex locks**, and **zero blocking system calls** on the real-time audio thread.

The crate is host-agnostic, standalone, and general-purpose: public APIs remain decoupled from specific plugin wrappers or audio servers, allowing downstream applications to integrate the engine cleanly into any audio processing graph.

---

## ⚡ Key Architectural Highlights

* **Pure Rust & Strict Zero-Allocation Real-Time Safety:** Engineered for deterministic audio callbacks under sub-millisecond deadlines. Guarantees zero heap drops, zero locks, and zero blocking syscalls on the audio thread (audited via `CountingAllocator`). Parameter changes and model hot-swaps communicate via lock-free SPSC channels, backed by a 3-tier garbage-collection cascade (*SPSC queue → 16-slot parking lot → overwrite ring*) ensuring safe off-RT resource deallocation without glitches or xruns.
* **Extremely Fast SIMD Inference & Zero-Vtable Dispatch:** Enforces `x86-64-v3` (AVX2/FMA/BMI2) baseline vectorization. Wide use of hand written SIMD code. The `dispatch_simd!` engine performs static compile-time monomorphization with zero vtables or indirect function pointer calls. Compute kernels employ tap-major memory layouts and instruction-level parallelism (`sum0..sum3`) unrolling. AVX2 represents the optimal production sweet spot for NAM channel geometries ($C \le 16$); an experimental AVX-512 backend is available for research and cross-ISA validation via the opt-in `avx512` feature.
* **Dual-Oracle Numerical Parity:** Formally validated against two co-equal reference oracles: canonical C++ NAMCore (ensuring 100% behavioral compatibility with existing market models) and double-precision f64 reference (measuring numerical ideality). Governed by `docs/quality-contract.json` (51 model baselines), typical WaveNet models achieve ESR $< 2.31 \times 10^{-14}$ (SNR $> 136\text{ dB}$, MR-STFT $< 6.5 \times 10^{-6}$), while ConvNet achieves ESR $< 9.33 \times 10^{-16}$ (SNR $> 150\text{ dB}$).
* **Const-Generic Optimization & Dynamic Topologies:** Provides 23 distinct model variants (16 static const-generic profiles + zero-allocation dynamic fallbacks). Channel counts ($CH=16, 12, 8, 4, 3$), kernel sizes, and receptive fields are known at compile time for canonical configurations, enabling aggressive LLVM loop unrolling and register allocation. Non-standard topologies automatically route to dynamic zero-alloc fallbacks (`WaveNetModelDyn`, `LstmModelDyn`, `WaveNetA2Dyn`).
* **Complete Native DSP Stack (Zero External Audio Crates):** Includes a native Minimum-Phase Polyphase FIR Sinc Resampler (256 phases × 64 taps, Kaiser $\beta=12$, $>105\text{ dB}$ stopband attenuation, zero pre-ringing, $0.7\text{–}1.3\text{ µs}$ execution), Uniform-Partitioned Overlap-Save (UPOLS) FFT cabinet IR convolution ($1.3\text{ µs}$ for 512-sample IRs), and multi-stage Half-Band FIR Anti-Aliasing Oversampling (2×/4×, $>100\text{ dB}$ stopband).
* **Adaptive Compute FSM & Pre-Transposed `.namb` v2 Container:** Dynamic CPU load-monitoring state machine with hysteresis that degrades model complexity under load (Full → Reduced → Minimal) using 32 ms click-free linear crossfades to prevent buffer dropouts. The binary `.namb` v2 container (Gate-Major LSTM, Interleaved-4 WaveNet) reduces model load time from $\sim 50\text{ ms}$ to $< 1\text{ ms}$ with mandatory IEEE 802.3 CRC32 integrity checks.
* **Denormal & Subnormal Armor:** Injects symmetric $-220\text{ dBFS}$ deterministic dither ($1.0 \times 10^{-11}$) combined with hardware MXCSR FTZ/DAZ reassertion on every processing block, preventing 10–100× CPU microcode penalties during digital silence with zero net DC drift.

---

## 🥊 Feature Showcase

| Feature / Attribute              | Technical Implementation                                                            | Benefit & Impact                                                                                             |
|:-------------------------------- |:----------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------ |
| **Inference Topologies**         | WaveNet (A1/A2), LSTM (1-layer & 2-layer), ConvNet, and Linear FIR/FFT              | Complete NAM ecosystem compatibility with native Rust execution speed                                        |
| **RT Safety Determinism**        | Zero heap drop, zero mutex locks, 3-tier lock-free GC cascade                       | Guaranteed real-time audio stability without dropouts/xruns under sub-millisecond deadlines                  |
| **SIMD Acceleration**            | Mandatory `x86-64-v3` (AVX2/FMA) baseline; static `dispatch_simd!` dispatch         | WaveNet Standard ≈ 43.3 µs / LSTM 1×16 ≈ 6.8 µs per 64-sample block (AMD Ryzen 7 5700U)                      |
| **Const-Generic Profiles**       | 23 model variants (16 static const-generic profiles + dynamic fallbacks)            | Compile-time LLVM loop unrolling and register allocation for canonical channel counts ($CH=16, 12, 8, 4, 3$) |
| **Numerical Parity**             | Dual-oracle validation: canonical C++ NAMCore $f32$ + double-precision $f64$        | Bit/float-exact accuracy matching reference models ($2.31 \times 10^{-14}$ to $9.33 \times 10^{-16}$ ESR)    |
| **Native Polyphase Resampler**   | 256 phases × 64 taps Kaiser sinc resampler (minimum-phase cepstrum, 0 pre-ringing)  | Pristine multi-rate conversion (>105 dB stopband, < 0.05 dB ripple) in 0.7–1.3 µs                            |
| **Cabinet IR Convolution**       | Uniform-Partitioned Overlap-Save (UPOLS) FFT convolution engine (.wav IRs)          | Ultra-low-overhead speaker cabinet simulation (1.3 µs for 512-sample IRs)                                    |
| **Oversampling & Anti-Aliasing** | Half-band polyphase FIR filters (`Off`, `2x`, `4x`, >100 dB stopband)               | Attenuates non-linear high-frequency foldover/aliasing in high-gain amp models                               |
| **Activation Math Modes**        | `Standard` (exact precision Taylor minimax) vs `Fast` (Padé polynomial minimax)     | User-selectable trade-off between floating-point precision (+89.5 dB SNR) and CPU cycles                     |
| **Adaptive Compute Container**   | Multi-profile `.namb` bundle support with runtime fallback switching                | Prevents audio dropouts by dynamically adjusting compute complexity under CPU spikes                         |
| **Binary `.namb` v2 Format**     | Pre-transposed memory layout (Gate-Major LSTM, Interleaved-4 WaveNet) with CRC32    | Reduces model loading / hot-swap time from ~50 ms to < 1 ms                                                  |
| **Denormal Armor**               | Symmetric −220 dBFS dither injection + hardware MXCSR FTZ/DAZ                       | Prevents 10–100× CPU microcode stalls on digital silence with zero DC drift                                  |
| **Comprehensive QA Suite**       | 2,000+ unit/integration tests, heap audit, soak, proptest, and Criterion benchmarks | Enterprise-grade software stability and strict regression protection                                         |

---

## 🧠 Supported Architectures

| Architecture            | Static Profiles                                           | Dynamic Fallback  | Notes                                        |
|:----------------------- |:--------------------------------------------------------- |:----------------- |:-------------------------------------------- |
| **WaveNet A1**          | Standard (CH=16), Lite (12), Feather (8), Nano (4)        | `WaveNetModelDyn` | Dilated causal 1D convs, gated tanh/sigmoid  |
| **WaveNet A2**          | Full (CH=8), Lite (CH=3), Cascade                         | `WaveNetA2Dyn`    | Headroom-optimized modern NAM architecture   |
| **LSTM**                | 10 profiles: 1-layer (hidden 3–40), 2-layer (hidden 8–24) | `LstmModelDyn`    | Pre-transposed gate-major SIMD GEMV kernels  |
| **ConvNet**             | Causal Conv1D + BatchNorm1D + activation                  | —                 | Fast feed-forward models (clean / overdrive) |
| **Linear**              | Direct FIR or Partitioned FFT convolution                 | —                 | Clean tone equalization & linear filters     |
| **Slimmable Container** | Multi-submodel bundles with runtime quality transitions   | —                 | Adaptive compute bundles with crossfading    |

---

## 🛠️ System Prerequisites & Build Requirements

| Dependency           | Minimum Requirement                           | Purpose                               |
|:-------------------- |:--------------------------------------------- |:------------------------------------- |
| **CPU Architecture** | `x86_64` with AVX2/FMA (`x86-64-v3` baseline) | SIMD vectorized DSP kernels           |
| **Rust Toolchain**   | $\ge 1.99.0$ (Edition 2024)                   | Public MSRV promise                   |
| **Build Tools**      | `build-essential`, `pkg-config`, `cmake`      | Host build tools & C++ parity oracles |

> **MSRV Policy:** `rust-version = "1.99.0"` in `Cargo.toml` is the guaranteed public MSRV promise. The project builds and validates on stable Rust.

### Installation of System Build Dependencies (Debian / Ubuntu / Pop!_OS)

```bash
sudo apt update && sudo apt install -y build-essential pkg-config cmake
```

### Mandatory Vector Target: `x86-64-v3`

NeuralAmpModeler-rs requires an `x86_64` processor supporting the **`x86-64-v3`** instruction set baseline (`avx`, `avx2`, `bmi1`, `bmi2`, `f16c`, `fma`, `lzcnt`, `movbe`). Build-time assertions enforce this target to guarantee SIMD performance across all DSP modules.

Because Cargo does not automatically propagate compiler target flags to downstream dependencies, applications depending on `NeuralAmpModeler-rs` must instruct `rustc` to target `x86-64-v3`:

**Option A — In `.cargo/config.toml` (Recommended for downstream crates):**

```toml
[build]
rustflags = ["-Ctarget-cpu=x86-64-v3"]
```

**Option B — Via environment variable:**

```bash
RUSTFLAGS="-Ctarget-cpu=x86-64-v3" cargo build --release
```

#### Pre-Flight Hardware Verification (`simd_probe`)

Verify your host CPU and toolchain compatibility directly using the built-in diagnostic probe:

```bash
cargo run --bin simd_probe
```

The probe validates CPU feature flags, operating system vector context support (`OSXSAVE`), reports the active monomorphized SIMD backend, and executes a synthetic inference smoke test with deterministic checksum verification.

---

## 🚀 Quick Start — Installation & Usage

### Add Dependency

Add `NeuralAmpModeler-rs` to your `Cargo.toml`:

```toml
[dependencies]
NeuralAmpModeler-rs = "0.8"
```

For test harnesses, audio generators, and perceptual fidelity measurement tools:

```toml
[dependencies]
NeuralAmpModeler-rs = { version = "0.8", features = ["testing"] }
```

### Minimal Code Example: Load & Process Audio

```rust,no_run
use std::path::Path;
use neural_amp_modeler_rs::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Capture system hardware capabilities (SIMD features, CPU topology)
    let sys = SystemSnapshot::capture();

    // 2. Load a .nam (JSON) or .namb (binary) model
    let mut model_pair = load_and_build_model(
        Path::new("models/BossWN-standard.nam"),
        &sys,
        false, // dual_mono: false = mono left-channel only
        LoadOptions::default(),
    )?;

    // 3. Process an audio block on the real-time thread (zero allocations, zero locks)
    let input_buffer = [0.0_f32; 64];
    let mut output_buffer = [0.0_f32; 64];

    if let Some(ref mut model) = model_pair.model_l {
        model.process(&input_buffer, &mut output_buffer);
    }

    Ok(())
}
```

### Full DSP Pipeline (Model + Cabinet IR + Oversampling)

For a complete end-to-end signal processing chain — neural amp model, cabinet impulse response (IR) convolution, and 4× polyphase anti-aliasing oversampling — see the [`offline_render`](examples/offline_render.rs) example:

```bash
cargo run --example offline_render -- path/to/model.nam
```

### Executable Examples

The crate includes runnable examples demonstrating key features:

| Example                                            | Description                                                                | Command                                             |
|:-------------------------------------------------- |:-------------------------------------------------------------------------- |:--------------------------------------------------- |
| [`load_model`](examples/load_model.rs)             | Off-RT model file loading (`.nam`/`.namb`) and SIMD state prewarming       | `cargo run --example load_model -- <model.nam>`     |
| [`inspect_model`](examples/inspect_model.rs)       | Inspects metadata, architecture, weights, and sample rates (Text/JSON)     | `cargo run --example inspect_model -- <model.nam>`  |
| [`offline_render`](examples/offline_render.rs)     | Full audio rendering with 4× polyphase oversampling (HQ mode)              | `cargo run --example offline_render -- <model.nam>` |
| [`cabsim`](examples/cabsim.rs)                     | Standalone cabinet impulse response (IR) convolution and resampling        | `cargo run --example cabsim -- <ir.wav>`            |
| [`diagnostics`](examples/diagnostics.rs)           | Diagnostic bundle (`DiagnosticBundle`) and log buffer (`LogBuffer`) export | `cargo run --example diagnostics`                   |
| [`math_activations`](examples/math_activations.rs) | Precision vs speed benchmarks for SIMD activations (`Standard` vs `Fast`)  | `cargo run --example math_activations`              |
| [`synthetic_model`](examples/synthetic_model.rs)   | In-memory `StaticModel` construction without external files                | `cargo run --example synthetic_model`               |

---

### Rustdoc Module Map

| Module                 | Purpose                                                                         |
|:---------------------- |:------------------------------------------------------------------------------- |
| [`loader`](src/loader) | Model deserialization and construction (`.nam`, `.namb`)                        |
| [`math`](src/math)     | SIMD math primitives, activation approximations, GEMV/FFT kernels               |
| [`models`](src/models) | Neural network architectures and `StaticModel` dispatch                         |
| [`dsp`](src/dsp)       | Complete DSP engine: polyphase resampler, noise gate, oversampling, cabsim      |
| [`common`](src/common) | System telemetry, diagnostics, lock-free SPSC channels                          |
| `testing`              | Off-RT testing utilities, perceptual metrics, and test fixtures (feature-gated) |

API Documentation:

* **Online:** [docs.rs/NeuralAmpModeler-rs](https://docs.rs/NeuralAmpModeler-rs)
* **Local:** `cargo doc --open`

---

## 🚩 Feature Flags

NeuralAmpModeler-rs provides modular Cargo feature flags to tailor capabilities, diagnostic tooling, and benchmarking:

| Feature              | Default     | Description                                                                                                                     | Category                |
|:-------------------- |:----------- |:------------------------------------------------------------------------------------------------------------------------------- |:----------------------- |
| `dual-mono`          | **Enabled** | Independent per-channel neural inference across the DSP pipeline for stereo/dual-mono signals. Disable for single-channel mono. | Production              |
| `heap-audit`         | Disabled    | Enables real-time allocation tracking via `CountingAllocator` to enforce zero-heap invariants.                                  | Diagnostics             |
| `testing`            | Disabled    | Exposes off-RT test utilities, audio signal generators, synthetic fixtures, and perceptual fidelity measurement oracles.        | Tooling / QA            |
| `fft-radix4-planner` | Disabled    | Exposes Radix-4 DIT FFT execution planning routines (`FftPlannerRadix4`) and benchmarks.                                        | Tooling / Benchmarks    |
| `rt-hardening`       | Disabled    | Opt-in Linux real-time system hardening (THP disable, `mlockall`, `SCHED_FIFO`, MXCSR DAZ/FTZ, CPU affinity). Linux-only.       | Production (Linux RT)   |
| `avx512`             | Disabled    | Experimental AVX-512 kernels and instruction dispatch for research and cross-ISA validation.                                    | Research / Experimental |

> **Note on `avx512`:** Production builds target the `x86-64-v3` (AVX2 + FMA) baseline. Benchmarks demonstrate that AVX2 delivers lower latency and higher throughput across canonical NAM channel geometries ($C \le 16$). The opt-in `avx512` feature is maintained for research, hardware benchmarking, and architectural exploration. See [`docs/architecture.md`](docs/architecture.md) and [`docs/benchmarks.md`](docs/benchmarks.md).

---

## 🏆 Quality & Performance

### Numerical Parity & Quality Contract

The engine's numerical accuracy is strictly enforced through continuous baseline regression testing (`docs/quality-contract.json`), tracking 51 model baseline envelopes against both the C++ NAMCore reference and a double-precision $f64$ mathematical oracle:

* **Dual-Oracle Parity:** WaveNet Standard models maintain an ESR of $2.31 \times 10^{-14}$ against NAMCore and $9.05 \times 10^{-15}$ against the $f64$ oracle ($136.4\text{ dB}$ SNR). ConvNet ReLU achieves $9.33 \times 10^{-16}$ ESR ($150.3\text{ dB}$ SNR).
* **Perceptual Validation:** All models verify multi-resolution STFT distance (MR-STFT $< 6.5 \times 10^{-6}$), spectral convergence, and LUFS loudness constancy.

### Measured CPU Headroom (Quality Contract SLA Baselines)

Execution times measured per 64-sample block at 48 kHz (1.33 ms real-time deadline budget) on an AMD Ryzen 7 5700U (AVX2 baseline, `rustc 1.98.1`):

| Component / Profile                            | Execution Time | RT Deadline Budget Used |
|:---------------------------------------------- |:-------------- |:----------------------- |
| **WaveNet Standard (CH=16)**                   | **43.3 µs**    | 3.2%                    |
| **WaveNet Lite (CH=12)**                       | **56.4 µs**    | 4.2%                    |
| **WaveNet Feather (CH=8)**                     | **19.9 µs**    | 1.5%                    |
| **WaveNet Nano (CH=4)**                        | **17.8 µs**    | 1.3%                    |
| **WaveNet A2 Full (CH=8)**                     | **25.6 µs**    | 1.9%                    |
| **WaveNet A2 Lite (CH=3)**                     | **22.6 µs**    | 1.7%                    |
| **LSTM 1-Layer (Hidden=16)**                   | **6.8 µs**     | 0.5%                    |
| **LSTM 2-Layer (Hidden=8)**                    | **7.1 µs**     | 0.5%                    |
| **ConvNet**                                    | **8.7 µs**     | 0.7%                    |
| **Linear FIR (RF=4 Direct)**                   | **0.3 µs**     | 0.02%                   |
| **DSP Resampler (44.1 kHz → 48 kHz)**          | **1.2 µs**     | 0.09%                   |
| **Cabinet IR Simulation (UPOLS 512)**          | **1.2 µs**     | 0.09%                   |
| **Full Pipeline Base (Model + CabSim, No OS)** | **43.6 µs**    | 3.3%                    |
| **Full Pipeline HQ (Model + CabSim + 4× OS)**  | **177.4 µs**   | 13.3%                   |

> Detailed benchmarking methodologies, throughput curves, and regression envelopes are documented in [`docs/benchmarks.md`](docs/benchmarks.md).

---

## 🧰 Developer & Maintainer Tooling

Crate consumers only require Cargo and a standard Rust toolchain. Developers contributing to `NeuralAmpModeler-rs` or validating parity against C++ reference implementations should set up the local mirror environment:

```bash
# Clone and configure pinned C++ NAMCore and Plugin mirrors (gitignored):
./utils/setup-third-party.sh

# Optional: Link a local directory of private community test models
NAM_COMMUNITY_MODELS_SRC=/path/to/models ./utils/setup-third-party.sh
```

### Automation Scripts (`./utils/`)

The `./utils/` suite automates formatting, static analysis, parity verification, and quality gates:

| Script                                                     | Purpose & Scope                                                                                                                       |
|:---------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------- |
| [`utils/lints.sh`](utils/lints.sh)                         | **Static Analysis Gate:** Runs `cargo fmt`, strict `clippy`, compilation checks, doc-tests, and SPDX header verification.             |
| [`utils/tests-quick.sh`](utils/tests-quick.sh)             | **Agile QA Suite:** Multi-phase test run covering unit tests, C++ parity quick checks, and parser fuzzing.                            |
| [`utils/quality-dashboard.sh`](utils/quality-dashboard.sh) | **Quality & Regression Gate:** Verifies benchmark timings and audio fidelity against `docs/quality-contract.json`.                    |
| [`utils/check-model.sh`](utils/check-model.sh)             | **Model Inspector CLI:** Inspects `.nam` and `.namb` files, outputting detailed reports, JSON, or manifest arrays.                    |
| [`utils/simd-probe.sh`](utils/simd-probe.sh)               | **SIMD Diagnostic Probe:** Inspects CPU features, OS vector context, and runs deterministic inference smoke tests.                    |
| [`utils/tests-long.sh`](utils/tests-long.sh)               | **Pre-Release Audit:** Exhaustive suite including soak tests, full proptest/fuzzing, cross-ISA, and heap audits (run by maintainers). |
| [`utils/setup-third-party.sh`](utils/setup-third-party.sh) | **Local Environment Bootstrap:** Clones and syncs pinned vendor mirrors into `third-party/`.                                          |
| [`utils/mod-update.sh`](utils/mod-update.sh)               | **Dependency Maintenance:** Updates the Rust toolchain and dependencies via `cargo upgrade`.                                          |

Standard verification workflow:

```bash
# 1. Static analysis & format checks
./utils/lints.sh

# 2. Agile QA validation
./utils/tests-quick.sh
```

---

## 📚 Architecture & Technical Documentation

Comprehensive architectural specifications and engineering guides are available in the [`docs/`](docs/) directory:

| Document                                                             | Topic                                                                          |
|:-------------------------------------------------------------------- |:------------------------------------------------------------------------------ |
| [`docs/architecture.md`](docs/architecture.md)                       | Engine architecture, SIMD kernels, memory layout, and DSP pipelines            |
| [`docs/audio_fidelity_map.md`](docs/audio_fidelity_map.md)           | Audio fidelity decisions, frequency response, and quality modes (Live vs HQ)   |
| [`docs/fastmath-approximations.md`](docs/fastmath-approximations.md) | Activation approximations (Padé/minimax), polynomial error bounds, and SNR     |
| [`docs/namb-spec.md`](docs/namb-spec.md)                             | Binary `.namb` v2 container specification, metadata schema, and memory layout  |
| [`docs/cpp_parity_map.md`](docs/cpp_parity_map.md)                   | Bit/float-exact parity verification against canonical C++ NeuralAmpModelerCore |
| [`docs/perceptual_validation.md`](docs/perceptual_validation.md)     | Perceptual audio metrics: ESR, MR-STFT, Spectral Convergence, and LUFS         |
| [`docs/benchmarks.md`](docs/benchmarks.md)                           | Criterion benchmarks, throughput profiles, and performance regression gates    |
| [`docs/testing.md`](docs/testing.md)                                 | Test suite organization, oracle hierarchy, and verification policies           |
| [`docs/fixtures.md`](docs/fixtures.md)                               | Golden vector formats, stress signal generation, and fixture discovery         |
| [`docs/functional-tests.md`](docs/functional-tests.md)               | Functional test matrix, execution protocols, and certification checklists      |
| [`docs/research-references.md`](docs/research-references.md)         | Scientific literature, DSP bibliography, and neural audio modeling research    |
| [`docs/quality-contract.json`](docs/quality-contract.json)           | Regression baseline thresholds: audio fidelity and benchmark SLA envelopes     |

---

## 🤝 Contributing

* **Testing, testing, testing!** Go on, "cargo add NeuralAmpModeler-rs" into your own project and make NeuralAmpModeler-rs live and evolve!
* **Feedback & Issues:** Submit detailed bug reports, questions, or feature requests via GitHub Issues.
* **Model Compatibility:** Test your favorite `.nam` models and impulse responses, and share your benchmarks or findings.
* **Code Contributions:** Help me to make NeuralAmpModeler-rs more useful and correct for the community!

---

## 🙏 Credits & Acknowledgments

* **Steven Atkinson** — Creator of [Neural Amp Modeler (NAM)](https://github.com/sdatkinson/neural-amp-modeler) for pioneering deep learning amplifier modeling and generously open-sourcing the ecosystem.
* **Mike Oliphant** — Author of [NeuralAudio](https://github.com/mikeoliphant/NeuralAudio), whose work provided early insights into WaveNet inference optimization.

---

## ⚖️ License & AI Transparency

### AI Transparency Note

The system architecture, DSP design decisions, mathematical verification frameworks, and project orchestration represent the creative and technical direction of the author (**Fábio Henrique de Lima Silva**). Implementation and iterative development were accelerated through human-directed AI pair programming.

### License

This project is licensed under the **Apache License, Version 2.0**. See [LICENSE.txt](LICENSE.txt) for details.
