<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# C++ ↔ Rust Parity Audit — NeuralAmpModelerCore × NeuralAmpModeler-rs

This document establishes the architectural, structural, and numerical parity mapping between the canonical C++ reference engine, **NeuralAmpModelerCore** ("NAMcore" vendored read-only at `third-party/NeuralAmpModelerCore/`), and the **NeuralAmpModeler-rs** Rust engine (`src/`).

Numerical correctness is evaluated along two complementary, co-equal axes:

- **Market Interoperability (NAMcore Oracle):** Guarantees that NeuralAmpModeler-rs produces audio identical (within floating-point tolerance) to existing models in the ecosystem, verified through committed golden vectors and live cross-validation.
- **Mathematical Ideality (f64 Oracle):** Measures deviation against exact mathematical formulas via an independent double-precision reference (`src/testing/reference_oracle/mod.rs`), cross-checked against NumPy f64 anchors.

Neither oracle has automatic prevalence over the other; disagreements trigger a `REVIEW_REQUIRED` governance event ([§1.2](#12-two-oracle-governance-policy)).

For an operational triage of active boundaries, known bugs, and accepted tradeoffs, see [§7 Known-Broken & Policy Ledger](#7-known-broken--policy-ledger).

---

## 0. Scope and Declared Exclusions

### 0.1 Architectural Scope & C++ Mirror Exclusions

The audit tracks the canonical C++ reference at tag **`v0.6.0`** (commit `0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842`, pinned in [`variables.env`](../variables.env)).

1. **Headers Audited via Observable Behavior:**

   - `NAM/film.h` (`class FiLM`): Feature-wise Linear Modulation. Conditioned scale and shift operations are exercised and verified through WaveNet A2 dynamic and fast paths ([§4.2](#42-measured-interop-metrics-fast--dynamic-paths)).
   - `NAM/gating_activations.h` (`class GatingActivation`, `BlendingActivation`, `IdentityActivation`): Gating and blend activations are audited via WaveNet A1/A2 activation tests and dynamic dispatch ([§3.4](#34-fail-closed-rejection-of-a2-features-in-a1), [§4.2](#42-measured-interop-metrics-fast--dynamic-paths)).
   - `NAM/ring_buffer.h` (`class RingBuffer`): Circular history buffer employed inside `Conv1D`. FIFO delay line semantics are verified directly in ConvNet ([§6](#6-other-architectures)) and WaveNet ([§3](#3-wavenet-a1-architecture)) convolution kernels.

2. **Plumbing and Host Infrastructure Out of Scope (Non-DSP):**

   - `NAM/get_dsp.h` / `get_dsp.cpp`: Model file loading and C++ polymorphic instantiation.
   - `NAM/model_config.h`: JSON parsing structs.
   - `NAM/registry.h`: C++ dynamic dispatch registry.
   - `NAM/util.h` / `util.cpp`: Host utility and string formatting routines.
   - `NAM/compiler.h`: C++ compiler macros (`NAM_FORCE_INLINE`).
   - `NAM/version.h`: C++ version declaration header (intentionally bypassed in favor of [`variables.env`](../variables.env) commit pinning).
     *Rationale:* The Rust engine implements its own strongly typed deserializer and static enum dispatcher (`src/loader/`, `src/models/mod.rs`), verified via independent test suites.

3. **External DSP Modules Out of Scope:**

   - `Dependencies/AudioDSPTools/dsp/*` and `ResamplingContainer/*`: Auxiliary tools in upstream repositories not included by `NAM/` core headers and not part of neural inference.

### 0.2 Numerical Scope & ISA Independence

This audit targets **numerical and architectural semantics**. NeuralAmpModeler-rs implements multiple SIMD dispatch backends (`scalar`, `sse4.2`, `avx2`, and opt-in `avx512`). All vector kernels are required to produce bit-exact or sub-ULP equivalent results to the scalar reference. ISA parity verification is enforced by dedicated cross-ISA test suites (`tests/models/isa_parity.rs`).

### 0.3 Architecture Audit Status

| Architecture         | Status                                                                                                                    | Reference Section                |
|:-------------------- |:------------------------------------------------------------------------------------------------------------------------- |:-------------------------------- |
| **LSTM**             | ✅ Fully Verified — Native f32 weights, bit-exact / sub-1e-11 interop parity vs NAMcore                                   | [§2](#2-lstm-architecture)       |
| **WaveNet A1**       | ✅ Fully Verified — Const-generic fast path & dynamic fallback pass golden gates; fail-closed A2 guards                   | [§3](#3-wavenet-a1-architecture) |
| **WaveNet A2**       | 🟡 Verified Dynamic/Fast paths — 🔴 Legacy fixture `wavenet_a2_max.nam` (reclassified upstream as feature test in v0.6.0) | [§4](#4-wavenet-a2-architecture) |
| **ConvNet**          | ✅ Identical — Full initialization and arithmetic parity (silence prewarm matches NAMcore)                                | [§6](#6-other-architectures)     |
| **Linear / CabSim**  | ✅ Verified — Affine linear models, `SlimmableContainer`, and FIR CabSim cross-validated (v0.6.0 multi-channel tracked)   | [§6](#6-other-architectures)     |
| **SlimmableWavenet** | 🟡 Verified — Channel-sliceable dynamic inference operational; no multi-size NAMcore parity claimed                       | [§6](#6-other-architectures)     |
| **Sequential**       | ✅ Fully Verified — Whole-chain live cross-validation vs C++ render (mono-interior/exterior composites, 44.1–192 kHz sweeps), f64 composition oracle, single-prewarm transient identity + detection-power control (NC-3.4) | [§6](#66-sequential-namcore-v060) |

---

## 1. Methodology & Dual-Oracle Governance

### 1.1 Correctness Axes

1. **Market Interoperability (NAMcore Oracle):** Assesses whether NeuralAmpModeler-rs reproduces the acoustic output of the C++ reference within floating-point tolerance across real community captures. Validated via committed golden fixtures (`tests/fixtures/*.bin`) and live C++ rendering (`tests/parity/cpp_parity.rs`).
2. **Mathematical Fidelity (f64 Oracle):** Assesses distance from pure mathematical evaluation using double precision (`f64`). Isolates precision losses resulting from `f32` accumulation, polynomial approximations, or algorithmic deviations.

### 1.2 Two-Oracle Governance Policy

NAMcore and the f64 oracle operate with **co-equal authority**:

| Oracle      | Question Answered                                                                     | Authority                          |
|:----------- |:------------------------------------------------------------------------------------- |:---------------------------------- |
| **NAMcore** | Does the engine match existing ecosystem captures? (Market interop)                   | Sole arbiter of interop parity     |
| **f64**     | What is the exact mathematical target, and how far is the engine from it? (Precision) | Sole arbiter of mathematical ideal |

**Disagreement Protocol:**
When NAMcore reports acceptable parity but the f64 oracle diverges significantly (or vice versa):

1. The divergence must be documented with measurements from both oracles.
2. A human reviewer must triage the root cause (C++ approximation, Rust structural difference, or oracle bug).
3. The catalog entry and test gates are marked `REVIEW_REQUIRED`.
4. Production code changes justified solely by improving one oracle while regressing the other are prohibited.

### 1.3 Reference Version Pinning

- **Pinned Reference:** Tag `v0.6.0` / commit `0b3d3c97b0859a3a8c92a8628c4dd89a25eb5842`.
- **Mirror Location:** `third-party/NeuralAmpModelerCore/` (populated via `utils/setup-third-party.sh`).
- **Build Entry Point:** `utils/ensure_namcore_render.sh` builds the release C++ `render` binary (`build/namcore_render/tools/render`).

### 1.4 Verification Layers

The fixture supply chain and test gates follow a 3-layer architecture detailed in [`docs/fixtures.md`](fixtures.md) and [`docs/testing.md`](testing.md):

- **Layer 0 (Generation):** `tests/fixtures/golden_gen_build.sh` queries the Rust catalog registry (`src/testing/catalog.rs::GOLDEN_GEN_CATALOG`) and executes the C++ `render` tool to emit `.bin` reference files.
- **Layer 1 (Committed Goldens):** `tests/models/golden_vectors.rs` validates the Rust engine against pre-committed `.bin` fixtures without requiring a C++ toolchain.
- **Layer 2 (Live Cross-Validation):** `tests/parity/cpp_parity.rs` builds and executes the C++ `render` tool on the fly, performing live comparison across multiple sample rates.
- **Integrity Gates:** `tests/fixtures/.golden_manifest.sha256` ensures fixture freshness. Thresholds are defended against regressions by meta-tests in `tests/models/threshold_calibration.rs`.

---

## 2. LSTM Architecture

Audited against `NAM/lstm.h`, `NAM/lstm.cpp`, and corresponding Rust modules:

- Models: [`src/models/lstm/`](../src/models/lstm/)
- Gate Kernels: [`src/math/lstm/gates.rs`](../src/math/lstm/gates.rs)
- Loader & Transpose: [`src/loader/dispatcher/lstm/`](../src/loader/dispatcher/lstm/), [`src/loader/transpose/lstm.rs`](../src/loader/transpose/lstm.rs)

### 2.0 Supported Sample Rates

| Model                         | Golden Vectors (v1) | Golden Vectors (v2)        | Live C++ Parity (v2)       |
|:----------------------------- |:-------------------:|:--------------------------:|:--------------------------:|
| BossLSTM-1×16                 | 48 kHz              | 44100, 48000, 88200, 96000 | 44100, 48000, 88200, 96000 |
| BossLSTM-2×8                  | 48 kHz              | 44100, 48000, 88200, 96000 | 44100, 48000, 88200, 96000 |
| LSTM Official (1×3)           | 48 kHz              | 48000                      | 48000                      |
| LSTM Synthetic (1/2/3 layers) | 48 kHz              | 48000                      | 44100, 48000, 88200, 96000 |

*Note: 192 kHz is excluded across all LSTM evaluations due to an upstream C++ overflow limitation ([§2.7](#27-192-khz-upstream-limitation)).*

### 2.1 Reference Algorithm (`NAM/lstm.cpp`)

A stack of $N$ recurrent cells, processing one sample at a time, followed by an affine head:

$$
\begin{aligned}
[i, f, g, o]^T &= W_l \cdot [x_t; h_{t-1}] + b_l \\
c_t &= \sigma(f) \odot c_{t-1} + \sigma(i) \odot \tanh(g) \\
h_t &= \sigma(o) \odot \tanh(c_t) \\
y_t &= W_{\text{head}} \cdot h_{t, \text{last}} + b_{\text{head}}
\end{aligned}
$$

Gate ordering in memory is fixed to **I, F, G, O** at strides `0, H, 2H, 3H` (`lstm.cpp:40-44`). By default (`Activation::using_fast_tanh = false`), C++ evaluates exact libm `expf` for $\sigma$ and $\tanh$.

### 2.2 Structural Implementation Mapping

| C++ Reference (`NeuralAmpModelerCore/`)                            | Rust Implementation (`src/`)                                           | Verdict                                                     |
|:------------------------------------------------------------------ |:---------------------------------------------------------------------- |:----------------------------------------------------------- |
| `LSTMCell::process_` gate math (`lstm.cpp:31-66`)                  | `math/lstm/gates.rs` + `models/lstm/layer_kernels.rs`                  | ✅ Exact match in gate ordering and arithmetic formulations |
| Row-major gate weights `[4H × (I+H)]` (`lstm.cpp:19-21`)           | `LstmLayer::input_hidden_weights` loaded via `read_lstm_weights_into`  | ✅ Byte-for-byte layout correspondence                      |
| Bias `[4H]`, init hidden `[H]`, init cell `[H]` (`lstm.cpp:22-28`) | `read_lstm_layer` sequentially parses bias, hidden-init, and cell-init | ✅ Exact sequential match                                   |
| 2-layer chain (`lstm.cpp:151-153`)                                 | `LstmModel2` software-pipelined chain (`model2.rs`)                    | ✅ Bit-exact sequential stacking, optimized for ILP         |
| Linear head projection (`lstm.cpp:161-164`)                        | `dot_product(..) + head_bias` with Kahan summation compensation        | ✅ Exact match (reduces floating-point accumulation drift)  |
| Prewarm length: `0.5 × sample_rate` (`lstm.cpp:125-132`)           | `prewarm_samples()` computes identical `(0.5 * sr) as usize`           | ✅ Exact formula match                                      |
| Reset prewarm opt-out (`dsp.cpp:130-139`)                          | `NamModel::reset()` respects `prewarm_on_reset()` (default `true`)     | ✅ Exact behavioral match                                   |

### 2.3 Weight Storage & Layout

Weights are laid out sequentially: `[W_ih, W_hh] -> bias -> hidden_init -> cell_init` per layer, followed by `head_weights -> head_bias`. `src/loader/dispatcher/lstm/weights.rs` and `src/loader/transpose/lstm.rs` parse this stream directly into memory. `WeightCursor::verify_exhausted()` enforces fail-closed validation against size mismatches.

### 2.4 Catalog Dispatch

[`src/loader/nam_json/topology/lstm.rs`](../src/loader/nam_json/topology/lstm.rs) routes valid topologies:

- `(1, 3)` → `LstmModel1<3, 4, 12>` (official bundled example)
- `(1, H)` where $H \in \{8, 12, 16, 24, 40\}$ → `LstmModel1<H, H+1, 4H>`
- `(2, H)` where $H \in \{8, 12, 16, 24\}$ → `LstmModel2<H, H+1, 2H+H, 4H>`
- Any other valid geometry → `LstmModelDyn` (heap-allocated dynamic layer chain)

Topologies declaring `num_layers == 0`, `num_layers > 16`, or `hidden_size > 1024` are rejected at detection with `Err(JsonError::UnsupportedTopology)`.

### 2.5 Precision & Measured Parity

Backbone weights are stored in native `f32` vectors (`AlignedVec<f32>`), matching C++ `Eigen::MatrixXf`. This avoids dequantization latency and guarantees high numerical convergence.

In Standard mode (universal default), activation functions execute exact polynomial exp-based kernels (error $\sim 2 \times 10^{-7}$). In Fast mode, Padé [5,4] rational $\tanh$ and minimax degree-17 $\sigma$ are used.

**Measured Interop Results (Standard Mode @ 48 kHz):**

| Model         | ESR vs. NAMcore | ESR vs. f64 Ideal | SNR (dB) | MR-STFT  | Parity Verdict             |
|:------------- |:---------------:|:-----------------:|:--------:|:--------:|:-------------------------- |
| BossLSTM-1×16 | 8.50e-12        | 8.90e-13          | 110.7    | 2.80e-05 | ✅ Near-bit-exact          |
| BossLSTM-2×8  | 1.00e-11        | 5.68e-13          | 110.0    | 1.57e-05 | ✅ Bit-exact / noise floor |
| LSTM Official | 7.86e-13        | 2.71e-12          | 121.0    | 3.08e-05 | ✅ Near-bit-exact          |

### 2.6 Scope Divergence: Mono Only

While C++ declares general multi-channel parameters, NeuralAmpModeler-rs models guitar and bass gear strictly as mono. Topologies declaring `in_channels` or `out_channels` $\notin \{1, \text{None}\}$ return `Err(JsonError::UnsupportedMultiChannel)`.

### 2.7 192 kHz Upstream Limitation

**Status:** Fundamental upstream C++ limitation.
The C++ NAMcore `render` tool produces `NaN` or crashes when processing LSTM models over continuous sequences at 192 kHz (e.g., 960,000 samples). The uncompensated recurrent state in Eigen accumulates exponential growth in `f32`.

- 192 kHz is formally excluded from LSTM golden generation and live cross-validation (`V2GenScope::Exclude192k` in `src/testing/catalog.rs`).
- All catalog LSTM captures are verified at `[44100, 48000, 88200, 96000]`.

---

## 3. WaveNet A1 Architecture

Audited against `NAM/wavenet/model.h`, `NAM/wavenet/model.cpp`, and corresponding Rust modules:

- Models: [`src/models/wavenet/`](../src/models/wavenet/)
- Dispatchers: [`src/loader/dispatcher/wavenet/`](../src/loader/dispatcher/wavenet/)
- Topology: [`src/loader/nam_json/topology/wavenet.rs`](../src/loader/nam_json/topology/wavenet.rs)

### 3.1 Architecture Overview: Const-Generic Monomorphization

In C++, `nam::wavenet::WaveNet` is a single generic Eigen-based implementation (`detail::LayerArray` / `detail::Layer`).
NeuralAmpModeler-rs introduces const-generic monomorphization for the four predominant channel geometries:

- **Standard:** 16 channels, dilations `[1, 2, ..., 512]` $\times 2$ arrays (`WaveNetModel<16, 3, 8>`)
- **Lite:** 12 channels, dilations `[1, ..., 64]` + `[128, ..., 512, 1, ..., 512]` (`WaveNetModel<12, 3, 6>`)
- **Feather:** 8 channels, Lite-shaped dilations (`WaveNetModel<8, 3, 4>`)
- **Nano:** 4 channels, Lite-shaped dilations (`WaveNetModel<4, 3, 2>`)
- **Dynamic Fallback:** Non-standard geometries route to `WaveNetModelDyn`.

### 3.2 Structural Implementation Mapping

| C++ Reference (`NAM/wavenet/model.cpp`)                                  | Rust Implementation (`src/models/wavenet/`)                            | Verdict                                                       |
|:------------------------------------------------------------------------ |:---------------------------------------------------------------------- |:------------------------------------------------------------- |
| `detail::LayerArray::ProcessInner` cascade (`model.cpp:450-511`)         | `WaveNetLayerArray::process_block_internal` (`layer_array.rs`)         | ✅ Rechannel → layer stack → head accumulate → head rechannel |
| `detail::LayerArray` head rechannel `Conv1D` (`model.cpp:399-404, 955-990`) | `HeadRechannelDyn` (`layer_array_dyn.rs`)                             | ✅ Bit-identical Dense ($k=1, dil=1$) or dilated Conv1d ($k>1 \lor dil>1$) |
| `detail::Layer::Process` conv + mixin + activation (`model.cpp:166-376`) | `WaveNetLayer::process_block_internal` (`layer.rs`)                    | ✅ Exact arithmetic match for ungated feedforward processing  |
| `detail::Layer` `layer1x1_post_film` execution (`model.cpp:285-291, 1228-1232`)| `WaveNetA2Dyn::process_frame`, `conv1d_ch{3,8}` simd                 | ✅ Unconditionally applied outside gating mode; fails closed if `!layer1x1` |
| `WaveNet::process` conditioning + stack flow (`model.cpp:744-832`)       | `WaveNetModel::process` / `WaveNetModelDyn::process`                   | ✅ Matching cascade flow                                      |
| Prewarm receptive field calculation (`model.cpp:436-442, 615-620`)       | `FreeWavenetGeometry::receptive_field()`, dynamic prewarm backfill    | ✅ Exact match with C++ receptive field summation ($\Sigma dil \cdot (k-1) + dil_{head} \cdot (k_{head}-1)$) |

### 3.3 Analytical Prewarm Shortcut

C++'s `DSP::prewarm()` iteratively processes zeros for the duration of the receptive field: $\mathcal{O}(\text{receptive\_field})$ passes.
Because an ungated causal feedforward stack fed by constant zero input converges to a deterministic fixed point, NeuralAmpModeler-rs calculates this fixed point **analytically in $\mathcal{O}(\text{layers})$**:

1. Evaluates a single zero-input frame through the rechannel layer.
2. For each layer in order, replicates the converged output across the entire history buffer (`copy_within`, `layer_array.rs`) before advancing.
3. Propagates the converged scalar to subsequent layers.

This shortcut produces mathematical equivalence to iterative settling while eliminating startup latency.

### 3.4 Fail-Closed Rejection of A2 Features in A1

The A1 path rejects configurations declaring A2 features at topology parsing time (`src/loader/nam_json/topology/wavenet.rs`):

- `gated: true` or `gating_mode` $\in \{\text{"gated"}, \text{"blended"}\}$
- `head1x1` active
- `layer1x1` active
- Any FiLM parameter definition
- Non-trivial `secondary_activation`

Topologies containing any of the above return `WavenetTopologyResult::Rejected("A2 feature not supported in WaveNet A1")`.

### 3.5 Measured Interop Metrics

Measured against NAMcore baseline at 48 kHz:

| Model                   | ESR vs. NAMcore | ESR vs. f64 Ideal | SNR (dB) | MR-STFT  | Parity Verdict     |
|:----------------------- |:---------------:|:-----------------:|:--------:|:--------:|:------------------ |
| WaveNet Standard (CH16) | 2.31e-14        | 9.05e-15          | 136.4    | 6.46e-06 | ✅ Bit-exact floor |
| WaveNet Feather (CH8)   | 4.74e-14        | 2.00e-14          | 133.2    | 8.86e-06 | ✅ Bit-exact floor |
| WaveNet Nano (CH4)      | 6.43e-14        | 3.05e-14          | 131.9    | 7.67e-06 | ✅ Bit-exact floor |
| EVH-5150-Lite (CH12)    | 7.87e-13        | 2.64e-13          | 121.0    | 4.31e-06 | ✅ Near-bit-exact  |
| WaveNet Official (CH3)  | 9.03e-14        | 6.13e-14          | 130.4    | 1.66e-05 | ✅ Bit-exact floor |

### 3.6 Formal `condition_dsp` Specification

`condition_dsp` embeds an auxiliary sub-model that modulates the main network:

- **C++ Sizing & Assertions:** `_condition_output` has dimensions `[condition_dsp->NumOutputChannels(), maxBufferSize]` (`model.cpp:656-660`). C++ enforces a construction-time assertion that `layer_array.condition_size == condition_dsp->NumOutputChannels()`, throwing `std::runtime_error` on mismatch.
- **Python Trainer Constraints:** The upstream trainer (`nam/models/wavenet/_wavenet.py:142-155`) only supports WaveNet as `condition_dsp`; any other architecture raises `NotImplementedError`.
- **Policy Rejection for LSTM Condition DSP:** Configurations specifying LSTM as `condition_dsp` are rejected fail-closed at load time (`Err("LSTM condition_dsp is not supported")`).
- **Head Scale Positioning:** Both production and validation oracles read `head_scale` strictly from the final position of the weight stream, matching upstream binary serializations.

---

## 4. WaveNet A2 Architecture

Audited against `NAM/wavenet/a2_fast.h/.cpp` (fast path), `NAM/wavenet/model.cpp` (generic fallback), and corresponding Rust modules:

- Models: [`src/models/a2/`](../src/models/a2/)
- Dynamic Cascade: [`src/models/a2/model/cascade/`](../src/models/a2/model/cascade/)
- Topology: [`src/loader/nam_json/topology/a2.rs`](../src/loader/nam_json/topology/a2.rs)

### 4.1 Topology Classification & Fast Path

`src/loader/nam_json/topology/a2.rs::is_a2_shape` replicates the 20 structural checks defined in `a2_fast.cpp:is_a2_shape()`:

- Layer count = 23, single layer array, no post-stack head.
- Channels $\in \{3, 8\}$ (Lite / Full) with `channels == bottleneck`.
- Exact dilation patterns and kernel sizes ($K=3$, degridded odd dilations).
- Activation = `LeakyReLU(0.01)`.
- Zero active FiLM slots, gating disabled, `head1x1` disabled, groups = 1.

Models matching these constraints route to `WaveNetA2<3>` or `WaveNetA2<8>`. Topologies with FiLM, gating, blending, cascade arrays, or non-unity groups route to `WaveNetA2Dyn` or `WaveNetA2Cascade`.

### 4.2 Measured Interop Metrics (Fast & Dynamic Paths)

Measured against NAMcore at 48 kHz:

| Variant / Fixture                | ESR vs. NAMcore | ESR vs. f64 Ideal | SNR (dB) | MR-STFT  | Parity Verdict     |
|:-------------------------------- |:---------------:|:-----------------:|:--------:|:--------:|:------------------ |
| WaveNet A2-Full (CH8, Fast)      | 1.12e-13        | 7.83e-14          | 129.5    | 1.68e-05 | ✅ Bit-exact floor |
| WaveNet A2-Lite (CH3, Fast)      | 6.43e-14        | 1.82e-14          | 131.9    | 9.54e-06 | ✅ Bit-exact floor |
| WaveNet A2-FiLM-Full (CH8, Dyn)  | 1.18e-14        | 8.75e-15          | 139.3    | 7.85e-06 | ✅ Bit-exact floor |
| WaveNet A2-FiLM-Lite (CH3, Dyn)  | 3.82e-13        | 1.61e-13          | 124.2    | 1.69e-05 | ✅ Near-bit-exact  |
| WaveNet A2 Dynamic Gated (CH8)   | 5.03e-11        | 1.00e-10          | 103.0    | 6.63e-05 | ✅ Parity verified |
| WaveNet A2 Dynamic Blended (CH3) | 5.35e-14        | 2.65e-14          | 132.7    | 9.97e-06 | ✅ Bit-exact floor |

### 4.3 🔴 Known Bug KB-A2-MAX: `wavenet_a2_max.nam`

**Status:** Permanent known bug. Under active fail-closed dispatch guard (`reject_wavenet_a2_max_class`).

The official flagship model `wavenet_a2_max.nam` triggers severe acoustic divergence against the C++ reference:

| Evaluation Pair             | Metric                            | Status                                           |
|:--------------------------- |:--------------------------------- |:------------------------------------------------ |
| **Rust f32 × C++ Golden**   | **SNR = 1.69 dB** (ESR ≈ 6.78e-1) | 🔴 Unacceptable parity (threshold $\ge 90$ dB)   |
| **Rust f32 × f64 Oracle**   | SNR = −3.30 dB (ESR ≈ 2.14)       | 🔴 Structural divergence from oracle             |
| **f64 Oracle × C++ Golden** | SNR = 0.53 dB (ESR ≈ 8.85e-1)     | 🔴 **Case D:** f64 oracle also diverges from C++ |

#### 4.3.1 Architectural Facts & Verified Invariants

`wavenet_a2_max.nam` combines multiple advanced topology features:

- **Main Network:** $\text{CH}=4, \text{BN}=4, \text{condition\_size}=8, K=4, 2 \text{ layers}$.
- **FiLM Modulation:** 8 FiLM slots active per layer ($16$ total), all with `shift = true`.
- **Grouped Convolutions:** `head1x1.groups = 2`, `groups_input_mixin = 4`, `layer1x1.groups = 2`.
- **Conditioning Sub-Model:** Nested 2-array `WavenetA2Cascade` with `head_size = 4` on Array 0 and `head_size = 8` on Array 1.

**Structural Verification Summary:**

1. **Weight Budget:** Consumes exactly **818** weights in the main network and **1052** weights in `condition_dsp` (Array 0: 617, Array 1: 434, head_scale: 1). Zero orphan or missing weights.
2. **FiLM Layout:** All 16 FiLM slots match theoretical offset formulas (`weights_layout.rs`) with zero memory overlap.
3. **Grouped Layouts:** Row-major group-to-output mapping verified in `build.rs`.
4. **Cascade Buffer Stride:** Multi-channel buffer indexing and slicing verified (`max_diff = 0.0` across 64-sample and 256-sample chunks).

#### 4.3.2 Root Divergence Analysis & Freeze Policy

Because `f64 Oracle × C++ Golden` diverges ($ESR \approx 0.89$, Case D), the mathematical reference oracle itself diverges from C++ NAMcore for multi-array cascades. Comparing Rust against f64 only measures internal consistency, not market parity.

The primary divergence area is isolated to **nested multi-array cascade conditioning and multi-head propagation graph interactions** within C++ Eigen. Speculative modifications to Rust production code without ground-truth intermediate tensor dumps have been halted.

#### 4.3.3 Reopening Criteria

The fail-closed guard `reject_wavenet_a2_max_class` remains permanent until:

1. Intermediate per-frame tensor dumps (Array 0 output, residual projection, Array 1 output, layer FiLM outputs) are extracted from instrumented C++ NAMcore.
2. A single hypothesis demonstrates isolated $\Delta\text{SNR} \gg 10\text{ dB}$.
3. Rust production × C++ golden reaches $\text{SNR} \ge 90.0\text{ dB}$ across standard test inputs with zero regression in neighboring models.

---

## 5. Shared DSP Engine Semantics

### 5.1 Sample Rate Default Policy & Decision DEC-01

- **C++ Reference:** Uses `NAM_UNKNOWN_EXPECTED_SAMPLE_RATE = -1.0` (`NAM/dsp.h:30`) when `sample_rate` is missing from the JSON config, which evaluates prewarm to 1 sample.
- **NeuralAmpModeler-rs Standalone Policy:** Defaults missing `sample_rate` to **48000.0 Hz** (`DEFAULT_SAMPLE_RATE` in `src/loader/loaded_model_pair.rs`).
- *Rationale:* Ensures models settle to their physical steady state during prewarm. Standard community models explicitly declare sample rate; this divergence affects only hand-crafted test configurations.

#### DEC-01 Decision Record — Missing Sample Rate in Sequential Models

When composing child models inside `Sequential`, interaction between missing rates and child rate reconciliation is governed by **DEC-01**:

- **Decision Choice:** **Option A (Recommended & Adopted)** — Propagate `Option<f32>` (where `None` represents unknown/undeclared rate) strictly through model configuration parsing up to the `Sequential` resolver. Standalone models preserve the 48000 Hz global default when instantiated individually.
- **Sequential Resolution Logic (mirrors C++ `sequential.cpp::resolve_expected_sample_rate` L36–62):**
  1. If top-level declares `sample_rate`, all child models with known rates must match it exactly. Children with missing rate (`None`) inherit the top-level rate.
  2. If top-level omits `sample_rate`, the expected rate is resolved to the first known child rate encountered. Subsequent children with known rates must match this resolved rate; children with missing rates are ignored during validation and adopt the resolved rate.
  3. Any conflict between known declared rates (child vs. child or child vs. top-level) triggers immediate fail-closed rejection (`Err(SampleRateMismatch)`).
  4. If all children and top-level omit `sample_rate`, the resolved rate falls back to 48000.0 Hz (`DEFAULT_SAMPLE_RATE`).

| Child 1 Rate | Child 2 Rate | Top-Level Rate | Resolved Rate / Behavior | Rationale |
|:---:|:---:|:---:|:---|:---|
| ∅ | ∅ | ∅ | **48000 Hz** | All rates undeclared; global steady-state default applied |
| ∅ | 44100 Hz | ∅ | **44100 Hz** | Child 1 unknown ignored; Child 2 establishes 44.1 kHz chain |
| 44100 Hz | 48000 Hz | ∅ | **REJECTED** | Conflict between known child rates (`SampleRateMismatch`) |
| ∅ | ∅ | 48000 Hz | **48000 Hz** | Top-level dictates 48 kHz; unknown children conform |
| 44100 Hz | ∅ | 48000 Hz | **REJECTED** | Conflict between Child 1 (44.1 kHz) and Top-Level (48 kHz) |

### 5.2 FastLUTActivation (Not Ported)

C++ provides an optional `FastLUTActivation` class (`NAM/activations.h:374-428`) for hardware lacking fast `expf`.

- Classified as **Not Applicable**.
- `FastLUTActivation` is never enabled in the C++ `render` tool (used exclusively in benchmarking tools).
- NeuralAmpModeler-rs's Standard activation mode provides exact-grade mathematical accuracy ($\sim 2 \times 10^{-7}$ error) without lookup tables.

---

## 6. Other Architectures

### 6.1 ConvNet

Implemented in [`src/models/convnet/mod.rs`](../src/models/convnet/mod.rs) and [`src/loader/dispatcher/convnet/`](../src/loader/dispatcher/convnet/):

- **Pre-fused BatchNorm:** Fuses batch normalization parameters directly into convolutional weights and biases during loading.
- **Silence Prewarm:** `ConvNetModel::prewarm()` processes `receptive_field_size + 1` silence samples through the full network, eliminating initialization transients and matching C++ `dsp.cpp:67-96`.
- **Measured Metrics:**
  - Live C++ Cross-Validation: $\text{ESR} = 4.20 \times 10^{-15}$ ($\text{SNR} = 143.8\text{ dB}$, $\text{MR-STFT} = 1.20 \times 10^{-6}$).
  - f64 Reference Oracle: $\text{ESR} = 3.57 \times 10^{-15}$ ($\text{SNR} = 144.5\text{ dB}$).
  - NumPy Anchor: $\text{ESR} = 5.23 \times 10^{-33}$ (bit-exact).

### 6.2 Affine Linear

Direct FIR affine models (receptive fields 2048, 4096, 8192):

- Measured against NAMcore (mono): $\text{ESR} = 1.70 \times 10^{-14}$ ($\text{SNR} = 137.7\text{ dB}$).

#### 6.2.1 NAMcore v0.6.0 Multichannel Topology & Weight Specification

Upstream NAMcore v0.6.0 (`linear.cpp` L120–147) generalized `Linear` to multichannel configurations:

- **Permitted Topologies:** Valid if `in_channels == out_channels` (arbitrary $N \ge 1$), `in_channels == 1` ($1 \to N$ split), or `out_channels == 1` ($N \to 1$ sum).
- **Prohibited Topologies:** Any configuration where both `in_channels > 1` and `out_channels > 1` with `in_channels != out_channels` (e.g. $2 \to 3$) is rejected fail-closed with `"Linear requires equal channel counts or one input/output channel"` (`linear.cpp` L127–128).
- **Kernel Count Formula:**
  $$\text{kernels} = (in == out) \ ? \ 1 : \max(in, out)$$
  *Crucial Invariant:* For $N \to N$ ($N > 1$), all $N$ channels **share a single impulse response** ($\text{kernels} = 1$).
- **Bias Count Formula:**
  $$\text{biases} = (in == out) \ ? \ 1 : out\_channels$$
- **Total Weight Count:**
  $$N_{\text{weights}} = \text{receptive\_field} \times \text{kernels} + (\text{bias} \ ? \ \text{biases} : 0)$$
- **Weight Layout:** Kernel $k \in [0, \text{kernels})$ occupies slice `weights[k * L .. (k + 1) * L)`. If `bias` is enabled, biases are placed at the end of the weight stream.
- **Execution & Partitioning:** Short receptive fields execute via time-domain direct convolution (`LinearImplementation::Direct`); long receptive fields select partitioned FFT convolution (`LinearImplementation::FFT`) via `select_implementation()` (`linear.h` L125).

#### 6.2.2 Decision DEC-02 — Sample Rate Adaptation of Linear Models

Upstream NAMcore v0.6.0 introduced arbitrary sample-rate adaptation inside `Linear::Reset(sampleRate)` (`linear.cpp` L165–185) via `_resample_impulse_response` (L46–72):

- **C++ Mechanism:** When `sampleRate != training_rate`, C++ resamples the stored impulse response taps using Catmull-Rom cubic Hermite interpolation with gain compensation `(original_rate / desired_rate)` and pads boundaries with zero.
- **NeuralAmpModeler-rs Mechanism:** NeuralAmpModeler-rs keeps stored impulse response taps identical to training data and performs sample-rate conversion on the streaming audio signal via the 256-phase minimum-phase / linear-phase polyphase sinc FIR resampler (`src/dsp/resampler/`, stopband attenuation $\ge 105\text{ dB}$, passband ripple $< 0.05\text{ dB}$).
- **Quantitative Measurement (DEC-02 Audit):**
  Comparison of C++ cubic Hermite resampled IR ($44.1\text{ kHz} \to 48.0\text{ kHz}$) against bandlimited sinc reference:
  - **256 taps:** $\text{ESR} = 4.63 \times 10^{-4}$ ($\text{SNR} = 33.34\text{ dB}$); Audible band (20 Hz – 20 kHz) $\text{Max } |\Delta\text{Mag}| = 32.11\text{ dB}$, $\text{Mean } |\Delta\text{Mag}| = 1.35\text{ dB}$.
  - **2048 taps:** $\text{ESR} = 7.34 \times 10^{-4}$ ($\text{SNR} = 31.34\text{ dB}$); Audible band (20 Hz – 20 kHz) $\text{Max } |\Delta\text{Mag}| = 22.74\text{ dB}$, $\text{Mean } |\Delta\text{Mag}| = 0.96\text{ dB}$.
- **Decision Outcome:** **Option (a) — Intentional Divergence Documented**.
  Resampling IR taps via cubic interpolation degrades audio fidelity down to $\sim 31\text{--}33\text{ dB}$ SNR with severe high-frequency roll-off and imaging near Nyquist. NeuralAmpModeler-rs's polyphase streaming audio resampler maintains $>105\text{ dB}$ fidelity without coloring the acoustic signature of captured cabinets or analog preamps. No baseline thresholds altered.

### 6.3 SlimmableContainer

Multi-model crossfade orchestrator ([`src/models/container.rs`](../src/models/container.rs)):

- Bundles multiple sub-models for dynamic compute scaling.
- Tested on official bundled capture `a2_example.nam` (CH 3→6): $\text{ESR} = 7.28 \times 10^{-14}$ vs. NAMcore ($\text{SNR} = 131.4\text{ dB}$).

### 6.4 Impulse Response CabSim

Uniform-partitioned overlap-save FIR convolution, cross-validated against `NeuralAmpModelerPlugin`'s `dsp::ImpulseResponse` in `tests/parity/cabsim_cpp_parity.rs`.

### 6.5 SlimmableWavenet

Dynamic single-network channel slicing ([`src/models/slimmable.rs`](../src/models/slimmable.rs)):

- Supports adaptive quality scaling via channel slicing (`allowed_channels`).
- Fully functional in loading and inference (`test_slimmable_wavenet_inference_and_breakpoints`).
- **Parity Scope:** Inference-only. Upstream NAMcore provides no channel-slicing API; multi-size C++ parity claims are architecturally out of scope.

### 6.6 Sequential (NAMcore v0.6.0)

Upstream v0.6.0 introduced the `"Sequential"` serial pipeline model architecture (`sequential.cpp`):

- **Architecture Registration:** Top-level identifier `"Sequential"` (case-sensitive; lowercase `"sequential"` rejected fail-closed).
- **Submodel Array:** `config.models` is mandatory and must be a non-empty array (`sequential.cpp` L82–89).
- **Top-Level Weights Empty Invariant:** Top-level `weights` array must be empty (`[]`); any weights present at top-level trigger immediate runtime error (`sequential.cpp` L224–227). All weights belong strictly to child models.
- **Child Envelope Completeness:** Each child model must be a complete, self-contained `.nam` envelope containing `"version"`, `"architecture"`, `"config"`, and `"weights"` (`sequential.cpp` L93–103). Legacy bare child configs are rejected fail-closed.
- **Recursive Composition:** Child models are constructed recursively via `nam::get_dsp(json)`, allowing nested `Sequential` children (`sequential.cpp` L104).
- **Stage Channel Alignment:** Stage $i$ output channels must match stage $i+1$ input channels (`output_channels(i) == input_channels(i+1)`) (`sequential.cpp` L64–78). Model input channels equal stage 0 input channels; model output channels equal the final stage output channels.
- **Sample Rate Reconciliation:** Resolved via `resolve_expected_sample_rate` (DEC-01): unknown rates (`-1` / `None`) are ignored; conflict between known child rates or between child rate and top-level rate triggers immediate error (`sequential.cpp` L36–62).
- **Prewarm Semantics:** `GetPrewarmSamples()` is the saturating sum of child prewarm sample counts (`sequential.cpp` L189–199). On `Reset()`, child `PrewarmOnReset` flags are saved and disabled, all children are reset, child flags are restored, and a single overall chain prewarm is executed (`sequential.cpp` L152–180). `SetPrewarmOnReset` propagates to all child models (L182–187).
- **Real-Time Safety & Buffering:** Intermediate ping-pong buffers (`stages - 1` buffers $\times$ stage output channels) are allocated off-RT during `SetMaxBufferSize()` (`sequential.cpp` L202–222). Hot-path `process()` operates zero-alloc. The final stage writes directly to the caller's output buffer.

**Rust Robustness Hardening (declared divergences):** Upstream applies no recursion or size limits to `Sequential` trees. The Rust loader fail-closes hostile topologies before any child allocation ([§6.6](#66-sequential-namcore-v060)): maximum nesting depth (`MAX_SEQUENTIAL_DEPTH` = 8, `E1311`), maximum total child count across the whole tree (`MAX_SEQUENTIAL_TOTAL_CHILDREN` = 64, checked arithmetic, `E1314`), and an aggregate child weight budget (`MAX_SEQUENTIAL_TOTAL_WEIGHTS`, the single-model float cap, `E1314`). Every child envelope is additionally validated with the strict `validate_envelope` schema (types enforced; C++ checks key presence only at `build_models` L93–103) — a superset that preserves upstream rejection outcomes with typed diagnostics.

**Rust Status (NC-3.2, 2026-10-07):** The chain engine is registered (`models::SequentialModel` behind `StaticModel::Sequential`, static enum dispatch — no `dyn` on the hot path; stages stored as plain `StaticModel` children). Construction performs the C++ ctor validations in order: presence (`E1306`), DEC-01 rate resolution (`E1309`), channel-link validation (`E1308`) — asymmetric `LinearOneToMany`/`LinearManyToOne` geometries (NC-2.2) compose freely. Prewarm mirrors the C++ lifecycle: saturating child sum (`usize::MAX` vs the C++ `i32::MAX` — engine-independent overflow point), child flag save/disable/reset/restore with the single chain-wide stabilization pass, and a split flow (`prewarm_reset`/`prewarm_step`/`prewarm_complete`) that is bit-exact against the integral path under arbitrary caller chunking. Inter-stage planes and channel pointer tables are pre-allocated off-RT (`set_max_buffer_size` and the constructor's engine-max default), the last stage writes directly into the caller's output, and blocks beyond the negotiated maximum are truncated with a symmetric `debug_assert!` (trait block contract) — the formal RT error contract and the heap-audit/deadline proofs are NC-3.3 deliverables; layer-skip adaptivity remains stage-local (does not descend into chains).

#### 6.6.1 Live C++ v0.6.0 Cross-Validation & Single-Prewarm Proof (NC-3.4, 2026-10-07)

The `tests/parity/sequential_cpp_parity.rs` module cross-validates whole chains against the C++ `render` tool at engine level and against the double-precision composition oracle, on committed deterministic fixtures (`tests/fixtures/models/sequential_*.nam`, NC-3.4 batch). The f64 oracle mirrors the C++ `get_dsp` recursion: each child is re-parsed as a standalone envelope, nested `Sequential` children recurse, `Linear` children route through `oracle_linear_multichannel` (channel planes), and neural children route through the shared f64 kernels; the chain-wide stabilization budget (`model.prewarm_samples()`, the exact mirror of the C++ saturating child sum) is replayed as the same zero-feed the engines execute inside `Reset`, and the pre-signal slice is dropped. The **single-prewarm invariant** is validated on the initial transient — the first 256 frames of the real signal, not just steady state — with a positive control proving the gate has detection power: a no-warm chain (prewarm disabled) drifts the transient six orders of magnitude above the warmed floor (2.291e-3 vs 7.451e-9, double-LSTM fixture).

Block-schedule invariance (fixed 64-frame wheel mirroring `render.cpp` vs irregular 1..=64 chunking) is asserted bit-equal on every run.

Measured parity (release; 64-frame blocks; v2 stress signal at each rate):

| Fixture (chain)                     | Rate              | Rust × C++ ESR¹ | Rust × C++ SNR | Rust × f64 ESR | Rust × f64 SNR | C++ × f64 ESR | C++ × f64 SNR | Transient² max abs diff | Prewarm frames |
|:------------------------------------|:------------------|:----------------|:---------------|:---------------|:---------------|:--------------|:--------------|:------------------------|:---------------|
| `sequential_linear2.nam` (L→L)      | 44.1k–192k sweep  | 3.21–3.27e-16   | 154.9–155.0    | 4.40–4.53e-16  | 153.4–153.6    | 3.85–3.94e-16 | 154.1         | 7.451e-9              | 0              |
| `sequential_linear_chain.nam` (L→L) | 48k               | 1.572e-15       | 148.0          | 1.997e-15      | 147.0          | 1.470e-15     | 148.3         | 3.725e-9              | 0              |
| `sequential_nested.nam` (L→S[L])    | 48k               | 1.572e-15       | 148.0          | 1.997e-15      | 147.0          | 1.470e-15     | 148.3         | 3.725e-9              | 0              |
| `sequential_multichannel.nam` (1→2→1)| 48k              | 3.749e-15       | 144.3          | 3.945e-15      | 144.0          | 3.480e-15     | 144.6         | 1.490e-8              | 0              |
| `sequential_linear_wavenet.nam` (L→WN)| 44.1k–192k sweep | 3.46e-15        | 144.6          | 3.86e-15       | 144.1          | 1.42e-15      | 148.5         | 2.328e-10             | 6              |
| `sequential_double_lstm.nam` (LSTM→LSTM) | 48k          | 1.023e-14       | 139.9          | 1.022e-14      | 139.9          | 9.217e-16     | 150.4         | 7.451e-9              | 48000          |
| positive control (no prewarm, double LSTM × same renders) | 48k | —      |                 |                 |               |              |               | 2.291e-3              | 0 (defect)     |

¹ ESR floors (error-to-signal ratio: noise-to-signal energy, 0 = bit-exact). Gates calibrated in the test module header (`MONO_*_LIMITS`) at the task-proposed norms — ESR < 1e-12 / SNR > 120 dB — with ≥ 260× / ≥ 20 dB margins over every measured cell, including the recurrent chain.
² First 256 signal frames after the single chain-wide warm (windowed SNR gates ≥ 110 dB round-trip at the f32 kernel noise; the WaveNet child's windowed SNR resolved to inf — zero abs diff within the window at the ULP of the chained response).

**Prewarm count arithmetic verified:** Linear contributes 0 (`dsp.h` default), WaveNet contributes 1 + Σ array RFs (`wavenet/model.cpp` L656–660, RUST mirror = RF-sum), LSTM contributes 0.5·declared rate (24000 @ 48 kHz, `lstm.cpp` L127–133). C++ `DSP::prewarm` feeds whole 64-frame chunks (overshoot tolerance, `dsp.cpp` L95–99) while the Rust chain feeds the exact saturating sum — the divergence is latent only: the chains' zero responds are count-invariant at the settling depth both engines feed (bias constants for Linear; tap-line tails for WaveNet any depth ≥ RF; geometric fixed-point convergence for LSTM), and the double-LSTM fixture additionally uses a 48000 = 750·64 frame budget, exact at the C++ chunk granularity. Initial transients therefore measured **identical** between engines at every fixture and rate.

**DEC-01/DEC-02 live semantics:** `sequential_linear2.nam`/`sequential_linear_wavenet.nam` declare no rate anywhere (all-unknown branch): the C++ `render` accepts any input WAV rate and runs at it; the Rust chain resets operationally at the WAV rate. The declared-rate double-LSTM chain keeps the DEC-02 contract live: the C++ render rejects non-48k inputs (exit 1, mirrored probe) while the Rust engine accepts and reconfigures — no reinstability at the operational rate.

**Note:** the small WaveNet child of `sequential_linear_wavenet.nam` (Linear → WaveNet) is canonical two-array A1 (correct f64-oracle routing requires `.layers.len() >= 2` for single-layer WaveNets, which the A1 oracle bail-outs to silence).

---

## 7. Known-Broken & Policy Ledger

### 7.1 🔴 Known Bug KB-A2-MAX

| Item / Model         | Symptom                                                             | Status                                                                      |
|:-------------------- |:------------------------------------------------------------------- |:--------------------------------------------------------------------------- |
| `wavenet_a2_max.nam` | Rust × C++ $\text{SNR} = 1.69\text{ dB}$; Case D triple divergence. | **KB-A2-MAX Frozen.** Fail-closed guard active. Reopen strictly via §4.3.3. |

- Guard: `reject_wavenet_a2_max_class` returns explicit error on model load.
- Unlock (test/diagnostic only): `NAM_A2_MAX_UNLOCK=1 cargo test ...`.
- CI Gate: `test_wavenet_a2_max_dispatch_rejected` must remain green.

### 7.2 🟡 Deliberate Engineering Tradeoffs

| Feature                        | Implementation Detail                                                                                              | Impact / Parity State                                                                 |
|:------------------------------ |:------------------------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------- |
| **Native f32 Weights**         | Retains full precision single-precision storage instead of quantization (bf16/f16c).                               | Bit-exact / noise-floor convergence with NAMcore (e.g., BossLSTM-2×8 ESR = 1.00e-11). |
| **Fast Activation Precision**  | Padé [5,4] $\tanh$ and minimax degree-17 $\sigma$ (opt-in).                                                        | $\sim 10\times$ faster activation kernels; Standard mode remains exact-grade default. |
| **Analytical WaveNet Prewarm** | $\mathcal{O}(\text{layers})$ fixed-point fill instead of $\mathcal{O}(\text{receptive\_field})$ iterative rollout. | Exact mathematical convergence with zero startup overhead.                            |
| **Linear Sample Rate (DEC-02)**| Preserves native IR taps and resamples audio via polyphase sinc FIR (`NamResampler`).                              | $>105\text{ dB}$ SNR vs $\sim 31\text{--}33\text{ dB}$ for C++ cubic tap interpolation ([§6.2.2](#622-decision-dec-02--sample-rate-adaptation-of-linear-models)). |
| **Sequential Rate (DEC-01)**   | Option A: ignores unknown rates inside Sequential resolution; defaults to 48 kHz if all absent.                   | Full upstream interop for multi-rate child chains ([§5.1](#dec-01-decision-record--missing-sample-rate-in-sequential-models)). |

### 7.3 🟠 Test Infrastructure Boundaries

- **Live Parity Enforcement:** All non-ignored `quick_parity_*` tests require `require_completed()`, failing hard on missing compilers or crashed render tools.
- **Fixture Resolution:** `golden_gen_build.sh` mirrors `tests/common/io_helpers.rs::model_path`, searching standard system paths, environment overrides, and non-distributable community directories.
- **Multichannel Render Oracle Fallback:** The upstream C++ `render` CLI supports mono input only and outputs channel 0 only. For multichannel **exterior** models ($1 \to N$, $N \to 1$, $N \to N$), live `render` cannot serve as oracle; the double-precision (f64) mathematical reference oracle and algebraic invariants from C++ `test_linear.cpp` serve as co-equal oracles. Chains whose **exterior** is mono but whose interior stages cross channel boundaries ($1 \to 2 \to 1$, `sequential_multichannel.nam`) remain fully dispatchable by live `render` and are cross-validated there (NC-3.4: ESR = 3.749e-15, SNR = 144.3 dB at 48 kHz).

### 7.4 🟡 Policy Rejections & Defensive Gaps

| ID  | Target / Feature                        | Classification        | Contract / Enforcement Mechanism                                                                                                                        |
|:--- |:--------------------------------------- |:--------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | `wavenet_condition_lstm.nam`            | **Policy Reject**     | Loader returns `Err("LSTM condition_dsp is not supported")`. Upstream toolchain cannot train or validate this geometry.                                 |
| P2  | WaveNet A1 models declaring A2 features | **Fail-Closed**       | Rejects `gated`, FiLM, `head1x1`, `layer1x1`, and secondary activations at topology detection ([§3.4](#34-fail-closed-rejection-of-a2-features-in-a1)). |
| P3  | LSTM degenerate topologies              | **Fail-Closed**       | Multi-channel → `Err(UnsupportedMultiChannel)`. Zero layers or overflow → `Err(UnsupportedTopology)`.                                                   |
| P4  | Multi-array receptive field summation   | **Canonical Sum**     | `prewarm_samples()` sums all array RFs + condition_dsp + post-stack head ([§3.2](#32-structural-implementation-mapping)).                               |
| P5  | `dsp_ch < condition_size` Broadcast     | **Rust Fallback**     | Broadcasts channel 0 across inputs if sub-model channels are fewer than parent condition size.                                                          |
| P6  | `SlimmableWavenet` Multi-Size Parity    | **Inference Only**    | Validated for loader and runtime inference; no upstream multi-size parity claimed ([§6.5](#65-slimmablewavenet)).                                       |
| P7  | A2 Fast-Path Fixtures Synthetic Only    | **Documented Caveat** | Fast-path parity validated on synthetic calibrated weights; community trained captures restricted by third-party licenses.                              |
| P8  | Linear $M \to N$ Channel Mismatch       | **Fail-Closed**       | Rejects topologies where both $in > 1$ and $out > 1$ with $in \neq out$ (e.g. $2 \to 3$) ([§6.2.1](#621-namcore-v060-multichannel-topology--weight-specification)). |
| P9  | Sequential Child & Top-Level Invariants | **Fail-Closed**       | Rejects non-empty top-level weights, non-matching inter-stage channels, and conflicting known sample rates ([§6.6](#66-sequential-namcore-v060)).       |

---

## See Also

- [`architecture.md`](architecture.md) — Core engine dispatch, SIMD microarchitecture, and pipeline design.
- [`audio_fidelity_map.md`](audio_fidelity_map.md) — In-depth analysis of audio precision floors, resampler, and numerical budgets.
- [`fixtures.md`](fixtures.md) — Golden vector generation, model hash manifests, and fixture supply chain.
- [`perceptual_validation.md`](perceptual_validation.md) — Numerical formulations of ESR, SNR, MR-STFT, and perceptual calibration.
- [`testing.md`](testing.md) — Comprehensive test architecture, suite phases, and quality gates.
