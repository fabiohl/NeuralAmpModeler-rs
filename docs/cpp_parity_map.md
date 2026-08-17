<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# C++ ↔ Rust Parity Audit — NeuralAmpModelerCore × NAM-rs

Ground-truth comparison between the canonical C++ reference, **NeuralAmpModelerCore**
("NAMcore", vendored read-only at `third-party/NeuralAmpModelerCore/`), and the NAM-rs
Rust engine (`src/`). Two independent oracles serve complementary roles: **NAMcore**
preserves market compatibility with existing models (interop parity), while the **f64
reference oracle** measures fidelity against the mathematical ideal (precision).
Neither oracle has automatic prevalence over the other — any disagreement between them
requires human analysis and `REVIEW_REQUIRED` marking (see [§1.2](#12-two-oracle-governance-policy)).

This document is audited **per architecture, in phases**, by reading the vendored C++ source
line-by-line against the Rust implementation — not by trusting prior write-ups. Each
architecture section carries a verification banner stating what was actually re-checked and
when. **For a single-page triage of what is actually broken vs. what is under control, read
[§7](#7-known-broken-ledger-sabidamente-broken) first.**

## 0. Audit Status

> **Audit status:** Strict fail-closed policy maintained. **KB-A2-MAX remains frozen**
> (fail-closed TR1.1; do not reopen without §4.4.3). Verified against current canonical source:
> §3.5 condition_dsp canonical RF summation, §3.6 fail-closed A1/A2 guards, §7 known-broken ledger.
> No production code change for Max.

| Architecture                | Status                                                                                                                                      | Section                                   |
|:--------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------- |:----------------------------------------- |
| **LSTM**                    | ✅ Fully Verified — Native f32 weights, bit-exact/sub-1e-11 interop parity vs NAMcore                                                       | [§2](#2-lstm-architecture)                |
| **WaveNet A1**              | ✅ Fully Verified — Const-generic fast path & dynamic fallback pass canonical golden gates; A1 A2-feature guard fail-closed (§3.6 FIXED)    | [§3](#3-wavenet-a1-architecture)          |
| **WaveNet A2**              | 🟡 Verified Dynamic/Fast paths — 🔴 Flagship `wavenet_a2_max.nam` **KB-A2-MAX** known bug (fail-closed TR1.1; prod×C++ **0.23 dB**; §4.4.3) | [§4](#4-wavenet-a2-architecture)          |
| **ConvNet**                 | ✅ IDENTICAL — Full Initialization & Arithmetic Parity (prewarm fix eliminates 2.54e-5 transient)                                           | [§6](#6-other-architectures-out-of-scope) |
| Linear / Container / Cabsim | ✅ Verified — Affine linear, SlimmableContainer, and IR Cabsim covered by targeted test suites                                              | [§6](#6-other-architectures-out-of-scope) |
| **SlimmableWavenet**        | 🟡 Loads + inference OK — inference-only; no multi-size NAMCore parity claim (§6 / §7.4)                                                    | [§6](#6-other-architectures-out-of-scope) |

## 1. Methodology

### 1.1 Two axes of correctness

1. **Market-compatibility parity** (NAMcore oracle) — does NAM-rs match NAMcore within
   float tolerance? Verified with committed golden vectors (`tests/fixtures/*.bin`, generated
   by the C++ `render` tool) and live cross-validation (`tests/parity/cpp_parity.rs`). This
   oracle preserves seamless interop with the existing NAM model ecosystem. All weights are
   native f32 (weight quantization was eliminated; NAMcore never quantized).

2. **Mathematical-ideal fidelity** (f64 oracle) — how far is NAM-rs from the exact
   mathematics? Measured against an independent f64 reference oracle
   (`src/testing/reference_oracle/mod.rs`), itself cross-checked against a third, independent
   NumPy f64 implementation. This oracle isolates the genuine precision floor — quantifying
   how much quality is lost to f32 arithmetic, activation approximations, and structural
   divergence — independently of any particular C++ implementation.

### 1.2 Two-Oracle Governance Policy

NAM-rs uses two oracles with complementary roles and **equal authority** — neither has
automatic prevalence over the other.

| Oracle      | Question Answered                                                                        | Authority                           |
|:----------- |:---------------------------------------------------------------------------------------- |:----------------------------------- |
| **NAMcore** | Does NAM-rs produce audio compatible with the existing model ecosystem? (market interop) | Sole arbiter of interop parity      |
| **f64**     | What is the mathematical ideal, and how far is NAM-rs from it? (precision floor)         | Sole arbiter of ideal-math fidelity |

**Disagreement protocol:**

When the two oracles disagree — i.e., NAMcore reports acceptable parity but f64
shows significant deviation from ideal math, or vice versa — **no oracle
automatically prevails.** The divergence must be:

1. **Documented** in this parity map with both oracle measurements.
2. **Triaged** by a human reviewer to determine the root cause (which may be
   a C++-side approximation, a Rust-side structural divergence that NAMcore
   also exhibits, or a genuine bug in either oracle).
3. **Marked `REVIEW_REQUIRED`** in the affected model's catalog entry and test
   gate until resolution.

A production-code change justified solely by improving one oracle's metric
while regressing the other is prohibited. Changes must either improve both
oracles or have a documented, human-reviewed rationale for the tradeoff.

**Historical note:** A prior audit round erroneously declared NAMcore the "sole
source of truth" and prescribed that the f64 oracle must always be fixed to
match C++ when they disagree (§4.5). This policy has been **superseded** —
NAMcore and f64 are co-equal oracles, and disagreements between them are
governance events, not automatic victories for either side.

### 1.3 Reference version

The vendored working copy at `third-party/NeuralAmpModelerCore/` is checked out at tag
`v0.5.4` (commit `1f42f88`; `NAM/version.h` still says `0.5.3` — the header wasn't bumped for
the tag). Some older committed golden vectors were generated at `v0.5.3` (`9c7b185`). This
patch-level drift is below the interop noise floor for all architectures except where explicitly
noted per-model. Regenerate goldens with `tests/fixtures/golden_gen_build.sh` when in doubt.

### 1.4 Fixture governance: `docs/fixtures.md`

[`docs/fixtures.md`](fixtures.md) is the canonical operational
supply-chain contract. Every parity claim in this document is operationalized through it.

| Layer                               | Mechanism                                                                                                                                                                                                                                                       | Hard-fail gate                                                          |
|:----------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------- |
| **Layer 0 — Generation pipeline**   | `tests/fixtures/golden_gen_build.sh` consumes the Rust golden registry `src/testing/catalog.rs::GOLDEN_GEN_CATALOG` (via `nam_golden_catalog emit-catalog`) — regenerates every `.bin` golden from the pinned NAMcore C++ `render` tool (no bash catalog array) | —                                                                       |
| **Layer 1 — Pre-committed goldens** | `tests/models/golden_vectors.rs` — compares Rust output against committed `.bin` files; no C++ toolchain required                                                                                                                                               | `utils/tests-quick.sh` Phase 2                                          |
| **Layer 2 — Live cross-validation** | `tests/parity/cpp_parity.rs` — builds C++ `render` tool and compares fresh output: `quick_parity` subset in `utils/tests-quick.sh` Phase 2; full `#[ignore]`d v1/v2 multi-SR matrix via `utils/tests-long.sh`                                                   | `utils/tests-quick.sh` Phase 2 (`quick_parity`) + `utils/tests-long.sh` |

**Freshness manifest:** `tests/fixtures/.golden_manifest.sha256` contains `sha256` of every
model *and* its golden. Verified as a **hard gate** in both `utils/tests-quick.sh` Phase 2
and `utils/tests-long.sh` (a stale `.nam` or `.bin` fails the suite).

**NAMcore mirror pinning:** `variables.env` pins the vendored C++ reference at
`NAM_CORE_COMMIT=1f42f88535884450104b8711d7595019afa0495b` (tag `v0.5.4`). Update
via `utils/setup-third-party.sh`. See [`docs/fixtures.md`](fixtures.md) for the full regeneration walkthrough.

**Calibrated thresholds:** per-model SNR/ESR gates cross-checked by
`tests/models/threshold_calibration.rs` (anti-placebo meta-tests, `// Measured:` provenance
comments). A claim not traceable to a catalog entry + manifest hash + calibrated threshold
is unverified.

#### 1.4.1 Runner execution and fidelity seals

**Quick suite (`utils/tests-quick.sh`)** — three phases; the parity gates live in
Phase 2 (`--release`, production float gate):

- **Layer 1 (`golden_vectors` v1 + `isa_parity` v2):** runs only when the committed
  `.bin` fixtures are present (`tests/fixtures/golden_wavenet_standard.bin` and
  `..._v2_48000.bin`). Missing fixtures record the gap
  `GAP: golden_vectors+isa_parity:missing_fixtures` and the oracle batch shrinks to
  the fixture-independent oracles (`reference_oracle_f64`, `spectral_fidelity`,
  `linear_fft_test`, `lstm_activation_precision`). Diagnostic:
  `tests/fixtures/golden_gen_build.sh`.
- **Layer 2 (`cpp_parity quick_parity`):** requires the NAMcore mirror
  (`ensure_third_party soft`) and a C++ compiler (`$CXX`, else `g++`/`clang++`).
  The `render` binary is ensured via the **single unified build entry point**
  `utils/ensure_namcore_render.sh` (`_lib.sh::ensure_namcore_render`):
  idempotent (skips cmake when `build/namcore_render/.build_config` matches the
  `$CXX:$BUILD_TYPE:$FLAGS` fingerprint; the binary is probed at
  `tools/render` → `Release/render` → `Debug/render`, matching
  `src/testing/fixtures.rs::render_bin_path`), builds Release with
  `-DNAM_ENABLE_A2_FAST=ON`, and logs `target/logs/cmake-configure.log` /
  `cmake-build.log`. The same helper is used by `tests-long.sh` Phase 0,
  `tests/fixtures/golden_gen_build.sh` and the Rust fallback in
  `tests/parity/cpp_parity.rs`. Missing mirror, missing compiler, missing
  cmake, or a failed CMake build each record a gap: `cpp_parity:no_namcore` /
  `cpp_parity:no_cxx` / `cpp_parity:no_cmake` / `cpp_parity:cmake_failed`.
- **Fail-closed status — a green seal is never emitted for skipped oracles.** Any
  recorded gap prints `FIDELITY: INCOMPLETE` and `OVERALL: PASSED_WITH_GAPS` (exit 0).
  `FIDELITY: OK` + `OVERALL: PASSED` is emitted **only** when every oracle actually
  executed (each phase is also post-checked by `assert_ran_tests`, so an empty test
  selection fails the suite). Freshness failures are `FIDELITY: FAIL` (exit 1).
  Therefore the absence of fixtures, of the NAMcore mirror, or of the C++ compiler
  **never** yields a green fidelity stamp.

| Status emitted                                       | Meaning                                                                             | Exit |
|:---------------------------------------------------- |:----------------------------------------------------------------------------------- |:----:|
| `FIDELITY: OK` / `OVERALL: PASSED`                   | All three phases executed; zero gaps                                                | 0    |
| `FIDELITY: INCOMPLETE` / `OVERALL: PASSED_WITH_GAPS` | At least one oracle skipped (fixture/NAMcore/compiler gap); `GAP:` lines in receipt | 0    |
| `FIDELITY: FAIL` / `OVERALL: FAIL`                   | Freshness gate failed, a test failed, or an unexpected error                        | 1    |
| `OVERALL: FAIL reason=strict_gaps`                   | Gaps present **and** `NAM_QUICK_STRICT=1` — gaps promoted to failure                | 1    |

- **`NAM_QUICK_STRICT=1`:** promotes every recorded gap to a hard failure
  (`OVERALL: FAIL reason=strict_gaps`, exit 1). Use for release gates where a skipped
  oracle is unacceptable. The `STRICT:` line in the receipt records the value used.
- **Receipt:** every run appends a machine-readable receipt to
  `target/logs/quick-receipt.txt` (`SUITE: tests-quick`, `STRICT: <0|1>`,
  `PHASE1/2/3: PASS <...>`, one `GAP: <id>` per skipped oracle, `OVERALL: <verdict>`),
  alongside the per-phase logs `target/logs/quick-phase{1,2,3}.log`. The script
  re-executes itself at low CPU/IO priority (`nice`/`ionice`) unless
  `NAM_NO_LOW_PRIORITY=1`.

**Long suite (`utils/tests-long.sh`)** — human-operated nightly/pre-release audit
(AI agents must never execute it; binding project rule):

- **Pre-flight gates:** requires the NAMcore mirror (`ensure_third_party hard` — a
  missing mirror aborts the suite). Golden/fixture presence is validated
  exclusively by the Rust gates: `catalog_preflight` (fixture catalog + V1
  golden matrix via `validate_v1_goldens` + V2 multi-SR matrix via
  `validate_v2_catalog`, both in `src/testing/catalog.rs`) and
  `check_freshness` (nam_freshness manifest gate) — missing required goldens
  fail the preflight (the former bash golden lists and the auto-rebuild were
  removed; regenerate with `tests/fixtures/golden_gen_build.sh`).
  It then ensures the C++ `render` binary via the unified `ensure_namcore_render`
  — a render build failure aborts the suite with
  `target/logs/cmake-configure.log` / `cmake-build.log` diagnostics.
- **Layer 2 full matrix:** the `#[ignore]`d `live_cross_validation_*` v1/v2 multi-SR
  tests and the full `cpp_parity` matrix run in the release parity phase.
- **Summary:** `FIDELITY: FAIL` (exit 1) when any fidelity-class phase failed;
  otherwise `FIDELITY: OK` with `OVERALL: FAILED` (exit 1, non-fidelity failure),
  `OVERALL: COMPLETED_WITH_GAPS` (exit 0 — skipped/inconclusive phases, e.g. RT jitter
  without capability; `--strict-pre-release` turns this into exit 1), or
  `OVERALL: PASSED` (exit 0, zero gaps). All phase logs land in `target/logs/`.

---

## 2. LSTM Architecture

Read against `NAM/lstm.h`, `NAM/lstm.cpp`, `NAM/dsp.h`, `NAM/dsp.cpp`, `NAM/activations.h/.cpp`
and the corresponding Rust modules (`src/models/lstm/`, `src/loader/dispatcher/lstm/`,
`src/loader/transpose/lstm.rs`, `src/math/lstm/gates.rs`).

### 2.0 Supported sample rates

| Model             | Golden Vectors (v1) | Golden Vectors (v2)        | Live C++ Parity (v2)       |
|:----------------- |:-------------------:|:--------------------------:|:--------------------------:|
| BossLSTM-1×16     | 48 kHz              | 44100, 48000, 88200, 96000 | 44100, 48000, 88200, 96000 |
| BossLSTM-2×8      | 48 kHz              | 44100, 48000, 88200, 96000 | 44100, 48000, 88200, 96000 |
| LSTM Official     | 48 kHz              | 48000                      | 48000                      |
| LSTM 1×10 (synt.) | 48 kHz              | 48000                      | 44100, 48000, 88200, 96000 |
| LSTM 2×24 (synt.) | 48 kHz              | 48000                      | 44100, 48000, 88200, 96000 |
| LSTM 3×8 (synt.)  | 48 kHz              | 48000                      | 44100, 48000, 88200, 96000 |
| lstm_dyn_test.nam | 48 kHz              | —                          | 48000                      |

**192 kHz is excluded from all LSTM testing** — both golden vectors and live cross-validation.
See [§2.9](#29-192-khz-limitation-lstm) for the formal limitation and root cause.

### 2.1 Reference algorithm (`NAM/lstm.cpp`)

A stack of `num_layers` LSTM cells, each processing one audio sample at a time, followed by a
linear head:

```text
for each layer i:
  ifgo = W_i · [input ; hidden_i] + b_i        // ifgo = [input_gate, forget_gate, cell_candidate, output_gate]
  c_i  = sigmoid(forget) * c_i + sigmoid(input) * tanh(cell_candidate)
  h_i  = sigmoid(output) * tanh(c_i)
output = head_weight · h_last + head_bias        // no activation
```

Gate order in the weight/state vectors is fixed: **I, F, G, O** at offsets `0, H, 2H, 3H`
(`lstm.cpp:40-44`).

By default (`Activation::using_fast_tanh = false`, `activations.cpp:16`), the gate
nonlinearities are `sigmoid(x) = 1/(1+exp(-x))` and `tanhf(x)` — both **exact** libm/expf-based
implementations (`lstm.cpp:57-65`, `activations.h:64-67`). The `fast_sigmoid`/`fast_tanh`
rational-approximation branch (`lstm.cpp:46-56`) is only enabled by `Activation::enable_fast_tanh()`,
which is called **only** from `tools/benchmodel*.cpp` — never from the `render` tool used to
generate goldens or run live cross-validation. **The C++ reference used for all LSTM parity
checks always uses exact math**, not its own fast-tanh approximation.

C++ generalizes the topology to arbitrary `in_channels`/`out_channels` and even `num_layers ==
0` (pure passthrough, `lstm.cpp:139-149`, zero-filling extra output channels). In practice every
known `.nam` LSTM model is mono in/out with `num_layers ∈ {1, 2}` — see [§2.6](#26-scope-divergence-mono-only-no-zero-layer-support).

### 2.2 Rust implementation

| C++ (`NeuralAmpModelerCore/`)                                                                       | Rust (`src/`)                                                                                                                                                           | Verdict                                                                                                                                       |
|:--------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------- |
| `LSTMCell::process_` gate math (`lstm.cpp:31-66`)                                                   | `math/lstm/gates.rs::fused_lstm_gates_{avx2,avx512}` + `models/lstm/layer_kernels.rs`                                                                                   | ✅ Match — same gate order, same `f·c + i·tanh(g)` / `o·tanh(c')` formulas                                                                    |
| Gate-major weight matrix `[4H × (I+H)]`, row-major (`lstm.cpp:19-21`)                               | `LstmLayer::input_hidden_weights: [[[u16; H]; IH]; 4]`, filled by `read_lstm_weights_into`                                                                              | ✅ Match — verified byte-for-byte against the constructor loop                                                                                |
| Bias `[4H]`, initial hidden `[H]`, initial cell `[H]` read order (`lstm.cpp:22-28`)                 | `read_lstm_layer` reads bias → hidden-init → cell-init in the same order                                                                                                | ✅ Match                                                                                                                                      |
| 2-layer chain: `layers[i].process_(layers[i-1].hidden)` (`lstm.cpp:151-153`)                        | `LstmModel2` software-pipelined chain (`model2.rs`) — layer2 consumes layer1's *previous*-step hidden state, reordered for throughput                                   | ✅ Match — mathematically identical sequential stacking, just reordered for instruction-level parallelism                                     |
| Head: `output = head_weight · h_last + head_bias`, no activation (`lstm.cpp:161-164`)               | `dot_product(..) + head_bias`, computed in **native f32 with Kahan compensation** (`use_f32_head = true` in every loader path) when quantized weights are not requested | ✅ Match (superset — Kahan compensation only *reduces* summation error vs. plain accumulation)                                                |
| `GetPrewarmSamples() = 0.5 × expected_sample_rate` (min 1) (`lstm.cpp:125-132`)                     | `prewarm_samples()` — identical formula (`models/lstm/mod.rs`)                                                                                                          | ✅ Match — **corrects a prior claim** in this document that Rust diverged here; both engines have the same opt-out flag with the same default |
| `DSP::Reset()` calls `prewarm()` only `if GetPrewarmOnReset()` (default `true`) (`dsp.cpp:130-139`) | `NamModel::reset()` calls `prewarm()` only `if self.prewarm_on_reset()` (default `true`)                                                                                | ✅ Match — **corrects a prior claim** in this document that Rust diverged here; both engines have the same opt-out flag with the same default |
| Backbone weights are plain `float` (Eigen), no quantization                                         | Gate weights are native f32, dispatched through f32-only GEMV kernels                                                                                                   | ✅ Match — see [§2.5](#25-native-f32-backbone-weights-and-activation-precision)                                                               |

### 2.3 Weight loading (`.nam` JSON / NAMB)

Layout is `input_hidden_weights[4H×IH] → bias[4H] → hidden_init[H] → cell_init[H]`, repeated per
layer, then `head_weights[H] → head_bias`. This is read identically in
`src/loader/dispatcher/lstm/weights.rs::read_lstm_layer{,_dyn}` and cross-checked against
`src/loader/transpose/lstm.rs` (used for the NAMB `GateMajorLstm` pre-transposed layout). Both
paths were read line-by-line against `LSTMCell`'s constructor (`lstm.cpp:9-29`) — confirmed
identical. `WeightCursor::verify_exhausted()` fails closed if the file has too many or too few
floats for the declared topology.

### 2.4 Catalog dispatch

| `(num_layers, hidden_size)` | Rust type                                            | Alias                              |
|:--------------------------- |:---------------------------------------------------- |:---------------------------------- |
| `(1, 3)`                    | `LstmModel1<3, 4, 12>`                               | `Lstm1x3` (official example model) |
| `(1, 8/12/16/24/40)`        | `LstmModel1<H, H+1, 4H>`                             | `Lstm1x{8,12,16,24,40}`            |
| `(2, 8/12/16/24)`           | `LstmModel2<H, H+1, 2H+H, 4H>`                       | `Lstm2x{8,12,16,24}`               |
| Anything else               | `LstmModelDyn` (heap-allocated, `Vec<LstmLayerDyn>`) | —                                  |

`get_lstm_topology` (`src/loader/nam_json/topology/lstm.rs`) rejects `num_layers == 0`,
`num_layers > 16`, and `hidden_size > 1024` with `Err(JsonError::UnsupportedTopology { … })`
(DoS / degenerate guards) — see [§2.6](#26-scope-divergence-mono-only-no-zero-layer-support).

### 2.5 Native f32 backbone weights and activation precision

**Backbone Weight Precision:** NAM-rs uses native `f32` weight storage across all LSTM layers, matching NAMcore's `Eigen::MatrixXf` representation (`NAM/lstm.h:38-39`). Eliminating historical weight quantization removed GEMV dequantization overhead, reducing per-sample latency while ensuring bit-exact interop parity for models such as `BossLSTM-2x8` (ESR = 0.00e0 vs NAMcore).

**Measured Interop Results (Standard Mode):**

| Model         | ESR (vs NAMcore) | ESR (vs f64 Ideal) | SNR (dB) | MR-STFT  | Status                     |
|:------------- |:----------------:|:------------------:|:--------:|:--------:|:-------------------------- |
| BossLSTM-1×16 | 8.50e-12         | 8.90e-13           | 110.7    | 2.80e-05 | ✅ Near-bit-exact          |
| BossLSTM-2×8  | 1.00e-11         | 5.68e-13           | 110.0    | 1.57e-05 | ✅ Bit-exact / noise floor |
| LSTM Official | 7.86e-13         | 2.71e-12           | 121.0    | 3.08e-05 | ✅ Near-bit-exact          |

**Key findings:**

- **BossLSTM-2×8 Parity:** Achieves bit-exact / noise-floor convergence with NAMcore (ESR = 1.00e-11 vs NAMcore, 5.68e-13 vs f64 oracle).
- **BossLSTM-1×16 Precision:** Residual error is dominated by activation function precision at high pre-activation magnitudes, which standard exact-grade math reduces to near-zero (ESR = 8.50e-12).
- **Activation Precision Tradeoff:** `ActivationPrecision::Fast` utilizes Padé [5,4] rational `tanh` (max error ~2.32e-3) and a minimax polynomial `sigmoid` (max error ~4.09e-4) for maximum throughput. `ActivationPrecision::Standard` (universal default) runs exact-grade polynomial exp-based math (error ~2e-7), matching C++ libm precision.

### 2.6 Scope divergence: mono-only, no zero-layer support

C++ `LSTM` generalizes to arbitrary `in_channels`/`out_channels` and `num_layers == 0`
(pass-through, `lstm.cpp:139-149`). NAM-rs restricts to mono in/out and rejects degenerate
topologies at detection time via `get_lstm_topology` (`src/loader/nam_json/topology/lstm.rs`):

- **Multi-channel `in_channels`/`out_channels` not in `{1, None}`** → `Err(JsonError::UnsupportedMultiChannel { architecture: "LSTM", field, value })`.
  A hypothetical LSTM model declaring `in_channels: 2` or `out_channels: 2` is rejected at
  topology detection with an explicit `Err` — fail-closed, observable in the public loader API.
  No known real-world `.nam` LSTM model uses multi-channel; NAM is guitar/bass amp modeling,
  always mono.

- **`num_layers == 0`** → `Err(JsonError::UnsupportedTopology { architecture: "LSTM",
  issue: "num_layers=0 (no valid model can have zero layers)", limit: 0 })`. Fail-closed and
  observable on the public loader API. No `LstmModelDyn` process path is entered.

- **`num_layers > MAX_LSTM_LAYERS` (16)** and **`hidden_size > MAX_LSTM_HIDDEN_SIZE` (1024)**
  → `Err(JsonError::UnsupportedTopology { … })` with the exceeded limit. DoS/OOM guard,
  same fail-closed `Err` style as `num_layers==0`.

- **`num_layers` or `hidden_size` absent from JSON** → `Ok(None)`. The model lacks LSTM
  structural keys — not a valid LSTM config (distinct from explicit zero/overflow rejects).

Previously this section described `in_channels`/`out_channels` as silently unvalidated and
`num_layers==0` as `Ok(None)`-only. Both are closed: multi-channel and degenerate bounds
return dedicated `Err` variants; only missing keys remain `Ok(None)`.

### 2.7 Measured interop drift

LSTM is the one topology whose interop error grows with **signal length** and **host sample
rate** — the recurrent cell state accumulates error over time.

**Native f32 weights & Standard activation:** BossLSTM-2×8 converges to bit-exact / noise-floor parity with NAMcore at 48 kHz (ESR = 1.00e-11). All models default to `Standard` (exact-grade) activation precision, collapsing interop gaps to near-zero across all catalog architectures.

**F64 Oracle Floors:** The model-specific f64-oracle floors (prewarm-paired, 24k prewarm + 4096 samples)
have been fully measured in Standard mode:

- **BossLSTM-1×16**: ESR vs f64 oracle = **8.90e-13 (SNR 110.7 dB)**
- **BossLSTM-2×8**: ESR vs f64 oracle = **5.68e-13 (SNR 110.0 dB)**

Gate constants (`tests/common/constants.rs`):

| Precision mode | Host rate | ESR cap | Margin over worst measured                        |
|:-------------- |:--------- |:-------:|:------------------------------------------------- |
| Fast           | ≤ 96 kHz  | 0.08    | ~1.3× (vs 6.09e-2 @ 96 kHz)                       |
| Fast           | > 96 kHz  | 0.20    | ~1.4× (vs 1.42e-1 @ 192 kHz)                      |
| Standard       | ≤ 96 kHz  | 0.30    | ~5× the Fast cap (covers the Fast→Standard delta) |
| Standard       | > 96 kHz  | 0.60    | Conservative headroom for 192 kHz recurrent drift |

`LSTM_ESR_LIMIT = 7.0e-3` (`tests/common/constants.rs`) is derived from measured
production-vs-oracle ESR with safety margin. Live cross-validation for catalog LSTM models
(`live_cross_validation_v2_lstm_1x16` / `_2x8`) exercises four sample rates
(44.1k, 48k, 88.2k, 96k) — 192 kHz is excluded for all LSTM topologies due to
the C++ upstream limitation documented in [§2.9](#29-192-khz-limitation-lstm).
The `Standard > 96 kHz` row in the table below is a structural guard, not a
claim of 192 kHz test coverage.

### 2.8 Test coverage and fixture quality

Verified directly against `tests/models/golden_vectors.rs`, `tests/parity/cpp_parity.rs`, and
`tests/parity/reference_oracle_f64.rs`:

- **Golden vectors** (`tests/models/golden_vectors.rs`, precomputed `.bin` from the C++ `render` tool,
  always active — not `#[ignore]`d): `test_golden_vectors_lstm_1x16`, `_lstm_2x8`,
  `_lstm_official`. Backing `.nam` files are **real community-trained models**, not synthetic:
  `BossLSTM-1x16.nam` and `BossLSTM-2x8.nam` (Boss Waza Tube Amp Expander community captures,
  compatible license, committed in `tests/fixtures/models/`), plus `lstm.nam` — NAMcore's own
  official bundled example (`example_models/lstm.nam`, 1×3, matches the `Lstm1x3` alias).
- **Live cross-validation** (`tests/parity/cpp_parity.rs`, `#[ignore]`d — requires the C++ toolchain,
  run via `utils/tests-long.sh`): `live_cross_validation_{,v2_}lstm_{1x16,2x8,official,dyn}` plus
  HF-mode variants. Uses the same real fixtures as the golden vectors.
- **No synthetic LSTM fixture exists in the active suite.** There is no equivalent of WaveNet's
  `BossWN-lite.nam` (obsolete synthetic, see §3.7) for LSTM — every committed LSTM golden is
  backed by a real trained model.
- **`LstmModelDyn`** (the non-catalog fallback) is exercised by `lstm_dyn_test.nam` — a small,
  deliberately non-catalog (hidden size outside `{3,8,12,16,24,40}`) **synthetic** fixture built
  to hit the dynamic dispatch path structurally; it is not a trained amp/pedal capture. This is
  the correct use of a synthetic fixture (topology/dispatch coverage), distinct from tone
  fidelity coverage (which the real Boss captures provide for the catalog path only — the
  dynamic LSTM path currently has **no real-model coverage**, since no known community LSTM
  export uses a non-catalog hidden size).

### 2.9 192 kHz Limitation: LSTM

**Status:** FUNDAMENTAL UPSTREAM LIMITATION — not a NAM-rs bug.

The C++ NAMcore `render` tool produces **NaN**
from recurrent-state overflow when processing LSTM models at 192 kHz. The root
cause is in the third-party Eigen-based render path
(`third-party/NeuralAmpModelerCore/`, read-only vendor tree) — the LSTM cell's
recurrent state accumulates over the 5-second v2 stress signal (960,000
uncompensated samples), exerting exponent growth that saturates f32 in the
Eigen computation graph.

**Evidence (2026-08-11):**

- The C++ `render` tool crashes or emits NaN for all LSTM models at 192 kHz
  (`BossLSTM-1x16`, `BossLSTM-2x8`, and every synthetic LSTM fixture).
- No committed `golden_lstm_*_v2_192000.bin` files exist — the golden registry
  (`src/testing/catalog.rs::GOLDEN_GEN_CATALOG`, `Exclude192k` scope for
  `BossLSTM-1x16`/`BossLSTM-2x8`) intentionally skips 192 kHz; the generator
  consumes that scope as `skip_srs=192000`.
- Live cross-validation (`tests/parity/cpp_parity.rs`) excludes 192 kHz via
  `v2_multi_sr_expected_rates()` → `V2MultiSRScope::Exclude192k`
  (`tests/common/io_helpers.rs:137-139`).
- Golden vector tests (`tests/models/golden_vectors.rs`) exclude 192 kHz via
  `v2_sample_rates_for()` → `V2_EX_192K_SAMPLE_RATES`
  (`src/testing/catalog.rs`).

**Governance (binding):**

1. **No golden vector shall be generated or regenerated at 192 kHz for LSTM**
   unless a new C++ upstream release fixes the root cause and a committed
   `.golden.bin` is produced from that release. Self-referencing Rust
   regeneration without C++ adjudication is prohibited (invariant per §4.5.1).
2. **No ESR/SNR gate shall be claimed at 192 kHz for LSTM.** The
   `ABSOLUTE_ESR_CAP_LSTM_HIRATE_HF` constant (`1e-4`, `cpp_parity.rs:468`)
   exists as a structural guard for the code path but is never exercised
   (dead code under current catalog scope).
3. **Every LSTM model** — catalogued (`BossLSTM-1x16`, `BossLSTM-2x8`) and
   synthetic (`lstm_1x10`, `lstm_2x24`, `lstm_3x8`, `lstm_dyn_test`) — is
   formally registered as supported at **`[44100, 48000, 88200, 96000]`**.
   `lstm.nam` (Official) and `lstm_dyn_test.nam` are additionally restricted
   to 48 kHz by the `condition_size`/`expected_sample_rate` contract of their
   `.nam` metadata — this is a *model* constraint, not an *architecture*
   constraint.
4. **`REVIEW_REQUIRED`** if any test path produces a passing result at
   192 kHz — such a result contradicts the documented upstream limitation
   and must be human-investigated before acceptance.

**Affected code locations:**

| Layer                | File                                                          | Mechanism                                                                 |
|:-------------------- |:------------------------------------------------------------- |:------------------------------------------------------------------------- |
| Golden registry      | `src/testing/catalog.rs` `GOLDEN_GEN_CATALOG`                 | `V2GenScope::Exclude192k` for LSTM entries (emitted as `skip_srs=192000`) |
| Golden vector tests  | `tests/models/golden_vectors.rs` via `src/testing/catalog.rs` | `v2_sample_rates_for()` → `V2_EX_192K_SAMPLE_RATES`                       |
| Live C++ parity      | `tests/common/io_helpers.rs` `v2_multi_sr_expected_rates`     | `V2MultiSRScope::Exclude192k` for all LSTM filenames                      |
| Long suite preflight | `catalog_preflight` (`cargo test --test models`)              | `validate_v2_catalog()` — Rust V2 gate (bash V2_CATALOG_SCOPE removed)    |

---

## 3. WaveNet A1 Architecture

Read against `NAM/wavenet/model.h`, `NAM/wavenet/model.cpp`, `NAM/wavenet/detail.h`, `NAM/activations.{h,cpp}`,
and the corresponding Rust modules (`src/models/wavenet/`, `src/loader/dispatcher/wavenet/`, `src/loader/nam_json/topology/wavenet.rs`).

### 3.1 C++ Reference Architecture vs. Rust Const-Generic Catalog

The vendored C++ source does not define separate SKU classes for different channel counts. `nam::wavenet::create_config` (`model.cpp:1227-1241`) has exactly two branches: (1) `a2_fast::is_a2_shape()` for the A2 fast path (§4), and (2) a single generic Eigen-based implementation (`detail::Layer` / `detail::LayerArray` / `WaveNet`) for all other WaveNet models regardless of channel count.

"Standard/Lite/Feather/Nano" (16/12/8/4 channels) are a **NAM-rs-side performance
optimization**: these four channel counts happen to cover the overwhelming majority of
real-world community WaveNet exports, so NAM-rs const-generic-specializes them
(`WaveNetModel<CH, K, HEAD>`) for SIMD throughput, falling back to a heap-allocated
`WaveNetModelDyn` for everything else. This is a legitimate and effective engineering strategy,
but it is **NAM-rs's own catalog, not a mirror of any C++-side concept** — the correct framing is
"const-generic fast path vs. generic fallback for the single C++ generic WaveNet class," not
"C++ SKU X maps to Rust SKU X."

### 3.2 Rust implementation

| C++ (`NeuralAmpModelerCore/`)                                                                                                                                      | Rust (`src/`)                                                                                                                                     | Verdict                                                                                                                                                                                          |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `detail::LayerArray::ProcessInner` — rechannel → layer loop → head accumulation → head rechannel (`model.cpp:450-511`)                                             | `WaveNetLayerArray::process_block_internal` (`layer_array.rs`)                                                                                    | ✅ Match — same rechannel → layer cascade → head-accumulate → head-rechannel structure                                                                                                           |
| `detail::Layer::Process` — dilated conv + input mixin, sum, activation, optional layer1x1/head1x1 residual+skip (`model.cpp:166-376`)                              | `WaveNetLayer::process_block_internal` (`layer.rs`)                                                                                               | ✅ Match for the **plain case** (no gating, no FiLM, no head1x1/layer1x1 variance) — the only case the const-generic fast path implements (§3.6)                                                 |
| `WaveNet::process` — condition → layer arrays → head_scale (`model.cpp:744-832`)                                                                                   | `WaveNetModel::process` / `WaveNetModelDyn::process` (`model.rs`, `model_dyn.rs`)                                                                 | ✅ Match for the no-condition_dsp, no-post-stack-head case                                                                                                                                       |
| Default activation: exact `tanh`/`sigmoid` (`Activation::using_fast_tanh = false`, never flipped by the `render` tool — same as LSTM, §2.1)                        | `ActivationPrecision::Fast` uses Padé[5,4] tanh / minimax-17 sigmoid (opt-in); `Standard` is exact-grade polynomial exp-based (universal default) | ⚠ Same intentional, bounded divergence documented for LSTM (§2.5); applies identically when `Fast` is active                                                                                     |
| `LayerArrayParams::get_receptive_field()` — per-array RF, **summed** across all arrays plus condition_dsp's own prewarm (`model.cpp:417-424`, `model.cpp:616-618`) | `WaveNetModel`/`WaveNetModelDyn` prewarm fill                                                                                                     | ✅ `prewarm_samples()` correctly sums all arrays' RFs + condition_dsp + post-stack head, matching C++ (§3.5 FIXED). `prewarm(&mut self, _)` discards arg and runs analytical fill — intentional. |

### 3.3 Weight loading

Layout (`rechannel → [conv1d+bias, input_mixin, layer1x1+bias]×N → head_rechannel+bias →
head_scale`, repeated per array) is read identically by
`src/loader/dispatcher/wavenet/{standard,lite,feather,nano}.rs` (catalog) and
`src/loader/dispatcher/wavenet/dynamic.rs` (free geometry) against C++'s `WaveNet::set_weights_`
/ `LayerArray::set_weights_` / `Layer::set_weights_` (`model.cpp:135-164, 525-531, 623-645`).
Confirmed identical read order. `WeightCursor::verify_exhausted()` fails closed on
under/over-provisioned weight files.

### 3.4 Catalog dispatch

`get_wavenet_topology` (`src/loader/nam_json/topology/wavenet.rs`) requires **exactly 2 layer
arrays**, `condition_size ≤ 1`, no gating on either array, and an exact dilation-pattern match to
classify a model as a catalog SKU:

| Channels                                                                                         | Dilation pattern (both arrays)                       | Rust type                |
|:------------------------------------------------------------------------------------------------ |:---------------------------------------------------- |:------------------------ |
| 16                                                                                               | `[1,2,4,...,512]` (both arrays, "Standard" pattern)  | `WaveNetModel<16, 3, 8>` |
| 12                                                                                               | `[1,...,64]` then `[128,...,512,1,...,512]` ("Lite") | `WaveNetModel<12, 3, 6>` |
| 8                                                                                                | same Lite-shaped dilation pattern, CH=8 ("Feather")  | `WaveNetModel<8, 3, 4>`  |
| 4                                                                                                | same Lite-shaped dilation pattern, CH=4 ("Nano")     | `WaveNetModel<4, 3, 2>`  |
| Anything else (any channel count, any array count, gated, `condition_size > 1`, post-stack head) | `WaveNetModelDyn`                                    |                          |

This is a NAM-rs-internal classification only — see §3.1. A `condition_dsp` sub-model is **not**
checked at all during catalog matching (see §3.6): a hypothetical 2-array, ungated, CH=16,
Standard-dilation model that also declares a `condition_dsp` JSON key would still match `Known(Standard)`
and be routed to the fast path, which has **zero `condition_dsp` handling** — it would be silently
dropped. No known real `.nam` model exhibits this combination.

**Activations:** `validate_layer_activations` (`static_factory.rs`) accepts both `Tanh` (C++ default) and `ReLU` for A1 models. `ReLU`-configured A1 models route through the generic `WaveNetModelDyn` path with ReLU applied identically to C++ semantics. The `mock_a2.nam` negative fixture uses `ReLU` config explicitly — but that is a zero-weight model tested only for `Err` rejection, not inference parity.

### 3.5 Prewarm: an elegant analytical shortcut

C++'s `DSP::prewarm()` is generic and iterative: it computes `mPrewarmSamples` once at
construction (`condition_dsp`'s own prewarm requirement **plus** the **sum** of every layer
array's receptive field, plus any post-stack head's receptive field, `model.cpp:615-620`), then
literally calls `process()` with zero input that many times (`dsp.cpp:67-101`).

NAM-rs's `WaveNetModel`/`WaveNetModelDyn::prewarm()` does **not** iterate. Because a purely
feedforward causal-conv stack driven by a constant input eventually converges to a constant
output at every layer, `prewarm_internal()` computes that fixed point **analytically in one pass**:
process a single zero-input frame through the rechannel (memoryless, exact for a single frame),
then for each layer in order, replicate that single already-correct value across the entire
history buffer (`copy_within`, `layer_array.rs:97-118`) *before* computing that layer's own
single-frame output — which is therefore also exactly the converged constant, propagating
correctness layer by layer. **This is a genuinely correct, elegant O(layers) alternative to
C++'s O(receptive_field) iteration for the plain feedforward case** — verified by working through
the recursion by hand. It is also why `prewarm_samples()`'s return value has zero
effect on WaveNet's own audio correctness: the trait's `prewarm(&mut self, _num_samples)` override
discards the argument entirely and always runs the full analytical fill regardless of what number
is passed in (`mod.rs:79-83, 107-109`).

- **`prewarm_samples()` correctly sums multi-array RFs (FIXED).** `WaveNetModel::prewarm_samples()`
  returns `array1.receptive_field_size + array2.receptive_field_size` — the sum of both arrays'
  RFs (`src/models/wavenet/mod.rs:85-87`). `WaveNetModelDyn::prewarm_samples()` returns
  `sum(arrays) + condition_dsp.prewarm_samples() + post_stack_head.receptive_field() - 1`
  (`src/models/wavenet/mod.rs:111-119`), matching C++'s canonical sum-of-all-components formula
  (`model.cpp:615-620`). **Note:** `prewarm(&mut self, _)` still discards the argument and always
  runs the full analytical fill — this is intentional and correct for WaveNet's feedforward
  structure, as the analytical shortcut (§3.5.1) is mathematically sound regardless of the number
  passed in.
- **`condition_dsp.prewarm` handling:** `WaveNetModelDyn::prewarm_internal()` invokes
  `cond_dsp.prewarm(cond_dsp.prewarm_samples())` (`model_dyn.rs`) for **supported** nested
  condition DSPs (WaveNet sub-models), matching C++'s `GetPrewarmSamples()` settle semantics.
  **LSTM-as-`condition_dsp` is not a supported production path** — public load rejects it
  fail-closed (`wavenet_condition_lstm.nam`; §3.9.4 / §7.4). Any residual code comments that
  mention “e.g. LSTM condition DSPs” describe historical/oracle exploration only, not the
  public contract.

### 3.6 Generic gating/FiLM/head1x1/layer1x1 — FIXED (fail-closed, 2026-08-10)

**Status:** FIXED — the gap described in previous revisions of this document (A1 `Free`/`Dynamic`
path silently processing gated/FiLM/head1x1/layer1x1 WaveNet models as if unconfigured) has been
closed. NAM-rs now **rejects** all such models at topology detection, fail-closed, before any
inference dispatch.

**Mechanism:** `get_wavenet_topology` (`src/loader/nam_json/topology/wavenet.rs`) iterates over
every layer config of every layer array and returns `WavenetTopologyResult::Rejected(...)` for
any of:

- `gated: true` (legacy boolean)
- `gating_mode` set to `"gated"` or `"blended"` (string-valued)
- `head1x1` active
- `layer1x1` active
- Any FiLM parameter object present
- `secondary_activation` set to a non-trivial value

Each rejection message follows the canonical pattern `"A2 feature not supported in WaveNet A1"`,
making the failure observable in logs and in the `Err` variant returned to the public loader API.
The guard covers **all** A1 paths — both catalog-SKU matching and the free-geometry
`WaveNetModelDyn` branch — so no model with A2-only topology features can reach inference.

**Regression tests** (`src/loader/nam_json_test.rs`):

- `test_wavenet_a1_rejects_gated_true`
- `test_wavenet_a1_rejects_gating_mode_non_none`
- `test_wavenet_a1_rejects_head1x1_active`
- `test_wavenet_a1_rejects_layer1x1_active`
- `test_wavenet_a1_rejects_film_active`
- `test_wavenet_a1_rejects_secondary_activation`

Each test constructs a minimal JSON config with the offending feature and asserts that topology
detection returns `WavenetTopologyResult::Rejected`. All six are active (not `#[ignore]`) and run
in the quick suite.

### 3.7 Test coverage and fixture quality

Verified directly against `tests/models/golden_vectors.rs`, `tests/parity/cpp_parity.rs`, and
`tests/parity/reference_oracle_f64.rs`:

- **Golden vectors, catalog SKUs — all real community models, all active (not ignored):**
  `test_golden_vectors_wavenet_{standard,feather,nano}` use `BossWN-{standard,feather,nano}.nam`
  (Boss Waza Tube Amp Expander community captures, compatible license). `test_golden_vectors_wavenet_lite`
  uses `EVH-5150-Lite.nam`, a **non-distributable real community capture** (gitignored, lives in
  `tests/fixtures/models-nondist/`, fetched separately — see [`docs/fixtures.md`](fixtures.md)
  §Non-Distributable Model Management). Its doc comment (`golden_vectors.rs:495-511`) and the
  test itself confirm the measured result **directly from current source**: SNR = 122.3 dB, ESR
  = 5.84e-13, thresholds SNR ≥ 105 dB / ESR ≤ 3.5e-11 — this specific figure is **independently
  re-confirmed in this audit pass**, not carried over.

- **Golden vectors, non-catalog:** `test_golden_vectors_wavenet_dyn` (`wavenet_dyn_free.nam`) and
  `test_golden_vectors_wavenet_condition_dsp` (`wavenet_condition_dsp.nam`) are **synthetic**,
  purpose-built to exercise `WaveNetModelDyn`'s free-geometry and `condition_dsp` dispatch paths
  structurally. `test_golden_vectors_wavenet_official` uses `wavenet_official.nam` — NAMcore's own
  bundled official example (`example_models/wavenet.nam`, CH=3, 2 arrays), a small but **real,
  officially-distributed** reference model, not synthetic.

- **Obsolete synthetic fixture, kept for traceability only:** `BossWN-lite.nam` (CH=12,
  artificially generated) is explicitly marked obsolete in [`docs/fixtures.md`](fixtures.md) — "no
  longer used in active tests," superseded by `EVH-5150-Lite.nam`. It is the historical source of
  the "SNR ≈ 0.9 dB" figure that `docs/testing.md` still (incorrectly) attributes to the current
  active test.

- **Live cross-validation** (`tests/parity/cpp_parity.rs`, `#[ignore]`d, requires C++ toolchain):
  `live_cross_validation_{,v2_}wavenet_{standard,feather,nano,lite,a1_standard,dyn}` plus HF
  variants, using the same real fixtures above. `live_cross_validation_nondist_models` and the
  named `live_cross_validation_v2_{app_evh,boss_bd2,slammin_marshall}` tests exercise **three
  additional real, non-distributable community captures** (`APP-EVH-Stealth100-Dialled-xSTD.nam`,
  `Boss BD-2 H2O Mod T-12_00 G-12_00.nam`, `SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam` —
  catalogued with SHA-256 + author attribution in `tests/fixtures/models-nondist/manifest.json`),
  covering custom-layer WaveNet and `SlimmableContainer` topologies beyond the four catalog SKUs.
  These gracefully no-op (printed `SKIP`, not a failure) when the non-distributable directory or
  the C++ toolchain is absent, consistent with `docs/testing.md`'s documented graceful-skip
  policy.

- **Net assessment:** WaveNet A1's test fixture quality is materially better than a synthetic-only
  suite would suggest — every catalog SKU and several custom real-world topologies are validated
  against genuine trained amp/pedal captures, not just structurally-synthetic weights. The
  remaining synthetic fixtures (`wavenet_dyn_free.nam`, `wavenet_condition_dsp.nam`) are correctly
  scoped to structural/dispatch coverage rather than claiming tone-fidelity validation.

### 3.8 Measured interop drift

Canonical golden-vector interop fidelity measured against NAMcore (`docs/quality-contract.json` baseline @ 48 kHz):

| Model                   | ESR (vs NAMcore) | ESR (vs f64 Ideal) | SNR (dB) | MR-STFT  | Mode |
|:----------------------- |:----------------:|:------------------:|:--------:|:--------:|:---- |
| WaveNet Standard (CH16) | 2.31e-14         | 9.05e-15           | 136.4    | 6.46e-06 | Live |
| WaveNet Feather (CH8)   | 4.74e-14         | 2.00e-14           | 133.2    | 8.86e-06 | Live |
| WaveNet Nano (CH4)      | 6.43e-14         | 3.05e-14           | 131.9    | 7.67e-06 | Live |
| EVH-5150-Lite (CH12)    | 7.87e-13         | 2.64e-13           | 121.0    | 4.31e-06 | Live |
| WaveNet A1 Standard     | 1.20e-13         | 1.05e-13           | 129.2    | 2.26e-06 | Live |
| WaveNet Official (CH3)  | 9.03e-14         | 6.13e-14           | 130.4    | 1.66e-05 | Live |

All WaveNet A1 catalog models pass their calibrated quality gates with multi-order-of-magnitude safety margins.

### 3.9 `condition_dsp` specification (canonical semantics)

> This section is the formal specification. It was derived by
> reading the C++ reference, the Python trainer, and the Rust production code side-by-side
> on 2026-07-14. All file:line citations reference NAMcore v0.5.4 (tag `1f42f88`).

#### 3.9.1 C++ semantics — `WaveNet::_process_condition` and sizing

**Source:** `third-party/NeuralAmpModelerCore/NAM/wavenet/model.cpp`

The `condition` matrix flowing through the WaveNet layer cascade is `_condition_output`
(`Eigen::MatrixXf`, `model.h:76`). Its dimensions are decided in `SetMaxBufferSize`
(`model.cpp:647-687`):

- **Without `condition_dsp`** (`model.cpp:652-654`): `_condition_output` is resized to
  `[_get_condition_dim(), maxBufferSize]`. `_get_condition_dim()` returns
  `NumInputChannels()` (`model.h:106`), which is always **1** for WaveNet (mono-in).
  So `_condition_output` = `[1 × maxBufferSize]` — a single row holding the raw input.

- **With `condition_dsp`** (`model.cpp:656-660`): `_condition_output` is resized to
  `[condition_dsp->NumOutputChannels(), maxBufferSize]`. The number of **rows** in the
  condition matrix is the condition DSP's output channel count, **not** the WaveNet's
  `in_channels`.

The `_process_condition` method (`model.cpp:699-729`) fills `_condition_output`:

- **Without `condition_dsp`** (`model.cpp:703-704`): copies `_condition_input.leftCols(num_frames)`
  into `_condition_output.leftCols(num_frames)`. Both are `[1 × num_frames]`.

- **With `condition_dsp`** (`model.cpp:710-728`):

  1. Input (`_condition_input`, shape `[condition_dim, num_frames]`, where
     `condition_dim = _get_condition_dim() = 1`) is copied row-by-row into
     pre-allocated contiguous DSP buffers (`model.cpp:710-715`).
  2. The condition DSP processes these buffers in-place (`model.cpp:718-719`).
  3. Output is copied back row-by-row from the DSP output buffers into
     `_condition_output` (`model.cpp:722-727`). The row count is
     `condition_dsp->NumOutputChannels()` — there is **no** broadcast, tile, or
     dimension coercion. Output channels are written 1:1.

**Construction-time validation** (`model.cpp:592-601`): when `condition_dsp` exists, C++
asserts that **every** layer array's `condition_size` matches
`condition_dsp->NumOutputChannels()` exactly — and throws `std::runtime_error` on mismatch:

```cpp
// model.cpp:594-601
if (layer_array_params[i].condition_size != this->_condition_dsp->NumOutputChannels())
{
    std::stringstream ss;
    ss << "condition_size of layer " << i << " ("
       << layer_array_params[i].condition_size
       << ") doesn't match output channels of condition DSP ("
       << this->_condition_dsp->NumOutputChannels() << "!\n";
    throw std::runtime_error(ss.str().c_str());
}
```

**Conclusion:** In C++, a model where `condition_dsp->NumOutputChannels() < condition_size`
is **rejected at construction**. The case `out_channels == condition_size` is guaranteed
by this check. There is no broadcast logic — dimensional matching is enforced structurally.

#### 3.9.2 How `_condition_output` is consumed by layers

In `WaveNet::process` (`model.cpp:744-832`):

- `_condition_output` (the multi-row condition matrix) is passed **as-is** to every
  layer array's `Process` method (`model.cpp:761,770`), alongside the layer inputs.
- Inside each `Layer::Process` (`model.cpp:166+`), the condition is consumed by
  `InputMixer` — a `Conv1D(kernel=1, in_channels=condition_size, out_channels=mid_channels)`
  — which projects `condition_size` channels to `mid_channels` (= `2*bottleneck` for
  gated, `bottleneck` for plain). FiLM modules also consume the condition channels
  directly.
- **No broadcasting between `_condition_output` and the layer internals.** The matrix
  already has the correct row count by construction (§3.9.1).

#### 3.9.3 Python trainer semantics (`neural-amp-modeler`, v0.13.0)

**Source:** `nam/models/wavenet/_wavenet.py` (tag `v0.13.0`)

The trainer's `WaveNet.parse_config` (`_wavenet.py:142-155`) handles `condition_dsp`:

```python
if condition_dsp_config.get("name") != "WaveNet":
    raise NotImplementedError("Only WaveNet condition DSP is supported")
condition_dsp = WaveNet.init_from_config(condition_dsp_config["config"])
```

- The Python trainer **only supports WaveNet as `condition_dsp`** — any other
  architecture (including LSTM) raises `NotImplementedError`.

- During training (`forward`, `_wavenet.py:189`):

  ```python
  c = x if self._condition_dsp is None else self._condition_dsp(x)
  ```

  The condition tensor `c` has shape `[B, condition_dsp_head_out_channels, L]` — exactly
  the head output of the condition-dsp WaveNet.

- Export (`export_config`, `_wavenet.py:176-195`): the `condition_dsp` sub-model is
  serialized as a complete `.nam` JSON object embedded inside the parent model's config
  under the `"condition_dsp"` key. The note at line 192 reads:

  ```python
  # Build condition_dsp export dict without running forward (condition_dsp
  # may have multiple output channels; WaveNet wrapper asserts 1 channel).
  ```

**Conclusion:** The Python trainer's `condition_dsp` output channels match the
`condition_size` of the parent's layer arrays by the trainer's own structural design
(the condition-dsp WaveNet's `head.out_channels` = layer arrays' `condition_size`).
There is **no** code path in the official trainer that produces a `condition_dsp`
output-channel-count mismatch — it would fail dimension checks during forward
computation.

#### 3.9.4 LSTM as `condition_dsp` — veredicto

The `wavenet_condition_lstm.nam` fixture (LSTM sub-model inside a WaveNet) represents
a configuration that:

1. **The Python trainer cannot produce** — raises `NotImplementedError` for non-WaveNet
   `condition_dsp` (§3.9.3).
2. **C++ NAMcore would reject at construction** — the LSTM's `NumOutputChannels() = 1`
   would fail the assertion `layer_array.condition_size == condition_dsp->NumOutputChannels()`
   when `condition_size = 3` (the standard WaveNet case) (§3.9.1).
3. **The C++ `render` tool does not support** this model — there is **no golden vector**
   generated by NAMcore for this fixture. The only committed golden for this model is the f64 oracle — which itself has a disputed `condition_dsp` semantic
   (the broadcast logic at `wavenet.rs:38-49` and `a2/dynamic_eval.rs:326-339`). This creates a circular
   dependency: the oracle's correctness for this fixture cannot be independently validated.

Rust production code behavior (`src/models/wavenet/model_dyn.rs:236-251`): when
`condition_dsp` output channels (`dsp_ch`) are fewer than the layer array's
`condition_size` (`cond`), the production engine **broadcasts** the first channel's
value across all condition channels. This broadcast is present in both the A1
dynamic path (`model_dyn.rs:240-247`) and the A2 dynamic/cascade paths (via the
same `condition_dsp_output` buffer). This is a **NAM-rs-specific behavior** with
no C++ precedent — it exists because NAM-rs loads models the upstream toolchain
rejects.

**Recommendation for T1.2 (oracle fix):** The f64 oracle's broadcast logic
(`wavenet.rs:38-49`, `a2/dynamic_eval.rs:326-339`) should match the production code's broadcast
**if** the production broadcast is deemed the intended semantics for models the
upstream toolchain cannot validate. Since the C++ golden cannot serve as arbiter
for the LSTM case, the "correct" broadcast behavior is a product decision documented
here — not a parity claim.

#### 3.9.5 Summary: canonical `condition_dsp` semantics

| Aspect                                 | C++ (NAMcore v0.5.4)                                                        | Python trainer (v0.13.0)                               | Rust production (NAM-rs)                                     |
|:-------------------------------------- |:--------------------------------------------------------------------------- |:------------------------------------------------------ |:------------------------------------------------------------ |
| `condition_dsp` matrix rows            | `condition_dsp->NumOutputChannels()`                                        | `condition_dsp.head.out_channels`                      | `condition_dsp.num_output_channels()`                        |
| Dimension enforcement                  | Hard assertion: `condition_size == NumOutputChannels()` (throw on mismatch) | Structural match (fails dimension check in forward)    | `assert` on max channels; broadcasts when `dsp_ch < cond`    |
| Broadcasting (dsp_ch < cond)           | **None** — construction rejected                                            | **None** — structural match prevents mismatch          | **Yes** — replicates channel 0 across all condition channels |
| Supported condition_dsp architectures  | Only WaveNet (and LSTM — but see §3.9.4 re: assertion rejection)            | Only WaveNet (raises `NotImplementedError` for others) | Any (LSTM accepted; see §3.9.4)                              |
| Reference for `condition_lstm` fixture | N/A — model rejected                                                        | N/A — model cannot be produced                         | Golden from `wavenet_condition_dsp.nam` (WaveNet sub-model)  |
| Key file:line references               | `model.cpp:592-601,652-660,699-729,744-770`                                 | `_wavenet.py:142-155,171-195`                          | `model_dyn.rs:236-251`, `model_dyn.rs:357-373`               |

#### 3.9.6 T6.1 Root Cause: `head_scale` read from JSON config instead of weight stream (WaveNet A1 oracle)

**Status:** FIXED (2026-07-14, T6.1).

**Root cause:** Both the Rust oracle (`src/testing/reference_oracle/wavenet.rs:33`)
and the Python anchor generator (`tests/fixtures/scripts/validate_oracle_f64.py:93`)
read `head_scale` from the JSON config field `model_data.config.head_scale`, not from
the last position of the weight stream — where production engines (Rust `build_wavenet_dynamic_inner`,
C++ `NAM/wavenet/model.cpp`) always read it.

For models generated by standard NAM trainers, the config `head_scale` and the
weight-stream `head_scale` are the same value, so the bug is hidden. For
test-script-generated models (`create_wavenet.py`), the weight stream may contain
random values that overwrite the config metadata. This caused the f64 oracle to
produce structurally wrong output for `wavenet_condition_dsp.nam`:

**`wavenet_condition_dsp.nam` — before vs. after T6.1:**

| Measurement                               | Before T6.1 (broken oracle) | After T6.1 (fixed oracle) |
|:----------------------------------------- |:--------------------------- |:------------------------- |
| Prod × Oracle ESR (paired, summary table) | 4.23e+01 (+16.3 dB)         | 6.33e-15 (−142.0 dB)      |
| Oracle × NumPy anchor ESR                 | N/A (circular: 4.96e-16)    | 3.18e-32 (−315.0 dB)      |
| Quality Dashboard tag                     | `[orac: f64 div]` TRIGGERED | **not triggered**         |
| Prod output (first 10)                    | ≈ +0.17 growing             | ≈ +0.17 growing           |
| Oracle output (first 10)                  | ≈ −0.033 flat               | ≈ +0.17 growing (matches) |

**What was wrong:** The main model's weight-stream head_scale = −0.1255 (the actual
weight at position 146), but the oracle used config head_scale = 0.02 — a sign inversion
and 6.27× magnitude error. The condition_dsp sub-model's weight-stream head_scale =
0.8649, but the oracle used 0.02 again — a 43.25× magnitude error. The combination of
both mismatches produced oracle output that was structurally unrelated to production.

**Fix (2 files, 2 languages):**

1. **Rust oracle** (`src/testing/reference_oracle/wavenet.rs:161-167`): After reading
   all array weights, read the last remaining weight from the cursor as `head_scale`.
   The config field is no longer used for computation.

2. **Python anchor generator** (`tests/fixtures/scripts/validate_oracle_f64.py:195-199`):
   Same fix — read `head_scale` from `weights[cursor]` after the per-array weight
   loop. The config field is no longer used for computation.

**Verification:** `test_summary_table` now shows ESR(WaveNetCondDSP) = 6.33e-15 (−142.0 dB),
matching the near-bit-exact floor of the WaveNet A1 family (1e-14 to 1e-12 range).
The regenerated Python anchor matches the Rust oracle at 3.18e-32 ESR — both now
read head_scale from the same weight-stream position and agree with the production
engine (itself golden-C++-confirmed at ESR 1.11e-14).

**Status after T6.1:**

- ✅ `wavenet_condition_dsp.nam` — oracle verified against production at the A1 floor
- ✅ Python anchor regenerated and validated against production, NOT circularly
- ✅ Quality Dashboard `[orac: f64 div]` tag eliminated for this model
- ✅ `docs/cpp_parity_map.md` §3.9 now records the definitive root cause

---

## 4. WaveNet A2 Architecture

> Read against `NAM/wavenet/a2_fast.{h,cpp}` (the C++ fast-path, in full) and cross-checked
> `src/loader/nam_json/topology/a2.rs`, `src/models/a2/model/static/process.rs`,
> `tests/models/golden_vectors.rs`, `tests/parity/cpp_parity.rs`, `tests/common/validation.rs`, and
> [`docs/fixtures.md`](fixtures.md) against each other and against current git history.
> §4.4–§4.6 (the `wavenet_a2_max.nam` investigation) were already
> established in the previous pass and are corroborated, not re-derived, here.

"A2" designates the newer WaveNet variant: `a2_fast.cpp` is C++'s **optimized, shape-restricted**
fast path (exactly 23 layers, fixed kernel/dilation pattern, CH∈{3,8}, LeakyReLU-only, no
gating/FiLM/head1x1 — `a2_fast.cpp:754-885`); anything not matching that exact shape falls
through to the same generic `NAM/wavenet/model.cpp` used by A1 (§3.1). NAM-rs mirrors this split
faithfully: `WaveNetA2<3>`/`WaveNetA2<8>` (fast path) vs. `WaveNetA2Dyn`/`WaveNetA2Cascade`
(everything else — FiLM, gating, blending, `condition_dsp`, multi-array cascade, `head1x1`).

### 4.1 Fast-path shape detection: a faithful, self-correcting mirror of C++

`src/loader/nam_json/topology/a2.rs::is_a2_shape` was read line-by-line against
`a2_fast.cpp::is_a2_shape` (lines 754-885) — every one of the 20 structural checks (layer count,
no post-stack head, `in_channels`/`input_size`/`condition_size`, `channels == bottleneck`,
`channels ∈ {3,8}`, exact kernel-size/dilation arrays, LeakyReLU(0.01) activation, gating,
head1x1, layer1x1 groups, layer-array head shape, all 8 FiLM slots, `groups_input*`,
non-slimmable) has a corresponding Rust check, in the same order, with a comment citing the C++
line number. This is the best-audited topology detector in the codebase.

**Notably, the code contains its own documented self-correction** (`topology/a2.rs:207-213`,
tagged `B.1.1 (F5)`): an earlier version of the Rust dispatcher apparently routed FiLM-active
models to the fast path anyway, producing measured divergence (CH=3 SNR 18.1 dB, CH=8 SNR 36.0
dB) against C++ — because C++'s `is_a2_shape` rejects any active FiLM slot and falls through to
the generic Eigen WaveNet, which the Rust fast path does not reproduce. The fix (already in the
current source) routes any model with FiLM, gating, `head1x1`, or non-1 groups to
`A2TopologyResult::Dynamic` instead, matching C++'s fallback exactly. This is now correct — but
the 18.1/36.0 dB figures remain the calibrated thresholds for the *dynamic* engine's FiLM
emulation itself (§4.2), since matching the shape-routing decision doesn't yet mean matching the
generic Eigen path's output bit-for-bit.

### 4.2 Fast path (A2-Full CH=8 / A2-Lite CH=3): structurally correct, only synthetic fixtures

`src/models/a2/model/static/process.rs` was read against `a2_fast.cpp`'s `A2FastModel<Channels>`:
rechannel → per-layer (dilated conv → bias → input mixin → LeakyReLU(0.01) → head-accumulate →
`layer1x1` residual) → head Conv1D(k=16, bias, `head_scale`). Structurally identical, including
weight-stream read order (`_load_weights`, `a2_fast.cpp:198-273`) matching
`src/models/a2/model/set_weights.rs` field-for-field.

Re-measured 2026-07-11 (`utils/tests-quick.sh` Phase 2, release): A2-Full ESR = 1.12e-13
(SNR 129.5 dB), A2-Lite ESR = 6.43e-14 (SNR 131.9 dB) against the committed NAMcore
goldens — both pass their calibrated gates (3.0e-11 / 3.5e-11, SNR ≥ 105 dB) with
2+ orders of magnitude of margin.

**Fixture quality caveat (new finding — see §4.6):** unlike LSTM and WaveNet A1, these figures are
**not** validated against a real trained community model. `wavenet_a2_full.nam` and
`wavenet_a2_lite.nam` are explicitly documented as **synthetic, calibrated weights** — [`docs/fixtures.md`](fixtures.md):
"Synthetic, NOT official FiLM models." There is currently no known real-world A2-Full
or A2-Lite `.nam` export in the test suite. The fast-path *code* is well-verified against C++
structurally; the fast-path *fixtures* only prove self-consistency of calibrated weights, not
tone-fidelity on a genuine trained model.

### 4.3 Measured interop drift on dynamic paths (Gating, Blending, FiLM)

Measured interop metrics for WaveNet A2 dynamic paths (`docs/quality-contract.json` baseline @ 48 kHz):

| Model / Variant                     | ESR (vs NAMcore) | ESR (vs f64 Ideal) | SNR (dB) | MR-STFT  | Mode |
|:----------------------------------- |:----------------:|:------------------:|:--------:|:--------:|:---- |
| WaveNet A2-Full (CH8)               | 1.46e-13         | 7.83e-14           | 128.3    | 1.68e-05 | Live |
| WaveNet A2-Lite (CH3)               | 8.36e-14         | 1.82e-14           | 130.8    | 9.54e-06 | Live |
| WaveNet A2-FiLM-Full (CH8)          | 1.18e-14         | 8.75e-15           | 139.3    | 7.85e-06 | Live |
| WaveNet A2-FiLM-Lite (CH3)          | 3.82e-13         | 1.61e-13           | 124.2    | 1.69e-05 | Live |
| WaveNet A2-FiLM-InputMixinPre (CH3) | 3.44e-14         | 2.21e-14           | 134.6    | 6.92e-06 | Live |
| WaveNet A2-FiLM Chaos Stress (CH3)  | 1.26e-14         | 1.03e-14           | 139.0    | 7.00e-06 | Live |
| WaveNet A2 Dynamic Gated (CH8)      | 5.03e-11         | 1.00e-10           | 103.0    | 6.63e-05 | Live |
| WaveNet A2 Dynamic Blended (CH3)    | 5.35e-14         | 2.65e-14           | 132.7    | 9.97e-06 | Live |

All dynamic path variants achieve near-bit-exact parity or expected approximation-bounded floors.

### 4.4 🔴 Known bug KB-A2-MAX: `wavenet_a2_max.nam` (Official Flagship)

**Status: PERMANENT KNOWN BUG** until reopening criteria in **§4.4.3**. Not scheduled for speculative iterations.

The fail-closed dispatch guard (`reject_wavenet_a2_max_class`, TR1.1) rejects `build_model` with `Err` citing **KB-A2-MAX**. No production f32 instance of this topology enters the public hot path.

**Authoritative metrics (HEAD H1+H2 tree, 2026-08-09):**

| Pair                  | Metric                           | Notes                                        |
|:--------------------- |:-------------------------------- |:-------------------------------------------- |
| prod f32 × C++ golden | **SNR = 0.23 dB**, ESR ≈ 9.49e-1 | `test_measure_a2_max_snr_vs_golden` + unlock |
| prod f32 × f64 oracle | ESR ≈ 10³–10⁴                    | paired FAILED — **prod ≉ f64**               |
| f64 × C++ golden      | ESR ≈ 1.0                        | H0 **Case D** — oracle also ≉ C++            |
| f64 × F32-sim (self)  | ~−134 dB                         | internal consistency only                    |

Historical: pre-R3 baseline 1.35 dB; H1-only peak 2.31 dB; H1+H2 tree **0.23 dB**. Weight budget **818 + 1052** exact. Neighbors green (A2-Full ~128 dB, condition_dsp ~139 dB, A2 matrix 103–140 dB).

**Investigation closed as residual work:** H1–H4 exhausted as dominant (§4.4.1); H6 FiLM slots excluded; H0 Case D; H5 nested cascade **candidate only** — secondary investigation “pre→post rechannel” was **rejected as next step** because production cascade already seeds post-rechannel (`cascade_head_finalize` → `cascade_seed_head_from_output`). Further work requires intermediate C++ dumps (§4.4.3), not residual hypothesis PRs.

**Discipline (binding):** C++ golden adjudicates market interop; f64 oracle adjudicates mathematical fidelity (per [§1.2](#12-two-oracle-governance-policy)); never remove guard while SNR < 90 dB; never regenerate golden to accommodate divergence.

#### 4.4.1 Investigation Log — Hypothesis Matrix (TR2.4)

The hypotheses below are ordered by likelihood × isolation cost — H1 must be cleared
before H2, etc. **Forbidden** to apply stacked fixes without isolating H1 first.
One dominant hypothesis per correction PR when possible.

| ID  | Hypothesis                                                | Experiment                                                                                                                                                                                                                                          | Confirmation Criterion                           | Status                                                                                                                                                                                    |
|:--- |:--------------------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------ |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H1  | `head1x1` groups reorder ≠ C++ Conv1x1                    | Compare weight count + layout of grouped `head1x1` path vs `mixin` path, which shares the same grouped-Conv1x1 layout. Patch trial: force `head1x1.groups=1` (ignore groups) on a test-only branch and re-measure SNR against golden C++.           | ΔSNR >> 0 (≥ 10 dB) in isolation                 | 🔴 isolated (PR-R3a). SNR 1.35→2.31 dB (Δ=+0.96 dB). Fix verified correct but Δ < 10 dB — not dominant alone. Proceed H2.                                                                 |
| H2  | `condition_dsp` output channel order / broadcast mismatch | Dump 8-channel `condition_dsp_output` after prewarm and compare frame-aligned against C++ expectation (or f64 reference at identical topology). Test whether `condition_dsp.num_output_channels()` matches `condition_size=8` structural invariant. | Structural mismatch in output channel layout     | 🔴 isolated (PR-R3b). `film_bias_count_generic` now accounts for shift (72 extra FiLM bias consumed, budget→818). Broadcast fixed for `dsp_ch>1`. SNR 0.23 dB — not dominant. Proceed H3. |
| H3  | Head kernel legacy (`head_kernel_size` ≠ 16)              | Force `head_kernel_size=1` vs `head_kernel_size=16` at parse time in a test-only branch; measure SNR against golden C++ for both. The C++ engine always uses head_kernel_size loaded from the weight stream (config field is metadata).             | SNR jumps across config change                   | 🔴 excluded (TR3.3). C++ hardcodes K=1 for legacy `head_size`/`head_bias` format (model.cpp:897). Rust `unwrap_or(1)` matches. No divergence.                                             |
| H4  | Softsign activation vs C++ expectation                    | Controlled swap of activation types in test branch. **Only if H1–H3 are all negative.** Trainer uses `LeakyReLU`; `wavenet_a2_max.nam` has softsign-derived field — verify this is a training artifact, not a load-time activation divergence.      | Activation config confirmed as metadata artifact | 🔴 excluded (TR3.3). Both C++ and Rust parse `"activation":{"type":"Softsign"}` identically. C++ uses `ActivationConfig::from_json`. No divergence.                                       |

##### H1 — Grouped head1x1 weight layout

**Symptom:** The A2 Max model declares `head1x1.groups = 2` and `groups_input_mixin = 4`.
The Rust dispatcher (`build.rs`) loads group-aware compact layouts for both `head1x1` and
`mixin` weights. The mixin path has been verified to produce a correct grouped Conv1x1
(shared `groups_input_mixin` semantics), but the `head1x1` path uses the same layout
pattern independently — any systematic layout error would manifest in the head accumulator,
which directly feeds the head convolution.

**File references:**

- Weight loading: `src/models/a2/model/dynamic/build.rs:253-282`
- Head accumulation: `src/models/a2/model/dynamic/process.rs:485-531`
- C++ reference: `NAM/dsp.cpp` (grouped Conv1x1 forward), `NAM/wavenet/model.cpp:273-297` (head1x1 weight set)

**Experiment protocol:**

1. Count all `head1x1` weight positions in the Rust `set_weights` path and compare with
   NAMCore Python reference (`third-party/.../generate_weights_a2.py`) for the A2 Max fixture.
2. Temporarily override `head1x1.groups = 1` in a test branch (skip group-aware layout)
   and re-measure SNR against the C++ golden.
3. If SNR jumps (Δ ≥ 10 dB), the grouped layout or the grouped inference loop is the
    dominant bug. If SNR unchanged, H1 is excluded.

**Result (2026-08-09, TR3.1 / PR-R3a):**

- Fix applied: group-major → per-output-channel row-major reorder in `build.rs` (matching mixin/l1x1 pattern).
- Budget 818 intact. All A2 regression gates (Full, Lite, FiLM, gated, blended, container, condition_dsp) pass without threshold relaxation.
- Measured SNR: **2.31 dB** (baseline 1.35 dB, Δ = +0.96 dB).
- **Verdict:** H1 fix is correct (no regressions, budget intact, Δ > 0) but **not dominant** (Δ < 10 dB criterion). Proceed to H2 (TR3.2).

##### H2 — condition_dsp output channel layout

**Symptom:** The `condition_dsp` sub-model is a 2-array WaveNet cascade with
`head_size=8` (the second array's head produces 8 output channels). These 8 channels
are fed as the condition vector to the main model. The channel ordering in
`condition_dsp_output` (interleaved `[ch0_f0, ch1_f0, ..., ch7_f0, ch0_f1, ...]`)
must match what C++ produces — any transposition, off-by-one channel stride, or
mono→multi-channel broadcast being **absent** (C++ uses strict 1:1 mapping per §3.9.1)
could cause large structural divergence.

**File references:**

- Condition DSP dispatch: `src/loader/dispatcher/wavenet/static_factory.rs:415-435`
- Condition DSP processing: `src/models/a2/model/dynamic/process.rs:89-108`
- Diagnostic dump: `src/testing/diagnostics/mod.rs` (TR2.3)
- C++ reference: `NAM/wavenet/model.cpp:652-729`

**Experiment protocol:**

1. `NAM_A2_MAX_UNLOCK=1 cargo test test_a2_max_diagnostic_dump_bit_stable -- --nocapture`
   to capture 8-channel condition_dsp output.
2. Generate equivalent C++ dump from NAMcore `render` tool with custom instrumentation
   (log `_condition_output` per frame) — or use the f64 oracle as initial comparison.
3. Compare frame-aligned output channels across tools. Any structural mismatch in
    ordering or values confirms H2.

**Result (2026-08-09, TR3.2 / PR-R3b):**

- Fix 1 (`weights_layout.rs`): `film_bias_count_generic` now accounts for `shift` — bias count = `channels * mult`. Consumes additional 72 FiLM bias values previously under-read, aligning weight stream to the full 818-weight fixture budget.
- Fix 2 (`process.rs`, `cascade/mod.rs`): Broadcast code for `dsp_ch > 1` (`dsp_ch < cond_size`) corrected — previously read mono-aligned `buf[f]` which is wrong for multi-channel sub-model output.
- Budget 818 intact. All A2 regression gates (Full, Lite, FiLM, condition_dsp, gated, blended, container) pass.
- Measured SNR: **0.23 dB** (drops from 2.31 dB baseline — desync previously compensated partially for another bug).
- **Verdict:** H2 fixes are correct (no regressions, budget aligned, broadcast fixed) but **not dominant** (Δ < 10 dB). Proceed to H3 (TR3.3).

##### H3 — Head kernel size handling

**Symptom:** The A2 Max fixture has `head_bias: true` in JSON. The Rust model loads
`head_kernel_size` from the dispatcher (cascade model sets `head_kernel_size` on each
sub-array). The C++ engine uses the weight-stream dimension to infer kernel size.
A mismatch in this value would cause the head convolution to read wrong ring-buffer
positions or apply wrong-shaped convolution, corrupting the output by up to tens of dB.

**File references:**

- Head kernel: `src/models/a2/model/dynamic/build.rs:316-317`, `process.rs:168-169`
- C++ reference: `NAM/wavenet/model.cpp:382-383`
- Fixture: `tests/fixtures/models/wavenet_a2_max.nam` — `head_kernel_size` in JSON

**Experiment protocol:**

1. Force `head_kernel_size=1` in a test-only branch (skip K=16 convolution).
2. Measure SNR against golden C++.
3. Force `head_kernel_size=16` explicitly (confirm current default).
4. Compare SNR delta. A large jump identifies the kernel as the dominant divergence.

##### H4 — Softsign activation configuration

**Symptom:** The A2 Max fixture JSON declares activation configurations that the
NAM trainer (`_wavenet.py`) always defaults to `LeakyReLU(negative_slope=0.01)`.
The `softsign` activation may appear in the JSON as a training artifact
(field in the exported config but never used at inference). If the Rust loader
incorrectly interprets this as a runtime activation override, it could cause
activation divergence on some layers.

**File references:**

- Activation parsing: `src/loader/nam_json/activation_parser.rs`
- Fixture: `tests/fixtures/models/wavenet_a2_max.nam` — `activation` fields
- C++ reference: `NAM/activations.h/.cpp`, `NAM/wavenet/model.cpp` (activation dispatch)

**Experiment protocol:**

1. Inspect the parsed activation config from the A2 Max JSON.
2. If `Softsign` is present, verify it is a training artifact (not used at inference
   in either C++ or Rust). Check `activation_parser.rs` handling.
3. **Only if H1–H3 are all negative:** force `LeakyReLU` uniformly and re-measure SNR.

##### Priority Ordering Rationale

H1 (head1x1 groups) is ranked highest because:

1. It is the most mechanically specific — a known layout discrepancy is testable in isolation.
2. Grouped convolutions are the primary structural difference between A2 Max and all
   other passing A2 models (which use groups=1 for both mixin and layer1x1).
3. The experiment is self-contained: override one config flag, measure SNR.

H2 (condition_dsp channels) is ranked second because the condition_dsp is the other
major structural novelty of A2 Max. If the condition signal reaching per-layer FiLM and
mixin is wrong, *everything* downstream is corrupted — fixing H1 before verifying H2
would be wasted effort.

H3 (head kernel) is lower priority because head kernel handling has been audited for
standard A2 models and found correct; the A2 Max case exercises the same code path.

H4 (softsign) is specifically gated behind H1–H3 because activation substitution bugs
are historically rare in this codebase, and the softsign field in A2 Max JSON has been
confirmed as a training artifact in prior audits.

#### 4.4.2 Post-R3 re-audit & Secondary Hypotheses (2026-08-09)

**Facts (do not restate the superseded TR3.4 “f64≈prod” claim):**

| Pair                         | Metric                           | Notes                                                       |
|:---------------------------- |:-------------------------------- |:----------------------------------------------------------- |
| prod f32 × C++ golden        | **SNR = 0.23 dB**, ESR ≈ 9.49e-1 | HEAD after H1+H2; meter `test_measure_a2_max_snr_vs_golden` |
| prod f32 × f64 oracle paired | **ESR = 5.88×10³** (test FAILED) | `test_oracle_a2_generic` + unlock — **prod ≉ f64**          |
| f64 × F32-sim (same oracle)  | ESR ≈ 4.18e-14 (−133.8 dB)       | `test_oracle_a2_max_standalone` only                        |

H1–H4 are exhausted as **dominant** causes (§4.4.1). Broadcast fix H2 does not run on A2 Max
if nested `condition_dsp` reports `dsp_ch == condition_size == 8`.

**Secondary hypothesis matrix:**

**H0 triple decomposition results (TR2b.1, 2026-08-09):**

Measured on `golden_wavenet_a2_max.bin` (n=2048, block=64, prewarm=2048, 48 kHz),
f64 oracle with `PrecisionConfig::default()` (F64Exact weights, Exact activations, Neumaier acc):

| Pair                  | ESR (linear) | ESR (dB) | SNR (dB) |
|:--------------------- |:------------ |:-------- |:-------- |
| prod f32 × C++ golden | 9.49e-1      | -0.2     | 0.23     |
| prod f32 × f64 oracle | 4.60e3       | +36.6    | -36.6    |
| f64 oracle × C++      | 1.00e0       | +0.0     | -0.00    |

**Classification: Case D** — all three pairs diverge significantly. prod×C++ (ESR≈0.95)
confirms the known gap. prod×f64 (ESR≈4600) confirms TR3.4's "prod≉f64". f64×C++ (ESR≈1.00)
indicates the f64 oracle output is essentially uncorrelated with the C++ golden —
the divergence is NOT solely a production f32 approximation error; the f64 oracle itself
diverges from the C++ reference. Multiple fault sources are active; prioritize
condition_dsp and FiLM per-slot investigation (H5/H6/H7).

Auto-generated test: `test_h0_triple_decomposition` in `tests/models/golden_vectors.rs`
(`#[ignore]`d). Run with:

```sh
cargo test --test models test_h0 -- --ignored --nocapture
```

| ID     | Focus                                                    | Status                                 |
|:------ |:-------------------------------------------------------- |:-------------------------------------- |
| **H0** | Triple decomposition prod / f64 / C++ (classify A/B/C/D) | ✅ **done (Case D)** — see table below |
| **H5** | Nested WaveNet condition_dsp (head 4→8, layout, prewarm) | ✅ **done (TR2b.2)** — see below       |
| **H6** | Per-slot FiLM weight cursor (8 films, shift, groups)     | ✅ **done (TR2b.3)** — see below       |

**H6 findings (TR2b.3, 2026-08-09):**

A2 Max topology: **CH=4, BN=4, cond=8, K=4, 2 layers**, head1x1_active=true,
head_accum_size=4, head1x1_h1_in=2, head_kernel_size=1, mixin_groups=4, l1x1_groups=2.

All 8 FiLM slots are active in both layers (16 total), all with `shift=true`. Groups vary:

| Slot | Name                  | Groups | Weights | Bias | Offset l0 | Offset l1 |
|:----:|:--------------------- |:------:|:-------:|:----:|:---------:|:---------:|
| 0    | conv_pre_film         | 2      | 32      | 8    | 104       | 508       |
| 1    | conv_post_film        | 4      | 16      | 8    | 144       | 548       |
| 2    | input_mixin_pre_film  | 4      | 32      | 16   | 168       | 572       |
| 3    | input_mixin_post_film | 2      | 32      | 8    | 216       | 620       |
| 4    | activation_pre_film   | 1      | 64      | 8    | 256       | 660       |
| 5    | activation_post_film  | 2      | 32      | 8    | 328       | 732       |
| 6    | layer1x1_post_film    | 8      | 8       | 8    | 368       | 772       |
| 7    | head1x1_post_film     | 4      | 16      | 8    | 384       | 788       |

**Result:** All 16 slots match `weights_layout.rs` formulas exactly. Total = 818 (mid-layer: 100 per layer ×2 + FiLM 304 per layer ×2 + head 6 = 818). **No slot overlap detected.** The +72 bias fix from H2 did NOT create a FiLM cursor misalignment.

Auto-generated test: `test_a2_max_film_slot_budget` in `tests/models/golden_vectors.rs` (`#[ignore]`d).
| **H7** | `num_output_channels` / condition stride contract | ✅ **done (TR2b.2)** — see below |

**H5+H7 findings (TR2b.2, 2026-08-09):**

- The nested `condition_dsp` is `StaticModel::WavenetA2Cascade` with **2 arrays** (cascade chain). Array[0] has `head_size=4`; the last array outputs 8 channels (`num_output_channels() == 8`).
- `dsp_ch == cond_size == 8` on A2 Max — the `dsp_ch < cond_size` branch in `process.rs:110` is **dead code**.
- **H2 broadcast is NOT active on this fixture.** The H1+H2 tree (0.23 dB SNR) cannot be attributed to the condition_dsp channel-broadcast fix.
- condition_dsp f32 production × f64 oracle: aggregate ESR = 3.93e2 (+25.9 dB). Per-sub-block ESR ranges from 19.5 dB to 33.4 dB — the condition_dsp output itself diverges significantly from the f64 ideal, even in isolation.
- The 8-channel output suggests the nested cascade correctly produces multi-channel conditioning, but the cascade's internal head propagation (pre-rechannel vs post-rechannel, §4.6) may contribute to the divergence.

Auto-generated test: `test_tr2b2_condition_dsp_contract` in `tests/models/golden_vectors.rs` (`#[ignore]`d).

**R2.bis conclusion — Hypothesis ranking (TR2b.4, 2026-08-09):**

All 4 hypotheses investigated. Ranking by evidence strength:

| Rank | ID        | Verdict                                                           | Evidence                                                                                                                                                                                                                                                                  |
|:----:|:--------- |:----------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 🥇 | **H5+H7** | **Strongest remaining candidate area** (not a validated fix plan) | Nested `WavenetA2Cascade` (head 4→8). Cond prod×f64 ESR=3.93e2 is weak adjudication (oracle ≉ C++). Cascade wire already post-rechannel. Future work needs C++ intermediate dumps (§4.4.3), not residual “seed fix” PRs.                                                  |
| 2    | **H0**    | Confirms H5/H7 dominance                                          | Case D: all 3 pairs diverge. f64×C++ (ESR≈1.00) proves the oracle itself diverges from C++ — the bug is not f32 approximation, it's a structural difference in the computation graph. The condition_dsp cascade is the only sub-component complex enough to explain this. |
| 3    | **H6**    | **Excluded**                                                      | All 16 FiLM slots (8 per layer × 2 layers) verified against `weights_layout.rs` formulas. Budget = 818 exact. Zero slot overlap. H2 +72 bias fix did NOT cause cursor misalignment.                                                                                       |

**Verdict (instrumentation only):** Nested cond is the strongest **remaining candidate** area, but the planned secondary hypothesis “fix cascade seed to post-rechannel” is **not validated** — code already finalizes then seeds. H0 Case D means f64 cannot adjudicate. **Decision (2026-08-09):** stop speculative A2 Max correction attempts; capitalize gains; formalize **KB-A2-MAX** (§4.4.3).

H1–H4 remain excluded. H6 is excluded. Secondary investigation closed without a C++-adjudicated root cause.

#### 4.4.3 Known bug KB-A2-MAX — freeze, capital gains, reopening criteria

**ID:** `KB-A2-MAX`
**Fixture:** `tests/fixtures/models/wavenet_a2_max.nam` + `golden_wavenet_a2_max.bin`
**Public contract:** `build_model` → `Err` (message contains `KB-A2-MAX` / `parity gap` / `fail-closed`).
**Unlock (diagnostics only):** `NAM_A2_MAX_UNLOCK=1` under `cfg(test)` or feature `testing`.
**Catalog:** `ApplicableOracle::KNOWN_GAP`.
**CI:** `test_wavenet_a2_max_dispatch_rejected` **active** (must stay green). Golden / meter / paired oracle / live remain `#[ignore]` with KB-A2-MAX reasons — **not** release gates.

**Topology (why hard):**

- Main array: CH=4, BN=4, `condition_size=8`, FiLM×8 slots/layer, head1x1 groups, `head_size=1`.
- Nested `condition_dsp`: WaveNet **2-array cascade**, array0 `head_size=4`, array1 `head_size=8` (multi-head path), own weights (1052) + main (818).
- C++ routes generic Eigen WaveNet; Rust routes `WaveNetA2Dyn` + nested `WavenetA2Cascade`.

**What was proven fixed / healthy (do not reopen without regression):**

| Asset                                                            | Evidence                           |
|:---------------------------------------------------------------- |:---------------------------------- |
| A2 Full / Lite / FiLM / Gated / Blended / Chaos                  | SNR ~103–140 dB vs NAMCore         |
| `wavenet_condition_dsp.nam` (standalone nested-ish cond)         | ~139 dB                            |
| Heterogeneous clone (condition_dsp, dyn_free, official, SLAMMIN) | PASS                               |
| Loader atomic Err (F1), clone (F2), ReLU container (F3)          | PASS                               |
| Budget A2 Max weights                                            | 818 + 1052 exact                   |
| H1 head1x1 group layout                                          | Correct (ΔSNR small; not dominant) |
| H6 FiLM per-slot cursor                                          | 16/16 formulas; no overlap         |
| Fail-closed + no smoke-green                                     | TR1.1 + TR1.2                      |

**What remains unknown (future investigation only):**

1. Root structural mismatch in nested multi-array condition_dsp and/or multi-head finalize vs C++ (needs **C++ intermediate tensors**, not f64 alone).
2. Whether oracle f64 graph matches C++ for cascade head_size&gt;1 (H0 f64×C++ ESR≈1).
3. Any residual main-net interaction after cond is bit-exact (only after 1–2).

**Reopening criteria (all required before removing guard):**

1. Intermediate C++ dumps or micro-goldens for nested cond output (8 ch × N) and optionally per-array heads.
2. One hypothesis PR with isolated ΔSNR ≫ 10 dB on `test_measure_a2_max_snr_vs_golden`.
3. prod×C++ SNR ≥ **90 dB** stable; neighbors unregressed; RT heap-audit ok.
4. Same merge (or golden-first): un-ignore golden + live; flip catalog off `KNOWN_GAP`; remove guard.

**Non-goals while frozen:** residual “try cascade.rs seed”, regenerate golden/anchor to pass, claim parity via f64 self-check, relax neighbor thresholds.

### 4.5 Known history — do not repeat

A prior audit round compared production output (`condition_size=8` values/frame) against the
f64 oracle's `condition_dsp` output (1 value/frame — a bug in the oracle, not production) and
concluded there was a critical 93 dB regression. Acting on that conclusion, it changed
production code to match the broken oracle, which reintroduced a real divergence from C++
that a prior fix had already corrected. That change was reverted. The load-bearing rule:
**never change production code to match one oracle without verifying the other.**
Disagreements between oracles are `REVIEW_REQUIRED` governance events per
[§1.2](#12-two-oracle-governance-policy) — the C++ golden adjudicates market interop, the
f64 oracle adjudicates mathematical fidelity, and neither automatically prevails.

### 4.5.1 Anchor Regeneration Policy

A f64 anchor (`tests/fixtures/f64_anchors/*.bin`) may only be regenerated when:

**(a)** a C++ golden vector exists for the same fixture **and** the paired
production×oracle test (`test_summary_table`) already passes within the
acceptance criterion **before** regeneration; **or**

**(b)** no C++ golden exists and the regeneration is accompanied by explicit
human review, documented in the commit message with before/after numbers.

Regenerating an anchor from the very oracle it is meant to validate constitutes
a circular comparison and does **not** constitute evidence of correctness.

### 4.6 Canonical C++ Layout Specifications

The structural spec table below documents C++ reference structures vs Rust implementations for ongoing dynamic engine parity work:

| Aspect                                               | C++ reference                                                                            | Rust reference                                                                                       | Verdict                                                                       |
|:---------------------------------------------------- |:---------------------------------------------------------------------------------------- |:---------------------------------------------------------------------------------------------------- |:----------------------------------------------------------------------------- |
| `Conv1x1` weight stream order                        | `NAM/dsp.cpp:384-393` — row-major `[out_ch][in_ch]` per group                            | `src/models/a2/model/dynamic/build.rs` (`transpose_dense_f32` + `head1x1_w[oc*h1_in+ic]` access)     | Tested both transposed and non-transposed variants against the golden         |
| `head1x1` weight count                               | `NAM/wavenet/detail.h:75-76` — `out_channels × (bottleneck/groups)`, bias `out_channels` | `build.rs` reads `head_accum_size × h1_in_size` (`head_accum_size == out_channels`)                  | ✅ Matches C++ formula                                                        |
| `head1x1` application loop (grouping)                | `NAM/dsp.cpp:449-646` — implicit block-diagonal GEMM                                     | `process.rs` explicit `grp → oc → ic` loop                                                           | ✅ Structurally equivalent                                                    |
| Cascade head propagation (multi-array)               | `NAM/wavenet/model.cpp:769` — propagates **post-rechannel** head output                  | `cascade/mod.rs` calls `cascade_head_finalize` then `cascade_seed_head_from_output` (post-rechannel) | ✅ Wire matches intent; multi-head K/bias path still under KB-A2-MAX scrutiny |
| `condition_dsp` interface (dimensions, pass-through) | `NAM/wavenet/model.cpp:699-729`                                                          | `process.rs:89-98`                                                                                   | ✅ Interface dimensions match (`condition_size` values/frame)                 |
| Head finalization, `head_size == 1`                  | `NAM/wavenet/model.cpp:382-383` — Conv1D(kernel=head_kernel_size, bias, head_scale)      | `A2HeadConv` (kernel=16, bias, head_scale)                                                           | ✅ Matches for `wavenet_a2_max.nam` (`head_size=1`, `kernel=16`)              |
| Head finalization, `head_size > 1`                   | Conv1D with kernel + bias + head_scale                                                   | Dense projection, no kernel/bias/head_scale                                                          | ⚠ Only equivalent to C++ when `head_kernel_size == 1 ∧ head_bias == false`    |

### 4.7 Test coverage and fixture quality

Verified directly against `tests/models/golden_vectors.rs`, `tests/parity/cpp_parity.rs`,
`tests/common/validation.rs`, and `tests/parity/reference_oracle_f64.rs`:

- **`tests/models/golden_vectors.rs`** (committed `.bin`) covers `golden_wavenet_a2_lite.bin` (Lite
  variant, `condition_size=1`, CH=3) and `golden_wavenet_a2_full.bin` (Full variant,
  `condition_size=1`, CH=8). Both pass bit-identically against prior baselines.

- **`tests/parity/cpp_parity.rs` (live, `#[ignore]`d)** has no
  `live_cross_validation_wavenet_a2_dyn` test because `WaveNetA2Dyn`'s scalar fallback path is
  a `nam-rs` internal extension for non-standard geometries not present in upstream C++ NAMcore.
  Cross-validation uses the synthetic dynamic builder anchor tests instead.

- **Real, official `.nam` files exercised:**

  - `a2_example.nam` — NAMcore's own official bundled example (`example_models/A2.nam`,
    `SlimmableContainer` with two WaveNet A2 submodels, CH 3→6). Golden-tested
    (`test_golden_vectors_a2_example_slimmable`) and live-cross-validated
    (`live_cross_validation_a2_example_slimmable`). **Passes** (ESR ~7.28e-14 vs NAMcore).
  - `wavenet_a2_max.nam` — Steve Atkinson's official flagship example (CC0). **Public contract
    (2026-08-09+): fail-closed.** `build_model` / `check-model` return `Err` citing **KB-A2-MAX**
    (`reject_wavenet_a2_max_class`, TR1.1). Active CI gate:
    `test_wavenet_a2_max_dispatch_rejected` (must stay green). Golden
    (`test_golden_vectors_wavenet_a2_max`), SNR meter, paired f64 oracle, and live
    `live_cross_validation_wavenet_a2_max` v1+v2 remain `#[ignore]`d with KB-A2-MAX reasons —
    **not** release gates. Unlock for diagnostics only: `NAM_A2_MAX_UNLOCK=1` under
    `cfg(test)` / feature `testing`. Authoritative metrics and reopen criteria: **§4.4 / §4.4.3 /
    §7.1**. Do **not** claim that the guard was removed or that production loads this model.

- **Every other A2 fixture is synthetic**, by explicit design and documentation
  ([`docs/fixtures.md`](fixtures.md)), not by omission:

  - `wavenet_a2_full.nam` / `wavenet_a2_lite.nam` (fast-path parity, calibrated weights) — explicitly
    labeled "**NOT official FiLM models**" to prevent future confusion with `wavenet_a2_max.nam`.
  - `wavenet_a2_film_{full,lite,chaos_stress,input_mixin_pre}.nam` (FiLM dynamic path),
    `a2_dynamic_gated_ch8.nam` / `a2_dynamic_blended_ch3.nam` (gating/blending),
    `wavenet_a2_container.nam` (`SlimmableContainer` joining the two fast-path submodels) — all
    generator-produced (`generate_a2_fixtures.py`), purpose-built to exercise one structural
    feature each against the C++ **generic** path (not `a2_fast`, which rejects all of them —
    §4.1). This is the correct, honest use of synthetic fixtures: proving the *feature* works,
    not claiming tone-fidelity on a trained model.
  - `mock_a2.nam` — a deliberate negative fixture (zero weights, `ReLU` config) used only to test
    the RT-safe model-load-failure path (`RT_STATUS_MODEL_LOAD_FAILED`), not inference at all.

- **Net assessment (2026-08-10):** A2 *shape-detection* (§4.1) and *dynamic-engine feature*
  fixtures are solid: gating/blending and FiLM paths measure **near-bit-exact** vs NAMcore after
  the identity-biased generator fix (§4.3 / §7.2 — historical 18–36 dB figures are obsolete).
  Fast-path Full/Lite goldens pass with multi-order margin but remain **synthetic-only** (no
  trained community A2-Full/Lite capture in-suite). The only real official single-network A2
  flagship (`wavenet_a2_max.nam`) is **intentionally rejected** (KB-A2-MAX); the only real
  official A2 topology with live+golden green status is the `SlimmableContainer` wrapper
  `a2_example.nam`. That makes A2 the weakest architecture on *genuine trained-model* coverage —
  by freeze policy, not by unnoticed silence.

- **Community-trained A2 search:** No publicly trained A2-Full/Lite
  model is incorporated; full/lite fixtures remain calibrated synthetic. A2 was released
  on 2026-06-02 and all known trained models reside on TONE3000 under the T3K license
  ("may not upload, republish, or distribute the data file without the author's permission") —
  incompatible with redistribution in Apache-2.0 fixtures. This is an ecosystem limitation
  (A2 is too new for a trained corpus to exist outside TONE3000), not an implementation
  gap.

---

## 5. Shared DSP Engine Semantics

Applies identically to LSTM, WaveNet A1, and A2 — all route through the common `NamModel` trait
and the C++ `DSP` base class.

| C++ (`NAM/dsp.h` / `dsp.cpp`)                                                                                              | Rust (`src/`)                                                                                                                                                                                                                        | Verdict                                                                                                                                                                                                                                                                           |
|:-------------------------------------------------------------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DSP::Reset(sr, maxBuf)` → `SetMaxBufferSize` + `prewarm()` iff `GetPrewarmOnReset()` (default `true`)                     | `NamModel::reset()` → `set_max_buffer_size` + `prewarm()` iff `prewarm_on_reset()` (default `true`)                                                                                                                                  | ✅ Match — verified for LSTM (§2.2) and WaveNet A1 (§3.5); A2 not re-verified this pass                                                                                                                                                                                           |
| `DSP::GetPrewarmSamples()` base returns `0`; overridden per-model; used by the **iterative** `DSP::prewarm()` loop         | `prewarm_samples()` per-model override                                                                                                                                                                                               | ✅ LSTM verified exact (`0.5 × sr`), load-bearing (drives real iteration). ✅ WaveNet A1 now correctly sums all arrays + condition_dsp + post-stack head (§3.5 FIXED); `prewarm()` discards arg and runs analytical fill. A2 not re-verified this pass.                           |
| `Activation::using_fast_tanh` default `false` (exact `tanh`/`sigmoid`); only flipped by benchmark tools, never by `render` | Activation precision selected via `ActivationPrecision::{Fast, Standard}`; `Fast` uses Padé/minimax approximations, not exact math. `Standard` (exact-grade polynomial, universal default) matches C++ exact math parity within 2e-7 | ⚠ **Intentional divergence, not a bug.** C++'s reference path used for goldens is exact math; NAM-rs's `Fast` mode trades a small, bounded approximation error for throughput. `Standard` (exact-grade default) narrows this to identical parity within measurement noise (§2.5). |

### 5.1 Sample Rate Default Policy (F-P3)

**Background:** the C++ NAMcore uses `NAM_UNKNOWN_EXPECTED_SAMPLE_RATE = -1.0`
(`NAM/dsp.h:30`) as a sentinel when the `sample_rate` field is absent from the `.nam`
JSON. When `expected_sample_rate == -1.0`, the LSTM prewarm computation
(`NAM/lstm.cpp:128`) produces `max(1, (int)(0.5 × -1.0)) = 1` sample — effectively
disabling prewarm.

**NAM-rs policy:** `sample_rate` absence defaults to **48000 Hz**
(`src/loader/loaded_model_pair.rs:13`, `pub(crate) const DEFAULT_SAMPLE_RATE: f32 = 48000.0`).
This value drives the real prewarm computation for LSTM (24000 samples at 48 kHz) and the
sample-rate-dependent logic in `DSP::Reset()` for all architectures.

**Rationale:** the 48000 Hz default produces a correct, functional prewarm rather than the
C++ sentinel's near-zero prewarm (1 sample). All known production `.nam` models include
`sample_rate` explicitly, so the default is only exercised by degenerate or hand-crafted
models. In those cases, NAM-rs's behavior is measurably "more correct" — the model settles
to its steady state — while C++'s sentinel produces effectively no prewarm at all.

**Divergence assessment:** This is an **intentional, documented, low-risk divergence**.
It does not affect any known production model (every real community `.nam` export includes
`sample_rate`). The behavior affects only the degenerate zero-`sample_rate` case, where
NAM-rs's prewarm is strictly superior. The C++ sentinel is not emulated, and emulating it
has no practical benefit for any real-world use case.

**Verification:** this policy is enforced at two levels:

1. **JSON parse:** `deserialize_sample_rate` (`src/loader/nam_json/validation/schema.rs`) rejects
   non-finite or ≤0 sample rates via `JsonError::InvalidSampleRate`, but `None` (absent
   field) passes through silently — it is handled downstream.
2. **Model build:** `build.rs:161` applies `unwrap_or(DEFAULT_SAMPLE_RATE)` to the parsed
   `Option<f32>`. LSTM dispatchers (`static_builder.rs:27,72`, `dynamic_builder.rs:32`)
   apply the same default independently for prewarm computation.

### 5.2 FastLUTActivation — Not Ported (F-P4-c)

**Background:** C++ NAMcore ships an optional `FastLUTActivation` class
(`NAM/activations.h:127-169`) that precomputes look-up tables for `tanh` and `sigmoid`
to accelerate inference on systems without fast `expf` hardware. It is controlled by
`Activation::enable_fast_tanh()` and `Activation::using_fast_tanh`.

**Status in NAM-rs:** `FastLUTActivation` is **not ported** and has no NAM-rs equivalent.
This is **not a parity gap** for the following reasons:

- `FastLUTActivation` is a **runtime optimization**, not a format/algorithm feature.
  The `.nam` file format has no field for "use lookup tables" — it is a local C++-side
  accelerator that produces the same mathematical output (within LUT precision) as exact
  `tanh`/`sigmoid` for identical weights.
- The NAMcore `render` tool (used for golden generation and live cross-validation) **never**
  enables it: `enable_fast_tanh()` is only called from benchmarking tools
  (`tools/benchmodel*.cpp`), not from `render.cpp`. This is confirmed in the NAMcore
  audited source (`activations.h:14`: `static bool using_fast_tanh = false` is the only
  initialization, and only `benchmodel*.cpp` flips it — verified by grep of all callers).
- NAM-rs's `ActivationPrecision::Standard` (universal default) already produces exact-grade
  `tanh`/`sigmoid` within 2×10⁻⁷ of C++'s exact math, making the LUT precision tradeoff
  irrelevant.

**Verdict:** FastLUTActivation is classified as **"Not Applicable"** — no port needed, no
parity gap, no audio divergence. Documented here for completeness and to prevent future
audit cycles from re-discovering and re-investigating it.

## 6. Other Architectures (Out of Scope)

ConvNet, Linear, `SlimmableContainer`, and the IR Cabsim convolution stage complete the model suite:

- **ConvNet — Paridade Total de Inicialização e Aritmética (✅ resolved 2026-07-28).** O vendored
  NAMcore implementa ConvNet (`NAM/convnet.cpp`), mas usando um formato flat com BatchNorm params
  brutos. O NAM-rs usa um formato nested por-bloco com BatchNorm pré-fundido scale/offset
  (`src/loader/dispatcher/convnet/mod.rs`). A divergência de ESR `2.54e-5` (SNR `45.9 dB`)
  previamente reportada era **exclusivamente um transiente de inicialização de estado (prewarm)**
  confinado às primeiras 62 amostras — o `ConvNetModel::prewarm()` preenchia zeros literais por
  bloco isolado, enquanto o NAMcore (`dsp.cpp:67-96`) processa `receptive_field_size + 1` amostras
  de silêncio através da rede inteira. A correção (`TASK-CONVNET-01`, 2026-07-28) replicou a
  semântica exata do NAMcore, eliminando o transiente.

  - **Métricas de paridade definitivas (pós-fix, 2026-07-28):**
    - **C++ cross-validation** (`quick_parity_convnet`): ESR = `4.20e-15` (SNR `143.8 dB`), MR-STFT = `1.20e-6`
    - **Oráculo f64** (`test_oracle_convnet`): ESR = `3.57e-15` (SNR `144.5 dB`, piso f32)
    - **Oráculo vs NumPy f64** (`test_oracle_vs_python_anchor_convnet`): ESR = `5.23e-33` (bit-exact)
    - **Self-golden** (`test_golden_vectors_convnet_test`): ESR = `0.00e0` (determinismo total)
  - **Gates de qualidade recalibrados:** SNR ≥ `120 dB`, ESR ≤ `1.0e-12`, MR-STFT ≤ `1.0e-4`
    (`TASK-CONVNET-05`).
  - **Teste de invariante:** `test_convnet_prewarm_fixed_point_invariant()` confirma que o
    estado pós-prewarm é um ponto fixo estacionário idêntico à convergência explícita.
  - CPU latency = `10.3 µs` (0.8% of RT budget) — inalterada (prewarm opera em caminho frio).

- **Linear (RF=2048 / 4096 / 8192).** Affine linear model. Baseline ESR vs NAMcore = `1.70e-14` (SNR `137.7 dB`), CPU latency = `0.3 µs` (0.0% of RT budget).

- **`SlimmableContainer`.** Multi-model crossfade orchestration, implemented and tested (`src/models/container.rs`, `tests/models/container_slimmable.rs`). Baseline A2 Example (CH=3→6) cross-validation vs NAMcore: ESR = `7.28e-14` (SNR `131.4 dB`), ESR vs f64 = `1.82e-14`, MR-STFT = `1.73e-05`.

- **IR Cabsim.** Impulse response convolution stage, cross-validated via `tests/parity/cabsim_cpp_parity.rs`.

- **`SlimmableWavenet`.** Channel-sliceable single-network WaveNet for adaptive compute quality scaling (`src/models/slimmable.rs`). Skeleton is **implemented and loads**: `clone_wavenet_for_slimmable_storage` + `slice_wavenet_model`, breakpoints from `allowed_channels`, and inference checks via `test_loader_gap_slimmable_wavenet` / `test_slimmable_wavenet_inference_and_breakpoints`. The fixture `slimmable_wavenet.nam` is **not** a negative reject mock — it builds successfully. **Inference-only; sem claim de paridade multi-size NAMCore** — NAMCore (`NeuralAmpModelerCore`) has no channel-slicing API; multi-size golden/live C++-adjudicated parity is architecturally infeasible (§7.4).

---

## 7. Known-Broken Ledger ("Sabidamente Broken")

Single-page triage. Everything in this document up to here is evidence; this chapter is the
verdict. Read this chapter alone if the only question is *"what's safe to ship, and what isn't."*
Severity tiers are ordered by how much they should worry a release decision, not by section order.

### 7.1 🔴 Known bug KB-A2-MAX — guard permanent until §4.4.3

Fail-closed TR1.1 remains **active**. T8.1 / R3 / secondary investigations do **not** constitute closure. Residual speculative correction attempts **cancelled** in favor of known-bug freeze.

| Model                | Symptom                                                                                                        | Status                                               |
|:-------------------- |:-------------------------------------------------------------------------------------------------------------- |:---------------------------------------------------- |
| `wavenet_a2_max.nam` | prod×C++ **0.23 dB**; H0 Case D (prod/f64/C++ all diverge); budget exact; H1–H4/H6 exhausted as dominant fixes | **KB-A2-MAX.** Guard active. Reopen only via §4.4.3. |

**Contract (frozen):**

- Guard `reject_wavenet_a2_max_class` — message cites **KB-A2-MAX** + fail-closed + SNR≈0.23 dB.
- Catalog `KNOWN_GAP`; unlock only `NAM_A2_MAX_UNLOCK=1` under test/testing.
- Active CI: `test_wavenet_a2_max_dispatch_rejected` (must pass).
- Ignored (not gates): golden, meter, paired f64, live v1/v2 — reasons cite KB-A2-MAX.
- Reopen only via §4.4.3. Neighbors (condition_dsp, A2 Full/Lite/FiLM, containers) stay green.

### 7.2 🟡 Known, measured, accepted tradeoffs — not bugs

Every item below is a deliberate engineering tradeoff with a calibrated, tested error budget. They
show up as nonzero numbers in the tables throughout this document, but they are not defects:

- **LSTM backbone weight representation** — Storage uses native `f32` weights across all gate matrices, matching NAMcore's `Eigen::MatrixXf`. Eliminating weight quantization simplified GEMV kernel dispatch and improved per-sample latency by 10-12%, while bringing `BossLSTM-2x8` to bit-exact convergence with NAMcore (ESR = 1.00e-11).
- **`ActivationPrecision::Fast`'s Padé/minimax activation approximations** vs. C++'s exact
  `tanh`/`sigmoid` — small, bounded, and identical in nature for LSTM and WaveNet A1/A2 (§2.5,
  §3.2, §5). `Standard` (exact-grade, universal default) collapsed this gap to match C++ parity
  within measurement noise.
- **A2 FiLM dynamic-engine interop gap** — previously identified as an interop gap of SNR 18.1–36.0 dB,
  this was shown to be caused by a zero-biased initialization in the synthetic weight generator.
  Following the fix to apply standard identity-biased weights, the gap collapsed to float32 precision
  limits (SNR 138+ dB / ESR ~1e-14), achieving near-bit-exact parity (§4.3).

### 7.3 🟠 Test-infrastructure caveats — parity coverage that can silently vanish

These do not produce wrong audio, but they can make the *evidence* for parity evaporate without failing CI:

- **Silent SKIP in v1 live cross-validation — Phase 2 `quick_parity_*` already fail-closed.** The
  `#[ignore]`d live tests (`run_v1` / `run_v1_hf`) in `tests/parity/cpp_parity.rs` still discard
  `ParityOutcome` via `let _ = run_render_comparison(...)`, allowing SKIP conditions (toolchain
  absent, model missing, render crash) to print `SKIP:` while the test reports `ok`. This is
  **not** the case for the non-ignored `quick_parity_*` tests in Phase 2
  (`utils/tests-quick.sh`): every `quick_parity_*` calls `require_completed`
  (`tests/parity/cpp_parity.rs:72`), which **panics** on any outcome except `Completed` —
  `SkippedModelNotFound`, `SkippedGarbageOutput`, `SkippedCppToolchainAbsent`, and all other
  non-`Completed` variants are hard failures. A meta-test in
  `tests/models/threshold_calibration.rs` (`test_no_discarded_parity_outcome`) guarantees zero
  occurrences of `let _ = run_render_comparison` in the entire test tree. **Residual:**
  optional nondist fixtures (e.g. `EVH-5150-Lite.nam`) can produce `SkippedModelNotFound` with
  explicit `allow` in the `#[ignore]`d `live_cross_validation_wavenet_lite` — this is a
  per-test policy decision, not a structural silent-failure hole. With C++ toolchain and
  committed fixtures present, a `quick_parity_*` test **cannot** pass green with zero
  comparisons.
- **Non-distributable / external catalog path drift.** `golden_gen_build.sh` resolves models
  through a shared `resolve_nam_model()` shell function that mirrors
  `tests/common/io_helpers.rs::model_path` — scanning five locations in order
  (`$NAM_MODELS_DIR`, `$NAM_THIRD_PARTY_DIR/community_models/`, `tests/fixtures/models-nondist`,
  `third-party/community_models/`, `tests/fixtures/models`). Community captures used in live tests
  (e.g. `EVH-5150-Lite.nam`, APP-EVH, Boss BD-2, SLAMMIN) are resolved transparently from
  `third-party/community_models/` when placed there. The stale "not found at … models-nondist" skip
  is eliminated — if a `.nam` file exists at any path, the gen script
  will find and render it. Freshness manifest still gates *present* artifacts; it does not
  prove every catalog entry was regenerated in the last run.
- **Synthetic goldens still pending offline build** (as of 2026-07-31 skip reasons): e.g.
  `lstm_1x10`, `lstm_2x24`, `lstm_3x8`, `convnet_{nobn,relu,silu}`, `linear_nobias` — structural
  fixtures without committed C++ goldens in the gen catalog path.
- **`quick_parity_convnet`** previously always skipped (§6 — architecture incompatibility), but
  after the prewarm initialization fix it now passes with ESR=4.20e-15 (SNR 143.8 dB),
  completing the 4-model quick-parity matrix at full coverage.
- **`wavenet_a2_film_input_mixin_pre.nam`** has been fully validated with committed C++ goldens and live cross-validation (`live_cross_validation_wavenet_a2_film_input_mixin_pre`), achieving ESR `3.44e-14` (SNR `134.6 dB`, MR-STFT `6.92e-06`) against NAMcore with calibrated gates (SNR ≥ `120.0 dB`, ESR ≤ `1.0e-11`, MR-STFT ≤ `1.0e-4`).
- **Performance quality-contract noise (not parity).** Dashboard runs may report RT latency
  over contract on WaveNet Standard/Feather/Lite while fidelity rows remain green. Treat as a
  separate performance track; do not relax ESR/SNR gates to “fix” a perf miss.

### 7.4 🟡 Policy rejects, defensive gaps, and open coverage (not KB-A2-MAX)

Items below are **not** the Max freeze. They are intentional product policy, low-severity
defensive holes, or incomplete evidence. This table is the parity-map ledger only.

| ID  | Item                                                                                                                | Class                        | Status / contract                                                                                                                                                                                                                      |
|:--- |:------------------------------------------------------------------------------------------------------------------- |:---------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P1  | `wavenet_condition_lstm.nam` (LSTM nested in WaveNet)                                                               | **Policy reject**            | Public load `Err` (“LSTM condition_dsp is not supported”). Upstream trainer cannot produce; C++ construction asserts channel match. Catalog `KnownGap`. CI: `test_policy_reject_condition_lstm` / reject path in golden tests. §3.9.4. |
| P2  | A1 free/dynamic path silently ignores `gated` / `gating_mode` / FiLM / `head1x1` / `layer1x1` when not routed to A2 | **Fail-closed implementado** | Fail-closed implementado — ver §3.6 FIXED.                                                                                                                                                                                             |
| P3  | LSTM `num_layers == 0` and implicit mono `in_channels`                                                              | **Fail-closed implementado** | §2.6 — multi-channel → `Err(UnsupportedMultiChannel)`. `num_layers==0` / bounds → `Err(UnsupportedTopology)`. Missing keys still `Ok(None)`.                                                                                           |
| P4  | WaveNet `prewarm_samples()` under-reports multi-array RF                                                            | **Corrigido**                | Corrigido — soma canônica; prewarm analítico inalterado. §3.5.                                                                                                                                                                         |
| P5  | `dsp_ch < condition_size` broadcast in Rust production                                                              | **Intentional Rust-only**    | §3.9 — C++/trainer reject mismatch; only relevant for models upstream cannot validate.                                                                                                                                                 |
| P6  | `SlimmableWavenet` multi-size vs NAMCore                                                                            | **Disclaimer**               | Inference-only; sem claim de paridade multi-size NAMCore. Load/inference tests remain. NAMCore has no channel-slicing API — multi-size C++-adjudicated parity architecturally infeasible (§6).                                         |
| P7  | A2 fast-path fixtures synthetic-only                                                                                | **Caveat documentado**       | Full/Lite parity is C++-backed on calibrated weights, not trained community captures (§4.2 / §4.7). Nenhum A2-Full/Lite treinado público incorporado em 2026-08-10; fixtures full/lite permanecem sintéticos calibrados.               |

**Non-goals of this ledger row:** reopening KB-A2-MAX, regenerating Max goldens to force a pass, or using f64 oracle as adjudicator (H0 Case D — §4.4.2).

---

## See Also

- [audio_fidelity_map.md](audio_fidelity_map.md) — off-spec DSP factors; §3 (LSTM recurrent drift) pairs with §2.5/§2.7 here
- [perceptual_validation.md](perceptual_validation.md) — metrics and gate-calibration policy
- §4.4 / §4.4.3 / §7.1 — **KB-A2-MAX freeze** (do not reopen without intermediate C++ dumps)
- §7.4 — policy rejects and defensive/coverage backlog (non-Max)
- `tests/parity/cpp_parity.rs` — live cross-validation against the C++ `render` tool
- `tests/parity/reference_oracle_f64.rs` — f64 oracle and independent NumPy anchor (decomposition tools, §1.2)
