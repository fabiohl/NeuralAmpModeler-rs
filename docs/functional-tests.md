<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Functional Testing Guide — DSP Engine

**Audience:** Developers and QA engineers working with the NeuralAmpModeler-rs crate.

---

## Executive Summary

This document defines manual functional verification procedures for the DSP engine core — model loading, inference correctness, sample-rate adaptation, RT-safe allocation, gate behavior, and resampler integrity. Automated test coverage for these areas is tracked in [testing.md](testing.md); the procedures below cover scenarios that benefit from manual inspection or targeted debugging workflows.

| Tier       | Name                     | Target Duration | When to Run                            |
|:---------- |:------------------------ |:--------------- |:-------------------------------------- |
| **Tier 1** | ⚡ Smoke Test            | ~2 min          | After code changes to core modules     |
| **Tier 2** | 🎯 Feature Verification  | ~10–15 min      | Sprint completion / feature merge      |
| **Tier 3** | 🛡️ Robustness & Stress   | ~20–30 min      | Pre-release audit                      |

---

## Tier 1: ⚡ Smoke Test (5 High-Yield Checks)

- [ ] **T1.1 Model Loading & Basic Inference:** Load a `.nam` (JSON) or `.namb` (binary bundle) model via `load_and_build_model()`, call `process()` with a 64-sample block of silence. *Expected:* function returns without panic; output buffer is populated (non-NaN, finite values).
- [ ] **T1.2 Static Dispatch Coverage:** Load one model from each architecture family (WaveNet A1, WaveNet A2, LSTM, ConvNet, Linear). Run a single block of inference. *Expected:* all five load and process successfully.
- [ ] **T1.3 Dynamic Model Fallback:** Load a model whose geometry does not match static profiles (e.g., LSTM 3×5, WaveNet CH=7). *Expected:* dispatches to the appropriate `Dyn` variant and produces output.
- [ ] **T1.4 Invalid Model Rejection:** Attempt to load a corrupted, missing, or 0-byte `.nam`/`.namb` file. *Expected:* returns a descriptive error (`Err(...)`) containing a valid `NamErrorCode`; does not panic.
- [ ] **T1.5 Sample Rate Adaptation:** Load a model with `sample_rate = 44100` in `LoadOptions`. Process a block. Change to `sample_rate = 96000`, call `reset()`, process again. *Expected:* no panics; sample rate metadata in diagnostics reflects the change.

---

## Tier 2: 🎯 Feature Verification

### Domain 2A: Model & Pipeline

- [ ] **2A.1 Block-Size Invariance:** Processing the same input split into [32+32] vs [64] samples must produce identical output (within floating-point tolerance). *Verifies:* sample-by-sample determinism, no hidden state leakage.
- [ ] **2A.2 Prewarm Correctness:** After `prewarm(n)`, the first `n` output samples must be deterministic and free of artifacts (no ramp-up transient that exceeds gate thresholds). *Verifies:* receptive-field buffer initialization.
- [ ] **2A.3 Reset Idempotency:** `reset()` → process → output A. `reset()` → process → output B. *Expected:* A == B (models are fully reset to initial state).
- [ ] **2A.4 Container Submodel Swap:** Load a `SlimmableContainer`, force `Full → Lite` submodel swap mid-inference. *Expected:* 32 ms crossfade produces no audible click; output remains within ESR tolerance.
- [ ] **2A.5 Model Hot-Swap via SPSC:** Send a `LoadModel` payload through a lock-free SPSC channel while processing audio. *Expected:* swap completes without blocking the audio thread; old model is queued for GC without leaks.

### Domain 2B: DSP Pipeline Stages

- [ ] **2B.1 Gate FSM Silence:** Process a block of silence → gate enters `FadingIn`/`FadingOut` → gate closes → output is all zeros. Feed signal → gate reopens with smooth ramp. *Verifies:* hysteresis thresholds and ramp coefficients.
- [ ] **2B.2 Oversampling Anti-Aliasing:** Run a sine sweep through a WaveNet model at 2× and 4× oversampling. *Expected:* aliasing artifacts measured via ASR are below documented thresholds in [audio_fidelity_map.md](audio_fidelity_map.md).
- [ ] **2B.3 Resampler Bypass at Native Rate:** When host sample rate equals model native sample rate (e.g., 48 kHz), `NamResampler` must engage zero-copy bypass with zero kernel overhead. *Verifies:* bypass path engages correctly.
- [ ] **2B.4 Resampler Multi-Rate:** Process at 44.1, 48, 88.2, 96, and 192 kHz. *Expected:* SNR remains above documented thresholds; no buffer overruns.
- [ ] **2B.5 Cabsim IR Convolution:** Load a `.wav` IR → process through `ConvEngine`. Clear the IR. *Expected:* with IR loaded, output differs from raw model output; after clearing, bypass path engages (output == raw model output).

### Domain 2C: RT-Safety & Allocation

- [ ] **2C.1 Zero-Allocation Hot Path:** Enable `heap-audit` feature. Process 10,000 blocks without a single allocation on the audio thread. *Expected:* `CountingAllocator` reports zero allocations in `process()`.
- [ ] **2C.2 Denormal Protection (FTZ/DAZ):** Process a long stretch of silence with `--features heap-audit`. *Expected:* no denormal-related slowdown; `set_daz_ftz()` assertion passes.
- [ ] **2C.3 Panic-Free Inference:** Feed out-of-range, NaN, or Inf values as input. *Expected:* clips to valid range; no unwrap/expect panics on hot path.
- [ ] **2C.4 Memory Footprint Stability:** Load and unload models 100× in a loop. *Expected:* RSS remains stable (< 2 MB growth after initial warmup); no leaked file descriptors.

---

## Tier 3: 🛡️ Robustness & Stress

- [ ] **3.1 Soak Test (10M+ frames):** Run continuous inference for >10 million frames with random block sizes (16–256). *Expected:* zero panics, zero leaks, output remains finite.
- [ ] **3.2 Concurrent Model Swap Storm:** Fire 1,000 SPSC `LoadModel` payloads in rapid succession while the audio thread processes. *Expected:* zero channel-full errors after GC drain; audio thread never blocks >100 µs.
- [ ] **3.3 Adaptive Compute FSM:** Artificially induce high CPU load (external stressor). *Expected:* Adaptive Compute downgrades model complexity (Full→Reduced→Minimal); when load drops, recovers to Full within hysteresis window.
- [ ] **3.4 Resampler Stress:** Randomly vary sample rate every 100 blocks (44.1→48→88.2→96→192→44.1). *Expected:* no buffer corruption; ESR remains stable through rate transitions.
- [ ] **3.5 Cabsim Partition Overflow:** Feed IRs of extreme length (up to 2^20 samples). *Expected:* partition count scales without allocation in hot path; FDL wrap-around is correct.

---

## Release Acceptance Criteria

- [ ] Zero panics or memory leaks across all tiers.
- [ ] Zero heap allocations on the hot path (Tier 2C.1).
- [ ] Block-size invariance maintained (Tier 2A.1).
- [ ] All five architecture families load and process (Tier 1.2).
- [ ] SPSC model swap completes without audio dropouts (Tier 3.2).

---

## Test Harness Quick Reference

```rust
use neural_amp_modeler_rs::common::diagnostics::SystemSnapshot;
use neural_amp_modeler_rs::loader::{load_and_build_model, LoadOptions};
use neural_amp_modeler_rs::models::NamModel;
use std::path::Path;

// Smoke test: load + process

let sys = SystemSnapshot::capture();
let mut mp = load_and_build_model(Path::new("model.nam"), &sys, false, LoadOptions::default())?;
let model = mp.model_l.as_mut().unwrap();

let input = vec![0.0f32; 128];
let mut output = vec![0.0f32; 128];
model.process(&input[..64], &mut output[..64]);

// Block-size invariance check
let sr = 48000;
let mut out_full = vec![0.0f32; 128];
let mut out_split = vec![0.0f32; 128];

model.reset(sr, 128)?;
model.process(&input[..64], &mut out_split[..64]);
model.process(&input[64..], &mut out_split[64..]);

// Reset, then process as one block
model.reset(sr, 128)?;
model.process(&input, &mut out_full);

let max_diff = out_full.iter().zip(&out_split)
    .map(|(a, b)| (a - b).abs())
    .fold(0.0f32, f32::max);
assert!(max_diff < 1e-7);
```
