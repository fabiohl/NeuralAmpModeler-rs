<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Host-precomputed FiLM modulation

Hosts that evaluate their control network once per block can install final
FiLM scale/shift vectors without evaluating that network again for every sample.
The existing per-channel SIMD modulation remains responsible for applying
`input * scale + shift`. A neutral update uses scale=1 and shift=0; no implicit
identity offset is added.

## Individual FiLM layers

Load a `FiLMLayer` off-thread as usual. `set_precomputed_modulation(scales,
shifts)` validates exact feature counts and finite values, then copies both
vectors into its prepared storage and enables the override. `process` skips the
condition projection while the override is active. The original condition
width/input safety requirements still apply, and projection weights/biases are
preserved. The precomputed shift is independent of the projection's `shift` flag.

`has_precomputed_modulation()` reports the mode.
`use_conditioned_modulation()` resumes the loaded projection on the next sample.
Cloning preserves both the override mode and its coefficient vectors.

## Atomic model updates

`WaveNetA2<CH>`, `WaveNetA2Dyn` and `WaveNetA2Cascade` provide:

```rust,ignore
let n = model.activation_modulation_size()?; // prepare n-element arrays off-thread
model.set_activation_modulation(&scales, &shifts)?;
// process one or more blocks with those coefficients
model.resume_activation_conditioning()?;
```

Every layer must already contain a loaded **activation-pre** FiLM projection.
The vectors concatenate features in layer order, or array then layer order for
a cascade. Each width is that insertion point's feature count (convolution output,
before gating), which can differ from residual width. Other FiLM insertion points,
projection weights, head scales, condition DSPs and streaming histories are
preserved. The query validates the complete prepared topology.

All destinations, both input lengths and every coefficient are checked before
any layer changes. A missing projection in the last array, or a non-finite final
coefficient, leaves earlier layers, mode flags and histories intact. Errors use
`FiLMModulationError`, with no heap-backed diagnostic strings.

Updates and resume perform no allocation/deallocation, locking, I/O or logging.
The caller serializes access through the same exclusive mutable model reference
used for processing. Apply changes at block boundaries and retain caller-owned
coefficient storage for the call. The override persists through processing and
reset/prewarm, until replaced or resumed. Reloading weights constructs fresh
FiLM layers in ordinary condition-projection mode, so hosts must reinstall any
override after a reload.

## Host and format responsibilities

This API does not encode controls, smooth targets, evaluate a hypernetwork,
apply a DC blocker or load an extension format. Those operations belong to the
host or a separate parametric model layer. In particular, block smoothing must
use the host's actual frame count, not an engine's internal subchunk count.

A schema-1 adapter can retain its original control-network coefficients and
prepare a canonical A2 inference model with constant activation-pre FiLM
projections. It then installs the evaluated gamma/beta once for each host block.
That conversion, skip-head/DC ordering, restored controls, reset and full-chain
performance need separate validation. No stationary snapshot speedup is implied
for a dynamic FiLM path.

## Checks

```sh
cargo test --lib modulation
cargo test --test rt_constraints --features heap-audit precomputed_modulation_is_heap_free
```

Tests compare cached modulation with ordinary condition projections, including
live changes, ring wraparound, specialized CH=3/8 kernels, SIMD remainders,
bottleneck/gating widths, cascade atomicity and resuming the loaded projection.
The allocator audit covers successful updates, rejected updates, resume and
processing. Existing A2/FiLM tests remain applicable to the default mode.
