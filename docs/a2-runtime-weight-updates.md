<!--
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
-->

# Runtime A2 weight updates

`WaveNetA2::<3>::update_weights`, `WaveNetA2::<8>::update_weights` and
`WaveNetA2Dyn::update_weights` replace coefficients in an already loaded A2
model. They accept the same
serialized NAM weight order as `set_weights`, including the head scale. All
convolution layouts are refreshed in place, including SIMD padding.

Load with `set_weights` and negotiate the processing block size off the audio
thread first. Keep the replacement weight stream in caller-owned storage. At a
block boundary, call `update_weights(&weights)` through the same exclusive mutable
model reference used for processing. There is no concurrent writer or internal
synchronization. Errors are typed (`A2WeightUpdateError`); format/report them off
the audio thread.

The update performs no heap allocation or deallocation, locking, I/O, logging,
reset, or prewarm. It validates the topology, exact weight count and finiteness
before writing anything. A rejected update leaves coefficients and streaming
state intact. A successful update preserves causal history, ring cursors and the
negotiated block size; subsequent samples use the new coefficients with that
existing history. This differs from resetting and replaying the model.

This is a bounded full-weight copy/transposition, not a constant-time operation.
Hosts must budget its cost at their intended update rate. Abrupt coefficient
changes can cause audible discontinuities; this API does not interpolate weights
or crossfade output. Input storage must remain available for the duration of the
call; producing or replacing that storage is the caller's responsibility.

## Scope and parametric models

This API is a first primitive for hypernetwork-produced weights, not a parametric
model loader. Supported layouts are:

- Canonical 23-layer static CH=3/8 models.
- Dense mono-output dynamic arrays with arbitrary loaded channel and bottleneck
  widths, input/condition widths, kernels, dilations and gating/blending modes.
  `runtime_weight_count()` validates the loaded destinations and returns the
  exact replacement-stream size. Nested condition-DSP weights are excluded from
  that stream and its state is preserved.

FiLM, grouped projections/convolutions, head1x1 and multi-output/cascade updates
are not supported. Other topologies return `UnsupportedTopology`; unloaded
models return `NotLoaded`. Checks use arithmetic that detects count overflow.

End-to-end HyperWaveNet support still needs the hypernetwork evaluator, remaining
main-network layouts, control metadata, parameter normalization and a loader.
Concat and FiLM conditioning need separate APIs. This change alone does not enable
loading or playing those parametric model families.

## Validation

The `runtime_updates` unit tests compare all stored coefficient layouts with the
existing loader and compare output bit for bit after processing a nonzero prefix,
including no-op updates, changed updates, varied blocks and history wraparound.
Both CH=3/8 specialized kernels and generic convolution fallback are covered.
Dynamic tests exercise repeated changes, noncanonical kernels, differing
bottlenecks, gating/blending and multichannel coefficient packing.
Invalid streams and unsupported models are rejected atomically.

The runtime-update heap tests use the shared counting allocator for repeated
successful, wrong-length and non-finite updates, including dynamic processing:

```sh
cargo test --lib runtime_updates
cargo test --test rt_constraints --features heap-audit runtime_weight_updates_are_heap_free
```
