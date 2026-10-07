#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
"""
Deterministic fixture generator for NAMCore v0.6.0 synthetic models.

Produces:
  linear_1x2.nam              — Multichannel Linear: 1->2 (kernels=2, biases=2, RF=8)
  linear_2x1.nam              — Multichannel Linear: 2->1 (kernels=2, biases=1, RF=8)
  linear_2x2_shared.nam       — Multichannel Linear: 2->2 (kernels=1 shared, biases=1, RF=8)
  linear_1x2_fft.nam          — Multichannel Linear FFT: 1->2 (RF=2048, kernels=2, biases=2)
  sequential_linear_chain.nam — Sequential pipeline of two Linear 1->1 stages (mono)
  sequential_multichannel.nam — Sequential pipeline (1->2 Linear followed by 2->1 Linear)
  sequential_nested.nam       — Sequential pipeline with nested child Sequential
  sequential_sr_homogeneous.nam — Sequential pipeline with matching child sample rates (48k, 48k)
  sequential_sr_mixed_unknown.nam — Sequential pipeline with child 1 unknown rate and child 2 at 48k
  sequential_sr_conflict.nam  — Sequential pipeline with conflicting sample rates (44.1k vs 48k)
  sequential_linear2.nam      — Sequential Linear 1->1 into Linear 1->1, sample_rate absent everywhere
  sequential_linear_wavenet.nam — Sequential Linear 1->1 into WaveNet 1->1, sample_rate absent everywhere
  sequential_double_lstm.nam  — Sequential of two LSTM 1x8 stages at an explicit 48 kHz rate
  wavenet_head_dilation.nam   — WaveNet with head.kernel_size=3 and head_dilation=2

All models use deterministic PRNG seeds, complete valid envelopes, and stable formatting.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List

OUTPUT_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def gen_floats(n: int, rng: random.Random, scale: float = 0.2) -> List[float]:
    return [round(rng.uniform(-1.0, 1.0) * scale, 6) for _ in range(n)]


# =============================================================================
# 1. Multichannel Linear Models (GAP-02)
# =============================================================================

def make_linear_nam(
    in_ch: int,
    out_ch: int,
    rf: int,
    bias: bool,
    rng: random.Random,
    sample_rate: float = 48000.0,
    model_name: str = "Linear",
) -> Dict[str, Any]:
    # Upstream NAMCore rules:
    # kernels = (in == out) ? 1 : max(in, out)
    # biases = (in == out) ? 1 : out
    kernels = 1 if in_ch == out_ch else max(in_ch, out_ch)
    biases = 1 if in_ch == out_ch else out_ch
    num_weights = rf * kernels + (biases if bias else 0)
    weights = gen_floats(num_weights, rng, scale=0.15)

    config: Dict[str, Any] = {
        "receptive_field": rf,
        "bias": bias,
    }
    if in_ch != 1 or out_ch != 1:
        config["in_channels"] = in_ch
        config["out_channels"] = out_ch

    model: Dict[str, Any] = {
        "version": "0.6.0",
        "architecture": "Linear",
        "config": config,
        "weights": weights,
        "metadata": {
            "name": model_name,
            "modeled_by": "generate_namcore_v060_fixtures.py",
        },
    }
    if sample_rate > 0:
        model["sample_rate"] = sample_rate
    return model


# =============================================================================
# 2. Sequential Models (GAP-01)
# =============================================================================

def make_sequential_nam(
    child_models: List[Dict[str, Any]],
    sample_rate: float = 48000.0,
    model_name: str = "Sequential",
) -> Dict[str, Any]:
    model: Dict[str, Any] = {
        "version": "0.6.0",
        "architecture": "Sequential",
        "config": {
            "models": child_models,
        },
        "weights": [],
        "metadata": {
            "name": model_name,
            "modeled_by": "generate_namcore_v060_fixtures.py",
        },
    }
    if sample_rate > 0:
        model["sample_rate"] = sample_rate
    return model


# =============================================================================
# 3. Sequential parity harnesses (GAP-01 / NC-3.4)
#
# `make_linear_nam`/`make_sequential_nam` above already carry the chain
# variants of the audited envelope; the builders below add the two
# load-bearing parity instruments the live C++ cross-validation needs:
#
# - All-unknown-rate chains (`sample_rate` omitted from the root envelope and
#   from every child): C++ resolves `GetExpectedSampleRate()` to the unknown
#   sentinel (-1) and renders at the input WAV rate, so one fixture serves the
#   full 44.1k/48k/96k (+88.2k/192k) sweep. The Rust mirror resolves DEC-01
#   identically (unknowns ignored), with the 48 kHz global default only
#   filling the diagnostic field — children without rate dependencies behave
#   rate-invariantly in both engines.
# - A declared-rate LSTM chain whose per-child stabilization counts
#   (0.5 * declared rate = 24000 @ 48 kHz) sum to an exact multiple of the C++
#   `DSP::prewarm` chunk granularity (64), so C++ whole-chunk feeding and the
#   Rust exact-count feeding traverse an identical zero-sample trajectory.
# =============================================================================

def make_wavenet_simple_nam(rng: random.Random, model_name: str = "WaveNet Stage") -> Dict[str, Any]:
    """Minimal canonical two-array A1 WaveNet child envelope without a
    declared sample rate.

    The two-array shape (array1 out CH=3 → array2 in CH=3, head out 1) is the
    one the shared f64 oracle composes canonically for A1 models: single-layer
    WaveNets route into the A2 oracle branch, and the A1 branch requires at
    least two arrays (`oracle_wavenet_forward_inner` bails to zeros for
    `layers.len() < 2`). The pair must see the same topology, hence:
      array1: input_size=1, head_size=3 (array2's input_size), head_bias=False
      array2: input_size=3, head_size=1 (final), head_bias=True
      root:   head_scale=1.0

    Per-array weight layout (C++ `LayerArray::set_weights_` / Rust mirror):
    per dilated layer: conv(ch*in*k + ch bias), input_mixin(cond*ch), 1x1
    mid(ch*ch + ch bias); then the array head Conv1D(ch*head*k_head [+bias]).
    """
    channels = 3
    dilations = [1, 2]
    kernel_size = 2
    condition_size = 1
    mid_ch = channels  # A1 1x1 mid mixer is ch->ch

    weights: List[float] = []
    # array1 (input_size 1)
    weights.extend(gen_floats(1 * channels, rng, scale=0.1))
    for _ in dilations:
        weights.extend(gen_floats(channels * (channels * kernel_size) + channels, rng, scale=0.1))
        weights.extend(gen_floats(condition_size * channels, rng, scale=0.1))
        weights.extend(gen_floats(mid_ch * channels + channels, rng, scale=0.1))
    weights.extend(gen_floats(channels * 3 * 1, rng, scale=0.1))  # head k=1, no bias
    # array2 (input_size 3)
    weights.extend(gen_floats(3 * channels, rng, scale=0.1))
    for _ in dilations:
        weights.extend(gen_floats(channels * (channels * kernel_size) + channels, rng, scale=0.1))
        weights.extend(gen_floats(condition_size * channels, rng, scale=0.1))
        weights.extend(gen_floats(mid_ch * channels + channels, rng, scale=0.1))
    weights.extend(gen_floats(channels * 1 * 1, rng, scale=0.1))  # head k=1
    weights.extend(gen_floats(1, rng, scale=0.1))                 # head bias
    weights.extend(gen_floats(1, rng, scale=0.1))                 # head_scale
    return {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "input_size": 1,
                    "condition_size": condition_size,
                    "head_size": 3,
                    "channels": channels,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": 3,
                    "condition_size": condition_size,
                    "head_size": 1,
                    "channels": channels,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 1.0,
        },
        "weights": weights,
        "metadata": {
            "name": model_name,
            "modeled_by": "generate_namcore_v060_fixtures.py",
        },
    }


def make_lstm_nam(
    rng: random.Random,
    num_layers: int,
    hidden_size: int,
    sample_rate: float = 48000.0,
    model_name: str = "LSTM Stage",
) -> Dict[str, Any]:
    """Mono legacy-format LSTM child envelope at a declared sample rate."""
    num_weights = 0
    for layer in range(num_layers):
        inp = 1 if layer == 0 else hidden_size
        ih = inp + hidden_size
        num_weights += 4 * hidden_size * ih  # input_hidden [Gate][H][IH]
        num_weights += 4 * hidden_size       # bias
        num_weights += hidden_size           # hidden_init
        num_weights += hidden_size           # cell_init
    num_weights += hidden_size               # head
    num_weights += 1                         # head bias

    weights = gen_floats(num_weights, rng, scale=0.1)

    return {
        "version": "0.6.0",
        "architecture": "LSTM",
        "sample_rate": sample_rate,
        "config": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "input_size": 1,
        },
        "weights": weights,
        "metadata": {
            "name": model_name,
            "modeled_by": "generate_namcore_v060_fixtures.py",
        },
    }


# =============================================================================
# 4. WaveNet with head_dilation and nested head.kernel_size (GAP-03)
# =============================================================================

def make_wavenet_head_dilation_nam(
    rng: random.Random,
    sample_rate: float = 48000.0,
) -> Dict[str, Any]:
    channels = 4
    bottleneck = 4
    input_size = 1
    condition_size = 1
    dilations = [1, 2]
    kernel_size = 2
    head_size = 1
    head_kernel_size = 3
    head_dilation = 2
    head_bias = True

    # Weight counting:
    # 1. Rechannel: input_size * channels
    rechannel_w = input_size * channels

    # 2. Layers: for each layer in dilations:
    #    conv: channels * (kernel_size * channels) + channels (bias)
    #    input_mixin: condition_size * channels (bias=false)
    #    layer1x1: bottleneck * channels + channels (bias=true)
    layer_w = 0
    for _ in dilations:
        conv_weights = channels * (channels * kernel_size) + channels
        in_mixin_weights = condition_size * channels
        layer1x1_weights = bottleneck * channels + channels
        layer_w += conv_weights + in_mixin_weights + layer1x1_weights

    # 3. Head rechannel Conv1D:
    #    in = bottleneck, out = head_size, kernel_size = head_kernel_size, bias = head_bias (1 if true else 0)
    head_rechannel_weights = bottleneck * head_size * head_kernel_size
    head_rechannel_bias = head_size if head_bias else 0

    # 4. Head scale
    head_scale_w = 1

    total_w = rechannel_w + layer_w + head_rechannel_weights + head_rechannel_bias + head_scale_w
    weights = gen_floats(total_w, rng, scale=0.1)

    return {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "sample_rate": sample_rate,
        "config": {
            "layers": [
                {
                    "input_size": input_size,
                    "condition_size": condition_size,
                    "channels": channels,
                    "bottleneck": bottleneck,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
                    "activation": "Tanh",
                    "head": {
                        "out_channels": head_size,
                        "kernel_size": head_kernel_size,
                        "head_dilation": head_dilation,
                        "bias": head_bias,
                    },
                }
            ],
            "head_scale": 1.0,
        },
        "weights": weights,
        "metadata": {
            "name": "WaveNet Head Dilation Fixture",
            "modeled_by": "generate_namcore_v060_fixtures.py",
        },
    }


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    rng = random.Random(42)

    # 1. Multichannel Linear (GAP-02 / NC-2.3)
    # 1.1 Direct (short RF = 8)
    l1x2 = make_linear_nam(1, 2, 8, True, rng, model_name="Linear 1->2")
    write_json(OUTPUT_DIR / "linear_1x2.nam", l1x2)

    l1x2_nobias = make_linear_nam(1, 2, 8, False, rng, model_name="Linear 1->2 (No Bias)")
    write_json(OUTPUT_DIR / "linear_1x2_nobias.nam", l1x2_nobias)

    l2x1 = make_linear_nam(2, 1, 8, True, rng, model_name="Linear 2->1")
    write_json(OUTPUT_DIR / "linear_2x1.nam", l2x1)

    l2x1_nobias = make_linear_nam(2, 1, 8, False, rng, model_name="Linear 2->1 (No Bias)")
    write_json(OUTPUT_DIR / "linear_2x1_nobias.nam", l2x1_nobias)

    l2x2 = make_linear_nam(2, 2, 8, True, rng, model_name="Linear 2->2 Shared IR")
    write_json(OUTPUT_DIR / "linear_2x2_shared.nam", l2x2)

    l2x2_nobias = make_linear_nam(2, 2, 8, False, rng, model_name="Linear 2->2 Shared IR (No Bias)")
    write_json(OUTPUT_DIR / "linear_2x2_shared_nobias.nam", l2x2_nobias)

    # 1.2 FFT (long RF = 2048)
    l1x2_fft = make_linear_nam(1, 2, 2048, True, rng, model_name="Linear 1->2 FFT")
    write_json(OUTPUT_DIR / "linear_1x2_fft.nam", l1x2_fft)

    l1x2_fft_nobias = make_linear_nam(1, 2, 2048, False, rng, model_name="Linear 1->2 FFT (No Bias)")
    write_json(OUTPUT_DIR / "linear_1x2_fft_nobias.nam", l1x2_fft_nobias)

    l2x1_fft = make_linear_nam(2, 1, 2048, True, rng, model_name="Linear 2->1 FFT")
    write_json(OUTPUT_DIR / "linear_2x1_fft.nam", l2x1_fft)

    l2x1_fft_nobias = make_linear_nam(2, 1, 2048, False, rng, model_name="Linear 2->1 FFT (No Bias)")
    write_json(OUTPUT_DIR / "linear_2x1_fft_nobias.nam", l2x1_fft_nobias)

    l2x2_fft = make_linear_nam(2, 2, 2048, True, rng, model_name="Linear 2->2 Shared IR FFT")
    write_json(OUTPUT_DIR / "linear_2x2_shared_fft.nam", l2x2_fft)

    l2x2_fft_nobias = make_linear_nam(2, 2, 2048, False, rng, model_name="Linear 2->2 Shared IR FFT (No Bias)")
    write_json(OUTPUT_DIR / "linear_2x2_shared_fft_nobias.nam", l2x2_fft_nobias)

    # 2. Sequential Models
    # A simple mono linear stage
    l1x1_a = make_linear_nam(1, 1, 4, True, rng, model_name="Stage A (1->1)")
    l1x1_b = make_linear_nam(1, 1, 4, True, rng, model_name="Stage B (1->1)")
    seq_chain = make_sequential_nam([l1x1_a, l1x1_b], model_name="Sequential Linear Chain")
    write_json(OUTPUT_DIR / "sequential_linear_chain.nam", seq_chain)

    # Multichannel sequential: 1->2 into 2->1
    seq_mc = make_sequential_nam([l1x2, l2x1], model_name="Sequential Multichannel 1->2->1")
    write_json(OUTPUT_DIR / "sequential_multichannel.nam", seq_mc)

    # Nested sequential
    seq_inner = make_sequential_nam([l1x1_b], model_name="Inner Sequential")
    seq_nested = make_sequential_nam([l1x1_a, seq_inner], model_name="Nested Sequential")
    write_json(OUTPUT_DIR / "sequential_nested.nam", seq_nested)

    # Sample rate variants
    # Homogeneous
    l1x1_48k_1 = make_linear_nam(1, 1, 4, True, rng, sample_rate=48000.0, model_name="Stage 48k 1")
    l1x1_48k_2 = make_linear_nam(1, 1, 4, True, rng, sample_rate=48000.0, model_name="Stage 48k 2")
    seq_sr_homo = make_sequential_nam([l1x1_48k_1, l1x1_48k_2], sample_rate=48000.0, model_name="Sequential Homogeneous SR")
    write_json(OUTPUT_DIR / "sequential_sr_homogeneous.nam", seq_sr_homo)

    # Mixed unknown
    l1x1_unknown = make_linear_nam(1, 1, 4, True, rng, sample_rate=-1.0, model_name="Stage Unknown SR")
    seq_sr_mixed = make_sequential_nam([l1x1_unknown, l1x1_48k_2], sample_rate=48000.0, model_name="Sequential Mixed Unknown SR")
    write_json(OUTPUT_DIR / "sequential_sr_mixed_unknown.nam", seq_sr_mixed)

    # Conflict (44.1k vs 48k)
    l1x1_44k = make_linear_nam(1, 1, 4, True, rng, sample_rate=44100.0, model_name="Stage 44.1k")
    seq_sr_conflict = make_sequential_nam([l1x1_44k, l1x1_48k_2], sample_rate=48000.0, model_name="Sequential Conflicting SR")
    write_json(OUTPUT_DIR / "sequential_sr_conflict.nam", seq_sr_conflict)

    # 4. WaveNet with head_dilation (continues the shared stream, exactly as
    # before; new fixtures below use independent streams so existing committed
    # fixture bytes can never churn from adding/removing NC-3.4 instruments)
    wn_head = make_wavenet_head_dilation_nam(rng, sample_rate=48000.0)
    write_json(OUTPUT_DIR / "wavenet_head_dilation.nam", wn_head)

    # 5. Sequential parity harnesses NC-3.4 (each fixture on its own seed)
    # Linear 1->1 into Linear 1->1 with the sample rate absent at the root and
    # in every child: the all-unknown DEC-01 branch, the multi-rate sweep
    # instrument (C++ render accepts any input WAV rate for it).
    lin2_a = make_linear_nam(1, 1, 4, True, random.Random(4042), sample_rate=-1.0, model_name="Stage A (1->1, no rate)")
    lin2_b = make_linear_nam(1, 1, 4, True, random.Random(4043), sample_rate=-1.0, model_name="Stage B (1->1, no rate)")
    seq_linear2 = make_sequential_nam([lin2_a, lin2_b], sample_rate=-1.0, model_name="Sequential Linear x2 (rates unknown)")
    write_json(OUTPUT_DIR / "sequential_linear2.nam", seq_linear2)

    # Linear into WaveNet, again fully rate-unknown: multi-rate sweep over a
    # neural child (the waveNet zero-feed trajectory is rate-invariant).
    lin_wn = make_linear_nam(1, 1, 4, True, random.Random(4044), sample_rate=-1.0, model_name="Stage Linear (no rate)")
    wn_stage = make_wavenet_simple_nam(random.Random(4045), model_name="Stage WaveNet (no rate)")
    seq_lin_wn = make_sequential_nam([lin_wn, wn_stage], sample_rate=-1.0, model_name="Sequential Linear->WaveNet (rates unknown)")
    write_json(OUTPUT_DIR / "sequential_linear_wavenet.nam", seq_lin_wn)

    # Two declared-rate LSTM stages: every child stabilizes 0.5 * 48000 =
    # 24000 samples, so the chain count sums 48000 = 750 * 64 and the C++
    # whole-chunk `DSP::prewarm` feed coincides with the exact-count Rust feed
    # (single-prewarm transient comparison at 48 kHz without chunk overshoot).
    lstm_a = make_lstm_nam(random.Random(4046), 1, 8, 48000.0, model_name="Stage LSTM 1x8 A")
    lstm_b = make_lstm_nam(random.Random(4047), 1, 8, 48000.0, model_name="Stage LSTM 1x8 B")
    seq_dbl_lstm = make_sequential_nam([lstm_a, lstm_b], sample_rate=48000.0, model_name="Sequential double LSTM 1x8")
    write_json(OUTPUT_DIR / "sequential_double_lstm.nam", seq_dbl_lstm)

    print("Successfully generated all NAMCore v0.6.0 synthetic fixtures.")


if __name__ == "__main__":
    main()
