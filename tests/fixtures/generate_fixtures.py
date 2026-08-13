#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
"""
Deterministic fixture generator for uncatalogued synthetic models — Complementary Golden Coverage.

Produces:
  lstm_1x10.nam             — LSTM 1-layer, hidden_size=10 (uncatalogued)
  lstm_2x24.nam             — LSTM 2-layer, hidden_size=24 (uncatalogued combo)
  lstm_3x8.nam              — LSTM 3-layer, hidden_size=8  (3-layer topology)
  convnet_nobn.nam          — ConvNet CH=8, batchnorm=false, Tanh
  convnet_relu.nam          — ConvNet CH=8, batchnorm=true, ReLU
  convnet_silu.nam          — ConvNet CH=8, batchnorm=true, SiLU
  linear_nobias.nam         — Linear RF=4, bias=false
  wavenet_a1_secondary_act.nam — A1 rejection fixture: valid A1 with non-null secondary_activation

All models use deterministic PRNG seeds and calibrated weight scales.

SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
"""

import json
import random
from pathlib import Path
from typing import List

OUTPUT_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def gen_weights(n: int, rng: random.Random, scale: float) -> List[float]:
    return [rng.uniform(-1.0, 1.0) * scale for _ in range(n)]


# =============================================================================
# 1. LSTM models — uncatalogued hidden sizes and 3-layer topology
#
# Weight layout: [Gate][H][IH] original layout
# Per layer:
#   input_hidden_weights: 4 * H * (input_size + H)
#   bias: 4 * H
#   hidden_init: H
#   cell_init: H
# Head:
#   head_weights: H
#   head_bias: 1
# =============================================================================

def count_lstm_weights(num_layers: int, hidden_size: int) -> int:
    h = hidden_size
    count = 0
    for i in range(num_layers):
        inp = 1 if i == 0 else h
        ih = inp + h
        count += 4 * h * ih  # input_hidden_weights [Gate][H][IH]
        count += 4 * h       # bias
        count += h            # hidden_init
        count += h            # cell_init
    count += h                # head_weights
    count += 1                # head_bias
    return count


def generate_lstm_weights(num_layers: int, hidden_size: int, rng: random.Random, scale: float, state_scale: float = 0.15) -> List[float]:
    h = hidden_size
    weights: List[float] = []

    for i in range(num_layers):
        inp = 1 if i == 0 else h
        ih = inp + h
        weights.extend(gen_weights(4 * h * ih, rng, scale=scale))
        weights.extend(gen_weights(4 * h, rng, scale=scale))
        weights.extend(gen_weights(h, rng, scale=scale * state_scale))
        weights.extend(gen_weights(h, rng, scale=scale * state_scale))

    weights.extend(gen_weights(h, rng, scale=scale))
    weights.extend(gen_weights(1, rng, scale=scale))
    return weights


def build_lstm_nam(weights: List[float], num_layers: int, hidden_size: int) -> dict:
    return {
        "version": "0.5.4",
        "architecture": "LSTM",
        "config": {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "input_size": 1,
        },
        "weights": weights,
        "metadata": {
            "name": f"LSTM Fixture ({num_layers}x{hidden_size})",
            "modeled_by": "tests/fixtures/generate_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# 2. ConvNet models — batchnorm variants and alternative activations
#
# C++ flat format: scalar `channels`, global `dilations`, `batchnorm` bool,
# `activation` string, fixed kernel_size=2.
#
# Weight layout:
# For each dilation d:
#   conv_w: ch * inp_ch * 2  (tap-interleaved)
#   if batchnorm: running_mean(ch), running_var(ch), gamma(ch), beta(ch), eps(1)
#   else: conv_bias(ch)
# Head: out_ch * ch (row-major) + out_ch bias
# =============================================================================

CONVNET_CH = 8
CONVNET_DILATIONS = [1, 2, 4, 8, 16, 32]
CONVNET_HEAD_OUT_CH = 1


def count_convnet_weights(batchnorm: bool) -> int:
    ch = CONVNET_CH
    count = 0
    for i, _d in enumerate(CONVNET_DILATIONS):
        in_ch = 1 if i == 0 else ch
        count += ch * in_ch * 2  # conv_w
        if batchnorm:
            count += ch * 4 + 1  # mean, var, gamma, beta, eps
        else:
            count += ch          # conv_b
    count += CONVNET_HEAD_OUT_CH * ch  # head.weight
    count += CONVNET_HEAD_OUT_CH       # head.bias
    return count


def generate_convnet_weights(batchnorm: bool, rng: random.Random, conv_scale: float = 0.20, bias_scale: float = 0.04, bn_scale: float = 0.02) -> List[float]:
    ch = CONVNET_CH
    weights: List[float] = []

    for i, _d in enumerate(CONVNET_DILATIONS):
        in_ch = 1 if i == 0 else ch
        for out_i in range(ch):
            for in_j in range(in_ch):
                weights.append(rng.uniform(-1.0, 1.0) * conv_scale)
                weights.append(rng.uniform(-1.0, 1.0) * conv_scale)

        if batchnorm:
            weights.extend(gen_weights(ch, rng, bn_scale))
            weights.extend([max(0.5, min(1.5, rng.uniform(0.98, 1.02))) for _ in range(ch)])
            weights.extend([max(0.9, min(1.1, rng.uniform(0.97, 1.03))) for _ in range(ch)])
            weights.extend(gen_weights(ch, rng, bn_scale))
            weights.append(1e-5)
        else:
            weights.extend(gen_weights(ch, rng, bias_scale))

    for out_i in range(CONVNET_HEAD_OUT_CH):
        for in_j in range(ch):
            weights.append(rng.uniform(-1.0, 1.0) * 0.3)
    for out_i in range(CONVNET_HEAD_OUT_CH):
        weights.append(rng.uniform(-0.1, 0.1))

    return weights


def build_convnet_nam(weights: List[float], batchnorm: bool, activation: str) -> dict:
    return {
        "version": "0.5.4",
        "architecture": "ConvNet",
        "config": {
            "channels": CONVNET_CH,
            "dilations": CONVNET_DILATIONS,
            "batchnorm": batchnorm,
            "activation": activation,
        },
        "weights": weights,
        "metadata": {
            "name": f"ConvNet Fixture (bn={batchnorm}, act={activation})",
            "modeled_by": "tests/fixtures/generate_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# 3. Linear model without bias
# =============================================================================

LINEAR_RF = 4


def build_linear_nobias_nam(weights: List[float]) -> dict:
    return {
        "version": "0.5.4",
        "architecture": "Linear",
        "config": {
            "receptive_field": LINEAR_RF,
            "bias": False,
        },
        "weights": weights,
        "metadata": {
            "name": "Linear Fixture (bias=false)",
            "modeled_by": "tests/fixtures/generate_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# 4. A1 rejection fixture — WaveNet A1 with non-trivial secondary_activation
#
# This is a valid WaveNet A1 model except for `secondary_activation: "Tanh"`
# in layer 0, which the topology validator must reject.
# =============================================================================

def build_a1_secondary_act_nam() -> dict:
    """Creates a valid minified WaveNet A1 model (CH=3, 2-layer) with a
    non-trivial secondary_activation set on layer 0."""
    # Minimal 2-layer A1 model: CH=[3,3], dilations=[1,2], no gating, Tanh
    return {
        "version": "0.5.4",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": 3,
                    "channels": 3,
                    "kernel_size": 3,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                    "secondary_activation": "Tanh",
                },
                {
                    "input_size": 3,
                    "condition_size": 1,
                    "head_size": 1,
                    "channels": 3,
                    "kernel_size": 3,
                    "dilations": [1, 2],
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head_scale": 0.02,
        },
        "weights": [],
        "metadata": {
            "name": "A1 secondary_activation rejection fixture",
            "modeled_by": "tests/fixtures/generate_fixtures.py",
        },
        "sample_rate": 48000,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # --- LSTM 1×10 (uncatalogued hidden size) ---
    rng = random.Random(101)
    nl, hs = 1, 10
    expected = count_lstm_weights(nl, hs)
    w = generate_lstm_weights(nl, hs, rng, scale=0.15)
    assert len(w) == expected, f"LSTM {nl}x{hs}: got {len(w)}, expected {expected}"
    doc = build_lstm_nam(w, nl, hs)
    out = OUTPUT_DIR / "lstm_1x10.nam"
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"Written {out}  ({len(w)} weights)")

    # --- LSTM 2×24 (uncatalogued combo) ---
    rng = random.Random(202)
    nl, hs = 2, 24
    expected = count_lstm_weights(nl, hs)
    w = generate_lstm_weights(nl, hs, rng, scale=0.15)
    assert len(w) == expected, f"LSTM {nl}x{hs}: got {len(w)}, expected {expected}"
    doc = build_lstm_nam(w, nl, hs)
    out = OUTPUT_DIR / "lstm_2x24.nam"
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"Written {out}  ({len(w)} weights)")

    # --- LSTM 3×8 (3-layer topology) ---
    rng = random.Random(303)
    nl, hs = 3, 8
    expected = count_lstm_weights(nl, hs)
    w = generate_lstm_weights(nl, hs, rng, scale=0.15)
    assert len(w) == expected, f"LSTM {nl}x{hs}: got {len(w)}, expected {expected}"
    doc = build_lstm_nam(w, nl, hs)
    out = OUTPUT_DIR / "lstm_3x8.nam"
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"Written {out}  ({len(w)} weights)")

    # --- ConvNet no BatchNorm, Tanh ---
    rng = random.Random(404)
    bn = False
    act = "Tanh"
    expected = count_convnet_weights(batchnorm=bn)
    w = generate_convnet_weights(batchnorm=bn, rng=rng, conv_scale=0.50, bias_scale=0.15)
    assert len(w) == expected, f"ConvNet nobn: got {len(w)}, expected {expected}"
    doc = build_convnet_nam(w, batchnorm=bn, activation=act)
    out = OUTPUT_DIR / "convnet_nobn.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w)} weights)")

    # --- ConvNet with ReLU ---
    rng = random.Random(505)
    bn = True
    act = "ReLU"
    expected = count_convnet_weights(batchnorm=bn)
    w = generate_convnet_weights(batchnorm=bn, rng=rng)
    assert len(w) == expected, f"ConvNet relu: got {len(w)}, expected {expected}"
    doc = build_convnet_nam(w, batchnorm=bn, activation=act)
    out = OUTPUT_DIR / "convnet_relu.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w)} weights)")

    # --- ConvNet with SiLU ---
    rng = random.Random(606)
    bn = True
    act = "SiLU"
    expected = count_convnet_weights(batchnorm=bn)
    w = generate_convnet_weights(batchnorm=bn, rng=rng)
    assert len(w) == expected, f"ConvNet silu: got {len(w)}, expected {expected}"
    doc = build_convnet_nam(w, batchnorm=bn, activation=act)
    out = OUTPUT_DIR / "convnet_silu.nam"
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Written {out}  ({len(w)} weights)")

    # --- Linear no bias ---
    rng = random.Random(707)
    # linear_nobias: RF=4, bias=false → only 4 weights (no bias scalar)
    rf = LINEAR_RF
    w = gen_weights(rf, rng, scale=0.30)
    doc = build_linear_nobias_nam(w)
    out = OUTPUT_DIR / "linear_nobias.nam"
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"Written {out}  ({len(w)} weights)")

    # --- A1 secondary_activation rejection fixture ---
    doc = build_a1_secondary_act_nam()
    out = OUTPUT_DIR / "wavenet_a1_secondary_act.nam"
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"Written {out}  (rejection fixture — empty weights)")

    print("Done. All synthetic topology fixtures generated.")


if __name__ == "__main__":
    main()
