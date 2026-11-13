#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
"""Deterministic post-stack head fixtures (S09 / P14).

Generates WaveNet A1 models accepted by NeuralAmpModelerCore with a canonical
post-stack `config.head` (`kernel_sizes` array), exercising:

  * a single stage and multiple stages,
  * intermediate widths distinct from the layer-array head sizes and from the
    final `out_channels`,
  * implicit head Conv1D biases,
  * non-linear activations (Tanh, ReLU, Softsign),
  * non-unit and negative `head_scale`.

Weights are emitted in the exact NAMCore stream order:
layer arrays -> post-stack head stages (weights + bias) -> head_scale.

The independent f64 anchors are produced by
`tests/fixtures/scripts/validate_oracle_f64.py` (NumPy implementation written
from the weight-format specification, not from the Rust production parser).
"""

import json
import math
import random
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
ANCHOR_DIR = HERE / "f64_anchors"
sys.path.insert(0, str(HERE / "scripts"))

from validate_oracle_f64 import wavenet_forward  # noqa: E402

NUM_SAMPLES = 256
INPUT_NAME = "post_stack_head_256_input.bin"


def gen_weights(n: int, rng: random.Random) -> list[float]:
    """Small deterministic f32-rounded weights."""
    return [round(rng.uniform(-1.0, 1.0) * 0.12, 6) for _ in range(n)]


def build_wavenet_post_head(
    *,
    name: str,
    channels: int,
    dilations: list[int],
    kernel_size: int,
    head_channels: int,
    out_channels: int,
    head_kernels: list[int],
    activation: str,
    head_scale: float,
    bias: bool,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    head0 = channels
    head1 = out_channels
    weights: list[float] = []

    # Array 0: input 1 -> channels, head projection to head0 (no bias).
    weights += gen_weights(1 * channels, rng)
    for _ in dilations:
        weights += gen_weights(channels * channels * kernel_size + channels, rng)
        weights += gen_weights(1 * channels, rng)
        weights += gen_weights(channels * channels + channels, rng)
    weights += gen_weights(channels * head0, rng)

    # Array 1: channels -> channels, head projection to head1 (with bias).
    weights += gen_weights(channels * channels, rng)
    for _ in dilations:
        weights += gen_weights(channels * channels * kernel_size + channels, rng)
        weights += gen_weights(1 * channels, rng)
        weights += gen_weights(channels * channels + channels, rng)
    weights += gen_weights(channels * head1, rng)
    weights += gen_weights(head1, rng)

    # Canonical post-stack head: per-stage [out][in][kernel] + bias[out].
    stage_in = head1
    for i, k in enumerate(head_kernels):
        stage_out = out_channels if i + 1 == len(head_kernels) else head_channels
        weights += gen_weights(stage_out * stage_in * k, rng)
        weights += gen_weights(stage_out, rng)
        stage_in = stage_out
    weights += [head_scale]

    model = {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {
            "layers": [
                {
                    "input_size": 1,
                    "condition_size": 1,
                    "head_size": head0,
                    "channels": channels,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": False,
                },
                {
                    "input_size": head0,
                    "condition_size": 1,
                    "head_size": head1,
                    "channels": channels,
                    "kernel_size": kernel_size,
                    "dilations": dilations,
                    "activation": "Tanh",
                    "gated": False,
                    "head_bias": True,
                },
            ],
            "head": {
                "channels": head_channels,
                "out_channels": out_channels,
                "kernel_sizes": head_kernels,
                "activation": activation,
                "bias": bias,
            },
            "head_scale": head_scale,
        },
        "weights": [float(w) for w in weights],
        "metadata": {"name": name, "modeled_by": "generate_post_stack_head_fixtures.py"},
        "sample_rate": 48000.0,
    }
    return model


def make_input() -> list[float]:
    """Deterministic 256-sample stimulus: chirp + low-E harmonics + impulse."""
    samples = []
    for i in range(NUM_SAMPLES):
        t = i / 48000.0
        chirp = 0.35 * math.sin(2.0 * math.pi * (220.0 + 2200.0 * t) * t)
        low_e = 0.25 * math.sin(2.0 * math.pi * 82.41 * t)
        harmonic = 0.12 * math.sin(2.0 * math.pi * 164.82 * t + 0.3)
        transient = 0.8 if i == 64 else 0.0
        samples.append(chirp + low_e + harmonic + transient)
    return samples


def write_input_bin(path: Path, samples: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(samples)))
        f.write(struct.pack(f"<{len(samples)}d", *samples))


def write_output_bin(path: Path, samples: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(samples)))
        f.write(struct.pack(f"<{len(samples)}d", *samples))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        dict(
            name="wavenet_post_head_k13",
            channels=3,
            dilations=[1, 2],
            kernel_size=2,
            head_channels=4,
            out_channels=1,
            head_kernels=[1, 3],
            activation="Tanh",
            head_scale=-0.7,
            bias=True,
            seed=901,
        ),
        dict(
            name="wavenet_post_head_single_k3",
            channels=2,
            dilations=[1],
            kernel_size=2,
            head_channels=3,
            out_channels=1,
            head_kernels=[3],
            activation="ReLU",
            head_scale=0.6,
            bias=True,
            seed=902,
        ),
        dict(
            name="wavenet_post_head_multistage",
            channels=3,
            dilations=[1, 2],
            kernel_size=2,
            head_channels=2,
            out_channels=1,
            head_kernels=[3, 1, 3],
            activation="Softsign",
            head_scale=-0.35,
            bias=True,
            seed=903,
        ),
    ]

    input_signal = make_input()
    write_input_bin(ANCHOR_DIR / INPUT_NAME, input_signal)

    import numpy as np

    x = np.array(input_signal, dtype=np.float64)
    for case in cases:
        name = case.pop("name")
        model = build_wavenet_post_head(name=name, **case)
        model_path = MODELS_DIR / f"{name}.nam"
        model_path.write_text(json.dumps(model, indent=2) + "\n")
        expected = wavenet_forward(model, x)
        write_output_bin(ANCHOR_DIR / f"{name}_256_f64.bin", expected.tolist())
        print(f"wrote {model_path} and f64 anchor ({len(expected)} samples)")


if __name__ == "__main__":
    main()
