#!/usr/bin/env python3
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
"""Deterministic WaveNet A1 geometry fixtures (S10 / P17).

Generates models accepted by NeuralAmpModelerCore that exercise the canonical
per-layer `kernel_sizes` array contract (`NAM/wavenet/model.cpp:1003-1032`):
each layer inside a layer array may use a different kernel size while the
legacy scalar `kernel_size` remains the homogeneous fallback.

The weight stream follows NAMCore's `SetWeights` order:
rechannel -> per layer (conv+bias, input_mixin, layer1x1+bias) -> head (bias on
the last array) -> head_scale.

The independent f64 anchor is produced by
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
INPUT_NAME = "wavenet_geometry_256_input.bin"


def gen_weights(n: int, rng: random.Random) -> list[float]:
    """Small deterministic f32-rounded weights."""
    return [round(rng.uniform(-1.0, 1.0) * 0.12, 6) for _ in range(n)]


def build_wavenet_per_layer_kernels(
    *,
    name: str,
    arrays: list[dict],
    head_scale: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    weights: list[float] = []

    for ai, arr in enumerate(arrays):
        ch = arr["channels"]
        input_size = arr["input_size"]
        cond = arr["condition_size"]
        kernels = arr["kernel_sizes"]
        head = arr["head_size"]

        weights += gen_weights(input_size * ch, rng)
        for k in kernels:
            weights += gen_weights(ch * ch * k + ch, rng)
            weights += gen_weights(cond * ch, rng)
            weights += gen_weights(ch * ch + ch, rng)
        weights += gen_weights(ch * head, rng)
        if arr["head_bias"]:
            weights += gen_weights(head, rng)

    weights += [head_scale]

    config_layers = []
    for arr in arrays:
        config_layers.append(
            {
                "input_size": arr["input_size"],
                "condition_size": arr["condition_size"],
                "head_size": arr["head_size"],
                "channels": arr["channels"],
                "kernel_sizes": arr["kernel_sizes"],
                "dilations": arr["dilations"],
                "activation": "Tanh",
                "gated": False,
                "head_bias": arr["head_bias"],
            }
        )

    return {
        "version": "0.6.0",
        "architecture": "WaveNet",
        "config": {"layers": config_layers, "head_scale": head_scale},
        "weights": [float(w) for w in weights],
        "metadata": {
            "name": name,
            "modeled_by": "generate_wavenet_geometry_fixtures.py",
        },
        "sample_rate": 48000.0,
    }


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


def write_bin(path: Path, samples: list[float]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(samples)))
        f.write(struct.pack(f"<{len(samples)}d", *samples))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)

    cases = [
        dict(
            name="wavenet_per_layer_kernels",
            arrays=[
                dict(
                    input_size=1,
                    condition_size=1,
                    head_size=2,
                    channels=3,
                    kernel_sizes=[2, 3, 1],
                    dilations=[1, 2, 3],
                    head_bias=False,
                ),
                dict(
                    input_size=3,
                    condition_size=1,
                    head_size=1,
                    channels=2,
                    kernel_sizes=[3, 2, 1],
                    dilations=[1, 2, 4],
                    head_bias=True,
                ),
            ],
            head_scale=0.75,
            seed=1001,
        ),
    ]

    input_signal = make_input()
    write_bin(ANCHOR_DIR / INPUT_NAME, input_signal)

    import numpy as np

    x = np.array(input_signal, dtype=np.float64)
    for case in cases:
        name = case.pop("name")
        model = build_wavenet_per_layer_kernels(name=name, **case)
        model_path = MODELS_DIR / f"{name}.nam"
        model_path.write_text(json.dumps(model, indent=2) + "\n")
        expected = wavenet_forward(model, x)
        write_bin(ANCHOR_DIR / f"{name}_256_f64.bin", expected.tolist())
        print(f"wrote {model_path} and f64 anchor ({len(expected)} samples)")


if __name__ == "__main__":
    main()
