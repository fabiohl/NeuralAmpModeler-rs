#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fabio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

"""Generate mrstft_golden.bin for parity testing against Rust compute_mr_stft.

Mirrors the Rust implementation in src/testing/perceptual.rs:
    - Multi-resolution STFT with windows [256, 1024, 4096] @ hop=window/4
    - Weights [0.1, 0.3, 0.5]
    - Per-frame relative floor at -80 dB below spectral peak (eps_abs=1e-8 fallback)

Golden binary format (little-endian):
    [f64 MR-STFT loss]  (8 bytes)
    [u32 num_samples]   (4 bytes)
    [f32 x N ref]       (N*4 bytes)
    [f32 x N test]      (N*4 bytes)
"""

import numpy as np
import struct
import os


WINDOW_SIZES = [256, 1024, 4096]
WEIGHTS = [0.1, 0.3, 0.5]
NUM_SAMPLES = 4800
SEED = 42
EPS_ABS = 1e-8
FLOOR_DB = -80
FLOOR_FACTOR = 10.0 ** (FLOOR_DB / 20.0)  # 10^(-80/20) = 1e-4


def hann_window(ws: int) -> np.ndarray:
    n = np.arange(ws, dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (ws - 1)))


def compute_mr_stft(reference: np.ndarray, test: np.ndarray) -> float:
    assert reference.shape == test.shape
    assert reference.dtype == np.float32

    total_loss = 0.0

    for ws, weight in zip(WINDOW_SIZES, WEIGHTS):
        hop = ws // 4
        if ws > len(reference):
            continue

        window = hann_window(ws)
        num_frames = (len(reference) - ws) // hop + 1
        if num_frames == 0:
            continue

        num_bins = ws // 2 + 1

        frame_losses = np.empty(num_frames, dtype=np.float64)

        for fi in range(num_frames):
            offset = fi * hop

            ref_frame = reference[offset : offset + ws].astype(np.float64) * window
            tst_frame = test[offset : offset + ws].astype(np.float64) * window

            ref_fft = np.fft.rfft(ref_frame)
            tst_fft = np.fft.rfft(tst_frame)

            mag_ref = np.abs(ref_fft)
            mag_tst = np.abs(tst_fft)

            frame_peak = max(np.max(mag_ref).item(), np.max(mag_tst).item())

            eps_frame = max(frame_peak * FLOOR_FACTOR, EPS_ABS)

            log_ref = np.log(np.maximum(mag_ref, eps_frame))
            log_tst = np.log(np.maximum(mag_tst, eps_frame))

            diff = np.abs(log_ref - log_tst)
            l1 = np.mean(diff)
            l2 = np.sqrt(np.mean(diff * diff))

            frame_losses[fi] = l1 + l2

        window_loss = np.mean(frame_losses).item()
        total_loss += weight * window_loss

    return float(total_loss)


def main() -> None:
    rng = np.random.RandomState(SEED)

    reference = (rng.randn(NUM_SAMPLES) * 0.3).astype(np.float32)
    test = (
        reference + (rng.randn(NUM_SAMPLES) * 0.001).astype(np.float32)
    ).astype(np.float32)

    loss = compute_mr_stft(reference, test)
    print(f"MR-STFT loss: {loss:.10e}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "mrstft_golden.bin")
    output_path = os.path.normpath(output_path)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<d", loss))
        f.write(struct.pack("<I", NUM_SAMPLES))
        f.write(reference.tobytes())
        f.write(test.tobytes())

    fsize = os.path.getsize(output_path)
    expected = 8 + 4 + NUM_SAMPLES * 4 * 2
    assert fsize == expected, f"size mismatch: {fsize} != {expected}"
    print(f"Wrote {fsize} bytes to {output_path}")


if __name__ == "__main__":
    main()
