#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fabio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
"""Generate reference vectors for resampler correctness tests.

Algorithm
---------
1. Generates a linear chirp from 20 Hz to 0.45×Nyquist (min of input/output rates).
2. Fades in/out 10% to avoid spectral leakage from hard edges.
3. Resamples with ffmpeg's libsoxr engine at maximum quality (precision=33 bits,
   Chebyshev passband).

External reference engine: libsoxr 0.1.x via ffmpeg (SoX Resampler library).
Settings documented in the shell wrapper script.

Output
------
For each rate pair (from_rate → to_rate):
  resampler_input_<from_rate>.f32      — raw f32 LE mono test signal
  resampler_ref_<from_rate>_to_<to_rate>.f32  — soxr reference output
"""

import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def generate_multitone(duration: float, sample_rate: int,
                      freqs: list[float]) -> list[float]:
    """Generate a multitone signal (sum of equal-amplitude sine waves).

    Each tone has amplitude 1/N where N is the number of tones,
    preventing clipping. Quarter-sine fade in/out over 10% of duration.
    """
    n_total = int(duration * sample_rate)
    n_fade = n_total // 10
    n_tones = len(freqs)
    amp = 1.0 / n_tones  # ensure no clipping

    samples = []
    for i in range(n_total):
        t = i / sample_rate
        value = 0.0
        for f in freqs:
            value += amp * math.sin(2.0 * math.pi * f * t)

        # Quarter-sine fade in
        if i < n_fade:
            value *= math.sin(0.5 * math.pi * i / n_fade)
        # Quarter-sine fade out
        if i >= n_total - n_fade:
            j = n_total - 1 - i
            value *= math.sin(0.5 * math.pi * j / n_fade)

        samples.append(float(value))

    return samples


def generate_chirp(duration: float, sample_rate: int,
                   f_start: float, f_end: float) -> list[float]:
    """Generate a linear chirp with quarter-sine fade in/out.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : int
        Sample rate in Hz.
    f_start : float
        Start frequency in Hz.
    f_end : float
        End frequency in Hz.

    Returns
    -------
    list[float]
        f32 samples in [-1.0, 1.0].
    """
    n_total = int(duration * sample_rate)
    n_fade = n_total // 10  # 10% fade in/out

    samples = []
    for i in range(n_total):
        t = i / sample_rate
        # Linear chirp instantaneous phase
        phase = 2.0 * math.pi * (f_start * t + 0.5 * (f_end - f_start) * t * t / duration)
        value = math.sin(phase)

        # Quarter-sine fade in
        if i < n_fade:
            value *= math.sin(0.5 * math.pi * i / n_fade)
        # Quarter-sine fade out
        if i >= n_total - n_fade:
            j = n_total - 1 - i
            value *= math.sin(0.5 * math.pi * j / n_fade)

        samples.append(float(value))

    return samples


def write_raw_f32(path: Path, samples: list[float]) -> None:
    """Write samples as raw f32 little-endian."""
    data = b"".join(struct.pack("<f", s) for s in samples)
    path.write_bytes(data)


def read_raw_f32(path: Path) -> list[float]:
    """Read raw f32 little-endian file."""
    data = path.read_bytes()
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))


def resample_soxr(input_path: Path, output_path: Path,
                  in_rate: int, out_rate: int) -> None:
    """Resample raw f32 mono via ffmpeg libsoxr at maximum quality.

    Uses:
      - resampler=soxr (libsoxr engine)
      - precision=33 (maximum bits, ~200 dB SNR theoretical)
      - cheby=1 (Chebyshev passband for min. aliasing)

    These are SWResampler options, passed as ffmpeg global flags.

    Format: raw f32 LE, mono.
    """
    cmd = [
        "ffmpeg",
        "-f", "f32le",
        "-ar", str(in_rate),
        "-ac", "1",
        "-i", str(input_path),
        "-resampler", "soxr",
        "-precision", "33",
        "-cheby", "1",
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(out_rate),
        "-y",
        str(output_path),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg failed for {in_rate}→{out_rate}:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


def main() -> None:
    fixture_dir = Path(__file__).resolve().parent

    # Rate pairs: (from_rate, to_rate)
    # Covers 44.1↔48↔96 kHz as required by T8.9
    rate_pairs: list[tuple[int, int]] = [
        (44100, 48000),
        (48000, 44100),
        (48000, 96000),  # output path (nam→pw with upsampling)
        (96000, 48000),  # input path (pw→nam with downsampling)
    ]

    # Collect unique input rates to avoid generating the same chirp twice
    input_rates: set[int] = {from_rate for from_rate, _ in rate_pairs}

    # Duration: 0.5 s (~24k samples @ 48k, ~96 KB per fixture).
    # Long enough for transient trim (4k) + alignment + SNR computation.
    duration = 0.5

    # Generate input multitone signals (one per unique input rate)
    print("Generating multitone signals...")
    num_tones = 10
    for rate in sorted(input_rates):
        nyquist = rate / 2.0
        # 10 tones from 100 Hz to 0.45×Nyquist, logarithmically spaced
        f_start = 100.0
        f_end = 0.45 * nyquist
        log_start = math.log10(f_start)
        log_end = math.log10(f_end)
        freqs = [10.0 ** (log_start + (log_end - log_start) * i / (num_tones - 1))
                 for i in range(num_tones)]
        signal = generate_multitone(duration, rate, freqs)
        path = fixture_dir / f"resampler_input_{rate}.f32"
        write_raw_f32(path, signal)
        print(f"  resampler_input_{rate}.f32: {len(signal)} samples, "
              f"tones at {[round(f) for f in freqs]} Hz")

    # Generate reference outputs via ffmpeg soxr
    print("\nGenerating reference outputs (ffmpeg soxr, precision=33)...")
    for from_rate, to_rate in rate_pairs:
        input_path = fixture_dir / f"resampler_input_{from_rate}.f32"
        ref_path = fixture_dir / f"resampler_ref_{from_rate}_to_{to_rate}.f32"

        if not input_path.exists():
            print(f"ERROR: missing input {input_path}", file=sys.stderr)
            sys.exit(1)

        resample_soxr(input_path, ref_path, from_rate, to_rate)

        ref_samples = read_raw_f32(ref_path)
        expected_len_approx = int(duration * to_rate)
        print(f"  resampler_ref_{from_rate}_to_{to_rate}.f32: "
              f"{len(ref_samples)} samples (expected ~{expected_len_approx})")

        # Sanity check: output should be non-zero (has energy)
        energy = sum(s * s for s in ref_samples) / len(ref_samples)
        if energy < 1e-8:
            print(f"ERROR: reference output for {from_rate}→{to_rate} "
                  f"has near-zero energy ({energy})", file=sys.stderr)
            sys.exit(1)

    print("\nDone. Fixtures generated successfully.")


if __name__ == "__main__":
    main()
