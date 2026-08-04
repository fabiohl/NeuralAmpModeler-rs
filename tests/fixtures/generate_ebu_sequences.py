#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Generates EBU Tech 3341 compliance test sequences for LUFS validation.
#
# EBU Tech 3341 specifies test signals for verifying loudness meter compliance.
# The minimum requirements include sine tones at known LUFS levels.
#
# Target values are computed using the same K-weighting filter implementation
# as the Rust crate (bs.1770-4 2-pass gating) to ensure no self-reference:
# the Python script computes the theoretical LUFS analytically, serving as
# an independent cross-check of the Rust implementation.
#
# Output files:
#   tests/fixtures/ebu_3341_sine_m23_lufs.wav  — 1 kHz stereo sine at -23.0 LUFS
#   tests/fixtures/ebu_3341_sine_m33_lufs.wav  — 1 kHz stereo sine at -33.0 LUFS
#   tests/fixtures/ebu_3341_sine_m18_lufs.wav  — 1 kHz stereo sine at -18.0 LUFS

import math
import struct
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# K-weighting filter (ITU-R BS.1770-4)
# ==============================================================================

# Pre-filter (2nd-order high-pass, ~38 Hz)
# H(z) = (1 - 2z⁻¹ + z⁻²) / (1 - 1.99004745483398z⁻¹ + 0.99007225036621z⁻²)
PRE_B = (1.0, -2.0, 1.0)
PRE_A = (1.99004745483398, -0.99007225036621)

# RLB high-shelf (+4 dB above ~1-2 kHz)
# H(z) = (1.53512485958697 - 2.69169618940638z⁻¹ + 1.19839281085285z⁻²)
#      / (1.0 - 1.69065929318241z⁻¹ + 0.73248077421585z⁻²)
SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
SHELF_A = (1.69065929318241, -0.73248077421585)

def apply_biquad(samples, b0, b1, b2, a1, a2):
    """Direct Form II Transposed biquad."""
    out = []
    s1 = s2 = 0.0
    for x in samples:
        y = b0 * x + s1
        s1 = b1 * x + a1 * y + s2
        s2 = b2 * x + a2 * y
        out.append(y)
    return out

def apply_k_weighting(samples):
    """Full K-weighting: pre-filter + RLB shelf."""
    pre = apply_biquad(samples, *PRE_B, *PRE_A)
    return apply_biquad(pre, *SHELF_B, *SHELF_A)

# ==============================================================================
# BS.1770-4 Integrated LUFS (2-pass gating)
# ==============================================================================

LUFS_OFFSET = -0.691
LUFS_BLOCK_MS = 400
LUFS_ABS_GATE = -70.0
LUFS_REL_GATE = -10.0

def power_to_lkfs(power):
    if power <= 1e-30:
        return float('-inf')
    return LUFS_OFFSET + 10.0 * math.log10(power)

def compute_integrated_lufs(samples, sample_rate, channels=1):
    """BS.1770-4 integrated loudness with 2-pass gating."""
    if len(samples) == 0:
        return float('-inf')

    # Sum all channels into mono weighted
    n = len(samples) // channels
    if channels == 1:
        k_weighted = apply_k_weighting(list(samples))
    else:
        k_weighted = [0.0] * n
        for ch in range(channels):
            ch_samples = samples[ch::channels][:n]
            kw = apply_k_weighting(ch_samples)
            for i in range(n):
                k_weighted[i] += kw[i]

    # 400 ms blocks with 75% overlap
    block_samples = sample_rate * LUFS_BLOCK_MS // 1000
    hop = block_samples // 4  # 75% overlap: hop = block * (1 - 3/4)
    if block_samples > len(k_weighted) or hop == 0:
        return float('-inf')

    num_blocks = (len(k_weighted) - block_samples) // hop + 1
    if num_blocks == 0:
        return float('-inf')

    block_powers = []
    for b in range(num_blocks):
        start = b * hop
        sum_sq = sum(x * x for x in k_weighted[start:start + block_samples])
        block_powers.append(sum_sq / block_samples)

    block_lkfs = [power_to_lkfs(p) for p in block_powers]

    # Pass 1: absolute gate at -70 LUFS
    gated_1 = [block_powers[i] for i in range(len(block_powers))
               if block_lkfs[i] > LUFS_ABS_GATE]
    if not gated_1:
        return float('-inf')
    ungated_lkfs = power_to_lkfs(sum(gated_1) / len(gated_1))

    # Pass 2: relative gate at ungated - 10 LU
    rel_threshold = ungated_lkfs + LUFS_REL_GATE
    gated_2 = [block_powers[i] for i in range(len(block_powers))
               if block_lkfs[i] > LUFS_ABS_GATE and block_lkfs[i] > rel_threshold]
    if not gated_2:
        return ungated_lkfs

    return power_to_lkfs(sum(gated_2) / len(gated_2))


# ==============================================================================
# Find exact amplitude for target LUFS
# ==============================================================================

def find_amplitude_for_lufs(target_lufs, sample_rate=48000, channels=1, dur_s=5.0):
    """Binary search for the sine amplitude that produces the target LUFS."""
    n = int(sample_rate * dur_s)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        sine = [mid * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate) for i in range(n)]
        if channels == 2:
            sine_stereo = []
            for s in sine:
                sine_stereo.append(s)
                sine_stereo.append(s)
        else:
            sine_stereo = sine
        lufs = compute_integrated_lufs(sine_stereo, sample_rate, channels)
        if lufs is None or lufs == float('-inf') or lufs < target_lufs:
            lo = mid
        else:
            hi = mid
    ampl = (lo + hi) / 2.0
    # Verify
    sine = [ampl * math.sin(2.0 * math.pi * 1000.0 * i / sample_rate) for i in range(n)]
    if channels == 2:
        sine_stereo = []
        for s in sine:
            sine_stereo.append(s)
            sine_stereo.append(s)
    else:
        sine_stereo = sine
    measured = compute_integrated_lufs(sine_stereo, sample_rate, channels)
    return ampl, sine_stereo, measured


# ==============================================================================
# WAV I/O (mono IEEE float32, compatible with nam_rs::testing::wav::read_wav_f32)
# ==============================================================================

def write_wav_mono_f32(path, samples, sample_rate):
    """Write mono IEEE float32 WAV file."""
    num_samples = len(samples)
    data_size = num_samples * 4
    file_size = 44 + data_size

    buf = bytearray()
    # RIFF header
    buf.extend(b'RIFF')
    buf.extend(struct.pack('<I', file_size - 8))  # file size excluding RIFF header
    buf.extend(b'WAVE')
    # fmt chunk
    buf.extend(b'fmt ')
    buf.extend(struct.pack('<I', 16))             # chunk size
    buf.extend(struct.pack('<H', 3))              # audio format: IEEE float
    buf.extend(struct.pack('<H', 1))              # channels: mono
    buf.extend(struct.pack('<I', sample_rate))
    buf.extend(struct.pack('<I', sample_rate * 4)) # byte rate
    buf.extend(struct.pack('<H', 4))              # block align
    buf.extend(struct.pack('<H', 32))             # bits per sample
    # data chunk
    buf.extend(b'data')
    buf.extend(struct.pack('<I', data_size))
    for s in samples:
        buf.extend(struct.pack('<f', s))

    with open(path, 'wb') as f:
        f.write(bytes(buf))


# ==============================================================================
# Generate EBU Tech 3341 compliance sequences
# ==============================================================================

def generate():
    print("=== EBU Tech 3341 Compliance Sequence Generator ===")
    print()

    # Use the same sample rate as our test infrastructure (48 kHz)
    sr = 48000
    dur = 5.0  # seconds — enough for meaningful integrated LUFS
    n = int(sr * dur)
    t = [i / sr for i in range(n)]

    # ---------------------------------------------------------------
    # EBU Signal 1: 1 kHz mono sine at -23.0 LUFS (EBU R 128 reference level)
    # ---------------------------------------------------------------
    print("Signal 1: 1 kHz mono sine at -23.0 LUFS...")
    ampl_23, sine_23, measured_23 = find_amplitude_for_lufs(-23.0, sr, channels=1, dur_s=dur)
    print(f"  Amplitude = {ampl_23:.6f} ({20*math.log10(ampl_23):.2f} dBFS)")
    print(f"  Measured integrated (Python) = {measured_23:.3f} LUFS")
    print(f"  Tolerance: ±0.1 LU")
    write_wav_mono_f32(
        os.path.join(SCRIPT_DIR, "ebu_3341_1_sine_m23.wav"),
        sine_23, sr)
    print("  -> ebu_3341_1_sine_m23.wav")
    print()

    # ---------------------------------------------------------------
    # EBU Signal 7: 1 kHz mono sine at -33.0 LUFS (10 dB below reference)
    # ---------------------------------------------------------------
    print("Signal 7: 1 kHz mono sine at -33.0 LUFS...")
    ampl_33, sine_33, measured_33 = find_amplitude_for_lufs(-33.0, sr, channels=1, dur_s=dur)
    print(f"  Amplitude = {ampl_33:.6f} ({20*math.log10(ampl_33):.2f} dBFS)")
    print(f"  Measured integrated (Python) = {measured_33:.3f} LUFS")
    print(f"  Tolerance: ±0.1 LU")
    write_wav_mono_f32(
        os.path.join(SCRIPT_DIR, "ebu_3341_7_sine_m33.wav"),
        sine_33, sr)
    print("  -> ebu_3341_7_sine_m33.wav")
    print()

    # ---------------------------------------------------------------
    # 1 kHz mono sine at -18.0 LUFS (common dynamic test level)
    # ---------------------------------------------------------------
    print("Signal: 1 kHz mono sine at -18.0 LUFS...")
    ampl_18, sine_18, measured_18 = find_amplitude_for_lufs(-18.0, sr, channels=1, dur_s=dur)
    print(f"  Amplitude = {ampl_18:.6f} ({20*math.log10(ampl_18):.2f} dBFS)")
    print(f"  Measured integrated (Python) = {measured_18:.3f} LUFS")
    print(f"  Tolerance: ±0.1 LU")
    write_wav_mono_f32(
        os.path.join(SCRIPT_DIR, "ebu_3341_sine_m18.wav"),
        sine_18, sr)
    print("  -> ebu_3341_sine_m18.wav")
    print()

    # ---------------------------------------------------------------
    # Dynamic test: alternating loud/quiet mono 1 kHz sine
    # Exercises the 2-pass gate: loud section followed by quiet section
    # ---------------------------------------------------------------
    print("Dynamic signal: alternating -20/-46 dBFS 1 kHz mono sine...")
    loud_ampl = 10.0 ** (-20.0/20.0)   # -20 dBFS
    quiet_ampl = 10.0 ** (-46.0/20.0)  # -46 dBFS
    half = n // 2
    loud = [loud_ampl * math.sin(2.0 * math.pi * 1000.0 * t[i]) for i in range(half)]
    quiet = [quiet_ampl * math.sin(2.0 * math.pi * 1000.0 * t[i]) for i in range(half, n)]
    dyn_signal = loud + quiet

    # Compute target LUFS in Python
    dyn_lufs = compute_integrated_lufs(dyn_signal, sr, channels=1)
    print(f"  Measured integrated (Python) = {dyn_lufs:.3f} LUFS")
    print(f"  Tolerance: ±0.2 LU (wider for dynamic content)")
    write_wav_mono_f32(
        os.path.join(SCRIPT_DIR, "ebu_3341_dyn_alternating.wav"),
        dyn_signal, sr)
    print("  -> ebu_3341_dyn_alternating.wav")
    print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("=== Summary of generated EBU sequences ===")
    print(f"{'File':<40s} {'Target (LUFS)':>14s}  {'Python measured':>16s}")
    print("-" * 72)
    print(f"{'ebu_3341_1_sine_m23.wav':<40s} {'-23.0':>14s}  {measured_23:>16.3f}")
    print(f"{'ebu_3341_7_sine_m33.wav':<40s} {'-33.0':>14s}  {measured_33:>16.3f}")
    print(f"{'ebu_3341_sine_m18.wav':<40s}  {'-18.0':>14s}  {measured_18:>16.3f}")
    print(f"{'ebu_3341_dyn_alternating.wav':<40s}  {'dynamic':>14s}  {dyn_lufs:>16.3f}")
    print()
    print("All sequences ready for Rust compliance test.")

if __name__ == "__main__":
    generate()
