#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

"""Reference oracle external validation — independent NumPy f64 anchor.

Independent f64 implementation of the NAM topology (WaveNet/LSTM/A2), used as
the external ground truth for the Rust f64 reference oracle.

INDEPENDENCE IS THE POINT (see docs/perceptual_validation.md, Gate Calibration
Policy Rule 6). This script must implement each architecture from the NAM weight
*format spec* — NOT by mirroring the Rust oracle's buffer layout or index
arithmetic. The weight byte order is a fixed fact (determine it empirically by
checking which reading matches the f32 production engine), but the computation
must stay idiomatic NumPy so that a shared conceptual bug cannot pass silently
in both. A historic violation (this script copying the old Rust shared-buffer
layout) made the S5 anchor circular and hid a real bug until T8.2; see T8.13.

Validation requirement (NOT optional): after any change to the oracle or this
script, the generated anchor must satisfy BOTH
  * ESR(anchor vs Rust oracle)      < 1e-12   (numerical agreement), AND
  * ESR(anchor vs production f32)    ≈ f32/f16c floor (independent agreement).
Regenerate the checked-in anchors in tests/fixtures/f64_anchors/ and keep the
three test_oracle_vs_python_anchor_* tests un-ignored.

ANCHOR REGENERATION POLICY (docs/cpp_parity_map.md §4.5.1):
A f64 anchor may only be regenerated when (a) a C++ golden exists AND
test_summary_table passes before regeneration; OR (b) no C++ golden exists
and the regeneration is documented in the commit with
before/after numbers. Regenerating from the oracle itself is circular.

Usage:
    python3 validate_oracle_f64.py <model.nam> <input.bin> --architecture <WaveNet|LSTM|A2>
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


def load_nam_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_weights_as_f64(model: dict) -> np.ndarray:
    """Load NAM model weights preserving f32 binary precision.

    JSON decimal representation of f32 values loses ~1e-8 in f64.
    Parse as f32 first, then convert to f64, matching the Rust oracle
    which stores weights as f32 and converts to f64 at compute time.
    """
    raw = np.array(model["weights"], dtype=np.float32)
    return np.array(raw, dtype=np.float64)


def read_input_bin(path: str) -> np.ndarray:
    """Read binary input signal: [u32 num_samples] [f64*N samples]."""
    with open(path, "rb") as f:
        num = struct.unpack("<I", f.read(4))[0]
        data = np.frombuffer(f.read(num * 8), dtype=np.float64)
    return data


def write_output_bin(path: str, data: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(data)))
        f.write(data.astype(np.float64).tobytes())


# ── WaveNet f64 model ─────────────────────────────────────────────────────

def wavenet_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """WaveNet forward pass in NumPy f64 — mono output."""
    return _wavenet_core(model, x, all_channels=False)


def wavenet_forward_all_channels(model: dict, x: np.ndarray) -> np.ndarray:
    """WaveNet forward pass — all head output channels (condition_dsp)."""
    return _wavenet_core(model, x, all_channels=True)


def _wavenet_core(model: dict, x: np.ndarray, *, all_channels: bool) -> np.ndarray:
    """WaveNet forward pass in NumPy f64.

    Independent f64 implementation of the NAM WaveNet topology (cross-checked
    against both the Rust oracle and the f32 production engine):
    - [out_ch][in_ch][kernel] conv weights; [head_ch][ch] head; [out][in] rechannel
    - Per-layer history buffers (each layer reads its own buffer, writes the
      residual sum into the next layer's buffer)
    - Array N seeds its head accumulator with array N-1's head projection
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    layers = config.get("layers", [])

    if not layers:
        return np.zeros_like(x)

    a_ch = [int(lc.get("channels", 8)) for lc in layers]
    a_head = [int(lc.get("head_size", 8)) for lc in layers]
    a_k = [int(lc.get("kernel_size", 3)) for lc in layers]
    a_dil = [lc.get("dilations", [1, 2, 4, 8]) for lc in layers]
    a_cond = [int(lc.get("condition_size", 1)) for lc in layers]

    a_rf = [sum((a_k[ai] - 1) * d for d in a_dil[ai]) for ai in range(len(layers))]
    max_rf = max(a_rf) + 64
    max_ch = max(a_ch)

    num_frames = len(x)
    cursor = 0

    # ── condition_dsp sub-model (T1.2) ──
    cond_dsp_out = None
    if not all_channels and config.get("condition_dsp") is not None:
        import copy
        cond_model = copy.deepcopy(config["condition_dsp"])
        dsp_arch = cond_model.get("architecture", None)
        raw = None
        if dsp_arch == "WaveNet":
            dsp_cfg = cond_model.get("config", {})
            dsp_layers_cfg = dsp_cfg.get("layers", [])
            is_a2 = dsp_layers_cfg and dsp_cfg.get("head_scale") is not None
            if is_a2 and len(dsp_layers_cfg) > 1:
                is_a2 = any(
                    l.get("head1x1", {}).get("active", False)
                    or any(l.get(k, {}).get("active", False) for k, _ in FILM_KEYS)
                    for l in dsp_layers_cfg
                ) or (
                    isinstance(dsp_layers_cfg[0].get("activation"), str)
                    and dsp_layers_cfg[0].get("activation") not in ("Tanh", "HardTanh", "FastTanh")
                )
            if not is_a2:
                raw = wavenet_forward_all_channels(cond_model, x)
            else:
                raw = forward_dispatch(cond_model, x)
        elif dsp_arch == "LSTM":
            raw = lstm_forward(cond_model, x)
        else:
            raw = forward_dispatch(cond_model, x)
        if raw is not None:
            first_cond = max(1, a_cond[0])
            if first_cond > 1 and len(raw) == num_frames:
                cond_dsp_out = np.tile(raw, (first_cond, 1)).T.ravel()
            else:
                cond_dsp_out = raw

    a_rech = []
    a_lws = []
    a_head_w = []
    a_head_b = []

    for ai in range(len(layers)):
        ch = a_ch[ai]
        head_ch = a_head[ai]
        k = a_k[ai]
        in_ch = 1 if ai == 0 else a_ch[ai - 1]

        # Rechannel weight: [out_ch][in_ch] (NAM order, matches production).
        rech_w = weights[cursor : cursor + in_ch * ch].reshape(ch, in_ch)
        cursor += in_ch * ch
        a_rech.append(rech_w)

        lws = []
        for dil in a_dil[ai]:
            cond_i = a_cond[ai]
            conv_w = weights[cursor : cursor + ch * ch * k].reshape(ch, ch, k)
            cursor += ch * ch * k
            conv_b = weights[cursor : cursor + ch]
            cursor += ch
            mixin_w = weights[cursor : cursor + cond_i * ch]
            cursor += cond_i * ch
            l1x1_w = weights[cursor : cursor + ch * ch].reshape(ch, ch)
            cursor += ch * ch
            l1x1_b = weights[cursor : cursor + ch]
            cursor += ch
            lws.append({
                "conv_w": conv_w,
                "conv_b": conv_b,
                "mixin_w": mixin_w,
                "l1x1_w": l1x1_w,
                "l1x1_b": l1x1_b,
                "dilation": dil,
            })
        a_lws.append(lws)

        # Head projection weight: [head_ch][ch] (NAM order, matches production).
        hw = weights[cursor : cursor + ch * head_ch].reshape(head_ch, ch)
        cursor += ch * head_ch
        a_head_w.append(hw)
        hb = np.zeros(head_ch, dtype=np.float64)
        if ai == len(layers) - 1:
            hb = weights[cursor : cursor + head_ch]
            cursor += head_ch
        a_head_b.append(hb)

    # head_scale is the LAST weight in the WaveNet weight stream.
    # The JSON config's head_scale field is metadata and may differ
    # from the weight-stream value (e.g. test-script-generated models
    # where random weights overwrite the config default). Production
    # engines always use the weight-stream value.
    head_scale = np.float64(weights[cursor])
    cursor += 1

    # Per-layer history buffers (matches production and the Rust oracle).
    # A single shared buffer is INCORRECT for dilated convolutions: with
    # layer-outer/frame-inner iteration, a layer's kernel would read past
    # frames already updated by that same layer's residual write, creating a
    # spurious IIR feedback. Each layer therefore reads its own buffer and
    # writes the residual sum into the next layer's buffer.
    bs = max_rf
    buf_size = max_rf + num_frames + 64

    ch_outs = []
    head_proj_out = []

    for ai in range(len(layers)):
        ch = a_ch[ai]
        head_ch = a_head[ai]
        num_li = len(a_lws[ai])
        # bufs[0] = rechannel output; bufs[li+1] = layer li output
        bufs = [np.zeros(buf_size * ch, dtype=np.float64) for _ in range(num_li + 1)]

        # Rechannel into bufs[0]
        if ai == 0:
            for f in range(num_frames):
                idx = bs + f
                for c in range(ch):
                    bufs[0][idx * ch + c] = x[f] * a_rech[ai][c, 0]
        else:
            prev_ch_out = ch_outs[ai - 1]
            for f in range(num_frames):
                idx = bs + f
                for c in range(ch):
                    bufs[0][idx * ch + c] = np.dot(a_rech[ai][c], prev_ch_out[f])

        ha = np.zeros((num_frames, ch), dtype=np.float64)

        # Layer-outer / frame-inner iteration
        for li, lw in enumerate(a_lws[ai]):
            hist = bufs[li]
            k = a_k[ai]
            dil = lw["dilation"]
            for f in range(num_frames):
                idx = bs + f

                # Conv1d + bias (reads this layer's own input buffer)
                cv = lw["conv_b"].copy()
                for oc in range(ch):
                    for kt in range(k):
                        off = dil * (kt + 1 - k)
                        ins = (idx + off) * ch
                        if ins >= 0 and ins + ch <= len(hist):
                            cv[oc] += np.dot(
                                hist[ins : ins + ch],
                                lw["conv_w"][oc, :, kt],
                            )

                # Mixin — use condition_dsp output when available
                cond_i = a_cond[ai]
                if cond_i == 1:
                    cv += x[f] * lw["mixin_w"]
                elif cond_dsp_out is not None:
                    off = f * cond_i
                    mix_mat = lw["mixin_w"].reshape(ch, cond_i)
                    cv += np.dot(mix_mat, cond_dsp_out[off : off + cond_i])
                else:
                    cv += x[f] * lw["mixin_w"]

                # Tanh
                cv = np.tanh(cv)

                # Head accumulate — cascaded seed from previous array
                if li == 0 and ai > 0:
                    ha[f] = head_proj_out[ai - 1][f] + cv
                elif li == 0:
                    ha[f] = cv
                else:
                    ha[f] = ha[f] + cv

                # L1x1 residual → next layer's buffer (bufs[li] + residual)
                for oc in range(ch):
                    val = lw["l1x1_b"][oc]
                    for ic in range(ch):
                        val += cv[ic] * lw["l1x1_w"][oc, ic]
                    bufs[li + 1][idx * ch + oc] = bufs[li][idx * ch + oc] + val

        # Save per-channel residual output (last layer's buffer) for next array
        ch_out = np.zeros((num_frames, ch), dtype=np.float64)
        last = bufs[num_li]
        for f in range(num_frames):
            idx = bs + f
            ch_out[f] = last[idx * ch : idx * ch + ch]
        ch_outs.append(ch_out)

        # Head projection: proj[f, hc] = sum_c ha[f, c] * head_w[hc, c]
        proj = ha @ a_head_w[ai].T + a_head_b[ai]
        head_proj_out.append(proj)

    if all_channels:
        output = head_proj_out[-1] * head_scale
        # Interleaved: [ch0_f0, ch1_f0, ..., chN_f0, ch0_f1, ...]
        return output.ravel(order='C')
    else:
        output = head_proj_out[-1][:, 0] * head_scale
        return output


# ── LSTM f64 model ─────────────────────────────────────────────────────────

def lstm_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """LSTM forward pass in NumPy f64."""
    config = model["config"]
    weights = load_weights_as_f64(model)
    h = int(config.get("hidden_size", 16))
    num_layers = int(config.get("num_layers", 1))
    num_frames = len(x)

    cursor = 0

    layers = []
    for l in range(num_layers):
        in_size = 1 if l == 0 else h
        ih = in_size + h
        h4 = 4 * h

        n_w = h4 * ih
        # NAM/PyTorch standard ("Original") LSTM weight layout: [gate][H][IH]
        # (rows = 4*hidden output units, cols = input+hidden). JSON .nam models
        # are always Original; the GateMajor [gate][IH][H] layout is binary-only.
        raw_w = weights[cursor : cursor + n_w].reshape(4, h, ih)
        cursor += n_w

        bias = weights[cursor : cursor + h4]
        cursor += h4
        hidden = weights[cursor : cursor + h]
        cursor += h
        cell = weights[cursor : cursor + h]
        cursor += h

        layers.append(
            {
                "w": raw_w,  # [4][ih][h]
                "bias": bias.reshape(4, h),
                "hidden": hidden.copy(),
                "cell": cell.copy(),
                "in_size": in_size,
            }
        )

    head_w = weights[cursor : cursor + h]
    cursor += h
    head_b = weights[cursor]
    cursor += 1

    output = np.zeros(num_frames, dtype=np.float64)

    for f in range(num_frames):
        x_val = x[f]

        for l, layer in enumerate(layers):
            ins = layer["in_size"]
            ih = ins + h

            state = np.zeros(ih, dtype=np.float64)
            if l == 0:
                state[0] = x_val
            else:
                state[:ins] = layers[l - 1]["hidden"]
            state[ins:] = layer["hidden"]

            # GEMV: gates[g, i] = bias[g, i] + sum_j state[j] * w[g, i, j]
            # ("Original" layout: w[gate, out_unit, in_index])
            gates = layer["bias"].copy()
            for g in range(4):
                for i in range(h):
                    for j in range(ih):
                        gates[g, i] += state[j] * layer["w"][g, i, j]

            # Fused LSTM gates
            for i in range(h):
                ig = 1.0 / (1.0 + np.exp(-gates[0, i]))
                fg = 1.0 / (1.0 + np.exp(-gates[1, i]))
                gv = np.tanh(gates[2, i])
                og = 1.0 / (1.0 + np.exp(-gates[3, i]))

                nc = fg * layer["cell"][i] + ig * gv
                hv = og * np.tanh(nc)

                layer["cell"][i] = nc
                layer["hidden"][i] = hv

        last_hidden = layers[-1]["hidden"]
        y = head_b
        for i in range(h):
            y += last_hidden[i] * head_w[i]
        output[f] = y

    return output


# ── ConvNet f64 model ──────────────────────────────────────────────────────

def convnet_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """ConvNet forward pass in NumPy f64.

    Independent f64 implementation of the NAM ConvNet topology, matching the
    Rust oracle (src/testing/reference_oracle/convnet.rs):
    - Supports both Layers format (pre-fused BN) and FlatCpp format (raw BN params)
    - [out_ch][in_ch][kernel] conv weight layout
    - Fused BatchNorm: scale * x + offset
    - Causal Conv1d with dilation
    - Per-block history buffers
    - Optional PostStackHead (Layers) or linear head (FlatCpp)
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))
    layers = config.get("layers", [])
    conv_channels = config.get("channels")
    conv_dilations = config.get("dilations")
    conv_batchnorm = config.get("batchnorm")

    is_flat_cpp = (not layers) and (conv_channels is not None) and (conv_dilations is not None) and (conv_batchnorm is not None)

    if not is_flat_cpp and not layers:
        return np.zeros_like(x)

    cursor = 0

    class BlockW:
        pass

    blocks = []
    if is_flat_cpp:
        ch = int(conv_channels)
        dilations = conv_dilations
        batchnorm = conv_batchnorm
        kernel = 2
        head_scale = np.float64(1.0)

        for i in range(len(dilations)):
            b = BlockW()
            b.out_ch = ch
            b.in_ch = 1 if i == 0 else ch
            b.kernel = kernel
            b.dilation = int(dilations[i])
            b.activation = "Tanh"

            n_conv_w = b.in_ch * b.out_ch * b.kernel
            b.conv_w = weights[cursor : cursor + n_conv_w].reshape(b.out_ch, b.in_ch, b.kernel)
            cursor += n_conv_w

            if batchnorm:
                running_mean = weights[cursor : cursor + b.out_ch]
                cursor += b.out_ch
                running_var = weights[cursor : cursor + b.out_ch]
                cursor += b.out_ch
                gamma = weights[cursor : cursor + b.out_ch]
                cursor += b.out_ch
                beta = weights[cursor : cursor + b.out_ch]
                cursor += b.out_ch
                eps = weights[cursor]
                cursor += 1

                b.bn_scale = gamma / np.sqrt(eps + running_var)
                b.bn_offset = beta - b.bn_scale * running_mean
                b.conv_b = np.zeros(b.out_ch, dtype=np.float64)
            else:
                b.conv_b = weights[cursor : cursor + b.out_ch]
                cursor += b.out_ch
                b.bn_scale = np.ones(b.out_ch, dtype=np.float64)
                b.bn_offset = np.zeros(b.out_ch, dtype=np.float64)

            blocks.append(b)

        # FlatCpp head: linear [1 × last_out_ch] + bias, no activation
        last_out_ch = blocks[-1].out_ch
        h_w = weights[cursor : cursor + last_out_ch].reshape(1, last_out_ch, 1)
        cursor += last_out_ch
        h_b = weights[cursor : cursor + 1]
        cursor += 1
        has_head = True
        h_in_ch = last_out_ch
        h_out_ch = 1
        h_kernel = 1
        h_activation = "Linear"
    else:
        for i, lc in enumerate(layers):
            b = BlockW()
            b.out_ch = int(lc.get("channels", 8))
            b.in_ch = 1 if i == 0 else int(layers[i - 1].get("channels", b.out_ch))
            b.kernel = int(lc.get("kernel_size", 3))
            use_dil = lc.get("dilations", [1])
            b.dilation = int(use_dil[0])  # support int or list
            b.activation = lc.get("activation", "Tanh")

            n_conv_w = b.in_ch * b.out_ch * b.kernel
            b.conv_w = weights[cursor : cursor + n_conv_w].reshape(b.out_ch, b.in_ch, b.kernel)
            cursor += n_conv_w

            b.conv_b = weights[cursor : cursor + b.out_ch]
            cursor += b.out_ch

            b.bn_scale = weights[cursor : cursor + b.out_ch]
            cursor += b.out_ch

            b.bn_offset = weights[cursor : cursor + b.out_ch]
            cursor += b.out_ch

            blocks.append(b)

        head_config = config.get("head")
        has_head = head_config is not None
        if has_head:
            last_out_ch = blocks[-1].out_ch
            h_in_ch = int(head_config.get("channels", last_out_ch))
            h_out_ch = int(head_config.get("out_channels", 1))
            h_kernel = int(head_config.get("kernel_size", 1))
            h_has_bias = head_config.get("bias", True)
            h_activation = head_config.get("activation", "Tanh")

            n_h_w = h_in_ch * h_out_ch * h_kernel
            h_w = weights[cursor : cursor + n_h_w].reshape(h_out_ch, h_in_ch, h_kernel)
            cursor += n_h_w

            if h_has_bias:
                h_b = weights[cursor : cursor + h_out_ch]
                cursor += h_out_ch
            else:
                h_b = np.zeros(h_out_ch, dtype=np.float64)

    num_frames = len(x)
    max_rf = max((b.kernel - 1) * b.dilation for b in blocks) + 64
    hist_size = max_rf + num_frames + 64

    # Per-block history buffers
    block_hists = [np.zeros(hist_size * b.in_ch, dtype=np.float64) for b in blocks]

    output = np.zeros(num_frames, dtype=np.float64)

    def apply_activation(data: np.ndarray, name: str) -> np.ndarray:
        if name == "Tanh":
            return np.tanh(data)
        elif name == "HardTanh":
            return np.clip(data, -1.0, 1.0)
        elif name == "FastTanh":
            return np.tanh(data)
        elif name == "ReLU":
            return np.maximum(data, 0.0)
        elif name == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-data))
        elif name == "SiLU":
            s = 1.0 / (1.0 + np.exp(-data))
            return data * s
        elif name == "HardSwish":
            relu6 = np.clip(data + 3.0, 0.0, 6.0)
            return data * relu6 / 6.0
        elif name == "Softsign":
            return data / (1.0 + np.abs(data))
        elif name in ("Linear", "Identity"):
            return data
        else:
            return np.tanh(data)

    for f in range(num_frames):
        hist_i = max_rf + f

        # Feed input into block 0 history
        block_hists[0][hist_i * blocks[0].in_ch] = x[f]

        last_out = None

        for bi, b in enumerate(blocks):
            hist = block_hists[bi]
            out_ch = b.out_ch
            in_ch = b.in_ch
            kernel = b.kernel
            dil = b.dilation

            conv_out = b.conv_b.copy()

            # Causal Conv1d
            for oc in range(out_ch):
                for kt in range(kernel):
                    off = dil * (kt + 1 - kernel)
                    ins = (hist_i + off) * in_ch
                    if ins >= 0 and ins + in_ch <= len(hist):
                        conv_out[oc] += np.dot(
                            hist[ins : ins + in_ch],
                            b.conv_w[oc, :, kt],
                        )

            # Fused BatchNorm: scale * x + offset
            conv_out = conv_out * b.bn_scale + b.bn_offset

            # Activation
            conv_out = apply_activation(conv_out, b.activation)

            # Pass to next block's history buffer
            if bi + 1 < len(blocks):
                next_in_ch = blocks[bi + 1].in_ch
                n_copy = min(out_ch, next_in_ch)
                block_hists[bi + 1][hist_i * next_in_ch : hist_i * next_in_ch + n_copy] = conv_out[:n_copy]

            if bi == len(blocks) - 1:
                last_out = conv_out

        block_out = last_out

        if has_head:
            h_out = h_b.copy()
            for oc in range(h_out_ch):
                for ic in range(h_in_ch):
                    h_out[oc] += block_out[ic] * h_w[oc, ic, 0]
            h_out = apply_activation(h_out, h_activation)
            y = h_out[0]
        else:
            y = block_out[0]

        output[f] = y * head_scale

    return output


# ── A2 f64 model (Generic topology — S13.2) ────────────────────────────────

# Legacy hardcoded constants (fallback when layer_raw lacks kernel_sizes/dilations)
A2_KS = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6]
A2_DIL = [1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239]
A2_NLAYERS = 23
A2_HEAD_K = 16

# ── FiLM A2 helpers ───────────────────────────────────────────────────────

FILM_KEYS = [
    ("conv_pre_film", 0),
    ("conv_post_film", 1),
    ("input_mixin_pre_film", 2),
    ("input_mixin_post_film", 3),
    ("activation_pre_film", 4),
    ("activation_post_film", 5),
    ("layer1x1_post_film", 6),
    ("head1x1_post_film", 7),
]


def film_weight_count(groups, cond_size, channels, shift):
    g = int(groups)
    ch_per_group = channels // g
    cond_per_group = cond_size // g
    out_per_group = ch_per_group * 2 if shift else ch_per_group
    return g * out_per_group * cond_per_group


def film_weight_count_generic(groups, cond_size, channels, shift):
    """Integer-safe variant for A2 generic (cond_size > 1)."""
    g = int(groups)
    mult = 2 if shift else 1
    return channels * mult * cond_size // g


def film_bias_count(channels, shift):
    return channels * 2 if shift else channels


def film_bias_count_generic(channels):
    """Simplified bias for A2 generic — always channels."""
    return channels


class FiLMSlot:
    def __init__(self, shift, groups, weights_arr, bias_arr, channels):
        self.shift = shift
        self.groups = int(groups)
        self.weights = np.array(weights_arr, dtype=np.float64)
        self.channels = channels
        expected_bias = channels * 2 if shift else channels
        if len(bias_arr) < expected_bias:
            bias_arr = np.pad(bias_arr, (0, expected_bias - len(bias_arr)))
        self.bias = np.array(bias_arr, dtype=np.float64)
        self.buf = np.zeros(channels * 2, dtype=np.float64)

    def apply(self, input_slice, condition):
        """Apply FiLM modulation in-place on input_slice (variable-length, up to channels)."""
        ch_constructed = self.channels
        g = self.groups
        ch_per_group = ch_constructed // g
        cond_per_group = len(condition) // g
        out_per_group = ch_per_group * 2 if self.shift else ch_per_group

        self.buf.fill(0.0)

        for grp in range(g):
            cond_off = grp * cond_per_group
            row_off = grp * out_per_group
            w_off = row_off * cond_per_group
            for row in range(out_per_group):
                if row < ch_per_group:
                    global_out = grp * ch_per_group + row
                else:
                    global_out = ch_constructed + grp * ch_per_group + (row - ch_per_group)
                s = float(self.bias[global_out])
                for k in range(cond_per_group):
                    s += (
                        self.weights[w_off + row * cond_per_group + k]
                        * condition[cond_off + k]
                    )
                self.buf[global_out] = s

        apply_len = min(len(input_slice), ch_constructed)
        for c in range(apply_len):
            scale = self.buf[c]
            shift_val = self.buf[c + ch_constructed] if self.shift else 0.0
            input_slice[c] = input_slice[c] * scale + shift_val


def apply_activation_a2(z: np.ndarray, act_config, activation_mode="exact"):
    """Apply f64 activation matching Rust oracle::oracle_apply_activation."""
    if activation_mode == "exact":
        sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-x))
        tanh_fn = np.tanh
    else:
        sigmoid_fn = lambda x: 1.0 / (1.0 + np.exp(-x))
        tanh_fn = np.tanh

    t = act_config.get("type", "Tanh") if isinstance(act_config, dict) else str(act_config)
    if t == "Tanh":
        return tanh_fn(z)
    elif t == "HardTanh":
        return np.clip(z, -1.0, 1.0)
    elif t == "FastTanh":
        return tanh_fn(z)
    elif t == "ReLU":
        return np.maximum(z, 0.0)
    elif t == "LeakyReLU":
        slope = float(act_config.get("negative_slope", 0.01)) if isinstance(act_config, dict) else 0.01
        return np.where(z < 0, z * slope, z)
    elif t == "Sigmoid":
        return sigmoid_fn(z)
    elif t == "SiLU":
        return z * sigmoid_fn(z)
    elif t == "HardSwish":
        relu6 = np.clip(z + 3.0, 0.0, 6.0)
        return z * relu6 / 6.0
    elif t == "Softsign":
        return z / (1.0 + np.abs(z))
    else:
        return tanh_fn(z)


def _extract_activation(activation_raw, li, num_layers):
    """Extract per-layer activation config (mirrors a2_read_activation in Rust)."""
    if isinstance(activation_raw, list) and li < len(activation_raw):
        return activation_raw[li]
    if isinstance(activation_raw, dict):
        return activation_raw
    return {"type": "LeakyReLU", "negative_slope": 0.01}


def _extract_gating_mode(layer_raw, li):
    """Extract per-layer gating mode (mirrors a2_read_gating_mode in Rust)."""
    gm = layer_raw.get("gating_mode")
    if isinstance(gm, list) and li < len(gm):
        return str(gm[li])
    return "none"


def _extract_secondary_activation(layer_raw, li):
    """Extract per-layer secondary activation (mirrors a2_read_secondary_activation in Rust)."""
    sec = layer_raw.get("secondary_activation")
    if isinstance(sec, list) and li < len(sec):
        val = sec[li]
        if val is None:
            return {"type": "Sigmoid"}
        return val
    if isinstance(sec, dict):
        return sec
    return {"type": "Sigmoid"}


def _extract_head1x1_active(layer_raw):
    h1 = layer_raw.get("head1x1")
    if isinstance(h1, dict):
        return h1.get("active", False)
    return False


def a2_forward(model: dict, x: np.ndarray) -> np.ndarray:
    """A2 forward pass in NumPy f64 — Multi-array cascade (S14.2).

    Supports open topologies, multi-array cascade, condition_dsp sub-model
    (dispatched via forward_dispatch, supporting any architecture: LSTM, A2,
    WaveNet, ConvNet), arbitrary channels, bottleneck, kernel_sizes, dilations,
    condition_size>1, head1x1, gating/blending, heterogeneous activations, and
    all 8 FiLM slots.
    Backward-compatible with legacy 23-layer fast-path A2 models and
    single-array A2 generic models.
    """
    config = model["config"]
    weights = load_weights_as_f64(model)
    head_scale = np.float64(config.get("head_scale", 1.0))

    layers_cfg = config.get("layers", [])
    if not layers_cfg:
        return np.zeros_like(x)
    num_arrays = len(layers_cfg)
    num_frames = len(x)
    if num_frames == 0:
        return np.zeros(0, dtype=np.float64)

    # ── condition_dsp sub-model ──
    # T1.2: Use wavenet_forward_all_channels for WaveNet A1 condition_dsp
    # to get ALL head output channels (matching C++ NumOutputChannels()).
    # A2 condition_dsp falls back to forward_dispatch (mono) until §4.4.
    cond_dsp_out = None
    if config.get("condition_dsp") is not None:
        import copy
        cond_model = copy.deepcopy(config["condition_dsp"])
        dsp_arch = cond_model.get("architecture", None)
        if dsp_arch == "WaveNet":
            dsp_cfg = cond_model.get("config", {})
            dsp_layers = dsp_cfg.get("layers", [])
            # Mirror Rust's is_a2_model: A1 (use all_channels) unless
            # head_scale present AND (single-array OR A2-specific features).
            is_a2 = dsp_layers and dsp_cfg.get("head_scale") is not None
            if is_a2 and len(dsp_layers) > 1:
                is_a2 = any(
                    l.get("head1x1", {}).get("active", False)
                    or any(l.get(k, {}).get("active", False) for k, _ in FILM_KEYS)
                    for l in dsp_layers
                ) or (
                    isinstance(dsp_layers[0].get("activation"), str)
                    and dsp_layers[0].get("activation") not in ("Tanh", "HardTanh", "FastTanh")
                )
            if is_a2:
                cond_dsp_out = forward_dispatch(cond_model, x)
            else:
                cond_dsp_out = wavenet_forward_all_channels(cond_model, x)
        else:
            cond_dsp_out = forward_dispatch(cond_model, x)

    # ── Read all arrays' weights ──
    cursor = 0

    class ArrayWeights:
        pass

    array_list = []
    for ai in range(num_arrays):
        layer_raw = layers_cfg[ai]
        ch = int(layer_raw["channels"])
        cond_size = int(layer_raw.get("condition_size", 1))

        # Read topology
        if "kernel_sizes" in layer_raw:
            kernel_sizes = list(layer_raw["kernel_sizes"])
        elif "kernel_size" in layer_raw and "dilations" in layer_raw:
            kernel_sizes = [int(layer_raw["kernel_size"])] * len(layer_raw["dilations"])
        else:
            kernel_sizes = list(A2_KS)
        dilations = list(layer_raw.get("dilations", A2_DIL))
        num_layers = len(kernel_sizes)
        if num_layers != len(dilations) or num_layers == 0:
            return np.zeros_like(x)
        bottleneck = int(layer_raw.get("bottleneck", ch))

        activation_raw = layer_raw.get("activation")
        head1x1_active = _extract_head1x1_active(layer_raw)

        # FiLM detection
        film_active = [False] * 8
        for key, idx in FILM_KEYS:
            cfg = layer_raw.get(key)
            if isinstance(cfg, dict) and cfg.get("active", False):
                film_active[idx] = True

        film_slot_configs = [None] * 8
        for key, idx in FILM_KEYS:
            cfg = layer_raw.get(key)
            if isinstance(cfg, dict) and cfg.get("active", False):
                film_slot_configs[idx] = {
                    "shift": cfg.get("shift", True),
                    "groups": cfg.get("groups", 1),
                }

        # Rechannel weights (1×ch for array 0, prev_ch×ch for cascade)
        in_ch = 1 if ai == 0 else (int(layers_cfg[ai - 1].get("channels", 8)))
        rw_count = in_ch * ch
        rechannel_w = weights[cursor : cursor + rw_count].copy()
        cursor += rw_count

        # Per-layer weights
        layer_weights = []
        for li in range(num_layers):
            ks = kernel_sizes[li]
            dil = dilations[li]
            gmode = _extract_gating_mode(layer_raw, li)
            use_gating = gmode in ("gated", "blended")
            conv_out = bottleneck * 2 if use_gating else bottleneck

            n_conv = ch * conv_out * ks
            conv_w = weights[cursor : cursor + n_conv].reshape(conv_out, ch, ks).copy()
            cursor += n_conv
            conv_b = weights[cursor : cursor + conv_out].copy()
            cursor += conv_out
            n_mixin = conv_out * cond_size
            mixin_w = weights[cursor : cursor + n_mixin].reshape(conv_out, cond_size).copy()
            cursor += n_mixin
            n_l1x1 = bottleneck * ch
            l1x1_w = weights[cursor : cursor + n_l1x1].reshape(ch, bottleneck).copy()
            cursor += n_l1x1
            l1x1_b = weights[cursor : cursor + ch].copy()
            cursor += ch

            film_slots = [None] * 8
            for slot_idx in range(8):
                if not film_active[slot_idx]:
                    continue
                scfg = film_slot_configs[slot_idx]
                g = int(scfg["groups"])
                shift = scfg["shift"]
                # C++ convention (slimmable.cpp): slot 2 → cond_size, slot 7 → head1x1_out
                film_ch = (1 if slot_idx == 2 else
                           1 if slot_idx == 7 else
                           ch)
                if cond_size > 1:
                    wc = film_weight_count_generic(g, cond_size, film_ch, shift)
                    bc = film_bias_count_generic(film_ch)
                else:
                    wc = film_weight_count(g, cond_size, film_ch, shift)
                    bc = film_bias_count(film_ch, shift)
                slot_w = weights[cursor : cursor + wc].copy()
                cursor += wc
                slot_b = weights[cursor : cursor + bc].copy()
                cursor += bc
                film_slots[slot_idx] = FiLMSlot(shift, g, slot_w, slot_b, film_ch)

            act = _extract_activation(activation_raw, li, num_layers)
            sec_act = _extract_secondary_activation(layer_raw, li)
            layer_weights.append({
                "conv_w": conv_w, "conv_b": conv_b, "mixin_w": mixin_w,
                "l1x1_w": l1x1_w, "l1x1_b": l1x1_b, "ks": ks, "dil": dil,
                "film": film_slots, "gating_mode": gmode,
                "activation": act, "secondary_activation": sec_act,
                "conv_out": conv_out,
            })

        # Head1x1 weights
        head1x1_w = None
        head1x1_b = None
        head1x1_in = None
        if head1x1_active:
            h1_groups = layer_raw.get("head1x1", {}).get("groups", 1)
            head1x1_in = bottleneck // h1_groups
            n_h1 = ch * head1x1_in
            head1x1_w = weights[cursor : cursor + n_h1].reshape(ch, head1x1_in).copy()
            cursor += n_h1
            head1x1_b = weights[cursor : cursor + ch].copy()
            cursor += ch

        # Head conv weights
        head_accum_size = int(layer_raw.get("head1x1", {}).get("out_channels", bottleneck)) if head1x1_active else bottleneck

        # S16.4 (T5.1): head layout format — A2 legacy (no explicit head_size in
        # JSON) vs rechannel (explicit head_size).  A2 uses K=16 Conv1D with
        # per-array bias+scale; rechannel uses a simple dense readout with
        # per-array bias (only when head_bias=True) and a global head_scale at
        # the end of the weight stream.
        head_size_raw = layer_raw.get("head_size")
        is_head_rechannel = head_size_raw is not None
        if is_head_rechannel:
            head_size = int(head_size_raw)
            hw_count = head_accum_size * head_size
            head_w = weights[cursor : cursor + hw_count].copy()
            cursor += hw_count
            head_bias_flag = bool(layer_raw.get("head_bias", False))
            if head_bias_flag:
                head_b_arr = weights[cursor : cursor + head_size].copy()
                cursor += head_size
            else:
                head_b_arr = np.zeros(head_size, dtype=np.float64)
            head_b = np.float64(head_b_arr[0]) if len(head_b_arr) == 1 else head_b_arr
        else:
            # Legacy A2 format: head_size not in JSON → A2_HEAD_K=16 Conv1D
            head_size = 1
            head_w_raw = weights[cursor : cursor + A2_HEAD_K * head_accum_size]
            cursor += A2_HEAD_K * head_accum_size
            head_w = np.zeros(A2_HEAD_K * head_accum_size, dtype=np.float64)
            for tap in range(A2_HEAD_K):
                for c in range(head_accum_size):
                    head_w[tap * head_accum_size + c] = head_w_raw[c * A2_HEAD_K + tap]
            head_b = np.float64(weights[cursor])
            cursor += 1
            _head_scale_val = np.float64(weights[cursor])
            cursor += 1

        arr = ArrayWeights()
        arr.head_accum_size = head_accum_size
        arr.ch = ch
        arr.bottleneck = bottleneck
        arr.cond_size = cond_size
        arr.head_size = head_size
        arr.head_is_rechannel = is_head_rechannel
        arr.rechannel_w = rechannel_w
        arr.layer_weights = layer_weights
        arr.head1x1_active = head1x1_active
        arr.head1x1_w = head1x1_w
        arr.head1x1_b = head1x1_b
        arr.head1x1_in = head1x1_in
        arr.head_w = head_w
        arr.head_b = head_b
        arr.kernel_sizes = kernel_sizes
        arr.dilations = dilations
        arr.num_layers = num_layers
        arr.film_active = film_active
        arr.layer_raw = layer_raw  # needed for head1x1 groups at runtime
        array_list.append(arr)

    # ── Buffers ──
    max_rf = 0
    for arr in array_list:
        max_ks = max(arr.kernel_sizes)
        max_dil = max(arr.dilations)
        max_rf = max(max_rf, (max_ks - 1) * max_dil + 64)
    hist_size = max_rf + num_frames + 64
    bs = max_rf

    # Per-array state buffers (lazily initialized per frame)
    hr_len = 1 << (max_rf + num_frames + 64).bit_length()
    ring_mask = hr_len - 1
    max_ch = max(a.ch for a in array_list)
    head_acc = np.zeros(hr_len * max_ch, dtype=np.float64)
    head_wp = 0

    # Cascade residual buffer
    cascade_residual = np.zeros(hist_size * max_ch, dtype=np.float64)

    for arr in array_list:
        arr.layer_bufs = [
            np.zeros(hist_size * arr.ch, dtype=np.float64) for _ in range(arr.num_layers)
        ]

    output = np.zeros(num_frames, dtype=np.float64)

    for f in range(num_frames):
        fi = bs + f
        x_val = x[f]
        head_col = head_wp
        head_wp += 1

        for ai, arr in enumerate(array_list):
            ch = arr.ch
            bottleneck = arr.bottleneck
            cond_size = arr.cond_size
            num_layers = arr.num_layers
            layer_raw = arr.layer_raw

            # Condition vector
            if cond_size == 1:
                condition = np.array([x_val], dtype=np.float64)
            elif cond_dsp_out is not None:
                off = f * cond_size
                condition = (
                    cond_dsp_out[off : off + cond_size].copy()
                    if off + cond_size <= len(cond_dsp_out)
                    else np.zeros(cond_size, dtype=np.float64)
                )
            else:
                condition = np.zeros(0, dtype=np.float64)

            # Per-array history buffers
            layer_bufs = arr.layer_bufs

            # Input: mono for array 0, cascade residual for others
            layer_in = np.zeros(ch, dtype=np.float64)
            if ai == 0:
                for c in range(ch):
                    layer_in[c] = x_val * arr.rechannel_w[c]
            else:
                prev_ch = array_list[ai - 1].ch
                rw = arr.rechannel_w
                for nc in range(ch):
                    s = 0.0
                    for ic in range(prev_ch):
                        s += cascade_residual[fi * max_ch + ic] * rw[ic * ch + nc]
                    layer_in[nc] = s

            for c in range(ch):
                layer_bufs[0][fi * ch + c] = layer_in[c]

            for li, lw in enumerate(arr.layer_weights):
                ks = lw["ks"]
                dil = lw["dil"]
                conv_out = lw["conv_out"]
                use_gating = lw["gating_mode"] == "gated"
                use_blending = lw["gating_mode"] == "blended"
                film = lw["film"]

                # conv_pre_film (slot 0)
                if film[0] is not None:
                    film[0].apply(layer_bufs[li][fi * ch : fi * ch + ch], condition)

                # Conv1d
                z = lw["conv_b"].copy()
                hist = layer_bufs[li]
                for oc in range(conv_out):
                    for kt in range(ks):
                        off = dil * (kt + 1 - ks)
                        ins = (fi + off) * ch
                        if ins >= 0 and ins + ch <= len(hist):
                            z[oc] += np.dot(
                                hist[ins : ins + ch], lw["conv_w"][oc, :, kt]
                            )

                # conv_post_film (slot 1)
                if film[1] is not None:
                    film[1].apply(z, condition)

                # Mixin — input_mixin_pre_film (slot 2) applied to condition
                # (self-modulation, C++ model.cpp:188-197) before the mixin.
                condition_mod = np.array(condition[:cond_size], dtype=np.float64).copy()
                if film[2] is not None:
                    film[2].apply(condition_mod, condition[:cond_size])
                mixin_contrib = np.zeros_like(z)
                if len(condition_mod) > 0:
                    k_used = min(cond_size, len(condition_mod))
                    for c in range(conv_out):
                        mixin_contrib[c] = np.dot(lw["mixin_w"][c, :k_used], condition_mod[:k_used])

                # input_mixin_post_film (slot 3)
                if film[3] is not None:
                    film[3].apply(mixin_contrib, condition)

                # Sum mixin to z
                z += mixin_contrib

                # activation_pre_film (slot 4)
                if film[4] is not None:
                    film[4].apply(z, condition)

                # Activation or Gating/Blending
                if use_gating:
                    half = bottleneck
                    z[:half] = apply_activation_a2(z[:half], lw["activation"])
                    z[half:half * 2] = apply_activation_a2(z[half:half * 2], lw["secondary_activation"])
                    z[:half] *= z[half:half * 2]
                    z_len = half
                elif use_blending:
                    half = bottleneck
                    original = z[:half].copy()
                    z[:half] = apply_activation_a2(z[:half], lw["activation"])
                    z[half:half * 2] = apply_activation_a2(z[half:half * 2], lw["secondary_activation"])
                    alpha = z[half:half * 2]
                    z[:half] = original + alpha * (z[:half] - original)
                    z_len = half
                else:
                    z[:bottleneck] = apply_activation_a2(z[:bottleneck], lw["activation"])
                    z_len = bottleneck

                # activation_post_film (slot 5)
                if film[5] is not None:
                    film[5].apply(z[:z_len], condition)

                # Head accumulate
                head_off = head_col * max_ch
                if arr.head1x1_active:
                    h1_groups = layer_raw.get("head1x1", {}).get("groups", 1)
                    ch_per_group = ch // h1_groups
                    h1x1_out = np.zeros(arr.head_accum_size, dtype=np.float64)
                    h1x1_out[:ch] = arr.head1x1_b
                    for grp in range(h1_groups):
                        for oc in range(grp * ch_per_group, (grp + 1) * ch_per_group):
                            for ic in range(arr.head1x1_in):
                                h1x1_out[oc] += (
                                    arr.head1x1_w[oc, ic]
                                    * z[grp * arr.head1x1_in + ic]
                                )
                    if film[7] is not None:
                        film[7].apply(h1x1_out, condition)
                    if li == 0 and ai == 0:
                        head_acc[head_off : head_off + arr.head_accum_size] = h1x1_out[:arr.head_accum_size]
                    else:
                        head_acc[head_off : head_off + arr.head_accum_size] += h1x1_out[:arr.head_accum_size]
                else:
                    if li == 0 and ai == 0:
                        head_acc[head_off : head_off + z_len] = z[:z_len]
                    else:
                        head_acc[head_off : head_off + z_len] += z[:z_len]

                # L1x1 residual
                if li < num_layers - 1:
                    residual = z[:bottleneck] @ lw["l1x1_w"].T + lw["l1x1_b"]
                    if use_blending:
                        if film[6] is not None:
                            film[6].apply(residual, condition)
                    layer_in = layer_in + residual
                    layer_bufs[li + 1][fi * ch : fi * ch + ch] = layer_in

            # Save residual for next array
            if ai + 1 < num_arrays:
                cascade_residual[fi * max_ch : fi * max_ch + ch] = layer_in

        # ── Head finalize (last array) ──
        last_arr = array_list[-1]
        lch = last_arr.head_accum_size
        if last_arr.head_is_rechannel:
            k = last_arr.head_size
        else:
            k = A2_HEAD_K if last_arr.head_size == 1 else last_arr.head_size
        cb = head_col - (k - 1)
        y = float(last_arr.head_b) if np.ndim(last_arr.head_b) == 0 else float(last_arr.head_b[0])
        for t in range(k):
            col = (cb + t) & ring_mask
            wo = t * lch
            y += np.dot(head_acc[col * max_ch : col * max_ch + lch], last_arr.head_w[wo : wo + lch])
        output[f] = y * head_scale

    return output


# ── Architecture detection and forward dispatch ──────────────────────────────

def detect_architecture(model: dict) -> str:
    """Detect model architecture from model dict (without --architecture flag).

    Used for recursive condition_dsp dispatching where the sub-model may be
    any architecture family (LSTM, A2, WaveNet, ConvNet), not just A2.
    """
    arch = model.get("architecture", None)
    if arch:
        return arch

    config = model.get("config", {})
    layers = config.get("layers", [])

    if not layers:
        if config.get("hidden_size") is not None:
            return "LSTM"
        if config.get("channels") and config.get("dilations"):
            return "ConvNet"
        return "Unknown"

    has_head_scale = "head_scale" in config
    has_head = bool(config.get("head"))
    if has_head_scale and not has_head and any(
        "dilations" in l and "channels" in l
        for l in layers
    ):
        return "A2"

    if any("dilations" in l for l in layers):
        return "WaveNet"

    return "Unknown"


def forward_dispatch(model: dict, x: np.ndarray) -> np.ndarray:
    """Generic forward dispatcher — routes to correct architecture-specific function.

    This is the recursive dispatch point for condition_dsp sub-models (T5.1).
    Unlike the command-line --architecture flag, this function auto-detects the
    architecture from the model dict, supporting sub-models of any type.
    """
    arch = detect_architecture(model)

    if arch == "WaveNet":
        return wavenet_forward(model, x)
    elif arch == "LSTM":
        return lstm_forward(model, x)
    elif arch == "A2":
        return a2_forward(model, x)
    elif arch == "ConvNet":
        return convnet_forward(model, x)
    else:
        print(f"Warning: unknown architecture '{arch}' in forward_dispatch, returning zeros ({len(x)} samples)", file=sys.stderr)
        return np.zeros_like(x)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="NAM f64 reference oracle — PyTorch/NumPy anchor"
    )
    ap.add_argument("model", help="Path to .nam JSON model file")
    ap.add_argument("input", help="Binary input: [u32 N] [f64*N]")
    ap.add_argument(
        "--architecture",
        choices=["WaveNet", "LSTM", "A2", "ConvNet"],
        default="WaveNet",
        help="Model architecture family",
    )
    ap.add_argument(
        "-o", "--output", help="Output file (binary: [u32 M] [f64*M])"
    )
    args = ap.parse_args()

    model = load_nam_json(args.model)
    signal = read_input_bin(args.input)

    arch = args.architecture
    if arch == "WaveNet":
        config = model.get("config", {})
        layers = config.get("layers", [])
        # A2 detection: head_scale present, no post-stack head, and layers
        # have dilations+channels (either with kernel_sizes array or
        # kernel_size scalar).
        # S16.4 (T5.1): condition_dsp models with Tanh activation route through
        # the WaveNet A1 oracle, matching Rust's is_a2_model routing.
        has_head_scale = "head_scale" in config
        has_head = bool(config.get("head"))
        has_cond_dsp = config.get("condition_dsp") is not None
        has_tanh = has_cond_dsp and any(
            l.get("activation") == "Tanh"
            or (isinstance(l.get("activation"), list) and "Tanh" in l.get("activation", []))
            for l in layers
        )
        if has_head_scale and not has_head and layers and any(
            "dilations" in l and "channels" in l
            for l in layers
        ) and not has_tanh:
            arch = "A2"

    print(f"Architecture: {arch}", file=sys.stderr)
    print(f"Input samples: {len(signal)}", file=sys.stderr)
    print(f"Weights: {len(model['weights'])}", file=sys.stderr)

    if arch == "WaveNet":
        output = wavenet_forward(model, signal)
    elif arch == "LSTM":
        output = lstm_forward(model, signal)
    elif arch == "A2":
        output = a2_forward(model, signal)
    elif arch == "ConvNet":
        output = convnet_forward(model, signal)
    else:
        print(f"Unknown architecture: {arch}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or (Path(args.model).stem + "_f64_oracle.bin")
    write_output_bin(out_path, output)
    print(f"Output written to {out_path} ({len(output)} samples)", file=sys.stderr)


if __name__ == "__main__":
    main()
