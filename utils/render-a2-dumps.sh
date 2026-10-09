#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# render-a2-dumps.sh — Reproduces the KB-A2-MAX (wavenet_a2_max.nam) C++ tensor
# dumps into tests/fixtures/dumps_a2_max/ (manifest.json kept alongside).
#
# Upstream NAMCore is a vendored (gitignored) mirror; the instrumentation lives
# in this repository as utils/namcore-a2-dumps.patch, applied and compiled in
# only for this run (CMake option NAM_A2_DUMPS=ON). Even when compiled in, the
# capture is inert unless NAM_A2_DUMP_DIR is set.
#
# Philosophy: offline I/O, compile-gated (default OFF), never active in the
# shared build dir build/namcore_render/.
#
# Usage:
#   ./utils/render-a2-dumps.sh
#
# Prerequisites:
#   - third-party/NeuralAmpModelerCore at the pinned NAM_CORE_COMMIT
#     (utils/setup-third-party.sh)
#   - cmake + C++20 compiler
#   - python3 (manifest generation + golden comparison)
#   - tests/fixtures/stress_signal.wav present (golden input, 2048 frames @ 48 kHz)
#   - tests/fixtures/golden_wavenet_a2_max.bin present (reproduction check)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$PROJECT_ROOT/third-party/NeuralAmpModelerCore"
PATCH="$PROJECT_ROOT/utils/namcore-a2-dumps.patch"
BUILD_DIR="${NAM_A2_DUMPS_BUILD_DIR:-$PROJECT_ROOT/build/namcore_a2_dumps}"
OUT_DIR="$PROJECT_ROOT/tests/fixtures/dumps_a2_max"
MODEL="$PROJECT_ROOT/tests/fixtures/models/wavenet_a2_max.nam"
INPUT="$PROJECT_ROOT/tests/fixtures/stress_signal.wav"
GOLDEN="$PROJECT_ROOT/tests/fixtures/golden_wavenet_a2_max.bin"
TMP_RENDER="${TMPDIR:-/tmp}/nam-a2-dumps-render.wav"
LOGS_DIR="$BUILD_DIR/logs"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -d "$VENDOR" ] || die "third-party/NeuralAmpModelerCore missing — run utils/setup-third-party.sh first"
[ -f "$PATCH" ] || die "utils/namcore-a2-dumps.patch missing"
[ -f "$MODEL" ] || die "wavenet_a2_max.nam missing"
[ -f "$INPUT" ] || die "tests/fixtures/stress_signal.wav missing (golden input; run tests/fixtures/golden_gen_build.sh)"
[ -f "$GOLDEN" ] || die "golden_wavenet_a2_max.bin missing"
command -v cmake >/dev/null || die "cmake not found"
command -v python3 >/dev/null || die "python3 not found"

# --- Apply the tracked instrumentation patch (idempotent) --------------------
command -v git >/dev/null || die "git not found"
if git -C "$VENDOR" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "instrumentation patch already applied"
elif git -C "$VENDOR" apply --check "$PATCH"; then
  git -C "$VENDOR" apply "$PATCH"
else
  die "vendor does not match the clean or fully instrumented patch; no files changed"
fi

# --- Build the dump-enabled render binary -----------------------------------
mkdir -p "$BUILD_DIR" "$LOGS_DIR"
BUILD_TYPE="${NAM_RENDER_BUILD_TYPE:-Release}"
CXX_COMPILER="${CXX:-}"
if [ -z "$CXX_COMPILER" ]; then
  if command -v g++ >/dev/null; then CXX_COMPILER=g++; else CXX_COMPILER=clang++; fi
fi
FLAGS="-w -fno-fast-math -ffp-contract=off"

cmake -S "$VENDOR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
  -DCMAKE_CXX_STANDARD=20 \
  -DCMAKE_CXX_FLAGS="$FLAGS" \
  -DNAM_ENABLE_A2_FAST=ON \
  -DNAM_A2_DUMPS=ON \
  > "$LOGS_DIR/cmake-configure.log" 2>&1 || { tail -20 "$LOGS_DIR/cmake-configure.log" >&2; die "cmake configure failed"; }
cmake --build "$BUILD_DIR" --target render -j"$(nproc 2>/dev/null || echo 2)" \
  > "$LOGS_DIR/cmake-build.log" 2>&1 || { tail -20 "$LOGS_DIR/cmake-build.log" >&2; die "cmake build failed"; }

RENDER_BIN="$BUILD_DIR/tools/render"
[ -x "$RENDER_BIN" ] || RENDER_BIN="$BUILD_DIR/Release/render"
[ -x "$RENDER_BIN" ] || die "render binary not found after build"

# --- Capture the dumps -------------------------------------------------------
mkdir -p "$OUT_DIR"
# Remove only generated capture files, never unrelated directory contents.
rm -f "$OUT_DIR/dump_array0.bin" "$OUT_DIR/dump_rechannel.bin" \
  "$OUT_DIR/dump_condition_dsp.bin" "$OUT_DIR/dump_film.bin" \
  "$OUT_DIR/manifest.json" "$OUT_DIR/capture_meta.txt"
export OUT_DIR
echo "→ rendering with capture"
NAM_A2_DUMP_DIR="$OUT_DIR" "$RENDER_BIN" "$MODEL" "$INPUT" "$TMP_RENDER" \
  > "$LOGS_DIR/render.log" 2>&1 || { cat "$LOGS_DIR/render.log" >&2; die "instrumented render failed"; }

# --- Reproduction check against the committed golden -------------------------
python3 - "$TMP_RENDER" "$GOLDEN" <<'EOF'
import struct, sys

def wav_data(path):
    data = open(path, 'rb').read()
    pos = 12
    while pos < len(data):
        chunk = data[pos:pos + 4]
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        if chunk == b'data':
            return data[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    raise SystemExit('wav data chunk not found')

rendered = wav_data(sys.argv[1])
golden = open(sys.argv[2], 'rb').read()
n = struct.unpack('<I', golden[:4])[0]
expected = golden[4 + 4 * n: 4 + 8 * n]
if rendered != expected:
    raise SystemExit('render does not reproduce golden_wavenet_a2_max.bin')
print(f'✓ RENDER REPRODUCED GOLDEN byte-identically ({n} frames)')
EOF

# --- Regenerate manifest.json -------------------------------------------------
python3 - <<'EOF'
import hashlib, json, os

outdir = os.environ['OUT_DIR']
frames = 2048
sample_rate = 48000
family_order = ["conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film",
                "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film"]
def film_channels(slot):
    # input_mixin_pre_film modulates the condition tensor itself (condition_size=8)
    return 8 if slot % 8 == 2 else 4

slots = []
off = 0
for slot in range(16):
    ch = film_channels(slot)
    slots.append({"slot": slot, "layer_ordinal": slot // 8, "family_index": slot % 8,
                  "family": family_order[slot % 8], "channels": ch,
                  "float_offset_in_frame": off,
                  "note": "post-modulation FiLM output (input*scale+shift); "
                          "slot 2/10 modulate the condition tensor itself (8 channels)"})
    off += ch
expected_total = {"dump_array0.bin": 3, "dump_rechannel.bin": 4,
                  "dump_condition_dsp.bin": 8, "dump_film.bin": off}
tensors = []
for name, channels in expected_total.items():
    raw = open(os.path.join(outdir, name), 'rb').read()
    assert len(raw) == frames * channels * 4, (name, len(raw), frames, channels)
    tensors.append({"file": name, "channels": channels,
                    "sha256": hashlib.sha256(raw).hexdigest()})
manifest = {
    "frames": frames,
    "sample_rate": sample_rate,
    "tensors": tensors,
    "model": "tests/fixtures/models/wavenet_a2_max.nam",
    "reference_input": "tests/fixtures/stress_signal.wav (golden input; equals the "
                       "golden_wavenet_a2_max.bin input section bit-exactly)",
    "reference_output": "golden_wavenet_a2_max.bin (instrumented render reproduced byte-identically)",
    "layout": "all tensors frame-major f32 little-endian",
    "frame_layouts": {name: "index = frame*%d + channel" % ch
                      for name, ch in expected_total.items()},
    "capture_scope": "rendered frames only; DSP::prewarm silent frames are excluded",
    "film_layout": {
        "packing": "frame-major: [frame: slot 0..15 concatenated]; "
                   "slots have variable channels; per-frame floats = %d" % off,
        "slot_order": "slot = layer_ordinal*8 + family_index (single main LayerArray, "
                      "2 layers); family order " + ", ".join(family_order),
        "film_slots": slots,
    },
    "dump_semantics": {
        "dump_array0.bin": "condition_dsp LayerArray 0 residual output (GetLayerOutputs; "
                           "last-layer residual = input + layer1x1), 3 channels — NOT the "
                           "head path",
        "dump_rechannel.bin": "condition_dsp LayerArray 0 _head_rechannel output "
                              "(Conv1D 6->4, kernel 1, no bias, dilation 1) after skip "
                              "accumulation; delivered to Array 1 as head_inputs",
        "dump_condition_dsp.bin": "main WaveNet _condition_output (already includes the "
                                  "condition DSP's own head_scale; NOT the main head_scale)",
        "dump_film.bin": "post-modulation outputs of the 16 main-net FiLM slots",
    },
}
with open(os.path.join(outdir, 'manifest.json'), 'w') as f:
    f.write(json.dumps(manifest, indent=2) + '\n')
print('✓ manifest.json written')
for t in tensors:
    print('  %s  ch=%d  sha256=%s' % (t['file'], t['channels'], t['sha256']))
EOF
# (OUT_DIR exported above so the manifest step can see it)

echo "Dumps ready: $OUT_DIR"
