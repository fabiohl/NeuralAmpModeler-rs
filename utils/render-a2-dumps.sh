#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# render-a2-dumps.sh — Reproduces the KB-A2-MAX tensor dumps. Classe D.
# Uso: utils/render-a2-dumps.sh
# Orçamento: 120
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
case "$#:${1:-}" in
  0:) ;;
  1:-h|1:--help) echo "Usage: utils/render-a2-dumps.sh"; exit 0 ;;
  *) echo "Usage: utils/render-a2-dumps.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh
rm -f "$LOG_DIR/render-a2-dumps-status.tsv"
trap 'rc=$?; [ "$rc" -eq 0 ] || [ -n "${DONE:-}" ] || { FAILS+=(render-a2-dumps); DONE=1; finish; }' EXIT
vernier() { set +x; FAILS+=(render-a2-dumps); DONE=1; finish || true; die "$*"; }

PROJECT_ROOT="$PWD"
DIFF_ROOT="${NAM_A2_DUMPS_ROOT:-$PROJECT_ROOT/target/vendor-a2-dumps}"
PATCH="$PROJECT_ROOT/utils/namcore-a2-dumps.patch"
BUILD_DIR="$PROJECT_ROOT/build/namcore_a2_dumps"
OUT_DIR="$PROJECT_ROOT/tests/fixtures/dumps_a2_max"
MODEL="$PROJECT_ROOT/tests/fixtures/models/wavenet_a2_max.nam"
INPUT="$PROJECT_ROOT/tests/fixtures/stress_signal.wav"
GOLDEN="$PROJECT_ROOT/tests/fixtures/golden_wavenet_a2_max.bin"
TMP_RENDER="${TMPDIR:-/tmp}/nam-a2-dumps-render.wav"
LOGS_DIR="$BUILD_DIR/logs"
DUMPS=(dump_array0.bin dump_rechannel.bin dump_condition_dsp.bin dump_film.bin)

VENDOR_MIRROR="$PROJECT_ROOT/third-party/NeuralAmpModelerCore"
CARRIER=" carrier=$LOG_DIR/render-a2-dumps-status.tsv"
[ -f "$PATCH" ] || vernier "utils/namcore-a2-dumps.patch missing$CARRIER"
[ -d "$VENDOR_MIRROR" ] || vernier "third-party/NeuralAmpModelerCore missing — run utils/setup-third-party.sh first$CARRIER"
PINNED=$(sed -n 's/^NAM_CORE_COMMIT="\([0-9a-f]*\)".*/\1/p' "$PROJECT_ROOT/variables.env")
[ -n "$PINNED" ] || vernier "NAM_CORE_COMMIT not found in variables.env$CARRIER"
[ -f "$MODEL" ] || vernier "wavenet_a2_max.nam missing$CARRIER"
[ -f "$INPUT" ] || vernier "tests/fixtures/stress_signal.wav missing (golden input; run tests/fixtures/golden_gen_build.sh)$CARRIER"
[ -f "$GOLDEN" ] || vernier "golden_wavenet_a2_max.bin missing$CARRIER"
command -v cmake >/dev/null || vernier "cmake not found$CARRIER"
command -v git >/dev/null || vernier "git not found$CARRIER"

# Vendor somente leitura: confere o pin e instrumenta uma cópia descartável em $DIFF_ROOT/<sha>.
before_head=$(git -C "$VENDOR_MIRROR" rev-parse HEAD) && before_status=$(git -C "$VENDOR_MIRROR" status --porcelain)
[ "$before_head" = "$PINNED" ] || vernier "vendor HEAD $before_head != pinned $PINNED$CARRIER"
[ -z "$before_status" ] || vernier "vendor dirty before run:$CARRIER$(printf '\n%s' "$before_status")"
VENDOR="$DIFF_ROOT/$PINNED"
mkdir -p "$DIFF_ROOT"
rm -rf "$VENDOR"
git clone -q --no-local "$VENDOR_MIRROR" "$VENDOR" && git -C "$VENDOR" checkout -q "$PINNED" \
  && { git -C "$VENDOR" submodule update --init --recursive -q || vernier "submodule update failed in $VENDOR$CARRIER"; } \
  && [ -z "$(git -C "$VENDOR" status --porcelain)" ] || vernier "cannot create clean disposable clone at $VENDOR$CARRIER"
git -C "$VENDOR" apply "$PATCH" || vernier "instrumentation patch does not apply to $PINNED$CARRIER"
after_head=$(git -C "$VENDOR_MIRROR" rev-parse HEAD) && after_status=$(git -C "$VENDOR_MIRROR" status --porcelain)
[ "$after_head" = "$before_head" ] && [ "$after_status" = "$before_status" ] \
  || vernier "vendor changed during run (HEAD $after_head)$CARRIER"

mkdir -p "$BUILD_DIR" "$LOGS_DIR"
cmake -S "$VENDOR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 -DNAM_ENABLE_A2_FAST=ON -DNAM_A2_DUMPS=ON \
  > "$LOGS_DIR/cmake-configure.log" 2>&1 || vernier "cmake configure failed"
cmake --build "$BUILD_DIR" --target render -j"${NAM_RENDER_JOBS:-$(nproc)}" \
  > "$LOGS_DIR/cmake-build.log" 2>&1 || vernier "cmake build failed"
RENDER_BIN=""
for cand in tools/render Release/render Debug/render; do
  [ -x "$BUILD_DIR/$cand" ] && RENDER_BIN="$BUILD_DIR/$cand" && break
done
[ -n "$RENDER_BIN" ] || vernier "render binary not found after build"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/dump_array0.bin" "$OUT_DIR/dump_rechannel.bin" \
  "$OUT_DIR/dump_condition_dsp.bin" "$OUT_DIR/dump_film.bin" "$OUT_DIR/SHA256SUMS" "$OUT_DIR/capture_meta.txt"
echo "→ rendering with capture"
NAM_A2_DUMP_DIR="$OUT_DIR" "$RENDER_BIN" "$MODEL" "$INPUT" "$TMP_RENDER" \
  > "$LOGS_DIR/render.log" 2>&1 || { cat "$LOGS_DIR/render.log" >&2; vernier "instrumented render failed"; }

# Reprodução byte-a-byte do golden: WAV canônico (cabeçalho de 44 bytes) contra a seção de saída f32 do golden.
n=$(od -An -t u4 -N4 "$GOLDEN" | tr -d ' ')
[ "$(dd if="$TMP_RENDER" bs=1 skip=36 count=4 2>/dev/null)" = data ] || vernier "render WAV without canonical 44-byte header"
[ "$(stat -c %s "$TMP_RENDER")" -eq $((44 + 4 * n)) ] || vernier "render WAV size differs from golden ($n frames)"
tail -c +45 "$TMP_RENDER" | cmp -s - <(dd if="$GOLDEN" bs=4 skip=$n count="$n" 2>/dev/null) \
  || vernier "render does not reproduce golden_wavenet_a2_max.bin"
echo "✓ RENDER REPRODUCED GOLDEN byte-identically ($n frames)"

(cd "$OUT_DIR" && sha256sum "${DUMPS[@]}") | tee "$OUT_DIR/SHA256SUMS"
echo "Dumps ready: $OUT_DIR"
echo "Disposable instrumented tree: $VENDOR"
DONE=1; finish
