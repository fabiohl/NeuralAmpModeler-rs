#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# ensure_namcore_render.sh — compila o renderizador C++ de referência. Classe D: stdout é apenas o caminho do binário; RESULT: vai para stderr (docs/testing.md §11.1).
# Uso: utils/ensure_namcore_render.sh [flags canônicas]
# Orçamento: 32          # +5: lib_init/finish com RESULT em stderr (S3-T4)
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/ensure_namcore_render.sh"; exit 0 ;;
    *) echo "Usage: utils/ensure_namcore_render.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh

T0=$SECONDS
lib_init >&2
if path=$(ensure_namcore_render "$@"); then
    printf '%s\n' "$path"
    finish >&2 || exit 1
else
    rc=$?
    FAILS+=(render)
    finish >&2 || true
    exit "$rc"
fi
