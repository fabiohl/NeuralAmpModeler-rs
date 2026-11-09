#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# ensure_namcore_render.sh — compila o renderizador C++ de referência. Classe D: stdout é apenas o caminho do binário; RESULT: vai para stderr (docs/testing.md §11.1).
# Uso: utils/ensure_namcore_render.sh [flags canônicas]
# Orçamento: 32          # +5: lib_init/finish com RESULT em stderr (S3-T4)
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/ensure_namcore_render.sh"; exit 0 ;;
    *) usage_fail "ensure_namcore_render.sh não aceita argumentos ($*)" >&2 ;;
esac

T0=$SECONDS
lib_init >&2
phase render >&2
if path=$(ensure_namcore_render "$@"); then
    emit "PHASE: render PASS" >&2
    printf '%s\n' "$path"
    finish >&2 || exit 1
else
    rc=$?
    FAILS+=(render); emit "PHASE: render FAIL rc=$rc" >&2
    finish >&2 || true
    exit "$rc"
fi
