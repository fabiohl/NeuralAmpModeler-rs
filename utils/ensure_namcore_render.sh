#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# ensure_namcore_render.sh — compila o renderizador C++ de referência. Classe D.
# Uso: utils/ensure_namcore_render.sh [flags canônicas]
# Orçamento: 27
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/ensure_namcore_render.sh"; exit 0 ;;
    *) echo "Usage: utils/ensure_namcore_render.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh

ensure_namcore_render "$@"
