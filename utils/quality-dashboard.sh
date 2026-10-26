#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# quality-dashboard.sh — Quality dashboard and contract verification. Classe A.
# Uso: utils/quality-dashboard.sh [--check <contrato> | --save]
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh


QUALITY_BIN=$(ensure_qa_bin nam_quality testing)

# Normalize arguments for nam_quality dashboard
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      exec "$QUALITY_BIN" dashboard --help
      ;;
    --check)
      args+=("$1")
      if [ $# -gt 1 ] && [[ "$2" != --* ]]; then
        args+=("$2")
        shift
      else
        args+=("docs/quality-contract.json")
      fi
      ;;
    *)
      args+=("$1")
      ;;
  esac
  shift
done

exec "$QUALITY_BIN" dashboard "${args[@]}"
