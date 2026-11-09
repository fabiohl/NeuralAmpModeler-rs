#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# quality-dashboard.sh — Quality dashboard and contract verification. Classe A.
# Uso: utils/quality-dashboard.sh [--check [<contrato>] [--fidelity-only] | --save]
# Orçamento: 30
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh
args=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) echo "Uso: utils/quality-dashboard.sh [--check [<contrato>] [--fidelity-only] | --save]"; exit 0 ;;
    --fidelity-only) shift ;;
    --check|--save) if [ $# -ge 2 ] && [[ "$2" != --* ]]; then shift 2; else shift; fi ;;
    *) usage_fail "quality-dashboard.sh: flag não reconhecida '$1'" ;;
  esac
done
lib_init
trap 'trap_fail $?' ERR
phase quality
rc=0; "$(ensure_qa_bin nam_quality testing)" dashboard "${args[@]}" || rc=$?
if [ "$rc" -eq 0 ]; then emit "PHASE: quality PASS"; else FAILS+=(quality); emit "PHASE: quality FAIL rc=$rc"; fi
frc=0; finish || frc=$?
[ "$rc" -ne 0 ] || rc=$frc
exit "$rc"
