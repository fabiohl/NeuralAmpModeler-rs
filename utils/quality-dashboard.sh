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
fidelity_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) echo "Uso: utils/quality-dashboard.sh [--check [<contrato>] [--fidelity-only] | --save]"; exit 0 ;;
    --fidelity-only) fidelity_only=1; shift ;;
    --check|--save) if [ $# -ge 2 ] && [[ "$2" != --* ]]; then shift 2; else shift; fi ;;
    *) usage_fail "quality-dashboard.sh: flag não reconhecida '$1'" ;;
  esac
done
[ "$fidelity_only" -eq 1 ] && SCOPE=fidelity-only
lib_init
trap 'trap_fail $?' ERR
run_step quality "$(ensure_qa_bin nam_quality testing)" dashboard "${args[@]}"
rc=0; finish || rc=$?
exit "$rc"
