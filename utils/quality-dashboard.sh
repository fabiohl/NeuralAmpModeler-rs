#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# quality-dashboard.sh — Quality dashboard and contract verification. Classe A.
# Uso: utils/quality-dashboard.sh [--check [<contrato>] [--fidelity-only] | --save]
# Orçamento: 15
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh
rc=0; "$(ensure_qa_bin nam_quality testing)" dashboard "$@" || rc=$?
[ "$rc" -eq 0 ] || FAILS+=(quality)
frc=0; finish || frc=$?
[ "$rc" -ne 0 ] || rc=$frc
exit "$rc"
