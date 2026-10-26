#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# tests-quick.sh — Execução da disciplina rápida de testes. Classe A.
# Uso: utils/tests-quick.sh
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

[ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] && { echo "Uso: utils/tests-quick.sh"; exit 0; }
trap 'echo "RESULT: FAIL ($((SECONDS - T0))s) fails=${PHASE_NAME:-step} gaps=${GAPS[*]:-none}"' ERR
rm -f "$LOG_DIR/skips.jsonl"

PHASE_NAME=freshness; phase freshness
log_fresh="$LOG_DIR/quick-freshness.log"; rc_fresh=0
cargo run --quiet --locked --features testing --bin nam_freshness -- artifacts-hard > "$log_fresh" 2>&1 || rc_fresh=$?
if [ "$rc_fresh" -eq 0 ]; then
  if grep -q "GENERATOR CHANGED" "$log_fresh"; then gap "generator-drift"; fi
  echo "PHASE: freshness PASS log=$log_fresh"
else
  FAILS+=("freshness"); echo "PHASE: freshness FAIL rc=$rc_fresh log=$log_fresh"; exit 1
fi

PHASE_NAME=tests; phase tests
log_tests="$LOG_DIR/quick-tests.log"; rc_tests=0
cargo test --locked --features testing --no-fail-fast > "$log_tests" 2>&1 || rc_tests=$?
grep -E 'FAILED|panicked at' "$log_tests" || true
awk '/test result: / { p+=$4; f+=$6; i+=$8 } END { printf "Tests summary: %d passed, %d failed, %d ignored\n", p, f, i }' "$log_tests"
if [ "$rc_tests" -eq 0 ] && ran_tests "$log_tests"; then
  echo "PHASE: tests PASS log=$log_tests"
else
  FAILS+=("tests"); echo "PHASE: tests FAIL rc=$rc_tests log=$log_tests"; exit 1
fi

if [ -s "$LOG_DIR/skips.jsonl" ]; then
  gap "cpp-oracle-absent:$(wc -l < "$LOG_DIR/skips.jsonl" | tr -d ' ')"
fi

finish
