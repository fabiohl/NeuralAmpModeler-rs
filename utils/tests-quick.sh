#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# tests-quick.sh — Execução da disciplina rápida de testes. Classe A.
# Uso: utils/tests-quick.sh
# Orçamento: 41
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) echo "Uso: utils/tests-quick.sh"; exit 0 ;; *) echo "RESULT: FAIL (uso)"; exit 2 ;; esac
trap 'rc=$?; FAILS+=("${PHASE_NAME:-step}"); emit "PHASE: ${PHASE_NAME:-step} FAIL rc=$rc log=${PHASE_LOG:-none}"; finish || true; exit 1' ERR
rm -f "$LOG_DIR/skips.jsonl" "$LOG_DIR/tests-quick-status.tsv"

PHASE_NAME=freshness; phase freshness
log_fresh="$LOG_DIR/quick-freshness.log"; rc_fresh=0; PHASE_LOG="$log_fresh"
cargo run --quiet --locked -p nam-qa --bin nam_freshness -- artifacts-hard > "$log_fresh" 2>&1 || rc_fresh=$?
case "$rc_fresh" in
  0) emit "PHASE: freshness PASS log=$log_fresh" ;;
  3) gap "generator-drift"; emit "PHASE: freshness PASS log=$log_fresh" ;;
  *) FAILS+=("freshness"); emit "PHASE: freshness FAIL rc=$rc_fresh log=$log_fresh"; finish || true; exit 1 ;;
esac

PHASE_NAME=tests; phase tests
log_tests="$LOG_DIR/quick-tests.log"; rc_tests=0; PHASE_LOG="$log_tests"
cargo test --locked --features testing,pinned --no-fail-fast > "$log_tests" 2>&1 || rc_tests=$?
grep -E 'FAILED|panicked at' "$log_tests" || true
awk '/test result: / { p+=$4; f+=$6; i+=$8 } END { printf "Tests summary: %d passed, %d failed, %d ignored\n", p, f, i }' "$log_tests"
if [ "$rc_tests" -eq 0 ] && ran_tests "$log_tests"; then
  emit "PHASE: tests PASS log=$log_tests"
else
  FAILS+=("tests"); emit "PHASE: tests FAIL rc=$rc_tests log=$log_tests"; finish || true; exit 1
fi

if [ -s "$LOG_DIR/skips.jsonl" ]; then
  gap "skip-capability:$(wc -l < "$LOG_DIR/skips.jsonl" | tr -d ' ')"
fi

finish || exit 1
