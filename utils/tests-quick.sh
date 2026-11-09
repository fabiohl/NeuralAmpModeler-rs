#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# tests-quick.sh — Execução da disciplina rápida de testes. Classe A.
# Uso: utils/tests-quick.sh
# Orçamento: 60
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) echo "Uso: utils/tests-quick.sh"; exit 0 ;; *) usage_fail "tests-quick.sh não aceita argumentos ($*)" ;; esac
lib_init
PHASE_NAME=init; PHASE_LOG="$LOG_DIR/quick-init.log"
trap 'trap_fail $?' ERR
SKIPS="$LOG_DIR/skips-${NAM_RUN_ID}.jsonl"

phase freshness
log_fresh="$LOG_DIR/quick-freshness.log"; rc_fresh=0; PHASE_LOG="$log_fresh"
cargo run --quiet --locked -p nam-qa --bin nam_freshness -- artifacts-hard > "$log_fresh" 2>&1 || rc_fresh=$?
case "$rc_fresh" in
  0) emit "PHASE: freshness PASS log=$log_fresh" ;;
  3) gap "freshness:generator-drift" ;;
  *) FAILS+=("freshness"); emit "PHASE: freshness FAIL rc=$rc_fresh log=$log_fresh" ;;
esac

phase tests
log_tests="$LOG_DIR/quick-tests.log"; rc_tests=0; PHASE_LOG="$log_tests"
cargo test --locked --features testing,pinned --no-fail-fast > "$log_tests" 2>&1 || rc_tests=$?
if [ "$rc_tests" -ne 0 ]; then
  grep -E '(panicked at|FAILED)' "$log_tests" || { echo "--- Cargo/build failure (tail of $log_tests) ---" >&2; tail -n 25 "$log_tests" >&2; }
fi
awk '/test result: / { p+=$4; f+=$6; i+=$8 } END { printf "Tests summary: %d passed, %d failed, %d ignored\n", p, f, i }' "$log_tests"
# Measured: x86_64=2228 passed (2026-11-08); threshold=2000
if [ "$rc_tests" -eq 0 ] && assert_ran_tests "$log_tests" 2000; then
  emit "PHASE: tests PASS log=$log_tests"
else
  FAILS+=("tests"); emit "PHASE: tests FAIL rc=$rc_tests log=$log_tests"; finish || exit 1; exit 1
fi

if [ -s "$SKIPS" ]; then
  blocking=$(grep -vc '"optional":true' "$SKIPS" 2>/dev/null || true)
  [ "${blocking:-0}" -gt 0 ] && gap "skip-capability:$blocking"
fi

finish || exit 1
