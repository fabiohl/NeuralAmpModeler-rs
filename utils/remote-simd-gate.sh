#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# remote-simd-gate.sh — remote AVX-512 parity and benchmarking tool. Classe D.
# Uso: utils/remote-simd-gate.sh [flags canônicas]
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

USE_SDE=0; CHECK_ONLY=0; SKIP_BENCH=0; RECEIPT_OUT="target/logs/remote-simd-receipt.jsonl"
mkdir -p "$LOG_DIR"
trap 'emit "RESULT: FAIL ($((SECONDS - T0))s) fails=${PHASE_NAME:-simd-gate} gaps=${GAPS[*]:-none}"' ERR

while [ $# -gt 0 ]; do
  case "$1" in
    --sde) USE_SDE=1; SKIP_BENCH=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --out) RECEIPT_OUT="${2:?--out requires a file}"; shift 2 ;;
    -h|--help) echo "Uso: utils/remote-simd-gate.sh [--sde] [--check-only] [--skip-bench] [--out <file>]"; exit 0 ;;
    *) echo "RESULT: FAIL (uso)"; exit 2 ;;
  esac
done

PHASE_NAME=preflight; phase preflight
log_probe="$LOG_DIR/remote-simd-probe.log"
if [ "$USE_SDE" -eq 1 ]; then
  if command -v sde64 >/dev/null 2>&1; then export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde64 -spr --"
  elif command -v sde >/dev/null 2>&1; then export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="sde -spr --"
  elif [[ ! "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-}" =~ sde ]]; then
    echo "ERROR: --sde specified but neither sde64 nor sde found in PATH" >&2
    FAILS+=("preflight"); emit "PHASE: preflight FAIL rc=2 log=$log_probe"; finish || true; exit 2
  fi
  echo "Hardware: Intel SDE Emulation" > "$log_probe"; emit "PHASE: preflight PASS log=$log_probe"
elif [[ "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-}" =~ sde ]]; then
  echo "Hardware: Intel SDE Emulation" > "$log_probe"; emit "PHASE: preflight PASS log=$log_probe"
else
  rc_probe=0
  cargo run --quiet --bin simd_probe -- --check-avx512 > "$log_probe" 2>&1 || rc_probe=$?
  case "$rc_probe" in
    0) emit "PHASE: preflight PASS log=$log_probe" ;;
    3) GAPS+=("avx512-not-detected"); emit "PHASE: preflight GAP avx512-not-detected"; finish; exit 0 ;;
    *) FAILS+=("preflight"); emit "PHASE: preflight FAIL rc=$rc_probe log=$log_probe"; finish || true; exit 1 ;;
  esac
fi

if [ "$CHECK_ONLY" -eq 1 ]; then finish; exit 0; fi

PHASE_NAME=parity; phase parity
log_parity="$LOG_DIR/remote-simd-parity.log"; rc_parity=0
cargo test --release --features avx512 --test parity -- isa_parity --include-ignored --test-threads=1 > "$log_parity" 2>&1 || rc_parity=$?
n_exercised=$(grep -c -E "^AVX512_MATRIX: EXERCISED" "$log_parity" || true)
if [ "$rc_parity" -eq 0 ] && ran_tests "$log_parity" && [ "$n_exercised" -gt 0 ]; then
  emit "PHASE: parity PASS log=$log_parity"
else
  FAILS+=("parity"); emit "PHASE: parity FAIL rc=${rc_parity:-1} log=$log_parity"; finish || true; exit 1
fi

if [ "$USE_SDE" -eq 0 ] && [ "$SKIP_BENCH" -eq 0 ]; then
  PHASE_NAME=bench; phase bench
  log_bench="$LOG_DIR/remote-simd-bench.log"; rc_bench=0
  cargo bench --features avx512 --bench inference_bench -- ISA_Compare > "$log_bench" 2>&1 || rc_bench=$?
  cargo run --quiet -p nam-qa --bin nam_remote_simd_receipt -- --criterion-dir "target/criterion" --out "$RECEIPT_OUT" >> "$log_bench" 2>&1 || rc_bench=$?
  if [ "$rc_bench" -eq 0 ]; then emit "PHASE: bench PASS log=$log_bench"
  else FAILS+=("bench"); emit "PHASE: bench FAIL rc=$rc_bench log=$log_bench"; finish || true; exit 1
  fi
fi

finish
