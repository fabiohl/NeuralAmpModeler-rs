#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# remote-simd-gate.sh — remote AVX-512 parity and benchmarking tool. Classe D.
# Uso: utils/remote-simd-gate.sh [flags canônicas]
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

USE_SDE=0; CHECK_ONLY=0; SKIP_BENCH=0; RECEIPT_OUT="target/logs/remote-simd-receipt.jsonl"
mkdir -p target/logs

while [ $# -gt 0 ]; do
  case "$1" in
    --sde) USE_SDE=1; SKIP_BENCH=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --skip-bench) SKIP_BENCH=1; shift ;;
    --out) RECEIPT_OUT="${2:?--out requires a file}"; shift 2 ;;
    -h|--help)
      echo "Uso: utils/remote-simd-gate.sh [--sde] [--check-only] [--skip-bench] [--out <file>]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [ "$USE_SDE" -eq 1 ] || [[ "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-}" =~ sde ]]; then
  echo "Hardware: Intel SDE Emulation"
else
  PROBE_OUT=$(cargo run --quiet --bin simd_probe 2>&1 || true)
  if ! echo "$PROBE_OUT" | grep -qE "avx512f:\s*YES"; then
    echo "AVX-512 features not detected on host CPU"
    exit 2
  fi
fi

echo "Running AVX-512 mathematical cross-ISA parity..."
cargo test --release --features avx512 --test parity -- isa_parity --include-ignored --test-threads=1 2>&1 | tee target/logs/remote-simd-parity.log

if [ "$USE_SDE" -eq 1 ]; then
  echo "parity-only (benchmarks skipped under SDE emulation)"
elif [ "$CHECK_ONLY" -eq 0 ] && [ "$SKIP_BENCH" -eq 0 ]; then
  echo "Running Criterion ISA comparison benchmarks..."
  cargo bench --features avx512 --bench inference_bench -- ISA_Compare
  if [ -f "src/bin/nam_remote_simd_receipt.rs" ]; then
    cargo run --quiet --features testing --bin nam_remote_simd_receipt -- --criterion-dir "target/criterion" --out "$RECEIPT_OUT" || true
  fi
fi
echo "RESULT: PASS remote-simd-gate"
