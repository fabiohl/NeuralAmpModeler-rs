#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# consumer-smoke.sh — Verifica contrato de consumidor externo (x86-64-v3 e aarch64). Classe D.
# Uso: utils/consumer-smoke.sh
# Orçamento: 45
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) echo "Uso: utils/consumer-smoke.sh"; exit 0 ;; *) echo "RESULT: FAIL (uso)"; exit 2 ;; esac
trap 'emit "RESULT: FAIL ($((SECONDS - T0))s) fails=${PHASE_NAME:-consumer-smoke} gaps=none"' ERR

PHASE_NAME=consumer-smoke; phase consumer-smoke
log="$LOG_DIR/consumer-smoke.log"
d=$(mktemp -d "${TMPDIR:-/tmp}/nam-consumer-XXXXXX")
trap 'rm -rf "$d"' EXIT
mkdir -p "$d/src" && : > "$d/src/lib.rs"
printf '[package]\nname = "nam-consumer"\nversion = "0.0.0"\nedition = "2024"\n\n[dependencies]\nNeuralAmpModeler-rs = { path = "%s" }\n' "$PWD" > "$d/Cargo.toml"
rc=0
(
  set -e
  unset CARGO_TARGET_DIR || true
  cd "$d"
  for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin; do
    rustup target list --installed | grep -qx "$t" || { echo "ERROR: target $t missing (run: rustup target add $t)" >&2; exit 1; }
  done
  if env -u RUSTFLAGS cargo check --target x86_64-unknown-linux-gnu > bare.log 2>&1; then
    echo "consumer x86 sem flag: verde (esperado FAIL)" >&2; exit 1
  fi
  grep -q "compiled as a dependency" bare.log || { echo "consumer x86 sem flag: mensagem de consumidor ausente" >&2; cat bare.log >&2; exit 1; }
  for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin; do
    flags=""; [ "$t" = x86_64-unknown-linux-gnu ] && flags="-Ctarget-cpu=x86-64-v3"
    RUSTFLAGS="$flags" cargo check --target "$t" || { echo "consumer $t: FAIL" >&2; exit 1; }
  done
) > "$log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  cat "$log" >&2 || true
  FAILS+=("consumer-smoke"); emit "PHASE: consumer-smoke FAIL rc=$rc log=$log"; finish || true; exit 1
fi
emit "PHASE: consumer-smoke PASS log=$log"
finish

