#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# consumer-smoke.sh — Contrato de consumidor externo (x86-64-v3/aarch64) por artefato estruturado. Classe D.
# Uso: utils/consumer-smoke.sh
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) echo "Uso: utils/consumer-smoke.sh"; exit 0 ;; *) echo "RESULT: FAIL (uso)"; exit 2 ;; esac
lib_init
trap 'rc=$?; FAILS+=("${PHASE_NAME:-consumer-smoke}"); emit "PHASE: ${PHASE_NAME:-consumer-smoke} FAIL rc=$rc log=${log:-$LOG_DIR/consumer-smoke.log}"; finish || true; exit 1' ERR

PHASE_NAME=consumer-smoke; phase consumer-smoke
log="$LOG_DIR/consumer-smoke.log"
d=$(mktemp -d "${TMPDIR:-/tmp}/nam-consumer-XXXXXX")
trap 'rm -rf "$d"' EXIT
mkdir -p "$d/src"
printf 'use neural_amp_modeler_rs::prelude::*;\npub fn smoke() { let _ = ProcessingParams::default(); let _ = NamErrorCode::FileNotFound; }\n' > "$d/src/lib.rs"
printf '[package]\nname = "nam-consumer"\nversion = "0.0.0"\nedition = "2024"\n\n[dependencies]\nNeuralAmpModeler-rs = { path = "%s" }\n' "$PWD" > "$d/Cargo.toml"
rc=0
(
  set -e
  unset CARGO_TARGET_DIR || true
  cd "$d"
  for t in x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin; do
    rustup target list --installed | grep -qx "$t" || { echo "ERROR: target $t missing (run: rustup target add $t)" >&2; exit 1; }
  done
  if env -u RUSTFLAGS cargo check --target x86_64-unknown-linux-gnu --message-format=json > bare.json 2> bare.log; then
    echo "consumer x86 sem flag: verde (esperado FAIL)" >&2; exit 1
  fi
  [ "$(tail -n 1 bare.json)" = '{"reason":"build-finished","success":false}' ] || { echo "consumer x86 sem flag: cargo não reportou build-finished=false" >&2; tail -n 1 bare.json >&2; exit 1; }
  contract=$(find target -type f -name nam-contract.txt -print -quit)
  [ -n "$contract" ] || { echo "consumer x86 sem flag: artefato de contrato ausente" >&2; cat bare.log >&2; exit 1; }
  code= role=
  while IFS='=' read -r k v; do
    case "$k" in code) code=$v ;; role) role=$v ;; esac
  done < "$contract"
  [ "$code" = x86-64-v3-missing ] && [ "$role" = dependency ] || {
    echo "consumer x86 sem flag: causa inesperada (code=${code:-?} role=${role:-?})" >&2
    cat "$contract" bare.log >&2; exit 1
  }
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
