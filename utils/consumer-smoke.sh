#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# consumer-smoke.sh — Contrato de consumidor externo sobre .crate extraído. Classe D.
# Uso: utils/consumer-smoke.sh
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) help_exit "Uso: utils/consumer-smoke.sh" ;; *) usage_fail "consumer-smoke.sh não aceita argumentos ($*)" ;; esac
lib_init
trap 'trap_fail $?' ERR

PHASE_NAME=consumer-smoke; phase consumer-smoke
log="$LOG_DIR/consumer-smoke.log"
d=$(mktemp -d "${TMPDIR:-/tmp}/nam-consumer-XXXXXX")
trap 'rm -rf "$d"' EXIT
rc=0
(
  set -e
  unset CARGO_TARGET_DIR || true
  pkg_args=(--locked --no-verify --target-dir "$d/pkg")
  [ -n "$(git status --porcelain 2>/dev/null || true)" ] && pkg_args+=(--allow-dirty)
  cargo package "${pkg_args[@]}"
  ver=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)
  mkdir -p "$d/crate" "$d/consumer/src"
  tar -xzf "$d/pkg/package/NeuralAmpModeler-rs-$ver.crate" -C "$d/crate"
  crate_dir="$d/crate/NeuralAmpModeler-rs-$ver"
  printf 'use neural_amp_modeler_rs::prelude::*;\npub fn smoke() { let _ = ProcessingParams::default(); let _ = NamErrorCode::FileNotFound; }\n' > "$d/consumer/src/lib.rs"
  printf '[package]\nname = "nam-consumer"\nversion = "0.0.0"\nedition = "2024"\n[dependencies]\nNeuralAmpModeler-rs = { path = "%s", default-features = false }\n[features]\ndefault = ["NeuralAmpModeler-rs/default"]\ntesting = ["NeuralAmpModeler-rs/testing"]\nfft-radix4-planner = ["NeuralAmpModeler-rs/fft-radix4-planner"]\nrt-hardening = ["NeuralAmpModeler-rs/rt-hardening"]\nheap-audit = ["NeuralAmpModeler-rs/heap-audit"]\n' "$crate_dir" > "$d/consumer/Cargo.toml"
  cd "$d/consumer"
  installed=$(rustup target list --installed)
  targets=(x86_64-unknown-linux-gnu aarch64-unknown-linux-gnu aarch64-apple-darwin)
  for t in "${targets[@]}"; do
    printf '%s\n' "$installed" | grep -qx "$t" || { echo "ERROR: target $t missing (run: rustup target add $t)" >&2; exit 1; }
  done
  for wt in x86_64-pc-windows-msvc aarch64-pc-windows-msvc; do
    if printf '%s\n' "$installed" | grep -qx "$wt"; then targets+=("$wt"); else echo "GAP:target-not-installed:$wt" >> "$d/gaps.txt"; fi
  done
  if env -u RUSTFLAGS cargo check --target x86_64-unknown-linux-gnu --message-format=json > bare.json 2> bare.log; then
    echo "consumer x86 sem flag: verde (esperado FAIL)" >&2; exit 1
  fi
  [ "$(tail -n 1 bare.json)" = '{"reason":"build-finished","success":false}' ] || { echo "consumer x86 sem flag: cargo não reportou build-finished=false" >&2; tail -n 1 bare.json >&2; exit 1; }
  contract=$(find target -type f -name nam-contract.txt -print -quit)
  [ -n "$contract" ] || { echo "consumer x86 sem flag: artefato de contrato ausente" >&2; cat bare.log >&2; exit 1; }
  code= role=
  while IFS='=' read -r k v; do case "$k" in code) code=$v ;; role) role=$v ;; esac; done < "$contract"
  [ "$code" = x86-64-v3-missing ] && [ "$role" = dependency ] || { echo "consumer x86 sem flag: causa inesperada (code=${code:-?} role=${role:-?})" >&2; cat "$contract" bare.log >&2; exit 1; }
  for t in "${targets[@]}"; do
    case "$t" in x86_64-*) flags="-Ctarget-cpu=x86-64-v3" ;; *) flags="" ;; esac
    RUSTFLAGS="$flags" cargo check --target "$t" || { echo "consumer $t: FAIL" >&2; exit 1; }
  done
  host_t=$(rustc -vV | sed -n 's/^host: //p')
  host_f=""; case "$host_t" in x86_64-*) host_f="-Ctarget-cpu=x86-64-v3" ;; esac
  for f in "--no-default-features" "--features testing,fft-radix4-planner,rt-hardening" "--no-default-features --features testing,fft-radix4-planner,rt-hardening,heap-audit"; do
    # shellcheck disable=SC2086
    RUSTFLAGS="$host_f" cargo check --target "$host_t" $f || { echo "consumer $host_t ($f): FAIL" >&2; exit 1; }
  done
  RUSTFLAGS="$host_f" cargo build --target "$host_t" || { echo "consumer link ($host_t): FAIL" >&2; exit 1; }
  cd "$crate_dir"
  RUSTFLAGS="$host_f" cargo build --locked --examples
  RUSTFLAGS="$host_f" cargo run --locked --example synthetic_model
  RUSTFLAGS="$host_f" cargo run --locked --example cabsim
  RUSTFLAGS="$host_f" cargo run --locked --example math_activations
) > "$log" 2>&1 || rc=$?
if [ "$rc" -ne 0 ]; then
  cat "$log" >&2 || true; FAILS+=("consumer-smoke"); emit "PHASE: consumer-smoke FAIL rc=$rc log=$log"; finish || true; exit 1
fi
if [ -s "$d/gaps.txt" ]; then
  while IFS= read -r g; do gap "consumer-smoke:${g#GAP:}"; done < "$d/gaps.txt"
else
  emit "PHASE: consumer-smoke PASS log=$log"
fi
finish
