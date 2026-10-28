#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# lints.sh — Análise estática, formatação e conformidade textual. Classe A.
# Uso: utils/lints.sh
# Orçamento: 100
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) echo "Uso: utils/lints.sh"; exit 0 ;; *) echo "RESULT: FAIL (uso)"; exit 2 ;; esac
trap 'emit "RESULT: FAIL ($((SECONDS - T0))s) fails=${PHASE_NAME:-step} gaps=none"' ERR

run_step() {
  local name=$1; shift; PHASE_NAME=$name; phase "$name"
  local log="$LOG_DIR/lints-$name.log" rc
  set +e; ( set -e; "$@" ) 2>&1 | tee "$log"; rc=${PIPESTATUS[0]}; set -e
  [ "$rc" -eq 0 ] || { FAILS+=("$name"); emit "PHASE: $name FAIL rc=$rc log=$log"; finish || true; exit 1; }
  emit "PHASE: $name PASS log=$log"
}

phase_fmt() {
  if [ "${CI:-}" = "true" ]; then cargo fmt --all -- --check; else cargo fmt --all && git diff --stat; fi
}
run_step fmt phase_fmt

MATRIX=(
  ""
  "--no-default-features"
  "--all-features"
  "--no-default-features --features testing,heap-audit,fft-radix4-planner,rt-hardening"
)
phase_check() {
  for args in "${MATRIX[@]}"; do
    # shellcheck disable=SC2086
    cargo check --locked --all-targets $args
  done
}
run_step check phase_check

phase_clippy() {
  for args in "${MATRIX[@]}"; do
    # shellcheck disable=SC2086
    cargo clippy --locked --all-targets $args -- -D warnings
  done
}
run_step clippy phase_clippy

phase_doc() { RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features; }
run_step doc phase_doc

phase_spdx() {
  local missing
  missing=$(git ls-files 'src/*.rs' 'tests/*.rs' 'benches/*.rs' 'examples/*.rs' 'utils/*.sh' 'tests/*.sh' 'tests/fixtures/*.py' 'build.rs' 'Cargo.toml' | xargs grep -L -E "SPDX-License-Identifier: (Apache-2\.0|MIT)" || true)
  if [ -n "$missing" ]; then echo "Missing/invalid SPDX in:" >&2; echo "$missing" >&2; exit 1; fi
}
run_step spdx phase_spdx

phase_policies() {
  if git grep -nF "#[test]" tests/common/ >/dev/null 2>&1; then echo "ERROR: #[test] in tests/common/" >&2; exit 1; fi
  local cargo_ver cargo_msrv
  cargo_ver=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "([0-9]+\.[0-9]+).*"/\1/')
  cargo_msrv=$(grep -m1 '^rust-version = ' Cargo.toml | sed -E 's/rust-version = "([^"]+)"/\1/')
  grep -qE "NeuralAmpModeler-rs = (\"$cargo_ver\"|\{ version = \"$cargo_ver\")" README.md || { echo "README version mismatch" >&2; exit 1; }
  grep -qE "badge/MSRV-$cargo_msrv" README.md || { echo "README MSRV mismatch" >&2; exit 1; }
  for feat in $(git grep -h -o -E 'doc\(cfg\(feature = "[^"]+"\)\)' src/ | sed -E 's/.*"([^"]+)".*/\1/' | sort -u); do
    grep -qE "^$feat\s*=" Cargo.toml || { echo "Unknown doc(cfg) feature: $feat" >&2; exit 1; }
  done
  git ls-files '*.rs' | xargs awk '
    FNR == 1 { prev = 0 }
    { s = $0; sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s) }
    s ~ /^#!?\[allow\(/ { if (!prev) { print FILENAME ":" FNR ": " s; bad = 1 }; next }
    s ~ /^(\/\/ SPDX|\/\/ Copyright|\/\/\/|\/\/!)/ { prev = 0; next }
    s ~ /^(\/\/([^\/!]|$)|\/\*)/ { prev = 1; next }
    s != "" { prev = 0 }
    END { exit bad }'
}
run_step policies phase_policies

phase_script_budget() { local f b; for f in $(find utils -name '*.sh'); do b=$(sed -n 's/^# Orçamento: *\([0-9]*\).*/\1/p;T;q' "$f"); [ -n "$b" ] && [ "$(wc -l < "$f")" -le "$b" ] || { echo "ORÇAMENTO: $f teto=${b:-AUSENTE} linhas=$(wc -l < "$f")" >&2; return 1; }; done; }
run_step script-budget phase_script_budget

finish
