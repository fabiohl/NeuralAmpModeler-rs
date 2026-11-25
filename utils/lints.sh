#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# lints.sh — Análise estática, formatação e conformidade textual. Classe A.
# Uso: utils/lints.sh
# Orçamento: 130
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

case "${1:-}" in "") ;; -h|--help) help_exit "Uso: utils/lints.sh" ;; *) usage_fail "lints.sh não aceita argumentos ($*)" ;; esac
lib_init
xtrace_side_channel "$LOG_DIR/lints-xtrace.log"
trap 'trap_fail $?' ERR

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
    cargo check --locked --workspace --all-targets $args
  done
}
run_step check phase_check

phase_clippy() {
  for args in "${MATRIX[@]}"; do
    # shellcheck disable=SC2086
    cargo clippy --locked --workspace --all-targets $args -- -D warnings
  done
}
run_step clippy phase_clippy

cross_clippy() {
  local target="$1"
  cargo clippy --locked --lib --target "$target" -- -D warnings
  cargo clippy --locked --workspace --lib --bins --target "$target" -- -D warnings
  cargo clippy --locked -p nam-preflight --all-targets --target "$target" -- -D warnings
}

HOST_ARCH="$(rustc -vV | sed -n 's/^host: //p')"
if [ "$HOST_ARCH" = aarch64-unknown-linux-gnu ]; then
  PHASE_NAME=clippy-x86_64
  if rustup target list --installed | grep -qx x86_64-unknown-linux-gnu; then
    run_step clippy-x86_64 cross_clippy x86_64-unknown-linux-gnu
  else
    gap "clippy-x86_64:native-host-x86-target-not-installed"
  fi
elif ! rustup target list --installed | grep -qx aarch64-unknown-linux-gnu; then
  PHASE_NAME=clippy-aarch64
  gap "clippy-aarch64:target-not-installed"
else
  PHASE_NAME=clippy-aarch64
  run_step clippy-aarch64 cross_clippy aarch64-unknown-linux-gnu
fi
installed_targets="$(rustup target list --installed)"
for spec in clippy-win-x64:x86_64-pc-windows-msvc clippy-win-arm64:aarch64-pc-windows-msvc; do
  p="${spec%%:*}"; t="${spec#*:}"; PHASE_NAME="$p"
  if printf '%s\n' "$installed_targets" | grep -qx "$t"; then
    run_step "$p" cross_clippy "$t"
  else
    gap "$p:target-not-installed"
  fi
done

phase_doc() { RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features; }
run_step doc phase_doc

phase_spdx() {
  local files f rc missing=()
  files=$(git ls-files '*.rs' '*.sh' '*.py' 'build.rs' 'Cargo.toml') || { echo "ERROR: git ls-files failed" >&2; exit 1; }
  [ -n "$files" ] || { echo "ERROR: SPDX enumeration returned no files" >&2; exit 1; }
  while IFS= read -r f; do
    rc=0; grep -qE "SPDX-License-Identifier: (Apache-2\.0|MIT)" "$f" || rc=$?
    case "$rc" in 0) ;; 1) missing+=("$f") ;; *) echo "ERROR: grep failed (rc=$rc) on $f" >&2; exit 1 ;; esac
  done <<< "$files"
  if [ "${#missing[@]}" -gt 0 ]; then echo "Missing/invalid SPDX in:" >&2; printf '%s\n' "${missing[@]}" >&2; exit 1; fi
}
run_step spdx phase_spdx

phase_policies() {
  if git grep -nF "#[test]" tests/common/ >/dev/null 2>&1; then echo "ERROR: #[test] in tests/common/" >&2; exit 1; fi
  local cargo_ver cargo_full cargo_msrv bad doc_cfg feat rc=0
  cargo_ver=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "([0-9]+\.[0-9]+).*"/\1/')
  cargo_full=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "([^"]+)"/\1/')
  cargo_msrv=$(grep -m1 '^rust-version = ' Cargo.toml | sed -E 's/rust-version = "([^"]+)"/\1/')
  grep -qE "NeuralAmpModeler-rs = (\"$cargo_ver\"|\{ version = \"$cargo_ver\")" README.md || { echo "README version mismatch" >&2; rc=1; }
  for f in README.md src/lib.rs; do
    bad=$(grep -nE 'NeuralAmpModeler-rs = ' "$f" | grep -vE "NeuralAmpModeler-rs = \"$cargo_ver(\.[0-9]+)?\"|version = \"$cargo_ver(\.[0-9]+)?\"" || true)
    [ -z "$bad" ] || { echo "$f: install snippet diverges from $cargo_ver" >&2; echo "$bad" >&2; rc=1; }
  done
  grep -m1 '^## \[' CHANGELOG.md | grep -qF "## [$cargo_full] - " || { echo "CHANGELOG header mismatch (expected $cargo_full)" >&2; rc=1; }
  grep -qE "badge/MSRV-$cargo_msrv" README.md || { echo "README MSRV mismatch" >&2; rc=1; }
  doc_cfg=$(git grep -h 'doc(cfg(' src/)
  for feat in $(grep -o -E 'feature = "[^"]+"' <<<"$doc_cfg" | sed -E 's/.*"([^"]+)".*/\1/' | sort -u); do
    grep -qE "^$feat\s*=" Cargo.toml || { echo "Unknown doc(cfg) feature: $feat" >&2; rc=1; }
  done
  git ls-files '*.rs' | xargs awk '
    FNR == 1 { prev = 0 }
    { s = $0; sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s) }
    s ~ /^#!?\[allow\(/ { if (!prev) { print FILENAME ":" FNR ": " s; bad = 1 }; next }
    s ~ /^(\/\/ SPDX|\/\/ Copyright|\/\/\/|\/\/!)/ { prev = 0; next }
    s ~ /^(\/\/([^\/!]|$)|\/\*)/ { prev = 1; next }
    s != "" { prev = 0 }
    END { exit bad }' || rc=1
  return "$rc"
}
run_step policies phase_policies

phase_script_budget() { local f b; for f in $(find utils tests/fixtures -name '*.sh'); do b=$(sed -n 's/^# Orçamento: *\([0-9]*\).*/\1/p;T;q' "$f"); [ -n "$b" ] && [ "$(wc -l < "$f")" -le "$b" ] || { echo "ORÇAMENTO: $f teto=${b:-AUSENTE} linhas=$(wc -l < "$f")" >&2; return 1; }; done; }
run_step script-budget phase_script_budget

finish
