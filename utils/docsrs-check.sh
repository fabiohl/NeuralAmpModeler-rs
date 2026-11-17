#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# docsrs-check.sh — Parity check for docs.rs targets under nightly rustdoc. Classe D.
# Uso: utils/docsrs-check.sh
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh
case "$#:${1:-}" in
  0:) ;;
  1:-h|1:--help) echo "Usage: utils/docsrs-check.sh"; exit 0 ;;
  *) usage_fail "docsrs-check.sh não aceita argumentos ($*)" ;;
esac

lib_init
require_nightly
trap 'trap_fail $?' ERR

docsrs_feature_scope=$(awk '
  /^\[/ { sec = ($0 == "[package.metadata.docs.rs]"); next }
  !sec { next }
  /^all-features *= *true/ { all = 1 }
  /^no-default-features *= *true/ { nodef = 1 }
  /^features *= *\[/ { f = substr($0, index($0, "[") + 1); sub(/\].*$/, "", f); gsub(/[" ,]+/, " ", f); feats = f }
  END {
    a = ""
    if (all) a = a " --all-features"
    if (nodef) a = a " --no-default-features"
    gsub(/^ +| +$/, "", feats)
    if (feats != "" && !all) { gsub(/ /, ",", feats); a = a " --features " feats }
    print a
  }' Cargo.toml)

docsrs_targets_raw=$(awk '
  /^\[/ { sec = ($0 == "[package.metadata.docs.rs]"); in_t = 0; next }
  !sec { next }
  /^targets *= *\[/ { in_t = 1 }
  in_t {
    line = $0
    while (match(line, /"[^"]+"/)) {
      t = t " " substr(line, RSTART + 1, RLENGTH - 2)
      line = substr(line, RSTART + RLENGTH)
    }
    if ($0 ~ /\]/) in_t = 0
  }
  END { gsub(/^ +| +$/, "", t); print t }' Cargo.toml)

docsrs_feature_args=()
docsrs_targets=()
read -r -a docsrs_feature_args <<< "$docsrs_feature_scope"
read -r -a docsrs_targets <<< "$docsrs_targets_raw"
[ "${#docsrs_targets[@]}" -gt 0 ] || { FAILS+=("metadata"); emit "PHASE: docsrs FAIL (no targets in package.metadata.docs.rs)"; finish || true; exit 1; }

export DOCS_RS=1
export RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }--cfg docsrs -D warnings"
installed_targets=$(rustup +nightly target list --installed)

for docsrs_target in "${docsrs_targets[@]}"; do
  PHASE_NAME="docsrs-$docsrs_target"
  if printf '%s\n' "$installed_targets" | grep -qx "$docsrs_target"; then
    run_step "$PHASE_NAME" cargo +nightly doc --locked --lib --no-deps --target "$docsrs_target" "${docsrs_feature_args[@]}"
  else
    gap "$PHASE_NAME:target-not-installed"
  fi
done
finish
