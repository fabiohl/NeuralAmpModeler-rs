#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# docsrs-check.sh — Parity check for docs.rs under nightly rustdoc. Classe D.
# Uso: utils/docsrs-check.sh
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
case "$#:${1:-}" in
  0:) ;;
  1:-h|1:--help) echo "Usage: utils/docsrs-check.sh"; exit 0 ;;
  *) echo "Usage: utils/docsrs-check.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh

PHASE_NAME=docsrs; phase docsrs
require_nightly

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

docsrs_feature_args=()
read -r -a docsrs_feature_args <<< "$docsrs_feature_scope"

export DOCS_RS=1
export RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }--cfg docsrs -D warnings"

for docsrs_target in "" aarch64-unknown-linux-gnu; do
  docsrs_target_args=()
  if [ -n "$docsrs_target" ]; then
    rustup +nightly target list --installed | grep -qx "$docsrs_target" || {
      echo "GAP: nightly target $docsrs_target not installed (rustup +nightly target add $docsrs_target)" >&2; exit 2;
    }
    docsrs_target_args=(--target "$docsrs_target")
  fi
  if ! cargo +nightly doc --locked --lib --no-deps "${docsrs_target_args[@]}" "${docsrs_feature_args[@]}"; then
    echo "FAIL: rustdoc emitted warnings/errors on target ${docsrs_target:-host}." >&2
    exit 1
  fi
done
ok "rustdoc emitted zero warnings on x86_64 and aarch64."
finish
