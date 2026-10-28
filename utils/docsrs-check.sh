#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# docsrs-check.sh — Parity check for docs.rs under nightly rustdoc. Classe D.
# Uso: utils/docsrs-check.sh
# Orçamento: 80
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

echo -e "=================================================================="
echo -e "      NeuralAmpModeler-rs docs.rs Parity Check (nightly)          "
echo -e "=================================================================="

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

if cargo +nightly doc --locked --lib --no-deps "${docsrs_feature_args[@]}"; then
  echo "OK: rustdoc (nightly, docsrs cfg, derived feature scope) emitted zero warnings."
else
  rc=$?
  echo "FAIL: rustdoc emitted warnings/errors (exit ${rc})." >&2
  exit 1
fi
