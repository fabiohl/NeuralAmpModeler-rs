#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# docsrs-check.sh — Parity check for docs.rs under nightly rustdoc. Classe D.
# Uso: utils/docsrs-check.sh
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

echo -e "=================================================================="
echo -e "      NeuralAmpModeler-rs docs.rs Parity Check (nightly)          "
echo -e "=================================================================="

require_nightly

# Derive the feature scope from `package.metadata.docs.rs` so the check mirrors
# exactly what docs.rs builds instead of duplicating the configuration here.
if ! command -v python3 >/dev/null 2>&1; then
    echo "GAP: python3 not found — cannot derive the docs.rs feature scope from Cargo.toml." >&2
    exit 2
fi

if ! docsrs_feature_scope=$(python3 - <<'PY'
import tomllib

with open("Cargo.toml", "rb") as f:
    data = tomllib.load(f)

meta = (
    data.get("package", {})
    .get("metadata", {})
    .get("docs", {})
    .get("rs", {})
)
args = []
if meta.get("all-features"):
    args.append("--all-features")
if meta.get("no-default-features"):
    args.append("--no-default-features")
features = meta.get("features") or []
if features and not meta.get("all-features"):
    args += ["--features", ",".join(features)]
print(" ".join(args))
PY
); then
    echo "GAP: could not parse package.metadata.docs.rs from Cargo.toml (requires python3 + tomllib)." >&2
    exit 2
fi

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
