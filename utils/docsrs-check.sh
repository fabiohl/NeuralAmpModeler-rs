#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# docs.rs parity check (opt-in, nightly) — NeuralAmpModeler-rs
# =============================================================================
#
# docs.rs renders the crate page with the nightly rustdoc and `--cfg docsrs`.
# This script mirrors that configuration locally so documentation regressions
# (broken intra-doc links, redundant explicit link targets, missing
# `doc(cfg(...))` feature seals) surface before publishing. Those diagnostics
# are emitted only by nightly rustdoc and are invisible to the stable toolchain
# used by `utils/lints.sh`.
#
# GOVERNANCE
#   Nightly is intentionally isolated here and is NOT wired into
#   `utils/lints.sh`, `utils/tests-quick.sh` or `utils/tests-long.sh`, all of
#   which stay strictly on the stable toolchain (rules/coding.md §1,
#   rules/quality.md §2). This is an opt-in, out-of-band check.
#
# FLAGS
#   DOCS_RS=1             : makes build.rs skip the x86-64-v3 preflight, the
#                           same environment variable docs.rs sets.
#   --cfg docsrs          : enables `#![feature(doc_cfg)]` and the
#                           `doc(cfg(...))` seals — mirrors
#                           `package.metadata.docs.rs.rustdoc-args`.
#   -D warnings           : any rustdoc warning fails the check (fail-closed).
#   --lib                 : required because the crate root emits a
#                           `compile_error!` when a non-`cfg(doc)` build skips
#                           the x86-64-v3 target-feature preflight.
#   <feature scope>       : derived at runtime from `package.metadata.docs.rs`
#                           (`all-features` / `no-default-features` / `features`)
#                           so this script cannot silently drift from Cargo.toml.
#   --no-deps             : documents the crate only; third-party crates are
#                           not re-documented (matches docs.rs scope and keeps
#                           `-D warnings` scoped to this crate).
#
# EXIT CODES
#   0  documentation builds with zero warnings
#   1  rustdoc warnings or documentation build failure
#   2  typed GAP (nightly toolchain or metadata tooling unavailable) —
#      fail-closed, never a silent success
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "=================================================================="
echo -e "      NeuralAmpModeler-rs docs.rs Parity Check (nightly)          "
echo -e "=================================================================="

if ! command -v rustup >/dev/null 2>&1; then
    echo "GAP: rustup not found — docs.rs check cannot run." >&2
    echo "  This is a typed gap: install a nightly toolchain with rustup." >&2
    exit 2
fi

if ! rustup toolchain list 2>/dev/null | grep -q "nightly"; then
    echo "GAP: no nightly toolchain installed — docs.rs check cannot run." >&2
    echo "  Install via: rustup toolchain install nightly --profile minimal" >&2
    exit 2
fi

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
