#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# check-model.sh — NAM Model Inspector
#
# Replaces check-model.py as the canonical model inspection tool.
# Delegates all format detection, classification, and metadata extraction
# to the official NeuralAmpModeler-rs loader via `inspect_model`.
#
# Supports .nam (JSON) and .namb (binary) files natively.
#
# Usage:
#   utils/check-model.sh [OPTIONS] <file.nam|.namb> [...]
#
# Options:
#   --json       Emit machine-readable JSON instead of human-readable text
#   --manifest   Batch mode: emit a JSON array with one entry per file
#   --release    Build in release mode (faster inference, recommended for large .namb)
#   --help, -h   Show this help message
#
# Examples:
#   utils/check-model.sh model.nam
#   utils/check-model.sh model.namb --json
#   utils/check-model.sh --manifest models/*.nam models/*.namb
#   utils/check-model.sh model.nam --release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Option parsing ────────────────────────────────────────────────────────────
RELEASE_FLAG=""
PASS_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --release)
            RELEASE_FLAG="--release"
            ;;
        --help|-h)
            cat >&2 <<HELP
NAM Model Inspector — powered by NeuralAmpModeler-rs (inspect_model example)

USAGE:
  utils/check-model.sh [OPTIONS] <file.nam|.namb> [...]

OPTIONS:
  --json       Emit machine-readable JSON instead of human-readable text
  --manifest   Batch mode: emit a JSON array with one entry per file
  --release    Build with optimizations (recommended for large .namb files)
  --help, -h   Show this help message

EXAMPLES:
  utils/check-model.sh model.nam
  utils/check-model.sh model.namb --json
  utils/check-model.sh --manifest models/*.nam models/*.namb
  utils/check-model.sh model.nam --release

NOTES:
  Supports both .nam (JSON) and .namb (binary) formats natively.
  All format detection and classification is performed by the official loader.
  Errors are reported via NamDiagnostic error codes — no silent failures.
HELP
            exit 0
            ;;
        *)
            PASS_ARGS+=("$arg")
            ;;
    esac
done

if [[ ${#PASS_ARGS[@]} -eq 0 ]]; then
    echo "Error: No model file(s) specified." >&2
    echo "Run 'utils/check-model.sh --help' for usage." >&2
    exit 1
fi

# ── Build the example (incremental — fast on rebuild) ─────────────────────────
cd "${CRATE_ROOT}"
cargo build ${RELEASE_FLAG} --example inspect_model --quiet 2>&1 \
    | grep -v "^$" \
    | grep -v "Compiling\|Finished\|Checking" \
    >&2 || true

# Determine binary path
if [[ -n "$RELEASE_FLAG" ]]; then
    BINARY="${CRATE_ROOT}/target/release/examples/inspect_model"
else
    BINARY="${CRATE_ROOT}/target/debug/examples/inspect_model"
fi

if [[ ! -x "$BINARY" ]]; then
    echo "Error: inspect_model binary not found at $BINARY" >&2
    echo "Try running: cargo build --example inspect_model" >&2
    exit 2
fi

# ── Execute the inspector ─────────────────────────────────────────────────────
exec "$BINARY" "${PASS_ARGS[@]}"
