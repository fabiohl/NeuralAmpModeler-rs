#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# check-model.sh — NAM Model Inspector
#
# Canonical model inspection tool for NeuralAmpModeler-rs.
# Delegates format detection, topology analysis, DSP profile, and metadata
# extraction to the official loader via the `inspect_model` example.
#
# Supports .nam (JSON) and .namb (binary) files natively.
#
# Usage:
#   utils/check-model.sh [OPTIONS] <file.nam|.namb|directory> [...]
#
# Options:
#   -d, --dir <DIR>  Recursively scan directory for compatible model files (.nam, .namb)
#   --json           Emit machine-readable JSON instead of human-readable text
#   --manifest       Batch mode: emit a JSON array with one entry per file
#   --release        Build/run with release optimizations (recommended for large .namb files)
#   --help, -h       Show this help message
#
# Examples:
#   utils/check-model.sh model.nam
#   utils/check-model.sh model.namb --json
#   utils/check-model.sh -d models/
#   utils/check-model.sh --dir=models/ --json
#   utils/check-model.sh --manifest -d models/
#   utils/check-model.sh --manifest models/*.nam models/*.namb
#   utils/check-model.sh model.nam --release

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Help message ──────────────────────────────────────────────────────────────
show_help() {
    cat >&2 <<HELP
NAM Model Inspector — powered by NeuralAmpModeler-rs (inspect_model)

USAGE:
  utils/check-model.sh [OPTIONS] <file.nam|.namb|directory> [...]

OPTIONS:
  -d, --dir <DIR>  Recursively scan directory for compatible model files (.nam, .namb)
  --json           Emit machine-readable JSON instead of human-readable text
  --manifest       Batch mode: emit a JSON array with one entry per file
  --release        Build/run with optimizations (recommended for large .namb files)
  --help, -h       Show this help message

EXAMPLES:
  utils/check-model.sh model.nam
  utils/check-model.sh model.namb --json
  utils/check-model.sh -d models/
  utils/check-model.sh --dir=models/ --json
  utils/check-model.sh --manifest -d models/
  utils/check-model.sh --manifest models/*.nam models/*.namb
  utils/check-model.sh model.nam --release

NOTES:
  Supports both .nam (JSON) and .namb (binary) formats natively.
  When a directory is specified, it is searched recursively for compatible models.
  All format detection and classification is performed by the official loader.
HELP
}

# ── Option parsing ────────────────────────────────────────────────────────────
RELEASE_FLAG=""
INSPECT_FLAGS=()
TARGET_FILES=()
DIRS_TO_SCAN=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release)
            RELEASE_FLAG="--release"
            shift
            ;;
        --json|--manifest)
            INSPECT_FLAGS+=("$1")
            shift
            ;;
        -d|--dir|--directory)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Error: Option '$1' requires a directory argument." >&2
                echo "Run 'utils/check-model.sh --help' for usage." >&2
                exit 1
            fi
            DIRS_TO_SCAN+=("$2")
            shift 2
            ;;
        -d=*|--dir=*|--directory=*)
            DIR_VAL="${1#*=}"
            if [[ -z "$DIR_VAL" ]]; then
                echo "Error: Option '${1%%=*}' requires a directory argument." >&2
                echo "Run 'utils/check-model.sh --help' for usage." >&2
                exit 1
            fi
            DIRS_TO_SCAN+=("$DIR_VAL")
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        -*)
            echo "Error: Unknown option '$1'." >&2
            echo "Run 'utils/check-model.sh --help' for usage." >&2
            exit 1
            ;;
        *)
            if [[ -d "$1" ]]; then
                DIRS_TO_SCAN+=("$1")
            else
                TARGET_FILES+=("$1")
            fi
            shift
            ;;
    esac
done

# ── Process directories ───────────────────────────────────────────────────────
for scan_dir in "${DIRS_TO_SCAN[@]}"; do
    if [[ ! -d "$scan_dir" ]]; then
        echo "Error: Directory not found: '$scan_dir'" >&2
        exit 1
    fi

    FOUND_IN_DIR=()
    while IFS= read -r -d $'\0' file; do
        [[ -n "$file" ]] && FOUND_IN_DIR+=("$file")
    done < <(find -L "$scan_dir" -type f \( -iname "*.nam" -o -iname "*.namb" \) -print0 2>/dev/null | sort -z)

    if [[ ${#FOUND_IN_DIR[@]} -eq 0 ]]; then
        echo "Error: No compatible model files (.nam, .namb) found in directory: '$scan_dir'" >&2
        exit 1
    fi

    TARGET_FILES+=("${FOUND_IN_DIR[@]}")
done

if [[ ${#TARGET_FILES[@]} -eq 0 ]]; then
    echo "Error: No model file(s) or directory specified." >&2
    echo "Run 'utils/check-model.sh --help' for usage." >&2
    exit 1
fi

# ── Execute the inspector via cargo run --locked ─────────────────────────────
cd "${CRATE_ROOT}"

# Direct, fast execution with cargo run --locked: handles incremental build
# and execution atomically without manual target path resolution.
exec cargo run --locked ${RELEASE_FLAG} --example inspect_model --quiet -- "${INSPECT_FLAGS[@]}" "${TARGET_FILES[@]}"
