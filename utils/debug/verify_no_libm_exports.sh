#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# verify_no_libm_exports.sh — ELF surface guard for libm symbol leakage.
#
# Scans the compiled Rust library artifact for global or weak exports of
# the 7 libm symbols that must remain hidden:
#   log10f, atan2f, acosf        (trampolines from global_asm!)
#   cbrt, cbrtf, fma, fmod       (weak symbols from compiler_builtins)
#
# Usage:
#   utils/debug/verify_no_libm_exports.sh [target-dir|target-file]
#     target-dir/file: optional override (default: target/release)
#
# Exit codes:
#   0 — clean: no libm symbols leaked
#   1 — leak detected

set -euo pipefail

TARGET_ARG="${1:-target/release}"

RUSTLIB_PATH=""
if [ -f "$TARGET_ARG" ]; then
    RUSTLIB_PATH="$TARGET_ARG"
elif [ -f "$TARGET_ARG/libneural_amp_modeler_rs.rlib" ]; then
    RUSTLIB_PATH="$TARGET_ARG/libneural_amp_modeler_rs.rlib"
else
    # Fallback: the most recent hashed rlib under target/<profile>/deps.
    # `cargo test --release` (tests-long.sh Phase 1) produces only hashed
    # artifacts there; a direct `cargo build` also emits the unhashed
    # top-level copy checked above. Either one is a valid scan target.
    RUSTLIB_PATH="$(ls -t "$TARGET_ARG"/deps/libneural_amp_modeler_rs-*.rlib 2>/dev/null | head -1 || echo "")"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

LIBM_SYMBOLS=("log10f" "atan2f" "acosf" "cbrt" "cbrtf" "fma" "fmod")

if [ ! -f "$RUSTLIB_PATH" ]; then
    echo -e "${YELLOW}Artifact not found: $RUSTLIB_PATH${NC}"
    echo -e "${YELLOW}Run 'cargo build --release' first. Skipping verification.${NC}"
    exit 0
fi

# Determine the best available introspection tool
NM=""
if command -v nm &>/dev/null; then
    NM="nm"
elif command -v llvm-nm &>/dev/null; then
    NM="llvm-nm"
fi

READELF=""
if command -v readelf &>/dev/null; then
    READELF="readelf"
elif command -v llvm-readelf &>/dev/null; then
    READELF="llvm-readelf"
fi

TMPDIR=""
cleanup_tmp() {
    if [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi
}
trap cleanup_tmp EXIT

LEAKED=""

if [ -n "$NM" ]; then
    # nm prints symbol-address + symbol-type + symbol-name.
    # 'T' = global text, 'W' = weak global. Hex address prefix ensures defined export.
    OUTPUT=$("$NM" "$RUSTLIB_PATH" 2>/dev/null || true)
    for sym in "${LIBM_SYMBOLS[@]}"; do
        if echo "$OUTPUT" | grep -qE "^[0-9a-fA-F]+[[:space:]]+[TW][[:space:]]+$sym[[:space:]]*$"; then
            LEAKED="$LEAKED $sym"
        fi
    done
else
    # Fallback: extract .o files using portable (cd "$TMPDIR" && ar x ...) and scan
    TMPDIR=$(mktemp -d)
    (cd "$TMPDIR" && ar x "$RUSTLIB_PATH" 2>/dev/null || true)
    for obj in "$TMPDIR"/*.o; do
        [ -f "$obj" ] || continue
        if [ -n "$READELF" ]; then
            OUTPUT=$("$READELF" -s "$obj" 2>/dev/null || true)
            for sym in "${LIBM_SYMBOLS[@]}"; do
                # Check for GLOBAL or WEAK bindings while excluding UND / UNDEF section index imports
                if echo "$OUTPUT" | grep -E "\b(GLOBAL|WEAK)\b" | grep -vE "\bUND(EF)?\b" | grep -qE "\b$sym\b"; then
                    if ! echo "$LEAKED" | grep -qw "$sym"; then
                        LEAKED="$LEAKED $sym"
                    fi
                fi
            done
        else
            # Last resort: grep raw strings in object files
            for sym in "${LIBM_SYMBOLS[@]}"; do
                if strings "$obj" 2>/dev/null | grep -qx "$sym"; then
                    if ! echo "$LEAKED" | grep -qw "$sym"; then
                        LEAKED="$LEAKED $sym"
                    fi
                fi
            done
        fi
    done
fi

LEAKED="${LEAKED# }"

if [ -n "$LEAKED" ]; then
    echo -e "${RED}${BOLD}❌ LIBSYMBOL LEAK DETECTED${NC}"
    echo -e "${RED}   The following libm symbols are exported as global or weak:${NC}"
    for sym in $LEAKED; do
        echo -e "       ${RED}$sym${NC}"
    done
    exit 1
fi

echo -e "${GREEN}✓ No libm symbols leaked (all 7 hidden).${NC}"
exit 0
