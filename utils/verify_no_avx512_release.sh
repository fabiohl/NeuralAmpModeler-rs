#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Binary artifact inspection guard: verifies zero AVX-512 symbols and zero
# EVEX/ZMM instructions in default (non-avx512) release compilation artifacts.
#
# Usage:
#   utils/verify_no_avx512_release.sh [path/to/artifact.rlib|.so|binary]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

TARGET="${1:-$PROJECT_DIR/target/release/libneural_amp_modeler_rs.rlib}"

if [ ! -f "$TARGET" ]; then
    echo -e "  ${YELLOW}Target artifact not found: $TARGET. Building default release...${NC}"
    cargo build --release --locked
fi

if [ ! -f "$TARGET" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Artifact not found after build: $TARGET${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve LLVM / binutils inspection tools (prefer matching rustc sysroot)
# ---------------------------------------------------------------------------
SYSROOT="$(rustc --print sysroot 2>/dev/null || true)"
HOST="$(rustc -vV 2>/dev/null | sed -n 's/host: //p' || true)"

NM_BIN=""
OBJDUMP_BIN=""

if [ -n "$SYSROOT" ] && [ -n "$HOST" ]; then
    if [ -x "$SYSROOT/lib/rustlib/$HOST/bin/llvm-nm" ]; then
        NM_BIN="$SYSROOT/lib/rustlib/$HOST/bin/llvm-nm"
    fi
    if [ -x "$SYSROOT/lib/rustlib/$HOST/bin/llvm-objdump" ]; then
        OBJDUMP_BIN="$SYSROOT/lib/rustlib/$HOST/bin/llvm-objdump"
    fi
fi

if [ -z "$NM_BIN" ]; then
    for tool in llvm-nm llvm-nm-21 nm; do
        if command -v "$tool" >/dev/null 2>&1; then
            NM_BIN="$(command -v "$tool")"
            break
        fi
    done
fi

if [ -z "$OBJDUMP_BIN" ]; then
    for tool in llvm-objdump llvm-objdump-21 objdump; do
        if command -v "$tool" >/dev/null 2>&1; then
            OBJDUMP_BIN="$(command -v "$tool")"
            break
        fi
    done
fi

if [ -z "$NM_BIN" ] && [ -z "$OBJDUMP_BIN" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Neither nm nor objdump found. Cannot inspect binary artifact.${NC}"
    exit 1
fi

echo -e "  ${CYAN}Scanning artifact:${NC} $TARGET"
[ -n "$NM_BIN" ] && echo -e "  ${CYAN}Using nm tool:${NC} $NM_BIN"
[ -n "$OBJDUMP_BIN" ] && echo -e "  ${CYAN}Using objdump tool:${NC} $OBJDUMP_BIN"

FAILURES=0

# ---------------------------------------------------------------------------
# 1. Symbol scan: check for forbidden AVX-512 symbols in symbol table
# ---------------------------------------------------------------------------
if [ -n "$NM_BIN" ]; then
    FORBIDDEN_SYMBOLS_REGEX='gemv_4gate_avx512|dot_product_4x_f32_avx512|Avx512Math|process_sample_avx512|process_avx512|simd_.*_avx512|_slice_avx512|_mm512_|avx512::'
    
    # Run nm with demangling; ignore errors if specific sub-object has no symbols (e.g. rmeta)
    NM_OUT="$("$NM_BIN" --demangle "$TARGET" 2>/dev/null || "$NM_BIN" -C "$TARGET" 2>/dev/null || true)"
    
    # Filter for symbol definitions (lines with symbol types like T, t, W, w, D, d, R, r)
    MATCHED_SYMBOLS="$(echo "$NM_OUT" | grep -iE "$FORBIDDEN_SYMBOLS_REGEX" || true)"
    
    if [ -n "$MATCHED_SYMBOLS" ]; then
        echo -e "  ${RED}${BOLD}❌ FAIL: Forbidden AVX-512 symbols detected in $TARGET:${NC}"
        echo "$MATCHED_SYMBOLS" | head -n 30 | sed 's/^/    /'
        FAILURES=$((FAILURES + 1))
    else
        ok "Symbol scan: zero AVX-512 symbols found."
    fi
fi

# ---------------------------------------------------------------------------
# 2. Disassembly scan: check for EVEX/ZMM machine instructions in .text
# ---------------------------------------------------------------------------
if [ -n "$OBJDUMP_BIN" ]; then
    FORBIDDEN_DISASM_REGEX='\bzmm[0-9]+\b|\bxmm(1[6-9]|2[0-9]|3[0-1])\b|\bymm(1[6-9]|2[0-9]|3[0-1])\b|\bvpdpbusd\b|\bvfmadd[0-9]*.*zmm\b|\bvmovups.*zmm\b|\bvbroadcast.*zmm\b'
    
    DISASM_OUT="$("$OBJDUMP_BIN" -d "$TARGET" 2>/dev/null || true)"
    
    MATCHED_DISASM="$(echo "$DISASM_OUT" | grep -iE "$FORBIDDEN_DISASM_REGEX" || true)"
    
    if [ -n "$MATCHED_DISASM" ]; then
        echo -e "  ${RED}${BOLD}❌ FAIL: Forbidden EVEX/ZMM instructions detected in $TARGET:${NC}"
        echo "$MATCHED_DISASM" | head -n 30 | sed 's/^/    /'
        FAILURES=$((FAILURES + 1))
    else
        ok "Disassembly scan: zero EVEX/ZMM instructions found."
    fi
fi

if [ "$FAILURES" -gt 0 ]; then
    echo -e "  ${RED}${BOLD}Binary inspection failed with $FAILURES violation(s).${NC}"
    exit 1
fi

ok "Binary scan passed: clean x86-64-v3 baseline without AVX-512 leaks."
exit 0
