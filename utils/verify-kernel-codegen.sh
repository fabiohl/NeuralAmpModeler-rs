#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# verify-kernel-codegen.sh — opt-in / operator static RT-codegen guard for the
# engine's own DSP kernels.
#
# The engine does not inspect its own `.rlib`; instead this guard disassembles a
# release binary that links and exercises the kernels (an `examples/` binary by
# default) and runs the unified NAM codegen filter
# (`utils/dsp-hotpath-filter.py`) with the engine symbol profile
# (`utils/codegen-profile.json`, T5.2): zero `zmm`, zero `libm` calls on the RT
# path and zero `alloc`/`__rust_alloc` calls among the selected kernel symbols.
#
# Opt-in by design: it builds a release binary and requires `objdump`, so it is
# not part of the default quick/lint gates.
#
# Usage:
#   utils/verify-kernel-codegen.sh [options]
#
# Options:
#   --bin PATH                  Inspect an existing binary (skips the build)
#   --example NAME              Release example to build (default: offline_render)
#   --profile PATH              Symbol profile (default: utils/codegen-profile.json)
#   --test-synthetic-violation  Inject synthetic failure for fail-closed proof
#
# Environment:
#   NAM_KERNEL_CODEGEN_BIN      Same as --bin
#   NAM_KERNEL_CODEGEN_EXAMPLE  Same as --example
#   NAM_KERNEL_CODEGEN_PROFILE  Same as --profile

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/_lib.sh"

# ── Memory guard (OOM containment) ──────────────────────────────────────────
# Builds a release example; contain the compile peak so it cannot trigger a
# global OOM that kills the IDE. Fail-open when containment is unavailable.
if [ "${NAM_MEM_GUARD:-0}" != "1" ] && [ "${NAM_NO_MEM_GUARD:-0}" != "1" ]; then
    export NAM_MEM_GUARD=1
    exec "$SCRIPT_DIR/mem-guard.sh" "$SCRIPT_PATH" "$@"
fi

FILTER="$SCRIPT_DIR/dsp-hotpath-filter.py"
PROFILE="${NAM_KERNEL_CODEGEN_PROFILE:-$SCRIPT_DIR/codegen-profile.json}"
EXAMPLE="${NAM_KERNEL_CODEGEN_EXAMPLE:-offline_render}"
TARGET_BIN="${NAM_KERNEL_CODEGEN_BIN:-}"
TEST_SYNTHETIC=0

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)
            TARGET_BIN="${2:?--bin requires a path}"
            shift 2
            ;;
        --example)
            EXAMPLE="${2:?--example requires a name}"
            shift 2
            ;;
        --profile)
            PROFILE="${2:?--profile requires a path}"
            shift 2
            ;;
        --test-synthetic-violation)
            TEST_SYNTHETIC=1
            shift
            ;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

OBJDUMP_TOOL="${OBJDUMP_TOOL:-$(command -v llvm-objdump || command -v objdump || true)}"
PYTHON_TOOL="${PYTHON_BIN:-$(command -v python3 || true)}"

if [ -z "$OBJDUMP_TOOL" ] || [ ! -x "$OBJDUMP_TOOL" ]; then
    die "Required disassembly tool 'objdump' not found or not executable (fail-closed)."
fi
if [ -z "$PYTHON_TOOL" ] || [ ! -x "$PYTHON_TOOL" ]; then
    die "Required script runner 'python3' not found or not executable (fail-closed)."
fi
if [ ! -f "$FILTER" ]; then
    die "Missing shared codegen filter: $FILTER (fail-closed)."
fi
if [ ! -f "$PROFILE" ]; then
    die "Missing codegen symbol profile: $PROFILE (fail-closed)."
fi

# Build the release example when no binary was supplied. Symbols are retained
# (strip disabled) so the kernel functions remain addressable in the dump.
if [ -z "$TARGET_BIN" ]; then
    echo -e "  ${BLUE}Building release example '${EXAMPLE}' with kernel symbols...${NC}"
    CARGO_PROFILE_RELEASE_STRIP=false cargo build --locked --release --example "$EXAMPLE"
    TARGET_BIN="$PROJECT_DIR/target/release/examples/$EXAMPLE"
fi

if [ ! -s "$TARGET_BIN" ]; then
    die "Kernel-codegen target binary missing or empty: $TARGET_BIN (fail-closed)."
fi

echo -e "  ${BLUE}Inspecting binary:${NC} $TARGET_BIN"
echo -e "  ${BLUE}Tools:${NC} objdump=$OBJDUMP_TOOL, python=$PYTHON_TOOL, profile=$PROFILE"

ASM_FULL="$(mktemp -t nam-kernel-codegen.XXXXXX.asm)"
ASM_FILTERED="$(mktemp -t nam-kernel-codegen-filtered.XXXXXX.asm)"
ASM_SUMMARY="$(mktemp -t nam-kernel-codegen-summary.XXXXXX.txt)"
trap 'rm -f "$ASM_FULL" "$ASM_FILTERED" "$ASM_SUMMARY"' EXIT

LC_ALL=C "$OBJDUMP_TOOL" -d --demangle --no-show-raw-insn "$TARGET_BIN" > "$ASM_FULL" 2>/dev/null || true
if [ ! -s "$ASM_FULL" ]; then
    die "Disassembly produced empty output for $TARGET_BIN (stripped binary?) (fail-closed)."
fi

FILTER_ARGS=(--full "$ASM_FULL" --filtered "$ASM_FILTERED" --summary "$ASM_SUMMARY" --profile "$PROFILE")
if [ "$TEST_SYNTHETIC" -eq 1 ]; then
    FILTER_ARGS+=(--test-synthetic-violation)
fi

if ! "$PYTHON_TOOL" "$FILTER" "${FILTER_ARGS[@]}"; then
    echo -e "  ${RED}${BOLD}Kernel codegen gate FAILED (see summary above).${NC}" >&2
    die "Kernel codegen verification failed for $TARGET_BIN"
fi

ok "Kernel codegen verification passed cleanly."
exit 0
