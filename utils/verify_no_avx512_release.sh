#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Fail-closed binary certification for the default (non-avx512) release build.
#
# Sprint 1 / F-ROB-01 protocol:
#   1. Rebuilds the default release artifact in a disposable isolated target
#      dir (`target/cert-release-XXXXXX`); a stale `target/release` rlib is
#      never reused.
#   2. Disables thin-LTO for the certification build so the `.rlib` archive
#      members are disassemblable ELF objects (thin-LTO stores LLVM IR, which
#      contains no machine code to certify).
#   3. Logs the SHA-256 of the produced artifact before inspection.
#   4. Requires `llvm-objdump` and `llvm-nm` (rustc sysroot preferred). A
#      missing tool, non-zero tool exit, empty tool output, or undecodable
#      archive member aborts with exit 1 — never a silent PASS.
#   5. Delegates the scan to the single Rust scanner (`nam_bin_guard scan`),
#      which inspects every ELF member of the `.rlib` and every executable
#      section for the EVEX (`0x62`) encoding prefix.
#
# Usage:
#   utils/verify_no_avx512_release.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_TOTAL=3
source "$SCRIPT_DIR/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}   NeuralAmpModeler-rs Binary Certification (Zero AVX-512)      ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

CERT_DIR="$(mktemp -d "$PROJECT_DIR/target/cert-release-XXXXXX")"
trap 'rm -rf "$CERT_DIR"' EXIT

phase "Building default release in isolated certification dir..."
(
    cd "$PROJECT_DIR"
    CARGO_TARGET_DIR="$CERT_DIR/target" CARGO_PROFILE_RELEASE_LTO=off \
        cargo build --release --locked --no-default-features --lib
)

TARGET="$CERT_DIR/target/release/libneural_amp_modeler_rs.rlib"
if [ ! -f "$TARGET" ]; then
    echo -e "  ${RED}${BOLD}ERROR: certification artifact was not produced: $TARGET${NC}"
    exit 1
fi

HASH="$(sha256sum "$TARGET" | cut -d' ' -f1)"
if [ -z "$HASH" ]; then
    die "failed to compute SHA-256 of $TARGET"
fi
echo -e "  ${CYAN}Certification artifact SHA-256:${NC} $HASH"

phase "Building QA scanner (nam_bin_guard)..."
(
    cd "$PROJECT_DIR"
    cargo build --quiet --locked --features testing --bin nam_bin_guard
)

GUARD_BIN="$PROJECT_DIR/target/debug/nam_bin_guard"
if [ ! -x "$GUARD_BIN" ]; then
    die "nam_bin_guard was not produced at $GUARD_BIN"
fi

phase "Scanning artifact for EVEX/AVX-512 instructions..."
if ! "$GUARD_BIN" scan "$TARGET"; then
    echo -e "  ${RED}${BOLD}Binary certification FAILED for sha256=$HASH${NC}"
    exit 1
fi

ok "Binary certification passed: zero EVEX/AVX-512 in isolated default release (sha256=$HASH)."
exit 0
