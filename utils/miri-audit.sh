#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Out-of-band Miri Concurrency & Aliasing Audit — NeuralAmpModeler-rs
# =============================================================================
#
# GOVERNANCE & POLICY NOTICE
#   In accordance with the project quality policy (docs/testing.md §13),
#   all CI and standard release verification gates (lints.sh, tests-quick.sh,
#   tests-long.sh) use strictly the stable Rust toolchain.
#
#   This script is an OPTIONAL, out-of-tree, opt-in periodic audit for unsafe
#   invariants, data races and Stacked Borrows / Tree Borrows aliasing soundness
#   investigated under the Miri MIR interpreter. It is intentionally NOT wired
#   into lints.sh / tests-quick.sh / tests-long.sh and requires an isolated
#   nightly toolchain.
#
# RELATION TO LOOM
#   Miri complements `loom` (utils/tests-long.sh phase 6) and does NOT replace
#   it: loom proves the absence of data races by exhaustively exploring the
#   atomic interleavings, whereas Miri proves pointer provenance and aliasing
#   validity (UB detection) for the concrete executions it interprets.
#
# TARGET MATRIX (each target must execute a real minimum number of tests; a
# declared target with zero executed tests is a hard failure — never a silent
# pass):
#   - common::spsc              SPSC ring, GC cascade/overflow, status flags
#   - dsp::pipeline::bridge     DspBridge double-buffer lock-free protocol
#   - dsp::mirror_buf           MirroredBuffer state machine (FFI alloc excluded)
#   - dsp::smoother             One-pole smoother primitives
#   - dsp::utils::delay_line    DelayLine ring arithmetic
#
# FLAGS
#   -Zmiri-disable-isolation : allows host clock / pseudo-sleeps in timing paths.
#   -Zmiri-ignore-leaks      : the GC overflow tests deliberately leak displaced
#                              items (the GcOverflowBuffer controlled-leak
#                              contract) and the corrupted-slot tests leak
#                              on purpose; without this flag Miri reports those
#                              intentional leaks as errors and drowns the
#                              provenance/aliasing signal we are auditing.
#
# EXIT CODES
#   0  all targets green
#   1  audit failure (UB, unsupported operation, or an empty/under-covered target)
#   2  typed GAP (nightly toolchain or miri component unavailable) — fail-closed,
#      never a silent success
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHASE_TOTAL=5
source "$SCRIPT_DIR/_lib.sh"

LOG_DIR="target/logs"
mkdir -p "$LOG_DIR"
MIRI_LOG="$LOG_DIR/miri-audit.log"
: > "$MIRI_LOG"

echo -e "${BLUE}${BOLD}=======================================================${NC}"
echo -e "${BLUE}${BOLD}   NeuralAmpModeler-rs Miri Concurrency & UB Audit     ${NC}"
echo -e "${BLUE}${BOLD}=======================================================${NC}"

# ── Toolchain preflight (fail-closed GAP) ────────────────────────────────────
if ! command -v rustup >/dev/null 2>&1; then
    echo -e "${YELLOW}GAP: rustup not found — Miri audit cannot run.${NC}"
    echo -e "  This is a typed gap: install a nightly toolchain with the miri component."
    exit 2
fi

if ! rustup toolchain list 2>/dev/null | grep -q "nightly"; then
    echo -e "${YELLOW}GAP: no nightly toolchain installed — Miri audit cannot run.${NC}"
    echo -e "  Install via:"
    echo -e "    ${BOLD}rustup toolchain install nightly --profile minimal${NC}"
    echo -e "    ${BOLD}rustup component add miri --toolchain nightly${NC}"
    exit 2
fi

if ! cargo +nightly miri --version >/dev/null 2>&1; then
    echo -e "${YELLOW}GAP: miri component missing on nightly — Miri audit cannot run.${NC}"
    echo -e "  Install via:"
    echo -e "    ${BOLD}rustup component add miri --toolchain nightly${NC}"
    exit 2
fi

echo -e "${GREEN}Miri environment detected:${NC}"
cargo +nightly miri --version | tee -a "$MIRI_LOG"
echo ""

# Ensure the Miri sysroot is pre-compiled/cached. Override [build] warnings =
# "deny" from .cargo/config.toml for the sysroot generator invocation.
CARGO_BUILD_WARNINGS=allow cargo +nightly miri setup >/dev/null 2>&1 || true

export MIRIFLAGS="${MIRIFLAGS:--Zmiri-disable-isolation -Zmiri-ignore-leaks}"

# run_target <label> <lib-filter> <min-tests>
# Runs one Miri target, tees its output into target/logs/miri-audit.log and
# fails closed when the target executes fewer than `min-tests` real tests.
run_target() {
    local label="$1" filter="$2" min="$3"
    phase "Miri target: ${label} (${filter})"

    local tlog
    tlog="$(mktemp)"

    if ! cargo +nightly miri test --lib "$filter" 2>&1 | tee "$tlog"; then
        cat "$tlog" >> "$MIRI_LOG"
        rm -f "$tlog"
        die "Miri target '${label}' failed (UB, unsupported operation, or test failure). See ${MIRI_LOG}."
    fi

    if ! assert_ran_tests "$tlog" "$min"; then
        cat "$tlog" >> "$MIRI_LOG"
        rm -f "$tlog"
        die "Miri target '${label}' executed fewer than ${min} test(s) — refusing a vacuous pass."
    fi

    cat "$tlog" >> "$MIRI_LOG"
    rm -f "$tlog"
    ok "Target '${label}': ≥ ${min} test(s) executed green"
}

# Minimums are the real counts observed on the pinned nightly (miri 0.1.0,
# 2026-10-11). Lowering a minimum requires re-measuring the target.
run_target "common::spsc"           "common::spsc"              28
run_target "dsp::pipeline::bridge"  "dsp::pipeline::bridge"      4
run_target "dsp::mirror_buf"        "dsp::mirror_buf"            3
run_target "dsp::smoother"          "dsp::smoother"              5
run_target "dsp::utils::delay_line" "dsp::utils::delay_line"    16

echo ""
ok "Miri concurrency and aliasing audit completed successfully (all targets green)."
echo -e "  Full log: ${MIRI_LOG}"
