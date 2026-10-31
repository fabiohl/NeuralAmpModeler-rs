#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# miri-audit.sh — Out-of-band Miri concurrency and UB audit. Classe D.
# Uso: utils/miri-audit.sh
# Orçamento: 100
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/miri-audit.sh"; exit 0 ;;
    *) echo "Usage: utils/miri-audit.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh
rm -f "target/logs/miri-audit-status.tsv"
trap 'rc=$?; [ "$rc" -eq 0 ] || [ "$rc" -eq 2 ] || [ -n "${DONE:-}" ] || { FAILS+=(miri-audit); DONE=1; finish; }' EXIT

LOG_DIR="target/logs"
mkdir -p "$LOG_DIR"
MIRI_LOG="$LOG_DIR/miri-audit.log"
: > "$MIRI_LOG"

echo "NeuralAmpModeler-rs Miri Concurrency & UB Audit"
echo ""

# ── Toolchain preflight (fail-closed GAP) ────────────────────────────────────
require_nightly miri

echo "Miri environment detected:"
cargo +nightly miri --version | tee -a "$MIRI_LOG"
echo ""

# Ensure the Miri sysroot is pre-compiled/cached. Override [build] warnings =
# "deny" from .cargo/config.toml for the sysroot generator invocation.
# Setup diagnostics are kept on stdout and in MIRI_LOG; a failed setup aborts (rc=1)
# before any target runs, so it can never be mistaken for a prepared sysroot.
setup_rc=0
CARGO_BUILD_WARNINGS=allow cargo +nightly miri setup 2>&1 | tee -a "$MIRI_LOG" || setup_rc=$?
[ "$setup_rc" -eq 0 ] || die "Miri setup failed (rc=${setup_rc}); no Miri target was executed. See ${MIRI_LOG}."
ok "Miri sysroot prepared (miri setup rc=0)"

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
echo "  Full log: ${MIRI_LOG}"
DONE=1; finish
