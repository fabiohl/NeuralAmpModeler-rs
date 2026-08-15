#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# repro_oversample_hang.sh — General-purpose safety wrapper for reproducing
# a suspected indefinite hang. Originally built to investigate and fix a
# real ELF symbol-interposition hang (see
# docs/postmortem-libm-symbol-interposition.md for the full writeup and the
# lessons learned) — kept as a general reusable tool for any future hang
# investigation, not specific to that one bug.
#
# Diagnostic Tool: not intended for automated CI gates. Use for manual
# debugging of thread hangs under resource limits.
#
# This is a *debug/investigation* tool, deliberately NOT part of the official
# `utils/tests-*.sh` suite. It never runs unattended CI/nightly checks; it
# exists solely so a human operator (or an AI) can run a single reproduction
# candidate command with:
#   1. Resource isolation (systemd-run --user --scope + cgroup v2 limits), so a
#      runaway process cannot OOM/starve the host or crash the graphical session.
#   2. An aggressive external kill timeout, since the internal cargo/test-harness
#      timeout is known to be unreliable once a real hang is in play.
#   3. A durable, timestamped log + exit-code artifact under target/debug-logs/,
#      plus an automatic dmesg tail, so every attempt is fully reproducible and
#      post-mortem-able without relying on scrollback.
#
# Usage:
#   utils/debug/repro_oversample_hang.sh <timeout_seconds> <label> -- <cargo full command...>
#
# Example:
#   utils/debug/repro_oversample_hang.sh 15 varA-stable-debug -- \
#     cargo test --lib -- "dsp::oversample::oversample_test::test_x2_aliasing_rejection" \
#       --ignored --nocapture --test-threads=1
#
# Exit codes:
#   0        — command completed normally within the timeout (success or an
#              ordinary test failure/assertion; inspect the log for details).
#   128+N    — the whole cgroup was terminated by `RuntimeMaxSec` (systemd
#              sends SIGTERM=15 first, hence the near-universally observed
#              143; escalates to SIGKILL=9/137 if the process ignores TERM).
#              Treat ANY exit >= 128 as HANG/RESOURCE-KILL.
#   2        — usage error (bad arguments); never confuse this with the above.
#   other    — whatever the wrapped command itself returned (ordinary pass/fail).
#
# CRITICAL FIX (verified live during the investigation this tool was built
# for — see docs/postmortem-libm-symbol-interposition.md §2 for the general
# lesson): an earlier version of this script ran
# `timeout -s KILL "$TIMEOUT_S" "$@"` *inside* the systemd-run scope. This
# was PROVEN LIVE to leak the actual test binary as an orphaned, still-running
# process pinned at ~100% CPU *after* the script had already printed
# "exit 124" and returned control to the caller — `timeout` only reliably
# signals its own direct child (`cargo`), not the grandchild test binary that
# `cargo test` spawns, and the scope's default `KillMode=process` does not
# clean up the rest of the cgroup on its own. This is exactly the mechanism
# that must be assumed capable of exhausting host resources unattended if
# this script is ever run without someone watching `ps`/`systemctl` afterward.
#
# FIX: the timeout is now expressed as `-p RuntimeMaxSec=<N>` on the
# `systemd-run` scope itself (no inner `timeout` process at all). systemd
# enforces this by sending SIGTERM then SIGKILL to *every* process in the
# scope's cgroup — verified empirically to leave zero residual processes. Do
# not reintroduce an inner `timeout -s KILL` — it is the confirmed root
# cause of a real, reproduced-live orphan/resource leak, not a theoretical
# concern.
#
# Secondary fix retained from the previous revision: no `| tee` pipe for the
# authoritative exit-status capture (a `${PIPESTATUS[0]}` read after `tee`
# separately produced a false-negative exit 0 for a log truncated exactly
# like a confirmed hang). Output is redirected straight to the log file and
# `$?` is read immediately, unambiguously.

set -uo pipefail

# Resolve project root dynamically so `target/debug-logs/` always lands at the
# repository root, regardless of the caller's current working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

usage() {
    echo "Usage: $0 <timeout_seconds> <label> -- <cargo full command...>" >&2
    echo "Example: $0 15 varA-stable-debug -- cargo test --lib -- <test_name> --ignored --nocapture --test-threads=1" >&2
    exit 2
}

if [ "$#" -lt 4 ]; then
    usage
fi

TIMEOUT_S="$1"
LABEL="$2"
shift 2

if [ "$1" = "--" ]; then
    shift
else
    echo -e "${RED}${BOLD}Error: expected '--' before the cargo command.${NC}" >&2
    usage
fi

if [ "$#" -eq 0 ]; then
    echo -e "${RED}${BOLD}Error: no cargo command provided after '--'.${NC}" >&2
    usage
fi

case "$TIMEOUT_S" in
    ''|*[!0-9]*)
        echo -e "${RED}${BOLD}Error: <timeout_seconds> must be a positive integer (received: '${TIMEOUT_S}').${NC}" >&2
        usage
        ;;
esac

# Restrict <label> to a safe charset: it becomes part of a filesystem path
# below (target/debug-logs/<label>.log|.exit). Without this check a crafted
# label containing '/' or '..' could traverse outside that directory and
# write/overwrite arbitrary files reachable by the current user.
case "$LABEL" in
    ''|*[!A-Za-z0-9_-]*)
        echo -e "${RED}${BOLD}Error: <label> must contain only [A-Za-z0-9_-] (received: '${LABEL}').${NC}" >&2
        usage
        ;;
esac

if ! command -v systemd-run >/dev/null 2>&1; then
    echo -e "${RED}${BOLD}Error: systemd-run not found (systemd user scope execution is unavailable in this environment).${NC}" >&2
    exit 2
fi

mkdir -p target/debug-logs
LOG="target/debug-logs/${LABEL}.log"
EXIT_FILE="target/debug-logs/${LABEL}.exit"

echo "=== $(date -Is) :: ${LABEL} :: RuntimeMaxSec=${TIMEOUT_S}s :: cmd: $* ===" >"$LOG"
cat "$LOG"

# No inner `timeout` process — see the CRITICAL FIX note above. The scope's
# own `RuntimeMaxSec` is the *only* time limit, and it is enforced by systemd
# against the whole cgroup (every descendant process), not just the direct
# child. No pipe into `tee` for the authoritative status capture either (see
# the "Secondary fix" note above) — `$?` right below is read directly.
systemd-run --user --scope --collect \
    -p MemoryMax=1G -p MemorySwapMax=0 -p CPUQuota=100% -p TasksMax=64 \
    -p "RuntimeMaxSec=${TIMEOUT_S}" \
    -- "$@" >>"$LOG" 2>&1
STATUS=$?

echo "$STATUS" >"$EXIT_FILE"

# Post-run safety net: confirm the cgroup really is gone and nothing from
# this invocation survived. This is not optional cosmetics — the exact
# failure mode this guards against (a still-running, ~100% CPU orphaned test
# binary, minutes after the script already printed a result and exited) was
# reproduced live with an earlier revision of this script.
LEFTOVER_PROC="$(pgrep -af 'target/(debug|release)/deps/' 2>/dev/null | grep -v "$$" || true)"
if [ -n "$LEFTOVER_PROC" ]; then
    echo -e "${RED}${BOLD}!!! WARNING: residual process(es) detected after script completion — kill manually: !!!${NC}" | tee -a "$LOG"
    echo "$LEFTOVER_PROC" | tee -a "$LOG"
fi

if [ "$STATUS" -ge 128 ]; then
    echo -e "${RED}${BOLD}!!! HANG/RESOURCE-KILL detected (exit ${STATUS}) !!!${NC}" | tee -a "$LOG"
elif [ "$STATUS" -eq 0 ]; then
    # Post-hoc consistency check: an exit 0 whose log ends mid-line exactly
    # like libtest's "test <name> ... " in-progress marker (no trailing
    # "ok"/"FAILED"/"ignored", no "test result:" summary anywhere in the
    # log) is almost certainly a false negative, not a real pass — flag it
    # loudly instead of trusting the exit code blindly.
    if ! grep -q "^test result:" "$LOG"; then
        echo -e "${RED}${BOLD}!!! SUSPICIOUS: exit 0 but no 'test result:' line in log — likely false negative. Treat as INCONCLUSIVE, not success. !!!${NC}" | tee -a "$LOG"
    else
        echo -e "${GREEN}${BOLD}--- Command completed normally (exit 0) ---${NC}" | tee -a "$LOG"
    fi
else
    echo -e "${YELLOW}${BOLD}--- Command finished with exit ${STATUS} (not a hang/OOM) ---${NC}" | tee -a "$LOG"
fi

echo "--- dmesg tail ---" | tee -a "$LOG"
dmesg -T 2>/dev/null | tail -n 20 | tee -a "$LOG"

exit "$STATUS"
