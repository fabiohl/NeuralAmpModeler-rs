#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Automated regression test suite for performance regression gate, stale log cleanup,
# and quality dashboard fail-closed protection.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ANSI style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$PROJECT_DIR"

echo -e "${BLUE}${BOLD}===============================================================${NC}"
echo -e "${BLUE}${BOLD}    Regression Guard Shell Test Suite (Automated T0.5)         ${NC}"
echo -e "${BLUE}${BOLD}===============================================================${NC}"

START_TIME=$(date +%s%N)
TOTAL_SCENARIOS=4

# ── Scenario 1: Absence of baseline in .performance-baselines/ ───────────────
echo -e "\n${BLUE}[1/${TOTAL_SCENARIOS}] Testing Scenario 1: Absence of baseline in .performance-baselines/${NC}"

# Backup real baseline directory if present temporarily
BASELINE_DIR="$PROJECT_DIR/.performance-baselines"
BACKUP_DIR="$TMP_DIR/performance-baselines-backup"
HAS_REAL_BASELINE=0
if [ -d "$BASELINE_DIR" ]; then
    HAS_REAL_BASELINE=1
    mv "$BASELINE_DIR" "$BACKUP_DIR"
fi

restore_real_baseline() {
    if [ "$HAS_REAL_BASELINE" -eq 1 ] && [ -d "$BACKUP_DIR" ]; then
        rm -rf "$BASELINE_DIR"
        mv "$BACKUP_DIR" "$BASELINE_DIR"
    fi
}
trap 'restore_real_baseline; rm -rf "$TMP_DIR"' EXIT

set +e
SCENARIO1_OUT=$("$PROJECT_DIR/utils/tests-performance-regression.sh" --check 2>&1)
SCENARIO1_EXIT=$?
set -e

if [ "$SCENARIO1_EXIT" -ne 1 ]; then
    echo -e "  ${RED}❌ Scenario 1 failed: expected exit code 1, got ${SCENARIO1_EXIT}${NC}"
    exit 1
fi

if ! echo "$SCENARIO1_OUT" | grep -q "MISSING_BASELINE"; then
    echo -e "  ${RED}❌ Scenario 1 failed: output does not contain MISSING_BASELINE${NC}"
    echo "$SCENARIO1_OUT"
    exit 1
fi
echo -e "  ${GREEN}✓ Scenario 1 passed: exit code 1 and MISSING_BASELINE confirmed.${NC}"

# ── Scenario 2: JSON-only --check (S2.T7); NOT_VERIFIED verdicts via fixture ─
echo -e "\n${BLUE}[2/${TOTAL_SCENARIOS}] Testing Scenario 2: JSON-only --check (S2.T7)${NC}"

# 2a: ASCII .txt contracts are rejected fail-closed (ERROR + exit 2) BEFORE
# any phase runs — the dashboard is not invoked for oracle execution.
set +e
SCENARIO2_OUT=$("$PROJECT_DIR/utils/quality-dashboard.sh" --check "$PROJECT_DIR/docs/quality-contract.txt" 2>&1)
SCENARIO2_EXIT=$?
set -e

if [ "$SCENARIO2_EXIT" -ne 2 ]; then
    echo -e "  ${RED}❌ Scenario 2a failed: expected exit 2 for .txt contract, got ${SCENARIO2_EXIT}${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi

if ! echo "$SCENARIO2_OUT" | grep -q "ERROR"; then
    echo -e "  ${RED}❌ Scenario 2a failed: output does not contain ERROR for .txt contract${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi
echo -e "  ${GREEN}✓ Scenario 2a passed: .txt contract rejected with ERROR + exit 2.${NC}"

# 2b: the verdict strings stay in English and are produced by the binary
# against the committed fixture (no oracle runs — the fixture freezes the
# S2 acceptance scenario: all phases PASS except regression_gate=NOT_VERIFIED,
# so PERFORMANCE: NOT_VERIFIED + CONTRACT VIOLATED with FIDELITY: OK,
# PERF-006 domain separation).
FIXTURE_REPORT="$PROJECT_DIR/tests/fixtures/qa/report_gate_not_verified.jsonl"
NAM_QUALITY_BIN="${NAM_QUALITY_BIN:-$PROJECT_DIR/target/debug/nam_quality}"
if [ ! -x "$NAM_QUALITY_BIN" ]; then
    ( cd "$PROJECT_DIR" && cargo build --quiet --features testing --bin nam_quality )
fi

set +e
SCENARIO2_OUT=$("$NAM_QUALITY_BIN" verify --contract "$PROJECT_DIR/docs/quality-contract.json" --report "$FIXTURE_REPORT" 2>&1)
SCENARIO2_EXIT=$?
set -e

if [ "$SCENARIO2_EXIT" -ne 1 ]; then
    echo -e "  ${RED}❌ Scenario 2b failed: expected exit code 1, got ${SCENARIO2_EXIT}${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi

if ! echo "$SCENARIO2_OUT" | grep -q "FIDELITY: OK"; then
    echo -e "  ${RED}❌ Scenario 2b failed: output does not contain FIDELITY: OK (PERF-006)${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi

if ! echo "$SCENARIO2_OUT" | grep -q "PERFORMANCE: NOT_VERIFIED"; then
    echo -e "  ${RED}❌ Scenario 2b failed: output does not contain PERFORMANCE: NOT_VERIFIED${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi

if ! echo "$SCENARIO2_OUT" | grep -q "CONTRACT VIOLATED"; then
    echo -e "  ${RED}❌ Scenario 2b failed: output does not contain CONTRACT VIOLATED${NC}"
    echo "$SCENARIO2_OUT"
    exit 1
fi
echo -e "  ${GREEN}✓ Scenario 2b passed: FIDELITY: OK, PERFORMANCE: NOT_VERIFIED & CONTRACT VIOLATED confirmed via fixture.${NC}"

# ── Scenario 3: Truncation of previous log at start of execution ─────────────
echo -e "\n${BLUE}[3/${TOTAL_SCENARIOS}] Testing Scenario 3: Log truncation at start of check_regression()${NC}"

STALE_LOG_FILE="$PROJECT_DIR/target/logs/regression-check.log"
mkdir -p "$PROJECT_DIR/target/logs"
DUMMY_MARKER="STALE_DUMMY_BENCHMARK_DATA_$(date +%s%N)"
echo "$DUMMY_MARKER" > "$STALE_LOG_FILE"

set +e
"$PROJECT_DIR/utils/tests-performance-regression.sh" --check >/dev/null 2>&1 || true
set -e

if [ -f "$STALE_LOG_FILE" ] && grep -q "$DUMMY_MARKER" "$STALE_LOG_FILE"; then
    echo -e "  ${RED}❌ Scenario 3 failed: stale dummy marker still found in $STALE_LOG_FILE${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Scenario 3 passed: stale log truncated at invocation start.${NC}"

# Restore real baseline (scenarios 1-3 move it aside).
restore_real_baseline

# ── Scenario 4: Nested ci-baseline dirs are sanitized; replace-copy never nests ─
echo -e "\n${BLUE}[4/${TOTAL_SCENARIOS}] Testing Scenario 4: Nested baseline sanitize + replace-copy via nam_perf_gate${NC}"

NAM_PERF_GATE="${NAM_PERF_GATE:-$PROJECT_DIR/target/debug/nam_perf_gate}"
if [ ! -x "$NAM_PERF_GATE" ]; then
    ( cd "$PROJECT_DIR" && cargo build --quiet --features testing --bin nam_perf_gate )
fi

BASELINE_NAME="ci-baseline"
BASELINE_DIR="$TMP_DIR/fake-baselines"
CRITERION_BASELINE_TARGET="$TMP_DIR/fake-criterion"
mkdir -p "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/$BASELINE_NAME/$BASELINE_NAME"
echo nested-deep > "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/$BASELINE_NAME/$BASELINE_NAME/marker.txt"
echo top-level > "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/marker.txt"
mkdir -p "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/$BASELINE_NAME"
echo stale-dest > "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/marker.txt"
echo already-nested > "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/$BASELINE_NAME/marker.txt"

# restore-baseline sanitizes the store and replace-copies into the criterion root
"$NAM_PERF_GATE" restore-baseline \
    --baseline-dir "$BASELINE_DIR" \
    --criterion-root "$CRITERION_BASELINE_TARGET" \
    --baseline "$BASELINE_NAME" >/dev/null

if find "$BASELINE_DIR" -mindepth 3 -type d -name "$BASELINE_NAME" 2>/dev/null | grep -q .; then
    echo -e "  ${RED}❌ Scenario 4 failed: nested baseline dirs remain after sanitize${NC}"
    find "$BASELINE_DIR" -type d -name "$BASELINE_NAME"
    exit 1
fi
if [ ! -f "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/marker.txt" ]; then
    echo -e "  ${RED}❌ Scenario 4 failed: top-level baseline marker lost during sanitize${NC}"
    exit 1
fi
if ! grep -qx 'top-level' "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/marker.txt"; then
    echo -e "  ${RED}❌ Scenario 4 failed: top-level marker was overwritten by nested content${NC}"
    cat "$BASELINE_DIR/RT_Dummy/$BASELINE_NAME/marker.txt"
    exit 1
fi

if find "$CRITERION_BASELINE_TARGET" -mindepth 3 -type d -name "$BASELINE_NAME" 2>/dev/null | grep -q .; then
    echo -e "  ${RED}❌ Scenario 4 failed: restore re-introduced nested baseline dirs${NC}"
    find "$CRITERION_BASELINE_TARGET" -type d -name "$BASELINE_NAME"
    exit 1
fi
if [ ! -f "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/marker.txt" ]; then
    echo -e "  ${RED}❌ Scenario 4 failed: restore did not place top-level baseline${NC}"
    exit 1
fi
if ! grep -qx 'top-level' "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/marker.txt"; then
    echo -e "  ${RED}❌ Scenario 4 failed: restore did not replace-copy top-level content${NC}"
    cat "$CRITERION_BASELINE_TARGET/RT_Dummy/$BASELINE_NAME/marker.txt"
    exit 1
fi
# Second restore must still not nest
"$NAM_PERF_GATE" restore-baseline \
    --baseline-dir "$BASELINE_DIR" \
    --criterion-root "$CRITERION_BASELINE_TARGET" \
    --baseline "$BASELINE_NAME" >/dev/null
nested_count=$(find "$CRITERION_BASELINE_TARGET" -mindepth 3 -type d -name "$BASELINE_NAME" 2>/dev/null | wc -l)
if [ "$nested_count" -ne 0 ]; then
    echo -e "  ${RED}❌ Scenario 4 failed: second restore nested baselines (count=${nested_count})${NC}"
    exit 1
fi
top_count=$(find "$CRITERION_BASELINE_TARGET" -mindepth 2 -maxdepth 2 -type d -name "$BASELINE_NAME" 2>/dev/null | wc -l)
if [ "$top_count" -ne 1 ]; then
    echo -e "  ${RED}❌ Scenario 4 failed: expected exactly 1 top-level baseline, got ${top_count}${NC}"
    find "$CRITERION_BASELINE_TARGET" -mindepth 2 -maxdepth 2 -type d -name "$BASELINE_NAME"
    exit 1
fi
echo -e "  ${GREEN}✓ Scenario 4 passed: sanitize + replace-copy keep a single flat baseline layer.${NC}"

END_TIME=$(date +%s%N)
ELAPSED_MS=$(awk -v start="$START_TIME" -v end="$END_TIME" 'BEGIN { printf "%.0f", (end - start) / 1000000 }')

echo -e "\n${GREEN}${BOLD}✓ All regression guard scenarios passed cleanly in ${ELAPSED_MS}ms.${NC}"
