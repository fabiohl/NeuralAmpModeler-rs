#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Performance Regression Gate — thin wrapper over nam_perf_gate (S3.T4)
# =============================================================================
#
# Canonical home of benchmark-based performance defense for NeuralAmpModeler-rs:
# runs the `regression_gate` Criterion suite (sample_size=100, measurement_time=5s),
# pinned to a designated CPU core, and compares current timings against a persisted
# statistical baseline. A regressing commit exits non-zero — protecting real-time
# DSP budgets with strict fail-closed safety.
#
# All logic (fingerprint, coverage, persist/restore, receipt) is delegated to
# `nam_perf_gate` (S3.T3). This script only orchestrates: taskset, cargo bench,
# and calls to the bin.
#
# Modes
# -----
#   --check (default)    Compare current build against saved baseline (read-only).
#                        Fails with MISSING_BASELINE if no baseline exists.
#                        Fails with BASELINE_COVERAGE_GAP if any bench lacks baseline.
#   --bootstrap-baseline Create a new baseline and environment fingerprint.
#                        Must be executed by a human operator.
#
# Environment variables
# ----------------------
#   NAM_BENCH_CORE       CPU core to pin via taskset (default: middle core).
#   NAM_BASELINE_NAME    Criterion baseline name (default: ci-baseline).
#
# Usage
# ------
#   utils/tests-performance-regression.sh [--check|--bootstrap-baseline]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

trap 'echo -e "\n${RED}${BOLD}❌ Unexpected error: Command \"$BASH_COMMAND\" failed at line $LINENO with status $?. Aborting.${NC}"; exit 1' ERR

# ── Configuration ────────────────────────────────────────────────────────────

NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
DEFAULT_CORE=$(( ${NUM_CORES:-1} / 2 ))
BENCH_CORE="${NAM_BENCH_CORE:-$DEFAULT_CORE}"
BASELINE_NAME="${NAM_BASELINE_NAME:-ci-baseline}"
MODE="${1:---check}"

BASELINE_DIR=".performance-baselines"
FINGERPRINT_FILE="${BASELINE_DIR}/baseline-fingerprint.json"
CRITERION_BASELINE_TARGET="target/criterion"

TASKSET=()
if command -v taskset >/dev/null 2>&1; then
    TASKSET=(taskset -c "${BENCH_CORE}")
else
    warn "taskset not found — running without core pinning."
fi

NAM_PERF_GATE="${NAM_PERF_GATE:-target/debug/nam_perf_gate}"
if [ ! -x "$NAM_PERF_GATE" ]; then
    cargo build --quiet --features testing --bin nam_perf_gate
fi

echo -e "${BLUE}${BOLD}  Performance Regression Gate${NC}" >&2
echo -e "  Core: ${YELLOW}${BENCH_CORE}${NC}  Baseline: ${YELLOW}${BASELINE_NAME}${NC}" >&2
echo -e "${BLUE}${BOLD}  Estimated time: ± 2.0 minutes${NC}" >&2

REGRESSION_RECEIPT_DIR="$PROJECT_DIR/target/logs"
REGRESSION_RECEIPT="${REGRESSION_RECEIPT_DIR}/regression_phase_receipt.jsonl"
mkdir -p "$REGRESSION_RECEIPT_DIR"
: > "$REGRESSION_RECEIPT"
DASHBOARD_PHASE_RECEIPT="$REGRESSION_RECEIPT"

NAM_RUN_ID="${NAM_RUN_ID:-$(date +%s%N-$$)}"
export NAM_RUN_ID

# ── Bootstrap baseline (human-only operation) ──────────────────────────────

bootstrap_baseline() {
    echo -e "\n${GREEN}${BOLD}[BOOTSTRAP] Creating new performance baseline...${NC}" >&2
    echo -e "  ${YELLOW}⚠ This operation must be performed by a human operator.${NC}"
    echo -e "  ${YELLOW}⚠ Automated/CI/agent-driven execution is prohibited.${NC}\n"

    "${TASKSET[@]}" cargo bench --bench regression_gate --features testing -- --save-baseline "$BASELINE_NAME"

    "$NAM_PERF_GATE" persist-baseline \
        --baseline-dir "$BASELINE_DIR" \
        --criterion-root "$CRITERION_BASELINE_TARGET" \
        --baseline "$BASELINE_NAME" >&2

    "$NAM_PERF_GATE" probe \
        --out "$FINGERPRINT_FILE" \
        --bench-core "$BENCH_CORE" \
        --baseline-dir "$BASELINE_DIR" > /dev/null

    "$NAM_PERF_GATE" receipt append \
        --phase-id "regression_baseline_created" \
        --status "PASS" \
        --exit-code 0 \
        --observed-records 1 \
        --expected-records 1 \
        --reason "" \
        --run-id "$NAM_RUN_ID" \
        --out "$REGRESSION_RECEIPT" >&2

    echo -e "  ${GREEN}✓${NC} Baseline '${BASELINE_NAME}' created and persisted." >&2
    echo -e "  Fingerprint saved to ${YELLOW}${FINGERPRINT_FILE}${NC}" >&2
    echo -e "  Baseline data persisted under ${YELLOW}${BASELINE_DIR}${NC}" >&2
}

# ── Check mode (strictly read-only) ────────────────────────────────────────

check_regression() {
    echo -e "\n${BLUE}${BOLD}[CHECK] Comparing against CI baseline...${NC}" >&2

    mkdir -p "$PROJECT_DIR/target/logs"
    LOG_FILE="$PROJECT_DIR/target/logs/regression-check.log"
    : > "$LOG_FILE"

    # Fingerprint comparison
    set +e
    "$NAM_PERF_GATE" compare \
        --baseline "$FINGERPRINT_FILE" \
        --bench-core "$BENCH_CORE" \
        --baseline-dir "$BASELINE_DIR" > "$LOG_FILE.fingerprint" 2>&1
    FINGERPRINT_STATUS=$?
    set -e

    if [ $FINGERPRINT_STATUS -ne 0 ]; then
        cat "$LOG_FILE.fingerprint"
        if grep -q "MISSING_BASELINE" "$LOG_FILE.fingerprint"; then
            REASON="MISSING_BASELINE"
        elif grep -q "INCOMPARABLE_ENVIRONMENT" "$LOG_FILE.fingerprint"; then
            REASON="INCOMPARABLE_ENVIRONMENT"
        else
            REASON="FINGERPRINT_ERROR"
        fi
        "$NAM_PERF_GATE" receipt append \
            --phase-id "regression_check" \
            --status "FAIL" \
            --exit-code 1 \
            --observed-records 0 \
            --expected-records 1 \
            --reason "$REASON" \
            --run-id "$NAM_RUN_ID" \
            --out "$REGRESSION_RECEIPT" >&2
        rm -f "$LOG_FILE.fingerprint"
        exit 1
    fi
    rm -f "$LOG_FILE.fingerprint"

    # Restore baseline
    "$NAM_PERF_GATE" restore-baseline \
        --baseline-dir "$BASELINE_DIR" \
        --criterion-root "$CRITERION_BASELINE_TARGET" \
        --baseline "$BASELINE_NAME" >&2

    # Run benchmarks
    mkdir -p target/logs
    LOG_FILE="target/logs/regression-check.log"

    set +e
    "${TASKSET[@]}" cargo bench --bench regression_gate --features testing \
        -- --baseline "$BASELINE_NAME" 2>&1 | tee "$LOG_FILE"
    BENCH_STATUS=$?
    set -e

    if grep -qiE 'has regressed' "$LOG_FILE" 2>/dev/null; then
        echo -e "\n${RED}${BOLD}❌ PERFORMANCE REGRESSION DETECTED${NC}"
        echo -e "  Review $LOG_FILE for details." >&2
        echo -e "  If the regression is intentional, re-save the baseline with:" >&2
        echo -e "    ${YELLOW}utils/tests-performance-regression.sh --bootstrap-baseline${NC}" >&2
        "$NAM_PERF_GATE" receipt append \
            --phase-id "regression_check" \
            --status "FAIL" \
            --exit-code 1 \
            --observed-records 0 \
            --expected-records 1 \
            --reason "REGRESSION_DETECTED" \
            --run-id "$NAM_RUN_ID" \
            --out "$REGRESSION_RECEIPT" >&2
        exit 1
    fi

    if [ $BENCH_STATUS -ne 0 ]; then
        echo -e "\n${RED}${BOLD}❌ Benchmark run failed (status=${BENCH_STATUS})${NC}"
        "$NAM_PERF_GATE" receipt append \
            --phase-id "regression_check" \
            --status "FAIL" \
            --exit-code "$BENCH_STATUS" \
            --observed-records 0 \
            --expected-records 1 \
            --reason "Benchmark run failed" \
            --run-id "$NAM_RUN_ID" \
            --out "$REGRESSION_RECEIPT" >&2
        exit 1
    fi

    # Coverage cross-check
    set +e
    "$NAM_PERF_GATE" coverage \
        --log "$LOG_FILE" \
        --root "$CRITERION_BASELINE_TARGET" \
        --baseline "$BASELINE_NAME" > "$LOG_FILE.coverage" 2>&1
    COVERAGE_STATUS=$?
    set -e

    if [ $COVERAGE_STATUS -ne 0 ]; then
        cat "$LOG_FILE.coverage"
        "$NAM_PERF_GATE" receipt append \
            --phase-id "regression_check" \
            --status "FAIL" \
            --exit-code 1 \
            --observed-records 0 \
            --expected-records 1 \
            --reason "BASELINE_COVERAGE_GAP" \
            --run-id "$NAM_RUN_ID" \
            --out "$REGRESSION_RECEIPT" >&2
        rm -f "$LOG_FILE.coverage"
        exit 1
    fi
    rm -f "$LOG_FILE.coverage"

    # Audit trail: record verified benchmark set
    local executed_count
    executed_count=$(grep -c "^Benchmarking " "$LOG_FILE" 2>/dev/null || echo 0)
    "$NAM_PERF_GATE" receipt append \
        --phase-id "regression_baseline_coverage" \
        --status "PASS" \
        --exit-code 0 \
        --observed-records "$executed_count" \
        --expected-records "$executed_count" \
        --reason "" \
        --run-id "$NAM_RUN_ID" \
        --out "$REGRESSION_RECEIPT" >&2

    echo -e "  ${GREEN}✓${NC} Coverage cross-check: all ${executed_count} executed benchmark(s) have baseline series." >&2
    echo -e "  ${GREEN}✓${NC} No performance regression detected." >&2
    "$NAM_PERF_GATE" receipt append \
        --phase-id "regression_check" \
        --status "PASS" \
        --exit-code 0 \
        --observed-records 1 \
        --expected-records 1 \
        --reason "" \
        --run-id "$NAM_RUN_ID" \
        --out "$REGRESSION_RECEIPT" >&2
}

# ── Main entry point ───────────────────────────────────────────────────────

case "$MODE" in
    --bootstrap-baseline)
        bootstrap_baseline
        ;;
    --check)
        check_regression
        ;;
    *)
        echo -e "${RED}Unknown mode: $MODE${NC}"
        echo "Usage: $0 [--check|--bootstrap-baseline]"
        exit 1
        ;;
esac
