#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Performance Regression Gate — statistical wall against DSP hot-path decay
# =============================================================================
#
# Canonical home of benchmark-based performance defense for NeuralAmpModeler-rs:
# runs the `regression_gate` Criterion suite (sample_size=100, measurement_time=5s),
# optionally pinned to a specific CPU core, and compares current timings against
# a persisted statistical baseline. A regressing commit exits non-zero — the audio
# engine has a strict real-time deadline (1.33 ms / 64 samples at 48 kHz), and this
# gate prevents performance regressions from silently consuming that budget.
#
# Full rationale, daily workflow, and troubleshooting live in docs/benchmarks.md
# ("Regression Gate" section).
#
# Modes
# -----
#   --check (default)    Compare the current build against the saved baseline.
#                        Strictly read-only: fails with MISSING_BASELINE if
#                        no baseline exists. Never auto-creates a baseline.
#   --bootstrap-baseline Create a new baseline and environment fingerprint.
#                        Must be executed by a human operator. Prohibited in
#                        automated/CI/agent-driven workflows.
#
# Environment variables
# ----------------------
#   NAM_BENCH_CORE       CPU core to pin via taskset (default: middle core).
#   NAM_BASELINE_NAME    Criterion baseline name (default: ci-baseline).
#
# Usage
# ------
#   utils/tests-performance-regression.sh [--check|--bootstrap-baseline]
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

trap 'echo -e "\n${RED}${BOLD}❌ Unexpected error: Command \"$BASH_COMMAND\" failed at line $LINENO with status $?. Aborting.${NC}"; exit 1' ERR

NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
DEFAULT_CORE=$(( ${NUM_CORES:-1} / 2 ))
BENCH_CORE="${NAM_BENCH_CORE:-$DEFAULT_CORE}"
BASELINE_NAME="${NAM_BASELINE_NAME:-ci-baseline}"
MODE="${1:---check}"

BASELINE_DIR="target/performance-baselines"
FINGERPRINT_FILE="${BASELINE_DIR}/baseline-fingerprint.json"
CRITERION_BASELINE_TARGET="target/criterion"

TASKSET=()
if command -v taskset >/dev/null 2>&1; then
    TASKSET=(taskset -c "${BENCH_CORE}")
else
    echo -e "  ${YELLOW}⚠ taskset not found — running without core pinning.${NC}"
fi

echo -e "${BLUE}${BOLD}  Performance Regression Gate${NC}"
echo -e "  Core: ${YELLOW}${BENCH_CORE}${NC}  Baseline: ${YELLOW}${BASELINE_NAME}${NC}"
echo -e "${BLUE}${BOLD}  Estimated time: ± 2.0 minutes${NC}"

# ── JSONL receipt infrastructure (T-E4.6-2) ──────────────────────────────────
REGRESSION_RECEIPT_DIR="$PROJECT_DIR/target/logs"
REGRESSION_RECEIPT="${REGRESSION_RECEIPT_DIR}/regression_phase_receipt.jsonl"
mkdir -p "$REGRESSION_RECEIPT_DIR"
: > "$REGRESSION_RECEIPT"
DASHBOARD_PHASE_RECEIPT="$REGRESSION_RECEIPT"

REGR_RUSTC_VER=$(rustc --version 2>/dev/null || echo 'unknown')
REGR_TARGET_TRIPLE=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
REGR_CPU_MODEL=$(detect_cpu_model)
REGR_CPU_MICROARCH=$(detect_cpu_microarch)
REGR_FREQ_GOV=$(detect_freq_governor)
REGR_GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

printf '{"kind":"build_metadata","pipeline":"performance-regression","cargo_profile":"release","target_triple":"%s","rustflags":"%s","rustc_version":"%s","cpu_model":"%s","cpu_microarch":"%s","frequency_governor":"%s","git_commit":"%s","bench_core":"%s"}\n' \
    "$REGR_TARGET_TRIPLE" "${RUSTFLAGS:-}" "$REGR_RUSTC_VER" "$REGR_CPU_MODEL" "$REGR_CPU_MICROARCH" \
    "$REGR_FREQ_GOV" "$REGR_GIT_COMMIT" "$BENCH_CORE" >> "$REGRESSION_RECEIPT"

# ── Environment detection helpers ───────────────────────────────────────────

detect_cpu_model() {
    grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown"
}

detect_cpu_microarch() {
    local flags
    flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
    if echo "$flags" | grep -q 'avx512f'; then
        echo "AVX-512"
    elif echo "$flags" | grep -q 'avx2'; then
        echo "AVX2 (x86-64-v3)"
    else
        echo "x86-64 (base)"
    fi
}

detect_physical_cores() {
    grep -c '^cpu cores' /proc/cpuinfo 2>/dev/null | head -1 || echo "$NUM_CORES"
}

detect_freq_governor() {
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown"
}

detect_git_commit() {
    git rev-parse HEAD 2>/dev/null || echo "unknown"
}

# ── Fingerprint generation ─────────────────────────────────────────────────

generate_fingerprint() {
    local cpu_model cpu_microarch phys_cores rustc_ver target_triple \
          rustflags build_profile freq_gov git_commit
    cpu_model=$(detect_cpu_model)
    cpu_microarch=$(detect_cpu_microarch)
    phys_cores=$(detect_physical_cores)
    rustc_ver=$(rustc --version 2>/dev/null || echo 'unknown')
    target_triple=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
    rustflags="${RUSTFLAGS:-}"
    build_profile="release"
    freq_gov=$(detect_freq_governor)
    git_commit=$(detect_git_commit)

    mkdir -p "$BASELINE_DIR"

    cat > "$FINGERPRINT_FILE" <<FINGERPRINT
{
  "cpu_model": "$cpu_model",
  "cpu_microarchitecture": "$cpu_microarch",
  "physical_cores": $phys_cores,
  "rustc_version": "$rustc_ver",
  "target_triple": "$target_triple",
  "rustflags": "$rustflags",
  "build_profile": "$build_profile",
  "frequency_governor": "$freq_gov",
  "git_commit": "$git_commit"
}
FINGERPRINT
}

# ── Fingerprint comparison ─────────────────────────────────────────────────
# Returns 0 if compatible, 1 if INCOMPARABLE.

compare_fingerprint() {
    if [ ! -f "$FINGERPRINT_FILE" ]; then
        echo -e "  ${RED}${BOLD}MISSING_FINGERPRINT${NC} No fingerprint found at $FINGERPRINT_FILE"
        echo -e "  Re-bootstrap the baseline with ${YELLOW}--bootstrap-baseline${NC}"
        return 1
    fi

    local cpu_microarch_now rustc_now target_now rustflags_now freq_gov_now
    cpu_microarch_now=$(detect_cpu_microarch)
    rustc_now=$(rustc --version 2>/dev/null || echo 'unknown')
    target_now=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
    rustflags_now="${RUSTFLAGS:-}"
    freq_gov_now=$(detect_freq_governor)

    local stored_cpu_microarch stored_rustc stored_target stored_rustflags
    stored_cpu_microarch=$(sed -n 's/.*"cpu_microarchitecture": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_rustc=$(sed -n 's/.*"rustc_version": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_target=$(sed -n 's/.*"target_triple": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_rustflags=$(sed -n 's/.*"rustflags": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")

    local incomparable=0

    if [ "$cpu_microarch_now" != "$stored_cpu_microarch" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} CPU microarchitecture mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_cpu_microarch${NC}"
        echo -e "    Current:  ${YELLOW}$cpu_microarch_now${NC}"
        incomparable=1
    fi

    if [ "$rustc_now" != "$stored_rustc" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} rustc version mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_rustc${NC}"
        echo -e "    Current:  ${YELLOW}$rustc_now${NC}"
        incomparable=1
    fi

    if [ "$target_now" != "$stored_target" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} target triple mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_target${NC}"
        echo -e "    Current:  ${YELLOW}$target_now${NC}"
        incomparable=1
    fi

    if [ "$rustflags_now" != "$stored_rustflags" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} RUSTFLAGS mismatch"
        echo -e "    Baseline: ${YELLOW}${stored_rustflags:-<none>}${NC}"
        echo -e "    Current:  ${YELLOW}${rustflags_now:-<none>}${NC}"
        incomparable=1
    fi

    if [ "$freq_gov_now" != "performance" ]; then
        echo -e "  ${YELLOW}WARNING${NC} CPU frequency governor is '${freq_gov_now}' (recommended: 'performance')"
    fi

    return $incomparable
}

# ── Baseline persistence ───────────────────────────────────────────────────
# Copies Criterion baselines from target/criterion to the persistent
# target/performance-baselines/ directory so they survive cargo clean.

persist_baseline() {
    mkdir -p "$BASELINE_DIR"
    if [ -d "$CRITERION_BASELINE_TARGET" ]; then
        find "$CRITERION_BASELINE_TARGET" -type d -name "$BASELINE_NAME" 2>/dev/null | while read -r baseline_path; do
            local rel_path="${baseline_path#$CRITERION_BASELINE_TARGET/}"
            local dest="$BASELINE_DIR/$rel_path"
            mkdir -p "$(dirname "$dest")"
            cp -a "$baseline_path" "$dest"
        done
        echo -e "  ${GREEN}✓${NC} Baseline persisted to ${BASELINE_DIR}"
    fi
}

# Restores baselines from the persistent directory back into target/criterion
# so Criterion can find them for --baseline comparisons.
restore_baseline() {
    if [ -d "$BASELINE_DIR" ]; then
        find "$BASELINE_DIR" -type d -name "$BASELINE_NAME" 2>/dev/null | while read -r persisted_path; do
            local rel_path="${persisted_path#$BASELINE_DIR/}"
            local dest="$CRITERION_BASELINE_TARGET/$rel_path"
            mkdir -p "$(dirname "$dest")"
            cp -a "$persisted_path" "$dest"
        done
    fi
}

# ── Bootstrap baseline (human-only operation) ──────────────────────────────

bootstrap_baseline() {
    echo -e "\n${GREEN}${BOLD}[BOOTSTRAP] Creating new performance baseline...${NC}"
    echo -e "  ${YELLOW}⚠ This operation must be performed by a human operator.${NC}"
    echo -e "  ${YELLOW}⚠ Automated/CI/agent-driven execution is prohibited.${NC}"
    echo ""

    "${TASKSET[@]}" cargo bench --bench regression_gate --features testing -- --save-baseline "$BASELINE_NAME"

    persist_baseline
    generate_fingerprint

    echo -e "${GREEN}✓ Baseline '${BASELINE_NAME}' created and persisted.${NC}"
    echo -e "  Fingerprint saved to ${YELLOW}${FINGERPRINT_FILE}${NC}"
    echo -e "  Baseline data persisted under ${YELLOW}${BASELINE_DIR}${NC}"
    dashboard_phase_receipt "regression_baseline_created" "PASS" 0 1 1 ""
}

# ── Check mode (strictly read-only) ────────────────────────────────────────

check_regression() {
    echo -e "\n${BLUE}${BOLD}[CHECK] Comparing against CI baseline...${NC}"

    if [ ! -f "$FINGERPRINT_FILE" ]; then
        echo -e "${RED}${BOLD}MISSING_BASELINE${NC} No baseline found under ${BASELINE_DIR}/"
        echo ""
        echo -e "  The performance baseline has not been bootstrapped."
        echo -e "  A human operator must run:"
        echo -e ""
        echo -e "    ${YELLOW}utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
        echo -e ""
        echo -e "  Automated/CI/agent-driven bootstrap is prohibited to prevent"
        echo -e "  a regressing branch from becoming the new reference."
        dashboard_phase_receipt "regression_check" "FAIL" 1 0 0 "MISSING_BASELINE"
        exit 1
    fi

    if ! compare_fingerprint; then
        echo ""
        echo -e "  ${RED}Performance comparison aborted: environment incompatible with baseline.${NC}"
        echo -e "  A human operator must re-bootstrap the baseline with:"
        echo -e ""
        echo -e "    ${YELLOW}utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
        echo ""
        echo -e "  accompanied by a formal justification for the environmental change."
        dashboard_phase_receipt "regression_check" "FAIL" 1 0 0 "INCOMPARABLE_ENVIRONMENT"
        exit 1
    fi

    restore_baseline

    mkdir -p target/logs
    LOG_FILE="target/logs/regression-check.log"

    set +e
    "${TASKSET[@]}" cargo bench --bench regression_gate --features testing \
        -- --baseline "$BASELINE_NAME" 2>&1 | tee "$LOG_FILE"
    BENCH_STATUS=$?
    set -e

    if grep -qiE 'has regressed' "$LOG_FILE" 2>/dev/null; then
        echo -e "\n${RED}${BOLD}❌ PERFORMANCE REGRESSION DETECTED${NC}"
        echo -e "  Review $LOG_FILE for details."
        echo -e "  If the regression is intentional, re-save the baseline with:"
        echo -e "    ${YELLOW}utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
        dashboard_phase_receipt "regression_check" "FAIL" 1 0 0 "REGRESSION_DETECTED"
        exit 1
    fi

    if [ $BENCH_STATUS -ne 0 ]; then
        echo -e "\n${RED}${BOLD}❌ Benchmark run failed (status=${BENCH_STATUS})${NC}"
        dashboard_phase_receipt "regression_check" "FAIL" "$BENCH_STATUS" 0 0 "Benchmark run failed"
        exit 1
    fi

    echo -e "${GREEN}✓ No performance regression detected.${NC}"
    dashboard_phase_receipt "regression_check" "PASS" 0 1 1 ""
}

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
