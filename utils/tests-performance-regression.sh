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
#                        Fail-closed coverage: any executed benchmark without
#                        a saved baseline series fails with
#                        BASELINE_COVERAGE_GAP (F-24b); the verified benchmark
#                        set is recorded in the receipt (F-24c).
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

# ── Environment detection helpers ───────────────────────────────────────────

detect_cpu_model() {
    grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown"
}

detect_cpu_microarch() {
    local flags
    flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null || true)
    if echo "$flags" | grep -q -w 'avx512f'; then
        echo "AVX-512"
    # Linux /proc/cpuinfo exposes LZCNT as "lzcnt" and/or "abm" (AMD ABM).
    elif echo "$flags" | grep -q -w 'avx' && \
         echo "$flags" | grep -q -w 'avx2' && \
         echo "$flags" | grep -q -w 'bmi1' && \
         echo "$flags" | grep -q -w 'bmi2' && \
         echo "$flags" | grep -q -w 'f16c' && \
         echo "$flags" | grep -q -w 'fma' && \
         (echo "$flags" | grep -q -w 'lzcnt' || echo "$flags" | grep -q -w 'abm') && \
         echo "$flags" | grep -q -w 'movbe'; then
        echo "x86-64-v3 (AVX2/FMA/F16C/BMI)"
    elif echo "$flags" | grep -q -w 'avx2'; then
        echo "AVX2 (incompleto / unsupported)"
    else
        echo "x86-64 (base)"
    fi
}

detect_physical_cores() {
    grep -m1 '^cpu cores' /proc/cpuinfo 2>/dev/null | awk '{print $4}' || echo "$NUM_CORES"
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
          rustflags build_profile freq_gov git_commit bench_core
    cpu_model=$(detect_cpu_model)
    cpu_microarch=$(detect_cpu_microarch)
    phys_cores=$(detect_physical_cores)
    rustc_ver=$(rustc --version 2>/dev/null || echo 'unknown')
    target_triple=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
    rustflags="${RUSTFLAGS:-}"
    build_profile="release"
    freq_gov=$(detect_freq_governor)
    git_commit=$(detect_git_commit)
    bench_core="${BENCH_CORE}"

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
  "git_commit": "$git_commit",
  "bench_core": "$bench_core"
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

    local cpu_model_now cpu_microarch_now rustc_now target_now rustflags_now \
          build_profile_now freq_gov_now bench_core_now
    cpu_model_now=$(detect_cpu_model)
    cpu_microarch_now=$(detect_cpu_microarch)
    rustc_now=$(rustc --version 2>/dev/null || echo 'unknown')
    target_now=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
    rustflags_now="${RUSTFLAGS:-}"
    build_profile_now="release"
    freq_gov_now=$(detect_freq_governor)
    bench_core_now="${BENCH_CORE}"
    local stored_cpu_model stored_cpu_microarch stored_build_profile \
          stored_rustc stored_target stored_rustflags stored_freq_gov stored_bench_core
    stored_cpu_model=$(sed -n 's/.*"cpu_model": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_cpu_microarch=$(sed -n 's/.*"cpu_microarchitecture": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_build_profile=$(sed -n 's/.*"build_profile": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_rustc=$(sed -n 's/.*"rustc_version": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_target=$(sed -n 's/.*"target_triple": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_rustflags=$(sed -n 's/.*"rustflags": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_freq_gov=$(sed -n 's/.*"frequency_governor": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")
    stored_bench_core=$(sed -n 's/.*"bench_core": *"\([^"]*\)".*/\1/p' "$FINGERPRINT_FILE")

    local incomparable=0

    if [ "$cpu_model_now" != "$stored_cpu_model" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} CPU model mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_cpu_model${NC}"
        echo -e "    Current:  ${YELLOW}$cpu_model_now${NC}"
        incomparable=1
    fi

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

    if [ "$build_profile_now" != "$stored_build_profile" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} build profile mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_build_profile${NC}"
        echo -e "    Current:  ${YELLOW}$build_profile_now${NC}"
        incomparable=1
    fi

    if [ "$freq_gov_now" != "performance" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} CPU frequency governor is '${freq_gov_now}' (baseline requires 'performance')"
        incomparable=1
    elif [ -n "$stored_freq_gov" ] && [ "$freq_gov_now" != "$stored_freq_gov" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} frequency governor mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_freq_gov${NC}"
        echo -e "    Current:  ${YELLOW}$freq_gov_now${NC}"
        incomparable=1
    fi

    if [ -n "$stored_bench_core" ] && [ "$bench_core_now" != "$stored_bench_core" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} bench_core pinning mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_bench_core${NC}"
        echo -e "    Current:  ${YELLOW}$bench_core_now${NC}"
        incomparable=1
    fi

    # F-24a: physical core count was recorded but never compared. Machines with
    # the same CPU model but different core counts (VMs / cgroups / host
    # reconfiguration) change DEFAULT_CORE — only indirectly caught via
    # bench_core when NAM_BENCH_CORE is unset. Compare it explicitly.
    local phys_cores_now stored_phys_cores
    phys_cores_now=$(detect_physical_cores)
    stored_phys_cores=$(sed -n 's/.*"physical_cores": *\([0-9][0-9]*\).*/\1/p' "$FINGERPRINT_FILE")
    if [ -n "$stored_phys_cores" ] && [ -n "$phys_cores_now" ] && [ "$phys_cores_now" != "$stored_phys_cores" ]; then
        echo -e "  ${RED}${BOLD}INCOMPARABLE_ENVIRONMENT${NC} physical core count mismatch"
        echo -e "    Baseline: ${YELLOW}$stored_phys_cores${NC}"
        echo -e "    Current:  ${YELLOW}$phys_cores_now${NC}"
        incomparable=1
    fi

    return $incomparable
}

# ── Baseline persistence ───────────────────────────────────────────────────
# Copies Criterion baselines from target/criterion to the persistent
# .performance-baselines/ directory so they survive cargo clean.
#
# IMPORTANT: only top-level baseline dirs are considered:
#   target/criterion/<bench>/$BASELINE_NAME
# Nested paths like .../ci-baseline/ci-baseline are corruption from older
# `cp -a src dest` when dest already existed (cp copies *into* dest). They
# must never be re-persisted or re-restored.

# List only depth-2 baseline directories: <root>/<bench>/$BASELINE_NAME
list_top_level_baselines() {
    local root="$1"
    if [ ! -d "$root" ]; then
        return 0
    fi
    find "$root" -mindepth 2 -maxdepth 2 -type d -name "$BASELINE_NAME" 2>/dev/null | sort
}

# ── Baseline coverage cross-check (F-24b / EP-05) ───────────────────────────
# Criterion silently skips baseline comparison for benchmarks without a saved
# baseline series ("no baseline found"). A new or renamed benchmark would pass
# the gate unverified. These helpers cross the benchmarks actually executed in
# a run against the restored baseline series so the gate can fail closed.

# Prints the benchmark IDs executed in a Criterion log (deduplicated, sorted).
# Criterion prints "Benchmarking <id>: ..." lines for every executed bench.
executed_bench_ids() {
    local log_file="$1"
    grep -oE '^Benchmarking [A-Za-z0-9_.]+' "$log_file" 2>/dev/null \
        | awk '{print $2}' | sort -u || true
}

# Prints the executed benchmark IDs that have NO baseline series under
# <criterion_root>/<id>/<baseline_name>. Empty output = full coverage.
# Exit codes: 0 = parsed fine (with or without gaps); 1 = parse failure
# (no executed benchmark found — the coverage check would be blind).
missing_baseline_coverage() {
    local log_file="$1" criterion_root="$2" baseline_name="${3:-ci-baseline}"
    local ids
    ids=$(executed_bench_ids "$log_file")
    if [ -z "$ids" ]; then
        return 1
    fi
    local id missing=""
    for id in $ids; do
        if [ ! -d "$criterion_root/$id/$baseline_name" ]; then
            missing="${missing}${missing:+ }$id"
        fi
    done
    echo "$missing"
    return 0
}

# Drop nested baseline dirs (depth >= 3) left by historical cp nesting.
sanitize_nested_baselines() {
    local root="$1"
    if [ ! -d "$root" ]; then
        return 0
    fi
    # Delete deepest first so parents can be removed cleanly.
    local nested
    nested=$(find "$root" -mindepth 3 -type d -name "$BASELINE_NAME" 2>/dev/null | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2- || true)
    if [ -z "$nested" ]; then
        return 0
    fi
    local n=0
    while IFS= read -r path; do
        [ -z "$path" ] && continue
        rm -rf "$path"
        n=$((n + 1))
    done <<< "$nested"
    if [ "$n" -gt 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} Removed ${n} nested '${BASELINE_NAME}/' dir(s) under ${root}"
    fi
}

# Replace-copy one directory tree: never nest into an existing dest.
replace_copy_dir() {
    local src="$1"
    local dest="$2"
    rm -rf "$dest"
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
    # Defensive: strip any nested baseline that may have been inside src.
    find "$dest" -mindepth 1 -type d -name "$BASELINE_NAME" -exec rm -rf {} + 2>/dev/null || true
}

persist_baseline() {
    mkdir -p "$BASELINE_DIR"
    if [ -d "$CRITERION_BASELINE_TARGET" ]; then
        sanitize_nested_baselines "$CRITERION_BASELINE_TARGET"
        local count=0
        while IFS= read -r baseline_path; do
            [ -z "$baseline_path" ] && continue
            local rel_path="${baseline_path#$CRITERION_BASELINE_TARGET/}"
            local dest="$BASELINE_DIR/$rel_path"
            replace_copy_dir "$baseline_path" "$dest"
            count=$((count + 1))
        done < <(list_top_level_baselines "$CRITERION_BASELINE_TARGET")
        sanitize_nested_baselines "$BASELINE_DIR"
        echo -e "  ${GREEN}✓${NC} Baseline persisted to ${BASELINE_DIR} (${count} series)"
    fi
}

# Restores baselines from the persistent directory back into target/criterion
# so Criterion can find them for --baseline comparisons.
restore_baseline() {
    if [ ! -d "$BASELINE_DIR" ]; then
        return 0
    fi
    sanitize_nested_baselines "$BASELINE_DIR"
    # Wipe prior Criterion state so leftover new/change/nested dirs cannot
    # pollute comparison or the next persist cycle.
    rm -rf "$CRITERION_BASELINE_TARGET"
    mkdir -p "$CRITERION_BASELINE_TARGET"
    local count=0
    while IFS= read -r persisted_path; do
        [ -z "$persisted_path" ] && continue
        local rel_path="${persisted_path#$BASELINE_DIR/}"
        local dest="$CRITERION_BASELINE_TARGET/$rel_path"
        replace_copy_dir "$persisted_path" "$dest"
        count=$((count + 1))
    done < <(list_top_level_baselines "$BASELINE_DIR")
    if [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}⚠${NC} No top-level '${BASELINE_NAME}' series found under ${BASELINE_DIR}"
    else
        echo -e "  ${GREEN}✓${NC} Restored ${count} baseline series into ${CRITERION_BASELINE_TARGET}"
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

    # Unconditionally clear any previous benchmark log prior to environment checks
    mkdir -p "$PROJECT_DIR/target/logs"
    LOG_FILE="$PROJECT_DIR/target/logs/regression-check.log"
    : > "$LOG_FILE"

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

    # ── Baseline coverage cross-check (F-24b) ────────────────────────────────
    # Every executed benchmark MUST have been compared against a baseline
    # series. A benchmark without one (new/renamed, or a restore that silently
    # lost a series) passes unverified otherwise — fail closed instead.
    local missing_series
    if ! missing_series=$(missing_baseline_coverage "$LOG_FILE" "$CRITERION_BASELINE_TARGET" "$BASELINE_NAME"); then
        echo -e "\n${RED}${BOLD}❌ BASELINE_COVERAGE_GAP${NC} — no executed benchmark could be parsed from $LOG_FILE"
        echo -e "  The coverage cross-check is blind: the Criterion log format changed or the run is empty."
        echo -e "  Fail-closed: nothing passes unverified. Investigate the log before re-running."
        dashboard_phase_receipt "regression_check" "FAIL" 1 0 0 "BASELINE_COVERAGE_GAP"
        exit 1
    fi
    if [ -n "$missing_series" ]; then
        echo -e "\n${RED}${BOLD}❌ BASELINE_COVERAGE_GAP${NC} — executed benchmark(s) without a saved baseline series:"
        for bench_id in $missing_series; do
            echo -e "    ${RED}${bench_id}${NC}"
        done
        echo -e "  A new or renamed benchmark must never pass unverified (fail-closed)."
        echo -e "  A human operator must re-bootstrap the baseline:"
        echo -e "    ${YELLOW}utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
        dashboard_phase_receipt "regression_check" "FAIL" 1 0 0 "BASELINE_COVERAGE_GAP"
        exit 1
    fi

    # (F-24c) Audit trail: record the verified benchmark set in the receipt so
    # coverage can be audited without re-reading Criterion's log.
    local executed_count coverage_list
    executed_count=$(executed_bench_ids "$LOG_FILE" | wc -l)
    coverage_list=$(executed_bench_ids "$LOG_FILE" | paste -sd ',' -)
    printf '{"kind":"baseline_coverage","phase_id":"regression_baseline_coverage","verified_benchmarks":"%s","count":%s,"run_id":"%s"}\n' \
        "$coverage_list" "$executed_count" "${NAM_RUN_ID:-}" >> "$REGRESSION_RECEIPT"
    echo -e "  ${GREEN}✓ Coverage cross-check: all ${executed_count} executed benchmark(s) have baseline series.${NC}"

    echo -e "${GREEN}✓ No performance regression detected.${NC}"
    dashboard_phase_receipt "regression_check" "PASS" 0 1 1 ""
}

# ── Main entry point ───────────────────────────────────────────────────────
# All top-level logic is encapsulated here so that every helper function is
# defined before its first invocation (hoisting), respecting set -euo pipefail.
main() {
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
        echo -e "  ${YELLOW}⚠ taskset not found — running without core pinning.${NC}"
    fi

    echo -e "${BLUE}${BOLD}  Performance Regression Gate${NC}"
    echo -e "  Core: ${YELLOW}${BENCH_CORE}${NC}  Baseline: ${YELLOW}${BASELINE_NAME}${NC}"
    echo -e "${BLUE}${BOLD}  Estimated time: ± 2.0 minutes${NC}"

    # ── JSONL receipt infrastructure (T-E4.6-2) ────────────────────────────
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

    NAM_RUN_ID="${NAM_RUN_ID:-$(date +%s%N-$$)}"
    export NAM_RUN_ID

    printf '{"kind":"build_metadata","pipeline":"performance-regression","cargo_profile":"release","target_triple":"%s","rustflags":"%s","rustc_version":"%s","cpu_model":"%s","cpu_microarch":"%s","effective_isa":"%s","frequency_governor":"%s","git_commit":"%s","bench_core":"%s","run_id":"%s"}\n' \
        "$REGR_TARGET_TRIPLE" "${RUSTFLAGS:-}" "$REGR_RUSTC_VER" "$REGR_CPU_MODEL" "$REGR_CPU_MICROARCH" "$REGR_CPU_MICROARCH" \
        "$REGR_FREQ_GOV" "$REGR_GIT_COMMIT" "$BENCH_CORE" "$NAM_RUN_ID" >> "$REGRESSION_RECEIPT"

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
}

main "$@"
