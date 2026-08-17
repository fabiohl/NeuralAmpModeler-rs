#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# quality-dashboard.sh — NeuralAmpModeler-rs Quality Dashboard
#
# Runs all fidelity suites and performance benchmarks, captures their outputs,
# and generates a comprehensive human-friendly report covering the full NeuralAmpModeler-rs
# universe: all architectures, models, quality modes, and ISAs.
#
# Usage:
#   ./utils/quality-dashboard.sh                        Standard dashboard (fidelity + performance)
#   ./utils/quality-dashboard.sh --fidelity-only        Fidelity tests only
#   ./utils/quality-dashboard.sh --bench-only           Benchmarks only
#   ./utils/quality-dashboard.sh --full                 Standard dashboard + delegated long-suite
#                                                       audit (utils/tests-long.sh, human-only)
#   ./utils/quality-dashboard.sh --save <json>          Promote the validated JSON contract after
#                                                       all fidelity phases pass (nam_quality save)
#   ./utils/quality-dashboard.sh --check <json>         Verify metrics against the JSON contract
#                                                       (nam_quality ingest + verify, S2.T7)
#
# Since Sprint S2 the contract authority is Rust + JSON: the wrapper only
# orchestrates phases; every contract interpretation is delegated to the
# `nam_quality` binary (ingest/verify/save). ASCII `.txt` contracts are
# rejected with ERROR + exit 2. The human render keeps reading the phase logs
# until Sprint S6 removes it — `--check`/`--save` never use the logs.

set -euo pipefail

export LC_ALL=C

# Save original invocation directory for path resolution before sourcing _lib.sh
INVOCATION_PWD="$(pwd)"

PHASE_TOTAL=0
source "$(dirname "$0")/_lib.sh"

# ── Argument parsing ────────────────────────────────────────────────────────

SAVE_FILE=""
CHECK_FILE=""
MODE="standard"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}✗ ERROR: --save requires a JSON contract path.${NC}" >&2
                exit 2
            fi
            SAVE_FILE="$2"
            shift 2
            ;;
        --check)
            if [ $# -lt 2 ] || [ -z "$2" ]; then
                echo -e "${RED}✗ ERROR: --check requires a JSON contract path.${NC}" >&2
                exit 2
            fi
            CHECK_FILE="$2"
            shift 2
            ;;
        --fidelity-only)
            MODE="fidelity"
            shift
            ;;
        --bench-only)
            MODE="bench"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        *)
            echo -e "${RED}✗ ERROR: unknown argument: $1${NC}" >&2
            echo "Run '$0 --help' for usage." >&2
            exit 2
            ;;
    esac
done

# Resolve relative file arguments against invocation directory or project root
if [ -n "$CHECK_FILE" ] && [[ "$CHECK_FILE" != /* ]]; then
    if [ -f "$INVOCATION_PWD/$CHECK_FILE" ]; then
        CHECK_FILE="$INVOCATION_PWD/$CHECK_FILE"
    elif [ -f "$PROJECT_DIR/$CHECK_FILE" ]; then
        CHECK_FILE="$PROJECT_DIR/$CHECK_FILE"
    elif [ -f "$PROJECT_DIR/docs/$(basename "$CHECK_FILE")" ]; then
        CHECK_FILE="$PROJECT_DIR/docs/$(basename "$CHECK_FILE")"
    fi
fi

if [ -n "$SAVE_FILE" ] && [[ "$SAVE_FILE" != /* ]]; then
    SAVE_FILE="$INVOCATION_PWD/$SAVE_FILE"
fi

# S2.T7: JSON-only contract authority — fail fast before any phase runs.
if [ -n "$CHECK_FILE" ] && [[ "$CHECK_FILE" != *.json ]]; then
    echo -e "${RED}✗ ERROR: --check requires a JSON contract (*.json, e.g. docs/quality-contract.json).${NC}" >&2
    echo -e "${RED}  ASCII .txt contracts are no longer supported since Sprint S2.${NC}" >&2
    exit 2
fi
if [ -n "$SAVE_FILE" ] && [[ "$SAVE_FILE" != *.json ]]; then
    echo -e "${RED}✗ ERROR: --save requires a JSON contract (*.json).${NC}" >&2
    exit 2
fi

# ── Setup ───────────────────────────────────────────────────────────────────

LOGDIR="target/logs/dashboard"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

JSONL_METRICS="${LOGDIR}/metrics.jsonl"
: "${NAM_METRICS_JSONL:=$JSONL_METRICS}"
DASHBOARD_PHASE_RECEIPT="${LOGDIR}/phase_receipt.jsonl"

TMPDIR="${TMPDIR:-/tmp}"
PARSEDIR="$(mktemp -d "$TMPDIR/nam-dashboard-XXXXXX")"
trap 'rm -rf "$PARSEDIR"' EXIT INT TERM

# ── System info detection ───────────────────────────────────────────────────

detect_isa() {
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

detect_cpu_model() {
    grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown"
}

ISA="$(detect_isa)"
CPU_MODEL="$(detect_cpu_model)"
NOW="$(date '+%Y-%m-%d %H:%M:%S %z')"
RUSTC_VER="$(rustc --version 2>/dev/null || echo 'unknown')"
CARGO_TARGET_TRIPLE="$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')"
CARGO_RUSTFLAGS="${RUSTFLAGS:-}"
CARGO_PROFILE="release"

# Locale-safe numeric printf — bash's printf is locale-aware for %f/%e/%g;
# in locales using commas for decimals, force C locale for numbers.

# Write build metadata to the JSONL receipt for provenance tracking.
# Records cargo_profile, target_triple, RUSTFLAGS, and rustc_version so
# comparisons across different build profiles can be detected and rejected.
write_build_metadata() {
    if [ -z "${DASHBOARD_PHASE_RECEIPT:-}" ] || [ ! -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        return 0
    fi
    local git_commit git_dirty
    git_commit=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then git_dirty="true"; else git_dirty="false"; fi
    printf '{"kind":"build_metadata","cargo_profile":"%s","target_triple":"%s","rustflags":"%s","rustc_version":"%s","git_commit":"%s","git_dirty_state":%s,"run_id":"%s","effective_isa":"%s"}\n' \
        "$CARGO_PROFILE" "$CARGO_TARGET_TRIPLE" "${CARGO_RUSTFLAGS:-}" "$RUSTC_VER" "$git_commit" "$git_dirty" "${NAM_RUN_ID:-$RUN_ID}" "$ISA" >> "$DASHBOARD_PHASE_RECEIPT"
}

# Unified metric formatter — locale-safe (LC_ALL=C), auto-detects scientific notation.
# Values containing [eE] use %.2e; numbers smaller than 0.0001 use %.2e; otherwise %.4f.

# Per-model mapping: dashboard label -> exact .nam fixture filename.
# Every golden model with an oracle measurement in test_summary_table is mapped 1:1.
# Models without oracle coverage transparently show N/A.

# Validate that a string is a finite decimal/scientific float (F-01/F-28).
# Accepts leading/trailing dot forms (".5", "1.") and e-notation; rejects the
# non-finite sentinels ("inf", "-inf", "nan"), empties, and anything else.
_is_finite_num() {
    local v="$1"
    [ -n "$v" ] || return 1
    case "${v,,}" in
        inf|-inf|+inf|infinity|-infinity|nan|-nan) return 1 ;;
    esac
    [[ "$v" =~ ^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]
}

# Backward-compatible alias used by f64-oracle log parsing: any metric accepted
# here must be a finite number (see F-28 — canonical regex, rejects non-finite).
_is_numeric_esr() {
    _is_finite_num "$1"
}

# Render an untrusted metric value safely for `echo -e` (F-01 defense-in-depth):
# strip backslashes and control characters so a corrupt value cannot inject
# escape sequences into the terminal/report.


# ── Data storage (global associative arrays) ────────────────────────────────

declare -A ESR_NAMCORE
declare -A ESR_F64_PAIRED
declare -A LATENCY_US
declare -A ISA_RESULTS

declare -a MODEL_BENCH_NAMES=()
declare -a DSP_BENCH_NAMES=()
declare -a ALL_BENCH_NAMES=()

SPECTRAL_PASSED_COUNT=0

# ── Mandatory dashboard phases (fail-closed gate) ───────────────────────────
# Phases listed here must receive PASS status in the phase receipt for the
# dashboard to be considered successful. Any FAIL or missing receipt triggers
# a contract violation in --check mode.
declare -A PHASE_MANDATORY=(
    ["golden_vectors"]="1"
    ["reference_oracle_f64"]="1"
    ["quick_parity"]="1"
)

# ── Coverage matrix axes (coverage governance) ─────────────────────────────
# Tracks per-axis coverage info for the coverage matrix summary.
COVERAGE_NAMCORE_PARITY=0
COVERAGE_F64_ORACLE=0
COVERAGE_ISA_OPTIMIZATIONS=0
COVERAGE_SPECTRAL_BASELINES=0
COVERAGE_RT_PERFORMANCE=0

# ── Duration tracking ───────────────────────────────────────────────────────

OVERALL_START=$(date +%s%N)
FIDELITY_DURATION_S=0
BENCH_DURATION_S=0

# ── Phase 0: freshness & third-party integrity (F-22) ───────────────────────
# Conservative freshness audit of the golden fixtures and NAMCore reference,
# per the PO binding directive: goldens are trustworthy by default and only
# require regeneration when a real change to a model file or the NAMCore
# reference occurred. check_freshness decides "stale" solely from hash
# provenance against the committed manifest — never from age. A real,
# unregenerated change fails with the typed reason STALE_FIXTURES (fail-closed,
# before any measurement phase runs). Missing third-party mirrors are a
# graceful, non-noisy SKIP (never an error).

run_phase0_freshness() {
    local freshness_log="$LOGDIR/freshness.log"

    set +e
    run_freshness_gate artifacts-hard > "$freshness_log" 2>&1
    local freshness_rc=$?
    set -e

    if [ "$freshness_rc" -ne 0 ]; then
        local reason="${FRESHNESS_REASON:-FRESHNESS_FAILED}"
        DASHBOARD_PHASE_HAD_FAILURE=1
        dashboard_phase_receipt "freshness" "FAIL" "$freshness_rc" 0 1 "$reason"
        echo -e "  ${RED}✗${NC} ${reason} — golden fixtures diverged from the committed manifest."
        echo -e "  ${RED}  Run './tests/fixtures/golden_gen_build.sh' to regenerate goldens and manifest.${NC}"
        echo -e "  ${RED}  Freshness gate detail: ${freshness_log}${NC}"
        return 1
    fi

    dashboard_phase_receipt "freshness" "PASS" "$freshness_rc" 0 1 ""
    echo -e "  ${GREEN}ok${NC} freshness gate passed (goldens trustworthy; no false alarms)"

    # NAMCore third-party reference — graceful, non-noisy skip when absent.
    local tp_log="$LOGDIR/third_party.log"
    set +e
    ensure_third_party soft > "$tp_log" 2>&1
    local tp_rc=$?
    set -e
    if [ "$tp_rc" -ne 0 ]; then
        dashboard_phase_receipt "third_party" "SKIP_CAPABILITY" 0 0 1 "third_party_absent"
        echo -e "  ${YELLOW}ⓘ NAMCore third-party mirror unavailable — C++ parity stages skip gracefully.${NC}"
    else
        dashboard_phase_receipt "third_party" "PASS" 0 1 1 ""
    fi

    return 0
}

# ── Run: golden_vectors ─────────────────────────────────────────────────────

run_golden_vectors() {
    local start_t end_t
    start_t=$(date +%s%N)
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" run_dashboard_phase "golden_vectors" 50 1 \
        cargo test --release --features testing --test models golden_vectors -- --test-threads=1 --nocapture
    end_t=$(date +%s%N)
    FIDELITY_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
}

# ── Run: reference_oracle_f64 ───────────────────────────────────────────────

run_reference_oracle() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "reference_oracle_f64" 10 \
        cargo test --release --features testing --test parity reference_oracle_f64 -- --test-threads=1 --nocapture
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
}

# ── Run: isa_parity ─────────────────────────────────────────────────────────

run_isa_parity() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "isa_parity" 5 \
        cargo test --release --features testing --test parity isa_parity -- --test-threads=1 --nocapture
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
}

# ── Run: spectral_fidelity ──────────────────────────────────────────────────

run_spectral_fidelity() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "spectral_fidelity" 5 \
        cargo test --release --features testing --test models spectral_fidelity -- --nocapture
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
}

# ── Run: lstm_activation_precision ──────────────────────────────────────────

run_activation_precision() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "lstm_activation_precision" 5 \
        cargo test --release --features testing --test models lstm_activation_precision -- --nocapture
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
}

# ── Run: quick_parity ────────────────────────────────────────────────────────

run_quick_parity() {
    local start_t end_t
    start_t=$(date +%s%N)
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" run_dashboard_phase "quick_parity" 50 1 \
        cargo test --release --features testing --test parity quick_parity -- --test-threads=1 --nocapture
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    write_build_metadata
}

# ── Run: regression_gate benchmarks ────────────────────────────────────────
# Invokes the performance regression gate (tests-performance-regression.sh --check)
# which validates fingerprint compatibility before running benchmarks.
#
# Single performance-status classifier (F-08 / EP-05): the typed receipt written
# by the regression script is the sole source of truth. "Performance not
# verified" (MISSING_BASELINE or INCOMPARABLE_ENVIRONMENT) has exactly ONE
# semantic — NOT_VERIFIED — displayed unambiguously (never green, never counted
# as PASS) but not aborting the default-mode run; `--check` against the quality
# contract fails on it (fail-closed alarm). Only a real regression or benchmark
# failure is a hard FAIL phase. The performance gate itself lives in
# tests-performance-regression.sh (per-sprint, human-triggered).
BENCH_PERF_NOT_VERIFIED=0
BENCH_NOT_VERIFIED_REASON=""

run_benchmarks() {
    local start_t end_t
    start_t=$(date +%s%N)

    local reg_script="$PROJECT_DIR/utils/tests-performance-regression.sh"
    local bench_log="$LOGDIR/regression_gate.log"
    : > "$bench_log"

    if [ -x "$reg_script" ]; then
        set +e
        NAM_RUN_ID="$RUN_ID" "$reg_script" --check > "$bench_log" 2>&1
        local reg_exit=$?
        set -e

        local reg_receipt_file="$PROJECT_DIR/target/logs/regression_phase_receipt.jsonl"
        local receipt_line=""
        local receipt_run_id=""
        local receipt_status=""
        local receipt_reason=""
        if [ -f "$reg_receipt_file" ]; then
            receipt_line=$(grep '"phase_id":"regression_check"' "$reg_receipt_file" 2>/dev/null | tail -1 || echo "")
            receipt_run_id=$(echo "$receipt_line" | grep -o '"run_id":"[^"]*"' | cut -d'"' -f4 || echo "")
            receipt_status=$(echo "$receipt_line" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "")
            receipt_reason=$(echo "$receipt_line" | grep -o '"reason":"[^"]*"' | cut -d'"' -f4 || echo "")
        fi

        # S6.T2: the single performance-status classifier lives in
        # `qa::classify` (F-08) — delegated via `nam_quality classify`; no
        # second 3-way copy here anymore.
        local perf_status
        perf_status=$("$NAM_QUALITY_BIN" classify --status "${receipt_status:-}" --reason "${receipt_reason:-}") \
            || perf_status="FAIL"
        case "$perf_status" in
            PASS)
                dashboard_phase_receipt "regression_gate" "PASS" 0 10 10 ""
                echo -e "  ${GREEN}ok${NC} regression_gate passed — no performance regression"
                ;;
            NOT_VERIFIED)
                BENCH_PERF_NOT_VERIFIED=1
                BENCH_NOT_VERIFIED_REASON="${receipt_reason:-UNKNOWN}"
                echo -e "  ${YELLOW}⚠ NOT_VERIFIED${NC} performance not verified (${BENCH_NOT_VERIFIED_REASON})"
                echo -e "  ${YELLOW}  Performance is not certified in this run; --check against the quality contract fails on this.${NC}"
                case "$BENCH_NOT_VERIFIED_REASON" in
                    MISSING_BASELINE)
                        echo -e "  ${YELLOW}  DIAGNOSTIC: No statistical baseline found in .performance-baselines/.${NC}"
                        echo -e "  ${YELLOW}  To bootstrap baseline (human operator only), run:${NC}"
                        echo -e "  ${YELLOW}    utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
                        ;;
                    INCOMPARABLE_ENVIRONMENT)
                        echo -e "  ${YELLOW}  DIAGNOSTIC: CPU/environment fingerprint differs from saved baseline.${NC}"
                        echo -e "  ${YELLOW}  To re-calibrate baseline (human operator only), run:${NC}"
                        echo -e "  ${YELLOW}    utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
                        ;;
                    BASELINE_COVERAGE_GAP)
                        echo -e "  ${YELLOW}  DIAGNOSTIC: Benchmark suite has new benches without baseline series.${NC}"
                        echo -e "  ${YELLOW}  To update baseline series (human operator only), run:${NC}"
                        echo -e "  ${YELLOW}    utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
                        ;;
                esac
                dashboard_phase_receipt "regression_gate" "NOT_VERIFIED" 0 0 10 "$receipt_reason"
                ;;
            *)
                DASHBOARD_PHASE_HAD_FAILURE=1
                dashboard_phase_receipt "regression_gate" "FAIL" "$reg_exit" 0 10 "Performance regression check failed"
                echo -e "  ${RED}✗${NC} regression_gate check failed (exit=${reg_exit})"
                echo -e "  ${YELLOW}  DIAGNOSTIC: Review target/logs/regression-check.log for details.${NC}"
                echo -e "  ${YELLOW}  If performance variation is due to background/thermal noise, re-bootstrap with:${NC}"
                echo -e "  ${YELLOW}    utils/tests-performance-regression.sh --bootstrap-baseline${NC}"
                ;;
        esac

        # Copy Criterion benchmark output from the regression script's log
        # strictly when reg_exit == 0 AND receipt_status == PASS AND receipt_run_id == RUN_ID.
        local criterion_log="target/logs/regression-check.log"
        if [ "$reg_exit" -eq 0 ] && [ "$receipt_status" = "PASS" ] && [ -n "$RUN_ID" ] && [ "$receipt_run_id" = "$RUN_ID" ] && [ -f "$criterion_log" ]; then
            cp "$criterion_log" "$bench_log"
        else
            : > "$bench_log"
        fi
    else
        echo -e "  ${YELLOW}⚠ regression script not found at $reg_script — skipping${NC}"
        dashboard_phase_receipt "regression_gate" "SKIP_CAPABILITY" 0 0 10 "regression_script_missing"
    fi

    end_t=$(date +%s%N)
    BENCH_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
}

# ── Parse: golden_vectors ───────────────────────────────────────────────────

parse_golden_vectors() {
    local log="$LOGDIR/golden_vectors.log"

    # S2.T7: the JSONL metric stream is consumed exclusively by the
    # `nam_quality` binary (ingest/verify) — no jq anywhere in this script.
    # The human render (transitory until Sprint S6) reads the phase log.
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/golden_vectors.parsed"
    LC_ALL=C awk '
    BEGIN { label=""; rate=""; mode="Live" }
    /^\[NeuralAmpModelerCore/ && /NAM-rs — / {
        line = $0
        sub(/^\[NeuralAmpModelerCore.*NAM-rs — /, "", line)
        sub(/\]$/, "", line)
        at_pos = index(line, " @ ")
        if (at_pos > 0) {
            lbl = substr(line, 1, at_pos - 1)
            rate_str = substr(line, at_pos + 3)
            gsub(/ Hz.*/, "", rate_str)
            rate = rate_str
        } else {
            lbl = line
            rate = "48000"
        }
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        if (lbl ~ /^(T[0-9]|T-)/) {
            label = ""
            next
        }
        label = lbl " @" rate
        mode = "Live"
        if (index($0, "(HQ)") > 0) mode = "HQ"
    }
    /^  ESR     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); esr_val = parts[1]
        esr_db = ""
        if (match($0, /\([-0-9.]+ dB\)/)) {
            esr_db = substr($0, RSTART+1, RLENGTH-5)
            gsub(/[[:space:]]+/, "", esr_db)
        }
        key = label " " mode
        printf "ESR_NAMCORE\t%s\t%s\n", key, esr_val
        printf "ESR_NAMCORE_DB\t%s\t%s\n", key, esr_db
    }
    /^  SNR     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); snr_val = parts[1]
        printf "SNR_DB\t%s\t%s\n", label " " mode, snr_val
    }
    /^  MSE     =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); mse_val = parts[1]
        printf "MSE\t%s\t%s\n", label " " mode, mse_val
    }
    /^  MR-STFT =/ && label != "" {
        split($0, a, "="); val_str = a[2]; gsub(/^[[:space:]]+/, "", val_str)
        split(val_str, parts, /[[:space:]]+/); mrstft_val = parts[1]
        printf "MRSTFT\t%s\t%s\n", label " " mode, mrstft_val
    }
    /^\[ConvNet Self-Golden/ {
        label = ""
        printf "ESR_NAMCORE\tConvNet Test @48000 Live\tN/A\n"
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric key value; do
        if [ "$metric" = "ESR_NAMCORE" ]; then
            ESR_NAMCORE["$key"]="$value"
        fi
    done < "$parsed"
}

# ── Parse: reference_oracle_f64 ─────────────────────────────────────────────

parse_oracle_f64() {
    local log="$LOGDIR/reference_oracle_f64.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/oracle_f64_summary.parsed"
    LC_ALL=C awk '
    BEGIN { in_table = 0 }
    /^=== ESR\(f32 vs f64 oracle\) Summary/ { in_table = 1; next }
    /^---/ && in_table { in_table = 2; next }
    in_table == 2 && (/^$/ || /^test /) { in_table = 0; next }
    in_table == 2 && /^(MODEL CLASS LABEL|PROD FIRST|ORACLE FIRST)/ { next }
    in_table == 2 && $1 ~ /\.nam$/ && $3 ~ /^[+-]?[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$/ {
        printf "ESR_F64_TABLE\t%s\t%s\t%s\t%s\n", $1, $2, $3, $4
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric filename family esr_lin esr_db; do
        [[ "$metric" == "ESR_F64_TABLE" ]] || continue
        if [ -n "$filename" ] && [ -n "$esr_lin" ]; then
            if _is_numeric_esr "$esr_lin"; then
                ESR_F64_PAIRED["$filename"]="$esr_lin"
            else
                echo "  ⚠ Dropping non-numeric f64 entry for '$filename': [$esr_lin] (malformed line in reference_oracle_f64.log)" >&2
            fi
        fi
    done < "$parsed"

    # Parse paired prewarm ESR lines using POSIX-compatible awk string processing
    LC_ALL=C awk '
    / ESR\(f32 vs oracle, prewarm-paired/ {
        line = $0
        lbl = line
        sub(/[[:space:]]*ESR\(f32 vs oracle, prewarm-paired.*/, "", lbl)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        esr = line
        sub(/.*:[[:space:]]*/, "", esr)
        sub(/[[:space:]]*\(.*/, "", esr)
        gsub(/[[:space:]]/, "", esr)
        esr_db = line
        sub(/.*\(/, "", esr_db)
        sub(/[[:space:]]*dB\).*/, "", esr_db)
        gsub(/[[:space:]]/, "", esr_db)
        if (lbl != "" && esr != "") {
            printf "%s\t%s\t%s\n", lbl, esr, esr_db
        }
    }
    ' "$log" > "$parsed" 2>/dev/null || true

    while IFS=$'\t' read -r label esr esr_db; do
        if [ -n "$label" ] && [ -n "$esr" ]; then
            if _is_numeric_esr "$esr"; then
                ESR_F64_PAIRED["$label"]="$esr"
            else
                echo "  ⚠ Dropping non-numeric f64 entry for family '$label': [$esr] (malformed line in reference_oracle_f64.log)" >&2
            fi
        fi
    done < "$parsed"

}

# ── Parse: isa_parity ───────────────────────────────────────────────────────

parse_isa_parity() {
    local log="$LOGDIR/isa_parity.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/isa_parity.parsed"
    LC_ALL=C awk '
    /\[ISA Matrix\]/ {
        line = $0
        sub(/.*\[ISA Matrix\][[:space:]]*/, "", line)
        if (line ~ /self-consistency/) {
            split(line, parts, "|")
            lbl = parts[1]; gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
            mse = "N/A"
            if (index(line, "MSE=") > 0) {
                mse = line
                sub(/.*MSE=/, "", mse)
                sub(/[[:space:]].*/, "", mse)
            }
            printf "%s | self-consistency\t%s\n", lbl, mse
        } else {
            split(line, parts, "|")
            lbl = parts[1]; gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
            isa_part = parts[2]
            ref_isa = isa_part; sub(/→.*/, "", ref_isa); gsub(/^[[:space:]]+|[[:space:]]+$/, "", ref_isa)
            test_isa = isa_part; sub(/.*→/, "", test_isa); gsub(/^[[:space:]]+|[[:space:]]+$/, "", test_isa)
            esr = "N/A"
            if (index(line, "ESR=") > 0) {
                esr = line
                sub(/.*ESR=/, "", esr)
                sub(/[[:space:]].*/, "", esr)
            }
            printf "%s | %s->%s\t%s\n", lbl, ref_isa, test_isa, esr
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r key val; do
        if [ -n "$key" ]; then
            ISA_RESULTS["$key"]="$val"
        fi
    done < "$parsed"
}

# ── Parse: spectral_fidelity ────────────────────────────────────────────────

parse_spectral_fidelity() {
    local log="$LOGDIR/spectral_fidelity.log"
    [ -f "$log" ] || return 0
    SPECTRAL_PASSED_COUNT=$(grep -c 'all spectral fidelity metrics within baseline tolerance' "$log" 2>/dev/null || true)
}


# ── Parse: regression_gate ──────────────────────────────────────────────────

parse_benchmarks() {
    local log="$LOGDIR/regression_gate.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/benchmarks.parsed"
    # Micro DSP benches run DSP_MICRO_BATCH (64) blocks per Criterion sample
    # so timer noise stays below the 2% wall. Report per-block latency here.
    LC_ALL=C awk '
    BEGIN {
        bench = ""
        micro_batch["RT_DSP_Resampler_44k1_to_48k"] = 64
        micro_batch["RT_DSP_Resampler_96k_to_48k"] = 64
        micro_batch["RT_DSP_CabSim_IR_Medium"] = 64
    }
    /^RT_/ && !/regression_gate/ && length($1) > 3 { bench = $1 }
    bench != "" && /time:.*\[/ {
        line = $0
        start_bracket = index(line, "[")
        end_bracket   = index(line, "]")
        if (start_bracket > 0 && end_bracket > start_bracket) {
            bracket_part = substr(line, start_bracket + 1, end_bracket - start_bracket - 1)
            split(bracket_part, parts, /[[:space:]]+/)
            if (parts[3] != "" && parts[4] != "") {
                median_val  = parts[3]
                median_unit = parts[4]
                if (median_unit == "ns")      us = median_val / 1000
                else if (median_unit == "µs") us = median_val
                else if (median_unit == "ms") us = median_val * 1000
                else if (median_unit == "s")  us = median_val * 1000000
                else                          us = median_val
                if (bench in micro_batch && micro_batch[bench] > 0) {
                    us = us / micro_batch[bench]
                }
                printf "LATENCY\t%s\t%.2f\n", bench, us
            }
        }
        bench = ""
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric bench latency; do
        if [[ "$metric" == "LATENCY" ]]; then
            LATENCY_US["$bench"]="$latency"
        fi
    done < "$parsed"

    MODEL_BENCH_NAMES=(
        RT_WaveNet_Std_CH16
        RT_WaveNet_Feather_CH8
        RT_WaveNet_Lite_CH12
        RT_WaveNet_Nano_CH4
        RT_A2_Full_CH8
        RT_A2_Lite_CH3
        RT_LSTM_1x16
        RT_LSTM_2x8
        RT_Linear
        RT_ConvNet
        RT_WaveNet_Dyn_Free
        RT_LSTM_Dyn_1x7
        RT_A2_Dyn_Gated_CH8
        RT_A2_Dyn_Blended_CH3
    )

    DSP_BENCH_NAMES=(
        RT_DSP_Resampler_44k1_to_48k
        RT_DSP_Resampler_96k_to_48k
        RT_DSP_CabSim_IR_Medium
        RT_DSP_Pipeline_Base_NoOS
        RT_DSP_Pipeline_HQ_4xOS
    )

    ALL_BENCH_NAMES=("${MODEL_BENCH_NAMES[@]}" "${DSP_BENCH_NAMES[@]}")
}








# ── Coverage matrix computation ─────────────────────────────────────────────
# Fills coverage axis counters from parsed data for the governance report.

compute_coverage() {
    set +u
    local namcore_count f64_count isa_count spectral_count rt_count

    namcore_count="${#ESR_NAMCORE[@]}"
    COVERAGE_NAMCORE_PARITY="$namcore_count"

    f64_count="${#ESR_F64_PAIRED[@]}"
    COVERAGE_F64_ORACLE="$f64_count"

    isa_count="${#ISA_RESULTS[@]}"
    COVERAGE_ISA_OPTIMIZATIONS="$isa_count"

    COVERAGE_SPECTRAL_BASELINES="${SPECTRAL_PASSED_COUNT:-0}"

    rt_count="${#ALL_BENCH_NAMES[@]}"
    COVERAGE_RT_PERFORMANCE="$rt_count"

    set -u

    if [ -n "${DASHBOARD_PHASE_RECEIPT:-}" ] && [ -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        printf '{"kind":"coverage_matrix","namcore_parity":%s,"f64_oracle":%s,"isa_optimizations":%s,"spectral_baselines":%s,"rt_performance":%s}\n' \
            "$COVERAGE_NAMCORE_PARITY" "$COVERAGE_F64_ORACLE" "$COVERAGE_ISA_OPTIMIZATIONS" \
            "$COVERAGE_SPECTRAL_BASELINES" "$COVERAGE_RT_PERFORMANCE" >> "$DASHBOARD_PHASE_RECEIPT"
    fi
}

# ── Phase count extraction from phase receipt ────────────────────────────────
# Extracts aggregated PHASE counts (passed/failed/ignored/filtered/skip_capability)
# from the phase receipt JSONL for governance tracking. These are phase-status
# tallies, NOT individual test tallies (F-25); real executed test counts are
# asserted per-phase by _lib.sh::assert_ran_tests against each phase log.
declare -A TEST_COUNTS=(
    ["passed"]="0"
    ["failed"]="0"
    ["ignored"]="0"
    ["filtered"]="0"
    ["skip_capability"]="0"
)

extract_test_counts() {
    if [ ! -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        return 0
    fi
    local skip_count pass_count fail_count
    skip_count=$(grep -c '"status":"SKIP_CAPABILITY"\|"status":"SKIP_OPTIONAL_FIXTURE"' "$DASHBOARD_PHASE_RECEIPT" 2>/dev/null || true)
    pass_count=$(grep -c '"status":"PASS"' "$DASHBOARD_PHASE_RECEIPT" 2>/dev/null || true)
    fail_count=$(grep -c '"status":"FAIL"' "$DASHBOARD_PHASE_RECEIPT" 2>/dev/null || true)
    TEST_COUNTS["skip_capability"]="${skip_count:-0}"
    TEST_COUNTS["passed"]="${pass_count:-0}"
    TEST_COUNTS["failed"]="${fail_count:-0}"

    if [ -n "${DASHBOARD_PHASE_RECEIPT:-}" ] && [ -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        printf '{"kind":"test_counts","passed":%s,"failed":%s,"ignored":%s,"filtered":%s,"skip_capability":%s}\n' \
            "${TEST_COUNTS[passed]}" "${TEST_COUNTS[failed]}" "${TEST_COUNTS[ignored]}" \
            "${TEST_COUNTS[filtered]}" "${TEST_COUNTS[skip_capability]}" >> "$DASHBOARD_PHASE_RECEIPT"
    fi
}




# ── Contract verification (S2.T7: Rust+JSON authority) ──────────────────────
# The wrapper no longer interprets contracts. Every --check/--save decision
# is delegated to the `nam_quality` binary (ingest/verify/save) built from
# this crate with the `testing` feature. No ASCII loader, no jq. The
# `nam_quality` CLI contract: exit 0 success, exit 1 run-time failure,
# exit 2 usage error.

NAM_QUALITY_BIN="${NAM_QUALITY_BIN:-$PROJECT_DIR/target/debug/nam_quality}"

ensure_nam_quality_bin() {
    if [ -x "$NAM_QUALITY_BIN" ]; then
        return 0
    fi
    if ! ( cd "$PROJECT_DIR" && cargo build --quiet --features testing --bin nam_quality >/dev/null 2>&1 ); then
        echo -e "  ${RED}${BOLD}❌ FATAL: failed to build nam_quality${NC}" >&2
        return 1
    fi
    return 0
}

# ── Latency stream for the ingest (transitory until the S3 Rust sink) ───────
# Converts the Criterion-log parse (LATENCY_US, still consumed by the human
# render) into the latency JSONL the ingest expects. The verify engine matches
# `RT_*` bench labels against the contract entry ids (verify.rs::match_latency).
write_latency_stream() {
    local out="$1"
    : > "$out"
    local bench
    for bench in "${!LATENCY_US[@]}"; do
        printf '{"kind":"latency","label":"%s","median_latency_us":%s}\n' \
            "$bench" "${LATENCY_US[$bench]}" >> "$out"
    done
}

# ── Quality contract ingestion (report.jsonl) ───────────────────────────────
# Merges the phase receipt + fidelity metrics JSONL (+ latency stream) into
# the verify report consumed by --check. Runs after the phases so the report
# always reflects the current run's receipts (incl. the long-suite audit).

run_quality_ingest() {
    local report_file="$1" latency_stream="$2"
    local -a ingest_args=(--receipt "$DASHBOARD_PHASE_RECEIPT")
    if [ -f "$NAM_METRICS_JSONL" ]; then
        ingest_args+=(--metrics "$NAM_METRICS_JSONL")
    fi
    if [ -s "$latency_stream" ]; then
        ingest_args+=(--latency "$latency_stream")
    fi
    if ! "$NAM_QUALITY_BIN" ingest "${ingest_args[@]}" --out "$report_file"; then
        echo -e "  ${RED}✗${NC} nam_quality ingest failed — report JSONL not generated (see stderr above)." >&2
        return 1
    fi
    return 0
}

# ── Quality contract verification (--check) ─────────────────────────────────
# Fail-closed, JSON-only: the contract must be `docs/quality-contract.json`
# and the report must come from the current run (ingest of the phase receipt
# + metrics JSONL). Verdict strings stay in English and are printed by the
# binary exactly as scripts grep them (`PERFORMANCE: NOT_VERIFIED`,
# `CONTRACT VIOLATED`, `FIDELITY: OK/FAIL`).

run_quality_check() {
    local check_file="$1" report_file="$2"
    if [ ! -f "$report_file" ]; then
        echo -e "  ${RED}✗${NC} Report missing (${report_file}) — run the dashboard phases first; --check verifies the current run." >&2
        return 1
    fi
    if ! "$NAM_QUALITY_BIN" verify --contract "$check_file" --report "$report_file"; then
        return 1
    fi
    return 0
}

# ── Extended long-suite audit (--full mode) ──────────────────────────────────
# F-06 / EP-05: --full must have a real, auditable effect. It delegates to
# utils/tests-long.sh — the canonical nightly/pre-release audit suite — after
# the standard dashboard phases complete. Human-only by design (long runtime,
# hard third-party requirements). The delegated suite's exit status gates the
# dashboard: a failed extended audit is a typed receipt FAIL (blocks --save and
# fails --check). NAM_LONG_SUITE_SCRIPT overrides the delegated script path.
run_extended_audit() {
    local long_script="${NAM_LONG_SUITE_SCRIPT:-$PROJECT_DIR/utils/tests-long.sh}"
    if [ ! -x "$long_script" ] && [ ! -f "$long_script" ]; then
        echo -e "  ${RED}✗${NC} extended audit script not found at $long_script"
        dashboard_phase_receipt "long_suite" "FAIL" 127 0 0 "long_suite_script_missing"
        return 0
    fi

    echo -e "${BLUE}${BOLD}-> Delegating extended long-suite audit to $long_script...${NC}"
    local long_exit
    set +e
    bash "$long_script" 2>&1 | tee "$LOGDIR/long_suite.log"
    long_exit=${PIPESTATUS[0]}
    set -e

    if [ "$long_exit" -ne 0 ]; then
        echo -e "  ${RED}✗${NC} extended long-suite audit failed (exit=${long_exit}, log: ${LOGDIR}/long_suite.log)"
        dashboard_phase_receipt "long_suite" "FAIL" "$long_exit" 0 0 "delegated tests-long.sh failed"
    else
        echo -e "  ${GREEN}ok${NC} extended long-suite audit passed"
        dashboard_phase_receipt "long_suite" "PASS" 0 0 0 ""
    fi
    return 0
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    RUN_ID="${NAM_RUN_ID:-$(date +%s%N-$$)}"
    export NAM_RUN_ID="$RUN_ID"

    if ! ensure_nam_quality_bin; then
        exit 1
    fi

    local run_phases=0
    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        run_phases=$((run_phases + 7))
    fi
    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        run_phases=$((run_phases + 1))
    fi
    if [ "$MODE" = "full" ]; then
        run_phases=$((run_phases + 1))
    fi
    PHASE_TOTAL=$((run_phases + 4))

    echo -e "${BLUE}${BOLD}===============================================================${NC}"
    echo -e "${BLUE}${BOLD}    NeuralAmpModeler-rs Quality Dashboard${NC}"
    echo -e "${BLUE}${BOLD}    Modo: ${MODE}${NC}"
    echo -e "${BLUE}${BOLD}===============================================================${NC}"

    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        phase "freshness (Fase 0) — golden & NAMCore integrity"
        if ! run_phase0_freshness; then
            exit 1
        fi

        phase "golden_vectors"
        run_golden_vectors

        phase "reference_oracle_f64"
        run_reference_oracle

        phase "isa_parity"
        run_isa_parity

        phase "spectral_fidelity"
        run_spectral_fidelity

        phase "lstm_activation_precision"
        run_activation_precision

        phase "quick_parity"
        run_quick_parity
    fi

    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        phase "regression_gate benchmarks"
        run_benchmarks
    fi

    phase "Parseando resultados"
    parse_golden_vectors
    parse_oracle_f64
    parse_isa_parity
    parse_spectral_fidelity
    parse_benchmarks

    # Coverage + test counts feed the `coverage_matrix`/`test_counts` kinds of
    # the receipt BEFORE the ingest; the human render itself moved to
    # `nam_quality render` (S6) and runs after the report is ingested.
    phase "Governanca (coverage + test counts)"
    compute_coverage
    extract_test_counts

    if [ "$MODE" = "full" ]; then
        phase "extended long-suite audit (delegated)"
        run_extended_audit
    fi

    local final_exit=0
    if [ "${DASHBOARD_PHASE_HAD_FAILURE:-0}" -ne 0 ]; then
        final_exit=1
    fi

    # ── S2.T7: Rust+JSON report ingestion ────────────────────────────────────
    # The report (phase receipt + fidelity metrics + latency stream) is the
    # single input of --check; the human render below (S6) pipes the same
    # report and never feeds the verdict.
    local report_file="$LOGDIR/report.jsonl"
    local latency_stream="$PARSEDIR/latency.jsonl"
    phase "Ingestando report JSONL (nam_quality ingest)"
    if [ -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        write_latency_stream "$latency_stream"
        if ! run_quality_ingest "$report_file" "$latency_stream"; then
            final_exit=1
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Phase receipt missing — report JSONL not generated."
    fi

    # ── S6.T2: the human render pipes the typed report (report.jsonl) ───────
    phase "Renderizando dashboard (nam_quality render)"
    if [ -f "$report_file" ]; then
        "$NAM_QUALITY_BIN" render --report "$report_file" --ansi || true
    fi

    # --save: transactional atomic write of the JSON contract — only promoted
    # when every fidelity phase PASSed (NOT_VERIFIED performance never blocks
    # saving; PERF-006). Automated/CI agents are strictly prohibited from
    # invoking --save (S7.T3 keeps it human-only).
    if [ -n "$SAVE_FILE" ]; then
        if ! "$NAM_QUALITY_BIN" save --contract "$SAVE_FILE" --receipt "$DASHBOARD_PHASE_RECEIPT"; then
            echo -e "  ${RED}✗${NC} Contract NOT saved (nam_quality save refused — see stderr above)."
            final_exit=1
        fi
    fi

    if [ -n "$CHECK_FILE" ]; then
        if ! run_quality_check "$CHECK_FILE" "$report_file"; then
            final_exit=1
        fi
    fi

    exit "$final_exit"
}

main "$@"
