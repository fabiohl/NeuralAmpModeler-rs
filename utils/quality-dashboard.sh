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
#   ./utils/quality-dashboard.sh --full                 Full long-suite audit (human-only)
#   ./utils/quality-dashboard.sh --save <filename>      Save plain-text copy alongside display
#   ./utils/quality-dashboard.sh --check <file>         Verify metrics against quality contract

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
            SAVE_FILE="$2"
            shift 2
            ;;
        --check)
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
            shift
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
    if echo "$flags" | grep -q 'avx512f'; then
        echo "AVX-512"
    elif echo "$flags" | grep -q 'avx2'; then
        echo "AVX2 (x86-64-v3)"
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
_nfmt() { LC_NUMERIC=C printf "$@"; }

# Write build metadata to the JSONL receipt for provenance tracking.
# Records cargo_profile, target_triple, RUSTFLAGS, and rustc_version so
# comparisons across different build profiles can be detected and rejected.
write_build_metadata() {
    if [ -z "${DASHBOARD_PHASE_RECEIPT:-}" ] || [ ! -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        return 0
    fi
    printf '{"kind":"build_metadata","cargo_profile":"%s","target_triple":"%s","rustflags":"%s","rustc_version":"%s"}\n' \
        "$CARGO_PROFILE" "$CARGO_TARGET_TRIPLE" "${CARGO_RUSTFLAGS:-}" "$RUSTC_VER" >> "$DASHBOARD_PHASE_RECEIPT"
}

# Unified metric formatter — locale-safe (LC_ALL=C), auto-detects scientific notation.
# Values containing [eE] use %.2e; numbers smaller than 0.0001 use %.2e; otherwise %.4f.
_fmt_metric() {
    local val="$1"
    [ -z "$val" ] || [ "$val" = "N/A" ] && { echo "N/A"; return; }
    if [[ "$val" =~ [eE] ]]; then
        LC_ALL=C printf "%.2e" "$val" 2>/dev/null || echo "$val"
    else
        LC_ALL=C awk -v v="$val" 'BEGIN {
            x = v + 0;
            ax = (x < 0) ? -x : x;
            if (ax != 0 && ax < 0.0001) printf "%.2e", x;
            else printf "%.4f", x;
        }' 2>/dev/null || echo "$val"
    fi
}

# Per-model mapping: dashboard label -> exact .nam fixture filename.
# Every golden model with an oracle measurement in test_summary_table is mapped 1:1.
# Models without oracle coverage transparently show N/A.
declare -A ESR_F64_MODEL_MAP=(
    # WaveNet standard family
    ["BossWN-standard"]="BossWN-standard.nam"
    ["BossWN-feather"]="BossWN-feather.nam"
    ["BossWN-nano"]="BossWN-nano.nam"
    ["EVH-5150-Lite"]="EVH-5150-Lite.nam"
    ["wavenet_a1_standard (Official)"]="wavenet_a1_standard.nam"
    ["WaveNet Condition DSP (CH=3, cond=3, dynamic path) C++ cross-reference"]="wavenet_condition_dsp.nam"
    ["WaveNet Official (CH=3, dynamic path) C++ cross-reference"]="wavenet_official.nam"
    ["WaveNetDyn Free-Shape (CH=7→4, dynamic path) C++ cross-reference"]="wavenet_dyn_free.nam"
    # LSTM family
    ["BossLSTM-1x16"]="BossLSTM-1x16.nam"
    ["BossLSTM-2x8"]="BossLSTM-2x8.nam"
    ["lstm (Official)"]="lstm.nam"
    ["LSTM-Dyn 1×7 (dynamic path) C++ cross-reference"]="lstm_dyn_test.nam"
    # A2 family
    ["WaveNet A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["WaveNet A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["Container A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Lite (CH=3) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["Container File A2-Full (CH=8) C++ cross-reference"]="wavenet_a2_full.nam"
    ["SlimmableContainer A2 Example (CH=3→6) C++ cross-reference"]="wavenet_a2_lite.nam"
    ["WaveNet A2 Dynamic Gated (CH=8, gated layers 3/23) C++ cross-reference"]="a2_dynamic_gated_ch8.nam"
    ["WaveNet A2 Dynamic Blended (CH=3, blended layers 2/23) C++ cross-reference"]="a2_dynamic_blended_ch3.nam"
    # A2-FiLM
    ["WaveNet A2-FiLM-Lite (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_lite.nam"
    ["WaveNet A2-FiLM Chaos Stress (CH=3, FiLM active) C++ cross-reference"]="wavenet_a2_film_chaos_stress.nam"
    ["WaveNet A2-FiLM-Full (CH=8, FiLM active) C++ cross-reference"]="wavenet_a2_film_full.nam"
    ["WaveNet A2-FiLM-InputMixinPre (CH=3, input_mixin_pre_film) C++ cross-reference"]="wavenet_a2_film_input_mixin_pre.nam"
    # ConvNet
    ["ConvNet Test"]="convnet_test.nam"
    # Quick parity labels
    ["Quick LSTM 1×16"]="BossLSTM-1x16.nam"
    ["Quick WaveNet CH16"]="BossWN-standard.nam"
    ["Quick A2-Full"]="wavenet_a2_full.nam"
    ["Quick ConvNet"]="convnet_test.nam"
)

# Validate that a string represents a valid scientific or decimal float ESR value.
# Guards against log corruption or interleaved stdout contaminating metric entries.
_is_numeric_esr() {
    local v="$1"
    [[ "$v" =~ ^[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]]
}

_lookup_esr_f64() {
    local golden_label="$1"

    local oracle_fixture="${ESR_F64_MODEL_MAP[$golden_label]:-}"
    if [ -n "$oracle_fixture" ]; then
        local val
        set +u; val="${ESR_F64_PAIRED[$oracle_fixture]:-}"; set -u
        if [ -n "$val" ] && _is_numeric_esr "$val"; then
            echo "$val"
            echo "exact"
            return
        fi
    fi

    # Fallback 1: Lookup golden_label directly
    local direct
    set +u; direct="${ESR_F64_PAIRED[$golden_label]:-}"; set -u
    if [ -n "$direct" ] && _is_numeric_esr "$direct"; then
        echo "$direct"
        echo "exact"
        return
    fi

    # Fallback 2: Lookup golden_label.nam
    local with_nam="${golden_label}.nam"
    set +u; direct="${ESR_F64_PAIRED[$with_nam]:-}"; set -u
    if [ -n "$direct" ] && _is_numeric_esr "$direct"; then
        echo "$direct"
        echo "exact"
        return
    fi

    echo "N/A"
    echo "none"
}

# ── Data storage (global associative arrays) ────────────────────────────────

declare -A ESR_NAMCORE
declare -A ESR_NAMCORE_DB
declare -A ESR_F64_COLD
declare -A ESR_F64_PAIRED
declare -A ESR_F64_DB_COLD
declare -A ESR_F64_DB_PAIRED
declare -A SNR_DB
declare -A MSE_VAL
declare -A MRSTFT
declare -A LATENCY_US
declare -A BENCH_MODEL_MAP
declare -A ISA_RESULTS
declare -A ACTIVATION_SNR
declare -A F64_DECOMPOSITION
declare -A MODEL_ESR_F64_TABLE
declare -A MODEL_MACS

declare -a MODEL_ORDER
declare -a ALL_BENCH_NAMES

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

# ── Coverage matrix axes (governança de cobertura) ──────────────────────────
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

# ── Run: golden_vectors ─────────────────────────────────────────────────────

run_golden_vectors() {
    local start_t end_t
    start_t=$(date +%s%N)
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" run_dashboard_phase "golden_vectors" 50 \
        "cargo test --release --features testing --test models golden_vectors -- --test-threads=1 --nocapture > \"$LOGDIR/golden_vectors.log\" 2>&1"
    end_t=$(date +%s%N)
    FIDELITY_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
}

# ── Run: reference_oracle_f64 ───────────────────────────────────────────────

run_reference_oracle() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "reference_oracle_f64" 10 \
        "cargo test --release --features testing --test parity reference_oracle_f64 -- --test-threads=1 --nocapture > \"$LOGDIR/reference_oracle_f64.log\" 2>&1"
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
        "cargo test --release --features testing --test parity isa_parity -- --test-threads=1 --nocapture > \"$LOGDIR/isa_parity.log\" 2>&1"
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
        "cargo test --release --features testing --test models spectral_fidelity -- --nocapture > \"$LOGDIR/spectral_fidelity.log\" 2>&1"
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
        "cargo test --release --features testing --test models lstm_activation_precision -- --nocapture > \"$LOGDIR/lstm_activation_precision.log\" 2>&1"
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
}

# ── Run: quick_parity ────────────────────────────────────────────────────────

run_quick_parity() {
    local start_t end_t
    start_t=$(date +%s%N)
    NAM_METRICS_JSONL="$NAM_METRICS_JSONL" run_dashboard_phase "quick_parity" 50 \
        "cargo test --release --features testing --test parity quick_parity -- --test-threads=1 --nocapture > \"$LOGDIR/quick_parity.log\" 2>&1"
    end_t=$(date +%s%N)
    local dur
    dur=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    FIDELITY_DURATION_S=$(awk -v a="$FIDELITY_DURATION_S" -v b="$dur" 'BEGIN { printf "%.1f", a + b }')
    write_build_metadata
}

# ── Run: regression_gate benchmarks ────────────────────────────────────────

run_benchmarks() {
    local start_t end_t
    start_t=$(date +%s%N)
    run_dashboard_phase "regression_gate" 10 \
        "cargo bench --bench regression_gate --features testing > \"$LOGDIR/regression_gate.log\" 2>&1"
    end_t=$(date +%s%N)
    BENCH_DURATION_S=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
}

# ── Parse: JSONL fidelity metrics ───────────────────────────────────────────

parse_jsonl_fidelity() {
    local jsonl="${NAM_METRICS_JSONL:-}"
    [ -n "$jsonl" ] && [ -f "$jsonl" ] || return 1

    local parsed="$PARSEDIR/jsonl_fidelity.parsed"

    if command -v jq >/dev/null 2>&1; then
        jq -r '
            select(.kind == "fidelity" or (.kind | not))
            | {
                label: .label,
                esr: .esr,
                esr_db: .esr_db,
                snr_db: .snr_db,
                mse: .mse,
                mrstft: .mrstft
              }
            | [.label, .esr, .esr_db, .snr_db, .mse, .mrstft]
            | @tsv' "$jsonl" 2>/dev/null | \
        LC_ALL=C awk -F'\t' 'NF >= 6 {
            printf "ESR_NAMCORE\t%s\t%s\n", $1, $2
            printf "ESR_NAMCORE_DB\t%s\t%s\n", $1, $3
            printf "SNR_DB\t%s\t%s\n", $1, $4
            printf "MSE\t%s\t%s\n", $1, $5
            printf "MRSTFT\t%s\t%s\n", $1, $6
        }' > "$parsed" 2>/dev/null
    else
        # POSIX awk extraction fallback
        LC_ALL=C awk '{
            label=""; esr=""; esr_db=""; snr_db=""; mse=""; mrstft=""; kind=""
            if (index($0, "\"label\"") > 0) { label = $0; sub(/.*"label"[[:space:]]*:[[:space:]]*"/, "", label); sub(/".*/, "", label) }
            if (index($0, "\"kind\"") > 0) { kind = $0; sub(/.*"kind"[[:space:]]*:[[:space:]]*"/, "", kind); sub(/".*/, "", kind) }
            if (kind != "" && kind != "fidelity") next
            if (index($0, "\"esr\"") > 0) { esr = $0; sub(/.*"esr"[[:space:]]*:[[:space:]]*/, "", esr); sub(/[,}].*/, "", esr); gsub(/[[:space:]]/, "", esr) }
            if (index($0, "\"esr_db\"") > 0) { esr_db = $0; sub(/.*"esr_db"[[:space:]]*:[[:space:]]*/, "", esr_db); sub(/[,}].*/, "", esr_db); gsub(/[[:space:]]/, "", esr_db) }
            if (index($0, "\"snr_db\"") > 0) { snr_db = $0; sub(/.*"snr_db"[[:space:]]*:[[:space:]]*/, "", snr_db); sub(/[,}].*/, "", snr_db); gsub(/[[:space:]]/, "", snr_db) }
            if (index($0, "\"mse\"") > 0) { mse = $0; sub(/.*"mse"[[:space:]]*:[[:space:]]*/, "", mse); sub(/[,}].*/, "", mse); gsub(/[[:space:]]/, "", mse) }
            if (index($0, "\"mrstft\"") > 0) { mrstft = $0; sub(/.*"mrstft"[[:space:]]*:[[:space:]]*/, "", mrstft); sub(/[,}].*/, "", mrstft); gsub(/[[:space:]]/, "", mrstft) }
            if (label != "" && esr != "") {
                printf "ESR_NAMCORE\t%s\t%s\n", label, esr
                printf "ESR_NAMCORE_DB\t%s\t%s\n", label, esr_db
                printf "SNR_DB\t%s\t%s\n", label, snr_db
                printf "MSE\t%s\t%s\n", label, mse
                printf "MRSTFT\t%s\t%s\n", label, mrstft
            }
        }' "$jsonl" > "$parsed" 2>/dev/null
    fi

    [ -s "$parsed" ] || return 1

    while IFS=$'\t' read -r metric key value; do
        case "$metric" in
            ESR_NAMCORE)    ESR_NAMCORE["$key"]="$value" ;;
            ESR_NAMCORE_DB) ESR_NAMCORE_DB["$key"]="$value" ;;
            SNR_DB)         SNR_DB["$key"]="$value" ;;
            MSE)            MSE_VAL["$key"]="$value" ;;
            MRSTFT)         MRSTFT["$key"]="$value" ;;
        esac
    done < "$parsed"

    # Label remapping for quick_parity -> golden_vectors key space
    declare -A _LMAP=(
        ["Quick ConvNet @48000 Live"]="ConvNet Test @48000 Live"
    )
    for _old in "${!_LMAP[@]}"; do
        local _new="${_LMAP[$_old]}"
        [ -n "${ESR_NAMCORE[$_old]:-}" ] && ESR_NAMCORE["$_new"]="${ESR_NAMCORE[$_old]}"
        [ -n "${ESR_NAMCORE_DB[$_old]:-}" ] && ESR_NAMCORE_DB["$_new"]="${ESR_NAMCORE_DB[$_old]}"
        [ -n "${SNR_DB[$_old]:-}" ] && SNR_DB["$_new"]="${SNR_DB[$_old]}"
        [ -n "${MSE_VAL[$_old]:-}" ] && MSE_VAL["$_new"]="${MSE_VAL[$_old]}"
        [ -n "${MRSTFT[$_old]:-}" ] && MRSTFT["$_new"]="${MRSTFT[$_old]}"
        unset "ESR_NAMCORE[$_old]" "ESR_NAMCORE_DB[$_old]" "SNR_DB[$_old]" "MSE_VAL[$_old]" "MRSTFT[$_old]"
    done

    local count
    set +u; count="${#ESR_NAMCORE[@]}"; set -u
    if [ -n "$count" ] && [ "$count" -gt 0 ]; then
        local sorted_keys
        set +u; sorted_keys=$(printf "%s\n" "${!ESR_NAMCORE[@]}" | sort -u); set -u
        while IFS= read -r key; do
            [ -n "$key" ] && MODEL_ORDER+=("$key")
        done <<< "$sorted_keys"
    fi

    return 0
}

# ── Parse: golden_vectors ───────────────────────────────────────────────────

parse_golden_vectors() {
    local log="$LOGDIR/golden_vectors.log"

    set +u; local count="${#ESR_NAMCORE[@]}"; set -u
    if parse_jsonl_fidelity; then
        set +u; count="${#ESR_NAMCORE[@]}"; set -u
        echo -e "  ${GREEN}ok${NC} metricas carregadas via JSONL (${count} entradas)"
        return 0
    fi

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
        case "$metric" in
            ESR_NAMCORE)    ESR_NAMCORE["$key"]="$value" ;;
            ESR_NAMCORE_DB) ESR_NAMCORE_DB["$key"]="$value" ;;
            SNR_DB)         SNR_DB["$key"]="$value" ;;
            MSE)            MSE_VAL["$key"]="$value" ;;
            MRSTFT)         MRSTFT["$key"]="$value" ;;
        esac
    done < "$parsed"

    set +u; local count="${#ESR_NAMCORE[@]}"; set -u
    if [ -n "$count" ] && [ "$count" -gt 0 ]; then
        local sorted_keys
        set +u; sorted_keys=$(printf "%s\n" "${!ESR_NAMCORE[@]}" | sort -u); set -u
        while IFS= read -r key; do
            [ -n "$key" ] && MODEL_ORDER+=("$key")
        done <<< "$sorted_keys"
    fi
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
                echo "  ⚠ Descartando entrada f64 nao-numerica para '$filename': [$esr_lin] (linha malformada em reference_oracle_f64.log)" >&2
            fi
        fi
        [ -n "$filename" ] && [ -n "$esr_db" ] && ESR_F64_DB_PAIRED["$filename"]="$esr_db"
        [ -n "$filename" ] && MODEL_ESR_F64_TABLE["$filename"]="${family}|${esr_lin}|${esr_db}"
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
                echo "  ⚠ Descartando entrada f64 nao-numerica para familia '$label': [$esr] (linha malformada em reference_oracle_f64.log)" >&2
            fi
        fi
        [ -n "$label" ] && [ -n "$esr_db" ] && ESR_F64_DB_PAIRED["$label"]="$esr_db"
    done < "$parsed"

    # Parse decomposition blocks
    LC_ALL=C awk '
    BEGIN { lbl=""; buf=""; in_decomp=0 }
    /Decomposition:/ {
        lbl = $0
        sub(/ Decomposition:.*/, "", lbl)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        buf = $0 "\n"
        in_decomp = 1
        next
    }
    in_decomp {
        if ($0 ~ /^[[:space:]]*(ESR|ΔESR|combined|Δ|accumulation|activation|weights)/) {
            buf = buf $0 "\n"
        } else {
            if (lbl != "" && buf != "") {
                gsub(/\n/, "@@", buf)
                printf "F64_DECOMP\t%s\t%s\n", lbl, buf
            }
            in_decomp = 0; lbl = ""; buf = ""
        }
    }
    END {
        if (lbl != "" && buf != "") {
            gsub(/\n/, "@@", buf)
            printf "F64_DECOMP\t%s\t%s\n", lbl, buf
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric key value; do
        [[ "$metric" == "F64_DECOMP" ]] || continue
        value="${value//@@/$'\n'}"
        F64_DECOMPOSITION["$key"]="$value"
    done < "$parsed"

    # Parse per-model f64 ESR from decomposition blocks for ESR_F64_COLD / ESR_F64_DB_COLD
    LC_ALL=C awk '
    /Decomposition:/ {
        lbl = $0
        sub(/Decomposition:.*/, "", lbl)
        sub(/.* \.\.\. /, "", lbl)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
        in_block = 1
        next
    }
    in_block {
        if ($0 ~ /ESR\(f32 vs f64 oracle\):/) {
            esr = $0
            sub(/.*ESR\(f32 vs f64 oracle\):[[:space:]]*/, "", esr)
            db = esr
            sub(/[[:space:]]*\(.*/, "", esr)
            sub(/.*\(/, "", db)
            sub(/[[:space:]]*dB\).*/, "", db)
            gsub(/[[:space:]]/, "", esr)
            gsub(/[[:space:]]/, "", db)
            printf "%s\t%s\t%s\n", lbl, esr, db
            in_block = 0
        } else if ($0 ~ /Decomposition:/) {
            lbl = $0
            sub(/Decomposition:.*/, "", lbl)
            sub(/.* \.\.\. /, "", lbl)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", lbl)
            in_block = 1
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r label esr db; do
        [ -n "$label" ] && [ -n "$esr" ] || continue
        if _is_numeric_esr "$esr"; then
            ESR_F64_COLD["$label"]="$esr"
            [ -n "$db" ] && ESR_F64_DB_COLD["$label"]="$db"
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
        [ -n "$key" ] && ISA_RESULTS["$key"]="$val"
    done < "$parsed"
}

# ── Parse: spectral_fidelity ────────────────────────────────────────────────

parse_spectral_fidelity() {
    local log="$LOGDIR/spectral_fidelity.log"
    [ -f "$log" ] || return 0
    SPECTRAL_PASSED_COUNT=$(grep -c 'all spectral fidelity metrics within baseline tolerance' "$log" 2>/dev/null || true)
}

# ── Parse: lstm_activation_precision ────────────────────────────────────────

parse_activation_precision() {
    local log="$LOGDIR/lstm_activation_precision.log"
    [ -f "$log" ] || return 0

    local parsed="$PARSEDIR/activation.parsed"
    LC_ALL=C awk '
    /Fast\(Pad/ && /Standard\(exact\)/ {
        line = $0
        model = line
        sub(/[[:space:]]*Fast\(Pad.*/, "", model)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", model)

        fast_snr = "N/A"; exact_snr = "N/A"; delta = "0.0"
        if (index(line, "Fast(Pad") > 0) {
            fast_snr = line
            sub(/.*Fast\(Pad[^)]*\):[[:space:]]*/, "", fast_snr)
            sub(/[[:space:]]*dB.*/, "", fast_snr)
        }
        if (index(line, "Standard(exact)") > 0) {
            exact_snr = line
            sub(/.*Standard\(exact\):[[:space:]]*/, "", exact_snr)
            sub(/[[:space:]]*dB.*/, "", exact_snr)
        }
        if (index(line, "Δ=") > 0) {
            delta = line
            sub(/.*Δ=/, "", delta)
            sub(/[[:space:]].*/, "", delta)
        }

        if (model != "") {
            printf "%s\t%s|%s|%s\n", model, fast_snr, exact_snr, delta
        }
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r model data; do
        [ -n "$model" ] && ACTIVATION_SNR["$model"]="$data"
    done < "$parsed"
}

# ── Parse: regression_gate ──────────────────────────────────────────────────

parse_benchmarks() {
    local log="$LOGDIR/regression_gate.log"
    [ -f "$log" ] || return 0

    BENCH_MODEL_MAP["RT_WaveNet_Std_CH16"]="WaveNet Standard CH16"
    BENCH_MODEL_MAP["RT_WaveNet_Feather_CH8"]="WaveNet Feather CH8"
    BENCH_MODEL_MAP["RT_WaveNet_Lite_CH12"]="WaveNet Lite CH12"
    BENCH_MODEL_MAP["RT_WaveNet_Nano_CH4"]="WaveNet Nano CH4"
    BENCH_MODEL_MAP["RT_A2_Full_CH8"]="A2 Full CH8"
    BENCH_MODEL_MAP["RT_A2_Lite_CH3"]="A2 Lite CH3"
    BENCH_MODEL_MAP["RT_LSTM_1x16"]="LSTM 1x16"
    BENCH_MODEL_MAP["RT_LSTM_2x8"]="LSTM 2x8"
    BENCH_MODEL_MAP["RT_Linear"]="Linear RF=2048"
    BENCH_MODEL_MAP["RT_ConvNet"]="ConvNet"

    # MACs constants: total_layers * CH^2 * K
    MODEL_MACS["WaveNet Standard CH16"]="15360"
    MODEL_MACS["WaveNet Feather CH8"]="3840"
    MODEL_MACS["WaveNet Lite CH12"]="8640"
    MODEL_MACS["WaveNet Nano CH4"]="960"

    local parsed="$PARSEDIR/benchmarks.parsed"
    LC_ALL=C awk '
    BEGIN { bench = "" }
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
                printf "LATENCY\t%s\t%.2f\n", bench, us
            }
        }
        bench = ""
    }
    ' "$log" > "$parsed"

    while IFS=$'\t' read -r metric bench latency; do
        [[ "$metric" == "LATENCY" ]] || continue
        LATENCY_US["$bench"]="$latency"
    done < "$parsed"

    for bn in RT_WaveNet_Std_CH16 RT_WaveNet_Feather_CH8 RT_WaveNet_Lite_CH12 \
              RT_WaveNet_Nano_CH4 RT_A2_Full_CH8 RT_A2_Lite_CH3 \
              RT_LSTM_1x16 RT_LSTM_2x8 RT_Linear RT_ConvNet; do
        ALL_BENCH_NAMES+=("$bn")
    done
}

# ── ESR verdict translation ─────────────────────────────────────────────────

esr_verdict() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "IDENTICO"
        else if (v+0 < 1e-5) print "IMPERCEPTIVEL"
        else if (v+0 < 1e-2) print "AUDIVEL APENAS COM A/B CIENTIFICO"
        else if (v+0 < 1e-1) print "AUDIVEL EM COMPARACAO DIRETA"
        else print "⚠ AUDIVEL"
    }'
}

esr_verdict_short() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "N/A"; return; }
    local cmp
    cmp=$(LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "1"
        else if (v+0 < 1e-5) print "2"
        else if (v+0 < 1e-2) print "3"
        else if (v+0 < 1e-1) print "4"
        else print "5"
    }')
    case "$cmp" in
        1) echo -e "${GREEN}IDENTICO${NC}" ;;
        2) echo -e "${GREEN}IMPERCEPTIVEL${NC}" ;;
        3) echo -e "${YELLOW}A/B CIENTIFICO${NC}" ;;
        4) echo -e "${YELLOW}AUDIVEL DIRETO${NC}" ;;
        *) echo -e "${RED}⚠ AUDIVEL${NC}" ;;
    esac
}

# Colorize an ESR numeric string with GREEN/YELLOW/RED ANSI codes.
_esr_color() {
    local esr="$1"
    [ -z "$esr" ] || [ "$esr" = "N/A" ] && { echo "$esr"; return; }
    local cmp
    cmp=$(LC_ALL=C awk -v v="$esr" 'BEGIN {
        if (v+0 < 1e-10) print "GREEN"
        else if (v+0 < 1e-5) print "GREEN"
        else if (v+0 < 1e-2) print "YELLOW"
        else if (v+0 < 1e-1) print "YELLOW"
        else print "RED"
    }')
    case "$cmp" in
        GREEN)  echo -e "${GREEN}${esr}${NC}" ;;
        YELLOW) echo -e "${YELLOW}${esr}${NC}" ;;
        RED)    echo -e "${RED}${esr}${NC}" ;;
        *)      echo "$esr" ;;
    esac
}

# Colorize a CPU budget percentage using headroom criteria.
_cpu_color() {
    local pct="$1"
    [ -z "$pct" ] || [ "$pct" = "N/A" ] && { echo "N/A"; return; }
    local f
    f=$(LC_ALL=C awk -v v="$pct" 'BEGIN { printf "%.0f", 100.0 - v }')
    if [ "$f" -gt 75 ]; then
        echo -e "${GREEN}${pct}%${NC}"
    elif [ "$f" -gt 50 ]; then
        echo -e "${GREEN}${pct}%${NC}"
    elif [ "$f" -gt 25 ]; then
        echo -e "${YELLOW}${pct}%${NC}"
    else
        echo -e "${RED}${pct}%${NC}"
    fi
}

budget_pct() {
    local latency_us="$1"
    [ -z "$latency_us" ] || [ "$latency_us" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v l="$latency_us" 'BEGIN { printf "%.1f", (l / 1333.0) * 100.0 }'
}

budget_folga() {
    local pct="$1"
    [ "$pct" = "N/A" ] && { echo "N/A"; return; }
    LC_ALL=C awk -v p="$pct" 'BEGIN { printf "%.1f", 100.0 - p }'
}

folga_color() {
    local folga="$1"
    [ "$folga" = "N/A" ] && { echo "N/A"; return; }
    local f
    f=$(LC_ALL=C awk -v v="$folga" 'BEGIN { printf "%.0f", v }')
    if [ "$f" -gt 75 ]; then
        echo -e "${GREEN}${folga}% ok${NC}"
    elif [ "$f" -gt 50 ]; then
        echo -e "${GREEN}${folga}%${NC}"
    elif [ "$f" -gt 25 ]; then
        echo -e "${YELLOW}${folga}%${NC}"
    else
        echo -e "${RED}${folga}% ⚠${NC}"
    fi
}

# ── Render: header ──────────────────────────────────────────────────────────

render_header() {
    local cpu_short="${CPU_MODEL:0:46}"
    printf "╔══════════════════════════════════════════════════════════════════╗\n"
    printf "║              NeuralAmpModeler-rs Quality Dashboard                            ║\n"
    printf "║              ------------------------------                      ║\n"
    printf "║              Medido em: %-25.25s                ║\n" "$NOW"
    printf "║              ISA: %-46.46s ║\n" "$ISA"
    printf "║              CPU: %-46.46s ║\n" "$cpu_short"
    printf "║              rustc: %-44.44s ║\n" "$RUSTC_VER"
    printf "╚══════════════════════════════════════════════════════════════════╝\n"
}

# ── Render: quick summary ───────────────────────────────────────────────────

render_quick_summary() {
    echo ""
    echo "🎯 RESUMO RAPIDO (para nao-cientistas)"
    echo "═══════════════════════════════════════"
    echo ""

    # Find representative models by scanning parsed keys
    local wn_std_key="" wn_feather_key="" lstm1_key="" lstm2_key=""
    local a2full_key="" a2lite_key="" convnet_key="" linear_key=""
    local a2film_key="" a1std_key=""

    set +u
    for key in "${!ESR_NAMCORE[@]}"; do
        case "$key" in
            *"BossWN-standard"*|*"WaveNet Std"*)   wn_std_key="$key" ;;
            *"BossWN-feather"*|*"WaveNet Feather"*) wn_feather_key="$key" ;;
            *"BossLSTM-1x16"*|*"LSTM 1x16"*)       lstm1_key="$key" ;;
            *"BossLSTM-2x8"*|*"LSTM 2x8"*)         lstm2_key="$key" ;;
            *"A2-Full"*|*"A2 Full"*)               a2full_key="$key" ;;
            *"A2-Lite"*|*"A2 Lite"*)               a2lite_key="$key" ;;
            *"ConvNet"*)                            convnet_key="$key" ;;
            *"linear_fft_rf2048"*|*"Linear FFT RF=2048"*) linear_key="$key" ;;
            *"A2-FiLM-Lite"*|*"FiLM.*Lite"*)       a2film_key="$key" ;;
            *"wavenet_a1_standard"*|*"A1 Standard"*) a1std_key="$key" ;;
        esac
    done
    set -u

    # Display entries (one per representative model family)
    _quick_entry() {
        local label="$1" icon="$2" key="$3" bench_name="$4"
        local esr_nam
        set +u; esr_nam="${ESR_NAMCORE[$key]:-N/A}"; set -u
        local esr_nam_display
        esr_nam_display=$(_fmt_metric "$esr_nam")
        # Extract model name for f64 lookup (strip rate and mode)
        local f64_label
        f64_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        local esr_f64 esr_f64_provenance
        { read -r esr_f64; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$f64_label")
        local esr_f64_display
        esr_f64_display=$(_fmt_metric "$esr_f64")
        local esr_f64_colored
        esr_f64_colored=$(_esr_color "$esr_f64_display")
        local verdict
        verdict=$(esr_verdict_short "$esr_nam")
        local latency
        set +u; latency="${LATENCY_US[$bench_name]:-N/A}"; set -u
        local pct_budget="N/A"
        local cpu_colored="N/A"
        if [ "$latency" != "N/A" ]; then
            pct_budget=$(budget_pct "$latency")
            cpu_colored=$(_cpu_color "$pct_budget")
        fi
        printf "  %s %-38s  vs NAMcore: %-10s %b  │  vs Ideal (f64): %-10s  │  ⚡ CPU: %s do budget\n" \
            "$icon" "${label:0:38}" "$esr_nam_display" "$verdict" "$esr_f64_colored" "$cpu_colored"
    }

    [ -n "$wn_std_key" ]   && _quick_entry "WaveNet Standard (CH16)"  "🎸" "$wn_std_key"    RT_WaveNet_Std_CH16
    [ -n "$a1std_key" ]    && _quick_entry "WaveNet A1 Standard"      "🎸" "$a1std_key"     RT_WaveNet_Std_CH16
    [ -n "$wn_feather_key" ] && _quick_entry "WaveNet Feather (CH8)"  "🎸" "$wn_feather_key" RT_WaveNet_Feather_CH8
    [ -n "$lstm1_key" ]    && _quick_entry "LSTM 1x16 (BossLSTM)"     "🎸" "$lstm1_key"     RT_LSTM_1x16
    [ -n "$lstm2_key" ]    && _quick_entry "LSTM 2x8 (BossLSTM)"      "🎸" "$lstm2_key"     RT_LSTM_2x8
    [ -n "$a2full_key" ]   && _quick_entry "A2 Full (CH8)"            "🎸" "$a2full_key"    RT_A2_Full_CH8
    [ -n "$a2lite_key" ]   && _quick_entry "A2 Lite (CH3)"            "🎸" "$a2lite_key"    RT_A2_Lite_CH3
    [ -n "$a2film_key" ]   && _quick_entry "A2-FiLM Lite (CH3)"       "🎸" "$a2film_key"    RT_A2_Lite_CH3
    [ -n "$convnet_key" ]  && _quick_entry "ConvNet"                  "🎸" "$convnet_key"   RT_ConvNet
    [ -n "$linear_key" ]   && _quick_entry "Linear (RF=2048)"         "🎸" "$linear_key"    RT_Linear

    echo ""
}

# ── Render: fidelity details table ──────────────────────────────────────────

# Classifies fidelity measurement rows as redundant (coverage from alternative entry points)
# vs canonical (primary golden_vector measurement).
_is_redundant_measurement() {
    local label="$1"
    [[ "$label" == Quick\ * ]]        && return 0
    [[ "$label" == Container\ * ]]    && return 0
    [[ "$label" == Container\ File\ * ]] && return 0
    [[ "$label" == T-* ]]             && return 0
    [[ "$label" == T[0-9]* ]]         && return 0
    return 1
}

# Generates context tags for fidelity rows in the red zone (ESR >= 0.1).
_red_zone_tags() {
    local label="$1" esr_nam="$2" esr_f64="$3"
    local tags=""

    local is_red=0
    if [ "$esr_nam" != "N/A" ]; then
        LC_ALL=C awk -v v="$esr_nam" 'BEGIN { if (v+0 >= 0.1) exit 0; exit 1 }' && is_red=1
    fi
    if [ "$is_red" -eq 0 ] && [ "$esr_f64" != "N/A" ]; then
        LC_ALL=C awk -v v="$esr_f64" 'BEGIN { if (v+0 >= 0.1) exit 0; exit 1 }' && is_red=1
    fi
    [ "$is_red" -eq 0 ] && { echo ""; return; }

    if [[ "$label" == *"condition_lstm"* ]] || [[ "$label" == *"Condition DSP LSTM"* ]]; then
        tags="$tags${RED}[EM INVESTIGAÇÃO]${NC} "
    fi

    if [ "$esr_f64" != "N/A" ] && [ "$esr_nam" != "N/A" ]; then
        local f64_div=0
        LC_ALL=C awk -v f64="$esr_f64" -v nam="$esr_nam" \
            'BEGIN { if (f64+0 >= 0.1 && f64+0 > nam*10.0) exit 0; exit 1 }' && f64_div=1
        if [ "$f64_div" -eq 1 ]; then
            tags="$tags${YELLOW}[orac: f64 div]${NC} "
        fi
    fi

    local gate
    case "$label" in
        *"condition_dsp"*|*"Condition DSP"*)         gate="1.0e-10" ;;
        *"Dynamic Blended"*)                         gate="1.0e-12" ;;
        *"Dynamic Gated"*)                           gate="1.0e-9" ;;
        *"condition_lstm"*|*"Condition DSP LSTM"*)   gate="fail-closed" ;;
        *"a2_max"*|*"A2 Max"*|*"A2-Max"*|*"KB-A2-MAX"*) gate="known-bug KB-A2-MAX" ;;
        *)                                           gate="0.1" ;;
    esac
    tags="$tags${YELLOW}[gate: ${gate}]${NC}"

    echo "$tags"
}

# Emits one row of the fidelity table.
_render_fidelity_row() {
    local display_key="$1" esr_nam_cell="$2" esr_f64_cell="$3" \
          snr_cell="$4" mrstft_cell="$5" mode="$6" tags="${7:-}"
    printf "  %-38s │ %-26b │ %-12s │ %-8s │ %-8s │ %-6s %s\n" \
        "$display_key" "$esr_nam_cell" "$esr_f64_cell" "$snr_cell" "$mrstft_cell" "$mode" "$tags"
}

# Renders the header row for fidelity tables.
_render_fidelity_header() {
    printf "  %-38s │ %-16s │ %-12s │ %-8s │ %-8s │ %s\n" \
        "Modelo" "ESR (vs NAMcore)" "ESR (vs f64)" "SNR dB" "MR-STFT" "Modo"
    printf "  %s │ %s │ %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..38})" "$(printf '─%.0s' {1..16})" \
        "$(printf '─%.0s' {1..12})" "$(printf '─%.0s' {1..8})" \
        "$(printf '─%.0s' {1..8})" "$(printf '─%.0s' {1..6})"
}

render_fidelity_details() {
    echo "📊 FIDELIDADE SONORA — Detalhes Tecnicos"
    echo "═════════════════════════════════════════"
    echo ""

    if [ ${#MODEL_ORDER[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum dado de fidelidade disponivel.${NC}"
        echo ""
        return
    fi

    # Pre-classify entries into canonical and redundant arrays
    local -a canonicals=()
    local -a redundants=()
    for key in "${MODEL_ORDER[@]}"; do
        local model_label
        model_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
        if _is_redundant_measurement "$model_label"; then
            redundants+=("$key")
        else
            canonicals+=("$key")
        fi
    done

    # ── Canonical table ──────────────────────────────────────────────────
    if [ ${#canonicals[@]} -gt 0 ]; then
        echo "  ── Fidelidade Canônica (golden_vectors) ──"
        echo ""
        _render_fidelity_header

        for key in "${canonicals[@]}"; do
            local esr_nam
            set +u; esr_nam="${ESR_NAMCORE[$key]:-N/A}"; set -u
            local esr_nam_short
            esr_nam_short=$(_fmt_metric "$esr_nam")
            local esr_color=""
            if [ "$esr_nam" != "N/A" ]; then
                esr_color=$(awk -v v="$esr_nam" 'BEGIN {
                    if (v+0 < 1e-10) print "GREEN"
                    else if (v+0 < 1e-5) print "GREEN"
                    else if (v+0 < 1e-2) print "YELLOW"
                    else if (v+0 < 1e-1) print "YELLOW"
                    else print "RED"
                }')
                case "$esr_color" in
                    GREEN)  esr_nam_short="${GREEN}${esr_nam_short}${NC}" ;;
                    YELLOW) esr_nam_short="${YELLOW}${esr_nam_short}${NC}" ;;
                    RED)    esr_nam_short="${RED}${esr_nam_short}${NC}" ;;
                esac
            fi
            local snr_val
            set +u; snr_val="${SNR_DB[$key]:-N/A}"; set -u
            local snr_formatted="$snr_val"
            [ "$snr_val" != "N/A" ] && snr_formatted=$(_nfmt "%.1f" "$snr_val")
            local mrstft
            set +u; mrstft="${MRSTFT[$key]:-N/A}"; set -u
            local mrstft_short
            mrstft_short=$(_fmt_metric "$mrstft")
            local model_label
            model_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
            local esr_f64 esr_f64_provenance
            { read -r esr_f64; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$model_label")
            local esr_f64_short
            esr_f64_short=$(_fmt_metric "$esr_f64")
            local esr_f64_colored
            esr_f64_colored=$(_esr_color "$esr_f64_short")
            local mode="Live"
            [[ "$key" == *" HQ"* ]] && mode="HQ"
            local display_key
            if [ "${IS_SAVING:-0}" = "1" ]; then
                display_key="$key"
            else
                display_key="${key:0:38}"
            fi

            local tags
            tags=$(_red_zone_tags "$model_label" "$esr_nam" "$esr_f64")

            _render_fidelity_row "$display_key" "$esr_nam_short" "$esr_f64_colored" \
                "$snr_formatted" "$mrstft_short" "$mode" "$tags"
        done
        echo ""
    fi

    # ── Redundant/coverage table ─────────────────────────────────────────
    if [ ${#redundants[@]} -gt 0 ]; then
        echo "  ── Cobertura Adicional (quick_parity, containers, regression gates) ──"
        echo "  (i) Estas medições validam os mesmos modelos por entry points alternativos."
        echo "       Linhas equivalentes da tabela canônica acima."
        echo ""
        _render_fidelity_header

        for key in "${redundants[@]}"; do
            local esr_nam
            set +u; esr_nam="${ESR_NAMCORE[$key]:-N/A}"; set -u
            local esr_nam_short
            esr_nam_short=$(_fmt_metric "$esr_nam")
            local esr_color=""
            if [ "$esr_nam" != "N/A" ]; then
                esr_color=$(awk -v v="$esr_nam" 'BEGIN {
                    if (v+0 < 1e-10) print "GREEN"
                    else if (v+0 < 1e-5) print "GREEN"
                    else if (v+0 < 1e-2) print "YELLOW"
                    else if (v+0 < 1e-1) print "YELLOW"
                    else print "RED"
                }')
                case "$esr_color" in
                    GREEN)  esr_nam_short="${GREEN}${esr_nam_short}${NC}" ;;
                    YELLOW) esr_nam_short="${YELLOW}${esr_nam_short}${NC}" ;;
                    RED)    esr_nam_short="${RED}${esr_nam_short}${NC}" ;;
                esac
            fi
            local snr_val
            set +u; snr_val="${SNR_DB[$key]:-N/A}"; set -u
            local snr_formatted="$snr_val"
            [ "$snr_val" != "N/A" ] && snr_formatted=$(_nfmt "%.1f" "$snr_val")
            local mrstft
            set +u; mrstft="${MRSTFT[$key]:-N/A}"; set -u
            local mrstft_short
            mrstft_short=$(_fmt_metric "$mrstft")
            local model_label
            model_label=$(echo "$key" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
            local esr_f64 esr_f64_provenance
            { read -r esr_f64; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$model_label")
            local esr_f64_short
            esr_f64_short=$(_fmt_metric "$esr_f64")
            local esr_f64_colored
            esr_f64_colored=$(_esr_color "$esr_f64_short")
            local mode="Live"
            [[ "$key" == *" HQ"* ]] && mode="HQ"
            local display_key
            if [ "${IS_SAVING:-0}" = "1" ]; then
                display_key="$key"
            else
                display_key="${key:0:38}"
            fi

            local tags
            tags=$(_red_zone_tags "$model_label" "$esr_nam" "$esr_f64")

            _render_fidelity_row "$display_key" "$esr_nam_short" "$esr_f64_colored" \
                "$snr_formatted" "$mrstft_short" "$mode" "$tags"
        done
        echo ""
    fi

    echo "  Legenda qualitativa (limites de audibilidade do ESR):"
    echo -e "    ${GREEN}verde${NC} = imperceptivel (ESR < 1e-5)"
    echo -e "    ${YELLOW}amarelo${NC} = audivel apenas com A/B cientifico (ESR < 1e-2)"
    echo -e "    ${RED}vermelho${NC} = ⚠ audivel — necessita investigacao (ESR >= 1e-1)"
    echo ""
}

# ── Render: performance ─────────────────────────────────────────────────────

# Compute median numerically without subshell loops.
_median() {
    if [ $# -eq 0 ]; then
        echo "N/A"
        return
    fi
    printf "%s\n" "$@" | LC_ALL=C sort -g | LC_ALL=C awk '
    { arr[NR] = $1 + 0 }
    END {
        if (NR == 0) { print "N/A"; exit }
        if (NR % 2 == 1) {
            printf "%.6f", arr[(NR + 1) / 2]
        } else {
            printf "%.6f", (arr[NR / 2] + arr[(NR / 2) + 1]) / 2.0
        }
    }'
}

render_performance() {
    echo "⚡ PERFORMANCE — Latencia por Bloco (64 amostras @ 48kHz)"
    echo "══════════════════════════════════════════════════════════"
    echo "  Deadline RT: 1333 µs (1.33 ms)"
    echo "  Eficiencia: µs por MMAC (mega-MACs) — menor e melhor"
    echo ""

    if [ ${#ALL_BENCH_NAMES[@]} -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum dado de performance disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-28s │ %-16s │ %-10s │ %-14s │ %s\n" \
        "Modelo" "Latencia Mediana" "% Budget" "µs/MMAC" "Folga"
    printf "  %s │ %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..28})" "$(printf '─%.0s' {1..16})" \
        "$(printf '─%.0s' {1..10})" "$(printf '─%.0s' {1..14})" \
        "$(printf '─%.0s' {1..18})"

    # Collect WaveNet efficiency values for headroom gate
    # Micro models (total MMAC < MIN_MMAC_THRESHOLD) are excluded from
    # median/outlier calculation because fixed per-block overhead dominates
    # their small denominator (F-18). Their µs/MMAC is still displayed.
    local MIN_MMAC_THRESHOLD="0.005"
    local -a wavenet_eff=()
    local -A efficiency_map

    for bn in "${ALL_BENCH_NAMES[@]}"; do
        local label latency pct folga folga_colored latency_display macs eff_display eff_val
        set +u
        label="${BENCH_MODEL_MAP[$bn]:-$bn}"
        latency="${LATENCY_US[$bn]:-N/A}"
        set -u
        pct="N/A"
        folga="N/A"
        folga_colored="N/A"
        if [ "$latency" != "N/A" ]; then
            pct=$(budget_pct "$latency")
            folga=$(budget_folga "$pct")
            folga_colored=$(folga_color "$folga")
        fi
        latency_display="$latency"
        if [ "$latency" != "N/A" ]; then
            latency_display=$(_nfmt "%.1f us" "$latency")
        fi

        # Calculate µs/MMAC
        set +u; macs="${MODEL_MACS[$label]:-}"; set -u
        eff_display="N/A"
        if [ -n "$macs" ] && [ "$latency" != "N/A" ]; then
            eff_val=$(LC_ALL=C awk -v lat="$latency" -v macs="$macs" 'BEGIN { printf "%.2f", lat / (macs / 1000000.0) }')
            eff_display=$(_nfmt "%.2f us/MMAC" "$eff_val")
            efficiency_map["$bn"]="$eff_val"
            if [[ "$label" == WaveNet* ]]; then
                local _model_mmacs
                _model_mmacs=$(LC_ALL=C awk -v macs="$macs" 'BEGIN { printf "%.6f", macs / 1000000.0 }')
                local _below_thr
                _below_thr=$(LC_ALL=C awk -v mm="$_model_mmacs" -v thr="$MIN_MMAC_THRESHOLD" \
                    'BEGIN { if (mm + 0 < thr + 0) print "1"; else print "0" }')
                if [ "$_below_thr" != "1" ]; then
                    wavenet_eff+=("$eff_val")
                fi
            fi
        fi

        printf "  %-28s │ %-16s │ %-10s │ %-14s │ %b\n" \
            "$label" "$latency_display" "${pct}%" "$eff_display" "$folga_colored"
    done

    echo ""
    echo "  (i) Folga > 50%:  Pode usar oversampling 2x sem xruns"
    echo "  (i) Folga > 75%:  Pode usar oversampling 4x sem xruns"
    echo "  (i) Folga < 25%:  ⚠ Risco de xruns com buffer de 64 amostras"
    echo ""

    # WaveNet efficiency check — excludes micro models where fixed per-block
    # overhead dominates the small MMAC denominator, leading to structurally
    # inflated µs/MMAC values (F-18). Models with total MMAC < MIN_MMAC_THRESHOLD
    # still display their µs/MMAC metric but are exempt from outlier detection.

    if [ ${#wavenet_eff[@]} -ge 2 ]; then
        local median_eff median_rounded
        median_eff=$(_median "${wavenet_eff[@]}")
        median_rounded=$(LC_ALL=C awk -v m="$median_eff" 'BEGIN { printf "%.2f", m }')
        for bn in "${ALL_BENCH_NAMES[@]}"; do
            local label
            set +u; label="${BENCH_MODEL_MAP[$bn]:-$bn}"; set -u
            [[ "$label" == WaveNet* ]] || continue
            local eff model_macs model_mmacs
            set +u; eff="${efficiency_map[$bn]:-}"; set -u
            [ -z "$eff" ] && continue
            set +u; model_macs="${MODEL_MACS[$label]:-0}"; set -u
            model_mmacs=$(LC_ALL=C awk -v macs="$model_macs" 'BEGIN { printf "%.6f", macs / 1000000.0 }')
            local below_threshold
            below_threshold=$(LC_ALL=C awk -v mm="$model_mmacs" -v thr="$MIN_MMAC_THRESHOLD" \
                'BEGIN { if (mm + 0 < thr + 0) print "1"; else print "0" }')
            if [ "$below_threshold" = "1" ]; then
                continue
            fi
            local is_outlier
            is_outlier=$(LC_ALL=C awk -v e="$eff" -v m="$median_eff" 'BEGIN { if (m > 0 && e > m * 2.0) print "1"; else print "0" }')
            if [ "$is_outlier" = "1" ]; then
                echo -e "  ${YELLOW}⚠ WARN:${NC} $label eficiencia $(LC_ALL=C printf "%.2f" "$eff") µs/MMAC — outlier >2× mediana ($median_rounded µs/MMAC)"
            fi
        done
        echo "  (i) µs/MMAC outlier detection excludes models with total MMAC < ${MIN_MMAC_THRESHOLD} (overhead-dominated regime)"
        echo ""
    fi
}

# ── Render: ISA parity ──────────────────────────────────────────────────────

render_isa_parity() {
    echo "🔬 ISA PARITY"
    echo "═════════════"
    echo ""

    local count
    set +u; count="${#ISA_RESULTS[@]}"; set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nao coberto no modo quick — rode tests-long para verificacao completa.${NC}"
        echo ""
        return
    fi

    local all_pass=true
    local self_consistency_count=0
    local cross_isa_count=0
    local cross_isa_pass=0

    set +u
    for key in "${!ISA_RESULTS[@]}"; do
        if [[ "$key" == *"self-consistency"* ]]; then
            self_consistency_count=$((self_consistency_count + 1))
        else
            cross_isa_count=$((cross_isa_count + 1))
            local esr="${ISA_RESULTS[$key]}"
            if [ -n "$esr" ] && [ "$esr" != "N/A" ]; then
                if awk -v v="$esr" 'BEGIN { exit (v+0 < 1e-8) ? 0 : 1 }'; then
                    cross_isa_pass=$((cross_isa_pass + 1))
                else
                    all_pass=false
                fi
            fi
        fi
    done
    set -u

    if $all_pass && [ "$cross_isa_count" -gt 0 ]; then
        echo -e "  AVX2 vs AVX-512: ${GREEN}bitwise identical ✅${NC}"
    elif [ "$cross_isa_count" -gt 0 ]; then
        echo -e "  AVX2 vs AVX-512: ${YELLOW}divergent on $((cross_isa_count - cross_isa_pass))/$cross_isa_count models ⚠${NC}"
    else
        echo "  AVX2 vs AVX-512: sem dados (CPU pode nao ter AVX-512)"
    fi

    echo "  Self-consistency checks: $self_consistency_count executados"
    echo ""

    if [ "$cross_isa_count" -gt 0 ]; then
        echo "  Detalhes cross-ISA:"
        set +u
        for key in "${!ISA_RESULTS[@]}"; do
            [[ "$key" == *"self-consistency"* ]] && continue
            local esr="${ISA_RESULTS[$key]}"
            local pass_str="⚠"
            if awk -v v="$esr" 'BEGIN { exit (v+0 < 1e-8) ? 0 : 1 }'; then
                pass_str="✅"
            fi
            printf "    %s  ESR=%s  %s\n" "$key" "$esr" "$pass_str"
        done
        set -u
        echo ""
    fi
}

# ── Render: activation precision ────────────────────────────────────────────

render_activation_precision() {
    echo "🎹 ACTIVATION PRECISION"
    echo "════════════════════════"
    echo ""

    local count
    set +u; count="${#ACTIVATION_SNR[@]}"; set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Nenhum resultado de activation precision disponivel.${NC}"
        echo ""
        return
    fi

    printf "  %-20s │ %-14s │ %-14s │ %s\n" \
        "Modelo" "Fast(Pade)" "Standard(exact)" "Δ SNR"
    printf "  %s │ %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..20})" "$(printf '─%.0s' {1..14})" \
        "$(printf '─%.0s' {1..14})" "$(printf '─%.0s' {1..10})"

    set +u
    for model in "${!ACTIVATION_SNR[@]}"; do
        local data="${ACTIVATION_SNR[$model]}"
        local fast_snr exact_snr delta
        IFS='|' read -r fast_snr exact_snr delta <<< "$data"

        local delta_val="${delta#+}"
        local delta_colored="${delta} dB"
        if awk -v v="$delta_val" 'BEGIN { exit (v+0 < 3.0) ? 0 : 1 }'; then
            delta_colored="${delta} dB"
        else
            delta_colored="${YELLOW}${delta} dB${NC}"
        fi

        printf "  %-20s │ %-14s │ %-14s │ %b\n" \
            "$model" "${fast_snr} dB" "${exact_snr} dB" "$delta_colored"
    done
    set -u
    echo ""

    local total=0 count_num=0
    set +u
    for model in "${!ACTIVATION_SNR[@]}"; do
        local data="${ACTIVATION_SNR[$model]}"
        local delta
        IFS='|' read -r _ _ delta <<< "$data"
        local delta_val="${delta#+}"
        if [ -n "$delta_val" ] && [ "$delta_val" != "N/A" ]; then
            total=$(awk -v t="$total" -v d="$delta_val" 'BEGIN { printf "%.2f", t + d }')
            count_num=$((count_num + 1))
        fi
    done
    set -u
    if [ "$count_num" -gt 0 ]; then
        local avg
        avg=$(awk -v t="$total" -v c="$count_num" 'BEGIN { printf "%.1f", t / c }')
        echo "  Ganho SNR medio com Standard(exact): +${avg} dB (sobre ${count_num} modelos LSTM)"
    fi
    echo ""
}

# ── Render: f64 decomposition ───────────────────────────────────────────────
#
# Methodology note: decomposition tests run the model cold (256 samples without prewarm).
# For architectures with large receptive fields (WaveNet, A2), initial transient fill
# dominates the raw error, so cold-start sum of sources may differ from total.
# Steady-state precision comparison is provided by prewarm-paired tables above.

_decomp_extract() {
    local block="$1" label_pattern="$2"
    LC_ALL=C awk -v pat="$label_pattern" '
    $0 ~ pat {
        val = $0
        sub(".*" pat, "", val)
        sub(/[[:space:]].*/, "", val)
        sub(/\).*/, "", val)
        print val
        exit
    }
    ' <<< "$block" 2>/dev/null
}

render_f64_decomposition() {
    local count
    set +u; count="${#F64_DECOMPOSITION[@]}"; set -u
    if [ -z "$count" ] || [ "$count" -eq 0 ]; then
        return
    fi
    echo "🔍 F64 ORACLE — Decomposicao de Fontes de Erro"
    echo "══════════════════════════════════════════════"
    echo ""
    echo "  (i) Estas medicoes sao cold-start (256 amostras, SEM prewarm) — NAO"
    echo "      comparaveis aos valores 'vs Ideal (f64)' da tabela de fidelidade"
    echo "      acima (medidos com warmup de 24k amostras). Para WaveNet/A2, o"
    echo "      campo receptivo e maior que a janela de 256 amostras, entao o"
    echo "      ESR total abaixo reflete majoritariamente o transiente de"
    echo "      preenchimento do buffer, nao o piso de precisao em regime"
    echo "      permanente. Ver docs/perceptual_validation.md#decomposition-cold-start."
    echo ""
    set +u
    for model in "${!F64_DECOMPOSITION[@]}"; do
        echo "  ${model}:"
        local block="${F64_DECOMPOSITION[$model]}"
        echo "$block" | while IFS= read -r line; do
            [ -n "$line" ] && echo "    $line"
        done || true

        local short_name
        short_name=$(echo "$model" | sed 's/.* \.\.\. //')
        local total combined
        total="${ESR_F64_COLD[$short_name]:-}"
        if [ -z "$total" ]; then
            total=$(_decomp_extract "$block" 'ESR\(f32 vs f64 oracle\):\s*')
        fi
        combined=$(_decomp_extract "$block" 'combined \(F16C\+Padé\+F32\):\s*')
        if [ -n "$total" ] && [ -n "$combined" ]; then
            local ratio_flag
            ratio_flag=$(LC_ALL=C awk -v t="$total" -v c="$combined" 'BEGIN {
                if (c == 0) { print "n/a"; exit }
                r = t / c; if (r < 1) r = 1 / r;
                printf "%.0f", r
            }' 2>/dev/null || echo "n/a")
            if [ "$ratio_flag" != "n/a" ] && [ "$ratio_flag" -gt 10 ] 2>/dev/null; then
                echo -e "    ${YELLOW}⚠ Rule 5 (Σ sources ≈ total, within 10×) violada: total/combinado ≈ ${ratio_flag}×.${NC}"
                echo -e "    ${YELLOW}  Esperado para modelos com campo receptivo > janela de medicao (cold-start).${NC}"
                echo -e "    ${YELLOW}  Nao usar este numero como piso de precisao calibrado sem medicao pareada-com-prewarm.${NC}"
            fi
        fi
        echo ""
    done
    set -u
}

# ── Render: spectral summary ────────────────────────────────────────────────

render_spectral_summary() {
    local count="${SPECTRAL_PASSED_COUNT:-0}"
    echo "📈 SPECTRAL FIDELITY"
    echo "═════════════════════"
    echo ""
    if [ "$count" -gt 0 ]; then
        echo -e "  ${GREEN}ok${NC} ${count} modelo(s) com metricas espectrais dentro da baseline."
    else
        echo -e "  ${YELLOW}(i) Nao coberto no modo quick — rode tests-long para verificacao completa.${NC}"
    fi
    echo ""
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

# ── Test count extraction from phase receipt ────────────────────────────────
# Extracts aggregated test counts (passed/failed/ignored/filtered/skip_capability)
# from the phase receipt JSONL for governance tracking.
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

# ── Render: coverage matrix ─────────────────────────────────────────────────

render_coverage_matrix() {
    echo "📋 MATRIZ DE COBERTURA POR EIXO (Governanca)"
    echo "════════════════════════════════════════════"
    echo ""

    local total_axes=5 covered_axes=0

    printf "  %-28s │ %-10s │ %s\n" "Eixo" "Registros" "Cobertura"
    printf "  %s │ %s │ %s\n" \
        "$(printf '─%.0s' {1..28})" "$(printf '─%.0s' {1..10})" "$(printf '─%.0s' {1..20})"

    local n
    n="${COVERAGE_NAMCORE_PARITY:-0}"
    if [ "$n" -gt 0 ]; then
        printf "  %-28s │ %-10s │ ${GREEN}%-20s${NC}\n" "NAMCore Parity" "$n" "coberto"
        covered_axes=$((covered_axes + 1))
    else
        printf "  %-28s │ %-10s │ ${YELLOW}%-20s${NC}\n" "NAMCore Parity" "$n" "nao coberto"
    fi

    n="${COVERAGE_F64_ORACLE:-0}"
    if [ "$n" -gt 0 ]; then
        printf "  %-28s │ %-10s │ ${GREEN}%-20s${NC}\n" "f64 Oracle Fidelity" "$n" "coberto"
        covered_axes=$((covered_axes + 1))
    else
        printf "  %-28s │ %-10s │ ${YELLOW}%-20s${NC}\n" "f64 Oracle Fidelity" "$n" "nao coberto"
    fi

    n="${COVERAGE_ISA_OPTIMIZATIONS:-0}"
    if [ "$n" -gt 0 ]; then
        printf "  %-28s │ %-10s │ ${GREEN}%-20s${NC}\n" "ISA Optimizations" "$n" "coberto"
        covered_axes=$((covered_axes + 1))
    else
        printf "  %-28s │ %-10s │ ${YELLOW}%-20s${NC}\n" "ISA Optimizations" "$n" "nao coberto"
    fi

    n="${COVERAGE_SPECTRAL_BASELINES:-0}"
    if [ "$n" -gt 0 ]; then
        printf "  %-28s │ %-10s │ ${GREEN}%-20s${NC}\n" "Spectral Baselines" "$n" "coberto"
        covered_axes=$((covered_axes + 1))
    else
        printf "  %-28s │ %-10s │ ${YELLOW}%-20s${NC}\n" "Spectral Baselines" "$n" "nao coberto"
    fi

    n="${COVERAGE_RT_PERFORMANCE:-0}"
    if [ "$n" -gt 0 ]; then
        printf "  %-28s │ %-10s │ ${GREEN}%-20s${NC}\n" "RT Performance" "$n" "coberto"
        covered_axes=$((covered_axes + 1))
    else
        printf "  %-28s │ %-10s │ ${YELLOW}%-20s${NC}\n" "RT Performance" "$n" "nao coberto"
    fi

    echo ""
    echo "  Cobertura: ${covered_axes}/${total_axes} eixos cobertos"
    echo ""

    echo "  Contagens de testes no receipt:"
    echo -e "    passed:          ${GREEN}${TEST_COUNTS[passed]}${NC}"
    echo -e "    failed:          ${RED}${TEST_COUNTS[failed]}${NC}"
    echo -e "    skip_capability: ${YELLOW}${TEST_COUNTS[skip_capability]}${NC}"
    echo -e "    ignored:         ${YELLOW}${TEST_COUNTS[ignored]}${NC}"
    echo -e "    filtered:        ${TEST_COUNTS[filtered]}"
    echo ""
}

render_footer() {
    local total_s
    local end_t
    end_t=$(date +%s%N)
    total_s=$(awk -v ns=$((end_t - OVERALL_START)) 'BEGIN { printf "%.1f", ns / 1000000000 }')
    echo "───────────────────────────────────────────────────────────────"
    echo -e "  Dashboard gerado em ${total_s}s (fidelidade: ${FIDELITY_DURATION_S}s, performance: ${BENCH_DURATION_S}s)"
    echo ""

    local skipped=0
    local order_count="${#MODEL_ORDER[@]}"
    local bench_count="${#ALL_BENCH_NAMES[@]}"
    local phase_failures="${DASHBOARD_PHASE_HAD_FAILURE:-0}"

    if [ "$phase_failures" -ne 0 ]; then
        echo -e "  ${RED}⚠ Uma ou mais fases do dashboard falharam (ver receipt: ${DASHBOARD_PHASE_RECEIPT}).${NC}"
        echo ""
        skipped=1
    fi
    if [ "$order_count" -eq 0 ] && [ "$MODE" != "bench" ]; then
        echo -e "  ${YELLOW}(i) Testes de fidelidade nao produziram dados parseaveis.${NC}"
        echo -e "  ${YELLOW}   Verifique se os modelos e golden vectors estao presentes.${NC}"
        skipped=1
    fi
    if [ "$bench_count" -eq 0 ] && [ "$MODE" != "fidelity" ]; then
        echo -e "  ${YELLOW}(i) Benchmarks nao produziram dados parseaveis.${NC}"
        skipped=1
    fi
    if [ "$skipped" -eq 1 ] && [ "$phase_failures" -eq 0 ]; then
        echo -e "  ${YELLOW}(i) Exit code 0 (graceful skip) — dados incompletos nao sao erros de infra.${NC}"
    fi
    echo ""
}

# ── Full dashboard render ───────────────────────────────────────────────────

render_dashboard() {
    render_header
    render_quick_summary
    render_fidelity_details
    render_performance
    render_isa_parity
    render_activation_precision
    render_f64_decomposition
    render_spectral_summary
    render_coverage_matrix
    render_footer
}

# ── Plain-text version (no ANSI) for --save ─────────────────────────────────

render_dashboard_plain() {
    export IS_SAVING=1
    set +o pipefail
    render_dashboard | sed "s/$(printf '\033')\[[0-9;]*m//g"
    set -o pipefail
    unset IS_SAVING
}

# ── Contract baseline storage ──────────────────────────────────────────────

declare -A CONTRACT_ESR
declare -A CONTRACT_ESR_F64
declare -A CONTRACT_SNR
declare -A CONTRACT_MRSTFT
declare -A CONTRACT_LATENCY

# ── Safety ceiling multipliers (three-tier threshold system) ───────────────
# baseline:   contract value (reference, no penalty if unchanged)
# noise:      baseline * ESR_NOISE_MULT (tolerance for numerical noise between runs)
# safety:     baseline * ESR_SAFETY_MULT (hard ceiling — violation is always a failure)
ESR_NOISE_MULT="10.0"
ESR_SAFETY_MULT="100.0"

# ── Load contract/baseline file ────────────────────────────────────────────
# Parses a plain-text quality contract file into associative arrays.

load_contract_baseline() {
    local file="$1"
    if [ ! -f "$file" ]; then
        if [ -f "docs/$(basename "$file")" ]; then
            file="docs/$(basename "$file")"
        elif [ -f "$(basename "$file")" ]; then
            file="$(basename "$file")"
        else
            echo "ERRO: Arquivo de contrato nao encontrado: ${file}" >&2
            exit 2
        fi
    fi

    local parsed="$PARSEDIR/contract_baseline.parsed"
    LC_ALL=C awk -F'│' '
    BEGIN { section = "" }
    /FIDELIDADE[[:space:]]+SONORA/ { section = "fidelity"; next }
    /PERFORMANCE/ { section = "performance"; next }
    /ACTIVATION|ISA[[:space:]]+PARITY|SPECTRAL[[:space:]]+FIDELITY|F64[[:space:]]+ORACLE/ { section = ""; next }

    section == "fidelity" && NF >= 5 {
        m = $1; gsub(/^[[:space:]]+|[[:space:]]+$/, "", m)
        esr_nam = $2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", esr_nam)
        esr_f64 = $3; gsub(/^[[:space:]]+|[[:space:]]+$/, "", esr_f64)
        snr = $4; gsub(/^[[:space:]]+|[[:space:]]+$/, "", snr)
        mrstft = $5; gsub(/^[[:space:]]+|[[:space:]]+$/, "", mrstft)

        if (m != "" && m !~ /^[─═]/ && m != "Modelo" && m != "Padrao") {
            if (esr_nam ~ /^[0-9.]/) printf "CONTRACT_ESR\t%s\t%s\n", m, esr_nam
            if (esr_f64 ~ /^[0-9.]/ || esr_f64 == "N/A") printf "CONTRACT_ESR_F64\t%s\t%s\n", m, esr_f64
            if (snr ~ /^[0-9.-]/) printf "CONTRACT_SNR\t%s\t%s\n", m, snr
            if (mrstft ~ /^[0-9.]/) printf "CONTRACT_MRSTFT\t%s\t%s\n", m, mrstft
        }
    }
    section == "performance" && NF >= 2 {
        m = $1; gsub(/^[[:space:]]+|[[:space:]]+$/, "", m)
        lat = $2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", lat); sub(/[[:space:]]*us$/, "", lat)

        if (m != "" && m !~ /^[─═]/ && m != "Modelo") {
            if (lat ~ /^[0-9.]/) printf "CONTRACT_LATENCY\t%s\t%s\n", m, lat
        }
    }
    ' "$file" > "$parsed"

    while IFS=$'\t' read -r metric key val; do
        case "$metric" in
            CONTRACT_ESR)      CONTRACT_ESR["$key"]="$val" ;;
            CONTRACT_ESR_F64)  CONTRACT_ESR_F64["$key"]="$val" ;;
            CONTRACT_SNR)      CONTRACT_SNR["$key"]="$val" ;;
            CONTRACT_MRSTFT)   CONTRACT_MRSTFT["$key"]="$val" ;;
            CONTRACT_LATENCY)  CONTRACT_LATENCY["$key"]="$val" ;;
        esac
    done < "$parsed"
}

# ── Contract verification ──────────────────────────────────────────────────
# Three-tier threshold system:
#   baseline:  contract value (reference point)
#   noise:     baseline × ESR_NOISE_MULT (run-to-run numerical tolerance)
#   safety:    baseline × ESR_SAFETY_MULT (hard ceiling, always a failure)
#
# Dual oracle: NAMCore parity and f64 oracle are checked independently.
# When NAMCore passes but f64 fails (or vice versa), REVIEW_REQUIRED is
# signaled — neither oracle wins automatically.
#
# Mandatory phase receipts must all show PASS before threshold analysis.

verify_contract() {
    local fidelity_violations=0
    local perf_violations=0
    local review_required=0

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  VERIFICACAO DE CONTRATO DE QUALIDADE"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""

    if [ ! -f "$DASHBOARD_PHASE_RECEIPT" ]; then
        echo -e "  ${RED}✗${NC} Receipt de fases ausente — nao e possivel verificar o contrato."
        echo ""
        return 1
    fi

    local phase_fail
    phase_fail=$(grep -c '"status":"FAIL"' "$DASHBOARD_PHASE_RECEIPT" 2>/dev/null || true)
    if [ -n "$phase_fail" ] && [ "$phase_fail" -gt 0 ]; then
        echo -e "  ${RED}✗${NC} ${phase_fail} fase(s) do dashboard falharam — ver receipt: ${DASHBOARD_PHASE_RECEIPT}"
        fidelity_violations=$((fidelity_violations + phase_fail))
    fi

    for phase_id in "${!PHASE_MANDATORY[@]}"; do
        local phase_status
        phase_status=$(grep "\"phase_id\":\"${phase_id}\"" "$DASHBOARD_PHASE_RECEIPT" 2>/dev/null | grep -o '"status":"[^"]*"' | cut -d'"' -f4 || echo "NOT_RUN")
        if [ "$phase_status" != "PASS" ]; then
            echo -e "  ${RED}✗ PHASE_FAILED${NC} Fase obrigatoria '${phase_id}': status=${phase_status} (requer PASS)"
            fidelity_violations=$((fidelity_violations + 1))
        fi
    done
    echo ""

    local esr_contract_count lat_contract_count
    set +u; esr_contract_count="${#CONTRACT_ESR[@]}"; lat_contract_count="${#CONTRACT_LATENCY[@]}"; set -u
    if [ -z "$esr_contract_count" ] || [ "$esr_contract_count" -eq 0 ]; then
        if [ -z "$lat_contract_count" ] || [ "$lat_contract_count" -eq 0 ]; then
            echo -e "  ${YELLOW}(i) Arquivo de contrato vazio ou sem metricas reconhecidas.${NC}"
            echo ""
            local total_violations=$((fidelity_violations + perf_violations))
            if [ "$total_violations" -gt 0 ]; then
                echo -e "  ${RED}CONTRATO VIOLADO — ${total_violations} violacao(oes) detectada(s).${NC}"
                echo ""
                return 1
            fi
            return 0
        fi
    fi

    if [ -n "$esr_contract_count" ] && [ "$esr_contract_count" -gt 0 ]; then
        echo "  FIDELIDADE — ${esr_contract_count} modelo(s) no contrato"
        echo "  ─────────────────────────────────────────────"
        echo ""

        set +u
        for contract_label in "${!CONTRACT_ESR[@]}"; do
            local dash_key=""
            if [ -n "${ESR_NAMCORE[$contract_label]:-}" ]; then
                dash_key="$contract_label"
            else
                local contract_short
                contract_short=$(echo "$contract_label" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
                for k in "${!ESR_NAMCORE[@]}"; do
                    local dash_label
                    dash_label=$(echo "$k" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')
                    if [ "$dash_label" = "$contract_short" ]; then
                        dash_key="$k"
                        break
                    fi
                done
            fi

            if [ -n "$dash_key" ]; then
                local namcore_ok=1

                local esr_cur="${ESR_NAMCORE[$dash_key]:-N/A}"
                local esr_ctr="${CONTRACT_ESR[$contract_label]}"

                if [ "$esr_cur" != "N/A" ] && [ "$esr_ctr" != "N/A" ] && [ -n "$esr_ctr" ]; then
                    local esr_cur_fmt
                    esr_cur_fmt=$(_fmt_metric "$esr_cur")
                    local noise_limit safety_limit
                    noise_limit=$(LC_ALL=C awk -v c="$esr_ctr" -v m="$ESR_NOISE_MULT" 'BEGIN { printf "%.2e", c*m }')
                    safety_limit=$(LC_ALL=C awk -v c="$esr_ctr" -v m="$ESR_SAFETY_MULT" 'BEGIN { printf "%.2e", c*m }')

                    local esr_noise_fail esr_safety_fail
                    esr_noise_fail=$(LC_ALL=C awk -v cur="$esr_cur" -v lim="$noise_limit" 'BEGIN { if (cur+0 > lim) print "1"; else print "0" }')
                    esr_safety_fail=$(LC_ALL=C awk -v cur="$esr_cur" -v lim="$safety_limit" 'BEGIN { if (cur+0 > lim) print "1"; else print "0" }')

                    if [ "$esr_safety_fail" = "1" ]; then
                        echo -e "    ${RED}✗ SAFETY CEILING${NC} ${contract_label}: ESR NAMCore ${esr_cur_fmt} > safety ${safety_limit} (baseline: ${esr_ctr})"
                        fidelity_violations=$((fidelity_violations + 1))
                        namcore_ok=0
                    elif [ "$esr_noise_fail" = "1" ]; then
                        echo -e "    ${YELLOW}⚠ NOISE ENVELOPE${NC} ${contract_label}: ESR NAMCore ${esr_cur_fmt} > noise ${noise_limit} (baseline: ${esr_ctr})"
                        fidelity_violations=$((fidelity_violations + 1))
                        namcore_ok=0
                    else
                        echo -e "    ${GREEN}ok${NC} ${contract_label}: ESR NAMCore ${esr_cur_fmt} (baseline: ${esr_ctr})"
                    fi
                fi

                local esr_f64_cur esr_f64_provenance
                { read -r esr_f64_cur; read -r esr_f64_provenance; } < <(_lookup_esr_f64 "$(echo "$contract_label" | sed 's/ @.*//; s/ Live$//; s/ HQ$//')")
                local esr_f64_ctr="${CONTRACT_ESR_F64[$contract_label]:-}"

                if [ -n "$esr_f64_ctr" ] && [ "$esr_f64_ctr" != "N/A" ] && [ "$esr_f64_cur" != "N/A" ] && [ -n "$esr_f64_cur" ]; then
                    local esr_f64_cur_fmt
                    esr_f64_cur_fmt=$(_fmt_metric "$esr_f64_cur")
                    local f64_noise_limit f64_safety_limit
                    f64_noise_limit=$(LC_ALL=C awk -v c="$esr_f64_ctr" -v m="$ESR_NOISE_MULT" 'BEGIN { printf "%.2e", c*m }')
                    f64_safety_limit=$(LC_ALL=C awk -v c="$esr_f64_ctr" -v m="$ESR_SAFETY_MULT" 'BEGIN { printf "%.2e", c*m }')

                    local f64_noise_fail f64_safety_fail
                    f64_noise_fail=$(LC_ALL=C awk -v cur="$esr_f64_cur" -v lim="$f64_noise_limit" 'BEGIN { if (cur+0 > lim) print "1"; else print "0" }')
                    f64_safety_fail=$(LC_ALL=C awk -v cur="$esr_f64_cur" -v lim="$f64_safety_limit" 'BEGIN { if (cur+0 > lim) print "1"; else print "0" }')

                    if [ "$f64_safety_fail" = "1" ]; then
                        echo -e "    ${RED}✗ SAFETY CEILING f64${NC} ${contract_label}: ESR f64 ${esr_f64_cur_fmt} > safety ${f64_safety_limit} (baseline f64: ${esr_f64_ctr})"
                        fidelity_violations=$((fidelity_violations + 1))
                        if [ "$namcore_ok" -eq 1 ]; then
                            echo -e "    ${YELLOW}[REVIEW_REQUIRED]${NC} NAMCore ESR ok, mas f64 viola safety ceiling. Oraculos divergem."
                            review_required=1
                        fi
                    elif [ "$f64_noise_fail" = "1" ]; then
                        echo -e "    ${YELLOW}⚠ NOISE ENVELOPE f64${NC} ${contract_label}: ESR f64 ${esr_f64_cur_fmt} > noise ${f64_noise_limit} (baseline f64: ${esr_f64_ctr})"
                        if [ "$namcore_ok" -eq 1 ]; then
                            echo -e "    ${YELLOW}[REVIEW_REQUIRED]${NC} NAMCore ESR ok, mas f64 degradou alem do noise envelope. Oraculos divergem."
                            review_required=1
                        fi
                        fidelity_violations=$((fidelity_violations + 1))
                    fi
                elif [ -n "$esr_f64_ctr" ] && [ "$esr_f64_ctr" != "N/A" ] && [ "$esr_f64_cur" = "N/A" ]; then
                    echo -e "    ${RED}MISSING${NC} ${contract_label}: ESR f64 nao medido mas presente no contrato (f64 baseline: ${esr_f64_ctr})"
                    fidelity_violations=$((fidelity_violations + 1))
                fi

                local snr_cur="${SNR_DB[$dash_key]:-N/A}"
                local snr_ctr="${CONTRACT_SNR[$contract_label]:-N/A}"
                if [ "$snr_cur" != "N/A" ] && [ "$snr_ctr" != "N/A" ] && [ -n "$snr_ctr" ]; then
                    local snr_cur_fmt="$snr_cur"
                    [ "$snr_cur" != "N/A" ] && snr_cur_fmt=$(_nfmt "%.1f" "$snr_cur")
                    local snr_fail
                    snr_fail=$(LC_ALL=C awk -v cur="$snr_cur" -v ctr="$snr_ctr" \
                        'BEGIN { if (cur+0 < ctr-6.0) print "1"; else print "0" }')
                    if [ "$snr_fail" = "1" ]; then
                        echo -e "    ${RED}✗${NC} ${contract_label}: SNR regrediu ${snr_cur_fmt} dB (contrato: ${snr_ctr} dB, limite: $(LC_ALL=C awk -v c="$snr_ctr" 'BEGIN { printf "%.1f", c-6.0 }') dB)"
                        fidelity_violations=$((fidelity_violations + 1))
                    fi
                fi

                local mrstft_cur="${MRSTFT[$dash_key]:-N/A}"
                local mrstft_ctr="${CONTRACT_MRSTFT[$contract_label]:-N/A}"
                if [ "$mrstft_cur" != "N/A" ] && [ "$mrstft_ctr" != "N/A" ] && [ -n "$mrstft_ctr" ]; then
                    local mrstft_cur_fmt
                    mrstft_cur_fmt=$(_fmt_metric "$mrstft_cur")
                    local mrstft_fail
                    mrstft_fail=$(LC_ALL=C awk -v cur="$mrstft_cur" -v ctr="$mrstft_ctr" \
                        'BEGIN { if (cur+0 > ctr*10.0) print "1"; else print "0" }')
                    if [ "$mrstft_fail" = "1" ]; then
                        echo -e "    ${RED}✗${NC} ${contract_label}: MR-STFT regrediu ${mrstft_cur_fmt} (contrato: ${mrstft_ctr}, limite: $(LC_ALL=C awk -v c="$mrstft_ctr" 'BEGIN { printf "%.4f", c*10.0 }'))"
                        fidelity_violations=$((fidelity_violations + 1))
                    fi
                fi
            else
                local is_optional_nondist=0
                case "$contract_label" in
                    *"EVH-5150-Lite"*) is_optional_nondist=1 ;;
                esac
                if [ "$is_optional_nondist" -eq 1 ]; then
                    echo -e "    ${YELLOW}(i) OPTIONAL_SKIPPED${NC} ${contract_label}: modelo nao-distribuivel ausente no ambiente local (teste ignorado graciosamente)"
                else
                    echo -e "    ${RED}MISSING_LABEL${NC} ${contract_label}: rotulo de contrato obrigatorio nao encontrado na execucao atual"
                    fidelity_violations=$((fidelity_violations + 1))
                fi
            fi
        done
        set -u
        echo ""
    fi

    if [ -n "$lat_contract_count" ] && [ "$lat_contract_count" -gt 0 ]; then
        echo "  PERFORMANCE — ${lat_contract_count} benchmark(s) no contrato"
        echo "  ─────────────────────────────────────────────────"
        echo ""

        set +u
        for contract_label in "${!CONTRACT_LATENCY[@]}"; do
            local matched=false
            for bn in "${ALL_BENCH_NAMES[@]}"; do
                local dash_label="${BENCH_MODEL_MAP[$bn]:-$bn}"
                local dash_norm="${dash_label//×/x}"
                local ctr_norm="${contract_label//×/x}"
                if [ "$dash_norm" = "$ctr_norm" ]; then
                    matched=true
                    local lat_cur="${LATENCY_US[$bn]:-N/A}"
                    local lat_ctr="${CONTRACT_LATENCY[$contract_label]}"

                    if [ "$lat_cur" != "N/A" ] && [ "$lat_ctr" != "N/A" ] && [ -n "$lat_ctr" ]; then
                        local lat_fail
                        lat_fail=$(LC_ALL=C awk -v cur="$lat_cur" -v ctr="$lat_ctr" \
                            'BEGIN { limit = ctr * 1.10; if (limit < ctr + 0.05) limit = ctr + 0.05; if (cur+0 > limit) print "1"; else print "0" }')
                        if [ "$lat_fail" = "1" ]; then
                            echo -e "    ${RED}✗${NC} ${contract_label}: latencia regrediu ${lat_cur} us (contrato: ${lat_ctr} us, limite: $(LC_ALL=C awk -v c="$lat_ctr" 'BEGIN { lim = c * 1.10; if (lim < c + 0.05) lim = c + 0.05; printf "%.1f", lim }') us)"
                            perf_violations=$((perf_violations + 1))
                        else
                            echo -e "    ${GREEN}ok${NC} ${contract_label}: latencia ${lat_cur} us (contrato: ${lat_ctr} us)"
                        fi
                    fi
                    break
                fi
            done
            if [ "$matched" = false ]; then
                echo -e "    ${RED}MISSING_LABEL${NC} ${contract_label}: rotulo de contrato nao encontrado na execucao atual"
                perf_violations=$((perf_violations + 1))
            fi
        done
        set -u
        echo ""
    fi

    if [ "$fidelity_violations" -gt 0 ]; then
        echo -e "  ${RED}FIDELIDADE: FAIL (${fidelity_violations} violacao(oes))${NC}"
        if [ "$review_required" -gt 0 ]; then
            echo -e "  ${YELLOW}[GOVERNANCA] REVIEW_REQUIRED — Divergencia NAMCore vs f64 detectada em modelo(s).${NC}"
            echo -e "  ${YELLOW}              Nenhum oraculo vence automaticamente. Investigar divergencia.${NC}"
        fi
        if [ "$perf_violations" -gt 0 ]; then
            echo -e "  ${RED}PERFORMANCE: FAIL (${perf_violations})${NC}"
        fi
        echo -e "  ${RED}CONTRATO VIOLADO${NC}"
        echo ""
        return 1
    fi

    if [ "$perf_violations" -gt 0 ]; then
        echo -e "  ${GREEN}FIDELIDADE: OK${NC}"
        echo -e "  ${RED}PERFORMANCE: FAIL (${perf_violations})${NC}"
        echo ""
        return 1
    fi

    if [ "$review_required" -gt 0 ]; then
        echo -e "  ${YELLOW}CONTRATO EM REVISAO — metricas numericas ok, mas divergencia oracular requer investigacao.${NC}"
        echo -e "  ${YELLOW}                     Nenhum oraculo vence automaticamente.${NC}"
        echo ""
        return 1
    fi

    echo -e "  ${GREEN}FIDELIDADE: OK${NC}"
    echo -e "  ${GREEN}PERFORMANCE: OK${NC}"
    echo ""
    return 0
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    local run_phases=0
    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
        run_phases=$((run_phases + 6))
    fi
    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "bench" ]; then
        run_phases=$((run_phases + 1))
    fi
    PHASE_TOTAL=$((run_phases + 2))

    echo -e "${BLUE}${BOLD}===============================================================${NC}"
    echo -e "${BLUE}${BOLD}    NeuralAmpModeler-rs Quality Dashboard${NC}"
    echo -e "${BLUE}${BOLD}    Modo: ${MODE}${NC}"
    echo -e "${BLUE}${BOLD}===============================================================${NC}"

    if [ "$MODE" = "standard" ] || [ "$MODE" = "full" ] || [ "$MODE" = "fidelity" ]; then
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
    parse_activation_precision
    parse_benchmarks

    phase "Renderizando dashboard"
    compute_coverage
    extract_test_counts
    render_dashboard

    if [ -n "$SAVE_FILE" ]; then
        render_dashboard_plain > "$SAVE_FILE"
        echo -e "${GREEN}ok${NC} Dashboard salvo em: ${SAVE_FILE} (plain text, sem ANSI)"
    fi

    local final_exit=0
    if [ "${DASHBOARD_PHASE_HAD_FAILURE:-0}" -ne 0 ]; then
        final_exit=1
    fi

    if [ -n "$CHECK_FILE" ]; then
        load_contract_baseline "$CHECK_FILE"
        if ! verify_contract; then
            final_exit=1
        fi
    fi

    exit "$final_exit"
}

main "$@"
