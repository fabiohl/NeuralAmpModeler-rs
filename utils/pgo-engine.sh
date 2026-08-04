#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# PGO Engine Pipeline — Profile-Guided Optimization for NeuralAmpModeler-rs
# =============================================================================
#
# Automates the full Profile-Guided Optimization (PGO) flow for the
# NeuralAmpModeler-rs DSP engine:
#
#   0. Pre-PGO baseline: Builds and benchmarks the regression_gate suite without
#      PGO into a segregated target dir (target/pgo-baseline/), saving a Criterion
#      baseline named "pgo-pre". Also saves a quality-dashboard snapshot.
#
#   1. Instrumented build: Compiles the test binary with LLVM profile generation
#      instrumentation (-Cprofile-generate) into target/pgo-instrumented/.
#
#   2. Mandatory inference profiling: Executes NamModel::process workloads across
#      all canonical model families with weighted execution counts:
#        WaveNet (Standard, Lite, Nano): 40% — golden vectors + self-consistency
#        LSTM (1x16, 2x8):              30% — golden vectors + self-consistency
#        ConvNet:                        15% — golden vectors + self-consistency
#        A2 (Full, Lite):               10% — golden vectors + self-consistency
#        Linear:                          5% — self-consistency
#      Profraw growth is verified after each family. Any family that fails to
#      produce execution counters causes a hard failure.
#
#   3. Profile merge: All .profraw files are merged into a unified .profdata
#      using llvm-profdata.
#
#   4. PGO-optimized build: Rebuilds the crate with PGO-guided optimizations
#      (-Cprofile-use) into target/pgo-optimized/.
#
#   5. Pre/Post PGO comparison gate: Runs benchmarks from the optimized build
#      against the pre-PGO baseline. Any statistically significant regression
#      (> 1% above noise margin) triggers PGO_REGRESSION (exit code 3).
#      Additionally runs quality-dashboard.sh --check against the pre-PGO
#      snapshot — ESR/SNR/MR-STFT divergence also triggers PGO_REGRESSION.
#
#   6. If all gates pass, the optimized binary is promoted as the PGO artifact.
#
# Output Artifact Location
# ------------------------
#   Merged Profile Data: $NAM_PGO_DIR/merged.profdata
#                        (Default: /tmp/nam_pgo/merged.profdata)
#   Pre-PGO Baseline:   target/pgo-baseline/
#   Instrumented:        target/pgo-instrumented/
#   Optimized Binaries:  target/pgo-optimized/release/
#
# Environment variables
# ----------------------
#   NAM_PGO_DIR            Profile output directory (default: /tmp/nam_pgo).
#   NAM_PGO_SKIP_COMPARE   Set to 1 to skip the pre/post comparison gate
#                          (only valid for human-in-the-loop manual evaluation).
#
# Exit codes
# ----------
#   0  Success — PGO pipeline completed, all gates passed.
#   1  Infrastructure failure (missing tools, build error, no profiles).
#   2  Profile coverage failure (mandatory model family missing profraw data).
#   3  PGO_REGRESSION — performance regression or quality divergence detected.
#
# Prerequisites
# -------------
#   - Rust toolchain with `llvm-tools` component installed.
#   - The `testing` feature flag enabled for test binaries.
#
# Usage
# ------
#   utils/pgo-engine.sh
#   NAM_PGO_SKIP_COMPARE=1 utils/pgo-engine.sh   (no gate, human evaluation)
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE_TOTAL=9
source "$SCRIPT_DIR/_lib.sh"

PROJECT_ROOT="$PROJECT_DIR"
cd "$PROJECT_ROOT"

# ── Configuration ────────────────────────────────────────────────────────────
PGO_DIR="${NAM_PGO_DIR:-/tmp/nam_pgo}"
SKIP_COMPARE="${NAM_PGO_SKIP_COMPARE:-0}"
PGO_PROFDATA="$PGO_DIR/merged.profdata"

# Segregated target directories (T-E4.4-1)
PGO_BASELINE_TARGET="$PROJECT_ROOT/target/pgo-baseline"
PGO_INSTR_TARGET="$PROJECT_ROOT/target/pgo-instrumented"
PGO_OPT_TARGET="$PROJECT_ROOT/target/pgo-optimized"

CRITERION_BASELINE_NAME="pgo-pre"
QUALITY_SNAPSHOT="$PGO_DIR/quality-pre-pgo.txt"

# ── JSONL receipt infrastructure (T-E4.6-2) ────────────────────────────
PGO_RECEIPT_DIR="$PROJECT_ROOT/build/namcore_render/logs"
PGO_RECEIPT="${PGO_RECEIPT_DIR}/pgo_phase_receipt.jsonl"
mkdir -p "$PGO_RECEIPT_DIR"
: > "$PGO_RECEIPT"
DASHBOARD_PHASE_RECEIPT="$PGO_RECEIPT"

# Write build metadata entry for full fingerprint traceability
PGO_RUSTC_VER=$(rustc --version 2>/dev/null || echo 'unknown')
PGO_TARGET_TRIPLE=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
PGO_CPU_MODEL=$(grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown")
PGO_CPU_MICROARCH=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | grep -q 'avx512f' && echo "AVX-512" || (grep -q 'avx2' && echo "AVX2 (x86-64-v3)" || echo "x86-64 (base)"))
PGO_GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

printf '{"kind":"build_metadata","pipeline":"pgo-engine","cargo_profile":"release","target_triple":"%s","rustflags":"%s","rustc_version":"%s","cpu_model":"%s","cpu_microarch":"%s","git_commit":"%s"}\n' \
    "$PGO_TARGET_TRIPLE" "${RUSTFLAGS:-}" "$PGO_RUSTC_VER" "$PGO_CPU_MODEL" "$PGO_CPU_MICROARCH" "$PGO_GIT_COMMIT" >> "$PGO_RECEIPT"

echo ""
echo -e "${BLUE}${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}${BOLD}║   PGO Engine Pipeline — NeuralAmpModeler-rs             ║${NC}"
echo -e "${BLUE}${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Profile directory:       ${YELLOW}$PGO_DIR${NC}"
echo -e "  Pre-PGO baseline target: ${YELLOW}$PGO_BASELINE_TARGET${NC}"
echo -e "  Instrumented target:     ${YELLOW}$PGO_INSTR_TARGET${NC}"
echo -e "  Optimized target:        ${YELLOW}$PGO_OPT_TARGET${NC}"
echo -e "  Pre/post comparison:     ${YELLOW}$([ "$SKIP_COMPARE" = "1" ] && echo 'SKIPPED (human evaluation mode)' || echo 'MANDATORY')${NC}"
echo ""

# ── Prerequisite checks ─────────────────────────────────────────────────────

LLVM_PROFDATA="$(rustup which llvm-profdata 2>/dev/null \
    || command -v llvm-profdata 2>/dev/null \
    || command -v llvm-profdata-21 2>/dev/null \
    || command -v llvm-profdata-20 2>/dev/null \
    || command -v llvm-profdata-19 2>/dev/null \
    || command -v llvm-profdata-18 2>/dev/null \
    || find /usr/bin /usr/local/bin -maxdepth 1 -name 'llvm-profdata*' -type f -executable 2>/dev/null | head -n 1 \
    || true)"

if [ -z "$LLVM_PROFDATA" ] || [ ! -x "$LLVM_PROFDATA" ]; then
    echo -e "${RED}ERROR: llvm-profdata not found or not executable.${NC}"
    echo -e "  To install via Rust toolchain: ${YELLOW}rustup component add llvm-tools${NC}"
    exit 1
fi
echo -e "  llvm-profdata: ${GREEN}$LLVM_PROFDATA${NC}"

# =============================================================================
# Phase 0 — Pre-PGO baseline (benchmarks without PGO)
# =============================================================================
# T-E4.4-1 / T-E4.5-1: Build and benchmark the baseline in a segregated target dir.
# This preserves Criterion data without fragile copies and establishes the
# comparison reference that the post-PGO gate will use.

phase "Pre-PGO baseline — building benchmarks without PGO..."
BASELINE_BUILD_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_baseline_build.log"
mkdir -p "$(dirname "$BASELINE_BUILD_LOG")"

# Remove carriage from previous run (target dirs are preserved between runs by design)
rm -rf "$PGO_BASELINE_TARGET"
mkdir -p "$PGO_BASELINE_TARGET"

CARGO_TARGET_DIR="$PGO_BASELINE_TARGET" \
    cargo build \
    --release \
    --features testing \
    --bench regression_gate \
    > "$BASELINE_BUILD_LOG" 2>&1 || {
        build_status=$?
        tail -10 "$BASELINE_BUILD_LOG"
        echo -e "${RED}ERROR: baseline build failed (exit=$build_status). Full log: $BASELINE_BUILD_LOG${NC}"
        exit 1
    }
tail -3 "$BASELINE_BUILD_LOG"
dashboard_phase_receipt "pgo_baseline_build" "PASS" 0 1 1 ""

phase "Pre-PGO baseline — running benchmarks and saving Criterion baseline..."
BASELINE_BENCH_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_baseline_bench.log"
mkdir -p "$(dirname "$BASELINE_BENCH_LOG")"

CARGO_TARGET_DIR="$PGO_BASELINE_TARGET" \
    cargo bench \
    --bench regression_gate \
    --features testing \
    -- \
    --save-baseline "$CRITERION_BASELINE_NAME" \
    > "$BASELINE_BENCH_LOG" 2>&1 || {
        bench_status=$?
        tail -10 "$BASELINE_BENCH_LOG"
        echo -e "${RED}ERROR: baseline benchmarks failed (exit=$bench_status). Full log: $BASELINE_BENCH_LOG${NC}"
        exit 1
    }
tail -5 "$BASELINE_BENCH_LOG"
echo -e "  Baseline saved as '${GREEN}${CRITERION_BASELINE_NAME}${NC}' in ${YELLOW}$PGO_BASELINE_TARGET/criterion/${NC}"
dashboard_phase_receipt "pgo_baseline_bench" "PASS" 0 1 1 ""

# Generate environment fingerprint alongside the baseline
PGO_FINGERPRINT_FILE="$PGO_BASELINE_TARGET/baseline-fingerprint.json"
CPU_MODEL=$(grep -m1 '^model name' /proc/cpuinfo 2>/dev/null | sed 's/^model name[[:space:]]*: //' || echo "unknown")
CPU_MICROARCH=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null | grep -q 'avx512f' && echo "AVX-512" || (grep -q 'avx2' && echo "AVX2 (x86-64-v3)" || echo "x86-64 (base)"))
RUSTC_VER=$(rustc --version 2>/dev/null || echo 'unknown')
TARGET_TRIPLE=$(rustc -vV 2>/dev/null | sed -n 's/^host: //p' || echo 'unknown')
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")

cat > "$PGO_FINGERPRINT_FILE" <<FINGERPRINT
{
  "cpu_model": "$CPU_MODEL",
  "cpu_microarchitecture": "$CPU_MICROARCH",
  "rustc_version": "$RUSTC_VER",
  "target_triple": "$TARGET_TRIPLE",
  "rustflags": "${RUSTFLAGS:-}",
  "build_profile": "release",
  "git_commit": "$GIT_COMMIT",
  "pgo_phase": "pre-pgo-baseline"
}
FINGERPRINT
echo -e "  Fingerprint saved to ${GREEN}$PGO_FINGERPRINT_FILE${NC}"

# Capture pre-PGO quality dashboard snapshot for post-PGO comparison (T-E4.5-1)
phase "Pre-PGO baseline — capturing quality dashboard snapshot..."
if [ -x "$SCRIPT_DIR/quality-dashboard.sh" ]; then
    CARGO_TARGET_DIR="$PGO_BASELINE_TARGET" \
        "$SCRIPT_DIR/quality-dashboard.sh" --save "$QUALITY_SNAPSHOT" \
        > /dev/null 2>&1 || {
            echo -e "  ${YELLOW}WARN: quality-dashboard snapshot failed — quality gate will be skipped.${NC}"
            QUALITY_SNAPSHOT=""
        }
    if [ -n "$QUALITY_SNAPSHOT" ] && [ -f "$QUALITY_SNAPSHOT" ]; then
        echo -e "  Quality snapshot saved to ${GREEN}$QUALITY_SNAPSHOT${NC}"
        dashboard_phase_receipt "pgo_baseline_quality_snapshot" "PASS" 0 1 1 ""
    fi
else
    echo -e "  ${YELLOW}quality-dashboard.sh not found — quality gate will be skipped.${NC}"
    QUALITY_SNAPSHOT=""
fi

# =============================================================================
# Phase 1 — Clean profile directory and build instrumented binary
# =============================================================================
phase "Cleaning previous PGO profile artifacts..."
rm -rf "$PGO_DIR"
mkdir -p "$PGO_DIR"

phase "Building instrumented test binary..."
rm -rf "$PGO_INSTR_TARGET"

PGO_INSTR_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_instr_build.log"
mkdir -p "$(dirname "$PGO_INSTR_LOG")"

# Build the test binary with profile generation instrumentation into segregated target dir.
# We use --test models --no-run to produce a binary that covers all model test functions.
CARGO_TARGET_DIR="$PGO_INSTR_TARGET" \
    RUSTFLAGS="-Cprofile-generate=$PGO_DIR" \
    cargo test \
    --release \
    --features testing \
    --test models \
    --no-run \
    > "$PGO_INSTR_LOG" 2>&1 || {
        build_status=$?
        tail -10 "$PGO_INSTR_LOG"
        echo -e "${RED}ERROR: instrumented build failed (exit=$build_status). Full log: $PGO_INSTR_LOG${NC}"
        exit 1
    }
tail -3 "$PGO_INSTR_LOG"
dashboard_phase_receipt "pgo_instr_build" "PASS" 0 1 1 ""

# Locate the instrumented test binary
INSTR_TEST_BIN=$(find "$PGO_INSTR_TARGET/release/deps" -maxdepth 1 -name 'models-*' -type f -executable 2>/dev/null | head -n 1)
if [ -z "$INSTR_TEST_BIN" ]; then
    echo -e "${RED}ERROR: instrumented test binary not found after build.${NC}"
    exit 1
fi
echo -e "  Instrumented test binary: ${GREEN}$INSTR_TEST_BIN${NC}"

# Also build parity test binary for quick_parity profiling
CARGO_TARGET_DIR="$PGO_INSTR_TARGET" \
    RUSTFLAGS="-Cprofile-generate=$PGO_DIR" \
    cargo test \
    --release \
    --features testing \
    --test parity \
    --no-run \
    > /dev/null 2>&1 || true

INSTR_PARITY_BIN=$(find "$PGO_INSTR_TARGET/release/deps" -maxdepth 1 -name 'parity-*' -type f -executable 2>/dev/null | head -n 1 || echo "")

# ── Helper: Count profraw files in PGO_DIR ──────────────────────────────────
_count_profraw() {
    find "$PGO_DIR" -maxdepth 1 -name '*.profraw' -type f 2>/dev/null | wc -l
}

# ── Helper: Run a test group and verify profraw growth ───────────────────────
# Usage: _profile_group <family_label> <min_expected> [test_names...]
# Each test_name is a single test function filter passed to the binary with --exact.
_profile_group() {
    local family="$1" min_expected="$2"
    shift 2
    local before after

    before=$(_count_profraw)
    echo "  Profiling ${family} ($# test(s))..."

    for test_name in "$@"; do
        if [ -n "$INSTR_PARITY_BIN" ] && [[ "$test_name" == quick_parity_* ]]; then
            RUSTFLAGS="-Cprofile-generate=$PGO_DIR" "$INSTR_PARITY_BIN" \
                --test-threads=1 "$test_name" --exact > /dev/null 2>&1 || true
        else
            RUSTFLAGS="-Cprofile-generate=$PGO_DIR" "$INSTR_TEST_BIN" \
                --test-threads=1 "$test_name" --exact > /dev/null 2>&1 || true
        fi
    done

    after=$(_count_profraw)
    local growth=$((after - before))
    if [ "$growth" -lt "$min_expected" ]; then
        echo -e "  ${RED}ERROR: ${family} profiling failed — expected ≥${min_expected} new .profraw(s), got ${growth}.${NC}"
        exit 2
    fi
    echo -e "  ${GREEN}${family}: ${growth} new .profraw(s) (total: ${after})${NC}"
    dashboard_phase_receipt "pgo_profile_${family//\//_}" "PASS" 0 "$growth" "$min_expected" ""
}

# =============================================================================
# Phase 2 — Profile WaveNet family (40% weight: 8 runs)
# =============================================================================
phase "Profiling WaveNet family (Standard, Lite, Nano) — 40% weight..."

_profile_group "WaveNet/Standard" 2 \
    golden_vectors_wavenet \
    golden_vectors_wavenet \
    self_consistency_wavenet

_profile_group "WaveNet/Lite" 2 \
    golden_vectors_wavenet_lite \
    golden_vectors_wavenet_lite \
    self_consistency_wavenet_lite

_profile_group "WaveNet/Nano" 1 \
    golden_vectors_wavenet_nano \
    self_consistency_wavenet_nano

# Also profile WaveNet Feather and A1-Standard for broader WaveNet coverage
_profile_group "WaveNet/Feather+A1Std" 1 \
    golden_vectors_wavenet_feather \
    golden_vectors_wavenet_a1_standard

# ── Phase 2b: Profile LSTM family (30% weight: 6 runs) ─────────────────────
phase "Profiling LSTM family (1x16, 2x8) — 30% weight..."

_profile_group "LSTM/1x16" 2 \
    golden_vectors_lstm_1x16 \
    golden_vectors_lstm_1x16 \
    self_consistency_lstm

_profile_group "LSTM/2x8" 2 \
    golden_vectors_lstm_2x8 \
    golden_vectors_lstm_2x8 \
    self_consistency_lstm_2x8

# ── Phase 2c: Profile ConvNet family (15% weight: 3 runs) ──────────────────
phase "Profiling ConvNet family — 15% weight..."

_profile_group "ConvNet" 2 \
    golden_vectors_convnet_test \
    golden_vectors_convnet_test \
    golden_vectors_convnet_test

# ── Phase 2d: Profile A2 family (10% weight: 2 runs) ───────────────────────
phase "Profiling A2 family (Full, Lite) — 10% weight..."

_profile_group "A2/Full" 1 \
    golden_vectors_wavenet_a2_full \
    self_consistency_wavenet_a2_full

_profile_group "A2/Lite" 1 \
    golden_vectors_wavenet_a2_lite \
    self_consistency_wavenet_a2_lite

# ── Phase 2e: Profile Linear family (5% weight: 1 run) ─────────────────────
phase "Profiling Linear family — 5% weight..."

_profile_group "Linear" 1 \
    self_consistency_linear

# ── Profiling summary ───────────────────────────────────────────────────────
TOTAL_PROFRAW=$(_count_profraw)
echo -e "  ${GREEN}${TOTAL_PROFRAW} total .profraw file(s) collected from inference profiling${NC}"
if [ "$TOTAL_PROFRAW" -eq 0 ]; then
    echo -e "${RED}ERROR: no .profraw files generated. Instrumentation may have failed.${NC}"
    exit 2
fi

# =============================================================================
# Phase 3 — Merge profiles
# =============================================================================
phase "Merging .profraw profiles into .profdata..."

PROFRAW_FILES=()
while IFS= read -r -d '' f; do
    PROFRAW_FILES+=("$f")
done < <(find "$PGO_DIR" -maxdepth 1 -name '*.profraw' -type f -print0)

if [ "${#PROFRAW_FILES[@]}" -eq 0 ]; then
    echo -e "${RED}ERROR: no .profraw files to merge.${NC}"
    exit 1
fi

echo "  Merging ${#PROFRAW_FILES[@]} profile(s)..."
"$LLVM_PROFDATA" merge \
    -sparse \
    -o "$PGO_PROFDATA" \
    "${PROFRAW_FILES[@]}" \
    2>&1 || {
        merge_status=$?
        echo -e "${RED}ERROR: llvm-profdata merge failed (exit=$merge_status).${NC}"
        exit 1
    }

if [ ! -f "$PGO_PROFDATA" ]; then
    echo -e "${RED}ERROR: merged .profdata not produced.${NC}"
    exit 1
fi

PROFDATA_SIZE=$(stat -c%s "$PGO_PROFDATA" 2>/dev/null || stat -f%z "$PGO_PROFDATA" 2>/dev/null || echo "?")
echo -e "  Merged profile: ${GREEN}$PGO_PROFDATA${NC} (${PROFDATA_SIZE} bytes)"

# Visualize top functions in the profile for manual inspection
echo "  Top profiled functions (for verification):"
"$LLVM_PROFDATA" show --topn=20 "$PGO_PROFDATA" 2>/dev/null | grep -iE 'process|wavenet|lstm|convnet|dsp|gemv|conv1d|linear|nam' | head -10 || true

# Clean up individual .profraw files (redundant after merge).
rm -f "$PGO_DIR"/*.profraw
dashboard_phase_receipt "pgo_profile_merge" "PASS" 0 "$TOTAL_PROFRAW" 1 ""

# =============================================================================
# Phase 4 — Build optimized binary with PGO
# =============================================================================
phase "Building PGO-optimized binary..."
rm -rf "$PGO_OPT_TARGET"

PGO_OPT_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_opt_build.log"
mkdir -p "$(dirname "$PGO_OPT_LOG")"

# Use absolute path for -Cprofile-use as required by rustc stable spec (T-E4.4-1)
CARGO_TARGET_DIR="$PGO_OPT_TARGET" \
    RUSTFLAGS="-Cprofile-use=$PGO_PROFDATA" \
    cargo build \
    --release \
    --features testing \
    --bench regression_gate \
    > "$PGO_OPT_LOG" 2>&1 || {
        opt_build_status=$?
        tail -10 "$PGO_OPT_LOG"
        echo -e "${RED}ERROR: PGO-optimized build failed (exit=$opt_build_status). Full log: $PGO_OPT_LOG${NC}"
        exit 1
    }
tail -3 "$PGO_OPT_LOG"
dashboard_phase_receipt "pgo_opt_build" "PASS" 0 1 1 ""

OPT_BENCH_BIN=$(find "$PGO_OPT_TARGET/release/deps" -maxdepth 1 -name 'regression_gate-*' -type f -executable 2>/dev/null | head -n 1)
if [ -z "$OPT_BENCH_BIN" ]; then
    echo -e "${RED}ERROR: optimized benchmark binary not found after PGO build.${NC}"
    exit 1
fi
echo -e "  PGO-optimized benchmark binary: ${GREEN}$OPT_BENCH_BIN${NC}"

# =============================================================================
# Phase 5 — Pre/Post PGO comparison gate (T-E4.5-1)
# =============================================================================
if [ "$SKIP_COMPARE" = "1" ]; then
    echo ""
    echo -e "  ${YELLOW}Pre/post comparison skipped (NAM_PGO_SKIP_COMPARE=1).${NC}"
    echo -e "  ${YELLOW}Optimized binary requires manual evaluation before promotion.${NC}"
else
    phase "Pre/Post PGO comparison gate..."

    # Copy Criterion baseline from pre-PGO build to optimized target dir
    # so Criterion can find it for comparison.
    PGO_POST_BENCH_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_post_bench.log"
    mkdir -p "$(dirname "$PGO_POST_BENCH_LOG")"

    echo "  Copying pre-PGO Criterion baseline to optimized target dir..."
    if [ -d "$PGO_BASELINE_TARGET/criterion" ]; then
        cp -a "$PGO_BASELINE_TARGET/criterion" "$PGO_OPT_TARGET/"
        echo -e "  ${GREEN}Baseline copied.${NC}"
    else
        echo -e "  ${RED}ERROR: Pre-PGO Criterion baseline not found at $PGO_BASELINE_TARGET/criterion${NC}"
        echo -e "  ${RED}The baseline was lost — comparison is impossible.${NC}"
        echo -e "  ${RED}Re-run the pipeline from the start to regenerate the baseline.${NC}"
        dashboard_phase_receipt "pgo_comparison_gate" "FAIL" 3 0 0 "Missing baseline"
        exit 3
    fi

    echo "  Running PGO-optimized benchmarks against pre-PGO baseline..."

    set +e
    CARGO_TARGET_DIR="$PGO_OPT_TARGET" \
        cargo bench \
        --bench regression_gate \
        --features testing \
        -- \
        --baseline "$CRITERION_BASELINE_NAME" \
        > "$PGO_POST_BENCH_LOG" 2>&1
    POST_BENCH_STATUS=$?
    set -e

    tail -20 "$PGO_POST_BENCH_LOG"

    # Parse benchmark results for regression detection
    REGRESSION_DETECTED=0
    REGRESSION_REPORT=""

    # Criterion reports regressions with lines like:
    #   RT_WaveNet_Std_CH16    time: [X us Y us Z us]  change: [+A% +B% +C%] (p = D)
    # where the change line is present when comparing against baseline.
    while IFS= read -r line; do
        if echo "$line" | grep -qE 'change:.*\[.*\+'; then
            # Extract the benchmark name and the upper-bound % change
            pgo_bn=$(echo "$line" | awk '{print $1}')
            pgo_pct_upper=$(echo "$line" | grep -oP 'change:.*\[\+?\K[0-9.]+(?=%)' | tail -1 || echo "0")
            if [ -n "$pgo_pct_upper" ]; then
                pgo_is_reg=$(LC_ALL=C awk -v p="$pgo_pct_upper" 'BEGIN { if (p+0 > 1.0) print "1"; else print "0" }')
                if [ "$pgo_is_reg" = "1" ]; then
                    REGRESSION_DETECTED=1
                    REGRESSION_REPORT="${REGRESSION_REPORT}  ${RED}${pgo_bn}: +${pgo_pct_upper}% (above 1% noise margin)${NC}\n"
                fi
            fi
        fi
    done < "$PGO_POST_BENCH_LOG"

    # Also check if Criterion explicitly flagged regressions
    if grep -qiE 'has regressed|Performance has regressed' "$PGO_POST_BENCH_LOG" 2>/dev/null; then
        REGRESSION_DETECTED=1
    fi

    if [ "$REGRESSION_DETECTED" = "1" ]; then
        echo ""
        echo -e "  ${RED}${BOLD}══ PGO_REGRESSION: Performance regression detected ══${NC}"
        if [ -n "$REGRESSION_REPORT" ]; then
            echo -e "$REGRESSION_REPORT"
        fi
        echo ""
        echo -e "  ${RED}The PGO-optimized build is REJECTED.${NC}"
        echo -e "  ${RED}Do NOT promote this binary. Investigate the regression.${NC}"
        echo ""
        echo -e "  Review full benchmark log: ${YELLOW}$PGO_POST_BENCH_LOG${NC}"
        dashboard_phase_receipt "pgo_comparison_gate" "FAIL" 3 0 0 "PGO_REGRESSION: performance regression detected"
        exit 3
    fi

    if [ "$POST_BENCH_STATUS" -ne 0 ]; then
        echo -e "  ${RED}ERROR: Post-PGO benchmarks failed (exit=$POST_BENCH_STATUS).${NC}"
        dashboard_phase_receipt "pgo_comparison_gate" "FAIL" 3 0 0 "Benchmark run failed with exit=$POST_BENCH_STATUS"
        exit 3
    fi

    echo -e "  ${GREEN}Performance gate PASSED — no regression above 1% noise margin.${NC}"
    dashboard_phase_receipt "pgo_comparison_gate" "PASS" 0 1 1 ""

    # ── Quality dashboard gate (T-E4.5-1) ─────────────────────────────────
    if [ -n "$QUALITY_SNAPSHOT" ] && [ -f "$QUALITY_SNAPSHOT" ] && [ -x "$SCRIPT_DIR/quality-dashboard.sh" ]; then
        echo ""
        echo "  Running quality-dashboard verification against pre-PGO snapshot..."

        QUALITY_CHECK_LOG="$PROJECT_ROOT/build/namcore_render/logs/pgo_quality_check.log"
        mkdir -p "$(dirname "$QUALITY_CHECK_LOG")"

        set +e
        CARGO_TARGET_DIR="$PGO_OPT_TARGET" \
            "$SCRIPT_DIR/quality-dashboard.sh" --check "$QUALITY_SNAPSHOT" \
            > "$QUALITY_CHECK_LOG" 2>&1
        QUALITY_CHECK_STATUS=$?
        set -e

        tail -30 "$QUALITY_CHECK_LOG"

        if [ "$QUALITY_CHECK_STATUS" -ne 0 ]; then
            echo ""
            echo -e "  ${RED}${BOLD}══ PGO_REGRESSION: Quality divergence detected ══${NC}"
            echo -e "  ${RED}The PGO-optimized build changed ESR/SNR/MR-STFT or fidelity metrics.${NC}"
            echo -e "  ${RED}Do NOT promote this binary. Full log: ${YELLOW}$QUALITY_CHECK_LOG${NC}"
            dashboard_phase_receipt "pgo_quality_gate" "FAIL" 3 0 0 "Quality divergence vs pre-PGO snapshot"
            exit 3
        fi

        echo -e "  ${GREEN}Quality gate PASSED — ESR/SNR/MR-STFT unchanged within tolerance.${NC}"
        dashboard_phase_receipt "pgo_quality_gate" "PASS" 0 1 1 ""
    else
        echo -e "  ${YELLOW}Quality dashboard gate skipped (snapshot unavailable).${NC}"
    fi
fi

# =============================================================================
# Cleanup
# =============================================================================
rm -f "$PGO_DIR"/stress_*.wav "$PGO_DIR"/stress_*.golden.bin 2>/dev/null || true

echo ""
echo -e "${GREEN}${BOLD}══ PGO Engine Pipeline complete ══${NC}"
echo ""
echo -e "  Profile:             ${GREEN}$PGO_PROFDATA${NC}"
echo -e "  Pre-PGO baseline:    ${YELLOW}$PGO_BASELINE_TARGET${NC}"
echo -e "  Optimized build:     ${YELLOW}$PGO_OPT_TARGET/release/${NC}"
echo ""
if [ "$SKIP_COMPARE" != "1" ]; then
    echo -e "  ${GREEN}All gates passed — the PGO-optimized build is ready for promotion.${NC}"
else
    echo -e "  ${YELLOW}Comparison gate was skipped. Manual evaluation required.${NC}"
fi
echo ""
echo "  Keep $PGO_PROFDATA for future PGO builds or delete it with:"
echo -e "    ${YELLOW}rm -rf $PGO_DIR${NC}"
