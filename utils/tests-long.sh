#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Nightly/pre-release audit suite — the final, extreme bug hunter of NeuralAmpModeler-rs.
# Runs the full advanced QA surface that `utils/lints.sh` (static analysis)
# and `utils/tests-quick.sh` (agile first line) deliberately leave out:
# numerical soak/endurance, full proptest/fuzz case counts, full C++
# NeuralAmpModelerCore parity matrix (multi-SR), cross-ISA determinism,
# RT-safety heap-audits, and the RT deadline/jitter gate.
# Everything measuring floats runs `--release` — the codegen path that
# ships to the end user (see docs/testing.md §2, Axis B).
#
# Non-duplication contract (docs/testing.md §2, §4):
#   - Does NOT repeat `lints.sh` (fmt/check/clippy) or `tests-quick.sh`
#     (structural debug tests, the 5 measurement oracles, capped parser
#     fuzzing). It only runs what those two intentionally leave `#[ignore]`d
#     or out of scope. Every phase below cross-references its quick-suite
#     counterpart so scope drift is visible at a glance.
#   - Does NOT repeat `utils/tests-performance-regression.sh` (per-push baseline
#     gate). Benchmarks are executed separately via `cargo bench`.
#
# Failure isolation: each phase runs to completion independently (§6.2) so
# one bad phase never hides the rest — a nightly run that dies on a shell
# bug would cost a full day of blind spots before the next window.
#
# Environment variables:
#   NAM_SKIP_GOLDEN_BUILD=1   Opt-out from automatic generation of missing golden vectors.
#                             By default, if goldens are missing and C++ toolchain +
#                             NeuralAmpModelerCore dependencies are present, they are
#                             automatically rebuilt during Phase 0.
#   NAM_AUTO_BUILD_GOLDENS    (Deprecated and ignored)
#   NAM_NEW_ARCH=1            Forces using `--test <entry> <test>` format regardless of disk checks.

set -euo pipefail

# AI NOTE: Due to the long runtime by design, AI agents MUST NOT execute this script directly.
# Ask the human operator to run it and report the results if needed.
#
# Known bug KB-A2-MAX (`wavenet_a2_max.nam`, docs/cpp_parity_map.md §4.4.3):
#   - NOT in REQUIRED_GOLDEN / V2_* release model lists below (by design).
#   - Public path is fail-closed; golden/live/paired tests stay #[ignore].
#   - Do not add a2_max to long parity phases as a green gate until §4.4.3 reopening.
#   - Neighbor A2 Full/Lite + condition_dsp remain first-class long/quick gates.

# ── Test-to-entry-point module mapping ──────────────────────────────────────
# Maps test names to their modular entry-point test files (tests/models.rs,
# tests/parity.rs, tests/perf_soak.rs, tests/rt_constraints.rs). When modular
# entry points are detected on disk or NAM_NEW_ARCH=1 is set, uses `--test <entry> <test>`.
# Otherwise falls back to flat `--test <test_name>`.
#
# To dry-run the entry-point command assembly without executing:
#   NAM_NEW_ARCH=1 utils/tests-long.sh

declare -A LONG_ENTRY_MAP=(
    [meta_coherence]="models"
    [proptest_parsers]="models"
    [proptest_math]="models"
    [gate_fsm_proptest]="models"
    [adaptive_fsm_proptest]="models"
    [lstm_model_dyn_validation]="models"
    [golden_vectors]="models"
    [linear_golden]="models"
    [spectral_fidelity]="models"
    [diagnostic_bundle]="models"
    [lstm_gate_bf16_parity]="parity"
    [lstm_scalar_bf16_parity]="parity"
    [cpp_parity]="parity"
    [cabsim_cpp_parity]="parity"
    [isa_parity]="parity"
    [t33_diagnostic_recurrent_drift_lstm_1x16]="parity"
    [t33b_diagnostic_recurrent_drift_lstm_1x16_paired]="parity"
    [soak_test]="perf_soak"
    [pipeline_soak]="perf_soak"
    [concurrency_stress]="perf_soak"
    [resampler_heap_audit]="rt_constraints"
    [cabsim_heap_audit]="rt_constraints"
    [a2_heap_audit]="rt_constraints"
    [rt_deadline]="rt_constraints"
    [rt_jitter]="rt_constraints"
)

_entry_files_exist() {
    for entry in models perf_soak parity rt_constraints; do
        [ -f "tests/${entry}.rs" ] || return 1
    done
    return 0
}

_test_flag() {
    local test_name="$1"
    if _entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
        local entry="${LONG_ENTRY_MAP[$test_name]:-models}"
        echo "--test $entry ${test_name}"
    else
        echo "--test $test_name"
    fi
}

# Shared style helpers (RED/GREEN/YELLOW/BLUE/BOLD/NC) + cd to project root.
source "$(dirname "$0")/_lib.sh"

# Setup defensive error trap (message-only; phase failures are isolated via
# `run_phase ... || true` below and never reach this trap — see §6.2).
trap 'echo -e "\n${RED}${BOLD}❌ Unexpected error: Command \"$BASH_COMMAND\" failed at line $LINENO with status $?. Aborting audit suite.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}=============================================================${NC}"
echo -e "${BLUE}${BOLD}    NeuralAmpModeler-rs Long-Duration Stress & Audit Suite   ${NC}"
echo -e "${BLUE}${BOLD}=============================================================${NC}"

# Setup target logs and timing tracker
mkdir -p target/logs/
# Targeted cleanup: remove only the log files generated by this suite's phases.
# Preserves regression_phase_receipt.jsonl, regression-check.log, dashboard/
# and any other artifacts from independent tools (Sprint S1.3 T1.3.1).
rm -f target/logs/catalog_preflight.log \
      target/logs/phase1-soak.log \
      target/logs/phase2-proptests-parity.log \
      target/logs/phase3-heap-audit.log \
      target/logs/phase4-rt-deadline.log \
      target/logs/phase5-rt-jitter.log \
      target/logs/phase6-loom.log

TIMED_TRACKER=$(mktemp)
trap 'rm -f "$TIMED_TRACKER"' EXIT

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# Verify NeuralAmpModelerCore presence (repo-local third-party mirror,
# resolved by _lib.sh; override via NAM_THIRD_PARTY_DIR or NAM_CORE_DIR).
# Auto-populate once if missing (hard: abort long suite without Core).
if ! ensure_third_party hard; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore not available — long suite requires it.${NC}"
    exit 1
fi

CURRENT_CORE_SHA=$(cd "$NAM_CORE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ NeuralAmpModelerCore found (version: $CURRENT_CORE_SHA).${NC}"

# ── Phase 0: Pre-flight check — C++ toolchain & golden files ──
echo -e "\n${BLUE}${BOLD}[Phase 0] Pre-flight: checking C++ toolchain and golden vectors...${NC}"

MISSING_GOLDENS=()
MISSING_OPTIONAL_GOLDENS=()
REQUIRED_CABSIM_GOLDENS=(
    "tests/fixtures/golden_cabsim_cpp_short.bin"
    "tests/fixtures/golden_cabsim_cpp_medium.bin"
    "tests/fixtures/golden_cabsim_cpp_long.bin"
)
# v1 golden vectors (48 kHz only) — DistributedCore
REQUIRED_GOLDEN_MODELS=(
    "wavenet_standard" "wavenet_feather" "wavenet_nano"
    "wavenet_a1_standard" "wavenet_a2_full" "wavenet_a2_lite"
    "lstm_1x16" "lstm_2x8" "lstm_official"
)
# v1 golden vectors — LocalNonDistributable (skip gracefully if absent)
NONDIST_GOLDEN_MODELS=(
    "wavenet_lite"
)
# ── V2 Multi-SR Catalog — Single Source of Truth ──
# Maps model golden_name prefixes to their sample-rate scope.
# This replaces the former parallel lists V2_ALL_SR_MODELS / V2_EX_192K_MODELS /
# V2_48K_MODELS / V2_NONDIST_ALL_SR_MODELS.
# Kept in sync with the CATALOG array in tests/fixtures/golden_gen_build.sh and
# the V2_CATALOG table in tests/models/golden_vectors.rs.
# Scope ∈ { multi, nondist_multi, ex192k, sr48k, known_gap }
#   multi          — all 5 sample rates (44100, 48000, 88200, 96000, 192000)
#   nondist_multi  — all 5 SRs, but the model is non-distributable (skip gracefully)
#   ex192k         — all except 192k (LSTM recurrent drift over 5s stress)
#   sr48k          — only 48000 Hz
#   known_gap      — documented upstream gap; absence is expected, never hard-fails
#
# Keys MUST match the golden_name prefixes used by golden_gen_build.sh CATALOG and
# tests/models/golden_vectors.rs V2_CATALOG (including the leading "golden_" prefix).

declare -A V2_CATALOG_SCOPE=(
    ["golden_wavenet_feather"]="multi"
    ["golden_wavenet_nano"]="multi"
    ["golden_wavenet_a1_standard"]="multi"
    ["golden_wavenet_lite"]="nondist_multi"
    ["golden_lstm_1x16"]="ex192k"
    ["golden_lstm_2x8"]="ex192k"
    ["golden_wavenet_standard"]="sr48k"
    ["golden_lstm_official"]="sr48k"
    ["golden_wavenet_a2_full"]="sr48k"
    ["golden_wavenet_a2_lite"]="sr48k"
    ["golden_wavenet_official"]="sr48k"
    ["golden_a2_example"]="sr48k"
    ["golden_wavenet_condition_dsp"]="sr48k"
    ["golden_wavenet_condition_lstm"]="known_gap"
    ["golden_wavenet_app_evh"]="sr48k"
    ["golden_wavenet_boss_bd2"]="sr48k"
    ["golden_wavenet_slammin_marshall"]="sr48k"
    ["golden_lstm_1x10"]="sr48k"
    ["golden_lstm_2x24"]="sr48k"
    ["golden_lstm_3x8"]="sr48k"
    ["golden_convnet_nobn"]="sr48k"
    ["golden_convnet_relu"]="sr48k"
    ["golden_convnet_silu"]="sr48k"
    ["golden_linear_nobias"]="sr48k"
)

V2_ALL_SR=(44100 48000 88200 96000 192000)
V2_EX_192K=(44100 48000 88200 96000)

_v2_check_golden() {
    local golden_name="$1"
    local scope="$2"
    local target_array_name="$3"
    local is_nondist="${4:-0}"
    local g

    case "$scope" in
        multi|nondist_multi)
            for sr in "${V2_ALL_SR[@]}"; do
                g="tests/fixtures/${golden_name}_v2_${sr}.bin"
                if [ "$is_nondist" = "1" ] || [ "$scope" = "nondist_multi" ]; then
                    if [ ! -f "$g" ]; then
                        MISSING_OPTIONAL_GOLDENS+=("$g")
                    fi
                else
                    if [ ! -f "$g" ]; then
                        eval "${target_array_name}+=(\"$g\")"
                    fi
                fi
            done
            ;;
        ex192k)
            for sr in "${V2_EX_192K[@]}"; do
                g="tests/fixtures/${golden_name}_v2_${sr}.bin"
                if [ ! -f "$g" ]; then
                    eval "${target_array_name}+=(\"$g\")"
                fi
            done
            ;;
        sr48k)
            g="tests/fixtures/${golden_name}_v2_48000.bin"
            if [ ! -f "$g" ]; then
                eval "${target_array_name}+=(\"$g\")"
            fi
            ;;
        known_gap)
            # Upstream KnownGap (e.g. condition_lstm): do not require the binary.
            ;;
    esac
}

# Check cabsim goldens
for g in "${REQUIRED_CABSIM_GOLDENS[@]}"; do
    if [ ! -f "$g" ]; then
        MISSING_GOLDENS+=("$g")
    fi
done

# Check v1 goldens — DistributedCore (hard fail if absent)
for m in "${REQUIRED_GOLDEN_MODELS[@]}"; do
    g="tests/fixtures/golden_${m}.bin"
    if [ ! -f "$g" ]; then
        MISSING_GOLDENS+=("$g")
    fi
done

# Check v1 goldens — LocalNonDistributable (skip gracefully if absent)
for m in "${NONDIST_GOLDEN_MODELS[@]}"; do
    g="tests/fixtures/golden_${m}.bin"
    if [ ! -f "$g" ]; then
        MISSING_OPTIONAL_GOLDENS+=("$g")
    fi
done

# Check v2 golden files per model catalog entries
for golden_name in "${!V2_CATALOG_SCOPE[@]}"; do
    scope="${V2_CATALOG_SCOPE[$golden_name]}"
    _v2_check_golden "$golden_name" "$scope" MISSING_GOLDENS 0
done

# Check C++ toolchain availability
MISSING_TOOLS=()
command -v cmake >/dev/null 2>&1 || MISSING_TOOLS+=("cmake")
command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || MISSING_TOOLS+=("g++/clang++ (C++20)")

if [ ${#MISSING_GOLDENS[@]} -gt 0 ] || [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
    echo -e "${RED}${BOLD}❌ Pre-flight failed — missing prerequisites:${NC}"
    if [ ${#MISSING_GOLDENS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Missing golden vectors (${#MISSING_GOLDENS[@]} file(s)):${NC}"
        for g in "${MISSING_GOLDENS[@]}"; do
            echo "    - $g"
        done
    fi
    if [ ${#MISSING_OPTIONAL_GOLDENS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Missing non-distributable golden vectors (${#MISSING_OPTIONAL_GOLDENS[@]} file(s)) — will skip gracefully:${NC}"
        for g in "${MISSING_OPTIONAL_GOLDENS[@]}"; do
            echo "    - $g (optional)"
        done
    fi
    if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
        echo -e "  ${YELLOW}Missing C++ tools:${NC}"
        for t in "${MISSING_TOOLS[@]}"; do
            echo "    - $t"
        done
    fi

    if [ -n "${NAM_AUTO_BUILD_GOLDENS+x}" ]; then
        echo -e "${YELLOW}⚠ WARNING: The NAM_AUTO_BUILD_GOLDENS variable is deprecated and ignored.${NC}"
        echo -e "${YELLOW}  Auto-build is now enabled by default when golden vectors are missing and C++ tools are present.${NC}"
        echo -e "${YELLOW}  To disable auto-build, set NAM_SKIP_GOLDEN_BUILD=1.${NC}"
    fi

    if [ "${NAM_SKIP_GOLDEN_BUILD:-0}" = "1" ]; then
        echo -e "${YELLOW}→ NAM_SKIP_GOLDEN_BUILD=1 — skipping automatic golden regeneration.${NC}"
        exit 1
    fi

    # Auto-build is default when C++ toolchain and NeuralAmpModelerCore are present
    if [ ${#MISSING_TOOLS[@]} -eq 0 ] && [ -d "$NAM_CORE_DIR" ]; then
        echo -e "\n${YELLOW}${BOLD}→ Automatically regenerating goldens (C++ toolchain + NeuralAmpModelerCore present)...${NC}"
        if ! bash tests/fixtures/golden_gen_build.sh; then
            echo -e "${RED}${BOLD}❌ golden_gen_build.sh failed. Fix dependencies and try again.${NC}"
            exit 1
        fi
        echo -e "${GREEN}✓ Golden vectors successfully regenerated.${NC}"
        # Re-validate golden files after generation
        MISSING_GOLDENS=()
        MISSING_OPTIONAL_GOLDENS=()
        for g in "${REQUIRED_CABSIM_GOLDENS[@]}"; do
            [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
        done
        for m in "${REQUIRED_GOLDEN_MODELS[@]}"; do
            g="tests/fixtures/golden_${m}.bin"
            [ ! -f "$g" ] && MISSING_GOLDENS+=("$g")
        done
        for m in "${NONDIST_GOLDEN_MODELS[@]}"; do
            g="tests/fixtures/golden_${m}.bin"
            [ ! -f "$g" ] && MISSING_OPTIONAL_GOLDENS+=("$g")
        done
        # Re-validate v2 golden files after generation
        for golden_name in "${!V2_CATALOG_SCOPE[@]}"; do
            scope="${V2_CATALOG_SCOPE[$golden_name]}"
            _v2_check_golden "$golden_name" "$scope" MISSING_GOLDENS 0
        done
        if [ ${#MISSING_GOLDENS[@]} -gt 0 ]; then
            echo -e "${RED}${BOLD}❌ Still missing goldens after golden_gen_build.sh:${NC}"
            for g in "${MISSING_GOLDENS[@]}"; do
                echo "    - $g"
            done
            echo -e "  ${YELLOW}V2 goldens may not be generated for all SRs (C++ render tool constraint).${NC}"
            exit 1
        fi
        if [ ${#MISSING_OPTIONAL_GOLDENS[@]} -gt 0 ]; then
            echo -e "  ${YELLOW}Non-distributable goldens still absent after regeneration (expected):${NC}"
            for g in "${MISSING_OPTIONAL_GOLDENS[@]}"; do
                echo "    - $g (optional, will skip gracefully)"
            done
        fi
    else
        if [ ${#MISSING_TOOLS[@]} -gt 0 ]; then
            echo -e "  ${YELLOW}→ Install: cmake >= 3.10, g++/clang++ with C++20 support${NC}"
        fi
        echo -e "  ${YELLOW}→ Run: ./utils/setup-third-party.sh${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ C++ prerequisites and golden vectors verified.${NC}"

# ── Catalog Preflight — Capability Receipt (Sprint 6: T-E3.1-1) ──
# Runs the Rust-side unified fixture catalog capability receipt BEFORE
# the freshness gate and long suite, so the operator has a complete
# typed inventory of every fixture's status (Available / MissingOptional /
# MissingRequired) with resolved paths.  Missing RequiredLocal fixtures
# are reported but do not gate here — the shell-level Phase 0 above and
# the freshness gate below provide hard-fail gating.
echo -e "\n${BLUE}${BOLD}→ Generating fixture catalog capability receipt (catalog_preflight)...${NC}"
if ! cargo test --features testing --release $(_test_flag catalog_preflight) -- --nocapture 2>&1 | tee target/logs/catalog_preflight.log; then
    echo -e "${YELLOW}⚠ catalog_preflight exited non-zero (test may have reported missing required fixtures).${NC}"
fi
# Extract MISSING-REQUIRED count for the summary
MISSING_REQUIRED_COUNT=$(grep -c 'MISSING-REQUIRED:' target/logs/catalog_preflight.log 2>/dev/null || true)
if [ "$MISSING_REQUIRED_COUNT" -gt 0 ]; then
    echo -e "${RED}${BOLD}❌ Catalog preflight: ${MISSING_REQUIRED_COUNT} RequiredLocal fixture(s) absent.${NC}"
    echo -e "  Check target/logs/catalog_preflight.log for the full capability receipt."
    exit 1
fi
echo -e "${GREEN}✓ All RequiredLocal fixtures present per catalog preflight.${NC}"

# ── Package Exclusion Verification (Sprint 6: T-E3.4-1) ──
# Confirms that non-distributable models (models-nondist/) and third-party
# vendor artifacts are excluded from the crate package.  Any leak of
# proprietary / license-restricted content into the distributed crate
# is a hard packaging failure.
echo -e "\n${BLUE}${BOLD}→ Verifying cargo package exclusion of non-distributable artifacts...${NC}"
PACKAGE_LIST=$(cargo package --list 2>/dev/null || true)
if echo "$PACKAGE_LIST" | grep -qE '(models-nondist|third-party/)'; then
    echo -e "${RED}${BOLD}❌ PACKAGE VIOLATION: non-distributable or third-party artifacts leaked into crate package.${NC}"
    echo "$PACKAGE_LIST" | grep -E '(models-nondist|third-party/)' || true
    exit 1
fi
echo -e "${GREEN}✓ Package exclusion verified — no models-nondist/ or third-party/ artifacts in crate.${NC}"

# ── Freshness gate (blocking, centralized) ──
# hard-fail: artifact integrity + generator provenance (stricter than quick's
# artifacts-hard, which only warns on generator-script drift).
# Bypass with NAM_BYPASS_FRESHNESS=1 for local developer convenience.
echo -e "\n→ Checking freshness of test fixtures and goldens..."
check_freshness hard-fail || exit 1

# ── Catalog↔test coherence gate (blocking) ──
# `meta_coherence` is a cheap, dependency-free governance test (no NAMCore, no
# goldens needed — it only parses golden_gen_build.sh + tests/*.rs). It has no
# home in tests-quick.sh (not a correctness or structural test) and would be
# silently orphaned ("on demand" only) without this hook. Runs here, before
# the ± 50 min battery, so a drifted catalog fails fast instead of burning a
# full nightly window before being noticed.
echo -e "\n${BLUE}${BOLD}→ Checking catalog↔test coherence (meta_coherence)...${NC}"
if ! cargo test --features testing --release $(_test_flag meta_coherence); then
    echo -e "${RED}${BOLD}❌ meta_coherence failed — golden catalog diverged from #[ignore] tests.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Golden catalog matches tests coherently.${NC}"

# ── Phase classification for fidelity/performance split ──────────────────────
# Fidelity phases: 0 (Pre-flight), 1 (Soak), 2 (Proptests/Parity/Golden),
#                  3 (Heap-Audit), 6 (Loom)
# Performance phases: 4 (RT Deadline), 5 (RT Jitter)
declare -A PHASE_CLASS=(
    [0]="fidelity"
    [1]="fidelity"
    [2]="fidelity"
    [3]="fidelity"
    [4]="performance"
    [5]="performance"
)

# Trackers for the final summary
declare -a PHASE_NAMES
declare -a PHASE_COMMANDS
declare -a PHASE_STATUS
declare -a PHASE_DURATIONS
declare -a PHASE_SUB_TIMINGS
declare -a PHASE_CLASSES
PHASE_COUNT=0
N_TOP_SLOWEST=5

# timed_cargo_test — runs a cargo test invocation, captures timing.
# Usage: timed_cargo_test <label> <cargo_test_args...>
# Appends per-invocation "TIMED: <seconds> <label>" lines to a temp tracker.
timed_cargo_test() {
    local label="$1"
    shift
    local start_t
    start_t=$(date +%s%N)
    cargo test --features testing "$@"
    local status=$?
    local end_t
    end_t=$(date +%s%N)
    local duration_ns=$((end_t - start_t))
    local duration_s
    duration_s=$(LC_NUMERIC=C awk -v ns="$duration_ns" 'BEGIN { printf "%.3f", ns / 1000000000 }')
    echo "TIMED: $duration_s $label" >> "$TIMED_TRACKER"
    return $status
}

# extract_sub_timings: reads the timed tracker, returns top-N slowest entries.
extract_sub_timings() {
    if [ ! -f "$TIMED_TRACKER" ] || [ ! -s "$TIMED_TRACKER" ]; then
        return
    fi
    grep '^TIMED:' "$TIMED_TRACKER" | \
        sed 's/^TIMED: //' | \
        sort -rn | \
        head -n "$N_TOP_SLOWEST"
}



assert_ran_tests() {
    local log_file="$1"
    local min_count="${2:-1}"

    local total_passed=0

    local passed
    if passed=$(grep -oP 'test result: ok\.\s+\K\d+(?=\s+passed)' "target/logs/$log_file" 2>/dev/null); then
        for p in $passed; do
            total_passed=$((total_passed + p))
        done
    fi

    local measured
    if measured=$(grep -oP '\K\d+(?=\s+measured)' "target/logs/$log_file" 2>/dev/null); then
        for m in $measured; do
            total_passed=$((total_passed + m))
        done
    fi

    if [ "$total_passed" -eq 0 ]; then
        local bench_count
        bench_count=$(grep -cP '^\S.*time:\s+\[' "target/logs/$log_file" 2>/dev/null || true)
        total_passed=$bench_count
    fi

    if [ "$total_passed" -lt "$min_count" ]; then
        echo -e "${RED}${BOLD}❌ Gate failed: phase executed 0 tests/benchmarks (empty selection or filter mismatch).${NC}"
        return 1
    fi
    echo -e "  Gate: ${total_passed} test(s)/benchmark(s) executed ≥ ${min_count}  ✓"
    return 0
}

run_phase() {
    local name="$1"
    local cmd="$2"
    local log_file="$3"

    echo -e "\n${BLUE}${BOLD}[Phase $((PHASE_COUNT+1))] $name...${NC}"
    echo -e "Executing: ${YELLOW}$cmd${NC}"
    echo -e "Log at: ${YELLOW}target/logs/$log_file${NC}"

    local start_time=$(date +%s)

    # Reset timed tracker for this phase
    : > "$TIMED_TRACKER"

    # Run command and capture output/status
    eval "$cmd" > "target/logs/$log_file" 2>&1
    local status=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    PHASE_NAMES[$PHASE_COUNT]="$name"
    PHASE_COMMANDS[$PHASE_COUNT]="$cmd"
    PHASE_DURATIONS[$PHASE_COUNT]="$duration"
    PHASE_CLASSES[$PHASE_COUNT]="${PHASE_CLASS[$PHASE_COUNT]:-fidelity}"

    # Capture sub-timings for this phase
    PHASE_SUB_TIMINGS[$PHASE_COUNT]="$(extract_sub_timings)"

    if [ $status -eq 77 ]; then
        echo -e "${YELLOW}⚠ SKIPPED (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="SKIPPED"
    elif [ $status -eq 0 ]; then
        echo -e "${GREEN}✓ Success (${duration}s)${NC}"
        PHASE_STATUS[$PHASE_COUNT]="PASSED"

        if ! assert_ran_tests "$log_file"; then
            echo -e "${RED}❌ Gate \"≥1 executed\" failed (${duration}s) — status promoted to FAILED.${NC}"
            PHASE_STATUS[$PHASE_COUNT]="FAILED"
            PHASE_COUNT=$((PHASE_COUNT + 1))
            return 1
        fi
    else
        echo -e "${RED}❌ Failure (${duration}s) - Status: $status${NC}"
        PHASE_STATUS[$PHASE_COUNT]="FAILED"
    fi

    PHASE_COUNT=$((PHASE_COUNT + 1))
    return $status
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase bodies — one function per phase. Each function is passed by name to
# run_phase (never inlined as a giant `;`-chained string) so the suite stays
# easy to scan and extend cleanly in an unattended nightly job.
# Every phase cross-references its non-overlapping tests-quick.sh counterpart.
# ═══════════════════════════════════════════════════════════════════════════

# --- Phase 1: Soak/Endurance (release, --ignored) ---
# tests-quick.sh runs 1 non-ignored decomposition test per suite (Phase 1,
# debug); every #[ignore]'d soak/endurance test (10M+ frames) lives here.
run_soak_phase() {
    local status=0
    timed_cargo_test "soak_test" --release --no-fail-fast $(_test_flag soak_test) -- --ignored --nocapture || status=1
    timed_cargo_test "pipeline_soak" --release --no-fail-fast $(_test_flag pipeline_soak) -- --ignored --nocapture --test-threads=1 || status=1
    return $status
}
run_phase "Soak Tests (Numerical Stability)" "run_soak_phase" "phase1-soak.log" || true

# --- Phase 2: Property-Based, FSM, Parity, Golden Vectors & ISA (release) ---
# The full, uncapped counterpart of tests-quick.sh Phase 2/3: every proptest
# runs at its full case count (Phase 3 caps proptest_parsers at 1000 cases),
# the C++/golden oracles run their full multi-SR/full-matrix scope (Phase 2
# only runs the v1/quick_parity subset), and cross-ISA + heavy/dyn parity —
# entirely absent from the quick suite — run here for the first time.
run_proptests_parity_phase() {
    local status=0
    # Full-count parser/math/gate/FSM fuzzing (quick caps or excludes these).
    timed_cargo_test "proptest_parsers" --release --no-fail-fast $(_test_flag proptest_parsers) -- --ignored || status=1
    timed_cargo_test "proptest_math" --release --no-fail-fast $(_test_flag proptest_math) -- --ignored || status=1
    timed_cargo_test "lstm_gate_bf16_parity" --release --no-fail-fast $(_test_flag lstm_gate_bf16_parity) -- --ignored || status=1
    timed_cargo_test "lstm_scalar_bf16_parity" --release --no-fail-fast $(_test_flag lstm_scalar_bf16_parity) -- --ignored || status=1
    timed_cargo_test "gate_fsm_proptest" --release --no-fail-fast $(_test_flag gate_fsm_proptest) -- --ignored || status=1
    timed_cargo_test "adaptive_fsm_proptest" --release --no-fail-fast $(_test_flag adaptive_fsm_proptest) -- --ignored || status=1
    # ModelDyn scalar-vs-SIMD parity proptests (arbitrary topologies) — no
    # quick-suite equivalent; LstmModelDyn parity is otherwise untested.
    timed_cargo_test "lstm_model_dyn_validation" --release --no-fail-fast $(_test_flag lstm_model_dyn_validation) -- --ignored --nocapture || status=1
    # Full C++ NAMCore live parity matrix + CabSim convolution parity
    # (quick's Phase 2 only runs the 3-model `quick_parity` subset).
    timed_cargo_test "cpp_parity" --release --no-fail-fast $(_test_flag cpp_parity) -- --ignored --nocapture || status=1
    timed_cargo_test "cabsim_cpp_parity" --release --no-fail-fast $(_test_flag cabsim_cpp_parity) -- --ignored --nocapture || status=1
    # Recurrent State Drift Diagnostics
    timed_cargo_test "t33_diagnostic_recurrent_drift_lstm_1x16" --release --no-fail-fast $(_test_flag t33_diagnostic_recurrent_drift_lstm_1x16) -- --ignored --nocapture || status=1
    timed_cargo_test "t33b_diagnostic_recurrent_drift_lstm_1x16_paired" --release --no-fail-fast $(_test_flag t33b_diagnostic_recurrent_drift_lstm_1x16_paired) -- --ignored --nocapture || status=1
    # Golden vectors v2 (multi-SR); v1 already covered by quick's Phase 2.
    # Filter MUST be a single libtest substring after `--`. Do not also pass
    # `golden_vectors` as a cargo TESTNAME: multiple filters are OR'd by libtest,
    # which previously pulled in KB-A2-MAX ignored diagnostics
    # (`test_golden_vectors_wavenet_a2_max`, H0/H5 meters) and failed Phase 2.
    # Match only multi-SR v2 goldens: test_golden_vectors_v2_*.
    timed_cargo_test "golden_vectors_v2" --release --no-fail-fast --test models \
        -- test_golden_vectors_v2_ --ignored --nocapture || status=1
    # Heavy/long receptive-field golden regression (quick only runs the
    # cheap non-ignored linear_golden cases).
    timed_cargo_test "linear_golden_heavy" --release --no-fail-fast $(_test_flag linear_golden) -- --ignored --nocapture || status=1
    # Full cross-ISA determinism matrix (AVX-512, VNNI+BF16 vs AVX2). Quick's
    # Phase 2 only asserts AVX2 self-consistency; gracefully skips per-model
    # when the running CPU lacks the target ISA (see skip_if_unsupported!
    # in tests/isa_parity.rs) — safe to run unconditionally on any machine.
    timed_cargo_test "isa_parity_full_matrix" --release --no-fail-fast $(_test_flag isa_parity) -- --ignored --test-threads=1 --nocapture || status=1
    # Spectral fidelity baseline immutability defense (Sprint S1.1 T1.1.2):
    # SHA-256 before/after guards the committed fixture against accidental
    # overwrite by the generator or any test-side mutation.
    local SF_BASELINE="tests/fixtures/spectral_fidelity_baseline.json"
    local SF_HASH_BEFORE
    SF_HASH_BEFORE=$(sha256sum "$SF_BASELINE" | awk '{print $1}')
    timed_cargo_test "spectral_fidelity_baselines" --release --no-fail-fast $(_test_flag spectral_fidelity) -- spectral_fidelity::model_baselines::baseline_ --skip generate_spectral_fidelity_baseline --ignored --nocapture || status=1
    local SF_HASH_AFTER
    SF_HASH_AFTER=$(sha256sum "$SF_BASELINE" | awk '{print $1}')
    if [ "$SF_HASH_BEFORE" != "$SF_HASH_AFTER" ]; then
        echo -e "\n${RED}${BOLD}❌ IMMUTABILITY VIOLATION: $SF_BASELINE was modified during the spectral fidelity subphase.${NC}" >&2
        echo -e "  SHA-256 before: ${YELLOW}$SF_HASH_BEFORE${NC}" >&2
        echo -e "  SHA-256 after:  ${RED}$SF_HASH_AFTER${NC}" >&2
        echo "  The committed baseline fixture must remain bitwise immutable under automated suites." >&2
        status=1
    fi
    # Random block-size sweep for the pipeline resampler chain.
    timed_cargo_test "lib_pipeline_block_proptest" --release --no-fail-fast --lib -- dsp::pipeline::pipeline_block_test::block_tests::test_random_block_sizes_proptest --ignored || status=1
    # Tier-3 "approx-vs-approx" consistency checks (Padé/poly NR1 vs NR2 vs
    # div_ps, AVX2 + AVX-512 for tanh and sigmoid): the f64 Oracle already
    # provides absolute correctness, so these only guard against silent
    # regressions between two approximate paths (docs/testing.md §8).
    # AVX-512 variants self-skip via `is_x86_feature_detected!` when unsupported.
    timed_cargo_test "activations_consistency" --release --no-fail-fast --lib -- "math::activations::" --ignored --nocapture || status=1
    # Gate FSM envelope continuity proptest (10k cases) — unit-level sibling
    # of tests/gate_fsm_proptest.rs, covers the DynamicHysteresis reversal
    # edge case specifically.
    timed_cargo_test "gate_envelope_continuity_proptest" --release --no-fail-fast --lib -- "dsp::gate::gate_test::tests::gate_envelope_continuity_on_reversal" --ignored --nocapture || status=1
    return $status
}
run_phase "Property-Based, Parity & Golden Vectors in Release" "run_proptests_parity_phase" "phase2-proptests-parity.log" || true

# Post-phase immutability gate (Sprint S1.1 T1.1.2): if the log contains the
# baseline-write signature, the generator was invoked — hard abort regardless
# of phase exit status.
if grep -qF "Baseline written to" target/logs/phase2-proptests-parity.log 2>/dev/null; then
    echo -e "${RED}${BOLD}❌ IMMUTABILITY VIOLATION: phase log contains 'Baseline written to' — the spectral fidelity baseline was overwritten during automated execution.${NC}"
    exit 1
fi

# --- Phase 3: RT-Safety Heap-Audit (release, heap-audit) ---
# Zero-alloc verification under the global counting allocator. No quick-suite
# equivalent — the `heap-audit` feature is exclusively a long-suite concern.
run_heap_audit_phase() {
    local status=0
    timed_cargo_test "resampler_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag resampler_heap_audit) || status=1
    timed_cargo_test "cabsim_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag cabsim_heap_audit) || status=1
    timed_cargo_test "a2_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag a2_heap_audit) || status=1
    timed_cargo_test "diagnostic_bundle_heap_audit" --release --no-fail-fast --features heap-audit $(_test_flag diagnostic_bundle) -- heap_audit || status=1
    return $status
}
run_phase "Resampler, Cabsim & A2 Heap-Audit" "run_heap_audit_phase" "phase3-heap-audit.log" || true

# --- Phase 4: RT Deadline Gate (deterministic, hard assertion) ---
# Absolute latency ceiling: assert!(p99 < 1.33 ms) for every model SKU.
# This is the definitive gate — a regression that pushes p99 past the
# audio buffer deadline fails the build deterministically.
run_rt_deadline_gate_phase() {
    local status=0
    timed_cargo_test "rt_deadline" --release --no-fail-fast $(_test_flag rt_deadline) -- --nocapture || status=1
    return $status
}
run_phase "RT Deadline Gate (deterministic)" "run_rt_deadline_gate_phase" "phase4-rt-deadline.log" || true

# --- Phase 5: RT Jitter Characterization (environmental telemetry) ---
# Characterizes tail latency under CPU contention. This is diagnostic
# telemetry — it does NOT assert deadlines under stress. An INCONCLUSIVE
# result is expected when environment preconditions (CPU pinning,
# performance governor, low background load) are not met.
run_rt_jitter_characterization_phase() {
    local status=0
    timed_cargo_test "rt_jitter" --release --no-fail-fast $(_test_flag rt_jitter) -- --ignored --nocapture || status=1
    return $status
}
run_phase "RT Jitter Characterization" "run_rt_jitter_characterization_phase" "phase5-rt-jitter.log" || true

# Post-phase: detect typed status from log and override phase status.
# The Rust test returns exit 0 even when internally bypassed (INCONCLUSIVE
# or SKIP_CAPABILITY) to avoid false FAIL — the log markers are authoritative.
# Invariant: exit-0 with internal measurement bypass SHALL NOT be promoted to PASS.
JITTER_LOG="target/logs/phase5-rt-jitter.log"
if grep -qF "[STATUS] SKIP_CAPABILITY" "$JITTER_LOG" 2>/dev/null; then
    JITTER_IDX=$((PHASE_COUNT - 1))
    PHASE_STATUS[$JITTER_IDX]="SKIP_CAPABILITY"
elif grep -qF "[STATUS] INCONCLUSIVE" "$JITTER_LOG" 2>/dev/null; then
    JITTER_IDX=$((PHASE_COUNT - 1))
    PHASE_STATUS[$JITTER_IDX]="INCONCLUSIVE"
fi

# --- Phase 6: Loom Concurrency Model Checking (release) ---
# Model-checks the SPSC/GC/DspBridge lock-free primitives under loom's
# exhaustive permutation engine. Runs with --cfg loom so the production
# atomic paths are replaced by loom's instrumented wrappers. No quick-suite
# equivalent — loom is too slow for a per-push gate and needs release mode
# to keep permutation-space exploration bounded within a few minutes.
run_loom_phase() {
    local status=0
    echo "  Compiling and running loom model checking with cfg=loom..."
    local loom_flags="${RUSTFLAGS:--Ctarget-cpu=x86-64-v3}"
    RUSTFLAGS="$loom_flags --cfg loom" timed_cargo_test "loom_tests" --release --no-fail-fast --test loom_tests -- --nocapture || status=1
    return $status
}
run_phase "Loom Concurrency Model Checking" "run_loom_phase" "phase6-loom.log" || true

# --- Print beautifully structured summary ---
echo -e "\n${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                  AUDIT SUMMARY REPORT                          ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
printf " | %-45s | %-10s | %-10s |\n" "Phase Name" "Duration" "Status"
printf " |-%-45s-|-%-10s-|-%-10s-|\n" "---------------------------------------------" "----------" "----------"

ANY_FAILED=0
ANY_FIDELITY_FAILED=0
RT_DEADLINE_STATUS="PASS"
RT_JITTER_STATUS="PASS"
for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    duration="${PHASE_DURATIONS[$i]}s"
    status="${PHASE_STATUS[$i]}"
    class="${PHASE_CLASSES[$i]:-fidelity}"

    if [ "$status" = "PASSED" ]; then
        status_colored="${GREEN}${status}${NC}"
    elif [ "$status" = "SKIPPED" ] || [ "$status" = "SKIP_CAPABILITY" ] || [ "$status" = "INCONCLUSIVE" ]; then
        status_colored="${YELLOW}${status}${NC}"
    else
        status_colored="${RED}${status}${NC}"
        ANY_FAILED=1
        if [ "$class" != "performance" ]; then
            ANY_FIDELITY_FAILED=1
        fi
    fi

    # ── Per-phase typed status tracking for disaggregated performance summary ──
    if [[ "$name" == *"RT Deadline"* ]]; then
        if [ "$status" = "FAILED" ]; then
            RT_DEADLINE_STATUS="FAIL"
        elif [ "$status" = "SKIPPED" ]; then
            RT_DEADLINE_STATUS="PASS"
        fi
    elif [[ "$name" == *"RT Jitter"* ]]; then
        case "$status" in
            PASSED)       RT_JITTER_STATUS="PASS" ;;
            INCONCLUSIVE) RT_JITTER_STATUS="INCONCLUSIVE" ;;
            SKIP_CAPABILITY) RT_JITTER_STATUS="SKIP_CAPABILITY" ;;
            FAILED)       RT_JITTER_STATUS="FAIL" ;;
            SKIPPED)      RT_JITTER_STATUS="INCONCLUSIVE" ;;
        esac
    fi

    printf " | %-45s | %-10s | %-19b |\n" "$name" "$duration" "$status_colored"
done

# --- Top-N slowest sub-timings per heavy phase ---
echo -e "\n${BLUE}${BOLD}  Top-$N_TOP_SLOWEST Slowest Items per Heavy Phase${NC}"
echo -e "${BLUE}${BOLD}  $(printf '━%.0s' {1..60})${NC}"

for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    sub_timings="${PHASE_SUB_TIMINGS[$i]}"

    if [ -n "$sub_timings" ]; then
        echo -e "\n  ${YELLOW}${BOLD}[$name]${NC}"
        rank=1
        while IFS= read -r line; do
            if [ -n "$line" ]; then
                t="${line%% *}"
                lbl="${line#* }"
                printf "    %2d. %8ss  %s\n" "$rank" "$t" "$lbl"
                rank=$((rank + 1))
            fi
        done <<< "$sub_timings"
    fi
done

echo -e "\n${BLUE}${BOLD}================================================================${NC}"

# Cleanup timed tracker temp file
rm -f "$TIMED_TRACKER"

# Must be defined before first invocation — Bash does not hoist functions.
render_performance_summary() {
    case "$RT_DEADLINE_STATUS" in
        FAIL) echo -e "${RED}RT_DEADLINE: ${RT_DEADLINE_STATUS}${NC}" ;;
        *)    echo -e "${GREEN}RT_DEADLINE: ${RT_DEADLINE_STATUS}${NC}" ;;
    esac
    case "$RT_JITTER_STATUS" in
        PASS)            echo -e "${GREEN}RT_JITTER: ${RT_JITTER_STATUS}${NC}" ;;
        INCONCLUSIVE|SKIP_CAPABILITY) echo -e "${YELLOW}RT_JITTER: ${RT_JITTER_STATUS}${NC}" ;;
        FAIL)            echo -e "${RED}RT_JITTER: ${RT_JITTER_STATUS}${NC}" ;;
    esac
    echo -e "${YELLOW}PERF_REGRESSION: NOT_RUN${NC}"
}

if [ $ANY_FAILED -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ All audit stages completed successfully!${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    render_performance_summary
    exit 0
else
    echo -e "${RED}${BOLD}❌ One or more audit stages failed. Check logs in target/logs/${NC}"
    if [ $ANY_FIDELITY_FAILED -eq 1 ]; then
        echo -e "${RED}FIDELITY: FAIL${NC}"
    else
        echo -e "${GREEN}FIDELITY: OK${NC}"
    fi
    render_performance_summary
    exit 1
fi
