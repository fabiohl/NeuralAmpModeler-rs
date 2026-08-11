#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Quick QA Suite for NeuralAmpModeler-rs — agile first line of defense.
#
# Division of responsibility among QA scripts:
#   * utils/lints.sh       — Static quality gate (fmt, SPDX, cargo check, clippy).
#                            Runs continuously. Not duplicated here.
#   * utils/tests-quick.sh — THIS script. Agile test suite (cargo test).
#                            Runs frequently during development (~2-3 min execution).
#                            Catches functional and precision regressions fast.
#   * utils/tests-long.sh  — Comprehensive test suite (soak, full proptests,
#                            full C++ parity, concurrency, benchmarks, RT).
#                            Time-consuming (~50 min), run nightly or pre-release.
#
# Philosophical design principles (docs/testing.md §7 — two orthogonal axes):
#   Axis A (Rigor):     ignored = long/rigorous; non-ignored = quick first line.
#   Axis B (Float path): structural → debug (fast, debug-assertions ON);
#                        measurement oracle → release (measures production path,
#                        preventing codegen "ghosts" — non-optimized, no FMA,
#                        no autovectorization).
#   Phase 1 keeps Axis A (non-ignored) in the appropriate Axis B (debug structural).
#   Phase 2 keeps measurement oracles in Axis B (production release).
#
# Phases:
#   1. Structural (debug) — unit (lib) + deterministic integration. Fast compilation,
#      debug-assertions ON. Verifies parser logic, state machines, loaders, SPSC,
#      bitwise determinism, FSM. Excludes the measurement oracles from §7
#      (→ Phase 2 release) and rt_deadline (→ long test suite).
#      Includes active known-bug policy tests (e.g. wavenet_a2_max dispatch rejected
#      — KB-A2-MAX / docs/cpp_parity_map.md §4.4.3). Does NOT run ignored A2 Max
#      golden/meter/oracle pairs (those need NAM_A2_MAX_UNLOCK=1 and are not gates).
#   2. Measurement oracles (release, docs/testing.md §7) — authoritative gate for
#      production floats: golden_vectors v1, cpp_parity quick_parity,
#      reference_oracle_f64, isa_parity (AVX2 self-consistency), spectral_fidelity.
#      Graceful skip for missing dependencies (NAMCore/goldens).
#      KB-A2-MAX fixtures stay #[ignore] / KNOWN_GAP — must not appear as green parity.
#   3. Agile parser fuzzing (release, --ignored) — proptest_parsers (Tier 1:
#      parser robustness/security) with reduced case count for speed
#      (configurable via NAM_QUICK_PROPTEST_CASES).
#
# Coverage notes:
#   - Clippy and cargo check remain in lints.sh (not duplicated here).
#   - CLAP block and feature-gated tests (heap-audit/clap) belong in tests-long.sh.
#   - proptest_math (Tier 3) and rt_deadline/rt_jitter(stress)/soak stay in tests-long.sh.
#   - Phase 1 explicitly maps structural integration tests or auto-detects entry points.
#     Auto-discovery of library unit tests is preserved.
#
# ── Skip conditions (graceful exit 0) ─────────────────────────────────────────
# The following skip scenarios are handled gracefully (exit code 0 with
# informational messages). They are designed for CI environments and developer
# machines that may not have all optional dependencies.
#
# Scenario                              Condition                           Consequence
# ────────────────────────────────────  ──────────────────────────────────  ──────────────────────────────────────────────────
# Golden vectors (v1/v2) absent         golden_wavenet_standard.bin and     golden_vectors + isa_parity skipped.
#                                       golden_wavenet_standard_v2_48000    f64 oracle, Spectral Fidelity, Linear FFT
#                                       .bin missing from tests/fixtures/   still run (mathematical-oracle tests, no
#                                                                           pre-computed goldens needed).
#                                       Tracked by: GOLDEN_RAN
#
# C++ toolchain not found               Neither g++ nor clang++ in PATH     cpp_parity entirely skipped.
#                                       Tracked by: CPP_PARITY_SKIPPED
#
# CMake configure / build failure       cmake fails to configure or         cpp_parity entirely skipped.
#                                       build the C++ render binary         Tracked by: CPP_PARITY_SKIPPED
#
# NAMCore not checked out               third-party/NeuralAmpModelerCore    cpp_parity entirely skipped.
#                                       Core directory absent               Tracked by: CPP_PARITY_SKIPPED
#
# All mandatory tests are NOT skippable — their failure always produces exit
# code 1. These include:
#   Phase 1: structural unit + integration tests (debug)
#   Phase 2: f64 oracle, Spectral Fidelity, Linear FFT (always run, no deps)
#   Phase 3: parser fuzzing (proptest_parsers)
#
# CI note: In a minimal CI environment without golden fixtures and without a
# C++ toolchain, this script exits 0 after running the non-skippable core.
# No false alarms are raised.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PHASE_TOTAL=3
source "$SCRIPT_DIR/_lib.sh"

# ── Freshness gate ────────────────────────────────────────────────────────────
# Centralized check_freshness() lives in _lib.sh.
# Called below with hard-fail mode — staleness blocks the test suite per
# the "Every Golden Must Be Able To Fail" principle.

# Re-execute with low CPU and I/O priority (nice and ionice) to prevent system overload.
# This can be bypassed by setting NAM_NO_LOW_PRIORITY=1.
if [ "${NAM_LOW_PRIORITY:-0}" != "1" ] && [ "${NAM_NO_LOW_PRIORITY:-0}" != "1" ]; then
    export NAM_LOW_PRIORITY=1
    CMD_PREFIX=""
    if command -v nice >/dev/null 2>&1; then
        CMD_PREFIX="nice -n 19"
    fi
    if command -v ionice >/dev/null 2>&1; then
        CMD_PREFIX="$CMD_PREFIX ionice -c 3"
    fi
    if [ -n "$CMD_PREFIX" ]; then
        echo -e "${YELLOW}ⓘ Restarting script with low priority (CPU/IO) to prevent system overload...${NC}"
        exec $CMD_PREFIX "$SCRIPT_PATH" "$@"
    fi
fi

trap 'echo -e "\n${RED}${BOLD}❌ Unexpected error: Command \"$BASH_COMMAND\" failed at line $LINENO with status $?. Aborting test suite.${NC}"; exit 1' ERR

echo -e "${BLUE}${BOLD}==========================${NC}"
echo -e "${BLUE}${BOLD}   NeuralAmpModeler-rs Quick QA Suite"
echo -e "${BLUE}${BOLD}==========================${NC}"

# ── Phase 1: Structural (debug) ─────────────────────────────────────────────
# Unit tests (lib, auto-discovered) + deterministic integration (explicit list).
# Excludes the 5 measurement oracles from §7 (→ Phase 2 release) and rt_deadline (→ long).
# perf_soak (concurrency_stress, spsc_pipeline, soak_test) remains in Phase 1:
# structural deterministic tests (~2s) validating concurrency/pipeline invariants.
# debug-assertions ON catches cheap invariants that --release would mask.
phase "Structural: unit + deterministic integration (debug)..."

# ── Test-to-entry-point mapping ─────────────────────────────────────────────
# Each structural test is assigned to its entry-point module (models, perf_soak, parity).
# When entry-point files exist, the script auto-detects and uses grouped test targets.
# Otherwise, it falls back to legacy flat tests/ layout (--test=<file>).
#
# To dry-run architecture detection without executing:
#   NAM_DRY_RUN_ARCH=1 utils/tests-quick.sh

declare -A STRUCT_ENTRY_MAP=(
    [a2_loader]="models"
    [activation_precision]="models"
    [adaptive_fsm_proptest]="models"
    [cabsim_golden]="models"
    [concurrency_stress]="perf_soak"
    [container_slimmable]="models"
    [diagnostic_bundle]="models"
    [ebu_lufs_compliance]="models"
    [fixture_b1_2_smoke]="models"
    [linear_golden]="models"
    [lstm_activation_precision]="models"
    [lstm_model_dyn_validation]="models"
    [mirror_buf_fault_injection]="models"
    [nam_infer_test]="models"
    [namb_v2_roundtrip]="models"
    [namb_v2_validation]="models"
    [nondist_validation]="models"
    [parity_primitives]="parity"
    [prewarm_test]="models"
    [proptest_math]="models"
    [self_consistency]="models"
    [soak_test]="perf_soak"
    [spsc_pipeline]="perf_soak"
    [threshold_calibration]="models"
    [wavenet_lite_block_invariance]="models"
    [wavenet_prewarm_edge]="models"
    [zero_alloc_infer]="models"
)

STRUCT_TESTS=(
    a2_loader activation_precision adaptive_fsm_proptest cabsim_golden
    concurrency_stress container_slimmable diagnostic_bundle ebu_lufs_compliance
    fixture_b1_2_smoke linear_golden lstm_activation_precision
    lstm_model_dyn_validation mirror_buf_fault_injection nam_infer_test
    namb_v2_roundtrip namb_v2_validation nondist_validation parity_primitives
    prewarm_test proptest_math self_consistency soak_test spsc_pipeline
    threshold_calibration wavenet_lite_block_invariance wavenet_prewarm_edge
    zero_alloc_infer
)

_structural_entry_files_exist() {
    for entry in models perf_soak parity rt_constraints; do
        [ -f "tests/${entry}.rs" ] || return 1
    done
    return 0
}

# ── Phase 2/3 measurement oracle → entry-point mapping ─────────────────────
declare -A MEASUREMENT_ENTRY_MAP=(
    [reference_oracle_f64]="parity"
    [spectral_fidelity]="models"
    [linear_fft_test]="models"
    [golden_vectors]="models"
    [isa_parity]="parity"
    [cpp_parity]="parity"
    [proptest_parsers]="models"
)

# Helper: builds cargo test args for measurement tests.
# Uses MEASUREMENT_ENTRY_MAP to find the right entry-point when structured layout exists.
_cargo_meas() {
    local targets="$1"
    local filters="$2"
    local -a libtest_args=("${@:3}")

    for arg in "${libtest_args[@]}"; do
        if [[ "$arg" =~ ^-[^-] ]]; then
            echo -e "${RED}${BOLD}❌ Error: malformed libtest argument '$arg' (use double --, not single -)${NC}" >&2
            exit 1
        fi
    done

    local -a tests=($targets)
    if _structural_entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
        local -A _eps=()
        local -a _filters=()
        for _t in "${tests[@]}"; do
            local _ep="${MEASUREMENT_ENTRY_MAP[$_t]:-models}"
            _eps[$_ep]=1
            _filters+=("${_t}::")
        done
        if [ -n "$filters" ]; then
            _filters+=("$filters")
        fi
        local -a _ep_flags=()
        for _ep in "${!_eps[@]}"; do
            _ep_flags+=("--test" "$_ep")
        done
        cargo test --features testing --release "${_ep_flags[@]}" -- "${_filters[@]}" "${libtest_args[@]}"
    else
        local -a _legacy_flags=()
        for _t in "${tests[@]}"; do
            _legacy_flags+=("--test" "$_t")
        done
        if [ -n "$filters" ]; then
            cargo test --features testing --release "${_legacy_flags[@]}" -- "$filters" "${libtest_args[@]}"
        else
            cargo test --features testing --release "${_legacy_flags[@]}" -- "${libtest_args[@]}"
        fi
    fi
}

if _structural_entry_files_exist || [ "${NAM_NEW_ARCH:-0}" = "1" ]; then
    # ── Grouped entry-points format ──────────────────────────────────────────
    # With entry-point modules, all non-ignored structural tests are grouped
    # per entry point — no filter needed. One compilation per entry point.
    #
    # `--skip <module>::` excludes measurement-oracle modules from this DEBUG run.
    # Without it, they ran twice (debug here + release in Phase 2), wasting time
    # and violating phase design (debug floats are non-optimized codegen).
    # The `module::` suffix ensures exact module-prefix matching.
    _struct_targets="models perf_soak parity"
    _struct_flags=()
    for _t in $_struct_targets; do
        _struct_flags+=("--test" "$_t")
    done
    cargo test --features testing --lib "${_struct_flags[@]}" -- \
        --skip golden_vectors:: --skip linear_fft_test:: \
        --skip spectral_fidelity:: --skip reference_oracle_f64:: \
        --skip cpp_parity:: --skip isa_parity:: \
        --skip rt_deadline:: --skip rt_jitter::
else
    # ── Legacy flat-file format ──────────────────────────────────────────────
    cargo test --features testing --lib "${STRUCT_TESTS[@]/#/--test=}"
fi

# ── Phase 2: Measurement Oracles (release, docs/testing.md §7) ──────────────
# Authoritative gate for production floats: measures the optimized codegen path.
# In debug mode, this would measure a non-optimized binary.
phase "Measurement oracles (release — production float gate)..."

# Freshness gate: artifact integrity hard-fails (missing/stale goldens, orphan
# models). Generator-script drift is warn-only here — hard-fail is reserved for
# tests-long / pre-release (see check_freshness artifacts-hard vs hard-fail).
if ! check_freshness artifacts-hard; then
    exit 1
fi

MEASUREMENT_STATUS=0
GOLDEN_RAN=false
CPP_PARITY_SKIPPED=false

# Combines measurement oracles into single cargo invocation per dependency branch,
# preventing repeated recompilation of the release library.
# Branch A — execute when committed dependencies (fixtures) are present.
# Branch B — adds golden_vectors (v1) + isa_parity (v2) when golden fixtures are present.
if [ -f "tests/fixtures/golden_wavenet_standard.bin" ] && [ -f "tests/fixtures/golden_wavenet_standard_v2_48000.bin" ]; then
    GOLDEN_RAN=true
    echo -e "  ${BLUE}→ f64 Oracle + Spectral + Linear FFT + Golden v1 + ISA parity (release, single compilation)...${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test golden_vectors isa_parity" \
        "" \
        --test-threads=1 --nocapture \
        || MEASUREMENT_STATUS=1
else
    echo -e "  ${YELLOW}ⓘ Golden vectors (v1/v2) not found — golden_vectors + isa_parity skipped.${NC}"
    echo -e "  ${YELLOW}  Run './tests/fixtures/golden_gen_build.sh' to generate them.${NC}"
    echo -e "  ${BLUE}→ f64 Oracle + Spectral Fidelity + Linear FFT (release, single compilation)...${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test" \
        "" \
        --nocapture \
        || MEASUREMENT_STATUS=1
fi

# C++ Parity — separate invocation because the `quick_parity` filter
# (required to run only the agile subset) would suppress other oracles if combined.
# Self-skips gracefully if C++ render binary cannot be compiled.
# The NAMCore mirror lives in repo-local third-party/ (resolved in _lib.sh).
# Auto-populate once if missing (soft: SKIP cpp_parity on failure).
ensure_third_party soft || true
if [ -d "$NAM_CORE_DIR" ]; then
    # ── Preventive render compilation ────────────────────────────
    # Build the C++ render binary before cargo test so the CMake build time
    # is isolated from the test output and doesn't trigger mid-phase.
    RENDER_BUILD_DIR="build/namcore_render"
    RENDER_BIN="$RENDER_BUILD_DIR/Release/render"
    if [ ! -f "$RENDER_BIN" ]; then
        RENDER_BIN="$RENDER_BUILD_DIR/Debug/render"
    fi
    SKIP_CPP_PARITY=false
    if [ ! -f "$RENDER_BIN" ]; then
        echo -e "  ${BLUE}→ Compiling C++ render binary preventively...${NC}"
        if [ -z "${CXX:-}" ]; then
            if command -v g++ >/dev/null 2>&1; then
                CXX=g++
            elif command -v clang++ >/dev/null 2>&1; then
                CXX=clang++
            fi
        fi
        if [ -z "$CXX" ]; then
            echo -e "  ${YELLOW}ⓘ C++ compiler not found — skipping cpp_parity.${NC}"
            SKIP_CPP_PARITY=true
        else
            mkdir -p "$RENDER_BUILD_DIR"
            NPROC_CMD=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
            if cmake -S "$NAM_CORE_DIR" -B "$RENDER_BUILD_DIR" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_COMPILER="$CXX" \
                -DCMAKE_CXX_STANDARD=20 \
                -DCMAKE_CXX_FLAGS="-w" \
                -DNAM_ENABLE_A2_FAST=ON > /dev/null 2>&1; then
                if cmake --build "$RENDER_BUILD_DIR" --target render -j"$NPROC_CMD" > /dev/null 2>&1; then
                    echo -e "  ${GREEN}✓ C++ render binary compiled successfully.${NC}"
                else
                    echo -e "  ${YELLOW}ⓘ cmake build failed — skipping cpp_parity.${NC}"
                    SKIP_CPP_PARITY=true
                fi
            else
                echo -e "  ${YELLOW}ⓘ cmake configure failed — skipping cpp_parity.${NC}"
                SKIP_CPP_PARITY=true
            fi
        fi
    fi

    if [ "$SKIP_CPP_PARITY" = true ]; then
        echo -e "  ${YELLOW}ⓘ cpp_parity skipped (C++ render binary unavailable).${NC}"
        CPP_PARITY_SKIPPED=true
    else
        echo -e "  ${BLUE}→ C++ Parity (quick_parity: LSTM + WaveNet CH16 + A2, live NAMCore)...${NC}"
        _cargo_meas "cpp_parity" \
            "quick_parity" \
            --nocapture || MEASUREMENT_STATUS=1
    fi
else
    echo -e "  ${YELLOW}ⓘ NeuralAmpModelerCore not found in $NAM_CORE_DIR.${NC}"
    echo -e "  ${YELLOW}  Run './utils/setup-third-party.sh' to populate the third-party mirrors.${NC}"
    echo -e "  ${YELLOW}  Skipping cpp_parity (live C++ parity).${NC}"
    CPP_PARITY_SKIPPED=true
fi

if [ "$MEASUREMENT_STATUS" -ne 0 ]; then
    echo -e "${RED}${BOLD}❌ Measurement oracle gate (release) failed.${NC}"
    echo -e "${RED}FIDELITY: FAIL${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A (use tests-long.sh)${NC}"
    exit 1
fi

# ── Phase 3: Agile Parser Fuzzing (release, --ignored) ───────────────────────
# Tier 1: parser robustness and safety. Reduced iteration count for first-line speed.
# Configurable via NAM_QUICK_PROPTEST_CASES environment variable.
phase "Agile parser fuzzing (release)..."
PROPTEST_CASES="${NAM_QUICK_PROPTEST_CASES:-1000}" \
    _cargo_meas "proptest_parsers" \
        "" \
        --ignored --nocapture

# ── Summary ──────────────────────────────────────────────────────────────────
if [ "$GOLDEN_RAN" = true ] && [ "$CPP_PARITY_SKIPPED" = false ]; then
    echo -e "${GREEN}${BOLD}================================================================${NC}"
    echo -e "${GREEN}${BOLD}      All quick tests passed! (structural + measurement)         ${NC}"
    echo -e "${GREEN}${BOLD}================================================================${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A (use tests-long.sh)${NC}"
elif [ "$GOLDEN_RAN" = true ]; then
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Quick tests passed (cpp_parity skipped —                     ${NC}"
    echo -e "${YELLOW}${BOLD}     C++ render binary unavailable)                             ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A (use tests-long.sh)${NC}"
elif [ "$CPP_PARITY_SKIPPED" = false ]; then
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Quick tests passed (golden_vectors + isa_parity              ${NC}"
    echo -e "${YELLOW}${BOLD}     skipped — generate golden vectors for full coverage)       ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A (use tests-long.sh)${NC}"
else
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${YELLOW}${BOLD}    Quick tests passed (golden_vectors + isa_parity              ${NC}"
    echo -e "${YELLOW}${BOLD}     and cpp_parity skipped — generate goldens and C++ render    ${NC}"
    echo -e "${YELLOW}${BOLD}     for full coverage)                                         ${NC}"
    echo -e "${YELLOW}${BOLD}================================================================${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A (use tests-long.sh)${NC}"
fi
