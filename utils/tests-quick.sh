#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Quick QA Suite — first-line cargo-test gate.
#
# Bash is the trigger + receipt printer. Cargo/Rust owns test selection,
# skip policy, and numeric gates. See docs/testing.md.
#
#   utils/lints.sh        static (not duplicated)
#   utils/tests-quick.sh  this script (~few minutes; hardware-dependent)
#   utils/tests-long.sh   ignored / soak / full matrix (nightly)
#
# Axis A: non-ignored = quick; #[ignore] = long.
# Axis B: structural = debug; float oracles = --release.
#
# Exit 1 on any failed test. Missing optional oracles are WARN (exit 0)
# unless NAM_QUICK_STRICT=1, which promotes those gaps to FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PHASE_TOTAL=3
source "$SCRIPT_DIR/_lib.sh"

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
        echo -e "${YELLOW}WARN: restarting with low CPU/IO priority (NAM_NO_LOW_PRIORITY=1 to skip)${NC}"
        exec $CMD_PREFIX "$SCRIPT_PATH" "$@"
    fi
fi

trap 'echo -e "\n${RED}${BOLD}FAIL: unexpected error: \"$BASH_COMMAND\" at line $LINENO status $?.${NC}"; exit 1' ERR

mkdir -p target/logs
rm -f target/logs/quick-phase1.log \
      target/logs/quick-phase2.log \
      target/logs/quick-phase3.log \
      target/logs/quick-receipt.txt

emit() {
    printf '%s\n' "$1" | tee -a target/logs/quick-receipt.txt
}

echo -e "${BLUE}${BOLD}NeuralAmpModeler-rs Quick QA${NC}"
emit "SUITE: tests-quick"
emit "STRICT: ${NAM_QUICK_STRICT:-0}"

# ── Phase 1: Structural (debug) ─────────────────────────────────────────────
# --lib plus every integration entry that holds non-ignored structural tests.
# Measurement-oracle modules are skipped here (Axis B: they run --release).
phase "Structural: unit + deterministic integration (debug)"

{
    # Axis-B: lstm_activation_precision SNR oracles are release-only (Phase 2).
    # Substring skip covers both `..._gain` and `..._gain_stress_v2`; the
    # module's structural checks still run in this phase.
    cargo test --features testing --lib \
        --test models --test perf_soak --test parity --test dsp_core \
        --test target_features_compliance_test --test libm_export_guard -- \
        --skip golden_vectors:: --skip linear_fft_test:: \
        --skip spectral_fidelity:: --skip reference_oracle_f64:: \
        --skip cpp_parity:: --skip isa_parity:: \
        --skip rt_deadline:: --skip rt_jitter:: \
        --skip lstm_activation_precision::test_lstm_activation_precision_gain
} 2>&1 | tee target/logs/quick-phase1.log

assert_ran_tests target/logs/quick-phase1.log 1
emit "PHASE1: PASS log=target/logs/quick-phase1.log"

# ── Phase 2: Measurement oracles (release) ──────────────────────────────────
phase "Measurement oracles (release — production float gate)"

if ! check_freshness artifacts-hard; then
    emit "PHASE2: FAIL reason=freshness"
    echo -e "${RED}FIDELITY: FAIL${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A${NC}"
    exit 1
fi

GOLDEN_RAN=0
CPP_PARITY_RAN=0
declare -a GAPS=()

_cargo_meas() {
    local -a tests=($1)
    shift
    local -a libtest_args=("$@")
    local -A eps=()
    local -a filters=()
    local t ep
    for t in "${tests[@]}"; do
        case "$t" in
            reference_oracle_f64|isa_parity|cpp_parity) ep="parity" ;;
            *) ep="models" ;;
        esac
        eps[$ep]=1
        filters+=("${t}::")
    done
    local -a flags=()
    for ep in "${!eps[@]}"; do
        flags+=(--test "$ep")
    done
    cargo test --features testing --release "${flags[@]}" -- "${filters[@]}" "${libtest_args[@]}"
}

if [ -f "tests/fixtures/golden_wavenet_standard.bin" ] \
    && [ -f "tests/fixtures/golden_wavenet_standard_v2_48000.bin" ]; then
    GOLDEN_RAN=1
    echo -e "  ${BLUE}→ f64 + spectral + linear_fft + golden v1 + isa_parity + lstm_activation_precision${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test golden_vectors isa_parity lstm_activation_precision" \
        --test-threads=1 --nocapture \
        2>&1 | tee target/logs/quick-phase2.log
else
    GAPS+=("golden_vectors+isa_parity:missing_fixtures")
    echo -e "${YELLOW}${BOLD}WARN: golden v1/v2 fixtures missing — golden_vectors + isa_parity SKIPPED${NC}"
    echo -e "${YELLOW}  DIAGNOSTIC: run tests/fixtures/golden_gen_build.sh${NC}"
    echo -e "  ${BLUE}→ f64 + spectral + linear_fft + lstm_activation_precision (no golden deps)${NC}"
    _cargo_meas "reference_oracle_f64 spectral_fidelity linear_fft_test lstm_activation_precision" \
        --nocapture \
        2>&1 | tee target/logs/quick-phase2.log
fi

assert_ran_tests target/logs/quick-phase2.log 1

ensure_third_party soft || true
SKIP_CPP=0
if RENDER_BIN="$(ensure_namcore_render)"; then
    :
else
    RC_RENDER=$?
    case "$RC_RENDER" in
        1) GAPS+=("cpp_parity:no_cxx") ;;
        2) GAPS+=("cpp_parity:no_cmake") ;;
        3) GAPS+=("cpp_parity:no_namcore") ;;
        4|5|6) GAPS+=("cpp_parity:cmake_failed") ;;
        *) GAPS+=("cpp_parity:cmake_failed") ;;
    esac
    SKIP_CPP=1
    echo -e "${YELLOW}${BOLD}WARN: C++ render binary unavailable (ensure_namcore_render exit $RC_RENDER) — cpp_parity SKIPPED${NC}"
    echo -e "${YELLOW}  DIAGNOSTIC: target/logs/cmake-configure.log target/logs/cmake-build.log${NC}"
fi
if [ "$SKIP_CPP" -eq 0 ]; then
    echo -e "  ${BLUE}→ cpp_parity quick_parity (render: $RENDER_BIN)${NC}"
    cargo test --features testing --release --test parity -- \
        quick_parity --nocapture \
        2>&1 | tee -a target/logs/quick-phase2.log
    CPP_PARITY_RAN=1
    assert_ran_tests target/logs/quick-phase2.log 1
fi

emit "PHASE2: PASS golden=${GOLDEN_RAN} cpp_parity=${CPP_PARITY_RAN} log=target/logs/quick-phase2.log"

# ── Phase 3: Parser fuzz (release, capped, --ignored) ───────────────────────
phase "Agile parser fuzzing (release, PROPTEST_CASES=${NAM_QUICK_PROPTEST_CASES:-1000})"
PROPTEST_CASES="${NAM_QUICK_PROPTEST_CASES:-1000}" \
    _cargo_meas "proptest_parsers" --ignored --nocapture \
    2>&1 | tee target/logs/quick-phase3.log
assert_ran_tests target/logs/quick-phase3.log 1
emit "PHASE3: PASS log=target/logs/quick-phase3.log"

# ── Receipt ─────────────────────────────────────────────────────────────────
if [ ${#GAPS[@]} -gt 0 ]; then
    for g in "${GAPS[@]}"; do
        emit "GAP: $g"
        echo -e "${YELLOW}${BOLD}WARN GAP: $g${NC}"
    done
    echo -e "${YELLOW}FIDELITY: INCOMPLETE${NC}"
    echo -e "${BLUE}PERFORMANCE: N/A${NC}"
    echo -e "${YELLOW}${BOLD}OVERALL: PASSED_WITH_GAPS${NC}"
    emit "OVERALL: PASSED_WITH_GAPS"
    if [ "${NAM_QUICK_STRICT:-0}" = "1" ]; then
        echo -e "${RED}${BOLD}FAIL: NAM_QUICK_STRICT=1 treats gaps as failure${NC}"
        emit "OVERALL: FAIL reason=strict_gaps"
        exit 1
    fi
    exit 0
fi

echo -e "${GREEN}FIDELITY: OK${NC}"
echo -e "${BLUE}PERFORMANCE: N/A${NC}"
emit "OVERALL: PASSED"
