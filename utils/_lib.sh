# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# _lib.sh — Common bash utilities for NeuralAmpModeler-rs scripts.
#
# Source with:
#   PHASE_TOTAL=<N>; source "$(dirname "$0")/_lib.sh"
# or for scripts not in utils/:
#   PHASE_TOTAL=<N>; source "$PROJECT_ROOT/utils/_lib.sh"
# or for scripts that manage their own working directory:
#   NAM_LIB_NO_CD=1 PHASE_TOTAL=<N>; source "$(dirname "$0")/_lib.sh"
#
# Then call:
#   phase "Description of the current step"
#   ok    "Success message"
#   warn  "Warning message"
#   die   "Fatal error message"

# ---------------------------------------------------------------------------
# Resolve project root dynamically relative to this helper script
# ---------------------------------------------------------------------------
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$LIB_DIR")"

if [ -z "$PROJECT_DIR" ]; then
    echo -e "\033[0;31m\033[1m[FATAL]\033[0m _lib.sh: could not resolve PROJECT_DIR." >&2
    exit 1
fi

# Automatically enter project root directory, unless NAM_LIB_NO_CD=1 is set
if [ "${NAM_LIB_NO_CD:-0}" != "1" ]; then
    cd "$PROJECT_DIR" || {
        echo -e "\033[0;31m\033[1m[FATAL]\033[0m _lib.sh: failed to cd into project root: $PROJECT_DIR" >&2
        exit 1
    }
fi

# Repo-local third-party area (gitignored): vendor mirrors and optional community_models.
THIRD_PARTY_DIR="${NAM_THIRD_PARTY_DIR:-$PROJECT_DIR/third-party}"
NAM_CORE_DIR="${NAM_CORE_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerCore}"
NAM_PLUGIN_DIR="${NAM_PLUGIN_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerPlugin}"
VARIABLES_ENV="${NAM_VARIABLES_ENV:-$PROJECT_DIR/variables.env}"
SETUP_THIRD_PARTY_SH="${SETUP_THIRD_PARTY_SH:-$PROJECT_DIR/utils/setup-third-party.sh}"

# ---------------------------------------------------------------------------
# ANSI style helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Standard output & control helpers
# ---------------------------------------------------------------------------
PHASE_NUM=0

phase() {
    PHASE_NUM=$((PHASE_NUM + 1))
    echo -e "\n${BLUE}${BOLD}[${PHASE_NUM}/${PHASE_TOTAL:-?}]${NC} $*"
}

ok() {
    echo -e "  ${GREEN}✓${NC} $*"
}

warn() {
    echo -e "  ${YELLOW}ⓘ${NC} $*"
}

die() {
    echo -e "${RED}${BOLD}[FATAL]${NC} $*" >&2
    exit 1
}

# ── Phase receipt machinery (fail-closed foundation) ──────────────────────────
# Each dashboard phase records a typed outcome in JSONL.
# Schema: phase_id, status (PASS|FAIL|SKIP_CAPABILITY|SKIP_OPTIONAL_FIXTURE|NOT_RUN),
#         exit_code, observed_records, expected_records, reason
DASHBOARD_PHASE_RECEIPT="${DASHBOARD_PHASE_RECEIPT:-}"
DASHBOARD_PHASE_HAD_FAILURE=0

# Register a phase receipt entry in the JSONL stream.
dashboard_phase_receipt() {
    local phase_id="$1" status="$2" exit_code="$3" \
          observed_records="$4" expected_records="${5:-0}" reason="${6:-}"
    if [ -z "$DASHBOARD_PHASE_RECEIPT" ]; then
        return 0
    fi
    local observed="${observed_records:-0}"
    local expected="${expected_records:-0}"
    local ecode="${exit_code:-0}"
    local run_id="${NAM_RUN_ID:-}"
    printf '{"phase_id":"%s","status":"%s","exit_code":%s,"observed_records":%s,"expected_records":%s,"reason":"%s","run_id":"%s"}\n' \
        "$phase_id" "$status" "$ecode" "$observed" "$expected" "$reason" "$run_id" >> "$DASHBOARD_PHASE_RECEIPT"
    if [ "$status" = "FAIL" ]; then
        DASHBOARD_PHASE_HAD_FAILURE=1
    fi
}

# ── Single performance-status classifier (F-08 / EP-05) ──────────────────────
# The ONE place where the performance verification outcome is classified.
# Inputs: <receipt_status> <receipt_reason>
# Output (stdout): PASS | NOT_VERIFIED | FAIL
classify_regression_outcome() {
    local status="${1:-}"
    local reason="${2:-}"
    case "${status}:${reason}" in
        PASS:*)
            echo "PASS"
            ;;
        *:MISSING_BASELINE|*:INCOMPARABLE_ENVIRONMENT)
            echo "NOT_VERIFIED"
            ;;
        *)
            echo "FAIL"
            ;;
    esac
}

# Count the number of JSONL metric records currently in a metrics file.
count_jsonl_records() {
    local jsonl="${1:-}"
    [ -n "$jsonl" ] && [ -f "$jsonl" ] || { echo 0; return 0; }
    wc -l < "$jsonl" 2>/dev/null || echo 0
}

# assert_ran_tests <log_file> [min_count]
# Verifies that a test/benchmark log proves real execution.
assert_ran_tests() {
    local log_file="$1"
    local min_count="${2:-1}"

    local total_passed=0

    local passed
    if passed=$(grep -oP 'test result: ok\.\s+\K\d+(?=\s+passed)' "$log_file" 2>/dev/null); then
        for p in $passed; do
            total_passed=$((total_passed + p))
        done
    fi

    local measured
    if measured=$(grep -oP '\K\d+(?=\s+measured)' "$log_file" 2>/dev/null); then
        for m in $measured; do
            total_passed=$((total_passed + m))
        done
    fi

    if [ "$total_passed" -eq 0 ]; then
        local bench_count
        bench_count=$(grep -cP '^\S.*time:\s+\[' "$log_file" 2>/dev/null || true)
        bench_count="${bench_count:-0}"
        total_passed=$bench_count
    fi

    if [ "$total_passed" -lt "$min_count" ]; then
        echo -e "${RED}${BOLD}❌ Gate failed: phase executed 0 tests/benchmarks (empty selection or filter mismatch).${NC}"
        return 1
    fi
    echo -e "  Gate: ${total_passed} test(s)/benchmark(s) executed ≥ ${min_count}  ✓"
    return 0
}

# Run a dashboard phase with strict exit code capture.
run_dashboard_phase() {
    local phase_id="$1" min_records="$2"
    shift 2

    local min_jsonl=0
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        min_jsonl="$1"
        shift
    fi

    local log_path="$LOGDIR/${phase_id}.log"

    echo -e "${BLUE}${BOLD}-> Running ${phase_id}...${NC}"

    local jsonl_before=0
    if [ "$min_jsonl" -gt 0 ]; then
        jsonl_before=$(count_jsonl_records "${NAM_METRICS_JSONL:-}")
    fi

    local start_t end_t exit_code
    start_t=$(date +%s%N)

    set +e
    eval "$@"
    exit_code=$?
    set -e

    end_t=$(date +%s%N)
    local dur_s
    dur_s=$(awk -v ns=$((end_t - start_t)) 'BEGIN { printf "%.1f", ns / 1000000000 }')

    local observed=0
    if [ -f "$log_path" ]; then
        observed=$(wc -l < "$log_path" 2>/dev/null || echo 0)
    fi

    local status="PASS" reason=""
    if [ "$exit_code" -ne 0 ]; then
        status="FAIL"
        reason="subprocess exited with code ${exit_code}"
        echo -e "  ${RED}✗${NC} ${phase_id} failed (exit_code=${exit_code}, ${dur_s}s, ${observed} lines)"
    elif [ "$observed" -lt "$min_records" ] && [ "$min_records" -gt 0 ]; then
        status="FAIL"
        reason="min_records=${min_records} not met (observed=${observed})"
        echo -e "  ${RED}✗${NC} ${phase_id} insufficient records: ${observed}/${min_records} (${dur_s}s)"
    elif ! assert_ran_tests "$log_path" 1; then
        status="FAIL"
        reason="no tests/benchmarks actually executed (empty selection or 100% skip)"
        echo -e "  ${RED}✗${NC} ${phase_id} asserted 0 executed tests/benchmarks (${dur_s}s)"
    elif [ "$min_jsonl" -gt 0 ]; then
        local jsonl_after jsonl_delta
        jsonl_after=$(count_jsonl_records "${NAM_METRICS_JSONL:-}")
        jsonl_delta=$((jsonl_after - jsonl_before))
        if [ "$jsonl_delta" -lt "$min_jsonl" ]; then
            status="FAIL"
            reason="jsonl_records=${jsonl_delta} below minimum ${min_jsonl} (phase emitted no measurement)"
            echo -e "  ${RED}✗${NC} ${phase_id} emitted ${jsonl_delta} JSONL metric record(s), minimum ${min_jsonl} (${dur_s}s)"
        else
            echo -e "  ${GREEN}ok${NC} ${phase_id} completed (${dur_s}s, ${observed} lines, ${jsonl_delta} metric record(s))"
        fi
    else
        echo -e "  ${GREEN}ok${NC} ${phase_id} completed (${dur_s}s, ${observed} lines)"
    fi

    dashboard_phase_receipt "$phase_id" "$status" "$exit_code" "$observed" "$min_records" "$reason"

    return 0
}

# Ensure vendor mirrors exist when a script needs them.
ensure_third_party() {
    local mode="${1:-soft}"

    if [ -d "$NAM_CORE_DIR" ] && [ -e "$NAM_CORE_DIR/.git" ]; then
        return 0
    fi

    if [ "${NAM_SKIP_THIRD_PARTY_SETUP:-0}" = "1" ]; then
        warn "NAM_SKIP_THIRD_PARTY_SETUP=1 — third-party auto-setup skipped."
        return 1
    fi

    if [ ! -x "$SETUP_THIRD_PARTY_SH" ] && [ ! -f "$SETUP_THIRD_PARTY_SH" ]; then
        warn "setup-third-party.sh not found at $SETUP_THIRD_PARTY_SH."
        return 1
    fi

    echo -e "  ${BLUE}→ third-party mirrors missing — running utils/setup-third-party.sh...${NC}"
    if ! bash "$SETUP_THIRD_PARTY_SH"; then
        warn "setup-third-party.sh failed."
        return 1
    fi

    # Re-resolve in case overrides were generated
    THIRD_PARTY_DIR="${NAM_THIRD_PARTY_DIR:-$PROJECT_DIR/third-party}"
    NAM_CORE_DIR="${NAM_CORE_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerCore}"
    NAM_PLUGIN_DIR="${NAM_PLUGIN_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerPlugin}"

    if [ -d "$NAM_CORE_DIR" ]; then
        ok "third-party ready ($NAM_CORE_DIR)."
        return 0
    fi

    if [ "$mode" = "hard" ]; then
        echo -e "  ${RED}${BOLD}❌ NeuralAmpModelerCore still missing at $NAM_CORE_DIR after setup.${NC}"
    else
        warn "NeuralAmpModelerCore still missing at $NAM_CORE_DIR — dependent stages will SKIP."
    fi
    return 1
}

# ── C++ render build — single entry point (S3-T01) ──────────────────────────
# Compiles/verifies the NAMCore `render` binary used by the cpp_parity tests
# (quick_parity + full live_cross_validation matrix) and by golden generation.
# This is THE single implementation of that build: utils/tests-quick.sh,
# utils/tests-long.sh, tests/fixtures/golden_gen_build.sh and the Rust parity
# fallback (tests/parity/cpp_parity.rs) all delegate here.
#
# Knobs:
#   CXX                     compiler; auto-detected g++ → clang++ when unset
#   NAM_RENDER_BUILD_TYPE   CMake build type (default: Release)
#   NAM_RENDER_BUILD_DIR    build dir (default: $PROJECT_DIR/build/namcore_render)
#   NAM_RENDER_JOBS         parallel jobs (default: nproc)
#   NAM_RENDER_FORCE=1      rebuild even when the cached fingerprint matches
#
# Idempotency: when the binary already exists and `$BUILD_DIR/.build_config`
# matches the current toolchain fingerprint (`$CXX:$BUILD_TYPE:$FLAGS`), no
# cmake invocation happens at all. A missing or mismatched fingerprint wipes
# the build dir and rebuilds from scratch (a compiler change invalidates the
# object files and the CMake cache).
#
# Logs: $PROJECT_DIR/target/logs/cmake-configure.log and cmake-build.log.
#
# Stdout: path of the render binary on success (progress goes to stderr).
# Exit codes:
#   0 binary ensured                    1 no C++ compiler available
#   2 cmake not found                   3 NAMCore vendor tree missing
#   4 cmake configure failed            5 cmake build failed
#   6 binary missing after a build that reported success
ensure_namcore_render() {
    local build_dir="${NAM_RENDER_BUILD_DIR:-$PROJECT_DIR/build/namcore_render}"
    local logs_dir="$PROJECT_DIR/target/logs"
    local build_type="${NAM_RENDER_BUILD_TYPE:-Release}"
    local flags="-w -fno-fast-math -ffp-contract=off"
    local jobs="${NAM_RENDER_JOBS:-}"
    local bin=""

    if [ "${NAM_RENDER_FORCE:-0}" = "1" ]; then
        rm -rf "$build_dir"
    fi

    if ! command -v cmake >/dev/null 2>&1; then
        echo -e "  ${YELLOW}${BOLD}WARN: cmake not found — cannot build C++ render binary${NC}" >&2
        return 2
    fi

    if [ -z "${CXX:-}" ]; then
        if command -v g++ >/dev/null 2>&1; then
            CXX=g++
        elif command -v clang++ >/dev/null 2>&1; then
            CXX=clang++
        else
            echo -e "  ${YELLOW}${BOLD}WARN: no C++ compiler — set CXX or install g++/clang++ (C++20)${NC}" >&2
            return 1
        fi
    fi
    if ! command -v "$CXX" >/dev/null 2>&1; then
        echo -e "  ${YELLOW}${BOLD}WARN: CXX=$CXX not found or not executable${NC}" >&2
        return 1
    fi

    if [ ! -d "$NAM_CORE_DIR" ]; then
        echo -e "  ${YELLOW}${BOLD}WARN: NAMCore missing at $NAM_CORE_DIR — run utils/setup-third-party.sh${NC}" >&2
        return 3
    fi

    # Probe existing binary — order mirrors src/testing/fixtures.rs::render_bin_path
    # (Makefiles generator drops the binary in `tools/render`, not `Release/`).
    local cand entry
    for cand in "tools/render" "Release/render" "Debug/render"; do
        if [ -x "$build_dir/$cand" ]; then
            bin="$build_dir/$cand"
            break
        fi
    done
    if [ -z "$bin" ] && [ -d "$build_dir" ]; then
        for entry in "$build_dir"/*/; do
            [ -d "$entry" ] || continue
            if [ -x "$entry/render" ]; then
                bin="$entry/render"
                break
            fi
        done
    fi

    local config_file="$build_dir/.build_config"
    local fingerprint="$CXX:$build_type:$flags"
    if [ -n "$bin" ] && [ -f "$config_file" ] \
        && [ "$(cat "$config_file" 2>/dev/null || true)" = "$fingerprint" ]; then
        echo -e "  ${BLUE}→ C++ render binary up-to-date: $bin${NC}" >&2
        echo "$bin"
        return 0
    fi

    if [ -n "$bin" ] || [ -d "$build_dir" ]; then
        echo -e "  ${YELLOW}render binary present but toolchain fingerprint changed (or missing) — rebuilding from scratch${NC}" >&2
        rm -rf "$build_dir"
    fi
    mkdir -p "$build_dir" "$logs_dir"

    if [ -z "$jobs" ]; then
        jobs=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
    fi
    echo -e "  ${BLUE}→ compiling C++ NAMCore render ($CXX, $build_type, -j$jobs)${NC}" >&2

    if ! cmake -S "$NAM_CORE_DIR" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_CXX_STANDARD=20 \
        -DCMAKE_CXX_FLAGS="$flags" \
        -DNAM_ENABLE_A2_FAST=ON > "$logs_dir/cmake-configure.log" 2>&1; then
        echo -e "${RED}${BOLD}ERROR: cmake configure failed — log: $logs_dir/cmake-configure.log${NC}" >&2
        tail -20 "$logs_dir/cmake-configure.log" >&2 2>/dev/null || true
        return 4
    fi

    if ! cmake --build "$build_dir" --target render -j"$jobs" \
        > "$logs_dir/cmake-build.log" 2>&1; then
        echo -e "${RED}${BOLD}ERROR: cmake build failed — log: $logs_dir/cmake-build.log${NC}" >&2
        tail -20 "$logs_dir/cmake-build.log" >&2 2>/dev/null || true
        return 5
    fi

    # Re-probe: the generator decides the final location (tools/ by default).
    bin=""
    for cand in "tools/render" "Release/render" "Debug/render"; do
        if [ -x "$build_dir/$cand" ]; then
            bin="$build_dir/$cand"
            break
        fi
    done
    if [ -z "$bin" ] && [ -d "$build_dir" ]; then
        for entry in "$build_dir"/*/; do
            [ -d "$entry" ] || continue
            if [ -x "$entry/render" ]; then
                bin="$entry/render"
                break
            fi
        done
    fi
    if [ -z "$bin" ]; then
        echo -e "${RED}${BOLD}ERROR: render binary not found under $build_dir after a successful build${NC}" >&2
        return 6
    fi

    printf '%s\n' "$fingerprint" > "$config_file"
    echo -e "  ${GREEN}OK C++ render binary: $bin${NC}" >&2
    echo "$bin"
    return 0
}

# ── Toolchain fingerprint check (F-I4 / Tarefa 3.2) ──────────────────────────
# Compares current toolchain against # TOOLCHAIN: lines in manifest.
check_toolchain_fingerprint() {
    local MANIFEST="tests/fixtures/.golden_manifest.sha256"

    if [ ! -f "$MANIFEST" ]; then
        return 0
    fi

    local CXX_NOW CMAKE_NOW GLIBC_NOW OS_NOW
    CXX_NOW=$(${CXX:-g++} --version 2>/dev/null | head -1 || echo "unknown")
    CMAKE_NOW=$(cmake --version 2>/dev/null | head -1 || echo "unknown")
    if GLIBC_NOW=$(ldd --version 2>/dev/null | head -1); then :; else
        GLIBC_NOW=$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "unknown")
    fi
    OS_NOW=$(uname -r 2>/dev/null || echo "unknown")

    local mismatch=0
    local F_CXX="" F_GLIBC="" F_CMAKE="" F_OS="" F_FLAGS=""
    while IFS= read -r line; do
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx:\ (.*)$ ]]      && F_CXX="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cmake:\ (.*)$ ]]     && F_CMAKE="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ glibc:\ (.*)$ ]]     && F_GLIBC="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ os:\ (.*)$ ]]        && F_OS="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx-flags:\ (.*)$ ]] && F_FLAGS="${BASH_REMATCH[1]}"
    done < "$MANIFEST"

    if [ -n "$F_CXX" ] && [ "$F_CXX" != "$CXX_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: compiler changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_CXX${NC}"
        echo -e "    ${YELLOW}now:      $CXX_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_GLIBC" ] && [ "$F_GLIBC" != "$GLIBC_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: glibc changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_GLIBC${NC}"
        echo -e "    ${YELLOW}now:      $GLIBC_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_CMAKE" ] && [ "$F_CMAKE" != "$CMAKE_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: cmake changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_CMAKE${NC}"
        echo -e "    ${YELLOW}now:      $CMAKE_NOW${NC}"
        mismatch=1
    fi
    if [ -n "$F_OS" ] && [ "$F_OS" != "$OS_NOW" ]; then
        echo -e "  ${YELLOW}⚠ TOOLCHAIN DRIFT: kernel changed since golden generation${NC}"
        echo -e "    ${YELLOW}manifest: $F_OS${NC}"
        echo -e "    ${YELLOW}now:      $OS_NOW${NC}"
        mismatch=1
    fi

    return "$mismatch"
}

# ── Centralized freshness gate (F-X4 / S3-T03) ───────────────────────────────
# Validates golden manifest integrity against models, fixtures and generators.
# The heavy lifting now lives in Rust (src/testing/freshness.rs) so the shell
# wrapper is just a thin, portable adapter.
check_freshness() {
    local mode="${1:-hard-fail}"
    if [ "${NAM_BYPASS_FRESHNESS:-0}" = "1" ]; then
        echo -e "  ${YELLOW}⚠ NAM_BYPASS_FRESHNESS=1 — freshness check skipped${NC}"
        return 0
    fi
    local bin="${NAM_FRESHNESS_BIN:-$PROJECT_DIR/target/debug/nam_freshness}"
    if [ ! -x "$bin" ]; then
        if ! ( cd "$PROJECT_DIR" && cargo build --quiet --features testing --bin nam_freshness >/dev/null 2>&1 ); then
            echo -e "  ${RED}${BOLD}❌ FATAL: failed to build nam_freshness${NC}" >&2
            return 1
        fi
    fi
    "$bin" --root "$PWD" "$mode"
}

# ── Typed freshness gate (F-22) ─────────────────────────────────────────────
# Reports a machine-readable outcome via the global FRESHNESS_REASON variable.
run_freshness_gate() {
    local mode="${1:-artifacts-hard}"
    local log
    log="$(mktemp "${TMPDIR:-/tmp}/nam-freshness-XXXXXX")"
    FRESHNESS_REASON="OK"
    check_freshness "$mode" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        if grep -q 'STALE' "$log"; then
            FRESHNESS_REASON="STALE_FIXTURES"
        elif grep -q 'MISSING' "$log"; then
            FRESHNESS_REASON="MISSING_FIXTURES"
        elif grep -q 'ORPHAN' "$log"; then
            FRESHNESS_REASON="ORPHAN_FIXTURE"
        else
            FRESHNESS_REASON="FRESHNESS_FAILED"
        fi
        cat "$log"
    fi
    rm -f "$log"
    return "$rc"
}
