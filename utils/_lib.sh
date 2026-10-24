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
#
# ── Contract ────────────────────────────────────────────────────────────────
# This library is a SHELL HUSK: it keeps only the glue that has no Rust home
# yet and delegates every interpretation to the QA binaries. No `eval`, no
# hand-serialized JSON (the old brace-printf is gone), no PCRE (grep -oP).
#
# Consumers (9 `source`s):
#   1. utils/tests-quick.sh
#   2. utils/lints.sh
#   3. utils/mod-update.sh
#   4. utils/setup-third-party.sh
#   5. utils/ensure_namcore_render.sh
#   6. tests/fixtures/golden_gen_build.sh
#   7. utils/quality-dashboard.sh
#   8. utils/tests-long.sh
#   9. utils/tests-performance-regression.sh
#
# What remains here:
#   * ANSI colors + phase/ok/warn/die
#   * PROJECT_DIR + third-party paths
#   * maybe_restart_low_priority (nice/ionice re-exec, opt-out via env)
#   * ensure_third_party (mirror provisioning)
#   * ensure_namcore_render (C++ render build)
#   * ensure_qa_bin (on-demand QA binary build & resolution with freshness guarantee)
#   * ensure_long_receipt_bin (on-demand nam_long_receipt build)
#   * Thin wrappers over the QA binaries:
#     - assert_ran_tests        → nam_long_receipt count-log
#     - assert_subphase_ran     → shared count-log core (subphase wording)
#     - dashboard_phase_receipt → nam_quality receipt append
#     - check_freshness → nam_freshness
#
# Delegated to Rust:
#   * classify_regression_outcome → src/testing/qa/classify.rs
#   * check_toolchain_fingerprint → freshness.rs ToolchainFingerprint
#
# Project-neutral helpers (ANSI colors, phase/ok/warn/die,
# format_duration_ms/phase_elapsed_str, emit/emit_to, assert_ran_tests/
# assert_subphase_ran, pick_bench_core) live in the canonical, byte-identical
# `_lib_core.sh` block (drift-pinned by utils/test-lib-drift.sh). This file
# adds only the engine-specific glue on top.

# Canonical project-neutral core (drift-pinned; see utils/test-lib-drift.sh).
source "$(dirname "${BASH_SOURCE[0]}")/_lib_core.sh"

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
# Low-priority re-exec (shared by lints.sh and tests-quick.sh)
# ---------------------------------------------------------------------------
# maybe_restart_low_priority <script_path> [args...]
#   Re-executes the caller under nice/ionice (best-effort, skipping whatever
#   is unavailable) so long QA runs stay polite to the operator's machine.
#   Skips when already restarted (NAM_LOW_PRIORITY=1) or explicitly disabled
#   (NAM_NO_LOW_PRIORITY=1).
#
#   Deliberately NOT used by quality-dashboard.sh nor
#   tests-performance-regression.sh: both drive the Criterion statistical
#   benchmarks, and a de-prioritized bench loses CPU to same-priority load on
#   the pinned core — inflating timings toward false regressions. Benchmarks
#   must run at normal priority; only build/test-only suites self-deprioritize.
maybe_restart_low_priority() {
    if [ "${NAM_LOW_PRIORITY:-0}" = "1" ] || [ "${NAM_NO_LOW_PRIORITY:-0}" = "1" ]; then
        return 0
    fi
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
        # Intentional unquoted expansion: CMD_PREFIX is "nice -n 19 [ionice -c 3]".
        exec $CMD_PREFIX "$@"
    fi
}

# ---------------------------------------------------------------------------
# Engine-specific output helpers (UI/governance helpers live in _lib_core.sh)
# ---------------------------------------------------------------------------
log_warn() {
    warn "$@"
}

json_escape_string() {
    # Escapes backslashes and double quotes for safe JSON string embedding.
    local s="$1"
    printf '%s' "$s" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

# ---------------------------------------------------------------------------
# QA Binary Freshness & Resolution Helper
# ---------------------------------------------------------------------------
# ensure_qa_bin <bin_name> [features] [override_var]
#   Resolves and guarantees freshness for an internal QA/testing binary via
#   cargo's machine-readable compiler-artifact message stream.
#
#   If the binary path was explicitly provided by the operator via environment
#   (e.g., NAM_QUALITY_BIN, NAM_PERF_GATE, etc.):
#     - Emits a warning to stderr: "operator-supplied binary, freshness not verified: ..."
#     - Verifies executable permissions (fail-closed if missing)
#     - Returns the path directly without invoking cargo.
#
#   Otherwise:
#     - Runs `cargo build --locked [--features <features>] --bin <bin_name> --message-format=json-render-diagnostics`
#     - Parses the compiler-artifact stream to locate the exact executable path.
#     - Fails closed with a clear diagnostic if compilation fails or the artifact is missing/unusable.
ensure_qa_bin() {
    local bin_name="$1"
    local features="${2:-testing}"
    local override_var="${3:-}"

    if [ -z "$override_var" ]; then
        case "$bin_name" in
            nam_quality) override_var="NAM_QUALITY_BIN" ;;
            nam_perf_gate) override_var="NAM_PERF_GATE" ;;
            nam_long_receipt) override_var="NAM_LONG_RECEIPT_BIN" ;;
            nam_freshness) override_var="NAM_FRESHNESS_BIN" ;;
        esac
    fi

    if [ -n "$override_var" ]; then
        local override_val="${!override_var:-}"
        if [ -n "$override_val" ]; then
            echo -e "  ${YELLOW}ⓘ${NC} operator-supplied binary, freshness not verified: $override_val ($override_var)" >&2
            if [ ! -x "$override_val" ]; then
                die "operator-supplied binary is not executable: $override_val"
            fi
            echo "$override_val"
            return 0
        fi
    fi

    local build_output build_rc=0
    local -a cargo_cmd=(cargo build --locked)
    if [ -n "$features" ]; then
        cargo_cmd+=(--features "$features")
    fi
    cargo_cmd+=(--bin "$bin_name" --message-format=json-render-diagnostics)

    build_output=$(cd "$PROJECT_DIR" && "${cargo_cmd[@]}" 2>&1) || build_rc=$?
    if [ "$build_rc" -ne 0 ]; then
        echo -e "  ${RED}${BOLD}❌ FATAL: cargo build failed for QA binary ${bin_name} (rc=${build_rc}):${NC}" >&2
        echo "$build_output" >&2
        die "compilation failed for QA binary ${bin_name}"
    fi

    local exe
    exe=$(printf '%s\n' "$build_output" \
        | grep -F '"reason":"compiler-artifact"' \
        | grep -F "\"name\":\"$bin_name\"" \
        | grep -F '"executable":"' \
        | tail -1 \
        | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p')

    if [ -z "$exe" ] || [ ! -x "$exe" ]; then
        echo -e "  ${RED}${BOLD}❌ FATAL: executable artifact not found in cargo compiler-artifact stream for ${bin_name}:${NC}" >&2
        echo "$build_output" >&2
        die "missing executable artifact for ${bin_name}"
    fi

    echo "$exe"
}

# ── Phase receipt machinery (fail-closed foundation) ──────────────────────────
# Each dashboard phase records a typed outcome in JSONL.
# Schema: phase_id, status (PASS|FAIL|SKIP_CAPABILITY|SKIP_OPTIONAL_FIXTURE|NOT_RUN|NOT_VERIFIED),
#         exit_code, observed_records, expected_records, reason, run_id
# The JSON line is serialized by `nam_quality receipt append` (serde) —
# the shell never hand-serializes.
DASHBOARD_PHASE_RECEIPT="${DASHBOARD_PHASE_RECEIPT:-}"
DASHBOARD_PHASE_HAD_FAILURE=0

# Register a phase receipt entry in the JSONL stream (delegates to the bin).
# Strictly fail-closed: a receipt-append failure aborts the script (a
# run whose receipt stream is lost must never be certified as clean).
dashboard_phase_receipt() {
    local phase_id="$1" status="$2" exit_code="${3:-0}" \
          observed_records="${4:-0}" expected_records="${5:-0}" reason="${6:-}"
    if [ -z "$DASHBOARD_PHASE_RECEIPT" ]; then
        warn "DASHBOARD_PHASE_RECEIPT not set — receipt stream lost"
        return 1
    fi
    if [ -z "${NAM_QUALITY_BIN:-}" ] || [ ! -x "$NAM_QUALITY_BIN" ]; then
        NAM_QUALITY_BIN=$(ensure_qa_bin nam_quality testing) || die "failed to resolve nam_quality — phase receipt for ${phase_id} not recorded (fail-closed)"
        export NAM_QUALITY_BIN
    fi
    local bin="$NAM_QUALITY_BIN"
    if ! "$bin" receipt append --phase-id "$phase_id" --status "$status" \
        --exit-code "$exit_code" --observed-records "$observed_records" \
        --expected-records "$expected_records" --reason "$reason" \
        --run-id "${NAM_RUN_ID:-}" --out "$DASHBOARD_PHASE_RECEIPT" >/dev/null 2>&1; then
        die "nam_quality receipt append failed for ${phase_id} — phase receipt not recorded (fail-closed)"
    fi
    if [ "$status" = "FAIL" ]; then
        DASHBOARD_PHASE_HAD_FAILURE=1
    fi
}

# Count the number of JSONL metric records currently in a metrics file.
# POSIX `wc -l` — plain line counting, no PCRE needed.
count_jsonl_records() {
    local jsonl="${1:-}"
    [ -n "$jsonl" ] && [ -f "$jsonl" ] || { echo 0; return 0; }
    wc -l < "$jsonl" 2>/dev/null || echo 0
}

# Structured long-audit receipt machinery (shared with tests-long.sh and
# remote-simd-gate.sh): resolve the emitter binary once and build it on demand.
LONG_RECEIPT_BIN=""

# ensure_long_receipt_bin
#   Idempotently guarantees $LONG_RECEIPT_BIN exists and is executable,
#   resolving it via ensure_qa_bin.
#   Fail-closed: returns 1 (never a silent skip) when the binary cannot be
#   provided — receipt emission is mandatory, not best-effort.
ensure_long_receipt_bin() {
    if [ -n "$LONG_RECEIPT_BIN" ] && [ -x "$LONG_RECEIPT_BIN" ]; then
        return 0
    fi
    LONG_RECEIPT_BIN=$(ensure_qa_bin nam_long_receipt testing) || return 1
    export LONG_RECEIPT_BIN
    return 0
}

# _nam_count_log_total <log_file>
#   Engine override of the canonical core hook: delegates the counting
#   (libtest `passed`/`measured` counters + Criterion `time:` fallback) to the
#   Rust QA binary (src/testing/receipt.rs::count_tests_executed_from_log) —
#   no `grep -oP`. Prints the total on stdout and returns 1 (fail-closed) when
#   the binary cannot be provided; the core gates (`assert_ran_tests` /
#   `assert_subphase_ran`) surface the failure with a generic notice.
_nam_count_log_total() {
    local log_file="$1"
    ensure_long_receipt_bin || return 1
    "$LONG_RECEIPT_BIN" count-log --log "$log_file" 2>/dev/null || echo 0
}

# Run a dashboard phase with strict exit code capture.
# The command is passed as an ARGUMENT ARRAY (no `eval`); the command's
# stdout+stderr are redirected internally to $LOGDIR/$phase_id.log.
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
    "$@" > "$log_path" 2>&1
    exit_code=$?
    set -e

    end_t=$(date +%s%N)
    local dur_ms=$(( (end_t - start_t) / 1000000 ))
    local dur_str
    dur_str=$(format_duration_ms "$dur_ms")

    local observed=0
    if [ -f "$log_path" ]; then
        observed=$(wc -l < "$log_path" 2>/dev/null || echo 0)
    fi

    local status="PASS" reason=""
    if [ "$exit_code" -ne 0 ]; then
        status="FAIL"
        reason="subprocess exited with code ${exit_code}"
        echo -e "  ${RED}✗${NC} ${phase_id} failed (exit_code=${exit_code}, ${dur_str}, ${observed} lines)"
    elif [ "$observed" -lt "$min_records" ] && [ "$min_records" -gt 0 ]; then
        status="FAIL"
        reason="min_records=${min_records} not met (observed=${observed})"
        echo -e "  ${RED}✗${NC} ${phase_id} insufficient records: ${observed}/${min_records} (${dur_str})"
    elif ! assert_ran_tests "$log_path" 1; then
        status="FAIL"
        reason="no tests/benchmarks actually executed (empty selection or 100% skip)"
        echo -e "  ${RED}✗${NC} ${phase_id} asserted 0 executed tests/benchmarks (${dur_str})"
    elif [ "$min_jsonl" -gt 0 ]; then
        local jsonl_after jsonl_delta
        jsonl_after=$(count_jsonl_records "${NAM_METRICS_JSONL:-}")
        jsonl_delta=$((jsonl_after - jsonl_before))
        if [ "$jsonl_delta" -lt "$min_jsonl" ]; then
            status="FAIL"
            reason="jsonl_records=${jsonl_delta} below minimum ${min_jsonl} (phase emitted no measurement)"
            echo -e "  ${RED}✗${NC} ${phase_id} emitted ${jsonl_delta} JSONL metric record(s), minimum ${min_jsonl} (${dur_str})"
        else
            echo -e "  ${GREEN}ok${NC} ${phase_id} completed (${dur_str}, ${observed} lines, ${jsonl_delta} metric record(s))"
        fi
    else
        echo -e "  ${GREEN}ok${NC} ${phase_id} completed (${dur_str}, ${observed} lines)"
    fi

    dashboard_phase_receipt "$phase_id" "$status" "$exit_code" "$observed" "$min_records" "$reason"

    if [ "$status" = "FAIL" ] || [ "${DASHBOARD_PHASE_HAD_FAILURE:-0}" -ne 0 ]; then
        return 1
    fi
    return 0
}

# Ensure vendor mirrors exist when a script needs them.
ensure_third_party() {
    local mode="${1:-soft}"

    # Ready only when the checkout AND its submodules are populated: a setup
    # run interrupted after the clone (e.g. a transient GitLab error while
    # fetching Dependencies/eigen) must be retried, not treated as ready.
    if [ -d "$NAM_CORE_DIR" ] && [ -e "$NAM_CORE_DIR/.git" ] \
        && [ -n "$(ls -A "$NAM_CORE_DIR/Dependencies/eigen" 2>/dev/null || true)" ] \
        && [ -n "$(ls -A "$NAM_CORE_DIR/Dependencies/AudioDSPTools" 2>/dev/null || true)" ]; then
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

# ── C++ render build — single entry point ───────────────────────────────────
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
    if [ -z "$build_dir" ] || [ "$build_dir" = "/" ] || [ "$build_dir" = "." ]; then
        die "invalid or dangerous NAM_RENDER_BUILD_DIR: '$build_dir'"
    fi
    local logs_dir="${NAM_RENDER_LOGS_DIR:-$PROJECT_DIR/target/logs}"
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

    local mc_cpp="$PROJECT_DIR/tests/fixtures/render_multichannel.cpp"
    local mc_bin="$build_dir/render_multichannel"
    if [ -f "$mc_cpp" ]; then
        local obj_files
        obj_files=$(find "$build_dir/CMakeFiles/NeuralAmpModelerCore.dir" -name "*.o" 2>/dev/null || true)
        if [ -n "$obj_files" ]; then
            "$CXX" -std=c++20 -O3 \
                -I "$NAM_CORE_DIR" \
                -I "$NAM_CORE_DIR/Dependencies/eigen" \
                -I "$NAM_CORE_DIR/Dependencies/AudioDSPTools" \
                -I "$NAM_CORE_DIR/Dependencies/nlohmann" \
                "$mc_cpp" $obj_files \
                -o "$mc_bin" > "$logs_dir/compile-render-multichannel.log" 2>&1 || true
        fi
    fi

    echo "$bin"
    return 0
}

# ── Centralized freshness gate ───────────────────────────────────────────────
# Validates golden manifest integrity against models, fixtures and generators.
# The heavy lifting now lives in Rust (src/testing/freshness.rs, incl. the
# `# TOOLCHAIN:` drift check that replaced the bash check_toolchain_fingerprint)
# so the shell wrapper is just a thin, portable adapter.
# Note on NAM_BYPASS_FRESHNESS=1: Callers (utils/tests-quick.sh,
# utils/tests-long.sh, utils/quality-dashboard.sh) record an explicit typed
# gap (e.g. freshness:bypassed_by_env / freshness_bypassed_by_env /
# bypass:freshness) in their receipts.
check_freshness() {
    local mode="${1:-hard-fail}"
    if [ "${NAM_BYPASS_FRESHNESS:-0}" = "1" ]; then
        echo -e "  ${YELLOW}⚠ NAM_BYPASS_FRESHNESS=1 — freshness check skipped (gap recorded by caller)${NC}"
        return 0
    fi
    local bin
    bin=$(ensure_qa_bin nam_freshness testing)
    "$bin" --root "$PWD" "$mode"
}
