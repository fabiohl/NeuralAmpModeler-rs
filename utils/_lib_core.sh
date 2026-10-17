# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# _lib_core.sh — canonical, project-neutral QA shell core for the NAM ecosystem.
#
# Sourced by each repository's `_lib.sh`. The region between the BEGIN/END
# markers below is byte-identical across NeuralAmpModeler-rs, NAM-Plug and
# NAM-Audio-Pipe; its integrity is pinned by the embedded `# sha256:` line and
# re-checked by `utils/test-lib-drift.sh` (drift = test failure).
#
# Keep the region project-neutral: no host/plugin/application names and no
# project-specific paths. Repo-specific helpers (project-root resolution,
# third-party setup, QA-binary resolution, ...) live in `_lib.sh`, outside the
# markers.

# >>> NAM-LIB-CORE BEGIN >>>
# sha256: 9047a6327095cd1b279f30eece87652453b7d7175b183805779f33c5324f4313
# Shared canonical block for the NAM ecosystem QA shell library.
# ---------------------------------------------------------------------------
# ANSI style helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Phase counter (global — not safe for concurrent subshells)
# ---------------------------------------------------------------------------
PHASE_NUM=0
PHASE_START_NS=0

# phase <description>
#   Increments and prints a phase header: [N/TOTAL] description
phase() {
    PHASE_NUM=$((PHASE_NUM + 1))
    PHASE_START_NS=$(date +%s%N 2>/dev/null || echo 0)
    echo -e "\n${BLUE}${BOLD}[${PHASE_NUM}/${PHASE_TOTAL:-?}]${NC} $*"
}

# die <message>
#   Prints a fatal error to stderr and exits with code 1.
die() {
    echo -e "${RED}${BOLD}[FATAL]${NC} $*" >&2
    exit 1
}

# ok <message>
#   Prints an indented green OK confirmation line.
ok() {
    echo -e "  ${GREEN}OK${NC} $*"
}

# warn <message>
#   Prints an indented yellow informational/warning line.
warn() {
    echo -e "  ${YELLOW}ⓘ${NC} $*"
}

# format_duration_ms <milliseconds>
#   Formats an interval in ms to human-readable scale:
#   < 1000 ms: "Xms" (e.g. 790ms, 45ms)
#   < 10000 ms: "X.YYs" (e.g. 1.35s)
#   >= 10000 ms: "X.Ys" (e.g. 65.2s)
format_duration_ms() {
    local ms="${1:-0}"
    if [ "$ms" -lt 1000 ]; then
        echo "${ms}ms"
    elif [ "$ms" -lt 10000 ]; then
        local sec=$(( ms / 1000 ))
        local dec=$(( (ms % 1000) / 10 ))
        printf "%d.%02ds\n" "$sec" "$dec"
    else
        local sec=$(( ms / 1000 ))
        local dec=$(( (ms % 1000) / 100 ))
        printf "%d.%ds\n" "$sec" "$dec"
    fi
}

# phase_elapsed_str
#   Returns the formatted duration elapsed since the last phase() call.
phase_elapsed_str() {
    if [ "${PHASE_START_NS:-0}" -ne 0 ]; then
        local now_ns dur_ms
        now_ns=$(date +%s%N 2>/dev/null || echo 0)
        if [ "$now_ns" -ge "$PHASE_START_NS" ]; then
            dur_ms=$(( (now_ns - PHASE_START_NS) / 1000000 ))
            format_duration_ms "$dur_ms"
            return 0
        fi
    fi
    echo "0ms"
}

# ---------------------------------------------------------------------------
# Typed receipt emitters
# ---------------------------------------------------------------------------
# emit_to <receipt_file> <line>
#   Appends one structured line to an arbitrary typed receipt file, mirroring
#   it on stdout. Single-line appends are atomic under O_APPEND.
emit_to() {
    local receipt_file="$1"
    shift
    mkdir -p target/logs
    printf '%s\n' "$*" | tee -a "$receipt_file"
}

# emit <receipt-line>
#   Appends one structured line to the typed quick-run receipt
#   (target/logs/quick-receipt.txt), mirroring it on stdout.
emit() {
    emit_to target/logs/quick-receipt.txt "$*"
}

# ---------------------------------------------------------------------------
# Test/benchmark execution gates
# ---------------------------------------------------------------------------
# _nam_count_log_total <log_file>
#   Project-neutral default backend for the gates below: sums the libtest
#   "N passed" / "N measured" counters and falls back to Criterion
#   "time: [ ... ]" lines. Prints the total on stdout and returns 0.
#   A repo may OVERRIDE this function after sourcing the core (e.g. to delegate
#   to a project QA binary) without editing this block.
_nam_count_log_total() {
    local log_file="$1"
    if [ ! -f "$log_file" ]; then
        echo 0
        return 0
    fi
    local total=0 n
    for n in $(grep -oE '[0-9]+ passed' "$log_file" 2>/dev/null | grep -oE '^[0-9]+'); do
        total=$((total + n))
    done
    for n in $(grep -oE '[0-9]+ measured' "$log_file" 2>/dev/null | grep -oE '^[0-9]+'); do
        total=$((total + n))
    done
    if [ "$total" -eq 0 ]; then
        n=$(grep -cE '^[^[:space:]].*time:[[:space:]]+\[' "$log_file" 2>/dev/null || true)
        total="${n:-0}"
    fi
    echo "$total"
}

# assert_ran_tests <log_file> [min_count]
#   Verifies that a test/benchmark log proves real execution: sums the executed
#   test/benchmark counters and fails the gate when the total falls below
#   min_count. Fail-closed against typo'd targets, empty filters and 100% skip
#   selection.
assert_ran_tests() {
    local log_file="$1" min_count="${2:-1}"
    local total
    if ! total=$(_nam_count_log_total "$log_file"); then
        warn "log counting backend unavailable — gate fails closed"
        return 1
    fi
    total="${total:-0}"
    if [ "$total" -lt "$min_count" ]; then
        echo -e "${RED}${BOLD}❌ Gate failed: phase executed 0 tests/benchmarks (empty selection or filter mismatch).${NC}"
        return 1
    fi
    echo -e "  Gate: ${total} test(s)/benchmark(s) executed ≥ ${min_count}  ✓"
    return 0
}

# assert_subphase_ran <phase_name> <log_file> [min_count]
#   Per-subphase counterpart of assert_ran_tests: proves that ONE mandatory
#   subphase of a multi-subphase phase executed at least min_count
#   tests/benchmarks. Invariant: a phase can never be registered PASSED if a
#   mandatory subphase executed zero tests due to #[cfg] compilation filters.
assert_subphase_ran() {
    local phase_name="$1" log_file="$2" min_count="${3:-1}"
    local total
    if ! total=$(_nam_count_log_total "$log_file"); then
        warn "log counting backend unavailable — subphase gate fails closed"
        return 1
    fi
    total="${total:-0}"
    if [ "$total" -lt "$min_count" ]; then
        echo -e "${RED}${BOLD}❌ Subphase gate failed: '${phase_name}' executed ${total} test(s)/benchmark(s) (< ${min_count}).${NC}"
        return 1
    fi
    echo -e "  Gate: subphase '${phase_name}' executed ${total} test(s)/benchmark(s) ≥ ${min_count}  ✓"
    return 0
}

# ---------------------------------------------------------------------------
# Benchmark core resolution
# ---------------------------------------------------------------------------
# pick_bench_core [sysfs_cpu_dir]
#   Resolves the deterministic benchmark CPU core according to policy:
#     1. $NAM_BENCH_CORE if explicitly set and non-empty.
#     2. First online isolated CPU from /sys/devices/system/cpu/isolated
#        (treating 8-9 as a physical core, preferring 8 with cpu9=offline intentional).
#     3. Fallback to $((nproc / 2)) with a warning and a typed gap when
#        `record_gap` is available.
pick_bench_core() {
    local sysfs_cpu="${1:-/sys/devices/system/cpu}"

    # 1. Explicit override
    if [ -n "${NAM_BENCH_CORE:-}" ]; then
        echo "$NAM_BENCH_CORE"
        return 0
    fi

    # 2. Inspect /sys/devices/system/cpu/isolated
    local isolated_file="$sysfs_cpu/isolated"
    if [ -f "$isolated_file" ]; then
        local raw_isolated
        raw_isolated=$(cat "$isolated_file" 2>/dev/null | tr -d '[:space:]')
        if [ -n "$raw_isolated" ]; then
            local -a candidate_cpus=()
            local IFS=','
            local -a parts=($raw_isolated)
            unset IFS
            for part in "${parts[@]}"; do
                if [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
                    local start="${BASH_REMATCH[1]}"
                    local end="${BASH_REMATCH[2]}"
                    for ((c = start; c <= end; c++)); do
                        candidate_cpus+=("$c")
                    done
                elif [[ "$part" =~ ^[0-9]+$ ]]; then
                    candidate_cpus+=("$part")
                fi
            done

            # Special policy: treat 8-9 as a physical core; if 8 is in isolated
            # and online, prefer 8 (cpu9=offline is intentional).
            local preferred_8=0
            for c in "${candidate_cpus[@]}"; do
                if [ "$c" -eq 8 ]; then
                    preferred_8=1
                    break
                fi
            done

            if [ "$preferred_8" -eq 1 ]; then
                local online_file="$sysfs_cpu/cpu8/online"
                if [ ! -f "$online_file" ] || [ "$(cat "$online_file" 2>/dev/null | tr -d '[:space:]')" = "1" ]; then
                    echo 8
                    return 0
                fi
            fi

            for c in "${candidate_cpus[@]}"; do
                local online_file="$sysfs_cpu/cpu$c/online"
                if [ ! -f "$online_file" ] || [ "$(cat "$online_file" 2>/dev/null | tr -d '[:space:]')" = "1" ]; then
                    echo "$c"
                    return 0
                fi
            done
        fi
    fi

    # 3. Fallback: nproc / 2
    local num_cores
    num_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
    local fallback_core=$(( ${num_cores:-1} / 2 ))
    echo -e "  \033[1;33mⓘ WARN: no online isolated CPU found in $sysfs_cpu/isolated; falling back to core $fallback_core (nproc/2)\033[0m" >&2
    if type record_gap >/dev/null 2>&1; then
        record_gap "cpu_not_isolated"
    fi
    echo "$fallback_core"
    return 0
}
# ---------------------------------------------------------------------------
# Downstream performance-regression gate (thin orchestration over nam_perf_gate)
# ---------------------------------------------------------------------------
# run_perf_regression_gate <script_dir> <bench_suite> <manifest_default> [args...]
#   Shared body of the per-repository performance regression wrappers, so the
#   downstream scripts stay thin configuration. The caller sources its
#   repository `_lib.sh` (which cd's into the project root and sources this
#   core) and passes:
#     <script_dir>        the wrapper's own directory
#     <bench_suite>       Criterion bench suite name (e.g. clap_bench, rt_gate)
#     <manifest_default>  default expected-id manifest path; NAM_BENCH_MANIFEST overrides
#     [args...]           the wrapper's command-line arguments
#   Mode parsing, engine gate resolution, core pinning, thermal cooldown,
#   bootstrap/check orchestration and fail-closed verdicts are identical across
#   repositories. Environment: NAM_PERF_GATE, NAM_ENGINE_DIR, NAM_BENCH_CORE,
#   NAM_BASELINE_NAME, NAM_BENCH_MANIFEST, NAM_THERMAL_COOLDOWN_S.
run_perf_regression_gate() {
    local script_dir="$1" bench_suite="$2" manifest_default="$3"
    shift 3

    # ── Mode parsing ─────────────────────────────────────────────────────────
    local mode="--check"
    local arg
    for arg in "$@"; do
        case "$arg" in
            --bootstrap-baseline) mode="--bootstrap-baseline" ;;
            --check) mode="--check" ;;
            --legacy)
                die "--legacy is retired: performance gates now strictly use nam_perf_gate"
                ;;
            --help|-h)
                echo "Usage: $0 [--check|--bootstrap-baseline]"
                return 0
                ;;
            *)
                echo "Unknown argument: $arg" >&2
                return 2
                ;;
        esac
    done

    # ── Configuration ────────────────────────────────────────────────────────
    local baseline_name="${NAM_BASELINE_NAME:-ci-baseline}"
    local baseline_dir=".performance-baselines"
    local fingerprint_file="${baseline_dir}/baseline-fingerprint.json"
    local criterion_root="target/criterion"
    local manifest="${NAM_BENCH_MANIFEST:-$manifest_default}"
    local log_file="target/logs/perf-regression.log"

    # CPU core resolution is the canonical pick_bench_core from this core:
    # NAM_BENCH_CORE → first online isolated CPU → nproc/2.
    local bench_core
    bench_core="$(pick_bench_core)"
    local -a taskset=()
    if command -v taskset >/dev/null 2>&1; then
        taskset=(taskset -c "${bench_core}")
    else
        warn "taskset not found — running without core pinning."
    fi

    mkdir -p "$baseline_dir" "$(dirname "$log_file")"

    # ── Engine gate resolution ───────────────────────────────────────────────
    #   1. NAM_PERF_GATE (operator-provided, used as-is);
    #   2. `nam_perf_gate` on PATH (e.g. `cargo install NeuralAmpModeler-rs
    #      --features testing --bin nam_perf_gate`);
    #   3. a build from a sibling engine checkout (NAM_ENGINE_DIR).
    resolve_perf_gate() {
        if [ -n "${NAM_PERF_GATE:-}" ]; then
            if [ ! -x "$NAM_PERF_GATE" ]; then
                die "NAM_PERF_GATE is not executable: $NAM_PERF_GATE"
            fi
            echo "$NAM_PERF_GATE"
            return 0
        fi
        if command -v nam_perf_gate >/dev/null 2>&1; then
            command -v nam_perf_gate
            return 0
        fi
        local engine="${NAM_ENGINE_DIR:-$PROJECT_DIR/../NeuralAmpModeler-rs}"
        if [ -f "$engine/Cargo.toml" ]; then
            echo -e "  ${BLUE}building engine gate from $engine...${NC}" >&2
            local build_out build_rc=0
            set +e
            build_out=$(cargo build --locked --manifest-path "$engine/Cargo.toml" \
                --features testing --bin nam_perf_gate --message-format=json-render-diagnostics 2>&1)
            build_rc=$?
            set -e
            if [ "$build_rc" -ne 0 ]; then
                echo "$build_out" >&2
                die "failed to build nam_perf_gate from $engine (rc=$build_rc)"
            fi
            local exe
            exe=$(printf '%s\n' "$build_out" \
                | grep -F '"reason":"compiler-artifact"' \
                | grep -F '"name":"nam_perf_gate"' \
                | grep -F '"executable":"' \
                | tail -1 \
                | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p')
            if [ -z "$exe" ] || [ ! -x "$exe" ]; then
                die "engine gate executable not found in cargo output for $engine"
            fi
            echo "$exe"
            return 0
        fi
        die "nam_perf_gate not found: set NAM_PERF_GATE, install it (cargo install NeuralAmpModeler-rs --features testing --bin nam_perf_gate), or provide an engine checkout at $engine"
    }

    local perf_gate
    perf_gate="$(resolve_perf_gate)"
    export NAM_PERF_GATE="$perf_gate"

    echo -e "${BLUE}${BOLD}  Performance Regression Gate (engine)${NC}"
    echo -e "  Core: ${YELLOW}${bench_core}${NC}  Baseline: ${YELLOW}${baseline_name}${NC}  Gate: ${CYAN}${perf_gate}${NC}"

    # ── Pre-build + benchmark execution machinery ────────────────────────────
    local bench_exe=""

    extract_bench_executable() {
        local json_out="$1"
        local rc=0
        bench_exe=$(printf '%s\n' "$json_out" \
            | grep -F '"reason":"compiler-artifact"' \
            | grep -F "\"name\":\"${bench_suite}\"" \
            | grep -F '"executable":' \
            | tail -1 \
            | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p') || rc=$?
        if [ "$rc" -ge 2 ]; then
            die "failed to extract benchmark executable from cargo JSON (rc=$rc)"
        fi
        return 0
    }

    prebuild_benchmark() {
        echo -e "  ${BLUE}Pre-compiling '${bench_suite}' before cooldown (all cores)...${NC}"
        bench_exe=""
        local json_out build_rc=0
        set +e
        json_out=$(cargo bench --bench "$bench_suite" --features testing --no-run --message-format=json 2>/dev/null)
        build_rc=$?
        set -e
        if [ "$build_rc" -ne 0 ]; then
            die "pre-compilation of '${bench_suite}' failed (rc=$build_rc)"
        fi
        extract_bench_executable "$json_out"
        if [ -z "$bench_exe" ] || [ ! -x "$bench_exe" ]; then
            bench_exe=$(ls -t "target/release/deps/${bench_suite}-"* 2>/dev/null | head -1 || true)
        fi
        if [ -n "$bench_exe" ] && [ -x "$bench_exe" ]; then
            ok "benchmark executable ready: $(basename "$bench_exe")"
        else
            warn "could not isolate the standalone benchmark binary — falling back to cargo bench"
        fi
    }

    run_benchmark_binary() {
        if [ -n "$bench_exe" ] && [ -x "$bench_exe" ]; then
            "${taskset[@]}" "$bench_exe" --bench "$@"
        else
            "${taskset[@]}" cargo bench --bench "$bench_suite" --features testing -- "$@"
        fi
    }

    apply_thermal_cooldown() {
        local cooldown="${NAM_THERMAL_COOLDOWN_S:-180}"
        if [[ "$cooldown" =~ ^[0-9]+$ ]] && [ "$cooldown" -gt 0 ]; then
            echo -e "${BLUE}[INFO] thermal cooldown: sleeping ${cooldown}s before microbenchmarks...${NC}"
            sleep "$cooldown"
        elif [ "$cooldown" = "0" ]; then
            echo -e "${BLUE}[INFO] thermal cooldown bypassed (NAM_THERMAL_COOLDOWN_S=0)${NC}"
        else
            warn "invalid NAM_THERMAL_COOLDOWN_S='${cooldown}', defaulting to 180s"
            sleep 180
        fi
    }

    # ── Bootstrap baseline (human operator only) ─────────────────────────────
    bootstrap_baseline() {
        echo -e "\n${GREEN}${BOLD}[BOOTSTRAP] Creating new performance baseline...${NC}"
        warn "This operation must be performed by a human operator on a calibrated host."

        prebuild_benchmark
        apply_thermal_cooldown

        : > "$log_file"
        if ! run_benchmark_binary --save-baseline "$baseline_name" 2>&1 | tee "$log_file"; then
            die "benchmark execution failed during baseline bootstrap"
        fi

        local persist_out
        persist_out=$("$perf_gate" persist-baseline \
            --baseline-dir "$baseline_dir" \
            --criterion-root "$criterion_root" \
            --baseline "$baseline_name")
        echo "$persist_out"
        if [[ "$persist_out" =~ "persisted 0 baseline series" ]]; then
            die "baseline bootstrap failed: 0 series persisted from $criterion_root into $baseline_dir (fail-closed)"
        fi
        "$perf_gate" probe \
            --out "$fingerprint_file" \
            --bench-core "$bench_core" \
            --baseline-dir "$baseline_dir" > /dev/null

        ok "baseline '${baseline_name}' persisted in $baseline_dir (fingerprint: $fingerprint_file)"
        echo "TEST_RESULT[perf_regression]=BOOTSTRAPPED"
    }

    # ── Check mode (strictly read-only) ──────────────────────────────────────
    check_regression() {
        echo -e "\n${BLUE}${BOLD}[CHECK] Comparing against the saved baseline...${NC}"

        # 1. Environment fingerprint (fail-closed: MISSING_BASELINE / INCOMPARABLE).
        set +e
        "$perf_gate" compare \
            --baseline "$fingerprint_file" \
            --bench-core "$bench_core" \
            --baseline-dir "$baseline_dir"
        local fp_status=$?
        set -e
        if [ "$fp_status" -ne 0 ]; then
            echo "TEST_RESULT[perf_regression]=FAIL:environment_or_baseline"
            return 1
        fi

        # 2. Restore the persisted Criterion series for the comparison run.
        local restore_out
        restore_out=$("$perf_gate" restore-baseline \
            --baseline-dir "$baseline_dir" \
            --criterion-root "$criterion_root" \
            --baseline "$baseline_name")
        echo "$restore_out"
        if [[ "$restore_out" =~ "no top-level" ]]; then
            die "performance comparison aborted: no top-level '$baseline_name' series found under $baseline_dir"
        fi

        # 3. Pre-build (all cores) then quiesce the pinned core.
        prebuild_benchmark
        apply_thermal_cooldown

        # 4. Measured comparison run.
        : > "$log_file"
        set +e
        run_benchmark_binary --baseline "$baseline_name" 2>&1 | tee "$log_file"
        local bench_rc=${PIPESTATUS[0]}
        set -e
        if [ "$bench_rc" -ne 0 ]; then
            echo "TEST_RESULT[perf_regression]=FAIL:benchmark_run (rc=$bench_rc)"
            return 1
        fi

        # 5. Per-benchmark coverage (expected groups + saved series) — fail-closed.
        if ! "$perf_gate" coverage \
            --log "$log_file" \
            --root "$criterion_root" \
            --baseline "$baseline_name" \
            --suite "$bench_suite" \
            --manifest "$manifest"; then
            echo "TEST_RESULT[perf_regression]=FAIL:baseline_coverage_gap"
            return 1
        fi

        # 6. Machine regression verdict over Criterion change/estimates.json.
        if ! "$perf_gate" verdict \
            --log "$log_file" \
            --root "$criterion_root" \
            --suite "$bench_suite" \
            --manifest "$manifest"; then
            echo "TEST_RESULT[perf_regression]=FAIL:regression_detected"
            return 1
        fi

        ok "no performance regression detected (machine verdict)"
        echo "TEST_RESULT[perf_regression]=PASS"
    }

    case "$mode" in
        --bootstrap-baseline) bootstrap_baseline ;;
        --check) check_regression ;;
        *) echo "Unknown mode: $mode" >&2; return 2 ;;
    esac
}
# <<< NAM-LIB-CORE END <<<
