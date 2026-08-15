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
# Structured audit receipt (Sprint S3-T04; preflight trace S6-T03): every
# completed phase appends one JSONL line (phase_id, name, status,
# duration_ms, tests_executed, gaps, timestamp) to target/logs/long-audit-
# receipt.jsonl via the Rust emitter `nam_long_receipt append`; the suite-level
# `overall` line is appended by `nam_long_receipt summary` before the verdict.
# Preflight steps (preflight-render, preflight-catalog, preflight-package,
# preflight-freshness, preflight-meta) emit their own lines ahead of Phase 1 —
# an abort before the first timed phase still leaves its FAILED line plus the
# derived `overall FAILED` verdict. Shell code never hand-serializes JSON
# (see src/testing/receipt.rs).
#
# Environment variables:
#   NAM_NEW_ARCH=1            Forces using `--test <entry> <test>` format regardless of disk checks.
#
# Golden/fixture presence is gated EXCLUSIVELY by the Rust preflights below:
#   * catalog_preflight — fixture catalog + V1 golden matrix (DistributedCore,
#     LocalNonDistributable, CabSim) via validate_v1_goldens() + V2 multi-SR
#     matrix via validate_v2_catalog() (src/testing/catalog.rs is the single
#     source of truth). Missing RequiredLocal fixtures abort fail-closed.
#   * check_freshness (nam_freshness) — .golden_manifest.sha256 integrity gate.
# The former Phase-0 bash golden lists and auto-rebuild knobs
# (NAM_SKIP_GOLDEN_BUILD / deprecated NAM_AUTO_BUILD_GOLDENS) were removed
# (Sprint S6-T01): regenerate goldens manually with
# tests/fixtures/golden_gen_build.sh when the preflight reports a missing file.

set -euo pipefail

STRICT_PRE_RELEASE=0
for arg in "$@"; do
    case "$arg" in
        --strict-pre-release)
            STRICT_PRE_RELEASE=1
            ;;
        --help|-h)
            echo "Usage: $0 [--strict-pre-release]"
            exit 0
            ;;
    esac
done

# AI NOTE: Due to the long runtime by design, AI agents MUST NOT execute this script directly.
# Ask the human operator to run it and report the results if needed.
#
# Known bug KB-A2-MAX (`wavenet_a2_max.nam`, docs/cpp_parity_map.md §4.4.3):
#   - NOT in the validated V2 golden catalog (src/testing/catalog.rs
#     GOLDEN_GEN_CATALOG, in_v2_catalog=false) by design.
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
    [catalog_preflight]="models"
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
    [cabsim_golden]="models"
    [oversampling_characterization]="models"
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

_test_flag() {
    local test_name="$1"
    local entry="${LONG_ENTRY_MAP[$test_name]:-models}"
    echo "--test $entry ${test_name}"
}

# Shared style helpers (RED/GREEN/YELLOW/BLUE/BOLD/NC) + cd to project root.
source "$(dirname "$0")/_lib.sh"

# CPU core pinning for performance-sensitive phases (RT Deadline, RT Jitter).
# Override with NAM_BENCH_CORE; defaults to the middle physical core.
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
DEFAULT_CORE=$(( ${NUM_CORES:-1} / 2 ))
BENCH_CORE="${NAM_BENCH_CORE:-$DEFAULT_CORE}"
HAS_TASKSET=0
if command -v taskset >/dev/null 2>&1; then
    HAS_TASKSET=1
fi

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
# long-audit-receipt.jsonl is regenerated by this suite (Sprint S3-T04).
rm -f target/logs/catalog_preflight.log \
      target/logs/meta_coherence.log \
      target/logs/package-list.err \
      target/logs/phase1-soak.log \
      target/logs/phase-libm-exports.log \
      target/logs/phase2-proptests-parity.log \
      target/logs/phase3-heap-audit.log \
      target/logs/phase4-rt-deadline.log \
      target/logs/phase5-rt-jitter.log \
      target/logs/phase-defense-scripts.log \
      target/logs/phase6-loom.log \
      target/logs/long-audit-receipt.jsonl

TIMED_TRACKER=$(mktemp)
trap 'rm -f "$TIMED_TRACKER"' EXIT

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# ── Structured long-audit receipt (Sprint S3-T04; preflight trace S6-T03) ──
# Every completed phase AND every preflight step appends one JSONL line
# (phase_id, name, status, duration_ms, tests_executed, gaps, timestamp) to
# target/logs/long-audit-receipt.jsonl via the Rust emitter
# `nam_long_receipt append` — no fragile bash JSON generation. The suite-level
# `overall` line is appended by `nam_long_receipt summary` before the verdict.
# Preflight steps (preflight-*) run ahead of Phase 1 (S6-T03 / RES-08): an
# abort there exits the suite immediately, so the FAILED line and the derived
# `overall FAILED` verdict are emitted before that exit — failures before
# Phase 1 still leave a machine-readable trace.
LONG_RECEIPT_FILE="target/logs/long-audit-receipt.jsonl"
LONG_RECEIPT_BIN="${NAM_LONG_RECEIPT_BIN:-$PROJECT_DIR/target/debug/nam_long_receipt}"
LONG_RECEIPT_FAILED=0

ensure_long_receipt_bin() {
    if [ -x "$LONG_RECEIPT_BIN" ]; then
        return 0
    fi
    if ! ( cd "$PROJECT_DIR" && cargo build --quiet --features testing --bin nam_long_receipt >/dev/null 2>&1 ); then
        echo -e "  ${RED}${BOLD}❌ FATAL: failed to build nam_long_receipt${NC}" >&2
        return 1
    fi
    return 0
}

# emit_preflight_receipt <phase_id> <name> <status> <duration_ms> [--log <path>] [--gaps <list>]
# Best-effort structured trace for one preflight step (S6-T03 / RES-08).
# A failure flags LONG_RECEIPT_FAILED (fail-closed at the final verdict) but
# never rewrites the preflight's own outcome.
emit_preflight_receipt() {
    local phase_id="$1" name="$2" status="$3" duration_ms="$4"
    shift 4
    if ! ensure_long_receipt_bin; then
        LONG_RECEIPT_FAILED=1
        return 1
    fi
    local rc=0
    "$LONG_RECEIPT_BIN" append \
        --phase-id "$phase_id" \
        --name "$name" \
        --status "$status" \
        --duration-ms "$duration_ms" \
        --out "$LONG_RECEIPT_FILE" "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo -e "  ${YELLOW}${BOLD}⚠ Long-audit receipt emission failed for $phase_id (rc=$rc)${NC}" >&2
        LONG_RECEIPT_FAILED=1
        return 1
    fi
    return 0
}

# abort_preflight <phase_id> <name> <duration_ms> [--log <path>] [--gaps <list>]
# Appends the failing preflight's receipt line (status FAILED), generates the
# suite-level `overall` verdict (FAILED) and exits 1. Preflights abort before
# any timed phase, so this is the only moment the overall line can be produced
# (S6-T03 / RES-08 acceptance: aborted preflight ⇒ line FAILED + overall FAILED).
abort_preflight() {
    local phase_id="$1" name="$2" duration_ms="$3"
    shift 3
    emit_preflight_receipt "$phase_id" "$name" "FAILED" "$duration_ms" "$@" || true
    if ensure_long_receipt_bin; then
        "$LONG_RECEIPT_BIN" summary --out "$LONG_RECEIPT_FILE" >/dev/null 2>&1 || true
    fi
    exit 1
}

# Verify NeuralAmpModelerCore presence (repo-local third-party mirror,
# resolved by _lib.sh; override via NAM_THIRD_PARTY_DIR or NAM_CORE_DIR).
# Auto-populate once if missing (hard: abort long suite without Core).
if ! ensure_third_party hard; then
    echo -e "${RED}${BOLD}❌ NeuralAmpModelerCore not available — long suite requires it.${NC}"
    exit 1
fi

CURRENT_CORE_SHA=$(cd "$NAM_CORE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
echo -e "${GREEN}✓ NeuralAmpModelerCore found (version: $CURRENT_CORE_SHA).${NC}"

# ── Pre-flight — golden/fixture presence & C++ render binary ──
# Golden/fixture presence is validated EXCLUSIVELY by the Rust gates below
# (no bash golden lists — Sprint S6-T01):
#   * catalog_preflight — fixture catalog + V1 golden matrix (DistributedCore
#     model goldens, LocalNonDistributable WaveNet Lite, CabSim convolution
#     goldens) via validate_v1_goldens() + V2 multi-SR matrix via
#     validate_v2_catalog(); both defined only in src/testing/catalog.rs.
#     Missing RequiredLocal fixtures abort fail-closed with exit 1 before any
#     timed phase.
#   * check_freshness hard-fail (nam_freshness) — .golden_manifest.sha256
#     integrity: missing/stale/orphaned goldens or models abort.
# The former v1 golden bash lists (model / non-distributable / CabSim arrays)
# and the Phase-0 auto-rebuild were removed: when the
# preflight reports a missing golden, regenerate with
# tests/fixtures/golden_gen_build.sh (C++ toolchain + NeuralAmpModelerCore).

# ── Render binary — unified build (S3-T01) ──
# Phase 3 (proptests/parity) requires the C++ render tool for the full
# live_cross_validation matrix. ensure_namcore_render is the single entry
# point (idempotent; skips cmake when the binary is up-to-date) and a render
# build failure here is a hard preflight failure with detailed diagnostics.
echo -e "\n${BLUE}${BOLD}→ Preflight: C++ render binary (preflight-render)...${NC}"
PF_RENDER_START=$(date +%s%N)
if RENDER_BIN="$(ensure_namcore_render)"; then
    :
else
    RC_RENDER=$?
    PF_RENDER_DUR=$(( ($(date +%s%N) - PF_RENDER_START) / 1000000 ))
    echo -e "${RED}${BOLD}❌ ensure_namcore_render failed (exit $RC_RENDER) — C++ render binary required by Phase 3 parity.${NC}"
    echo -e "  ${YELLOW}Diagnostic logs: target/logs/cmake-configure.log target/logs/cmake-build.log${NC}"
    abort_preflight "preflight-render" "C++ render binary (preflight)" "$PF_RENDER_DUR"
fi
PF_RENDER_DUR=$(( ($(date +%s%N) - PF_RENDER_START) / 1000000 ))
echo -e "${GREEN}✓ C++ render binary: $RENDER_BIN${NC}"
emit_preflight_receipt "preflight-render" "C++ render binary (preflight)" "PASSED" "$PF_RENDER_DUR" || true

# ── Catalog Preflight — Capability Receipt (Sprint 6: T-E3.1-1; V1 gate S6-T01; V2 gate S3-T02) ──
# Runs the Rust-side unified fixture catalog capability receipt AND the V1
# golden catalog validation (src/testing/catalog.rs::validate_v1_goldens:
# DistributedCore model goldens + LocalNonDistributable WaveNet Lite + CabSim
# convolution goldens) AND the V2 golden catalog validation
# (validate_v2_catalog) BEFORE the freshness gate and long suite, so the
# operator has a complete typed inventory of every fixture's status
# (Available / MissingOptional / MissingRequired) with resolved paths. The V1
# and V2 golden matrices are defined exclusively in Rust — this gate is the
# only presence preflight (the former v1 golden bash lists and the
# V2_CATALOG_SCOPE shell checks were removed). Missing RequiredLocal fixtures
# fail this gate hard (the cargo test exits 1); the grep below is
# defense-in-depth for the summary.
echo -e "\n${BLUE}${BOLD}→ Generating fixture + V1/V2 catalog capability receipt (preflight-catalog)...${NC}"
PF_CATALOG_START=$(date +%s%N)
if ! cargo test --features testing --release $(_test_flag catalog_preflight) -- --nocapture 2>&1 | tee target/logs/catalog_preflight.log; then
    PF_CATALOG_DUR=$(( ($(date +%s%N) - PF_CATALOG_START) / 1000000 ))
    echo -e "${RED}${BOLD}FAIL: catalog_preflight — see target/logs/catalog_preflight.log${NC}"
    abort_preflight "preflight-catalog" "Fixture + V1/V2 catalog preflight" "$PF_CATALOG_DUR" --log target/logs/catalog_preflight.log
fi
# Extract MISSING-REQUIRED count for the summary
MISSING_REQUIRED_COUNT=$(grep -c 'MISSING-REQUIRED:' target/logs/catalog_preflight.log 2>/dev/null || true)
if [ "$MISSING_REQUIRED_COUNT" -gt 0 ]; then
    PF_CATALOG_DUR=$(( ($(date +%s%N) - PF_CATALOG_START) / 1000000 ))
    echo -e "${RED}${BOLD}❌ Catalog preflight: ${MISSING_REQUIRED_COUNT} RequiredLocal fixture(s) absent.${NC}"
    echo -e "  Check target/logs/catalog_preflight.log for the full capability receipt."
    abort_preflight "preflight-catalog" "Fixture + V1/V2 catalog preflight" "$PF_CATALOG_DUR" --log target/logs/catalog_preflight.log
fi
PF_CATALOG_DUR=$(( ($(date +%s%N) - PF_CATALOG_START) / 1000000 ))
echo -e "${GREEN}✓ All RequiredLocal fixtures present per catalog preflight.${NC}"
emit_preflight_receipt "preflight-catalog" "Fixture + V1/V2 catalog preflight" "PASSED" "$PF_CATALOG_DUR" --log target/logs/catalog_preflight.log || true

# ── Package Exclusion Verification (Sprint 6: T-E3.4-1) ──
# Confirms that non-distributable models (models-nondist/) and third-party
# vendor artifacts are excluded from the crate package.  Any leak of
# proprietary / license-restricted content into the distributed crate
# is a hard packaging failure.
#
# Dirty-tree handling: the vendor mirrors under third-party/ are nested git
# checkouts. Cargo's status walker reports their contents as uncommitted
# files even though /third-party is gitignored, which makes plain
# `cargo package --list` refuse on a dirty working tree. `--allow-dirty`
# proceeds, but the walker then lists those git-ignored files as phantom
# entries. The gate therefore verifies the exclusion contract itself: a
# models-nondist/ or third-party/ entry is a real leak only if it is NOT
# git-ignored (ignored entries are walker artifacts — the include whitelist
# in Cargo.toml governs the actual tarball, and cargo itself refuses to
# package a dirty tree without --allow-dirty).
echo -e "\n${BLUE}${BOLD}→ Verifying cargo package exclusion of non-distributable artifacts (preflight-package)...${NC}"
PF_PKG_START=$(date +%s%N)
if ! PACKAGE_LIST=$(cargo package --list --allow-dirty 2>target/logs/package-list.err); then
    PF_PKG_DUR=$(( ($(date +%s%N) - PF_PKG_START) / 1000000 ))
    echo -e "${RED}${BOLD}FAIL: cargo package --list failed — see target/logs/package-list.err${NC}"
    cat target/logs/package-list.err
    abort_preflight "preflight-package" "Cargo package exclusion preflight" "$PF_PKG_DUR"
fi
LEAKS=()
while IFS= read -r entry; do
    case "$entry" in
        models-nondist/*|third-party/*)
            if ! git check-ignore -q --no-index -- "$entry"; then
                LEAKS+=("$entry")
            fi
            ;;
    esac
done <<< "$PACKAGE_LIST"
if [ ${#LEAKS[@]} -gt 0 ]; then
    PF_PKG_DUR=$(( ($(date +%s%N) - PF_PKG_START) / 1000000 ))
    echo -e "${RED}${BOLD}❌ PACKAGE VIOLATION: non-distributable or third-party artifacts leaked into crate package.${NC}"
    printf '  %s\n' "${LEAKS[@]}"
    abort_preflight "preflight-package" "Cargo package exclusion preflight" "$PF_PKG_DUR" --gaps "package_leak"
fi
PF_PKG_DUR=$(( ($(date +%s%N) - PF_PKG_START) / 1000000 ))
echo -e "${GREEN}✓ Package exclusion verified — no models-nondist/ or third-party/ artifacts in crate.${NC}"
emit_preflight_receipt "preflight-package" "Cargo package exclusion preflight" "PASSED" "$PF_PKG_DUR" || true

# ── Freshness gate (blocking, centralized) ──
# hard-fail: artifact integrity + generator provenance (stricter than quick's
# artifacts-hard, which only warns on generator-script drift).
# Bypass with NAM_BYPASS_FRESHNESS=1 for local developer convenience.
echo -e "\n→ Checking freshness of test fixtures and goldens (preflight-freshness)..."
PF_FRESH_START=$(date +%s%N)
if ! check_freshness hard-fail; then
    PF_FRESH_DUR=$(( ($(date +%s%N) - PF_FRESH_START) / 1000000 ))
    abort_preflight "preflight-freshness" "Fixture/golden freshness preflight" "$PF_FRESH_DUR"
fi
PF_FRESH_DUR=$(( ($(date +%s%N) - PF_FRESH_START) / 1000000 ))
emit_preflight_receipt "preflight-freshness" "Fixture/golden freshness preflight" "PASSED" "$PF_FRESH_DUR" || true

# ── Catalog↔test coherence gate (blocking) ──
# `meta_coherence` is a cheap, dependency-free governance test (no NAMCore, no
# goldens needed — it only parses golden_gen_build.sh + tests/*.rs). It has no
# home in tests-quick.sh (not a correctness or structural test) and would be
# silently orphaned ("on demand" only) without this hook. Runs here, before
# the ± 50 min battery, so a drifted catalog fails fast instead of burning a
# full nightly window before being noticed.
echo -e "\n${BLUE}${BOLD}→ Checking catalog↔test coherence (preflight-meta)...${NC}"
PF_META_START=$(date +%s%N)
if ! cargo test --features testing --release $(_test_flag meta_coherence) 2>&1 | tee target/logs/meta_coherence.log; then
    PF_META_DUR=$(( ($(date +%s%N) - PF_META_START) / 1000000 ))
    echo -e "${RED}${BOLD}❌ meta_coherence failed — golden catalog diverged from #[ignore] tests.${NC}"
    abort_preflight "preflight-meta" "Catalog↔test coherence preflight" "$PF_META_DUR" --log target/logs/meta_coherence.log
fi
PF_META_DUR=$(( ($(date +%s%N) - PF_META_START) / 1000000 ))
echo -e "${GREEN}✓ Golden catalog matches tests coherently.${NC}"
emit_preflight_receipt "preflight-meta" "Catalog↔test coherence preflight" "PASSED" "$PF_META_DUR" --log target/logs/meta_coherence.log || true

# ── Phase classification for fidelity/performance split ──────────────────────
# run_phase indices: 0 soak, 1 defense, 2 proptests, 3 heap,
#                    4 deadline, 5 jitter, 6 loom
declare -A PHASE_CLASS=(
    [0]="fidelity"
    [1]="fidelity"
    [2]="fidelity"
    [3]="fidelity"
    [4]="performance"
    [5]="performance"
    [6]="fidelity"
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



assert_phase_ran() {
    assert_ran_tests "target/logs/$1" "${2:-1}"
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

        if ! assert_phase_ran "$log_file"; then
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

# ── Structured long-audit receipt: per-phase emission ───────────────────────
# Receipt file/bin/ensure_long_receipt_bin/emit_preflight_receipt/abort_preflight
# are defined at the top (preflight steps emit before Phase 1 — S6-T03).

# emit_long_phase_receipt <phase_idx> <log_file>
# Appends the just-completed phase's structured receipt line. Emission must
# run AFTER any post-phase status override (phases 4/5) so the line carries
# the authoritative status. A failure flags LONG_RECEIPT_FAILED (fail-closed
# at the final verdict) but never rewrites the phase's own outcome.
emit_long_phase_receipt() {
    local idx="$1" log_file="$2"
    if ! ensure_long_receipt_bin; then
        LONG_RECEIPT_FAILED=1
        return 1
    fi
    local rc=0
    "$LONG_RECEIPT_BIN" append \
        --phase-id "phase$((idx + 1))" \
        --name "${PHASE_NAMES[$idx]}" \
        --status "${PHASE_STATUS[$idx]}" \
        --duration-ms "$(( PHASE_DURATIONS[$idx] * 1000 ))" \
        --log "target/logs/$log_file" \
        --out "$LONG_RECEIPT_FILE" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo -e "  ${YELLOW}${BOLD}⚠ Long-audit receipt emission failed for phase $((idx + 1)) (rc=$rc)${NC}" >&2
        LONG_RECEIPT_FAILED=1
        return 1
    fi
    return 0
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
    timed_cargo_test "concurrency_stress" --release --no-fail-fast $(_test_flag concurrency_stress) -- --ignored --nocapture || status=1
    return $status
}
run_phase "Soak Tests (Numerical Stability)" "run_soak_phase" "phase1-soak.log" || true
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase1-soak.log" || true

# --- Defense scripts + ELF surface (absorbed from utils/tests + utils/debug) ---

# run_bash_scripts_unit_tests — inline Bash unit-test suite for the defense
# scripts (formerly the standalone utils/tests/test_scripts.sh, removed in
# S4-T06). Exercises utils/_lib.sh and isolated functions of
# utils/quality-dashboard.sh and utils/tests-performance-regression.sh against
# their own failure modes (F-01, F-02, F-08, F-21, F-22, F-24, F-27, S3-T01):
#   * metric sanitization (null/empty/non-finite sentinels)   (F-01, F-28)
#   * toolchain fingerprint without TOOLCHAIN manifest lines   (F-02)
#   * single performance-status classifier (NOT_VERIFIED)      (F-08)
#   * real test-execution assertion & JSONL record counting    (F-21)
#   * conservative freshness gate (STALE/MISSING/ORPHAN/OK)    (F-22)
#   * baseline coverage cross-check helpers                     (F-24)
#   * extended long-suite delegation (--full)                   (F-06)
#   * single jq JSONL parser with canonical edge cases         (F-27)
#   * unified C++ render build (ensure_namcore_render)         (S3-T01)
#
# The suite runs in a subshell with `set +euo pipefail` (the original was a
# standalone `bash` process, so assertions inspect the rc of commands that are
# expected to fail); the subshell also isolates every sandbox directory and
# global variable change from this suite. Exit status: 0 when every test
# passes (or skips), 1 on any failure.
run_bash_scripts_unit_tests() {
    (
        set +euo pipefail

UTILS_DIR="$LIB_DIR"
LIB_SH="$UTILS_DIR/_lib.sh"
DASHBOARD_SH="$UTILS_DIR/quality-dashboard.sh"
PERF_REGRESSION_SH="$UTILS_DIR/tests-performance-regression.sh"

# ── Load shared library ──────────────────────────────────────────────────────
PHASE_TOTAL=1
# shellcheck disable=SC1091
source "$LIB_SH"

# ── Isolated functions extracted from quality-dashboard.sh / ─────────────────
# ── tests-performance-regression.sh. Each is a self-contained definition      ──
# ── pulled by its `^name() {` … `^}` block, so the test suite never executes  ──
# ── the scripts' own `main "$@"` or load-time side effects. Extraction        ──
# ── failure is fatal: it means the script was refactored and this suite       ──
# ── must follow.                                                              ──
extract_define() {
    local file="$1" name="$2" body
    body="$(sed -n "/^${name}() {/,/^}/p" "$file")"
    if [ -z "$body" ]; then
        echo "ERROR: cannot extract function '${name}' from ${file}" >&2
        return 1
    fi
    eval "$body"
}
extract_define "$DASHBOARD_SH" _nfmt                || exit 1
extract_define "$DASHBOARD_SH" _fmt_metric          || exit 1
extract_define "$DASHBOARD_SH" _is_finite_num       || exit 1
extract_define "$DASHBOARD_SH" _is_numeric_esr      || exit 1
extract_define "$DASHBOARD_SH" _safe_render         || exit 1
extract_define "$DASHBOARD_SH" detect_isa           || exit 1
extract_define "$DASHBOARD_SH" detect_cpu_model     || exit 1
extract_define "$DASHBOARD_SH" parse_jsonl_fidelity || exit 1
extract_define "$DASHBOARD_SH" run_phase0_freshness || exit 1
extract_define "$DASHBOARD_SH" run_extended_audit   || exit 1
extract_define "$PERF_REGRESSION_SH" executed_bench_ids      || exit 1
extract_define "$PERF_REGRESSION_SH" missing_baseline_coverage || exit 1

# Globals consumed by parse_jsonl_fidelity (declared at dashboard load time).
declare -A ESR_NAMCORE ESR_NAMCORE_DB SNR_DB MSE_VAL MRSTFT
declare -a MODEL_ORDER

# ── Test harness ──────────────────────────────────────────────────────────────
TOTAL=0 PASSED=0 FAILED=0 SKIPPED=0
FAILED_NAMES=()

pass() { TOTAL=$((TOTAL + 1)); PASSED=$((PASSED + 1)); printf '  %sok%s   %s\n' "$GREEN" "$NC" "$1"; }
fail() { TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1)); FAILED_NAMES+=("$1"); printf '  %sFAIL%s %s\n' "$RED" "$NC" "$1"; }
skip() { SKIPPED=$((SKIPPED + 1)); printf '  %sSKIP%s  %s\n' "$YELLOW" "$NC" "$1"; }

expect_rc()       { local name="$1" want="$2" got="$3"; if [ "$want" -eq "$got" ]; then pass "$name"; else fail "$name (expected rc=$want, got rc=$got)"; fi; }
expect_str()      { local name="$1" want="$2" got="$3"; if [ "$want" = "$got" ]; then pass "$name"; else fail "$name (expected [$want], got [$got])"; fi; }
expect_nonempty() { local name="$1" got="$2"; if [ -n "$got" ]; then pass "$name"; else fail "$name (expected non-empty output)"; fi; }

# Assert a phase receipt line holds a given status and optional reason substring.
receipt_has() {
    local phase_id="$1" status="$2" reason="${3:-}" line
    line="$(grep "\"phase_id\":\"${phase_id}\"" "${DASHBOARD_PHASE_RECEIPT:-}" 2>/dev/null | tail -1)"
    [ -n "$line" ] || return 1
    case "$line" in *"\"status\":\"${status}\""*) ;; *) return 1 ;; esac
    [ -z "$reason" ] || case "$line" in *"${reason}"*) return 0 ;; *) return 1 ;; esac
    return 0
}

# Capture run_phase0_freshness's exit status from a sandbox without letting its
# internal `set -e` (plus a non-zero return) terminate the capture subshell.
capture_phase0_rc() {  # $1 = sandbox dir; prints the exit code
    ( set -u; cd "$1" || exit 9; if run_phase0_freshness >/dev/null 2>&1; then echo 0; else echo 1; fi )
}

# Freshness sandbox: a minimal self-consistent golden manifest + one model, so
# check_freshness/run_freshness_gate resolve their relative paths against it.
make_freshness_sandbox() {
    local sb="$1" sha
    mkdir -p "$sb/tests/fixtures/models"
    printf 'model-a\n' > "$sb/tests/fixtures/models/model_a.nam"
    sha="$(sha256sum "$sb/tests/fixtures/models/model_a.nam" | cut -d' ' -f1)"
    {
        echo "# Golden freshness manifest — test fixture"
        echo "${sha} 0000000000000000000000000000000000000000000000000000000000000000 model_a.nam golden_a.bin"
        echo "# MODEL-REGISTRY: model_a.nam"
    } > "$sb/tests/fixtures/.golden_manifest.sha256"
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ""
echo "${BOLD}=== _is_finite_num / _is_numeric_esr (F-01, F-28) ===${NC}"

for v in 0 0.0 0.5 .5 1. 1.5e-3 -1.5E3 +3.14 3.14e2 42 12345678901234567890; do
    _is_finite_num "$v"; rc=$?
    expect_rc "_is_finite_num accepts '${v}'" 0 "$rc"
done
for v in "" " " inf -inf +inf Infinity -infinity nan -nan NaN null N/A abc 1.2.3 0x10 1e e5 .; do
    _is_finite_num "$v"; rc=$?
    expect_rc "_is_finite_num rejects '${v}'" 1 "$rc"
done
_is_numeric_esr ".5"; rc=$?; expect_rc "_is_numeric_esr accepts '.5'" 0 "$rc"
_is_numeric_esr "inf"; rc=$?; expect_rc "_is_numeric_esr rejects 'inf'" 1 "$rc"

echo ""
echo "${BOLD}=== _safe_render / _fmt_metric / _nfmt (F-01 defense-in-depth) ===${NC}"

expect_str    "_safe_render strips backslash+control chars" "abc" "$(_safe_render $'a\\b\nc')"
expect_str    "_safe_render keeps plain text"              "plain-123" "$(_safe_render 'plain-123')"
expect_str    "_fmt_metric renders N/A"                    "N/A" "$(_fmt_metric N/A)"
expect_str    "_fmt_metric renders empty as N/A"           "N/A" "$(_fmt_metric '')"
expect_str    "_fmt_metric renders 0.5"                    "0.5000" "$(_fmt_metric 0.5)"
expect_str    "_fmt_metric renders 2"                      "2.0000" "$(_fmt_metric 2)"
expect_str    "_fmt_metric renders scientific notation"    "1.50e-03" "$(_fmt_metric 1.5e-3)"
expect_str    "_nfmt forces C locale for decimals"         "1.50" "$(_nfmt '%.2f' 1.5)"

echo ""
echo "${BOLD}=== count_jsonl_records (F-21) ===${NC}"

expect_str "count_jsonl_records absent file -> 0"  "0" "$(count_jsonl_records "$WORK/nope.jsonl")"
: > "$WORK/empty.jsonl"
expect_str "count_jsonl_records empty file -> 0"   "0" "$(count_jsonl_records "$WORK/empty.jsonl")"
printf 'a\nb\nc\n' > "$WORK/three.jsonl"
expect_str "count_jsonl_records three lines -> 3"  "3" "$(count_jsonl_records "$WORK/three.jsonl")"

echo ""
echo "${BOLD}=== assert_ran_tests (F-21) ===${NC}"

printf 'test result: ok. 50 passed. 2 failed.\n' > "$WORK/pass.log"
assert_ran_tests "$WORK/pass.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests counts 50 passed -> 0" 0 "$rc"

printf 'test result: ok. 0 passed. 0 failed.\n' > "$WORK/zero.log"
assert_ran_tests "$WORK/zero.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests 0 passed (100% skip) -> 1" 1 "$rc"

printf 'running tests...\nall filtered out (early return)\n' > "$WORK/skip.log"
assert_ran_tests "$WORK/skip.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests skip-only log -> 1" 1 "$rc"

assert_ran_tests "$WORK/absent.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests missing file -> 1" 1 "$rc"

printf 'bench time: [1.2 ms]\nbench time: [3.4 ms]\n' > "$WORK/bench.log"
assert_ran_tests "$WORK/bench.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests Criterion time fallback -> 0" 0 "$rc"

printf 'x 5 measured\n' > "$WORK/meas.log"
assert_ran_tests "$WORK/meas.log" 1 >/dev/null 2>&1; rc=$?
expect_rc "assert_ran_tests 'N measured' -> 0" 0 "$rc"

echo ""
echo "${BOLD}=== run_dashboard_phase (F-21) ===${NC}"

LOGDIR="$WORK/logdir"; mkdir -p "$LOGDIR"
NAM_METRICS_JSONL="$WORK/metrics.jsonl"
DASHBOARD_PHASE_RECEIPT="$WORK/receipt.jsonl"

( run_dashboard_phase "t_pass" 1 'printf "test result: ok. 3 passed.\n" > "$LOGDIR/t_pass.log" 2>&1' >/dev/null 2>&1 )
receipt_has t_pass PASS && pass "run_dashboard_phase: real execution -> PASS" || fail "run_dashboard_phase: real execution -> PASS"

( run_dashboard_phase "t_zero" 1 'printf "test result: ok. 0 passed.\n" > "$LOGDIR/t_zero.log" 2>&1' >/dev/null 2>&1 )
receipt_has t_zero FAIL "no tests/benchmarks actually executed" && pass "run_dashboard_phase: 0 passed -> FAIL" || fail "run_dashboard_phase: 0 passed -> FAIL"

( run_dashboard_phase "t_jsonl" 1 1 'printf "test result: ok. 3 passed.\n" > "$LOGDIR/t_jsonl.log" 2>&1' >/dev/null 2>&1 )
receipt_has t_jsonl FAIL "jsonl_records" && pass "run_dashboard_phase: min_jsonl not met -> FAIL" || fail "run_dashboard_phase: min_jsonl not met -> FAIL"

: > "$NAM_METRICS_JSONL"
( run_dashboard_phase "t_jsonl2" 1 1 'printf "test result: ok. 3 passed.\n" > "$LOGDIR/t_jsonl2.log" 2>&1; printf "x\n" >> "$NAM_METRICS_JSONL"' >/dev/null 2>&1 )
receipt_has t_jsonl2 PASS && pass "run_dashboard_phase: min_jsonl met -> PASS" || fail "run_dashboard_phase: min_jsonl met -> PASS"

( run_dashboard_phase "t_exit" 0 'false' >/dev/null 2>&1 )
receipt_has t_exit FAIL "subprocess exited" && pass "run_dashboard_phase: subprocess failure -> FAIL" || fail "run_dashboard_phase: subprocess failure -> FAIL"

echo ""
echo "${BOLD}=== check_toolchain_fingerprint (F-02) ===${NC}"

sb="$(mktemp -d "$WORK/tc-absent-XXXXXX")"; mkdir -p "$sb/tests/fixtures"
( set -u; cd "$sb"; check_toolchain_fingerprint >/dev/null 2>&1 ); rc=$?
expect_rc "check_toolchain_fingerprint: absent manifest -> 0" 0 "$rc"

sb="$(mktemp -d "$WORK/tc-noline-XXXXXX")"; mkdir -p "$sb/tests/fixtures"
printf '# just a comment, no TOOLCHAIN lines\n' > "$sb/tests/fixtures/.golden_manifest.sha256"
( set -u; cd "$sb"; check_toolchain_fingerprint >/dev/null 2>&1 ); rc=$?
expect_rc "check_toolchain_fingerprint: manifest without TOOLCHAIN -> 0 (F-02)" 0 "$rc"

sb="$(mktemp -d "$WORK/tc-empty-XXXXXX")"; mkdir -p "$sb/tests/fixtures"
: > "$sb/tests/fixtures/.golden_manifest.sha256"
( set -u; cd "$sb"; check_toolchain_fingerprint >/dev/null 2>&1 ); rc=$?
expect_rc "check_toolchain_fingerprint: empty manifest -> 0" 0 "$rc"

sb="$(mktemp -d "$WORK/tc-drift-XXXXXX")"; mkdir -p "$sb/tests/fixtures"
printf '# TOOLCHAIN: cxx: definitely-not-a-real-compiler XYZ-999\n' > "$sb/tests/fixtures/.golden_manifest.sha256"
( set -u; cd "$sb"; check_toolchain_fingerprint >/dev/null 2>&1 ); rc=$?
expect_rc "check_toolchain_fingerprint: mismatched cxx -> 1 (drift detected)" 1 "$rc"

echo ""
echo "${BOLD}=== run_freshness_gate (F-22) ===${NC}"

sb_ok="$(mktemp -d "$WORK/fr-ok-XXXXXX")"; make_freshness_sandbox "$sb_ok"
out="$( ( set -u; cd "$sb_ok"; run_freshness_gate artifacts-hard >/dev/null 2>&1; printf '%s|%s' "$?" "${FRESHNESS_REASON:-UNSET}" ) )"
expect_str "run_freshness_gate: consistent manifest -> rc=0 reason=OK" "0|OK" "$out"

sb_stale="$(mktemp -d "$WORK/fr-stale-XXXXXX")"; make_freshness_sandbox "$sb_stale"
printf 'tamper\n' >> "$sb_stale/tests/fixtures/models/model_a.nam"
out="$( ( set -u; cd "$sb_stale"; run_freshness_gate artifacts-hard >/dev/null 2>&1; printf '%s|%s' "$?" "${FRESHNESS_REASON:-UNSET}" ) )"
expect_str "run_freshness_gate: model hash drift -> rc=1 reason=STALE_FIXTURES" "1|STALE_FIXTURES" "$out"

sb_miss="$(mktemp -d "$WORK/fr-miss-XXXXXX")"; make_freshness_sandbox "$sb_miss"
printf '# EXPECTED: missing_golden.bin\n' >> "$sb_miss/tests/fixtures/.golden_manifest.sha256"
out="$( ( set -u; cd "$sb_miss"; run_freshness_gate artifacts-hard >/dev/null 2>&1; printf '%s|%s' "$?" "${FRESHNESS_REASON:-UNSET}" ) )"
expect_str "run_freshness_gate: missing expected golden -> rc=1 reason=MISSING_FIXTURES" "1|MISSING_FIXTURES" "$out"

sb_orph="$(mktemp -d "$WORK/fr-orph-XXXXXX")"; make_freshness_sandbox "$sb_orph"
printf 'orphan\n' > "$sb_orph/tests/fixtures/models/orphan.nam"
out="$( ( set -u; cd "$sb_orph"; run_freshness_gate artifacts-hard >/dev/null 2>&1; printf '%s|%s' "$?" "${FRESHNESS_REASON:-UNSET}" ) )"
expect_str "run_freshness_gate: unregistered model -> rc=1 reason=ORPHAN_FIXTURE" "1|ORPHAN_FIXTURE" "$out"

echo ""
echo "${BOLD}=== run_phase0_freshness receipts (F-22) ===${NC}"

sb_p0="$(mktemp -d "$WORK/p0-XXXXXX")"; make_freshness_sandbox "$sb_p0"
NAM_CORE_DIR="$WORK/namcore"; mkdir -p "$NAM_CORE_DIR/.git"
rm -f "$DASHBOARD_PHASE_RECEIPT"
expect_rc "run_phase0_freshness: present core -> rc 0" 0 "$(capture_phase0_rc "$sb_p0")"
receipt_has freshness PASS && pass "run_phase0_freshness: freshness receipt PASS" || fail "run_phase0_freshness: freshness receipt PASS"
receipt_has third_party PASS && pass "run_phase0_freshness: third_party receipt PASS" || fail "run_phase0_freshness: third_party receipt PASS"

rm -f "$DASHBOARD_PHASE_RECEIPT"
NAM_CORE_DIR="$WORK/nonexistent-core"; NAM_SKIP_THIRD_PARTY_SETUP=1
expect_rc "run_phase0_freshness: absent core -> rc 0 (graceful skip)" 0 "$(capture_phase0_rc "$sb_p0")"
receipt_has third_party SKIP_CAPABILITY third_party_absent && pass "run_phase0_freshness: third_party receipt SKIP_CAPABILITY/third_party_absent" || fail "run_phase0_freshness: third_party receipt SKIP_CAPABILITY/third_party_absent"
unset NAM_SKIP_THIRD_PARTY_SETUP

sb_p0_stale="$(mktemp -d "$WORK/p0s-XXXXXX")"; make_freshness_sandbox "$sb_p0_stale"
printf 'x\n' >> "$sb_p0_stale/tests/fixtures/models/model_a.nam"
NAM_CORE_DIR="$WORK/namcore"
rm -f "$DASHBOARD_PHASE_RECEIPT"
expect_rc "run_phase0_freshness: stale fixtures -> rc 1" 1 "$(capture_phase0_rc "$sb_p0_stale")"
receipt_has freshness FAIL STALE_FIXTURES && pass "run_phase0_freshness: freshness receipt FAIL/STALE_FIXTURES" || fail "run_phase0_freshness: freshness receipt FAIL/STALE_FIXTURES"

echo ""
echo "${BOLD}=== ensure_third_party / detect_isa / detect_cpu_model (tool detection) ===${NC}"

NAM_CORE_DIR="$WORK/namcore"
ensure_third_party soft >/dev/null 2>&1; rc=$?
expect_rc "ensure_third_party: present core -> 0" 0 "$rc"
NAM_CORE_DIR="$WORK/nonexistent-core"; NAM_SKIP_THIRD_PARTY_SETUP=1
ensure_third_party soft >/dev/null 2>&1; rc=$?
expect_rc "ensure_third_party: absent core + skip flag -> 1" 1 "$rc"
unset NAM_SKIP_THIRD_PARTY_SETUP

expect_nonempty "detect_isa returns a known ISA string" "$(detect_isa)"
expect_nonempty "detect_cpu_model returns a model string" "$(detect_cpu_model)"

echo ""
echo "${BOLD}=== parse_jsonl_fidelity (F-27 canonical JSONL) ===${NC}"

if command -v jq >/dev/null 2>&1; then
    PARSEDIR="$WORK/parsedir"; mkdir -p "$PARSEDIR"
    cat > "$WORK/canonical.jsonl" <<'EOF'
{"kind":"fidelity","label":"Model A @48000 Live","esr":null,"esr_db":"","snr_db":"1.5e2","mse":"1.2e-5","mrstft":"inf"}
{"kind":"fidelity","label":"Model B","esr":"","esr_db":null,"snr_db":"-inf","mse":null,"mrstft":"nan"}
{"label":"Model C @44100","esr":"0.0001","esr_db":"-40.0","snr_db":"50.0","mse":"3.0e-7","mrstft":"0.001"}
{"label":null,"esr":"1","esr_db":"2","snr_db":"3","mse":"4","mrstft":"5"}
EOF
    NAM_METRICS_JSONL="$WORK/canonical.jsonl" parse_jsonl_fidelity >/dev/null 2>&1; rc=$?
    expect_rc "parse_jsonl_fidelity parses canonical JSONL -> 0" 0 "$rc"
    expect_str "jq normalizes null esr -> N/A"          "N/A"   "${ESR_NAMCORE["Model A @48000 Live"]}"
    expect_str "jq normalizes empty esr_db -> N/A"      "N/A"   "${ESR_NAMCORE_DB["Model A @48000 Live"]}"
    expect_str "jq preserves e-notation string"         "1.5e2" "${SNR_DB["Model A @48000 Live"]}"
    expect_str "jq preserves non-finite sentinel (inf)" "inf"   "${MRSTFT["Model A @48000 Live"]}"
    expect_str "jq normalizes null esr on Model B"      "N/A"   "${ESR_NAMCORE["Model B"]}"
    expect_str "jq preserves non-finite sentinel (nan)" "nan"   "${MRSTFT["Model B"]}"
    expect_str "jq keeps label with spaces/@ as key"    "0.0001" "${ESR_NAMCORE["Model C @44100"]}"
    _is_finite_num "${MRSTFT["Model A @48000 Live"]}"; rc=$?
    expect_rc "_is_finite_num rejects 'inf' sentinel from JSONL" 1 "$rc"
    _is_finite_num "${SNR_DB["Model A @48000 Live"]}"; rc=$?
    expect_rc "_is_finite_num accepts e-notation from JSONL" 0 "$rc"
    expect_str "parse_jsonl_fidelity drops null-label records (3 labels)" "3" "${#MODEL_ORDER[@]}"
else
    skip "parse_jsonl_fidelity (jq unavailable in PATH)"
fi

echo ""
echo "${BOLD}=== classify_regression_outcome (F-08 single NOT_VERIFIED semantics) ===${NC}"

expect_str "classify PASS -> PASS"                       "PASS"          "$(classify_regression_outcome PASS '')"
expect_str "classify FAIL:MISSING_BASELINE -> NOT_VERIFIED" "NOT_VERIFIED" "$(classify_regression_outcome FAIL MISSING_BASELINE)"
expect_str "classify FAIL:INCOMPARABLE_ENVIRONMENT -> NOT_VERIFIED" "NOT_VERIFIED" "$(classify_regression_outcome FAIL INCOMPARABLE_ENVIRONMENT)"
expect_str "classify FAIL:REGRESSION_DETECTED -> FAIL"   "FAIL"          "$(classify_regression_outcome FAIL REGRESSION_DETECTED)"
expect_str "classify FAIL:Benchmark run failed -> FAIL" "FAIL"          "$(classify_regression_outcome FAIL 'Benchmark run failed')"
expect_str "classify empty receipt -> FAIL (fail-closed)" "FAIL"        "$(classify_regression_outcome '' '')"
expect_str "classify SKIP_CAPABILITY never promoted -> FAIL" "FAIL"     "$(classify_regression_outcome SKIP_CAPABILITY whatever)"

echo ""
echo "${BOLD}=== executed_bench_ids / missing_baseline_coverage (F-24) ===${NC}"

crit="$WORK/crit-root"
mkdir -p "$crit/RT_A/ci-baseline" "$crit/RT_B/ci-baseline"
cat > "$WORK/crit.log" <<'EOF'
Benchmarking RT_A: Warming up for 1.0000 s
Benchmarking RT_A: Collecting 100 samples
Benchmarking RT_B: Warming up for 1.0000 s
Benchmarking RT_C: Warming up for 1.0000 s
EOF

out="$(missing_baseline_coverage "$WORK/crit.log" "$crit" ci-baseline)"; rc=$?
expect_rc "missing_baseline_coverage: parse ok -> 0" 0 "$rc"
expect_str "missing_baseline_coverage: RT_C without series listed" "RT_C" "$out"

executed_bench_ids "$WORK/crit.log" > "$WORK/ids.txt"; rc=$?
expect_str "executed_bench_ids dedups and sorts" "RT_A
RT_B
RT_C" "$(cat "$WORK/ids.txt")"

mkdir -p "$crit/RT_C/ci-baseline"
out="$(missing_baseline_coverage "$WORK/crit.log" "$crit" ci-baseline)"; rc=$?
expect_str "missing_baseline_coverage: full coverage -> empty" "" "$out"
expect_rc "missing_baseline_coverage: full coverage -> 0" 0 "$rc"

printf 'garbage log with no criterion lines\n' > "$WORK/nobench.log"
out="$(missing_baseline_coverage "$WORK/nobench.log" "$crit" ci-baseline)"; rc=$?
expect_rc "missing_baseline_coverage: unparseable log -> 1 (blind gate)" 1 "$rc"

out="$(missing_baseline_coverage "$WORK/absent-crit.log" "$crit" ci-baseline)"; rc=$?
expect_rc "missing_baseline_coverage: absent log -> 1" 1 "$rc"

echo ""
echo "${BOLD}=== run_extended_audit (F-06 --full delegation) ===${NC}"

LOGDIR="$WORK/logdir2"; mkdir -p "$LOGDIR"
DASHBOARD_PHASE_RECEIPT="$WORK/receipt2.jsonl"; rm -f "$DASHBOARD_PHASE_RECEIPT"
DASHBOARD_PHASE_HAD_FAILURE=0

cat > "$WORK/stub-long-ok.sh" <<'EOF'
#!/bin/bash
echo "stub long suite ran"
exit 0
EOF
cat > "$WORK/stub-long-fail.sh" <<'EOF'
#!/bin/bash
echo "stub long suite failed"
exit 3
EOF

NAM_LONG_SUITE_SCRIPT="$WORK/stub-long-ok.sh"
run_extended_audit >/dev/null 2>&1; rc=$?
expect_rc "run_extended_audit: stub exit 0 -> rc 0" 0 "$rc"
receipt_has long_suite PASS && pass "run_extended_audit: receipt long_suite PASS" || fail "run_extended_audit: receipt long_suite PASS"

NAM_LONG_SUITE_SCRIPT="$WORK/stub-long-fail.sh"
run_extended_audit >/dev/null 2>&1; rc=$?
expect_rc "run_extended_audit: stub exit 3 -> rc 0 (flag, not abort)" 0 "$rc"
receipt_has long_suite FAIL "delegated tests-long.sh failed" && pass "run_extended_audit: receipt long_suite FAIL" || fail "run_extended_audit: receipt long_suite FAIL"
expect_rc "run_extended_audit: FAIL sets DASHBOARD_PHASE_HAD_FAILURE" 1 "${DASHBOARD_PHASE_HAD_FAILURE:-0}"

NAM_LONG_SUITE_SCRIPT="$WORK/definitely-missing-script.sh"
run_extended_audit >/dev/null 2>&1; rc=$?
expect_rc "run_extended_audit: missing script -> rc 0 (typed receipt)" 0 "$rc"
receipt_has long_suite FAIL long_suite_script_missing && pass "run_extended_audit: missing script -> FAIL/long_suite_script_missing" || fail "run_extended_audit: missing script -> FAIL/long_suite_script_missing"
unset NAM_LONG_SUITE_SCRIPT

echo ""
echo "${BOLD}=== ensure_namcore_render (S3-T01 unified C++ render build) ===${NC}"

# NOTE: run_extended_audit (above) leaves `set -e` active in this shell, so
# every non-zero-rc capture below uses an if/else context (never `cmd; rc=$?`).

mkdir -p "$WORK/emptybin"
if PATH="$WORK/emptybin" ensure_namcore_render >/dev/null 2>&1; then rc=0; else rc=$?; fi
expect_rc "ensure_namcore_render: cmake missing -> rc 2" 2 "$rc"

if CXX="$WORK/missing-cxx" ensure_namcore_render >/dev/null 2>&1; then rc=0; else rc=$?; fi
expect_rc "ensure_namcore_render: invalid CXX -> rc 1" 1 "$rc"

CXX_REAL="$(command -v g++ 2>/dev/null || command -v clang++ 2>/dev/null || true)"
if [ -n "$CXX_REAL" ]; then
    if CXX="$CXX_REAL" NAM_CORE_DIR="$WORK/nonexistent-core" ensure_namcore_render >/dev/null 2>&1; then rc=0; else rc=$?; fi
    expect_rc "ensure_namcore_render: missing NAMCore -> rc 3" 3 "$rc"
else
    skip "ensure_namcore_render: missing NAMCore (no C++ compiler in PATH)"
fi

# Fake cmake + fake compiler: exercises configure/build/fingerprint/idempotency
# without a real C++ build. The fake cmake logs every invocation and fabricates
# a render binary at `tools/render` (the Makefiles-generator layout).
mkdir -p "$WORK/bin" "$WORK/namcore-mock" "$WORK/rb"
FAKE_CMAKE="$WORK/bin/cmake"
cat > "$FAKE_CMAKE" <<'EOF'
#!/bin/bash
echo "invoked: $*" >> "${CMAKE_CALL_LOG:-/dev/null}"
BUILD_D=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        -B|--build) shift; BUILD_D="$1" ;;
    esac
    shift
done
mkdir -p "$BUILD_D/tools"
printf '#!/bin/bash\nexit 0\n' > "$BUILD_D/tools/render"
chmod +x "$BUILD_D/tools/render"
exit 0
EOF
chmod +x "$FAKE_CMAKE"
printf '#!/bin/bash\nexit 0\n' > "$WORK/bin/fake-cxx"
chmod +x "$WORK/bin/fake-cxx"
printf '#!/bin/bash\nexit 0\n' > "$WORK/bin/fake-clang"
chmod +x "$WORK/bin/fake-clang"

CMAKE_CALL_LOG="$WORK/cmake-calls.log"; rm -f "$CMAKE_CALL_LOG"

out="$(PATH="$WORK/bin:$PATH" CXX=fake-cxx NAM_CORE_DIR="$WORK/namcore-mock" NAM_RENDER_BUILD_DIR="$WORK/rb" CMAKE_CALL_LOG="$CMAKE_CALL_LOG" ensure_namcore_render 2>/dev/null)"; rc=$?
expect_rc "ensure_namcore_render: cold build -> rc 0" 0 "$rc"
expect_str "ensure_namcore_render: cold build prints binary path" "$WORK/rb/tools/render" "$out"
expect_str "ensure_namcore_render: cold build writes .build_config" "fake-cxx:Release:-w -fno-fast-math -ffp-contract=off" "$(cat "$WORK/rb/.build_config" 2>/dev/null)"
CALLS_COLD=$(wc -l < "$CMAKE_CALL_LOG")
expect_str "ensure_namcore_render: cold build invokes cmake twice" "2" "$CALLS_COLD"

out="$(PATH="$WORK/bin:$PATH" CXX=fake-cxx NAM_CORE_DIR="$WORK/namcore-mock" NAM_RENDER_BUILD_DIR="$WORK/rb" CMAKE_CALL_LOG="$CMAKE_CALL_LOG" ensure_namcore_render 2>/dev/null)"; rc=$?
expect_rc "ensure_namcore_render: warm run -> rc 0" 0 "$rc"
CALLS_WARM=$(wc -l < "$CMAKE_CALL_LOG")
expect_str "ensure_namcore_render: warm run skips cmake entirely (idempotent)" "$CALLS_COLD" "$CALLS_WARM"

out="$(PATH="$WORK/bin:$PATH" CXX=fake-clang NAM_CORE_DIR="$WORK/namcore-mock" NAM_RENDER_BUILD_DIR="$WORK/rb" CMAKE_CALL_LOG="$CMAKE_CALL_LOG" ensure_namcore_render 2>/dev/null)"; rc=$?
expect_rc "ensure_namcore_render: compiler change -> rc 0 (rebuild)" 0 "$rc"
CALLS_SWITCH=$(wc -l < "$CMAKE_CALL_LOG")
expect_str "ensure_namcore_render: compiler change triggers fresh build" "$((CALLS_COLD + 2))" "$CALLS_SWITCH"
expect_str "ensure_namcore_render: fingerprint follows CXX" "fake-clang:Release:-w -fno-fast-math -ffp-contract=off" "$(cat "$WORK/rb/.build_config" 2>/dev/null)"

if PATH="$WORK/bin:$PATH" CXX=fake-clang NAM_CORE_DIR="$WORK/namcore-mock" NAM_RENDER_BUILD_DIR="$WORK/rb" CMAKE_CALL_LOG="$CMAKE_CALL_LOG" NAM_RENDER_FORCE=1 ensure_namcore_render >/dev/null 2>&1; then rc=0; else rc=$?; fi
expect_rc "ensure_namcore_render: NAM_RENDER_FORCE=1 -> rc 0" 0 "$rc"
CALLS_FORCE=$(wc -l < "$CMAKE_CALL_LOG")
expect_str "ensure_namcore_render: NAM_RENDER_FORCE=1 forces reconfigure+build" "$((CALLS_SWITCH + 2))" "$CALLS_FORCE"

if PATH="$WORK/bin:$PATH" CXX=fake-clang NAM_CORE_DIR="$WORK/namcore-mock" NAM_RENDER_BUILD_DIR="$WORK/rb" CMAKE_CALL_LOG="$CMAKE_CALL_LOG" NAM_RENDER_BUILD_TYPE=Debug ensure_namcore_render >/dev/null 2>&1; then rc=0; else rc=$?; fi
expect_rc "ensure_namcore_render: build-type change -> rc 0 (rebuild)" 0 "$rc"
expect_str "ensure_namcore_render: fingerprint follows build type" "fake-clang:Debug:-w -fno-fast-math -ffp-contract=off" "$(cat "$WORK/rb/.build_config" 2>/dev/null)"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  bash defense unit tests — passed=${PASSED} failed=${FAILED} skipped=${SKIPPED} (total=${TOTAL})"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failures:"
    for f in "${FAILED_NAMES[@]}"; do echo "    - ${f}"; done
    echo "  RESULT: FAIL"
    exit 1
fi
echo "  RESULT: PASS"
exit 0
    )
}

run_defense_scripts_phase() {
    local status=0
    echo "  → run_bash_scripts_unit_tests (bash defense helpers)"
    run_bash_scripts_unit_tests || status=1
    echo "  → libm_export_guard (Rust ELF surface, release)"
    timed_cargo_test "libm_export_guard" --release --no-fail-fast --test libm_export_guard -- --nocapture || status=1
    echo "  → oversample hang bound (lib unit, 60s execution ceiling)"
    # The 60s ceiling bounds the TEST (hang defense), not the compile: a cold
    # release build of the lib unit-test binary can exceed 60s (observed
    # ~2 min on the certifying host) and was previously killed mid-compile,
    # failing the phase without ever running the test. Compile first without
    # a ceiling, then run the filtered test under the ceiling.
    cargo test --features testing --release --lib --no-run || status=1
    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=KILL 60 cargo test --features testing --release --lib -- \
            dsp::oversample::oversample_test:: --nocapture --test-threads=1 \
            || status=1
    else
        cargo test --features testing --release --lib -- \
            dsp::oversample::oversample_test:: --nocapture --test-threads=1 \
            || status=1
    fi
    return $status
}
run_phase "Defense scripts + libm + oversample bound" "run_defense_scripts_phase" "phase-defense-scripts.log" || true
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase-defense-scripts.log" || true

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
    timed_cargo_test "cabsim_golden_heavy" --release --no-fail-fast $(_test_flag cabsim_golden) -- --ignored --nocapture || status=1
    timed_cargo_test "oversampling_characterization" --release --no-fail-fast $(_test_flag oversampling_characterization) -- --ignored --nocapture || status=1
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
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase2-proptests-parity.log" || true

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
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase3-heap-audit.log" || true

# --- Phase 4: RT Deadline Gate (deterministic, hard assertion) ---
# Absolute latency ceiling: assert!(p99 < 1.33 ms) for every model SKU.
# This is the definitive gate — a regression that pushes p99 past the
# audio buffer deadline fails the build deterministically.
#
# Runs under explicit CPU core pinning (taskset -c $BENCH_CORE) so that
# the preflight check inside rt_deadline.rs can certify the environment.
# When preflight detects an uncontrolled environment, tail/max values are
# classified as INCONCLUSIVE_ENVIRONMENT and the gate is not certified.
run_rt_deadline_gate_phase() {
    local status=0
    local start_t
    start_t=$(date +%s%N)
    local flag=$(_test_flag rt_deadline)
    if [ "$HAS_TASKSET" = "1" ] && [ -n "${BENCH_CORE:-}" ]; then
        taskset -c "${BENCH_CORE}" cargo test --features testing --release --no-fail-fast $flag -- --nocapture || status=$?
    else
        cargo test --features testing --release --no-fail-fast $flag -- --nocapture || status=$?
    fi
    local end_t
    end_t=$(date +%s%N)
    local duration_ns=$((end_t - start_t))
    local duration_s
    duration_s=$(LC_NUMERIC=C awk -v ns="$duration_ns" 'BEGIN { printf "%.3f", ns / 1000000000 }')
    echo "TIMED: $duration_s rt_deadline" >> "$TIMED_TRACKER"
    return $status
}
run_phase "RT Deadline Gate (deterministic)" "run_rt_deadline_gate_phase" "phase4-rt-deadline.log" || true

# Post-phase: detect INCONCLUSIVE_ENVIRONMENT from the deadline log.
# When the Rust preflight reports uncontrolled environment, the test exits 0
# (no assertion) but the log contains the marker — this is NOT a PASS.
DEADLINE_LOG="target/logs/phase4-rt-deadline.log"
if grep -qF "INCONCLUSIVE_ENVIRONMENT" "$DEADLINE_LOG" 2>/dev/null; then
    DEADLINE_IDX=$((PHASE_COUNT - 1))
    PHASE_STATUS[$DEADLINE_IDX]="INCONCLUSIVE"
fi
# Emit after the status override so the receipt carries the authoritative status.
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase4-rt-deadline.log" || true

# --- Phase 5: RT Jitter Characterization (environmental telemetry) ---
# Characterizes tail latency under CPU contention. This is diagnostic
# telemetry — it does NOT assert deadlines under stress. An INCONCLUSIVE
# result is expected when environment preconditions (CPU pinning,
# performance governor, low background load) are not met.
run_rt_jitter_characterization_phase() {
    # Same core pin as Phase 4: jitter preflight requires single-CPU affinity.
    local status=0
    local start_t
    start_t=$(date +%s%N)
    local flag=$(_test_flag rt_jitter)
    if [ "$HAS_TASKSET" = "1" ] && [ -n "${BENCH_CORE:-}" ]; then
        taskset -c "${BENCH_CORE}" cargo test --features testing --release --no-fail-fast $flag -- --ignored --nocapture || status=$?
    else
        cargo test --features testing --release --no-fail-fast $flag -- --ignored --nocapture || status=$?
    fi
    local end_t
    end_t=$(date +%s%N)
    local duration_ns=$((end_t - start_t))
    local duration_s
    duration_s=$(LC_NUMERIC=C awk -v ns="$duration_ns" 'BEGIN { printf "%.3f", ns / 1000000000 }')
    echo "TIMED: $duration_s rt_jitter" >> "$TIMED_TRACKER"
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
# Emit after the status override so the receipt carries the authoritative status.
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase5-rt-jitter.log" || true

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
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase6-loom.log" || true

# --- Print beautifully structured summary ---
echo -e "\n${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                  AUDIT SUMMARY REPORT                          ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
printf " | %-45s | %-10s | %-10s |\n" "Phase Name" "Duration" "Status"
printf " |-%-45s-|-%-10s-|-%-10s-|\n" "---------------------------------------------" "----------" "----------"

ANY_FAILED=0
ANY_FIDELITY_FAILED=0
COUNT_PASSED=0
COUNT_FAILED=0
COUNT_INCONCLUSIVE=0
COUNT_SKIP=0
COUNT_NOT_RUN=0

RT_DEADLINE_STATUS="PASS"
RT_JITTER_STATUS="PASS"
for ((i=0; i<PHASE_COUNT; i++)); do
    name="${PHASE_NAMES[$i]}"
    duration="${PHASE_DURATIONS[$i]}s"
    status="${PHASE_STATUS[$i]}"
    class="${PHASE_CLASSES[$i]:-fidelity}"

    case "$status" in
        PASSED)
            COUNT_PASSED=$((COUNT_PASSED + 1))
            status_colored="${GREEN}${status}${NC}"
            ;;
        SKIPPED|SKIP_CAPABILITY)
            COUNT_SKIP=$((COUNT_SKIP + 1))
            status_colored="${YELLOW}${status}${NC}"
            ;;
        INCONCLUSIVE)
            COUNT_INCONCLUSIVE=$((COUNT_INCONCLUSIVE + 1))
            status_colored="${YELLOW}${status}${NC}"
            ;;
        NOT_RUN)
            COUNT_NOT_RUN=$((COUNT_NOT_RUN + 1))
            status_colored="${YELLOW}${status}${NC}"
            ;;
        *)
            COUNT_FAILED=$((COUNT_FAILED + 1))
            status_colored="${RED}${status}${NC}"
            ANY_FAILED=1
            if [ "$class" != "performance" ]; then
                ANY_FIDELITY_FAILED=1
            fi
            ;;
    esac

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

# ── Structured long-audit receipt: suite-level `overall` line (Sprint S3-T04) ──
# Derives the verdict from the phase lines already emitted (PASSED /
# FAILED / COMPLETED_WITH_GAPS) and appends it before the final verdict.
# Failure to produce a valid receipt is fail-closed: the suite exits 1 below.
if [ "$PHASE_COUNT" -gt 0 ] && ensure_long_receipt_bin; then
    if ! "$LONG_RECEIPT_BIN" summary --out "$LONG_RECEIPT_FILE"; then
        echo -e "  ${YELLOW}${BOLD}⚠ Long-audit receipt summary emission failed${NC}" >&2
        LONG_RECEIPT_FAILED=1
    fi
else
    LONG_RECEIPT_FAILED=1
fi

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

HAS_GAPS=0
if [ "$COUNT_INCONCLUSIVE" -gt 0 ] || [ "$COUNT_SKIP" -gt 0 ] || [ "$COUNT_NOT_RUN" -gt 0 ] || \
   [ "$RT_JITTER_STATUS" != "PASS" ] || [ "$RT_DEADLINE_STATUS" != "PASS" ]; then
    HAS_GAPS=1
fi

if [ $ANY_FAILED -ne 0 ]; then
    echo -e "${RED}${BOLD}❌ One or more audit stages failed. Check logs in target/logs/${NC}"
    if [ $ANY_FIDELITY_FAILED -eq 1 ]; then
        echo -e "${RED}FIDELITY: FAIL${NC}"
    else
        echo -e "${GREEN}FIDELITY: OK${NC}"
    fi
    render_performance_summary
    echo -e "${RED}${BOLD}OVERALL: FAILED${NC}"
    exit 1
elif [ $HAS_GAPS -eq 1 ]; then
    echo -e "${YELLOW}${BOLD}⚠ Audit completed with declared gaps (inconclusive / skipped / unexecuted stages).${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    render_performance_summary
    echo -e "${YELLOW}${BOLD}OVERALL: COMPLETED_WITH_GAPS${NC}"

    if [ "${LONG_RECEIPT_FAILED:-0}" -eq 1 ]; then
        echo -e "${RED}${BOLD}❌ Long-audit receipt emission failed — target/logs/long-audit-receipt.jsonl is missing or incomplete.${NC}"
        exit 1
    fi

    if [ "${STRICT_PRE_RELEASE:-0}" -eq 1 ]; then
        echo -e "${RED}${BOLD}❌ --strict-pre-release mode: failing audit due to declared gaps (INCONCLUSIVE/NOT_RUN).${NC}"
        exit 1
    else
        exit 0
    fi
else
    echo -e "${GREEN}${BOLD}✓ All audit stages completed successfully!${NC}"
    echo -e "${GREEN}FIDELITY: OK${NC}"
    render_performance_summary
    echo -e "${GREEN}${BOLD}OVERALL: PASSED${NC}"

    if [ "${LONG_RECEIPT_FAILED:-0}" -eq 1 ]; then
        echo -e "${RED}${BOLD}❌ Long-audit receipt emission failed — target/logs/long-audit-receipt.jsonl is missing or incomplete.${NC}"
        exit 1
    fi

    exit 0
fi
