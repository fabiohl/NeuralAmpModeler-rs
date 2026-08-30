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
# Structured audit receipt: every
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
# (NAM_SKIP_GOLDEN_BUILD / deprecated NAM_AUTO_BUILD_GOLDENS) were removed:
# regenerate goldens manually with tests/fixtures/golden_gen_build.sh when
# the preflight reports a missing file.

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
# and any other artifacts from independent tools.
# long-audit-receipt.jsonl is regenerated by this suite.
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
      target/logs/subphase-isa-parity.log \
      target/logs/long-audit-receipt.jsonl

# Cleanup accumulated live-test artifacts from previous runs (41+ MB WAVs)
rm -rf tests/fixtures/.temp_live/

# ── Structured long-audit receipt ──────────────────────────────────────────
# Every completed phase AND every preflight step appends one JSONL line
# (phase_id, name, status, duration_ms, tests_executed, gaps, timestamp) to
# target/logs/long-audit-receipt.jsonl via the Rust emitter
# `nam_long_receipt append` — no fragile bash JSON generation. The suite-level
# `overall` line is appended by `nam_long_receipt summary` before the verdict.
# Preflight steps (preflight-*) run ahead of Phase 1: an
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
# (no bash golden lists):
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

# ── Render binary — unified build ──
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

# ── Catalog Preflight — Capability Receipt ──
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

# ── Package Exclusion Verification ──
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
# the approximately 10-minute battery, so a drifted catalog fails fast instead of burning a
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
# The fidelity/performance split itself moved to Rust in S5:
# src/testing/receipt.rs::PERFORMANCE_PHASE_IDS (phase5 = RT Deadline,
# phase6 = RT Jitter) — the human summary derives FIDELITY from the receipt.

# Trackers for the final summary
declare -a PHASE_NAMES
declare -a PHASE_STATUS
declare -a PHASE_DURATIONS
PHASE_COUNT=0

# timed_cargo_test — runs a cargo test invocation, propagates its status.
# Usage: timed_cargo_test <label> <cargo_test_args...>
timed_cargo_test() {
    local label="$1"
    shift
    cargo test --features testing "$@"
    local status=$?
    return $status
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

    # Run command and capture output/status
    eval "$cmd" > "target/logs/$log_file" 2>&1
    local status=$?

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    PHASE_NAMES[$PHASE_COUNT]="$name"
    PHASE_DURATIONS[$PHASE_COUNT]="$duration"

    # T2.4: the legacy exit-code 77 skip convention is dead — skips are now
    # conveyed exclusively by typed `[STATUS]` log markers, picked up by
    # `nam_long_receipt append --log` (detect_gap_markers) and recorded as
    # gaps in the structured receipt. `run_phase` only distinguishes PASSED
    # from FAILED; a phase that internally bypassed its measurement exits 0
    # but its typed markers downgrade the suite verdict to
    # COMPLETED_WITH_GAPS — never a clean PASSED.
    if [ $status -eq 0 ]; then
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
# Appends the just-completed phase's structured receipt line; `--log` makes
# `nam_long_receipt append` detect typed gap markers in the phase log
# (detect_gap_markers — the S5 carrier of measurement bypasses; the old
# post-phase PHASE_STATUS overrides for phases 4/5 are gone). A failure
# flags LONG_RECEIPT_FAILED (fail-closed at the final verdict) but never
# rewrites the phase's own outcome.
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

# Defense phase (EP-05 / R-05): the inline bash unit-test suite
# acceptance cases — F-01 metric sanitization, F-08 single NOT_VERIFIED
# classifier, F-21 executed-tests counter, F-22 freshness gate, F-24 baseline
# coverage, F-27 canonical JSONL parse, C++ render build — now live in
# the Rust harness tests/qa_defense.rs (one `--test qa_defense` call replaces
# ~470 lines of sed/eval extraction). The bash-only orchestration tests
# (run_dashboard_phase, run_phase0_freshness, run_extended_audit, render
# helpers) died with their functions; the dashboard conversion owns
# their Rust replacements.

run_defense_scripts_phase() {
    local status=0
    echo "  → qa_defense (Rust defense harness: F-01/F-08/F-21/F-22/F-24/F-27)"
    timed_cargo_test "qa_defense" --no-fail-fast --test qa_defense || status=1
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
    #
    # T2.4: the matrix runs under its OWN subphase log so the mandatory-
    # subphase gate can prove real execution. On a default local runner
    # (no `--features avx512`) the cross-ISA cases compile out via `#[cfg]` —
    # the AVX2 self-consistency cases still prove the subphase ran, and the
    # `AVX512_OPT_IN:` declaration below (echoed into the phase log, parsed
    # by nam_long_receipt detect_gap_markers) records the missing opt-in as a
    # typed gap instead of silently promoting a zero-case matrix to PASSED.
    local SUBLOG="target/logs/subphase-isa-parity.log"
    cargo test --features testing --release --no-fail-fast $(_test_flag isa_parity) -- \
        --include-ignored --test-threads=1 --nocapture 2>&1 | tee -a "$SUBLOG" \
        >> "target/logs/phase2-proptests-parity.log" || status=1
    assert_subphase_ran "isa_parity_full_matrix" "$SUBLOG" 1 || status=1
    if grep -qE "AVX-512" "$SUBLOG"; then
        echo "AVX512_OPT_IN: RUN (cross-ISA AVX-512 matrix compiled and exercised)"
    else
        echo "AVX512_OPT_IN: NOT_RUN (default runner without --features avx512)"
    fi
    # Spectral fidelity baseline immutability defense:
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

# Post-phase immutability gate: if the log contains the
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
    local flag=$(_test_flag rt_deadline)
    if [ "$HAS_TASKSET" = "1" ] && [ -n "${BENCH_CORE:-}" ]; then
        taskset -c "${BENCH_CORE}" cargo test --features testing --release --no-fail-fast $flag -- --nocapture || status=$?
    else
        cargo test --features testing --release --no-fail-fast $flag -- --nocapture || status=$?
    fi
    return $status
}
run_phase "RT Deadline Gate (deterministic)" "run_rt_deadline_gate_phase" "phase4-rt-deadline.log" || true

# The receipt carries the bypass typed in the log: when the Rust preflight
# reports an uncontrolled environment the test exits 0 (no assertion) but the
# log contains the INCONCLUSIVE_ENVIRONMENT marker. `nam_long_receipt append
# --log` picks it up via detect_gap_markers (S5 — the PHASE_STATUS override
# is gone; the marker is the only carrier, and the summary derives
# COMPLETED_WITH_GAPS from it).
emit_long_phase_receipt "$((PHASE_COUNT - 1))" "phase4-rt-deadline.log" || true

# --- Phase 5: RT Jitter Characterization (environmental telemetry) ---
# Characterizes tail latency under CPU contention. This is diagnostic
# telemetry — it does NOT assert deadlines under stress. An INCONCLUSIVE
# result is expected when environment preconditions (performance governor,
# low background load) are not met.
run_rt_jitter_characterization_phase() {
    # Unlike Phase 4 (which pins to a single core for isolated deadline gating),
    # Phase 5 characterizes multi-worker CPU contention and must NOT be confined
    # to a single core. The process must see all available cores so stress-1,
    # stress-2, and saturate-N execute real concurrent load across CPUs.
    local status=0
    local flag=$(_test_flag rt_jitter)
    cargo test --features testing --release --no-fail-fast $flag -- --ignored --nocapture || status=$?
    return $status
}
run_phase "RT Jitter Characterization" "run_rt_jitter_characterization_phase" "phase5-rt-jitter.log" || true

# Guard against regression: if host has multiple cores, Phase 5 must never
# report 'Single core affinity' bypass.
if [ "${NUM_CORES:-1}" -ge 2 ] && grep -q "Single core affinity" "target/logs/phase5-rt-jitter.log" 2>/dev/null; then
    echo -e "  ${RED}${BOLD}❌ REGRESSION: Phase 5 reported 'Single core affinity' on a machine with ${NUM_CORES} cores!${NC}" >&2
fi

# The Rust test returns exit 0 even when internally bypassed (INCONCLUSIVE
# or SKIP_CAPABILITY) to avoid false FAIL — the [STATUS] log markers are
# authoritative and `nam_long_receipt append --log` records them as typed
# gaps (S5: the PHASE_STATUS override is gone). Invariant: exit-0 with
# internal measurement bypass SHALL NOT be promoted to PASS.
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

# ── Human summary ──────────────────────────────────────────────────────────
# The verdict is derived ONCE, in Rust: `nam_long_receipt summary` appends
# the suite-level `overall` line to long-audit-receipt.jsonl (PASSED /
# FAILED / COMPLETED_WITH_GAPS, with declared gaps) and prints the human
# one-liners — WARNING/ERROR alarms per phase plus OVERALL / FIDELITY /
# RT_DEADLINE / RT_JITTER / PERF_REGRESSION. Bash only echoes those lines
# verbatim and maps `OVERALL:` to the exit code; it never reclassifies logs
# (the giant ASCII table, the top-N slowest block and the per-phase status
# reclassification are gone — forensics live in the JSONL:
# humans get alarms, agents get data).
if [ "$PHASE_COUNT" -gt 0 ] && ensure_long_receipt_bin; then
    if ! SUMMARY_TEXT="$("$LONG_RECEIPT_BIN" summary --out "$LONG_RECEIPT_FILE")"; then
        echo -e "  ${YELLOW}${BOLD}⚠ Long-audit receipt summary emission failed${NC}" >&2
        LONG_RECEIPT_FAILED=1
    fi
else
    LONG_RECEIPT_FAILED=1
fi

if [ "${LONG_RECEIPT_FAILED:-0}" -eq 1 ]; then
    echo -e "${RED}${BOLD}❌ Long-audit receipt emission failed — target/logs/long-audit-receipt.jsonl is missing or incomplete.${NC}"
    exit 1
fi

echo -e "\n${BLUE}${BOLD}================ AUDIT SUMMARY ================${NC}"
printf '%s\n' "$SUMMARY_TEXT"

echo -e "\n${GREEN}${BOLD}================================================================================${NC}"
echo -e "  ${BOLD}Audit Artifacts saved:${NC}"
echo -e "    - JSONL Receipt: ${CYAN}$LONG_RECEIPT_FILE${NC}"
echo -e "    - Phase Logs:    ${CYAN}target/logs/phase*.log${NC}"
echo -e "${GREEN}${BOLD}================================================================================${NC}\n"

case "$SUMMARY_TEXT" in
    *"OVERALL: FAILED"*)
        echo -e "${RED}${BOLD}❌ One or more audit stages failed. Check target/logs/long-audit-receipt.jsonl${NC}"
        exit 1
        ;;
    *"OVERALL: COMPLETED_WITH_GAPS"*)
        echo -e "${YELLOW}${BOLD}⚠ Audit completed with declared gaps (inconclusive / skipped / unexecuted stages).${NC}"
        if [ "${STRICT_PRE_RELEASE:-0}" -eq 1 ]; then
            echo -e "${RED}${BOLD}❌ --strict-pre-release mode: failing audit due to declared gaps (INCONCLUSIVE/NOT_RUN).${NC}"
            exit 1
        fi
        exit 0
        ;;
    *)
        echo -e "${GREEN}${BOLD}✓ All audit stages completed successfully!${NC}"
        exit 0
        ;;
esac
