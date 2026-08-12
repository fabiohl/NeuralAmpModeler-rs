# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# _lib.sh — Common bash utilities for NeuralAmpModeler-rs scripts.
#
# Source with:
#   PHASE_TOTAL=<N>; source "$(dirname "$0")/_lib.sh"
# or for scripts not in utils/:
#   PHASE_TOTAL=<N>; source "$PROJECT_ROOT/utils/_lib.sh"
#
# Then call:
#   phase "Description of the current step"

# ANSI style helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PHASE_NUM=0

phase() {
    PHASE_NUM=$((PHASE_NUM + 1))
    echo -e "\n${BLUE}${BOLD}[${PHASE_NUM}/${PHASE_TOTAL:-?}]${NC} $*"
}

# ── Phase receipt machinery (fail-closed foundation) ──────────────────────────
# Each dashboard phase records a typed outcome in JSONL.
# Schema: phase_id, status (PASS|FAIL|SKIP_CAPABILITY|SKIP_OPTIONAL_FIXTURE|NOT_RUN),
#         exit_code, observed_records, expected_records, reason

DASHBOARD_PHASE_RECEIPT="${DASHBOARD_PHASE_RECEIPT:-}"
DASHBOARD_PHASE_HAD_FAILURE=0

# Register a phase receipt entry in the JSONL stream.
# Must be called with DASHBOARD_PHASE_RECEIPT set to a writable file path.
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

# Run a dashboard phase with strict exit code capture.
#
# Usage: run_dashboard_phase <phase_id> <min_expected_records> [command...]
#
# Executes the command, captures its exit code, counts output lines in the
# expected log file (derived from LOGDIR/<phase_id>.log convention used by
# quality-dashboard.sh), and writes a typed receipt entry.
#
# The command MUST include shell redirection to its log file (e.g. `> "$LOGDIR/xy.log" 2>&1`).
# The wrapper does NOT suppress failures with || true — the exit code is captured and
# the phase continues collecting further phases. A global failure flag is set for the
# final exit code.
#
# Returns: 0 always (to keep the dashboard collecting), but sets DASHBOARD_PHASE_HAD_FAILURE=1
#          when exit_code != 0 or observed_records < min_expected_records.
run_dashboard_phase() {
    local phase_id="$1" min_records="$2"
    shift 2

    local log_path="$LOGDIR/${phase_id}.log"

    echo -e "${BLUE}${BOLD}-> Running ${phase_id}...${NC}"

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
    else
        echo -e "  ${GREEN}ok${NC} ${phase_id} completed (${dur_s}s, ${observed} lines)"
    fi

    dashboard_phase_receipt "$phase_id" "$status" "$exit_code" "$observed" "$min_records" "$reason"

    return 0
}

# Resolve project root dynamically relative to this helper script
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$LIB_DIR")"

# Repo-local third-party area (gitignored): vendor mirrors
# (NeuralAmpModelerCore, NeuralAmpModelerPlugin) and optional community_models.
# Populated by utils/setup-third-party.sh. Pins live in variables.env.
# Override base with NAM_THIRD_PARTY_DIR, or individual trees with NAM_CORE_DIR /
# NAM_PLUGIN_DIR, when the layout differs.
THIRD_PARTY_DIR="${NAM_THIRD_PARTY_DIR:-$PROJECT_DIR/third-party}"
NAM_CORE_DIR="${NAM_CORE_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerCore}"
NAM_PLUGIN_DIR="${NAM_PLUGIN_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerPlugin}"
VARIABLES_ENV="${NAM_VARIABLES_ENV:-$PROJECT_DIR/variables.env}"
SETUP_THIRD_PARTY_SH="${SETUP_THIRD_PARTY_SH:-$PROJECT_DIR/utils/setup-third-party.sh}"

# Automatically enter the project root directory
cd "$PROJECT_DIR"

# Ensure vendor mirrors exist when a script needs them.
#
# Modes:
#   soft  — if Core is missing, run setup-third-party.sh once; on failure or
#           still-missing Core, print a warning and return 1 (caller may SKIP).
#   hard  — same attempt; if Core is still missing afterward, return 1 and the
#           caller should abort (used by long / golden pipelines).
#
# community_models remains optional and is never required here.
# Skip auto-setup with NAM_SKIP_THIRD_PARTY_SETUP=1.
ensure_third_party() {
    local mode="${1:-soft}"

    if [ -d "$NAM_CORE_DIR" ] && [ -e "$NAM_CORE_DIR/.git" ]; then
        return 0
    fi

    if [ "${NAM_SKIP_THIRD_PARTY_SETUP:-0}" = "1" ]; then
        echo -e "  ${YELLOW}ⓘ NAM_SKIP_THIRD_PARTY_SETUP=1 — third-party auto-setup skipped.${NC}"
        return 1
    fi

    if [ ! -x "$SETUP_THIRD_PARTY_SH" ] && [ ! -f "$SETUP_THIRD_PARTY_SH" ]; then
        echo -e "  ${YELLOW}ⓘ setup-third-party.sh not found at $SETUP_THIRD_PARTY_SH.${NC}"
        return 1
    fi

    echo -e "  ${BLUE}→ third-party mirrors missing — running utils/setup-third-party.sh...${NC}"
    if ! bash "$SETUP_THIRD_PARTY_SH"; then
        echo -e "  ${YELLOW}ⓘ setup-third-party.sh failed.${NC}"
        return 1
    fi

    # Re-resolve in case the script used overrides / created trees.
    THIRD_PARTY_DIR="${NAM_THIRD_PARTY_DIR:-$PROJECT_DIR/third-party}"
    NAM_CORE_DIR="${NAM_CORE_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerCore}"
    NAM_PLUGIN_DIR="${NAM_PLUGIN_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerPlugin}"

    if [ -d "$NAM_CORE_DIR" ]; then
        echo -e "  ${GREEN}✓ third-party ready ($NAM_CORE_DIR).${NC}"
        return 0
    fi

    if [ "$mode" = "hard" ]; then
        echo -e "  ${RED}${BOLD}❌ NeuralAmpModelerCore still missing at $NAM_CORE_DIR after setup.${NC}"
    else
        echo -e "  ${YELLOW}ⓘ NeuralAmpModelerCore still missing at $NAM_CORE_DIR — dependent stages will SKIP.${NC}"
    fi
    return 1
}

# ── Toolchain fingerprint check (F-I4 / Tarefa 3.2) ──────────────────────────
# Reads the # TOOLCHAIN: lines from .golden_manifest.sha256 and compares
# against the current toolchain.  Mismatch emits a YELLOW warning but returns
# 0 (does not block the test suite) — the fingerprint is diagnostic, not
# authoritative.  See docs/cpp_parity_map.md §1.3 for drift context.
check_toolchain_fingerprint() {
    local MANIFEST="tests/fixtures/.golden_manifest.sha256"

    if [ ! -f "$MANIFEST" ]; then
        return 0
    fi

    local CXX_NOW
    CXX_NOW=$(${CXX:-g++} --version 2>/dev/null | head -1 || echo "unknown")
    local CMAKE_NOW
    CMAKE_NOW=$(cmake --version 2>/dev/null | head -1 || echo "unknown")
    local GLIBC_NOW
    if GLIBC_NOW=$(ldd --version 2>/dev/null | head -1); then :; else
        GLIBC_NOW=$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "unknown")
    fi
    local OS_NOW
    OS_NOW=$(uname -r 2>/dev/null || echo "unknown")

    local mismatch=0
    while IFS= read -r line; do
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx:\ (.*)$ ]]      && local F_CXX="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cmake:\ (.*)$ ]]     && local F_CMAKE="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ glibc:\ (.*)$ ]]     && local F_GLIBC="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ os:\ (.*)$ ]]        && local F_OS="${BASH_REMATCH[1]}"
        [[ "$line" =~ ^#\ TOOLCHAIN:\ cxx-flags:\ (.*)$ ]] && local F_FLAGS="${BASH_REMATCH[1]}"
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

# ── Centralized freshness gate (F-X4 / Tarefa 3.4) ──────────────────────────
# Unified freshness validator for golden manifest integrity.
# Usage: check_freshness <mode>
#   mode = hard-fail       → artifact integrity + generator provenance hard-fail
#                            (long suite / pre-release)
#   mode = artifacts-hard  → artifact integrity hard-fail; generator drift warn-only
#                            (quick suite — daily first line)
#   mode = warn-only       → all issues YELLOW; always returns 0
# Bypass: NAM_BYPASS_FRESHNESS=1 skips the entire check (returns 0).
# Validates:
#   1. Manifest existence                         (artifact)
#   2. EXPECTED golden files missing from disk    (artifact)
#   3. Catalog model↔golden SHA pairs             (artifact)
#   4. Standalone fixtures (hash integrity)       (artifact)
#   5. Generator scripts changed                  (provenance; severity by mode)
#   6. Reverse-check: orphan .nam in models/      (artifact)
#   7. Toolchain fingerprint drift                (always warn-only)
check_freshness() {
    local mode="${1:-hard-fail}"
    if [ "${NAM_BYPASS_FRESHNESS:-0}" = "1" ]; then
        echo -e "  ${YELLOW}⚠ NAM_BYPASS_FRESHNESS=1 — freshness check skipped${NC}"
        return 0
    fi

    local PREFIX=""
    local FAIL_PREFIX=""
    local STALE_PREFIX=""
    local GEN_HARD=0
    case "$mode" in
        warn-only)
            PREFIX="${YELLOW}⚠"
            FAIL_PREFIX="${YELLOW}${BOLD}⚠"
            STALE_PREFIX="${YELLOW}▲"
            GEN_HARD=0
            ;;
        artifacts-hard)
            PREFIX="${RED}${BOLD}❌"
            FAIL_PREFIX="${RED}${BOLD}❌"
            STALE_PREFIX="${RED}▲"
            GEN_HARD=0
            ;;
        hard-fail|*)
            PREFIX="${RED}${BOLD}❌"
            FAIL_PREFIX="${RED}${BOLD}❌"
            STALE_PREFIX="${RED}▲"
            GEN_HARD=1
            mode="hard-fail"
            ;;
    esac

    local MANIFEST="tests/fixtures/.golden_manifest.sha256"
    local MODELS_DIR="tests/fixtures/models"
    local FIXTURES_DIR="tests/fixtures"

    if [ ! -f "$MANIFEST" ]; then
        echo -e "${FAIL_PREFIX} Freshness manifest missing: $MANIFEST${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to generate goldens and manifest.${NC}"
        # Missing manifest is artifact integrity — hard in hard-fail and artifacts-hard.
        [ "$mode" = "warn-only" ] && return 0
        return 1
    fi

    local STALE_COUNT=0
    local MISSING_COUNT=0
    local GEN_STALE_COUNT=0
    local ORPHAN_COUNT=0
    local SECTION="catalog"
    declare -A REGISTERED_MODELS  # for reverse-check

    while IFS= read -r line; do
        if [[ "$line" =~ ^#.*FIXTURES ]]; then
            SECTION="fixtures"
            continue
        fi
        if [[ "$line" =~ ^#.*GENERATORS ]]; then
            SECTION="generators"
            continue
        fi

        if [[ "$line" =~ ^#\ EXPECTED:\ (.+)$ ]]; then
            local expected_file="${BASH_REMATCH[1]}"
            if [ ! -f "$FIXTURES_DIR/$expected_file" ]; then
                echo -e "  ${STALE_PREFIX} MISSING: $expected_file — expected golden file not found on disk${NC}"
                MISSING_COUNT=$((MISSING_COUNT + 1))
            fi
            continue
        fi

        if [[ "$line" =~ ^#\ MODEL-REGISTRY:\ (.+)$ ]]; then
            REGISTERED_MODELS["${BASH_REMATCH[1]}"]=1
            continue
        fi

        [[ "$line" =~ ^# ]] && continue
        [[ -z "$line" ]] && continue

        if [ "$SECTION" = "fixtures" ] || [ "$SECTION" = "generators" ]; then
            read -r expected_sha file_path <<< "$line"
            if [ "$SECTION" = "fixtures" ]; then
                local fixture_path="$FIXTURES_DIR/$file_path"
                if [ -f "$fixture_path" ]; then
                    local CURRENT_FIXTURE_SHA
                    CURRENT_FIXTURE_SHA=$(sha256sum "$fixture_path" | cut -d' ' -f1)
                    if [ "$CURRENT_FIXTURE_SHA" != "$expected_sha" ]; then
                        echo -e "  ${STALE_PREFIX} STALE: $file_path — fixture hash changed${NC}"
                        STALE_COUNT=$((STALE_COUNT + 1))
                    fi
                else
                    echo -e "  ${STALE_PREFIX} MISSING: $file_path — fixture file not found on disk${NC}"
                    MISSING_COUNT=$((MISSING_COUNT + 1))
                fi
            else
                local gen_path="$file_path"
                if [ -f "$gen_path" ]; then
                    local CURRENT_GEN_SHA
                    CURRENT_GEN_SHA=$(sha256sum "$gen_path" | cut -d' ' -f1)
                    if [ "$CURRENT_GEN_SHA" != "$expected_sha" ]; then
                        echo -e "  ${YELLOW}⚠ GENERATOR CHANGED: $gen_path — fixtures may be stale; re-run golden_gen_build.sh${NC}"
                        GEN_STALE_COUNT=$((GEN_STALE_COUNT + 1))
                    fi
                fi
            fi
            continue
        fi

        # ── Catalog entries (4 fields: model_sha golden_sha model_name golden_name) ──
        read -r expected_model_sha expected_golden_sha nam_file golden_file <<< "$line"
        REGISTERED_MODELS["$nam_file"]=1
        local MODEL_PATH="$MODELS_DIR/$nam_file"
        if [ -f "$MODEL_PATH" ]; then
            local CURRENT_MODEL_SHA
            CURRENT_MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
            if [ "$CURRENT_MODEL_SHA" != "$expected_model_sha" ]; then
                echo -e "  ${STALE_PREFIX} STALE: $nam_file — model modified since golden was generated${NC}"
                STALE_COUNT=$((STALE_COUNT + 1))
            fi
        fi
    done < "$MANIFEST"

    # ── Reverse-check: scan models/ for .nam files not in manifest ──
    for nam_path in "$MODELS_DIR"/*.nam; do
        [ -f "$nam_path" ] || continue
        local nam_name
        nam_name=$(basename "$nam_path")
        if [ -z "${REGISTERED_MODELS[$nam_name]:-}" ]; then
            echo -e "  ${STALE_PREFIX} ORPHAN: $nam_name — model file not registered in freshness manifest${NC}"
            ORPHAN_COUNT=$((ORPHAN_COUNT + 1))
        fi
    done

    local ARTIFACT_INTEGRITY_OK=1
    local GENERATOR_PROVENANCE_OK=1
    local TOOLCHAIN_PROVENANCE_OK=1
    local HAD_FAILURE=0

    if [ "$MISSING_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $MISSING_COUNT expected file(s) missing.${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to generate missing golden vectors.${NC}"
        ARTIFACT_INTEGRITY_OK=0
        HAD_FAILURE=1
    fi
    if [ "$STALE_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $STALE_COUNT file(s) stale (artifact integrity).${NC}"
        echo -e "  ${PREFIX} Run './tests/fixtures/golden_gen_build.sh' to regenerate fixtures and manifest.${NC}"
        ARTIFACT_INTEGRITY_OK=0
        HAD_FAILURE=1
    fi
    if [ "$ORPHAN_COUNT" -gt 0 ]; then
        echo -e "  ${PREFIX} $ORPHAN_COUNT model(s) not registered in manifest.${NC}"
        echo -e "  ${PREFIX} Add them to the CATALOG in golden_gen_build.sh and regenerate.${NC}"
        ARTIFACT_INTEGRITY_OK=0
        HAD_FAILURE=1
    fi

    if [ "$GEN_STALE_COUNT" -gt 0 ]; then
        GENERATOR_PROVENANCE_OK=0
        if [ "$GEN_HARD" -eq 1 ]; then
            echo -e "  ${PREFIX} $GEN_STALE_COUNT generator(s) changed — fixture provenance stale.${NC}"
            echo -e "  ${PREFIX} Re-run './tests/fixtures/golden_gen_build.sh' to regenerate goldens from current generators.${NC}"
            echo -e "  ${PREFIX} Golden files may still be *internally consistent* but their provenance no longer matches the generator.${NC}"
            HAD_FAILURE=1
        else
            echo -e "  ${YELLOW}⚠ $GEN_STALE_COUNT generator(s) changed — provenance drift (non-blocking in this mode).${NC}"
            echo -e "  ${YELLOW}  Goldens may still be internally consistent. Re-run './tests/fixtures/golden_gen_build.sh' before long/pre-release.${NC}"
        fi
    fi

    check_toolchain_fingerprint
    local tf_mismatch=$?
    if [ "$tf_mismatch" -eq 1 ]; then
        TOOLCHAIN_PROVENANCE_OK=0
    fi

    if [ "$HAD_FAILURE" -eq 1 ]; then
        [ "$mode" = "warn-only" ] && return 0
        return 1
    fi

    local summary="Freshness gate passed"
    local details=""
    if [ "$ARTIFACT_INTEGRITY_OK" -eq 1 ]; then
        details="${details}artifact_integrity=OK "
    fi
    if [ "$GENERATOR_PROVENANCE_OK" -eq 1 ]; then
        details="${details}generator_provenance=OK "
    else
        details="${details}generator_provenance=DRIFT "
    fi
    if [ "$TOOLCHAIN_PROVENANCE_OK" -eq 1 ]; then
        details="${details}toolchain_provenance=OK"
    else
        details="${details}toolchain_provenance=DRIFT"
    fi
    echo -e "  ${GREEN}✓ ${summary} (${details}).${NC}"
    return 0
}

