#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# =============================================================================
# Remote SIMD Gate & Receipt Generator (AVX-512 / AVX10) — T4.1.1
# =============================================================================
#
# Automated harness for executing cross-ISA parity validation and latency
# benchmarks on remote machines or VMs equipped with AVX-512 hardware (AMD Zen 4/5,
# Intel Xeon / Sapphire Rapids).
#
# Sequential Execution Phases:
#   [0/4] Phase 0: Hardware Preflight (check avx512f and avx512vl support)
#   [1/4] Phase 1: Cross-Mathematical Parity (cargo test --test isa_parity)
#   [2/4] Phase 2: ISA Comparison Benchmarking (cargo bench -- isa_compare)
#   [3/4] Phase 3: Structured JSON Receipt & Table Generation (target/remote-simd-receipt.json)
#
# Exit Codes:
#   0: Success (all parity tests passed, all primary SKUs achieve >= 12% speedup with p < 0.05).
#   1: Failure (parity violation, performance regression, or gate failure).
#   2: Clean skip (AVX-512 features not detected on host CPU during Preflight).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PHASE_TOTAL=4
source "$SCRIPT_DIR/_lib.sh"

mkdir -p target/logs

trap 'echo -e "\n${RED}${BOLD}FAIL: unexpected error: \"$BASH_COMMAND\" at line $LINENO status $?.${NC}"; exit 1' ERR

# ── Structured receipt emitter (T2.4: cross-ISA matrix registered in a
# dedicated receipt and validated end-to-end, fail-closed) ───────────────────
# The remote gate appends machine-readable lines (phase_id, name, status,
# duration_ms, tests_executed, gaps, timestamp) to target/logs/
# remote-simd-receipt.jsonl via the same `nam_long_receipt` emitter used by
# the long suite (src/testing/receipt.rs) — never hand-serialized JSON.
LONG_RECEIPT_BIN="${NAM_LONG_RECEIPT_BIN:-$PROJECT_DIR/target/debug/nam_long_receipt}"

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

emit_remote_receipt() {
    local phase_id="$1" name="$2" status="$3" duration_ms="$4" log_file="$5"
    if ! ensure_long_receipt_bin; then
        echo -e "  ${YELLOW}${BOLD}⚠ remote-simd receipt emission unavailable (nam_long_receipt missing)${NC}" >&2
        return 1
    fi
    local rc=0
    "$LONG_RECEIPT_BIN" append \
        --phase-id "$phase_id" \
        --name "$name" \
        --status "$status" \
        --duration-ms "$duration_ms" \
        --log "$log_file" \
        --out "$REMOTE_RECEIPT_JSONL" || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo -e "  ${YELLOW}${BOLD}⚠ remote-simd receipt emission failed for $phase_id (rc=$rc)${NC}" >&2
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Default Configuration & CLI Options
# ---------------------------------------------------------------------------
COOLDOWN_SECS="${NAM_COOLDOWN_SECS:-180}"
RECEIPT_OUT="target/logs/remote-simd-receipt.json"
REMOTE_RECEIPT_JSONL="target/logs/remote-simd-receipt.jsonl"
PARITY_LOG="target/logs/remote-simd-parity.log"
CHECK_ONLY=0
SKIP_PARITY=0
SKIP_BENCH=0
USE_SDE=0

print_usage() {
    echo "Usage: utils/remote-simd-gate.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --sde                Run in Intel SDE emulation mode (auto-configures runner, skips bench/cooldown)"
    echo "  --check-only         Run Phase 0 (Preflight) only and exit"
    echo "  --skip-cooldown      Skip 180s thermal cooldown intervals"
    echo "  --cooldown <SECS>    Specify custom thermal cooldown in seconds (default: 180)"
    echo "  --skip-parity        Skip Phase 1 (mathematical parity test)"
    echo "  --skip-bench         Skip Phase 2 (Criterion ISA comparison bench)"
    echo "  --out <FILE>         Destination path for receipt JSON (default: target/logs/remote-simd-receipt.json)"
    echo "  --help, -h           Show this usage summary"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --sde)
            USE_SDE=1
            COOLDOWN_SECS=0
            SKIP_BENCH=1
            export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-sde64 -spr --}"
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --skip-cooldown)
            COOLDOWN_SECS=0
            shift
            ;;
        --cooldown)
            if [ -n "${2:-}" ]; then
                COOLDOWN_SECS="$2"
                shift 2
            else
                echo -e "${RED}Error: --cooldown requires a numeric argument in seconds.${NC}" >&2
                exit 1
            fi
            ;;
        --skip-parity)
            SKIP_PARITY=1
            shift
            ;;
        --skip-bench)
            SKIP_BENCH=1
            shift
            ;;
        --out)
            if [ -n "${2:-}" ]; then
                RECEIPT_OUT="$2"
                shift 2
            else
                echo -e "${RED}Error: --out requires a file path.${NC}" >&2
                exit 1
            fi
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            print_usage
            exit 1
            ;;
    esac
done

echo -e "${BLUE}${BOLD}================================================================================"
echo -e "          NeuralAmpModeler-rs Remote SIMD Gating Suite (AVX-512)"
echo -e "================================================================================${NC}"

# ---------------------------------------------------------------------------
# [0/4] Phase 0: Hardware Preflight
# ---------------------------------------------------------------------------
phase "Phase 0: Hardware Preflight (AVX-512 Detection)..."

CPU_MODEL="unknown"
HAS_AVX512=0

# 1. Check if running under Intel SDE runner or explicit SDE mode
if [ "$USE_SDE" -eq 1 ] || [[ "${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-}" =~ sde ]]; then
    HAS_AVX512=1
    CPU_MODEL="Intel SDE Emulation (${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER:-sde64 -spr --})"
    if [ "$USE_SDE" -eq 0 ]; then
        COOLDOWN_SECS=0
        SKIP_BENCH=1
    fi
elif [ -f /proc/cpuinfo ]; then
    CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^[ \t]*//' || echo "unknown")
    # T2.1/T2.4: the reachable AVX-512 kernels require the full capability
    # matrix — F + VL + BW + DQ (a partial subset can SIGILL).
    if grep -qw "avx512f" /proc/cpuinfo && grep -qw "avx512vl" /proc/cpuinfo \
        && grep -qw "avx512bw" /proc/cpuinfo && grep -qw "avx512dq" /proc/cpuinfo; then
        HAS_AVX512=1
    fi
elif command -v lscpu >/dev/null 2>&1; then
    CPU_MODEL=$(lscpu | grep -m1 "Model name:" | cut -d: -f2 | sed 's/^[ \t]*//' || echo "unknown")
    CPU_FLAGS=$(lscpu | grep -iE '^(Flags|Opções):' || true)
    if echo "$CPU_FLAGS" | grep -qw "avx512f" && echo "$CPU_FLAGS" | grep -qw "avx512vl" \
        && echo "$CPU_FLAGS" | grep -qw "avx512bw" && echo "$CPU_FLAGS" | grep -qw "avx512dq"; then
        HAS_AVX512=1
    fi
fi

if [ "$HAS_AVX512" -ne 1 ]; then
    warn "Host CPU: $CPU_MODEL"
    warn "AVX-512 features (avx512f + avx512vl + avx512bw + avx512dq) not detected on this machine."
    warn "Clean skip: This gate is designed for execution on AVX-512 hardware or under Intel SDE (--sde)."
    exit 2
fi

ok "AVX-512 environment verified: $CPU_MODEL (avx512f + avx512vl + avx512bw + avx512dq present)."

if [ "$CHECK_ONLY" -eq 1 ]; then
    ok "Preflight check completed successfully (--check-only requested)."
    exit 0
fi

# ---------------------------------------------------------------------------
# [1/4] Phase 1: Cross-Mathematical Parity
# ---------------------------------------------------------------------------
phase "Phase 1: Cross-Mathematical Parity Validation..."

if [ "$SKIP_PARITY" -eq 1 ]; then
    warn "Skipping Phase 1 (--skip-parity specified)."
else
    if [ "$COOLDOWN_SECS" -gt 0 ]; then
        echo -e "  ${YELLOW}Hardware thermal stabilization cooling period (${COOLDOWN_SECS}s)...${NC}"
        sleep "$COOLDOWN_SECS"
    fi

    echo -e "  ${BLUE}Running: cargo test --release --features avx512 --test parity -- isa_parity --include-ignored --test-threads=1 --nocapture${NC}"
    echo "AVX512_OPT_IN: RUN (remote-simd-gate compiles with --features avx512)" > "$PARITY_LOG"
    parity_start=$(date +%s%N)
    parity_rc=0
    cargo test --release --features avx512 --test parity -- isa_parity --include-ignored --test-threads=1 --nocapture 2>&1 | tee -a "$PARITY_LOG" || parity_rc=$?
    parity_dur=$(( ($(date +%s%N) - parity_start) / 1000000 ))

    # T2.4 mandatory-subphase gate: the cross-ISA matrix must prove real
    # execution (≥1 test) — never a silent zero-case PASS from a filter/compile
    # mismatch. Under the AVX-512 opt-in build the matrix always has cases.
    assert_subphase_ran "remote_simd_isa_parity" "$PARITY_LOG" 1 || parity_rc=1

    if [ "$parity_rc" -ne 0 ]; then
        emit_remote_receipt "remote-simd-parity" "Remote SIMD Cross-ISA Parity Matrix" "FAILED" "$parity_dur" "$PARITY_LOG" || true
        echo -e "${RED}${BOLD}❌ Mathematical cross-ISA parity FAILED (rc=$parity_rc).${NC}"
        exit 1
    fi
    emit_remote_receipt "remote-simd-parity" "Remote SIMD Cross-ISA Parity Matrix" "PASSED" "$parity_dur" "$PARITY_LOG" || true
    ok "Mathematical cross-ISA parity verified against AVX2 reference."
fi

# ---------------------------------------------------------------------------
# [2/4] Phase 2: ISA Comparison Benchmarking
# ---------------------------------------------------------------------------
phase "Phase 2: ISA Comparison Benchmarking (Criterion)..."

if [ "$SKIP_BENCH" -eq 1 ]; then
    warn "Skipping Phase 2 (--skip-bench specified)."
else
    if [ "$COOLDOWN_SECS" -gt 0 ]; then
        echo -e "  ${YELLOW}Hardware thermal stabilization cooling period (${COOLDOWN_SECS}s)...${NC}"
        sleep "$COOLDOWN_SECS"
    fi

    echo -e "  ${BLUE}Running: cargo bench --features avx512 --bench inference_bench -- ISA_Compare${NC}"
    cargo bench --features avx512 --bench inference_bench -- ISA_Compare
    ok "Criterion ISA comparison benchmarks completed."
fi

# ---------------------------------------------------------------------------
# [3/4] Phase 3: Structured JSON Receipt & Table Generation
# ---------------------------------------------------------------------------
phase "Phase 3: Structured Receipt Generation..."

if [ "$SKIP_BENCH" -eq 1 ]; then
    warn "Skipping Phase 3 receipt generation & ROI check (--skip-bench specified)."
else
    ROI_START=$(date +%s%N)
    RECEIPT_STATUS=0
    cargo run --quiet --features testing --bin nam_remote_simd_receipt -- \
        --criterion-dir "target/criterion" \
        --out "$RECEIPT_OUT" \
        --table \
        --check || RECEIPT_STATUS=$?
    ROI_DUR=$(( ($(date +%s%N) - ROI_START) / 1000000 ))
    ROI_LOG="target/logs/remote-simd-roi.log"

    if [ "$RECEIPT_STATUS" -ne 0 ]; then
        echo "ROI criteria not met (exit $RECEIPT_STATUS) — see $RECEIPT_OUT" > "$ROI_LOG"
        emit_remote_receipt "remote-simd-roi" "Remote SIMD ROI Receipt Check" "FAILED" "$ROI_DUR" "$ROI_LOG" || true
        warn "Remote SIMD audit receipt saved to: $RECEIPT_OUT"
        echo -e "\n${YELLOW}${BOLD}================================================================================"
        echo -e "  Remote SIMD Gate: Parity verified (100% PASS), but ROI criteria not met."
        echo -e "  Verdict: Gate Check FAILED (ROI policy directs fallback to AVX2)."
        echo -e "  Artifact saved: ${CYAN}$RECEIPT_OUT${NC}"
        echo -e "${YELLOW}${BOLD}================================================================================${NC}\n"
        exit 1
    fi

    emit_remote_receipt "remote-simd-roi" "Remote SIMD ROI Receipt Check" "PASSED" "$ROI_DUR" "$ROI_LOG" || true
    ok "Remote SIMD audit receipt saved to: $RECEIPT_OUT"
fi

# ── End-to-end receipt validation (T2.4) ────────────────────────────────────
# The dedicated remote-simd JSONL receipt (parity matrix + ROI check, when
# run) is validated fail-closed: every line must match the LongPhaseReceipt
# schema, and the derived overall verdict is printed by `summary`. A corrupt
# or missing receipt fails the gate.
if ensure_long_receipt_bin; then
    if ! "$LONG_RECEIPT_BIN" validate --out "$REMOTE_RECEIPT_JSONL"; then
        echo -e "${RED}${BOLD}❌ Remote SIMD receipt validation FAILED (target/logs/remote-simd-receipt.jsonl).${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}✓ Remote SIMD receipt validated (target/logs/remote-simd-receipt.jsonl)${NC}"
    SUMMARY_TEXT="$("$LONG_RECEIPT_BIN" summary --out "$REMOTE_RECEIPT_JSONL")" || true
    printf '%s\n' "$SUMMARY_TEXT" | sed 's/^/    /'
else
    echo -e "${RED}${BOLD}❌ nam_long_receipt unavailable — remote SIMD receipt NOT emitted (fail-closed).${NC}"
    exit 1
fi

echo -e "\n${GREEN}${BOLD}================================================================================${NC}"
echo -e "${GREEN}${BOLD}✓ Remote SIMD Gate passed all mathematical and parity criteria!${NC}"
echo -e "  ${BOLD}AVX512_OPT_IN: RUN (remote-simd-gate compiles with --features avx512)${NC}"
if [ "$SKIP_BENCH" -eq 0 ]; then
    echo -e "  ${BOLD}Artifact saved:${NC} ${CYAN}$RECEIPT_OUT${NC}"
fi
echo -e "${GREEN}${BOLD}================================================================================${NC}\n"
