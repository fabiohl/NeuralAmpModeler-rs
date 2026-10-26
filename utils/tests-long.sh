#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# tests-long.sh — long-duration audit, stress, and verification suite. Classe B.
# Uso: utils/tests-long.sh [--strict] [--only <fase>[,<fase>]]
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -o pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

STRICT=0
ONLY_PHASES=()
SCRIPT_NAME="long"
LOG_DIR="target/logs"
RECEIPT_OUT="$LOG_DIR/long-audit-receipt.jsonl"
CARGO_TARGET_DIR="target/long"
export CARGO_TARGET_DIR

while [ $# -gt 0 ]; do
  case "$1" in
    --strict)
      STRICT=1
      export NAM_REQUIRE_CPP_ORACLE=1
      export NAM_RT_STRICT=1
      shift
      ;;
    --only)
      IFS=',' read -r -a ONLY_PHASES <<< "${2:?--only requires a list}"
      shift 2
      ;;
    --only=*)
      IFS=',' read -r -a ONLY_PHASES <<< "${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Uso: utils/tests-long.sh [--strict] [--only <fase>[,<fase>]]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

rm -rf "$CARGO_TARGET_DIR"
mkdir -p "$LOG_DIR"
rm -f "$RECEIPT_OUT"

ensure_long_receipt_bin || {
  echo "ERROR: nam_long_receipt binary could not be built" >&2
  exit 1
}
PERF_GATE_BIN=$(ensure_qa_bin nam_perf_gate testing) || {
  echo "ERROR: nam_perf_gate binary could not be built" >&2
  exit 1
}

is_phase_selected() {
  local pid="$1" pname="$2"
  [ ${#ONLY_PHASES[@]} -eq 0 ] && return 0
  for o in "${ONLY_PHASES[@]}"; do
    o=$(echo "$o" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    local p_id=$(echo "$pid" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    local p_name=$(echo "$pname" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    if [ "$o" = "$p_id" ] || [ "$o" = "$p_name" ]; then
      return 0
    fi
  done
  return 1
}

record_receipt() {
  local pid="$1" pname="$2" status="$3" dur_ms="$4" log_file="${5:-}"
  "$LONG_RECEIPT_BIN" append \
    --phase-id "$pid" \
    --name "$pname" \
    --status "$status" \
    --duration-ms "$dur_ms" \
    ${log_file:+--log "$log_file"} \
    --out "$RECEIPT_OUT" || {
      FAILS+=("receipt-append-$pid")
      echo "WARN: receipt append failed for $pid" >&2
    }
}

exec_preflight() {
  local pid="$1" pname="$2" to_s="$3"; shift 3
  local log="$LOG_DIR/long-$pid.log"
  if ! is_phase_selected "$pid" "$pname"; then
    echo "PHASE: $pid SKIPPED"
    record_receipt "$pid" "$pname" "SKIPPED" 0
    return 0
  fi
  phase "$pid"
  local t0=$(date +%s%N)
  local rc=0
  local to_val=${NAM_PHASE_TIMEOUT_S:-$to_s}
  if [ "$(type -t "$1")" = "function" ]; then
    "$@" > "$log" 2>&1 || rc=$?
  else
    timeout --kill-after=30 "$to_val" "$@" > "$log" 2>&1 || rc=$?
  fi
  local t1=$(date +%s%N)
  local dur_ms=$(( (t1 - t0) / 1000000 ))
  [ "$dur_ms" -lt 0 ] && dur_ms=0
  local status="PASSED"
  if [ "$rc" -eq 3 ] && [ "$STRICT" -eq 0 ]; then
    status="SKIPPED"
    GAPS+=("$pid:optional-dependency-missing")
    echo "PHASE: $pid GAP optional-dependency-missing"
  elif [ "$rc" -ne 0 ]; then
    status="FAILED"
    FAILS+=("$pid")
    echo "PHASE: $pid FAIL rc=$rc log=$log"
  else
    echo "PHASE: $pid PASS log=$log"
  fi
  record_receipt "$pid" "$pname" "$status" "$dur_ms" "$log"
}

exec_timed_phase() {
  local pid="$1" pname="$2" to_s="$3" req_tests="$4"; shift 4
  local log="$LOG_DIR/long-$pid.log"
  if ! is_phase_selected "$pid" "$pname"; then
    echo "PHASE: $pid SKIPPED"
    record_receipt "$pid" "$pname" "SKIPPED" 0
    return 0
  fi
  phase "$pid"
  local t0=$(date +%s%N)
  local rc=0
  local to_val=${NAM_PHASE_TIMEOUT_S:-$to_s}
  if [ "$(type -t "$1")" = "function" ]; then
    "$@" > "$log" 2>&1 || rc=$?
  else
    timeout --kill-after=30 "$to_val" "$@" > "$log" 2>&1 || rc=$?
  fi
  local t1=$(date +%s%N)
  local dur_ms=$(( (t1 - t0) / 1000000 ))
  [ "$dur_ms" -lt 0 ] && dur_ms=0
  local status="PASSED"
  if [ "$rc" -ne 0 ] || { [ "$req_tests" -eq 1 ] && ! ran_tests "$log"; }; then
    status="FAILED"
    FAILS+=("$pid")
    echo "PHASE: $pid FAIL rc=$rc log=$log"
  else
    echo "PHASE: $pid PASS log=$log"
  fi
  record_receipt "$pid" "$pname" "$status" "$dur_ms" "$log"
}

# --- Preflight 1: render ---
run_preflight_render() {
  local render_bin
  render_bin=$(ensure_namcore_render 2>&1) || {
    local rc=$?
    if [ "$rc" -eq 3 ]; then
      echo "NAMCore mirror missing in third-party"
      return 3
    fi
    return "$rc"
  }
  echo "C++ NAMCore render binary ready at $render_bin"
  return 0
}
exec_preflight "preflight-render" "preflight-render" 300 run_preflight_render

# --- Preflight 2: catalog ---
exec_preflight "preflight-catalog" "preflight-catalog" 120 \
  cargo test --release --features testing --test models catalog_preflight -- --nocapture

# --- Preflight 3: package ---
run_preflight_package() {
  cargo package --list --allow-dirty | grep -Ev '(target/|\.git/)'
}
exec_preflight "preflight-package" "preflight-package" 60 run_preflight_package

# --- Preflight 4: freshness ---
exec_preflight "preflight-freshness" "preflight-freshness" 60 check_freshness hard-fail

# --- Preflight 5: meta ---
exec_preflight "preflight-meta" "preflight-meta" 60 \
  cargo test --release --features testing --test models test_ignored_models_are_in_catalog -- --nocapture

# --- Preflight 6: simd-probe ---
exec_preflight "preflight-simd-probe" "preflight-simd-probe" 60 cargo run --quiet --bin simd_probe

# --- Phase 1: catch-all ---
run_phase_catch_all() {
  cargo test --release --features testing --no-fail-fast -- --include-ignored --skip rt_deadline:: --skip rt_jitter:: --skip soak_test::
}
exec_timed_phase "phase1" "catch-all" 600 1 run_phase_catch_all

# --- Phase 2: soak ---
run_phase_soak() {
  cargo test --release --features testing --test perf_soak soak_test:: -- --ignored --test-threads=1
}
exec_timed_phase "phase2" "soak" 600 1 run_phase_soak

# --- Phase 3: isa-codegen ---
run_phase_isa_codegen() {
  cargo run --quiet --bin simd_probe || return 1
  cargo test --release --features testing --test isa_contract -- --nocapture || return 1
  if command -v objdump >/dev/null 2>&1 && [ -f "utils/verify-kernel-codegen.sh" ]; then
    bash utils/verify-kernel-codegen.sh 2>&1 || true
  fi
  return 0
}
exec_timed_phase "phase3" "isa-codegen" 300 1 run_phase_isa_codegen

# --- Phase 4: heap-audit ---
run_phase_heap_audit() {
  cargo test --release --features testing,heap-audit -- --ignored
}
exec_timed_phase "phase4" "heap-audit" 300 1 run_phase_heap_audit

# --- Phase 5: rt-deadline ---
run_phase_rt_deadline() {
  local pick_args=()
  [ "$STRICT" -eq 1 ] && pick_args+=("--strict")
  local core
  core=$("$PERF_GATE_BIN" pick-core "${pick_args[@]}") || return $?
  if [ "$core" != "GAP" ]; then
    export NAM_BENCH_CORE="$core"
  fi
  cargo test --release --features testing --test rt_constraints rt_deadline:: --nocapture
}
exec_timed_phase "phase5" "rt-deadline" 300 1 run_phase_rt_deadline

# --- Phase 6: rt-jitter ---
run_phase_rt_jitter() {
  cargo test --release --features testing --test rt_constraints rt_jitter:: --nocapture
}
exec_timed_phase "phase6" "rt-jitter" 300 1 run_phase_rt_jitter

# --- Phase 7: loom ---
run_phase_loom() {
  RUSTFLAGS="${RUSTFLAGS:-} --cfg loom" cargo test --release --test loom_tests -- --nocapture
}
exec_timed_phase "phase7" "loom" 600 1 run_phase_loom

# --- Phase 8: perf ---
run_phase_perf() {
  local pick_args=()
  [ "$STRICT" -eq 1 ] && pick_args+=("--strict")
  local core
  core=$("$PERF_GATE_BIN" pick-core "${pick_args[@]}") || return $?
  local perf_args=(run --check docs/quality-contract.json)
  [ "$core" != "GAP" ] && perf_args+=(--bench-core "$core")
  [ "$STRICT" -eq 1 ] && perf_args+=(--strict)
  "$PERF_GATE_BIN" "${perf_args[@]}"
}
exec_timed_phase "phase8" "perf" 600 0 run_phase_perf

ONLY_ARGS=()
if [ ${#ONLY_PHASES[@]} -gt 0 ]; then
  joined_only=$(IFS=,; echo "${ONLY_PHASES[*]}")
  ONLY_ARGS=(--only "$joined_only")
fi

echo "Emitting suite summary to receipt..."
"$LONG_RECEIPT_BIN" summary "${ONLY_ARGS[@]}" --out "$RECEIPT_OUT" || {
  FAILS+=("receipt-summary")
  echo "PHASE: receipt-summary FAIL"
}

VALIDATE_ARGS=()
[ "$STRICT" -eq 1 ] && VALIDATE_ARGS+=("--strict")
"$LONG_RECEIPT_BIN" validate "${VALIDATE_ARGS[@]}" "${ONLY_ARGS[@]}" --out "$RECEIPT_OUT" || {
  FAILS+=("receipt-validation")
  echo "PHASE: receipt-validation FAIL"
}

finish
