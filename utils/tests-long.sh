#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# tests-long.sh — long-duration audit, stress, and verification suite. Classe B.
# Uso: utils/tests-long.sh [--strict] [--only <fase>[,<fase>]]
# Orçamento: 296
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -o pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

STRICT=0
ONLY_PHASES=()
SCRIPT_NAME="long"
LOG_DIR="target/logs"
RECEIPT_OUT="$LOG_DIR/long-audit-receipt.jsonl"
LONG_ROOT="target/long"
export CARGO_TARGET_DIR="$LONG_ROOT/release"

usage() {
  cat <<'EOF'
Uso: utils/tests-long.sh [--strict] [--only <fase>[,<fase>]]
  --strict         GAP converte-se em falha
  --only <fases>   roda só as fases nomeadas (preserva target/long)
  preflight-render      render C++ de referência (GAP sem --strict)
  preflight-catalog     catálogo de modelos
  preflight-package     cargo package --locked (verify) + consumer-smoke (x86/aarch64 sem .cargo do repo)
  preflight-freshness   frescor de artefatos
  preflight-simd-probe  sonda de ISA
  catch-all             --include-ignored; pinados ficam em alvos com required-features
  soak                  soak_pinned (1 thread)
  isa-codegen           simd_probe e isa_contract
  heap-audit            heap audit (target/long/heap-audit)
  rt-deadline           prazos RT no núcleo escolhido por pick-core
  rt-jitter             jitter RT
  loom                  testes loom (target/long/loom)
  perf                  nam_perf_gate run --check
EOF
}

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
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [ ${#ONLY_PHASES[@]} -eq 0 ]; then rm -rf "$LONG_ROOT"; fi
mkdir -p "$LOG_DIR"
rm -f "$RECEIPT_OUT" "$LOG_DIR/skips.jsonl"; xtrace_side_channel "$LOG_DIR/long-xtrace.log"

ensure_long_receipt_bin || {
  echo "ERROR: nam_long_receipt binary could not be built" >&2
  exit 1
}
PERF_GATE_BIN=$(ensure_qa_bin nam_perf_gate testing) || {
  echo "ERROR: nam_perf_gate binary could not be built" >&2
  exit 1
}

is_phase_selected() {
  local pid="$1" pname="$2" o
  [ ${#ONLY_PHASES[@]} -eq 0 ] && return 0
  for o in "${ONLY_PHASES[@]}"; do
    { [ "$o" = "$pid" ] || [ "$o" = "$pname" ]; } && return 0
  done
  return 1
}

record_receipt() {
  local pid="$1" pname="$2" status="$3" dur_ms="$4" log_file="${5:-}" reason="${6:-}"
  "$LONG_RECEIPT_BIN" append \
    --phase-id "$pid" \
    --name "$pname" \
    --status "$status" \
    --duration-ms "$dur_ms" \
    ${log_file:+--log "$log_file"} \
    ${reason:+--reason "$reason"} \
    --out "$RECEIPT_OUT" || {
      FAILS+=("receipt-append-$pid")
      echo "WARN: receipt append failed for $pid" >&2
    }
}

run_cmd() { local rc=0; timeout --kill-after=30 "$to_val" "$@" || rc=$?; case "$rc" in 124|137) timed_out=1 ;; esac; return "$rc"; }

# Extracts only the libtest diagnostic headers so the failure cause survives
# even when the last 120 log lines are all `test ... ok`. The full log stays
# untouched and published by the artifact.
extract_failures() {
  local log="$1" out="$2"
  awk '
    /^---- .* stdout ----$/ || /panicked at/ || /^failures:/ || /^error: .* target/ || /test result: FAILED/ {
      print
      want_message = /panicked at/
      next
    }
    want_message { print; want_message = 0 }
  ' "$log" > "$out" 2>/dev/null || true
}

exec_phase() {
  local pid="$1" pname="$2" to_s="$3" req_tests="$4"; shift 4
  local log="$LOG_DIR/long-$pname.log"
  if ! is_phase_selected "$pid" "$pname"; then
    emit "PHASE: $pname SKIPPED"
    record_receipt "$pid" "$pname" "SKIPPED" 0
    return 0
  fi
  phase "$pname"
  export CARGO_TARGET_DIR="$LONG_ROOT/release"
  local t0=$(date +%s%N)
  local rc=0 note="" timed_out=0 reason=""
  local to_val=${NAM_PHASE_TIMEOUT_S:-$to_s}
  PHASE_GAP=""
  "$@" > "$log" 2>&1 || rc=$?
  [ "$timed_out" -eq 1 ] && rc=124
  local t1=$(date +%s%N)
  local dur_ms=$(( (t1 - t0) / 1000000 ))
  [ "$dur_ms" -lt 0 ] && dur_ms=0
  local status="PASSED"
  if [ "$rc" -eq 3 ] && [ "$pid" = "phase8" ]; then
    status="SKIPPED"; GAPS+=("$pname:perf-baseline-missing"); emit "PHASE: $pname GAP perf-baseline-missing"
  elif [ "$rc" -eq 3 ] && [ "$pid" = "preflight-render" ] && [ "$STRICT" -eq 0 ]; then
    status="SKIPPED"
    GAPS+=("$pname:optional-dependency-missing")
    emit "PHASE: $pname GAP optional-dependency-missing"
  elif [ "$rc" -ne 0 ] || { [ "$req_tests" -eq 1 ] && ! ran_tests "$log"; }; then
    status="FAILED"; FAILS+=("$pname")
    case "$rc" in 124|137) note=" (timeout)"; reason="TIMEOUT" ;; esac
    [ -z "$reason" ] && reason="TEST_FAILURE"
    [ "$pid" = "phase8" ] && [ "$rc" -eq 4 ] && reason="MISSING_BASELINE"
    [ "$pid" = "preflight-freshness" ] && [ "$rc" -eq 5 ] && reason="GENERATOR_DRIFT"
    [ "${pid#preflight-}" != "$pid" ] && [ "$reason" = "TEST_FAILURE" ] && reason="PREFLIGHT_FAILURE"
    emit "PHASE: $pname FAIL rc=$rc$note log=$log"
    if [ -f "$log" ]; then
      local fail_out="$LOG_DIR/long-$pname-failures.txt"
      extract_failures "$log" "$fail_out"
      [ -s "$fail_out" ] && { echo "--- [FAILURES: $pname] ---" >&2; cat "$fail_out" >&2; echo "--- [END FAILURES: $pname] ---" >&2; }
      echo "--- [FAIL LOG: $pname (rc=$rc)] ---" >&2; tail -n 120 "$log" >&2 || true; echo "--- [END LOG: $pname] ---" >&2
    fi
  elif [ -n "$PHASE_GAP" ]; then
    status="SKIPPED"; [ "${PHASE_GAP%%:*}" = "skip-capability" ] && status="SKIP_CAPABILITY"
    GAPS+=("$pname:$PHASE_GAP")
    emit "PHASE: $pname GAP $PHASE_GAP"
  else
    emit "PHASE: $pname PASS log=$log"
  fi
  record_receipt "$pid" "$pname" "$status" "$dur_ms" "$log" "$reason"
}

# --- Preflight 1: render ---
run_preflight_render() {
  local render_bin rc=0 smoke
  render_bin=$(run_cmd bash utils/ensure_namcore_render.sh 2>/dev/null | tail -n 1) || rc=$?
  [ "$rc" -eq 3 ] && { echo "NAMCore mirror missing in third-party"; return 3; }
  [ "$rc" -ne 0 ] && return "$rc"; [ -x "$render_bin" ] || return 6
  smoke=$("$render_bin" 2>&1) || rc=$?
  [ -n "$smoke" ] && [ "$rc" -eq 1 ] || return 6
  echo "C++ NAMCore render binary ready at $render_bin commit=$(sed -n 's/^core_commit=//p' build/namcore_render/.build_provenance)"
}
exec_phase "preflight-render" "preflight-render" 300 0 run_preflight_render

# --- Preflight 2: catalog ---
run_preflight_catalog() { run_cmd cargo test --release --features testing --test models catalog_preflight -- --nocapture; }
exec_phase "preflight-catalog" "preflight-catalog" 120 1 run_preflight_catalog

# --- Preflight 3: package ---
run_preflight_package() { run_cmd cargo package --locked --allow-dirty && run_cmd bash utils/consumer-smoke.sh; }
exec_phase "preflight-package" "preflight-package" 600 0 run_preflight_package

# --- Preflight 4: freshness ---
run_preflight_freshness() { run_cmd cargo run --quiet --locked -p nam-qa --bin nam_freshness -- --root "$PWD" hard-fail; }
exec_phase "preflight-freshness" "preflight-freshness" 60 0 run_preflight_freshness

# --- Preflight 5: simd-probe ---
run_preflight_simd_probe() { run_cmd cargo run --quiet --bin simd_probe; }
exec_phase "preflight-simd-probe" "preflight-simd-probe" 60 0 run_preflight_simd_probe

# --- Phase 1: catch-all ---
run_phase_catch_all() {
  local a=() rc=0
  rm -f "$LOG_DIR/fidelity-metrics.jsonl"
  [ "$STRICT" -eq 0 ] && [[ " ${GAPS[*]} " == *" preflight-render:"* ]] && a+=(--skip cpp_parity::live_cross_validation)
  run_cmd cargo test --release --features testing --no-fail-fast -- --include-ignored "${a[@]}" || rc=$?
  if [ -s "$LOG_DIR/skips.jsonl" ]; then
    local total optional blocking
    total=$(wc -l < "$LOG_DIR/skips.jsonl")
    optional=$(grep -c '"optional":true' "$LOG_DIR/skips.jsonl" 2>/dev/null || true)
    optional=${optional:-0}
    blocking=$((total - optional))
    [ "$blocking" -gt 0 ] && PHASE_GAP="skip-capability:$blocking"
  fi
  return "$rc"
}
exec_phase "phase1" "catch-all" 600 1 run_phase_catch_all

# --- Phase 2: soak ---
run_phase_soak() {
  run_cmd cargo test --release --features testing,pinned --test soak_pinned -- --include-ignored --test-threads=1
}
exec_phase "phase2" "soak" 600 1 run_phase_soak

# --- Phase 3: isa-codegen ---
run_phase_isa_codegen() {
  run_cmd cargo run --quiet --bin simd_probe || return 1
  run_cmd cargo test --release --features testing --test isa_contract -- --nocapture || return 1
  return 0
}
exec_phase "phase3" "isa-codegen" 300 1 run_phase_isa_codegen

# --- Phase 4: heap-audit (alvos e filtro: ver docs/testing.md; perfil próprio) ---
run_phase_heap_audit() {
  export CARGO_TARGET_DIR="$LONG_ROOT/heap-audit"
  run_cmd cargo test --release --features testing,heap-audit --test rt_constraints --test cabsim_stereo \
    --test models -- --include-ignored heap_audit
}
exec_phase "phase4" "heap-audit" 300 1 run_phase_heap_audit

# --- Phase 5: rt-deadline ---
run_phase_rt_deadline() {
  local pick_args=(); [ "$STRICT" -eq 1 ] && pick_args+=("--strict")
  local core; core=$(run_cmd "$PERF_GATE_BIN" pick-core "${pick_args[@]}") || return $?
  [ "$core" = "GAP" ] && { PHASE_GAP=no-bench-core; [ "$STRICT" -eq 1 ] && return 1; return 0; }
  export NAM_BENCH_CORE="$core"
  run_cmd cargo test --release --features testing,pinned --test rt_pinned rt_deadline:: -- --include-ignored --nocapture
}
exec_phase "phase5" "rt-deadline" 300 1 run_phase_rt_deadline

# --- Phase 6: rt-jitter ---
run_phase_rt_jitter() {
  if [ -z "${NAM_BENCH_CORE:-}" ]; then
    local pick_args=(); [ "$STRICT" -eq 1 ] && pick_args+=("--strict")
    local core; core=$(run_cmd "$PERF_GATE_BIN" pick-core "${pick_args[@]}") || return $?
    if [ "$core" = "GAP" ]; then PHASE_GAP=no-bench-core; [ "$STRICT" -eq 1 ] && return 1; else export NAM_BENCH_CORE="$core"; fi
  fi
  run_cmd cargo test --release --features testing,pinned --test rt_pinned rt_jitter:: -- --include-ignored --nocapture
}
exec_phase "phase6" "rt-jitter" 300 1 run_phase_rt_jitter

# --- Phase 7: loom ---
run_phase_loom() {
  export CARGO_TARGET_DIR="$LONG_ROOT/loom"
  (unset RUSTFLAGS; run_cmd cargo test --release --test loom_tests --config "target.'cfg(target_arch = \"x86_64\")'.rustflags=['--cfg','loom']" -- --nocapture)
}
exec_phase "phase7" "loom" 600 1 run_phase_loom

# --- Phase 8: perf ---
run_phase_perf() {
  local pick_args=()
  [ "$STRICT" -eq 1 ] && pick_args+=("--strict")
  local core
  core=$(run_cmd "$PERF_GATE_BIN" pick-core "${pick_args[@]}") || return $?
  local perf_args=(run --check docs/quality-contract.json)
  [ "$core" != "GAP" ] && perf_args+=(--bench-core "$core")
  [ "$STRICT" -eq 1 ] && perf_args+=(--strict)
  run_cmd "$PERF_GATE_BIN" "${perf_args[@]}"
}
exec_phase "phase8" "perf" 600 0 run_phase_perf

ONLY_ARGS=()
if [ ${#ONLY_PHASES[@]} -gt 0 ]; then
  joined_only=$(IFS=,; echo "${ONLY_PHASES[*]}")
  ONLY_ARGS=(--only "$joined_only")
fi

echo "Emitting suite summary to receipt..."
"$LONG_RECEIPT_BIN" summary "${ONLY_ARGS[@]}" --out "$RECEIPT_OUT" || {
  FAILS+=("receipt-summary")
  emit "PHASE: receipt-summary FAIL"
}

VALIDATE_ARGS=()
[ "$STRICT" -eq 1 ] && VALIDATE_ARGS+=("--strict")
"$LONG_RECEIPT_BIN" validate "${VALIDATE_ARGS[@]}" "${ONLY_ARGS[@]}" --out "$RECEIPT_OUT" || {
  rc=$?
  if [ "$rc" -ne 4 ]; then FAILS+=("receipt-validation"); emit "PHASE: receipt-validation FAIL"; fi
}

finish
