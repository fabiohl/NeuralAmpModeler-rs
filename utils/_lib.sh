#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# _lib.sh — helpers locais de NeuralAmpModeler-rs. Sem cópia pinada, sem hash.
# Orçamento: 180

LOG_DIR=target/logs; [ -d "$LOG_DIR" ] || { command -v mkdir >/dev/null 2>&1 && mkdir -p "$LOG_DIR"; } || true
export NAM_RUN_ID="${NAM_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
[ "$(cat "$LOG_DIR/.summary-run" 2>/dev/null)" = "$NAM_RUN_ID" ] || { : > "$LOG_DIR/summary.txt"; printf '%s\n' "$NAM_RUN_ID" > "$LOG_DIR/.summary-run"; }
T0=$SECONDS; FAILS=(); GAPS=(); STRICT=0

LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$LIB_DIR")"; cd "$PROJECT_DIR"
THIRD_PARTY_DIR="$PROJECT_DIR/third-party"
CORE_DIR="$THIRD_PARTY_DIR/NeuralAmpModelerCore"
SETUP_THIRD_PARTY_SH="$PROJECT_DIR/utils/setup-third-party.sh"

warn() { echo "WARN: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
emit() { echo "$*"; printf '%s\n' "$*" >> "$LOG_DIR/summary.txt"; }
ok()   { echo "OK: $*"; }

phase() { emit "PHASE: $1"; }
gap()   { GAPS+=("$1"); emit "PHASE: $1 GAP"; }

run_phase() {
  local name=$1 to=${NAM_PHASE_TIMEOUT_S:-$2} log="$LOG_DIR/${SCRIPT_NAME:-run}-$1.log"; shift 2
  phase "$name"
  timeout --kill-after=30 "$to" "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && { [ "${REQUIRE_TESTS:-1}" = 0 ] || ran_tests "$log"; }; then
    emit "PHASE: $name PASS log=$log"
  else
    FAILS+=("$name"); emit "PHASE: $name FAIL rc=$rc log=$log"
  fi
}

ran_tests() {
  local - log="$1" clean n_run n_start n_res n_fail n_pass
  set +x
  clean=$(sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$log" 2>/dev/null || true)
  n_run=$(grep -c -E '^\s*(Running|Doc-tests)\b' <<<"$clean" || true)
  n_start=$(grep -c -E '^\s*running [0-9]+ tests?' <<<"$clean" || true)
  n_res=$(grep -c -E '^test result: (ok|FAILED)\.' <<<"$clean" || true)
  n_fail=$(grep -c 'test result: FAILED' <<<"$clean" || true)
  n_pass=$(grep -c -E 'test result: ok\. [1-9][0-9]* passed' <<<"$clean" || true)
  [ "$n_run" -gt 0 ] && [ "$n_res" -ge "$n_run" ] && [ "$n_start" -eq "$n_res" ] \
    && [ "$n_fail" -eq 0 ] && [ "$n_pass" -gt 0 ]
}

finish() {
  local s=PASS
  [ ${#GAPS[@]} -gt 0 ] && s=PASS_WITH_GAPS
  { [ ${#FAILS[@]} -gt 0 ] || { [ "$STRICT" = 1 ] && [ ${#GAPS[@]} -gt 0 ]; }; } && s=FAIL
  emit "RESULT: $s ($((SECONDS - T0))s) fails=${FAILS[*]:-none} gaps=${GAPS[*]:-none}"
  [ "$s" != FAIL ]
}

require_nightly() {
  local comp="${1:-}"
  command -v rustup >/dev/null 2>&1 || { echo "GAP: rustup not found" >&2; exit 2; }
  rustup toolchain list 2>/dev/null | grep -q "nightly" || {
    echo "GAP: no nightly toolchain installed" >&2; exit 2;
  }
  if [ -n "$comp" ]; then
    cargo +nightly "$comp" --version >/dev/null 2>&1 || {
      echo "GAP: $comp component missing on nightly" >&2; exit 2;
    }
  fi
}
assert_ran_tests() { ran_tests "$1"; }


ensure_third_party() {
  local mode="${1:-soft}"
  [ -d "$CORE_DIR/.git" ] && [ -n "$(ls -A "$CORE_DIR/Dependencies/eigen" 2>/dev/null)" ] \
    && [ -n "$(ls -A "$CORE_DIR/Dependencies/AudioDSPTools" 2>/dev/null)" ] && return 0
  [ -f "$SETUP_THIRD_PARTY_SH" ] || { warn "setup-third-party.sh missing"; return 1; }
  echo "→ third-party mirrors missing — running utils/setup-third-party.sh..."
  bash "$SETUP_THIRD_PARTY_SH" || return 1
  [ -d "$CORE_DIR" ] && return 0
  [ "$mode" = "hard" ] && { echo "ERROR: NeuralAmpModelerCore missing" >&2; return 1; }
  return 1
}

ensure_namcore_render() {
  local build_dir="$PROJECT_DIR/build/namcore_render"
  local logs_dir="$PROJECT_DIR/target/logs"
  local build_type="Release"
  local flags="-w -fno-fast-math -ffp-contract=off"
  local jobs="${NAM_RENDER_JOBS:-$(nproc 2>/dev/null || echo 2)}"
  local bin="" config_file="$build_dir/.build_config"
  command -v cmake >/dev/null 2>&1 || { warn "cmake not found"; return 2; }
  local cxx="${CXX:-}"
  if [ -z "$cxx" ]; then
    command -v g++ >/dev/null 2>&1 && cxx=g++ || { command -v clang++ >/dev/null 2>&1 && cxx=clang++; } || true
  fi
  [ -n "$cxx" ] && command -v "$cxx" >/dev/null 2>&1 || { warn "no C++ compiler found"; return 1; }
  [ -d "$CORE_DIR" ] || { warn "NAMCore missing at $CORE_DIR"; return 3; }
  for cand in tools/render Release/render Debug/render; do
    [ -x "$build_dir/$cand" ] && bin="$build_dir/$cand" && break
  done
  local fingerprint="$cxx:$build_type:$flags"
  if [ -n "$bin" ] && [ -f "$config_file" ] && [ "$(cat "$config_file" 2>/dev/null)" = "$fingerprint" ]; then
    echo "$bin"; return 0
  fi
  rm -rf "$build_dir"
  mkdir -p "$build_dir" "$logs_dir"
  echo "→ compiling C++ NAMCore render ($cxx, $build_type, -j$jobs)" >&2
  cmake -S "$CORE_DIR" -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" \
    -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_FLAGS="$flags" \
    -DNAM_ENABLE_A2_FAST=ON > "$logs_dir/cmake-configure.log" 2>&1 || return 4
  cmake --build "$build_dir" --target render -j"$jobs" > "$logs_dir/cmake-build.log" 2>&1 || return 5
  bin=""
  for cand in tools/render Release/render Debug/render; do
    [ -x "$build_dir/$cand" ] && bin="$build_dir/$cand" && break
  done
  [ -z "$bin" ] && return 6
  echo "$fingerprint" > "$config_file"
  local mc_cpp="$PROJECT_DIR/tests/fixtures/render_multichannel.cpp"
  if [ -f "$mc_cpp" ]; then
    local objs
    objs=$(find "$build_dir/tools/CMakeFiles/render.dir" -name "*.o" ! -name "render.cpp.o" 2>/dev/null || true)
    if [ -n "$objs" ]; then
      "$cxx" -std=c++20 -O3 -I "$CORE_DIR" -I "$CORE_DIR/Dependencies/eigen" \
        -I "$CORE_DIR/Dependencies/AudioDSPTools" -I "$CORE_DIR/Dependencies/nlohmann" \
        "$mc_cpp" $objs -o "$build_dir/render_multichannel" > "$logs_dir/compile-mc.log" 2>&1 || {
        warn "render_multichannel compilation failed (see $logs_dir/compile-mc.log)"
      }
    else
      warn "NAMCore object files not found in $build_dir/tools/CMakeFiles/render.dir"
    fi
  fi
  echo "$bin"
}

ensure_qa_bin() {
  local bin_name="$1" features="${2:-testing}"
  local build_out
  build_out=$(cd "$PROJECT_DIR" && cargo build --locked --workspace ${features:+--features "$features"} --bin "$bin_name" --message-format=json-render-diagnostics 2>&1) || {
    echo "ERROR: cargo build failed for QA binary $bin_name" >&2; return 1
  }
  local exe
  exe=$(printf '%s\n' "$build_out" | grep -F '"reason":"compiler-artifact"' | grep -F "\"name\":\"$bin_name\"" | grep -F '"executable":"' | tail -1 | sed -n 's/.*"executable":"\([^"]*\)".*/\1/p')
  [ -n "$exe" ] && [ -x "$exe" ] || { echo "ERROR: executable artifact not found in cargo compiler-artifact stream for $bin_name" >&2; return 1; }
  echo "$exe"
}

LONG_RECEIPT_BIN=""
ensure_long_receipt_bin() {
  [ -n "$LONG_RECEIPT_BIN" ] && [ -x "$LONG_RECEIPT_BIN" ] && return 0
  LONG_RECEIPT_BIN=$(ensure_qa_bin nam_long_receipt testing) || return 1
  export LONG_RECEIPT_BIN
}

check_freshness() {
  local mode="${1:-hard-fail}"
  local bin
  bin=$(ensure_qa_bin nam_freshness testing) || return 1
  "$bin" --root "$PROJECT_DIR" "$mode"
}
