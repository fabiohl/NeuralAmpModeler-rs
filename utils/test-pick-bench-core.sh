#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# test-pick-bench-core.sh — Unit tests for pick_bench_core helper with simulated sysfs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_lib.sh"

TEST_TMP="$(mktemp -d /tmp/nam_bench_core_test.XXXXXX)"
trap 'rm -rf "$TEST_TMP"' EXIT

# --- Test 1: Explicit NAM_BENCH_CORE override ---
res=$(NAM_BENCH_CORE=4 pick_bench_core)
if [ "$res" -ne 4 ]; then
    echo "FAIL: Test 1 expected 4, got $res" >&2
    exit 1
fi
echo "  ✓ Test 1: explicit NAM_BENCH_CORE override passed"

# --- Test 2: Simulated sysfs with isolated 8-9 (cpu9 offline, cpu8 online) ---
SYSFS_DIR="$TEST_TMP/sysfs_isolated_8_9"
mkdir -p "$SYSFS_DIR/cpu8" "$SYSFS_DIR/cpu9"
echo "8-9" > "$SYSFS_DIR/isolated"
echo "1" > "$SYSFS_DIR/cpu8/online"
echo "0" > "$SYSFS_DIR/cpu9/online"

res=$(unset NAM_BENCH_CORE; pick_bench_core "$SYSFS_DIR")
if [ "$res" -ne 8 ]; then
    echo "FAIL: Test 2 expected 8, got $res" >&2
    exit 1
fi
echo "  ✓ Test 2: isolated 8-9 (cpu9 offline) preferred core 8 passed"

# --- Test 3: Simulated sysfs with empty isolated file (fallback branch) ---
SYSFS_EMPTY="$TEST_TMP/sysfs_empty"
mkdir -p "$SYSFS_EMPTY"
touch "$SYSFS_EMPTY/isolated"

expected_fallback=$(( $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) / 2 ))
err_out=$(mktemp)
res=$(unset NAM_BENCH_CORE; pick_bench_core "$SYSFS_EMPTY" 2>"$err_out")
if [ "$res" -ne "$expected_fallback" ]; then
    echo "FAIL: Test 3 expected fallback $expected_fallback, got $res" >&2
    exit 1
fi
if ! grep -q "WARN: no online isolated CPU found" "$err_out"; then
    echo "FAIL: Test 3 expected warning on stderr, got:" >&2
    cat "$err_out" >&2
    exit 1
fi
rm -f "$err_out"
echo "  ✓ Test 3: fallback to nproc/2 on empty isolated with warning passed"

# --- Test 4: Simulated sysfs with isolated cpu9 offline only (fallback branch) ---
SYSFS_OFFLINE="$TEST_TMP/sysfs_offline"
mkdir -p "$SYSFS_OFFLINE/cpu9"
echo "9" > "$SYSFS_OFFLINE/isolated"
echo "0" > "$SYSFS_OFFLINE/cpu9/online"

err_out=$(mktemp)
res=$(unset NAM_BENCH_CORE; pick_bench_core "$SYSFS_OFFLINE" 2>"$err_out")
if [ "$res" -ne "$expected_fallback" ]; then
    echo "FAIL: Test 4 expected fallback $expected_fallback, got $res" >&2
    exit 1
fi
rm -f "$err_out"
echo "  ✓ Test 4: fallback when isolated CPU is offline passed"

# --- Test 5: Simulated sysfs with range 3-5 (cpu3 online) ---
SYSFS_RANGE="$TEST_TMP/sysfs_range"
mkdir -p "$SYSFS_RANGE/cpu3" "$SYSFS_RANGE/cpu4" "$SYSFS_RANGE/cpu5"
echo "3-5" > "$SYSFS_RANGE/isolated"
echo "1" > "$SYSFS_RANGE/cpu3/online"

res=$(unset NAM_BENCH_CORE; pick_bench_core "$SYSFS_RANGE")
if [ "$res" -ne 3 ]; then
    echo "FAIL: Test 5 expected 3, got $res" >&2
    exit 1
fi
echo "  ✓ Test 5: range isolated core selection passed"

echo "All pick_bench_core unit tests passed successfully!"
