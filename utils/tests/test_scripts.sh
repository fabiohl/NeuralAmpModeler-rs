#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# test_scripts.sh — Stable Bash unit-test suite for the defense scripts.
#
# Exercises utils/_lib.sh and isolated functions of utils/quality-dashboard.sh
# and utils/tests-performance-regression.sh against their own failure modes
# (F-01, F-02, F-08, F-21, F-22, F-24, F-27):
#   * metric sanitization (null/empty/non-finite sentinels)   (F-01, F-28)
#   * toolchain fingerprint without TOOLCHAIN manifest lines   (F-02)
#   * single performance-status classifier (NOT_VERIFIED)      (F-08)
#   * real test-execution assertion & JSONL record counting    (F-21)
#   * conservative freshness gate (STALE/MISSING/ORPHAN/OK)    (F-22)
#   * baseline coverage cross-check helpers                     (F-24)
#   * extended long-suite delegation (--full)                   (F-06)
#   * single jq JSONL parser with canonical edge cases         (F-27)
#
# Usage:
#   bash utils/tests/test_scripts.sh
#
# Exit status: 0 when every test passes (or skips), 1 on any failure.

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$(dirname "$TEST_DIR")"
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

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  test_scripts.sh — passed=${PASSED} failed=${FAILED} skipped=${SKIPPED} (total=${TOTAL})"
if [ "$FAILED" -gt 0 ]; then
    echo "  Failures:"
    for f in "${FAILED_NAMES[@]}"; do echo "    - ${f}"; done
    echo "  RESULT: FAIL"
    exit 1
fi
echo "  RESULT: PASS"
exit 0
