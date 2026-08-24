#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and static analysis script for NeuralAmpModeler-rs.
# Runs all cargo checks first (fmt, check, clippy, doc) covering the maximum
# feature spectrum dynamically, followed by static textual and policy checks.
#
# Dynamic feature matrix (broad, strict and resilient):
#   All Features (catch-all) : --all-targets --all-features
#   Pure Core                : --lib --no-default-features
#   No Default Features      : --all-targets --no-default-features
#   Individual feature axes  : dynamic-engine, stereo, testing, heap-audit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PHASE_TOTAL=8
source "$SCRIPT_DIR/_lib.sh"

if [ "${NAM_LOW_PRIORITY:-0}" != "1" ] && [ "${NAM_NO_LOW_PRIORITY:-0}" != "1" ]; then
    export NAM_LOW_PRIORITY=1
    CMD_PREFIX=""
    if command -v nice >/dev/null 2>&1; then
        CMD_PREFIX="nice -n 19"
    fi
    if command -v ionice >/dev/null 2>&1; then
        CMD_PREFIX="$CMD_PREFIX ionice -c 3"
    fi
    if [ -n "$CMD_PREFIX" ]; then
        echo -e "${YELLOW}WARN: restarting with low CPU/IO priority (NAM_NO_LOW_PRIORITY=1 to skip)${NC}"
        exec $CMD_PREFIX "$SCRIPT_PATH" "$@"
    fi
fi

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                 NeuralAmpModeler-rs Linting & Quality Suite                 ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# ---------------------------------------------------------------------------
# [1/7] Code formatting (cargo fmt — applies in-place formatting immediately)
# ---------------------------------------------------------------------------
phase "Applying code formatting (cargo fmt in-place)..."
cargo fmt --all
ok "Code formatting applied."

# ---------------------------------------------------------------------------
# [2/7] Compilation checks (cargo check) — maximum feature & target matrix
# ---------------------------------------------------------------------------
phase "Executing compilation checks (cargo check)..."

echo -e "  ${YELLOW}${BOLD}Checking: All Targets + All Features (broad catch-all)...${NC}"
cargo check --locked --all-targets --all-features

echo -e "  ${YELLOW}${BOLD}Checking: Pure Core (lib, no default features)...${NC}"
cargo check --locked --lib --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: All Targets (no default features)...${NC}"
cargo check --locked --all-targets --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (dynamic-engine)...${NC}"
cargo check --locked --all-targets --no-default-features --features dynamic-engine

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (stereo)...${NC}"
cargo check --locked --all-targets --no-default-features --features stereo

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (testing)...${NC}"
cargo check --locked --all-targets --no-default-features --features testing

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (heap-audit)...${NC}"
cargo check --locked --all-targets --no-default-features --features heap-audit

ok "All compilation check permutations passed."

# ---------------------------------------------------------------------------
# [3/7] Static analysis (cargo clippy) — strict, maximum feature matrix
# ---------------------------------------------------------------------------
phase "Executing strict static analysis (cargo clippy)..."

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets + All Features (broad catch-all)...${NC}"
cargo clippy --locked --all-targets --all-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Pure Core (lib, no default features)...${NC}"
cargo clippy --locked --lib --no-default-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets (no default features)...${NC}"
cargo clippy --locked --all-targets --no-default-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Feature Axis (dynamic-engine)...${NC}"
cargo clippy --locked --all-targets --no-default-features --features dynamic-engine -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Feature Axis (stereo)...${NC}"
cargo clippy --locked --all-targets --no-default-features --features stereo -- -D warnings

ok "All static analysis permutations passed cleanly with zero warnings."

# ---------------------------------------------------------------------------
# [4/7] Documentation validation (cargo doc + cargo test --doc)
# ---------------------------------------------------------------------------
phase "Validating documentation (cargo doc + cargo test --doc)..."

echo -e "  ${YELLOW}${BOLD}Building docs (--no-deps, zero warnings, all features)...${NC}"
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

echo -e "  ${YELLOW}${BOLD}Running doc-tests (all features)...${NC}"
cargo test --locked --doc --all-features

ok "Documentation and doc-tests validated."

# ---------------------------------------------------------------------------
# [5/7] SPDX license header validation (deterministic, no external tooling)
# ---------------------------------------------------------------------------
phase "Validating SPDX license headers..."

rs_dirs=( src tests )
[ -d benches ] && rs_dirs+=( benches )
[ -d examples ] && rs_dirs+=( examples )

# T2.4: enumeration is fail-closed — a failing `find` (or any step below)
# aborts the script via `set -e` instead of being swallowed by `|| true`
# into an empty scope that would let missing files pass silently.
spdx_scope=$(
    {
        find "${rs_dirs[@]}" -type f -name '*.rs'
        if [ -d tests/fixtures ]; then find tests/fixtures -type f -name '*.py'; fi
        find utils tests -type f -name '*.sh'
        if [ -f build.rs ]; then echo build.rs; fi
        if [ -f Cargo.toml ]; then echo Cargo.toml; fi
    }
)

# Missing SPDX header: scan file-by-file (no xargs, so grep errors are never
# masked into an empty result) — an unreadable file counts as missing.
missing=""
while IFS= read -r f; do
    [ -n "$f" ] || continue
    if ! grep -q "SPDX-License-Identifier" "$f"; then
        missing+="$f"$'\n'
    fi
done <<< "$spdx_scope"
if [ -n "$missing" ]; then
    echo -e "  ${RED}${BOLD}Missing SPDX header in files:${NC}"
    echo "$missing" | sed 's/^/    /'
    exit 1
fi

# Invalid SPDX identifier (expected Apache-2.0 or MIT): same file-by-file
# scan — a grep error on one file is a hard failure, never a silent pass.
invalid=""
while IFS= read -r f; do
    [ -n "$f" ] || continue
    if grep -q "SPDX-License-Identifier" "$f" \
        && ! grep -qE "SPDX-License-Identifier: (Apache-2\.0|MIT)" "$f"; then
        invalid+="$f"$'\n'
    fi
done <<< "$spdx_scope"
if [ -n "$invalid" ]; then
    echo -e "  ${RED}${BOLD}Invalid SPDX identifier (expected Apache-2.0 or MIT):${NC}"
    echo "$invalid" | sed 's/^/    /'
    exit 1
fi
ok "All files have valid SPDX headers (Apache-2.0, MIT)."

# ---------------------------------------------------------------------------
# [6/7] Anti-pattern check: #[test] in tests/common/
# ---------------------------------------------------------------------------
phase "Checking anti-pattern #[test] in tests/common/..."
if [ -d "tests/common" ] && grep -rnF "#[test]" tests/common/ >/dev/null 2>&1; then
    echo -e "  ${RED}${BOLD}ERROR: '#[test]' found in tests/common/ (redundant executions):${NC}"
    grep -rnF "#[test]" tests/common/ | sed 's/^/    /'
    exit 1
fi
ok "No '#[test]' in tests/common/."

# ---------------------------------------------------------------------------
# [7/7] Undocumented #[allow(clippy::)] check (enforce allow_attributes policy)
# ---------------------------------------------------------------------------
phase "Checking for undocumented #[allow(clippy::)] suppressions..."

undocumented_allows=""
while IFS= read -r rs_file; do
    prev_was_comment=false
    while IFS= read -r line; do
        trimmed="${line#"${line%%[! ]*}"}"
        if [[ "$trimmed" =~ ^\#\[allow\(clippy:: ]]; then
            if ! $prev_was_comment; then
                undocumented_allows+="$rs_file: $trimmed"$'\n'
            fi
            prev_was_comment=false
        elif [[ "$trimmed" =~ ^//|^# ]]; then
            prev_was_comment=true
        elif [ -n "$trimmed" ]; then
            prev_was_comment=false
        fi
    done < "$rs_file"
done < <(printf '%s\n' "$spdx_scope" | grep '\.rs$')

if [ -n "$undocumented_allows" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Undocumented #[allow(clippy::)] found (add a justification comment above):${NC}"
    echo "$undocumented_allows" | sed 's/^/    /'
    exit 1
fi
ok "All #[allow(clippy::)] suppressions are documented."

# ---------------------------------------------------------------------------
# [8/8] Binary scan: zero EVEX/ZMM and zero AVX-512 symbols in default release
# ---------------------------------------------------------------------------
phase "Validating binary artifact (zero AVX-512 in default release build)..."
"$SCRIPT_DIR/verify_no_avx512_release.sh"
ok "Binary artifact is clean of AVX-512 symbols and EVEX instructions."

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD} Quality suite completed successfully!                          ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
