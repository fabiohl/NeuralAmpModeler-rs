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

PHASE_TOTAL=7
source "$(dirname "$0")/_lib.sh"

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

spdx_scope=$(
    {
        find "${rs_dirs[@]}" -type f -name '*.rs'
        [ -d tests/fixtures ] && find tests/fixtures -type f -name '*.py'
        find utils tests -type f -name '*.sh'
        test -f build.rs && echo build.rs
        test -f Cargo.toml && echo Cargo.toml
    } || true
)

missing=$(printf '%s\n' "$spdx_scope" | xargs -r grep -L "SPDX-License-Identifier" 2>/dev/null || true)
if [ -n "$missing" ]; then
    echo -e "  ${RED}${BOLD}Missing SPDX header in files:${NC}"
    echo "$missing" | sed 's/^/    /'
    exit 1
fi

invalid=$(printf '%s\n' "$spdx_scope" \
    | xargs -r grep -l "SPDX-License-Identifier" 2>/dev/null \
    | xargs -r grep -LE "SPDX-License-Identifier: (Apache-2\.0|MIT)" 2>/dev/null || true)
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

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD} Quality suite completed successfully!                          ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
