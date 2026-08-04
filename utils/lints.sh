#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Standard quality control and static analysis script for NeuralAmpModeler-rs.
# Runs all cargo checks first (fmt, clippy, check, doc) covering the maximum
# feature spectrum dynamically, followed by static textual and ELF surface checks.
#
# Dynamic feature matrix (broad and resilient to Cargo.toml changes):
#   All Features (catch-all) : --all-targets --all-features
#   Pure Core                : --lib --no-default-features
#   No Default Features      : --all-targets --no-default-features

set -euo pipefail

PHASE_TOTAL=6
source "$(dirname "$0")/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                 NeuralAmpModeler-rs Linting & Quality Suite                 ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# ---------------------------------------------------------------------------
# [1/6] Code formatting (cargo fmt)
# ---------------------------------------------------------------------------
phase "Applying code formatting (cargo fmt)..."
cargo fmt --all

# ---------------------------------------------------------------------------
# [2/6] Static analysis (cargo clippy) — strict, broad feature matrix
# ---------------------------------------------------------------------------
phase "Executing strict static analysis (cargo clippy)..."

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets + All Features (broad catch-all)...${NC}"
cargo clippy --all-targets --all-features

echo -e "  ${YELLOW}${BOLD}Clippy: Pure Core (lib, no features)...${NC}"
cargo clippy --lib --no-default-features

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets (no default features)...${NC}"
cargo clippy --all-targets --no-default-features

# ---------------------------------------------------------------------------
# [3/6] Compilation checks (cargo check) — broad feature matrix
# ---------------------------------------------------------------------------
phase "Executing compilation checks (cargo check)..."

echo -e "  ${YELLOW}${BOLD}Checking: All Targets + All Features (broad catch-all)...${NC}"
cargo check --all-targets --all-features

echo -e "  ${YELLOW}${BOLD}Checking: Pure Core (lib, no features)...${NC}"
cargo check --lib --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: All Targets (no default features)...${NC}"
cargo check --all-targets --no-default-features

# ---------------------------------------------------------------------------
# [4/6] Documentation validation (cargo doc + cargo test --doc)
# ---------------------------------------------------------------------------
phase "Validating documentation (cargo doc + cargo test --doc)..."

echo -e "  ${YELLOW}${BOLD}Building docs (--no-deps, zero warnings, all features)...${NC}"
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

echo -e "  ${YELLOW}${BOLD}Running doc-tests (all features)...${NC}"
cargo test --doc --all-features

# ---------------------------------------------------------------------------
# [5/6] SPDX license header validation (deterministic, no external tooling)
# ---------------------------------------------------------------------------
phase "Validating SPDX license headers..."
spdx_scope=$(
    {
        find src benches tests -type f -name '*.rs'
        find utils -type f -name '*.sh'
        echo "build.rs"
        echo "Cargo.toml"
    } || true
)
# (a) Files missing the SPDX-License-Identifier marker entirely.
missing=$(printf '%s\n' "$spdx_scope" | xargs grep -L "SPDX-License-Identifier" 2>/dev/null || true)
if [ -n "$missing" ]; then
    echo -e "  ${RED}${BOLD}Missing SPDX header in files:${NC}"
    echo "$missing" | sed 's/^/    /'
    exit 1
fi
# (b) Files whose SPDX identifier is not an approved license (Apache-2.0 | MIT).
invalid=$(printf '%s\n' "$spdx_scope" \
    | xargs grep -l "SPDX-License-Identifier" 2>/dev/null \
    | xargs grep -LE "SPDX-License-Identifier: (Apache-2\.0|MIT)" 2>/dev/null || true)
if [ -n "$invalid" ]; then
    echo -e "  ${RED}${BOLD}Invalid SPDX identifier (expected Apache-2.0 or MIT):${NC}"
    echo "$invalid" | sed 's/^/    /'
    exit 1
fi
echo -e "  ${GREEN}OK${NC} — all files have valid SPDX headers (Apache-2.0, MIT)."

# ---------------------------------------------------------------------------
# [6/6] Anti-pattern check: #[test] in tests/common/
# ---------------------------------------------------------------------------
phase "Checking anti-pattern #[test] in tests/common/..."
if grep -rnF "#[test]" tests/common/ >/dev/null 2>&1; then
    echo -e "  ${RED}${BOLD}ERROR: '#[test]' found in tests/common/ (redundant executions):${NC}"
    grep -rnF "#[test]" tests/common/ | sed 's/^/    /'
    exit 1
fi
echo -e "  ${GREEN}OK${NC} — no '#[test]' in tests/common/."
