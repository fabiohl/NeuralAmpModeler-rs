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
#   Individual feature axes  : fft-radix4-planner, dual-mono, testing, heap-audit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

PHASE_TOTAL=9
source "$SCRIPT_DIR/_lib.sh"

# Shared helper in _lib.sh; skips itself when already restarted or disabled.
# (quality-dashboard.sh / tests-performance-regression.sh intentionally do NOT
# deprioritize: they drive the statistical benchmarks — see _lib.sh.)
maybe_restart_low_priority "$SCRIPT_PATH" "$@"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}                 NeuralAmpModeler-rs Linting & Quality Suite                 ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"
SUITE_START=$(date +%s%N)

# ---------------------------------------------------------------------------
# [1/8] Code formatting (cargo fmt — applies in-place formatting immediately)
# ---------------------------------------------------------------------------
phase "Applying code formatting (cargo fmt in-place)..."
cargo fmt --all
ok "Code formatting applied ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [2/8] Compilation checks (cargo check) — maximum feature & target matrix
# ---------------------------------------------------------------------------
phase "Executing compilation checks (cargo check)..."

echo -e "  ${YELLOW}${BOLD}Checking: All Targets + All Features (broad catch-all)...${NC}"
cargo check --locked --all-targets --all-features

echo -e "  ${YELLOW}${BOLD}Checking: Pure Core (lib, no default features)...${NC}"
cargo check --locked --lib --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: All Targets (no default features)...${NC}"
cargo check --locked --all-targets --no-default-features

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (fft-radix4-planner)...${NC}"
cargo check --locked --all-targets --no-default-features --features fft-radix4-planner

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (dual-mono)...${NC}"
cargo check --locked --all-targets --no-default-features --features dual-mono

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (testing)...${NC}"
cargo check --locked --all-targets --no-default-features --features testing

echo -e "  ${YELLOW}${BOLD}Checking: Feature Axis (heap-audit)...${NC}"
cargo check --locked --all-targets --no-default-features --features heap-audit

ok "All compilation check permutations passed ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [3/8] Static analysis (cargo clippy) — strict, maximum feature matrix
# ---------------------------------------------------------------------------
phase "Executing strict static analysis (cargo clippy)..."

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets + All Features (broad catch-all)...${NC}"
cargo clippy --locked --all-targets --all-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Pure Core (lib, no default features)...${NC}"
cargo clippy --locked --lib --no-default-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: All Targets (no default features)...${NC}"
cargo clippy --locked --all-targets --no-default-features -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Feature Axis (fft-radix4-planner)...${NC}"
cargo clippy --locked --all-targets --no-default-features --features fft-radix4-planner -- -D warnings

echo -e "  ${YELLOW}${BOLD}Clippy: Feature Axis (dual-mono)...${NC}"
cargo clippy --locked --all-targets --no-default-features --features dual-mono -- -D warnings

ok "All static analysis permutations passed cleanly with zero warnings ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [4/8] Documentation validation (cargo doc + cargo test --doc)
# ---------------------------------------------------------------------------
phase "Validating documentation (cargo doc + cargo test --doc)..."

echo -e "  ${YELLOW}${BOLD}Building docs (--no-deps, zero warnings, all features)...${NC}"
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features

echo -e "  ${YELLOW}${BOLD}Running doc-tests (all features)...${NC}"
cargo test --locked --doc --all-features

ok "Documentation and doc-tests validated ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [5/8] SPDX license header validation (deterministic, no external tooling)
# ---------------------------------------------------------------------------
phase "Validating SPDX license headers..."

rs_dirs=( src tests )
[ -d benches ] && rs_dirs+=( benches )
[ -d examples ] && rs_dirs+=( examples )

# enumeration is fail-closed — a failing `find` (or any step below)
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
ok "All files have valid SPDX headers (Apache-2.0, MIT) ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [6/8] Anti-pattern check: #[test] in tests/common/
# ---------------------------------------------------------------------------
phase "Checking anti-pattern #[test] in tests/common/..."
if [ ! -d "tests/common" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Directory tests/common/ is missing. Cannot verify anti-patterns.${NC}"
    exit 1
fi
# Fail-closed grep handling (mirrors the SPDX scan above): rc==1 means "no
# match" and passes; rc>=2 is a tool error (e.g. unreadable file) and must
# fail the gate — a grep error can never be mistaken for a clean tree.
grep_rc=0
grep -rnF "#[test]" tests/common/ >/dev/null 2>&1 || grep_rc=$?
if [ "$grep_rc" -ge 2 ]; then
    echo -e "  ${RED}${BOLD}ERROR: grep failed scanning tests/common/ (rc=$grep_rc) — gate fails closed.${NC}"
    exit 1
fi
if [ "$grep_rc" -eq 0 ]; then
    echo -e "  ${RED}${BOLD}ERROR: '#[test]' found in tests/common/ (redundant executions):${NC}"
    grep -rnF "#[test]" tests/common/ | sed 's/^/    /'
    exit 1
fi
ok "No '#[test]' in tests/common/ ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [7/8] Undocumented #[allow(...)] / #![allow(...)] check (enforce allow_attributes policy)
# ---------------------------------------------------------------------------
phase "Checking for undocumented #[allow(...)] / #![allow(...)] suppressions..."

undocumented_allows=""
while IFS= read -r rs_file; do
    prev_was_comment=false
    while IFS= read -r line; do
        trimmed="${line#"${line%%[! ]*}"}"
        # Every #[allow(...)] (item-level) and #![allow(...)] (inner) attribute
        # requires a justification comment — the pattern list is intentionally
        # open-ended so newly-used lint names (deprecated, unused,
        # unreachable_code, ...) cannot bypass the gate.
        if [[ "$trimmed" =~ ^#!\[allow\(|^#\[allow\( ]]; then
            if ! $prev_was_comment; then
                undocumented_allows+="$rs_file: $trimmed"$'\n'
            fi
            # Stacked/consecutive allow attributes share the preceding justification comment
        elif [[ "$trimmed" =~ ^//\ SPDX|^//\ Copyright ]]; then
            prev_was_comment=false
        elif [[ "$trimmed" =~ ^//!|^/// ]]; then
            # Doc comments describe item/module purpose, not justification for suppressions
            prev_was_comment=false
        elif [[ "$trimmed" =~ ^//|^\/\* ]]; then
            prev_was_comment=true
        elif [ -n "$trimmed" ]; then
            prev_was_comment=false
        fi
    done < "$rs_file"
done < <(printf '%s\n' "$spdx_scope" | grep '\.rs$')

if [ -n "$undocumented_allows" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Undocumented #[allow(...)] / #![allow(...)] found (add a justification comment above):${NC}"
    echo "$undocumented_allows" | sed 's/^/    /'
    exit 1
fi
ok "All #[allow(...)] / #![allow(...)] suppressions are documented ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [8/8] Static validation: doc(cfg(feature = "...")) feature names exist in Cargo.toml
# ---------------------------------------------------------------------------
phase "Validating doc(cfg) feature names against Cargo.toml..."

doc_cfg_errors=""
cargo_features=$(python3 -c "
import tomllib
with open('Cargo.toml', 'rb') as f:
    data = tomllib.load(f)
for k in data.get('features', {}).keys():
    print(k)
")

# Fail-closed: git grep exit 1 = no files found, which is itself a gate failure.
doc_cfg_matches=""
if ! doc_cfg_matches=$(git grep -n -o -E 'doc\(cfg\(feature = "[^"]+"\)\)' src/); then
    echo -e "  ${RED}${BOLD}ERROR: No doc(cfg(feature = ...)) annotations found in src/ (or git grep failed).${NC}"
    exit 1
fi

while IFS=: read -r file line match; do
    [ -n "$file" ] || continue
    feat=$(echo "$match" | sed -E 's/.*doc\(cfg\(feature = "([^"]+)".*/\1/')
    if ! echo "$cargo_features" | grep -qx "$feat"; then
        doc_cfg_errors+="$file:$line: feature '$feat' not found in Cargo.toml [features]"$'\n'
    fi
done <<< "$doc_cfg_matches"

if [ -n "$doc_cfg_errors" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Invalid feature name(s) in doc(cfg):${NC}"
    echo "$doc_cfg_errors" | sed 's/^/    /'
    exit 1
fi
ok "All doc(cfg) feature annotations match declared features in Cargo.toml ($(phase_elapsed_str))."

# ---------------------------------------------------------------------------
# [9/9] Static validation: README crate version and MSRV against Cargo.toml
# ---------------------------------------------------------------------------
phase "Validating README crate version and MSRV against Cargo.toml..."

cargo_version=$(grep -m1 '^version = ' Cargo.toml | sed -E 's/version = "([0-9]+\.[0-9]+).*"/\1/')
cargo_msrv=$(grep -m1 '^rust-version = ' Cargo.toml | sed -E 's/rust-version = "([^"]+)"/\1/')

if [ -z "$cargo_version" ]; then
    echo -e "  ${RED}${BOLD}ERROR: Could not extract package version from Cargo.toml${NC}"
    exit 1
fi

# Extract version numbers cited in README dependency blocks
# Matches both 'NeuralAmpModeler-rs = "X.Y"' and '{ version = "X.Y", ... }'
readme_version_matches=$(grep -nE 'NeuralAmpModeler-rs = ("([0-9]+\.[0-9]+)"|\{ version = "([0-9]+\.[0-9]+)")' README.md || true)
if [ -z "$readme_version_matches" ]; then
    echo -e "  ${RED}${BOLD}ERROR: No NeuralAmpModeler-rs dependency version specifications found in README.md${NC}"
    exit 1
fi

readme_ver_errors=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    ver=$(echo "$line" | grep -oE '"[0-9]+\.[0-9]+"' | tr -d '"')
    if [ "$ver" != "$cargo_version" ]; then
        readme_ver_errors+="  $line (expected $cargo_version, found $ver)"$'\n'
    fi
done <<< "$readme_version_matches"

if [ -n "$readme_ver_errors" ]; then
    echo -e "  ${RED}${BOLD}ERROR: README.md dependency version diverged from Cargo.toml ($cargo_version):${NC}"
    echo "$readme_ver_errors"
    exit 1
fi

# Validate MSRV badge matches rust-version in Cargo.toml
if [ -n "$cargo_msrv" ]; then
    readme_msrv=$(grep -oE 'badge/MSRV-[0-9]+\.[0-9]+(\.[0-9]+)?' README.md | sed -E 's/badge\/MSRV-//' || true)
    if [ -z "$readme_msrv" ]; then
        echo -e "  ${RED}${BOLD}ERROR: MSRV badge not found in README.md (expected MSRV-$cargo_msrv)${NC}"
        exit 1
    fi
    if [ "$readme_msrv" != "$cargo_msrv" ]; then
        echo -e "  ${RED}${BOLD}ERROR: README.md MSRV badge ($readme_msrv) diverged from Cargo.toml ($cargo_msrv)${NC}"
        exit 1
    fi
fi
ok "README version ($cargo_version) and MSRV ($cargo_msrv) match Cargo.toml ($(phase_elapsed_str))."

SUITE_END=$(date +%s%N)
TOTAL_DUR_MS=$(( (SUITE_END - SUITE_START) / 1000000 ))
TOTAL_DUR_STR=$(format_duration_ms "$TOTAL_DUR_MS")

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD} Quality suite completed successfully in ${TOTAL_DUR_STR}!         ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
