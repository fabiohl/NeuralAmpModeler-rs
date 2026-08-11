#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Supply chain update utility for NeuralAmpModeler-rs.
# Updates the Rust toolchain, Cargo package indexes, and dependencies in
# Cargo.toml / Cargo.lock.
#
# Vendor mirrors and optional community models live under third-party/ and are
# prepared by utils/setup-third-party.sh (not this script).

set -euo pipefail

PHASE_TOTAL=3
source "$(dirname "$0")/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}   NeuralAmpModeler-rs Supply Chain Update Pipeline             ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

# 1. Update Rust Toolchain
phase "Updating the active Rust toolchain (rustup)..."
if command -v rustup &>/dev/null; then
    rustup update
else
    echo -e "${YELLOW}Warning: rustup not found. Skipping toolchain update.${NC}"
fi

# 2. Upgrade dependencies in Cargo.toml
phase "Upgrading dependency definitions (Cargo.toml)..."
if cargo --list | grep -q "upgrade"; then
    cargo upgrade --verbose
else
    echo -e "${YELLOW}Warning: cargo-edit (cargo upgrade) not found.${NC}"
    echo -e "${YELLOW}Install with: cargo install cargo-edit${NC}"
fi

# 3. Update Cargo.lock
phase "Updating resolved versions in Cargo.lock..."
cargo update --verbose

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}          Supply chain update complete.                         ${NC}"
echo -e "${GREEN}${BOLD}  (Vendor mirrors: run ./utils/setup-third-party.sh if needed.) ${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
