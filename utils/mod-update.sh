#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# mod-update.sh — atualiza toolchain, dependências e lockfile. Classe D.
# Uso: utils/mod-update.sh [flags canônicas]
# Orçamento: 70
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
#
# Supply chain update utility for NeuralAmpModeler-rs.
# Updates the Rust toolchain, Cargo package indexes, and dependencies in
# Cargo.toml / Cargo.lock.
#
# Vendor mirrors and optional community models live under third-party/ and are
# prepared by utils/setup-third-party.sh (not this script).

set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) help_exit "Usage: utils/mod-update.sh" ;;
    *) usage_fail "mod-update.sh não aceita argumentos ($*)" ;;
esac
lib_init
trap 'trap_fail $?' ERR

echo "NeuralAmpModeler-rs Supply Chain Update Pipeline"

# 1. Update Rust Toolchain
phase "Updating the active Rust toolchain (rustup)..."
if command -v rustup &>/dev/null; then
    rustup update
else
    warn "rustup not found. Skipping toolchain update."; gap "rustup-missing"
fi

# 2. Upgrade dependencies in Cargo.toml
phase "Upgrading dependency definitions (Cargo.toml)..."
if command -v cargo-upgrade >/dev/null 2>&1; then
    cargo upgrade --verbose
else
    warn "cargo-edit (cargo upgrade) not found. Install with: cargo install cargo-edit"; gap "cargo-upgrade-missing"
fi

# 3. Update Cargo.lock
phase "Updating resolved versions in Cargo.lock..."
cargo update --verbose

ok "Supply chain update finished."
echo "  (Vendor mirrors: run ./utils/setup-third-party.sh if needed.)"
finish
