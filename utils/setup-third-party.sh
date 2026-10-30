#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# setup-third-party.sh — clona os espelhos C++ fixados. Classe D.
# Uso: utils/setup-third-party.sh [flags canônicas]
# Orçamento: 141
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
#
# Populate NeuralAmpModeler-rs/third-party/ with external artifacts needed for
# full local development and parity/golden pipelines.
#
# Intended for a freshly cloned repository (or after wiping third-party/).
# The crate and most tests skip gracefully when these artifacts are absent;
# this script is only required when working on NeuralAmpModeler-rs itself
# (parity against NAMCore, golden regeneration, cabsim C++ cross-validation,
# optional community-model integration tests).
#
# What this prepares (all under $PROJECT_DIR/third-party/, gitignored):
#   NeuralAmpModelerCore/   — pinned C++ NAMCore mirror (render tool / parity)
#   NeuralAmpModelerPlugin/ — pinned C++ plugin mirror (IR / cabsim reference)
#   community_models/       — optional symlink to a local non-distributable
#                             model archive (never cloned; never committed)
#
# Pins and clone URLs: variables.env (NAM_CORE_*, NAM_PLUGIN_*).
# Overrides: NAM_COMMUNITY_MODELS_SRC (absolute path to link as community_models).

set -euo pipefail
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/setup-third-party.sh"; exit 0 ;;
    *) echo "Usage: utils/setup-third-party.sh" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
. utils/_lib.sh

if ! command -v git >/dev/null 2>&1; then
    die "git binary is required but not found in PATH."
fi

VARIABLES_ENV="${VARIABLES_ENV:-$PROJECT_DIR/variables.env}"
if [ ! -f "$VARIABLES_ENV" ]; then
    die "variables.env not found at $VARIABLES_ENV (expected at the NeuralAmpModeler-rs repository root)."
fi
# shellcheck source=/dev/null
source "$VARIABLES_ENV"

mkdir -p "$THIRD_PARTY_DIR"

sync_git_pin() {
    local name="$1" dir="$2" repo="$3" tag="$4" commit="$5"

    if [ -e "$dir/.git" ]; then
        echo -e "  Found $name at $dir — updating to pin..."
        (
            cd "$dir"
            git fetch --depth 1 origin tag "$tag" 2>/dev/null || git fetch origin "$tag" 2>/dev/null || git fetch origin
            if ! git checkout "$commit" 2>/dev/null; then
                git fetch --depth 1 origin "$commit" 2>/dev/null || git fetch origin "$commit" 2>/dev/null || git fetch origin
                git checkout "$commit"
            fi
            git clean -df
        )
        ok "$name synced ($tag @ $commit)."
    elif [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null || true)" ]; then
        die "$dir exists but is not a git checkout; remove it and re-run."
    else
        echo -e "  Cloning $name for the first time..."
        rm -rf "$dir"
        git clone --depth 1 --branch "$tag" "$repo" "$dir"
        (
            cd "$dir"
            if ! git checkout "$commit" 2>/dev/null; then
                git fetch --depth 1 origin "$commit" 2>/dev/null || git fetch origin "$commit" 2>/dev/null || git fetch origin
                git checkout "$commit"
            fi
        )
        ok "$name cloned ($tag @ $commit)."
    fi
}

PLUGIN_DIR="$THIRD_PARTY_DIR/NeuralAmpModelerPlugin"

# 1. NeuralAmpModelerCore
phase "Syncing NeuralAmpModelerCore mirror..."
sync_git_pin "NeuralAmpModelerCore" "$CORE_DIR" \
    "$NAM_CORE_REPO" "$NAM_CORE_TAG" "$NAM_CORE_COMMIT"

for sub in eigen AudioDSPTools; do
    sub_path="$CORE_DIR/Dependencies/$sub"
    if [ ! -d "$sub_path" ] || [ -z "$(ls -A "$sub_path" 2>/dev/null || true)" ]; then
        echo "  Initializing submodule $sub for NeuralAmpModelerCore..."
        (cd "$CORE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done
ok "NeuralAmpModelerCore submodules ready."

# 2. NeuralAmpModelerPlugin
phase "Syncing NeuralAmpModelerPlugin mirror..."
sync_git_pin "NeuralAmpModelerPlugin" "$PLUGIN_DIR" \
    "$NAM_PLUGIN_REPO" "$NAM_PLUGIN_TAG" "$NAM_PLUGIN_COMMIT"

sub_plugin_path="$PLUGIN_DIR/AudioDSPTools"
if [ ! -d "$sub_plugin_path" ] || [ ! -d "$sub_plugin_path/Dependencies/eigen/Eigen" ]; then
    echo -e "  Initializing submodule AudioDSPTools for NeuralAmpModelerPlugin..."
    (cd "$PLUGIN_DIR" && git submodule update --init --recursive AudioDSPTools)
    ok "NeuralAmpModelerPlugin submodules ready."
else
    ok "NeuralAmpModelerPlugin submodules already present."
fi

# 3. community_models (optional local symlink — never fetched from git)
phase "Checking community_models (optional non-distributable archive)..."
COMMUNITY_LINK="$THIRD_PARTY_DIR/community_models"
if [ -e "$COMMUNITY_LINK" ] || [ -L "$COMMUNITY_LINK" ]; then
    if [ -L "$COMMUNITY_LINK" ]; then
        target="$(readlink -f "$COMMUNITY_LINK" 2>/dev/null || readlink "$COMMUNITY_LINK")"
        ok "community_models → $target"
    else
        ok "community_models present at $COMMUNITY_LINK"
    fi
elif [ -n "${NAM_COMMUNITY_MODELS_SRC:-}" ]; then
    if [ ! -d "$NAM_COMMUNITY_MODELS_SRC" ]; then
        die "NAM_COMMUNITY_MODELS_SRC is not a directory: $NAM_COMMUNITY_MODELS_SRC"
    fi
    ln -s "$NAM_COMMUNITY_MODELS_SRC" "$COMMUNITY_LINK"
    ok "Linked community_models → $NAM_COMMUNITY_MODELS_SRC"
else
    warn "community_models not configured (optional)."
    echo "  For local non-distributable test models, create a symlink:"
    echo "    ln -s /path/to/your/nam_models $COMMUNITY_LINK"
    echo "  Or re-run with NAM_COMMUNITY_MODELS_SRC=/path/to/your/nam_models"
fi

ok "third-party environment ready under: $THIRD_PARTY_DIR"
echo "RESULT: PASS ($((SECONDS - T0))s) fails=none gaps=none"
