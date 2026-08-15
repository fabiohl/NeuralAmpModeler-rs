#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
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
# Overrides: NAM_THIRD_PARTY_DIR, NAM_CORE_DIR, NAM_PLUGIN_DIR, NAM_VARIABLES_ENV,
#            NAM_COMMUNITY_MODELS_SRC (absolute path to link as community_models).

set -euo pipefail

PHASE_TOTAL=3
source "$(dirname "$0")/_lib.sh"

echo -e "${BLUE}${BOLD}================================================================${NC}"
echo -e "${BLUE}${BOLD}   NeuralAmpModeler-rs — third-party environment setup          ${NC}"
echo -e "${BLUE}${BOLD}================================================================${NC}"

if ! command -v git >/dev/null 2>&1; then
    die "git binary is required but not found in PATH."
fi

if [ ! -f "$VARIABLES_ENV" ]; then
    echo -e "${RED}${BOLD}❌ variables.env not found at $VARIABLES_ENV.${NC}"
    echo -e "${YELLOW}Expected at the NeuralAmpModeler-rs repository root (override via NAM_VARIABLES_ENV).${NC}"
    exit 1
fi
# shellcheck source=/dev/null
source "$VARIABLES_ENV"

mkdir -p "$THIRD_PARTY_DIR"

sync_git_pin() {
    local name="$1"
    local dir="$2"
    local repo="$3"
    local tag="$4"
    local commit="$5"

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
        echo -e "${RED}${BOLD}❌ $dir exists but is not a git checkout.${NC}"
        echo -e "${YELLOW}Remove it and re-run, or point NAM_CORE_DIR / NAM_PLUGIN_DIR elsewhere.${NC}"
        exit 1
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

# 1. NeuralAmpModelerCore
phase "Syncing NeuralAmpModelerCore mirror..."
sync_git_pin "NeuralAmpModelerCore" "$NAM_CORE_DIR" \
    "$NAM_CORE_REPO" "$NAM_CORE_TAG" "$NAM_CORE_COMMIT"

for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ ! -d "$sub_path" ] || [ -z "$(ls -A "$sub_path" 2>/dev/null || true)" ]; then
        echo "  Initializing submodule $sub for NeuralAmpModelerCore..."
        (cd "$NAM_CORE_DIR" && git submodule update --init "Dependencies/$sub")
    fi
done
ok "NeuralAmpModelerCore submodules ready."

# 2. NeuralAmpModelerPlugin
phase "Syncing NeuralAmpModelerPlugin mirror..."
sync_git_pin "NeuralAmpModelerPlugin" "$NAM_PLUGIN_DIR" \
    "$NAM_PLUGIN_REPO" "$NAM_PLUGIN_TAG" "$NAM_PLUGIN_COMMIT"

sub_plugin_path="$NAM_PLUGIN_DIR/Dependencies/AudioDSPTools"
if [ ! -d "$sub_plugin_path" ] || [ -z "$(ls -A "$sub_plugin_path" 2>/dev/null || true)" ]; then
    echo -e "  Initializing submodules for NeuralAmpModelerPlugin..."
    (cd "$NAM_PLUGIN_DIR" && git submodule update --init --recursive Dependencies/AudioDSPTools 2>/dev/null || (cd "$NAM_PLUGIN_DIR" && git submodule update --init --recursive))
    ok "NeuralAmpModelerPlugin submodules initialized."
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
    echo -e "  ${YELLOW}  For local non-distributable test models, create a symlink:${NC}"
    echo -e "  ${YELLOW}    ln -s /path/to/your/nam_models $COMMUNITY_LINK${NC}"
    echo -e "  ${YELLOW}  Or re-run with NAM_COMMUNITY_MODELS_SRC=/path/to/your/nam_models${NC}"
fi

echo -e "${GREEN}${BOLD}================================================================${NC}"
echo -e "${GREEN}${BOLD}  third-party environment ready under:${NC}"
echo -e "${GREEN}${BOLD}    $THIRD_PARTY_DIR${NC}"
echo -e "${GREEN}${BOLD}================================================================${NC}"
