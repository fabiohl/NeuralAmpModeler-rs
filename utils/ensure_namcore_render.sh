#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# ensure_namcore_render.sh — CLI wrapper for the single, unified C++ render
# build (S3-T01). All build logic lives in _lib.sh::ensure_namcore_render; this
# script only sources it and forwards the exit code.
#
# Compiles (idempotently) the NAMCore `render` binary into
# build/namcore_render/ and prints its path on success.
#
# Usage:
#   ./utils/ensure_namcore_render.sh
#
# Knobs: CXX, NAM_RENDER_BUILD_TYPE, NAM_RENDER_BUILD_DIR, NAM_RENDER_JOBS,
# NAM_RENDER_FORCE=1 (see _lib.sh for the documented exit codes and logs).

set -euo pipefail

PHASE_TOTAL=1
# shellcheck disable=SC1091
source "$(dirname "$0")/_lib.sh"

ensure_namcore_render "$@"
