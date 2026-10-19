#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# Runs a command inside a transient systemd user scope with memory containment
# and elevated OOM preference. Thin launcher delegating to mem_guard_wrap in _lib_core.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/_lib_core.sh" ]; then
    source "$SCRIPT_DIR/_lib_core.sh"
elif [ -f "$SCRIPT_DIR/lib/_lib_core.sh" ]; then
    source "$SCRIPT_DIR/lib/_lib_core.sh"
else
    echo "mem-guard: _lib_core.sh not found relative to $SCRIPT_DIR" >&2
    exit 1
fi

mem_guard_wrap "$@"
