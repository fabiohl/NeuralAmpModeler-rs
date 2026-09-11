#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# SIMD Diagnostic & Capability Probe wrapper.
# Runs the `simd_probe` CLI with or without the opt-in `avx512` feature.
# See src/bin/simd_probe.rs for the full probe semantics.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
if [[ "${1:-}" == "--avx512" ]]; then
    cargo run --quiet --features avx512 --bin simd_probe
else
    cargo run --quiet --bin simd_probe
fi
