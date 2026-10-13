#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
#
# test-lib-drift.sh — fail-closed drift guard for the shared QA shell assets.
#
# 1. The region between the NAM-LIB-CORE BEGIN/END markers in `_lib_core.sh` is
#    pinned by an embedded `# sha256:` line. This test recomputes the digest of
#    that region (excluding the digest line itself) and fails when it no longer
#    matches, i.e. whenever the shared block diverges from the canonical version.
# 2. The shared RT-codegen filter `dsp-hotpath-filter.py` is byte-identical
#    across repositories except for its SPDX header line; its body digest
#    (header excluded) is pinned below.
# Because every repository carries the same digests, a green test in all three
# proves the shared bodies are byte-identical. Intentionally changing either
# asset requires updating its digest (a deliberate, reviewable act).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_FILE="$SCRIPT_DIR/_lib_core.sh"
FILTER_FILE="$SCRIPT_DIR/dsp-hotpath-filter.py"
# sha256 of the shared filter body with its SPDX header line removed.
FILTER_BODY_SHA256="8b70604524e3a935e6681319a6328c3b3c775f0eff095bad2e1bfd6671d4abdb"

if [ ! -f "$CORE_FILE" ]; then
    echo "FAIL: core lib not found: $CORE_FILE" >&2
    exit 1
fi

embedded="$(grep -m1 '^# sha256: ' "$CORE_FILE" | awk '{print $3}' || true)"
if [ -z "$embedded" ]; then
    echo "FAIL: no embedded '# sha256:' digest in $CORE_FILE" >&2
    exit 1
fi

if ! grep -q 'NAM-LIB-CORE BEGIN' "$CORE_FILE" || ! grep -q 'NAM-LIB-CORE END' "$CORE_FILE"; then
    echo "FAIL: NAM-LIB-CORE BEGIN/END markers missing in $CORE_FILE" >&2
    exit 1
fi

# Extract the canonical region, dropping the marker-adjacent digest line.
extract_body() {
    awk '
        /NAM-LIB-CORE BEGIN/ { inb=1; next }
        /NAM-LIB-CORE END/ { inb=0; next }
        inb && !/^# sha256: / { print }
    ' "$1"
}

if command -v sha256sum >/dev/null 2>&1; then
    computed="$(extract_body "$CORE_FILE" | sha256sum | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    computed="$(extract_body "$CORE_FILE" | shasum -a 256 | awk '{print $1}')"
else
    echo "FAIL: no sha256 tool (sha256sum/shasum) available (fail-closed)" >&2
    exit 1
fi

if [ "$computed" != "$embedded" ]; then
    echo "FAIL: shell core drift detected in $CORE_FILE" >&2
    echo "      embedded: $embedded" >&2
    echo "      computed: $computed" >&2
    exit 1
fi

echo "  OK shell core digest verified ($computed)"

# ── Shared RT-codegen filter body (SPDX header excluded) ─────────────────────
if [ ! -f "$FILTER_FILE" ]; then
    echo "FAIL: shared filter not found: $FILTER_FILE" >&2
    exit 1
fi

# The first line is the shebang and the second the SPDX header; only the SPDX
# line may legitimately differ across repositories, so it is dropped before
# hashing — everything else must be byte-identical.
extract_filter_body() {
    awk '!/^# SPDX-License-Identifier:/' "$1"
}

if command -v sha256sum >/dev/null 2>&1; then
    filter_computed="$(extract_filter_body "$FILTER_FILE" | sha256sum | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    filter_computed="$(extract_filter_body "$FILTER_FILE" | shasum -a 256 | awk '{print $1}')"
else
    echo "FAIL: no sha256 tool (sha256sum/shasum) available (fail-closed)" >&2
    exit 1
fi

if [ "$filter_computed" != "$FILTER_BODY_SHA256" ]; then
    echo "FAIL: RT-codegen filter body drift detected in $FILTER_FILE" >&2
    echo "      expected: $FILTER_BODY_SHA256" >&2
    echo "      computed: $filter_computed" >&2
    exit 1
fi

echo "  OK shared filter body digest verified ($filter_computed)"
exit 0
