#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
# ensure_lanczos_probe.sh — compila o probe Lanczos C++ de comparação (S12) a partir do espelho vendor. Classe D: stdout é apenas o caminho do binário; RESULT: vai para stderr.
# Uso: utils/ensure_lanczos_probe.sh [flags canônicas]
# Orçamento: 64        # +9: fingerprint com espelho plugin/NAMCore + receipt .build_provenance
set -x; PS4='+ ${BASH_SOURCE##*/}:${LINENO}: '
set -euo pipefail
cd "$(dirname "$0")/.."
. utils/_lib.sh

ensure_lanczos_probe() {
    local build_dir="$PROJECT_DIR/build/resampler_probe" logs_dir="$PROJECT_DIR/target/logs"
    local bin="$build_dir/lanczos_probe" config_file="$build_dir/.build_config"
    local plugin_dir="${NAM_PLUGIN_DIR:-$THIRD_PARTY_DIR/NeuralAmpModelerPlugin}"
    local core_dir="${NAM_CORE_DIR:-$CORE_DIR}"
    local container_dir="$plugin_dir/AudioDSPTools/dsp/ResamplingContainer"
    local src="$PROJECT_DIR/tests/resampler_cxx/lanczos_probe.cpp"
    local flags="-w -O2 -std=c++20 -ffp-contract=off -fno-fast-math"
    local cxx="${CXX:-$(command -v g++ >/dev/null 2>&1 && echo g++ || echo clang++)}"
    command -v "$cxx" >/dev/null 2>&1 || { warn "no C++ compiler found ($cxx)"; return 1; }
    [ -f "$src" ] || { warn "probe source missing: $src"; return 2; }
    [ -f "$container_dir/ResamplingContainer.h" ] || { warn "vendor ResamplingContainer.h missing at $container_dir"; return 3; }
    local core_commit plugin_commit cxx_ver cont_md5 lan_md5 src_md5
    core_commit=$(git -C "$core_dir" rev-parse HEAD 2>/dev/null || echo unknown)
    plugin_commit=$(git -C "$plugin_dir" rev-parse HEAD 2>/dev/null || echo absent)
    cxx_ver=$("$cxx" --version 2>/dev/null | head -n 1)
    cont_md5=$(md5sum "$container_dir/ResamplingContainer.h" | awk '{print $1}')
    lan_md5=$(md5sum "$container_dir/Dependencies/LanczosResampler.h" | awk '{print $1}')
    src_md5=$(md5sum "$src" | awk '{print $1}')
    local fingerprint="probe=$cxx:$flags:core=$core_commit:plugin=$plugin_commit:cont=$cont_md5:lan=$lan_md5:src=$src_md5:cc=$cxx_ver"
    if [ -x "$bin" ] && [ -f "$config_file" ] && [ "$(cat "$config_file" 2>/dev/null)" = "$fingerprint" ]; then
        printf '%s' "$bin"
        return 0
    fi
    mkdir -p "$build_dir" "$logs_dir"
    echo "→ compiling Lanczos probe ($cxx, $flags)" >&2
    "$cxx" $flags "-I$container_dir" "$src" -o "$bin" > "$logs_dir/lanczos-probe-build.log" 2>&1 || return 4
    printf '%s\n' "$fingerprint" > "$config_file"
    printf 'core_commit=%s\nplugin_commit=%s\ncont_md5=%s\nlan_md5=%s\nprobe_src_md5=%s\ncompiler=%s\nflags=%s\n' \
        "$core_commit" "$plugin_commit" "$cont_md5" "$lan_md5" "$src_md5" "$cxx" "$flags" > "$build_dir/.build_provenance"
    printf '%s' "$bin"
    return 0
}

case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) echo "Usage: utils/ensure_lanczos_probe.sh"; exit 0 ;;
    *) usage_fail "ensure_lanczos_probe.sh não aceita argumentos ($*)" >&2 ;;
esac

T0=$SECONDS
lib_init >&2
phase probe >&2
if path=$(ensure_lanczos_probe); then
    emit "PHASE: probe PASS" >&2
    printf '%s\n' "$path"
    finish >&2 || exit 1
else
    rc=$?
    FAILS+=(probe); emit "PHASE: probe FAIL rc=$rc" >&2
    finish >&2 || true
    exit "$rc"
fi
