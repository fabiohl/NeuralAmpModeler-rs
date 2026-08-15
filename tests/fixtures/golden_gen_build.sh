#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

# golden_gen_build.sh — Builds the NeuralAmpModelerCore render tool, clones
# NeuralAmpModelerPlugin (C++ IR reference), and generates all golden vectors.
#
# Canonical reference: NeuralAmpModelerCore (tag pinned in variables.env),
# NeuralAmpModelerPlugin (IR reference, tag also pinned in variables.env).
# All goldens (A1/LSTM/WaveNet/A2/ConvNet/Dyn) are rendered from a single pinned
# commit.  Pinned versions (commits, tags, repo URLs) live in
# variables.env — sourced by both this script and utils/setup-third-party.sh.
# A mismatch between the vendored working copy and the pin in variables.env causes
# this script's version-mismatch guard (below) to hard-fail. Some older committed
# goldens were rendered at v0.5.3 (9c7b185); the patch-level diff is below the
# interop noise floor for all architectures except where explicitly noted in
# docs/cpp_parity_map.md §1.3.
#
# Prerequisites:
#   - cmake >= 3.10, g++ or clang++ with C++20
#   - cargo (Rust; stress signal generation and WAV→golden conversion are now Rust native)
#   - python3 (for generating synthetic A2 dynamic/FiLM fixtures)
#   - git (to clone NeuralAmpModelerCore and NeuralAmpModelerPlugin if needed)
#
# Reproducibility:
#   Upstream commits are pinned in variables.env (NAM_CORE_COMMIT,
#   NAM_PLUGIN_COMMIT). Update there when regenerating goldens from a newer
#   upstream version.
#
# Python is required for generating synthetic A2 dynamic/FiLM fixtures (generate_a2_fixtures.py).
#
# Usage:
#   ./tests/fixtures/golden_gen_build.sh
#
# Repo-local vendor mirrors (managed by utils/setup-third-party.sh):
#   third-party/NeuralAmpModelerCore/   (~143 MB) — C++ upstream render engine
#   third-party/NeuralAmpModelerPlugin/ (~164 MB) — C++ upstream plugin (IR reference)
#   third-party/community_models/       (optional symlink) — non-distributable test models
#   build/namcore_render/               (~6 MB)   — C++ build artifacts (repo-local, gitignored)
#   The third-party base directory can be overridden via NAM_THIRD_PARTY_DIR.
#
# Output (tests/fixtures/):
#   golden_wavenet_standard.bin, golden_wavenet_lite.bin, golden_wavenet_feather.bin, golden_wavenet_nano.bin
#   golden_lstm_1x16.bin, golden_lstm_2x8.bin, golden_lstm_official.bin
#   golden_wavenet_a2_full.bin, golden_wavenet_a2_lite.bin
#   (A2 goldens are cross-reference Rust↔C++ v0.5.4 via ESR/SNR scale-invariant
#    gate — self-goldens removed in T2.6.)
#   golden_convnet_test.bin, golden_wavenet_dyn_free.bin, golden_lstm_dyn_test.bin
#   (ConvNet and dynamic model goldens from dynamic architecture fixtures — sample_rate=48000)
#   golden_cabsim_cpp_short.bin, golden_cabsim_cpp_medium.bin,
#   golden_cabsim_cpp_long.bin
#   (C++ dsp::ImpulseResponse reference for cabsim cross-validation)
#   golden_a2_dynamic_gated_ch8.bin, golden_a2_dynamic_blended_ch3.bin,
#   golden_wavenet_a2_film_lite.bin, golden_wavenet_a2_film_full.bin
#   (Synthetic A2 dynamic/FiLM goldens — v1 only, generated from Python fixtures)
#   golden_linear_fft_rf320.bin, golden_linear_fft_rf2048.bin,
#   golden_linear_fft_rf4096.bin, golden_linear_fft_rf8192.bin
#   (Linear FFT partitioned convolution goldens — v1 + v2@48k)
#   golden_lstm_1x10.bin, golden_lstm_2x24.bin, golden_lstm_3x8.bin
#   (LSTM uncatalogued hidden sizes and 3-layer topology)
#   golden_convnet_nobn.bin, golden_convnet_relu.bin, golden_convnet_silu.bin
#   (ConvNet batchnorm and activation variants)
#   golden_linear_nobias.bin
#   (Linear without bias)
#
# These files must be committed so that the Rust golden vector tests
# run without C++ recompilation.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/namcore_render"
LOGS_DIR="$BUILD_DIR/logs"
MODELS_DIR="$SCRIPT_DIR/models"
FIXTURES_DIR="$SCRIPT_DIR"
mkdir -p "$LOGS_DIR"

# Load common utilities (phase helper, color vars, third-party resolution:
# THIRD_PARTY_DIR, NAM_CORE_DIR, NAM_PLUGIN_DIR, VARIABLES_ENV).
PHASE_TOTAL=13
source "$PROJECT_ROOT/utils/_lib.sh"

# Shell shared resolve — mirrors src/testing/fixtures.rs::model_path order.
# Usage: resolve_nam_model <filename>  →  echoes absolute path || return 1
resolve_nam_model() {
    local filename="$1"
    local p

    # (1) NAM_MODELS_DIR env var (explicit override)
    if [ -n "${NAM_MODELS_DIR:-}" ]; then
        p="$NAM_MODELS_DIR/$filename"
        if [ -f "$p" ]; then
            echo "$p"
            return 0
        fi
    fi

    # (2) third-party/community_models/ (respects NAM_THIRD_PARTY_DIR via _lib.sh)
    p="$THIRD_PARTY_DIR/community_models/$filename"
    if [ -f "$p" ]; then
        echo "$p"
        return 0
    fi

    # (3) tests/fixtures/models-nondist (local non-distributable override)
    p="$FIXTURES_DIR/models-nondist/$filename"
    if [ -f "$p" ]; then
        echo "$p"
        return 0
    fi

    # (4) tests/fixtures/models (default — distributed with the repository)
    p="$MODELS_DIR/$filename"
    if [ -f "$p" ]; then
        echo "$p"
        return 0
    fi

    return 1
}

# Load pinned versions from single source of truth (variables.env).
if [ ! -f "$VARIABLES_ENV" ]; then
    echo "ERROR: variables.env not found at $VARIABLES_ENV."
    echo "Expected at NeuralAmpModeler-rs root (override via NAM_VARIABLES_ENV)."
    exit 1
fi
source "$VARIABLES_ENV"

# =============================================================================
# Prerequisite checks
# =============================================================================
echo "=== Golden Vector Generator (NeuralAmpModelerCore) ==="

for cmd in cmake cargo python3; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '$cmd' not found. Install with: sudo apt install cmake cargo python3"
        exit 1
    fi
done

# Check C++20 compiler
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ &>/dev/null; then
        CXX=g++
    elif command -v clang++ &>/dev/null; then
        CXX=clang++
    else
        echo "ERROR: C++ compiler not found. Install g++ or clang++."
        exit 1
    fi
fi
echo "  C++ Compiler: $CXX"

# =============================================================================
# Ensure vendor mirrors (Core + Plugin) are present
# =============================================================================
phase "Ensuring third-party vendor mirrors..."
if ! ensure_third_party hard; then
    echo "ERROR: third-party vendor mirrors unavailable after setup."
    exit 1
fi

# =============================================================================
# Verify NeuralAmpModelerPlugin and dependencies
# =============================================================================
phase "Verifying NeuralAmpModelerPlugin (C++ IR reference)..."
if [ ! -d "$NAM_PLUGIN_DIR" ]; then
    echo "ERROR: NeuralAmpModelerPlugin not found at $NAM_PLUGIN_DIR."
    echo "Please run './utils/setup-third-party.sh' to download and setup dependencies."
    exit 1
fi

CURRENT_PLUGIN_SHA=$(cd "$NAM_PLUGIN_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_PLUGIN_SHA" != "$NAM_PLUGIN_COMMIT" ]; then
    echo "ERROR: NeuralAmpModelerPlugin version mismatch ($NAM_PLUGIN_TAG @ $NAM_PLUGIN_COMMIT expected, installed: $CURRENT_PLUGIN_SHA)."
    echo "Please run './utils/setup-third-party.sh' to synchronize dependencies."
    exit 1
fi

AUDIO_DSP_TOOLS_DIR="$NAM_PLUGIN_DIR/AudioDSPTools"
if [ ! -f "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" ] || [ ! -d "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen/Eigen" ]; then
    echo "ERROR: Submodules for NeuralAmpModelerPlugin are missing."
    echo "Please run './utils/setup-third-party.sh' to initialize submodules."
    exit 1
fi
echo "  NeuralAmpModelerPlugin verified ($NAM_PLUGIN_TAG @ $NAM_PLUGIN_COMMIT, submodules present)"

# =============================================================================
# Verify NeuralAmpModelerCore (standard)
# =============================================================================
phase "Verifying NeuralAmpModelerCore..."
if [ ! -d "$NAM_CORE_DIR" ]; then
    echo "ERROR: NeuralAmpModelerCore not found at $NAM_CORE_DIR."
    echo "Please run './utils/setup-third-party.sh' to download and setup dependencies."
    exit 1
fi

CURRENT_CORE_SHA=$(cd "$NAM_CORE_DIR" && git rev-parse HEAD 2>/dev/null || echo "unknown")
if [ "$CURRENT_CORE_SHA" != "$NAM_CORE_COMMIT" ]; then
    echo "ERROR: NeuralAmpModelerCore version mismatch ($NAM_CORE_TAG @ $NAM_CORE_COMMIT expected, installed: $CURRENT_CORE_SHA)."
    echo "Please run './utils/setup-third-party.sh' to synchronize dependencies."
    exit 1
fi

for sub in eigen AudioDSPTools; do
    sub_path="$NAM_CORE_DIR/Dependencies/$sub"
    if [ ! -d "$sub_path" ] || [ -z "$(ls -A "$sub_path" 2>/dev/null)" ]; then
        echo "ERROR: Submodule $sub is missing in NeuralAmpModelerCore."
        echo "Please run './utils/setup-third-party.sh' to initialize submodules."
        exit 1
    fi
done
echo "  NeuralAmpModelerCore verified ($NAM_CORE_TAG @ $NAM_CORE_COMMIT, submodules present)"

# =============================================================================
# Generate A2 dynamic/FiLM synthetic fixtures
# =============================================================================
phase "Generating A2 dynamic/FiLM fixtures (Python)..."
A2_FIXTURES_PY="$FIXTURES_DIR/generate_a2_fixtures.py"
if [ ! -f "$A2_FIXTURES_PY" ]; then
    echo "ERROR: generate_a2_fixtures.py not found at $A2_FIXTURES_PY"
    exit 1
fi
python3 "$A2_FIXTURES_PY"
echo "  A2 dynamic/FiLM .nam fixtures regenerated in $MODELS_DIR/"

# =============================================================================
# Generate synthetic topology fixtures (LSTM 1×10/2×24/3×8, ConvNet
# nobn/ReLU/SiLU, Linear nobias)
# =============================================================================
phase "Generating synthetic topology fixtures (Python)..."
S3_FIXTURES_PY="$FIXTURES_DIR/generate_fixtures.py"
if [ ! -f "$S3_FIXTURES_PY" ]; then
    echo "ERROR: generate_fixtures.py not found at $S3_FIXTURES_PY"
    exit 1
fi
python3 "$S3_FIXTURES_PY"
echo "  Synthetic .nam fixtures regenerated in $MODELS_DIR/"

# =============================================================================
# Build render tool (single unified binary at v0.5.4 with A2-fast)
# =============================================================================
# Delegates to the single entry point _lib.sh::ensure_namcore_render (S3-T01),
# shared with tests-quick.sh / tests-long.sh / tests/parity/cpp_parity.rs.
# Idempotent: skips cmake entirely when the binary is up-to-date. The vendor
# tree is never patched (read-only boundary); `-w` in the unified flags keeps
# the pinned NAMCore v0.5.4 compiling under GCC >= 15 without touching it.
phase "Building render tool..."
BUILD_TYPE="${BUILD_TYPE:-Release}"
export NAM_RENDER_BUILD_TYPE="$BUILD_TYPE"
if RENDER_BIN="$(ensure_namcore_render)"; then
    :
else
    RC_RENDER=$?
    echo "ERROR: ensure_namcore_render failed (exit=$RC_RENDER)."
    echo "  Diagnostics: $PROJECT_ROOT/target/logs/cmake-configure.log, $PROJECT_ROOT/target/logs/cmake-build.log"
    exit 1
fi
echo "  Render: $RENDER_BIN"

# =============================================================================
# Build Rust tools (gen_stress + wav_to_golden + nam_golden_catalog)
# =============================================================================
phase "Building Rust tools (gen_stress + wav_to_golden + nam_golden_catalog)..."

RUST_LOG="$LOGS_DIR/rust_build.log"
cargo build --release --features testing --bin gen_stress --bin wav_to_golden \
    --bin nam_golden_catalog > "$RUST_LOG" 2>&1 || {
    rust_status=$?
    tail -5 "$RUST_LOG"
    echo "ERROR: cargo build failed (exit=$rust_status). Full log: $RUST_LOG"
    exit 1
}
tail -3 "$RUST_LOG"
GEN_STRESS="$PROJECT_ROOT/target/release/gen_stress"
WAV_TO_GOLDEN="$PROJECT_ROOT/target/release/wav_to_golden"
GOLDEN_CATALOG_BIN="$PROJECT_ROOT/target/release/nam_golden_catalog"

if [ ! -f "$GEN_STRESS" ]; then
    echo "ERROR: Failed to build gen_stress binary."
    exit 1
fi
echo "  gen_stress: $GEN_STRESS"
echo "  wav_to_golden: $WAV_TO_GOLDEN"

# =============================================================================
# Golden registry — loaded from Rust (single source of truth, Sprint S3-T02)
# =============================================================================
# The canonical model↔golden registry (39 entries) lives in
# src/testing/catalog.rs::GOLDEN_GEN_CATALOG — the former static bash catalog
# array was removed so model lists are never duplicated in shell scripts.
# `nam_golden_catalog emit-catalog` serializes it in the same line format the
# loops below already parse:
#
#   nam_file : golden_name : label : v2_scope[:skip_srs[:skip_reason]]
#     v2_scope ∈ {all, 48k_only, none}
#       all      — v2 multi-SR for all 5 sample rates (respecting skip_srs)
#       48k_only — v2 only at 48000 Hz (model declares expected_sample_rate=48000)
#       none     — no v2 golden generation for this model
#     skip_srs (optional) — sample rates NOT to generate in v2 (e.g. 192000)
#     skip_reason (optional) — if non-empty, skip model entirely in both v1 and v2
#     loops with an explanatory message. Also suppresses # EXPECTED: lines in the
#     freshness manifest (F-C9, Tarefa T3.2).
#
# Rationale for v2_scope=none (A2 dynamic/FiLM models):
#   The 4 dynamic/FiLM models (a2_dynamic_gated_ch8, a2_dynamic_blended_ch3,
#   wavenet_a2_film_lite, wavenet_a2_film_full) are intentionally v2_scope=none
#   for two independent technical reasons:
#
#   1. C++ upstream limitation: the a2_fast.cpp render path rejects FiLM-conditioned
#      models and falls back to the Eigen-based generic WaveNet engine. The generic
#      engine does not consistently support multi-sample-rate rendering for FiLM
#      architectures — attempting v2 multi-SR renders for these models would produce
#      unreliable (or rejected) C++ reference outputs.
#
#   2. Dynamic engine coverage is a superset: these models are routed through
#      WaveNetA2Dyn (the dynamic engine with native FiLM support) at test time.
#      The dynamic engine handles arbitrary free geometries — geometry variance
#      subsumes sample-rate variance in practice. Live multi-SR cross-validation
#      is exercised via cpp_parity (live C++ toolchain) for dynamic engines, and
#      the v1 golden at 48 kHz provides the essential committed cross-reference.
#      Generating v2 multi-SR goldens here would produce ~28 MB of binary files
#      without any Rust test consumer (golden_vectors v2 skips tests whose
#      corresponding catalog entry has v2_scope=none).
#
#   This rationale is the single source of truth — docs/testing.md and
#   tests/fixtures/README.md reference this comment rather than duplicating it.
echo "  Loading golden registry from Rust (src/testing/catalog.rs)..."
mapfile -t CATALOG < <("$GOLDEN_CATALOG_BIN" emit-catalog)
if [ ${#CATALOG[@]} -lt 15 ]; then
    echo "ERROR: nam_golden_catalog emit-catalog returned ${#CATALOG[@]} entries (< 15 expected)."
    exit 1
fi
echo "  Golden registry: ${#CATALOG[@]} entries loaded from Rust."

# =============================================================================
# Generate stress WAV signals
# =============================================================================
phase "Generating stress signals..."

STRESS_WAV="$FIXTURES_DIR/stress_signal.wav"
"$GEN_STRESS" --version v1 --output "$STRESS_WAV" 2>&1
echo "  v1: $STRESS_WAV"

echo "  Generating v2 multi-SR stress signals..."
V2_STRESS_WAVS=()
for sr in 44100 48000 88200 96000 192000; do
    v2_wav="$FIXTURES_DIR/stress_signal_v2_${sr}.wav"
    "$GEN_STRESS" --version v2 --sample-rate "$sr" --output "$v2_wav" 2>&1
    echo "    v2 @ ${sr} Hz: $v2_wav"
    V2_STRESS_WAVS+=("$sr:$v2_wav")
done

# =============================================================================
# Run render for each model → WAV output → .golden.bin
# =============================================================================
phase "Running render for each model (v1)..."

# $CATALOG is loaded above from src/testing/catalog.rs (Rust single source of
# truth, S3-T02). Models with skip_reason set are skipped cleanly in both v1
# and v2 loops (F-C9, Tarefa T3.2).

TEMP_DIR="$FIXTURES_DIR/.temp_golden"
mkdir -p "$TEMP_DIR"

for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file golden_name label v2_scope skip_srs skip_reason <<< "$entry"
    OUTPUT_WAV="$TEMP_DIR/${golden_name}.wav"
    GOLDEN_BIN="$FIXTURES_DIR/${golden_name}.bin"

    if [ -n "$skip_reason" ]; then
        echo "  SKIP: $label ($nam_file) — skip_reason=$skip_reason"
        continue
    fi

    if ! MODEL_PATH=$(resolve_nam_model "$nam_file"); then
        echo "  SKIP: $nam_file not found"
        continue
    fi

    echo "  Processing $label ($nam_file)..."

    TEMP_RENDER_LOG="$TEMP_DIR/${golden_name}_v1_render.log"
    render_status=0
    "$RENDER_BIN" "$MODEL_PATH" "$STRESS_WAV" "$OUTPUT_WAV" > "$TEMP_RENDER_LOG" 2>&1 || render_status=$?
    tail -1 "$TEMP_RENDER_LOG"
    cat "$TEMP_RENDER_LOG" >> "$LOGS_DIR/render_v1.log"
    rm -f "$TEMP_RENDER_LOG"
    set -o pipefail
    if [ "$render_status" -ne 0 ] || [ ! -f "$OUTPUT_WAV" ]; then
        echo "  ERROR: Render failed for $label (exit=$render_status). Full log: $LOGS_DIR/render_v1.log"
        continue
    fi

    # Convert WAV output → .golden.bin (Rust native replacement for Python block)
    "$WAV_TO_GOLDEN" \
        --input "$OUTPUT_WAV" \
        --reference "$STRESS_WAV" \
        --output "$GOLDEN_BIN" 2>&1

done

# =============================================================================
# Generate v2 multi-SR goldens (one per model × sample_rate)
# =============================================================================
phase "Generating v2 multi-SR golden vectors..."

# v2 iterates the same Rust-sourced $CATALOG (src/testing/catalog.rs).
# Models with v2_scope="none" are skipped entirely;
# v2_scope="48k_only" only produces the 48 kHz golden;
# v2_scope="all" generates all 5 sample rates respecting skip_srs.
#
# NOTE ON SAMPLE-RATE SKIPS DURING RENDER: models whose .nam declares
# `expected_sample_rate` (e.g. WaveNet Standard CH=16, Official, LSTM Official,
# A2-Full, A2-Lite — all 48 kHz) make the C++ render tool reject other SRs with
# "Input WAV sample rate (X) does not match model expected rate (48000 Hz)". The
# v2_scope="48k_only" tag prevents those rejections by only running 48 kHz.

for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file golden_name label v2_scope skip_srs skip_reason <<< "$entry"

    if [ -n "$skip_reason" ]; then
        echo "  SKIP v2: $label ($nam_file) — skip_reason=$skip_reason"
        continue
    fi

    if [ "$v2_scope" = "none" ]; then
        echo "  SKIP v2: $label ($nam_file) — v2_scope=none"
        continue
    fi

    if ! MODEL_PATH=$(resolve_nam_model "$nam_file"); then
        echo "  SKIP v2: $nam_file not found"
        continue
    fi

    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr v2_wav <<< "$sr_entry"

        if [ "$v2_scope" = "48k_only" ] && [ "$sr" -ne 48000 ]; then
            continue
        fi

        if [ -n "$skip_srs" ] && [[ ",${skip_srs}," == *",${sr},"* ]]; then
            echo "    $label @ ${sr} Hz (v2)... SKIP (excluded SR for this model)"
            continue
        fi

        v2_golden="$FIXTURES_DIR/${golden_name}_v2_${sr}.bin"
        v2_out_wav="$TEMP_DIR/${golden_name}_v2_${sr}.wav"

        echo "    $label @ ${sr} Hz (v2)..."

        TEMP_RENDER_LOG="$TEMP_DIR/${golden_name}_v2_${sr}_render.log"
        set +o pipefail
        render_status=0
        "$RENDER_BIN" "$MODEL_PATH" "$v2_wav" "$v2_out_wav" > "$TEMP_RENDER_LOG" 2>&1 || render_status=$?
        tail -1 "$TEMP_RENDER_LOG"
        cat "$TEMP_RENDER_LOG" >> "$LOGS_DIR/render_v2.log"
        rm -f "$TEMP_RENDER_LOG"
        set -o pipefail
        if [ "$render_status" -ne 0 ] || [ ! -f "$v2_out_wav" ]; then
            echo "    SKIP: render failed for $label @ ${sr} Hz (likely SR mismatch in C++ tool). Full log: $LOGS_DIR/render_v2.log"
            continue
        fi

        "$WAV_TO_GOLDEN" \
            --input "$v2_out_wav" \
            --reference "$v2_wav" \
            --output "$v2_golden" 2>&1
    done
done

# =============================================================================
# Build and run C++ IR reference (dsp::ImpulseResponse) → golden_cabsim_cpp_*.bin
# =============================================================================
phase "Building C++ IR reference (dsp::ImpulseResponse)..."

AUDIO_DSP_TOOLS_DIR="$NAM_PLUGIN_DIR/AudioDSPTools"
IR_BIN="$FIXTURES_DIR/render_ir"

# Timestamp check: force rebuild if render_ir.cpp is newer than the cached binary.
# Prevents "phantom fix" bugs where source patches silently go unused.
if [ -f "$IR_BIN" ] && [ "$FIXTURES_DIR/render_ir.cpp" -nt "$IR_BIN" ]; then
    echo "  Source render_ir.cpp is newer than binary — forcing rebuild"
    rm -f "$IR_BIN"
fi

if [ -f "$IR_BIN" ]; then
    echo "  IR reference binary already exists: $IR_BIN"
else
    echo "  Compiling render_ir.cpp..."
    IR_LOG="$LOGS_DIR/render_ir_build.log"
    "$CXX" -std=c++17 -O2 \
        -I "$AUDIO_DSP_TOOLS_DIR" \
        -I "$AUDIO_DSP_TOOLS_DIR/Dependencies/eigen" \
        -I "$AUDIO_DSP_TOOLS_DIR/Dependencies/nlohmann" \
        -D "FIXTURES_DIR=\"$FIXTURES_DIR\"" \
        "$FIXTURES_DIR/render_ir.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/dsp.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/ImpulseResponse.cpp" \
        "$AUDIO_DSP_TOOLS_DIR/dsp/wav.cpp" \
        -o "$IR_BIN" \
        -lstdc++fs \
        > "$IR_LOG" 2>&1 || {
        ir_status=$?
        tail -5 "$IR_LOG"
        echo "ERROR: Failed to build render_ir binary (exit=$ir_status). Full log: $IR_LOG"
        exit 1
    }
    tail -5 "$IR_LOG"

    if [ ! -f "$IR_BIN" ]; then
        echo "  ERROR: Failed to build render_ir binary."
        echo "  Check that the g++ compiler and Eigen headers are available."
        exit 1
    fi
fi

echo "  Running render_ir to generate C++ IR golden vectors..."
"$IR_BIN"

# =============================================================================
# Cleanup
# =============================================================================
phase "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Golden vectors generated successfully ==="
echo "  v1 files at $FIXTURES_DIR/:"
for entry in "${CATALOG[@]}"; do
    IFS=':' read -r _ golden_name _ ___ ___ <<< "$entry"
    [ -f "$FIXTURES_DIR/${golden_name}.bin" ] && echo "    ${golden_name}.bin"
done
for cpp_file in golden_cabsim_cpp_short.bin golden_cabsim_cpp_medium.bin \
                 golden_cabsim_cpp_long.bin; do
    [ -f "$FIXTURES_DIR/$cpp_file" ] && echo "    $cpp_file"
done
echo "  v2 multi-SR files at $FIXTURES_DIR/:"
for entry in "${CATALOG[@]}"; do
    IFS=':' read -r _ golden_name label v2_scope ___ <<< "$entry"
    count=0
    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr _ <<< "$sr_entry"
        v2_file="$FIXTURES_DIR/${golden_name}_v2_${sr}.bin"
        if [ -f "$v2_file" ]; then
            count=$((count + 1))
        fi
    done
    if [ "$count" -gt 0 ]; then
        echo "    ${golden_name}_v2_*.bin  ($count sample rates) — $label"
    fi
done
# =============================================================================
# Generate freshness manifest (.nam ↔ golden)
# =============================================================================
phase "Generating freshness manifest..."

MANIFEST="$FIXTURES_DIR/.golden_manifest.sha256"
echo "# Golden freshness manifest — auto-generated by golden_gen_build.sh" > "$MANIFEST"
echo "# Format: sha256(model.nam) sha256(golden.bin) model_filename golden_filename" >> "$MANIFEST"
echo "# Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$MANIFEST"

# ── Toolchain fingerprint (F-I4 / Tarefa 3.2) ──
# Collect compiler, glibc, cmake, flags and OS info for cross-machine
# drift diagnosis.  Divergence emits a warning at test time but does NOT
# block the suite — the fingerprint is informational, not authoritative.
CXX_VER=$("$CXX" --version 2>/dev/null | head -1 || echo "unknown")
CMAKE_VER=$(cmake --version 2>/dev/null | head -1 || echo "unknown")
if GLIBC_VER=$(ldd --version 2>/dev/null | head -1); then
    :  # ldd worked
else
    GLIBC_VER=$(getconf GNU_LIBC_VERSION 2>/dev/null || echo "unknown")
fi
OS_INFO=$(uname -r 2>/dev/null || echo "unknown")
CXX_FLAGS_USED="-w -fno-fast-math -ffp-contract=off"                     # see §build → CMAKE_CXX_FLAGS
TARGET_FLAGS_USED="-O3"                                                  # -Ofast replaced by sed in vendorized CMakeLists.txt
echo "# TOOLCHAIN: cxx: $CXX_VER" >> "$MANIFEST"
echo "# TOOLCHAIN: cmake: $CMAKE_VER" >> "$MANIFEST"
echo "# TOOLCHAIN: glibc: $GLIBC_VER" >> "$MANIFEST"
echo "# TOOLCHAIN: os: $OS_INFO" >> "$MANIFEST"
echo "# TOOLCHAIN: cxx-flags: $CXX_FLAGS_USED $TARGET_FLAGS_USED" >> "$MANIFEST"

# ── v1 goldens ──
for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file golden_name label v2_scope skip_srs skip_reason <<< "$entry"
    MODEL_PATH="$MODELS_DIR/$nam_file"
    GOLDEN_PATH="$FIXTURES_DIR/${golden_name}.bin"
    if [ -f "$MODEL_PATH" ] && [ -f "$GOLDEN_PATH" ]; then
        MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
        GOLDEN_SHA=$(sha256sum "$GOLDEN_PATH" | cut -d' ' -f1)
        echo "$MODEL_SHA $GOLDEN_SHA $nam_file ${golden_name}.bin" >> "$MANIFEST"
    fi
done

# ── v2 multi-SR goldens ──
for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file golden_name label v2_scope skip_srs skip_reason <<< "$entry"
    if [ "$v2_scope" = "none" ]; then
        continue
    fi
    MODEL_PATH="$MODELS_DIR/$nam_file"
    if [ ! -f "$MODEL_PATH" ]; then
        continue
    fi
    MODEL_SHA=$(sha256sum "$MODEL_PATH" | cut -d' ' -f1)
    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr v2_wav <<< "$sr_entry"
        if [ "$v2_scope" = "48k_only" ] && [ "$sr" -ne 48000 ]; then
            continue
        fi
        if [ -n "$skip_srs" ] && [[ ",${skip_srs}," == *",${sr},"* ]]; then
            continue
        fi
        v2_golden="$FIXTURES_DIR/${golden_name}_v2_${sr}.bin"
        if [ -f "$v2_golden" ]; then
            GOLDEN_SHA=$(sha256sum "$v2_golden" | cut -d' ' -f1)
            echo "$MODEL_SHA $GOLDEN_SHA $nam_file ${golden_name}_v2_${sr}.bin" >> "$MANIFEST"
        fi
    done
done

# ── EXPECTED golden files (Freshness Gate, F-C9 / Tarefa T3.2) ──
# Every file listed below MUST exist on disk. The check_freshness() function
# in utils/tests-quick.sh reads these lines and fails hard if any are missing.
# Models with skip_reason set are intentionally excluded — they are known
# incompatible, so no golden file is expected for them.
echo "" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"
echo "# EXPECTED golden files — every entry listed here MUST exist on disk." >> "$MANIFEST"
echo "# If a file is missing, run './tests/fixtures/golden_gen_build.sh' to regenerate." >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"

for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file golden_name label v2_scope skip_srs skip_reason <<< "$entry"
    if [ -n "$skip_reason" ]; then
        continue
    fi
    # v1 golden always expected
    echo "# EXPECTED: ${golden_name}.bin" >> "$MANIFEST"

    # v2 goldens
    if [ "$v2_scope" = "none" ]; then
        continue
    fi
    for sr_entry in "${V2_STRESS_WAVS[@]}"; do
        IFS=':' read -r sr v2_wav <<< "$sr_entry"
        if [ "$v2_scope" = "48k_only" ] && [ "$sr" -ne 48000 ]; then
            continue
        fi
        if [ -n "$skip_srs" ] && [[ ",${skip_srs}," == *",${sr},"* ]]; then
            continue
        fi
        echo "# EXPECTED: ${golden_name}_v2_${sr}.bin" >> "$MANIFEST"
    done
done

echo "  Freshness manifest: $MANIFEST"

# =============================================================================
# ── Fixture integrity (standalone data fixtures not tied to .nam models) ──
# F-X3 / Tarefa 3.3: expands manifest coverage to all synthetic fixture data
# so that generator changes can be detected and fixtures flagged stale.
# Format: sha256(fixture) fixture
# =============================================================================
echo "" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"
echo "# FIXTURES — standalone data fixtures (sha256 checksums)" >> "$MANIFEST"
echo "# Any mismatch between committed SHA and file-on-disk triggers a hard fail." >> "$MANIFEST"
echo "# Regenerate with: ./tests/fixtures/golden_gen_build.sh" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"

FIXTURE_FILES=(
    # cabsim C++ reference goldens (generated by render_ir.cpp)
    "golden_cabsim_cpp_short.bin"
    "golden_cabsim_cpp_medium.bin"
    "golden_cabsim_cpp_long.bin"
    # MR-STFT golden (generated by scripts/gen_mrstft_golden.py)
    "mrstft_golden.bin"
    # Resampler reference vectors (generated by generate_resampler_reference.py)
    "resampler_input_44100.f32"
    "resampler_input_48000.f32"
    "resampler_input_96000.f32"
    "resampler_ref_44100_to_48000.f32"
    "resampler_ref_48000_to_44100.f32"
    "resampler_ref_48000_to_96000.f32"
    "resampler_ref_96000_to_48000.f32"
    # EBU 3341/R128 test sequences (generated by generate_ebu_sequences.py)
    "ebu_3341_1_sine_m23.wav"
    "ebu_3341_7_sine_m33.wav"
    "ebu_3341_dyn_alternating.wav"
    "ebu_3341_sine_m18.wav"
    # Stress signals (generated by src/bin/gen_stress.rs)
    "stress_signal.wav"
    "stress_signal_v2_44100.wav"
    "stress_signal_v2_48000.wav"
    "stress_signal_v2_88200.wav"
    "stress_signal_v2_96000.wav"
    "stress_signal_v2_192000.wav"
    # Spectral fidelity baselines
    "spectral_fidelity_baseline.json"
)
# f64 anchors (generated by scripts/validate_oracle_f64.py)
for anchor in "$FIXTURES_DIR"/f64_anchors/*.bin; do
    [ -f "$anchor" ] || continue
    FIXTURE_FILES+=("f64_anchors/$(basename "$anchor")")
done

for fixture in "${FIXTURE_FILES[@]}"; do
    fixture_path="$FIXTURES_DIR/$fixture"
    if [ -f "$fixture_path" ]; then
        FIXTURE_SHA=$(sha256sum "$fixture_path" | cut -d' ' -f1)
        echo "$FIXTURE_SHA $fixture" >> "$MANIFEST"
    fi
done

# =============================================================================
# ── Generator integrity (scripts/builders that produce fixtures) ──
# F-X3 / Tarefa 3.3: hash the source code of every fixture generator.
# If a generator changes, the freshness gate warns that associated fixtures
# should be regenerated even if their own hashes still match.
# Format: sha256(generator) generator
# =============================================================================
echo "" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"
echo "# GENERATORS — fixture generator scripts (sha256 checksums)" >> "$MANIFEST"
echo "# If a generator hash differs from the committed SHA, its fixtures may be" >> "$MANIFEST"
echo "# stale — re-generate with: ./tests/fixtures/golden_gen_build.sh" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"

GENERATOR_FILES=(
    # Shell builder
    "tests/fixtures/golden_gen_build.sh"
    # C++ renderers
    "tests/fixtures/render_ir.cpp"
    # Python fixture generators
    "tests/fixtures/generate_a2_fixtures.py"
    "tests/fixtures/generate_b1_2_fixtures.py"
    "tests/fixtures/generate_ebu_sequences.py"
    "tests/fixtures/generate_resampler_reference.py"
    "tests/fixtures/scripts/gen_mrstft_golden.py"
    "tests/fixtures/scripts/validate_oracle_f64.py"
    # Rust binaries that produce fixtures
    "src/bin/gen_stress.rs"
    "src/bin/wav_to_golden.rs"
    # Model inspection/hashing tools
    "utils/check-model.py"
)

for gen_file in "${GENERATOR_FILES[@]}"; do
    gen_path="$PROJECT_ROOT/$gen_file"
    if [ -f "$gen_path" ]; then
        GEN_SHA=$(sha256sum "$gen_path" | cut -d' ' -f1)
        echo "$GEN_SHA $gen_file" >> "$MANIFEST"
    fi
done

# =============================================================================
# ── Model registry (all CATALOG models, including skip_reason) ──
# F-X4 / Tarefa 3.4: complete listing of every .nam model known to this
# build script.  Used by the reverse-check in check_freshness() to detect
# orphaned .nam files in models/ that are not registered here.
# =============================================================================
echo "" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"
echo "# MODEL-REGISTRY — every .nam model in the CATALOG (reverse-check reference)" >> "$MANIFEST"
echo "# =============================================================================" >> "$MANIFEST"
for entry in "${CATALOG[@]}"; do
    IFS=':' read -r nam_file _ __ ___ ___ <<< "$entry"
    echo "# MODEL-REGISTRY: $nam_file" >> "$MANIFEST"
done
# Test-only model fixtures (no direct CATALOG entry; used by container/slimming/structural tests)
EXTRA_MODELS=(
    "BossWN-lite.nam"
    "linear_test.nam"
    "mock_a2.nam"
    "slimmable_container.nam"
    "slimmable_wavenet.nam"
    "wavenet.nam"
    "wavenet_a2_container.nam"
)
for nam in "${EXTRA_MODELS[@]}"; do
    if [ -f "$MODELS_DIR/$nam" ]; then
        echo "# MODEL-REGISTRY: $nam" >> "$MANIFEST"
    fi
done

echo ""
echo "Commit these files so that the Rust golden vector tests work."
echo "v2 files are large (~18 MB per model across 5 SRs). Git LFS or strategic" 
echo "subset selection is recommended for repo size management."
