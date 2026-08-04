// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// render_ir.cpp — C++ reference binary for IR cabsim cross-validation.
//
// Generates synthetic IRs and input signals deterministically using a PCG PRNG
// (bit-identical to the Rust SimplePcg in tests/cabsim_golden.rs), then
// convolves them via dsp::ImpulseResponse (AudioDSPTools) and emits golden
// vectors in the binary format:
//   [u32 N LE] [f32×N in LE] [f32×N out LE]
//
// Scenarios: short (seed=42), medium (seed=137), long (seed=31337).
// A theoretical "stress" scenario with 32768-sample IR is deliberately not
// implemented: dsp::ImpulseResponse caps impulse response length at 8192
// samples (mMaxLength) and would silently truncate, making cross-reference
// validation against C++ meaningless. See tests/cabsim_cpp_parity.rs.
//
// Build: see the "Building C++ IR reference" section of golden_gen_build.sh
// (referenced by name, not step number, to survive renumbering).

#include "dsp/ImpulseResponse.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// ── Simple PCG PRNG — bit-identical to Rust tests/cabsim_golden.rs ────────

struct SimplePcg {
  uint64_t state;
  uint64_t inc;

  explicit SimplePcg(uint64_t seed) {
    state = seed + 1442695040888963407ULL;
    inc = 1;
  }

  float next_f32() {
    uint64_t old_state = state;
    state = old_state * 6364136223846793005ULL + (inc | 1);
    uint32_t xorshifted =
        static_cast<uint32_t>(((old_state >> 18) ^ old_state) >> 27);
    uint32_t rot = static_cast<uint32_t>(old_state >> 59);
    uint32_t rotated;
    if (rot == 0) {
      rotated = xorshifted;
    } else {
      rotated = (xorshifted >> rot) | (xorshifted << (32 - rot));
    }
    return static_cast<float>(static_cast<double>(rotated) * 2.3283064e-10);
  }

  float next_f32_signed() { return next_f32() * 2.0f - 1.0f; }
};

// ── Synthetic IR and signal generators — same formulas as cabsim_golden.rs ─

std::vector<float> synth_ir_deterministic(size_t len, float freq, float decay,
                                          SimplePcg &rng) {
  const float noise_level = 0.02f;
  const float sample_rate = 48000.0f;
  std::vector<float> ir(len);
  for (size_t n = 0; n < len; ++n) {
    float t = static_cast<float>(n) / sample_rate;
    ir[n] =
        std::sin(6.283185307179586f * freq * t) * std::exp(-decay * t) +
        noise_level * rng.next_f32_signed();
  }
  return ir;
}

std::vector<float> synth_signal_deterministic(size_t len, SimplePcg &rng) {
  std::vector<float> sig(len);
  for (size_t i = 0; i < len; ++i) {
    float t = static_cast<float>(i) / 48000.0f;
    sig[i] = 0.7f * std::sin(6.283185307179586f * 220.0f * t) +
             0.35f * std::sin(6.283185307179586f * 554.37f * t) +
             0.18f * std::sin(6.283185307179586f * 880.0f * t) +
             0.05f * rng.next_f32_signed();
  }
  return sig;
}

// ── Golden binary I/O ──────────────────────────────────────────────────────

void write_golden_binary(const std::string &path,
                         const std::vector<float> &input,
                         const std::vector<float> &output) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "ERROR: cannot open " << path << " for writing\n";
    std::exit(1);
  }

  uint32_t N = static_cast<uint32_t>(input.size());
  out.write(reinterpret_cast<const char *>(&N), sizeof(N));
  out.write(reinterpret_cast<const char *>(input.data()),
            input.size() * sizeof(float));
  out.write(reinterpret_cast<const char *>(output.data()),
            output.size() * sizeof(float));

  if (!out) {
    std::cerr << "ERROR: failed writing " << path << "\n";
    std::exit(1);
  }
}

// ── One scenario pipeline ──────────────────────────────────────────────────

void run_scenario(uint64_t seed, size_t ir_len, size_t sig_len, float freq,
                  float decay, const std::string &output_path) {
  SimplePcg rng(seed);

  std::vector<float> ir = synth_ir_deterministic(ir_len, freq, decay, rng);
  std::vector<float> signal = synth_signal_deterministic(sig_len, rng);

  dsp::ImpulseResponse::IRData irData;
  irData.mRawAudio = std::move(ir);
  irData.mRawAudioSampleRate = 48000.0;

  dsp::ImpulseResponse irProc(irData, 48000.0);

  std::vector<double> input_double(signal.begin(), signal.end());
  double *in_ptr = input_double.data();
  double **inputs = &in_ptr;

  double **outputs = irProc.Process(inputs, 1, sig_len);

  std::vector<float> output_float(sig_len);
  for (size_t i = 0; i < sig_len; ++i) {
    output_float[i] = static_cast<float>(outputs[0][i]);
  }

  write_golden_binary(output_path, signal, output_float);
  std::cout << "  " << output_path << " (" << sig_len << " samples)"
            << std::endl;
}

// ── Main ───────────────────────────────────────────────────────────────────

int main() {
  const std::string fixtures_dir =
#ifdef FIXTURES_DIR
      FIXTURES_DIR;
#else
      ".";
#endif

  std::cout << "=== C++ IR Reference Golden Generator ===" << std::endl;

  run_scenario(42, 64, 256, 600.0f, 12.0f,
               fixtures_dir + "/golden_cabsim_cpp_short.bin");
  run_scenario(137, 512, 1024, 350.0f, 6.0f,
               fixtures_dir + "/golden_cabsim_cpp_medium.bin");
  run_scenario(31337, 8192, 16384, 200.0f, 2.0f,
               fixtures_dir + "/golden_cabsim_cpp_long.bin");
  std::cout << "=== Done ===" << std::endl;
  return 0;
}
