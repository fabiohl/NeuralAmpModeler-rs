// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.
//
// lanczos_probe.cpp — Offline comparison probe for the vendor Lanczos resampler (S12).
//
// Compiles against the READ-ONLY vendor mirrors and uses the REAL production
// classes — no replicated logic:
//   * `dsp::ResamplingContainer<T, 1, 12>` — exactly the plugin's resampling
//     wrapper (`NeuralAmpModeler.h:187`, with NAM_SAMPLE=double by default
//     since nothing defines NAM_SAMPLE_FLOAT), including Reset() silence
//     warm-up and per-block PassthroughFunc;
//   * `dsp::LanczosResampler<T, 1, 12>` — one isolated stage (per-stage view).
// The identical md5 of the vendor headers in both mirrors (NAMCore and plugin)
// is recorded in the .build_provenance receipt by utils/ensure_lanczos_probe.sh.
//
// Modes (raw f32 mono LE files):
//   single <in_rate> <out_rate> <in.f32> <out.f32> [float|double]
//   pair   <in_rate> <out_rate> <in.f32> <out.f32> [float|double]
//   bench  <in_rate> <out_rate> <in.f32> <reps>    [float|double]
//
// Machine-readable stdout (one token per line):
//   RATE_TYPE <T>
//   SAMPLES_OUT <n>                                    (single/pair)
//   LATENCY_IN <in-samples>                            (pair only, container.GetLatency())
//   STAGE_PAIR_NS_BEGIN / STAGE_NS <median> <min> / CONTAINER_NS <median> <min> / STAGE_PAIR_NS_END   (bench)
//
// The vendor LanczosResampler.h carries its own GPLv3-or-later/MIT dual-exception
// provenance; this probe is a measurement oracle, never linked into the Rust
// engine nor shipped to consumers — analogous to tools/render.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <ostream>

// Vendor headers reference these without providing them (their iPlug2/WDL
// platform includes are commented out upstream); supply them here so the
// read-only vendor tree stays untouched.
#define DEFAULT_BLOCK_SIZE 256
namespace iplug
{
constexpr double PI = 3.14159265358979323846;
} // namespace iplug

#include "ResamplingContainer.h"

namespace
{

constexpr size_t kChunk = 512; // plugin-callback-scale blocks
constexpr int kMaxBlock = 512; // Reset(blockSize) upper bound used here
constexpr int kDrainBlocks = 32;

std::vector<float> read_raw_f32(const char* path)
{
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "lanczos_probe: cannot open input %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes < 0 || bytes % 4 != 0)
    {
        std::fprintf(stderr, "lanczos_probe: invalid raw f32 size %ld\n", bytes);
        std::fclose(f);
        std::exit(2);
    }
    std::vector<float> data(static_cast<size_t>(bytes) / 4);
    const size_t got = std::fread(data.data(), 4, data.size(), f);
    std::fclose(f);
    if (got != data.size())
    {
        std::fprintf(stderr, "lanczos_probe: short read on %s\n", path);
        std::exit(2);
    }
    return data;
}

void write_raw_f32(const char* path, const std::vector<float>& data)
{
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "lanczos_probe: cannot open output %s\n", path);
        std::exit(2);
    }
    const size_t put = std::fwrite(data.data(), 4, data.size(), f);
    std::fclose(f);
    if (put != data.size())
    {
        std::fprintf(stderr, "lanczos_probe: short write on %s\n", path);
        std::exit(2);
    }
}

// One isolated vendor stage over the file: kChunk pushes interleaved with pops,
// per-block RenormalizePhases (as the container does), and a zero flush to
// expose the filter tail.
template <typename T_>
std::vector<float> run_single(dsp::LanczosResampler<T_, 1, 12>& r, const std::vector<float>& in)
{
    std::vector<float> out;
    T_ in_chunk[kChunk] = {static_cast<T_>(0.0f)};
    T_ out_chunk[kChunk] = {static_cast<T_>(0.0f)};
    T_* in_p[1] = {in_chunk};
    T_* out_p[1] = {out_chunk};

    size_t pushed = 0;
    int flushed = 0;
    bool flushing = false;

    while (true)
    {
        if (!flushing)
        {
            if (pushed < in.size())
            {
                const size_t take = std::min(kChunk, in.size() - pushed);
                for (size_t s = 0; s < take; ++s)
                {
                    in_chunk[s] = static_cast<T_>(in[pushed + s]);
                }
                r.PushBlock(in_p, take);
                pushed += take;
            }
            else
            {
                flushing = true;
            }
        }
        if (flushing && flushed < 2048)
        {
            for (size_t s = 0; s < kChunk; ++s)
            {
                in_chunk[s] = static_cast<T_>(0.0f);
            }
            const size_t take = std::min(kChunk, static_cast<size_t>(2048 - flushed));
            r.PushBlock(in_p, take);
            flushed += static_cast<int>(take);
        }

        const size_t got = r.PopBlock(out_p, kChunk);
        if (got > 0)
        {
            for (size_t s = 0; s < got; ++s)
            {
                out.push_back(static_cast<float>(out_chunk[s]));
            }
        }
        r.RenormalizePhases();

        if (flushing && flushed >= 2048 && got == 0)
        {
            break; // zero flush fully pushed and the pop condition is spent
        }
    }
    return out;
}

// The real plugin container with a passthrough encapsulated DSP: whole-file
// flow of `ProcessBlock` calls with callback-scale blocks, then zero-drain.
template <typename T_>
std::vector<float> run_pair(double in_rate, double out_rate, const std::vector<float>& in)
{
    dsp::ResamplingContainer<T_, 1, 12> container(out_rate);
    container.Reset(in_rate, kMaxBlock);
    std::printf("LATENCY_IN %d\n", container.GetLatency());
    std::vector<float> out;
    std::vector<T_> chunk_in(kChunk, static_cast<T_>(0.0f));
    std::vector<T_> chunk_out(kChunk, static_cast<T_>(0.0f));
    T_* ip[1] = {chunk_in.data()};
    T_* op[1] = {chunk_out.data()};

    auto passthrough = [](T_** i, T_** o, int n) {
        std::memcpy(o[0], i[0], static_cast<size_t>(n) * sizeof(T_));
    };

    size_t pushed = 0;
    while (pushed < in.size())
    {
        const size_t take = std::min(kChunk, in.size() - pushed);
        for (size_t s = 0; s < take; ++s)
        {
            chunk_in[s] = static_cast<T_>(in[pushed + s]);
        }
        container.ProcessBlock(ip, op, static_cast<int>(take), passthrough);
        for (size_t s = 0; s < take; ++s)
        {
            out.push_back(static_cast<float>(chunk_out[s]));
        }
        pushed += take;
    }
    // Offline tail: silence callbacks until the container's buffered audio is
    // fully flushed (the audio the plugin would receive in later callbacks);
    // the drained tail is discarded — the dumped stream is exactly the first
    // `in.size()` plugin-visible output samples.
    for (int b = 0; b < kDrainBlocks; ++b)
    {
        for (size_t s = 0; s < kChunk; ++s)
        {
            chunk_in[s] = static_cast<T_>(0.0f);
        }
        container.ProcessBlock(ip, op, static_cast<int>(kChunk), passthrough);
    }
    if (out.size() > in.size())
    {
        out.resize(in.size());
    }
    return out;
    return out;
}

double steady_ns()
{
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

// Median/min ns per output sample of one isolated stage over `reps`
// fresh-engine runs; construction and table init excluded from the window.
template <typename T_>
void bench_stage(double in_rate, double out_rate, const std::vector<float>& in, long reps)
{
    {
        dsp::LanczosResampler<T_, 1, 12> warm(out_rate > in_rate ? out_rate : in_rate,
                                              out_rate > in_rate ? in_rate : out_rate);
        (void)warm; // guarantees the shared kernel table is built once
    }
    std::vector<double> per_out_ns;
    per_out_ns.reserve(static_cast<size_t>(reps));
    for (long rep = 0; rep < reps; ++rep)
    {
        dsp::LanczosResampler<T_, 1, 12> r(in_rate, out_rate);
        std::vector<T_> chunk_in(kChunk, static_cast<T_>(0.0f));
        std::vector<T_> chunk_out(kChunk, static_cast<T_>(0.0f));
        T_* ip[1] = {chunk_in.data()};
        T_* op[1] = {chunk_out.data()};
        double out_samples = 0.0;
        size_t pushed = 0;
        const double t0 = steady_ns();
        while (true)
        {
            if (pushed < in.size())
            {
                const size_t take = std::min(kChunk, in.size() - pushed);
                for (size_t s = 0; s < take; ++s)
                {
                    chunk_in[s] = static_cast<T_>(in[pushed + s]);
                }
                r.PushBlock(ip, take);
                pushed += take;
                r.RenormalizePhases();
            }
            const size_t got = r.PopBlock(op, kChunk);
            out_samples += static_cast<double>(got);
            if (pushed >= in.size() && got == 0)
            {
                break;
            }
        }
        const double ns = (steady_ns() - t0) / out_samples;
        per_out_ns.push_back(ns);
    }
    std::sort(per_out_ns.begin(), per_out_ns.end());
    std::printf("STAGE_NS %.0f %.0f\n", per_out_ns[per_out_ns.size() / 2], per_out_ns.front());
}

// Full passthrough container cost (both stages + plumbing) in ns per output
// sample, over `reps` fresh-container runs.
template <typename T_>
void bench_container(double in_rate, double out_rate, const std::vector<float>& in, long reps)
{
    std::vector<double> per_out_ns;
    per_out_ns.reserve(static_cast<size_t>(reps));
    for (long rep = 0; rep < reps; ++rep)
    {
        dsp::ResamplingContainer<T_, 1, 12> container(out_rate);
        container.Reset(in_rate, kMaxBlock);
        std::vector<T_> chunk_in(kChunk, static_cast<T_>(0.0f));
        std::vector<T_> chunk_out(kChunk, static_cast<T_>(0.0f));
        T_* ip[1] = {chunk_in.data()};
        T_* op[1] = {chunk_out.data()};
        auto passthrough = [](T_** i, T_** o, int n) {
            std::memcpy(o[0], i[0], static_cast<size_t>(n) * sizeof(T_));
        };
        size_t pushed = 0;
        const double t0 = steady_ns();
        while (true)
        {
            if (pushed < in.size())
            {
                const size_t take = std::min(kChunk, in.size() - pushed);
                for (size_t s = 0; s < take; ++s)
                {
                    chunk_in[s] = static_cast<T_>(in[pushed + s]);
                }
                container.ProcessBlock(ip, op, static_cast<int>(take), passthrough);
                pushed += take;
            }
            if (pushed >= in.size())
            {
                break;
            }
        }
        // Each ProcessBlock yields exactly its nFrames of output, so the
        // output count equals pushed samples; report ns per external sample.
        const double ns = (steady_ns() - t0) / static_cast<double>(in.size());
        per_out_ns.push_back(ns);
    }
    std::sort(per_out_ns.begin(), per_out_ns.end());
    std::printf("CONTAINER_NS %.0f %.0f\n", per_out_ns[per_out_ns.size() / 2],
                per_out_ns.front());
}

void usage()
{
    std::fprintf(stderr,
                 "usage: lanczos_probe single|pair <in_rate> <out_rate> <in.f32> <out.f32> "
                 "[float|double]\n"
                 "       lanczos_probe bench <in_rate> <out_rate> <in.f32> <reps> "
                 "[float|double]\n");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        usage();
        return 2;
    }
    const std::string mode = argv[1];

    if (mode == "single" || mode == "pair")
    {
        if (argc < 6 || argc > 7)
        {
            usage();
            return 2;
        }
        const double in_rate = std::atof(argv[2]);
        const double out_rate = std::atof(argv[3]);
        const bool use_float = argc == 7 && std::string(argv[6]) == "float";
        std::printf("RATE_TYPE %s\n", use_float ? "float" : "double");
        std::vector<float> in = read_raw_f32(argv[4]);
        if (in.empty())
        {
            std::fprintf(stderr, "lanczos_probe: empty input\n");
            return 2;
        }
        if (mode == "single")
        {
            std::vector<float> out;
            if (use_float)
            {
                dsp::LanczosResampler<float, 1, 12> r(in_rate, out_rate);
                out = run_single(r, in);
            }
            else
            {
                dsp::LanczosResampler<double, 1, 12> r(in_rate, out_rate);
                out = run_single(r, in);
            }
            std::printf("SAMPLES_OUT %zu\n", out.size());
            write_raw_f32(argv[5], out);
        }
        else
        {
            std::vector<float> out = use_float ? run_pair<float>(in_rate, out_rate, in)
                                               : run_pair<double>(in_rate, out_rate, in);
            std::printf("SAMPLES_OUT %zu\n", out.size());
            write_raw_f32(argv[5], out);
        }
        return 0;
    }

    if (mode == "bench")
    {
        if (argc < 6 || argc > 7)
        {
            usage();
            return 2;
        }
        const double in_rate = std::atof(argv[2]);
        const double out_rate = std::atof(argv[3]);
        const long reps = std::atol(argv[5]);
        const bool use_float = argc == 7 && std::string(argv[6]) == "float";
        std::printf("RATE_TYPE %s\n", use_float ? "float" : "double");
        if (reps < 1 || reps > 100000)
        {
            std::fprintf(stderr, "lanczos_probe: reps out of range\n");
            return 2;
        }
        std::vector<float> in = read_raw_f32(argv[4]);
        if (in.empty())
        {
            std::fprintf(stderr, "lanczos_probe: empty input\n");
            return 2;
        }
        std::printf("STAGE_PAIR_NS_BEGIN\n");
        if (use_float)
        {
            bench_stage<float>(in_rate, out_rate, in, reps);
            bench_container<float>(in_rate, out_rate, in, reps);
        }
        else
        {
            bench_stage<double>(in_rate, out_rate, in, reps);
            bench_container<double>(in_rate, out_rate, in, reps);
        }
        std::printf("STAGE_PAIR_NS_END\n");
        return 0;
    }

    usage();
    return 2;
}
