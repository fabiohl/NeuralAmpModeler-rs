// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// render_multichannel.cpp — C++ reference binary for multichannel NAM model cross-validation.
//
// Reads a multi-channel binary input vector, executes nam::DSP::process() in C++,
// and writes the multi-channel output vector in the binary format:
//   [u32 channels LE] [u32 frames LE] [f32×frames ch0 LE] [f32×frames ch1 LE] ...

#include "NAM/dsp.h"
#include "NAM/get_dsp.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: render_multichannel <model.nam> <input.bin> <output.bin> [--in-place] [--chunk <size>] [--irregular]\n";
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string input_path = argv[2];
  const std::string output_path = argv[3];

  bool in_place = false;
  int chunk_size = 64;
  bool irregular = false;

  for (int i = 4; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "--in-place")
    {
      in_place = true;
    }
    else if (arg == "--irregular")
    {
      irregular = true;
    }
    else if (arg == "--chunk" && i + 1 < argc)
    {
      chunk_size = std::atoi(argv[++i]);
    }
  }

  // 1. Load model via NAMCore
  auto model = nam::get_dsp(std::filesystem::path(model_path));
  if (!model)
  {
    std::cerr << "Error: failed to load model [" << model_path << "]\n";
    return 1;
  }

  const int model_in_channels = model->NumInputChannels();
  const int model_out_channels = model->NumOutputChannels();

  // 2. Read binary input: [u32 channels LE] [u32 frames LE] [f32*frames per channel]
  std::ifstream in_file(input_path, std::ios::binary);
  if (!in_file)
  {
    std::cerr << "Error: cannot open input file [" << input_path << "]\n";
    return 1;
  }

  uint32_t file_channels = 0, file_frames = 0;
  in_file.read(reinterpret_cast<char*>(&file_channels), 4);
  in_file.read(reinterpret_cast<char*>(&file_frames), 4);

  if (static_cast<int>(file_channels) != model_in_channels)
  {
    std::cerr << "Error: input channel mismatch (file: " << file_channels
              << ", model: " << model_in_channels << ")\n";
    return 1;
  }

  std::vector<std::vector<float>> input_f32(file_channels, std::vector<float>(file_frames));
  for (uint32_t ch = 0; ch < file_channels; ++ch)
  {
    in_file.read(reinterpret_cast<char*>(input_f32[ch].data()), file_frames * sizeof(float));
  }
  in_file.close();

  // Convert input to NAM_SAMPLE
  std::vector<std::vector<NAM_SAMPLE>> input_audio(file_channels, std::vector<NAM_SAMPLE>(file_frames));
  for (uint32_t ch = 0; ch < file_channels; ++ch)
  {
    for (size_t i = 0; i < file_frames; ++i)
    {
      input_audio[ch][i] = static_cast<NAM_SAMPLE>(input_f32[ch][i]);
    }
  }

  // 3. Prepare output buffers
  std::vector<std::vector<NAM_SAMPLE>> output_audio(model_out_channels, std::vector<NAM_SAMPLE>(file_frames, 0.0));

  // 4. Configure DSP model
  const double sample_rate = model->GetExpectedSampleRate() > 0 ? model->GetExpectedSampleRate() : 48000.0;
  model->SetPrewarmOnReset(false);
  model->Reset(sample_rate, 127);

  // 5. Process in chunks
  if (in_place)
  {
    if (model_in_channels != model_out_channels)
    {
      std::cerr << "Error: --in-place requires in_channels == out_channels (got in="
                << model_in_channels << ", out=" << model_out_channels << ")\n";
      return 1;
    }

    std::vector<NAM_SAMPLE*> io_ptrs(model_in_channels);
    size_t offset = 0;
    while (offset < file_frames)
    {
      size_t count = irregular ? std::min<size_t>((offset % 127) + 1, file_frames - offset)
                               : std::min<size_t>(chunk_size, file_frames - offset);
      for (int ch = 0; ch < model_in_channels; ++ch)
      {
        io_ptrs[ch] = input_audio[ch].data() + offset;
      }
      model->process(io_ptrs.data(), io_ptrs.data(), static_cast<int>(count));
      offset += count;
    }
    // Copy in-place modified input audio to output_audio
    for (int ch = 0; ch < model_out_channels; ++ch)
    {
      output_audio[ch] = input_audio[ch];
    }
  }
  else
  {
    std::vector<NAM_SAMPLE*> in_ptrs(model_in_channels);
    std::vector<NAM_SAMPLE*> out_ptrs(model_out_channels);
    size_t offset = 0;
    while (offset < file_frames)
    {
      size_t count = irregular ? std::min<size_t>((offset % 127) + 1, file_frames - offset)
                               : std::min<size_t>(chunk_size, file_frames - offset);
      for (int ch = 0; ch < model_in_channels; ++ch)
      {
        in_ptrs[ch] = input_audio[ch].data() + offset;
      }
      for (int ch = 0; ch < model_out_channels; ++ch)
      {
        out_ptrs[ch] = output_audio[ch].data() + offset;
      }
      model->process(in_ptrs.data(), out_ptrs.data(), static_cast<int>(count));
      offset += count;
    }
  }

  // 6. Write binary output: [u32 channels LE] [u32 frames LE] [f32*frames per channel]
  std::ofstream out_file(output_path, std::ios::binary);
  if (!out_file)
  {
    std::cerr << "Error: cannot open output file [" << output_path << "]\n";
    return 1;
  }

  uint32_t out_channels_u32 = static_cast<uint32_t>(model_out_channels);
  uint32_t out_frames_u32 = file_frames;
  out_file.write(reinterpret_cast<const char*>(&out_channels_u32), 4);
  out_file.write(reinterpret_cast<const char*>(&out_frames_u32), 4);

  std::vector<float> out_f32(file_frames);
  for (int ch = 0; ch < model_out_channels; ++ch)
  {
    for (size_t i = 0; i < file_frames; ++i)
    {
      out_f32[i] = static_cast<float>(output_audio[ch][i]);
    }
    out_file.write(reinterpret_cast<const char*>(out_f32.data()), file_frames * sizeof(float));
  }
  out_file.close();

  return 0;
}
