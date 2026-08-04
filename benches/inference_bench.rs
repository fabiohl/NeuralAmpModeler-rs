// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! End-to-end neural network inference latency benchmarks for the NAM-rs engine.
//!
//! Measures the processing time of 1 DSP block (64 samples at 48 kHz = 1.33 ms
//! deadline) for WaveNet, LSTM, A2, ConvNet, and Linear neural networks.
//!
//! ## Available benchmarks
//!
//! | ID                                      | Description                             | Practical context                        |
//! | --------------------------------------- | --------------------------------------- | ---------------------------------------- |
//! | `WaveNet_Standard_CH16_64samp_48kHz`    | Complete WaveNet Standard inference     | ~284 KB model, 10+10 dilated layers      |
//! | `LSTM_2x16_64samp_48kHz`                | LSTM 2 layers × 16 hidden inference     | Heaviest supported recurrent network     |
//! | `A2Full_CH8_64samp_48kHz`               | A2-Full (CH=8) inference                | AVX2 col-major T=4 broadcast-FMA         |
//! | `A2Lite_CH3_64samp_48kHz`               | A2-Lite (CH=3) inference                | u16 interleaved GEMV, CPU-efficient      |
//! | `WaveNet_Dynamic_CH5_64samp_48kHz`      | WaveNet Dynamic free-geom inference     | Fallback for non-cataloged WaveNet geom  |
//! | `LSTM_Dynamic_1x7_64samp_48kHz`         | LSTM Dynamic 1×7 inference              | Fallback for non-cataloged LSTM geom     |
//! | `ConvNet_Model_64samp_48kHz`            | ConvNet end-to-end model inference      | Full pipeline: 2 blocks CH=8→4 + head    |
//! | `A2Dyn_Gated_64samp_48kHz`              | A2 Dynamic CH=4 gated inference         | Fallback for non-cataloged A2 geom       |
//!
//! ## Interpreting the results
//!
//! - The real-time deadline at 48 kHz with a 64-sample buffer is **1.33 ms**.
//! - If any inference benchmark exceeds this deadline, the engine will cause
//!   xruns (buffer underruns) in production with this buffer size.
//!
//! ## Running
//!
//! ```sh
//! cargo bench --bench inference_bench
//! ```

mod common;

#[path = "inference/wavenet_bench.rs"]
mod wavenet;

#[path = "inference/lstm_bench.rs"]
mod lstm;

#[path = "inference/a2_bench.rs"]
mod a2;

#[path = "inference/misc_bench.rs"]
mod misc;

use criterion::{criterion_group, criterion_main};

criterion_group!(
    name = benches;
    config = criterion::Criterion::default().sample_size(50);
    targets = wavenet::bench_wavenet_standard_process,
    wavenet::bench_wavenet_p10_small_block_sizes,
    wavenet::bench_wavenet_standard_block_sizes,
    lstm::bench_lstm_2x16_process,
    lstm::bench_lstm_2x16_block_sizes,
    lstm::bench_lstm_1x8_comparison,
    lstm::bench_lstm_2x16_comparison,
    lstm::bench_lstm_1x40_process,
    lstm::bench_lstm_2x24_process,
    lstm::bench_lstm_1x40_comparison,
    lstm::bench_lstm_2x24_comparison,
    a2::bench_a2_full_process,
    a2::bench_a2_full_block_sizes,
    a2::bench_a2_lite_process,
    a2::bench_a2_lite_block_sizes,
    wavenet::bench_prewarm_wavenet_standard,
    lstm::bench_prewarm_lstm_2x16,
    a2::bench_prewarm_a2_full,
    a2::bench_prewarm_a2_lite,
    misc::bench_linear_model_dot_product,
    misc::bench_container_crossfade_64samp,
    wavenet::bench_wavenet_dynamic_process,
    misc::bench_lstm_dynamic_process,
    misc::bench_wavenet_a2_dyn_gated_process,
    wavenet::bench_wavenet_comparison,
    a2::bench_a2_comparison,
    misc::bench_nondist_models,
    misc::bench_convnet_model_process
);

criterion_main!(benches);
