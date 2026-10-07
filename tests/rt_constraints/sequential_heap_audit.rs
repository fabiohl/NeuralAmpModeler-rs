// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Test for SequentialModel Heap-Audit Coverage (RT-Safety).
//
//  Mirrors C++ `test_sequential.cpp:191 test_sequential_process_is_realtime_
//  safe_after_warmup` (zero allocations after a warm process call) and
//  extends the invariant to the multichannel `process_raw` pointer-table hot
//  path (NC-2.2 geometries) and to the split chain-stabilization drain
//  (`prewarm_step`), which runs between audio callbacks on the RT thread.
//
//  Invariant: zero alloc/realloc/free on the chain hot path, observed with
//  `TrackingGuard` over the harness's counting global allocator. The smoke
//  test `audit_harness_counts_allocations` (harness root) certifies the
//  counting facility itself in every feature configuration.
//
//  Marked `#[ignore]` per the testing rules — runs exclusively in
//  `utils/tests-long.sh` phase 3 (`--release --features heap-audit -- --ignored`).

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use neural_amp_modeler_rs::loader::nam_json::{LinearImplementation, LinearTopology};
    use neural_amp_modeler_rs::models::linear::LinearModel;
    use neural_amp_modeler_rs::models::lstm::LstmModel1;
    use neural_amp_modeler_rs::models::sequential::SequentialModel;
    use neural_amp_modeler_rs::models::{NamModel, StaticModel};

    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };

    // =========================================================================
    // Sequential Helper — synthetic Linear stages (public API surface)
    // =========================================================================

    /// Builds a static Linear stage with one FIR kernel per channel pair
    /// (`max(in, out)` kernels, identity semantics for `[1.0, 0.0]` taps),
    /// bias-free so the audit holds a strictly bounded arithmetic surface.
    fn linear_stage(in_ch: usize, out_ch: usize, kernel: &[f32]) -> StaticModel {
        let rf = kernel.len();
        let fan = in_ch.max(out_ch);
        let mut weights = Vec::with_capacity(fan * rf);
        for _ in 0..fan {
            weights.extend_from_slice(kernel);
        }
        let topo = LinearTopology {
            in_channels: in_ch,
            out_channels: out_ch,
            receptive_field: rf,
            has_bias: false,
            implementation: LinearImplementation::Direct,
        };
        StaticModel::Linear(Box::new(
            LinearModel::new_with_topology(topo, weights, Vec::new())
                .expect("Linear stage allocation"),
        ))
    }

    /// Zero-weight LSTM stage: carries a non-zero chain stabilization budget
    /// (24000 samples at the LSTM's internal 48 kHz) so the split-drain
    /// audit exercises the recurrent feed path, not merely a zero-budget
    /// fast-out.
    fn lstm_stage() -> StaticModel {
        StaticModel::Lstm1x3(Box::new(LstmModel1::<3, 4, 12>::new()))
    }

    /// Two-stage mono chain mirroring the C++ realtime-safety fixture weights
    /// (`{0.5}` → `{-2.0}`, `test_sequential.cpp:192`).
    fn two_stage_mono_chain() -> Box<SequentialModel> {
        let stages = vec![linear_stage(1, 1, &[0.5]), linear_stage(1, 1, &[-2.0])];
        Box::new(SequentialModel::new(stages, vec![None, None], Some(48000.0)).expect("chain"))
    }

    /// Asserts the full zero-heap invariant (alloc/realloc/free) and fails
    /// with a readable message pointing at the audited path.
    fn assert_zero_heap(allocs: usize, reallocs: usize, deallocs: usize, label: &str) {
        assert_eq!(
            allocs, 0,
            "Heap allocations detected on {label} hot-path! count={allocs}"
        );
        assert_eq!(
            reallocs, 0,
            "Heap reallocations detected on {label} hot-path! count={reallocs}"
        );
        assert_eq!(
            deallocs, 0,
            "Heap deallocations detected on {label} hot-path! count={deallocs}"
        );
    }

    /// Deterministic pseudo-audio block (stable across runs/platforms).
    fn test_input(num_samples: usize) -> Vec<f32> {
        (0..num_samples)
            .map(|i| (i as f32 * 0.017_3).sin() * 0.5)
            .collect()
    }

    // =========================================================================
    // Audit Tests
    // =========================================================================

    /// Mirrors test_sequential.cpp:191 `test_sequential_process_is_realtime_
    /// safe_after_warmup`: one warm call first (the C++ notes Linear lazily
    /// initializes an output buffer on its first process; the Rust chain has
    /// an analogous first-touch casing on the interior plane memory), then a
    /// sustained mixed-block audit window with the tracking guard active.
    #[test]
    #[ignore]
    fn test_sequential_process_heap_audit_after_warmup() {
        let mut chain = two_stage_mono_chain();
        chain.reset(48000, 64).expect("chain reset at 64 frames");

        let block_sizes = [1usize, 7, 32, 64];
        let inputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| test_input(bs)).collect();
        let mut outputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();

        // Warmup: boot the full chain path once per block size.
        for (bi, input) in inputs.iter().enumerate() {
            chain.process(input, &mut outputs[bi]);
        }
        for &o in outputs.iter().flatten() {
            assert!(o.is_finite(), "identity-weighted chain must stay finite");
        }

        // Audit — mixed-block scheduling (1..64 frames) on the hot path.
        let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
        let (allocs, reallocs, deallocs) = {
            let _guard = TrackingGuard::new();
            for _ in 0..iters {
                for (bi, input) in inputs.iter().enumerate() {
                    chain.process(
                        std::hint::black_box(input),
                        std::hint::black_box(&mut outputs[bi]),
                    );
                }
            }
            (get_alloc_count(), get_realloc_count(), get_dealloc_count())
        };
        assert_zero_heap(allocs, reallocs, deallocs, "Sequential mono process");
    }

    /// Extends the invariant to the multichannel pointer-table path (C++ ping-
    /// pong over `_stage_buffer_ptrs`): an asymmetric `1 -> 2 -> 2 -> 1` chain
    /// (NC-2.2 geometries) driven through `process_raw` with stack channel
    /// arrays — the exact invocation form the RT host uses for multichannel
    /// chains — over boundary planes sized by each stage's output channels.
    #[test]
    #[ignore]
    fn test_sequential_process_raw_multichannel_heap_audit() {
        let stages = vec![
            linear_stage(1, 2, &[0.75, -0.125]),
            linear_stage(2, 2, &[1.0, 0.5]),
            linear_stage(2, 1, &[-2.0]),
        ];
        let mut chain =
            SequentialModel::new(stages, vec![None, None, None], Some(48000.0)).expect("chain");
        chain.reset(48000, 64).expect("chain reset at 64 frames");

        // The chain face is mono (stage 0 in = 1 channel, last stage out = 1
        // channel): the 2-channel interior hops happen inside the chain over
        // the prebuilt boundary plane pointer tables.
        let input_mono = test_input(64);
        let mut output = vec![0.0f32; 64];
        let in_arr: [*const f32; 1] = [input_mono.as_ptr()];
        let out_arr: [*mut f32; 1] = [output.as_mut_ptr()];

        // Warmup over preallocated pointer tables.
        // SAFETY: `in_arr`/`out_arr` provide the chain's `in_channels`/`out_
        // channels` (1/1) valid borrows of 64 elements each (prewarm-reset
        // contract of `process_raw`).
        unsafe { chain.process_raw(in_arr.as_ptr(), out_arr.as_ptr(), 64) };

        // Audit — the entire chain hot path: stage 0 reads the caller plane,
        // interior stages run over the prebuilt boundary pointer tables, the
        // last stage writes the caller output directly.
        let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
        let (allocs, reallocs, deallocs) = {
            let _guard = TrackingGuard::new();
            for _ in 0..iters {
                // SAFETY: same pointer-table contract as the warm call.
                unsafe {
                    chain.process_raw(in_arr.as_ptr(), out_arr.as_ptr(), 64);
                }
            }
            (get_alloc_count(), get_realloc_count(), get_dealloc_count())
        };
        for &o in output.iter() {
            assert!(o.is_finite(), "multichannel chain must stay finite");
        }
        assert_zero_heap(
            allocs,
            reallocs,
            deallocs,
            "Sequential multichannel process_raw",
        );
    }

    /// Audits the split chain-stabilization drain: `prewarm_reset` disarms
    /// every child and arms the saturating sum; `prewarm_step` executes the
    /// zero-feed chain pass between audio callbacks on the RT thread — it
    /// must drain strictly over preallocated planes with zero heap traffic.
    #[test]
    #[ignore]
    fn test_sequential_prewarm_step_split_heap_audit() {
        // LSTM child (24000-sample budget at 48 kHz) + FIR tail: the chain
        // budget is non-zero and the feed exercises the recurrent stage.
        let stages = vec![lstm_stage(), linear_stage(1, 1, &[0.5, -0.25])];
        let mut chain =
            SequentialModel::new(stages, vec![None, None], Some(48000.0)).expect("chain");
        assert_eq!(chain.prewarm_samples(), 24000usize, "LSTM child budget");

        // Warmup (off-guard): arms the split flow and passes the first steps.
        chain.prewarm_reset();
        assert!(!chain.prewarm_complete(), "split stabilization armed");
        let mut drained = chain.prewarm_step(64);
        for _ in 0..371 {
            drained = std::hint::black_box(chain.prewarm_step(std::hint::black_box(64)));
        }
        assert!(drained < 24000, "drain must be mid-flight after warmup");

        // Audit — the remaining RT-side drain in irregular chunks.
        let chunk_cycle = [1usize, 33, 64, 4096, 2048, 777];
        let (allocs, reallocs, deallocs) = {
            let _guard = TrackingGuard::new();
            let mut chunk_index = 0usize;
            while !chain.prewarm_complete() {
                chain.prewarm_step(chunk_cycle[chunk_index % chunk_cycle.len()]);
                chunk_index += 1;
            }
            (get_alloc_count(), get_realloc_count(), get_dealloc_count())
        };
        assert!(chain.prewarm_complete(), "split drain must complete");
        assert_zero_heap(allocs, reallocs, deallocs, "Sequential split prewarm drain");
    }
}
