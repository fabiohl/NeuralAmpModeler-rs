// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Test for WaveNetA2 Heap-Audit Coverage (RT-Safety).
//
//  Validates zero heap allocations on the A2 hot-path using CountingAllocator.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use neural_amp_modeler_rs::loader::dispatcher::build_model;
    use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
    use neural_amp_modeler_rs::models::NamModel;
    use neural_amp_modeler_rs::models::a2::{A2_KERNEL_SIZES, WaveNetA2, a2_weight_count};

    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    // =============================================================================
    // A2 Helper — Synthetic Weights
    // =============================================================================

    /// Assembles synthetic A2 weight stream.
    fn a2_synth_weights<const CH: usize>(weight_val: f32) -> Vec<f32> {
        let num_weights = a2_weight_count::<CH>();
        let mut w = Vec::with_capacity(num_weights);

        w.extend(std::iter::repeat_n(weight_val, CH));
        for &k in &A2_KERNEL_SIZES {
            w.extend(std::iter::repeat_n(weight_val, CH * CH * k));
            w.extend(std::iter::repeat_n(0.0f32, CH));
            w.extend(std::iter::repeat_n(weight_val, CH));
            w.extend(std::iter::repeat_n(weight_val, CH * CH));
            w.extend(std::iter::repeat_n(0.0f32, CH));
        }
        w.extend(std::iter::repeat_n(weight_val, 16 * CH));
        w.push(0.0);
        w.push(0.02);

        assert_eq!(w.len(), num_weights);
        w
    }

    /// Builds a synthetically-weighted A2 model.
    fn build_a2<const CH: usize>(weight_val: f32) -> WaveNetA2<CH> {
        let weights = a2_synth_weights::<CH>(weight_val);
        let mut model = WaveNetA2::<CH>::new().expect("Failed to create WaveNetA2");
        model.set_weights(&weights).expect("A2 set_weights failed");
        model
    }

    // =============================================================================
    // Audit Tests
    // =============================================================================

    fn run_a2_audit<const CH: usize>(label: &str) {
        let mut model = build_a2::<CH>(0.01);
        model.prewarm();

        let block_sizes = [1usize, 16, 32, 48, 64];

        // Pre-allocate buffers outside the audit guard
        let mut inputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();
        let mut outputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();

        let mut sample_offset = 0usize;
        for (bi, &block_size) in block_sizes.iter().enumerate() {
            for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                let t = (sample_offset + i) as f32;
                *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
            }
            model.process(&inputs[bi], &mut outputs[bi]);
            sample_offset += block_size;
        }

        // Audit — must have zero allocations on hot-path
        let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
        let count = {
            let _guard = TrackingGuard::new();
            for _ in 0..iters {
                for (bi, &block_size) in block_sizes.iter().enumerate() {
                    for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                        let t = (sample_offset + i) as f32;
                        *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
                    }
                    model.process(
                        std::hint::black_box(&inputs[bi]),
                        std::hint::black_box(&mut outputs[bi]),
                    );
                    sample_offset += block_size;
                }
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on A2-{label} hot-path! count={}",
            count
        );
    }

    #[test]
    fn test_a2_full_heap_audit() {
        run_a2_audit::<8>("Full");
    }

    #[test]
    fn test_a2_lite_heap_audit() {
        run_a2_audit::<3>("Lite");
    }

    /// Dynamic/cascade path heap-audit using `wavenet_a2_max.nam`.
    ///
    /// Exercises `WaveNetA2Dyn` (cascaded FiLM condition_dsp with groups > 1,
    /// head1x1, skip_last_residual=false) on the production model graph to certify
    /// zero heap allocations on the hot-path after prewarm.
    #[test]
    #[ignore]
    fn test_a2_dyn_max_heap_audit() {
        let path = crate::common::model_path("wavenet_a2_max.nam");
        if !path.exists() {
            println!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:wavenet_a2_max.nam\"");
            return;
        }
        let json_data = std::fs::read_to_string(&path).expect("Failed to read wavenet_a2_max.nam");
        let model_data = parse_nam_json(&json_data).expect("Failed to parse wavenet_a2_max.nam");
        let mut model = build_model(&model_data).expect("Dispatcher failed for wavenet_a2_max.nam");
        model.prewarm(2048);

        let block_sizes = [1usize, 16, 32, 48, 64];

        // Pre-allocate buffers outside the audit guard
        let mut inputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();
        let mut outputs: Vec<Vec<f32>> = block_sizes.iter().map(|&bs| vec![0.0f32; bs]).collect();

        let mut sample_offset = 0usize;
        for (bi, &block_size) in block_sizes.iter().enumerate() {
            for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                let t = (sample_offset + i) as f32;
                *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
            }
            model.process(&inputs[bi], &mut outputs[bi]);
            sample_offset += block_size;
        }

        // Audit — must have zero allocations on hot-path
        let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
        let count = {
            let _guard = TrackingGuard::new();
            for _ in 0..iters {
                for (bi, &block_size) in block_sizes.iter().enumerate() {
                    for (i, v) in inputs[bi].iter_mut().enumerate().take(block_size) {
                        let t = (sample_offset + i) as f32;
                        *v = (2.0 * std::f32::consts::PI * 440.0 * t / 48000.0).sin();
                    }
                    model.process(
                        std::hint::black_box(&inputs[bi]),
                        std::hint::black_box(&mut outputs[bi]),
                    );
                    sample_offset += block_size;
                }
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on A2-Max dynamic hot-path! count={}",
            count
        );
    }
}
