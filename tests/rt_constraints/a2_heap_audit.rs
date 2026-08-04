// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Test for WaveNetA2 Heap-Audit Coverage (RT-Safety).
//
//  Validates zero heap allocations on the A2 hot-path using CountingAllocator.

#[cfg(feature = "heap-audit")]
mod audit_tests {
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
}
