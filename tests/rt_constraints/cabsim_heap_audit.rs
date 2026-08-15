// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Test for IR Cabsim Heap-Audit Coverage (RT-Safety).
//
//  Validates zero heap allocations on the UPOLS convolution hot-path
//  using CountingAllocator.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine;

    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    // =============================================================================
    // Synthetic IR helpers
    // =============================================================================

    fn synth_ir(len: usize, freq: f32, decay: f32) -> Vec<f32> {
        let sample_rate = 48000u32;
        (0..len)
            .map(|n| {
                let t = n as f32 / sample_rate as f32;
                (std::f32::consts::TAU * freq * t).sin() * (-decay * t).exp()
            })
            .collect()
    }

    // =============================================================================
    // Audit
    // =============================================================================

    fn run_cabsim_audit(ir_len: usize, partition_size: usize, label: &str) {
        let ir = synth_ir(ir_len, 440.0, 10.0);

        let mut engine = ConvEngine::new(&ir, partition_size)
            .expect("construction should succeed for test-sized buffers");

        let iters = if cfg!(debug_assertions) { 100 } else { 2000 };
        let mut input = vec![0.0f32; partition_size];
        let mut output = vec![0.0f32; partition_size];

        // Warm-up: fill FDL
        for i in 0..engine.num_partitions().max(1) {
            for (j, v) in input.iter_mut().enumerate() {
                *v = ((i * partition_size + j) as f32 * 0.001).sin();
            }
            engine.process(&input, &mut output, None);
        }

        let count = {
            let _guard = TrackingGuard::new();
            for iter in 0..iters {
                for (j, v) in input.iter_mut().enumerate() {
                    *v = ((iter * partition_size + j) as f32 * 0.001).sin();
                }
                engine.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    None,
                );
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on cabsim-{label} hot-path! count={count}",
        );
    }

    #[test]
    fn test_cabsim_short_ir_heap_audit() {
        run_cabsim_audit(64, 64, "short");
    }

    #[test]
    fn test_cabsim_medium_ir_heap_audit() {
        run_cabsim_audit(512, 64, "medium");
    }

    #[test]
    fn test_cabsim_long_ir_heap_audit() {
        run_cabsim_audit(4096, 128, "long");
    }

    #[test]
    fn test_cabsim_passthrough_heap_audit() {
        let mut engine =
            ConvEngine::new(&[], 64).expect("construction should succeed for test-sized buffers");
        assert!(engine.is_passthrough());

        let iters = if cfg!(debug_assertions) { 100 } else { 2000 };
        let mut input = vec![0.0f32; 64];
        let mut output = vec![0.0f32; 64];

        for (j, v) in input.iter_mut().enumerate() {
            *v = (j as f32 * 0.01).sin();
        }
        engine.process(&input, &mut output, None);

        let count = {
            let _guard = TrackingGuard::new();
            for iter in 0..iters {
                for (j, v) in input.iter_mut().enumerate() {
                    *v = ((iter * 64 + j) as f32 * 0.01).sin();
                }
                engine.process(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                    None,
                );
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Heap allocations detected on cabsim-passthrough hot-path! count={count}",
        );
    }
}
