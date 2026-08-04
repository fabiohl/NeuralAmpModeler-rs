// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Integration Test for NamResampler Heap-Audit Coverage.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use neural_amp_modeler_rs::dsp::resampler::NamResampler;

    use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

    fn generate_noise(samples: usize) -> Vec<f32> {
        (0..samples).map(|i| (i as f32 * 0.1234).sin()).collect()
    }

    fn run_resampler_audit(host_rate: u32, nam_rate: u32, chunk_size: usize, is_mono: bool) {
        let mut resampler =
            NamResampler::new(host_rate, nam_rate, chunk_size).expect("Failed to create resampler");

        let in_l = generate_noise(chunk_size);
        let in_r = generate_noise(chunk_size);
        let mut out_l = vec![0.0f32; chunk_size * 4];
        let mut out_r = vec![0.0f32; chunk_size * 4];

        // Warmup
        for _ in 0..5 {
            if is_mono {
                resampler.process_input_mono(&in_l, &mut out_l, &mut out_r);
                resampler.process_output_mono(&in_l, &mut out_l, &mut out_r);
            } else {
                resampler.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
                resampler.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
            }
        }

        // Audit
        let count = {
            let _guard = TrackingGuard::new();
            let iters = if cfg!(debug_assertions) { 50 } else { 1000 };
            for _ in 0..iters {
                if is_mono {
                    resampler.process_input_mono(&in_l, &mut out_l, &mut out_r);
                    resampler.process_output_mono(&in_l, &mut out_l, &mut out_r);
                } else {
                    resampler.process_input(&in_l, &in_r, &mut out_l, &mut out_r);
                    resampler.process_output(&in_l, &in_r, &mut out_l, &mut out_r);
                }
            }
            get_alloc_count()
        };

        assert_eq!(
            count, 0,
            "Allocations detected during resampling! (host_rate={}, nam_rate={}, mono={})",
            host_rate, nam_rate, is_mono
        );
    }

    #[test]
    fn test_resampler_heap_audit_all_scenarios() {
        let scenarios = [
            // ratio 1:1 (passthrough)
            (48000, 48000, 32, true),
            (48000, 48000, 32, false),
            (48000, 48000, 1024, true),
            (48000, 48000, 1024, false),
            // 96k -> 48k (downsample)
            (96000, 48000, 32, true),
            (96000, 48000, 32, false),
            (96000, 48000, 1024, true),
            (96000, 48000, 1024, false),
            // 44.1k -> 48k (upsample)
            (44100, 48000, 32, true),
            (44100, 48000, 32, false),
            (44100, 48000, 1024, true),
            (44100, 48000, 1024, false),
        ];

        for &(pw, nam, chunk, is_mono) in &scenarios {
            run_resampler_audit(pw, nam, chunk, is_mono);
        }
    }
}
