// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// Integration Test for split `prewarm_reset` / `prewarm_step` Heap-Audit Coverage (RT-Safety).
//
// Verifies zero heap allocations, zero reallocations, zero deallocations, and
// monotonic convergence across every model family.

#[cfg(feature = "heap-audit")]
mod prewarm_audit_tests {
    use neural_amp_modeler_rs::SystemSnapshot;
    use neural_amp_modeler_rs::loader::{LoadOptions, load_and_build_model};
    use neural_amp_modeler_rs::models::{NamModel, StaticModel};

    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };
    use crate::common::validation::report_heap_alloc;

    fn load_fixture(name: &str) -> Option<Box<StaticModel>> {
        let path = crate::common::model_path(name);
        if !path.exists() {
            println!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:{name}\"");
            return None;
        }
        let sys = SystemSnapshot::capture();
        let pair = load_and_build_model(&path, &sys, false, LoadOptions::default())
            .unwrap_or_else(|e| panic!("Failed to load {name}: {e}"));
        Some(pair.model_l.expect("fixture yields model_l"))
    }

    fn audit_split_prewarm(label: &str, model: &mut StaticModel) {
        // Warmup off-guard so lazy CPUID/dispatch initialization isn't conflated
        // with per-step heap traffic, then re-arm inside the audited window.
        model.prewarm_reset();
        let _ = model.prewarm_step(1);

        let (allocs, reallocs, deallocs, steps) = {
            let _guard = TrackingGuard::new();
            model.prewarm_reset();
            assert!(
                !model.prewarm_complete(),
                "{label}: prewarm_reset must arm pending work"
            );
            let mut steps = 0usize;
            while !model.prewarm_complete() {
                std::hint::black_box(model.prewarm_step(std::hint::black_box(64)));
                steps += 1;
                assert!(
                    steps < 10_000,
                    "{label}: prewarm_step(64) did not converge within 10_000 steps"
                );
            }
            (
                get_alloc_count(),
                get_realloc_count(),
                get_dealloc_count(),
                steps,
            )
        };

        assert!(steps > 0, "{label}: expected at least 1 prewarm step");
        assert!(
            model.prewarm_complete(),
            "{label}: prewarm_complete() must be true after convergence"
        );
        assert_eq!(
            allocs, 0,
            "Heap allocations detected on {label} prewarm_reset/prewarm_step! allocs={allocs}"
        );
        assert_eq!(
            reallocs, 0,
            "Heap reallocations detected on {label} prewarm_reset/prewarm_step! reallocs={reallocs}"
        );
        assert_eq!(
            deallocs, 0,
            "Heap deallocations detected on {label} prewarm_reset/prewarm_step! deallocs={deallocs}"
        );
        report_heap_alloc(&format!("prewarm-step-{label}"), allocs);
    }

    #[test]
    fn test_all_model_families_prewarm_step_heap_audit() {
        let cases: &[(&str, &str)] = &[
            ("WaveNet", "BossWN-standard.nam"),
            ("WaveNetDyn", "BossWN-feather.nam"),
            ("WaveNetA2", "wavenet_a2_full.nam"),
            ("WaveNetA2Dyn", "a2_dynamic_gated_ch8.nam"),
            ("WaveNetA2DynCond", "wavenet_a2_film_full.nam"),
            ("WaveNetA2Cascade", "wavenet_a2_max.nam"),
            ("ConvNet", "convnet_relu.nam"),
            ("Lstm", "BossLSTM-1x16.nam"),
            ("LstmDyn", "lstm_dyn_test.nam"),
            ("Linear", "linear_test.nam"),
            ("Slimmable", "slimmable_container.nam"),
        ];

        let mut audited = 0usize;
        for &(label, fixture) in cases {
            if let Some(mut model) = load_fixture(fixture) {
                audit_split_prewarm(label, &mut model);
                audited += 1;
            }
        }
        assert!(
            audited > 0,
            "At least one model fixture must be audited for prewarm_step heap safety"
        );
    }
}
