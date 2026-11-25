// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Heap audit for prepared FiLM modulation, rejection, resume and inference.

#[cfg(feature = "heap-audit")]
mod audit_tests {
    use crate::common::alloc_audit::{
        TrackingGuard, get_alloc_count, get_dealloc_count, get_realloc_count,
    };
    use neural_amp_modeler_rs::models::a2::{A2_KERNEL_SIZES, WaveNetA2};

    fn audit_precomputed_modulation<const CH: usize>() {
        let mut weights = Vec::new();
        weights.extend(std::iter::repeat_n(0.01, CH));
        for &k in &A2_KERNEL_SIZES {
            weights.extend(std::iter::repeat_n(
                0.01,
                CH * CH * k + CH + CH + CH * CH + CH,
            ));
            weights.extend(std::iter::repeat_n(0.0, CH * 2));
            weights.extend(std::iter::repeat_n(1.0, CH));
            weights.extend(std::iter::repeat_n(0.0, CH));
        }
        weights.extend(std::iter::repeat_n(0.01, CH * 16 + 2));
        let mut model = WaveNetA2::<CH>::new().expect("create FiLM A2");
        model.set_layer_raw(Some(serde_json::json!({"activation_pre_film": {"active": true, "shift": true, "groups": 1}})));
        model.set_weights(&weights).expect("load FiLM A2");
        let n = model.activation_modulation_size().expect("prepared FiLM");
        let scales = vec![1.1; n];
        let shifts = vec![0.01; n];
        let mut invalid = shifts.clone();
        invalid[n - 1] = f32::NAN;
        let mut output = [0.0; 128];
        let (allocs, reallocs, deallocs) = {
            let _guard = TrackingGuard::new();
            for _ in 0..100 {
                assert!(model.set_activation_modulation(&scales, &shifts).is_ok());
                assert!(model.set_activation_modulation(&scales, &invalid).is_err());
                assert!(model.set_activation_modulation(&[], &shifts).is_err());
                model.process(&[0.1; 128], &mut output);
                assert!(model.resume_activation_conditioning().is_ok());
                model.process(&[0.1; 128], &mut output);
            }
            (get_alloc_count(), get_realloc_count(), get_dealloc_count())
        };
        assert_eq!(
            (allocs, reallocs, deallocs),
            (0, 0, 0),
            "precomputed FiLM updates and processing must not use the heap"
        );
        assert!(output.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn precomputed_modulation_is_heap_free() {
        audit_precomputed_modulation::<3>();
        audit_precomputed_modulation::<8>();
    }
}
