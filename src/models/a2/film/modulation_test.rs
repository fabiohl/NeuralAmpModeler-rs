// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::FiLMConfig;

fn layer(ch: usize) -> FiLMLayer {
    FiLMLayer::load(
        FiLMConfig {
            active: true,
            shift: true,
            groups: 1,
        },
        2,
        ch,
        vec![0.1; ch * 4],
        vec![0.2; ch * 2],
    )
    .unwrap()
}

#[test]
fn precomputed_modulation_matches_final_scales_and_resumes_projection() {
    for ch in 1..=17 {
        let mut actual = layer(ch);
        let mut conditioned = layer(ch);
        let scales: Vec<_> = (0..ch).map(|i| 0.3 + i as f32 * 0.1).collect();
        let shifts: Vec<_> = (0..ch).map(|i| -0.2 + i as f32 * 0.03).collect();
        actual.set_precomputed_modulation(&scales, &shifts).unwrap();
        assert!(actual.has_precomputed_modulation());
        for n in [ch, ch.div_ceil(2)] {
            let mut output = vec![0.4; n];
            // SAFETY: condition has cond_size=2 elements; input is <= ch.
            unsafe {
                actual.process(&mut output, &[99.0, -55.0]);
            }
            for (i, &value) in output.iter().enumerate() {
                assert!((value - (0.4 * scales[i] + shifts[i])).abs() < 1e-6);
            }
        }
        let mut cloned = actual.clone();
        let mut a = vec![0.4; ch];
        let mut b = a.clone();
        // SAFETY: both condition vectors have cond_size=2; inputs have ch elements.
        unsafe {
            actual.process(&mut a, &[0.0; 2]);
            cloned.process(&mut b, &[1.0; 2]);
        }
        assert_eq!(a, b);
        actual.use_conditioned_modulation();
        assert!(!actual.has_precomputed_modulation());
        let mut a = vec![0.4; ch];
        let mut b = a.clone();
        // SAFETY: both valid loaded layers have matching input/condition extents.
        unsafe {
            actual.process(&mut a, &[0.5, 0.8]);
            conditioned.process(&mut b, &[0.5, 0.8]);
        }
        assert_eq!(a, b);
    }
}

#[test]
fn grouped_projection_resumes_without_retaining_override_coefficients() {
    for (shift, groups, expected) in [(false, 2, [1.0, 1.0, 0.0]), (true, 4, [2.0, 1.0, 1.0])] {
        let condition = [0.0; 4];
        let mut layer = FiLMLayer::load(
            FiLMConfig {
                active: true,
                shift,
                groups,
            },
            condition.len(),
            3,
            vec![0.0; 4],
            vec![1.0; if shift { 6 } else { 3 }],
        )
        .unwrap();
        let mut before = [1.0; 3];
        // SAFETY: Both slices match the loaded dimensions.
        unsafe { layer.process(&mut before, &condition) };
        assert_eq!(before, expected);

        layer
            .set_precomputed_modulation(&[2.0; 3], &[3.0; 3])
            .unwrap();
        let mut overridden = [1.0; 3];
        // SAFETY: Both slices match the loaded dimensions.
        unsafe { layer.process(&mut overridden, &condition) };
        assert_eq!(overridden, [5.0; 3]);

        layer.use_conditioned_modulation();
        layer.use_conditioned_modulation();
        let mut resumed = [1.0; 3];
        // SAFETY: Both slices match the loaded dimensions.
        unsafe { layer.process(&mut resumed, &condition) };
        assert_eq!(resumed, before);
    }
}

#[test]
fn precomputed_modulation_rejects_before_changing_mode_or_values() {
    let mut actual = layer(3);
    let weights = actual.weights.to_vec();
    let bias = actual.bias.to_vec();
    assert_eq!(
        actual.set_precomputed_modulation(&[1.0; 2], &[0.0; 3]),
        Err(FiLMModulationError::SizeMismatch {
            expected: 3,
            scales: 2,
            shifts: 3
        })
    );
    assert!(!actual.has_precomputed_modulation());
    actual
        .set_precomputed_modulation(&[2.0; 3], &[0.3; 3])
        .unwrap();
    let before = actual.scale_shift_buf.to_vec();
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for index in [0, 1, 2] {
            let mut scales = [1.0; 3];
            scales[index] = value;
            let mut shifts = [0.0; 3];
            shifts[index] = value;
            assert_eq!(
                actual.set_precomputed_modulation(&scales, &[0.0; 3]),
                Err(FiLMModulationError::NonFiniteScale { index })
            );
            assert_eq!(
                actual.set_precomputed_modulation(&[1.0; 3], &shifts),
                Err(FiLMModulationError::NonFiniteShift { index })
            );
            assert_eq!(&*actual.scale_shift_buf, before);
            assert!(actual.has_precomputed_modulation());
        }
    }
    assert_eq!(&*actual.weights, weights);
    assert_eq!(&*actual.bias, bias);
    actual.channels = 4;
    assert_eq!(
        actual.set_precomputed_modulation(&[1.0; 4], &[0.0; 4]),
        Err(FiLMModulationError::InvalidTopology)
    );
    assert_eq!(&*actual.scale_shift_buf, before);
}
