// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::{ActivationType, GatingMode, WaveNetA2Dyn};

fn unloaded(bias: bool, final_head: usize) -> WaveNetA2Cascade {
    let mut first = WaveNetA2Dyn::new(
        1,
        3,
        3,
        5,
        3,
        3,
        2,
        &[2, 6],
        &[1, 2],
        vec![ActivationType::Tanh; 2],
        vec![GatingMode::None; 2],
        vec![None; 2],
    )
    .unwrap();
    first.head_bias = bias;
    let second = WaveNetA2Dyn::new(
        3,
        5,
        5,
        final_head,
        5,
        5,
        3,
        &[3, 2],
        &[1, 3],
        vec![ActivationType::Tanh; 2],
        vec![GatingMode::None; 2],
        vec![None; 2],
    )
    .unwrap();
    WaveNetA2Cascade::try_new(vec![first, second], None, 1).unwrap()
}

fn count(model: &WaveNetA2Cascade) -> usize {
    model
        .arrays
        .iter()
        .map(|a| {
            a.input_channels * a.channels
                + a.kernel_sizes
                    .iter()
                    .map(|&k| {
                        a.channels * a.bottleneck * k
                            + a.bottleneck
                            + a.bottleneck * a.condition_size
                            + a.channels * a.bottleneck
                            + a.channels
                    })
                    .sum::<usize>()
                + a.head_size * a.head_kernel_size * a.head_accum_size
                + if a.head_size == 1 {
                    2
                } else if a.head_bias {
                    a.head_size
                } else {
                    0
                }
        })
        .sum()
}

fn weights(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 16) as f32 / 65_535.0 - 0.5) * 0.2
        })
        .collect()
}

fn load_reference(model: &mut WaveNetA2Cascade, weights: &[f32], scale: f32) {
    let mut pos = 0;
    for array in &mut model.arrays {
        array
            .load_weights_inner(weights, &mut pos, weights.len())
            .unwrap();
        // These are parent-level scales, not serialized array coefficients.
        array.head_rechannel_scale.fill(0.75);
    }
    assert_eq!(pos, weights.len());
    model.head_scale = scale;
}

fn snapshot(model: &WaveNetA2Cascade, coefficients: bool) -> Vec<u32> {
    let mut values = Vec::new();
    if coefficients {
        values.push(model.head_scale.to_bits());
    }
    for a in &model.arrays {
        if coefficients {
            values.extend(a.rechannel_w_f32.iter().map(|v| v.to_bits()));
            for l in &a.layers {
                let crate::models::a2::A2Conv1d::Standard(c) = &l.conv else {
                    panic!("dense fixture required")
                };
                for field in [&c.weights, &c.bias, &l.mixin_w, &l.l1x1_w, &l.l1x1_b] {
                    values.extend(field.iter().map(|v| v.to_bits()));
                }
            }
            for field in [
                &a.head_rechannel_w,
                &a.head_rechannel_b,
                &a.head_rechannel_scale,
            ] {
                values.extend(field.iter().map(|v| v.to_bits()));
            }
            if let Some(h) = &a.head_conv {
                values.extend(h.head_w.iter().map(|v| v.to_bits()));
                values.extend([h.head_b.to_bits(), h.head_scale.to_bits()]);
            }
        }
        values.extend(a.head_accum.iter().map(|v| v.to_bits()));
        for b in &a.layer_buffers {
            values.extend(b[..b.size()].iter().map(|v| v.to_bits()));
        }
    }
    values
}

#[test]
fn cascade_runtime_updates_match_loader_and_preserve_all_histories() {
    for bias in [false, true] {
        for final_head in [1, 2] {
            let mut actual = unloaded(bias, final_head);
            let mut reference = unloaded(bias, final_head);
            let initial = weights(count(&actual), 17);
            load_reference(&mut actual, &initial, 0.7);
            load_reference(&mut reference, &initial, 0.7);
            assert_eq!(actual.runtime_weight_count(), Ok(initial.len()));
            let input: Vec<_> = (0..128).map(|i| (i as f32 * 0.137).sin()).collect();
            for block in 0..100 {
                let size = [1, 13, 32, 64, 128][block % 5];
                if block % 9 == 0 {
                    let next = weights(initial.len(), block as u32 + 1);
                    let history = snapshot(&actual, false);
                    let starts: Vec<_> = actual
                        .arrays
                        .iter()
                        .map(|a| (a.head_write_pos, a.layer_buffer_starts.clone()))
                        .collect();
                    let pointers: Vec<_> = actual
                        .arrays
                        .iter()
                        .flat_map(|a| a.layer_buffers.iter().map(|b| b.as_ptr()))
                        .collect();
                    let scale = 0.3 + block as f32 * 0.001;
                    actual.update_weights(&next, scale).unwrap();
                    assert_eq!(snapshot(&actual, false), history);
                    assert_eq!(
                        actual
                            .arrays
                            .iter()
                            .map(|a| (a.head_write_pos, a.layer_buffer_starts.clone()))
                            .collect::<Vec<_>>(),
                        starts
                    );
                    assert_eq!(
                        actual
                            .arrays
                            .iter()
                            .flat_map(|a| a.layer_buffers.iter().map(|b| b.as_ptr()))
                            .collect::<Vec<_>>(),
                        pointers
                    );
                    load_reference(&mut reference, &next, scale);
                    assert_eq!(snapshot(&actual, true), snapshot(&reference, true));
                }
                let mut output = [0.0; 256];
                let mut expected = [0.0; 256];
                actual.process(&input[..size], &mut output[..size * final_head]);
                reference.process(&input[..size], &mut expected[..size * final_head]);
                assert!(output.iter().all(|v| v.is_finite()));
                assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
            }
        }
    }
}

#[test]
fn cascade_runtime_updates_reject_before_mutating_any_array() {
    let mut model = unloaded(true, 1);
    assert_eq!(
        model.runtime_weight_count(),
        Err(A2WeightUpdateError::NotLoaded)
    );
    let initial = weights(count(&model), 17);
    load_reference(&mut model, &initial, 0.7);
    model.process(&[0.1; 64], &mut [0.0; 64]);
    let before = snapshot(&model, true);
    for len in [initial.len() - 1, initial.len() + 1] {
        assert_eq!(
            model.update_weights(&vec![0.0; len], 0.8),
            Err(A2WeightUpdateError::WeightCountMismatch {
                expected: initial.len(),
                actual: len
            })
        );
        assert_eq!(snapshot(&model, true), before);
    }
    let mut invalid = weights(initial.len(), 42);
    let last = invalid.len() - 1;
    invalid[last] = f32::NAN;
    assert_eq!(
        model.update_weights(&invalid, 0.8),
        Err(A2WeightUpdateError::NonFiniteWeight { index: last })
    );
    assert_eq!(snapshot(&model, true), before);
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            model.update_weights(&initial, scale),
            Err(A2WeightUpdateError::NonFiniteHeadScale)
        );
        assert_eq!(snapshot(&model, true), before);
    }
    model.arrays[1].l1x1_groups = 2;
    assert_eq!(
        model.update_weights(&weights(initial.len(), 42), 0.8),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    assert_eq!(snapshot(&model, true), before);
}
