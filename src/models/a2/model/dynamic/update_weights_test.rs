// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::{A2Conv1d, A2ConvCh, ActivationType};

fn make_model(
    ch: usize,
    bn: usize,
    gating: GatingMode,
    kernel: usize,
    head_kernel: usize,
) -> WaveNetA2Dyn {
    WaveNetA2Dyn::new(
        1,
        ch,
        bn,
        1,
        bn,
        bn,
        head_kernel,
        &[2, kernel, 3],
        &[1, 2, 4],
        vec![ActivationType::Tanh; 3],
        vec![gating; 3],
        vec![
            if gating == GatingMode::None {
                None
            } else {
                Some(ActivationType::Sigmoid)
            };
            3
        ],
    )
    .unwrap()
}

fn count(model: &WaveNetA2Dyn) -> usize {
    model.input_channels * model.channels
        + model
            .kernel_sizes
            .iter()
            .enumerate()
            .map(|(i, &k)| {
                let conv_out = model.bottleneck
                    * if model.gating_modes[i] == GatingMode::None {
                        1
                    } else {
                        2
                    };
                conv_out * model.channels * k
                    + conv_out
                    + conv_out * model.condition_size
                    + model.channels * model.bottleneck
                    + model.channels
            })
            .sum::<usize>()
        + model.head_kernel_size * model.head_accum_size
        + 2
}

fn weights(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 16) as f32 / 65_535.0 - 0.5) * 0.2
        })
        .collect()
}

fn stored_weights(model: &WaveNetA2Dyn) -> Vec<f32> {
    let mut stored = model.rechannel_w_f32.to_vec();
    for layer in &model.layers {
        let A2Conv1d::Standard(conv) = &layer.conv else {
            panic!("dense model required")
        };
        stored.extend_from_slice(&conv.weights);
        stored.extend_from_slice(&conv.bias);
        match &layer.conv_ch {
            Some(A2ConvCh::Ch3(conv)) => {
                stored.extend_from_slice(&conv.weights);
                stored.extend_from_slice(&conv.bias);
            }
            Some(A2ConvCh::Ch8(conv)) => {
                stored.extend_from_slice(&conv.weights);
                stored.extend_from_slice(&conv.bias);
            }
            None => {}
        }
        stored.extend_from_slice(&layer.mixin_w);
        stored.extend_from_slice(&layer.l1x1_w);
        stored.extend_from_slice(&layer.l1x1_b);
    }
    let head = model.head_conv.as_ref().unwrap();
    stored.extend_from_slice(&head.head_w);
    stored.extend([head.head_b, head.head_scale]);
    stored
}

#[test]
fn dynamic_runtime_updates_match_loader_with_live_changes() {
    for (ch, bn, gating, kernel, head_kernel) in [
        (3, 3, GatingMode::None, 6, 16),
        (8, 8, GatingMode::None, 15, 16),
        (5, 7, GatingMode::None, 2, 1),
        (4, 3, GatingMode::Gated, 7, 3),
        (9, 5, GatingMode::Blended, 1, 1),
    ] {
        let mut actual = make_model(ch, bn, gating, kernel, head_kernel);
        let mut reference = make_model(ch, bn, gating, kernel, head_kernel);
        let initial = weights(count(&actual), 17);
        actual.set_weights(&initial).unwrap();
        reference.set_weights(&initial).unwrap();
        actual.set_max_buffer_size(128).unwrap();
        reference.set_max_buffer_size(128).unwrap();
        assert_eq!(actual.runtime_weight_count(), Ok(initial.len()));
        let input: Vec<_> = (0..128).map(|i| (i as f32 * 0.137).sin()).collect();
        for block in 0..160 {
            let size = [1, 13, 32, 64, 128][block % 5];
            if block % 7 == 0 {
                let next = weights(initial.len(), block as u32 + 1);
                let head = actual.head_accum.to_vec();
                let starts = actual.layer_buffer_starts.clone();
                let cursor = actual.head_write_pos;
                let histories: Vec<Vec<f32>> = actual
                    .layer_buffers
                    .iter()
                    .map(|b| b[..b.size()].to_vec())
                    .collect();
                let pointers: Vec<_> = actual.layer_buffers.iter().map(|b| b.as_ptr()).collect();
                actual.update_weights(&next).unwrap();
                reference.set_weights(&next).unwrap();
                assert_eq!(stored_weights(&actual), stored_weights(&reference));
                assert_eq!(&*actual.head_accum, head);
                assert_eq!(actual.layer_buffer_starts, starts);
                assert_eq!(actual.head_write_pos, cursor);
                assert_eq!(actual.negotiated_max_buffer_size, 128);
                for ((buffer, saved), pointer) in
                    actual.layer_buffers.iter().zip(histories).zip(pointers)
                {
                    assert_eq!(&buffer[..buffer.size()], saved);
                    assert_eq!(buffer.as_ptr(), pointer);
                }
            }
            let mut output = [0.0; 128];
            let mut expected = [0.0; 128];
            actual.process(&input[..size], &mut output[..size]);
            reference.process(&input[..size], &mut expected[..size]);
            assert!(output.iter().all(|v| v.is_finite()));
            assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}

#[test]
fn dynamic_runtime_updates_pack_multichannel_mixin_and_rechannel() {
    let mut actual = make_model(5, 7, GatingMode::None, 6, 16);
    let mut reference = make_model(5, 7, GatingMode::None, 6, 16);
    for model in [&mut actual, &mut reference] {
        model.input_channels = 4;
        model.condition_size = 4;
    }
    let initial = weights(count(&actual), 17);
    let next = weights(initial.len(), 42);
    actual.set_weights(&initial).unwrap();
    reference.set_weights(&next).unwrap();
    actual.update_weights(&next).unwrap();
    assert_eq!(stored_weights(&actual), stored_weights(&reference));
}

#[test]
fn dynamic_runtime_updates_reject_atomically() {
    let mut model = make_model(5, 7, GatingMode::None, 2, 1);
    assert_eq!(
        model.runtime_weight_count(),
        Err(A2WeightUpdateError::NotLoaded)
    );
    let initial = weights(count(&model), 17);
    model.set_weights(&initial).unwrap();
    let before = stored_weights(&model);
    for n in [0, initial.len() - 1, initial.len() + 1] {
        assert_eq!(
            model.update_weights(&vec![0.0; n]),
            Err(A2WeightUpdateError::WeightCountMismatch {
                expected: initial.len(),
                actual: n
            })
        );
    }
    for index in [0, initial.len() / 2, initial.len() - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = initial.clone();
            invalid[index] = value;
            assert_eq!(
                model.update_weights(&invalid),
                Err(A2WeightUpdateError::NonFiniteWeight { index })
            );
            assert_eq!(stored_weights(&model), before);
        }
    }
    for field in [0, 1, 2] {
        match field {
            0 => model.head1x1_active = true,
            1 => model.groups_input = 2,
            _ => model.mixin_groups = 2,
        }
        assert_eq!(
            model.update_weights(&initial),
            Err(A2WeightUpdateError::UnsupportedTopology)
        );
        assert_eq!(stored_weights(&model), before);
        model.head1x1_active = false;
        model.groups_input = 1;
        model.mixin_groups = 1;
    }
    model.condition_size = usize::MAX;
    assert_eq!(
        model.update_weights(&initial),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    assert_eq!(stored_weights(&model), before);
    model.condition_size = 1;
    model.input_channels = usize::MAX;
    assert_eq!(
        model.runtime_weight_count(),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    model.input_channels = 1;

    // A late destination mismatch must not partially overwrite earlier layers.
    model.layers[2].l1x1_b = crate::math::common::AlignedVec::new(4, 0.0).unwrap();
    let before = stored_weights(&model);
    assert_eq!(
        model.update_weights(&weights(initial.len(), 42)),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    assert_eq!(stored_weights(&model), before);
}
