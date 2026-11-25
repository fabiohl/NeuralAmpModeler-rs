// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

fn weights<const CH: usize>(seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..a2_weight_count::<CH>())
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) as f32 / 65_535.0 - 0.5) * 0.08
        })
        .collect()
}

fn loaded<const CH: usize>(weights: &[f32]) -> WaveNetA2<CH> {
    let mut model = WaveNetA2::new().unwrap();
    model.set_weights(weights).unwrap();
    model.set_max_buffer_size(128).unwrap();
    model
}

fn compare_stream<const CH: usize>(a: &mut WaveNetA2<CH>, b: &mut WaveNetA2<CH>) {
    let input: Vec<_> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
    for size in [1, 13, 32, 64, 128].into_iter().cycle().take(150) {
        let mut actual = [0.0; 128];
        let mut expected = [0.0; 128];
        a.process(&input[..size], &mut actual[..size]);
        b.process(&input[..size], &mut expected[..size]);
        assert!(actual.iter().all(|v| v.is_finite()));
        assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
    }
}

fn continuation<const CH: usize>(fallback: bool) {
    let initial = weights::<CH>(17);
    let next = weights::<CH>(42);
    let mut actual = loaded::<CH>(&initial);
    let mut reference = loaded::<CH>(&initial);
    if fallback {
        for layer in actual.layers.iter_mut().chain(reference.layers.iter_mut()) {
            layer.conv_ch = None;
        }
    }
    compare_stream(&mut actual, &mut reference);
    // A no-op update must preserve dirty state, including wrapped history.
    actual.update_weights(&initial).unwrap();
    compare_stream(&mut actual, &mut reference);

    let head = actual.head_accum.to_vec();
    assert!(head.iter().any(|&v| v != 0.0));
    let starts = actual.layer_buffer_starts.clone();
    let cursor = actual.head_write_pos;
    let histories: Vec<Vec<f32>> = actual
        .layer_buffers
        .iter()
        .map(|b| b[..b.size()].to_vec())
        .collect();
    let pointers: Vec<_> = actual.layer_buffers.iter().map(|b| b.as_ptr()).collect();
    actual.update_weights(&next).unwrap();
    assert_eq!(&*actual.head_accum, head);
    assert_eq!(actual.layer_buffer_starts, starts);
    assert_eq!(actual.head_write_pos, cursor);
    assert_eq!(actual.negotiated_max_buffer_size, 128);
    for ((buffer, saved), pointer) in actual.layer_buffers.iter().zip(histories).zip(pointers) {
        assert_eq!(&buffer[..buffer.size()], saved);
        assert_eq!(buffer.as_ptr(), pointer);
    }
    // The established loader is the coefficient-layout oracle; it also leaves
    // the external streaming rings intact when called without reset.
    reference.set_weights(&next).unwrap();
    if fallback {
        for layer in &mut reference.layers {
            layer.conv_ch = None;
        }
    }
    assert_eq!(&*actual.rechannel_w_f32, &*reference.rechannel_w_f32);
    for (a, b) in actual.layers.iter().zip(&reference.layers) {
        let (A2Conv1d::Standard(a_conv), A2Conv1d::Standard(b_conv)) = (&a.conv, &b.conv) else {
            panic!("canonical convolutions required")
        };
        assert_eq!(&*a_conv.weights, &*b_conv.weights);
        assert_eq!(&*a_conv.bias, &*b_conv.bias);
        assert_eq!(&*a.mixin_w, &*b.mixin_w);
        assert_eq!(&*a.l1x1_w, &*b.l1x1_w);
        assert_eq!(&*a.l1x1_b, &*b.l1x1_b);
        match (&a.conv_ch, &b.conv_ch) {
            (Some(A2ConvCh::Ch3(a)), Some(A2ConvCh::Ch3(b))) => {
                assert_eq!(&*a.weights, &*b.weights);
                assert_eq!(&*a.bias, &*b.bias);
            }
            (Some(A2ConvCh::Ch8(a)), Some(A2ConvCh::Ch8(b))) => {
                assert_eq!(&*a.weights, &*b.weights);
                assert_eq!(&*a.bias, &*b.bias);
            }
            (None, None) => {}
            _ => panic!("specialized layouts must match"),
        }
    }
    let a = actual.head_conv.as_ref().unwrap();
    let b = reference.head_conv.as_ref().unwrap();
    assert_eq!(&*a.head_w, &*b.head_w);
    assert_eq!(a.head_b.to_bits(), b.head_b.to_bits());
    assert_eq!(a.head_scale.to_bits(), b.head_scale.to_bits());
    compare_stream(&mut actual, &mut reference);
}

#[test]
fn runtime_updates_preserve_continuation_ch3() {
    continuation::<3>(false);
    continuation::<3>(true);
}

#[test]
fn runtime_updates_preserve_continuation_ch8() {
    continuation::<8>(false);
    continuation::<8>(true);
}

fn rejected_updates<const CH: usize>() {
    let initial = weights::<CH>(17);
    let mut actual = loaded::<CH>(&initial);
    let mut reference = loaded::<CH>(&initial);
    compare_stream(&mut actual, &mut reference);
    for length in [initial.len() - 1, initial.len() + 1] {
        assert_eq!(
            actual.update_weights(&vec![0.1; length]),
            Err(A2WeightUpdateError::WeightCountMismatch {
                expected: initial.len(),
                actual: length
            })
        );
    }
    for index in [0, initial.len() / 2, initial.len() - 1] {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut invalid = initial.clone();
            invalid[index] = value;
            assert_eq!(
                actual.update_weights(&invalid),
                Err(A2WeightUpdateError::NonFiniteWeight { index })
            );
        }
    }
    compare_stream(&mut actual, &mut reference);
    actual.layers[0].mixin_groups = 2;
    assert_eq!(
        actual.update_weights(&initial),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    actual.layers[0].mixin_groups = 1;
    compare_stream(&mut actual, &mut reference);
    actual.layers[0].conv_pre_film = Some(
        crate::models::a2::FiLMLayer::load(
            crate::models::a2::FiLMConfig {
                active: true,
                shift: true,
                groups: 1,
            },
            1,
            CH,
            vec![0.0; 2 * CH],
            vec![0.0; 2 * CH],
        )
        .unwrap(),
    );
    assert_eq!(
        actual.update_weights(&initial),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
    actual.layers[0].conv_pre_film = None;
    compare_stream(&mut actual, &mut reference);
    actual.layers[0].l1x1_b = crate::math::common::AlignedVec::new(CH - 1, 0.0).unwrap();
    assert_eq!(
        actual.update_weights(&initial),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
}

#[test]
fn runtime_updates_reject_atomically() {
    rejected_updates::<3>();
    rejected_updates::<8>();
    assert_eq!(
        WaveNetA2::<3>::new().unwrap().update_weights(&[]),
        Err(A2WeightUpdateError::NotLoaded)
    );
    assert_eq!(
        WaveNetA2::<4>::new().unwrap().update_weights(&[]),
        Err(A2WeightUpdateError::UnsupportedTopology)
    );
}
