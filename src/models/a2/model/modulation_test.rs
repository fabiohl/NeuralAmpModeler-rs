// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::a2::{ActivationType, FiLMConfig, FiLMLayer, GatingMode, a2_weight_count};

fn projection(ch: usize) -> FiLMLayer {
    let mut bias = vec![0.0; ch * 2];
    bias[..ch].fill(1.0);
    FiLMLayer::load(
        FiLMConfig {
            active: true,
            shift: true,
            groups: 1,
        },
        1,
        ch,
        vec![0.0; ch * 2],
        bias,
    )
    .unwrap()
}

fn prepare(layers: &mut [A2Layer]) {
    for layer in layers {
        layer.activation_pre_film = Some(projection(layer.conv.out_ch()));
    }
}

fn set_reference(layers: &mut [A2Layer], scales: &[f32], shifts: &[f32], pos: &mut usize) {
    for layer in layers {
        let film = layer.activation_pre_film.as_mut().unwrap();
        let end = *pos + film.channels;
        film.bias[..film.channels].copy_from_slice(&scales[*pos..end]);
        film.bias[film.channels..].copy_from_slice(&shifts[*pos..end]);
        *pos = end;
    }
}

fn static_parity<const CH: usize>() {
    let mut actual = WaveNetA2::<CH>::new().unwrap();
    let mut reference = WaveNetA2::<CH>::new().unwrap();
    let weights = vec![0.01; a2_weight_count::<CH>()];
    actual.set_weights(&weights).unwrap();
    reference.set_weights(&weights).unwrap();
    prepare(&mut actual.layers);
    prepare(&mut reference.layers);
    let n = actual.activation_modulation_size().unwrap();
    assert_eq!(n, 23 * CH);
    let input: Vec<_> = (0..128).map(|i| (i as f32 * 0.137).sin()).collect();
    for block in 0..150 {
        let scales: Vec<_> = (0..n).map(|i| 0.8 + (i + block) as f32 * 0.001).collect();
        let shifts: Vec<_> = (0..n)
            .map(|i| (i as f32 * 0.1 + block as f32).sin() * 0.01)
            .collect();
        let head = actual.head_accum.to_vec();
        let starts = actual.layer_buffer_starts.clone();
        let cursor = actual.head_write_pos;
        let histories: Vec<_> = actual
            .layer_buffers
            .iter()
            .map(|b| b[..b.size()].to_vec())
            .collect();
        actual.set_activation_modulation(&scales, &shifts).unwrap();
        assert_eq!(&*actual.head_accum, head);
        assert_eq!(actual.layer_buffer_starts, starts);
        assert_eq!(actual.head_write_pos, cursor);
        for (b, saved) in actual.layer_buffers.iter().zip(histories) {
            assert_eq!(&b[..b.size()], saved);
        }
        set_reference(&mut reference.layers, &scales, &shifts, &mut 0);
        let size = [1, 13, 32, 64, 128][block % 5];
        let mut output = [0.0; 128];
        let mut expected = [0.0; 128];
        actual.process(&input[..size], &mut output[..size]);
        reference.process(&input[..size], &mut expected[..size]);
        assert!(output.iter().all(|v| v.is_finite()));
        assert_eq!(output.map(f32::to_bits), expected.map(f32::to_bits));
    }
    actual.resume_activation_conditioning().unwrap();
    set_reference(&mut reference.layers, &vec![1.0; n], &vec![0.0; n], &mut 0);
    let mut a = [0.0; 128];
    let mut b = [0.0; 128];
    actual.process(&input, &mut a);
    reference.process(&input, &mut b);
    assert_eq!(a, b);
    // Reset clears causal state, while the committed override remains active.
    actual
        .set_activation_modulation(&vec![1.0; n], &vec![0.0; n])
        .unwrap();
    actual.prewarm_on_reset = false;
    reference.prewarm_on_reset = false;
    actual.reset(48_000, 128).unwrap();
    reference.reset(48_000, 128).unwrap();
    assert!(actual.layers.iter().all(|l| {
        l.activation_pre_film
            .as_ref()
            .unwrap()
            .has_precomputed_modulation()
    }));
    actual.process(&input, &mut a);
    reference.process(&input, &mut b);
    assert_eq!(a, b);
}

#[test]
fn activation_modulation_matches_condition_projection_with_live_changes() {
    static_parity::<3>();
    static_parity::<8>();
}

fn dynamic(input: usize, head: usize, count: usize) -> WaveNetA2Dyn {
    let mut model = WaveNetA2Dyn::new(
        input,
        3,
        3,
        head,
        3,
        3,
        1,
        &[2],
        &[1],
        vec![ActivationType::Tanh],
        vec![GatingMode::None],
        vec![None],
    )
    .unwrap();
    model.set_weights(&vec![0.01; count]).unwrap();
    prepare(&mut model.layers);
    model
}

#[test]
fn cascade_activation_modulation_is_atomic_across_layers_and_arrays() {
    let mut model =
        WaveNetA2Cascade::try_new(vec![dynamic(1, 3, 51), dynamic(3, 1, 50)], None, 1).unwrap();
    assert_eq!(model.activation_modulation_size(), Ok(6));
    model
        .set_activation_modulation(&[2.0; 6], &[0.1; 6])
        .unwrap();
    let before = model.arrays[0].layers[0]
        .activation_pre_film
        .as_ref()
        .unwrap()
        .clone();
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut shifts = [0.0; 6];
        shifts[5] = bad;
        assert_eq!(
            model.set_activation_modulation(&[1.0; 6], &shifts),
            Err(FiLMModulationError::NonFiniteShift { index: 5 })
        );
    }
    assert_eq!(
        model.set_activation_modulation(&[1.0; 5], &[0.0; 6]),
        Err(FiLMModulationError::SizeMismatch {
            expected: 6,
            scales: 5,
            shifts: 6
        })
    );
    let removed = model.arrays[1].layers[0].activation_pre_film.take();
    assert_eq!(
        model.set_activation_modulation(&[1.0; 6], &[0.0; 6]),
        Err(FiLMModulationError::NotPrepared)
    );
    assert_eq!(
        model.resume_activation_conditioning(),
        Err(FiLMModulationError::NotPrepared)
    );
    model.arrays[1].layers[0].activation_pre_film = removed;
    let actual = model.arrays[0].layers[0]
        .activation_pre_film
        .as_mut()
        .unwrap();
    let mut control = before;
    let mut a = [0.2; 3];
    let mut b = a;
    // SAFETY: condition width is 1 and inputs match the prepared 3-channel FiLM layers.
    unsafe {
        actual.process(&mut a, &[0.1]);
        control.process(&mut b, &[0.1]);
    }
    assert_eq!(a, b);
    model.resume_activation_conditioning().unwrap();
    assert!(model.arrays.iter().all(|a| a.layers.iter().all(|l| {
        !l.activation_pre_film
            .as_ref()
            .unwrap()
            .has_precomputed_modulation()
    })));
}

#[test]
fn activation_modulation_requires_loaded_film() {
    let mut model = WaveNetA2::<3>::new().unwrap();
    assert_eq!(
        model.set_activation_modulation(&[], &[]),
        Err(FiLMModulationError::NotPrepared)
    );
    model
        .set_weights(&vec![0.01; a2_weight_count::<3>()])
        .unwrap();
    assert_eq!(
        model.activation_modulation_size(),
        Err(FiLMModulationError::NotPrepared)
    );
}

#[test]
fn dynamic_activation_modulation_uses_pre_gating_feature_width() {
    fn loaded() -> WaveNetA2Dyn {
        let mut model = WaveNetA2Dyn::new(
            1,
            5,
            3,
            1,
            3,
            3,
            1,
            &[2, 2],
            &[1, 2],
            vec![ActivationType::Tanh; 2],
            vec![GatingMode::Gated; 2],
            vec![Some(ActivationType::Sigmoid); 2],
        )
        .unwrap();
        model.set_weights(&vec![0.01; 194]).unwrap();
        prepare(&mut model.layers);
        model
    }
    let mut actual = loaded();
    let mut reference = loaded();
    assert_eq!(actual.activation_modulation_size(), Ok(12));
    let scales: Vec<_> = (0..12).map(|i| 0.7 + i as f32 * 0.03).collect();
    let shifts: Vec<_> = (0..12).map(|i| -0.02 + i as f32 * 0.001).collect();
    actual.set_activation_modulation(&scales, &shifts).unwrap();
    set_reference(&mut reference.layers, &scales, &shifts, &mut 0);
    let mut a = [0.0; 128];
    let mut b = a;
    for _ in 0..20 {
        actual.process(&[0.1; 128], &mut a);
        reference.process(&[0.1; 128], &mut b);
        assert_eq!(a, b);
    }
}
