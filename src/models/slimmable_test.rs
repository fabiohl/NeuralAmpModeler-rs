// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    Conv1dDyn, DenseLayerDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn, WaveNetLayerDyn,
    WaveNetLayerState, WaveNetModelDyn,
};

const TEST_KERNEL: usize = 3;
const TEST_DILATION: usize = 2;
const CH_FULL: usize = 8;
const CH_SLIM: usize = 4;

fn make_conv1d(in_ch: usize, out_ch: usize) -> Conv1dDyn {
    let kernel = TEST_KERNEL;
    let num_blocks = out_ch.div_ceil(4);
    let weights_len = num_blocks * 4 * in_ch * kernel;
    let mut weights = AlignedVec::new(weights_len, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    for i in 0..weights_len {
        weights[i] = (i + 1) as f32;
    }
    let mut bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    for i in 0..out_ch {
        bias[i] = (i + 100) as f32;
    }
    Conv1dDyn {
        weights,
        bias,
        do_bias: true,
        dilation: TEST_DILATION,
        in_ch,
        out_ch,
        num_blocks,
        interleave_width: 4,
        kernel,
    }
}

fn make_dense(in_ch: usize, out_ch: usize) -> DenseLayerDyn {
    let mut weights = AlignedVec::new(in_ch * out_ch, 0.0f32)
        .expect("allocation should succeed for test-sized buffers");
    for in_c in 0..in_ch {
        for out_c in 0..out_ch {
            weights[in_c * out_ch + out_c] = ((in_c * out_ch + out_c) as f32) + 1.0;
        }
    }
    let mut bias =
        AlignedVec::new(out_ch, 0.0f32).expect("allocation should succeed for test-sized buffers");
    for i in 0..out_ch {
        bias[i] = (i + 200) as f32;
    }
    DenseLayerDyn {
        in_ch,
        out_ch,
        weights,
        bias,
        do_bias: true,
    }
}

fn make_wavenet_layer(ch: usize) -> WaveNetLayerDyn {
    let conv1d = make_conv1d(ch, ch);
    let input_mixin = make_dense(1, ch);
    let one_by_one = make_dense(ch, ch);
    WaveNetLayerDyn::new(ch, conv1d, input_mixin, one_by_one).unwrap()
}

fn make_wavenet_array(
    in_ch: usize,
    ch: usize,
    head: usize,
    dilations: &[usize],
) -> WaveNetLayerArrayDyn {
    let rechannel = make_dense(in_ch, ch);
    let num_layers = dilations.len();
    let mut layers = Vec::with_capacity(num_layers);
    let mut states = Vec::with_capacity(num_layers);
    for (alloc_num, &d) in dilations.iter().enumerate() {
        let mut layer = make_wavenet_layer(ch);
        layer.conv1d.dilation = d;
        let rf = (TEST_KERNEL - 1) * d;
        states.push(WaveNetLayerState::new(ch, rf, alloc_num).unwrap());
        layers.push(layer);
    }
    let head_rechannel = make_dense(ch, head);
    let receptive_field_size: usize = dilations.iter().map(|&d| (TEST_KERNEL - 1) * d).sum();
    let block_size = ch;
    WaveNetLayerArrayDyn {
        in_ch,
        cond: 1,
        ch,
        k: TEST_KERNEL,
        head,
        layers,
        states,
        rechannel,
        head_rechannel,
        array_outputs: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        head_accum: AlignedVec::new(ch * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        head_outputs: AlignedVec::new(head * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        receptive_field_size,
        block_size,
        block_buffer: AlignedVec::new(block_size * WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        effective_layers: num_layers,
    }
}

fn make_full_model(ch: usize, head: usize) -> WaveNetModelDyn {
    let dilations = [1, 2, 4];
    let array1 = make_wavenet_array(1, ch, head, &dilations);
    let array2 = make_wavenet_array(ch, head, 1, &dilations);
    let rf = array1.receptive_field_size.max(array2.receptive_field_size);
    WaveNetModelDyn {
        ch,
        k: TEST_KERNEL,
        head,
        arrays: vec![array1, array2],
        head_scale: 0.02,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0)
            .expect("allocation should succeed for test-sized buffers"),
        prewarm_on_reset: true,
        slimmable_capable: true,
        allowed_channels: None,
        pending_slim_channel: None,
    }
}

// =====================================================================
// slice_conv1d tests
// =====================================================================

#[test]
fn test_slice_conv1d_dims() {
    let conv = make_conv1d(CH_FULL, CH_FULL);
    let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM).unwrap();
    assert_eq!(sliced.in_ch, CH_SLIM);
    assert_eq!(sliced.out_ch, CH_SLIM);
    assert_eq!(sliced.kernel, TEST_KERNEL);
    assert_eq!(sliced.dilation, TEST_DILATION);
    assert!(sliced.do_bias);
    assert_eq!(sliced.num_blocks, CH_SLIM.div_ceil(4));
    assert_eq!(
        sliced.weights.len(),
        sliced.num_blocks * 4 * CH_SLIM * TEST_KERNEL
    );
    assert_eq!(sliced.bias.len(), CH_SLIM);
}

#[test]
fn test_slice_conv1d_weights_match() {
    let conv = make_conv1d(CH_FULL, CH_FULL);
    let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM).unwrap();

    for b in 0..sliced.num_blocks {
        for k in 0..TEST_KERNEL {
            for in_c in 0..CH_SLIM {
                let src_idx = ((b * TEST_KERNEL + k) * CH_FULL + in_c) * 4;
                let dst_idx = ((b * TEST_KERNEL + k) * CH_SLIM + in_c) * 4;
                assert_eq!(
                    &sliced.weights[dst_idx..dst_idx + 4],
                    &conv.weights[src_idx..src_idx + 4],
                    "mismatch at block={} k={} in_c={}",
                    b,
                    k,
                    in_c
                );
            }
        }
    }
}

#[test]
fn test_slice_conv1d_bias_match() {
    let conv = make_conv1d(CH_FULL, CH_FULL);
    let sliced = slice_conv1d(&conv, CH_SLIM, CH_SLIM).unwrap();
    assert_eq!(&sliced.bias[..CH_SLIM], &conv.bias[..CH_SLIM]);
}

#[test]
fn test_slice_conv1d_bigger_in_ch_returns_err() {
    let conv = make_conv1d(CH_FULL, CH_FULL);
    let Err(err) = slice_conv1d(&conv, CH_FULL + 1, CH_FULL) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::InvalidInputChannels {
            requested: CH_FULL + 1,
            available: CH_FULL
        }
    );
}

#[test]
fn test_slice_conv1d_bigger_out_ch_returns_err() {
    let conv = make_conv1d(CH_FULL, CH_FULL);
    let Err(err) = slice_conv1d(&conv, CH_FULL, CH_FULL + 1) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::InvalidOutputChannels {
            requested: CH_FULL + 1,
            available: CH_FULL
        }
    );
}

// =====================================================================
// slice_dense tests
// =====================================================================

#[test]
fn test_slice_dense_dims() {
    let dense = make_dense(CH_FULL, CH_FULL);
    let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM).unwrap();
    assert_eq!(sliced.in_ch, CH_SLIM);
    assert_eq!(sliced.out_ch, CH_SLIM);
    assert_eq!(sliced.do_bias, dense.do_bias);
    assert_eq!(sliced.weights.len(), CH_SLIM * CH_SLIM);
    assert_eq!(sliced.bias.len(), CH_SLIM);
}

#[test]
fn test_slice_dense_weights_match() {
    let dense = make_dense(CH_FULL, CH_FULL);
    let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM).unwrap();

    for in_c in 0..CH_SLIM {
        for out_c in 0..CH_SLIM {
            let src_idx = in_c * CH_FULL + out_c;
            let dst_idx = in_c * CH_SLIM + out_c;
            assert_eq!(
                sliced.weights[dst_idx], dense.weights[src_idx],
                "mismatch at in_c={} out_c={}",
                in_c, out_c
            );
        }
    }
}

#[test]
fn test_slice_dense_bias_match() {
    let dense = make_dense(CH_FULL, CH_FULL);
    let sliced = slice_dense(&dense, CH_SLIM, CH_SLIM).unwrap();
    assert_eq!(&sliced.bias[..CH_SLIM], &dense.bias[..CH_SLIM]);
}

#[test]
fn test_slice_dense_asymmetric() {
    let dense = make_dense(8, 12);
    let sliced = slice_dense(&dense, 4, 6).unwrap();
    assert_eq!(sliced.in_ch, 4);
    assert_eq!(sliced.out_ch, 6);
    assert_eq!(sliced.weights.len(), 24);
    assert_eq!(sliced.bias.len(), 6);
    for in_c in 0..4usize {
        for out_c in 0..6usize {
            assert_eq!(
                sliced.weights[in_c * 6 + out_c],
                dense.weights[in_c * 12 + out_c]
            );
        }
    }
}

#[test]
fn test_slice_dense_bigger_in_ch_returns_err() {
    let dense = make_dense(CH_FULL, CH_FULL);
    let Err(err) = slice_dense(&dense, CH_FULL + 1, CH_FULL) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::InvalidInputChannels {
            requested: CH_FULL + 1,
            available: CH_FULL
        }
    );
}

#[test]
fn test_slice_dense_bigger_out_ch_returns_err() {
    let dense = make_dense(CH_FULL, CH_FULL);
    let Err(err) = slice_dense(&dense, CH_FULL, CH_FULL + 1) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::InvalidOutputChannels {
            requested: CH_FULL + 1,
            available: CH_FULL
        }
    );
}

// =====================================================================
// slice_wavenet_layer tests
// =====================================================================

#[test]
fn test_slice_wavenet_layer_dims() {
    let layer = make_wavenet_layer(CH_FULL);
    let sliced = slice_wavenet_layer(&layer, CH_SLIM).unwrap();

    assert_eq!(sliced.conv1d.in_ch, CH_SLIM);
    assert_eq!(sliced.conv1d.out_ch, CH_SLIM);
    assert_eq!(sliced.input_mixin.in_ch, 1);
    assert_eq!(sliced.input_mixin.out_ch, CH_SLIM);
    assert_eq!(sliced.one_by_one.in_ch, CH_SLIM);
    assert_eq!(sliced.one_by_one.out_ch, CH_SLIM);
    assert_eq!(sliced.scratch_mixin.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.scratch_conv.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
}

#[test]
fn test_slice_wavenet_layer_weights_preserved() {
    let layer = make_wavenet_layer(CH_FULL);
    let sliced = slice_wavenet_layer(&layer, CH_SLIM).unwrap();

    let conv_sliced = slice_conv1d(&layer.conv1d, CH_SLIM, CH_SLIM).unwrap();
    let mixin_sliced = slice_dense(&layer.input_mixin, 1, CH_SLIM).unwrap();
    let obo_sliced = slice_dense(&layer.one_by_one, CH_SLIM, CH_SLIM).unwrap();

    assert_eq!(&*sliced.conv1d.weights, &*conv_sliced.weights);
    assert_eq!(&*sliced.input_mixin.weights, &*mixin_sliced.weights);
    assert_eq!(&*sliced.one_by_one.weights, &*obo_sliced.weights);
}

// =====================================================================
// slice_wavenet_array tests
// =====================================================================

#[test]
fn test_slice_wavenet_array_dims() {
    let dilations = [1, 2, 4];
    let array = make_wavenet_array(1, CH_FULL, 4, &dilations);
    let mut alloc_num = 0;
    let sliced = slice_wavenet_array(&array, 1, CH_SLIM, &mut alloc_num).unwrap();

    assert_eq!(sliced.in_ch, 1);
    assert_eq!(sliced.ch, CH_SLIM);
    assert_eq!(sliced.head, 4);
    assert_eq!(sliced.cond, 1);
    assert_eq!(sliced.layers.len(), 3);
    assert_eq!(sliced.states.len(), 3);
    assert_eq!(sliced.rechannel.in_ch, 1);
    assert_eq!(sliced.rechannel.out_ch, CH_SLIM);
    assert_eq!(sliced.head_rechannel.in_ch, CH_SLIM);
    assert_eq!(sliced.head_rechannel.out_ch, 4);
    assert_eq!(sliced.array_outputs.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.head_accum.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.block_size, CH_SLIM);
    assert_eq!(sliced.effective_layers, 3);
}

#[test]
fn test_slice_wavenet_array_preserves_weights() {
    let dilations = [1, 2, 4];
    let array = make_wavenet_array(1, CH_FULL, 4, &dilations);
    let mut alloc_num = 0;
    let sliced = slice_wavenet_array(&array, 1, CH_SLIM, &mut alloc_num).unwrap();

    let rec_expected = slice_dense(&array.rechannel, 1, CH_SLIM).unwrap();
    assert_eq!(&*sliced.rechannel.weights, &*rec_expected.weights);

    for (i, (orig, slic)) in array.layers.iter().zip(sliced.layers.iter()).enumerate() {
        let conv_expected = slice_conv1d(&orig.conv1d, CH_SLIM, CH_SLIM).unwrap();
        assert_eq!(
            &*slic.conv1d.weights, &*conv_expected.weights,
            "conv1d mismatch at layer {}",
            i
        );
    }

    let head_expected = slice_dense(&array.head_rechannel, CH_SLIM, 4).unwrap();
    assert_eq!(&*sliced.head_rechannel.weights, &*head_expected.weights);
}

// =====================================================================
// slice_wavenet_model / slice_channels tests
// =====================================================================

#[test]
fn test_slice_wavenet_model_dims() {
    let model = make_full_model(CH_FULL, CH_SLIM);
    let sliced = slice_wavenet_model(&model, CH_SLIM).unwrap();

    assert_eq!(sliced.ch, CH_SLIM);
    assert_eq!(sliced.head, CH_SLIM);
    assert_eq!(sliced.k, TEST_KERNEL);
    assert_eq!(sliced.head_scale, 0.02);
    assert_eq!(sliced.arrays.len(), 2);

    assert_eq!(sliced.arrays[0].ch, CH_SLIM);
    assert_eq!(sliced.arrays[0].in_ch, 1);
    assert_eq!(sliced.arrays[1].ch, CH_SLIM);
    assert_eq!(sliced.arrays[1].in_ch, CH_SLIM);
    assert_eq!(sliced.arrays[0].effective_layers, 3);
    assert_eq!(sliced.arrays[1].effective_layers, 3);
}

#[test]
fn test_slice_wavenet_model_through_method() {
    let model = make_full_model(CH_FULL, CH_SLIM);
    let sliced = model.slice_channels(CH_SLIM).unwrap();
    assert_eq!(sliced.ch, CH_SLIM);
    assert_eq!(sliced.arrays.len(), 2);
    assert_eq!(sliced.arrays[0].ch, CH_SLIM);
    assert_eq!(sliced.arrays[1].ch, CH_SLIM);
}

#[test]
fn test_slice_wavenet_model_preserves_inference_shape() {
    let mut model = make_full_model(CH_FULL, CH_SLIM);
    let sliced = slice_wavenet_model(&model, CH_SLIM).unwrap();

    model.prewarm();

    let input = vec![0.5f32; 64];
    let mut output_full = vec![0.0f32; 64];
    let mut output_slim = vec![0.0f32; 64];

    model.process(&input, &mut output_full);

    let mut sliced_mut = sliced;
    sliced_mut.prewarm();
    sliced_mut.process(&input, &mut output_slim);

    assert_eq!(output_full.len(), output_slim.len());
}

#[test]
fn test_slice_wavenet_model_zero_ch_returns_err() {
    let model = make_full_model(CH_FULL, CH_SLIM);
    let Err(err) = slice_wavenet_model(&model, 0) else {
        panic!("expected Err");
    };
    assert_eq!(err, SlicingError::ZeroChannelCount);
}

#[test]
fn test_slice_wavenet_model_too_large_ch_returns_err() {
    let model = make_full_model(CH_FULL, CH_SLIM);
    let Err(err) = slice_wavenet_model(&model, CH_FULL + 1) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::RequestedChannelExceedsModel {
            requested: CH_FULL + 1,
            available: CH_FULL
        }
    );
}

#[test]
fn test_slice_wavenet_model_arrays_different_ch() {
    let model = make_full_model(8, 4);
    let sliced = slice_wavenet_model(&model, 4).unwrap();
    assert_eq!(sliced.ch, 4);
    assert_eq!(sliced.arrays[0].ch, 4);
    assert_eq!(sliced.arrays[0].in_ch, 1);
    assert_eq!(sliced.arrays[1].ch, 4);
    assert_eq!(sliced.arrays[1].in_ch, 4);
}

#[test]
fn test_slice_wavenet_model_exceeds_min_array_ch_returns_err() {
    let model = make_full_model(8, 4);
    let Err(err) = slice_wavenet_model(&model, 5) else {
        panic!("expected Err");
    };
    assert_eq!(
        err,
        SlicingError::RequestedChannelExceedsMinArray {
            requested: 5,
            min_array_ch: 4
        }
    );
}

#[test]
fn test_clone_exact_heterogeneous_model() {
    let model = make_full_model(8, 4);
    let cloned = model.clone_exact();
    assert_eq!(cloned.ch, 8);
    assert_eq!(cloned.arrays.len(), 2);
    assert_eq!(cloned.arrays[0].ch, 8);
    assert_eq!(cloned.arrays[1].ch, 4);
    assert_eq!(cloned.head_scale, model.head_scale);
}

#[test]
fn test_clone_wavenet_for_slimmable_storage_heterogeneous() {
    let model = make_full_model(8, 4);
    let storage_clone = crate::models::slimmable::clone_wavenet_for_slimmable_storage(&model);
    assert!(storage_clone.is_ok());
    if let Ok(boxed_sm) = storage_clone
        && let crate::models::StaticModel::WavenetDyn(m) = *boxed_sm
    {
        assert_eq!(m.ch, 8);
        assert_eq!(m.arrays[0].ch, 8);
        assert_eq!(m.arrays[1].ch, 4);
    } else {
        panic!("expected StaticModel::WavenetDyn");
    }
}
