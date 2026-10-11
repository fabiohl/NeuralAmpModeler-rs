// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::common::AlignedVec;
use crate::models::wavenet::{
    Conv1dDyn, DenseLayerDyn, HeadRechannelDyn, WAVENET_MAX_NUM_FRAMES, WaveNetLayerArrayDyn,
    WaveNetLayerDyn, WaveNetLayerState, WaveNetModelDyn,
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
        head_rechannel: HeadRechannelDyn::Dense(head_rechannel),
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
        prewarm_pending: false,
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
    let sliced = slice_wavenet_array(&array, 1, CH_SLIM, 4, &mut alloc_num).unwrap();

    assert_eq!(sliced.in_ch, 1);
    assert_eq!(sliced.ch, CH_SLIM);
    assert_eq!(sliced.head, 4);
    assert_eq!(sliced.cond, 1);
    assert_eq!(sliced.layers.len(), 3);
    assert_eq!(sliced.states.len(), 3);
    assert_eq!(sliced.rechannel.in_ch, 1);
    assert_eq!(sliced.rechannel.out_ch, CH_SLIM);
    match &sliced.head_rechannel {
        HeadRechannelDyn::Dense(dense) => {
            assert_eq!(dense.in_ch, CH_SLIM);
            assert_eq!(dense.out_ch, 4);
        }
        HeadRechannelDyn::Conv1d { .. } => panic!("expected Dense head_rechannel"),
    }
    assert_eq!(sliced.array_outputs.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.head_accum.len(), CH_SLIM * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.head_outputs.len(), 4 * WAVENET_MAX_NUM_FRAMES);
    assert_eq!(sliced.block_size, CH_SLIM);
    assert_eq!(sliced.effective_layers, 3);
}

#[test]
fn test_slice_wavenet_array_preserves_weights() {
    let dilations = [1, 2, 4];
    let array = make_wavenet_array(1, CH_FULL, 4, &dilations);
    let mut alloc_num = 0;
    let sliced = slice_wavenet_array(&array, 1, CH_SLIM, 4, &mut alloc_num).unwrap();

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

    let (head_weights, head_expected) = match (&array.head_rechannel, &sliced.head_rechannel) {
        (HeadRechannelDyn::Dense(orig), HeadRechannelDyn::Dense(slic)) => {
            let exp = slice_dense(orig, CH_SLIM, 4).unwrap();
            (&slic.weights, exp.weights)
        }
        _ => panic!("expected Dense head_rechannel"),
    };
    assert_eq!(&**head_weights, &*head_expected);
}

// =====================================================================
// slice_wavenet_model / slice_channels tests
// =====================================================================

#[test]
fn test_slice_wavenet_model_dims() {
    let model = make_full_model(CH_FULL, CH_SLIM);
    let sliced = slice_wavenet_model(&model, CH_SLIM).unwrap();

    assert_eq!(sliced.ch, CH_SLIM);
    assert_eq!(sliced.head, 1);
    assert_eq!(sliced.k, TEST_KERNEL);
    assert_eq!(sliced.head_scale, 0.02);
    assert_eq!(sliced.arrays.len(), 2);

    assert_eq!(sliced.arrays[0].ch, CH_SLIM);
    assert_eq!(sliced.arrays[0].in_ch, 1);
    assert_eq!(sliced.arrays[0].head, CH_SLIM);
    assert_eq!(sliced.arrays[1].ch, CH_SLIM);
    assert_eq!(sliced.arrays[1].in_ch, CH_SLIM);
    assert_eq!(sliced.arrays[1].head, 1);
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
        SlicingError::TargetExceedsArrayChannels {
            array_idx: 0,
            requested: CH_FULL + 1,
            available: CH_FULL,
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
        SlicingError::TargetExceedsArrayChannels {
            array_idx: 1,
            requested: 5,
            available: 4,
        }
    );
}

// =====================================================================
// T4.1c: F-20 reproduction & heterogeneous per-array slicing tests
// =====================================================================

/// (a) F-20 reproduction & verification:
/// Standard NAM WaveNet has array0 `ch=16, head=8` and array1 `ch=8, head=1`.
/// Slicing with per-array targets `[4, 4]` produces:
/// - array0: `in_ch=1, ch=4, head=4` (intermediate head matches next array channels!)
/// - array1: `in_ch=4, ch=4, head=1`
///
/// Verifies the chaining invariant: `prev.head == curr.ch == 4`, preventing
/// slice buffer length divergence during `tanh_and_accumulate_with_seed`.
#[test]
fn test_f20_chain_head_and_seed_alignment() {
    let dilations = [1, 2, 4];
    let array0 = make_wavenet_array(1, 16, 8, &dilations);
    let array1 = make_wavenet_array(16, 8, 1, &dilations);
    let rf = array0.receptive_field_size.max(array1.receptive_field_size);
    let model = WaveNetModelDyn {
        ch: 16,
        k: TEST_KERNEL,
        head: 1,
        arrays: vec![array0, array1],
        head_scale: 0.02,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        prewarm_on_reset: true,
        prewarm_pending: false,
        slimmable_capable: true,
        allowed_channels: None,
        pending_slim_channel: None,
    };

    let mut sliced = slice_wavenet_model_per_array(&model, &[4, 4]).unwrap();
    assert_eq!(sliced.ch, 4);
    assert_eq!(sliced.arrays[0].ch, 4);
    assert_eq!(
        sliced.arrays[0].head, 4,
        "Array 0 head must be chained to Array 1 ch"
    );
    assert_eq!(sliced.arrays[1].in_ch, 4);
    assert_eq!(sliced.arrays[1].ch, 4);
    assert_eq!(sliced.arrays[1].head, 1);

    // Buffer length invariants
    assert_eq!(
        sliced.arrays[0].head_outputs.len(),
        4 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(
        sliced.arrays[1].head_accum.len(),
        4 * WAVENET_MAX_NUM_FRAMES
    );

    // Inference executes without panic, bounds violation, or NaN/Inf
    sliced.prewarm();
    let input = vec![0.3f32; 128];
    let mut output = vec![0.0f32; 128];
    sliced.process(&input, &mut output);

    for &s in &output {
        assert!(s.is_finite(), "Output sample must be finite");
    }
}

/// (b) 2 arrays heterogêneos: buffer dimensions, chaining, and validation error cases.
#[test]
fn test_heterogeneous_per_array_slicing_and_validations() {
    let dilations = [1, 2, 4];
    let array0 = make_wavenet_array(1, 16, 8, &dilations);
    let array1 = make_wavenet_array(16, 8, 1, &dilations);
    let rf = array0.receptive_field_size.max(array1.receptive_field_size);
    let model = WaveNetModelDyn {
        ch: 16,
        k: TEST_KERNEL,
        head: 1,
        arrays: vec![array0, array1],
        head_scale: 0.02,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        prewarm_on_reset: true,
        prewarm_pending: false,
        slimmable_capable: true,
        allowed_channels: Some(vec![vec![4, 8, 16], vec![2, 4, 8]]),
        pending_slim_channel: None,
    };

    // Valid heterogeneous target: [8, 4]
    let sliced = slice_wavenet_model_per_array(&model, &[8, 4]).unwrap();
    assert_eq!(sliced.arrays[0].ch, 8);
    assert_eq!(sliced.arrays[0].in_ch, 1);
    assert_eq!(
        sliced.arrays[0].head, 4,
        "Intermediate head chains to target[1]"
    );
    assert_eq!(
        sliced.arrays[1].in_ch, 8,
        "Array 1 input chains to target[0]"
    );
    assert_eq!(sliced.arrays[1].ch, 4);
    assert_eq!(sliced.arrays[1].head, 1);

    // Buffer dimension verification
    assert_eq!(
        sliced.arrays[0].array_outputs.len(),
        8 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(
        sliced.arrays[0].head_accum.len(),
        8 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(
        sliced.arrays[0].head_outputs.len(),
        4 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(
        sliced.arrays[1].array_outputs.len(),
        4 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(
        sliced.arrays[1].head_accum.len(),
        4 * WAVENET_MAX_NUM_FRAMES
    );
    assert_eq!(sliced.arrays[1].head_outputs.len(), WAVENET_MAX_NUM_FRAMES);

    // Error case 1: Length mismatch
    let Err(err_len) = slice_wavenet_model_per_array(&model, &[8]) else {
        panic!("expected Err");
    };
    assert_eq!(
        err_len,
        SlicingError::ArrayCountMismatch {
            requested: 1,
            available: 2
        }
    );

    // Error case 2: Target exceeds loaded array channel count
    let Err(err_exceed) = slice_wavenet_model_per_array(&model, &[32, 4]) else {
        panic!("expected Err");
    };
    assert_eq!(
        err_exceed,
        SlicingError::TargetExceedsArrayChannels {
            array_idx: 0,
            requested: 32,
            available: 16
        }
    );

    // Error case 3: Target not in allowed list
    let Err(err_not_allowed) = slice_wavenet_model_per_array(&model, &[6, 4]) else {
        panic!("expected Err");
    };
    assert_eq!(
        err_not_allowed,
        SlicingError::TargetNotInAllowedList {
            array_idx: 0,
            requested: 6,
            allowed: vec![4, 8, 16]
        }
    );

    // Error case 4: Fixed array target mismatch
    let mut model_with_fixed = model.clone();
    model_with_fixed.allowed_channels = Some(vec![vec![4, 8, 16], vec![]]); // Array 1 fixed
    let Err(err_fixed) = slice_wavenet_model_per_array(&model_with_fixed, &[8, 4]) else {
        panic!("expected Err");
    };
    assert_eq!(
        err_fixed,
        SlicingError::FixedArrayTargetMismatch {
            array_idx: 1,
            requested: 4,
            fixed: 8
        }
    );

    // Valid fixed array target: array 1 retains 8
    let ok_fixed = slice_wavenet_model_per_array(&model_with_fixed, &[8, 8]).unwrap();
    assert_eq!(ok_fixed.arrays[0].ch, 8);
    assert_eq!(ok_fixed.arrays[0].head, 8);
    assert_eq!(ok_fixed.arrays[1].ch, 8);
}

/// (c) clone_exact and slice preserve allowed_channels and reset pending_slim_channel.
#[test]
fn test_clone_and_slice_preserve_allowed_channels() {
    let dilations = [1, 2, 4];
    let array0 = make_wavenet_array(1, 16, 8, &dilations);
    let array1 = make_wavenet_array(16, 8, 1, &dilations);
    let rf = array0.receptive_field_size.max(array1.receptive_field_size);
    let model = WaveNetModelDyn {
        ch: 16,
        k: TEST_KERNEL,
        head: 1,
        arrays: vec![array0, array1],
        head_scale: 0.02,
        receptive_field_size: rf,
        condition_dsp: None,
        condition_dsp_output: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        post_stack_head: None,
        head_output_scratch: AlignedVec::new(WAVENET_MAX_NUM_FRAMES, 0.0).unwrap(),
        prewarm_on_reset: true,
        prewarm_pending: false,
        slimmable_capable: true,
        allowed_channels: Some(vec![vec![4, 8, 16], vec![2, 4, 8]]),
        pending_slim_channel: Some(vec![8, 4]),
    };

    let cloned = model.clone_exact();
    assert_eq!(cloned.allowed_channels, model.allowed_channels);
    assert_eq!(cloned.pending_slim_channel, Some(vec![8, 4]));

    let sliced = slice_wavenet_model_per_array(&model, &[8, 4]).unwrap();
    assert_eq!(sliced.allowed_channels, model.allowed_channels);
    assert_eq!(
        sliced.pending_slim_channel, None,
        "Pending slim channel must be reset to None on sliced model"
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

// =====================================================================
// slimmable tier mapping / breakpoints
// =====================================================================

/// Tier mapping for `allowed_channels = [4, 8, 16]` (three tiers): the
/// `floor(ratio * num_tiers)` rule places each tier on an equal third of the
/// quality axis, so `0.30 -> 4` (the old `round(ratio * (n - 1))` rule wrongly
/// yielded `8`).
#[test]
fn test_slimmable_tier_index_four_eight_sixteen() {
    let allowed = [4usize, 8, 16];
    let pick = |r: f64| allowed[slimmable_tier_index(r, allowed.len())];

    assert_eq!(pick(0.0), 4);
    assert_eq!(pick(0.30), 4);
    assert_eq!(pick(0.34), 8);
    assert_eq!(pick(2.0 / 3.0), 16);
    assert_eq!(pick(0.99), 16);
    assert_eq!(pick(1.0), 16);
    assert_eq!(pick(2.0), 16, "out-of-range ratio clamps to the top tier");
    assert_eq!(pick(-1.0), 4, "negative ratio clamps to the bottom tier");
}

#[test]
fn test_slimmable_tier_index_monotonic_and_bounded() {
    for n in 1..=16usize {
        let mut prev = 0usize;
        for k in 0..=1000 {
            let r = k as f64 / 1000.0;
            let idx = slimmable_tier_index(r, n);
            assert!(idx < n, "index {idx} out of range for n={n}");
            assert!(idx >= prev, "non-monotonic at n={n}, r={r}");
            prev = idx;
        }
        assert_eq!(slimmable_tier_index(1.0, n), n - 1);
        assert_eq!(slimmable_tier_index(0.0, n), 0);
    }
}

/// T5.4-8: `f64` ratio preserves C++ `ratio_to_channels` parity at tier
/// boundaries that `f32` rounding corrupts. With `num_tiers = 10`, the exact
/// `0.7f64 * 10 = 7.0` selects tier 7, whereas the `f32`-rounded `0.7` widens
/// to `6.999…` and would select tier 6 (one below C++).
#[test]
fn test_slimmable_tier_index_f64_boundary_parity() {
    assert_eq!(slimmable_tier_index(0.7f64, 10), 7);
    assert_eq!(slimmable_tier_index(0.9f64, 10), 9);
    assert_eq!(slimmable_tier_index(0.35f64, 20), 7);
    assert_eq!(slimmable_tier_index(0.95f64, 20), 19);
    // The f32-rounded value is what previously drifted a tier lower.
    assert_eq!(slimmable_tier_index(0.7f32 as f64, 10), 6);
}

#[test]
fn test_slimmable_channel_breakpoints() {
    assert!(slimmable_channel_breakpoints(0).is_empty());
    assert!(slimmable_channel_breakpoints(1).is_empty());

    let bp3 = slimmable_channel_breakpoints(3);
    assert_eq!(&*bp3, &[1.0 / 3.0, 2.0 / 3.0]);

    let bp4 = slimmable_channel_breakpoints(4);
    assert_eq!(&*bp4, &[0.25, 0.5, 0.75]);

    for window in bp4.windows(2) {
        assert!(
            window[0] < window[1],
            "breakpoints must be strictly increasing"
        );
    }
}
