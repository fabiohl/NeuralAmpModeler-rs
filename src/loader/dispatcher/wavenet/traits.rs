// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::math::common::AlignedVec;
use crate::models::wavenet::Conv1dDyn;
use crate::models::wavenet::{Conv1d, DenseLayer, DenseLayerDyn};

use super::layout::select_interleave_width;

/// Output type for convolution weights, unifying `Conv1d<IN,OUT,K>` and `Conv1dDyn`.
pub(crate) trait ConvWeightsOutput: Sized {
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        k_size: usize,
    ) -> Self;
}

impl<const IN: usize, const OUT: usize, const K: usize> ConvWeightsOutput for Conv1d<IN, OUT, K> {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        _in_ch: usize,
        _out_ch: usize,
        _k_size: usize,
    ) -> Self {
        let interleave_width = select_interleave_width(OUT);
        let num_blocks = OUT.div_ceil(interleave_width);
        let padded_total = num_blocks * interleave_width * IN * K;
        assert!(
            weights.len() >= padded_total,
            "Conv1d weights buffer is too small"
        );
        Conv1d {
            weights,
            bias,
            do_bias,
            dilation,
        }
    }
}

impl ConvWeightsOutput for Conv1dDyn {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        dilation: usize,
        in_ch: usize,
        out_ch: usize,
        k_size: usize,
    ) -> Self {
        let interleave_width = select_interleave_width(out_ch);
        let num_blocks_effective = out_ch.div_ceil(interleave_width);
        let padded_total = num_blocks_effective * interleave_width * in_ch * k_size;
        assert!(
            weights.len() >= padded_total,
            "Conv1d weights buffer is too small"
        );
        Conv1dDyn {
            weights,
            bias,
            do_bias,
            dilation,
            in_ch,
            out_ch,
            num_blocks: out_ch.div_ceil(4),
            interleave_width,
            kernel: k_size,
        }
    }
}

/// Output type for dense layer weights, unifying `DenseLayer<IN,OUT>` and `DenseLayerDyn`.
pub(crate) trait DenseWeightsOutput: Sized {
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
    ) -> Self;
}

impl<const IN: usize, const OUT: usize> DenseWeightsOutput for DenseLayer<IN, OUT> {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        _in_size: usize,
        _out_size: usize,
    ) -> Self {
        DenseLayer {
            weights,
            bias,
            do_bias,
        }
    }
}

impl DenseWeightsOutput for DenseLayerDyn {
    #[inline(always)]
    fn from_parts(
        weights: AlignedVec<f32>,
        bias: AlignedVec<f32>,
        do_bias: bool,
        in_size: usize,
        out_size: usize,
    ) -> Self {
        DenseLayerDyn {
            in_ch: in_size,
            out_ch: out_size,
            weights,
            bias,
            do_bias,
        }
    }
}
