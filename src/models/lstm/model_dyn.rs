// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic LSTM model with arbitrary number of layers and hidden size.
//!
//! This is the runtime fallback for topologies that do not match the 10 static
//! const-generic profiles (`LstmModel1` / `LstmModel2`).

use super::layer_dyn::LstmLayerDyn;
use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

/// LSTM model with dynamically-allocated layers and head weights.
///
/// Composed of a `Vec<LstmLayerDyn>` chained sequentially:
///   - Layer 0 receives the mono audio input (input_size=1).
///   - Layers 1..N receive the previous layer's hidden state as input.
///   - Final output = `dot(hidden_last, head_weights_f32) + head_bias`.
///
/// ## SIMD
/// `process()` calls `process_avx2` only. There is no AVX-512 or BF16
/// arm on the dynamic LSTM path.
pub struct LstmModelDyn {
    /// Number of input channels (audio frames are interleaved with this stride).
    pub in_channels: usize,
    /// Number of output channels (audio frames are emitted with this stride).
    pub out_channels: usize,
    /// Dynamically-sized LSTM layers.
    ///
    /// Layer 0 has `input_size=in_channels`, subsequent layers have
    /// `input_size=hidden_size` (stacked recurrent input).
    pub layers: Vec<LstmLayerDyn>,
    /// Output head (linear projection) weights, shaped `[out_channels × hidden_size]`.
    pub head_weights_f32: AlignedVec<f32>,
    /// Output head biases, shaped `[out_channels]`.
    pub head_bias: AlignedVec<f32>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Expected sample rate (Hz) for prewarm calculation. Default: `48000.0`.
    pub expected_sample_rate: f64,
    /// Zeroed-sample stabilization work still pending for the deferred split
    /// pass armed by [`Self::prewarm_reset`](super::NamModel::prewarm_reset).
    /// Always `0` for freshly built models; the integral
    /// [`Self::prewarm`](super::NamModel::prewarm) / [`Self::reset`](super::NamModel::reset)
    /// paths neither consult nor alter it.
    pub prewarm_pending: usize,
}

use crate::loader::nam_json::validation::{
    MAX_LSTM_HIDDEN_SIZE, MAX_LSTM_IO_CHANNELS, MAX_LSTM_LAYERS,
};

impl LstmModelDyn {
    /// Creates a new zero-initialized dynamic LSTM model with custom channel geometry.
    ///
    /// Allocates `num_layers` × `LstmLayerDyn` with the appropriate
    /// input_size (`in_channels` for layer 0, `hidden_size` for subsequent layers) and
    /// pre-allocates `head_weights_f32` and `head_bias`.
    pub fn new_io(
        num_layers: usize,
        hidden_size: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self, NamErrorCode> {
        if num_layers == 0 || num_layers > MAX_LSTM_LAYERS {
            return Err(NamErrorCode::InvalidFeatureTopology);
        }
        if hidden_size == 0 || hidden_size > MAX_LSTM_HIDDEN_SIZE {
            return Err(NamErrorCode::InvalidFeatureTopology);
        }
        if in_channels == 0 || in_channels > MAX_LSTM_IO_CHANNELS {
            return Err(NamErrorCode::InvalidFeatureTopology);
        }
        if out_channels == 0 || out_channels > MAX_LSTM_IO_CHANNELS {
            return Err(NamErrorCode::InvalidFeatureTopology);
        }

        let head_len = out_channels
            .checked_mul(hidden_size)
            .ok_or(NamErrorCode::InvalidFeatureTopology)?;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let input_size = if i == 0 { in_channels } else { hidden_size };
            layers.push(LstmLayerDyn::new(input_size, hidden_size)?);
        }

        Ok(Self {
            in_channels,
            out_channels,
            layers,
            head_weights_f32: AlignedVec::new(head_len, 0.0f32)?,
            head_bias: AlignedVec::new(out_channels, 0.0f32)?,
            prewarm_on_reset: true,
            expected_sample_rate: 48000.0,
            prewarm_pending: 0,
        })
    }

    /// Creates a new zero-initialized dynamic LSTM model with mono 1→1 I/O.
    pub fn new(num_layers: usize, hidden_size: usize) -> Result<Self, NamErrorCode> {
        Self::new_io(num_layers, hidden_size, 1, 1)
    }

    // =====================================================================
    // AVX2 specialization (x86-64-v3 baseline)
    // =====================================================================

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        if self.layers.is_empty() || self.in_channels == 0 || self.out_channels == 0 {
            return;
        }
        let in_ch = self.in_channels;
        let out_ch = self.out_channels;
        let n_layers = self.layers.len();
        let n_frames = (input.len() / in_ch).min(output.len() / out_ch);

        // SAFETY: `layers` is non-empty (early return above), so `layers_ptr`
        // points to a live Vec of `n_layers` elements; the indices 0, `i-1`, `i`
        // and `n_layers-1` for `i in 1..n_layers` are all in bounds, and each
        // deref aliases a distinct layer slot. `n_frames` ensures all input and
        // output slice accesses remain strictly in bounds.
        // AVX2+FMA+F16C are guaranteed by `#[target_feature]`.
        unsafe {
            let layers_ptr = self.layers.as_mut_ptr();

            for s in 0..n_frames {
                let frame_in = &input[s * in_ch..(s + 1) * in_ch];
                (*layers_ptr).process_sample_avx2(frame_in);

                for i in 1..n_layers {
                    let prev = &*layers_ptr.add(i - 1);
                    let hidden = &prev.state[prev.input_size..];
                    (*layers_ptr.add(i)).process_sample_avx2(hidden);
                }

                let last = &*layers_ptr.add(n_layers - 1);
                let h = last.get_hidden_state();
                let h_len = h.len();
                for c in 0..out_ch {
                    let w = &self.head_weights_f32[c * h_len..(c + 1) * h_len];
                    let dot = crate::math::common::scalar_ref::dot_product_f32_native_kahan4(h, w);
                    output[s * out_ch + c] = dot + self.head_bias[c];
                }
            }
        }
    }

    // =====================================================================
    // Dispatch hub
    // =====================================================================

    /// Processes an audio block through the dynamic LSTM model.
    ///
    /// Routes to the baseline x86-64-v3 AVX2 kernel (dynamic LSTM is a fallback
    /// topology where AVX-512 duplication has low ROI).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        if self.in_channels == 0 || self.out_channels == 0 {
            return;
        }
        let n_frames = (input.len() / self.in_channels).min(output.len() / self.out_channels);
        let in_len = n_frames * self.in_channels;
        let out_len = n_frames * self.out_channels;
        let input = &input[..in_len];
        let output = &mut output[..out_len];
        // SAFETY: the slices were re-sliced to `n_frames * channels`, satisfying
        // `process_avx2`'s length contract; the AVX2 kernel is reachable only
        // on x86-64-v3 targets.
        unsafe {
            self.process_avx2(input, output);
        }
    }

    /// Scalar processing (fallback).
    ///
    /// Exclusively for parity tests.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        if self.layers.is_empty() || self.in_channels == 0 || self.out_channels == 0 {
            return;
        }
        let in_ch = self.in_channels;
        let out_ch = self.out_channels;
        let n_layers = self.layers.len();
        let n_frames = (input.len() / in_ch).min(output.len() / out_ch);

        let layers_ptr = self.layers.as_mut_ptr();

        let hidden_size = if n_layers > 1 {
            self.layers[0].hidden_size
        } else {
            0
        };
        let mut hidden_scratch = vec![0.0f32; hidden_size];

        for s in 0..n_frames {
            // SAFETY: `layers` is non-empty (early return above), so `layers_ptr`
            // points to a live Vec of `n_layers` elements; indices 0, `i-1`, `i`
            // and `n_layers-1` for `i in 1..n_layers` are in bounds.
            unsafe {
                let frame_in = &input[s * in_ch..(s + 1) * in_ch];
                (*layers_ptr).process_sample_scalar(frame_in);

                for i in 1..n_layers {
                    let prev = &*layers_ptr.add(i - 1);
                    let hidden = &prev.state[prev.input_size..];
                    hidden_scratch[..hidden.len()].copy_from_slice(hidden);
                    (*layers_ptr.add(i)).process_sample_scalar(&hidden_scratch[..hidden.len()]);
                }

                let last = &*layers_ptr.add(n_layers - 1);
                let hidden_last = last.get_hidden_state();
                let h_len = hidden_last.len();
                for c in 0..out_ch {
                    let w = &self.head_weights_f32[c * h_len..(c + 1) * h_len];
                    let dot = crate::math::common::scalar_ref::dot_product_f32_native_kahan4(
                        hidden_last,
                        w,
                    );
                    output[s * out_ch + c] = dot + self.head_bias[c];
                }
            }
        }
    }

    /// Resets all layers' internal states (hidden, cell, gates) to zero.
    pub fn reset_states(&mut self) {
        for layer in &mut self.layers {
            layer.reset_states();
        }
    }

    /// Resets only the input slots of all layers, preserving hidden and cell state.
    ///
    /// Used during prewarm to avoid discarding the initial `_xh` and `_c` states
    /// loaded from the NAM file.
    pub fn reset_input_slots(&mut self) {
        for layer in &mut self.layers {
            layer.reset_input_slot();
        }
    }
}
