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
///   - Final output = `dot(hidden_last, head_weights) + head_bias`.
///
/// ## SIMD
/// `process()` calls `process_avx2` only. There is no AVX-512 or BF16
/// arm on the dynamic LSTM path.
pub struct LstmModelDyn {
    /// Dynamically-sized LSTM layers.
    ///
    /// Layer 0 has `input_size=1` (mono audio), subsequent layers have
    /// `input_size=hidden_size` (stacked recurrent input).
    pub layers: Vec<LstmLayerDyn>,
    /// Output head (linear projection) weights.
    pub head_weights: AlignedVec<f32>,
    /// Output head weights.
    pub head_weights_f32: AlignedVec<f32>,
    /// Output head bias.
    pub head_bias: f32,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Expected sample rate (Hz) for prewarm calculation. Default: `48000.0`.
    pub expected_sample_rate: f64,
}

impl LstmModelDyn {
    /// Creates a new zero-initialized dynamic LSTM model.
    ///
    /// Allocates `num_layers` × `LstmLayerDyn` with the appropriate
    /// input_size (1 for layer 0, `hidden_size` for the rest) and
    /// pre-allocates `head_weights` / `head_weights_f32`.
    pub fn new(num_layers: usize, hidden_size: usize) -> Result<Self, NamErrorCode> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let input_size = if i == 0 { 1 } else { hidden_size };
            layers.push(LstmLayerDyn::new(input_size, hidden_size)?);
        }

        Ok(Self {
            layers,
            head_weights: AlignedVec::new(hidden_size, 0.0f32)?,
            head_weights_f32: AlignedVec::new(hidden_size, 0.0f32)?,
            head_bias: 0.0,
            prewarm_on_reset: true,
            expected_sample_rate: 48000.0,
        })
    }

    // =====================================================================
    // AVX2 specialization (x86-64-v3 baseline)
    // =====================================================================

    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn process_avx2(&mut self, input: &[f32], output: &mut [f32]) {
        if self.layers.is_empty() {
            return;
        }
        // SAFETY: `layers` is non-empty (early return above), so `layers_ptr`
        // points to a live Vec of `n_layers` elements; the indices 0, `i-1`, `i`
        // and `n_layers-1` for `i in 1..n_layers` are all in bounds, and each
        // deref aliases a distinct layer slot. The caller guarantees
        // `output.len() >= input.len()`, so every `output[s]` write (s <
        // `input.len()`) is in bounds; `input[s]` is read in bounds by iteration.
        // AVX2+FMA+F16C are guaranteed by `#[target_feature]`.
        unsafe {
            let n_layers = self.layers.len();
            debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
            let layers_ptr = self.layers.as_mut_ptr();

            for (s, &val) in input.iter().enumerate() {
                (*layers_ptr).process_sample_avx2(&[val]);

                for i in 1..n_layers {
                    let prev = &*layers_ptr.add(i - 1);
                    let hidden = &prev.state[prev.input_size..];
                    (*layers_ptr.add(i)).process_sample_avx2(hidden);
                }

                let last = &*layers_ptr.add(n_layers - 1);
                let h = last.get_hidden_state();
                let dot = crate::math::common::scalar_ref::dot_product_f32_native_kahan4(
                    h,
                    &self.head_weights_f32,
                );
                output[s] = dot + self.head_bias;
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
        // Length contract: clamp to the shorter buffer; never index past
        // `output.len()` even with asymmetric caller buffers.
        let n = input.len().min(output.len());
        let input = &input[..n];
        let output = &mut output[..n];
        // SAFETY: the slices were re-sliced to the common length `n`, so
        // `input.len() == output.len()`, satisfying `process_avx2`'s length
        // contract; the AVX2 kernel is reachable only on x86-64-v3 targets.
        unsafe {
            self.process_avx2(input, output);
        }
    }

    /// Scalar processing (fallback).
    ///
    /// Exclusively for parity tests. Extremely slow.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        if self.layers.is_empty() {
            return;
        }
        let n_layers = self.layers.len();
        let n = input.len().min(output.len());

        debug_assert!(n_layers > 0, "LstmModelDyn requires at least one layer");
        let layers_ptr = self.layers.as_mut_ptr();

        for s in 0..n {
            // SAFETY: `layers` is non-empty (early return above), so `layers_ptr`
            // points to a live Vec of `n_layers` elements; indices 0, `i-1`, `i`
            // and `n_layers-1` for `i in 1..n_layers` are in bounds. `s < n <=
            // input.len() = output.len()` (clamped above), so the reads/writes on
            // the slice bounds are in bounds.
            unsafe {
                (*layers_ptr).process_sample_scalar(&[input[s]]);

                for i in 1..n_layers {
                    let prev = &*layers_ptr.add(i - 1);
                    let hidden = &prev.state[prev.input_size..];
                    let hidden_copy: Vec<f32> = hidden.to_vec();
                    (*layers_ptr.add(i)).process_sample_scalar(&hidden_copy);
                }

                let last = &*layers_ptr.add(n_layers - 1);
                let hidden_last = last.get_hidden_state();
                let dot = crate::math::common::scalar_ref::dot_product_f32_native_kahan4(
                    hidden_last,
                    &self.head_weights_f32,
                );
                output[s] = dot + self.head_bias;
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
