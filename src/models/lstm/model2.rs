// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! 2-layer LSTM model with pipelining and SIMD dispatch.

use super::layer::LstmLayer;

macro_rules! define_lstm2_process_pipelined {
    (
        $fn_name:ident,
        $target_meta:meta,
        $layer_proc:ident,
    ) => {
        #[$target_meta]
        unsafe fn $fn_name(&mut self, input: &[f32], output: &mut [f32]) {
            unsafe {
                let len = input.len();
                if len >= 1 {
                    self.layer1.$layer_proc(&[input[0]]);
                    let mut prev_h1 = [0.0; H];
                    prev_h1.copy_from_slice(self.layer1.get_hidden_state());

                    for i in 1..len {
                        self.layer1.$layer_proc(&[input[i]]);
                        self.layer2.$layer_proc(&prev_h1);
                        let h_f32 = self.layer2.get_hidden_state();
                        output[i - 1] =
                            $crate::math::common::scalar_ref::dot_product_f32_native_kahan(
                                h_f32,
                                &self.head_weights_f32,
                            ) + self.head_bias;
                        prev_h1.copy_from_slice(self.layer1.get_hidden_state());
                    }

                    self.layer2.$layer_proc(&prev_h1);
                    let h_f32 = self.layer2.get_hidden_state();
                    output[len - 1] =
                        $crate::math::common::scalar_ref::dot_product_f32_native_kahan(
                            h_f32,
                            &self.head_weights_f32,
                        ) + self.head_bias;
                }
            }
        }
    };
}

/// 2-layer LSTM model.
pub struct LstmModel2<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> {
    /// Model layer 1.
    pub layer1: LstmLayer<1, H, H1_IH, H_H4>,
    /// Model layer 2.
    pub layer2: LstmLayer<H, H, H2_IH, H_H4>,
    /// Output head weights.
    pub head_weights: [f32; H],
    /// Output head weights.
    pub head_weights_f32: [f32; H],
    /// Output head bias.
    pub head_bias: f32,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Expected sample rate (Hz) for prewarm calculation. Default: `48000.0`.
    pub expected_sample_rate: f64,
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize>
    LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    /// Creates a new 2-layer LSTM model.
    pub fn new() -> Self {
        Self {
            layer1: LstmLayer::new(),
            layer2: LstmLayer::new(),
            head_weights: [0.0f32; H],
            head_weights_f32: [0.0f32; H],
            head_bias: 0.0,
            prewarm_on_reset: true,
            expected_sample_rate: 48000.0,
        }
    }
    define_lstm2_process_pipelined!(
        process_avx2,
        target_feature(enable = "avx2,fma,f16c"),
        process_sample_avx2,
    );

    define_lstm2_process_pipelined!(
        process_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        process_sample_avx512,
    );
    /// Processes an audio block through the model (SIMD dispatch).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // Length contract: clamp to the shorter buffer; never index past
        // `output.len()` even with asymmetric caller buffers.
        let n = input.len().min(output.len());
        let input = &input[..n];
        let output = &mut output[..n];
        unsafe {
            crate::math::common::dispatch_simd!(
                @self,
                process_avx512,
                process_avx512,
                process_avx2,
                input,
                output
            );
        }
    }
    /// Scalar processing (fallback).
    ///
    /// # Note
    /// Exclusively for parity tests. Extremely slow.
    pub fn process_scalar(&mut self, input: &[f32], output: &mut [f32]) {
        let n = input.len().min(output.len());
        for i in 0..n {
            self.layer1.process_sample_scalar(&[input[i]]);
            self.layer2
                .process_sample_scalar(self.layer1.get_hidden_state());
            let hidden2 = self.layer2.get_hidden_state();
            let dot = crate::math::common::scalar_ref::dot_product_f32_native_kahan(
                hidden2,
                &self.head_weights_f32,
            );
            output[i] = dot + self.head_bias;
        }
    }
    /// Resets the internal states.
    pub fn reset_states(&mut self) {
        self.layer1.reset_states();
        self.layer2.reset_states();
    }
}

impl<const H: usize, const H1_IH: usize, const H2_IH: usize, const H_H4: usize> Default
    for LstmModel2<H, H1_IH, H2_IH, H_H4>
{
    fn default() -> Self {
        Self::new()
    }
}
