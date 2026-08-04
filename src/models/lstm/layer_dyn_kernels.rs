// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dynamic LSTM layer processing kernels — SIMD paths and scalar fallback.
//!
//! Reuses the existing `gemv_4gate_*` and `fused_lstm_gates_dyn_*` functions
//! from `crate::math::gemm` and `crate::math::lstm`, which infer dimensions
//! directly from slice lengths at runtime.

use crate::dispatch_simd;
use crate::math::activations::ActivationPrecision;
use crate::math::activations::activation_precision;
use crate::math::activations::scalar_minimax_sigmoid;
use crate::math::activations::sigmoid::high_fidelity::scalar_sigmoid_poly;
use crate::math::activations::tanh::high_fidelity::scalar_tanh_poly;
use crate::math::activations::tanh::scalar_pade_tanh;
use core::arch::x86_64::*;

use super::layer_dyn::LstmLayerDyn;

// =========================================================================
// Scalar fallback — reference implementation for validation
// =========================================================================

impl LstmLayerDyn {
    /// Scalar processing (fallback) for tests and benchmarks.
    ///
    /// This is the fully manual version, used as a reference to ensure that
    /// the SIMD-accelerated paths produce mathematically identical results.
    #[inline(always)]
    pub fn process_sample_scalar(&mut self, input: &[f32]) {
        let i = self.input_size;
        let h = self.hidden_size;
        let ih = i + h;

        self.state[..i].copy_from_slice(&input[..i]);

        // Manual weight-input multiplication (4 gates).
        for k in 0..4 {
            let target_gate_offset = k * h;
            let w_start = k * ih * h;
            for hi in 0..h {
                let mut sum = 0.0;
                for j in 0..ih {
                    let s = self.state[j];
                    let w = self.input_hidden_weights[w_start + j * h + hi];
                    sum += w * s;
                }
                self.gates[target_gate_offset + hi] = sum + self.bias[target_gate_offset + hi];
            }
        }

        // Manual activation of the LSTM gates.
        let is_hf = activation_precision() == ActivationPrecision::Standard;
        for j in 0..h {
            let gf = self.gates[j + h];
            let gi = self.gates[j];
            let gg = self.gates[j + 2 * h];
            let go = self.gates[j + 3 * h];
            let cs = self.cell_state[j];
            let cs_err = self.cell_error[j];

            if is_hf {
                let f = scalar_sigmoid_poly(gf);
                let in_gate = scalar_sigmoid_poly(gi);
                let g = scalar_tanh_poly(gg);
                let o = scalar_sigmoid_poly(go);

                let f_cs = f * cs;
                let f_err = f * cs_err;
                let i_g = in_gate * g;
                let y = i_g - f_err;
                let new_cs = f_cs + y;
                let new_cs_err = (new_cs - f_cs) - y;

                self.cell_state[j] = new_cs;
                self.cell_error[j] = new_cs_err;
                self.state[i + j] = o * scalar_tanh_poly(new_cs);
            } else {
                let f = scalar_minimax_sigmoid(gf);
                let in_gate = scalar_minimax_sigmoid(gi);
                let g = scalar_pade_tanh(gg);
                let o = scalar_minimax_sigmoid(go);

                let f_cs = f * cs;
                let f_err = f * cs_err;
                let i_g = in_gate * g;
                let y = i_g - f_err;
                let new_cs = f_cs + y;
                let new_cs_err = (new_cs - f_cs) - y;

                self.cell_state[j] = new_cs;
                self.cell_error[j] = new_cs_err;
                self.state[i + j] = o * scalar_pade_tanh(new_cs);
            }
        }
    }
}

// =========================================================================
// AVX2 specialization (x86-64-v3 baseline)
// =========================================================================

impl LstmLayerDyn {
    /// Processes a sample through the dynamic LSTM layer using AVX2+FMA+F16C.
    ///
    /// # Safety
    /// The caller must have verified AVX2+FMA+F16C CPU support.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn process_sample_avx2(&mut self, input: &[f32]) {
        let i = self.input_size;
        let h = self.hidden_size;
        let ih = i + h;
        let stride = ih * h;

        // 1. Copy audio sample into the consolidated state buffer.
        self.state[..i].copy_from_slice(&input[..i]);

        // 2. Prefetch state to minimise DRAM stalls during the GEMV.
        _mm_prefetch::<_MM_HINT_T0>(self.state.as_ptr().cast::<i8>());

        // 3. Compute the 4 LSTM gates via the AVX2 GEMV kernel.
        unsafe {
            crate::math::gemm::gemv_4gate_avx2(
                &self.state[..ih],
                &self.input_hidden_weights[0..stride],
                &self.input_hidden_weights[stride..2 * stride],
                &self.input_hidden_weights[2 * stride..3 * stride],
                &self.input_hidden_weights[3 * stride..4 * stride],
                &self.bias[..4 * h],
                &mut self.gates[..4 * h],
                true,
            );
        }

        // 4. Update cell state and hidden state via fused-gate kernel.
        let (gates_slice, _rest) = &mut self.gates.split_at_mut(4 * h);
        let (cell_slice, _) = &mut self.cell_state.split_at_mut(h);
        let (cell_err_slice, _) = &mut self.cell_error.split_at_mut(h);
        let hidden_slice = &mut self.state[i..];
        unsafe {
            crate::math::lstm::fused_lstm_gates_dyn_avx2(
                gates_slice,
                cell_slice,
                cell_err_slice,
                hidden_slice,
                h,
            );
        }
    }
}

// =========================================================================
// AVX-512 specialization (F+VL)
// =========================================================================

impl LstmLayerDyn {
    /// Processes a sample through the dynamic LSTM layer using AVX-512F+VL.
    ///
    /// # Safety
    /// The caller must have verified AVX-512F+VL CPU support.
    #[target_feature(enable = "avx512f,avx512vl")]
    pub unsafe fn process_sample_avx512(&mut self, input: &[f32]) {
        let i = self.input_size;
        let h = self.hidden_size;
        let ih = i + h;
        let stride = ih * h;

        self.state[..i].copy_from_slice(&input[..i]);

        _mm_prefetch::<_MM_HINT_T0>(self.state.as_ptr().cast::<i8>());

        unsafe {
            crate::math::gemm::gemv_4gate_avx512(
                &self.state[..ih],
                &self.input_hidden_weights[0..stride],
                &self.input_hidden_weights[stride..2 * stride],
                &self.input_hidden_weights[2 * stride..3 * stride],
                &self.input_hidden_weights[3 * stride..4 * stride],
                &self.bias[..4 * h],
                &mut self.gates[..4 * h],
                true,
            );
        }

        let (gates_slice, _rest) = &mut self.gates.split_at_mut(4 * h);
        let (cell_slice, _) = &mut self.cell_state.split_at_mut(h);
        let (cell_err_slice, _) = &mut self.cell_error.split_at_mut(h);
        let hidden_slice = &mut self.state[i..];
        unsafe {
            crate::math::lstm::fused_lstm_gates_dyn_avx512(
                gates_slice,
                cell_slice,
                cell_err_slice,
                hidden_slice,
                h,
            );
        }
    }
}

// =========================================================================
// Dispatch helper — selects the right kernel at call site
// =========================================================================

impl LstmLayerDyn {
    /// Dispatches to the highest available SIMD kernel based on global config.
    ///
    /// Calls the matching `#[target_feature]` method, which is safe because
    /// the global `SIMD_MATH` configuration was validated against the CPU
    /// at application startup.
    #[inline]
    pub fn process(&mut self, input: &[f32]) {
        unsafe {
            dispatch_simd!(
                @self,
                process_sample_avx512,
                process_sample_avx512,
                process_sample_avx2,
                input
            );
        }
    }
}
