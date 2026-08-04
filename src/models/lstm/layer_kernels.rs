// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! LSTM layer kernel dispatch — SIMD processing paths and scalar fallback.

use core::arch::x86_64::*;

use super::layer::LstmLayer;
use crate::math::activations::ActivationPrecision;
use crate::math::activations::activation_precision;
use crate::math::activations::scalar_minimax_sigmoid;
use crate::math::activations::sigmoid::high_fidelity::scalar_sigmoid_poly;
use crate::math::activations::tanh::high_fidelity::scalar_tanh_poly;
use crate::math::activations::tanh::scalar_pade_tanh;

macro_rules! define_lstm_process {
    (
        $fn_name:ident,
        $target_meta:meta,
        $gemv_4gate:path,
        $step:expr,
        $load:ident,
        $store:ident,
        $add:ident,
        $mul:ident,
        $tanh:path,
        $sigmoid:path,
        $fused_gates:path,
    ) => {
        /// Processes a sample through the LSTM layer.
        ///
        /// # Safety
        /// The caller must guarantee support for the specified SIMD instructions.
        // NOTE: $target_meta allows injecting #[inline(always)] for AVX2 (our x86-64-v3 baseline)
        // or #[target_feature] for higher extensions, ensuring correct codegen.
        #[$target_meta]
        pub unsafe fn $fn_name(&mut self, input: &[f32]) {
            unsafe {
                // 1. Feed the model's 'memory' with the new audio fragment.
                self.state[..I].copy_from_slice(&input[..I]);

                // 2. 'Prefetch': Notify the processor to fetch the weights from RAM
                // slightly ahead of when they're needed, preventing the computation from stalling for data.
                _mm_prefetch::<{ _MM_HINT_T0 }>(self.state.as_ptr().cast::<i8>());

                // 3. Matrix-Vector Multiplication (GEMV):
                // Here we multiply the input and previous state by the weights (the trained 'brain').
                // The result activates the 4 'gates' of the LSTM: Forget, Input, Candidate, and Output.
                $gemv_4gate(
                    &self.state.0,
                    self.input_hidden_weights[0].as_flattened(),
                    self.input_hidden_weights[1].as_flattened(),
                    self.input_hidden_weights[2].as_flattened(),
                    self.input_hidden_weights[3].as_flattened(),
                    &self.bias.0,
                    &mut self.gates.0,
                    true,
                );

                // 4. Map where each of the 4 gates starts in our calculation buffer.
                let f_offset = H; // Forget Gate
                let g_offset = 2 * H; // Update Gate (Cell Candidate)
                let o_offset = 3 * H; // Output Gate
                let h_offset = I; // Where we store the final result for the next step

                let mut i = 0;
                // 5. Main Loop (SIMD): Process several neurons in parallel (8 or 16 at a time).
                while i + $step <= H {
                    // Load the pre-computed values of the 4 gates and current memory (cell state).
                    let g_f = $load(self.gates.as_ptr().add(i + f_offset));
                    let g_i = $load(self.gates.as_ptr().add(i));
                    let g_g = $load(self.gates.as_ptr().add(i + g_offset));
                    let g_o = $load(self.gates.as_ptr().add(i + o_offset));
                    let c_s = $load(self.cell_state.as_ptr().add(i));
                    let c_err = $load(self.cell_error.as_ptr().add(i));

                    // 'Fused Gates': The LSTM magic happens here.
                    // We decide what to forget from old memory and what to learn from the new input.
                    let (new_c_s, new_c_err, h_s) = $fused_gates(g_f, g_i, g_g, g_o, c_s, c_err);

                    // Save the new memory (long-term) and the new output (short-term).
                    $store(self.cell_state.as_mut_ptr().add(i), new_c_s);
                    $store(self.cell_error.as_mut_ptr().add(i), new_c_err);
                    $store(self.state.as_mut_ptr().add(h_offset + i), h_s);

                    i += $step;
                }

                // 6. 'Tail' Handling: If the number of neurons is not an exact multiple of the
                // parallel processing, we handle the last elements individually.
                if i < H {
                    let tail_len = H - i;
                    let mut temp_gf = [0.0; $step];
                    let mut temp_gi = [0.0; $step];
                    let mut temp_gg = [0.0; $step];
                    let mut temp_go = [0.0; $step];
                    let mut temp_cs = [0.0; $step];
                    let mut temp_cerr = [0.0; $step];

                    for j in 0..tail_len {
                        temp_gf[j] = self.gates[i + j + f_offset];
                        temp_gi[j] = self.gates[i + j];
                        temp_gg[j] = self.gates[i + j + g_offset];
                        temp_go[j] = self.gates[i + j + o_offset];
                        temp_cs[j] = self.cell_state[i + j];
                        temp_cerr[j] = self.cell_error[i + j];
                    }

                    let g_f = $load(temp_gf.as_ptr());
                    let g_i = $load(temp_gi.as_ptr());
                    let g_g = $load(temp_gg.as_ptr());
                    let g_o = $load(temp_go.as_ptr());
                    let c_s = $load(temp_cs.as_ptr());
                    let c_err = $load(temp_cerr.as_ptr());

                    let (new_c_s, new_c_err, h_val) = $fused_gates(g_f, g_i, g_g, g_o, c_s, c_err);

                    let mut out_cs = [0.0; $step];
                    let mut out_err = [0.0; $step];
                    let mut out_h = [0.0; $step];
                    $store(out_cs.as_mut_ptr(), new_c_s);
                    $store(out_err.as_mut_ptr(), new_c_err);
                    $store(out_h.as_mut_ptr(), h_val);

                    for j in 0..tail_len {
                        self.cell_state[i + j] = out_cs[j];
                        self.cell_error[i + j] = out_err[j];
                        self.state[i + j + h_offset] = out_h[j];
                    }
                }
            }
        }
    };
}

impl<const I: usize, const H: usize, const IH: usize, const H4: usize> LstmLayer<I, H, IH, H4> {
    // --- Hardware (SIMD) Specializations ---
    // We create different versions of the same logic to get the most out of each CPU.

    // 1. AVX2 Specialization (Default for x86-64-v3):
    // Uses 256-bit registers (processing 8 floats in parallel) with fast
    // Tanh and Sigmoid approximators via AVX2 SIMD.
    define_lstm_process!(
        process_sample_avx2,
        target_feature(enable = "avx2,fma,f16c"),
        crate::math::gemm::gemv_4gate_avx2,
        8,
        _mm256_loadu_ps,
        _mm256_storeu_ps,
        _mm256_add_ps,
        _mm256_mul_ps,
        crate::math::activations::simd_tanh_avx2,
        crate::math::activations::simd_sigmoid_avx2,
        crate::math::lstm::fused_lstm_gates_avx2,
    );

    // 2. AVX-512 (F/VL) Specialization:
    // Uses 512-bit registers (processing 16 floats at once). Ideal for
    // servers or newer Intel/AMD CPUs that support extended vector instructions.
    define_lstm_process!(
        process_sample_avx512,
        target_feature(enable = "avx512f,avx512vl"),
        crate::math::gemm::gemv_4gate_avx512,
        16,
        _mm512_loadu_ps,
        _mm512_storeu_ps,
        _mm512_add_ps,
        _mm512_mul_ps,
        crate::math::activations::simd_tanh_avx512,
        crate::math::activations::simd_sigmoid_avx512,
        crate::math::lstm::fused_lstm_gates_avx512,
    );

    /// Scalar processing (fallback) for tests and benchmarks.
    ///
    /// This is the 'manual' and slow version, used only as a reference to ensure
    /// the ultra-fast versions above have no mathematical errors.
    #[inline(always)]
    pub fn process_sample_scalar(&mut self, input: &[f32]) {
        let ih = I + H;
        let h = H;
        self.state[..I].copy_from_slice(&input[..I]);

        // Manual weight-input multiplication (4 gates).
        for k in 0..4 {
            let target_gate_offset = k * h;
            for i in 0..h {
                let mut sum = 0.0;
                for (j, &s) in self.state.iter().enumerate().take(ih) {
                    let w = self.input_hidden_weights[k][j][i];
                    sum += w * s;
                }
                self.gates[target_gate_offset + i] = sum + self.bias[target_gate_offset + i];
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
                self.state[I + j] = o * scalar_tanh_poly(new_cs);
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
                self.state[I + j] = o * scalar_pade_tanh(new_cs);
            }
        }
    }
}
