// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dual-Frame Processing of Causal WaveNet Convolution.
//!
//! Extension of `Conv1d` with methods that process two frames simultaneously
//! (Temporal Tiling), maximizing weight reuse in registers.
//!
//! ## Cohesion justification (no split)
//!
//! This file is a cohesive unit: a single `impl Conv1d` block extending the
//! static convolution with dual-frame (Temporal Tiling) processing.

use super::conv_input::{store_4_accums, store_8_accums, store_16_accums};
use super::conv1d::Conv1d;
use crate::loader::dispatcher::wavenet::layout::select_interleave_width;
use crate::math::common::{SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple};

impl<const IN: usize, const OUT: usize, const K: usize> Conv1d<IN, OUT, K> {
    /// Fused variant that processes two frames simultaneously, adding Mixin vectors
    /// (conditioning) directly to the accumulators.
    /// This approach maximizes the utilization of weights loaded into registers (Temporal Tiling).
    ///
    /// Uses full-precision f32 weights via `M::dot_product_*_f32_dual_accumulate`
    /// (AVX2/FMA or AVX-512 kernel) with fused bias+mixin initialization and
    /// K-tap accumulation in a single SIMD kernel call.
    ///
    /// # Safety
    /// `layer_buffer` and `mixin` must have appropriate sizes.
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "WaveNet dual conv1d kernel requiring many dimension/stride parameters for causal dilated convolution"
    )]
    pub unsafe fn process_dual_frame_with_mixin<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame_f0: &mut [f32],
        out_frame_f1: &mut [f32],
        frame_idx_f0: usize,
        frame_idx_f1: usize,
        mixin_f0: &[f32],
        mixin_f1: &[f32],
    ) {
        let interleave_width = select_interleave_width(OUT);
        let num_blocks = OUT.div_ceil(interleave_width);

        let mut in_taps_f0 = [[0.0f32; IN]; K];
        let mut in_taps_f1 = [[0.0f32; IN]; K];
        for k in 0..K {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (K as isize));
            let in_start_f0 = ((frame_idx_f0 as isize) + offset) as usize * IN;
            let in_start_f1 = ((frame_idx_f1 as isize) + offset) as usize * IN;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    layer_buffer.as_ptr().add(in_start_f0),
                    in_taps_f0[k].as_mut_ptr(),
                    IN,
                );
                std::ptr::copy_nonoverlapping(
                    layer_buffer.as_ptr().add(in_start_f1),
                    in_taps_f1[k].as_mut_ptr(),
                    IN,
                );
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(
                        layer_buffer.as_ptr().add(in_start_f0),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                } else {
                    prefetch_strategy_simple(
                        layer_buffer.as_ptr().add(in_start_f0),
                        self.dilation * IN,
                        k,
                        K,
                        self.dilation,
                    );
                }
            }
        }

        let flat_taps_f0: &[f32] =
            unsafe { core::slice::from_raw_parts(in_taps_f0.as_ptr() as *const f32, K * IN) };
        let flat_taps_f1: &[f32] =
            unsafe { core::slice::from_raw_parts(in_taps_f1.as_ptr() as *const f32, K * IN) };

        for b in 0..num_blocks {
            let out_c = b * interleave_width;
            let w = interleave_width.min(OUT - out_c);
            let w_start = b * K * IN * interleave_width;

            match interleave_width {
                16 => {
                    let mut init_f0 = [0.0f32; 16];
                    let mut init_f1 = [0.0f32; 16];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        if self.do_bias {
                            *item_f0 = self.bias[out_c + j] + mixin_f0[out_c + j];
                            *item_f1 = self.bias[out_c + j] + mixin_f1[out_c + j];
                        } else {
                            *item_f0 = mixin_f0[out_c + j];
                            *item_f1 = mixin_f1[out_c + j];
                        }
                    }
                    let w_slice: &[[f32; 16]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_16x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe { store_16_accums(out_frame_f0, out_c, r_f0, OUT) };
                    unsafe { store_16_accums(out_frame_f1, out_c, r_f1, OUT) };
                }
                8 => {
                    let mut init_f0 = [0.0f32; 8];
                    let mut init_f1 = [0.0f32; 8];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        if self.do_bias {
                            *item_f0 = self.bias[out_c + j] + mixin_f0[out_c + j];
                            *item_f1 = self.bias[out_c + j] + mixin_f1[out_c + j];
                        } else {
                            *item_f0 = mixin_f0[out_c + j];
                            *item_f1 = mixin_f1[out_c + j];
                        }
                    }
                    let w_slice: &[[f32; 8]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_8x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe { store_8_accums(out_frame_f0, out_c, r_f0, OUT) };
                    unsafe { store_8_accums(out_frame_f1, out_c, r_f1, OUT) };
                }
                _ => {
                    let mut init_f0 = [0.0f32; 4];
                    let mut init_f1 = [0.0f32; 4];
                    for (j, (item_f0, item_f1)) in init_f0
                        .iter_mut()
                        .zip(init_f1.iter_mut())
                        .enumerate()
                        .take(w)
                    {
                        if self.do_bias {
                            *item_f0 = self.bias[out_c + j] + mixin_f0[out_c + j];
                            *item_f1 = self.bias[out_c + j] + mixin_f1[out_c + j];
                        } else {
                            *item_f0 = mixin_f0[out_c + j];
                            *item_f1 = mixin_f1[out_c + j];
                        }
                    }
                    let w_slice: &[[f32; 4]] = unsafe {
                        let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                        core::slice::from_raw_parts(ptr, K * IN)
                    };
                    let (r_f0, r_f1) = unsafe {
                        M::dot_product_4x_f32_dual_accumulate(
                            w_slice,
                            flat_taps_f0,
                            flat_taps_f1,
                            &init_f0,
                            &init_f1,
                        )
                    };
                    unsafe { store_4_accums(out_frame_f0, out_c, r_f0, OUT) };
                    unsafe { store_4_accums(out_frame_f1, out_c, r_f1, OUT) };
                }
            }
        }
    }
}
