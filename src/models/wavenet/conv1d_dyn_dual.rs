// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Dual-frame convolution for the dynamic (runtime-dimensional) WaveNet path.
//!
//! # Performance Rationale — Scalar Accumulation vs Static Path
//!
//! Unlike the static path (`conv1d_dual.rs`), which uses fused SIMD K-tap accumulation
//! kernels (`dot_product_*x_f32_accumulate`) that keep accumulators in YMM registers
//! across K taps and perform a single reduction + store, this dynamic path is limited
//! to **scalar-init accumulators** (`[0.0f32; iw]`) because the channel count (`out_ch`)
//! is not known at compile-time and cannot drive LLVM const-propagation.
//!
//! Each tap accumulates directly via raw pointer into the per-tap slice without
//! copying tap data into intermediate buffers.
//!
//! Remaining overhead vs static path: ~38% slower per block at CH5/64samp.
//! Root cause: scalar acc init + K scalar partial sums per interleave block.
//! Future optimization: const-generic dispatch for common non-standard shapes
//! at the loader level (see `model_dyn.rs`).
//!
//! **Decision**: not a priority. Dynamic models are rare in the NAM ecosystem.
//! The standard trainers produce CH=3/8 (A2) or CH=4/8/12/16 (A1), which all
//! hit const-generic static paths. Revisit only if non-standard shapes become common.

use super::common::MAX_KERNEL;
use super::conv1d_dyn::Conv1dDyn;
use crate::math::common::{SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple};

impl Conv1dDyn {
    #[inline(always)]
    #[expect(
        clippy::too_many_arguments,
        reason = "WaveNet dynamic dual conv1d kernel requiring many dimension/stride parameters for adaptive dilated convolution"
    )]
    pub(crate) unsafe fn process_dual_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_f0: &mut [f32],
        out_f1: &mut [f32],
        idx_f0: usize,
        idx_f1: usize,
        mixin_f0: Option<&[f32]>,
        mixin_f1: Option<&[f32]>,
    ) {
        let iw = self.interleave_width;
        let num_blocks = self.out_ch.div_ceil(iw);
        let mut tap_ptrs_f0 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let mut tap_ptrs_f1 = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = self.kernel.min(MAX_KERNEL);

        for (k, (tap_f0, tap_f1)) in tap_ptrs_f0
            .iter_mut()
            .zip(tap_ptrs_f1.iter_mut())
            .enumerate()
            .take(k_limit)
        {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (self.kernel as isize));
            // F-01: same warm-up clamp as `Conv1dDyn::process_single_frame` —
            // `idx + offset` below the receptive-field threshold must not wrap
            // through `as usize` into an out-of-bounds pointer.
            let in_start_f0 = ((idx_f0 as isize) + offset).max(0) as usize * self.in_ch;
            let in_start_f1 = ((idx_f1 as isize) + offset).max(0) as usize * self.in_ch;

            unsafe {
                *tap_f0 = layer_buffer.as_ptr().add(in_start_f0);
                *tap_f1 = layer_buffer.as_ptr().add(in_start_f1);

                if self.dilation >= 128 {
                    prefetch_strategy_2stage(
                        *tap_f0,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                } else {
                    prefetch_strategy_simple(
                        *tap_f0,
                        self.dilation * self.in_ch,
                        k,
                        self.kernel,
                        self.dilation,
                    );
                }
            }
        }

        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let do_bias = self.do_bias;

        match iw {
            16 => {
                for b in 0..num_blocks {
                    let out_c = b * 16;
                    let w = 16.min(self.out_ch - out_c);
                    let mut acc_f0 = [0.0f32; 16];
                    let mut acc_f1 = [0.0f32; 16];
                    for (j, (item_f0, item_f1)) in
                        acc_f0.iter_mut().zip(acc_f1.iter_mut()).enumerate().take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    for k in 0..kernel {
                        let w_start = b * kernel * in_ch * 16 + k * in_ch * 16;
                        let w_slice: &[[f32; 16]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let tap_f0 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f0.get_unchecked(k), in_ch)
                        };
                        let tap_f1 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f1.get_unchecked(k), in_ch)
                        };
                        let (r_f0, r_f1) = unsafe {
                            M::dot_product_16x_f32_dual_accumulate(
                                w_slice, tap_f0, tap_f1, &acc_f0, &acc_f1,
                            )
                        };
                        acc_f0 = r_f0;
                        acc_f1 = r_f1;
                    }
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in
                            acc_f0.iter().zip(acc_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
            8 => {
                for b in 0..num_blocks {
                    let out_c = b * 8;
                    let w = 8.min(self.out_ch - out_c);
                    let mut acc_f0 = [0.0f32; 8];
                    let mut acc_f1 = [0.0f32; 8];
                    for (j, (item_f0, item_f1)) in
                        acc_f0.iter_mut().zip(acc_f1.iter_mut()).enumerate().take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    for k in 0..kernel {
                        let w_start = b * kernel * in_ch * 8 + k * in_ch * 8;
                        let w_slice: &[[f32; 8]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let tap_f0 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f0.get_unchecked(k), in_ch)
                        };
                        let tap_f1 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f1.get_unchecked(k), in_ch)
                        };
                        let (r_f0, r_f1) = unsafe {
                            M::dot_product_8x_f32_dual_accumulate(
                                w_slice, tap_f0, tap_f1, &acc_f0, &acc_f1,
                            )
                        };
                        acc_f0 = r_f0;
                        acc_f1 = r_f1;
                    }
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in
                            acc_f0.iter().zip(acc_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
            _ => {
                for b in 0..num_blocks {
                    let out_c = b * 4;
                    let w = 4.min(self.out_ch - out_c);
                    let mut acc_f0 = [0.0f32; 4];
                    let mut acc_f1 = [0.0f32; 4];
                    for (j, (item_f0, item_f1)) in
                        acc_f0.iter_mut().zip(acc_f1.iter_mut()).enumerate().take(w)
                    {
                        let m0 = if let Some(m) = mixin_f0 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        let m1 = if let Some(m) = mixin_f1 {
                            if out_c + j < m.len() {
                                m[out_c + j]
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        };
                        if do_bias {
                            *item_f0 = self.bias[out_c + j] + m0;
                            *item_f1 = self.bias[out_c + j] + m1;
                        } else {
                            *item_f0 = m0;
                            *item_f1 = m1;
                        }
                    }
                    for k in 0..kernel {
                        let w_start = b * kernel * in_ch * 4 + k * in_ch * 4;
                        let w_slice: &[[f32; 4]] = unsafe {
                            let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                            core::slice::from_raw_parts(ptr, in_ch)
                        };
                        let tap_f0 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f0.get_unchecked(k), in_ch)
                        };
                        let tap_f1 = unsafe {
                            core::slice::from_raw_parts(*tap_ptrs_f1.get_unchecked(k), in_ch)
                        };
                        let (r_f0, r_f1) = unsafe {
                            M::dot_product_4x_f32_dual_accumulate(
                                w_slice, tap_f0, tap_f1, &acc_f0, &acc_f1,
                            )
                        };
                        acc_f0 = r_f0;
                        acc_f1 = r_f1;
                    }
                    unsafe {
                        for (j, (&v_f0, &v_f1)) in
                            acc_f0.iter().zip(acc_f1.iter()).enumerate().take(w)
                        {
                            *out_f0.get_unchecked_mut(out_c + j) = v_f0;
                            *out_f1.get_unchecked_mut(out_c + j) = v_f1;
                        }
                    }
                }
            }
        }
    }
}
