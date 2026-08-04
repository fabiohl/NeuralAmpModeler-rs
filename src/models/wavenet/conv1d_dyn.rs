// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Runtime-dimensional convolution components for WaveNet architectures.
//!
//! Contains the fundamental convolution structures that operate with
//! runtime-defined dimensions, serving as a foundation for A2 architecture
//! stages and static WaveNet test/stress kernels.

use crate::math::common::{
    AlignedVec, SimdMath, prefetch_strategy_2stage, prefetch_strategy_simple,
};

use super::common::MAX_KERNEL;

/// Structure for causal 1D convolution with dynamic dimensions.
#[derive(Clone)]
#[repr(align(64))]
pub struct Conv1dDyn {
    /// Full-precision f32 convolution weights `[OUT][KERNEL][IN]` (interleaved).
    pub weights: AlignedVec<f32>,
    /// Bias vector `[OUT]`.
    pub bias: AlignedVec<f32>,
    /// Flag indicating whether bias should be applied.
    pub do_bias: bool,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
    /// Number of 4-channel blocks (used by dual-frame path).
    pub num_blocks: usize,
    /// Interleave width (4, 8, or 16) used for single-frame processing.
    pub interleave_width: usize,
    /// Physical kernel size.
    pub kernel: usize,
}

impl Conv1dDyn {
    /// F32-native single-frame convolution (full-precision f32 weights).
    ///
    /// # Safety
    /// The caller must guarantee that `layer_buffer` and `out_frame` have sizes
    /// compatible with the layer dimensions.
    #[inline(always)]
    pub unsafe fn process_single_frame<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        let in_ch = self.in_ch;
        let kernel = self.kernel;
        let mut tap_ptrs = [core::ptr::null::<f32>(); MAX_KERNEL];
        let k_limit = kernel.min(MAX_KERNEL);

        for (k, tap) in tap_ptrs.iter_mut().enumerate().take(k_limit) {
            let offset = (self.dilation as isize) * ((k as isize) + 1 - (kernel as isize));
            let in_start = ((frame_idx as isize) + offset) as usize * in_ch;
            unsafe {
                *tap = layer_buffer.as_ptr().add(in_start);
                if self.dilation >= 128 {
                    prefetch_strategy_2stage(*tap, self.dilation * in_ch, k, kernel, self.dilation);
                } else {
                    prefetch_strategy_simple(*tap, self.dilation * in_ch, k, kernel, self.dilation);
                }
            }
        }

        unsafe {
            match self.interleave_width {
                16 => {
                    self.process_blocks_16_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin)
                }
                8 => self.process_blocks_8_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin),
                _ => self.process_blocks_4_nocopy::<M>(out_frame, kernel, &tap_ptrs, in_ch, mixin),
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_4_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(4);
        for b in 0..num_blocks {
            let out_c = b * 4;
            let w = 4.min(self.out_ch - out_c);
            let (mu0, mu1, mu2, mu3) = unsafe { Self::load_mixin_4(mixin, out_c) };

            let mut acc = [0.0f32; 4];
            unsafe {
                if self.do_bias {
                    acc[0] = *self.bias.get_unchecked(out_c) + mu0;
                    acc[1] = if out_c + 1 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 1)
                    } else {
                        0.0
                    } + mu1;
                    acc[2] = if out_c + 2 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 2)
                    } else {
                        0.0
                    } + mu2;
                    acc[3] = if out_c + 3 < self.out_ch {
                        *self.bias.get_unchecked(out_c + 3)
                    } else {
                        0.0
                    } + mu3;
                } else {
                    acc[0] = mu0;
                    acc[1] = mu1;
                    acc[2] = mu2;
                    acc[3] = mu3;
                }
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 4 + k * in_ch * 4;
                let w_slice: &[[f32; 4]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 4];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                acc = unsafe { M::dot_product_4x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            unsafe {
                if out_c + 3 < self.out_ch {
                    *out_frame.get_unchecked_mut(out_c) = acc[0];
                    *out_frame.get_unchecked_mut(out_c + 1) = acc[1];
                    *out_frame.get_unchecked_mut(out_c + 2) = acc[2];
                    *out_frame.get_unchecked_mut(out_c + 3) = acc[3];
                } else {
                    for (lane, &val) in acc.iter().enumerate().take(w) {
                        *out_frame.get_unchecked_mut(out_c + lane) = val;
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_8_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(8);
        for b in 0..num_blocks {
            let out_c = b * 8;
            let w = 8.min(self.out_ch - out_c);
            let mut acc = [0.0f32; 8];
            unsafe {
                for (j, item) in acc.iter_mut().enumerate().take(w) {
                    let v_bias = if self.do_bias {
                        *self.bias.get_unchecked(out_c + j)
                    } else {
                        0.0
                    };
                    let v_mixin = if let Some(m) = mixin {
                        if out_c + j < m.len() {
                            *m.get_unchecked(out_c + j)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    *item = v_bias + v_mixin;
                }
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 8 + k * in_ch * 8;
                let w_slice: &[[f32; 8]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 8];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                acc = unsafe { M::dot_product_8x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            unsafe {
                for (j, &item) in acc.iter().enumerate().take(w) {
                    *out_frame.get_unchecked_mut(out_c + j) = item;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn process_blocks_16_nocopy<M: SimdMath>(
        &self,
        out_frame: &mut [f32],
        kernel: usize,
        tap_ptrs: &[*const f32],
        in_ch: usize,
        mixin: Option<&[f32]>,
    ) {
        let num_blocks = self.out_ch.div_ceil(16);
        for b in 0..num_blocks {
            let out_c = b * 16;
            let w = 16.min(self.out_ch - out_c);
            let mut acc = [0.0f32; 16];
            unsafe {
                for (j, item) in acc.iter_mut().enumerate().take(w) {
                    let v_bias = if self.do_bias {
                        *self.bias.get_unchecked(out_c + j)
                    } else {
                        0.0
                    };
                    let v_mixin = if let Some(m) = mixin {
                        if out_c + j < m.len() {
                            *m.get_unchecked(out_c + j)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    *item = v_bias + v_mixin;
                }
            }

            for k in 0..kernel {
                let w_start = b * kernel * in_ch * 16 + k * in_ch * 16;
                let w_slice: &[[f32; 16]] = unsafe {
                    let ptr = self.weights.as_ptr().add(w_start) as *const [f32; 16];
                    core::slice::from_raw_parts(ptr, in_ch)
                };
                let tap_slice =
                    unsafe { core::slice::from_raw_parts(*tap_ptrs.get_unchecked(k), in_ch) };
                acc = unsafe { M::dot_product_16x_f32_accumulate(w_slice, tap_slice, &acc) };
            }

            unsafe {
                for (j, &item) in acc.iter().enumerate().take(w) {
                    *out_frame.get_unchecked_mut(out_c + j) = item;
                }
            }
        }
    }

    /// F32-native block processing (full-precision f32 weights).
    ///
    /// Same dual-frame tiling as the generic block processing, but uses full-precision
    /// f32 weights and scalar dot products.
    #[inline(always)]
    pub(crate) unsafe fn process_block<M: SimdMath>(
        &self,
        layer_buffer: &[f32],
        block: &mut [f32],
        buffer_start: usize,
        num_frames: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(num_frames * self.out_ch, block.len());
        let mut i = 0;

        let mut chunks = block.chunks_exact_mut(2 * self.out_ch);
        for chunk in chunks.by_ref() {
            let (out_f0, out_f1) = chunk.split_at_mut(self.out_ch);

            let (m_f0, m_f1) = if let Some(m) = mixin {
                let start0 = i * self.out_ch;
                let end0 = (start0 + self.out_ch).min(m.len());
                let start1 = (i + 1) * self.out_ch;
                let end1 = (start1 + self.out_ch).min(m.len());
                (
                    if start0 < m.len() {
                        Some(&m[start0..end0])
                    } else {
                        None
                    },
                    if start1 < m.len() {
                        Some(&m[start1..end1])
                    } else {
                        None
                    },
                )
            } else {
                (None, None)
            };

            unsafe {
                self.process_dual_frame::<M>(
                    layer_buffer,
                    out_f0,
                    out_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    m_f0,
                    m_f1,
                );
            }
            i += 2;
        }

        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let m = mixin.map(|m| &m[i * self.out_ch..(i + 1) * self.out_ch]);
            unsafe {
                self.process_single_frame::<M>(layer_buffer, rem, buffer_start + i, m);
            }
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn load_mixin_4(mixin: Option<&[f32]>, out_c: usize) -> (f32, f32, f32, f32) {
        if let Some(m) = mixin {
            if out_c + 3 < m.len() {
                unsafe {
                    (
                        *m.get_unchecked(out_c),
                        *m.get_unchecked(out_c + 1),
                        *m.get_unchecked(out_c + 2),
                        *m.get_unchecked(out_c + 3),
                    )
                }
            } else {
                let mut v = [0.0f32; 4];
                for (i, val) in v.iter_mut().enumerate() {
                    if out_c + i < m.len() {
                        unsafe {
                            *val = *m.get_unchecked(out_c + i);
                        }
                    }
                }
                (v[0], v[1], v[2], v[3])
            }
        } else {
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}
