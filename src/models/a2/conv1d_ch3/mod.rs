// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]

//! CH=3 A2 fast-path convolution with **f32 native weights** for optimization.
//!
//! ## Problem solved
//!
//! The original `process_single_ch3_unrolled` used `_mm_cvtph_ps` (f16 → f32 decode)
//! inside each of the 18/45 FMAs, adding a pipeline stall per instruction. This caused
//! A2-Lite (CH=3, ~48.7 µs) to be **58% slower** than A2-Full (CH=8, ~30.9 µs) despite
//! having ~6.5× fewer weights — inverting the expected CPU economy.
//!
//! ## Solution
//!
//! Mirrors the approach for CH=8: store weights in **f32 native col-major-per-tap**
//! layout (`w[k * 16 + in * 4 + out]`, 4-padded for SIMD alignment with one zero lane).
//! The hot-path becomes a plain `_mm_loadu_ps` + `_mm_fmadd_ps` — identical in structure
//! to the CH=8 kernel but using SSE/128-bit registers (CH=3+1pad = 4 lanes = 1 XMM).
//!
//! ## Post-conv AVX2 batching (`layer_forward_ch3_block`)
//!
//! Mixin, LeakyReLU, head and l1x1 are batched in pairs of frames using `__m256`
//! (2 frames × 4 channels = 8 floats = 1 YMM register), exploiting the x86-64-v3 AVX2
//! baseline. This eliminates per-frame scalar loops for the post-conv operations.
//!
//! ## Layout
//!
//! NAM JSON order: `raw[out_ch][in_ch][kernel]`
//! Col-major-per-tap: `w[k * 16 + in * 4 + out]`
//!   - `k` : tap index (0..K-1)
//!   - `in`: input channel (0..2), stride 4
//!   - `out`: output channel (0..2), lane 3 = 0.0 (padding)
//!
//! For a given `(k, in)`, 4 floats are contiguous → one `_mm_loadu_ps` load.
//!
//! ## Source of truth
//! - `a2_fast.cpp` (strategy `Channels=3`, GEMV unrolled)
//! - `conv1d_ch8.rs` for the f32 col-major pattern

/// CH=3 dilated causal Conv1D alias — see `super::conv1d_ch::A2Conv1dCh` for docs.
pub type A2Conv1dCh3 = super::conv1d_ch::A2Conv1dCh<3>;

mod scalar;
mod simd;
pub use scalar::*;
pub use simd::*;

// Legacy dispatch (kept for use in conv1d_dyn.rs hot-path compat)
// Uses the new f32-native kernel when A2Conv1dCh3 is available via model dispatch.
// The old _mm_cvtph_ps path in Conv1dDyn::process_single_ch3_unrolled is now
// superseded by layer_forward_ch3_block / model.rs dispatch.
// =============================================================================

impl crate::models::wavenet::conv1d_dyn::Conv1dDyn {
    /// Fully unrolled CH=3 GEMV — f32-native path for CH=3 convolutions.
    ///
    /// Dispatches to `process_single_ch3_k6` or `process_single_ch3_k15`
    /// depending on the kernel size.
    #[cfg(test)]
    #[inline(always)]
    pub(crate) unsafe fn process_single_ch3_unrolled(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        debug_assert_eq!(self.out_ch, 3);
        debug_assert!(self.kernel == 6 || self.kernel == 15);

        match self.kernel {
            6 => self.process_single_ch3_k6(layer_buffer, out_frame, frame_idx, mixin),
            15 => self.process_single_ch3_k15(layer_buffer, out_frame, frame_idx, mixin),
            _ => {
                debug_assert!(
                    false,
                    "A2 CH3 conv kernel must be 6 or 15; got {} — silencing output frame",
                    self.kernel
                );
                out_frame[..self.out_ch].fill(0.0);
            }
        }
    }

    /// Unrolled K=6 GEMV for 3-channel input/output (f32-native).
    #[cfg(test)]
    #[target_feature(enable = "avx")]
    unsafe fn process_single_ch3_k6(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        use core::arch::x86_64::*;
        let in_ch = self.in_ch;
        let d = self.dilation as isize;
        let k_limit = self.kernel as isize; // 6

        let buf = layer_buffer.as_ptr();

        let t0 = ((frame_idx as isize) + d * (1_isize - k_limit)) as usize * in_ch;
        let t1 = ((frame_idx as isize) + d * (2_isize - k_limit)) as usize * in_ch;
        let t2 = ((frame_idx as isize) + d * (3_isize - k_limit)) as usize * in_ch;
        let t3 = ((frame_idx as isize) + d * (4_isize - k_limit)) as usize * in_ch;
        let t4 = ((frame_idx as isize) + d * (5_isize - k_limit)) as usize * in_ch;
        let t5 = ((frame_idx as isize) + d * (6_isize - k_limit)) as usize * in_ch;

        let (b0, b1, b2, b3) = Self::load_mixin_4(mixin, 0);
        let mut acc = if self.do_bias {
            _mm_setr_ps(
                *self.bias.get_unchecked(0) + b0,
                *self.bias.get_unchecked(1) + b1,
                *self.bias.get_unchecked(2) + b2,
                b3,
            )
        } else {
            _mm_setr_ps(b0, b1, b2, b3)
        };

        let w_ptr = self.weights.as_ptr();

        macro_rules! fma3_k6 {
            ($tap_base:ident, $k:expr) => {{
                let wp = w_ptr.add(($k * in_ch + 0) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 0));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            {
                let wp = w_ptr.add(($k * in_ch + 1) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 1));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            {
                let wp = w_ptr.add(($k * in_ch + 2) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 2));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }};
        }

        fma3_k6!(t0, 0);
        fma3_k6!(t1, 1);
        fma3_k6!(t2, 2);
        fma3_k6!(t3, 3);
        fma3_k6!(t4, 4);
        fma3_k6!(t5, 5);

        _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
    }

    /// Unrolled K=15 GEMV for 3-channel input/output (f32-native).
    #[cfg(test)]
    #[target_feature(enable = "avx")]
    unsafe fn process_single_ch3_k15(
        &self,
        layer_buffer: &[f32],
        out_frame: &mut [f32],
        frame_idx: usize,
        mixin: Option<&[f32]>,
    ) {
        use core::arch::x86_64::*;
        let in_ch = self.in_ch;
        let d = self.dilation as isize;
        let k_limit = self.kernel as isize; // 15

        let buf = layer_buffer.as_ptr();

        macro_rules! tap {
            ($idx:expr) => {
                ((frame_idx as isize) + d * (($idx as isize) + 1 - k_limit)) as usize * in_ch
            };
        }

        let t0 = tap!(0);
        let t1 = tap!(1);
        let t2 = tap!(2);
        let t3 = tap!(3);
        let t4 = tap!(4);
        let t5 = tap!(5);
        let t6 = tap!(6);
        let t7 = tap!(7);
        let t8 = tap!(8);
        let t9 = tap!(9);
        let t10 = tap!(10);
        let t11 = tap!(11);
        let t12 = tap!(12);
        let t13 = tap!(13);
        let t14 = tap!(14);

        let (b0, b1, b2, b3) = Self::load_mixin_4(mixin, 0);
        let mut acc = if self.do_bias {
            _mm_setr_ps(
                *self.bias.get_unchecked(0) + b0,
                *self.bias.get_unchecked(1) + b1,
                *self.bias.get_unchecked(2) + b2,
                b3,
            )
        } else {
            _mm_setr_ps(b0, b1, b2, b3)
        };

        let w_ptr = self.weights.as_ptr();

        macro_rules! fma3_k15 {
            ($tap_base:ident, $k:expr) => {{
                let wp = w_ptr.add(($k * in_ch + 0) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 0));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            {
                let wp = w_ptr.add(($k * in_ch + 1) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 1));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }
            {
                let wp = w_ptr.add(($k * in_ch + 2) * 4);
                let wv = _mm_loadu_ps(wp);
                let sv = _mm_set1_ps(*buf.add($tap_base + 2));
                acc = _mm_fmadd_ps(wv, sv, acc);
            }};
        }

        fma3_k15!(t0, 0);
        fma3_k15!(t1, 1);
        fma3_k15!(t2, 2);
        fma3_k15!(t3, 3);
        fma3_k15!(t4, 4);
        fma3_k15!(t5, 5);
        fma3_k15!(t6, 6);
        fma3_k15!(t7, 7);
        fma3_k15!(t8, 8);
        fma3_k15!(t9, 9);
        fma3_k15!(t10, 10);
        fma3_k15!(t11, 11);
        fma3_k15!(t12, 12);
        fma3_k15!(t13, 13);
        fma3_k15!(t14, 14);

        _mm_storeu_ps(out_frame.as_mut_ptr(), acc);
    }
}

#[cfg(test)]
#[path = "conv1d_ch3_test.rs"]
mod tests;
