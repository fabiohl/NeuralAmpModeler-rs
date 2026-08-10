// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

// SAFETY: Caller guarantees alignment, bounds, and AVX2+FMA ISA availability for SIMD kernels.
#![allow(unsafe_op_in_unsafe_fn, clippy::missing_safety_doc)]

//! A2 Head convolution (k=16, bias, head_scale).
//!
//! Implements the head rechannel convolution from `a2_fast.cpp:722-743`:
//! `Conv1D(Bottleneck → 1, K=16, bias)` read from ring with tail-mirror,
//! followed by multiplication by `head_scale`.
//!
//! ## Data layout
//!
//! - Weights: `[K][Channels]` column-major per tap (Channels f32 values per tap).
//! - Head history ring buffer: `[Channels][cols]` column-major.
//! - Ring access: `col & ring_mask` (pow2 ring, mask = size - 1).
//!
//! ## Source of truth
//!
//! - `NAM/wavenet/a2_fast.cpp:117-136` (member declarations)
//! - `NAM/wavenet/a2_fast.cpp:262-275` (weight loading order)
//! - `NAM/wavenet/a2_fast.cpp:716-745` (`_head_forward`)

use crate::math::common::AlignedVec;
use crate::math::common::hsum_avx2;
use core::arch::x86_64::*;

/// Head convolution for the A2 WaveNet architecture.
///
/// Applies `Conv1D(Bottleneck → 1, K=16, bias)` over the head history ring buffer
/// and multiplies the result by `head_scale`.
///
/// The head history accumulates the post-activation outputs from all 23 layers
/// (first layer assigns, subsequent layers add). This struct applies a final
/// causal convolution over that accumulator to produce the output signal.
#[derive(Clone)]
pub struct A2HeadConv {
    /// Head weights: `[KERNEL_SIZE][Channels]` f32, stored column-major per tap.
    /// At tap `k`, weight for channel `c` is at index `k * num_channels + c`.
    pub head_w: AlignedVec<f32>,
    /// Head bias (single scalar).
    pub head_b: f32,
    /// Head scale (multiplied after convolution).
    pub head_scale: f32,
    /// Number of channels = bottleneck size (3 for Lite, 8 for Full).
    pub num_channels: usize,
    /// Kernel size of the head convolution (= 16 for A2).
    pub kernel_size: usize,
}

impl A2HeadConv {
    /// A2 canonical head kernel size.
    pub const HEAD_KERNEL_SIZE: usize = 16;

    /// Creates a new `A2HeadConv` from pre-loaded weights.
    ///
    /// `head_w` must contain exactly `HEAD_KERNEL_SIZE * num_channels` f32 values,
    /// stored column-major per tap as loaded by `_load_weights` (see `a2_fast.cpp:262-275`).
    pub fn new(head_w: AlignedVec<f32>, head_b: f32, head_scale: f32, num_channels: usize) -> Self {
        Self::new_with_kernel(
            head_w,
            head_b,
            head_scale,
            num_channels,
            Self::HEAD_KERNEL_SIZE,
        )
    }

    /// Creates a new `A2HeadConv` with a non-standard kernel size.
    ///
    /// Legacy models may have `head_kernel_size = 1` (1×1 rechannel, no temporal
    /// context). The SIMD fast-paths require `kernel_size == 16`; for other sizes,
    /// the scalar fallback is used.
    pub fn new_with_kernel(
        head_w: AlignedVec<f32>,
        head_b: f32,
        head_scale: f32,
        num_channels: usize,
        kernel_size: usize,
    ) -> Self {
        assert_eq!(
            head_w.len(),
            kernel_size * num_channels,
            "head_w must have kernel_size * num_channels elements"
        );
        Self {
            head_w,
            head_b,
            head_scale,
            num_channels,
            kernel_size,
        }
    }

    /// Processes a block of `num_frames` through the head convolution.
    ///
    /// `head_history` is a contiguous col-major buffer (`Channels` rows × N columns).
    /// Ring access uses `col & ring_mask` (pow2 ring). `head_write_pos` is the position
    /// where the *next* batch of frames will be written (already advanced past this batch).
    ///
    /// # Panics
    /// Debug: asserts that `output` has at least `num_frames` elements.
    #[inline(always)]
    pub fn process(
        &self,
        head_history: &[f32],
        head_write_pos: usize,
        ring_mask: usize,
        num_frames: usize,
        output: &mut [f32],
    ) {
        debug_assert!(output.len() >= num_frames);
        debug_assert!(head_history.len() >= (ring_mask + 1) * self.num_channels);

        // Dispatch to SIMD kernels when the canonical A2 kernel size is active.
        if self.kernel_size == Self::HEAD_KERNEL_SIZE {
            match self.num_channels {
                8 => {
                    // SAFETY: x86-64-v3 guarantees AVX2+FMA.
                    unsafe {
                        head_process_ch8_avx2(
                            &self.head_w,
                            self.head_b,
                            self.head_scale,
                            head_history,
                            head_write_pos,
                            ring_mask,
                            num_frames,
                            output,
                        );
                    }
                    return;
                }
                3 => {
                    // SAFETY: x86-64-v3 guarantees FMA.
                    unsafe {
                        head_process_ch3_sse(
                            &self.head_w,
                            self.head_b,
                            self.head_scale,
                            head_history,
                            head_write_pos,
                            ring_mask,
                            num_frames,
                            output,
                        );
                    }
                    return;
                }
                _ => {}
            }
        }

        // Scalar fallback.
        let k = self.kernel_size;
        let ch = self.num_channels;
        let base = head_write_pos.wrapping_sub(num_frames);

        for (f, out_val) in output.iter_mut().take(num_frames).enumerate() {
            let col_base = base.wrapping_add(f);
            let mut y = self.head_b;

            for t in 0..k {
                let col = col_base.wrapping_sub(k - 1 - t) & ring_mask;
                let src_off = col * ch;
                let w_off = t * ch;

                // SAFETY: head_w length is validated in new() (assert_eq), and
                // head_history length is validated by the debug_assert above.
                // w_off + c < head_w.len() because t < k and c < ch.
                // src_off + c < head_history.len() because col <= ring_mask.
                for c in 0..ch {
                    unsafe {
                        y += *self.head_w.get_unchecked(w_off + c)
                            * *head_history.get_unchecked(src_off + c);
                    }
                }
            }

            *out_val = y * self.head_scale;
        }
    }
}

// =============================================================================
// AVX2+FMA kernel for CH=8
// =============================================================================

/// Kernel SIMD AVX2+FMA para A2 Head Conv com CH=8.
///
/// Processa `num_frames` frames usando T=4 frame-tiling:
/// carrega os K=16 pesos uma vez por tap e acumula
/// via broadcast FMA em 4 acumuladores `__m256` simultâneos.
///
/// Ao final de cada tile de 4 frames, `hsum_avx2` reduz cada acumulador,
/// adiciona `head_b` e multiplica por `head_scale`.
///
/// # Safety
/// - Requer AVX2+FMA (`target_feature`).
/// - `head_w` deve ter pelo menos `K * 8` elementos.
/// - `head_history` deve ter `(ring_mask + 1) * 8` elementos.
/// - `output` deve ter pelo menos `num_frames` elementos.
/// - `num_channels` implícito = 8 (chamador deve garantir).
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn head_process_ch8_avx2(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
) {
    let k = A2HeadConv::HEAD_KERNEL_SIZE;
    const T: usize = 4;
    let n_tiled = (num_frames / T) * T;
    let base = head_write_pos.wrapping_sub(num_frames);

    for f in (0..n_tiled).step_by(T) {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        let col_base_f = base.wrapping_add(f);

        for t in 0..k {
            let w_v = _mm256_loadu_ps(head_w.as_ptr().add(t * 8));

            let col0 = col_base_f.wrapping_sub(k - 1 - t) & ring_mask;
            let h0 = _mm256_loadu_ps(head_history.as_ptr().add(col0 * 8));
            a0 = _mm256_fmadd_ps(w_v, h0, a0);

            let col1 = col_base_f.wrapping_add(1).wrapping_sub(k - 1 - t) & ring_mask;
            let h1 = _mm256_loadu_ps(head_history.as_ptr().add(col1 * 8));
            a1 = _mm256_fmadd_ps(w_v, h1, a1);

            let col2 = col_base_f.wrapping_add(2).wrapping_sub(k - 1 - t) & ring_mask;
            let h2 = _mm256_loadu_ps(head_history.as_ptr().add(col2 * 8));
            a2 = _mm256_fmadd_ps(w_v, h2, a2);

            let col3 = col_base_f.wrapping_add(3).wrapping_sub(k - 1 - t) & ring_mask;
            let h3 = _mm256_loadu_ps(head_history.as_ptr().add(col3 * 8));
            a3 = _mm256_fmadd_ps(w_v, h3, a3);
        }

        output[f] = (hsum_avx2(a0) + head_b) * head_scale;
        output[f + 1] = (hsum_avx2(a1) + head_b) * head_scale;
        output[f + 2] = (hsum_avx2(a2) + head_b) * head_scale;
        output[f + 3] = (hsum_avx2(a3) + head_b) * head_scale;
    }

    // Scalar tail for remaining frames (< T)
    for (f, out_val) in output.iter_mut().take(num_frames).enumerate().skip(n_tiled) {
        *out_val = a2_head_single_frame_scalar_ref(
            head_w,
            head_b,
            head_scale,
            8,
            head_history,
            head_write_pos,
            ring_mask,
            num_frames,
            f,
        );
    }
}

// =============================================================================
// SSE+FMA kernel for CH=3
// =============================================================================

/// Kernel SIMD SSE+FMA para A2 Head Conv com CH=3.
///
/// Processa `num_frames` frames, um por vez (sem frame-tiling),
/// usando `_mm_setr_ps` para empacotar 3 pesos + 0 e 3 valores
/// do histórico + 0 em registradores `__m128`, acumulando via
/// `_mm_fmadd_ps` sobre K=16 taps.
///
/// A redução final usa `_mm_hadd_ps` × 2 + `_mm_cvtss_f32`,
/// seguida de `(y + head_b) * head_scale`.
///
/// # Safety
/// - Requer SSE+FMA (`target_feature`).
/// - `head_w` deve ter pelo menos `K * 3` elementos.
/// - `head_history` deve ter `(ring_mask + 1) * 3` elementos.
/// - `output` deve ter pelo menos `num_frames` elementos.
/// - `num_channels` implícito = 3 (chamador deve garantir).
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
#[target_feature(enable = "fma")]
pub unsafe fn head_process_ch3_sse(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
) {
    let k = A2HeadConv::HEAD_KERNEL_SIZE;
    let base = head_write_pos.wrapping_sub(num_frames);
    let w_ptr = head_w.as_ptr();
    let h_ptr = head_history.as_ptr();

    for (f, out_val) in output.iter_mut().take(num_frames).enumerate() {
        let mut acc = _mm_setzero_ps();
        let col_base = base.wrapping_add(f);

        for t in 0..k {
            let col = col_base.wrapping_sub(k - 1 - t) & ring_mask;
            let src_off = col * 3;
            let w_off = t * 3;

            let w_v = _mm_setr_ps(
                *w_ptr.add(w_off),
                *w_ptr.add(w_off + 1),
                *w_ptr.add(w_off + 2),
                0.0,
            );
            let h_v = _mm_setr_ps(
                *h_ptr.add(src_off),
                *h_ptr.add(src_off + 1),
                *h_ptr.add(src_off + 2),
                0.0,
            );

            acc = _mm_fmadd_ps(w_v, h_v, acc);
        }

        let hadd1 = _mm_hadd_ps(acc, acc);
        let hadd2 = _mm_hadd_ps(hadd1, hadd1);
        let y = _mm_cvtss_f32(hadd2);

        *out_val = (y + head_b) * head_scale;
    }
}

// =============================================================================
// Scalar reference for parity testing (oracle)
// =============================================================================

/// Scalar reference for a single frame of head convolution.
///
/// Matches `A2HeadConv::process` single-frame logic exactly.
/// Used as an oracle for unit tests and SIMD parity verification.
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn a2_head_single_frame_scalar_ref(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    num_channels: usize,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    frame: usize,
) -> f32 {
    let k = A2HeadConv::HEAD_KERNEL_SIZE;
    let base = head_write_pos.wrapping_sub(num_frames);
    let col_base = base.wrapping_add(frame);

    let mut y = head_b;

    for t in 0..k {
        let col = col_base.wrapping_sub(k - 1 - t) & ring_mask;
        let src_off = col * num_channels;
        let w_off = t * num_channels;

        for c in 0..num_channels {
            y += head_w[w_off + c] * head_history[src_off + c];
        }
    }

    y * head_scale
}

/// Scalar reference for a full block of head convolution.
///
/// Computes the head output for `num_frames` using the same algorithm
/// as `A2HeadConv::process`. Useful for validating the block-level path.
#[expect(
    clippy::too_many_arguments,
    reason = "FFI design or complex DSP kernel signature required by construction"
)]
pub fn a2_head_block_scalar_ref(
    head_w: &[f32],
    head_b: f32,
    head_scale: f32,
    num_channels: usize,
    head_history: &[f32],
    head_write_pos: usize,
    ring_mask: usize,
    num_frames: usize,
    output: &mut [f32],
) {
    for (f, out_val) in output.iter_mut().take(num_frames).enumerate() {
        *out_val = a2_head_single_frame_scalar_ref(
            head_w,
            head_b,
            head_scale,
            num_channels,
            head_history,
            head_write_pos,
            ring_mask,
            num_frames,
            f,
        );
    }
}

#[cfg(test)]
#[path = "head_test.rs"]
mod tests;
