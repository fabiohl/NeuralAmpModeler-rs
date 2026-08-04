// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Const-generic A2 fast-path convolution with f32 native weights.
//!
//! Unifies the previously-separate `A2Conv1dCh3` and `A2Conv1dCh8` into a single
//! `A2Conv1dCh<const CH: usize>` struct. The aligned SIMD stride per input channel
//! is `CH_PAD = CH.next_power_of_two()`, giving a total of `kernel * CH_PAD * CH_PAD`
//! weights and `CH_PAD` bias elements (zero-padded when `CH < CH_PAD`).
//!
//! ## Layout
//!
//! NAM JSON order: `raw[out_ch][in_ch][kernel]`
//! Col-major-per-tap: `w[k * CH_PAD² + in * CH_PAD + out]`
//!
//! ## SIMD kernels
//!
//! The SIMD kernels remain in `conv1d_ch3/simd.rs` (SSE 128-bit for CH=3) and
//! `conv1d_ch8/simd.rs` (AVX2 256-bit for CH=8), accessible via the type aliases
//! `A2Conv1dCh3` (= `A2Conv1dCh<3>`) and `A2Conv1dCh8` (= `A2Conv1dCh<8>`).

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

/// CH-padded SIMD stride per input channel.
///
/// CH=3 → CH_PAD=4 (SSE-ready), CH=8 → CH_PAD=8 (AVX2-ready).
pub const fn ch_pad<const CH: usize>() -> usize {
    CH.next_power_of_two()
}

/// CH-channel dilated causal Conv1D weights in col-major-per-tap layout with f32 precision.
///
/// Layout: `w[k * CH_PAD² + in_ch * CH_PAD + out_ch]`
/// - `k`: kernel tap index (0..K-1)
/// - `in_ch`: input channel (0..CH-1)
/// - `out_ch`: output channel (0..CH-1), lanes ≥ CH are zero-padded
///
/// Supported `CH` values: 3 (Lite/Nano) and 8 (Full/Standard).
#[derive(Clone)]
#[repr(align(64))]
pub struct A2Conv1dCh<const CH: usize> {
    /// Col-major-per-tap f32 weights: `kernel * CH_PAD²` elements.
    pub weights: AlignedVec<f32>,
    /// Bias vector — `CH_PAD` elements (CH valid + zero padding).
    pub bias: AlignedVec<f32>,
    /// Temporal dilation factor.
    pub dilation: usize,
    /// Kernel size (6 or 15 for A2).
    pub kernel: usize,
}

impl<const CH: usize> A2Conv1dCh<CH> {
    /// Builds an A2 conv1d from the NAM JSON weight stream (f32, raw order).
    ///
    /// `raw_weights` is in NAM JSON row-major order: `[out_ch][in_ch][kernel]`.
    /// This constructor permutes once (at load time) to col-major-per-tap:
    /// `w[k * CH_PAD² + in * CH_PAD + out]`.
    ///
    /// `raw_bias` must contain exactly `CH` elements; when `CH_PAD > CH`, the
    /// remaining lanes are zero-filled for SIMD alignment.
    pub fn new(
        raw_weights: &[f32],
        out_ch: usize,
        in_ch: usize,
        kernel: usize,
        dilation: usize,
        raw_bias: &[f32],
    ) -> Result<Self, NamErrorCode> {
        let ch_pad = ch_pad::<CH>();
        debug_assert_eq!(out_ch, CH);
        debug_assert_eq!(in_ch, CH);
        debug_assert!(kernel == 6 || kernel == 15);
        debug_assert_eq!(raw_weights.len(), out_ch * in_ch * kernel);
        debug_assert_eq!(raw_bias.len(), CH);

        let stride = ch_pad * ch_pad;
        let mut weights = AlignedVec::new(kernel * stride, 0.0f32)?;
        for out in 0..out_ch {
            for inp in 0..in_ch {
                for k in 0..kernel {
                    let src = out * in_ch * kernel + inp * kernel + k;
                    let dst = k * stride + inp * ch_pad + out;
                    weights[dst] = raw_weights[src];
                }
            }
        }

        let mut bias = AlignedVec::new(ch_pad, 0.0f32)?;
        bias[..CH].copy_from_slice(&raw_bias[..CH]);

        Ok(Self {
            weights,
            bias,
            dilation,
            kernel,
        })
    }
}
