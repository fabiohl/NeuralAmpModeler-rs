// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WAVENET_MAX_NUM_FRAMES, WavenetProcessContext};
use super::conv1d::Conv1d;
use super::dense::DenseLayer;
use crate::math::common::{AlignedVec, SimdMath};
use crate::models::a2::activations::ActivationType;

#[cfg(test)]
pub(crate) use telemetry_vars::*;

#[cfg(test)]
mod telemetry_vars {
    use std::cell::RefCell;
    use std::time::Duration;

    thread_local! {
        pub static ACC_MIXIN: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_CONV: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_TANH: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static ACC_ONE_BY_ONE: RefCell<Duration> = const { RefCell::new(Duration::ZERO) };
        pub static TELEMETRY_ACTIVE: RefCell<bool> = const { RefCell::new(false) };
    }
}

/// Complete Convolutional Cell (WaveNet Layer).
#[derive(Clone)]
pub struct WaveNetLayer<const COND: usize, const CH: usize, const K: usize> {
    /// This layer's parametric dilated Causal 1D convolution mesh.
    pub conv1d: Conv1d<CH, CH, K>,
    /// Conditional injection mixing network.
    pub input_mixin: DenseLayer<COND, CH>,
    /// 1x1 decompression linear affine transform of the layer.
    pub one_by_one: DenseLayer<CH, CH>,
    /// Activation function for this layer (default: Tanh).
    pub activation: ActivationType,
    /// Pre-allocated scratch buffer for conditioning mixin output
    /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_mixin: AlignedVec<f32>,
    /// Pre-allocated scratch buffer for Conv1D + mixin intermediate results
    /// (size: `CH * WAVENET_MAX_NUM_FRAMES`).
    pub scratch_conv: AlignedVec<f32>,
}

impl<const COND: usize, const CH: usize, const K: usize> WaveNetLayer<COND, CH, K> {
    /// Processes a full WaveNet layer, iterating `FastMath` in AVX2.
    ///
    /// Conv1D uses full-precision f32 weights; DenseLayer uses standard quantized paths.
    ///
    /// # Performance Warning (CH=12 Memory Stride Barrier)
    /// For models with `CH=12` (WaveNet Lite), the 12-float channel stride (48 bytes) does not
    /// align with the CPU's 64-byte cache line boundary. This mismatch induces frequent
    /// cache-line splits and stalls the FMA pipelines. Although specialized SIMD kernels have been
    /// written (e.g., SIMD 8+4 store path and 12x12 GEMM), the physical stride remains a bottleneck
    /// preventing the engine from reaching ≤ 38 µs. The ultimate fix is a homogeneous pad-to-16
    /// layout (CH=16 static).
    ///
    /// # Safety
    /// Math dispatch via pointer to inlined intrinsic functions.
    ///
    /// # RT-safety (F-10)
    /// The audio hot-path is 100% panic-free: the previous `assert!(...)` with
    /// `format!` allocations were removed. `num_frames` is clamped defensively
    /// to the scratch capacities, so an oversized frame count is truncated
    /// (fail-closed) instead of aborting the RT thread. Host contract
    /// violations are flagged at the pipeline boundary (`capture.rs`,
    /// `RT_STATUS_HOST_CONTRACT_VIOLATION`); the clamp here is the last line of
    /// defense and never allocates or panics.
    #[inline]
    pub unsafe fn process_block_internal<M: SimdMath>(&mut self, ctx: WavenetProcessContext<'_>) {
        let WavenetProcessContext {
            condition,
            head_input,
            output,
            layer_buffer,
            buffer_start,
            num_frames,
            seed,
            is_first_layer,
            ..
        } = ctx;

        // SAFETY: `num_frames` is clamped to `WAVENET_MAX_NUM_FRAMES` (F-10) and the
        // `debug_assert!`s below bound `num_frames * CH` within `scratch_mixin`/`scratch_conv`
        // (pre-allocated as `CH * WAVENET_MAX_NUM_FRAMES`), so the `get_unchecked` scratch slices
        // and the `layer_buffer` residual slice (covered by the caller contract on
        // `buffer_start`/`num_frames`) are in bounds; the inner `unsafe fn` calls
        // (`process_block`, dual/single-frame conv, activation and residual kernels) are reached
        // under those same validated preconditions with `M` dispatched by runtime CPUID checks.
        unsafe {
            // F-10: fail-closed clamp — never panic on an oversized `num_frames`,
            // truncate it to the pre-allocated scratch capacity instead. The
            // scratch buffers are constructed as `CH * WAVENET_MAX_NUM_FRAMES`,
            // so `min(WAVENET_MAX_NUM_FRAMES)` is equivalent to
            // `min(scratch.len() / CH)` while staying division-free on the
            // audio hot-path.
            let num_frames = num_frames.min(WAVENET_MAX_NUM_FRAMES);
            debug_assert!(
                num_frames * CH <= self.scratch_mixin.len(),
                "process_block_internal: num_frames*CH ({}) exceeds scratch_mixin capacity ({})",
                num_frames * CH,
                self.scratch_mixin.len(),
            );
            debug_assert!(
                num_frames * CH <= self.scratch_conv.len(),
                "process_block_internal: num_frames*CH ({}) exceeds scratch_conv capacity ({})",
                num_frames * CH,
                self.scratch_conv.len(),
            );

            #[cfg(test)]
            let t_start = if crate::models::wavenet::layer::TELEMETRY_ACTIVE.with(|a| *a.borrow()) {
                Some(std::time::Instant::now())
            } else {
                None
            };

            let mixin_out = self.scratch_mixin.get_unchecked_mut(..num_frames * CH);
            self.input_mixin
                .process_block::<M>(condition, mixin_out, num_frames);

            #[cfg(test)]
            let t_mixin = if let Some(ts) = t_start {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_MIXIN
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            let conv_slice = self.scratch_conv.get_unchecked_mut(..num_frames * CH);

            let mut i = 0;
            let mut chunks = conv_slice.chunks_exact_mut(2 * CH);
            for chunk in chunks.by_ref() {
                let (out_frame_f0, out_frame_f1) = chunk.split_at_mut(CH);

                let mix_idx_f0 = i * CH;
                let mix_idx_f1 = (i + 1) * CH;
                let mixin_f0 = &mixin_out[mix_idx_f0..mix_idx_f0 + CH];
                let mixin_f1 = &mixin_out[mix_idx_f1..mix_idx_f1 + CH];

                self.conv1d.process_dual_frame_with_mixin::<M>(
                    layer_buffer,
                    out_frame_f0,
                    out_frame_f1,
                    buffer_start + i,
                    buffer_start + i + 1,
                    mixin_f0,
                    mixin_f1,
                );
                i += 2;
            }

            let rem = chunks.into_remainder();
            if !rem.is_empty() {
                let mix_idx = i * CH;
                let mixin_slice = &mixin_out[mix_idx..mix_idx + CH];

                self.conv1d.process_single_frame_with_mixin::<M>(
                    layer_buffer,
                    rem,
                    buffer_start + i,
                    mixin_slice,
                );
            }

            #[cfg(test)]
            let t_conv = if let Some(ts) = t_mixin {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_CONV
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            if let Some(s) = seed {
                match &self.activation {
                    ActivationType::ReLU => {
                        M::relu_and_accumulate_with_seed(head_input, conv_slice, s)
                    }
                    _ => M::tanh_and_accumulate_with_seed(head_input, conv_slice, s),
                }
            } else if is_first_layer {
                match &self.activation {
                    ActivationType::ReLU => M::relu_and_overwrite_block(head_input, conv_slice),
                    _ => M::tanh_and_overwrite_block(head_input, conv_slice),
                }
            } else {
                match &self.activation {
                    ActivationType::ReLU => M::relu_and_accumulate_block(head_input, conv_slice),
                    _ => M::tanh_and_accumulate_block(head_input, conv_slice),
                }
            }

            #[cfg(test)]
            let t_tanh = if let Some(ts) = t_conv {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_TANH
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
                Some(now)
            } else {
                None
            };

            let lb_offset = buffer_start * CH;
            let residual_slice = layer_buffer.get_unchecked(lb_offset..lb_offset + num_frames * CH);

            self.one_by_one.process_residual_batch::<M>(
                conv_slice,
                residual_slice,
                output,
                num_frames,
            );

            #[cfg(test)]
            if let Some(ts) = t_tanh {
                let now = std::time::Instant::now();
                crate::models::wavenet::layer::ACC_ONE_BY_ONE
                    .with(|a| *a.borrow_mut() += now.duration_since(ts));
            }
        }
    }
}
