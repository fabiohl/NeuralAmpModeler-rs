// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ConvNet feed-forward model — chains ConvNetBlock layers sequentially
//! with an optional post-stack head.

use crate::math::common::{AlignedVec, SimdMath};
use crate::models::wavenet::PostStackHead;
use crate::models::wavenet::common::WAVENET_MAX_NUM_FRAMES;

use super::block::ConvNetBlock;

/// ConvNet feed-forward model.
///
/// Composed of a sequence of [`ConvNetBlock`]s chained sequentially,
/// followed by an optional [`PostStackHead`] and a final `head_scale` gain.
///
/// Unlike WaveNet, ConvNet has no gating, no rechannel projections,
/// no condition_dsp, and no dual-array architecture. Each block's
/// output is the input of the next block directly.
#[repr(align(64))]
pub struct ConvNetModel {
    /// Sequential ConvNet blocks.
    pub blocks: Vec<ConvNetBlock>,
    /// Final voltage compensation scale (Target Output Scale).
    pub head_scale: f32,
    /// Total receptive field (sum of all block RFs + head RF contribution).
    pub receptive_field_size: usize,
    /// Optional post-stack head sub-object (Conv1D + activation).
    pub post_stack_head: Option<PostStackHead>,
    /// Scratch buffer for post-stack head output.
    pub head_output_scratch: AlignedVec<f32>,
    /// Ping-pong scratch buffers for block-to-block signal relay.
    pub(crate) scratch_a: AlignedVec<f32>,
    pub(crate) scratch_b: AlignedVec<f32>,
    /// Whether to execute prewarm during `reset()`. Default: `true`.
    pub prewarm_on_reset: bool,
    /// Optional C++ flat-format linear head weights (no activation).
    /// Used when `post_stack_head` is `None` but the model has a separate
    /// linear projection from `in_ch → out_ch`.
    pub linear_head: Option<LinearHead>,
}

/// Simple linear projection head for C++ flat ConvNet format.
///
/// The weight matrix is consumed by the batched GEMV kernel
/// `M::gemv_with_bias_f32`, which expects column-major layout
/// `weights[in_c * out_ch + out_c]`. For `out_ch == 1` (the only current
/// construction path — C++ flat ConvNet always serializes a mono head) this
/// coincides with row-major `weight[o * in_ch + i]`.
#[derive(Clone)]
#[repr(align(64))]
pub struct LinearHead {
    /// Column-major weight matrix: `weights[in_c * out_ch + out_c]`.
    pub weight: AlignedVec<f32>,
    /// Bias vector: out_ch.
    pub bias: AlignedVec<f32>,
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of output channels.
    pub out_ch: usize,
}

impl ConvNetModel {
    /// Returns the number of input channels expected by the first block.
    pub fn in_channels(&self) -> usize {
        self.blocks.first().map(|b| b.conv.in_ch).unwrap_or(1)
    }

    /// Returns the number of output channels produced by the model.
    pub fn out_channels(&self) -> usize {
        if let Some(ref head) = self.post_stack_head {
            head.out_channels()
        } else if let Some(ref linear) = self.linear_head {
            linear.out_ch
        } else {
            self.blocks.last().map(|b| b.conv.out_ch).unwrap_or(1)
        }
    }

    /// Resolves the full forward pass and produces waveform samples in zero allocation (DSP).
    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        // SAFETY: `dispatch_simd!` dispatches on runtime CPUID feature checks to a
        // matching `#[target_feature]` backend; `input`/`output` are the same valid
        // slices passed to this safe wrapper.
        unsafe { crate::math::common::dispatch_simd!(self, process_internal, input, output) };
    }

    /// SIMD-dispatched processing kernel.
    ///
    /// Chains blocks sequentially: block 0 receives raw input, each subsequent
    /// block receives the output of the previous block. After all blocks,
    /// the result is passed through the optional post-stack head and
    /// scaled by `head_scale`.
    #[inline(always)]
    unsafe fn process_internal<M: SimdMath>(&mut self, input: &[f32], output: &mut [f32]) {
        // Each frame writes `out_ch` interleaved output samples; clamp the frame
        // count so the output is never indexed beyond its actual length.
        let out_ch = self.out_channels().max(1);
        let total_frames = input.len().min(output.len() / out_ch);
        if total_frames == 0 || self.blocks.is_empty() {
            output[..total_frames].fill(0.0);
            return;
        }

        let mut pos = 0;

        while pos < total_frames {
            let num_frames = (total_frames - pos).min(WAVENET_MAX_NUM_FRAMES);
            let in_slice = &input[pos..pos + num_frames];

            let num_blocks = self.blocks.len();
            // SAFETY: &mut self borrows blocks exclusively; Vec won't reallocate
            // while this mutable reference exists. Calling as_mut_ptr() is sound
            // because the Vec is never resized, dropped, or aliased during the
            // function body. All pointer arithmetic respects i ∈ 0..num_blocks,
            // and blocks_ptr.add(i) points to a valid initialized ConvNetBlock.
            let blocks_ptr = self.blocks.as_mut_ptr();

            // Block 0 writes to scratch_a
            // SAFETY: blocks_ptr is valid (see above). blocks is non-empty
            // (guarded at L90). i=0 is within bounds.
            let first_out_ch = unsafe { (*blocks_ptr).conv.out_ch };
            let dst_a = &mut self.scratch_a[..num_frames * first_out_ch];
            // SAFETY: blocks_ptr and dst_a are valid; M provides safe SIMD dispatch.
            unsafe {
                (*blocks_ptr).process_block_internal::<M>(in_slice, dst_a, num_frames);
            }

            let mut src_is_a = true;

            for i in 1..num_blocks {
                // SAFETY: i < num_blocks by loop bound. blocks_ptr.add(i) yields
                // a valid pointer to an initialized ConvNetBlock in the Vec.
                let curr = unsafe { &mut *blocks_ptr.add(i) };
                let curr_out_ch = curr.conv.out_ch;

                if src_is_a {
                    // SAFETY: `i - 1 < num_blocks` (i ∈ 1..num_blocks), so
                    // `blocks_ptr.add(i - 1)` points to a valid initialized
                    // ConvNetBlock; reading its `conv.out_ch` is valid.
                    let prev_out_ch = unsafe { (*blocks_ptr.add(i - 1)).conv.out_ch };
                    let src = &self.scratch_a[..num_frames * prev_out_ch];
                    let dst = &mut self.scratch_b[..num_frames * curr_out_ch];
                    // SAFETY: curr is valid (see above); src/dst are valid
                    // slices within scratch_a/scratch_b which are separate
                    // allocations.
                    unsafe {
                        curr.process_block_internal::<M>(src, dst, num_frames);
                    }
                } else {
                    // SAFETY: `i - 1 < num_blocks` (i ∈ 1..num_blocks), so
                    // `blocks_ptr.add(i - 1)` points to a valid initialized
                    // ConvNetBlock; reading its `conv.out_ch` is valid.
                    let prev_out_ch = unsafe { (*blocks_ptr.add(i - 1)).conv.out_ch };
                    let src = &self.scratch_b[..num_frames * prev_out_ch];
                    let dst = &mut self.scratch_a[..num_frames * curr_out_ch];
                    // SAFETY: Same invariants as the if branch; src and dst
                    // are reversed between scratch_a/scratch_b.
                    unsafe {
                        curr.process_block_internal::<M>(src, dst, num_frames);
                    }
                }

                src_is_a = !src_is_a;
            }

            let last_result_in_a = (num_blocks - 1).is_multiple_of(2);
            // SAFETY: num_blocks > 0 (guarded at L90). num_blocks-1 < num_blocks,
            // so blocks_ptr.add(num_blocks-1) is in bounds.
            let last_out_ch = unsafe { (*blocks_ptr.add(num_blocks - 1)).conv.out_ch };
            let last_slice = if last_result_in_a {
                &self.scratch_a[..num_frames * last_out_ch]
            } else {
                &self.scratch_b[..num_frames * last_out_ch]
            };

            if let Some(ref mut head_proc) = self.post_stack_head {
                let head_out_ch = head_proc.out_channels();
                let head_scratch = &mut self.head_output_scratch[..num_frames * head_out_ch];
                // SAFETY: `last_slice` has `num_frames * last_out_ch` elements and
                // `head_scratch` has `num_frames * head_out_ch`, matching the head's
                // input/output channel dimensions (validated at construction);
                // `process_block`'s documented preconditions hold.
                unsafe {
                    head_proc.process_block(last_slice, head_scratch, num_frames);
                }
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * out_ch];
                out_slice.copy_from_slice(head_scratch);
                // SAFETY: `out_slice` is a valid mutable slice of `num_frames * out_ch`
                // elements and `M` matches the CPU ISA; `apply_gain`'s documented
                // precondition holds.
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            } else if let Some(ref linear) = self.linear_head {
                let lh_out_ch = linear.out_ch;
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * lh_out_ch];
                // SAFETY: `gemv_with_bias_f32`'s documented preconditions hold:
                // `last_slice` and `out_slice` divide exactly by `num_frames`
                // (`in_len = last_out_ch`, `out_len = lh_out_ch`), `linear.weight`
                // has `in_len * out_len` elements and `linear.bias` at least `out_len`
                // (both validated at construction), `num_frames > 0`, and `M` matches
                // the CPU ISA.
                unsafe {
                    M::gemv_with_bias_f32(
                        last_slice,
                        &linear.weight,
                        &linear.bias,
                        out_slice,
                        num_frames,
                    );
                }
                // SAFETY: `out_slice` is a valid mutable slice of `num_frames * lh_out_ch`
                // elements and `M` matches the CPU ISA; `apply_gain`'s documented
                // precondition holds.
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            } else {
                let out_start = pos * out_ch;
                let out_slice = &mut output[out_start..out_start + num_frames * out_ch];
                out_slice.copy_from_slice(last_slice);
                // SAFETY: `out_slice` is a valid mutable slice of `num_frames * out_ch`
                // elements and `M` matches the CPU ISA; `apply_gain`'s documented
                // precondition holds.
                unsafe {
                    M::apply_gain(out_slice, self.head_scale);
                }
            }

            pos += num_frames;
        }
    }

    /// Stabilizes the model by processing silence (Zero Input) for pre-warm.
    ///
    /// Replicates `nam::DSP::prewarm()` semantics from NAMcore (`dsp.cpp:67-96`):
    /// processes `receptive_field_size + 1` samples of silence through the
    /// entire network, leaving internal states at the zero-input fixed point.
    #[cold]
    pub fn prewarm(&mut self) {
        let n = self.receptive_field_size + 1;
        let zeros = vec![0.0f32; n];
        let mut sink = vec![0.0f32; n * self.out_channels()];
        self.process(&zeros, &mut sink);
    }
}

#[cfg(test)]
#[path = "convnet_model_test.rs"]
mod tests;
