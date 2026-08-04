// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WaveNetLayerState, WavenetProcessContext};
use super::dense_dyn::DenseLayerDyn;
use super::layer_dyn::WaveNetLayerDyn;
use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

/// Grouped Multi-Layer WaveNet Unit with runtime dimensions.
#[derive(Clone)]
pub struct WaveNetLayerArrayDyn {
    /// Number of input channels.
    pub in_ch: usize,
    /// Number of conditioning channels (runtime-configurable; 1 for classic WaveNet A1).
    pub cond: usize,
    /// Number of internal channels.
    pub ch: usize,
    /// Kernel size.
    pub k: usize,
    /// Head projection size (skip-connection output dimension).
    pub head: usize,
    /// Vec with structural topology (length defines blocks).
    pub layers: Vec<WaveNetLayerDyn>,
    /// RingBuffer states, one for each Layer in the system.
    pub states: Vec<WaveNetLayerState>,
    /// Initial `Dense` tensor opening.
    pub rechannel: DenseLayerDyn,
    /// Final tensor closure generating Head projection.
    pub head_rechannel: DenseLayerDyn,
    /// Pre-allocated temporary output array.
    pub array_outputs: AlignedVec<f32>,
    /// CH-sized intermediate accumulator for layer contributions before the Head projection.
    pub head_accum: AlignedVec<f32>,
    /// Allocated global Linear projection memory (HEAD-sized).
    pub head_outputs: AlignedVec<f32>,
    /// Dimensional field size (global receptive field) for routing.
    pub receptive_field_size: usize,
    /// Shared activation buffer size.
    pub block_size: usize,
    /// Temporary accumulator for blocks (pre-allocated).
    pub block_buffer: AlignedVec<f32>,
    /// Active number of layers for soft-degrade. Set to `layers.len()` by default.
    pub effective_layers: usize,
}

impl WaveNetLayerArrayDyn {
    /// Sets the effective number of layers for soft-degrade.
    /// Clamped to [1, layers.len()].
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.effective_layers = n.min(self.layers.len()).max(1);
    }

    /// Array's central processing. Fully shielded against allocations.
    ///
    /// When `prev_head_outputs` is `Some`, it is passed as `seed` in the
    /// first layer's `WavenetProcessContext` — the layer then fuses seed
    /// accumulation with tanh in a single SIMD pass. This eliminates the
    /// dedicated `copy_from_slice` of `num_frames * ch` bytes.
    ///
    /// This implements the C++ cascaded head pattern (array N seeds its head
    /// accumulator with the post-head_rechannel output of array N−1).
    ///
    /// # Safety
    /// State pointers iterate internally without bounds checks.
    #[inline(always)]
    pub unsafe fn process_block_internal<M: SimdMath, const PREWARM: bool>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
        prev_head_outputs: Option<&[f32]>,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();
        let ch = self.ch;
        let head = self.head;

        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * ch],
                num_frames,
            );

            let num_layers = self.effective_layers;
            let last_layer = num_layers - 1;

            for (i, layer) in self.layers.iter_mut().enumerate() {
                if i >= num_layers {
                    break;
                }
                let current_state = &mut *states_ptr.add(i);

                if PREWARM {
                    let start_idx = current_state.buffer_start * ch;
                    let src_range = start_idx..start_idx + ch;

                    for offset in 1..=current_state.receptive_field_size {
                        debug_assert!(
                            current_state.buffer_start >= offset,
                            "backfill underflow: bs={}, off={}",
                            current_state.buffer_start,
                            offset
                        );
                        let dst_start = current_state.buffer_start - offset;
                        let dst_idx = dst_start * ch;
                        current_state
                            .layer_buffer
                            .copy_within(src_range.clone(), dst_idx);
                    }
                }

                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * ch],
                        output: &mut self.array_outputs[0..num_frames * ch],
                        layer_buffer: &current_state.layer_buffer[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                        is_first_layer: i == 0,
                        seed: if i == 0 { prev_head_outputs } else { None },
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let n_start = next_state.buffer_start * ch;
                    let next_layer_buffer =
                        &mut next_state.layer_buffer[n_start..n_start + num_frames * ch];

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * ch],
                        output: next_layer_buffer,
                        layer_buffer: &current_state.layer_buffer[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                        is_first_layer: i == 0,
                        seed: if i == 0 { prev_head_outputs } else { None },
                    });
                }

                if !PREWARM {
                    current_state.advance_frames(num_frames, ch);
                }
            }

            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * ch],
                &mut self.head_outputs[0..num_frames * head],
                num_frames,
            );
        }
    }

    /// Processes data in Pre-warm mode to initialize and stabilize temporal memory.
    ///
    /// [SCIENTIFIC EXPLANATION]
    /// Causal convolution neural networks like WaveNet have an internal state that actively
    /// depends on N past steps (Receptive Field). When loading a fresh model, the allocated
    /// network memory (Ring Buffers) contains "pure zeros" or computational garbage.
    /// Pre-warm feeds a continuous inert signal (Absolute Silence) into the network so as to
    /// fill the entire past window. The resulting cold-start transients "drain"
    /// silently into oblivion, ensuring the first audio sample when turning on the device
    /// sounds organic and stable, without pops or clicks.
    ///
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[inline(always)]
    pub unsafe fn prewarm_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        prev_head_outputs: Option<&[f32]>,
    ) {
        unsafe {
            self.process_block_internal::<M, true>(layer_inputs, condition, 1, prev_head_outputs);
        }
    }
}
