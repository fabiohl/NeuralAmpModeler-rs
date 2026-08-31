// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WaveNetLayerState, WavenetProcessContext};
use super::dense::DenseLayer;
use super::layer::WaveNetLayer;
use crate::math::common::{AlignedVec, SimdMath};
use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};

/// Grouped Multi-Layer WaveNet Unit.
#[derive(Clone)]
pub struct WaveNetLayerArray<
    const IN: usize,
    const COND: usize,
    const CH: usize,
    const K: usize,
    const HEAD: usize,
> {
    /// Vec with structural topology (length defines blocks).
    pub layers: Vec<WaveNetLayer<COND, CH, K>>,
    /// RingBuffer states, one for each Layer in the system.
    pub states: Vec<WaveNetLayerState>,
    /// Initial `Dense` tensor opening.
    pub rechannel: DenseLayer<IN, CH>,
    /// Final tensor closure generating Head projection.
    pub head_rechannel: DenseLayer<CH, HEAD>,

    /// Pre-allocated temporary output array.
    /// Array output temporary accumulator.
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

impl<const IN: usize, const COND: usize, const CH: usize, const K: usize, const HEAD: usize>
    WaveNetLayerArray<IN, COND, CH, K, HEAD>
{
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
    /// dedicated `copy_from_slice` of `num_frames * CH` bytes.
    ///
    /// This implements the C++ cascaded head pattern (array N seeds its head
    /// accumulator with the post-head_rechannel output of array N−1).
    ///
    /// # Safety
    /// State pointers iterate internally without bounds checks.
    #[inline]
    pub unsafe fn process_block_internal<M: SimdMath, const PREWARM: bool>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        num_frames: usize,
        prev_head_outputs: Option<&[f32]>,
    ) {
        debug_assert_eq!(self.layers.len(), self.states.len());
        let states_ptr = self.states.as_mut_ptr();

        // SAFETY: `states_ptr` is derived from `self.states` with `layers.len() == states.len()`
        // (the `debug_assert_eq!` above), and every index used is `< num_layers <= states.len()`
        // (`states_ptr.add(0)`, `add(i)`, and `add(i + 1)` guarded by `i + 1 < num_layers`), so
        // each `&mut *` reborrow is valid and targets a distinct state slot per layer iteration;
        // the `_mm_prefetch` calls only touch the addresses, not the contents.
        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * CH;
            self.rechannel.process_block::<M>(
                layer_inputs,
                &mut state_0.layer_buffer[start..start + num_frames * CH],
                num_frames,
            );

            let num_layers = self.effective_layers;
            let last_layer = num_layers - 1;

            // [STEP 4: Layer Inference Cascade]
            for (i, layer) in self.layers.iter_mut().enumerate() {
                if i >= num_layers {
                    break;
                }
                let current_state = &mut *states_ptr.add(i);

                if PREWARM {
                    // [STEP 4.1: Static State Propagation (Backfill)]
                    // In prewarm mode, we replicate the current sample to the entire history (Receptive Field).
                    // This ensures the network stabilizes instantly to the stationary value.
                    let start_idx = current_state.buffer_start * CH;
                    let src_range = start_idx..start_idx + CH;

                    for offset in 1..=current_state.receptive_field_size {
                        debug_assert!(
                            current_state.buffer_start >= offset,
                            "backfill underflow: bs={}, off={}",
                            current_state.buffer_start,
                            offset
                        );
                        // SAFETY: Guaranteed by the WaveNetLayerState::new constructor, which enforces buffer_start >= receptive_field_size.
                        let dst_start = current_state.buffer_start - offset;
                        let dst_idx = dst_start * CH;
                        current_state
                            .layer_buffer
                            .copy_within(src_range.clone(), dst_idx);
                    }
                }

                // Software Prefetch the next state in the cascade.
                // Bring the cache line of state i+1 (and i+2 if possible) into L1
                // while the processor resolves the arithmetic pipeline of the current layer.
                if i + 1 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 1) as *const i8);
                }
                if i + 2 < num_layers {
                    _mm_prefetch::<_MM_HINT_T0>(states_ptr.add(i + 2) as *const i8);
                }

                if i == last_layer {
                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * CH],
                        output: &mut self.array_outputs[0..num_frames * CH],
                        layer_buffer: &current_state.layer_buffer[..],
                        buffer_start: current_state.buffer_start,
                        num_frames,
                        block: &mut self.block_buffer[0..num_frames * self.block_size],
                        is_first_layer: i == 0,
                        seed: if i == 0 { prev_head_outputs } else { None },
                    });
                } else {
                    let next_state = &mut *states_ptr.add(i + 1);
                    let n_start = next_state.buffer_start * CH;
                    let next_layer_buffer =
                        &mut next_state.layer_buffer[n_start..n_start + num_frames * CH];

                    layer.process_block_internal::<M>(WavenetProcessContext {
                        condition,
                        head_input: &mut self.head_accum[0..num_frames * CH],
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
                    current_state.advance_frames(num_frames, CH);
                }
            }

            // [STEP 5: Dimensional Closure (Head Rechannel)]
            // The dense matrix funnels the accumulator (sum of skip-connections from all layers,
            // of size `CH`) into a smaller `HEAD` dimension (e.g., 16 -> 8 or 16 -> 1).
            // Uses f32 native precision for this critical final projection (tonal fidelity),
            // while the backbone runs quantized (BF16/F16) for performance.
            self.head_rechannel.process_block::<M>(
                &self.head_accum[0..num_frames * CH],
                &mut self.head_outputs[0..num_frames * HEAD],
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
    /// # Safety
    /// Call this via `dispatch_simd!` macro only.
    #[inline(always)]
    pub unsafe fn prewarm_internal<M: SimdMath>(
        &mut self,
        layer_inputs: &[f32],
        condition: &[f32],
        prev_head_outputs: Option<&[f32]>,
    ) {
        // SAFETY: `prewarm_internal` is an `unsafe fn` only reachable through the `dispatch_simd!`
        // macro (per its docs), which dispatches on runtime CPUID feature checks to a matching
        // `#[target_feature]` backend `M`; the buffers are the same validated slices passed to
        // this safe wrapper.
        unsafe {
            // Unified via shared code: we process 1 frame with the prewarm flag active.
            // [STEP 4.1] inside `process_block_internal` handles the backfill.
            self.process_block_internal::<M, true>(layer_inputs, condition, 1, prev_head_outputs);
        }
    }
}
