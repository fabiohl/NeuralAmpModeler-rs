// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::common::{WAVENET_MAX_NUM_FRAMES, WaveNetLayerState, WavenetProcessContext};
use super::conv1d_dyn::Conv1dDyn;
use super::dense_dyn::DenseLayerDyn;
use super::layer_dyn::WaveNetLayerDyn;
use crate::arch::simd::{_MM_HINT_T0, _mm_prefetch};
use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;
use crate::math::common::SimdMath;

/// Final tensor closure generating Head projection (Dense or 1D Conv).
#[derive(Clone)]
pub enum HeadRechannelDyn {
    /// 1x1 matrix multiplication (legacy models with head kernel=1, dilation=1).
    Dense(DenseLayerDyn),
    /// Temporal 1D convolution with delay line state (head kernel > 1 or dilation > 1).
    Conv1d {
        /// 1D convolution layer weights and configuration.
        conv: Conv1dDyn,
        /// State buffer for receptive field history.
        state: WaveNetLayerState,
    },
}

impl From<DenseLayerDyn> for HeadRechannelDyn {
    fn from(dense: DenseLayerDyn) -> Self {
        HeadRechannelDyn::Dense(dense)
    }
}

impl HeadRechannelDyn {
    /// Returns whether the selected head projection still satisfies the
    /// contract of its kernels for `ch` channels and `head` outputs.
    #[inline]
    pub(crate) fn is_sound(&self, ch: usize, head: usize) -> bool {
        match self {
            HeadRechannelDyn::Dense(dense) => {
                dense.in_ch == ch && dense.out_ch == head && dense.is_sound()
            }
            HeadRechannelDyn::Conv1d { conv, state } => {
                conv.in_ch == ch
                    && conv.out_ch == head
                    && conv.is_sound()
                    && conv
                        .backreach()
                        .is_some_and(|back| back <= state.receptive_field_size)
                    && state.is_sound_for(ch)
            }
        }
    }
}

/// Grouped Multi-Layer WaveNet Unit with runtime dimensions.
#[derive(Clone)]
pub struct WaveNetLayerArrayDyn {
    /// Number of input channels.
    pub(crate) in_ch: usize,
    /// Number of conditioning channels (runtime-configurable; 1 for classic WaveNet A1).
    pub(crate) cond: usize,
    /// Number of internal channels.
    pub(crate) ch: usize,
    /// Kernel size.
    pub(crate) k: usize,
    /// Head projection size (skip-connection output dimension).
    pub(crate) head: usize,
    /// Vec with structural topology (length defines blocks).
    pub(crate) layers: Vec<WaveNetLayerDyn>,
    /// RingBuffer states, one for each Layer in the system.
    pub(crate) states: Vec<WaveNetLayerState>,
    /// Initial `Dense` tensor opening.
    pub(crate) rechannel: DenseLayerDyn,
    /// Final tensor closure generating Head projection.
    pub(crate) head_rechannel: HeadRechannelDyn,
    /// Pre-allocated temporary output array.
    pub(crate) array_outputs: AlignedVec<f32>,
    /// CH-sized intermediate accumulator for layer contributions before the Head projection.
    pub(crate) head_accum: AlignedVec<f32>,
    /// Allocated global Linear projection memory (HEAD-sized).
    pub(crate) head_outputs: AlignedVec<f32>,
    /// Dimensional field size (global receptive field) for routing.
    pub(crate) receptive_field_size: usize,
    /// Shared activation buffer size.
    pub(crate) block_size: usize,
    /// Temporary accumulator for blocks (pre-allocated).
    pub(crate) block_buffer: AlignedVec<f32>,
    /// Active number of layers for soft-degrade. Set to `layers.len()` by default.
    pub(crate) effective_layers: usize,
}

impl WaveNetLayerArrayDyn {
    /// Constructs a validated dynamic `WaveNetLayerArrayDyn`.
    ///
    /// Allocates internal working buffers and validates all topology,
    /// layer/state pairing, and tensor invariants once at construction.
    ///
    /// # Errors
    /// Returns [`NamErrorCode::InvalidModelTopology`] if `is_sound()` fails, or
    /// [`NamErrorCode::OutOfMemory`] on allocation failure.
    #[expect(
        clippy::too_many_arguments,
        reason = "Runtime-dimensioned WaveNet layer array constructor initializes geometry, projections, and working buffers"
    )]
    pub fn from_parts(
        in_ch: usize,
        cond: usize,
        ch: usize,
        k: usize,
        head: usize,
        layers: Vec<WaveNetLayerDyn>,
        states: Vec<WaveNetLayerState>,
        rechannel: DenseLayerDyn,
        head_rechannel: HeadRechannelDyn,
        receptive_field_size: usize,
    ) -> Result<Self, NamErrorCode> {
        let frame_elems = ch
            .checked_mul(WAVENET_MAX_NUM_FRAMES)
            .ok_or(NamErrorCode::InvalidModelTopology)?;
        let head_elems = head
            .checked_mul(WAVENET_MAX_NUM_FRAMES)
            .ok_or(NamErrorCode::InvalidModelTopology)?;
        let num_layers = layers.len();
        let array = Self {
            in_ch,
            cond,
            ch,
            k,
            head,
            layers,
            states,
            rechannel,
            head_rechannel,
            array_outputs: AlignedVec::new(frame_elems, 0.0f32)?,
            head_accum: AlignedVec::new(frame_elems, 0.0f32)?,
            head_outputs: AlignedVec::new(head_elems, 0.0f32)?,
            receptive_field_size,
            block_size: ch,
            block_buffer: AlignedVec::new(frame_elems, 0.0f32)?,
            effective_layers: num_layers,
        };
        if !array.is_sound() {
            return Err(NamErrorCode::InvalidModelTopology);
        }
        Ok(array)
    }

    /// Returns the number of input channels.
    #[inline(always)]
    pub fn in_ch(&self) -> usize {
        self.in_ch
    }

    /// Returns the number of conditioning channels.
    #[inline(always)]
    pub fn cond(&self) -> usize {
        self.cond
    }

    /// Returns the number of internal channels.
    #[inline(always)]
    pub fn ch(&self) -> usize {
        self.ch
    }

    /// Returns the kernel size.
    #[inline(always)]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Returns the head projection output width.
    #[inline(always)]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Returns the layers of this array.
    #[inline(always)]
    pub fn layers(&self) -> &[WaveNetLayerDyn] {
        &self.layers
    }

    /// Returns the delay-line states of this array.
    #[inline(always)]
    pub fn states(&self) -> &[WaveNetLayerState] {
        &self.states
    }

    /// Returns the input rechannel dense layer.
    #[inline(always)]
    pub fn rechannel(&self) -> &DenseLayerDyn {
        &self.rechannel
    }

    /// Returns the head rechannel projection.
    #[inline(always)]
    pub fn head_rechannel(&self) -> &HeadRechannelDyn {
        &self.head_rechannel
    }

    /// Returns the pre-allocated array output buffer.
    #[inline(always)]
    pub fn array_outputs(&self) -> &[f32] {
        &self.array_outputs
    }

    /// Returns the intermediate head accumulator buffer.
    #[inline(always)]
    pub fn head_accum(&self) -> &[f32] {
        &self.head_accum
    }

    /// Returns the head projection output buffer.
    #[inline(always)]
    pub fn head_outputs(&self) -> &[f32] {
        &self.head_outputs
    }

    /// Returns the receptive field size of this array.
    #[inline(always)]
    pub fn receptive_field_size(&self) -> usize {
        self.receptive_field_size
    }

    /// Returns the block buffer width.
    #[inline(always)]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the active number of layers for soft-degrade.
    #[inline(always)]
    pub fn effective_layers(&self) -> usize {
        self.effective_layers
    }
    /// Sets the effective number of layers for soft-degrade.
    /// Clamped to [1, layers.len()].
    #[inline(always)]
    pub fn set_effective_layers(&mut self, n: usize) {
        self.effective_layers = n.min(self.layers.len()).max(1);
    }

    /// Returns whether this array can be dispatched to its unchecked kernels.
    ///
    /// Covers the invariants that the public fields can otherwise break after
    /// construction: layer/state pairing, effective-layer bounds, scratch
    /// capacities, tensor geometry and the delay-line windows of every state
    /// (including the optional convolution head).
    pub fn is_sound(&self) -> bool {
        if self.in_ch == 0 || self.cond == 0 || self.ch == 0 || self.head == 0 || self.k == 0 {
            return false;
        }
        if self.layers.is_empty()
            || self.layers.len() != self.states.len()
            || self.effective_layers == 0
            || self.effective_layers > self.layers.len()
        {
            return false;
        }
        let Some(frame_elems) = self.ch.checked_mul(WAVENET_MAX_NUM_FRAMES) else {
            return false;
        };
        let Some(head_elems) = self.head.checked_mul(WAVENET_MAX_NUM_FRAMES) else {
            return false;
        };
        let Some(block_elems) = self.block_size.checked_mul(WAVENET_MAX_NUM_FRAMES) else {
            return false;
        };
        if self.array_outputs.len() < frame_elems
            || self.head_accum.len() < frame_elems
            || self.head_outputs.len() < head_elems
            || self.block_buffer.len() < block_elems
        {
            return false;
        }
        if self.rechannel.in_ch != self.in_ch
            || self.rechannel.out_ch != self.ch
            || !self.rechannel.is_sound()
        {
            return false;
        }
        if !self.head_rechannel.is_sound(self.ch, self.head) {
            return false;
        }
        self.layers
            .iter()
            .zip(self.states.iter())
            .all(|(layer, state)| {
                layer.is_sound(self.ch, self.cond)
                    && layer
                        .conv1d
                        .backreach()
                        .is_some_and(|back| back <= state.receptive_field_size)
                    && state.is_sound_for(self.ch)
            })
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
        // Defense in depth: never derive the loop bound from a value that can
        // exceed the real vector lengths. `effective_layers` is public, so a
        // safe caller may set it above `layers.len()`/`states.len()`.
        let num_layers = self
            .effective_layers
            .min(self.layers.len())
            .min(self.states.len());
        if num_layers == 0 {
            return;
        }
        let states_ptr = self.states.as_mut_ptr();
        let ch = self.ch;
        let head = self.head;

        // SAFETY: `states_ptr` is derived from `self.states` with `layers.len() == states.len()`
        // (the `debug_assert_eq!` above), and every index used is `< num_layers <= states.len()`
        // (`states_ptr.add(0)`, `add(i)`, and `add(i + 1)` guarded by `i + 1 < num_layers`), so
        // each `&mut *` reborrow is valid and targets a distinct state slot per layer iteration;
        // the `_mm_prefetch` calls only touch the addresses, not the contents.
        unsafe {
            let state_0 = &mut *states_ptr.add(0);
            let start = state_0.buffer_start * ch;
            self.rechannel.process_block::<M>(
                layer_inputs,
                state_0.layer_buffer.window_mut(start, num_frames * ch),
                num_frames,
            );

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
                        next_state.layer_buffer.window_mut(n_start, num_frames * ch);

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

            match &mut self.head_rechannel {
                HeadRechannelDyn::Dense(dense) => {
                    dense.process_block::<M>(
                        &self.head_accum[0..num_frames * ch],
                        &mut self.head_outputs[0..num_frames * head],
                        num_frames,
                    );
                }
                HeadRechannelDyn::Conv1d { conv, state } => {
                    let start = state.buffer_start * ch;
                    state
                        .layer_buffer
                        .write_slice_at(start, &self.head_accum[0..num_frames * ch]);

                    if PREWARM {
                        let src_range = start..start + ch;
                        for offset in 1..=state.receptive_field_size {
                            debug_assert!(
                                state.buffer_start >= offset,
                                "head backfill underflow: bs={}, off={}",
                                state.buffer_start,
                                offset
                            );
                            let dst_start = state.buffer_start - offset;
                            state
                                .layer_buffer
                                .copy_within(src_range.clone(), dst_start * ch);
                        }
                    }

                    conv.process_block::<M>(
                        &state.layer_buffer,
                        &mut self.head_outputs[0..num_frames * head],
                        state.buffer_start,
                        num_frames,
                        None,
                    );

                    if !PREWARM {
                        state.advance_frames(num_frames, ch);
                    }
                }
            }
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
        // SAFETY: `prewarm_internal` is an `unsafe fn` only reachable through the `dispatch_simd!`
        // macro (per its docs), which dispatches on runtime CPUID feature checks to a matching
        // `#[target_feature]` backend `M`; the buffers are the same validated slices passed to
        // this safe wrapper.
        unsafe {
            self.process_block_internal::<M, true>(layer_inputs, condition, 1, prev_head_outputs);
        }
    }
}
