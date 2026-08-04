// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Common types, constants, and layer state for WaveNet modules.

use crate::dsp::mirror_buf::MirroredBuffer;

/// Maximum frames to process in one callback pulse.
pub const WAVENET_MAX_NUM_FRAMES: usize = 64;
/// Circular temporal padding of memory buffers in the Ring Buffers framework.
pub const LAYER_ARRAY_BUFFER_PADDING: usize = 24;
/// Maximum supported kernel size.
pub const MAX_KERNEL: usize = 16;

/// Processing context to optimize parameter passing on the WaveNet hot-path.
/// Unifies the needs of static (const generic) models.
pub struct WavenetProcessContext<'a> {
    /// Conditioning (sidechain) buffer.
    pub condition: &'a [f32],
    /// Head accumulator (Skip-Connection).
    pub head_input: &'a mut [f32],
    /// Layer output buffer (for the next layer or final output).
    pub output: &'a mut [f32],
    /// Circular buffer of the current layer (delay line).
    pub layer_buffer: &'a [f32],
    /// Starting index in the circular buffer.
    pub buffer_start: usize,
    /// Number of frames to process.
    pub num_frames: usize,
    /// On-stack temporary buffer for intermediate calculations.
    pub block: &'a mut [f32],
    /// Indicates if this is the first layer of the array.
    pub is_first_layer: bool,
    /// Optional seed for fused seed+tanh accumulation on the first layer.
    /// Eliminates the separate `copy_from_slice` of `prev_head_outputs`.
    pub seed: Option<&'a [f32]>,
}

/// Manages the buffer memory of a WaveNet cell.
///
/// 64B (cache line) alignment is sufficient because this struct lives exclusively
/// on the DSP thread — there is no inter-thread sharing that would require 128B anti-false-sharing.
#[repr(align(64))]
#[derive(Clone)]
pub struct WaveNetLayerState {
    /// Mirrored Buffer (zero allocations in DSP context, rewind elimination).
    pub layer_buffer: MirroredBuffer<f32>,
    /// Numeric pointer of the current frame (advances with each processed frame).
    pub buffer_start: usize,
    /// Physical dimension of the receptive vector space (size of the dilation history).
    pub receptive_field_size: usize,
}

impl WaveNetLayerState {
    /// Static allocator constructor for State (execute before DSP Thread).
    pub fn new(
        channels: usize,
        receptive_field_size: usize,
        alloc_num: usize,
    ) -> std::io::Result<Self> {
        // [STEP 1: Calculate Temporal Buffer Size]
        // The buffer needs to accommodate the receptive field and block padding.
        // Page rounding is done internally by MirroredBuffer.
        let min_buffer_frames =
            receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES;

        let buffer = MirroredBuffer::<f32>::new_aligned(min_buffer_frames * channels, channels)?;

        let actual_buffer_frames = buffer.size() / channels;

        // [STEP 2: Initial Offset (Allocated Jittering)]
        // Position the initial pointer in the second half of the virtual mapping (offset N).
        // This allows looking backwards (receptive field) without crossing the virtual buffer start.
        let jitter = (alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1;
        let start = actual_buffer_frames * 2 - (WAVENET_MAX_NUM_FRAMES * jitter);

        if start < receptive_field_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "buffer_start ({}) is smaller than receptive_field_size ({}); \
                     increase LAYER_ARRAY_BUFFER_PADDING or reduce RF",
                    start, receptive_field_size
                ),
            ));
        }

        Ok(Self {
            layer_buffer: buffer,
            buffer_start: start,
            receptive_field_size,
        })
    }

    /// Fallible clone for activation paths where panic must not cross FFI.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self {
            layer_buffer: self.layer_buffer.try_clone()?,
            buffer_start: self.buffer_start,
            receptive_field_size: self.receptive_field_size,
        })
    }

    /// Executes one Ring Buffer pointer step. If it reaches the margin, it wraps back to the start.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.size() / channels;

        // [MIRRORED BUFFER]
        // If the next max-size block (64) could overflow the 2N mapping limit,
        // we rewind the pointer to the first half (maintaining virtual address parity).
        // This ensures [buffer_start .. buffer_start + 64] is always a safe access.
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames * 2 {
            self.buffer_start -= buffer_frames;
        }
    }
}
