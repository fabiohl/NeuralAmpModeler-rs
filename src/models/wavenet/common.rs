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
/// **Release-stable invariant (R-2 / A4 owner):** `buffer_start` is a
/// construction-validated counter that always satisfies
/// `buffer_start >= receptive_field_size`:
/// - [`WaveNetLayerState::new`] returns `Err` when the initial
///   `buffer_start < receptive_field_size`, and sizes the mirrored buffer to at
///   least `receptive_field_size + (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES`
///   frames (page-rounded up by `MirroredBuffer`), so the wrap in
///   [`WaveNetLayerState::advance_frames`] lands at
///   `buffer_frames - 63 >= receptive_field_size + 1537`;
/// - [`WaveNetLayerState::advance_frames`] only increases `buffer_start` (or
///   subtracts a full `buffer_frames` at the wrap margin, keeping the pointer in
///   the second half of the 2N mapping), preserving the inequality.
///
/// The wavenet array builders pass the *array-level* receptive field — the sum
/// `Σ (K - 1) * dilation_i` — to every layer state, so
/// `receptive_field_size >= (K - 1) * dilation` holds for each layer of the
/// array. Together with `buffer_start >= receptive_field_size`, this is the
/// release-stable owner of the causal tap-read invariant
/// `frame_idx >= dilation * (K - 1)` that the `conv1d*` kernels document as
/// their correctness precondition (`conv1d.rs` / `conv1d_dual.rs`; the kernels
/// also carry an F-01 `.max(0)` clamp so even a violating caller cannot read
/// out of bounds in release).
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
    ///
    /// # Errors
    /// Returns `Err` when any intermediate size computation overflows `usize`
    /// (checked arithmetic, F-RES2-03) or when the resolved `buffer_start`
    /// lands below `receptive_field_size`. No `unwrap()`/`expect()` on this
    /// path: every product/difference that feeds the checked allocator
    /// (`MirroredBuffer::<f32>::new_aligned`, itself checked) is validated
    /// first, so a hostile dimension rejected upstream (F-RES2-02) — or a
    /// direct caller with extreme values — can never wrap into a smaller
    /// buffer in release builds (overflow-checks off). For valid inputs the
    /// arithmetic is bit-identical to the previous unchecked version.
    pub fn new(
        channels: usize,
        receptive_field_size: usize,
        alloc_num: usize,
    ) -> std::io::Result<Self> {
        // [STEP 1: Calculate Temporal Buffer Size]
        // The buffer needs to accommodate the receptive field and block padding.
        // Page rounding is done internally by MirroredBuffer.
        let pad_frames = (LAYER_ARRAY_BUFFER_PADDING + 1)
            .checked_mul(WAVENET_MAX_NUM_FRAMES)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "WaveNetLayerState size overflows usize: (LAYER_ARRAY_BUFFER_PADDING + 1) * WAVENET_MAX_NUM_FRAMES = ({} + 1) * {}",
                        LAYER_ARRAY_BUFFER_PADDING, WAVENET_MAX_NUM_FRAMES
                    ),
                )
            })?;
        let min_buffer_frames = receptive_field_size.checked_add(pad_frames).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WaveNetLayerState size overflows usize: receptive_field_size ({receptive_field_size}) + pad_frames ({pad_frames})"
                ),
            )
        })?;

        let requested_elems = min_buffer_frames.checked_mul(channels).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WaveNetLayerState size overflows usize: min_buffer_frames ({min_buffer_frames}) * channels ({channels})"
                ),
            )
        })?;
        let buffer = MirroredBuffer::<f32>::new_aligned(requested_elems, channels)?;

        let actual_buffer_frames = buffer.size() / channels;

        // [STEP 2: Initial Offset (Allocated Jittering)]
        // Position the initial pointer in the second half of the virtual mapping (offset N).
        // This allows looking backwards (receptive field) without crossing the virtual buffer start.
        let jitter = (alloc_num % LAYER_ARRAY_BUFFER_PADDING) + 1;
        let jitter_frames = WAVENET_MAX_NUM_FRAMES.checked_mul(jitter).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WaveNetLayerState offset overflows usize: WAVENET_MAX_NUM_FRAMES ({WAVENET_MAX_NUM_FRAMES}) * jitter ({jitter})"
                ),
            )
        })?;
        let doubled = actual_buffer_frames.checked_mul(2).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WaveNetLayerState offset overflows usize: actual_buffer_frames ({actual_buffer_frames}) * 2"
                ),
            )
        })?;
        let start = doubled.checked_sub(jitter_frames).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "WaveNetLayerState offset underflows: {doubled} - {jitter_frames} (actual_buffer_frames={actual_buffer_frames}, jitter={jitter})"
                ),
            )
        })?;

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

    /// Returns whether this delay line can still be consumed by the unchecked
    /// kernels for `channels` input channels and a full
    /// `WAVENET_MAX_NUM_FRAMES` block.
    ///
    /// All fields of this type are public, so a safe caller may replace
    /// `buffer_start`, `receptive_field_size` or the whole `layer_buffer` after
    /// construction. The kernels read the mirrored view through raw pointers
    /// and `get_unchecked`, which makes this predicate the release-stable
    /// gate that the safe model entry points consult before dispatching.
    #[inline]
    pub(crate) fn is_sound_for(&self, channels: usize) -> bool {
        self.start_is_sound(self.buffer_start, channels)
    }

    /// Immutable variant of [`Self::is_sound_for`] for a candidate start.
    ///
    /// Used by the safe `restore_buffer_starts` APIs to validate the whole
    /// slice before applying any value, keeping restores atomic.
    #[inline]
    pub(crate) fn start_is_sound(&self, candidate: usize, channels: usize) -> bool {
        if channels == 0 || self.receptive_field_size > isize::MAX as usize {
            return false;
        }
        let size = self.layer_buffer.size();
        if size == 0 || !size.is_multiple_of(channels) {
            return false;
        }
        let frames = size / channels;
        // `advance_frames` preserves the `buffer_start >= receptive_field`
        // invariant across chunks (including the wrap path) only with this
        // margin. Constructors size the buffer with a larger margin; this
        // release-stable bound also covers replacement `MirroredBuffer`s.
        let Some(min_frames) = self
            .receptive_field_size
            .checked_add(WAVENET_MAX_NUM_FRAMES)
        else {
            return false;
        };
        if frames < min_frames {
            return false;
        }
        let Some(virtual_frames) = frames.checked_mul(2) else {
            return false;
        };
        let Some(block_end) = candidate.checked_add(WAVENET_MAX_NUM_FRAMES) else {
            return false;
        };
        candidate >= self.receptive_field_size && block_end <= virtual_frames
    }

    /// Fallible clone for activation paths where panic must not cross FFI.
    pub fn try_clone(&self) -> std::io::Result<Self> {
        Ok(Self {
            layer_buffer: self.layer_buffer.try_clone()?,
            buffer_start: self.buffer_start,
            receptive_field_size: self.receptive_field_size,
        })
    }

    /// Clears the delay-line storage and restores the ring cursor, reproducing
    /// the freshly-constructed state.
    ///
    /// The full `2N` virtual view of [`MirroredBuffer`] is zeroed: both halves
    /// alias the same physical pages, so a single `fill` clears the storage
    /// exactly once while keeping both mirrored halves zero. The cursor is set
    /// to `receptive_field_size` (not a literal zero) because the causal read
    /// reaches `buffer_start - (K - 1) * dilation` and this struct's
    /// construction invariant requires `buffer_start >= receptive_field_size`.
    /// With an all-zero delay line every tap reads zero, so the exact cursor
    /// value is irrelevant to the output: the cleared state is bit-equivalent
    /// to a cold model.
    ///
    /// Zero-allocation: in-place fill only.
    #[inline]
    pub fn clear(&mut self) {
        self.layer_buffer.fill(0.0);
        self.buffer_start = self.receptive_field_size;
    }

    /// Executes one Ring Buffer pointer step. If it reaches the margin, it wraps back to the start.
    pub fn advance_frames(&mut self, num_frames: usize, channels: usize) {
        self.buffer_start += num_frames;
        let buffer_frames = self.layer_buffer.size() / channels;

        // SAFETY: the construction-time invariant (see the struct docs) guarantees
        // `buffer_start >= receptive_field_size` before and after the wrap below:
        // the mirrored buffer holds `buffer_frames >= receptive_field_size + 1600`
        // frames and the wrap only fires when `buffer_start + 64 > 2 * buffer_frames`,
        // landing at `buffer_start - buffer_frames >= buffer_frames - 63 >=
        // receptive_field_size + 1537`. This debug net is a redundant tripwire for
        // struct-literal misuse that bypasses `WaveNetLayerState::new`.
        debug_assert!(
            self.buffer_start >= self.receptive_field_size,
            "advance_frames underflow: buffer_start ({}) < receptive_field_size ({})",
            self.buffer_start,
            self.receptive_field_size
        );

        // [MIRRORED BUFFER]
        // If the next max-size block (64) could overflow the 2N mapping limit,
        // we rewind the pointer to the first half (maintaining virtual address parity).
        // This ensures [buffer_start .. buffer_start + 64] is always a safe access.
        if self.buffer_start + WAVENET_MAX_NUM_FRAMES > buffer_frames * 2 {
            self.buffer_start -= buffer_frames;
        }
        debug_assert!(
            self.buffer_start >= self.receptive_field_size,
            "advance_frames post-wrap underflow: buffer_start ({}) < receptive_field_size ({})",
            self.buffer_start,
            self.receptive_field_size
        );
    }
}
