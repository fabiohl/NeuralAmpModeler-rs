// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Delay line for the polyphase FIR resampler.
//!
//! Implements the "double-buffer" technique for contiguous SIMD access
//! without circular wrap logic in the hot path.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

use super::super::sinc_kernel::TAPS_PER_PHASE;

/// Delay line size (double-buffer) to ensure contiguous access.
/// Maintains 2 copies of history to avoid wrap logic in the hot-path SIMD.
const DELAY_LINE_LEN: usize = TAPS_PER_PHASE * 2;

/// FIR filter state for one channel (mono).
///
/// Uses the "double-buffer" technique: the sample history is kept in two
/// contiguous copies. When inserting a new sample, it is written to both
/// `[write_pos]` and `[write_pos + TAPS_PER_PHASE]`. This ensures that
/// any window of `TAPS_PER_PHASE` consecutive samples from `write_pos`
/// is always contiguous — eliminating the need for circular wrap logic
/// in the SIMD inner loop.
pub(crate) struct DelayLine {
    /// Sample buffer (size = DELAY_LINE_LEN = 2 × TAPS_PER_PHASE).
    pub buf: AlignedVec<f32>,
    /// Write position (0..TAPS_PER_PHASE-1, wrapping).
    pub pos: usize,
}

impl DelayLine {
    pub fn new() -> Result<Self, NamErrorCode> {
        Ok(Self {
            buf: AlignedVec::new(DELAY_LINE_LEN, 0.0f32)?,
            pos: 0,
        })
    }

    /// Inserts a sample into the delay line (double-write for contiguity).
    #[inline(always)]
    pub fn push(&mut self, sample: f32) {
        let pos = self.pos;
        debug_assert!(pos < TAPS_PER_PHASE);
        unsafe {
            *self.buf.get_unchecked_mut(pos) = sample;
            *self.buf.get_unchecked_mut(pos + TAPS_PER_PHASE) = sample;
        }
        self.pos += 1;
        if self.pos >= TAPS_PER_PHASE {
            self.pos = 0;
        }
    }

    /// Returns a pointer to TAPS_PER_PHASE contiguous samples (most recent first).
    #[inline(always)]
    pub fn window_ptr(&self) -> *const f32 {
        debug_assert!(self.pos < TAPS_PER_PHASE);
        debug_assert!(self.pos + TAPS_PER_PHASE <= DELAY_LINE_LEN);
        unsafe { self.buf.as_ptr().add(self.pos) }
    }
}
