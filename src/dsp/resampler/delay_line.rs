// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Delay line for the polyphase FIR resampler.
//!
//! Implements the "double-buffer" technique for contiguous SIMD access
//! without circular wrap logic in the hot path.
//!
//! # Release-stable window invariant (R-2 / A5)
//!
//! The write position always satisfies `pos < TAPS_PER_PHASE`, which makes every
//! `TAPS_PER_PHASE`-sample window starting at `pos` fully in-bounds
//! (`pos + TAPS_PER_PHASE <= DELAY_LINE_LEN = buf.len()`). The owner of this
//! invariant is this module: the fields are private, so only `new`/`push`/`clear`
//! can mutate the state, and each provably preserves it:
//!
//! - `new`: `pos = 0`;
//! - `push`: writes at `pos` and `pos + TAPS_PER_PHASE` (both in bounds because
//!   `pos < TAPS_PER_PHASE`), then advances and wraps to 0 at `TAPS_PER_PHASE`;
//! - `clear`: `pos = 0`.
//!
//! The `debug_assert!`s are a redundant debug tripwire only; in release the
//! guarantee is structural (private state + module-owned transitions), with zero
//! runtime checks in the hot path.

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
///
/// R-2 / A5: the fields are private, so the invariant
/// `pos + TAPS_PER_PHASE <= DELAY_LINE_LEN` is owned release-stable by this
/// module's own transitions (`new`/`push`/`clear`), not by caller discipline.
/// See the module-level documentation for the full proof.
pub(crate) struct DelayLine {
    /// Sample buffer (size = DELAY_LINE_LEN = 2 × TAPS_PER_PHASE).
    buf: AlignedVec<f32>,
    /// Write position (0..TAPS_PER_PHASE-1, wrapping).
    ///
    /// Invariant: `pos < TAPS_PER_PHASE`, maintained release-stable by the wrap
    /// in `push` and the reset to 0 in `new`/`clear` (module-owned; private field).
    pos: usize,
}

impl DelayLine {
    pub fn new() -> Result<Self, NamErrorCode> {
        Ok(Self {
            buf: AlignedVec::new(DELAY_LINE_LEN, 0.0f32)?,
            pos: 0,
        })
    }

    /// Inserts a sample into the delay line (double-write for contiguity).
    ///
    /// Maintains the module-owned invariant `pos < TAPS_PER_PHASE` release-stable:
    /// the write position wraps to 0 at `TAPS_PER_PHASE`. The `debug_assert!` is a
    /// redundant debug tripwire.
    #[inline(always)]
    pub fn push(&mut self, sample: f32) {
        let pos = self.pos;
        debug_assert!(pos < TAPS_PER_PHASE);
        // SAFETY: `pos < TAPS_PER_PHASE` is owned release-stable by this module
        // (the wrap at the end of this fn, plus `new`/`clear` resetting to 0; the
        // fields are private, so no external caller can corrupt `pos` — the
        // `debug_assert!` above is a redundant tripwire only), and `buf` has
        // `2 * TAPS_PER_PHASE` elements (DELAY_LINE_LEN), so both
        // `get_unchecked_mut` indices (`pos`, `pos + TAPS_PER_PHASE`) are in bounds.
        unsafe {
            *self.buf.get_unchecked_mut(pos) = sample;
            *self.buf.get_unchecked_mut(pos + TAPS_PER_PHASE) = sample;
        }
        self.pos += 1;
        if self.pos >= TAPS_PER_PHASE {
            self.pos = 0;
        }
    }

    /// Clears all history (zero-fills the double buffer) without deallocating.
    ///
    /// RT-safe: zero allocations. Used by resampler reset paths.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }

    /// Returns a pointer to `TAPS_PER_PHASE` contiguous samples (most recent first).
    ///
    /// # Contract
    ///
    /// The returned pointer is valid for exactly `TAPS_PER_PHASE` reads of `f32`
    /// (`pos + TAPS_PER_PHASE <= DELAY_LINE_LEN = buf.len()`). This holds
    /// release-stable by construction — `pos < TAPS_PER_PHASE` is owned by this
    /// module's state transitions (`new`/`push`/`clear`; fields are private) — so
    /// no runtime check is needed on the hot path. The `debug_assert!`s below are
    /// a redundant debug tripwire.
    #[inline(always)]
    pub fn window_ptr(&self) -> *const f32 {
        debug_assert!(self.pos < TAPS_PER_PHASE);
        debug_assert!(self.pos + TAPS_PER_PHASE <= DELAY_LINE_LEN);
        // SAFETY: `pos < TAPS_PER_PHASE` is owned release-stable by the module state
        // transitions (`new` initializes to 0, `push` wraps at TAPS_PER_PHASE, `clear`
        // resets to 0; the fields are private so no external code can corrupt `pos`),
        // hence `pos + TAPS_PER_PHASE <= DELAY_LINE_LEN = buf.len()` — the pointer is
        // non-null, aligned to f32, and valid for `TAPS_PER_PHASE` reads (the
        // `debug_assert!`s above are a redundant debug tripwire only).
        unsafe { self.buf.as_ptr().add(self.pos) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A5 / R-2 — the module-owned invariant survives arbitrary push cycles in both
    /// debug and release: `pos` always wraps into `0..TAPS_PER_PHASE`, so the
    /// `window_ptr` read window `pos + TAPS_PER_PHASE` never exceeds the double
    /// buffer. `clear` restores the initial state.
    #[test]
    fn test_delay_line_push_wrap_keeps_window_in_bounds() {
        let mut dl = DelayLine::new().expect("allocation should succeed in tests");
        for count in 0..(TAPS_PER_PHASE * 4) {
            dl.push(count as f32);
            assert!(
                dl.pos < TAPS_PER_PHASE,
                "push {count}: pos {} must stay below TAPS_PER_PHASE {}",
                dl.pos,
                TAPS_PER_PHASE
            );
            assert!(
                dl.pos + TAPS_PER_PHASE <= DELAY_LINE_LEN,
                "push {count}: window ending at {} exceeds delay line {}",
                dl.pos + TAPS_PER_PHASE,
                DELAY_LINE_LEN
            );
            let ptr = dl.window_ptr();
            // SAFETY: `ptr` is derived from `dl.buf.as_ptr()` by `window_ptr`, so both
            // pointers belong to the same allocation and `offset_from` is defined.
            let offset = unsafe { ptr.offset_from(dl.buf.as_ptr()) };
            assert_eq!(
                offset, dl.pos as isize,
                "push {count}: window must start exactly at pos"
            );
        }
        dl.clear();
        assert_eq!(dl.pos, 0, "clear must reset the write position");
        // SAFETY: `window_ptr()` returns a pointer derived from `dl.buf.as_ptr()`
        // (same allocation), so `offset_from` is defined.
        let offset = unsafe { dl.window_ptr().offset_from(dl.buf.as_ptr()) };
        assert_eq!(
            offset, 0,
            "window must restart at the buffer base after clear"
        );
    }

    /// A5 / R-2 — at the last legal write position the window reaches the last
    /// element of the double buffer (exclusive end `pos + TAPS_PER_PHASE ==
    /// DELAY_LINE_LEN - 1`); the full `TAPS_PER_PHASE` read stays in bounds in both
    /// debug and release.
    #[test]
    fn test_delay_line_window_reaches_buffer_end() {
        let mut dl = DelayLine::new().expect("allocation should succeed in tests");
        // Drive pos to TAPS_PER_PHASE - 1, the maximum legal write position.
        for _ in 0..(TAPS_PER_PHASE - 1) {
            dl.push(1.0);
        }
        assert_eq!(dl.pos, TAPS_PER_PHASE - 1);
        let ptr = dl.window_ptr();
        // SAFETY: `ptr.add(TAPS_PER_PHASE)` is the exclusive end of the window within
        // `buf` (same allocation), which makes `offset_from` valid.
        let end = unsafe { ptr.add(TAPS_PER_PHASE).offset_from(dl.buf.as_ptr()) };
        assert_eq!(
            end,
            (DELAY_LINE_LEN - 1) as isize,
            "window must end at the last element of the delay line"
        );
        // SAFETY: `pos + TAPS_PER_PHASE < DELAY_LINE_LEN == buf.len()`, so the whole
        // window `[pos, pos + TAPS_PER_PHASE)` is inside `buf`.
        for i in 0..TAPS_PER_PHASE {
            // SAFETY: `i < TAPS_PER_PHASE` bounds the read inside the window proven
            // in-bounds above (ends at `pos + TAPS_PER_PHASE < buf.len()`).
            let _ = unsafe { ptr.add(i).read() };
        }
    }

    /// A5 / R-2 — release-only worst case: even if a module-internal bug drove `pos`
    /// to its maximum possible value (`TAPS_PER_PHASE`), the double buffer still
    /// yields a fully in-bounds read window (the second copy), never an out-of-bounds
    /// read. In debug the tripwire fires instead
    /// (`test_delay_line_window_ptr_debug_tripwire`).
    #[cfg(not(debug_assertions))]
    #[test]
    fn test_delay_line_window_worst_case_internal_state_release() {
        let mut dl = DelayLine::new().expect("allocation should succeed in tests");
        dl.pos = TAPS_PER_PHASE; // module-internal worst case; window = second copy
        let ptr = dl.window_ptr();
        // SAFETY: `ptr.add(TAPS_PER_PHASE)` is the one-past-end of `buf` (same allocation).
        let end = unsafe { ptr.add(TAPS_PER_PHASE).offset_from(dl.buf.as_ptr()) };
        assert_eq!(end, DELAY_LINE_LEN as isize);
        // SAFETY: the window `[TAPS_PER_PHASE, 2*TAPS_PER_PHASE)` = `[DELAY_LINE_LEN/2,
        // DELAY_LINE_LEN)` lies entirely inside `buf` (len = DELAY_LINE_LEN), as proven by
        // the offset check above.
        let mut acc = 0.0f32;
        for i in 0..TAPS_PER_PHASE {
            // SAFETY: `i < TAPS_PER_PHASE` bounds the read inside the window proven
            // in-bounds above (ends at `2*TAPS_PER_PHASE == buf.len()`).
            acc += unsafe { ptr.add(i).read() };
        }
        assert_eq!(
            acc, 0.0,
            "worst-case window reads the zero-initialized second copy"
        );
    }

    /// A5 / R-2 — debug tripwire: the `debug_assert!` still fires when the write
    /// position is corrupted past `TAPS_PER_PHASE - 1` (module-internal mutation
    /// only; the fields are private). In release the same state is structurally
    /// impossible via the safe API and, if it ever occurred internally, the double
    /// buffer would still keep the read window in bounds
    /// (`test_delay_line_window_worst_case_internal_state_release`).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "self.pos < TAPS_PER_PHASE")]
    fn test_delay_line_window_ptr_debug_tripwire() {
        let mut dl = DelayLine::new().expect("allocation should succeed in tests");
        dl.pos = TAPS_PER_PHASE;
        let _ = dl.window_ptr();
    }
}
