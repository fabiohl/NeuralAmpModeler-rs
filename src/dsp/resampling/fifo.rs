// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Bounded, pre-allocated FIFO for the streaming resample adapter.
//!
//! RT-safe: all storage is allocated in [`SampleFifo::new`]; the hot-path
//! operations (`push`, `pop_into`, `unpop`, `clear`) never allocate.
//! The FIFO is a circular buffer of contiguous `f32` samples with explicit
//! read/length bookkeeping, splitting wrap-around segments into contiguous
//! copies. No `unsafe` is used: all segment copies go through safe slice
//! `copy_from_slice`.

use crate::common::diagnostics::NamErrorCode;
use crate::math::common::AlignedVec;

/// Bounded circular FIFO of `f32` samples (single channel).
pub struct SampleFifo {
    buf: AlignedVec<f32>,
    /// Read position (0..capacity).
    read: usize,
    /// Number of valid samples currently stored.
    len: usize,
}

impl SampleFifo {
    /// Creates a FIFO with the given capacity (at least 1 sample).
    ///
    /// # Errors
    /// Returns [`NamErrorCode::OutOfMemory`] on allocation failure.
    pub fn new(capacity: usize) -> Result<Self, NamErrorCode> {
        let capacity = capacity.max(1);
        Ok(Self {
            buf: AlignedVec::new(capacity, 0.0f32)?,
            read: 0,
            len: 0,
        })
    }

    /// Total storage capacity in samples.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.cap()
    }

    /// Number of samples currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when no samples are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of sample slots currently free.
    #[inline]
    pub fn free(&self) -> usize {
        self.capacity() - self.len
    }

    /// Appends up to `src.len()` samples, returning the number actually stored
    /// (never more than [`free`](SampleFifo::free)). Excess is left unread in
    /// `src` — the FIFO is bounded and never overflows.
    #[inline]
    pub fn push(&mut self, src: &[f32]) -> usize {
        let n = src.len().min(self.free());
        if n == 0 {
            return 0;
        }
        let cap = self.capacity();
        let write = (self.read + self.len) % cap;
        let first = (cap - write).min(n);
        self.buf[write..write + first].copy_from_slice(&src[..first]);
        if first < n {
            self.buf[..n - first].copy_from_slice(&src[first..n]);
        }
        self.len += n;
        n
    }

    /// Removes up to `dst.len()` samples into `dst`, returning the number moved
    /// (never more than [`len`](SampleFifo::len)). The removed samples remain
    /// physically in the buffer until overwritten, so they can be restored with
    /// [`unpop`](SampleFifo::unpop).
    #[inline]
    pub fn pop_into(&mut self, dst: &mut [f32]) -> usize {
        let n = dst.len().min(self.len);
        if n == 0 {
            return 0;
        }
        let cap = self.capacity();
        let first = (cap - self.read).min(n);
        dst[..first].copy_from_slice(&self.buf[self.read..self.read + first]);
        if first < n {
            dst[first..n].copy_from_slice(&self.buf[..n - first]);
        }
        self.read = (self.read + n) % cap;
        self.len -= n;
        n
    }

    /// Restores `n` samples previously removed by [`pop_into`](SampleFifo::pop_into).
    ///
    /// # Panics
    /// In debug builds, panics if `n > capacity - len` (i.e. the popped slots
    /// were already overwritten or `n` exceeds the capacity). Release builds
    /// clamp `n` to the representable range.
    #[inline]
    pub fn unpop(&mut self, n: usize) {
        let cap = self.capacity();
        let clamped = n.min(cap - self.len);
        debug_assert_eq!(n, clamped, "unpop: {n} samples cannot be restored");
        self.read = (self.read + cap - clamped) % cap;
        self.len += clamped;
    }

    /// Discards all stored samples (does not zero the storage).
    #[inline]
    pub fn clear(&mut self) {
        self.read = 0;
        self.len = 0;
    }
}

#[cfg(test)]
#[path = "fifo_test.rs"]
mod fifo_test;
