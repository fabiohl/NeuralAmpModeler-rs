// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Generic RT-safe variable delay line.
//!
//! Bounded ring buffer that delays a sample stream by a fixed number of
//! samples. The buffer is allocated once off-RT with 64-byte aligned storage;
//! the per-sample path performs no allocation, no locking and no logging.
//!
//! # Real-time contract
//!
//! - `new` / `with_capacity` / `resize` allocate and must be called off-RT
//!   only (all are marked `#[cold]`).
//! - `push` / `pop` / `set_delay` are O(1) index arithmetic and `reset` an
//!   O(n) fill — all over pre-allocated storage, zero allocations, zero
//!   locks.
//! - Per-sample usage is `push` then `pop`: after pushing input `x[i]`, `pop`
//!   returns `x[i - delay]` (zero-primed: `T::default()` for `i < delay`).
//!
//! Block usage is [`DelayLine::process_block`]: it applies the exact same
//! `push`/`pop` semantics to a whole slice in one fused pass.
//!
//! # Capacity model
//!
//! The ring holds `capacity + 1` slots (exposed by [`DelayLine::capacity`]) so
//! a `push`-then-`pop` pair yields exactly `delay` samples of latency
//! (`delay == 0` is a passthrough). `set_delay` retargets the latency anywhere
//! up to `capacity` without touching the heap; `resize` reallocates and must
//! stay off-RT. `reset` (and `resize`) clear history to `T::default()`.

use crate::math::common::AlignedVec;

/// Generic RT-safe delay line.
///
/// `T` must be `Copy` (bitwise-duplicable ring slots, matching `AlignedVec`)
/// and `Default` (zero-priming value; `0.0` for `f32`).
#[derive(Debug)]
pub struct DelayLine<T: Copy> {
    buf: AlignedVec<T>,
    /// Index of the next slot to write.
    head: usize,
    /// Applied delay in samples (`< capacity` by construction).
    delay: usize,
}

impl<T: Copy + Default> DelayLine<T> {
    /// Creates a delay line with `max_len` samples of latency (off-RT only).
    ///
    /// Allocates `max_len + 1` aligned slots via [`AlignedVec`] and primes
    /// them with `T::default()`.
    #[cold]
    pub fn new(max_len: usize) -> Self {
        let cap = max_len.saturating_add(1).max(1);
        let buf = AlignedVec::from_vec(vec![T::default(); cap])
            .expect("DelayLine::new: aligned allocation failed");
        Self {
            buf,
            head: 0,
            delay: max_len,
        }
    }

    /// Creates a delay line pre-allocated for `capacity` samples of latency
    /// (off-RT only).
    ///
    /// Allocates `capacity + 1` aligned slots via [`AlignedVec`] once and
    /// primes them with `T::default()`, so the effective delay can later be
    /// retargeted at any time with [`DelayLine::set_delay`] without ever
    /// reallocating. `delay` is clamped to `capacity` (the maximum latency
    /// the ring supports).
    #[cold]
    pub fn with_capacity(capacity: usize, delay: usize) -> Self {
        let cap = capacity.saturating_add(1).max(1);
        let buf = AlignedVec::from_vec(vec![T::default(); cap])
            .expect("DelayLine::with_capacity: aligned allocation failed");
        Self {
            buf,
            head: 0,
            delay: delay.min(cap.saturating_sub(1)),
        }
    }

    /// Pushes one sample into the line (RT-safe, zero allocations).
    #[inline(always)]
    pub fn push(&mut self, sample: T) {
        let cap = self.buf.len().max(1);
        // SAFETY: `head < cap == buf.len()` is maintained by construction
        // (`head` is always written modulo `cap` below and reset by
        // `new`/`resize`; `cap >= 1`), so the index is in bounds.
        debug_assert!(self.head < cap);
        self.buf[self.head] = sample;
        self.head += 1;
        if self.head >= cap {
            self.head = 0;
        }
    }

    /// Returns the sample from `delay` pushes ago (RT-safe, zero allocations).
    ///
    /// Does not consume: repeated `pop` calls without an intervening `push`
    /// return the same delayed sample. Prime value is `T::default()`.
    #[inline(always)]
    pub fn pop(&mut self) -> T {
        let cap = self.buf.len().max(1);
        debug_assert!(self.head < cap);
        // Read `delay` behind the just-written slot: `head` points at the
        // next write, so the newest sample lives at `head - 1`.
        let rp = (self.head + cap * 2 - 1 - self.delay.min(cap - 1)) % cap;
        self.buf[rp]
    }

    /// Reconfigures the delay to `new_len` samples (off-RT only).
    ///
    /// Reallocates the ring and clears history to `T::default()`.
    /// Must never be called on the RT thread.
    #[cold]
    pub fn resize(&mut self, new_len: usize) {
        let cap = new_len.saturating_add(1).max(1);
        self.buf = AlignedVec::from_vec(vec![T::default(); cap])
            .expect("DelayLine::resize: aligned allocation failed");
        self.head = 0;
        self.delay = new_len;
    }

    /// Retargets the applied delay to `delay` samples in place
    /// (RT-safe, zero allocations).
    ///
    /// Only the read-offset arithmetic changes: the aligned storage, the
    /// write head and the in-flight history are preserved, so the transition
    /// is sample-continuous and free of reallocation. Values above
    /// [`DelayLine::capacity`] clamp to `capacity`.
    #[inline(always)]
    pub fn set_delay(&mut self, delay: usize) {
        let cap = self.buf.len().max(1);
        self.delay = delay.min(cap.saturating_sub(1));
    }

    /// Clears the entire sample history (RT-safe, zero allocations).
    ///
    /// Refills the ring with `T::default()` and rewinds the write head to the
    /// first slot, restoring the zero-primed state of a freshly constructed
    /// line while keeping the current capacity and delay.
    #[inline(always)]
    pub fn reset(&mut self) {
        // `fill` rewrites the aligned storage in place through `DerefMut` —
        // no allocation, no temporary buffer.
        self.buf.fill(T::default());
        self.head = 0;
    }

    /// Returns the applied delay in samples.
    #[inline(always)]
    pub fn latency_samples(&self) -> usize {
        self.delay
    }

    /// Returns the pre-allocated latency capacity in samples (the exclusive
    /// upper bound applied by [`DelayLine::set_delay`]).
    ///
    /// The aligned ring owns `capacity + 1` slots; the extra slot is what
    /// lets `pop` reach exactly `capacity` samples behind the write head.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.buf.len().saturating_sub(1)
    }

    /// Delays a whole block in one fused pass (RT-safe, zero allocations).
    ///
    /// Bit-identical to calling [`DelayLine::push`] then [`DelayLine::pop`]
    /// for each of the first `n = min(input.len(), output.len())` samples
    /// (including zero-priming for the first `delay` samples of a fresh or
    /// [`DelayLine::reset`] line), but replaces the per-sample index
    /// arithmetic with bulk copies: the block is copied into the ring
    /// through at most two `copy_from_slice` segments per safe chunk and
    /// copied out the same way, with the read point resolved once per
    /// chunk. `delay == 0` short-circuits into a direct passthrough copy.
    ///
    /// A chunk is never longer than `ring_len - delay` slots: that is the
    /// maximum history that can be pushed into the ring before the read
    /// points of the same chunk overtake the write points and would race
    /// ahead of the just-written samples. Within a chunk the write stream
    /// wraps the ring at most once (two segments); the read stream starts
    /// exactly `delay` slots behind the write head and progresses in lock
    /// step, so it also wraps at most once. Blocks longer than the safe
    /// chunk (e.g. delays near [`DelayLine::capacity`]) simply pay more
    /// chunk iterations while staying copy-shaped.
    ///
    /// `input` must not overlap `output` (overlapping in-place use is not
    /// supported); the ring owns internal storage, so both caller slices
    /// can freely alias an earlier `reset`/`set_delay` call site.
    ///
    /// # Real-time safety
    ///
    /// Pure index arithmetic plus slice copies over pre-allocated aligned
    /// storage: no allocation, no lock, no logging. Invalid slice lengths
    /// are handled by the `min` reduction — never by panicking.
    #[inline]
    pub fn process_block(&mut self, input: &[T], output: &mut [T]) {
        let n = input.len().min(output.len());
        if n == 0 {
            return;
        }
        let ring = self.buf.len().max(1);
        debug_assert!(self.head < ring);
        let delay = self.delay.min(ring - 1);

        if delay == 0 {
            // Passthrough: `pop` on a fresh line is the just-pushed sample,
            // so the whole block is a plain copy; the ring only has to
            // absorb the newest `ring` samples to stay coherent for later
            // `set_delay` retargets.
            let mut head = self.head;
            let mut w = 0;
            while w < n {
                let take = (ring - head).min(n - w);
                self.buf[head..head + take].copy_from_slice(&input[w..w + take]);
                head += take;
                if head >= ring {
                    head -= ring;
                }
                w += take;
            }
            self.head = head;
            output[..n].copy_from_slice(&input[..n]);
            return;
        }

        // With a deeper delay, every read point trails the write head by
        // exactly `delay` slots. A chunk longer than `ring - delay` lets
        // reads overtake writes through the wrap: the chunk's remaining
        // pushes reach the slots that its trailing zero-priming-tail reads
        // (samples `i < delay`) still expect to see as pre-chunk history,
        // so those reads would return freshly overwritten in-chunk
        // samples. Capped chunks keep every read slot strictly behind all
        // writes of the same chunk.
        let chunk_len = ring - delay;
        let mut head = self.head;
        let mut done = 0;
        while done < n {
            let chunk = (n - done).min(chunk_len);
            let chunk_head = head;
            // Write the chunk into the ring (≤ 2 segments per chunk: split
            // at the sole wrap crossing).
            let mut w = 0;
            let mut wh = head;
            while w < chunk {
                let take = (ring - wh).min(chunk - w);
                self.buf[wh..wh + take].copy_from_slice(&input[done + w..done + w + take]);
                wh += take;
                if wh >= ring {
                    wh -= ring;
                }
                w += take;
            }
            head = wh;
            // Read the chunk out from the point `delay` slots behind the
            // write start: after pushing input[i] the newest sample lives
            // at `head - 1`, so the read of the chunk's first sample sits
            // at `chunk_head + ring - delay` (mod `ring`) and advances one
            // slot per sample, wrapping at most once per chunk.
            let mut rp = (chunk_head + ring - delay) % ring;
            let mut r = 0;
            while r < chunk {
                let take = (ring - rp).min(chunk - r);
                output[done + r..done + r + take].copy_from_slice(&self.buf[rp..rp + take]);
                rp += take;
                if rp >= ring {
                    rp -= ring;
                }
                r += take;
            }
            done += chunk;
        }
        self.head = head;
    }

    /// Copies the whole ring state (storage, write cursor and applied
    /// delay) from `src` into `self` (RT-safe, zero allocations).
    ///
    /// Both lines must have been configured with the same ring length
    /// (identical `capacity` requests); the debug assertion enforces it and
    /// the release path clamps defensively to the shorter ring. The typical
    /// use is re-seeding a second line from a line that received a mirrored
    /// mono signal, so its future reads continue seamlessly from the same
    /// history instead of replaying stale or zero-primed content.
    ///
    /// # Real-time safety
    ///
    /// One slice copy over pre-allocated aligned storage: no allocation, no
    /// lock, no logging.
    #[inline]
    pub fn copy_state_from(&mut self, src: &Self) {
        let ring = self.buf.len().max(1);
        let src_ring = src.buf.len().max(1);
        debug_assert_eq!(
            ring, src_ring,
            "copy_state_from requires identical ring lengths"
        );
        let common = ring.min(src_ring);
        self.buf[..common].copy_from_slice(&src.buf[..common]);
        self.head = src.head.min(ring - 1);
        self.delay = src.delay.min(ring - 1);
    }
}

#[cfg(test)]
#[path = "delay_line_test.rs"]
mod delay_line_test;
