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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_zero_is_passthrough() {
        let mut line = DelayLine::<f32>::new(0);
        assert_eq!(line.latency_samples(), 0);
        for i in 0..64 {
            let x = i as f32 * 0.01;
            line.push(x);
            assert_eq!(
                line.pop(),
                x,
                "delay=0 must return the just-pushed sample at {i}"
            );
        }
    }

    #[test]
    fn test_delay_n_shifts_by_n_with_zero_priming() {
        for &delay in &[1usize, 12, 64, 256] {
            let mut line = DelayLine::<f32>::new(delay);
            assert_eq!(line.latency_samples(), delay);
            let n = delay + 64;
            let input: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 + 0.5).collect();
            for (i, &x) in input.iter().enumerate() {
                line.push(x);
                let expected = if i >= delay { input[i - delay] } else { 0.0 };
                assert_eq!(line.pop(), expected, "delay={delay} shift mismatch at {i}");
            }
        }
    }

    #[test]
    fn test_push_pop_circular_wrap_exact_shift() {
        let delay = 128usize;
        let mut line = DelayLine::<f32>::new(delay);
        let n = 2048;
        let input: Vec<f32> = (0..n).map(|i| (i % 997) as f32 * 0.001).collect();
        for (i, &x) in input.iter().enumerate() {
            line.push(x);
            let expected = if i >= delay { input[i - delay] } else { 0.0 };
            assert_eq!(line.pop(), expected, "wrap shift mismatch at {i}");
        }
    }

    #[test]
    fn test_resize_clears_and_updates_latency() {
        let mut line = DelayLine::<f32>::new(8);
        for i in 0..32 {
            line.push(i as f32);
            let _ = line.pop();
        }
        line.resize(32);
        assert_eq!(line.latency_samples(), 32);
        // Fresh history: first `delay` pops after resize are zero-primed.
        for i in 0..32 {
            line.push(0.5);
            assert_eq!(line.pop(), 0.0, "post-resize priming at {i}");
        }
        line.push(0.5);
        assert_eq!(
            line.pop(),
            0.5,
            "steady state after re-prime must shift correctly"
        );
        line.resize(0);
        assert_eq!(line.latency_samples(), 0);
        line.push(1.25);
        assert_eq!(line.pop(), 1.25);
    }

    #[test]
    fn test_push_pop_is_zero_alloc() {
        use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

        let mut line = DelayLine::<f32>::new(64);
        // Warm once outside the guard.
        line.push(1.0);
        let _ = line.pop();
        let _guard = TrackingGuard::new();
        for i in 0..256 {
            line.push(i as f32 * 0.001);
            let _ = line.pop();
        }
        assert_eq!(
            get_alloc_count(),
            0,
            "push/pop must not allocate on the RT path"
        );
    }

    #[test]
    fn test_with_capacity_preallocates_capacity_plus_one_and_clamps() {
        let line = DelayLine::<f32>::with_capacity(3200, 64);
        assert_eq!(
            line.capacity(),
            3200,
            "capacity accessor must match request"
        );
        assert_eq!(line.latency_samples(), 64);

        // Requested delay above capacity clamps to the maximum usable latency.
        let mut clamped = DelayLine::<f32>::with_capacity(8, 100);
        assert_eq!(clamped.capacity(), 8);
        assert_eq!(clamped.latency_samples(), 8);

        // The `capacity + 1` ring slots must sustain the maximum latency
        // (`delay == capacity`) with exact shifts across many wrap-arounds.
        let delay = clamped.capacity();
        let n = 4 * (delay + 1);
        let input: Vec<f32> = (0..n).map(|i| (i % 251) as f32 * 0.01).collect();
        for (i, &x) in input.iter().enumerate() {
            clamped.push(x);
            let expected = if i >= delay { input[i - delay] } else { 0.0 };
            assert_eq!(clamped.pop(), expected, "max-delay shift mismatch at {i}");
        }

        // Degenerate request still yields a usable passthrough line.
        let mut degenerate = DelayLine::<f32>::with_capacity(0, 0);
        assert_eq!(degenerate.capacity(), 0);
        degenerate.push(1.25);
        assert_eq!(degenerate.pop(), 1.25);
    }

    #[test]
    fn test_set_delay_dynamic_transition_preserves_circular_integrity() {
        let mut line = DelayLine::<f32>::with_capacity(64, 4);
        // Mid-stream retargets, including delay == capacity (max latency).
        let changes = [(0usize, 4usize), (10, 31), (60, 0), (100, 63), (160, 7)];
        let n = 256;
        let input: Vec<f32> = (0..n).map(|i| (i % 173) as f32 * 0.01 + 0.25).collect();

        let mut next_change = 0;
        let mut delay = 0usize;
        for (i, &x) in input.iter().enumerate() {
            if next_change < changes.len() && changes[next_change].0 == i {
                delay = changes[next_change].1;
                line.set_delay(delay);
                assert_eq!(line.latency_samples(), delay, "retarget at {i}");
                next_change += 1;
            }
            line.push(x);
            // History is continuous across retargets: `pop` must return the
            // sample pushed exactly `delay` steps ago (zero-primed before).
            let expected = if i >= delay { input[i - delay] } else { 0.0 };
            assert_eq!(
                line.pop(),
                expected,
                "dynamic delay mismatch at i={i} (delay={delay})"
            );
        }

        // Retargets above capacity clamp without corrupting the ring.
        line.set_delay(10_000);
        assert_eq!(line.latency_samples(), 64);
        line.set_delay(7);
        assert_eq!(line.latency_samples(), 7);
        line.push(-2.5);
        assert_eq!(line.pop(), input[n - 7], "post-clamp shift mismatch");
    }

    #[test]
    fn test_reset_clears_history_and_re_primes() {
        let mut line = DelayLine::<f32>::with_capacity(32, 8);
        for i in 0..64 {
            line.push(i as f32 * 0.5);
            let _ = line.pop();
        }

        line.reset();
        assert_eq!(line.capacity(), 32, "reset must preserve capacity");
        assert_eq!(line.latency_samples(), 8, "reset must preserve delay");

        // Full wipe: pops are zero-primed again, then shift exactly.
        for i in 0..24 {
            line.push(0.75);
            let expected = if i >= 8 { 0.75 } else { 0.0 };
            assert_eq!(line.pop(), expected, "post-reset priming at {i}");
        }

        // A second reset rewinds the head from an arbitrary wrap position:
        // the ring behaves exactly like a freshly constructed line.
        line.reset();
        let input: Vec<f32> = (0..96).map(|i| (i % 71) as f32 * 0.02).collect();
        for (i, &x) in input.iter().enumerate() {
            line.push(x);
            let expected = if i >= 8 { input[i - 8] } else { 0.0 };
            assert_eq!(line.pop(), expected, "post-reset shift mismatch at {i}");
        }
    }

    #[test]
    fn test_set_delay_and_reset_are_zero_alloc() {
        use crate::common::alloc_audit::{TrackingGuard, get_alloc_count};

        let mut line = DelayLine::<f32>::with_capacity(3200, 16);
        // Warm once outside the guard.
        line.push(1.0);
        let _ = line.pop();
        let _guard = TrackingGuard::new();
        for i in 0..512 {
            line.set_delay(i % 3201);
            line.push(i as f32 * 0.001);
            let _ = line.pop();
            if i % 128 == 0 {
                line.reset();
            }
        }
        assert_eq!(
            line.capacity(),
            3200,
            "RT-path retargets must not reallocate"
        );
        assert_eq!(
            get_alloc_count(),
            0,
            "set_delay/reset/push/pop must not allocate on the RT path"
        );
    }
}
