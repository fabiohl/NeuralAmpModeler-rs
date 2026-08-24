// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared allocation audit infrastructure for RT-Safety verification.
//!
//! Provides `CountingAllocator` (the "Memory Watchdog") and `TrackingGuard`
//! used to prove that hot-path DSP code performs zero heap allocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::AtomicBool;

/// Flag controlling whether heap allocation tracking is enabled.
pub static AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TRACKING_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
}

/// Returns the current thread's heap allocation count.
pub fn get_alloc_count() -> usize {
    ALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

/// Sets the current thread's heap allocation count.
pub fn set_alloc_count(val: usize) {
    let _ = ALLOC_COUNT_TLS.try_with(|count| count.set(val));
}

/// Checks if heap allocation tracking is active on the current thread.
pub fn is_tracking_active() -> bool {
    TRACKING_ACTIVE
        .try_with(|active| active.get())
        .unwrap_or(false)
}

/// Sets heap allocation tracking active on the current thread.
pub fn set_tracking_active(active: bool) {
    let _ = TRACKING_ACTIVE.try_with(|a| a.set(active));
}

/// The "Memory Watchdog": intercepts all memory requests from the program.
///
/// Implements `GlobalAlloc` directly — register as `#[global_allocator]`
/// with `static GLOBAL: CountingAllocator = CountingAllocator;`.
pub struct CountingAllocator;

impl CountingAllocator {
    /// Intercepts allocation: increments `ALLOC_COUNT` if on the watched thread.
    ///
    /// # Safety
    ///
    /// The caller must ensure `layout` is valid (non-zero size, non-ZST
    /// with alignment ≤ size). This delegates to the system allocator.
    pub unsafe fn alloc(layout: Layout) -> *mut u8 {
        if is_tracking_active() {
            let _ = ALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        // SAFETY: `layout` validity is the caller's documented precondition of
        // this `unsafe fn`; `System` upholds the `GlobalAlloc` contract and
        // returns a pointer that `System.dealloc` with the same layout can free.
        unsafe { System.alloc(layout) }
    }

    /// Delegates deallocation to the system allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must have been previously allocated via `CountingAllocator::alloc`
    /// with the same `layout`.
    pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by a matching `CountingAllocator::alloc`
        // with the same `layout` (documented precondition of this `unsafe fn`);
        // `System.dealloc` is then called with a valid `ptr`/`layout` pair.
        unsafe { System.dealloc(ptr, layout) }
    }
}

// SAFETY: every method forwards to `CountingAllocator::alloc`/`dealloc`, which
// delegate to `System`'s `GlobalAlloc`; the trait contract is upheld for the
// documented preconditions (valid `layout` for `alloc`; `ptr`/`layout` pairing
// for `dealloc`).
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` validity is guaranteed by the `GlobalAlloc` caller;
        // `Self::alloc` forwards it to the system allocator.
        unsafe { Self::alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated with the same `layout` (trait contract);
        // `Self::dealloc` forwards it to the system allocator.
        unsafe { Self::dealloc(ptr, layout) }
    }
}

/// The "Switch": turns on the watchdog when created and turns it off when destroyed.
pub struct TrackingGuard {
    _private: (),
}

impl TrackingGuard {
    /// Starts watching the current thread.
    pub fn new() -> Self {
        set_tracking_active(true);
        set_alloc_count(0);
        Self { _private: () }
    }
}

impl Default for TrackingGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        set_tracking_active(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn fresh_state() {
        set_tracking_active(false);
        set_alloc_count(0);
        AUDIT_ENABLED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn tracking_guard_new_sets_active() {
        fresh_state();
        assert!(!is_tracking_active());

        let guard = TrackingGuard::new();

        assert!(is_tracking_active());

        drop(guard);
    }

    #[test]
    fn tracking_guard_new_resets_alloc_count() {
        set_alloc_count(42);

        let guard = TrackingGuard::new();

        assert_eq!(get_alloc_count(), 0);

        drop(guard);
    }

    #[test]
    fn tracking_guard_drop_clears_tracking_active() {
        fresh_state();

        let guard = TrackingGuard::new();
        assert!(is_tracking_active());

        drop(guard);

        assert!(!is_tracking_active());
    }

    #[test]
    fn tracking_guard_default_works() {
        fresh_state();

        let guard = TrackingGuard::default();

        assert!(is_tracking_active());
        assert_eq!(get_alloc_count(), 0);

        drop(guard);
    }

    #[test]
    fn multiple_guards_work() {
        fresh_state();

        let g1 = TrackingGuard::new();
        assert!(is_tracking_active());

        drop(g1);
        assert!(!is_tracking_active());

        let g2 = TrackingGuard::new();
        assert!(is_tracking_active());

        drop(g2);
        assert!(!is_tracking_active());
    }

    #[test]
    fn alloc_count_is_zero_after_tracking_guard() {
        fresh_state();

        let _g = TrackingGuard::new();
        assert_eq!(get_alloc_count(), 0);
    }

    #[test]
    fn parallel_allocation_tracking_isolation() {
        use std::thread;

        fresh_state();

        let num_threads = 8;
        let mut handles = Vec::new();

        for i in 0..num_threads {
            handles.push(thread::spawn(move || {
                // Each thread starts its own TrackingGuard
                let _guard = TrackingGuard::new();

                // Allocate some boxes to trigger allocations
                let mut v = Vec::new();
                for j in 0..(i + 1) * 10 {
                    v.push(Box::new(j));
                }

                // Read local alloc count
                let count = get_alloc_count();
                // Ensure at least some allocations were captured
                assert!(
                    count >= (i + 1) * 10,
                    "Thread {} should have detected allocations, got {}",
                    i,
                    count
                );
                count
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.join().unwrap());
        }

        // Ensure different threads saw different allocation counts corresponding to their patterns,
        // and did not corrupt each other.
        for i in 1..num_threads {
            assert!(
                results[i] > results[i - 1],
                "Thread allocation counts should be isolated and distinct, got: {:?}",
                results
            );
        }
    }
}
