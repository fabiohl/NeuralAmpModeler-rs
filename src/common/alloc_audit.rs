// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared allocation audit infrastructure for RT-Safety verification.
//!
//! Provides `CountingAllocator` (the "Memory Watchdog") and `TrackingGuard`
//! used to prove that hot-path DSP code performs zero heap allocations and
//! deallocations.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::AtomicBool;

/// Flag controlling whether heap allocation tracking is enabled.
pub static AUDIT_ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TRACKING_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
    static DEALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
    static REALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
}

/// Returns the current thread's heap allocation count.
pub fn get_alloc_count() -> usize {
    ALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

/// Sets the current thread's heap allocation count.
pub fn set_alloc_count(val: usize) {
    let _ = ALLOC_COUNT_TLS.try_with(|count| count.set(val));
}

/// Returns the current thread's heap deallocation count.
pub fn get_dealloc_count() -> usize {
    DEALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

/// Sets the current thread's heap deallocation count.
pub fn set_dealloc_count(val: usize) {
    let _ = DEALLOC_COUNT_TLS.try_with(|count| count.set(val));
}

/// Returns the current thread's heap reallocation count.
pub fn get_realloc_count() -> usize {
    REALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

/// Sets the current thread's heap reallocation count.
pub fn set_realloc_count(val: usize) {
    let _ = REALLOC_COUNT_TLS.try_with(|count| count.set(val));
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

    /// Intercepts deallocation: increments `DEALLOC_COUNT` if on the watched thread,
    /// then delegates deallocation to the system allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must have been previously allocated via `CountingAllocator::alloc`
    /// with the same `layout`.
    pub unsafe fn dealloc(ptr: *mut u8, layout: Layout) {
        if is_tracking_active() {
            let _ = DEALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        // SAFETY: `ptr` was returned by a matching `CountingAllocator::alloc`
        // with the same `layout` (documented precondition of this `unsafe fn`);
        // `System.dealloc` is then called with a valid `ptr`/`layout` pair.
        unsafe { System.dealloc(ptr, layout) }
    }

    /// Intercepts reallocation: increments `REALLOC_COUNT` and `ALLOC_COUNT` if on
    /// the watched thread, then delegates reallocation to the system allocator.
    ///
    /// # Safety
    ///
    /// `ptr` must have been previously allocated via `CountingAllocator::alloc`
    /// with the same `layout`, and `new_size` must fit alignment requirements.
    pub unsafe fn realloc(ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_tracking_active() {
            let _ = REALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
            let _ = ALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        // SAFETY: `ptr` was allocated with `layout` and `new_size` meets `GlobalAlloc` requirements.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

// SAFETY: every method forwards to `CountingAllocator::alloc`/`dealloc`/`realloc`, which
// delegate to `System`'s `GlobalAlloc`; the trait contract is upheld for the
// documented preconditions (valid `layout` for `alloc`; `ptr`/`layout` pairing
// for `dealloc` and `realloc`).
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
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` was allocated with `layout` (trait contract);
        // `Self::realloc` forwards it to the system allocator.
        unsafe { Self::realloc(ptr, layout, new_size) }
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
        set_dealloc_count(0);
        set_realloc_count(0);
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
        set_alloc_count(0);
        set_dealloc_count(0);
        set_realloc_count(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn fresh_state() {
        set_tracking_active(false);
        set_alloc_count(0);
        set_dealloc_count(0);
        set_realloc_count(0);
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
        set_dealloc_count(42);
        set_realloc_count(42);

        let guard = TrackingGuard::new();

        assert_eq!(get_alloc_count(), 0);
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);

        drop(guard);
    }

    #[test]
    fn tracking_guard_drop_clears_tracking_active_and_counts() {
        fresh_state();

        let guard = TrackingGuard::new();
        assert!(is_tracking_active());
        set_alloc_count(10);
        set_dealloc_count(20);
        set_realloc_count(30);

        drop(guard);

        assert!(!is_tracking_active());
        assert_eq!(get_alloc_count(), 0);
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);
    }

    #[test]
    fn tracking_guard_default_works() {
        fresh_state();

        let guard = TrackingGuard::default();

        assert!(is_tracking_active());
        assert_eq!(get_alloc_count(), 0);
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);

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
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);
    }

    #[test]
    fn box_destruction_increments_dealloc_count() {
        fresh_state();
        let ptr = Box::into_raw(Box::new(42u8));
        let guard = TrackingGuard::new();
        assert_eq!(get_dealloc_count(), 0);

        // SAFETY: `ptr` was obtained from `Box::into_raw` and is valid for reclamation.
        unsafe {
            let _ = Box::from_raw(ptr);
        }
        assert_eq!(get_dealloc_count(), 1);

        drop(guard);
    }

    #[test]
    fn tracking_guard_records_alloc_and_dealloc_negative() {
        fresh_state();
        let guard = TrackingGuard::new();
        assert_eq!(get_alloc_count(), 0);
        assert_eq!(get_dealloc_count(), 0);

        {
            let b = Box::new(42u8);
            assert_eq!(get_alloc_count(), 1);
            drop(b);
        }
        assert_eq!(get_dealloc_count(), 1);

        drop(guard);
    }

    #[test]
    fn getters_and_setters_work() {
        fresh_state();
        assert_eq!(get_alloc_count(), 0);
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);

        set_alloc_count(15);
        set_dealloc_count(25);
        set_realloc_count(35);

        assert_eq!(get_alloc_count(), 15);
        assert_eq!(get_dealloc_count(), 25);
        assert_eq!(get_realloc_count(), 35);
    }

    #[test]
    fn counting_allocator_realloc_increments_counts() {
        fresh_state();
        set_tracking_active(true);

        let layout = Layout::from_size_align(64, 8).unwrap();
        // SAFETY: `layout` is valid (non-zero size, alignment ≤ size).
        let ptr = unsafe { CountingAllocator::alloc(layout) };
        assert_eq!(get_alloc_count(), 1);
        assert_eq!(get_dealloc_count(), 0);
        assert_eq!(get_realloc_count(), 0);

        // SAFETY: `ptr` was allocated with `layout` and `128` fits alignment.
        let new_ptr = unsafe { CountingAllocator::realloc(ptr, layout, 128) };
        assert_eq!(get_alloc_count(), 2);
        assert_eq!(get_realloc_count(), 1);
        assert_eq!(get_dealloc_count(), 0);

        let new_layout = Layout::from_size_align(128, 8).unwrap();
        // SAFETY: `new_ptr` was allocated by `realloc` with `new_layout`.
        unsafe { CountingAllocator::dealloc(new_ptr, new_layout) };
        assert_eq!(get_dealloc_count(), 1);

        set_tracking_active(false);
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
