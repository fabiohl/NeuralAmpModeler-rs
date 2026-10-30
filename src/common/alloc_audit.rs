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
#[path = "alloc_audit_test.rs"]
mod tests;
