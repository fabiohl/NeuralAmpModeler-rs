// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

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

    // Explicit cap: this is a concurrency-isolation test, not a topology probe.
    let num_threads = thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(2);
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
