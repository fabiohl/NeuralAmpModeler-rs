// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared CountingAllocator infrastructure for heap-audit integration tests.
//!
//! Provides a local `CountingAllocator`, `TrackingGuard` (RAII gate that
//! starts/stops allocation counting), and `get_alloc_count()`.
//!
//! Each test binary registers its own `#[global_allocator]` referencing
//! [`CountingAllocator`]; this module only provides the shared type.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACKING_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
}

fn is_tracking_active() -> bool {
    TRACKING_ACTIVE
        .try_with(|active| active.get())
        .unwrap_or(false)
}

fn set_tracking_active(active: bool) {
    let _ = TRACKING_ACTIVE.try_with(|a| a.set(active));
}

fn get_local_alloc_count() -> usize {
    ALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

fn set_local_alloc_count(val: usize) {
    let _ = ALLOC_COUNT_TLS.try_with(|count| count.set(val));
}

pub struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_tracking_active() {
            let _ = ALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

pub struct TrackingGuard;

impl TrackingGuard {
    pub fn new() -> Self {
        set_tracking_active(true);
        set_local_alloc_count(0);
        Self
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        set_tracking_active(false);
    }
}

pub fn get_alloc_count() -> usize {
    get_local_alloc_count()
}
