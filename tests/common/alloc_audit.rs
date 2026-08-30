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
    static DEALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
    static REALLOC_COUNT_TLS: Cell<usize> = const { Cell::new(0) };
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

fn get_local_dealloc_count() -> usize {
    DEALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

fn set_local_dealloc_count(val: usize) {
    let _ = DEALLOC_COUNT_TLS.try_with(|count| count.set(val));
}

fn get_local_realloc_count() -> usize {
    REALLOC_COUNT_TLS.try_with(|count| count.get()).unwrap_or(0)
}

fn set_local_realloc_count(val: usize) {
    let _ = REALLOC_COUNT_TLS.try_with(|count| count.set(val));
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
        if is_tracking_active() {
            let _ = DEALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_tracking_active() {
            let _ = REALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
            let _ = ALLOC_COUNT_TLS.try_with(|count| {
                count.set(count.get() + 1);
            });
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

pub struct TrackingGuard;

impl TrackingGuard {
    pub fn new() -> Self {
        set_tracking_active(true);
        set_local_alloc_count(0);
        set_local_dealloc_count(0);
        set_local_realloc_count(0);
        Self
    }
}

impl Drop for TrackingGuard {
    fn drop(&mut self) {
        set_tracking_active(false);
        set_local_alloc_count(0);
        set_local_dealloc_count(0);
        set_local_realloc_count(0);
    }
}

pub fn get_alloc_count() -> usize {
    get_local_alloc_count()
}

pub fn set_alloc_count(val: usize) {
    set_local_alloc_count(val)
}

pub fn get_dealloc_count() -> usize {
    get_local_dealloc_count()
}

pub fn set_dealloc_count(val: usize) {
    set_local_dealloc_count(val)
}

pub fn get_realloc_count() -> usize {
    get_local_realloc_count()
}

pub fn set_realloc_count(val: usize) {
    set_local_realloc_count(val)
}
