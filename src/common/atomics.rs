// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loom-compatible atomic type aliases for lock-free production structures.
//!
//! Every atomic used in a production concurrency protocol must be routed
//! through these aliases so the Loom model-checking engine can instrument it.
//!
//! Under a Loom build (`--cfg loom`, see `utils/tests-long.sh` phase 6) the
//! aliases resolve to `loom::sync::atomic` wrappers: Loom's permutation engine
//! then explores every thread interleaving and fails the test if an ordering
//! violation or data race exists. Outside Loom builds they resolve to the
//! standard library atomics with zero runtime overhead (a plain re-export).
//!
//! `Ordering` is deliberately *not* aliased: `loom::sync::atomic::Ordering` is
//! a re-export of `core::sync::atomic::Ordering`, so production code should
//! keep importing `core::sync::atomic::Ordering` unchanged and only swap the
//! atomic *types* through this module.
//!
//! `AtomicBool` is intentionally absent: the one production `AtomicBool` is the
//! process-global `SHUTDOWN` flag in `spsc/status.rs`, which must stay on the
//! standard library (loom's `AtomicBool::new` is not `const`, so it cannot back
//! a `static`), and no other production protocol uses it.

#[cfg(not(loom))]
pub use core::sync::atomic::AtomicI32;
/// Atomic 32-bit signed integer: loom-instrumented under `--cfg loom`.
#[cfg(loom)]
pub use loom::sync::atomic::AtomicI32;

#[cfg(not(loom))]
pub use core::sync::atomic::AtomicI64;
/// Atomic 64-bit signed integer: loom-instrumented under `--cfg loom`.
#[cfg(loom)]
pub use loom::sync::atomic::AtomicI64;

#[cfg(not(loom))]
pub use core::sync::atomic::AtomicU32;
/// Atomic 32-bit unsigned integer: loom-instrumented under `--cfg loom`.
#[cfg(loom)]
pub use loom::sync::atomic::AtomicU32;

#[cfg(not(loom))]
pub use core::sync::atomic::AtomicU64;
/// Atomic 64-bit unsigned integer: loom-instrumented under `--cfg loom`.
#[cfg(loom)]
pub use loom::sync::atomic::AtomicU64;

#[cfg(not(loom))]
pub use core::sync::atomic::AtomicUsize;
/// Atomic pointer-width unsigned integer: loom-instrumented under `--cfg loom`.
#[cfg(loom)]
pub use loom::sync::atomic::AtomicUsize;

use core::sync::atomic::Ordering;

/// Extension trait providing cross-compatibility between Loom and modern `core::sync::atomic`.
///
/// Modern `core::sync::atomic` deprecated `fetch_update` in favor of `try_update`,
/// whereas `loom` only provides `fetch_update`. This trait bridges the two.
pub trait AtomicExt<T> {
    /// Performs an atomic update compatible with both `std` and `loom`.
    fn atomic_update<F>(&self, set_order: Ordering, fetch_order: Ordering, f: F) -> Result<T, T>
    where
        F: FnMut(T) -> Option<T>;

    /// Atomically increments the value saturating at maximum.
    fn saturating_inc(&self, set_order: Ordering, fetch_order: Ordering) -> Result<T, T>;
}

impl AtomicExt<u32> for AtomicU32 {
    #[inline]
    fn atomic_update<F>(&self, set_order: Ordering, fetch_order: Ordering, f: F) -> Result<u32, u32>
    where
        F: FnMut(u32) -> Option<u32>,
    {
        #[cfg(not(loom))]
        {
            self.try_update(set_order, fetch_order, f)
        }
        #[cfg(loom)]
        {
            self.fetch_update(set_order, fetch_order, f)
        }
    }

    #[inline]
    fn saturating_inc(&self, set_order: Ordering, fetch_order: Ordering) -> Result<u32, u32> {
        self.atomic_update(set_order, fetch_order, |v| Some(v.saturating_add(1)))
    }
}
