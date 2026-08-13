// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Utility for 64-byte aligned memory allocation.
//!
//! Ensures that dynamic buffers (weights, accumulators) respect cache line
//! boundaries and AVX2/AVX-512 requirements, avoiding unaligned load/store
//! penalties.
//!
//! For huge-page-backed allocations (> 1 MiB), see `huge_alloc::HugePageVec`.

use std::alloc::{Layout, alloc, dealloc};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

use crate::common::diagnostics::NamErrorCode;

/// Marker trait for types where the all-zero bit pattern is a valid representation.
///
/// # Safety
///
/// The implementor guarantees that calling `std::mem::zeroed::<Self>()` produces
/// a well-defined, valid value of `Self`. All IEEE 754 floating-point types and
/// all primitive integers satisfy this invariant (0.0 is all-zero bits, and 0
/// is all-zero bits).
pub unsafe trait Zeroable: Copy {
    /// Returns `true` when all bytes of this value are zero.
    fn is_zero(&self) -> bool;
}

macro_rules! impl_zeroable_int {
    ($($t:ty),*) => {
        $(
            // SAFETY: for all primitive integer types, the all-zero bit pattern
            // is the canonical representation of the value 0 — this is guaranteed
            // by the Rust language reference.
            unsafe impl Zeroable for $t {
                #[inline(always)]
                fn is_zero(&self) -> bool {
                    *self == 0
                }
            }
        )*
    };
}

impl_zeroable_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

// SAFETY: IEEE 754 represents +0.0 as all-zero bits (`to_bits() == 0`).
// `std::mem::zeroed::<f32>()` is thus the canonical +0.0, valid for all
// arithmetic operations.
unsafe impl Zeroable for f32 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.to_bits() == 0
    }
}

// SAFETY: IEEE 754 represents +0.0 as all-zero bits (`to_bits() == 0`).
// `std::mem::zeroed::<f64>()` is thus the canonical +0.0, valid for all
// arithmetic operations.
unsafe impl Zeroable for f64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.to_bits() == 0
    }
}

/// Wrapper to guarantee 64-byte alignment (Cache Line / AVX-512).
///
/// Stack-allocated counterpart to `AlignedVec`. Use this for fixed-size
/// const-generic buffers on struct fields; use `AlignedVec` for dynamic
/// heap-allocated buffers.
#[repr(align(64))]
#[derive(Clone, Copy, Debug)]
pub struct Aligned64<T>(pub T);

impl<T> Deref for Aligned64<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Aligned64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T: Default> Default for Aligned64<T> {
    #[inline(always)]
    fn default() -> Self {
        Aligned64(T::default())
    }
}

/// A 64-byte aligned buffer (Cache Line / AVX-512).
///
/// Layout (24 bytes): pointer + length + capacity. Zero overhead for standard allocations.
/// For huge-page-backed variants, use `huge_alloc::HugePageVec` instead.
///
/// `T: Copy` is a design invariant (F-10): this container allocates raw
/// contiguous memory and deallocates it in `Drop` **without** running
/// per-element destructors (`drop_in_place`). Restricting `T` to `Copy` makes
/// it impossible to store types with non-trivial `Drop` glue, so no element
/// destructor can ever be silently skipped.
#[derive(Debug)]
pub struct AlignedVec<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

impl<T: Copy> AlignedVec<T> {
    /// The default guaranteed alignment (64 bytes).
    pub const ALIGN: usize = 64;

    /// Allocates a 64-byte aligned block for `capacity` elements.
    ///
    /// Returns the raw pointer and the `Layout` used (for later deallocation).
    /// This is the single allocation entry-point; all constructors route through it.
    fn alloc_slot(capacity: usize) -> Result<(*mut T, Layout), NamErrorCode> {
        if capacity == 0 {
            return Ok((NonNull::dangling().as_ptr(), Layout::new::<()>()));
        }

        let element_size = std::mem::size_of::<T>();
        let byte_size = capacity
            .checked_mul(element_size)
            .ok_or(NamErrorCode::OutOfMemory)?;

        let layout = Layout::from_size_align(byte_size, Self::ALIGN)
            .map_err(|_| NamErrorCode::OutOfMemory)?;

        // SAFETY: layout is derived from valid size (capacity * element_size, checked against
        // overflow) with 64-byte alignment (Self::ALIGN). alloc returns either a valid aligned
        // pointer or null.
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(NamErrorCode::OutOfMemory);
        }

        Ok((ptr as *mut T, layout))
    }

    /// Creates a new aligned buffer and fills it with an initial value.
    ///
    /// Think of a shelf where each compartment has exactly the size the processor
    /// prefers to read (64 bytes). This function reserves the space and places a "default value"
    /// in each spot, leaving everything ready for immediate use.
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation fails.
    pub fn new(len: usize, default: T) -> Result<Self, NamErrorCode>
    where
        T: Zeroable,
    {
        let mut vec = Self::with_capacity(len)?;
        // SAFETY: ptr is non-null with cap ≥ len (from with_capacity).
        // write_bytes sets raw memory — Zeroable guarantees all-zero bits is a valid T.
        // Element-by-element write for non-zero default: T: Copy prevents double-free,
        // and each slot receives a valid bitwise copy of default.
        unsafe {
            let ptr = vec.ptr.as_ptr();
            if default.is_zero() {
                std::ptr::write_bytes(ptr as *mut u8, 0u8, len * std::mem::size_of::<T>());
            } else {
                for i in 0..len {
                    ptr.add(i).write(default);
                }
            }
        }
        vec.len = len;
        Ok(vec)
    }

    /// Reserves memory space with the required alignment, but without filling the data yet.
    ///
    /// It's like reserving an exclusive parking lot: the space is there, guaranteed and
    /// organized in 64-byte blocks, but the "slots" are still empty.
    /// This special alignment allows the processor to read data at maximum
    /// speed, without needing to "adjust" the memory position.
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation fails.
    pub fn with_capacity(capacity: usize) -> Result<Self, NamErrorCode> {
        let (ptr, _layout) = Self::alloc_slot(capacity)?;
        Ok(Self {
            ptr: NonNull::new(ptr).unwrap_or(NonNull::dangling()),
            len: 0,
            cap: capacity,
        })
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns the allocated capacity (max number of elements without reallocation).
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Resizes the buffer to a new length, filling new elements with `default`.
    /// If `new_len <= self.len`, this is a no-op (never shrinks).
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation for the new capacity fails.
    pub fn resize(&mut self, new_len: usize, default: T) -> Result<(), NamErrorCode>
    where
        T: Zeroable,
    {
        if new_len <= self.len {
            return Ok(());
        }
        let mut new_vec = Self::with_capacity(new_len)?;
        // SAFETY: old_ptr is valid for self.len reads (self is initialized).
        // new_vec.ptr is non-null with cap ≥ new_len (freshly allocated). Source and
        // destination are separate allocations — no overlap possible.
        // copy_nonoverlapping moves self.len elements atomically.
        // write_bytes fills remaining slots with zero bytes; Zeroable guarantees
        // all-zero bits is a valid T.
        // Element-by-element write for non-zero default covers remaining slots.
        unsafe {
            let old_ptr = self.ptr.as_ptr();
            let new_ptr = new_vec.ptr.as_ptr();
            std::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);
            if default.is_zero() {
                let fill_start = new_ptr.add(self.len) as *mut u8;
                let fill_bytes = (new_len - self.len) * std::mem::size_of::<T>();
                std::ptr::write_bytes(fill_start, 0u8, fill_bytes);
            } else {
                for i in self.len..new_len {
                    new_ptr.add(i).write(default);
                }
            }
        }
        new_vec.len = new_len;
        let old = std::mem::replace(self, new_vec);
        drop(old);
        Ok(())
    }

    /// An optimized conversion from regular vector to aligned buffer.
    ///
    /// Uses a direct memory copy to ensure the transfer is as fast as possible,
    /// maintaining the 64-byte alignment.
    ///
    /// # Errors
    /// Returns `NamErrorCode::OutOfMemory` if allocation fails.
    pub fn from_vec(v: Vec<T>) -> Result<Self, NamErrorCode> {
        if v.is_empty() {
            return Self::with_capacity(0);
        }
        let mut aligned = Self::with_capacity(v.len())?;
        // SAFETY: v.as_ptr() is valid for v.len() reads (Vec invariant). aligned.ptr is non-null
        // with cap = v.len() (freshly allocated by with_capacity). T: Copy guarantees the source
        // and destination do not overlap (separate allocations), so copy_nonoverlapping is sound.
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), aligned.ptr.as_ptr(), v.len());
        }
        aligned.len = v.len();
        Ok(aligned)
    }
}

/// Allows accessing the buffer data as if it were a regular Rust slice.
///
/// This simplifies usage, as you can use functions that expect a normal slice
/// without needing complicated conversions.
impl<T: Copy> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: self.ptr is non-null from with_capacity (or dangling for zero-cap which is guarded
            // by the len==0 branch above). self.len ≤ self.cap and all elements [0..len) are initialized.
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

/// Allows accessing and modifying buffer data as a regular list.
///
/// Ensures freedom to read and write content in a simple and direct way.
impl<T: Copy> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: self.ptr is non-null with cap > 0. self.len ≤ self.cap, all elements in [0..len)
            // are initialized. &mut self guarantees exclusive access — no aliasing possible.
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

/// Ensures memory is returned to the system when the object is destroyed.
///
/// It's the "cleaner" that frees the reserved space as soon as you finish using
/// the buffer, preventing memory waste (so-called "memory leaks").
impl<T: Copy> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            // Layout construction cannot fail here: self.cap was validated during allocation
            // when alloc_slot produced a valid layout with the same size + alignment.
            if let Ok(layout) =
                Layout::from_size_align(self.cap * std::mem::size_of::<T>(), Self::ALIGN)
            {
                // SAFETY: self.ptr was allocated by alloc_slot using an identical layout
                // (size=cap*size_of::<T>(), align=64). The layout is recalculated identically
                // here. Drop runs exactly once (Rust ownership semantics). No per-element
                // destructors need to run: `T: Copy` (struct bound, F-10) guarantees the
                // elements carry no `Drop` glue, so deallocating the raw block is complete.
                unsafe {
                    dealloc(self.ptr.as_ptr() as *mut u8, layout);
                }
            }
        }
    }
}

/// Creates an identical copy of the buffer and its entire contents.
///
/// Reserves new memory space with the same alignment and performs a single
/// `copy_nonoverlapping` blit — no element-by-element iteration.
impl<T: Copy> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        // The layout (self.len * size_of::<T>(), 64) is identical to a layout that was
        // already validated during the original allocation. Therefore, alloc_slot cannot
        // return Err here — only a true system OOM could fail, which is unrecoverable.
        let (ptr, _layout) = Self::alloc_slot(self.len)
            .expect("clone: layout identical to validated original allocation");
        let mut new_vec = Self {
            ptr: NonNull::new(ptr).unwrap_or(NonNull::dangling()),
            len: 0,
            cap: self.len,
        };
        // SAFETY: self.ptr is valid for self.len reads. new_vec.ptr is non-null with
        // cap = self.len (freshly allocated). Source and destination are separate
        // allocations — no overlap possible. T: Copy guarantees that bitwise duplication
        // is semantically equivalent to Clone::clone.
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_vec.ptr.as_ptr(), self.len);
        }
        new_vec.len = self.len;
        new_vec
    }
}

/// Tells Rust that it is safe to send and share this buffer between different threads.
///
/// This is essential so that audio processing can be distributed across
/// multiple processor cores with full safety.
// SAFETY: AlignedVec owns its heap allocation exclusively. When T: Send, all elements can be
// safely transferred across thread boundaries. Destructors run on the receiving thread only.
unsafe impl<T: Copy + Send> Send for AlignedVec<T> {}
// SAFETY: AlignedVec's heap allocation is never mutated through shared references
// (Deref provides &[T]). When T: Sync, shared reads of elements across threads are safe.
unsafe impl<T: Copy + Sync> Sync for AlignedVec<T> {}

#[cfg(test)]
#[path = "aligned_test.rs"]
mod tests;
