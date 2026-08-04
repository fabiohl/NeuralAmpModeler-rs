// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::AlignedVec;
use crate::common::diagnostics::NamErrorCode;

#[test]
fn aligned_from_vec_empty_is_safe() {
    let v: Vec<f32> = Vec::new();
    let aligned = AlignedVec::from_vec(v).unwrap();
    assert!(aligned.is_empty());
    assert_eq!(aligned.len(), 0);
    assert_eq!(aligned.cap(), 0);
}

#[test]
fn aligned_from_vec_preserves_data() {
    let v = vec![1.0f32, 2.0, 3.0];
    let aligned = AlignedVec::from_vec(v).unwrap();
    assert_eq!(aligned.len(), 3);
    assert_eq!(aligned.cap(), 3);
    assert_eq!(aligned[0], 1.0);
    assert_eq!(aligned[1], 2.0);
    assert_eq!(aligned[2], 3.0);
}

#[test]
fn aligned_from_vec_empty_capacity_zero() {
    let v: Vec<f64> = Vec::new();
    let aligned = AlignedVec::from_vec(v).unwrap();
    assert!(aligned.is_empty());
    assert_eq!(aligned.len(), 0);
    assert_eq!(aligned.cap(), 0);
}

#[test]
fn with_capacity_sets_cap() {
    let v = AlignedVec::<f32>::with_capacity(100).unwrap();
    assert_eq!(v.len(), 0);
    assert_eq!(v.cap(), 100);
    assert!(v.is_empty());
}

#[test]
fn with_capacity_zero() {
    let v = AlignedVec::<f32>::with_capacity(0).unwrap();
    assert_eq!(v.len(), 0);
    assert_eq!(v.cap(), 0);
    assert!(v.is_empty());
}

#[test]
fn drop_excess_capacity_leak_free() {
    {
        let _v = AlignedVec::<f32>::with_capacity(1024).unwrap();
    }
}

#[test]
fn new_sets_len_eq_cap() {
    let v = AlignedVec::new(64, 0.0f32).unwrap();
    assert_eq!(v.len(), 64);
    assert_eq!(v.cap(), 64);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[63], 0.0);
}

#[test]
fn clone_preserves_len_cap() {
    let original = AlignedVec::new(42, 1.0f64).unwrap();
    let cloned = original.clone();
    assert_eq!(cloned.len(), 42);
    assert_eq!(cloned.cap(), 42);
    assert_eq!(cloned[0], 1.0);
    assert_eq!(cloned[41], 1.0);
}

#[test]
fn resize_grows_cap_and_len() {
    let mut v = AlignedVec::new(2, 0.0f32).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v.cap(), 2);
    v.resize(8, 1.0f32).unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(v.cap(), 8);
    assert_eq!(v[0], 0.0);
    assert_eq!(v[7], 1.0);
}

#[test]
fn resize_noop_preserves_dimensions() {
    let mut v = AlignedVec::new(8, 0.0f32).unwrap();
    v.resize(4, 1.0f32).unwrap();
    assert_eq!(v.len(), 8);
    assert_eq!(v.cap(), 8);
}

#[test]
fn alignment_is_64_bytes() {
    let v = AlignedVec::<f32>::new(1, 0.0f32).unwrap();
    assert_eq!(
        v.as_ptr() as usize % AlignedVec::<f32>::ALIGN,
        0,
        "pointer must be 64-byte aligned"
    );
}

#[test]
fn new_alloc_oom_on_layout_overflow() {
    let result = AlignedVec::<f32>::new(isize::MAX as usize, 0.0f32);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), NamErrorCode::OutOfMemory);
}

#[test]
fn with_capacity_oom_on_layout_overflow() {
    let result = AlignedVec::<f32>::with_capacity(isize::MAX as usize);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), NamErrorCode::OutOfMemory);
}
