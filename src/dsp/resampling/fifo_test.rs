// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_push_pop_basic() {
    let mut fifo = SampleFifo::new(8).unwrap();
    assert_eq!(fifo.capacity(), 8);
    assert!(fifo.is_empty());

    assert_eq!(fifo.push(&[1.0, 2.0, 3.0]), 3);
    assert_eq!(fifo.len(), 3);
    assert_eq!(fifo.free(), 5);

    let mut out = [0.0f32; 2];
    assert_eq!(fifo.pop_into(&mut out), 2);
    assert_eq!(out, [1.0, 2.0]);
    assert_eq!(fifo.len(), 1);
}

#[test]
fn test_push_pop_wraparound() {
    let mut fifo = SampleFifo::new(4).unwrap();
    assert_eq!(fifo.push(&[1.0, 2.0, 3.0]), 3);
    let mut out = [0.0f32; 2];
    assert_eq!(fifo.pop_into(&mut out), 2);
    assert_eq!(out, [1.0, 2.0]);

    // Push more than the free head space to force a wrap.
    assert_eq!(fifo.push(&[4.0, 5.0, 6.0]), 3);
    let mut out2 = [0.0f32; 4];
    assert_eq!(fifo.pop_into(&mut out2), 4);
    assert_eq!(out2, [3.0, 4.0, 5.0, 6.0]);
    assert!(fifo.is_empty());
}

#[test]
fn test_bounded_overflow_returns_partial() {
    let mut fifo = SampleFifo::new(4).unwrap();
    assert_eq!(fifo.push(&[1.0, 2.0, 3.0, 4.0]), 4);
    assert_eq!(fifo.push(&[5.0, 6.0]), 0, "full FIFO accepts nothing");
    assert_eq!(fifo.len(), 4);

    let mut out = [0.0f32; 2];
    assert_eq!(fifo.pop_into(&mut out), 2);
    assert_eq!(fifo.push(&[7.0, 8.0]), 2, "partial acceptance");
    assert_eq!(fifo.len(), 4);
}

#[test]
fn test_unpop_restores_popped() {
    let mut fifo = SampleFifo::new(8).unwrap();
    fifo.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut out = [0.0f32; 3];
    assert_eq!(fifo.pop_into(&mut out), 3);
    assert_eq!(out, [1.0, 2.0, 3.0]);
    assert_eq!(fifo.len(), 2);

    fifo.unpop(3);
    assert_eq!(fifo.len(), 5);
    let mut out2 = [0.0f32; 5];
    assert_eq!(fifo.pop_into(&mut out2), 5);
    assert_eq!(out2, [1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_unpop_after_wraparound() {
    let mut fifo = SampleFifo::new(4).unwrap();
    fifo.push(&[1.0, 2.0, 3.0]);
    let mut out = [0.0f32; 2];
    fifo.pop_into(&mut out);
    fifo.push(&[4.0, 5.0]); // forces wrap: storage now [5.0, 2.0, 3.0, 4.0]? no: [4,5] wrap
    // read index after first pop = 2; write = (2 + 1) % 4 = 3 → buf[3]=4, wrap buf[0]=5
    // storage: [5.0, X, 3.0, 4.0]
    let mut out2 = [0.0f32; 2];
    assert_eq!(fifo.pop_into(&mut out2), 2);
    assert_eq!(out2, [3.0, 4.0]);

    // Restore the two popped (physical slots hold 3.0, 4.0 at indices 2, 3).
    fifo.unpop(2);
    assert_eq!(fifo.len(), 3);
    let mut out3 = [0.0f32; 3];
    assert_eq!(fifo.pop_into(&mut out3), 3);
    assert_eq!(out3, [3.0, 4.0, 5.0]);
}

#[test]
fn test_clear() {
    let mut fifo = SampleFifo::new(4).unwrap();
    fifo.push(&[1.0, 2.0, 3.0]);
    fifo.clear();
    assert!(fifo.is_empty());
    assert_eq!(fifo.push(&[9.0]), 1);
    let mut out = [0.0f32; 1];
    assert_eq!(fifo.pop_into(&mut out), 1);
    assert_eq!(out, [9.0]);
}

#[test]
fn test_zero_capacity_clamped() {
    let fifo = SampleFifo::new(0).unwrap();
    assert_eq!(fifo.capacity(), 1);
}
