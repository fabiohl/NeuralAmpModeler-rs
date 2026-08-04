// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use neural_amp_modeler_rs::dsp::mirror_buf::{MirroredBuffer, set_simulate_fail};

#[test]
#[cfg(target_os = "linux")]
fn test_mirror_buf_mmap_failure_injection() {
    // Activating failure simulation
    set_simulate_fail(true);

    let res = MirroredBuffer::<f32>::new(1024);
    assert!(
        res.is_err(),
        "Expected MirroredBuffer::new to fail when SIMULATE_FAIL is active"
    );

    // Deactivating failure simulation
    set_simulate_fail(false);

    let res = MirroredBuffer::<f32>::new(1024);
    assert!(
        res.is_ok(),
        "Expected MirroredBuffer::new to succeed when SIMULATE_FAIL is inactive"
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_mirror_buf_try_clone_under_fault() {
    let buf = MirroredBuffer::<f32>::new(1024).expect("initial allocation should succeed");

    set_simulate_fail(true);
    let result = buf.try_clone();
    assert!(
        result.is_err(),
        "try_clone should return Err under simulated mmap failure"
    );

    set_simulate_fail(false);
    let cloned = buf.try_clone();
    assert!(
        cloned.is_ok(),
        "try_clone should succeed without failure simulation"
    );
}
