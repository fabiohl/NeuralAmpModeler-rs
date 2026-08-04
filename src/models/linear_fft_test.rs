// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn new_zero_tail_partitions_when_p_equals_n() {
    let weights = vec![1.0f32; 256];
    let state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.p, 256);
    assert_eq!(state.n, 256);
    assert_eq!(state.num_partitions, 0);
    assert_eq!(state.num_bins, 257);
    assert_eq!(state.h_fdl_re.len(), 0);
    assert_eq!(state.fdl_re.len(), 0);
    assert_eq!(state.tail_output_buf.len(), 256);
}

#[test]
fn new_one_tail_partition_when_n_equals_2p() {
    let weights = vec![1.0f32; 512];
    let state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.p, 256);
    assert_eq!(state.n, 512);
    assert_eq!(state.num_partitions, 1);
    assert_eq!(state.num_bins, 257);
    // Flat length = K × num_bins = 1 × 257
    assert_eq!(state.h_fdl_re.len(), 257);
    assert_eq!(state.fdl_re.len(), 257);
}

#[test]
fn new_partitions_for_ir_8192_p_512() {
    let weights = vec![0.0f32; 8192];
    let state = LinearFftState::new(512, &weights)
        .expect("construction should succeed for test-sized buffers");
    // ceil((8192-512)/512) = ceil(7680/512) = 15
    assert_eq!(state.num_partitions, 15);
    assert_eq!(state.num_bins, 513);
    // Flat length = 15 × 513 = 7695
    assert_eq!(state.h_fdl_re.len(), 7695);
    assert_eq!(state.fdl_re.len(), 7695);
    assert_eq!(state.tail_output_buf.len(), 512);
    assert_eq!(state.input_buf.len(), 1024);
}

#[test]
fn new_partitions_uneven_last_block() {
    // N=1000, P=256: ceil((1000-256)/256) = ceil(744/256) = 3
    let weights = vec![0.0f32; 1000];
    let state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 3);
    // Last partition covers IR[768..1000] = 232 samples, zero-padded to 512
}

#[test]
#[should_panic(expected = "power of two")]
fn new_panics_on_non_power_of_two_p() {
    let weights = vec![0.0f32; 1024];
    LinearFftState::new(300, &weights).expect("construction should succeed for test-sized buffers");
}

#[test]
#[should_panic(expected = "P (1024) must be ≤ N (512)")]
fn new_panics_when_p_greater_than_n() {
    let weights = vec![0.0f32; 512];
    LinearFftState::new(1024, &weights)
        .expect("construction should succeed for test-sized buffers");
}

#[test]
fn reset_zeros_runtime_buffers() {
    let mut weights = vec![0.0f32; 256];
    weights[0] = 1.0;
    let mut state = LinearFftState::new(128, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 1);

    // Dirty the runtime buffers
    state.tail_output_buf[0] = 99.0;
    state.fdl_re[0] = 99.0;
    state.fdl_im[0] = 99.0;
    state.sample_counter = 42;
    state.fdl_write_idx = 0;

    state.reset();

    assert_eq!(state.sample_counter, 0);
    assert_eq!(state.fdl_write_idx, state.num_partitions.saturating_sub(1));
    for v in state.tail_output_buf.iter() {
        assert!((*v).abs() < f32::EPSILON);
    }
    // FDL must be zeroed
    for v in state.fdl_re.iter() {
        assert!((*v).abs() < f32::EPSILON);
    }
    // h_fdl should NOT be zeroed (pre-computed IR spectra)
    assert!(!state.h_fdl_re.is_empty());
}

#[test]
fn debug_format_does_not_leak_internal_buffers() {
    let weights = vec![1.0f32; 512];
    let state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    let dbg = format!("{state:?}");
    assert!(dbg.contains("LinearFftState"));
    assert!(dbg.contains("p"));
    assert!(dbg.contains("n"));
    assert!(dbg.contains("..")); // finish_non_exhaustive marker
}

#[test]
fn h_spectra_nonzero_for_nonzero_ir_tail() {
    let mut weights = vec![0.0f32; 512];
    // Put an impulse in the tail at index P (256)
    weights[256] = 1.0;
    let state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 1);
    // The spectrum of an impulse delayed by 0 within its block is all ones (real)
    // But it's zero-padded to 512 then FFT'd of 512 real → 257 complex bins
    // DC should be 1.0 (sum of all samples = 1.0)
    // Partition 0, bin 0 = state.h_fdl_re[0]
    assert!((state.h_fdl_re[0] - 1.0).abs() < 1e-5);
}

#[test]
fn weights_head_portion_not_used_for_spectra() {
    // IR: head [1,2,3,4], tail [5,6,7,8] with P=4
    let weights: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let state = LinearFftState::new(4, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 1);
    // The head portion [1,2,3,4] should NOT affect h_fdl.
    // Only tail [5,6,7,8] is transformed (zero-padded to 8).
    // DC of [5,6,7,8,0,0,0,0] = 5+6+7+8 = 26
    // Partition 0, bin 0 = state.h_fdl_re[0]
    assert!((state.h_fdl_re[0] - 26.0).abs() < 1e-4);
}

#[test]
fn process_tail_block_noop_when_zero_partitions() {
    let weights = vec![1.0f32; 256];
    let mut state = LinearFftState::new(256, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 0);
    // Should not panic when called with no tail partitions
    state.process_tail_block(&[0.0f32; 512]);
    // tail_output_buf should remain all zeros
    for &v in state.tail_output_buf.iter() {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn process_tail_block_first_call_uses_current_spectrum() {
    // P=2, N=4, IR=[1.0, 2.0, 3.0, 4.0], tail=[3.0, 4.0]
    let weights = vec![1.0, 2.0, 3.0, 4.0];
    let mut state = LinearFftState::new(2, &weights)
        .expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 1);
    // Fix: first call uses current input spectrum.
    // conv([3,4], [1,2,3,4]) at positions [2,3] = [3*3+4*2=17, 3*4+4*3=24]
    state.process_tail_block(&[1.0, 2.0, 3.0, 4.0]);
    assert!(
        (state.tail_output_buf[0] - 17.0).abs() < 1e-4,
        "expected 17, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 24.0).abs() < 1e-4,
        "expected 24, got {}",
        state.tail_output_buf[1]
    );
}

#[test]
fn process_tail_block_numerical_correctness() {
    // P=2, N=4, IR = [0.5, 1.5, 3.0, 4.0]
    // head = [0.5, 1.5], tail = [3.0, 4.0]
    let ir = vec![0.5, 1.5, 3.0, 4.0];
    let p = 2;
    let mut state =
        LinearFftState::new(p, &ir).expect("construction should succeed for test-sized buffers");

    // Block 1: feed [1, 2, 3, 4] — uses current input spectrum.
    // conv([3,4], [1,2,3,4]) valid region [2,3] = [17, 24]
    let block1 = [1.0, 2.0, 3.0, 4.0];
    state.process_tail_block(&block1);
    assert!(
        (state.tail_output_buf[0] - 17.0).abs() < 1e-4,
        "block 1[0]: expected 17, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 24.0).abs() < 1e-4,
        "block 1[1]: expected 24, got {}",
        state.tail_output_buf[1]
    );

    // Block 2: feed [5, 6, 7, 8] — uses current input spectrum.
    // conv([3,4], [5,6,7,8]) valid region [2,3] = [45, 52]
    let block2 = [5.0, 6.0, 7.0, 8.0];
    state.process_tail_block(&block2);
    assert!(
        (state.tail_output_buf[0] - 45.0).abs() < 1e-4,
        "block 2[0]: expected 45, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 52.0).abs() < 1e-4,
        "block 2[1]: expected 52, got {}",
        state.tail_output_buf[1]
    );
}

#[test]
fn process_tail_block_multiple_partitions() {
    // P=2, N=8, K=ceil((8-2)/2)=3
    // IR = [h0, h1, t0, t1, t2, t3, t4, t5]
    let ir: Vec<f32> = vec![
        0.0, 0.0, // head (unused by FFT tail)
        1.0, 0.0, // tail partition 0 (delays 2,3)
        0.0, 1.0, // tail partition 1 (delays 4,5)
        0.5, 0.5, // tail partition 2 (delays 6,7)
    ];
    let p = 2;
    let mut state =
        LinearFftState::new(p, &ir).expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 3);

    // Block 1: feed [0, 0, 1, 0]
    // k=0: conv([1,0], [0,0,1,0]) at [2,3] = [1, 0]
    // k=1: FDL zeros → [0, 0]
    // k=2: FDL zeros → [0, 0]
    state.process_tail_block(&[0.0, 0.0, 1.0, 0.0]);
    assert!(
        (state.tail_output_buf[0] - 1.0).abs() < 1e-4,
        "block 1[0]: expected 1, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 0.0).abs() < 1e-4,
        "block 1[1]: expected 0, got {}",
        state.tail_output_buf[1]
    );

    // Block 2: feed [0, 1, 0, 0]
    // k=0: conv([1,0], [0,1,0,0]) at [2,3] = [0, 0]
    // k=1: FDL[2]=block 1's [0,0,1,0]. conv([0,1], [0,0,1,0]) at [2,3] = [0, 1]
    // k=2: FDL zeros → [0, 0]
    state.process_tail_block(&[0.0, 1.0, 0.0, 0.0]);
    assert!(
        (state.tail_output_buf[0] - 0.0).abs() < 1e-4,
        "block 2[0]: expected 0, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 1.0).abs() < 1e-4,
        "block 2[1]: expected 1, got {}",
        state.tail_output_buf[1]
    );

    // Block 3: feed [1, 0, 0, 0]
    // k=0: conv([1,0], [1,0,0,0]) at [2,3] = [0, 0]
    // k=1: FDL[0]=block 2's [0,1,0,0]. conv([0,1], [0,1,0,0]) at [2,3] = [1, 0]
    // k=2: FDL[2]=block 1's [0,0,1,0]. conv([0.5,0.5], [0,0,1,0]) at [2,3] = [0.5, 0.5]
    // Total: [1.5, 0.5]
    state.process_tail_block(&[1.0, 0.0, 0.0, 0.0]);
    assert!(
        (state.tail_output_buf[0] - 1.5).abs() < 1e-4,
        "block 3[0]: expected 1.5, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 0.5).abs() < 1e-4,
        "block 3[1]: expected 0.5, got {}",
        state.tail_output_buf[1]
    );
}

#[test]
fn process_tail_block_fdl_circular_advance() {
    // P=2, N=8, K=3
    let ir = vec![0.0f32; 8];
    let p = 2;
    let mut state =
        LinearFftState::new(p, &ir).expect("construction should succeed for test-sized buffers");
    assert_eq!(state.num_partitions, 3);
    assert_eq!(state.fdl_write_idx, 2); // K-1

    // Call 1: write at 2, advance to 0
    state.process_tail_block(&[1.0; 4]);
    assert_eq!(state.fdl_write_idx, 0);

    // Call 2: write at 0, advance to 1
    state.process_tail_block(&[2.0; 4]);
    assert_eq!(state.fdl_write_idx, 1);

    // Call 3: write at 1, advance to 2
    state.process_tail_block(&[3.0; 4]);
    assert_eq!(state.fdl_write_idx, 2);

    // Call 4: wraps back to 0
    state.process_tail_block(&[4.0; 4]);
    assert_eq!(state.fdl_write_idx, 0);
}

#[test]
fn process_tail_block_reset_then_process() {
    let ir = vec![1.0, 2.0, 3.0, 4.0];
    let mut state =
        LinearFftState::new(2, &ir).expect("construction should succeed for test-sized buffers");

    // Prime FDL with one block
    state.process_tail_block(&[1.0; 4]);
    assert_eq!(state.fdl_write_idx, 0);

    // Reset should clear FDL and counters
    state.reset();
    assert_eq!(state.fdl_write_idx, 0); // K-1 for K=1
    assert_eq!(state.sample_counter, 0);

    // After reset, first call uses current spectrum.
    // conv([3,4], [5,6,7,8]) at [2,3] = [3*7+4*6=45, 3*8+4*7=52]
    state.process_tail_block(&[5.0, 6.0, 7.0, 8.0]);
    assert!(
        (state.tail_output_buf[0] - 45.0).abs() < 1e-4,
        "expected 45, got {}",
        state.tail_output_buf[0]
    );
    assert!(
        (state.tail_output_buf[1] - 52.0).abs() < 1e-4,
        "expected 52, got {}",
        state.tail_output_buf[1]
    );
}
