// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::math::dsp::fft::FftPlanner;

const EPS_F32: f32 = 1e-5;
const EPS_F64: f64 = 1e-9;

fn assert_slice_approx_f32(a: &[f32], b: &[f32], eps: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "mismatch at idx {i}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

fn assert_slice_approx_f64(a: &[f64], b: &[f64], eps: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "mismatch at idx {i}: {x} vs {y} (diff {})",
            (x - y).abs()
        );
    }
}

fn make_random_input_f32(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    use std::num::Wrapping;
    let mut s = Wrapping(seed);
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for _ in 0..n {
        s = s * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
        let bits = s.0;
        let val: f32 = ((bits & 0xFFFFFF) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        re.push(val);
        s = s * Wrapping(6364136223846793005u64) + Wrapping(1442695040888963407u64);
        let bits2 = s.0;
        let val2: f32 = ((bits2 & 0xFFFFFF) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
        im.push(val2);
    }
    (re, im)
}

#[test]
#[should_panic(expected = "Radix-4 FFT requires N ≥ 4")]
fn rejects_n_less_than_4() {
    FftPlannerRadix4::<f32>::new(2);
}

// -----------------------------------------------------------------
// process panics on wrong buffer length
// -----------------------------------------------------------------

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_wrong_re_len() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re = vec![0.0f32; 15];
    let mut im = vec![0.0f32; 16];
    planner.process(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "im length mismatch")]
fn process_wrong_im_len() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re = vec![0.0f32; 16];
    let mut im = vec![0.0f32; 15];
    planner.process(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_both_short() {
    let planner = FftPlannerRadix4::<f32>::new(64);
    let mut re = vec![0.0f32; 16];
    let mut im = vec![0.0f32; 16];
    planner.process(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_empty_buffers() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re: Vec<f32> = Vec::new();
    let mut im: Vec<f32> = Vec::new();
    planner.process(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_wrong_len_f64() {
    let planner = FftPlannerRadix4::<f64>::new(64);
    let mut re = vec![0.0f64; 63];
    let mut im = vec![0.0f64; 64];
    planner.process(&mut re, &mut im);
}

// -----------------------------------------------------------------
// process_inverse panics on wrong buffer length
// -----------------------------------------------------------------

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_inverse_wrong_re_len() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re = vec![0.0f32; 15];
    let mut im = vec![0.0f32; 16];
    planner.process_inverse(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "im length mismatch")]
fn process_inverse_wrong_im_len() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re = vec![0.0f32; 16];
    let mut im = vec![0.0f32; 17];
    planner.process_inverse(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_inverse_both_short() {
    let planner = FftPlannerRadix4::<f32>::new(64);
    let mut re = vec![0.0f32; 32];
    let mut im = vec![0.0f32; 32];
    planner.process_inverse(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_inverse_empty_buffers() {
    let planner = FftPlannerRadix4::<f32>::new(16);
    let mut re: Vec<f32> = Vec::new();
    let mut im: Vec<f32> = Vec::new();
    planner.process_inverse(&mut re, &mut im);
}

#[test]
#[should_panic(expected = "re length mismatch")]
fn process_inverse_wrong_len_f64() {
    let planner = FftPlannerRadix4::<f64>::new(64);
    let mut re = vec![0.0f64; 63];
    let mut im = vec![0.0f64; 65];
    planner.process_inverse(&mut re, &mut im);
}

#[test]
fn roundtrip_n64_f32() {
    let n = 64;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_random_input_f32(n, 42);
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f32(&re, &re_orig, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im, &im_orig, EPS_F32 * 10.0);
}

#[test]
fn roundtrip_n4_f32() {
    let n = 4;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_random_input_f32(n, 42);
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f32(&re, &re_orig, EPS_F32);
    assert_slice_approx_f32(&im, &im_orig, EPS_F32);
}

#[test]
fn roundtrip_n16_f32() {
    let n = 16;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_random_input_f32(n, 42);
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f32(&re, &re_orig, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im, &im_orig, EPS_F32 * 10.0);
}

#[test]
fn parity_vs_radix2_n4_f32() {
    let n = 4;
    let r4 = FftPlannerRadix4::<f32>::new(n);
    let r2 = FftPlanner::<f32>::new(n);
    let (mut re4, mut im4) = make_random_input_f32(n, 99);
    let (mut re2, mut im2) = (re4.clone(), im4.clone());
    r4.process(&mut re4, &mut im4);
    r2.process(&mut re2, &mut im2);
    assert_slice_approx_f32(&re4, &re2, EPS_F32);
    assert_slice_approx_f32(&im4, &im2, EPS_F32);
}

#[test]
fn parity_vs_radix2_n16_f32() {
    let n = 16;
    let r4 = FftPlannerRadix4::<f32>::new(n);
    let r2 = FftPlanner::<f32>::new(n);
    let (mut re4, mut im4) = make_random_input_f32(n, 99);
    let (mut re2, mut im2) = (re4.clone(), im4.clone());
    r4.process(&mut re4, &mut im4);
    r2.process(&mut re2, &mut im2);
    assert_slice_approx_f32(&re4, &re2, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im4, &im2, EPS_F32 * 10.0);
}

#[test]
fn impulse_n16_f32() {
    let n = 16;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    re[0] = 1.0;
    planner.process(&mut re, &mut im);
    for i in 0..n {
        assert!((re[i] - 1.0).abs() < EPS_F32, "re[{i}]");
        assert!(im[i].abs() < EPS_F32, "im[{i}]");
    }
}

#[test]
fn roundtrip_n256_f32() {
    let n = 256;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_random_input_f32(n, 42);
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f32(&re, &re_orig, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im, &im_orig, EPS_F32 * 10.0);
}

#[test]
fn roundtrip_n1024_f32() {
    let n = 1024;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let (mut re, mut im) = make_random_input_f32(n, 42);
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f32(&re, &re_orig, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im, &im_orig, EPS_F32 * 10.0);
}

#[test]
fn parity_vs_radix2_n64_f32() {
    let n = 64;
    let r4 = FftPlannerRadix4::<f32>::new(n);
    let r2 = FftPlanner::<f32>::new(n);
    let (mut re4, mut im4) = make_random_input_f32(n, 99);
    let (mut re2, mut im2) = (re4.clone(), im4.clone());
    r4.process(&mut re4, &mut im4);
    r2.process(&mut re2, &mut im2);
    assert_slice_approx_f32(&re4, &re2, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im4, &im2, EPS_F32 * 10.0);
}

#[test]
fn parity_vs_radix2_n256_f32() {
    let n = 256;
    let r4 = FftPlannerRadix4::<f32>::new(n);
    let r2 = FftPlanner::<f32>::new(n);
    let (mut re4, mut im4) = make_random_input_f32(n, 99);
    let (mut re2, mut im2) = (re4.clone(), im4.clone());
    r4.process(&mut re4, &mut im4);
    r2.process(&mut re2, &mut im2);
    assert_slice_approx_f32(&re4, &re2, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im4, &im2, EPS_F32 * 10.0);
}

#[test]
fn parity_vs_radix2_n1024_f32() {
    let n = 1024;
    let r4 = FftPlannerRadix4::<f32>::new(n);
    let r2 = FftPlanner::<f32>::new(n);
    let (mut re4, mut im4) = make_random_input_f32(n, 99);
    let (mut re2, mut im2) = (re4.clone(), im4.clone());
    r4.process(&mut re4, &mut im4);
    r2.process(&mut re2, &mut im2);
    assert_slice_approx_f32(&re4, &re2, EPS_F32 * 10.0);
    assert_slice_approx_f32(&im4, &im2, EPS_F32 * 10.0);
}

#[test]
fn impulse_n64_f32() {
    let n = 64;
    let planner = FftPlannerRadix4::<f32>::new(n);
    let mut re = vec![0.0f32; n];
    let mut im = vec![0.0f32; n];
    re[0] = 1.0;
    planner.process(&mut re, &mut im);
    for i in 0..n {
        let diff = re[i] - 1.0f32;
        assert!(diff.abs() < EPS_F32, "re[{i}] = {} != 1.0", re[i]);
        assert!(im[i].abs() < EPS_F32, "im[{i}] = {} != 0.0", im[i]);
    }
}

#[test]
fn roundtrip_n64_f64() {
    let n = 64;
    let planner = FftPlannerRadix4::<f64>::new(n);
    let mut re: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17).sin()).collect();
    let mut im: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31).cos()).collect();
    let re_orig = re.clone();
    let im_orig = im.clone();
    planner.process(&mut re, &mut im);
    planner.process_inverse(&mut re, &mut im);
    assert_slice_approx_f64(&re, &re_orig, EPS_F64 * 100.0);
    assert_slice_approx_f64(&im, &im_orig, EPS_F64 * 100.0);
}
