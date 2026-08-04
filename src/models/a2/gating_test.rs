// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn test_gating_mode_default() {
    assert_eq!(GatingMode::default(), GatingMode::None);
}

#[test]
fn test_gating_apply_tanh_sigmoid() {
    let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
    let mut buf = vec![0.5f32, -0.3, 0.8, -0.6];
    let mut golden = buf.clone();

    // Golden: apply activations to each half independently, then multiply.
    ActivationType::Tanh.apply(&mut golden[..2]);
    ActivationType::Sigmoid.apply(&mut golden[2..]);
    golden[0] *= golden[2];
    golden[1] *= golden[3];

    config.apply_gating(&mut buf);

    // Compare only the first half (output channels).
    assert!((buf[0] - golden[0]).abs() < 1e-7);
    assert!((buf[1] - golden[1]).abs() < 1e-7);
}

#[test]
fn test_blending_apply_tanh_sigmoid() {
    let mut config =
        BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2).unwrap();
    let mut buf = [0.5f32, -0.3f32, 0.8f32, -0.6f32];
    let orig = buf;
    let mut golden = [0.0f32; 2];

    // Golden: apply activations, then blend.
    let mut activated = [orig[0], orig[1]];
    ActivationType::Tanh.apply(&mut activated);
    let mut alpha = [orig[2], orig[3]];
    ActivationType::Sigmoid.apply(&mut alpha);
    for i in 0..2 {
        golden[i] = activated[i].mul_add(alpha[i], orig[i] * (1.0 - alpha[i]));
    }

    config.apply_blending(&mut buf);

    // Compare only the first half (output channels).
    assert!((buf[0] - golden[0]).abs() < 1e-7);
    assert!((buf[1] - golden[1]).abs() < 1e-7);
}

#[test]
fn test_gating_channel_pairs_zero_input() {
    let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
    let mut buf = vec![0.0f32; 8]; // ch=4
    config.apply_gating(&mut buf);
    // tanh(0)=0, sigmoid(0)=0.5, product=0
    for v in buf.iter().take(4) {
        assert!((*v).abs() < 1e-7);
    }
}

#[test]
fn test_blending_channel_pairs_zero_input() {
    let mut config =
        BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 4).unwrap();
    let mut buf = vec![0.0f32; 8]; // ch=4
    config.apply_blending(&mut buf);
    // tanh(0)=0, sigmoid(0)=0.5, blended = 0 * 0.5 + 0 * 0.5 = 0
    for v in buf.iter().take(4) {
        assert!((*v).abs() < 1e-7);
    }
}

#[test]
fn test_gating_unity_gate() {
    let config = GatingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid);
    let mut buf = vec![0.5f32, -0.3f32, 10.0f32, 10.0f32]; // ch=2
    let mut expected = [0.5f32, -0.3f32];
    ActivationType::Tanh.apply(&mut expected);
    // sigmoid(10) ≈ 1, so gate multiplier ≈ 1
    config.apply_gating(&mut buf);
    assert!((buf[0] - expected[0]).abs() < 1e-4);
    assert!((buf[1] - expected[1]).abs() < 1e-4);
}

#[test]
fn test_blending_alpha_one() {
    let mut config =
        BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2).unwrap();
    let mut buf = vec![0.5f32, -0.3f32, 10.0f32, 10.0f32]; // ch=2
    let mut expected = [0.5f32, -0.3f32];
    ActivationType::Tanh.apply(&mut expected);
    // sigmoid(10) ≈ 1 → blended ≈ activated_input
    config.apply_blending(&mut buf);
    assert!((buf[0] - expected[0]).abs() < 1e-4);
    assert!((buf[1] - expected[1]).abs() < 1e-4);
}

#[test]
fn test_blending_alpha_zero() {
    let mut config =
        BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2).unwrap();
    let orig = [0.5f32, -0.3f32];
    let mut buf = [orig[0], orig[1], -10.0f32, -10.0f32]; // ch=2, alpha≈0
    let mut golden = orig;

    // Golden with actual sigmoid: alpha ≈ 0 → blended ≈ original.
    // For x ≤ -8, sigmoid(x) = 0 in our approximation.
    let mut alpha = [-10.0f32, -10.0f32];
    ActivationType::Sigmoid.apply(&mut alpha);
    assert!(alpha[0].abs() < 1e-4);

    let mut activated = orig;
    ActivationType::Tanh.apply(&mut activated);
    for i in 0..2 {
        golden[i] = activated[i].mul_add(alpha[i], orig[i] * (1.0 - alpha[i]));
    }

    config.apply_blending(&mut buf);
    assert!((buf[0] - golden[0]).abs() < 1e-4);
    assert!((buf[1] - golden[1]).abs() < 1e-4);
}

#[test]
fn test_blending_scratch_preserved() {
    let mut config =
        BlendingActivationConfig::new(ActivationType::Tanh, ActivationType::Sigmoid, 2).unwrap();
    assert_eq!(config.channels(), 2);

    // Apply twice with same config — scratch must not leak state.
    let mut buf1 = [0.5f32, 0.3f32, 0.8f32, 0.6f32];
    let mut buf2 = [0.5f32, 0.3f32, 0.8f32, 0.6f32];
    config.apply_blending(&mut buf1);
    config.apply_blending(&mut buf2);
    assert_eq!(&buf1[..2], &buf2[..2]);
}
