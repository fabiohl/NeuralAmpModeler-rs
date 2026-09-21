// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn parse_linear_implementation_case_insensitive() {
    assert_eq!("auto".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("Auto".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("AUTO".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("direct".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("Direct".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("DIRECT".parse(), Ok(LinearImplementation::Direct));
    assert_eq!("fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("Fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("FFT".parse(), Ok(LinearImplementation::Fft));
}

#[test]
fn parse_linear_implementation_aliases_partitioned_fft() {
    // C++ NAMcore aliases for partitioned FFT convolution
    assert_eq!("partitioned_fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("PARTITIONED_FFT".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("Partitioned_Fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("partitioned-fft".parse(), Ok(LinearImplementation::Fft));
    assert_eq!("PARTITIONED-FFT".parse(), Ok(LinearImplementation::Fft));
}

#[test]
fn parse_linear_implementation_aliases_legacy() {
    // C++ NAMcore aliases for legacy/old → Auto
    assert_eq!("legacy".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("LeGaCy".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("LEGACY".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("old".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("OLD".parse(), Ok(LinearImplementation::Auto));
    assert_eq!("Old".parse(), Ok(LinearImplementation::Auto));
}

#[test]
fn parse_linear_implementation_invalid() {
    // legacy/old are now valid aliases — no longer Err
    assert_eq!("unknown".parse::<LinearImplementation>(), Err(()));
    assert_eq!("".parse::<LinearImplementation>(), Err(()));
}

#[test]
fn validate_head_none_when_absent() {
    let config = NamConfig {
        head: None,
        ..Default::default()
    };
    assert_eq!(config.validate_head(4).unwrap(), None);
}

#[test]
fn validate_head_valid_with_defaults() {
    let config = NamConfig {
        head: Some(serde_json::json!({})),
        ..Default::default()
    };
    let head = config.validate_head(8).unwrap().expect("valid head");
    assert_eq!(head.channels, None);
    assert_eq!(head.out_channels, None);
    assert_eq!(head.kernel_size, None);
}

#[test]
fn validate_head_rejects_exceeded_ceilings() {
    use crate::loader::nam_json::validation::{
        MAX_HEAD_CHANNELS, MAX_HEAD_KERNEL_SIZE, MAX_HEAD_OUT_CHANNELS,
    };

    let config_channels = NamConfig {
        head: Some(serde_json::json!({ "channels": MAX_HEAD_CHANNELS + 1 })),
        ..Default::default()
    };
    assert!(config_channels.validate_head(4).is_err());

    let config_out_channels = NamConfig {
        head: Some(serde_json::json!({ "out_channels": MAX_HEAD_OUT_CHANNELS + 1 })),
        ..Default::default()
    };
    assert!(config_out_channels.validate_head(4).is_err());

    let config_kernel_zero = NamConfig {
        head: Some(serde_json::json!({ "kernel_size": 0 })),
        ..Default::default()
    };
    assert!(config_kernel_zero.validate_head(4).is_err());

    let config_kernel_max = NamConfig {
        head: Some(serde_json::json!({ "kernel_size": MAX_HEAD_KERNEL_SIZE + 1 })),
        ..Default::default()
    };
    assert!(config_kernel_max.validate_head(4).is_err());
}
