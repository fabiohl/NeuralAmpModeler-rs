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
