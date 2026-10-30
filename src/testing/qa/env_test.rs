// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Unit tests for host environment probing and hardware classification (R-12, R-23, S6-T03).

use super::*;

#[test]
fn test_hardware_class_classification_ryzen_baseline() {
    let hw = classify_hardware_class(
        "AMD Ryzen 7 5700U with Radeon Graphics",
        ISA_X86_64_V3,
        "x86_64-unknown-linux-gnu",
    );
    assert_eq!(hw, HardwareClass::X86Zen3Laptop);
    assert_eq!(hw.as_str(), "x86_zen3_laptop");
}

#[test]
fn test_hardware_class_classification_amd_zen() {
    let hw = classify_hardware_class(
        "AMD EPYC 7763 64-Core Processor",
        ISA_X86_64_V3,
        "x86_64-unknown-linux-gnu",
    );
    assert_eq!(hw, HardwareClass::X86AmdZen);
    assert_eq!(hw.as_str(), "x86_amd_zen");
}

#[test]
fn test_hardware_class_classification_intel_client() {
    let hw = classify_hardware_class(
        "12th Gen Intel(R) Core(TM) i7-12700H",
        ISA_X86_64_V3,
        "x86_64-unknown-linux-gnu",
    );
    assert_eq!(hw, HardwareClass::X86IntelClient);
    assert_eq!(hw.as_str(), "x86_intel_client");
}

#[test]
fn test_hardware_class_classification_avx512_server() {
    let hw = classify_hardware_class(
        "Intel(R) Xeon(R) Platinum 8488C",
        ISA_AVX512,
        "x86_64-unknown-linux-gnu",
    );
    assert_eq!(hw, HardwareClass::X86Avx512Server);
    assert_eq!(hw.as_str(), "x86_avx512_server");
}

#[test]
fn test_hardware_class_classification_apple_silicon() {
    let hw = classify_hardware_class(
        "Apple M2 Pro",
        ISA_AARCH64_APPLE_SILICON,
        "aarch64-apple-darwin",
    );
    assert_eq!(hw, HardwareClass::AArch64AppleSilicon);
    assert_eq!(hw.as_str(), "aarch64_apple_silicon");
}

#[test]
fn test_hardware_class_classification_aarch64_linux_server() {
    let hw = classify_hardware_class("Neoverse-N1", ISA_AARCH64_NEON, "aarch64-unknown-linux-gnu");
    assert_eq!(hw, HardwareClass::AArch64LinuxServer);
    assert_eq!(hw.as_str(), "aarch64_linux_server");
}

#[test]
fn test_hardware_class_roundtrip_names() {
    let classes = [
        HardwareClass::X86Zen3Laptop,
        HardwareClass::X86AmdZen,
        HardwareClass::X86IntelClient,
        HardwareClass::X86Avx512Server,
        HardwareClass::X86GenericV3,
        HardwareClass::AArch64AppleSilicon,
        HardwareClass::AArch64LinuxServer,
        HardwareClass::AArch64Generic,
        HardwareClass::Unknown,
    ];
    for &c in &classes {
        let name = c.as_str();
        assert_eq!(HardwareClass::from_str_name(name), c);
    }
}
