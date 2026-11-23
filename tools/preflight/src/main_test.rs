// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;

#[test]
fn x86_64_v3_evaluator_detects_missing_features() {
    let all_yes = [
        FeatureCheck {
            name: "avx",
            present: true,
        },
        FeatureCheck {
            name: "avx2",
            present: true,
        },
        FeatureCheck {
            name: "bmi1",
            present: true,
        },
        FeatureCheck {
            name: "bmi2",
            present: true,
        },
        FeatureCheck {
            name: "f16c",
            present: true,
        },
        FeatureCheck {
            name: "fma",
            present: true,
        },
        FeatureCheck {
            name: "lzcnt",
            present: true,
        },
        FeatureCheck {
            name: "movbe",
            present: true,
        },
    ];
    let ok = evaluate_x86_64_v3_features(&all_yes);
    assert!(ok.passed);
    assert!(ok.missing.is_empty());

    let mut missing_fma_movbe = all_yes;
    missing_fma_movbe[5].present = false;
    missing_fma_movbe[7].present = false;
    let gap = evaluate_x86_64_v3_features(&missing_fma_movbe);
    assert!(!gap.passed);
    assert_eq!(gap.missing, vec!["fma", "movbe"]);
}

#[test]
fn windows_support_classification_covers_client_server_and_eos() {
    assert_eq!(
        classify_windows_support(10, 26300, true).as_str(),
        "supported-windows-11-client"
    );
    assert_eq!(
        classify_windows_support(10, 26100, true),
        WindowsSupportStatus::SupportedWindows11Client
    );
    assert_eq!(
        classify_windows_support(10, 22631, true),
        WindowsSupportStatus::LegacyWindows11Client
    );
    assert_eq!(
        classify_windows_support(10, 26100, false),
        WindowsSupportStatus::WindowsServerCiOnly
    );
    assert_eq!(
        classify_windows_support(10, 19045, true),
        WindowsSupportStatus::OutOfScope
    );
}

#[test]
fn windows_product_name_normalizes_legacy_win10_registry_string_on_win11() {
    assert_eq!(
        normalize_windows_product_name("Windows 10 Pro", 10, 26300, true),
        "Windows 11 Pro"
    );
    assert_eq!(
        normalize_windows_product_name("Windows 10 Pro", 10, 19045, true),
        "Windows 10 Pro"
    );
    assert_eq!(
        normalize_windows_product_name("Windows Server 2025 Datacenter", 10, 26100, false),
        "Windows Server 2025 Datacenter"
    );
}

#[test]
fn arm64_evaluator_requires_neon_and_keeps_lse_informational() {
    let ok = evaluate_arm64_features(true, Some(true), false, Some(false));
    assert!(ok.neon_ok);
    assert!(!ok.lse_present);

    let missing = evaluate_arm64_features(true, Some(false), true, Some(true));
    assert!(!missing.neon_ok);
}
