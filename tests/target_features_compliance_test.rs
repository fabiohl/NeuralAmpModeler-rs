// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Off-RT compliance meta-test for x86-64-v3 target features.
//!
//! Validates at compile-time and run-time that the active crate build enforces the full
//! psABI x86-64-v3 instruction set baseline: `avx`, `avx2`, `bmi1`, `bmi2`,
//! `f16c`, `fma`, `lzcnt`, `movbe`.

const _: () = {
    assert!(
        cfg!(target_feature = "avx"),
        "Target feature 'avx' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "avx2"),
        "Target feature 'avx2' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "bmi1"),
        "Target feature 'bmi1' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "bmi2"),
        "Target feature 'bmi2' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "f16c"),
        "Target feature 'f16c' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "fma"),
        "Target feature 'fma' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "lzcnt"),
        "Target feature 'lzcnt' must be enabled (psABI x86-64-v3 required)"
    );
    assert!(
        cfg!(target_feature = "movbe"),
        "Target feature 'movbe' must be enabled (psABI x86-64-v3 required)"
    );
};

#[test]
fn test_x86_64_v3_target_features_compliance() {
    let features = [
        ("avx", cfg!(target_feature = "avx")),
        ("avx2", cfg!(target_feature = "avx2")),
        ("bmi1", cfg!(target_feature = "bmi1")),
        ("bmi2", cfg!(target_feature = "bmi2")),
        ("f16c", cfg!(target_feature = "f16c")),
        ("fma", cfg!(target_feature = "fma")),
        ("lzcnt", cfg!(target_feature = "lzcnt")),
        ("movbe", cfg!(target_feature = "movbe")),
    ];

    for (name, enabled) in features {
        assert!(
            enabled,
            "Target feature '{name}' must be enabled for psABI x86-64-v3 compliance"
        );
    }
}

#[test]
fn test_layer_kernels_feature_requirements_satisfied() {
    // Layer kernels (e.g. in src/models/lstm/layer_kernels.rs) use #[target_feature(enable = "avx2,fma,f16c")].
    // Verify statically that all three required features are satisfied by the baseline compile flags.
    let avx2_enabled = cfg!(target_feature = "avx2");
    let fma_enabled = cfg!(target_feature = "fma");
    let f16c_enabled = cfg!(target_feature = "f16c");

    assert!(
        avx2_enabled && fma_enabled && f16c_enabled,
        "LSTM layer kernels require target_feature avx2, fma, f16c."
    );
}
