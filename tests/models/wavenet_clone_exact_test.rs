// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Acceptance test for Task T3.1: Structural exact clone (`clone_exact` / `clone_wavenet_for_slimmable_storage`)
//! verifying zero panics / SIGABRT on models with heterogeneous channel geometries.

use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::NamModelData;
use neural_amp_modeler_rs::models::StaticModel;
use neural_amp_modeler_rs::models::slimmable::clone_wavenet_for_slimmable_storage;
use std::fs;
use std::path::{Path, PathBuf};

fn find_model_path(basename: &str) -> PathBuf {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let p1 = manifest_dir.join("tests/fixtures/models").join(basename);
    if p1.exists() {
        return p1;
    }
    let p2 = manifest_dir
        .parent()
        .unwrap_or(manifest_dir)
        .join("third-party/nam_t3k")
        .join(basename);
    if p2.exists() {
        return p2;
    }
    panic!("Model file not found: {basename}");
}

fn load_wavenet_dyn(
    basename: &str,
) -> Option<Box<neural_amp_modeler_rs::models::wavenet::WaveNetModelDyn>> {
    let path = find_model_path(basename);
    let content = fs::read_to_string(&path).expect("Failed to read model JSON");
    let data: NamModelData = serde_json::from_str(&content).expect("Failed to parse model JSON");
    let static_model = build_model(&data).expect("Failed to build model");

    match *static_model {
        StaticModel::WavenetDyn(w) => Some(w),
        _ => None,
    }
}

#[test]
fn test_clone_exact_wavenet_condition_dsp() {
    let model = load_wavenet_dyn("wavenet_condition_dsp.nam")
        .expect("wavenet_condition_dsp.nam should load as WavenetDyn");

    let exact_clone = model.clone_exact();
    assert_eq!(exact_clone.ch, model.ch);
    assert_eq!(exact_clone.arrays.len(), model.arrays.len());
    for (a_orig, a_clone) in model.arrays.iter().zip(exact_clone.arrays.iter()) {
        assert_eq!(a_orig.ch, a_clone.ch);
        assert_eq!(a_orig.in_ch, a_clone.in_ch);
    }

    let storage_clone = clone_wavenet_for_slimmable_storage(&model);
    assert!(
        storage_clone.is_ok(),
        "Storage clone must succeed without panic"
    );
}

#[test]
fn test_clone_exact_wavenet_dyn_free() {
    let model = load_wavenet_dyn("wavenet_dyn_free.nam")
        .expect("wavenet_dyn_free.nam should load as WavenetDyn");

    let exact_clone = model.clone_exact();
    assert_eq!(exact_clone.ch, model.ch);
    assert_eq!(exact_clone.arrays.len(), model.arrays.len());
    for (a_orig, a_clone) in model.arrays.iter().zip(exact_clone.arrays.iter()) {
        assert_eq!(a_orig.ch, a_clone.ch);
        assert_eq!(a_orig.in_ch, a_clone.in_ch);
    }

    let storage_clone = clone_wavenet_for_slimmable_storage(&model);
    assert!(
        storage_clone.is_ok(),
        "Storage clone must succeed without panic"
    );
}

#[test]
fn test_clone_exact_wavenet_official() {
    let model = load_wavenet_dyn("wavenet_official.nam")
        .expect("wavenet_official.nam should load as WavenetDyn");

    let exact_clone = model.clone_exact();
    assert_eq!(exact_clone.ch, model.ch);
    assert_eq!(exact_clone.arrays.len(), model.arrays.len());
    for (a_orig, a_clone) in model.arrays.iter().zip(exact_clone.arrays.iter()) {
        assert_eq!(a_orig.ch, a_clone.ch);
        assert_eq!(a_orig.in_ch, a_clone.in_ch);
    }

    let storage_clone = clone_wavenet_for_slimmable_storage(&model);
    assert!(
        storage_clone.is_ok(),
        "Storage clone must succeed without panic"
    );
}

#[test]
fn test_clone_exact_slammin_marshall() {
    let basename = "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam";
    let model = load_wavenet_dyn(basename).expect("SLAMMIN_MARSHALL should load as WavenetDyn");

    let exact_clone = model.clone_exact();
    assert_eq!(exact_clone.ch, model.ch);
    assert_eq!(exact_clone.arrays.len(), model.arrays.len());
    for (a_orig, a_clone) in model.arrays.iter().zip(exact_clone.arrays.iter()) {
        assert_eq!(a_orig.ch, a_clone.ch);
        assert_eq!(a_orig.in_ch, a_clone.in_ch);
    }

    let storage_clone = clone_wavenet_for_slimmable_storage(&model);
    assert!(
        storage_clone.is_ok(),
        "Storage clone must succeed without panic"
    );
}
