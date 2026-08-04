// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use super::*;
use crate::models::StaticModel;

fn make_lstm() -> Box<StaticModel> {
    Box::new(StaticModel::Lstm1x8(Box::default()))
}

#[test]
fn test_valid_max_value_passes() {
    let submodels = vec![(0.5, make_lstm()), (1.0, make_lstm())];
    assert!(ContainerModel::new(submodels, 48000).is_ok());
}

#[test]
fn test_reject_max_value_nan() {
    let submodels = vec![(f32::NAN, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected NaN rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value=NaN"),
            "Expected NaN rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_inf() {
    let submodels = vec![(f32::INFINITY, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected Inf rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value"),
            "Expected Inf rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_neg_inf() {
    let submodels = vec![(f32::NEG_INFINITY, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected -Inf rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value"),
            "Expected -Inf rejection, got: {e}"
        ),
    }
}

#[test]
fn test_reject_max_value_negative() {
    let submodels = vec![(-0.5, make_lstm()), (1.0, make_lstm())];
    match ContainerModel::new(submodels, 48000) {
        Ok(_) => panic!("Expected negative rejection"),
        Err(e) => assert!(
            e.to_string().contains("invalid max_value=-0.5"),
            "Expected negative rejection, got: {e}"
        ),
    }
}

#[test]
fn test_slimmable_size_zero_selects_first_submodel() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.0, None);

    assert_eq!(container.pending_index(), Some(0));
}

#[test]
fn test_slimmable_size_one_selects_last_submodel() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_active_index(0);

    container.set_slimmable_size(1.0, None);

    assert_eq!(container.pending_index(), Some(2));
}

#[test]
fn test_slimmable_size_between_thresholds() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
}

#[test]
fn test_slimmable_size_same_value_noop() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.5, None);
    assert_eq!(container.pending_index(), Some(1));

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
}

#[test]
fn test_slimmable_size_same_active_noop() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_active_index(0);

    container.set_slimmable_size(0.2, None);

    assert!(container.pending_index().is_none());
}

#[test]
fn test_slimmable_size_change_during_crossfade() {
    let submodels = vec![(0.3, make_lstm()), (0.6, make_lstm()), (1.0, make_lstm())];
    let mut container = ContainerModel::new(submodels, 48000).unwrap();

    container.set_slimmable_size(0.2, None);
    assert_eq!(container.pending_index(), Some(0));

    container.set_slimmable_size(0.5, None);

    assert_eq!(container.pending_index(), Some(1));
    assert_eq!(container.active_index(), 0);
}
