// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RT-safe `RtStatusFlags` injection through the `StaticModel` dispatcher.

use super::StaticModel;
use crate::common::spsc::RtStatusFlags;
use crate::models::a2::WaveNetA2;
use std::sync::Arc;

#[test]
fn inject_rt_status_rt_propagates_to_a2_lite() {
    let flags = Arc::new(RtStatusFlags::new());
    let strong_before = Arc::strong_count(&flags);
    let mut model = StaticModel::WavenetA2Lite(Box::new(WaveNetA2::<3>::new().unwrap()));

    model.inject_rt_status_rt(&flags);

    assert_eq!(Arc::strong_count(&flags), strong_before + 1);
    match &model {
        StaticModel::WavenetA2Lite(inner) => assert!(
            inner
                .rt_status
                .as_ref()
                .is_some_and(|stored| Arc::ptr_eq(stored, &flags)),
            "the A2 layer must hold the exact injected instance"
        ),
        _ => unreachable!("variant constructed above"),
    }

    // Repeated RT injection is a pointer-identity no-op.
    let after_adopt = Arc::strong_count(&flags);
    for _ in 0..64 {
        model.inject_rt_status_rt(&flags);
    }
    assert_eq!(Arc::strong_count(&flags), after_adopt);
}

#[test]
fn inject_rt_status_rt_propagates_to_a2_full() {
    let flags = Arc::new(RtStatusFlags::new());
    let mut model = StaticModel::WavenetA2Full(Box::new(WaveNetA2::<8>::new().unwrap()));

    model.inject_rt_status_rt(&flags);

    match &model {
        StaticModel::WavenetA2Full(inner) => assert!(
            inner
                .rt_status
                .as_ref()
                .is_some_and(|stored| Arc::ptr_eq(stored, &flags))
        ),
        _ => unreachable!("variant constructed above"),
    }
}

#[test]
fn inject_rt_status_rt_is_noop_for_non_a2_variants() {
    let flags = Arc::new(RtStatusFlags::new());
    let strong_before = Arc::strong_count(&flags);
    let mut model = StaticModel::Linear(Box::new(
        crate::models::linear::LinearModel::new(
            vec![1.0; 8],
            0.0,
            crate::loader::nam_json::LinearImplementation::Direct,
        )
        .unwrap(),
    ));

    model.inject_rt_status_rt(&flags);

    assert_eq!(Arc::strong_count(&flags), strong_before);
}
