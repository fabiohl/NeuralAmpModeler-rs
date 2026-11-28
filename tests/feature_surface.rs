// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Contract tests for the `testing` feature surface.
//!
//! Integration tests build the engine with `feature = "testing"` (see the
//! self dev-dependency). Enabling those test utilities must not silently
//! change production behaviour: the crash directory stays the real one and
//! the gate-disable/pipeline-diagnostic hooks stay inert until explicitly
//! armed by a caller.

use std::sync::atomic::Ordering;

#[cfg(not(windows))]
#[test]
#[serial_test::serial(feature_surface_env)]
fn testing_feature_keeps_the_production_crash_directory() {
    let fake_home = std::env::temp_dir().join(format!("nam-feature-home-{}", std::process::id()));
    let previous_crash_dir = std::env::var_os("NAM_CRASH_DIR");
    let previous_home = std::env::var_os("HOME");

    // SAFETY: serialized by `serial_test`; values are restored below.
    unsafe {
        std::env::remove_var("NAM_CRASH_DIR");
        std::env::set_var("HOME", &fake_home);
    }

    let resolved = neural_amp_modeler_rs::common::panic_hook::crash_directory()
        .expect("POSIX HOME must resolve a production crash directory");
    let expected = fake_home.join(".cache").join("neural-amp-modeler-rs");
    assert_eq!(
        resolved, expected,
        "feature `testing` must not redirect the production crash path"
    );

    // SAFETY: serialized by `serial_test`; restore the ambient environment.
    unsafe {
        match previous_crash_dir {
            Some(value) => std::env::set_var("NAM_CRASH_DIR", value),
            None => std::env::remove_var("NAM_CRASH_DIR"),
        }
        match previous_home {
            Some(value) => std::env::set_var("HOME", value),
            None => std::env::remove_var("HOME"),
        }
    }
}

#[test]
fn testing_feature_exposes_inert_hooks_until_armed() {
    assert!(
        !neural_amp_modeler_rs::dsp::pipeline::DISABLE_GATE.load(Ordering::Relaxed),
        "DISABLE_GATE must stay off unless a caller explicitly sets it"
    );
}
