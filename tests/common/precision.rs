// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use std::sync::{Mutex, MutexGuard};

use neural_amp_modeler_rs::math::activations::{
    ActivationPrecision, activation_precision, set_activation_tls,
};

static PRECISION_MUTEX: Mutex<()> = Mutex::new(());

/// RAII guard that sets thread-local activation precision on construction
/// and restores the original mode on drop.
///
/// Holds the global PRECISION_MUTEX for the guard's lifetime, serializing
/// concurrent tests that switch activation precision.
///
/// Uses thread-local storage instead of process-wide global atomic. Tests running in parallel threads each have independent TLS.
pub struct PrecisionGuard {
    original_mode: ActivationPrecision,
    _lock: MutexGuard<'static, ()>,
}

impl PrecisionGuard {
    /// Acquires the global precision mutex and sets `ActivationPrecision` to `mode`.
    ///
    /// Returns a guard that restores the previous precision mode on drop.
    #[must_use]
    pub fn new(mode: ActivationPrecision) -> Self {
        let _lock = PRECISION_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let original_mode = activation_precision();
        set_activation_tls(mode);
        Self {
            original_mode,
            _lock,
        }
    }
}

impl Drop for PrecisionGuard {
    fn drop(&mut self) {
        set_activation_tls(self.original_mode);
    }
}
