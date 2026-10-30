// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! RAII Guards for forcing specific instruction set dispatch in testing & benchmarking.

use crate::math::common::dispatch::detect::TEST_ISA_OVERRIDE;
#[cfg(nam_avx512)]
use crate::math::common::encode_isa_override;
use crate::math::common::{InstructionSet, IsaOverrideError, set_test_isa_override};
use std::sync::atomic::Ordering;

/// Off-RT RAII guard that forces SIMD instruction dispatch to operate
/// exclusively on [`InstructionSet::Avx2`] for the lifetime of the guard.
///
/// On [`Drop::drop`], the previous instruction set override setting is restored.
///
/// AVX2 is the mandatory `x86-64-v3` baseline, so this guard is unconditionally
/// safe to install on every supported host.
#[derive(Debug)]
pub struct ForceAvx2Guard {
    prev_override: u8,
}

impl ForceAvx2Guard {
    /// Creates a new guard, forcing [`InstructionSet::Avx2`] SIMD dispatch.
    pub fn new() -> Self {
        let prev = set_test_isa_override(InstructionSet::Avx2)
            .expect("AVX2 override must always be installable on the x86-64-v3 baseline");
        Self {
            prev_override: prev,
        }
    }
}

impl Default for ForceAvx2Guard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ForceAvx2Guard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

/// Off-RT RAII guard that forces SIMD instruction dispatch to operate
/// exclusively on [`InstructionSet::Avx512`] for the lifetime of the guard.
///
/// On [`Drop::drop`], the previous instruction set override setting is restored.
#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
#[derive(Debug)]
pub struct ForceAvx512Guard {
    prev_override: u8,
}

#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
impl ForceAvx512Guard {
    /// Fallibly creates a new guard, forcing [`InstructionSet::Avx512`] SIMD
    /// dispatch — but ONLY after validating that the host CPU supports the
    /// complete AVX-512 capability matrix (`F+VL+BW+DQ`) required by the
    /// Reachable kernels.
    ///
    /// On a host with a partial AVX-512 subset (e.g. F+VL without BW/DQ —
    /// common under VMs/hypervisors) the override is NOT installed and a
    /// structured [`IsaOverrideError::UnsupportedIsa`] is returned. This is
    /// the fail-closed path: safe Rust can never force dispatch towards
    /// instructions that would raise `SIGILL`.
    ///
    /// Callers in tests/harnesses should map the error to a typed skip
    /// (e.g. `SKIP_CAPABILITY`) instead of panicking.
    pub fn try_new() -> Result<Self, IsaOverrideError> {
        set_test_isa_override(InstructionSet::Avx512).map(|prev| Self {
            prev_override: prev,
        })
    }

    /// Unchecked constructor for emulation and controlled fault-injection
    /// harnesses (e.g. `remote-simd-gate.sh` under Intel SDE, or tests that
    /// exercise the override machinery without executing kernels).
    ///
    /// # Safety
    /// The caller must guarantee that dispatching [`InstructionSet::Avx512`]
    /// kernels is safe for the lifetime of the guard: either the host CPU
    /// supports the full `F+VL+BW+DQ` matrix, the code runs under an
    /// instruction emulator (SDE), or no SIMD kernel is actually dispatched
    /// while the override is active. Violating this precondition causes
    /// `SIGILL` (illegal instruction).
    pub unsafe fn new_unchecked() -> Self {
        // SAFETY: guaranteed by the caller per the # Safety contract above.
        let prev = TEST_ISA_OVERRIDE.swap(
            encode_isa_override(InstructionSet::Avx512),
            Ordering::SeqCst,
        );
        Self {
            prev_override: prev,
        }
    }
}

#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
impl Default for ForceAvx512Guard {
    fn default() -> Self {
        Self::try_new()
            .expect("AVX-512 dispatch override requires the full host capability set (F+VL+BW+DQ)")
    }
}

#[cfg(nam_avx512)]
#[cfg_attr(docsrs, doc(cfg(feature = "avx512")))]
impl Drop for ForceAvx512Guard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

/// Generic RAII guard that forces SIMD instruction dispatch to operate
/// on the specified [`InstructionSet`] for the lifetime of the guard.
#[derive(Debug)]
pub struct IsaGuard {
    prev_override: u8,
}

impl IsaGuard {
    /// Fallibly creates a new guard, forcing the specified ISA for SIMD
    /// Dispatch after validating host CPU capability.
    ///
    /// Returns [`IsaOverrideError`] when the requested ISA cannot be executed
    /// on the current host (e.g. AVX-512 on a CPU with a partial subset) so
    /// safe Rust never installs an override that would `SIGILL`.
    pub fn try_set(isa: InstructionSet) -> Result<Self, IsaOverrideError> {
        set_test_isa_override(isa).map(|prev| Self {
            prev_override: prev,
        })
    }
}

impl Drop for IsaGuard {
    fn drop(&mut self) {
        TEST_ISA_OVERRIDE.store(self.prev_override, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[path = "isa_guard_test.rs"]
mod tests;
