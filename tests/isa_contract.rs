// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Contractual test: a standard build (without `--features avx512`) never
//! dispatches AVX-512 code, even on hardware that supports the full AVX-512
//! capability set. AVX-512 is only selected when the crate is explicitly
//! compiled with the opt-in `avx512` feature AND the host CPU provides the
//! complete F+VL+BW+DQ matrix.

use neural_amp_modeler_rs::math::common::{InstructionSet, effective_instruction_set};

#[test]
fn contract_default_build_never_dispatches_avx512() {
    #[cfg(not(feature = "avx512"))]
    {
        // Even if the physical CPU supports AVX-512, the default engine dispatch
        // is contractually restricted to AVX2 (x86-64-v3 baseline).
        let effective = effective_instruction_set();
        assert_eq!(
            effective,
            InstructionSet::Avx2,
            "Contract violation: default build cannot dispatch above AVX2"
        );
    }

    #[cfg(feature = "avx512")]
    {
        // When explicitly compiled with --features avx512, select AVX-512
        // only if physical hardware provides the complete F+VL+BW+DQ matrix.
        let effective = effective_instruction_set();
        if neural_amp_modeler_rs::math::common::has_full_avx512() {
            assert_eq!(effective, InstructionSet::Avx512);
        } else {
            assert_eq!(effective, InstructionSet::Avx2);
        }
    }
}

#[test]
fn contract_simd_math_initialized_off_rt() {
    #[cfg(feature = "avx512")]
    {
        // Under opt-in avx512, SIMD_MATH LazyLock is accessible off-RT (R-03).
        let config = *neural_amp_modeler_rs::math::common::SIMD_MATH;
        if neural_amp_modeler_rs::math::common::has_full_avx512() {
            assert_eq!(config.instruction_set, InstructionSet::Avx512);
            assert!(config.is_avx512);
        } else {
            assert_eq!(config.instruction_set, InstructionSet::Avx2);
            assert!(!config.is_avx512);
        }
    }
}
