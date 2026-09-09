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
        // Mesmo que a CPU física suporte AVX-512, o dispatch padrão do motor
        // é contratualmente restrito a AVX2 (x86-64-v3 baseline).
        let effective = effective_instruction_set();
        assert_eq!(
            effective,
            InstructionSet::Avx2,
            "Violação de contrato: build padrão não pode despachar acima de AVX2"
        );
    }

    #[cfg(feature = "avx512")]
    {
        // Se compilado explicitamente com --features avx512, seleciona AVX-512
        // apenas se o hardware físico tiver a matriz F+VL+BW+DQ completa.
        let effective = effective_instruction_set();
        if neural_amp_modeler_rs::math::common::has_full_avx512() {
            assert_eq!(effective, InstructionSet::Avx512);
        } else {
            assert_eq!(effective, InstructionSet::Avx2);
        }
    }
}
