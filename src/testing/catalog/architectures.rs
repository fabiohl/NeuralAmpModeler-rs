// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Canonical reference architecture manifesto for profiling and benchmarking.

use serde::{Deserialize, Serialize};

// =============================================================================
// Canonical Reference Architecture Manifesto
// =============================================================================
//
// Single source of truth for representative model architecture fixtures across
// all supported inference topology families. Used for profiling matrices (PGO/BOLT),
// benchmarking, and architectural coverage validation.

/// Classification of neural network and acoustic DSP model architecture families.
///
/// Represents the distinct inference topology families supported by the engine.
/// Used for canonical profiling matrices, benchmarking suites, and architecture
/// coverage verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArchitectureFamily {
    /// WaveNet A1 architecture (Standard, Lite, Feather, Nano, dynamic fallback).
    #[serde(rename = "wavenet_a1")]
    WaveNetA1,
    /// WaveNet A2 architecture (FiLM, gating, head1x1, bottleneck, cascade).
    #[serde(rename = "wavenet_a2")]
    WaveNetA2,
    /// LSTM recurrent architecture (single/multi-layer, SIMD-accelerated gates).
    #[serde(rename = "lstm")]
    Lstm,
    /// ConvNet feed-forward dilated convolution architecture.
    #[serde(rename = "convnet")]
    ConvNet,
    /// Linear FIR / Linear-FFT partitioned convolution architecture.
    #[serde(rename = "linear")]
    Linear,
    /// Sequential chain architecture: serial composition of independent
    /// validated stages (freshly-registered model family).
    #[serde(rename = "sequential")]
    Sequential,
}

impl ArchitectureFamily {
    /// Array of all six architecture families supported by the engine.
    pub const ALL: &'static [ArchitectureFamily] = &[
        ArchitectureFamily::WaveNetA1,
        ArchitectureFamily::WaveNetA2,
        ArchitectureFamily::Lstm,
        ArchitectureFamily::ConvNet,
        ArchitectureFamily::Linear,
        ArchitectureFamily::Sequential,
    ];

    /// Returns a static slice of all architecture families.
    pub fn all() -> &'static [ArchitectureFamily] {
        Self::ALL
    }

    /// Machine-readable identifier string (`snake_case`, matching profiling and telemetry conventions).
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::WaveNetA1 => "wavenet_a1",
            Self::WaveNetA2 => "wavenet_a2",
            Self::Lstm => "lstm",
            Self::ConvNet => "convnet",
            Self::Linear => "linear",
            Self::Sequential => "sequential",
        }
    }

    /// Human-readable display name for this architectural family.
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::WaveNetA1 => "WaveNet A1",
            Self::WaveNetA2 => "WaveNet A2",
            Self::Lstm => "LSTM",
            Self::ConvNet => "ConvNet",
            Self::Linear => "Linear",
            Self::Sequential => "Sequential",
        }
    }
}

impl std::fmt::Display for ArchitectureFamily {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.display_name())
    }
}

/// Error returned when parsing an unrecognized architecture family name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseArchitectureFamilyError(pub String);

impl std::fmt::Display for ParseArchitectureFamilyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown architecture family: {:?}", self.0)
    }
}

impl std::error::Error for ParseArchitectureFamilyError {}

impl std::str::FromStr for ArchitectureFamily {
    type Err = ParseArchitectureFamilyError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "wavenet_a1" | "waveneta1" | "wavenet" => Ok(Self::WaveNetA1),
            "wavenet_a2" | "waveneta2" | "a2" => Ok(Self::WaveNetA2),
            "lstm" => Ok(Self::Lstm),
            "convnet" => Ok(Self::ConvNet),
            "linear" | "linear_fft" => Ok(Self::Linear),
            "sequential" => Ok(Self::Sequential),
            _ => Err(ParseArchitectureFamilyError(s.to_string())),
        }
    }
}

/// Canonical metadata specification for a reference model architecture fixture.
///
/// Designed as a host-agnostic, consumer-neutral descriptor for profiling
/// workloads, benchmarking suites, and architecture drift validation.
///
/// Contains purely descriptive metadata and resolution helpers without
/// coupling to any specific host audio runtime or framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchitectureFixtureSpec {
    /// Architectural family represented by this fixture.
    pub family: ArchitectureFamily,
    /// Basename of the model file within the fixture directory (e.g. `"convnet_test.nam"`).
    pub nam_file: &'static str,
    /// Canonical workspace-relative path to the fixture file.
    pub canonical_path: &'static str,
    /// Suggested audio sample rate in Hz for profiling and benchmarking (e.g. 48000).
    pub suggested_sample_rate: u32,
    /// Suggested audio quantum / block size in samples for real-time profiling (e.g. 64).
    pub suggested_quantum: usize,
    /// Human-readable description of this architecture fixture.
    pub description: &'static str,
}

impl ArchitectureFixtureSpec {
    /// Resolves the filesystem path to this fixture file using the engine's fixture resolver.
    ///
    /// Resolves through [`crate::testing::fixtures::model_path`].
    pub fn resolve_path(&self) -> std::path::PathBuf {
        crate::testing::fixtures::model_path(self.nam_file)
    }

    /// Checks whether this fixture file exists on disk in the current environment.
    pub fn exists(&self) -> bool {
        self.resolve_path().exists()
    }
}

/// Canonical static array of reference architecture fixture specifications.
///
/// Covers all six supported model architecture families with at least one
/// committed, verified fixture file:
/// - [`ArchitectureFamily::WaveNetA1`]: WaveNet A1 Standard (`wavenet_a1_standard.nam`)
/// - [`ArchitectureFamily::WaveNetA2`]: WaveNet A2 Full (`wavenet_a2_full.nam`)
/// - [`ArchitectureFamily::Lstm`]: LSTM 1×16 (`BossLSTM-1x16.nam`)
/// - [`ArchitectureFamily::ConvNet`]: ConvNet 6-block (`convnet_test.nam`)
/// - [`ArchitectureFamily::Linear`]: Linear direct FIR (`linear_test.nam`) and Linear FFT (`linear_fft_rf320.nam`)
/// - [`ArchitectureFamily::Sequential`]: Sequential two-stage chain (`sequential_linear_chain.nam`)
pub static REFERENCE_ARCHITECTURES: &[ArchitectureFixtureSpec] = &[
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::WaveNetA1,
        nam_file: "wavenet_a1_standard.nam",
        canonical_path: "tests/fixtures/models/wavenet_a1_standard.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "WaveNet A1 Standard reference model (16 channels, dilation depth 10)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::WaveNetA2,
        nam_file: "wavenet_a2_full.nam",
        canonical_path: "tests/fixtures/models/wavenet_a2_full.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "WaveNet A2 Full reference model (8 channels, gated activation)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::Lstm,
        nam_file: "BossLSTM-1x16.nam",
        canonical_path: "tests/fixtures/models/BossLSTM-1x16.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "LSTM 1×16 reference model (1 layer, 16 hidden units)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::ConvNet,
        nam_file: "convnet_test.nam",
        canonical_path: "tests/fixtures/models/convnet_test.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "ConvNet reference model (8 channels, 6 blocks)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::Linear,
        nam_file: "linear_test.nam",
        canonical_path: "tests/fixtures/models/linear_test.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "Linear FIR direct convolution reference model (RF=4)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::Linear,
        nam_file: "linear_fft_rf320.nam",
        canonical_path: "tests/fixtures/models/linear_fft_rf320.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "Linear FFT partitioned frequency-domain convolution reference model (RF=320)",
    },
    ArchitectureFixtureSpec {
        family: ArchitectureFamily::Sequential,
        nam_file: "sequential_linear_chain.nam",
        canonical_path: "tests/fixtures/models/sequential_linear_chain.nam",
        suggested_sample_rate: 48000,
        suggested_quantum: 64,
        description: "Sequential two-stage chain reference model (validated serial composition)",
    },
];

/// Returns the canonical slice of reference architecture fixture specifications.
///
/// Guaranteed to contain at least one representative model fixture for each of the
/// six supported neural network architecture families ([`ArchitectureFamily`]):
/// WaveNet A1, WaveNet A2, LSTM, ConvNet, Linear, and Sequential.
///
/// # Examples
///
/// ```
/// use neural_amp_modeler_rs::testing::catalog::{reference_architectures, ArchitectureFamily};
///
/// let specs = reference_architectures();
/// assert!(!specs.is_empty());
///
/// for family in ArchitectureFamily::ALL {
///     assert!(
///         specs.iter().any(|spec| spec.family == *family),
///         "missing reference architecture for family {family:?}"
///     );
/// }
/// ```
pub fn reference_architectures() -> &'static [ArchitectureFixtureSpec] {
    REFERENCE_ARCHITECTURES
}

/// Finds the first reference architecture fixture specification for a given family.
///
/// Returns `None` if no fixture is registered for the specified family.
///
/// # Examples
///
/// ```
/// use neural_amp_modeler_rs::testing::catalog::{reference_architecture_for, ArchitectureFamily};
///
/// let convnet_spec = reference_architecture_for(ArchitectureFamily::ConvNet)
///     .expect("ConvNet reference architecture must exist");
/// assert_eq!(convnet_spec.family, ArchitectureFamily::ConvNet);
/// assert_eq!(convnet_spec.nam_file, "convnet_test.nam");
/// ```
pub fn reference_architecture_for(
    family: ArchitectureFamily,
) -> Option<&'static ArchitectureFixtureSpec> {
    REFERENCE_ARCHITECTURES
        .iter()
        .find(|spec| spec.family == family)
}
