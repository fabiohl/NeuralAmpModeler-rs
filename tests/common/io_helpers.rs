// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

#![allow(dead_code)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use neural_amp_modeler_rs::models::NamModel;

/// Classification of a test fixture's origin, determining its distribution
/// policy and whether its absence is recoverable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureOrigin {
    /// Distributed with the repository — always required.
    DistributedCore,
    /// Local non-distributable models (community models, proprietary IRs).
    /// Required for full-environment testing, may be absent in CI.
    LocalNonDistributable,
    /// Third-party vendor files under `third-party/` (read-only).
    /// Required only when the vendor toolchain is available.
    ThirdPartyVendor,
}

/// Execution profile for a fixture — drives preflight behavior.
///
/// - `RequiredLocal`: fixture is committed or expected in the local checkout.
///   Absence is a hard preflight failure, aborting before tests run.
/// - `OptionalExternal`: fixture is non-distributable or vendor-controlled.
///   Absence is a graceful typed skip (`MissingOptional`), never a panic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProfile {
    /// Must exist in the local environment — hard preflight failure if absent.
    RequiredLocal,
    /// May be absent — skip gracefully with `MissingOptional` status.
    OptionalExternal,
}

/// Applicable oracles encoded as bitflags for a single fixture entry.
///
/// Each fixture may be validated against one or both reference oracles.
pub struct ApplicableOracle;

impl ApplicableOracle {
    pub const NAMCORE_F32: u8 = 1 << 0;
    pub const F64: u8 = 1 << 1;
    pub const BOTH: u8 = Self::NAMCORE_F32 | Self::F64;
}

/// Typed outcome of a fixture availability check, replacing ad-hoc
/// `.expect()` panics on missing external files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureStatus {
    /// Fixture is present and usable.
    Available,
    /// Fixture is optional and not found — test can skip with a
    /// typed status, never panic via `.expect()`.
    MissingOptional,
    /// Fixture is required and not found — test must fail.
    MissingRequired,
}

impl FixtureStatus {
    pub fn is_available(self) -> bool {
        matches!(self, FixtureStatus::Available)
    }
}

/// Single entry in the unified fixture catalog.
///
/// Each entry declares its filename, origin, execution profile,
/// applicable oracles, and a human-readable description.
///
/// The catalog replaces ad-hoc `.expect()` calls and hardcoded path
/// assumptions with a single declarative source of truth consulted
/// before any fixture I/O operation.
#[derive(Debug, Clone)]
pub struct FixtureEntry {
    pub name: &'static str,
    pub origin: FixtureOrigin,
    pub execution_profile: ExecutionProfile,
    pub applicable_oracles: u8,
    pub description: &'static str,
}

/// Unified catalog of all external test fixtures.
///
/// Centralizes the policy for fixture availability — replacing the three
/// implicit policies (green-on-skip, assert Completed, panic expect) with
/// a single declarative source.
///
/// # Usage
/// ```ignore
/// let status = FIXTURE_CATALOG.check("BossWN-standard.nam");
/// match status {
///     FixtureStatus::Available => { /* proceed */ }
///     FixtureStatus::MissingOptional => { /* skip with typed reason */ }
///     FixtureStatus::MissingRequired => { panic!("required fixture absent"); }
/// }
/// ```
pub struct FixtureCatalog {
    entries: &'static [FixtureEntry],
}

/// Declares the contracted sample-rate scope for a model in live v2 multi-SR
/// cross-validation (`cpp_parity::run_v2_multi_sr`).
///
/// Mirrors the `v2_scope` / `skip_srs` fields in
/// `tests/fixtures/golden_gen_build.sh` catalog, applied to live C++↔Rust
/// parity tests.
///
/// The default (`AllRates`) means the model is validated at 44.1k, 48k,
/// 88.2k, 96k, and 192k. Restricted scopes must be declared here with a
/// documented reason — silent partial completion is prohibited by Gate
/// Calibration Policy Rule 7.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V2MultiSRScope {
    AllRates,
    Exclude192k,
    SR48kOnly,
}

/// Returns the set of sample rates that `run_v2_multi_sr` must validate for
/// a given model, as contracted in the golden-build catalog.
///
/// Models not listed here default to `AllRates` (all five
/// `SUPPORTED_SAMPLE_RATES`).
pub fn v2_multi_sr_expected_rates(model_filename: &str) -> Vec<u32> {
    let scope = match model_filename {
        // ── LSTM: Exclude 192k ──────────────────────────────────────────
        // C++ render at 192 kHz produces NaN from recurrent-state overflow
        // in the Eigen-based WaveNet path (third-party upstream, read-only).
        "BossLSTM-1x16.nam" | "lstm_1x16.nam"
        | "BossLSTM-2x8.nam" | "lstm_2x8.nam"
        // Uncatalogued synthetic LSTM model — same 192k crash in C++ render
        | "lstm_1x10.nam" | "lstm_2x24.nam" | "lstm_3x8.nam" => {
            V2MultiSRScope::Exclude192k
        }

        // ── 48k-only (catalog contract) ─────────────────────────────────
        | "BossWN-standard.nam" | "wavenet_standard.nam"
        | "wavenet_a2_full.nam" | "wavenet_a2_lite.nam"
        | "wavenet_official.nam" | "lstm.nam" | "lstm_official.nam"
        | "wavenet_condition_dsp.nam" | "wavenet_condition_lstm.nam"
        | "convnet_test.nam" | "convnet_nobn.nam" | "convnet_relu.nam" | "convnet_silu.nam"
        | "wavenet_dyn_free.nam" | "lstm_dyn_test.nam"
        | "a2_dynamic_gated_ch8.nam" | "a2_dynamic_blended_ch3.nam"
        | "a2_example.nam" | "wavenet_a2_max.nam"
        | "EVH-5150-Lite.nam" | "APP-EVH-Stealth100-Dialled-xSTD.nam"
        | "Boss BD-2 H2O Mod T-12_00 G-12_00.nam"
        | "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam"
        | "linear_nobias.nam" | "linear_test.nam" => {
            V2MultiSRScope::SR48kOnly
        }

        _ => V2MultiSRScope::AllRates,
    };

    match scope {
        V2MultiSRScope::AllRates => vec![44100, 48000, 88200, 96000, 192000],
        V2MultiSRScope::Exclude192k => vec![44100, 48000, 88200, 96000],
        V2MultiSRScope::SR48kOnly => vec![48000],
    }
}

impl FixtureCatalog {
    /// Check whether a fixture file (relative to `tests/fixtures/models/`) is available.
    ///
    /// Returns `FixtureStatus` based on the fixture's execution profile and disk presence:
    /// - `RequiredLocal` → `MissingRequired` if absent
    /// - `OptionalExternal` → `MissingOptional` if absent
    pub fn check(&self, filename: &str) -> FixtureStatus {
        let model_path = model_path(filename);
        if model_path.exists() {
            return FixtureStatus::Available;
        }
        for entry in self.entries {
            if entry.name == filename {
                return match entry.execution_profile {
                    ExecutionProfile::RequiredLocal => FixtureStatus::MissingRequired,
                    ExecutionProfile::OptionalExternal => FixtureStatus::MissingOptional,
                };
            }
        }
        FixtureStatus::MissingRequired
    }

    /// Check whether a golden binary file (relative to `tests/fixtures/`) is available.
    pub fn check_golden(&self, golden_filename: &str) -> FixtureStatus {
        let golden_path = fixtures_dir().join(golden_filename);
        if golden_path.exists() {
            return FixtureStatus::Available;
        }
        for entry in self.entries {
            if entry.name == golden_filename {
                return match entry.execution_profile {
                    ExecutionProfile::RequiredLocal => FixtureStatus::MissingRequired,
                    ExecutionProfile::OptionalExternal => FixtureStatus::MissingOptional,
                };
            }
        }
        FixtureStatus::MissingRequired
    }

    /// Check if a model filename corresponds to an optional fixture.
    /// When true, runners may skip gracefully instead of panicking.
    pub fn is_optional(&self, filename: &str) -> bool {
        self.entries.iter().any(|e| {
            e.name == filename && e.execution_profile == ExecutionProfile::OptionalExternal
        })
    }

    /// Return an iterator over all catalog entries.
    pub fn entries(&self) -> std::slice::Iter<'_, FixtureEntry> {
        self.entries.iter()
    }

    /// Generate a capability receipt — lists every catalog entry with its
    /// resolved path and current status.  Designed for preflight emission
    /// before running the long test suite.
    pub fn capability_receipt(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Fixture Catalog Capability Receipt ===".to_string());
        lines.push(format!(
            "{:<60} {:<22} {:<20} {:?}",
            "NAME", "STATUS", "PROFILE", "ORACLES"
        ));
        lines.push("-".repeat(130));

        for entry in self.entries {
            let (status, _resolved) = if entry.name.ends_with(".bin") {
                let path = fixtures_dir().join(entry.name);
                let s = if path.exists() {
                    FixtureStatus::Available
                } else {
                    self.check_golden(entry.name)
                };
                (s, path)
            } else {
                let path = model_path(entry.name);
                let s = self.check(entry.name);
                (s, path)
            };

            let oracle_str = if entry.applicable_oracles & ApplicableOracle::NAMCORE_F32 != 0
                && entry.applicable_oracles & ApplicableOracle::F64 != 0
            {
                "NAMCore+f64"
            } else if entry.applicable_oracles & ApplicableOracle::NAMCORE_F32 != 0 {
                "NAMCore"
            } else if entry.applicable_oracles & ApplicableOracle::F64 != 0 {
                "f64"
            } else {
                "none"
            };

            lines.push(format!(
                "{:<60} {:<22} {:<20} {:?}",
                entry.name,
                format!("{:?}", status),
                format!("{:?}", entry.execution_profile),
                oracle_str,
            ));
        }
        lines.push("-".repeat(130));
        lines.join("\n")
    }
}

/// Static catalog of all external test fixtures.
///
/// Every `.nam` model and `.bin` golden referenced by tests should have an
/// entry here. The catalog replaces ad-hoc `.expect()` calls with typed
/// `FixtureStatus` checks.
#[rustfmt::skip]
pub static FIXTURE_CATALOG: FixtureCatalog = FixtureCatalog {
    entries: &[
        // Distributed Core models — required-local, NAMCore oracle
        FixtureEntry { name: "BossWN-standard.nam",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet Standard CH=16" },
        FixtureEntry { name: "BossWN-feather.nam",         origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet Feather CH=8" },
        FixtureEntry { name: "BossWN-nano.nam",            origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet Nano CH=4" },
        FixtureEntry { name: "BossLSTM-1x16.nam",          origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM 1×16" },
        FixtureEntry { name: "BossLSTM-2x8.nam",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM 2×8" },
        FixtureEntry { name: "lstm.nam",                   origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM Official (H=3)" },
        FixtureEntry { name: "wavenet_a1_standard.nam",    origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet A1 Standard Official" },
        FixtureEntry { name: "wavenet_a2_full.nam",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet A2 Full CH=8" },
        FixtureEntry { name: "wavenet_a2_lite.nam",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet A2 Lite CH=3" },
        FixtureEntry { name: "wavenet_a2_container.nam",   origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "SlimmableContainer A2" },
        FixtureEntry { name: "a2_example.nam",             origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "SlimmableContainer A2 Example" },
        FixtureEntry { name: "wavenet_condition_dsp.nam",  origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNet Condition DSP" },
        FixtureEntry { name: "wavenet_condition_lstm.nam", origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Condition LSTM (fail-closed)" },
        FixtureEntry { name: "wavenet_official.nam",       origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Official free-geom CH=3" },
        FixtureEntry { name: "wavenet_dyn_free.nam",       origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "WaveNetDyn Free-Shape" },
        FixtureEntry { name: "lstm_dyn_test.nam",          origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM-Dyn 1×7" },
        FixtureEntry { name: "linear_test.nam",            origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "Linear RF=4" },
        FixtureEntry { name: "convnet_test.nam",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "ConvNet Test" },
        FixtureEntry { name: "convnet_nobn.nam",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "ConvNet No BatchNorm" },
        FixtureEntry { name: "convnet_relu.nam",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "ConvNet ReLU" },
        FixtureEntry { name: "convnet_silu.nam",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "ConvNet SiLU" },
        FixtureEntry { name: "linear_nobias.nam",          origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "Linear No Bias" },
        FixtureEntry { name: "lstm_1x10.nam",              origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM 1×10 (uncatalogued synthetic topology)" },
        FixtureEntry { name: "lstm_2x24.nam",              origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM 2×24" },
        FixtureEntry { name: "lstm_3x8.nam",               origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "LSTM 3×8" },
        FixtureEntry { name: "slimmable_container.nam",    origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "Slimmable Container 3 submodels" },
        FixtureEntry { name: "slimmable_wavenet.nam",      origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "Slimmable single-net WaveNet (fail-closed)" },
        FixtureEntry { name: "wavenet_a2_max.nam",         origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet A2 Max (fail-closed §7.1)" },
        FixtureEntry { name: "a2_dynamic_gated_ch8.nam",   origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2 Dynamic Gated CH=8" },
        FixtureEntry { name: "a2_dynamic_blended_ch3.nam", origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2 Dynamic Blended CH=3" },
        FixtureEntry { name: "wavenet_a2_film_lite.nam",   origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2-FiLM-Lite CH=3" },
        FixtureEntry { name: "wavenet_a2_film_full.nam",   origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2-FiLM-Full CH=8" },
        FixtureEntry { name: "wavenet_a2_film_input_mixin_pre.nam", origin: FixtureOrigin::DistributedCore, execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2-FiLM-InputMixinPre CH=3" },
        FixtureEntry { name: "wavenet_a2_film_chaos_stress.nam", origin: FixtureOrigin::DistributedCore, execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::BOTH, description: "A2-FiLM Chaos Stress CH=3" },

        // Local Non-Distributable models — optional-external, NAMCore oracle
        FixtureEntry { name: "EVH-5150-Lite.nam",              origin: FixtureOrigin::LocalNonDistributable, execution_profile: ExecutionProfile::OptionalExternal, applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "EVH 5150 Lite (community, CH=12)" },
        FixtureEntry { name: "APP-EVH-Stealth100-Dialled-xSTD.nam", origin: FixtureOrigin::LocalNonDistributable, execution_profile: ExecutionProfile::OptionalExternal, applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "APP EVH Stealth 100" },
        FixtureEntry { name: "Boss BD-2 H2O Mod T-12_00 G-12_00.nam", origin: FixtureOrigin::LocalNonDistributable, execution_profile: ExecutionProfile::OptionalExternal, applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "Boss BD-2 H2O Mod" },
        FixtureEntry { name: "SLAMMIN_MARSHALL_J45_VN9_TREBLEBOOSTER_P4_C.nam", origin: FixtureOrigin::LocalNonDistributable, execution_profile: ExecutionProfile::OptionalExternal, applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "SLAMMIN MARSHALL JTM 45" },

        // Golden vector files — distributed core, NAMCore oracle
        FixtureEntry { name: "golden_wavenet_standard.bin",       origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Standard v1 golden" },
        FixtureEntry { name: "golden_wavenet_feather.bin",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Feather v1 golden" },
        FixtureEntry { name: "golden_wavenet_nano.bin",           origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Nano v1 golden" },
        FixtureEntry { name: "golden_wavenet_lite.bin",           origin: FixtureOrigin::LocalNonDistributable, execution_profile: ExecutionProfile::OptionalExternal, applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Lite v1 golden" },
        FixtureEntry { name: "golden_lstm_1x16.bin",              origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "LSTM 1×16 v1 golden" },
        FixtureEntry { name: "golden_lstm_2x8.bin",               origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "LSTM 2×8 v1 golden" },
        FixtureEntry { name: "golden_lstm_official.bin",          origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "LSTM Official v1 golden" },
        FixtureEntry { name: "golden_wavenet_a1_standard.bin",    origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet A1 Standard v1 golden" },
        FixtureEntry { name: "golden_wavenet_a2_full.bin",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet A2 Full v1 golden" },
        FixtureEntry { name: "golden_wavenet_a2_lite.bin",        origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet A2 Lite v1 golden" },
        FixtureEntry { name: "golden_wavenet_condition_dsp.bin",  origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Condition DSP v1 golden" },
        FixtureEntry { name: "golden_wavenet_official.bin",       origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "WaveNet Official v1 golden" },
        FixtureEntry { name: "golden_a2_example.bin",             origin: FixtureOrigin::DistributedCore,       execution_profile: ExecutionProfile::RequiredLocal,  applicable_oracles: ApplicableOracle::NAMCORE_F32, description: "SlimmableContainer A2 Example v1 golden" },
    ],
};
///
/// Returns `Some((input, expected_output))` or `None` if the file does not exist
/// or is malformed.
///
/// ## Format
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn read_golden_bin(path: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let data = fs::read(path).ok()?;

    if data.len() < 12 {
        eprintln!(
            "WARN: golden file {path:?} too small ({} bytes)",
            data.len()
        );
        return None;
    }

    let num_samples = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;

    let expected_size = 4 + num_samples * 4 * 2;
    if data.len() < expected_size {
        eprintln!(
            "WARN: golden {path:?} declares {num_samples} samples but has {} bytes (expected {expected_size})",
            data.len()
        );
        return None;
    }

    let input_start = 4;
    let output_start = 4 + num_samples * 4;

    let input: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = input_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    let output: Vec<f32> = (0..num_samples)
        .map(|i| {
            let offset = output_start + i * 4;
            f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ])
        })
        .collect();

    Some((input, output))
}

/// Writes a `.golden.bin` file with the standard binary format:
///
/// ```text
/// [u32 num_samples LE]
/// [f32×N input samples LE]
/// [f32×N expected output LE]
/// ```
pub fn write_golden_bin(path: &Path, input: &[f32], output: &[f32]) -> std::io::Result<()> {
    assert_eq!(
        input.len(),
        output.len(),
        "write_golden_bin: input and output must have same length"
    );
    let mut file = std::fs::File::create(path)?;
    let num_samples = input.len() as u32;
    file.write_all(&num_samples.to_le_bytes())?;
    for sample in input {
        file.write_all(&sample.to_le_bytes())?;
    }
    for sample in output {
        file.write_all(&sample.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

/// Resolves the path to the `tests/fixtures/` directory.
///
/// Search order:
/// 1. `NAM_FIXTURES_DIR` environment variable (explicit override)
/// 2. `CARGO_MANIFEST_DIR/tests/fixtures` (default)
pub fn fixtures_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_FIXTURES_DIR") {
        let p = PathBuf::from(&dir);
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Resolves the path to a test model in `tests/fixtures/models/`.
///
/// Search order:
/// 1. `NAM_MODELS_DIR` environment variable (explicit override)
/// 2. `NAM_THIRD_PARTY_DIR` + `/nam_t3k/` (workspace vendor area)
/// 3. `tests/fixtures/models-nondist` (local non-distributable override)
/// 4. `../third-party/nam_t3k/` (workspace third-party model archive)
/// 5. `tests/fixtures/models` (default — distributed with the repository)
pub fn model_path(filename: &str) -> PathBuf {
    if let Ok(dir) = std::env::var("NAM_MODELS_DIR") {
        let p = PathBuf::from(&dir).join(filename);
        if p.exists() {
            return p;
        }
    }
    if let Ok(tp_dir) = std::env::var("NAM_THIRD_PARTY_DIR") {
        let p = PathBuf::from(&tp_dir).join("nam_t3k").join(filename);
        if p.exists() {
            return p;
        }
    }
    let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut nondist = base.clone();
    nondist.push("tests/fixtures/models-nondist");
    nondist.push(filename);
    if nondist.exists() {
        return nondist;
    }
    let mut t3k = base.clone();
    t3k.push("../third-party/nam_t3k");
    t3k.push(filename);
    if t3k.exists() {
        return t3k;
    }
    base.push("tests/fixtures/models");
    base.push(filename);
    base
}

/// Processes an input block through the model in chunks of `block_size`.
pub fn process_in_blocks(
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    input: &[f32],
    output: &mut [f32],
    block_size: usize,
) {
    let total = input.len();
    let mut pos = 0;
    while pos < total {
        let end = (pos + block_size).min(total);
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
}
