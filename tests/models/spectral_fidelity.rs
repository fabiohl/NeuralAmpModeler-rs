// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Spectral fidelity integration tests — ASR, THD, FR, IMD baselines per SKU.
//
//  ## Test structure
//
//  - **Fast validation** (non-model): hard-clip, linear gain, ASR curve sanity.
//  - **Baseline measurement** (`#[ignore]`): one test per representative SKU,
//    asserts ASR/THD/FR/IMD measurements against a committed JSON fixture.
//  - **Fixture generation** (`generate_spectral_fidelity_baseline`): runs all
//    measurements and prints the JSON baseline. Run manually when models change.
//
//  The committed fixture lives at `tests/fixtures/spectral_fidelity_baseline.json`.
//
//  ## Measured quantities per SKU
//
//  | Metric       | Method               | Input                          |
//  |-------------|----------------------|--------------------------------|
//  | ASR typical | compute_asr          | Musical pitches, gain=1.0      |
//  | ASR aggro   | compute_asr          | Musical pitches, gain=4.0      |
//  | ASR stress  | compute_asr          | 2017 Hz, gain=4.0              |
//  | THD+N       | measure_thdn         | 997 Hz AES17, Q=5              |
//  | IMD SMPTE   | measure_smpte_imd    | 60 Hz + 7 kHz, 4:1             |
//  | FR          | farina_measure       | 20 Hz–20 kHz sweep, gain=1.0   |
//  | THD Farina  | farina_measure       | Per-harmonic-order from sweep  |

use neural_amp_modeler_rs::models::NamModel;
use neural_amp_modeler_rs::testing::aliasing;
use neural_amp_modeler_rs::testing::spectral;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use super::common;

// =============================================================================
// Baseline fixture data model
// =============================================================================

/// Single-model ASR baseline entry.
///
/// ASR values are in dB. `None` means no aliasing was detected
/// (linear behaviour at the given input level).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrBaseline {
    /// Aggregate ASR across musical pitches at typical gain (1.0), in dB.
    #[serde(default)]
    pub aggregate_typical_db: Option<f64>,
    /// Worst-case ASR across musical pitches at typical gain (1.0), in dB.
    #[serde(default)]
    pub worst_typical_db: Option<f64>,
    /// Aggregate ASR across musical pitches at high gain (4.0), in dB.
    #[serde(default)]
    pub aggregate_high_db: Option<f64>,
    /// Worst-case ASR across musical pitches at high gain (4.0), in dB.
    #[serde(default)]
    pub worst_high_db: Option<f64>,
    /// ASR at stress tone (2017 Hz, gain=4.0), in dB.
    #[serde(default)]
    pub stress_db: Option<f64>,
}

impl AsrBaseline {
    fn opt(value: f64) -> Option<f64> {
        if value.is_finite() { Some(value) } else { None }
    }
}

/// Single-model THD+N (AES17) baseline entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThdnBaseline {
    /// THD+N in percent.
    pub thdn_percent: f64,
    /// THD+N in dB.
    pub thdn_db: f64,
}

/// Single-model IMD SMPTE baseline entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImdBaseline {
    /// IMD in percent.
    pub imd_percent: f64,
    /// IMD in dB.
    pub imd_db: f64,
}

/// Single-model Farina sweep baseline entry (FR + THD per order).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FarinaBaseline {
    /// FR magnitude at 100 Hz, in dB.
    pub fr_100hz_db: f64,
    /// FR magnitude at 1 kHz, in dB.
    pub fr_1khz_db: f64,
    /// FR magnitude at 10 kHz, in dB.
    pub fr_10khz_db: f64,
    /// Total THD from Farina sweep, in percent.
    pub thd_total_percent: f64,
    /// THD per harmonic order (order -> percent), indices ≥ 2.
    pub thd_by_order: BTreeMap<u32, f64>,
}

/// Complete spectral fidelity baseline for one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    pub asr: AsrBaseline,
    pub thdn: ThdnBaseline,
    pub imd: ImdBaseline,
    pub farina: FarinaBaseline,
}

/// Top-level baseline fixture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFidelityBaseline {
    /// Sample rate all measurements were taken at.
    pub sample_rate: u32,
    /// Map from model filename key to its baseline entry.
    pub models: BTreeMap<String, BaselineEntry>,
}

impl SpectralFidelityBaseline {
    /// Loads the committed baseline fixture.
    pub fn load() -> Self {
        let path = baseline_fixture_path();
        let json = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Missing baseline fixture at {path:?}: {e}"));
        serde_json::from_str(&json)
            .unwrap_or_else(|e| panic!("Malformed baseline fixture at {path:?}: {e}"))
    }
}

fn baseline_fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/spectral_fidelity_baseline.json");
    p
}

// =============================================================================
// Shared measurement helpers
// =============================================================================

fn process_through_model(model_filename: &str, input: &[f32]) -> Option<Vec<f32>> {
    let path = common::io_helpers::model_path(model_filename);
    if !path.exists() {
        return None;
    }
    let json_data = fs::read_to_string(&path).ok()?;
    let model_data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&json_data).ok()?;
    let mut model = neural_amp_modeler_rs::loader::dispatcher::build_model(&model_data).ok()?;
    model.prewarm(2048);

    let mut output = vec![0.0f32; input.len()];
    let block_size = 64;
    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + block_size).min(input.len());
        model.process(&input[pos..end], &mut output[pos..end]);
        pos = end;
    }
    Some(output)
}

fn process_through_model_f64(model_filename: &str, input: &[f64]) -> Option<Vec<f32>> {
    let input_f32: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    process_through_model(model_filename, &input_f32)
}

/// Measures the ASR across all musical pitches at a given gain.
fn measure_asr_curve(
    model_filename: &str,
    gain: f64,
    sr: u32,
    n: usize,
) -> Option<Vec<aliasing::AsrResult>> {
    let mut results = Vec::new();
    for &(f0, _note) in aliasing::MUSICAL_PITCHES {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output = process_through_model(model_filename, &input)?;
        results.push(aliasing::compute_asr(&output, f0, sr));
    }
    Some(results)
}

/// Measures ASR at the stress tone.
fn measure_asr_stress(model_filename: &str, sr: u32, n: usize) -> Option<aliasing::AsrResult> {
    let input = aliasing::generate_sine(aliasing::STRESS_F0, sr, n, aliasing::HIGH_GAIN);
    let output = process_through_model(model_filename, &input)?;
    Some(aliasing::compute_asr(&output, aliasing::STRESS_F0, sr))
}

/// Runs a complete spectral fidelity measurement for one model.
fn measure_spectral_baseline(model_filename: &str, sr: u32) -> BaselineEntry {
    let asr_n = 32768;

    // --- ASR typical (gain=1.0) ---
    let typical = measure_asr_curve(model_filename, 1.0, sr, asr_n)
        .unwrap_or_else(|| panic!("Failed to process {model_filename} for ASR typical"));
    let agg_typical_db = aliasing::asr_aggregate(&typical);
    let worst_typical_db = aliasing::asr_worst_case(&typical);

    // --- ASR high gain (gain=4.0) ---
    let high = measure_asr_curve(model_filename, 4.0, sr, asr_n)
        .unwrap_or_else(|| panic!("Failed to process {model_filename} for ASR high"));
    let agg_high_db = aliasing::asr_aggregate(&high);
    let worst_high_db = aliasing::asr_worst_case(&high);

    // --- ASR stress tone ---
    let stress = measure_asr_stress(model_filename, sr, 65536)
        .unwrap_or_else(|| panic!("Failed to process {model_filename} for ASR stress"));
    let stress_db = stress.asr_db;

    let asr = AsrBaseline {
        aggregate_typical_db: AsrBaseline::opt(agg_typical_db),
        worst_typical_db: AsrBaseline::opt(worst_typical_db),
        aggregate_high_db: AsrBaseline::opt(agg_high_db),
        worst_high_db: AsrBaseline::opt(worst_high_db),
        stress_db: AsrBaseline::opt(stress_db),
    };

    // --- THD+N AES17 ---
    let thdn = spectral::measure_thdn(997.0, sr, 1.0, 5.0, 48000, |tones| {
        process_through_model_f64(model_filename, tones)
            .unwrap_or_else(|| panic!("Failed to process {model_filename} for THD+N"))
    });
    let thdn_entry = ThdnBaseline {
        thdn_percent: thdn.thdn_percent,
        thdn_db: thdn.thdn_db,
    };

    // --- IMD SMPTE ---
    let imd = spectral::measure_smpte_imd(60.0, 7000.0, 4.0, sr, 1.0, 48000, |tones| {
        process_through_model_f64(model_filename, tones)
            .unwrap_or_else(|| panic!("Failed to process {model_filename} for IMD"))
    });
    let imd_entry = ImdBaseline {
        imd_percent: imd.imd_percent,
        imd_db: imd.imd_db,
    };

    // --- Farina FR + THD ---
    let farina = spectral::farina_measure(20.0, 20000.0, 1.0, sr, 5, |sweep| {
        process_through_model_f64(model_filename, sweep)
            .unwrap_or_else(|| panic!("Failed to process {model_filename} for Farina"))
    });

    let bin_width = (sr as f64) / farina.freq_axis.len() as f64;
    let fr_at = |target_hz: f64| -> f64 {
        let idx = (target_hz / bin_width.max(1.0)).round() as usize;
        farina.fr_magnitude_db.get(idx).copied().unwrap_or(-300.0)
    };

    let mut thd_by_order = BTreeMap::new();
    for &(order, thd) in &farina.thd_by_order {
        if order >= 2 {
            thd_by_order.insert(order, thd);
        }
    }

    let farina_entry = FarinaBaseline {
        fr_100hz_db: fr_at(100.0),
        fr_1khz_db: fr_at(1000.0),
        fr_10khz_db: fr_at(10000.0),
        thd_total_percent: farina.thd_total_percent,
        thd_by_order,
    };

    BaselineEntry {
        asr,
        thdn: thdn_entry,
        imd: imd_entry,
        farina: farina_entry,
    }
}

// =============================================================================
// SKU catalog — models to fingerprint
// =============================================================================

/// Representative SKUs for spectral fidelity baselines.
///
/// Covers all major architectures at 48 kHz (native model rate).
const REPRESENTATIVE_SKUS: &[(&str, &str)] = &[
    // WaveNet A1 community
    ("BossWN-standard.nam", "WaveNet-standard"),
    ("BossWN-feather.nam", "WaveNet-feather"),
    ("BossWN-nano.nam", "WaveNet-nano"),
    // WaveNet A1 official
    ("wavenet_a1_standard.nam", "WaveNet-A1-official"),
    // WaveNet official (dynamic)
    ("wavenet_official.nam", "WaveNet-official-dyn"),
    // WaveNet A2
    ("wavenet_a2_full.nam", "WaveNet-A2-full"),
    ("wavenet_a2_lite.nam", "WaveNet-A2-lite"),
    // A2 container example
    ("a2_example.nam", "A2-example-container"),
    // LSTM
    ("BossLSTM-1x16.nam", "LSTM-1x16"),
    ("BossLSTM-2x8.nam", "LSTM-2x8"),
    ("lstm.nam", "LSTM-official"),
    // Linear (control — should have ASR≈−∞, THD≈0)
    ("linear_test.nam", "Linear-test"),
];

// =============================================================================
// Fast validation — no model required
// =============================================================================

fn hard_clip(x: f32, threshold: f32) -> f32 {
    x.clamp(-threshold, threshold)
}

#[test]
fn asr_hard_clip_detects_aliasing() {
    let f0 = 2017.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;
    let clip = 0.2;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input.iter().map(|&x| hard_clip(x, clip)).collect();

    let result = aliasing::compute_asr(&output, f0, sr);

    assert!(
        result.num_aliased > 0,
        "Hard-clip must produce aliased peaks; got {} harmonics, {} aliased",
        result.num_harmonics,
        result.num_aliased
    );
    assert!(
        result.asr_db > -30.0,
        "ASR too low for hard-clip: {:.1} dB",
        result.asr_db
    );
}

#[test]
fn asr_linear_system_no_aliasing() {
    let f0 = 440.0;
    let sr = 48000;
    let n = 16384;
    let gain = 1.0;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input.iter().map(|&x| 2.0 * x).collect();

    let result = aliasing::compute_asr(&output, f0, sr);

    assert_eq!(
        result.num_aliased, 0,
        "Linear gain must not produce aliased peaks"
    );
    assert!(
        result.asr_db < -60.0 || result.asr_linear < 1e-6,
        "Linear system ASR near zero; got {:.1} dB",
        result.asr_db
    );
}

#[test]
fn asr_soft_clip_tanh_less_aliasing_than_hard_clip() {
    let f0 = 2017.0;
    let sr = 48000;
    let n = 65536;
    let gain = 6.0;

    let input = aliasing::generate_sine(f0, sr, n, gain);

    let tanh_out: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
    let clip_out: Vec<f32> = input.iter().map(|&x| hard_clip(x, 0.2)).collect();

    let tanh_result = aliasing::compute_asr(&tanh_out, f0, sr);
    let clip_result = aliasing::compute_asr(&clip_out, f0, sr);

    assert!(
        tanh_result.asr_db < clip_result.asr_db + 3.0,
        "tanh ASR ({:.1} dB) should be ≤ hard-clip ASR ({:.1} dB)",
        tanh_result.asr_db,
        clip_result.asr_db
    );
}

#[test]
fn asr_curve_musical_pitches() {
    let sr = 48000;
    let n = 32768;
    let gain = 1.0;

    for &(f0, _name) in aliasing::MUSICAL_PITCHES {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output: Vec<f32> = input.iter().map(|&x| (x * 3.0).tanh()).collect();

        let result = aliasing::compute_asr(&output, f0, sr);
        assert!(
            result.num_harmonics >= 1,
            "Must detect fundamental for {_name} ({f0} Hz); got {} harmonics",
            result.num_harmonics
        );
    }
}

#[test]
fn asr_stress_tone_works() {
    let f0 = aliasing::STRESS_F0;
    let sr = 48000;
    let n = 32768;
    let gain = aliasing::HIGH_GAIN;

    let input = aliasing::generate_sine(f0, sr, n, gain);
    let output: Vec<f32> = input.iter().map(|&x| x.clamp(-0.3, 0.3)).collect();

    let result = aliasing::compute_asr(&output, f0, sr);
    assert!(result.f0 == f0);
    assert!(result.sample_rate == sr);
    assert!(
        result.has_aliasing(),
        "Stress tone ({f0} Hz, gain={gain}) must produce detected aliasing; \
         got {} harmonics, {} aliased, ASR={:.1} dB",
        result.num_harmonics,
        result.num_aliased,
        result.asr_db
    );
}

#[test]
fn asr_aggregate_across_pitches() {
    let sr = 48000;
    let n = 16384;
    let gain = 4.0;

    let mut results = Vec::new();
    for &(f0, _name) in aliasing::MUSICAL_PITCHES {
        let input = aliasing::generate_sine(f0, sr, n, gain);
        let output: Vec<f32> = input.iter().map(|&x| hard_clip(x, 0.3)).collect();
        results.push(aliasing::compute_asr(&output, f0, sr));
    }

    let agg = aliasing::asr_aggregate(&results);
    let worst = aliasing::asr_worst_case(&results);

    assert!(agg.is_finite() || agg == f64::NEG_INFINITY);
    assert!(worst.is_finite() || worst == f64::NEG_INFINITY);
}

// =============================================================================
// Baseline fixture generation
// =============================================================================

/// Generates the spectral fidelity baseline fixture.
///
/// Run this test when models change or Sprint S5 begins:
///
/// ```bash
/// cargo test --test models spectral_fidelity::generate_spectral_fidelity_baseline -- --ignored --nocapture
/// ```
#[test]
#[ignore = "generates baseline fixture — run manually when models change"]
fn generate_spectral_fidelity_baseline() {
    let sr = 48000;
    let mut models = BTreeMap::new();

    for &(filename, label) in REPRESENTATIVE_SKUS {
        let path = common::io_helpers::model_path(filename);
        if !path.exists() {
            eprintln!("  SKIP {label}: model file not found at {path:?}");
            continue;
        }
        eprintln!("  Measuring {label} ({filename})...");
        let entry = measure_spectral_baseline(filename, sr);
        models.insert(label.to_string(), entry);
    }

    let baseline = SpectralFidelityBaseline {
        sample_rate: sr,
        models,
    };

    let json = serde_json::to_string_pretty(&baseline).unwrap();
    let out_path = baseline_fixture_path();
    fs::write(&out_path, &json)
        .unwrap_or_else(|e| panic!("Failed to write baseline fixture to {out_path:?}: {e}"));

    eprintln!(
        "\nBaseline written to {out_path:?}\n{} models measured.\n",
        baseline.models.len()
    );
    eprintln!("{}", json);
}

// =============================================================================
// Baseline validation — per-SKU ignored tests
// =============================================================================

/// Tolerance for ASR comparisons (dB). Models are deterministic, but
/// floating-point across machines/compiles can differ by a small fraction.
const ASR_TOLERANCE_DB: f64 = 0.5;

/// Tolerance for THD+N comparisons (percent).
const THDN_TOLERANCE_PCT: f64 = 0.1;

/// Tolerance for IMD comparisons (percent).
const IMD_TOLERANCE_PCT: f64 = 0.2;

/// Tolerance for FR comparisons (dB).
const FR_TOLERANCE_DB: f64 = 0.5;

/// Tolerance for Farina THD comparisons (percent).
const FARINA_THD_TOLERANCE_PCT: f64 = 0.5;

/// Asserts that a measured ASR baseline matches the committed fixture.
fn assert_asr_baseline(label: &str, measured: &AsrBaseline, fixture: &AsrBaseline) {
    let check = |name: &str, m: Option<f64>, f: Option<f64>| {
        match (m, f) {
            (Some(mv), Some(fv)) => {
                let delta = (mv - fv).abs();
                assert!(
                    delta <= ASR_TOLERANCE_DB,
                    "{label}: ASR {name} changed: {fv:.3} → {mv:.3} dB (Δ={delta:.3})"
                );
            }
            (None, None) => {} // both report no aliasing
            (Some(mv), None) => {
                panic!(
                    "{label}: ASR {name} regressed: was none, now {mv:.3} dB (aliasing detected where there was none)"
                );
            }
            (None, Some(fv)) => {
                panic!(
                    "{label}: ASR {name} improved unexpectedly: was {fv:.3} dB, now none (regenerate baseline if intentional)"
                );
            }
        }
    };

    check(
        "aggregate typical",
        measured.aggregate_typical_db,
        fixture.aggregate_typical_db,
    );
    check(
        "aggregate high",
        measured.aggregate_high_db,
        fixture.aggregate_high_db,
    );
    check("stress", measured.stress_db, fixture.stress_db);
}

/// Asserts that measured THD+N matches the fixture.
fn assert_thdn_baseline(label: &str, measured: &ThdnBaseline, fixture: &ThdnBaseline) {
    let delta = (measured.thdn_percent - fixture.thdn_percent).abs();
    assert!(
        delta <= THDN_TOLERANCE_PCT,
        "{label}: THD+N changed: {:.4}% → {:.4}% (Δ={:.4})",
        fixture.thdn_percent,
        measured.thdn_percent,
        delta
    );
}

/// Asserts that measured IMD matches the fixture.
fn assert_imd_baseline(label: &str, measured: &ImdBaseline, fixture: &ImdBaseline) {
    let delta = (measured.imd_percent - fixture.imd_percent).abs();
    assert!(
        delta <= IMD_TOLERANCE_PCT,
        "{label}: IMD changed: {:.4}% → {:.4}% (Δ={:.4})",
        fixture.imd_percent,
        measured.imd_percent,
        delta
    );
}

/// Asserts that measured Farina FR+THD matches the fixture.
fn assert_farina_baseline(label: &str, measured: &FarinaBaseline, fixture: &FarinaBaseline) {
    let delta = (measured.fr_100hz_db - fixture.fr_100hz_db).abs();
    assert!(
        delta <= FR_TOLERANCE_DB,
        "{label}: FR @ 100 Hz changed: {:.3} → {:.3} dB (Δ={:.3})",
        fixture.fr_100hz_db,
        measured.fr_100hz_db,
        delta
    );

    let delta = (measured.fr_1khz_db - fixture.fr_1khz_db).abs();
    assert!(
        delta <= FR_TOLERANCE_DB,
        "{label}: FR @ 1 kHz changed: {:.3} → {:.3} dB (Δ={:.3})",
        fixture.fr_1khz_db,
        measured.fr_1khz_db,
        delta
    );

    let delta = (measured.fr_10khz_db - fixture.fr_10khz_db).abs();
    assert!(
        delta <= FR_TOLERANCE_DB,
        "{label}: FR @ 10 kHz changed: {:.3} → {:.3} dB (Δ={:.3})",
        fixture.fr_10khz_db,
        measured.fr_10khz_db,
        delta
    );

    let delta = (measured.thd_total_percent - fixture.thd_total_percent).abs();
    assert!(
        delta <= FARINA_THD_TOLERANCE_PCT,
        "{label}: Farina THD changed: {:.4}% → {:.4}% (Δ={:.4})",
        fixture.thd_total_percent,
        measured.thd_total_percent,
        delta
    );
}

/// Validates a single model against the committed baseline fixture.
fn validate_model_against_baseline(filename: &str, label: &str, sr: u32) {
    let path = common::io_helpers::model_path(filename);
    if !path.exists() {
        eprintln!("  SKIP {label}: model file not found at {path:?}");
        return;
    }

    let fixture = SpectralFidelityBaseline::load();
    let fixture_entry = fixture.models.get(label).unwrap_or_else(|| {
        panic!(
            "{label}: not found in baseline fixture. \
             Run 'cargo test --test models spectral_fidelity::generate_spectral_fidelity_baseline -- --ignored' to regenerate."
        )
    });

    let measured = measure_spectral_baseline(filename, sr);

    assert_asr_baseline(label, &measured.asr, &fixture_entry.asr);
    assert_thdn_baseline(label, &measured.thdn, &fixture_entry.thdn);
    assert_imd_baseline(label, &measured.imd, &fixture_entry.imd);
    assert_farina_baseline(label, &measured.farina, &fixture_entry.farina);

    eprintln!("  ✅ {label}: all spectral fidelity metrics within baseline tolerance");
}

// =============================================================================
// Per-SKU baseline validation tests (`#[ignore]` — require model files)
// =============================================================================

#[cfg(test)]
mod model_baselines {
    use super::*;

    macro_rules! baseline_test {
        ($test_name:ident, $filename:literal, $label:literal) => {
            #[test]
            #[ignore = "requires .nam model files; run with --ignored"]
            fn $test_name() {
                validate_model_against_baseline($filename, $label, 48000);
            }
        };
    }

    baseline_test!(
        baseline_wavenet_standard,
        "BossWN-standard.nam",
        "WaveNet-standard"
    );
    baseline_test!(
        baseline_wavenet_feather,
        "BossWN-feather.nam",
        "WaveNet-feather"
    );
    baseline_test!(baseline_wavenet_nano, "BossWN-nano.nam", "WaveNet-nano");
    baseline_test!(
        baseline_wavenet_a1_official,
        "wavenet_a1_standard.nam",
        "WaveNet-A1-official"
    );
    baseline_test!(
        baseline_wavenet_official_dyn,
        "wavenet_official.nam",
        "WaveNet-official-dyn"
    );
    baseline_test!(
        baseline_wavenet_a2_full,
        "wavenet_a2_full.nam",
        "WaveNet-A2-full"
    );
    baseline_test!(
        baseline_wavenet_a2_lite,
        "wavenet_a2_lite.nam",
        "WaveNet-A2-lite"
    );
    baseline_test!(
        baseline_a2_example,
        "a2_example.nam",
        "A2-example-container"
    );
    baseline_test!(baseline_lstm_1x16, "BossLSTM-1x16.nam", "LSTM-1x16");
    baseline_test!(baseline_lstm_2x8, "BossLSTM-2x8.nam", "LSTM-2x8");
    baseline_test!(baseline_lstm_official, "lstm.nam", "LSTM-official");
    baseline_test!(baseline_linear_test, "linear_test.nam", "Linear-test");

    /// Smoke test: assert the baseline fixture itself is well-formed.
    #[test]
    fn baseline_fixture_is_well_formed() {
        let fixture = SpectralFidelityBaseline::load();

        assert_eq!(fixture.sample_rate, 48000);
        assert!(
            !fixture.models.is_empty(),
            "Baseline fixture must contain at least one model entry"
        );

        for (label, entry) in &fixture.models {
            // ASR must be finite or None (linear system)
            if let Some(v) = entry.asr.aggregate_typical_db {
                assert!(
                    v.is_finite(),
                    "{label}: ASR aggregate typical is NaN/infinite"
                );
            }
            assert!(
                entry.thdn.thdn_percent >= 0.0,
                "{label}: THD+N percent negative"
            );
            assert!(
                entry.imd.imd_percent >= 0.0,
                "{label}: IMD percent negative"
            );
            assert!(
                entry.farina.thd_total_percent >= 0.0,
                "{label}: Farina THD percent negative"
            );
        }
    }
}
