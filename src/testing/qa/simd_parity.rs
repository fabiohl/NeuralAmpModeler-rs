// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Mandatory cross-ISA parity matrix contract and fail-closed JSONL reconciler.
//!
//! Defines the exact 10 mandatory AVX2 -> AVX-512 comparisons at 48 kHz
//! (7 Fast + 3 Standard/HF) in `tests/parity/isa_parity.rs` — excluding
//! deprecated VNNI/BF16 legacy cases that fold into the `f32` AVX-512 kernels —
//! and verifies completeness, run identity, uniqueness, and numerical budgets.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

static RECORD_WRITE_LOCK: Mutex<()> = Mutex::new(());

/// Activation precision path exercised by a cross-ISA parity case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimdActivationPrecision {
    /// Fast Padé/minimax activation kernels.
    Fast,
    /// Standard exact-grade (HF) polynomial kernels.
    Standard,
}

impl SimdActivationPrecision {
    /// Human-readable label for tables and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Standard => "standard",
        }
    }
}

/// Machine-readable execution outcome for one mandatory cross-ISA parity case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimdParityStatus {
    /// Both reference and test ISAs ran and met the ESR budget.
    Passed,
    /// Both ISAs ran, but the comparison violated the ESR budget or produced non-finite metrics.
    Failed,
    /// Host CPU or build lacked required AVX-512 capability.
    SkippedCapability,
    /// Mandatory model or golden fixture was missing on disk.
    MissingFixture,
    /// Loading, parsing, ISA guard installation, or model construction failed.
    Error,
}

/// Static specification of one mandatory AVX2 -> AVX-512 parity comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimdParityCaseSpec {
    /// Stable canonical case identifier.
    pub case_id: &'static str,
    /// Human-readable display label.
    pub label: &'static str,
    /// Model filename under `tests/fixtures/models/`.
    pub model: &'static str,
    /// Golden base name (without `_v2_<sr>.bin`).
    pub golden: &'static str,
    /// Golden fixture filename under `tests/fixtures/`.
    pub fixture: &'static str,
    /// Activation precision path.
    pub precision: SimdActivationPrecision,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Reference ISA name.
    pub ref_isa: &'static str,
    /// Target ISA under test.
    pub test_isa: &'static str,
    /// Strict upper bound for cross-ISA ESR (`esr < max_esr`).
    pub max_esr: f64,
}

/// Conservative cross-ISA ESR budget for WaveNet models (Fast and Standard/HF).
pub const WN_CROSS_ISA_ESR_BUDGET: f64 = 1e-3;
/// Conservative cross-ISA ESR budget for LSTM models (Fast and Standard/HF).
pub const LSTM_CROSS_ISA_ESR_BUDGET: f64 = 1e-2;
/// Conservative cross-ISA ESR budget for WaveNet A2 models (Fast).
pub const A2_CROSS_ISA_ESR_BUDGET: f64 = 1e-3;

/// Canonical 10-case mandatory cross-ISA contract (7 Fast + 3 Standard/HF at 48 kHz).
///
/// Deprecated VNNI/BF16 legacy cases fold directly into `Avx512Math` (`f32`) and
/// are intentionally excluded so they cannot inflate or mask matrix coverage.
pub const MANDATORY_AVX512_PARITY_CASES: [SimdParityCaseSpec; 10] = [
    SimdParityCaseSpec {
        case_id: "wn-standard-fast",
        label: "WN-Std @ 48000 Hz",
        model: "BossWN-standard.nam",
        golden: "golden_wavenet_standard",
        fixture: "golden_wavenet_standard_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: WN_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "wn-feather-fast",
        label: "WN-Feather @ 48000 Hz",
        model: "BossWN-feather.nam",
        golden: "golden_wavenet_feather",
        fixture: "golden_wavenet_feather_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: WN_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "wn-nano-fast",
        label: "WN-Nano @ 48000 Hz",
        model: "BossWN-nano.nam",
        golden: "golden_wavenet_nano",
        fixture: "golden_wavenet_nano_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: WN_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "lstm-1x16-fast",
        label: "LSTM-1x16 @ 48000 Hz",
        model: "BossLSTM-1x16.nam",
        golden: "golden_lstm_1x16",
        fixture: "golden_lstm_1x16_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: LSTM_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "lstm-2x8-fast",
        label: "LSTM-2x8 @ 48000 Hz",
        model: "BossLSTM-2x8.nam",
        golden: "golden_lstm_2x8",
        fixture: "golden_lstm_2x8_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: LSTM_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "a2-full-fast",
        label: "A2-Full @ 48000 Hz",
        model: "wavenet_a2_full.nam",
        golden: "golden_wavenet_a2_full",
        fixture: "golden_wavenet_a2_full_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: A2_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "a2-lite-fast",
        label: "A2-Lite @ 48000 Hz",
        model: "wavenet_a2_lite.nam",
        golden: "golden_wavenet_a2_lite",
        fixture: "golden_wavenet_a2_lite_v2_48000.bin",
        precision: SimdActivationPrecision::Fast,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: A2_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "lstm-1x16-standard",
        label: "LSTM-1x16 @ 48000 Hz (HF)",
        model: "BossLSTM-1x16.nam",
        golden: "golden_lstm_1x16",
        fixture: "golden_lstm_1x16_v2_48000.bin",
        precision: SimdActivationPrecision::Standard,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: LSTM_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "lstm-2x8-standard",
        label: "LSTM-2x8 @ 48000 Hz (HF)",
        model: "BossLSTM-2x8.nam",
        golden: "golden_lstm_2x8",
        fixture: "golden_lstm_2x8_v2_48000.bin",
        precision: SimdActivationPrecision::Standard,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: LSTM_CROSS_ISA_ESR_BUDGET,
    },
    SimdParityCaseSpec {
        case_id: "wn-standard-standard",
        label: "WN-Std @ 48000 Hz (HF)",
        model: "BossWN-standard.nam",
        golden: "golden_wavenet_standard",
        fixture: "golden_wavenet_standard_v2_48000.bin",
        precision: SimdActivationPrecision::Standard,
        sample_rate: 48_000,
        ref_isa: "AVX2",
        test_isa: "AVX-512",
        max_esr: WN_CROSS_ISA_ESR_BUDGET,
    },
];

/// Looks up a mandatory cross-ISA case by its canonical `case_id`.
#[must_use]
pub fn find_mandatory_case(case_id: &str) -> Option<&'static SimdParityCaseSpec> {
    MANDATORY_AVX512_PARITY_CASES
        .iter()
        .find(|spec| spec.case_id == case_id)
}

/// Structured JSONL record emitted for one cross-ISA parity comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimdParityRecord {
    /// Run identifier matching `NAM_RUN_ID`.
    pub run_id: String,
    /// Canonical case identifier from [`MANDATORY_AVX512_PARITY_CASES`].
    pub case_id: String,
    /// Model filename evaluated.
    pub model: String,
    /// Golden fixture filename evaluated.
    pub fixture: String,
    /// Activation precision path (`fast` or `standard`).
    pub precision: SimdActivationPrecision,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Reference ISA name (`"AVX2"`).
    pub ref_isa: String,
    /// Target ISA under test (`"AVX-512"`).
    pub test_isa: String,
    /// Case outcome.
    pub status: SimdParityStatus,
    /// Error-to-signal ratio (finite when comparison ran).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub esr: Option<f64>,
    /// Mean squared error (finite when comparison ran).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mse: Option<f64>,
    /// Maximum absolute sample error (finite when comparison ran).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_abs_err: Option<f64>,
    /// ESR upper budget for the case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<f64>,
    /// Number of IEEE-754 bit mismatches observed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitwise_mismatches: Option<usize>,
    /// Number of compared output samples.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_samples: Option<usize>,
    /// Optional diagnostic reason or error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Appends one [`SimdParityRecord`] line to `path`, failing closed on any I/O or serialization error.
pub fn append_simd_parity_record(path: &Path, record: &SimdParityRecord) -> Result<(), String> {
    if record.run_id.trim().is_empty() {
        return Err("cannot write SIMD parity record with empty run_id".to_owned());
    }
    let line = serde_json::to_string(record)
        .map_err(|e| format!("failed to serialize SIMD parity record: {e}"))?;

    let _guard = RECORD_WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create SIMD parity output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("failed to open SIMD parity sink {}: {e}", path.display()))?;
    writeln!(file, "{line}")
        .and_then(|()| file.flush())
        .map_err(|e| format!("failed to write SIMD parity sink {}: {e}", path.display()))
}

/// Summary produced by [`reconcile_simd_parity_file`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SimdParityReconciliation {
    /// Expected run identifier.
    pub run_id: String,
    /// Number of mandatory cases in the contract.
    pub expected_cases: usize,
    /// Number of distinct valid contract records observed.
    pub observed_cases: usize,
    /// Number of mandatory cases that passed all checks.
    pub passed_cases: usize,
    /// Fail-closed contract violations detected during reconciliation.
    pub violations: Vec<String>,
    /// Parsed records in file order.
    pub records: Vec<SimdParityRecord>,
}

impl SimdParityReconciliation {
    /// Returns `true` only when every mandatory case passed with zero violations.
    #[must_use]
    pub fn is_pass(&self) -> bool {
        self.violations.is_empty() && self.passed_cases == self.expected_cases
    }
}

/// Reconciles `path` against [`MANDATORY_AVX512_PARITY_CASES`] and `expected_run_id`.
///
/// Fails closed on missing/unreadable/directory/empty files, malformed JSONL,
/// unknown fields, stale/mismatched `run_id`, unexpected/duplicate cases,
/// metadata mismatches, non-passing statuses, non-finite/out-of-budget metrics,
/// and any missing mandatory case.
pub fn reconcile_simd_parity_file(
    path: &Path,
    expected_run_id: &str,
) -> Result<SimdParityReconciliation, String> {
    let expected_run_id = expected_run_id.trim();
    if expected_run_id.is_empty() {
        return Err("expected_run_id must not be empty".to_owned());
    }

    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Err(format!("SIMD parity JSONL not found: {}", path.display()));
        }
        Err(e) => {
            return Err(format!(
                "cannot stat SIMD parity JSONL {}: {e}",
                path.display()
            ));
        }
    }

    let meta = fs::metadata(path).map_err(|e| {
        format!(
            "SIMD parity JSONL {} is unreadable or a broken symlink: {e}",
            path.display()
        )
    })?;
    if meta.is_dir() {
        return Err(format!(
            "SIMD parity JSONL path {} is a directory",
            path.display()
        ));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| format!("failed to read SIMD parity JSONL {}: {e}", path.display()))?;
    if content.is_empty() {
        return Err(format!(
            "SIMD parity JSONL {} is empty (0 records)",
            path.display()
        ));
    }

    let mut violations = Vec::new();
    let mut records = Vec::new();
    let mut seen: HashSet<&'static str> = HashSet::new();
    let mut passed_cases = 0usize;

    for (idx, raw_line) in content.split_terminator('\n').enumerate() {
        let line_no = idx + 1;
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.trim().is_empty() {
            return Err(format!(
                "malformed empty JSONL line at {}:{}",
                path.display(),
                line_no
            ));
        }
        let record: SimdParityRecord = serde_json::from_str(line).map_err(|e| {
            format!(
                "malformed SIMD parity JSONL at {}:{}: {e}",
                path.display(),
                line_no
            )
        })?;

        if record.run_id != expected_run_id {
            violations.push(format!(
                "line {line_no}: stale/mismatched run_id '{}' (expected '{expected_run_id}')",
                record.run_id
            ));
        }

        let Some(spec) = find_mandatory_case(&record.case_id) else {
            violations.push(format!(
                "line {line_no}: unexpected non-contract case_id '{}'",
                record.case_id
            ));
            records.push(record);
            continue;
        };

        if !seen.insert(spec.case_id) {
            violations.push(format!(
                "line {line_no}: duplicate case_id '{}'",
                spec.case_id
            ));
            records.push(record);
            continue;
        }

        validate_record_against_spec(line_no, &record, spec, &mut violations, &mut passed_cases);
        records.push(record);
    }

    for spec in &MANDATORY_AVX512_PARITY_CASES {
        if !seen.contains(spec.case_id) {
            violations.push(format!(
                "missing mandatory case '{}' ({} / {} / {} Hz)",
                spec.case_id,
                spec.model,
                spec.precision.as_str(),
                spec.sample_rate
            ));
        }
    }

    Ok(SimdParityReconciliation {
        run_id: expected_run_id.to_owned(),
        expected_cases: MANDATORY_AVX512_PARITY_CASES.len(),
        observed_cases: seen.len(),
        passed_cases,
        violations,
        records,
    })
}

fn validate_record_against_spec(
    line_no: usize,
    record: &SimdParityRecord,
    spec: &SimdParityCaseSpec,
    violations: &mut Vec<String>,
    passed_cases: &mut usize,
) {
    let mut meta_ok = true;
    if record.model != spec.model
        || record.fixture != spec.fixture
        || record.precision != spec.precision
        || record.sample_rate != spec.sample_rate
        || record.ref_isa != spec.ref_isa
        || record.test_isa != spec.test_isa
    {
        meta_ok = false;
        violations.push(format!(
            "line {line_no}: contract metadata mismatch for '{}' (model={}, fixture={}, precision={}, sr={}, {}->{})",
            spec.case_id,
            record.model,
            record.fixture,
            record.precision.as_str(),
            record.sample_rate,
            record.ref_isa,
            record.test_isa
        ));
    }

    if record.status != SimdParityStatus::Passed {
        let detail = record.detail.as_deref().unwrap_or("none");
        violations.push(format!(
            "line {line_no}: case '{}' did not pass (status={:?}, detail={detail})",
            spec.case_id, record.status
        ));
        return;
    }

    let metrics_ok = matches!(
        (
            record.esr,
            record.mse,
            record.max_abs_err,
            record.budget,
            record.total_samples
        ),
        (Some(esr), Some(mse), Some(mae), Some(budget), Some(samples))
            if esr.is_finite()
                && mse.is_finite()
                && mae.is_finite()
                && budget.is_finite()
                && esr >= 0.0
                && mse >= 0.0
                && mae >= 0.0
                && esr < spec.max_esr
                && (budget - spec.max_esr).abs() <= 1e-15
                && samples > 0
    );

    if !metrics_ok {
        violations.push(format!(
            "line {line_no}: invalid or out-of-budget metrics for '{}' (esr={:?}, mse={:?}, max_abs_err={:?}, budget={:?}, samples={:?})",
            spec.case_id,
            record.esr,
            record.mse,
            record.max_abs_err,
            record.budget,
            record.total_samples
        ));
        return;
    }

    if meta_ok {
        *passed_cases += 1;
    }
}
