// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Contract verify engine — literal port of `verify_contract`
//! (`utils/quality-dashboard.sh:2113-2488`).
//!
//! Verdicts are strictly domain-separated (PERF-006 / docs §9.3):
//! `regression_gate` failures classify as `PERFORMANCE: NOT_VERIFIED` and can
//! **never** flip `FIDELITY` to FAIL. Fidelity phases (`golden_vectors`,
//! `reference_oracle_f64`, `quick_parity`) must be `PASS`.
//!
//! Literal-port notes:
//! - ESR envelope limits are rounded with `printf "%.2e"` semantics BEFORE
//!   the comparison — the rounded value is the effective gate
//!   (`quality-dashboard.sh:2241-2250`, S1.T2 note). SNR, MR-STFT and latency
//!   compare the exact limit.
//! - The f64 ESR oracle gets **no** syntactic finite gate (the bash compares
//!   it directly via `awk cur+0`); `inf`/`nan` flow through f64 comparison
//!   semantics. Unparseable garbage is fail-closed instead of awk's
//!   coercion to `0.0` (documented divergence).
//! - The performance domain is only consulted when the contract carries
//!   latency entries (bash quirk preserved): with an empty performance
//!   contract a failing `regression_gate` does not flip the verdict.
//! - Fidelity matching is by `id`: report labels resolve to contract entries
//!   by exact label, with old-report labels resolved **only** through the
//!   `ids.rs` alias table — never ad-hoc string matching.
//!
//! The report format (one JSON object per line) is the future `nam_quality
//! verify --report` input: phase records (`phase_id`/`status`, bash receipt
//! shape), fidelity records (S2.T1 `FidelityRecord`, optionally with the
//! paired `esr_f64`), latency records (`kind` `latency`, `label`,
//! `median_latency_us`). Unknown kinds (S2.T6) are skipped.

use std::collections::HashMap;

use serde_json::Value;

use super::ids;
use super::metrics::{FidelityRecord, MetricValue, fidelity_from_json, is_finite_num};
use super::{EsrNamcoreEnvelope, PerformanceEntry, QualityContract};

/// Mandatory fidelity phases: any status other than `PASS` fails the
/// fidelity domain (missing records count as `NOT_RUN`).
pub const MANDATORY_FIDELITY_PHASES: [&str; 3] =
    ["golden_vectors", "reference_oracle_f64", "quick_parity"];

/// Phase whose outcome belongs to the performance domain (PERF-006).
pub const REGRESSION_GATE_PHASE: &str = "regression_gate";

/// One phase outcome of the report (bash receipt record shape, no `kind`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseRecord {
    /// Phase id, e.g. `golden_vectors`.
    pub phase_id: String,
    /// Receipt status, e.g. `PASS`, `FAIL`, `NOT_VERIFIED`, `NOT_RUN`.
    pub status: String,
    /// Number of records actually observed by the phase (`0` when the receipt
    /// predates the F-07 records field — a phase with `expected_records > 0`
    /// observing zero records is a harness-integrity violation).
    pub observed_records: u64,
    /// Number of records the phase promised to observe (`0` = no promise).
    pub expected_records: u64,
}

/// One latency measurement of the report.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct LatencyRecord {
    /// Bench/model label (matched against the contract label or id).
    pub label: String,
    /// Median block latency in microseconds (finite).
    pub median_latency_us: f64,
}

/// Parsed verify report: phases + fidelity + latency streams.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyReport {
    /// Phase outcome records.
    pub phases: Vec<PhaseRecord>,
    /// Fidelity metric records (S2.T1 shape).
    pub fidelity: Vec<FidelityRecord>,
    /// Latency measurement records.
    pub latency: Vec<LatencyRecord>,
}

/// Parses a verify report stream, fail-closed on malformed lines.
///
/// Routing: records with `phase_id` are phase records, records with
/// `median_latency_us` are latency records, everything else goes through the
/// fidelity canonicalization of S2.T1 (kind filter + label drop). Unknown
/// kinds (`build_metadata` provenance, S2.T6 sink kinds) are skipped.
pub fn parse_verify_report(input: &str) -> Result<VerifyReport, VerifyError> {
    let mut report = VerifyReport {
        phases: Vec::new(),
        fidelity: Vec::new(),
        latency: Vec::new(),
    };
    for (line_no, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(line).map_err(|source| VerifyError::MalformedLine {
                line: line_no + 1,
                source,
            })?;
        if value.get("phase_id").is_some() {
            report.phases.push(phase_from_json(&value, line_no + 1)?);
        } else if value.get("median_latency_us").is_some() {
            report.latency.push(latency_from_json(&value, line_no + 1)?);
        } else if let Some(record) = fidelity_from_json(&value) {
            report.fidelity.push(record);
        }
    }
    Ok(report)
}

/// Parses a verify report file, fail-closed on unreadable input.
pub fn parse_verify_report_file(
    path: impl AsRef<std::path::Path>,
) -> Result<VerifyReport, VerifyError> {
    let input = std::fs::read_to_string(path)?;
    parse_verify_report(&input)
}

/// Typed error of the verify report ingest.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    /// A report line is not valid JSON.
    #[error("malformed report line {line}: {source}")]
    MalformedLine {
        /// 1-based line number in the report.
        line: usize,
        /// The underlying JSON parse error.
        source: serde_json::Error,
    },
    /// A phase record lacks string `phase_id`/`status`.
    #[error("phase record on line {line} must have string `phase_id` and `status`")]
    InvalidPhaseRecord {
        /// 1-based line number in the report.
        line: usize,
    },
    /// A latency record lacks string `label` or finite `median_latency_us`.
    #[error(
        "latency record on line {line} must have string `label` and finite numeric `median_latency_us`"
    )]
    InvalidLatencyRecord {
        /// 1-based line number in the report.
        line: usize,
    },
    /// The report file could not be read.
    #[error("cannot read verify report: {0}")]
    Io(#[from] std::io::Error),
}

/// Extracts the F-07 record counts from a phase record, defaulting to `0/0`
/// for reports that predate the records gate (no promise ⇒ no check).
///
/// Shared by the strict verify parser and the lenient render parser so the
/// gate and the report can never drift on these fields.
pub(crate) fn parse_record_counts(value: &Value) -> (u64, u64) {
    let observed_records = value
        .get("observed_records")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let expected_records = value
        .get("expected_records")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    (observed_records, expected_records)
}

fn phase_from_json(value: &Value, line: usize) -> Result<PhaseRecord, VerifyError> {
    let phase_id = value
        .get("phase_id")
        .and_then(Value::as_str)
        .ok_or(VerifyError::InvalidPhaseRecord { line })?;
    let status = value
        .get("status")
        .and_then(Value::as_str)
        .ok_or(VerifyError::InvalidPhaseRecord { line })?;
    let (observed_records, expected_records) = parse_record_counts(value);
    Ok(PhaseRecord {
        phase_id: phase_id.to_string(),
        status: status.to_string(),
        observed_records,
        expected_records,
    })
}

fn latency_from_json(value: &Value, line: usize) -> Result<LatencyRecord, VerifyError> {
    let label = value
        .get("label")
        .and_then(Value::as_str)
        .ok_or(VerifyError::InvalidLatencyRecord { line })?;
    let median_latency_us = value
        .get("median_latency_us")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or(VerifyError::InvalidLatencyRecord { line })?;
    Ok(LatencyRecord {
        label: label.to_string(),
        median_latency_us,
    })
}

/// Fidelity domain verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FidelityVerdict {
    /// All mandatory phases and metric envelopes passed.
    Ok,
    /// At least one fidelity violation (phases or metrics).
    Fail {
        /// Total fidelity violation count.
        violations: u32,
    },
}

impl FidelityVerdict {
    /// Whether the fidelity domain passed.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// Performance domain verdict (PERF-006).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceVerdict {
    /// Latency contract certified (regression_gate `PASS` + envelopes ok).
    Ok,
    /// `regression_gate != PASS`: the latency contract cannot be certified.
    /// Never a fidelity concern.
    NotVerified,
    /// Latency envelopes violated.
    Fail {
        /// Total performance violation count.
        violations: u32,
    },
}

impl PerformanceVerdict {
    /// Whether the performance domain passed.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }

    /// Whether the performance domain is `NOT_VERIFIED`.
    pub fn is_not_verified(&self) -> bool {
        matches!(self, Self::NotVerified)
    }
}

/// Full verify result of one contract+report pair.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOutcome {
    /// Fidelity domain verdict.
    pub fidelity: FidelityVerdict,
    /// Performance domain verdict.
    pub performance: PerformanceVerdict,
    /// NAMCore vs f64 oracle divergences requiring human review.
    pub review_required: u32,
    /// Per-contract-entry fidelity checks (render material).
    pub fidelity_checks: Vec<FidelityCheck>,
    /// Per-contract-entry latency checks (render material).
    pub perf_checks: Vec<PerfCheck>,
}

impl VerifyOutcome {
    /// `0` only when both domains pass and no oracle review is pending —
    /// mirror of the bash `return 0/1` of `verify_contract`.
    pub fn exit_code(&self) -> i32 {
        if self.fidelity.is_ok() && self.performance.is_ok() && self.review_required == 0 {
            0
        } else {
            1
        }
    }

    /// `0` when only the fidelity domain passes and no oracle review is pending,
    /// ignoring whether performance was certified.
    pub fn fidelity_exit_code(&self) -> i32 {
        if self.fidelity.is_ok() && self.review_required == 0 {
            0
        } else {
            1
        }
    }
}

/// Fidelity check of one contract entry.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityCheck {
    /// Contract id of the entry (canonical match key).
    pub id: String,
    /// Contract label of the entry.
    pub label: String,
    /// Resolution outcome of the entry.
    pub outcome: FidelityOutcome,
}

/// Resolution outcome of one contract fidelity entry.
#[derive(Debug, Clone, PartialEq)]
pub enum FidelityOutcome {
    /// `optional: true` entry absent from the report — no violation.
    OptionalSkipped,
    /// Mandatory entry absent from the report — fail-closed violation.
    MissingLabel,
    /// Entry matched to a report record; per-metric results follow.
    Measured(Vec<MetricCheck>),
}

/// One metric check of a measured fidelity entry.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MetricCheck {
    /// Which metric was checked.
    pub metric: Metric,
    /// Outcome of the check.
    pub outcome: MetricOutcome,
}

/// Fidelity metric identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Linear ESR against the NAMCore reference.
    EsrNamcore,
    /// ESR against the ideal f64 oracle.
    EsrF64,
    /// SNR in dB.
    SnrDb,
    /// MR-STFT metric.
    Mrstft,
}

/// Outcome of one metric check.
#[derive(Debug, Clone, PartialEq)]
pub enum MetricOutcome {
    /// Within the envelope.
    Ok,
    /// ESR above the safety ceiling (fatal ceiling, PERF-009).
    SafetyCeiling {
        /// Measured value.
        current: f64,
        /// Effective (rounded) ceiling.
        limit: f64,
        /// Contract baseline.
        baseline: f64,
    },
    /// ESR above the noise envelope but below the safety ceiling.
    NoiseEnvelope {
        /// Measured value.
        current: f64,
        /// Effective (rounded) ceiling.
        limit: f64,
        /// Contract baseline.
        baseline: f64,
    },
    /// SNR below the floor / MR-STFT above the ceiling.
    Envelope {
        /// Measured value.
        current: f64,
        /// Exact envelope limit.
        limit: f64,
        /// Contract baseline.
        baseline: f64,
    },
    /// Metric malformed in the report (missing/empty or non-finite).
    Malformed(MalformedReason),
    /// f64 baseline present in the contract but not measured in the report.
    Missing,
}

/// Why a metric was malformed in the report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MalformedReason {
    /// `null`/empty/missing field — the `N/A` state.
    Missing,
    /// Non-finite or unparseable literal, verbatim.
    NonFinite(String),
}

/// Latency check of one contract performance entry.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct PerfCheck {
    /// Contract id of the entry.
    pub id: String,
    /// Contract label of the entry.
    pub label: String,
    /// Outcome of the check.
    pub result: PerfResult,
}

/// Outcome of one latency check.
#[derive(Debug, Clone, PartialEq)]
pub enum PerfResult {
    /// Within the latency envelope.
    Ok {
        /// Measured median latency in µs.
        median_us: f64,
    },
    /// Median latency above the exact envelope limit.
    Regressed {
        /// Measured median latency in µs.
        median_us: f64,
        /// Envelope limit in µs (`max(baseline × 1.10, baseline + 0.05)`).
        limit_us: f64,
        /// Contract baseline in µs.
        baseline_us: f64,
    },
    /// Contract entry matched to no report record.
    MissingLabel,
    /// Benchmark matched but carried no data (future sink; see S2.T6/T7).
    MissingLatency,
}

/// Verifies a contract against a report — literal port of the bash
/// `verify_contract` (`quality-dashboard.sh:2113-2488`).
pub fn verify_contract(contract: &QualityContract, report: &VerifyReport) -> VerifyOutcome {
    let mut fidelity_violations: u32 = 0;
    let mut perf_violations: u32 = 0;
    let mut review_required: u32 = 0;

    // ── Phase scan (PERF-006 / docs §9.3) ─────────────────────────────────
    // `regression_gate` failures belong to the performance domain and are
    // accounted exactly once, via `PERFORMANCE: NOT_VERIFIED` below. Every
    // other phase with status FAIL is a fidelity violation. Mandatory
    // fidelity phases that are neither PASS nor FAIL (NOT_RUN, SKIP, …) also
    // count — without double-counting the FAILs of the first scan.
    //
    // F-07 record-count gate: a phase claiming `PASS`/`NOT_RUN` while
    // observing fewer records than it expected is a harness-integrity
    // violation — an empty or suppressed record stream (e.g. a freshness log
    // with zero records) can never be certified. The performance domain
    // (`regression_gate`) is excluded: its record shortfall is already
    // represented by the `NOT_VERIFIED` status (PERF-006), and skip statuses
    // (`SKIP_CAPABILITY`/`SKIP_OPTIONAL_FIXTURE`/`NOT_VERIFIED`) declare that
    // the phase deliberately did not run.
    for phase in &report.phases {
        if phase.status == "FAIL" && phase.phase_id != REGRESSION_GATE_PHASE {
            fidelity_violations += 1;
        }
        // F-07 record-count gate: a phase claiming `PASS`/`NOT_RUN` while
        // observing fewer records than it expected is a harness-integrity
        // violation — an empty or suppressed record stream (e.g. a freshness
        // log with zero records) can never be certified. The performance
        // domain (`regression_gate`) is excluded: its record shortfall is
        // already represented by the `NOT_VERIFIED` status (PERF-006), and
        // skip statuses (`SKIP_CAPABILITY`/`SKIP_OPTIONAL_FIXTURE`/`NOT_VERIFIED`)
        // declare that the phase deliberately did not run. Mandatory phases
        // with a non-PASS status are accounted by the mandatory scan below,
        // so the shortfall applies to them only when they claim `PASS` — one
        // broken phase always yields exactly one violation.
        let mandatory = MANDATORY_FIDELITY_PHASES.contains(&phase.phase_id.as_str());
        let shortfall =
            phase.expected_records > 0 && phase.observed_records < phase.expected_records;
        let clean_claim = phase.status == "PASS" || (phase.status == "NOT_RUN" && !mandatory);
        if phase.phase_id != REGRESSION_GATE_PHASE && clean_claim && shortfall {
            fidelity_violations += 1;
        }
    }
    for phase_id in MANDATORY_FIDELITY_PHASES {
        let status = phase_status(report, phase_id);
        if status != "PASS" && status != "FAIL" {
            fidelity_violations += 1;
        }
    }

    // ── Fidelity domain ────────────────────────────────────────────────────
    let mut fidelity_checks = Vec::new();
    if !contract.fidelity.is_empty() {
        // Report lookup by label; exact labels win over old-report aliases.
        let mut report_by_label: HashMap<&str, &FidelityRecord> = HashMap::new();
        for record in &report.fidelity {
            report_by_label
                .entry(record.label.as_str())
                .or_insert(record);
        }
        for record in &report.fidelity {
            if let Some(canonical) = ids::resolve_fidelity_alias(&record.label) {
                report_by_label.entry(canonical).or_insert(record);
            }
        }

        let envelopes = &contract.envelopes;
        for entry in &contract.fidelity {
            let id = entry.id.clone();
            let label = entry.label.clone();
            let Some(record) = report_by_label.get(entry.label.as_str()).copied() else {
                if entry.optional {
                    fidelity_checks.push(FidelityCheck {
                        id,
                        label,
                        outcome: FidelityOutcome::OptionalSkipped,
                    });
                } else {
                    fidelity_violations += 1;
                    fidelity_checks.push(FidelityCheck {
                        id,
                        label,
                        outcome: FidelityOutcome::MissingLabel,
                    });
                }
                continue;
            };

            let mut metrics = Vec::new();
            let mut namcore_ok = true;

            // ESR against the NAMCore reference.
            let esr_baseline = entry.esr_namcore;
            if esr_baseline.is_finite() {
                match &record.esr {
                    MetricValue::Na | MetricValue::Null => {
                        fidelity_violations += 1;
                        namcore_ok = false;
                        metrics.push(MetricCheck {
                            metric: Metric::EsrNamcore,
                            outcome: MetricOutcome::Malformed(MalformedReason::Missing),
                        });
                    }
                    MetricValue::Raw(raw) if !is_finite_num(raw) => {
                        fidelity_violations += 1;
                        namcore_ok = false;
                        metrics.push(MetricCheck {
                            metric: Metric::EsrNamcore,
                            outcome: MetricOutcome::Malformed(MalformedReason::NonFinite(
                                raw.clone(),
                            )),
                        });
                    }
                    MetricValue::Raw(raw) => {
                        let current = parse_finite(raw);
                        let (noise_limit, safety_limit) =
                            esr_limits(esr_baseline, &envelopes.esr_namcore);
                        if current > safety_limit {
                            fidelity_violations += 1;
                            namcore_ok = false;
                            metrics.push(MetricCheck {
                                metric: Metric::EsrNamcore,
                                outcome: MetricOutcome::SafetyCeiling {
                                    current,
                                    limit: safety_limit,
                                    baseline: esr_baseline,
                                },
                            });
                        } else if current > noise_limit {
                            fidelity_violations += 1;
                            namcore_ok = false;
                            metrics.push(MetricCheck {
                                metric: Metric::EsrNamcore,
                                outcome: MetricOutcome::NoiseEnvelope {
                                    current,
                                    limit: noise_limit,
                                    baseline: esr_baseline,
                                },
                            });
                        } else {
                            metrics.push(MetricCheck {
                                metric: Metric::EsrNamcore,
                                outcome: MetricOutcome::Ok,
                            });
                        }
                    }
                }
            }

            // ESR against the f64 oracle + oracle-divergence review.
            if let Some(f64_baseline) = entry.esr_f64.filter(|v| v.is_finite()) {
                match record.esr_f64.as_raw() {
                    None => {
                        fidelity_violations += 1;
                        metrics.push(MetricCheck {
                            metric: Metric::EsrF64,
                            outcome: MetricOutcome::Missing,
                        });
                    }
                    Some(raw) => match raw.parse::<f64>() {
                        Err(_) => {
                            // Fail-closed: the bash awk coerced garbage to 0.
                            fidelity_violations += 1;
                            metrics.push(MetricCheck {
                                metric: Metric::EsrF64,
                                outcome: MetricOutcome::Malformed(MalformedReason::NonFinite(
                                    raw.to_string(),
                                )),
                            });
                        }
                        Ok(current) => {
                            let (noise_limit, safety_limit) =
                                esr_limits(f64_baseline, &envelopes.esr_namcore);
                            if current > safety_limit {
                                fidelity_violations += 1;
                                if namcore_ok {
                                    review_required += 1;
                                }
                                metrics.push(MetricCheck {
                                    metric: Metric::EsrF64,
                                    outcome: MetricOutcome::SafetyCeiling {
                                        current,
                                        limit: safety_limit,
                                        baseline: f64_baseline,
                                    },
                                });
                            } else if current > noise_limit {
                                fidelity_violations += 1;
                                if namcore_ok {
                                    review_required += 1;
                                }
                                metrics.push(MetricCheck {
                                    metric: Metric::EsrF64,
                                    outcome: MetricOutcome::NoiseEnvelope {
                                        current,
                                        limit: noise_limit,
                                        baseline: f64_baseline,
                                    },
                                });
                            } else {
                                if !namcore_ok {
                                    review_required += 1;
                                }
                                metrics.push(MetricCheck {
                                    metric: Metric::EsrF64,
                                    outcome: MetricOutcome::Ok,
                                });
                            }
                            // Directional divergence (bash :2363-2379): both
                            // baselines strictly positive and both currents
                            // finite; opposite 0.85/1.15 moves need review.
                            if esr_baseline > 0.0
                                && let Some(cur_n_raw) = record.esr.as_finite()
                            {
                                let cur_n = parse_finite(cur_n_raw);
                                let rn = cur_n / esr_baseline;
                                let rf = current / f64_baseline;
                                if (rn < 0.85 && rf > 1.15) || (rn > 1.15 && rf < 0.85) {
                                    review_required += 1;
                                }
                            }
                        }
                    },
                }
            }

            // SNR (exact floor: baseline − 6.0 dB).
            //
            // A JSON `null` SNR is **not** a violation: the canonical sink
            // (`tests/common/validation.rs::json_snr_db`) emits `null` for
            // the positive non-finite SNR — bit-identical perfect parity
            // (`+∞` dB), which is above any floor (P0.T3). An *absent* or
            // empty `snr_db` (the `N/A` state) still fails closed, and the
            // non-finite string sentinels (`"-inf"`, `"nan"`) stay
            // malformed — the sink never writes them for a positive SNR, so
            // their presence signals a foreign or corrupt writer.
            if let Some(snr_baseline) = entry.snr_db.filter(|v| v.is_finite()) {
                match &record.snr_db {
                    MetricValue::Null => {
                        metrics.push(MetricCheck {
                            metric: Metric::SnrDb,
                            outcome: MetricOutcome::Ok,
                        });
                    }
                    MetricValue::Na => {
                        fidelity_violations += 1;
                        metrics.push(MetricCheck {
                            metric: Metric::SnrDb,
                            outcome: MetricOutcome::Malformed(MalformedReason::Missing),
                        });
                    }
                    MetricValue::Raw(raw) if !is_finite_num(raw) => {
                        fidelity_violations += 1;
                        metrics.push(MetricCheck {
                            metric: Metric::SnrDb,
                            outcome: MetricOutcome::Malformed(MalformedReason::NonFinite(
                                raw.clone(),
                            )),
                        });
                    }
                    MetricValue::Raw(raw) => {
                        let current = parse_finite(raw);
                        let limit = snr_baseline - envelopes.snr_db_drop;
                        if current < limit {
                            fidelity_violations += 1;
                            metrics.push(MetricCheck {
                                metric: Metric::SnrDb,
                                outcome: MetricOutcome::Envelope {
                                    current,
                                    limit,
                                    baseline: snr_baseline,
                                },
                            });
                        } else {
                            metrics.push(MetricCheck {
                                metric: Metric::SnrDb,
                                outcome: MetricOutcome::Ok,
                            });
                        }
                    }
                }
            }

            // MR-STFT (exact ceiling: baseline × 10.0).
            if entry.mrstft.is_finite() {
                match &record.mrstft {
                    MetricValue::Na | MetricValue::Null => {
                        fidelity_violations += 1;
                        metrics.push(MetricCheck {
                            metric: Metric::Mrstft,
                            outcome: MetricOutcome::Malformed(MalformedReason::Missing),
                        });
                    }
                    MetricValue::Raw(raw) if !is_finite_num(raw) => {
                        fidelity_violations += 1;
                        metrics.push(MetricCheck {
                            metric: Metric::Mrstft,
                            outcome: MetricOutcome::Malformed(MalformedReason::NonFinite(
                                raw.clone(),
                            )),
                        });
                    }
                    MetricValue::Raw(raw) => {
                        let current = parse_finite(raw);
                        let limit = entry.mrstft * envelopes.mrstft_mult;
                        if current > limit {
                            fidelity_violations += 1;
                            metrics.push(MetricCheck {
                                metric: Metric::Mrstft,
                                outcome: MetricOutcome::Envelope {
                                    current,
                                    limit,
                                    baseline: entry.mrstft,
                                },
                            });
                        } else {
                            metrics.push(MetricCheck {
                                metric: Metric::Mrstft,
                                outcome: MetricOutcome::Ok,
                            });
                        }
                    }
                }
            }

            fidelity_checks.push(FidelityCheck {
                id,
                label,
                outcome: FidelityOutcome::Measured(metrics),
            });
        }
    }

    // ── Performance domain (PERF-006) ─────────────────────────────────────
    let mut perf_checks = Vec::new();
    let performance = if contract.performance.is_empty() {
        // Bash literal: without latency entries the performance domain is not
        // consulted, even when regression_gate != PASS (quirk preserved).
        PerformanceVerdict::Ok
    } else if phase_status(report, REGRESSION_GATE_PHASE) != "PASS" {
        PerformanceVerdict::NotVerified
    } else {
        let envelopes = &contract.envelopes;
        for entry in &contract.performance {
            let id = entry.id.clone();
            let label = entry.label.clone();
            let Some(record) = match_latency(entry, &report.latency) else {
                perf_violations += 1;
                perf_checks.push(PerfCheck {
                    id,
                    label,
                    result: PerfResult::MissingLabel,
                });
                continue;
            };
            let baseline_us = entry.median_latency_us;
            let limit_us = (baseline_us * envelopes.latency_mult)
                .max(baseline_us + envelopes.latency_floor_us);
            if record.median_latency_us > limit_us {
                perf_violations += 1;
                perf_checks.push(PerfCheck {
                    id,
                    label,
                    result: PerfResult::Regressed {
                        median_us: record.median_latency_us,
                        limit_us,
                        baseline_us,
                    },
                });
            } else {
                perf_checks.push(PerfCheck {
                    id,
                    label,
                    result: PerfResult::Ok {
                        median_us: record.median_latency_us,
                    },
                });
            }
        }
        if perf_violations > 0 {
            PerformanceVerdict::Fail {
                violations: perf_violations,
            }
        } else {
            PerformanceVerdict::Ok
        }
    };

    VerifyOutcome {
        fidelity: if fidelity_violations > 0 {
            FidelityVerdict::Fail {
                violations: fidelity_violations,
            }
        } else {
            FidelityVerdict::Ok
        },
        performance,
        review_required,
        fidelity_checks,
        perf_checks,
    }
}

/// Status of one phase — the bash `$(...)` assignment keeps the LAST
/// matching receipt record; absent records read as `NOT_RUN`.
fn phase_status<'a>(report: &'a VerifyReport, phase_id: &str) -> &'a str {
    report
        .phases
        .iter()
        .rev()
        .find(|p| p.phase_id == phase_id)
        .map(|p| p.status.as_str())
        .unwrap_or("NOT_RUN")
}

/// ESR envelope limits of one oracle — bash `:2241-2250` semantics: the
/// limits are rounded with `printf "%.2e"` BEFORE the comparison, so the
/// rounded values are the effective gates.
fn esr_limits(baseline: f64, env: &EsrNamcoreEnvelope) -> (f64, f64) {
    let noise = (baseline * env.noise_mult).max(baseline + env.noise_floor_abs);
    let safety = (baseline * env.safety_mult).max(env.safety_floor_abs);
    (round_printf_2e(noise), round_printf_2e(safety))
}

/// Replicates `printf "%.2e"` rounding: mantissa rounded to 2 digits after
/// the decimal point, then re-parsed — the same string round trip the bash
/// performs (`awk 'BEGIN { printf "%.2e", lim }'` fed back into the
/// comparison).
fn round_printf_2e(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        return v;
    }
    format!("{v:.2e}").parse::<f64>().unwrap_or(v)
}

/// Parses a raw metric that passed the `is_finite_num` syntactic gate.
///
/// Values like `"1e400"` yield `inf` — the same fail-closed semantics of the
/// bash `awk cur+0` comparisons.
fn parse_finite(raw: &str) -> f64 {
    raw.parse::<f64>()
        .expect("is_finite_num accepted a non-parseable literal")
}

/// Matches a contract performance entry against the report latency records:
/// exact label, exact id, the `RT_*` bench-label alias table
/// (`ids::resolve_rt_contract_id` — the Criterion bench names differ from the
/// contract ids for `RT_Linear` and the DSP benches), or bash-style
/// normalized equality (`×`→`x`, `→`→`->`, collapsed double spaces,
/// case-insensitive).
fn match_latency<'a>(
    entry: &PerformanceEntry,
    report: &'a [LatencyRecord],
) -> Option<&'a LatencyRecord> {
    report.iter().find(|record| {
        record.label == entry.label
            || record.label == entry.id
            || ids::resolve_rt_contract_id(&record.label) == Some(entry.id.as_str())
            || normalize_bench_label(&record.label) == normalize_bench_label(&entry.label)
    })
}

fn normalize_bench_label(label: &str) -> String {
    label
        .replace('×', "x")
        .replace('→', "->")
        .replace("  ", " ")
        .to_lowercase()
}

#[cfg(test)]
#[path = "verify_test.rs"]
mod verify_test;
