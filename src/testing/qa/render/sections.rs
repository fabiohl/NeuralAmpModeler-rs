// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! ANSI/Plain section renderers for the quality dashboard.

use super::super::metrics::{FidelityRecord, MetricValue};
use super::super::phases;
use super::super::verify::LatencyRecord;
use super::types::{ActivationRow, IsaRow, QualityReport, RenderStyle};

pub(super) struct Palette {
    green: &'static str,
    yellow: &'static str,
    red: &'static str,
    bold: &'static str,
    nc: &'static str,
}

impl Palette {
    pub(super) fn new(style: RenderStyle) -> Self {
        match style {
            RenderStyle::Ansi => Self {
                green: "\x1b[0;32m",
                yellow: "\x1b[1;33m",
                red: "\x1b[0;31m",
                bold: "\x1b[1m",
                nc: "\x1b[0m",
            },
            RenderStyle::Plain => Self {
                green: "",
                yellow: "",
                red: "",
                bold: "",
                nc: "",
            },
        }
    }

    fn paint(&self, color: &str, text: &str) -> String {
        format!("{color}{text}{}", self.nc)
    }

    fn paint_class(&self, class: &str, text: &str) -> String {
        match class {
            "green" => self.green(text),
            "yellow" => self.yellow(text),
            "red" => self.red(text),
            _ => text.to_string(),
        }
    }

    fn green(&self, text: &str) -> String {
        self.paint(self.green, text)
    }

    fn yellow(&self, text: &str) -> String {
        self.paint(self.yellow, text)
    }

    fn red(&self, text: &str) -> String {
        self.paint(self.red, text)
    }

    fn bold(&self, text: &str) -> String {
        self.paint(self.bold, text)
    }
}

// ── Formatting helpers ──────────────────────────────────────────────────────

/// `printf "%.2e"` for small/scientific values, `%.4f` otherwise (mirror of
/// the bash `_fmt_metric`); `N/A` and non-finite literals pass through.
fn fmt_metric(mv: &MetricValue) -> String {
    match mv {
        MetricValue::Na | MetricValue::Null => "N/A".to_string(),
        MetricValue::Raw(raw) => match raw.parse::<f64>() {
            Ok(v) if v.is_finite() => {
                if raw.contains(['e', 'E']) || (v != 0.0 && v.abs() < 0.0001) {
                    format!("{v:.2e}")
                } else {
                    format!("{v:.4}")
                }
            }
            _ => raw.clone(),
        },
    }
}

/// SNR is rendered with one decimal (mirror of the bash `_nfmt "%.1f"`).
fn fmt_snr(mv: &MetricValue) -> String {
    match mv {
        MetricValue::Na | MetricValue::Null => "N/A".to_string(),
        MetricValue::Raw(raw) => match raw.parse::<f64>() {
            Ok(v) if v.is_finite() => format!("{v:.1}"),
            _ => raw.clone(),
        },
    }
}

/// Finite numeric value behind a metric, if any.
fn metric_f64(mv: &MetricValue) -> Option<f64> {
    match mv {
        MetricValue::Na | MetricValue::Null => None,
        MetricValue::Raw(raw) => raw.parse::<f64>().ok().filter(|v| v.is_finite()),
    }
}

/// Visible width of a string, ignoring ANSI escape sequences.
fn visible_width(s: &str) -> usize {
    let mut width = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            while let Some(&n) = chars.peek() {
                chars.next();
                if n == 'm' {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
}

/// Left-pads `s` to `width` visible columns (ANSI-aware).
fn pad(s: &str, width: usize) -> String {
    let visible = visible_width(s);
    if visible >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - visible))
    }
}

/// Truncates a label to `max` characters (the dashboard truncates to 38).
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max).collect()
    }
}

/// Strips the trailing ` @rate` and ` Live`/` HQ` decorations, mirroring the
/// bash `sed 's/ @.*//; s/ Live$//; s/ HQ$//'`.
fn model_label(label: &str) -> String {
    let mut out = label.to_string();
    if let Some(pos) = out.find(" @") {
        out.truncate(pos);
    }
    for suffix in [" Live", " HQ"] {
        if let Some(stripped) = out.strip_suffix(suffix) {
            out = stripped.to_string();
        }
    }
    out
}

// ── Qualitative color/verdict thresholds (mirror of the bash helpers) ──────

/// ESR color class: green < 1e-5, yellow < 1e-1, red otherwise.
fn esr_color_class(esr: f64) -> &'static str {
    if esr < 1e-5 {
        "green"
    } else if esr < 1e-1 {
        "yellow"
    } else {
        "red"
    }
}

/// Short qualitative ESR verdict (English — the bash used PT here).
fn esr_verdict_short(esr: f64) -> &'static str {
    if esr == 0.0 {
        "ZERO-ERROR"
    } else if esr < 1e-10 {
        "NOISE-FLOOR"
    } else if esr < 1e-5 {
        "IMPERCEPTIBLE"
    } else if esr < 1e-2 {
        "A/B SCIENTIFIC"
    } else if esr < 1e-1 {
        "AUDIBLE DIRECT"
    } else {
        "⚠ AUDIBLE"
    }
}

/// CPU budget percentage (RT deadline 1333 µs).
fn budget_pct(latency_us: f64) -> f64 {
    (latency_us / 1333.0) * 100.0
}

/// CPU headroom percentage.
fn budget_headroom(pct: f64) -> f64 {
    100.0 - pct
}

/// CPU budget color class by headroom (green > 50, yellow > 25, red otherwise).
fn cpu_color_class(pct: f64) -> &'static str {
    let headroom = budget_headroom(pct);
    if headroom > 50.0 {
        "green"
    } else if headroom > 25.0 {
        "yellow"
    } else {
        "red"
    }
}

// ── Section renderers ───────────────────────────────────────────────────────

pub(super) fn render_header(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    let h = &report.header;
    let dirty = if h.git_dirty { "dirty" } else { "clean" };
    out.push_str(&p.bold("===============================================================\n"));
    out.push_str("   NeuralAmpModeler-rs Quality Dashboard\n");
    out.push_str("   ------------------------------\n");
    if let Some(measured_at) = &h.measured_at {
        out.push_str(&format!("   Measured at: {measured_at}\n"));
    }
    if let Some(cpu) = &h.cpu_model {
        out.push_str(&format!("   CPU:    {cpu}\n"));
    }
    out.push_str(&format!("   Commit: {} ({dirty})\n", h.git_commit));
    out.push_str(&format!("   Run ID: {}\n", h.run_id));
    match &h.host_isa {
        Some(host) if host != &h.effective_isa => {
            out.push_str(&format!(
                "   ISA:    {} (active engine dispatch; host CPU: {host})\n",
                h.effective_isa
            ));
        }
        _ => {
            out.push_str(&format!("   ISA:    {}\n", h.effective_isa));
        }
    }
    out.push_str(&format!("   rustc:  {}\n", h.rustc));
    out.push_str(&format!(
        "   profile: {} · target: {}\n",
        h.cargo_profile, h.target_triple
    ));
    out.push_str(&p.bold("===============================================================\n"));
    out
}

/// Representative quick-summary models (display label, latency bench id,
/// case-insensitive label substrings) — mirrors the bash quick-summary map.
const QUICK_REPS: &[(&str, &str, &[&str])] = &[
    (
        "WaveNet Standard (CH16)",
        "RT_WaveNet_Std_CH16",
        &["BossWN-standard", "WaveNet Std"],
    ),
    (
        "WaveNet A1 Standard",
        "RT_WaveNet_Std_CH16",
        &["wavenet_a1_standard", "A1 Standard"],
    ),
    (
        "WaveNet Feather (CH8)",
        "RT_WaveNet_Feather_CH8",
        &["BossWN-feather", "WaveNet Feather"],
    ),
    (
        "LSTM 1x16 (BossLSTM)",
        "RT_LSTM_1x16",
        &["BossLSTM-1x16", "LSTM 1x16"],
    ),
    (
        "LSTM 2x8 (BossLSTM)",
        "RT_LSTM_2x8",
        &["BossLSTM-2x8", "LSTM 2x8"],
    ),
    (
        "A2 Full (CH8)",
        "RT_A2_Full_CH8",
        &["WaveNet A2-Full", "A2-Full", "A2 Full"],
    ),
    (
        "A2 Lite (CH3)",
        "RT_A2_Lite_CH3",
        &["WaveNet A2-Lite", "A2-Lite", "A2 Lite"],
    ),
    (
        "A2-FiLM Lite (CH3)",
        "RT_A2_Lite_CH3",
        &["A2-FiLM-Lite", "A2-FiLM Lite"],
    ),
    ("ConvNet", "RT_ConvNet", &["ConvNet"]),
    (
        "Linear (RF=2048)",
        "RT_Linear_Fft_RF2048",
        &["linear_fft_rf2048", "Linear FFT RF=2048"],
    ),
];

fn find_representative<'a>(
    fidelity: &'a [FidelityRecord],
    patterns: &[&str],
) -> Option<&'a FidelityRecord> {
    for pat in patterns {
        let needle = pat.to_lowercase();
        let mut best: Option<(u8, &'a FidelityRecord)> = None;
        for record in fidelity {
            if !record.label.to_lowercase().contains(&needle) {
                continue;
            }
            let canonical = !is_redundant_measurement(&model_label(&record.label));
            let is_48k = record.label.contains("@48000");
            let is_proxy = record
                .f64_oracle_provenance
                .as_deref()
                .is_some_and(|p| p.starts_with("proxy:"));
            let has_real_f64 = metric_f64(&record.esr_f64).is_some() && !is_proxy;
            let tier = match (canonical, is_48k, has_real_f64) {
                (true, true, true) => 1,
                (true, true, false) => 2,
                (true, false, true) => 3,
                (true, false, false) => 4,
                (false, _, _) => 5,
            };
            if best.is_none_or(|(best_tier, _)| tier < best_tier) {
                best = Some((tier, record));
                if tier == 1 {
                    break;
                }
            }
        }
        if let Some((_, record)) = best {
            return Some(record);
        }
    }
    None
}

fn latency_by_bench<'a>(latency: &'a [LatencyRecord], bench_id: &str) -> Option<&'a LatencyRecord> {
    latency.iter().find(|record| record.label == bench_id)
}

pub(super) fn render_quick_summary(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push('\n');
    out.push_str("QUICK SUMMARY (non-specialist)\n");
    out.push_str("═══════════════════════════════════════\n\n");

    let mut has_proxy = false;
    for (display, bench_id, patterns) in QUICK_REPS {
        let Some(record) = find_representative(&report.fidelity, patterns) else {
            continue;
        };
        has_proxy |= record
            .f64_oracle_provenance
            .as_deref()
            .is_some_and(|p| p.starts_with("proxy:"));
        let esr_nam = fmt_metric(&record.esr);
        let esr_nam_colored = match metric_f64(&record.esr) {
            Some(esr) => p.paint_class(esr_color_class(esr), &esr_nam),
            None => esr_nam.clone(),
        };
        let verdict = match metric_f64(&record.esr) {
            Some(esr) => {
                let text = esr_verdict_short(esr);
                p.paint_class(esr_color_class(esr), text)
            }
            None => "N/A".to_string(),
        };
        let esr_f64 = fmt_f64_esr(record);
        let esr_f64_colored = match metric_f64(&record.esr_f64) {
            Some(esr) => p.paint_class(esr_color_class(esr), &esr_f64),
            None => esr_f64.clone(),
        };
        let cpu = match latency_by_bench(&report.latency, bench_id) {
            Some(record) => {
                let pct = budget_pct(record.median_latency_us);
                p.paint_class(cpu_color_class(pct), &format!("{pct:.1}%"))
            }
            None => "N/A".to_string(),
        };

        out.push_str(&format!(
            "  {:<24}  vs NAMCore: {:>10} {:<16}  |  vs Ideal (f64): {:>10}  |  CPU: {} of budget\n",
            display,
            esr_nam_colored,
            pad(&verdict, 16),
            esr_f64_colored,
            cpu,
        ));
    }
    if has_proxy {
        out.push_str("  * = f64 ESR measured on a substitute fixture (proxy), not on this model\n");
    }
    out.push('\n');
    out
}

fn is_redundant_measurement(label: &str) -> bool {
    let t_digit = label.chars().next().is_some_and(|c| c == 'T')
        && label.chars().nth(1).is_some_and(|c| c.is_ascii_digit());
    label.starts_with("Quick ")
        || label.starts_with("Container ")
        || label.starts_with("Container File ")
        || label.starts_with("T-")
        || t_digit
}

fn fidelity_mode(label: &str) -> &str {
    if label.contains(" HQ") { "HQ" } else { "Live" }
}

pub(super) fn render_fidelity_details(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("AUDIO FIDELITY — Technical Details\n");
    out.push_str("═════════════════════════════════════════\n\n");

    if report.fidelity.is_empty() {
        out.push_str(&p.yellow("  (i) No fidelity data available.\n"));
        out.push('\n');
        return out;
    }

    let mut canonicals: Vec<&FidelityRecord> = Vec::new();
    let mut redundants: Vec<&FidelityRecord> = Vec::new();
    for record in &report.fidelity {
        if is_redundant_measurement(&model_label(&record.label)) {
            redundants.push(record);
        } else {
            canonicals.push(record);
        }
    }

    let header_line = || {
        format!(
            "  {:<38} | {:>10} | {:>10} | {:>8} | {:>8} | {:<4}\n",
            "Model", "ESR NAMCore", "ESR f64", "SNR dB", "MR-STFT", "Mode"
        )
    };
    let rule_line = || {
        format!(
            "  {:-<38}-+-{:-<10}-+-{:-<10}-+-{:-<8}-+-{:-<8}-+-{:-<4}\n",
            "", "", "", "", "", ""
        )
    };

    if !canonicals.is_empty() {
        out.push_str("  ── Canonical Fidelity (golden_vectors) ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for record in &canonicals {
            out.push_str(&fidelity_row(record, p));
        }
        out.push('\n');
    }

    if !redundants.is_empty() {
        out.push_str("  ── Additional Coverage (quick_parity, containers, regression gates) ──\n");
        out.push_str(
            "  (i) These measurements validate the same models via alternate entry points.\n",
        );
        out.push_str("       Equivalent rows from the canonical table above.\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for record in &redundants {
            out.push_str(&fidelity_row(record, p));
        }
        out.push('\n');
    }

    out.push_str("  Qualitative legend (ESR audibility bounds):\n");
    out.push_str(&format!(
        "    {} = imperceptible (ESR < 1e-5)\n",
        p.green("green")
    ));
    out.push_str(&format!(
        "    {} = audible only with scientific A/B (ESR < 1e-2)\n",
        p.yellow("yellow")
    ));
    out.push_str(&format!(
        "    {} = audible — needs investigation (ESR >= 1e-1)\n",
        p.red("red")
    ));
    out.push_str("    * = f64 ESR measured on a substitute fixture (proxy), not on this model\n");
    out.push('\n');
    out
}

fn fmt_f64_esr(record: &FidelityRecord) -> String {
    let mut value = fmt_metric(&record.esr_f64);
    if record
        .f64_oracle_provenance
        .as_deref()
        .is_some_and(|p| p.starts_with("proxy:"))
    {
        value.push('*');
    }
    value
}

fn fidelity_row(record: &FidelityRecord, p: &Palette) -> String {
    let esr_nam = fmt_metric(&record.esr);
    let esr_nam_colored = match metric_f64(&record.esr) {
        Some(esr) => p.paint_class(esr_color_class(esr), &esr_nam),
        None => esr_nam,
    };
    let esr_f64 = fmt_f64_esr(record);
    let esr_f64_colored = match metric_f64(&record.esr_f64) {
        Some(esr) => p.paint_class(esr_color_class(esr), &esr_f64),
        None => esr_f64,
    };
    let snr = fmt_snr(&record.snr_db);
    let mrstft = fmt_metric(&record.mrstft);
    let mode = fidelity_mode(&record.label);
    let display_key = truncate(&record.label, 38);
    format!(
        "  {:<38} | {:>10} | {:>10} | {:>8} | {:>8} | {:<4}\n",
        pad(&display_key, 38),
        esr_nam_colored,
        esr_f64_colored,
        snr,
        mrstft,
        mode
    )
}

/// Bench id → human display label (mirror of the bash `BENCH_MODEL_MAP`).
const BENCH_DISPLAY: &[(&str, &str)] = &[
    ("RT_WaveNet_Std_CH16", "WaveNet Standard CH16"),
    ("RT_WaveNet_Feather_CH8", "WaveNet Feather CH8"),
    ("RT_WaveNet_Lite_CH12", "WaveNet Lite CH12"),
    ("RT_WaveNet_Nano_CH4", "WaveNet Nano CH4"),
    ("RT_A2_Full_CH8", "A2 Full CH8"),
    ("RT_A2_Lite_CH3", "A2 Lite CH3"),
    ("RT_LSTM_1x16", "LSTM 1x16"),
    ("RT_LSTM_2x8", "LSTM 2x8"),
    ("RT_Linear_Direct_RF4", "Linear RF=4 Direct"),
    ("RT_Linear_Fft_RF2048", "Linear FFT RF=2048"),
    ("RT_ConvNet", "ConvNet"),
    ("RT_WaveNet_Dyn_Free", "WaveNet Dyn Free"),
    ("RT_LSTM_Dyn_1x7", "LSTM Dyn 1x7"),
    ("RT_A2_Dyn_Gated_CH8", "A2 Dyn Gated CH8"),
    ("RT_A2_Dyn_Blended_CH3", "A2 Dyn Blended CH3"),
    ("RT_A2_Max_CH4", "A2 Max CH4"),
    ("RT_DSP_Resampler_44k1_to_48k", "DSP Resampler 44.1k->48k"),
    ("RT_DSP_Resampler_96k_to_48k", "DSP Resampler 96k->48k"),
    ("RT_DSP_CabSim_IR_Medium", "DSP CabSim IR Medium"),
    ("RT_DSP_Pipeline_Base_NoOS", "DSP Pipeline Base (No OS)"),
    ("RT_DSP_Pipeline_HQ_4xOS", "DSP Pipeline HQ (4x OS)"),
    ("RT_DSP_Pipeline_AdaptiveOn", "DSP Pipeline (Adaptive On)"),
];

/// Median latency (µs) for a bench id, if a matching latency record exists.
fn latency_us(report: &QualityReport, bench_id: &str) -> Option<f64> {
    report
        .latency
        .iter()
        .find(|record| record.label == bench_id)
        .map(|record| record.median_latency_us)
}

pub(super) fn render_performance(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("PERFORMANCE — Block Latency (µs per 64-sample block @ 48kHz)\n");
    out.push_str("══════════════════════════════════════════════════════════\n");
    out.push_str("  RT deadline: 1333 µs (1.33 ms)\n\n");

    if report.performance_not_verified() {
        out.push_str(&p.yellow("  ⚠ NOT_VERIFIED — performance not verified against baseline.\n"));
        out.push_str(&p.yellow(
            "    Performance is not certified in this run; full --check against the quality contract requires verified latency (skipped under --fidelity-only).\n\n",
        ));
        if !report.latency.is_empty() {
            out.push_str(
                &p.yellow("    Raw measurements for reference (not comparable to baseline):\n\n"),
            );
        }
    }

    let mut model_rows: Vec<(&str, Option<f64>)> = Vec::new();
    let mut dsp_rows: Vec<(&str, Option<f64>)> = Vec::new();
    for (bench_id, display) in BENCH_DISPLAY {
        let is_dsp = display.starts_with("DSP ");
        let lat = latency_us(report, bench_id);
        if is_dsp {
            dsp_rows.push((display, lat));
        } else {
            model_rows.push((display, lat));
        }
    }

    let header_line = || {
        format!(
            "  {:<28} | {:>16} | {:>10} | {:>18}\n",
            "Model / Component", "Median Latency", "% Budget", "Headroom"
        )
    };
    let rule_line = || format!("  {:-<28}-+-{:-<16}-+-{:-<10}-+-{:-<18}\n", "", "", "", "");
    let verified = !report.performance_not_verified();
    let perf_row = |display: &str, lat: Option<f64>| {
        let (latency, pct, headroom) = match lat {
            Some(lat) => {
                let pct = budget_pct(lat);
                let headroom = budget_headroom(pct);
                // NOT_VERIFIED is never green (S6 invariant): raw values only.
                let (pct, headroom) = if verified {
                    (
                        p.paint_class(cpu_color_class(pct), &format!("{pct:.1}%")),
                        p.paint_class(cpu_color_class(pct), &format!("{headroom:.1}%")),
                    )
                } else {
                    (format!("{pct:.1}%"), format!("{headroom:.1}%"))
                };
                let lat_str = if lat < 0.1 {
                    format!("{lat:.3} us")
                } else {
                    format!("{lat:.1} us")
                };
                (lat_str, pct, headroom)
            }
            None => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
        };
        format!(
            "  {:<28} | {:>16} | {:>10} | {:>18}\n",
            pad(display, 28),
            latency,
            pct,
            headroom
        )
    };

    let has_any = !model_rows.is_empty() || !dsp_rows.is_empty();
    if !has_any {
        out.push_str(&p.yellow("  (i) No performance data available.\n\n"));
        return out;
    }

    if !model_rows.is_empty() {
        out.push_str("  ── Model Inference Core ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for (display, lat) in &model_rows {
            out.push_str(&perf_row(display, *lat));
        }
        out.push('\n');
    }
    if !dsp_rows.is_empty() {
        out.push_str("  ── DSP Infrastructure ──\n\n");
        out.push_str(&header_line());
        out.push_str(&rule_line());
        for (display, lat) in &dsp_rows {
            out.push_str(&perf_row(display, *lat));
        }
        out.push('\n');
        out.push_str(
            "  (i) DSP micro-kernels (Resampler, CabSim) run 64 blocks per Criterion\n\
             \x20     iteration and are normalized here to µs per 64-sample block; Pipeline\n\
             \x20     rows already process one 64-sample block per iteration.\n\n",
        );
    }

    out.push_str("  (i) Headroom > 50%:  2x oversampling usually safe without xruns\n");
    out.push_str("  (i) Headroom > 75%:  4x oversampling usually safe without xruns\n");
    out.push_str("  (i) Headroom < 25%:  xrun risk with a 64-sample buffer\n\n");
    out
}

pub(super) fn render_isa_parity(report: &QualityReport, p: &Palette) -> String {
    // The QA report must distinguish *self-consistency* (the local
    // dashboard phase `isa_self_consistency`, AVX2 vs AVX2, MSE=0 tautology)
    // from *real cross-ISA parity* (the remote gate `remote-simd-gate.sh`,
    // tracked as phase `isa_parity_cross_isa`). The local runner declares the
    // cross-ISA matrix as a SKIP_CAPABILITY gap instead of letting an
    // `isa_parity: PASS` line imply inter-ISA comparison.
    let mut out = String::new();
    out.push_str("ISA SELF-CONSISTENCY / CROSS-ISA\n");
    out.push_str("══════════════════════════════════\n\n");

    if report.isa.is_empty() {
        // No per-model ISA rows: reflect what the phases actually declared.
        let self_phase = phases::ISA_SELF_CONSISTENCY_PHASE;
        let cross_phase = phases::ISA_PARITY_CROSS_ISA_PHASE;
        if report.phase_status(self_phase) == "PASS" {
            out.push_str(
                "  (i) isa_self_consistency passed but no per-model metric rows recorded.\n",
            );
        } else if report.phase_status(self_phase) == "FAIL" {
            out.push_str(&p.red("  FAIL isa_self_consistency (AVX2 determinism violated).\n\n"));
            return out;
        } else {
            out.push_str(&p.yellow("  (i) isa_self_consistency not run.\n"));
        }
        match report.phase_status(cross_phase) {
            "PASS" => {
                out.push_str(
                    "  ok cross-ISA parity (AVX2 vs AVX-512) passed on the remote gate.\n",
                );
            }
            "SKIP_CAPABILITY" => {
                out.push_str(&p.yellow(&format!(
                    "  (i) cross-ISA parity NOT executed here — declared gap \
                     `{}` (multi-target/remote gate only).\n",
                    phases::CROSS_ISA_GAP_REASON
                )));
            }
            _ => {
                out.push_str(&p.yellow(
                    "  (i) cross-ISA parity not covered — run the remote SIMD gate \
                     (utils/remote-simd-gate.sh) for the real inter-ISA matrix.\n",
                ));
            }
        }
        out.push('\n');
        return out;
    }

    let self_consistency: Vec<&IsaRow> = report
        .isa
        .iter()
        .filter(|row| row.ref_isa == row.test_isa)
        .collect();
    let cross_isa: Vec<&IsaRow> = report
        .isa
        .iter()
        .filter(|row| row.ref_isa != row.test_isa)
        .collect();
    let cross_isa_pass = cross_isa
        .iter()
        .filter(|row| match row.esr.as_ref().and_then(metric_f64) {
            Some(esr) => esr < 1e-8,
            None => false,
        })
        .count();

    if cross_isa_pass == cross_isa.len() && !cross_isa.is_empty() {
        out.push_str(&p.green("  AVX2 vs AVX-512: within ESR budget (near-lossless)\n"));
    } else if !cross_isa.is_empty() {
        out.push_str(&p.yellow(&format!(
            "  AVX2 vs AVX-512: divergent on {}/{} models\n",
            cross_isa.len() - cross_isa_pass,
            cross_isa.len()
        )));
    } else {
        out.push_str(&p.yellow(
            "  cross-ISA parity: no data — declared gap `cross_isa_matrix_requires_avx512_remote`\n",
        ));
    }

    out.push_str(&format!(
        "  Self-consistency checks: {} executed\n",
        self_consistency.len()
    ));

    if !self_consistency.is_empty() {
        out.push_str("  Self-consistency details (AVX2 vs AVX2):\n");
        for row in &self_consistency {
            let mse_text = fmt_metric(&row.mse);
            let pass = match metric_f64(&row.mse) {
                Some(mse) => mse == 0.0,
                None => false,
            };
            let mark = if pass { p.green("ok") } else { p.yellow("⚠") };
            out.push_str(&format!(
                "    {} | MSE={} | {}\n",
                row.label, mse_text, mark
            ));
        }
        out.push('\n');
    }

    if !cross_isa.is_empty() {
        out.push_str("  Cross-ISA details:\n");
        for row in &cross_isa {
            let esr_text = row
                .esr
                .as_ref()
                .map(fmt_metric)
                .unwrap_or_else(|| "N/A".to_string());
            let pass = match row.esr.as_ref().and_then(metric_f64) {
                Some(esr) => esr < 1e-8,
                None => false,
            };
            let mark = if pass { p.green("ok") } else { p.yellow("⚠") };
            out.push_str(&format!("    {}  ESR={}  {}\n", row.label, esr_text, mark));
        }
        out.push('\n');
    }
    out
}

fn render_activation_table_body<'a>(
    title: &str,
    rows: impl IntoIterator<Item = &'a ActivationRow>,
    p: &Palette,
) -> String {
    let mut out = String::new();
    out.push_str(title);
    out.push('\n');
    out.push_str(&"═".repeat(title.chars().count()));
    out.push_str("\n\n");

    let rows: Vec<&'a ActivationRow> = rows.into_iter().collect();
    if rows.is_empty() {
        out.push_str(&p.yellow("  (i) No activation-precision results available.\n\n"));
        return out;
    }

    out.push_str(&format!(
        "  {:<20} | {:>14} | {:>14} | {:>10}\n",
        "Model", "Fast(Pade)", "Standard(exact)", "Δ SNR"
    ));
    out.push_str(&format!(
        "  {:-<20}-+-{:-<14}-+-{:-<14}-+-{:-<10}\n",
        "", "", "", ""
    ));

    let mut total: f64 = 0.0;
    let mut count: u64 = 0;
    for row in rows {
        let fast = fmt_snr(&row.snr_fast_db);
        let exact = fmt_snr(&row.snr_exact_db);
        let gain = fmt_snr(&row.gain_db);
        let gain_colored = match metric_f64(&row.gain_db) {
            Some(v) if v >= 3.0 => gain.clone(),
            Some(_) => p.yellow(&gain),
            None => gain.clone(),
        };
        if let Some(v) = metric_f64(&row.gain_db) {
            total += v;
            count += 1;
        }
        out.push_str(&format!(
            "  {:<20} | {:>14} | {:>14} | {:>10}\n",
            pad(&truncate(&row.model, 20), 20),
            format!("{fast} dB"),
            format!("{exact} dB"),
            gain_colored
        ));
    }
    if count > 0 {
        out.push_str(&format!(
            "  Mean SNR gain with Standard(exact): +{:.1} dB (over {count} LSTM model(s))\n",
            total / count as f64
        ));
    }
    out.push('\n');
    out
}

pub(super) fn render_activation_precision(report: &QualityReport, p: &Palette) -> String {
    if report.activation.is_empty() {
        return render_activation_table_body("ACTIVATION PRECISION", &[], p);
    }

    let has_explicit_oracle = report.activation.iter().any(|r| r.oracle.is_some());
    let mut seen = std::collections::HashSet::new();
    let has_duplicate_model = report.activation.iter().any(|r| !seen.insert(&r.model));

    if !has_explicit_oracle && !has_duplicate_model {
        // Single population (or backward-compatible single table)
        return render_activation_table_body("ACTIVATION PRECISION", &report.activation, p);
    }

    let mut namcore_rows = Vec::new();
    let mut f64_rows = Vec::new();

    if has_explicit_oracle {
        for row in &report.activation {
            match row.oracle.as_deref() {
                Some("f64") => f64_rows.push(row),
                _ => namcore_rows.push(row),
            }
        }
    } else {
        // Partition by first-seen vs repeated model name
        let mut first_seen = std::collections::HashSet::new();
        for row in &report.activation {
            if first_seen.insert(&row.model) {
                namcore_rows.push(row);
            } else {
                f64_rows.push(row);
            }
        }
    }

    let has_namcore = !namcore_rows.is_empty();
    let has_f64 = !f64_rows.is_empty();

    let mut out = String::new();
    if has_namcore {
        let title = if !has_f64 {
            "ACTIVATION PRECISION"
        } else {
            "ACTIVATION PRECISION — vs NAMCore (Warm, Goldens)"
        };
        out.push_str(&render_activation_table_body(title, namcore_rows, p));
    }
    if has_f64 {
        let title = if !has_namcore {
            "ACTIVATION PRECISION"
        } else {
            "ACTIVATION PRECISION — vs f64 Reference Oracle (Cold-Start / Stress)"
        };
        out.push_str(&render_activation_table_body(title, f64_rows, p));
    }
    out
}

pub(super) fn render_f64_decomposition(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    if report.f64_decomp.is_empty() {
        return out;
    }
    out.push_str("F64 ORACLE — Error Source Decomposition\n");
    out.push_str("══════════════════════════════════════════════\n\n");
    out.push_str("  (i) These measurements are cold-start (256 samples, NO prewarm) — NOT\n");
    out.push_str("      comparable to the 'vs Ideal (f64)' values in the fidelity table\n");
    out.push_str("      above (measured with 24k-sample warmup).\n");
    out.push_str("      For WaveNet/A2 the receptive field can exceed the 256-sample\n");
    out.push_str("      window, so the measured total may include transient buffer fill-in;\n");
    out.push_str("      for other models the measured total is close to the real f32-vs-f64\n");
    out.push_str("      floor while `combined` is inflated by simulated F16C quantization,\n");
    out.push_str("      which the production f32 path does not execute.\n");
    out.push_str("      See docs/perceptual_validation.md#decomposition-cold-start.\n\n");

    for block in &report.f64_decomp {
        out.push_str(&format!("  {}:\n", block.label));
        out.push_str(&format!(
            "    ESR(f32 vs f64 oracle): {}\n",
            fmt_metric(&block.esr_f32_vs_f64)
        ));
        for (name, value) in [
            ("quant F16C", &block.esr_quant_f16c),
            ("quant BF16", &block.esr_quant_bf16),
            ("activation", &block.esr_activation),
            ("accumulation", &block.esr_accumulation),
            ("combined (F16C+Padé+F32)", &block.esr_combined),
        ] {
            if let Some(value) = value {
                out.push_str(&format!("    {name}: {}\n", fmt_metric(value)));
            }
        }

        let total = metric_f64(&block.esr_f32_vs_f64);
        let combined = block.esr_combined.as_ref().and_then(metric_f64);
        if let (Some(total), Some(combined)) = (total, combined)
            && combined != 0.0
            && total != 0.0
        {
            let ratio = (total / combined).abs().max((combined / total).abs());
            if ratio > 10.0 {
                let arch = block.architecture.to_ascii_lowercase();
                let label = block.label.to_ascii_lowercase();
                let is_transient_architecture = arch.contains("wavenet")
                    || arch.contains("a2")
                    || arch.contains("convnet")
                    || label.contains("wavenet")
                    || label.contains("a2")
                    || label.contains("convnet");

                let quant_f16c = block.esr_quant_f16c.as_ref().and_then(metric_f64);
                let quant_dominates_combined =
                    quant_f16c.map(|q| q >= 0.5 * combined).unwrap_or(false);

                if combined > total && quant_dominates_combined {
                    // `combined` includes the simulated F16C weight quantization,
                    // which the production f32 path never executes. The measured
                    // total is the real f32-vs-f64 error, so a large
                    // combined/total ratio here is an artifact of the simulation,
                    // not a cold-start transient.
                    out.push_str(&format!(
                        "    Note: combined/total ≈ {ratio:.0}x — combined is dominated by the simulated F16C weight quantization (absent from the production f32 path); the measured total is the real f32-vs-f64 error.\n"
                    ));
                } else if total > combined && is_transient_architecture {
                    out.push_str(&format!(
                        "    Rule 5 notice: total/combined ≈ {ratio:.0}x (expected cold-start buffer fill-in transient: receptive field exceeds the 256-sample, prewarm=0 window).\n"
                    ));
                    out.push_str(
                        "      Confirm against the paired warmup measurement before treating this as a precision floor.\n",
                    );
                } else {
                    let ratio_label = if combined > total {
                        "combined/total"
                    } else {
                        "total/combined"
                    };
                    out.push_str(&p.yellow(&format!(
                        "    Rule 5 (Σ sources ≈ total, within 10x) violated: {ratio_label} ≈ {ratio:.0}x.\n"
                    )));
                    out.push_str(&p.yellow(
                        "      Sources and total disagree in a regime that neither the simulated-quantization nor the cold-start transient explains.\n",
                    ));
                    out.push_str(&p.yellow(
                        "      Do not treat this number as a calibrated precision floor without paired prewarm measurement.\n",
                    ));
                }
            }
        }
        out.push('\n');
    }
    out
}

pub(super) fn render_spectral_summary(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("SPECTRAL FIDELITY\n");
    out.push_str("═════════════════\n\n");
    let count = report.coverage.map(|c| c.spectral_baselines).unwrap_or(0);
    if count > 0 {
        out.push_str(&p.green(&format!(
            "  ok {count} model(s) with spectral metrics inside baseline.\n"
        )));
    } else if report.phase_status("spectral_fidelity") == "PASS" {
        // The phase receipt passed; "not covered" would be a lie (P0.T3).
        out.push_str(
            "  ok spectral_fidelity phase passed; no per-model baseline counts in this report.\n",
        );
    } else {
        out.push_str(
            &p.yellow("  (i) Not covered in quick mode — run tests-long for full verification.\n"),
        );
    }
    out.push('\n');
    out
}

pub(super) fn render_coverage_matrix(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("COVERAGE MATRIX BY AXIS (Governance)\n");
    out.push_str("════════════════════════════════════════════\n\n");

    let Some(coverage) = report.coverage else {
        out.push_str(&p.yellow("  (i) No coverage data available.\n\n"));
        return out;
    };

    let mut covered = 0u64;
    out.push_str(&format!(
        "  {:<28} | {:>10} | {:<20}\n",
        "Axis", "Records", "Coverage"
    ));
    out.push_str(&format!("  {:-<28}-+-{:-<10}-+-{:-<20}\n", "", "", ""));

    for (axis, records) in [
        ("NAMCore Parity", coverage.namcore_parity),
        ("f64 Oracle Fidelity", coverage.f64_oracle),
        ("ISA Optimizations", coverage.isa_optimizations),
        ("Spectral Baselines", coverage.spectral_baselines),
        ("RT Performance", coverage.rt_performance),
    ] {
        let status = if records > 0 {
            covered += 1;
            p.green("covered")
        } else {
            p.yellow("not covered")
        };
        out.push_str(&format!(
            "  {:<28} | {:>10} | {:<20}\n",
            pad(axis, 28),
            records,
            status
        ));
    }
    out.push('\n');
    out.push_str(&format!("  Coverage: {covered}/5 axes covered\n\n"));

    if let Some(counts) = report.test_counts {
        out.push_str("  Phases in the receipt:\n");
        out.push_str(&format!(
            "    passed:          {}\n",
            p.green(&counts.passed.to_string())
        ));
        out.push_str(&format!(
            "    failed:          {}\n",
            p.red(&counts.failed.to_string())
        ));
        out.push_str(&format!(
            "    skip_capability: {}\n",
            p.yellow(&counts.skip_capability.to_string())
        ));
        out.push_str(&format!("    ignored:         {}\n", counts.ignored));
        out.push_str(&format!("    filtered:        {}\n\n", counts.filtered));
    }
    out
}

pub(super) fn render_footer(report: &QualityReport, p: &Palette) -> String {
    let mut out = String::new();
    out.push_str("───────────────────────────────────────────────────────────────\n");

    if report.performance_not_verified() {
        out.push_str(
            &p.yellow("  Performance NOT_VERIFIED — fidelity gates validated independently.\n\n"),
        );
    }

    let phase_failures = report.phases.iter().any(|phase| phase.status == "FAIL");
    if phase_failures {
        out.push_str(&p.red("  One or more dashboard phases failed.\n\n"));
    }
    out
}
