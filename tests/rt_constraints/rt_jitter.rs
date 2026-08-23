// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  RT jitter characterization — environmental telemetry, NOT a deadline gate.
//
//  Characterizes tail latency under concurrent CPU load to quantify how
//  scheduler contention affects the DSP thread. This is diagnostic telemetry:
//  it does NOT assert that deadlines are met under stress — that is the
//  exclusive job of `rt_deadline.rs`.
//
//  The baseline test (no stress) reports p50/p90/p99/p99.9/exact_max latency and
//  counts deadline violations. It performs an `rt_preflight` environment check
//  and emits `INCONCLUSIVE` when preconditions (CPU pinning, performance
//  governor, low background load) are not satisfied.
//
//  Stress tests (1 thread, 2 threads, saturation) report delta against the
//  baseline of the same execution — they characterize resilience, they do NOT
//  hard-assert zero violations.
//
//  ## Load scenarios (T3.1 / F-ROB-05)
//
//  - `stress-1`: exactly 1 CPU-burn worker during the whole measurement.
//  - `stress-2`: exactly 2 CPU-burn workers during the whole measurement.
//  - `saturate-N`: N = max(1, affinity_cores - 1) workers — saturation keeps a
//    single core for the DSP thread whenever the host provides more than one.
//
//  Invariant: a scenario labeled `stress-N` MUST generate active load in
//  exactly N concurrent worker threads during the entire DSP measurement. When
//  the process affinity mask exposes fewer than N cores, the scenario is
//  declared `SKIP_CAPABILITY` (typed marker) — it never runs with zero load
//  under a misleading label.
//
//  ## Running
//
//  ```sh
//  cargo test --release --test rt_jitter -- --ignored --nocapture
//  ```
//
//  Marked `#[ignore]` — runs during `tests-long.sh` Phase 5.

use super::common;
use common::rt_helpers::{self, RtPreflightStatus};
use common::*;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use neural_amp_modeler_rs::dsp::telemetry::LatencyHistogram;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;

const RT_DEADLINE_US: u64 = 1330;
const BLOCK_SIZE: usize = 64;
const MEASURE_BLOCKS: usize = 2048;

#[derive(Debug, Default)]
struct JitterStats {
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    p999_us: u64,
    exact_max_us: u64,
    violations: u64,
    total_blocks: usize,
}

impl JitterStats {
    fn violation_pct(&self) -> f64 {
        (self.violations as f64 / self.total_blocks as f64) * 100.0
    }

    /// Number of CPU cores the process can actually use, given its affinity
    /// mask (e.g. `taskset -c N`). `1` means "single core available by
    /// affinity" — the T3.1 rollback condition for multi-worker stress.
    fn affinity_cores() -> usize {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn stress_worker(running: Arc<AtomicBool>) {
    let mut x: f64 = 0.0;
    while running.load(Ordering::Relaxed) {
        for _ in 0..256 {
            x = (x + 0.123456789).fract();
            let x2 = x * x;
            let sin_approx = x * (1.0 - x2 * (1.0 / 6.0 - x2 / 120.0));
            let cos_approx = 1.0 - x2 * (0.5 - x2 * (1.0 / 24.0 - x2 / 720.0));
            std::hint::black_box(sin_approx + cos_approx);
        }
    }
}

fn load_and_prewarm(filename: &str) -> Option<neural_amp_modeler_rs::models::StaticModel> {
    let path = model_path(filename);
    if !path.exists() {
        eprintln!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:{filename}\"");
        return None;
    }
    let json_data = fs::read_to_string(&path).ok()?;
    let model_data = parse_nam_json(&json_data).ok()?;
    let mut model = build_model(&model_data).ok()?;
    model.prewarm(2048);
    Some(*model)
}

fn measure_latency(
    label: &str,
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    stress_threads: usize,
) -> JitterStats {
    if stress_threads == 0 {
        println!("[{label}] Baseline — {MEASURE_BLOCKS} blocks, no stress");
    } else {
        println!("[{label}] Stress: {stress_threads} worker(s), {MEASURE_BLOCKS} blocks");
    }
    println!(
        "[{label}] Core affinity: {} core(s) available via affinity mask",
        JitterStats::affinity_cores()
    );
    println!(
        "[{label}] Effective load: {stress_threads} CPU-burn worker(s) active during measurement"
    );

    let running = Arc::new(AtomicBool::new(true));
    let mut workers = Vec::with_capacity(stress_threads);

    for _ in 0..stress_threads {
        let r = running.clone();
        workers.push(thread::spawn(move || stress_worker(r)));
    }

    if stress_threads > 0 {
        thread::sleep(std::time::Duration::from_millis(100));
    }

    let input = generate_sine_440hz(BLOCK_SIZE);
    let mut output = vec![0.0f32; BLOCK_SIZE];
    let hist = LatencyHistogram::new();
    let mut violations: u64 = 0;

    for _ in 0..MEASURE_BLOCKS {
        let start = std::time::Instant::now();
        model.process(&input, &mut output);
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        hist.record(elapsed_ns);

        if elapsed_ns > RT_DEADLINE_US * 1000 {
            violations += 1;
        }

        assert!(
            output.iter().all(|s| s.is_finite()),
            "[{label}] Non-finite output under stress"
        );
    }

    running.store(false, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }

    let stats = JitterStats {
        p50_us: hist.get_percentile(0.50) / 1000,
        p90_us: hist.get_percentile(0.90) / 1000,
        p99_us: hist.get_percentile(0.99) / 1000,
        p999_us: hist.get_percentile(0.999) / 1000,
        exact_max_us: hist.get_exact_max() / 1000,
        violations,
        total_blocks: MEASURE_BLOCKS,
    };

    println!(
        "[{label}] P50={}μs  P90={}μs  P99={}μs  P99.9={}μs  exact_max={}μs  \
         violations={}/{} ({:.2}%)",
        stats.p50_us,
        stats.p90_us,
        stats.p99_us,
        stats.p999_us,
        stats.exact_max_us,
        stats.violations,
        stats.total_blocks,
        stats.violation_pct(),
    );

    stats
}

fn delta_pct(current: u64, baseline: u64) -> String {
    if baseline == 0 {
        return "N/A".to_string();
    }
    let delta = ((current as f64 - baseline as f64) / baseline as f64) * 100.0;
    format!("{delta:+.1}%")
}

fn emit_receipt(label: &str, stress_workers: usize, stats: &JitterStats) {
    println!(
        "[RECEIPT] status=PASS label={label} stress_workers={stress_workers} \
         p50_us={} p90_us={} p99_us={} p99.9_us={} exact_max_us={} violations={}/{}",
        stats.p50_us,
        stats.p90_us,
        stats.p99_us,
        stats.p999_us,
        stats.exact_max_us,
        stats.violations,
        stats.total_blocks,
    );
}

fn emit_skip_receipt(label: &str, stress_workers: usize, reason: &str) {
    println!(
        "[RECEIPT] status=SKIP_CAPABILITY label={label} stress_workers={stress_workers} \
         reason=\"{reason}\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A \
         exact_max_us=N/A violations=N/A"
    );
}

/// Runs one jitter characterization scenario at the given stress level.
///
/// T3.1: the stress count is explicit — a scenario labeled `stress-N` spawns
/// exactly N CPU-burn workers for the whole measurement. When the process
/// affinity exposes fewer cores than the requested worker count, the scenario
/// is declared `SKIP_CAPABILITY` (typed marker) instead of running with
/// misleading zero/partial load.
fn run_jitter_characterization(
    label: &str,
    model: &mut neural_amp_modeler_rs::models::StaticModel,
    baseline: Option<&JitterStats>,
    stress_threads: usize,
) -> Option<JitterStats> {
    if let Some(b) = baseline {
        println!("\n[{label}] CHARACTERIZATION — comparing against baseline of same execution");
        println!(
            "[{label}] Baseline violations={}/{} ({:.2}%)",
            b.violations,
            b.total_blocks,
            b.violation_pct()
        );
        println!(
            "[{label}] Baseline P50={}μs  P90={}μs  P99={}μs  P99.9={}μs  exact_max={}μs",
            b.p50_us, b.p90_us, b.p99_us, b.p999_us, b.exact_max_us
        );
    } else {
        println!("\n[{label}] CHARACTERIZATION — environmental jitter telemetry");
        println!("[{label}] No baseline to compare against (first measurement)");
    }

    let affinity_cores = JitterStats::affinity_cores();
    if stress_threads > affinity_cores {
        println!("[STATUS] SKIP_CAPABILITY reason=\"Single core affinity\"");
        println!(
            "[{label}] Requested {stress_threads} stress worker(s) but only \
             {affinity_cores} core(s) available via affinity mask — skipping"
        );
        emit_skip_receipt(label, stress_threads, "Single core affinity");
        return None;
    }

    let stats = measure_latency(label, model, stress_threads);

    if let Some(b) = baseline {
        println!(
            "[{label}] Δ P50: {}  Δ P90: {}  Δ P99: {}  Δ P99.9: {}  Δ max: {}  \
             Δ violations: {}→{}",
            delta_pct(stats.p50_us, b.p50_us),
            delta_pct(stats.p90_us, b.p90_us),
            delta_pct(stats.p99_us, b.p99_us),
            delta_pct(stats.p999_us, b.p999_us),
            delta_pct(stats.exact_max_us, b.exact_max_us),
            b.violations,
            stats.violations,
        );
    }

    emit_receipt(label, stress_threads, &stats);
    Some(stats)
}

// ── Jitter characterization — baseline (no stress) ─────────────────────────

#[test]
#[ignore]
fn test_jitter_characterization_baseline_wavenet_standard() {
    let preflight = rt_helpers::rt_preflight();
    rt_helpers::print_preflight(&preflight);

    if preflight.status != RtPreflightStatus::Pass {
        println!("[STATUS] INCONCLUSIVE — skipping jitter characterization");
        println!("  Environment preconditions not met; jitter telemetry would be");
        println!("  invalid without CPU isolation and performance governor.");
        println!(
            "[RECEIPT] status=INCONCLUSIVE reason=\"preflight_failed\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A exact_max_us=N/A violations=N/A"
        );
        return;
    }

    println!("[STATUS] PASS — preflight OK");
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        let baseline = measure_latency("WaveNet-Std-char-baseline", &mut model, 0);

        // Emit typed receipt with percentiles as completion marker
        println!(
            "[RECEIPT] status=PASS label=WaveNet-Std-char-baseline stress_workers=0 \
             p50_us={} p90_us={} p99_us={} p99.9_us={} exact_max_us={} violations={}/{}",
            baseline.p50_us,
            baseline.p90_us,
            baseline.p99_us,
            baseline.p999_us,
            baseline.exact_max_us,
            baseline.violations,
            baseline.total_blocks,
        );

        // Run stress characterizations with delta against this baseline.
        // T3.1: stress-1 and stress-2 carry 1 and 2 real CPU-burn workers.
        run_jitter_characterization("WaveNet-Std-char-stress-1", &mut model, Some(&baseline), 1);
        run_jitter_characterization("WaveNet-Std-char-stress-2", &mut model, Some(&baseline), 2);

        // Saturation: resilience characterization, no hard assertion on zero
        // violations. Kept isolated and dynamic: one worker per core minus one
        // (the DSP thread keeps its own core whenever the host allows).
        let num_cpus = JitterStats::affinity_cores();
        let sat_threads = num_cpus.saturating_sub(1).max(1);
        println!(
            "\n[WaveNet-Std-char-saturate-{sat_threads}] CHARACTERIZATION — saturation resilience"
        );
        println!(
            "[WaveNet-Std-char-saturate-{sat_threads}] Baseline violations={}/{} ({:.2}%)",
            baseline.violations,
            baseline.total_blocks,
            baseline.violation_pct()
        );
        let sat_stats = run_jitter_characterization(
            "WaveNet-Std-char-saturate",
            &mut model,
            Some(&baseline),
            sat_threads,
        );
        if let Some(sat_stats) = sat_stats {
            println!(
                "[WaveNet-Std-char-saturate] Δ violations: {}→{}  \
                 Δ P99: {}  Δ max: {}",
                baseline.violations,
                sat_stats.violations,
                delta_pct(sat_stats.p99_us, baseline.p99_us),
                delta_pct(sat_stats.exact_max_us, baseline.exact_max_us),
            );
            println!(
                "[WaveNet-Std-char-saturate] Violations under saturation — \
                 environmental characterization, NOT a deadline gate."
            );
        }
    } else {
        println!("[STATUS] SKIP_CAPABILITY reason=\"model_not_found:BossWN-standard.nam\"");
        println!(
            "[RECEIPT] status=SKIP_CAPABILITY reason=\"model_not_found\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A exact_max_us=N/A violations=N/A"
        );
    }
}
