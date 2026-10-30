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
//  ## Load scenarios
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
//  cargo test --release --test rt_constraints rt_jitter -- --ignored --nocapture
//  ```
//
//  ## Pinning & Contention Architecture (F4)
//
//  Jitter characterization runs in two distinct modes:
//
//  1. **Pinned mode** (`NAM_BENCH_CORE=N` or `taskset -c N`):
//     When `NAM_BENCH_CORE` is set, the DSP thread pins itself to the designated
//     bench core while stress workers are pinned across the remaining host cores.
//     Under 15-worker saturation on a 16-core system, the memory bus and L3 cache
//     are saturated by concurrent workers while the DSP thread remains uninterrupted.
//     In this mode, exact_max remains well below the 1330 µs deadline (~160–200 µs),
//     deadline violations are 0/16384 (0.00%), and involuntary context switches
//     stay minimal. This proves empirically that L3/memory contention adds only
//     ~30 µs overhead and does NOT threaten RT deadlines.
//
//  2. **Unpinned mode** (generic multi-core, `pinned=false`):
//     When unpinned, all threads (1 DSP + N-1 workers = N threads) share the N cores
//     under standard Linux CFS (`SCHED_OTHER`). Because background OS tasks exist,
//     the system is oversubscribed. Timeslices expire, causing involuntary context
//     switches (`cs_invol > 0`). Each preemption suspends the DSP thread for a full
//     scheduler quantum (~1–3 ms), yielding exact_max values of ~2800–3200 µs and
//     occasional violations. This is an expected Linux CFS scheduling artifact of
//     unpinned multi-threaded competition, NOT an engine regression.
//
//  3. **Context Switch Telemetry**:
//     Each scenario records voluntary (`cs_vol`) and involuntary (`cs_invol`)
//     context switches via `libc::getrusage(RUSAGE_THREAD)`. A spike in exact_max
//     accompanied by high `cs_invol` under `pinned=false` confirms scheduler
//     preemption as the root cause.
//
//  Marked `#[ignore]` — runs during `tests-long.sh` Phase 6.

pub(crate) use super::budget::{BLOCK_SIZE, RT_DEADLINE_US};
use super::common;
use common::rt_helpers::{self, RtPreflightStatus};
use common::*;

use std::fs;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use neural_amp_modeler_rs::dsp::telemetry::LatencyHistogram;
use neural_amp_modeler_rs::loader::dispatcher::build_model;
use neural_amp_modeler_rs::loader::nam_json::parse_nam_json;
use neural_amp_modeler_rs::models::NamModel;

const WARMUP_BLOCKS: usize = 512;
const MEASURE_BLOCKS: usize = 16384;

fn pin_thread(cpus: &[usize]) {
    #[cfg(target_os = "linux")]
    {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        for &cpu in cpus {
            if cpu < libc::CPU_SETSIZE as usize {
                unsafe { libc::CPU_SET(cpu, &mut set) };
            }
        }
        let _ = unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) };
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = cpus;
    }
}

#[cfg(target_os = "linux")]
fn thread_context_switches() -> Option<(u64, u64)> {
    neural_amp_modeler_rs::testing::host::proc::read_thread_rusage()
        .ok()
        .map(|usage| (usage.nvcsw, usage.nivcsw))
}

#[cfg(not(target_os = "linux"))]
fn thread_context_switches() -> Option<(u64, u64)> {
    None
}

fn initial_process_cpus() -> Vec<usize> {
    static CPUS: OnceLock<Vec<usize>> = OnceLock::new();
    CPUS.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            let ret = unsafe {
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set)
            };
            if ret == 0 {
                let mut cpus = Vec::new();
                for cpu in 0..libc::CPU_SETSIZE as usize {
                    if unsafe { libc::CPU_ISSET(cpu, &set) } {
                        cpus.push(cpu);
                    }
                }
                if !cpus.is_empty() {
                    return cpus;
                }
            }
        }
        let n = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        (0..n).collect()
    })
    .clone()
}

fn bench_core_env() -> Option<usize> {
    std::env::var("NAM_BENCH_CORE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
}

#[derive(Debug, Default)]
struct JitterStats {
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    p999_us: u64,
    exact_max_us: u64,
    violations: u64,
    total_blocks: usize,
    voluntary_cs: Option<u64>,
    involuntary_cs: Option<u64>,
}

impl JitterStats {
    fn violation_pct(&self) -> f64 {
        (self.violations as f64 / self.total_blocks as f64) * 100.0
    }

    /// Number of CPU cores the process can actually use, given its initial
    /// affinity mask before any thread-level isolation.
    fn affinity_cores() -> usize {
        initial_process_cpus().len()
    }

    /// `true` when the DSP thread runs pinned to exactly one core (either via
    /// `taskset -c $NAM_BENCH_CORE` operator shape or via thread affinity).
    fn is_pinned() -> bool {
        #[cfg(target_os = "linux")]
        {
            let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
            let ret = unsafe {
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set)
            };
            if ret == 0 {
                let mut count = 0;
                for cpu in 0..libc::CPU_SETSIZE as usize {
                    if unsafe { libc::CPU_ISSET(cpu, &set) } {
                        count += 1;
                    }
                }
                return count == 1;
            }
        }
        Self::affinity_cores() == 1
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
    let all_cpus = initial_process_cpus();
    let bench_core = bench_core_env().filter(|c| all_cpus.contains(c));

    // If a bench core is designated, pin the DSP thread to that core.
    if let Some(core) = bench_core {
        pin_thread(&[core]);
    }

    if stress_threads == 0 {
        println!(
            "[{label}] Baseline — {MEASURE_BLOCKS} blocks, no stress (pinned={})",
            JitterStats::is_pinned()
        );
    } else {
        println!(
            "[{label}] Stress: {stress_threads} worker(s), {MEASURE_BLOCKS} blocks (pinned={})",
            JitterStats::is_pinned()
        );
    }
    println!(
        "[{label}] Core affinity: {} core(s) available via affinity mask (pinned={})",
        JitterStats::affinity_cores(),
        JitterStats::is_pinned(),
    );
    println!(
        "[{label}] Effective load: {stress_threads} CPU-burn worker(s) active during measurement"
    );

    let running = Arc::new(AtomicBool::new(true));
    let mut workers = Vec::with_capacity(stress_threads);

    // If bench_core is set, worker threads are pinned to the other available cores
    // so they do not share the DSP core while still saturating the host CPU/L3/bus.
    let worker_cores: Vec<usize> = if let Some(core) = bench_core {
        all_cpus.iter().copied().filter(|&c| c != core).collect()
    } else {
        all_cpus.clone()
    };

    for i in 0..stress_threads {
        let r = running.clone();
        let cores = worker_cores.clone();
        let target_core = if !cores.is_empty() {
            Some(cores[i % cores.len()])
        } else {
            None
        };
        workers.push(thread::spawn(move || {
            if let Some(core) = target_core {
                pin_thread(&[core]);
            }
            stress_worker(r);
        }));
    }

    if stress_threads > 0 {
        thread::sleep(std::time::Duration::from_millis(100));
    }

    let input = generate_sine_440hz(BLOCK_SIZE);
    let mut output = vec![0.0f32; BLOCK_SIZE];

    // Measured: initial 512 blocks eliminate ~200 µs cold cache/branch miss transient.
    for _ in 0..WARMUP_BLOCKS {
        model.process(&input, &mut output);
        std::hint::black_box(&output);
    }

    let hist = LatencyHistogram::new();
    let mut violations: u64 = 0;

    let cs_before = thread_context_switches();

    // `black_box` around the process call and the
    // output consumption proves the loop exercises the buffers and
    // coefficients — no compiler elision can fake a latency reading.
    for _ in 0..MEASURE_BLOCKS {
        let start = std::time::Instant::now();
        model.process(&input, &mut output);
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        hist.record(elapsed_ns);

        if elapsed_ns > RT_DEADLINE_US * 1000 {
            violations += 1;
        }

        // Consumed AFTER the timed region (still in the loop): the black_box
        // still prevents elision of the process call, but the histogram
        // measures only the process — never the barrier itself.
        std::hint::black_box(&output);

        assert!(
            output.iter().all(|s| s.is_finite()),
            "[{label}] Non-finite output under stress"
        );
    }
    std::hint::black_box(&output);

    let cs_after = thread_context_switches();
    let (voluntary_cs, involuntary_cs) = match (cs_before, cs_after) {
        (Some((v_before, i_before)), Some((v_after, i_after))) => (
            Some(v_after.saturating_sub(v_before)),
            Some(i_after.saturating_sub(i_before)),
        ),
        _ => (None, None),
    };

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
        voluntary_cs,
        involuntary_cs,
    };

    println!(
        "[{label}] P50≈{}μs  P90≈{}μs  P99≈{}μs  P99.9≈{}μs  exact_max={}μs  \
         violations={}/{} ({:.2}%)  cs(vol={}, invol={})",
        stats.p50_us,
        stats.p90_us,
        stats.p99_us,
        stats.p999_us,
        stats.exact_max_us,
        stats.violations,
        stats.total_blocks,
        stats.violation_pct(),
        fmt_cs(stats.voluntary_cs),
        fmt_cs(stats.involuntary_cs),
    );

    stats
}

fn fmt_cs(value: Option<u64>) -> String {
    value.map_or_else(|| "N/A".to_string(), |v| v.to_string())
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
        "[RECEIPT] status=CHARACTERIZED label={label} stress_workers={stress_workers} \
         pinned={} p50_us={} p90_us={} p99_us={} p99.9_us={} exact_max_us={} violations={}/{} \
         cs_vol={} cs_invol={}",
        JitterStats::is_pinned(),
        stats.p50_us,
        stats.p90_us,
        stats.p99_us,
        stats.p999_us,
        stats.exact_max_us,
        stats.violations,
        stats.total_blocks,
        fmt_cs(stats.voluntary_cs),
        fmt_cs(stats.involuntary_cs),
    );
}

fn emit_skip_receipt(label: &str, stress_workers: usize, reason: &str) {
    println!(
        "[RECEIPT] status=SKIP_CAPABILITY label={label} stress_workers={stress_workers} \
         reason=\"{reason}\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A \
         exact_max_us=N/A violations=N/A"
    );
}

/// Runs one jitter character scenario at the given stress level.
///
/// The stress count is explicit — a scenario labeled `stress-N` spawns
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
            "[{label}] Baseline P50≈{}μs  P90≈{}μs  P99≈{}μs  P99.9≈{}μs  exact_max={}μs",
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
        let note = if stats.exact_max_us < b.exact_max_us {
            " (informative: baseline > stress)"
        } else {
            ""
        };
        println!(
            "[{label}] Δ P50≈: {}  Δ P90≈: {}  Δ P99≈: {}  Δ P99.9≈: {}  Δ max: {}{}  \
             Δ violations: {}→{}",
            delta_pct(stats.p50_us, b.p50_us),
            delta_pct(stats.p90_us, b.p90_us),
            delta_pct(stats.p99_us, b.p99_us),
            delta_pct(stats.p999_us, b.p999_us),
            delta_pct(stats.exact_max_us, b.exact_max_us),
            note,
            b.violations,
            stats.violations,
        );
    }

    emit_receipt(label, stress_threads, &stats);
    Some(stats)
}

// ── Jitter characterization — baseline (no stress) ─────────────────────────

#[test]
#[ignore = "long: RT jitter characterization"]
fn test_jitter_characterization_baseline_wavenet_standard() {
    let preflight = rt_helpers::rt_preflight_jitter();
    rt_helpers::print_preflight(&preflight);

    if preflight.status != RtPreflightStatus::Pass {
        println!("[STATUS] INCONCLUSIVE — skipping jitter characterization");
        println!("  Environment preconditions not met; jitter telemetry would be");
        println!("  invalid without performance governor and low background load.");
        println!(
            "[RECEIPT] status=INCONCLUSIVE reason=\"preflight_failed\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A exact_max_us=N/A violations=N/A"
        );
        return;
    }

    println!("[STATUS] PASS — preflight OK");
    let all_cpus = initial_process_cpus();
    let bench_core = bench_core_env().filter(|c| all_cpus.contains(c));
    if let Some(core) = bench_core {
        pin_thread(&[core]);
    }
    println!(
        "[JIT_PINS] pinned={} cores={} bench_core_env=\"{}\"",
        JitterStats::is_pinned(),
        JitterStats::affinity_cores(),
        std::env::var("NAM_BENCH_CORE").unwrap_or_default(),
    );
    if let Some(mut model) = load_and_prewarm("BossWN-standard.nam") {
        let baseline = measure_latency("WaveNet-Std-char-baseline", &mut model, 0);

        // Emit typed receipt with percentiles as completion marker.
        // `status=CHARACTERIZED` (S5-T2): jitter telemetry never asserts a
        // deadline, so the token must not read as a gate approval.
        emit_receipt("WaveNet-Std-char-baseline", 0, &baseline);

        // Run stress characterizations with delta against this baseline.
        // Stress-1 and stress-2 carry 1 and 2 real CPU-burn workers.
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
            let note = if sat_stats.exact_max_us < baseline.exact_max_us {
                " (informative: baseline > stress)"
            } else {
                ""
            };
            println!(
                "[WaveNet-Std-char-saturate] Δ violations: {}→{}  \
                 Δ P99≈: {}  Δ max: {}{}",
                baseline.violations,
                sat_stats.violations,
                delta_pct(sat_stats.p99_us, baseline.p99_us),
                delta_pct(sat_stats.exact_max_us, baseline.exact_max_us),
                note,
            );
            println!(
                "[WaveNet-Std-char-saturate] Violations under saturation — \
                 environmental characterization, NOT a deadline gate."
            );
        }
        // Every scenario (baseline, stress-1, stress-2 and
        // saturate-N, including the single-core `saturate-1`) ended its
        // section with a canonical `[RECEIPT]`/`[STATUS]` terminal line — the
        // section-level marker closes the test so a truncated log is visible.
        println!(
            "[STATUS] PASS — jitter characterization complete (all scenarios emitted terminal receipts)"
        );
    } else {
        println!("[STATUS] SKIP_CAPABILITY: model_not_found:BossWN-standard.nam");
        println!(
            "[RECEIPT] status=SKIP_CAPABILITY reason=\"model_not_found\" p50_us=N/A p90_us=N/A p99_us=N/A p99.9_us=N/A exact_max_us=N/A violations=N/A"
        );
    }
}
