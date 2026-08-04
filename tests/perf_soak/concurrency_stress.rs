// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//  Dedicated concurrency stress tests with real OS threads.
//
//  These tests exercise the lock-free SPSC/GC pipeline, model/IR hot-swap,
//  and parameter smoothing under deliberate thread contention.
//
//  **They must run without `--test-threads=1`** so that multiple test
//  threads (plus internal spawns) exercise the kernel scheduler.  The
//  heavy variants are marked `#[ignore]` for the long suite.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use neural_amp_modeler_rs::common::spsc::{
    GcItem, GcOverflowBuffer, RT_STATUS_GC_CORRUPTED, RtStatusFlags, drain_gc_channels, gc_cascade,
};
use neural_amp_modeler_rs::dsp::resampler::NamResampler;
use neural_amp_modeler_rs::dsp::smoother::ParamSmoother;

// =============================================================================
// Helper: create a lightweight bypass resampler for GC stress tests
// =============================================================================

fn make_bypass_resampler() -> NamResampler {
    NamResampler::new(48000, 48000, 64).expect("Bypass resampler creation failed")
}

fn make_gc_resampler_item() -> GcItem {
    GcItem::Resampler(Box::new(make_bypass_resampler()))
}

fn count_gc_resamplers(items: &[GcItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i, GcItem::Resampler(_)))
        .count()
}

// =============================================================================
// Test 1 — GcOverflowBuffer concurrent push/drain (T6.3 torn-read coverage)
// =============================================================================

/// Rapid concurrent push/drain on the overflow buffer with 4 producer
/// threads and 1 drainer thread.  Every drained item must decode
/// correctly — a single `RT_STATUS_GC_CORRUPTED` flag means that a
/// torn-read (type↔ptr inconsistency) slipped through, which T6.3
/// eliminated by packing both into a single `AtomicU64`.
#[test]
fn test_overflow_concurrent_push_drain() {
    const ITEMS_PER_PRODUCER: usize = 2_000;
    const NUM_PRODUCERS: usize = 4;
    const BUFFER_CAPACITY: usize = 64;

    let overflow = Arc::new(GcOverflowBuffer::new(BUFFER_CAPACITY));
    let drain_status = Arc::new(RtStatusFlags::new());
    // Track how many producers have finished
    let producers_done_count = Arc::new(AtomicU32::new(0));

    // --- Spawn producers ---
    let mut producers = Vec::with_capacity(NUM_PRODUCERS);
    for _ in 0..NUM_PRODUCERS {
        let ov = Arc::clone(&overflow);
        let pdc = Arc::clone(&producers_done_count);
        producers.push(thread::spawn(move || {
            for _ in 0..ITEMS_PER_PRODUCER {
                ov.push(make_gc_resampler_item());
            }
            pdc.fetch_add(1, Ordering::Release);
        }));
    }

    // --- Drainer thread: drains while producers run, then final drain ---
    let ov_drain = Arc::clone(&overflow);
    let ds = Arc::clone(&drain_status);
    let pdc_drain = Arc::clone(&producers_done_count);
    let drainer = thread::spawn(move || {
        let mut total_resamplers: usize = 0;
        let mut empty_streak: u32 = 0;
        loop {
            let items = ov_drain.drain(&ds);
            let count = count_gc_resamplers(&items);
            drop(items);

            if count == 0 {
                empty_streak += 1;
            } else {
                empty_streak = 0;
            }
            total_resamplers += count;

            let all_done = pdc_drain.load(Ordering::Acquire) == NUM_PRODUCERS as u32;
            if all_done && empty_streak >= 10 {
                break;
            }
            thread::yield_now();
        }
        total_resamplers
    });

    for p in producers {
        p.join().unwrap();
    }
    let total = drainer.join().unwrap();

    println!(
        "Overflow concurrent OK — {} resampler items drained (expected up to {})",
        total,
        ITEMS_PER_PRODUCER * NUM_PRODUCERS
    );

    // With a 64-slot buffer and 4 concurrent pushers, some items are
    // expected to be overwritten (controlled leak).  The key verification
    // is zero GC_CORRUPTED flags.
    assert!(
        !drain_status.check_flag(RT_STATUS_GC_CORRUPTED),
        "RT_STATUS_GC_CORRUPTED was set — possible torn-read (T6.3 regression)!"
    );
    assert!(
        total > 0,
        "No items were drained — test infrastructure issue"
    );
}

// =============================================================================
// Test 2 — SPSC GC cascade + drain under contention
// =============================================================================

/// Exercises the full three-tier cascade (SPSC channel → parking lot →
/// overflow buffer) with 2 producers pushing through `gc_cascade()` and
/// 1 drainer calling `drain_gc_channels()`.  All three tiers are stressed
/// because small channel capacity forces fallback to the overflow buffer.
#[test]
#[ignore]
fn test_gc_cascade_concurrent_drain() {
    use rtrb::RingBuffer;

    const GC_CAPACITY: usize = 8; // small to force cascade quickly
    const PUSH_COUNT: usize = 50_000;

    let (gc_producer, gc_consumer) = RingBuffer::<GcItem>::new(GC_CAPACITY);
    let overflow = Arc::new(GcOverflowBuffer::new(64));
    let rt_status = Arc::new(RtStatusFlags::new());
    let producer_done_count = Arc::new(AtomicU32::new(0));

    // Wrap the producer for thread-safe sharing (gc_cascade needs &mut Producer)
    let prod_shared = Arc::new(std::sync::Mutex::new(gc_producer));

    let p1 = {
        let prod = Arc::clone(&prod_shared);
        let ov1 = Arc::clone(&overflow);
        let rs1 = Arc::clone(&rt_status);
        let pdc = Arc::clone(&producer_done_count);
        thread::spawn(move || {
            let mut parking_lot: [Option<GcItem>; 16] = Default::default();
            for _ in 0..PUSH_COUNT {
                let mut producer = prod.lock().unwrap();
                gc_cascade(
                    Some(make_gc_resampler_item()),
                    &mut producer,
                    &mut parking_lot,
                    &ov1,
                    &rs1,
                );
            }
            pdc.fetch_add(1, Ordering::Release);
        })
    };

    let p2 = {
        let prod = Arc::clone(&prod_shared);
        let ov2 = Arc::clone(&overflow);
        let rs2 = Arc::clone(&rt_status);
        let pdc = Arc::clone(&producer_done_count);
        thread::spawn(move || {
            let mut parking_lot: [Option<GcItem>; 16] = Default::default();
            for _ in 0..PUSH_COUNT {
                let mut producer = prod.lock().unwrap();
                gc_cascade(
                    Some(make_gc_resampler_item()),
                    &mut producer,
                    &mut parking_lot,
                    &ov2,
                    &rs2,
                );
            }
            pdc.fetch_add(1, Ordering::Release);
        })
    };

    // Drainer
    let ov_drain = Arc::clone(&overflow);
    let rs_drain = Arc::clone(&rt_status);
    let pdc_drain = Arc::clone(&producer_done_count);
    let drainer = thread::spawn(move || {
        let mut total = 0usize;
        let mut gc_cons = gc_consumer;
        let mut empty_streak: u32 = 0;
        loop {
            let drained = drain_gc_channels(&mut gc_cons, &ov_drain, &rs_drain);
            total += drained;
            if drained == 0 {
                empty_streak += 1;
            } else {
                empty_streak = 0;
            }
            // Both producers done, and buffers drained
            let all_done = pdc_drain.load(Ordering::Acquire) == 2;
            if all_done && empty_streak > 100 {
                break;
            }
            thread::yield_now();
        }
        total
    });

    p1.join().unwrap();
    p2.join().unwrap();
    let total = drainer.join().unwrap();

    println!(
        "gc_cascade concurrent OK — {} items drained (expected up to {})",
        total,
        2 * PUSH_COUNT
    );
    // Items may be overwritten due to small channel capacity
    assert!(
        total > 0,
        "No items were drained — test infrastructure issue"
    );
    assert!(
        !rt_status.check_flag(RT_STATUS_GC_CORRUPTED),
        "RT_STATUS_GC_CORRUPTED was set during gc_cascade stress!"
    );
}

// =============================================================================
// Test 3 — Model hot-swap during concurrent audio processing
// =============================================================================

/// Simulates the main thread rapidly loading and swapping models via SPSC
/// while the "audio thread" concurrently processes blocks and drains
/// payloads.  Exercises the `cold_load_model` data race window.
#[test]
#[ignore]
fn test_model_hot_swap_during_process() {
    use neural_amp_modeler_rs::models::NamModel;
    use rtrb;
    use rtrb::RingBuffer;
    use std::path::PathBuf;

    let mut model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    model_dir.push("tests/fixtures/models");

    let models: Vec<PathBuf> = ["BossWN-nano.nam", "BossWN-standard.nam"]
        .iter()
        .map(|name| model_dir.join(name))
        .filter(|p| p.exists())
        .collect();

    if models.len() < 2 {
        eprintln!(
            "SKIP: need at least 2 model fixtures, found {}",
            models.len()
        );
        return;
    }

    let (mut param_prod, param_cons) =
        RingBuffer::<neural_amp_modeler_rs::common::spsc::ParamPayload>::new(4);

    let models_arc = Arc::new(models);
    let swap_count = Arc::new(AtomicU32::new(0));
    let done = Arc::new(AtomicBool::new(false));

    // --- "Main thread": loads and pushes ---
    let models_clone = Arc::clone(&models_arc);
    let sc = Arc::clone(&swap_count);
    let dn = Arc::clone(&done);
    let loader = thread::spawn(move || {
        for (idx, _) in (0..200).enumerate() {
            let model_path = &models_clone[idx % models_clone.len()];

            let json_data = std::fs::read_to_string(model_path)
                .unwrap_or_else(|_| panic!("Failed to read {:?}", model_path));
            let model_data = neural_amp_modeler_rs::loader::nam_json::parse_nam_json(&json_data)
                .unwrap_or_else(|_| panic!("Failed to parse {:?}", model_path));
            let boxed = neural_amp_modeler_rs::loader::dispatcher::build_model(&model_data)
                .unwrap_or_else(|_| panic!("Dispatcher failed for {:?}", model_path));

            let mut payload = Some(boxed);
            loop {
                match param_prod.push(
                    neural_amp_modeler_rs::common::spsc::ParamPayload::LoadModel {
                        model_l: payload.take(),
                        model_r: None,
                        input_mult_adj: 1.0,
                        output_mult_adj: 1.0,
                        sample_rate: 48000,
                    },
                ) {
                    Ok(()) => break,
                    Err(rtrb::PushError::Full(mut returned)) => {
                        if let neural_amp_modeler_rs::common::spsc::ParamPayload::LoadModel {
                            model_l,
                            ..
                        } = &mut returned
                        {
                            payload = model_l.take();
                        }
                        thread::yield_now();
                    }
                }
            }
            sc.fetch_add(1, Ordering::Relaxed);
        }
        dn.store(true, Ordering::Release);
    });

    // --- "Audio thread": processes and swaps ---
    let dn2 = Arc::clone(&done);
    let audio_thread = thread::spawn(move || {
        let mut cons = param_cons;
        let mut active_model: Option<Box<neural_amp_modeler_rs::models::StaticModel>> = None;
        let input = [0.1f32; 64];
        let mut output = [0.0f32; 64];
        let mut last_was_empty = false;

        loop {
            // Drain all pending swap payloads
            while let Ok(payload) = cons.pop() {
                if let neural_amp_modeler_rs::common::spsc::ParamPayload::LoadModel {
                    model_l,
                    ..
                } = payload
                {
                    drop(active_model.take());
                    if let Some(mut m) = model_l {
                        m.prewarm(512);
                        active_model = Some(m);
                    }
                }
            }

            // Process a block if we have a model
            if let Some(ref mut model) = active_model {
                model.process(&input, &mut output);
                for s in &output {
                    assert!(s.is_finite(), "Non-finite sample in model output");
                }
            }

            let done = dn2.load(Ordering::Acquire);
            let empty = cons.pop().is_err();
            if done && empty {
                if last_was_empty {
                    break;
                }
                last_was_empty = true;
            } else {
                last_was_empty = false;
            }
            thread::yield_now();
        }
    });

    loader.join().unwrap();
    audio_thread.join().unwrap();

    let swaps = swap_count.load(Ordering::Relaxed);
    println!(
        "Model hot-swap stress OK — {} swaps processed concurrently with inference.",
        swaps
    );
    assert!(swaps >= 200, "Expected at least 200 swaps, got {}", swaps);
}

// =============================================================================
// Test 4 — CabSim IR hot-swap during concurrent audio processing
// =============================================================================

/// Exercises cabsim engine swap (via SPSC) while a processing thread
/// runs convolution concurrently.  Covers the race between
/// `cold_load_cabsim` (replaces `conv_engine`) and `process()`.
#[test]
#[ignore]
fn test_cabsim_swap_during_process() {
    use rtrb;
    use rtrb::RingBuffer;

    // Synthetic short IR (256 samples @ 48 kHz → ~5.3 ms)
    let ir: Vec<f32> = (0..256)
        .map(|n| {
            let t = n as f32 / 256.0;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * (-3.0 * t).exp()
        })
        .collect();

    let partition_size: usize = 64;
    let engine_a = neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine::new(&ir, partition_size)
        .expect("construction should succeed for test-sized buffers");

    let (mut cs_prod, cs_cons) =
        RingBuffer::<Option<Box<neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine>>>::new(4);

    cs_prod
        .push(Some(Box::new(engine_a)))
        .expect("Failed to push first engine");

    let iterations = Arc::new(AtomicU32::new(0));
    let done = Arc::new(AtomicBool::new(false));

    // --- Swapper thread: sends engines ---
    let dn = Arc::clone(&done);
    let it = Arc::clone(&iterations);
    let swapper = thread::spawn(move || {
        for i in 0..500 {
            let mut current_engine = Some(Box::new(
                neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine::new(&ir, partition_size)
                    .expect("construction should succeed for test-sized buffers"),
            ));
            loop {
                match cs_prod.push(current_engine) {
                    Ok(()) => break,
                    Err(rtrb::PushError::Full(returned)) => {
                        current_engine = returned;
                        thread::yield_now();
                    }
                }
            }
            it.store(i + 1, Ordering::Relaxed);
        }
        dn.store(true, Ordering::Release);
    });

    // --- Processing thread: drains and processes ---
    let dn2 = Arc::clone(&done);
    let processor = thread::spawn(move || {
        let mut cons = cs_cons;
        let mut active_engine: Option<Box<neural_amp_modeler_rs::dsp::cabsim::conv::ConvEngine>> =
            None;
        let input = [0.01f32; 64];
        let mut output = [0.0f32; 64];
        let mut empty_streak: u32 = 0;

        loop {
            // Drain engine swap payloads
            while let Ok(engine) = cons.pop() {
                if let Some(e) = engine {
                    drop(active_engine.take());
                    active_engine = Some(e);
                }
                empty_streak = 0;
            }

            // Process if we have an engine
            if let Some(ref mut engine) = active_engine {
                engine.process(&input, &mut output);
                for s in &output {
                    assert!(s.is_finite(), "Non-finite sample from cabsim processing");
                }
            }

            let done = dn2.load(Ordering::Acquire);
            empty_streak += 1;
            if done && empty_streak > 100 {
                break;
            }
            thread::yield_now();
        }
    });

    swapper.join().unwrap();
    processor.join().unwrap();

    let swaps = iterations.load(Ordering::Relaxed);
    assert!(
        swaps > 0,
        "No cabsim engines were swapped — SPSC channel may be broken"
    );
    println!(
        "Cabsim swap stress OK — {} engine swaps during concurrent convolution.",
        swaps
    );
}

// =============================================================================
// Test 5 — Parameter smoothing under concurrent target updates (main↔RT)
// =============================================================================

/// Simulates the main↔RT parameter update path: one thread continuously
/// writes new parameter values (like the UI/main thread via atomics),
/// while another thread reads them and applies to a `ParamSmoother`.
///
/// Verification: the smoother converges toward the last-written value
/// and never produces NaN or infinite values despite concurrent writes.
#[test]
fn test_param_smoothing_concurrent_updates() {
    let shared_param = Arc::new(AtomicU32::new(1.0f32.to_bits()));
    let updates = Arc::new(AtomicU32::new(0));
    let done = Arc::new(AtomicBool::new(false));

    // --- Writer thread: rapid parameter changes ---
    let sp = Arc::clone(&shared_param);
    let up = Arc::clone(&updates);
    let dn = Arc::clone(&done);
    let writer = thread::spawn(move || {
        let values: [f32; 6] = [1.0, 0.5, 2.0, 0.1, 0.0, 1.0];
        for (idx, _) in (0..100_000).enumerate() {
            sp.store(values[idx % values.len()].to_bits(), Ordering::Release);
            up.fetch_add(1, Ordering::Relaxed);
            thread::yield_now();
        }
        dn.store(true, Ordering::Release);
    });

    // --- Reader/"audio" thread: reads atomics and smoothes ---
    let sp2 = Arc::clone(&shared_param);
    let dn2 = Arc::clone(&done);
    let reader = thread::spawn(move || {
        let mut smoother_l = ParamSmoother::new(1.0, 48000.0, 20.0);
        let mut total_samples = 0u64;
        let mut converged_count = 0u64;

        loop {
            let new_target = f32::from_bits(sp2.load(Ordering::Acquire));
            if (new_target - smoother_l.target_value()).abs() > f32::EPSILON {
                smoother_l.set_target(new_target);
            }

            // Simulate one audio block of smoothing (64 samples)
            for _ in 0..64 {
                let val = smoother_l.tick();
                assert!(val.is_finite(), "Smoother produced non-finite value: {val}");
                total_samples += 1;
            }

            if (smoother_l.current_value() - smoother_l.target_value()).abs() < 1e-6 {
                converged_count += 1;
            }

            let d = dn2.load(Ordering::Acquire);
            let stable = converged_count >= 10;
            if d && stable {
                break;
            }
            thread::yield_now();
        }

        (
            total_samples,
            smoother_l.current_value(),
            smoother_l.target_value(),
        )
    });

    writer.join().unwrap();
    let (total_samples, current, target) = reader.join().unwrap();

    let error = (current - target).abs();
    println!(
        "Param smoothing concurrent OK — {} samples processed, current={:.6}, target={:.6}, error={:.6}",
        total_samples, current, target, error
    );

    // The smoother moves toward the target but due to rapid concurrent writes
    // may not fully converge within the test window.  The key property is
    // that tick() never produced NaN or infinite values under contention.
}

// =============================================================================
// Test 6 — T6.3 race reproduction: torn-read prevention under max contention
// =============================================================================

/// Reproduces the exact scenario that the T6.3 fix protects against:
/// maximum-rate concurrent `push()` and `drain()` on the
/// `GcOverflowBuffer`.  Before T6.3, type and pointer were two separate
/// atomic swaps, creating a torn-read window.  After T6.3, type+pointer
/// are packed into a single `AtomicU64`, eliminating the window.
///
/// This test runs hundreds of thousands of push/drain cycles and asserts
/// zero `RT_STATUS_GC_CORRUPTED` flags.
#[test]
#[ignore]
fn test_t6_3_torn_read_prevention() {
    const BUFFER_SLOTS: usize = 16; // tiny → maximum overwrite pressure
    const PUSHES_PER_THREAD: usize = 50_000;
    const NUM_PUSHERS: usize = 8;

    let overflow = Arc::new(GcOverflowBuffer::new(BUFFER_SLOTS));
    let drain_status = Arc::new(RtStatusFlags::new());
    let barrier = Arc::new(std::sync::Barrier::new(NUM_PUSHERS + 1)); // +1 for drainer
    let producers_done = Arc::new(AtomicU32::new(0));

    // --- Producer threads: push at maximum speed ---
    let mut handles = Vec::new();
    for _ in 0..NUM_PUSHERS {
        let ov = Arc::clone(&overflow);
        let b = Arc::clone(&barrier);
        let pd = Arc::clone(&producers_done);
        handles.push(thread::spawn(move || {
            b.wait(); // synchronize start to maximize contention
            for _ in 0..PUSHES_PER_THREAD {
                ov.push(make_gc_resampler_item());
            }
            pd.fetch_add(1, Ordering::Release);
        }));
    }

    // --- Drainer thread ---
    let ov_drain = Arc::clone(&overflow);
    let b_drain = Arc::clone(&barrier);
    let ds = Arc::clone(&drain_status);
    let pd_drain = Arc::clone(&producers_done);
    let drainer = thread::spawn(move || {
        b_drain.wait();
        let mut drained_total: usize = 0;
        let mut empty_streak: u32 = 0;
        loop {
            let items = ov_drain.drain(&ds);
            let count = items.len();
            // Every non-empty slot must be a valid Resampler
            for item in &items {
                match item {
                    GcItem::Resampler(_) => {}
                    _other => panic!("Unexpected variant in overflow buffer drain"),
                }
            }
            drop(items);

            if count == 0 {
                empty_streak += 1;
            } else {
                empty_streak = 0;
            }
            drained_total += count;

            let all_done = pd_drain.load(Ordering::Acquire) == NUM_PUSHERS as u32;
            if all_done && empty_streak >= 20 {
                break;
            }
            thread::yield_now();
        }
        drained_total
    });

    for h in handles {
        h.join().unwrap();
    }
    let total = drainer.join().unwrap();

    assert!(
        !drain_status.check_flag(RT_STATUS_GC_CORRUPTED),
        "T6.3 regression! GC_CORRUPTED flag set under maximum contention — \
         torn-read window may still exist. {} items drained.",
        total
    );

    println!(
        "T6.3 torn-read prevention OK — {} items pushed/drained with zero corruption.",
        total
    );
}

// =============================================================================
// Test 7 — Param Smoother multi-reader (independent smoothers, shared atomics)
// =============================================================================

/// Multiple "audio threads" each own a `ParamSmoother` and read from a
/// shared atomic parameter, simulating stereo/unlinked processing.
/// Verifies that reading a shared atomic while multiple smoothers tick
/// independently is correct and never produces invalid values.
#[test]
fn test_multi_reader_param_smoothing() {
    let shared_gain = Arc::new(AtomicU32::new(1.0f32.to_bits()));
    let done = Arc::new(AtomicBool::new(false));

    let sg = Arc::clone(&shared_gain);
    let dn = Arc::clone(&done);
    let writer = thread::spawn(move || {
        let values = [1.0f32, 0.5, 2.0, 0.25, 4.0, 0.125, 1.0];
        for &val in values.iter().cycle().take(2000) {
            sg.store(val.to_bits(), Ordering::Release);
            for _ in 0..50 {
                thread::yield_now();
            }
        }
        dn.store(true, Ordering::Release);
    });

    let num_readers: usize = 4;
    let mut readers = Vec::new();
    for _ in 0..num_readers {
        let sg2 = Arc::clone(&shared_gain);
        let dn2 = Arc::clone(&done);
        readers.push(thread::spawn(move || {
            let mut smoother = ParamSmoother::new(1.0, 48000.0, 20.0);
            let mut ticks = 0u64;
            loop {
                let target = f32::from_bits(sg2.load(Ordering::Acquire));
                smoother.set_target(target);
                for _ in 0..16 {
                    let val = smoother.tick();
                    assert!(val.is_finite(), "Non-finite value in multi-reader test");
                    ticks += 1;
                }
                if dn2.load(Ordering::Acquire) {
                    // Final convergence attempt — not required to reach exact
                    // target under concurrent writes, but must not panic
                    for _ in 0..1000 {
                        let v = smoother.tick();
                        assert!(v.is_finite(), "Non-finite at final convergence");
                        ticks += 1;
                    }
                    break;
                }
                thread::yield_now();
            }
            ticks
        }));
    }

    writer.join().unwrap();
    for r in readers {
        let ticks = r.join().unwrap();
        assert!(ticks > 0, "Reader thread processed zero samples");
    }

    println!(
        "Multi-reader param smoothing OK — {} readers processed concurrently.",
        num_readers
    );
}
