// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Loom concurrency test suite for `NeuralAmpModeler-rs`.
//!
//! Model-checks the **production** lock-free primitives under loom's exhaustive
//! permutation engine: [`RtStatusFlags`] atomic handshake methods
//! (`set_flag_release`/`check_flag_acquire`/`check_and_clear_flag`),
//! [`GcOverflowBuffer`] AcqRel slot protocol, the [`gc_cascade`]/[`drain_gc_channels`]
//! SPSC→parking→overflow cascade, and the double-buffered [`DspBridge`] write/read
//! handshake.
//!
//! The production atomics are routed through `src/common/atomics.rs` so that, under
//! `--cfg loom` (see `utils/tests-long.sh` phase 6), they resolve to loom's
//! instrumented wrappers — a divergence in production memory orderings now fails
//! this suite immediately instead of hiding behind mirrored mocks (G-01).

#![cfg(loom)]

use loom::cell::UnsafeCell;
use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use loom::thread;

use neural_amp_modeler_rs::common::spsc::{
    GcItem, GcOverflowBuffer, RT_STATUS_NEEDS_RESAMPLER_REBUILD, RtStatusFlags, drain_gc_channels,
    gc_cascade,
};
use neural_amp_modeler_rs::dsp::pipeline::{DspBridge, DspBridgeReader, DspBridgeWriter};

// =============================================================================
// Loom self-tests: canonical handshake pattern (not production-coupled)
// =============================================================================
//
// These two keep the loom harness itself honest and document the canonical
// producer/consumer handshake idiom (Release store of the flag, Acquire load by
// the consumer, payload behind a loom `UnsafeCell`). The production-coupled
// variants of the same protocol live further down (`test_rt_status_*`).

#[test]
fn test_handshake_correct_passes() {
    loom::model(|| {
        let flag = Arc::new(AtomicBool::new(false));
        let cell = Arc::new(UnsafeCell::new(0));

        let flag_clone = flag.clone();
        let cell_clone = cell.clone();

        let t1 = thread::spawn(move || {
            cell_clone.with_mut(|ptr| unsafe {
                *ptr = 42;
            });
            flag_clone.store(true, Ordering::Release);
        });

        let t2 = thread::spawn(move || {
            if flag.load(Ordering::Acquire) {
                let val = cell.with(|ptr| unsafe { *ptr });
                assert_eq!(val, 42);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

#[test]
fn test_handshake_relaxed_fails() {
    let result = std::panic::catch_unwind(|| {
        loom::model(|| {
            let flag = Arc::new(AtomicBool::new(false));
            let cell = Arc::new(UnsafeCell::new(0));

            let flag_clone = flag.clone();
            let cell_clone = cell.clone();

            let t1 = thread::spawn(move || {
                cell_clone.with_mut(|ptr| unsafe {
                    *ptr = 42;
                });
                flag_clone.store(true, Ordering::Relaxed);
            });

            let t2 = thread::spawn(move || {
                if flag.load(Ordering::Relaxed) {
                    let val = cell.with(|ptr| unsafe { *ptr });
                    assert_eq!(val, 42);
                }
            });

            t1.join().unwrap();
            t2.join().unwrap();
        });
    });
    assert!(
        result.is_err(),
        "Expected loom to catch a data race under Relaxed ordering"
    );
}

// =============================================================================
// Production RtStatusFlags handshake (T3.1) — replaces the LoomRtStatus mock
// =============================================================================
//
// Exercises the *real* `RtStatusFlags` data-handshake protocol: the RT side
// publishes `requested_host_rate` then raises the flag with `set_flag_release`;
// the main side observes with `check_flag_acquire` and must read the matching
// payload. Both tests directly drive the production methods documented in
// `status.rs`.

#[test]
fn test_rt_status_production_handshake() {
    loom::model(|| {
        let status = Arc::new(RtStatusFlags::new());
        let status_rt = status.clone();
        let status_main = status.clone();

        let t_rt = thread::spawn(move || {
            status_rt
                .requested_host_rate
                .store(48_000, Ordering::Release);
            status_rt.set_flag_release(RT_STATUS_NEEDS_RESAMPLER_REBUILD);
        });

        let t_main = thread::spawn(move || {
            let mut observed = Vec::new();
            for _ in 0..4 {
                if status_main.check_flag_acquire(RT_STATUS_NEEDS_RESAMPLER_REBUILD) {
                    observed.push(status_main.requested_host_rate.load(Ordering::Acquire));
                    status_main.clear_flag_relaxed(RT_STATUS_NEEDS_RESAMPLER_REBUILD);
                }
            }
            for rate in observed {
                assert_eq!(
                    rate, 48_000,
                    "multi-field handshake violated: flag observed without matching data"
                );
            }
        });

        t_rt.join().unwrap();
        t_main.join().unwrap();
    });
}

/// Demonstrates that the production `set_flag`/`check_flag` (Relaxed) path does
/// **not** order a handshake payload.
///
/// Production `requested_host_rate` is an `AtomicU32`, so the pure-production
/// type cannot reproduce the failure (atomics are race-free in loom; G-01's
/// aggravation is exactly that the mocked `UnsafeCell` hazard does not exist
/// in-crate). To still certify the *ordering requirement* for downstream
/// consumers, the flag side uses the real `RtStatusFlags` while the payload is
/// a loom `UnsafeCell` — loom then flags the race because the Relaxed flag
/// publication establishes no happens-before edge.
#[test]
fn test_rt_status_production_relaxed_fails() {
    let result = std::panic::catch_unwind(|| {
        loom::model(|| {
            let status = Arc::new(RtStatusFlags::new());
            let data = Arc::new(UnsafeCell::new(0u32));

            let status_rt = status.clone();
            let data_rt = data.clone();
            let status_main = status.clone();
            let data_main = data.clone();

            let t_rt = thread::spawn(move || {
                data_rt.with_mut(|ptr| unsafe {
                    *ptr = 48_000;
                });
                // Relaxed publication — NOT `set_flag_release`.
                status_rt.set_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD);
            });

            let t_main = thread::spawn(move || {
                if status_main.check_flag(RT_STATUS_NEEDS_RESAMPLER_REBUILD) {
                    let _ = data_main.with(|ptr| unsafe { *ptr });
                }
            });

            t_rt.join().unwrap();
            t_main.join().unwrap();
        });
    });
    assert!(
        result.is_err(),
        "Expected loom to catch the data race: `set_flag` (Relaxed) does not order \
         the payload write — only `set_flag_release` + `check_flag_acquire` do"
    );
}

// =============================================================================
// Production GcOverflowBuffer slot protocol (T3.1) — replaces LoomGcOverflowBuffer
// =============================================================================
//
// Directly exercises `GcOverflowBuffer::push`/`drain` with the real AcqRel slot
// handoff. Each item carries a loom `UnsafeCell` payload written by the producer
// before `push` and read by the consumer after `drain`: the slot `swap(AcqRel)`
// pair must be the happens-before edge that makes the payload visible — if the
// ordering regressed to Relaxed, loom flags the race immediately.

#[test]
fn test_gc_overflow_production() {
    loom::model(|| {
        let buffer = Arc::new(GcOverflowBuffer::new(3));
        let rt_status = Arc::new(RtStatusFlags::new());

        let buffer_prod = buffer.clone();
        let t1 = thread::spawn(move || {
            for item_id in 1..=3u32 {
                let cell = Box::new(UnsafeCell::new(0u32));
                cell.with_mut(|ptr| unsafe {
                    *ptr = item_id * 10;
                });
                buffer_prod.push(GcItem::LoomProbe(cell));
            }
        });

        let buffer_cons = buffer.clone();
        let rt_cons = rt_status.clone();
        let t2 = thread::spawn(move || {
            let drained = buffer_cons.drain(&rt_cons);
            for item in drained {
                match item {
                    GcItem::LoomProbe(cell) => {
                        let val = cell.with(|ptr| unsafe { *ptr });
                        assert!(
                            val >= 10 && val % 10 == 0,
                            "drained payload corrupted: {val}"
                        );
                    }
                    _ => unreachable!("only LoomProbe items are pushed"),
                }
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    });
}

// =============================================================================
// Production DspBridge write/read handshake (T3.1) — replaces LoomDspBridge mock
// =============================================================================
//
// Drives the real `DspBridge` double-buffer protocol through the real
// `DspBridgeWriter`/`DspBridgeReader` faces (production atomics under loom).
// The reader must only ever observe fully-published blocks (`n_samples` + L/R
// payload published before the Release generation store), never a torn buffer.
//
// Note on scope: the bridge's audio buffers are plain `[f32; MAX_BRIDGE_BUF]`
// arrays (by production design), so loom validates the *atomic handshake* of the
// real code rather than instrumenting the buffer bytes — the front/back-buffer
// non-overlap is exactly the invariant the generation/active_read_idx orderings
// must enforce. Each published block carries a distinct payload and the reader
// asserts the per-block invariant, so a stale generation or mismatched
// front-buffer selection is caught even without buffer-byte instrumentation
// (a plain-memory data race on the buffers themselves would require loom
// `UnsafeCell` payloads, which the production type does not use).

#[test]
fn test_dsp_bridge_production() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        // SAFETY of the raw-pointer handoff: `bridge` is Box-allocated in this
        // closure's frame and outlives both spawned threads (they are joined
        // before the closure returns), so the `DspBridgeWriter`/`Reader`
        // pointers reconstructed inside the threads stay valid for their whole
        // lifetime.
        //
        // The bridge is initialized in-place on the heap (not via a struct
        // literal) because `DspBridge` is 128 KiB of `[f32; MAX_BRIDGE_BUF]`
        // buffers: a stack-resident literal would overflow loom's coroutine
        // stack. Byte-zeroing is valid here — the type is a plain aggregate of
        // atomics and `f32` arrays (no Drop, no interior pointers) — and the
        // atomic fields are then reset to their canonical `new(0)` state.
        let mut bridge = {
            let mut boxed = Box::<DspBridge>::new_uninit();
            unsafe {
                std::ptr::write_bytes(boxed.as_mut_ptr(), 0, 1);
                let b = boxed.as_mut_ptr();
                (*b).active_read_idx = AtomicUsize::new(0);
                (*b).generation = AtomicU64::new(0);
                (*b).consumed_gen = AtomicU64::new(0);
                (*b).dropped_frames = AtomicU32::new(0);
                boxed.assume_init()
            }
        };
        let ptr = (&mut *bridge as *mut DspBridge) as usize;

        let writer = unsafe { DspBridgeWriter::new(ptr as *mut DspBridge) };
        let reader = unsafe { DspBridgeReader::new(ptr as *mut DspBridge) };

        let writer_t = writer;
        let t1 = thread::spawn(move || {
            // Distinct payload per published generation: a reader that observes a
            // stale or mismatched block (torn handshake) would not match the
            // per-block invariant below.
            for i in 1..=3u32 {
                let l = [i as f32, (i * 2) as f32, (i * 3) as f32];
                let r = [(i + 10) as f32, (i + 20) as f32, (i + 30) as f32];
                writer_t.write_block(&l, &r, l.len(), false);
            }
        });

        let reader_t = reader;
        let t2 = thread::spawn(move || {
            let mut last_gen = 0u64;
            for _ in 0..3 {
                reader_t.read_block(&mut last_gen, |l, r| {
                    assert!(
                        l.len() >= 3 && r.len() >= 3,
                        "reader observed a block shorter than any published block"
                    );
                    // Every published block obeys l = [k, 2k, 3k], r = [k+10, k+20, k+30].
                    // A torn/stale read (mixing values across generations or reading a
                    // partially-published buffer) breaks these relationships.
                    assert_eq!(l[0] * 2.0, l[1], "reader observed a torn or stale L block");
                    assert_eq!(l[0] * 3.0, l[2], "reader observed a torn or stale L block");
                    assert_eq!(r[0] - l[0], 10.0, "reader observed a torn or stale R block");
                });
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
        drop(bridge);
    });
}

// =============================================================================
// Production gc_cascade / drain_gc_channels (T3.1) — replaces LoomSwapMesh mock
// =============================================================================
//
// Exercises the real `gc_cascade` (SPSC → 16-slot parking → overflow) and
// `drain_gc_channels` against a real `rtrb` SPSC channel, the production
// `GcOverflowBuffer` (Tier 3) and `RtStatusFlags`. Every item cascaded must be
// drained exactly once — no leak, no double-drop — under every explored
// schedule. Items use the plain `LoomTag` payload because the SPSC tier runs on
// `rtrb`'s std atomics (invisible to loom); a loom cell there would report a
// spurious race that the real happens-before edge of `rtrb` provides.

#[test]
fn test_gc_cascade_production() {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(|| {
        // SPSC capacity 1 + parking 16 + overflow 1 == 18 in-flight items, so
        // the 18th cascade necessarily reaches Tier 3 (overflow) without ever
        // overwriting (no leak), while a concurrent drain frees headroom.
        let (mut gc_prod, mut gc_cons) = rtrb::RingBuffer::new(1);
        let overflow = Arc::new(GcOverflowBuffer::new(1));
        let rt_status = Arc::new(RtStatusFlags::new());

        let overflow_rt = overflow.clone();
        let rt_rt = rt_status.clone();
        let t_rt = thread::spawn(move || {
            let mut parking: [Option<GcItem>; 16] = Default::default();
            for id in 1..=18u32 {
                gc_cascade(
                    Some(GcItem::LoomTag(Box::new(id))),
                    &mut gc_prod,
                    &mut parking,
                    &overflow_rt,
                    &rt_rt,
                );
            }
            (parking, gc_prod)
        });

        let overflow_main = overflow.clone();
        let rt_main = rt_status.clone();
        let t_main = thread::spawn(move || {
            let mut parking: [Option<GcItem>; 16] = Default::default();
            let count = drain_gc_channels(&mut gc_cons, &overflow_main, &mut parking, &rt_main);
            (count, gc_cons)
        });

        let (mut rt_parking, _gc_prod) = t_rt.join().unwrap();
        let (mut drained, mut gc_cons) = t_main.join().unwrap();

        // Final off-RT drain after the RT producer has stopped (R-04 handoff of
        // the RT parking lot).
        drained += drain_gc_channels(&mut gc_cons, &overflow, &mut rt_parking, &rt_status);

        assert_eq!(
            drained, 18,
            "every cascaded item must be drained exactly once \
             (no leak, no double-drop) — drained: {drained}"
        );
    });
}
