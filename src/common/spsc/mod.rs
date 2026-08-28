// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Shared data structures and flags for lock-free thread coordination.
//!
//! This module defines all the "invisible wiring" that connects the CLI interface (the
//! musician's remote control) with the DSP audio engine (the neural amplifier running in real-time).
//!
//! ## Main components
//!
//! - **`SHUTDOWN`**: global flag signaling all threads that the program should terminate.
//! - **`ParamPayload`**: parameter "packets" (input/output gain, model swap)
//!   that the CLI sends to the DSP without locking any thread (lock-free via SPSC ring buffer).
//! - **`RtStatusFlags`**: atomic flags for silent RT→Main communication (no I/O in callback).
//!   Allows the DSP engine to report its status without ever calling `println!`.
//! - **`NamResampler` SPSC channel**: the main thread builds the resampler (with memory
//!   allocations) outside real-time and sends it to the DSP callback via a lock-free channel.

use rtrb::{Consumer, Producer, RingBuffer};
use std::sync::Arc;

mod gc;
mod payload;
mod status;

pub use gc::*;
pub use payload::*;
pub use status::*;

/// Default capacity for the main SPSC parameter channel.
///
/// 64 was chosen empirically: it provides enough headroom for rapid CLI control
/// commands while keeping memory usage negligible (64 × sizeof(ParamPayload) ≈ 2 KiB).
pub const SPSC_CAPACITY: usize = 64;

/// SPSC initialization result: parameter channels, model GC,
/// RT-safe resampler channel, cab-sim IR channel, and atomic status flags.
pub struct SpscChannels {
    /// CLI→DSP parameter producer.
    pub param_producer: Producer<ParamPayload>,
    /// CLI→DSP parameter consumer (moved to the RT callback).
    pub param_consumer: Consumer<ParamPayload>,
    /// GC producer: DSP thread sends obsolete items for drop outside RT.
    pub gc_producer: Producer<GcItem>,
    /// GC consumer: background thread executes `drop()`.
    pub gc_consumer: Consumer<GcItem>,
    /// Fallback buffer for GC overflow (overwrite).
    pub gc_overflow: Arc<GcOverflowBuffer>,
    /// Resampler producer: main thread builds and sends to the RT callback.
    /// Carries versioned envelopes (`ResamplerSwapPayload`) so the RT side can
    /// discard stale builds when the host renegotiates the clock mid-rebuild.
    pub resampler_producer: Producer<Box<ResamplerSwapPayload>>,
    /// Resampler consumer: RT callback drains to replace the active resampler.
    pub resampler_consumer: Consumer<Box<ResamplerSwapPayload>>,
    /// Cab-sim IR producer: main thread loads and sends a versioned cab-sim
    /// payload envelope (`CabSimSwapPayload`) to the RT callback (F-RB-004 / T7.1).
    pub cabsim_producer: Producer<Box<CabSimSwapPayload>>,
    /// Cab-sim IR consumer: RT callback drains to replace the active pair.
    pub cabsim_consumer: Consumer<Box<CabSimSwapPayload>>,
    /// Slimmable model producer: main thread builds and sends an atomic
    /// L/R pair (`SlimModelPair`) to the RT callback (F-RB-005).
    pub slimmable_producer: Producer<Box<SlimModelPair>>,
    /// Slimmable model consumer: RT callback drains atomic pairs to swap the
    /// active L/R models together.
    pub slimmable_consumer: Consumer<Box<SlimModelPair>>,
    /// Oversampling engine producer: main thread builds and sends OS engines to RT callback.
    pub os_producer: Producer<Box<crate::dsp::oversample::OsEnginePair>>,
    /// Oversampling engine consumer: RT callback drains to hot-swap OS engines.
    pub os_consumer: Consumer<Box<crate::dsp::oversample::OsEnginePair>>,
    /// Atomic status flags shared between RT and Main (zero I/O in callback).
    pub rt_status: Arc<RtStatusFlags>,
}

/// Creates and returns the complete lock-free SPSC mesh for the pipeline.
///
/// Includes channels for:
/// - CLI→DSP parameters (`ParamPayload`)
/// - Obsolete model GC (Drop-Delegation)
/// - Pre-built resamplers (Main→RT, zero allocation in callback)
/// - RT→Main atomic status flags
///
/// `capacity` should preferably be a power of 2.
pub fn setup_spsc(capacity: usize) -> SpscChannels {
    let (param_prod, param_cons) = RingBuffer::new(capacity);
    let (gc_prod, gc_cons) = RingBuffer::new(capacity * 4); // Quadrupled capacity for safe garbage collection
    // Resampler channel: small capacity (only 1 in transit at a time, typically)
    let (rs_prod, rs_cons) = RingBuffer::new(4);
    // Cab-sim IR channel: small capacity (only 1 in transit at a time)
    let (cs_prod, cs_cons) = RingBuffer::new(4);
    // Slimmable model channel: small capacity (only 1 in transit at a time)
    let (sl_prod, sl_cons) = RingBuffer::new(4);
    // Oversampling engine channel: small capacity (only L+R pair in transit at a time)
    let (os_prod, os_cons) = RingBuffer::new(4);
    let rt_status = Arc::new(RtStatusFlags::new());
    // The overflow buffer should be large enough to accommodate model swap spikes.
    // We use 64 as a base, or the requested capacity if higher.
    let gc_overflow = Arc::new(GcOverflowBuffer::new(capacity.max(64)));

    SpscChannels {
        param_producer: param_prod,
        param_consumer: param_cons,
        gc_producer: gc_prod,
        gc_consumer: gc_cons,
        gc_overflow,
        resampler_producer: rs_prod,
        resampler_consumer: rs_cons,
        cabsim_producer: cs_prod,
        cabsim_consumer: cs_cons,
        slimmable_producer: sl_prod,
        slimmable_consumer: sl_cons,
        os_producer: os_prod,
        os_consumer: os_cons,
        rt_status,
    }
}

#[cfg(test)]
#[path = "spsc_test.rs"]
mod spsc_test;
