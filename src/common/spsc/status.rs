// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

use crate::common::atomics::{AtomicI32, AtomicI64, AtomicU32, AtomicU64};
use core::sync::atomic::Ordering;

/// Global flag for coordinated graceful shutdown across all threads.
/// Set to `true` by the CTRL+C handler.
///
/// Kept on the standard-library `AtomicBool` in every build (instead of
/// `common::atomics`): `loom::sync::atomic::AtomicBool::new` is not `const`, and
/// `SHUTDOWN` is a process-global control-plane flag that is never part of a
/// modeled handshake — it is simply set by the signal handler and polled by the
/// main loop, so there is nothing for the Loom permutation engine to validate.
pub static SHUTDOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Flag indicating that the DSP thread needs a new `NamResampler`.
pub const RT_STATUS_NEEDS_RESAMPLER_REBUILD: u64 = 1 << 0;
/// Indicates whether the last resampler rebuild attempt by the main thread failed.
pub const RT_STATUS_RESAMPLER_REBUILD_FAILED: u64 = 1 << 1;
/// `true` if the DSP thread confirmed operation under `SCHED_FIFO`.
pub const RT_STATUS_RT_IS_FIFO: u64 = 1 << 2;
/// Flag indicating that saturation (clipping) occurred on the output audio.
pub const RT_STATUS_HAS_CLIPPED: u64 = 1 << 3;
/// Flag indicating that the current buffer is completely silent (Gate closed).
pub const RT_STATUS_IS_SILENT: u64 = 1 << 4;
/// Flag indicating that the GC channel overflowed.
pub const RT_STATUS_GC_OVERFLOW: u64 = 1 << 5;
/// Flag indicating that the gate is transitioning (Fade-In or Fade-Out).
pub const RT_STATUS_IS_FADING: u64 = 1 << 6;
/// Flag indicating that a critical model load failure occurred on the RT thread.
pub const RT_STATUS_MODEL_LOAD_FAILED: u64 = 1 << 7;
/// Flag indicating that a heap allocation occurred on the RT thread (detected by heap-audit).
pub const RT_STATUS_HEAP_ALLOC: u64 = 1 << 8;
/// Flag indicating that the RT callback should pause DSP processing until
/// the resampler is replaced (during hot-plug or sample rate change).
pub const RT_STATUS_RESAMP_SWAP_PENDING: u64 = 1 << 10;
/// Flag indicating that at least one huge-page allocation succeeded
/// (set by main thread after alloc, checked by telemetry for logging).
pub const RT_STATUS_HUGEPAGE_OK: u64 = 1 << 11;
/// Soft-degrade: model running in Reduced mode
/// (fewer WaveNet layers or LSTM single-layer).
pub const RT_STATUS_DEGRADE_REDUCED: u64 = 1 << 12;
/// Soft-degrade: model running in Minimal mode
/// (maximum reduction — passthrough for LSTM, half-WaveNet).
pub const RT_STATUS_DEGRADE_MINIMAL: u64 = 1 << 13;
/// Flag indicating that a cabsim rebuild is needed
/// (partition_size no longer matches current buffer size).
pub const RT_STATUS_NEEDS_CABSIM_REBUILD: u64 = 1 << 9;
/// Flag indicating that a corrupted/malformed GC item was detected
/// (unknown type_id or inconsistent type+ptr in overflow buffer).
pub const RT_STATUS_GC_CORRUPTED: u64 = 1 << 14;
/// Flag indicating that the A2 static variant triggered the scalar fallback path
/// (no CH=3 or CH=8 conv available). Set by the RT thread for telemetry polling.
pub const RT_STATUS_A2_FALLBACK_TRIGGERED: u64 = 1 << 15;
/// Flag indicating that a WaveNet slimmable slice_channels rebuild failed on the RT thread.
/// Replaces `log::error!` for RT-zero-IO compliance.
pub const RT_STATUS_SLIMMABLE_SLICE_FAILED: u64 = 1 << 16;
/// Flag indicating that a WaveNet slimmable rebuild is needed (set by RT, cleared by main).
pub const RT_STATUS_NEEDS_SLIMMABLE_REBUILD: u64 = 1 << 17;
/// Flag indicating that Transparent Huge Pages (THP) advice was used
/// (madvise MADV_HUGEPAGE + MADV_COLLAPSE), as opposed to explicit HugeTLB 2 MB pages.
pub const RT_STATUS_THP_ACTIVE: u64 = 1 << 18;
/// Flag indicating that an oversampling factor change is needed (set by RT, cleared by main).
pub const RT_STATUS_NEEDS_OS_REBUILD: u64 = 1 << 19;
/// Flag indicating that the audio host provided a buffer violating the FFI contract
/// (misaligned byte count, offset out of bounds). Set by the RT callback.
pub const RT_STATUS_HOST_CONTRACT_VIOLATION: u64 = 1 << 20;
/// Flag indicating that `ContainerModel::set_slimmable_size` failed to reset a submodel.
/// Replaces `log::error!` for RT-zero-IO compliance. Set by RT callback, read by main thread.
pub const RT_STATUS_SLIMMABLE_RESET_FAILED: u64 = 1 << 21;
/// Flag indicating that the host quantum (buffer size) changed.
/// Set by the RT callback whenever the per-cycle sample count differs from the
/// previous cycle. The main thread reads `requested_buffer_frames`, logs, and clears.
pub const RT_STATUS_NEEDS_QUANTUM_LOG: u64 = 1 << 23;

/// Flag indicating that the GC cascade reached Tier 3 (overflow buffer).
/// Set whenever an item is parked in the overflow buffer, regardless of overwrite.
pub const RT_STATUS_GC_TIER3: u64 = 1 << 22;

/// Flag indicating that the SPSC command queue drain limit was reached (64 events) on the RT thread.
pub const RT_STATUS_SPSC_DRAIN_TRUNCATED: u64 = 1 << 24;

/// Flag indicating that a CabSim sub-block violated the adapter contract
/// (`input.len() > partition_size` or `input.len() != output.len()` during host
/// quantum renegotiation). Set by the RT thread; the adapter clamps defensively
/// and never panics.
pub const RT_STATUS_CABSIM_CONTRACT_VIOLATION: u64 = 1 << 25;

/// Flag indicating that non-finite audio samples (NaN or Inf) were detected on an input buffer.
/// Set by the RT thread; the callback silences the corrupted block and recovers safely.
pub const RT_STATUS_NON_FINITE_INPUT_DETECTED: u64 = 1 << 26;

/// Flag indicating that the GUI→host output event queue backpressured: a gesture
/// (begin/change/end) event could not be pushed to the host in one flush cycle.
/// The caller must retain the corresponding gesture bit for retry on the next
/// flush/process; this flag is telemetry for housekeeping/log off-RT so the
/// saturation is explicit, never an invisible loss.
pub const RT_STATUS_GUI_EVENT_BACKPRESSURE: u64 = 1 << 27;

/// Flag indicating that a structural command (model load, IR swap, oversample
/// rebuild, state restore) was deferred to the next audio callback because the
/// per-callback structural budget (F-RT-007 command budgeting) was exhausted.
/// Deferral preserves FIFO ordering and never loses the payload; the consumer
/// parks the command and applies it at the start of the next callback.
pub const RT_STATUS_STRUCTURAL_DEFERRED: u64 = 1 << 28;

/// Flag indicating that a deferred structural command was superseded by a newer
/// same-kind command already queued in the SPSC ring (command coalescing). The
/// superseded command's resources are discarded off-RT through the GC cascade —
/// never dropped on the audio thread and never applied (latest-wins semantics).
pub const RT_STATUS_STRUCTURAL_SUPERSEDED: u64 = 1 << 29;

/// Flag indicating that the scalar parameter command queue still held elements
/// after the per-callback drain budget (`MAX_PARAM_BUDGET`) was exhausted
/// (F-RB-011 / T2.5). The RT callback consumed its fixed quota for this quantum
/// and the remainder is processed by the next callback; the flag is telemetry
/// for the main thread so the saturation is explicit, never an invisible loss.
pub const RT_STATUS_PARAM_QUEUE_BACKLOG: u64 = 1 << 30;

/// Atomic status flags for silent RT→Main communication.
///
/// The DSP thread sets atomic flags instead of calling `println!`/`eprintln!`.
/// The main thread reads these flags periodically and prints logs to the user.
/// This ensures **zero I/O** occurs in the RT callback.
///
/// ### Bitmask Map (`status_bits`)
///
/// | Bit | Constant | Description |
/// | :--- | :--- | :--- |
/// | 0 | `NEEDS_RESAMPLER_REBUILD` | DSP thread requests new resampler |
/// | 1 | `RESAMPLER_REBUILD_FAILED` | Resampler rebuild failed |
/// | 2 | `RT_IS_FIFO` | SCHED_FIFO active confirmed |
/// | 3 | `HAS_CLIPPED` | Output saturation (clipping) |
/// | 4 | `IS_SILENT` | Buffer completely silent (Gate Closed) |
/// | 5 | `GC_OVERFLOW` | Garbage Collection channel overflow |
/// | 6 | `IS_FADING` | Gate transitioning (Fading In/Out) |
/// | 7 | `MODEL_LOAD_FAILED` | Model load failure on RT thread |
/// | 8 | `HEAP_ALLOC` | Heap allocation detected on RT thread |
/// | 9 | `NEEDS_CABSIM_REBUILD` | DSP thread requests cabsim engine rebuild |
/// | 10 | `RESAMP_SWAP_PENDING` | RT callback paused awaiting resampler swap |
/// | 11 | `HUGEPAGE_OK` | Huge-page allocation confirmed active |
/// | 12 | `DEGRADE_REDUCED` | Soft-degrade active — Reduced mode |
/// | 13 | `DEGRADE_MINIMAL` | Soft-degrade active — Minimal mode |
/// | 14 | `GC_CORRUPTED` | GC overflow buffer corrupted (unknown type/ptr) |
/// | 15 | `A2_FALLBACK_TRIGGERED` | A2 static variant fell back to scalar zero-output path |
/// | 16 | `SLIMMABLE_SLICE_FAILED` | WaveNet slimmable slice_channels rebuild failed |
/// | 17 | `NEEDS_SLIMMABLE_REBUILD` | DSP thread requests slimmable model rebuild |
/// | 18 | `THP_ACTIVE` | Transparent huge pages(madvise) active — not explicit HugeTLB |
/// | 19 | `NEEDS_OS_REBUILD` | DSP thread requests oversampling engine rebuild |
/// | 20 | `HOST_CONTRACT_VIOLATION` | Host buffer FFI contract violated (misaligned or OOB) |
/// | 21 | `SLIMMABLE_RESET_FAILED` | ContainerModel submodel reset failed on RT thread |
/// | 22 | `GC_TIER3` | GC cascade reached Tier 3 (overflow buffer) — item parked |
/// | 23 | `NEEDS_QUANTUM_LOG` | Host quantum (buffer size in frames) changed |
/// | 24 | `SPSC_DRAIN_TRUNCATED` | SPSC command queue drain limit reached (64 events) |
/// | 25 | `CABSIM_CONTRACT_VIOLATION` | CabSim sub-block exceeded partition or input/output mismatch |
/// | 26 | `NON_FINITE_INPUT_DETECTED` | Non-finite input sample (NaN/Inf) detected and contained |
/// | 27 | `GUI_EVENT_BACKPRESSURE` | GUI→host output event queue full; gesture bits retained for retry |
/// | 28 | `STRUCTURAL_DEFERRED` | Structural command deferred to next callback (budget exhausted) |
/// | 29 | `STRUCTURAL_SUPERSEDED` | Deferred structural command superseded by newer same-kind; discarded off-RT |
/// | 30 | `PARAM_QUEUE_BACKLOG` | Scalar param queue still non-empty after the per-callback drain budget |
#[repr(align(128))]
pub struct RtStatusFlags {
    /// Effective sample rate active on the DSP thread after resampler rebuild.
    /// Set by the DSP thread upon consuming a new `NamResampler` from the SPSC channel.
    /// Value `0` indicates no pending update.
    pub active_rate: AtomicU32,
    /// Rate change notification for logging purposes.
    /// Value `0` indicates no change since the last poll.
    pub active_rate_changed: AtomicU32,

    /// Target rate detected by the DSP thread from the audio host but not yet applied (awaiting rebuild).
    /// The main thread reads this value to know which rate to build.
    /// Value `0` indicates no pending request.
    pub requested_host_rate: AtomicU32,

    /// Target rate of the loaded model (NAM). The usual default is 48000.
    pub requested_nam_rate: AtomicU32,

    /// Monotonic generation counter for resampler swap requests (F-RB-004).
    ///
    /// Incremented with `Ordering::Release` by the RT thread in `sync_rate`
    /// every time a rebuild is requested, *after* publishing the new
    /// `requested_host_rate` / `requested_nam_rate`. The main thread captures it
    /// as the version of the `ResamplerSwapPayload` it builds, and the RT drain
    /// installs a payload only while its generation still matches this counter —
    /// otherwise the envelope is stale and goes to the GC cascade without
    /// unmuting. This eliminates the lost-wakeup where a rate renegotiation
    /// arriving during a rebuild was silently erased.
    pub requested_rate_generation: AtomicU64,

    /// Generation of the resampler currently applied on the DSP thread.
    ///
    /// Stored with `Ordering::Release` by the RT thread when it installs a
    /// `ResamplerSwapPayload` whose generation matches
    /// [`Self::requested_rate_generation`]. The invariant
    /// `applied_rate_generation == requested_rate_generation` must hold before
    /// the callback clears `RT_STATUS_RESAMP_SWAP_PENDING` (unmutes) — a stale
    /// resampler can never substitute the most recent request.
    pub applied_rate_generation: AtomicU64,

    /// Request generation of a failed resampler rebuild attempt by the main thread.
    ///
    /// Stored with `Ordering::Release` by the main thread when a resampler rebuild
    /// attempt fails for `requested_rate_generation`. The RT callback checks
    /// this value against its current `requested_rate_generation` before performing
    /// a safe fail-open unmute: if a newer generation request B arrived while
    /// generation A failed, the failure of A is ignored and B remains pending.
    pub resampler_failed_generation: AtomicU64,

    /// Effective RT priority confirmed by `pthread_getschedparam`.
    /// Value `-1` indicates the check has not yet been performed.
    /// Set on the cold-path of the DSP thread's first frame.
    pub rt_priority: AtomicI32,

    /// Atomic counter of DSP overloads (virtual XRUNs).
    /// Incremented by the RT callback if processing exceeds 85% of the time budget.
    pub dsp_overloads: AtomicU32,

    /// Processing time of the last DSP cycle in ticks (RDTSC).
    /// Read by the main thread and converted to Duration via Anchor.
    pub dsp_cycle_time: AtomicU64,

    /// Number of samples processed in the last cycle (for budget calculation).
    pub last_n_samples: AtomicU32,

    /// Latency histogram for statistical analysis (P50, P95, P99).
    pub latency_hist: crate::dsp::telemetry::LatencyHistogram,

    /// Total degradation transitions that have occurred (Full↔Reduced↔Minimal).
    pub degrade_transitions_total: AtomicU32,

    /// Atomic bitmask containing binary states (needs_rebuild, clipped, silent, etc).
    /// Reduces Cache Bouncing by condensing multiple states into a single cache line.
    pub status_bits: AtomicU64,

    /// Confirmed RT priority.
    pub confirmed_priority: AtomicI32,
    /// Confirmed RT scheduling policy.
    pub rt_policy: AtomicI32,
    /// Pinned physical CPU core (or -1 if not pinned).
    pub rt_cpu: AtomicI32,
    /// Accumulated OR of all RT_STATUS_* flags ever seen since startup.
    pub flags_seen: AtomicU64,
    /// Total count of virtual XRUNs/overloads.
    pub xruns: AtomicU32,
    /// Total count of GC items successfully drained.
    pub drains: AtomicU32,
    /// Requested partition size for cabsim rebuild (set by RT thread).
    pub requested_cabsim_partition_size: AtomicU32,
    /// Requested host output rate for cabsim rebuild (set by RT thread).
    ///
    /// The cab-sim stage runs at the host output rate (after the return
    /// resampler), so the IR must be recalibrated whenever that rate
    /// changes. The RT thread publishes the applied host rate here before
    /// raising `RT_STATUS_NEEDS_CABSIM_REBUILD` (F-RB-006 rate
    /// calibration); value `0` indicates no request was ever published.
    pub requested_cabsim_host_rate: AtomicU32,
    /// Monotonic generation counter for cabsim rebuild requests (F-RB-004
    /// pattern).
    ///
    /// Incremented with `Release` by the RT thread whenever it raises
    /// `RT_STATUS_NEEDS_CABSIM_REBUILD`, after publishing the requested
    /// partition size and host rate. The main thread captures it with
    /// `Acquire` before building and re-arms the flag if the generation
    /// advanced during the build, so a renegotiation arriving mid-rebuild
    /// is never erased (lost-wakeup guard).
    pub requested_cabsim_generation: AtomicU64,
    /// Generation of the cab-sim pair currently applied on the DSP thread.
    ///
    /// Stored with `Ordering::Release` by the RT thread when it installs a
    /// `CabSimSwapPayload` whose generation matches
    /// [`Self::requested_cabsim_generation`].
    pub applied_cabsim_generation: AtomicU64,
    /// Requested slimmable channel count (set by RT thread, read by main thread).
    /// Value `0` indicates no pending request.
    pub requested_slimmable_ch: AtomicU32,
    /// Slimmable rebuild generation (set by RT thread, read by main thread).
    ///
    /// Incremented with `Release` by the RT callback whenever it requests a
    /// slimmable rebuild (`NEEDS_SLIMMABLE_REBUILD`), mirroring the resampler
    /// `requested_rate_generation` protocol (F-RB-004). The main thread captures
    /// the value with `Acquire` before building and stamps the delivered
    /// [`crate::common::spsc::SlimModelPair`], so the RT drain can discard stale
    /// pairs and guarantee L/R are always swapped from the latest request.
    pub requested_slimmable_generation: AtomicU64,
    /// Requested oversampling factor (0=Off, 1=X2, 2=X4) for engine rebuild.
    /// Set by RT thread, read and cleared by main thread after rebuild.
    pub requested_os_factor: AtomicU32,

    /// Host quantum (buffer size in frames) detected by the RT callback.
    /// Stored by the RT thread whenever `n_samples` differs from the previous cycle.
    /// Read by the main thread for quantum-renegotiation logging.
    pub requested_buffer_frames: AtomicU32,

    /// Previous quantum value used by the main loop to detect and log changes.
    /// Updated by the main thread after logging. Not accessed by the RT thread.
    pub previous_buffer_frames: AtomicU32,

    /// Incremented by the RT callback when capture (source)
    /// `dequeue_buffer()` returns `None` — host buffer miss on the input side.
    pub input_buffer_miss: AtomicU32,
    /// Incremented by the playback thread when `dequeue_buffer()`
    /// returns `None` — host buffer miss on the output side.
    pub output_buffer_miss: AtomicU32,
    /// Incremented by the playback callback each time the bridge produced no
    /// new DSP block (capture paused, resampler rebuild pending, clock drift or
    /// quantum miss) and the deterministic silence policy delivered a recycled
    /// output buffer filled with `0.0f32` (G-RB-001 / T4.2). Telemetry only —
    /// the hardware never repeats stale audio.
    pub playback_bridge_starvation: AtomicU32,

    /// Last sample rate negotiated by the capture stream's `param_changed`
    /// listener (`0` = never negotiated). Written on the PipeWire ThreadLoop
    /// thread (cold path, not the RT data thread); read by the playback
    /// listener for the cross-stream rate comparison and by the main loop for
    /// diagnostics (G-RB-001 / T4.3).
    pub capture_negotiated_rate: AtomicU32,

    /// Last sample rate negotiated by the playback stream's `param_changed`
    /// listener (`0` = never negotiated). Written on the PipeWire ThreadLoop
    /// thread (cold path, not the RT data thread); read by the capture
    /// listener for the cross-stream rate comparison and by the main loop for
    /// diagnostics (G-RB-001 / T4.3).
    pub playback_negotiated_rate: AtomicU32,

    /// Sticky latch guarding the capture stream SPA format contract.
    pub capture_format_ok: AtomicU32,

    /// Sticky latch guarding the playback stream SPA format contract.
    pub playback_format_ok: AtomicU32,

    /// Active state of the capture stream (1 = Streaming, 0 = Paused/Unconnected/Error).
    pub capture_active: AtomicU32,

    /// Active state of the playback stream (1 = Streaming, 0 = Paused/Unconnected/Error).
    pub playback_active: AtomicU32,

    /// Aggregate sticky latch guarding the strict SPA format contract (G-RB-001 / T4.3).
    ///
    /// `1` = both stream formats are valid (`F32P` planar stereo); `0` = a divergent format
    /// was negotiated on either stream.
    pub format_contract_ok: AtomicU32,

    /// Host clock `time.now` from the last capture stream time() call (nanoseconds).
    pub capture_host_now: AtomicI64,
    /// Host clock `time.ticks` from the last capture stream time() call.
    pub capture_host_ticks: AtomicU64,
    /// Host clock `time.delay` from the last capture stream time() call (ticks).
    pub capture_host_delay: AtomicI64,
    /// Host clock `time.now` from the last playback stream time() call (nanoseconds).
    pub playback_host_now: AtomicI64,
    /// Host clock `time.ticks` from the last playback stream time() call.
    pub playback_host_ticks: AtomicU64,
    /// Host clock `time.delay` from the last playback stream time() call (ticks).
    pub playback_host_delay: AtomicI64,

    /// Total count of structural commands deferred by the audio callback because
    /// the per-callback structural budget was exhausted (F-RT-007). Monotonic
    /// telemetry counter — never reset; accompanies `RT_STATUS_STRUCTURAL_DEFERRED`.
    pub structural_deferred_total: AtomicU32,
    /// Total count of deferred structural commands superseded by a newer
    /// same-kind command and discarded off-RT via the GC cascade (command
    /// coalescing). Accompanies `RT_STATUS_STRUCTURAL_SUPERSEDED`.
    pub structural_superseded_total: AtomicU32,

    /// errno from `pthread_setaffinity_np` (0 = success).
    pub rt_affinity_err: AtomicI32,
    /// errno from `pthread_setschedparam` (0 = success).
    pub rt_sched_err: AtomicI32,
    /// errno from `pthread_getschedparam` (0 = success).
    pub rt_getsched_err: AtomicI32,
    /// Target CPU requested for affinity pinning (-1 = not set).
    pub rt_target_cpu: AtomicI32,
}

impl RtStatusFlags {
    /// Creates a new instance with zero/sentinel initial values.
    #[cold]
    pub fn new() -> Self {
        Self {
            active_rate: AtomicU32::new(0),
            active_rate_changed: AtomicU32::new(0),
            requested_host_rate: AtomicU32::new(0),
            requested_nam_rate: AtomicU32::new(48_000),
            requested_rate_generation: AtomicU64::new(0),
            applied_rate_generation: AtomicU64::new(0),
            resampler_failed_generation: AtomicU64::new(0),
            rt_priority: AtomicI32::new(-1),
            dsp_overloads: AtomicU32::new(0),
            dsp_cycle_time: AtomicU64::new(0),
            last_n_samples: AtomicU32::new(0),
            latency_hist: crate::dsp::telemetry::LatencyHistogram::new(),
            degrade_transitions_total: AtomicU32::new(0),
            status_bits: AtomicU64::new(0),
            confirmed_priority: AtomicI32::new(-1),
            rt_policy: AtomicI32::new(-1),
            rt_cpu: AtomicI32::new(-1),
            flags_seen: AtomicU64::new(0),
            xruns: AtomicU32::new(0),
            drains: AtomicU32::new(0),
            requested_cabsim_partition_size: AtomicU32::new(0),
            requested_cabsim_host_rate: AtomicU32::new(0),
            requested_cabsim_generation: AtomicU64::new(0),
            applied_cabsim_generation: AtomicU64::new(0),
            requested_slimmable_ch: AtomicU32::new(0),
            requested_slimmable_generation: AtomicU64::new(0),
            requested_os_factor: AtomicU32::new(0),
            requested_buffer_frames: AtomicU32::new(0),
            previous_buffer_frames: AtomicU32::new(0),
            input_buffer_miss: AtomicU32::new(0),
            output_buffer_miss: AtomicU32::new(0),
            playback_bridge_starvation: AtomicU32::new(0),
            capture_negotiated_rate: AtomicU32::new(0),
            playback_negotiated_rate: AtomicU32::new(0),
            capture_format_ok: AtomicU32::new(1),
            playback_format_ok: AtomicU32::new(1),
            capture_active: AtomicU32::new(1),
            playback_active: AtomicU32::new(1),
            format_contract_ok: AtomicU32::new(1),
            capture_host_now: AtomicI64::new(0),
            capture_host_ticks: AtomicU64::new(0),
            capture_host_delay: AtomicI64::new(0),
            playback_host_now: AtomicI64::new(0),
            playback_host_ticks: AtomicU64::new(0),
            playback_host_delay: AtomicI64::new(0),
            structural_deferred_total: AtomicU32::new(0),
            structural_superseded_total: AtomicU32::new(0),
            rt_affinity_err: AtomicI32::new(0),
            rt_sched_err: AtomicI32::new(0),
            rt_getsched_err: AtomicI32::new(0),
            rt_target_cpu: AtomicI32::new(-1),
        }
    }

    /// Whether audio is unmuted across both streams (capture and playback format contracts
    /// valid AND both streams active).
    #[inline(always)]
    pub fn is_audio_unmuted(&self) -> bool {
        self.capture_format_ok.load(Ordering::Relaxed) != 0
            && self.playback_format_ok.load(Ordering::Relaxed) != 0
            && self.capture_active.load(Ordering::Relaxed) != 0
            && self.playback_active.load(Ordering::Relaxed) != 0
    }

    /// Sets one or more flags in the bitmask with `Relaxed` ordering.
    ///
    /// **Telemetry-only flags**: use this when the flag does not gate any
    /// associated data — the consumer only needs to observe the bit eventually,
    /// and a torn observation is harmless. This is the in-crate default: every
    /// internal `RT_STATUS_*` set uses `Relaxed` because those bits are
    /// self-contained status signals (no payload follows them).
    #[inline(always)]
    pub fn set_flag(&self, flag: u64) {
        self.status_bits.fetch_or(flag, Ordering::Relaxed);
    }

    /// Clears one or more flags in the bitmask with `Relaxed` ordering.
    ///
    /// `Relaxed` is correct here because these bits are written by the RT
    /// thread with `set_flag`/`set_flag_release` and are never read back by it —
    /// the main thread is the only reader, and no happens-before edge is needed
    /// to reset a bit the producer never re-reads.
    #[inline(always)]
    pub fn clear_flag(&self, flag: u64) {
        self.status_bits.fetch_and(!flag, Ordering::Relaxed);
    }

    /// Checks whether a flag is active with `Relaxed` ordering.
    ///
    /// Use this only for telemetry flags (see [`Self::set_flag`]). For a flag
    /// that gates associated data, use [`Self::check_flag_acquire`] so the
    /// Acquire barrier orders the payload read.
    #[inline(always)]
    pub fn check_flag(&self, flag: u64) -> bool {
        (self.status_bits.load(Ordering::Relaxed) & flag) != 0
    }

    /// Sets one or more flags with `Release` ordering — producer side of a
    /// **data handshake** (public API for downstream consumers).
    ///
    /// # Synchronization protocol (RT producer → main/consumer)
    ///
    /// Use this when the flag signals that associated *data* (e.g.
    /// `requested_host_rate`, `requested_slimmable_ch`) is ready to be read:
    ///
    /// 1. Producer writes the data fields (`Relaxed` or `Release` is sufficient
    ///    for the individual fields, since the flag publication below orders
    ///    them — `Release` is conventional for the last write).
    /// 2. Producer calls [`Self::set_flag_release`] on the gating flag. The
    ///    Release barrier publishes *all* prior writes on this thread to any
    ///    thread that observes the flag with Acquire.
    /// 3. Consumer observes the flag with [`Self::check_flag_acquire`] and then
    ///    reads the data fields — the Acquire barrier makes every write
    ///    sequenced-before the matching `set_flag_release` visible.
    /// 4. Consumer resets the flag with [`Self::clear_flag`] /
    ///    [`Self::clear_flag_relaxed`] (the RT producer never reads the bit back).
    ///
    /// The invariant — a consumer that sees the flag set must also see the
    /// full data payload — is enforced by the Release/Acquire pair, and is
    /// model-checked by the Loom suite (`tests/loom_tests.rs`, `--cfg loom`).
    ///
    /// ## Why these methods exist as public API
    ///
    /// They have **zero in-crate callers** by design: internal `RT_STATUS_*`
    /// signaling is telemetry-only (`Relaxed`), so the crate itself never needs
    /// the barrier. They are retained for *downstream hosts* that publish a
    /// value + flag pair across their own RT/Main boundary (e.g. a plugin that
    /// writes `requested_host_rate` then raises `RT_STATUS_NEEDS_RESAMPLER_REBUILD`).
    /// This is the only sanctioned way for a consumer to publish associated
    /// data through `RtStatusFlags`.
    #[inline(always)]
    pub fn set_flag_release(&self, flag: u64) {
        self.status_bits.fetch_or(flag, Ordering::Release);
    }

    /// Clears one or more flags with `Relaxed` ordering.
    ///
    /// Use this when the consumer clears flags the RT producer never acquires —
    /// the main thread simply resets its own flags after acting on them.
    /// No happens-before edge is needed because the RT producer only sets
    /// these flags (via `fetch_or`), never reads them back. Alias of
    /// [`Self::clear_flag`], kept under a distinct name to document the intent.
    #[inline(always)]
    pub fn clear_flag_relaxed(&self, flag: u64) {
        self.status_bits.fetch_and(!flag, Ordering::Relaxed);
    }

    /// Clears one or more flags with `Release` ordering (data handshake).
    ///
    /// Consumer-side reset in a data handshake *before* publishing new data:
    /// e.g. the RT thread lowers a handshake flag with Release so the main
    /// thread's subsequent Acquire observes the lowering. Paired with
    /// [`Self::set_flag_release`] / [`Self::check_flag_acquire`] in the
    /// protocol documented there.
    #[inline(always)]
    pub fn clear_flag_release(&self, flag: u64) {
        self.status_bits.fetch_and(!flag, Ordering::Release);
    }

    /// Checks whether a flag is active with `Acquire` ordering — consumer side
    /// of a **data handshake**.
    ///
    /// Use this when the flag gates access to data written by the producer.
    /// The Acquire barrier guarantees all writes sequenced-before the matching
    /// [`Self::set_flag_release`] are visible on this thread. See that method
    /// for the full protocol.
    #[inline(always)]
    pub fn check_flag_acquire(&self, flag: u64) -> bool {
        (self.status_bits.load(Ordering::Acquire) & flag) != 0
    }

    /// Checks whether a flag is active and clears it atomically in a single
    /// operation, recording it into `flags_seen`.
    ///
    /// Returns `true` if the flag was active.
    ///
    /// # Ordering note
    ///
    /// The clear uses `Relaxed` because this is a **telemetry drain**, not a
    /// data handshake: it is meant for flags the main thread polls and resets
    /// (e.g. `RT_STATUS_GC_OVERFLOW`), where no payload follows the flag. It
    /// also ORs the bit into `flags_seen` so the set of flags ever raised is
    /// preserved even after the bit is cleared.
    ///
    /// For a flag that gates associated data, do **not** use this: first
    /// [`Self::check_flag_acquire`] (orders the payload read), then
    /// [`Self::clear_flag_relaxed`] (or `clear_flag_release` if new data is
    /// about to be published).
    #[inline(always)]
    pub fn check_and_clear_flag(&self, flag: u64) -> bool {
        let old = self.status_bits.fetch_and(!flag, Ordering::Relaxed);
        let active = (old & flag) != 0;
        if active {
            self.flags_seen.fetch_or(flag, Ordering::Relaxed);
        }
        active
    }
}

impl Default for RtStatusFlags {
    fn default() -> Self {
        Self::new()
    }
}
