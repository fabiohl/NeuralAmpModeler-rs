# ADR: Real-Time Host Hardening on macOS

- **Status:** Accepted by the operator on 2026-10-28
- **Date:** 2026-10-28
- **Feature:** `rt-hardening`
- **Targets:** `aarch64-apple-darwin` (supported); `x86_64-apple-darwin` remains
  unsupported per DEC-01.
- **Related:** `src/rt_hardening/`, `docs/architecture.md` §2, Sprint S09 (T09.1–T09.4).

## Context

On Linux, the opt-in `rt-hardening` feature gives audio hosts a small,
host-agnostic bring-up surface: `mlockall`, `SCHED_FIFO`, IRQ-aware affinity,
PM-QoS, THP control and DAZ/FTZ. The engine itself never names a host project.
macOS has no equivalent surface today: the module is compiled only for Linux,
so Apple Silicon hosts cannot request the platform's own real-time mechanisms
without adding host-specific code outside the crate.

XNU does not provide Linux CPU affinity, `SCHED_FIFO`, PM-QoS or THP controls.
Its real-time contract is built from different primitives:

- `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` for the
  interactive QoS class.
- `thread_policy_set(THREAD_TIME_CONSTRAINT_POLICY)` with period, computation
  and constraint expressed in Mach absolute-time ticks.
- `mach_timebase_info` to convert nanoseconds to Mach ticks.
- `mlock` (`mlock_region`) to prevent page faults in locked regions.
- Audio Workgroups (`os_workgroup_join` / `os_workgroup_leave`), supplied by
  the host, to describe the real-time audio workload to the scheduler.

The design must not depend on CoreAudio headers or link against CoreAudio. The
host owns the `os_workgroup_t` it obtains from its audio device and passes the
opaque pointer to the engine.

## Decision

Extend the existing `rt-hardening` feature to macOS through a separate
`src/rt_hardening/macos.rs` implementation, gated by
`#[cfg(all(feature = "rt-hardening", target_os = "macos"))]`. The Linux module
and its public API stay unchanged. The macOS surface is:

| Function | Contract |
|:---------|:---------|
| `set_ftz_daz()` | Sets `FPCR.FZ` on the calling thread. Same contract as the Linux helper. |
| `mlock_region(addr, len)` | `unsafe` `mlock` for a host-owned model/scratch region; off-RT only. |
| `set_user_interactive_qos()` | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`; off-RT or RT thread setup. |
| `promote_time_constraint(buffer_frames, sample_rate, &RtStatusFlags)` | Applies DAZ/FTZ, interactive QoS and `THREAD_TIME_CONSTRAINT_POLICY`; returns the applied `ThreadTiming`. |
| `compute_thread_timing(buffer_frames, sample_rate, numer, denom)` | Pure, testable tick conversion; no syscall. |
| `join_audio_workgroup(workgroup)` | `unsafe` join of the calling thread to a host-supplied `os_workgroup_t`; returns an RAII guard that leaves on drop. |

### Timing policy

`period` is derived from the host quantum:

```text
period_ns = buffer_frames * 1_000_000_000 / sample_rate
period_ticks = period_ns * timebase.denom / timebase.numer
```

`computation` and `constraint` are both set to `period_ticks`: the audio
callback is budgeted for the full quantum and is not expected to be preempted
inside the block. `preemptible` is `false`. Values that do not fit in `u32`
Mach ticks fail with `EINVAL` rather than silently wrapping. A zero
`buffer_frames`, zero `sample_rate`, or zero timebase component also fails with
`EINVAL`.

### Telemetry mapping

The existing `RtStatusFlags` fields remain the single telemetry contract:

- `rt_policy` = `THREAD_TIME_CONSTRAINT_POLICY` on success, `-1` on failure.
- `rt_priority` / `confirmed_priority` = `QOS_CLASS_USER_INTERACTIVE` (not a
  Mach scheduler priority).
- `rt_sched_err` = the first failing `pthread_*` or Mach error code.
- `rt_tid` = `pthread_self()`; `rt_cpu` = `-1` (no portable current-CPU API is
  used).
- `RT_STATUS_RT_IS_FIFO` is never set on macOS. Hosts must inspect
  `rt_policy`, not the Linux-named flag.

### Memory locking

`mlock_region` locks bounded host-owned model or scratch allocations in RAM via
`libc::mlock` (off-RT only). `mlockall_current` is omitted on macOS because
`_mlockall` availability and behavior in `libSystem.B.dylib` are not verified
on an Apple host. The engine does not reach into host buffers.

### Audio Workgroups

`join_audio_workgroup` accepts the opaque `os_workgroup_t` obtained by the host
from its audio device. It is `unsafe` because the pointer must be a live
workgroup and the call must run on the thread that will process audio. The
returned guard stores the join token and calls `os_workgroup_leave` exactly
once, on drop or via `leave()`. The engine does not create, retain or release
the workgroup object itself.

### Deliberately absent

- CPU affinity, `SCHED_FIFO`, PM-QoS and THP: not part of the XNU contract.
- CoreAudio framework linkage or header use: the host owns the workgroup.
- A new public telemetry flag or a new dependency: the existing fields and
  `libc` are sufficient.
- Windows cfg changes: the module uses `target_os` gates, so a future Windows
  port is not blocked (DEC-07).

## Alternatives considered

1. **Reuse the Linux module with `cfg(unix)`.** Rejected: most Linux calls do
   not exist on XNU, and a broad `unix` gate would block the future Windows
   port.
2. **Require the host to implement all macOS calls.** Rejected: duplicates the
   same fragile FFI in every host and loses the engine's tested conversion and
   error mapping.
3. **Link CoreAudio to read the workgroup automatically.** Rejected: couples
   the engine to a host framework and violates the API-neutrality rule.
4. **Only set QoS, without a time-constraint policy.** Rejected: QoS alone does
   not describe the callback period/deadline to the scheduler.

## Consequences

- The feature remains opt-in; default builds and the `testing` permutation are
  unchanged on macOS.
- `cargo check --target aarch64-apple-darwin --features rt-hardening` becomes a
  required cross-check. `x86_64-apple-darwin` is not claimed.
- Real jitter measurement on M-series hardware is a hardware gate and cannot be
  satisfied on the Linux development host.
- Hosts gain one optional unsafe entry point for Audio Workgroups but no
  CoreAudio dependency.

## Gates

- `G-L` lints, `G-T` targeted tests, `G-C` cross-check for
  `aarch64-apple-darwin`, `G-DOC` docs/rustdoc, `G-HW` jitter on M-series
  (operator), and `G-Q` quick suite once at sprint closure.
