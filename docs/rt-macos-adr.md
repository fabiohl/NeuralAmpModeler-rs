# ADR: Real-Time Host Hardening on macOS

- **Status:** Accepted
- **Feature:** `rt-hardening`
- **Targets:** `aarch64-apple-darwin` (supported); `x86_64-apple-darwin` is not
  validated or supported (see [architecture.md](architecture.md) §1.6 (Portability) and the
  README platform notes).
- **Related:** [`src/rt_hardening/`](../src/rt_hardening/),
  [architecture.md](architecture.md) §2 (Host Hardening).

## Context

On Linux, the opt-in `rt-hardening` feature gives audio hosts a small,
host-agnostic bring-up surface: `mlockall`, `SCHED_FIFO`, IRQ-aware affinity,
PM-QoS, THP control and DAZ/FTZ. The engine never names a host project. Before
this decision the module compiled only for Linux, so Apple Silicon hosts could
not request the platform's own real-time mechanisms without host-specific code
outside the crate.

XNU does not provide Linux CPU affinity, `SCHED_FIFO`, PM-QoS or THP controls.
Its real-time contract is built from different primitives:

- `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` for the
  interactive QoS class.
- `thread_policy_set(THREAD_TIME_CONSTRAINT_POLICY)` with period, computation
  and constraint expressed in Mach absolute-time ticks.
- `mach_timebase_info` to convert nanoseconds to Mach ticks.
- `mlock` (exposed as `mlock_region`) to prevent page faults in locked regions.
- Audio Workgroups (`os_workgroup_join` / `os_workgroup_leave`), supplied by
  the host, to describe the real-time audio workload to the scheduler.

The design must not depend on CoreAudio headers or link against CoreAudio. The
host owns the `os_workgroup_t` it obtains from its audio device and passes the
opaque pointer to the engine.

## Decision

Extend the existing `rt-hardening` feature to macOS through a separate
[`src/rt_hardening/macos.rs`](../src/rt_hardening/macos.rs) implementation,
gated by `#[cfg(all(feature = "rt-hardening", target_os = "macos"))]`. The pure
tick-conversion arithmetic lives in
[`src/rt_hardening/timebase.rs`](../src/rt_hardening/timebase.rs), free of libc
and system calls, so it is testable on any host. The Linux module and its
public API stay unchanged. The macOS surface is:

| Function | Contract |
| :--------- | :--------- |
| `set_ftz_daz()` | Enables denormal flushing on the calling thread (`FPCR.FZ` on aarch64). Same contract as the Linux helper. |
| `mlock_region(addr, len)` | `unsafe` `mlock` for a host-owned model/scratch region; off-RT only. |
| `set_user_interactive_qos()` | `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`; off-RT or RT thread setup. |
| `promote_time_constraint(buffer_frames, sample_rate, &RtStatusFlags)` | Applies DAZ/FTZ, interactive QoS and `THREAD_TIME_CONSTRAINT_POLICY`; publishes telemetry and returns the applied `ThreadTiming`. Off-RT or RT thread setup only. |
| `compute_thread_timing(buffer_frames, sample_rate, numer, denom)` | Pure, testable tick conversion; no syscall. Re-exported from `timebase`. |
| `join_audio_workgroup(workgroup)` | `unsafe` join of the calling thread to a host-supplied `os_workgroup_t`; returns an RAII guard that leaves on drop. |

### Timing policy

`period` is derived from the host quantum:

```text
period_ns = buffer_frames * 1_000_000_000 / sample_rate
period_ticks = period_ns * timebase.denom / timebase.numer
```

`computation` and `constraint` are both set to `period_ticks`: the audio
callback is budgeted for the full quantum and is not expected to be preempted
inside the block. `preemptible` is `false`. A period that does not fit in `u32`
Mach ticks, a zero `buffer_frames`, a zero `sample_rate`, a zero timebase
component, or a quantum shorter than one nanosecond is rejected with an
`InvalidInput` error rather than silently wrapping. `mach_timebase_info` failures
and zero timebase components reported by the OS are also returned as errors.

### Telemetry mapping

The existing `RtStatusFlags` fields remain the single telemetry contract:

- `rt_policy` = `THREAD_TIME_CONSTRAINT_POLICY` on success, `-1` on failure.
- `rt_priority` / `confirmed_priority` = `QOS_CLASS_USER_INTERACTIVE` on success
  (a QoS class, not a Mach scheduler priority).
- `rt_sched_err` = `0` on success, otherwise the first failing `pthread_*` or
  Mach error code.
- `rt_tid` = `pthread_self()`; `rt_cpu` = `-1` (no portable current-CPU API is
  used).
- `RT_STATUS_RT_IS_FIFO` is never set on macOS. Hosts must inspect
  `rt_policy`, not the Linux-named flag.

### Memory locking

`mlock_region` locks bounded host-owned model or scratch allocations in RAM via
`libc::mlock` (off-RT only). `mlockall_current` is intentionally omitted on macOS because the availability
and behavior of `mlockall` in `libSystem.B.dylib` have not been verified on an
Apple host. The engine does not reach into host buffers.

### Audio Workgroups

`join_audio_workgroup` accepts the opaque `os_workgroup_t` obtained by the host
from its audio device. It is `unsafe` because the pointer must be a live
workgroup and the call must run on the thread that will process audio. The
returned [`AudioWorkgroupJoin`](../src/rt_hardening/macos.rs) guard stores the
join token and calls `os_workgroup_leave` exactly once, on drop or via
`leave()`. The guard is neither `Send` nor `Sync` because the token belongs to
the joining thread. The engine does not create, retain or release the workgroup
object itself. A null workgroup is rejected with `EINVAL`.

### Deliberately absent

- CPU affinity, `SCHED_FIFO`, PM-QoS and THP: not part of the XNU contract.
- CoreAudio framework linkage or header use: the host owns the workgroup.
- A new public telemetry flag or a new dependency: the existing fields and
  `libc` are sufficient.
- Windows changes: the module uses `target_os` gates rather than `cfg(unix)`, so
  a future Windows port is not blocked.

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
- `cargo check --target aarch64-apple-darwin --features rt-hardening` is the
  required cross-compilation check. It proves that the module compiles, not that
  it behaves correctly at run time. `x86_64-apple-darwin` is not claimed.
- The `macos-arm64` CI runner executes the quick suite natively on Apple Silicon
  (see [testing.md](testing.md)); the thread-policy calls themselves are not
  exercised there under real-time load.
- Real jitter measurement on M-series hardware remains an operator hardware gate
  and cannot be satisfied on the Linux development host. Until it is recorded,
  macOS real-time behavior is unmeasured.
- Hosts gain one optional unsafe entry point for Audio Workgroups but no
  CoreAudio dependency.

## Verification

- **Static analysis:** `utils/lints.sh`, which includes a `cargo check` with the
  `rt-hardening` feature.
- **Unit tests:** the pure tick conversion
  ([`timebase_test.rs`](../src/rt_hardening/timebase_test.rs)) runs on any host.
- **Cross-check:** `cargo check --target aarch64-apple-darwin --features rt-hardening`.
- **Documentation:** rustdoc for the macOS module and
  [architecture.md](architecture.md) §2.
- **Hardware (operator):** jitter measurement on M-series hardware.
- **Quick suite:** `utils/tests-quick.sh`, once per milestone acceptance.
