// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Minimal `perf_event_open` PMU plumbing for RT metric harnesses.
//!
//! libc has no `perf_event_open` binding, so this uses the raw syscall (as the
//! standalone `ab_opt_bench` does) to expose just `cycles` and `instructions`
//! per-thread, letting a harness compute IPC-per-block — a governor/thermal
//! immune degradation signal. When the kernel's `perf_event_paranoid` level
//! forbids opening the counters, the set reports a typed unavailability reason
//! instead of a misleading zero.

// `libc::Ioctl` is `c_ulong` on glibc and `c_int` on musl/Android; the request
// constants must use the target's own type or the call fails to compile there.
const PERF_EVENT_IOC_ENABLE: libc::Ioctl = 0x2400;
const PERF_EVENT_IOC_DISABLE: libc::Ioctl = 0x2401;
const PERF_EVENT_IOC_RESET: libc::Ioctl = 0x2403;
const PERF_FLAG_FD_CLOEXEC: u64 = 0x8;
const PERF_TYPE_HARDWARE: u32 = 0;
const PERF_COUNT_HW_CPU_CYCLES: u64 = 0;
const PERF_COUNT_HW_INSTRUCTIONS: u64 = 1;

/// Linux `perf_event_attr` (first 112 bytes — enough for type/size/config and
/// the `disabled`/`exclude_kernel`/`exclude_hv` control flags).
#[repr(C)]
struct PerfEventAttr {
    type_: u32,
    size: u32,
    config: u64,
    sample_period: u64,
    sample_type: u64,
    read_format: u64,
    flags: u64,
    wakeup_events: u32,
    bp_type: u32,
    config1: u64,
    config2: u64,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
}

impl PerfEventAttr {
    fn hardware(config: u64) -> Self {
        Self {
            type_: PERF_TYPE_HARDWARE,
            size: std::mem::size_of::<Self>() as u32,
            config,
            sample_period: 0,
            sample_type: 0,
            read_format: 0,
            // disabled=1 (bit 0), exclude_kernel=1 (bit 5), exclude_hv=1 (bit 6).
            flags: 1 | (1 << 5) | (1 << 6),
            wakeup_events: 0,
            bp_type: 0,
            config1: 0,
            config2: 0,
            branch_sample_type: 0,
            sample_regs_user: 0,
            sample_stack_user: 0,
            clockid: 0,
        }
    }
}

/// One best-effort PMU counter; the fd is `None` when unavailable.
struct Pmc {
    name: &'static str,
    fd: Option<i32>,
    reason: Option<String>,
}

impl Pmc {
    fn open(name: &'static str, config: u64) -> Self {
        let attr = PerfEventAttr::hardware(config);
        // SAFETY: `attr` is a fully initialized value of the correct layout;
        // the syscall only reads it and returns an fd (or -1). The syscall
        // number comes from libc so every supported Linux architecture uses
        // its own ABI instead of a hard-coded x86-64 value.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &raw const attr,
                0,  // pid: current thread
                -1, // cpu: any (harness pins the CPU)
                -1, // group_fd: none
                PERF_FLAG_FD_CLOEXEC,
            )
        };
        if fd >= 0 {
            Self {
                name,
                fd: Some(fd as i32),
                reason: None,
            }
        } else {
            Self {
                name,
                fd: None,
                reason: Some(std::io::Error::last_os_error().to_string()),
            }
        }
    }

    fn ioctl(&self, request: libc::Ioctl) -> Result<(), std::io::Error> {
        let Some(fd) = self.fd else {
            return Ok(());
        };
        // SAFETY: `fd` is a valid perf fd owned by this counter.
        let rc = unsafe { libc::ioctl(fd, request, 0) };
        if rc < 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    fn read(&self) -> Result<u64, std::io::Error> {
        let Some(fd) = self.fd else {
            return Err(std::io::Error::from(std::io::ErrorKind::NotConnected));
        };
        let mut val: u64 = 0;
        // SAFETY: `fd` is a valid perf fd and `val` is a writable u64 large
        // enough for a single counter value.
        let n = unsafe {
            libc::read(
                fd,
                &raw mut val as *mut libc::c_void,
                std::mem::size_of::<u64>(),
            )
        };
        if n < 0 {
            return Err(std::io::Error::last_os_error());
        }
        if n != std::mem::size_of::<u64>() as isize {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("short perf counter read ({n} bytes)"),
            ));
        }
        Ok(val)
    }
}

impl Drop for Pmc {
    fn drop(&mut self) {
        if let Some(fd) = self.fd {
            // SAFETY: `fd` is a valid fd owned by this counter.
            unsafe {
                libc::close(fd);
            }
        }
    }
}

/// Per-thread cycles/instructions counter pair.
pub struct PmuCounters {
    cycles: Pmc,
    instructions: Pmc,
    /// First `ioctl`/`read` failure after opening, recorded so a failed counter
    /// is never mistaken for a valid `(0, 0)` sample.
    runtime_reason: std::sync::Mutex<Option<String>>,
}

impl PmuCounters {
    /// Opens the cycles/instructions pair for the calling thread.
    pub fn open() -> Self {
        Self {
            cycles: Pmc::open("cycles", PERF_COUNT_HW_CPU_CYCLES),
            instructions: Pmc::open("instructions", PERF_COUNT_HW_INSTRUCTIONS),
            runtime_reason: std::sync::Mutex::new(None),
        }
    }

    fn lock_runtime_reason(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.runtime_reason
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record_error(&self, source: &str, error: &std::io::Error) {
        let mut slot = self.lock_runtime_reason();
        if slot.is_none() {
            *slot = Some(format!("{source}:{error}"));
        }
    }

    /// Resets and enables both counters.
    ///
    /// A failed reset/enable is recorded and makes [`Self::cycles_instructions`]
    /// return `None` until the next successful [`Self::start`].
    pub fn start(&self) {
        *self.lock_runtime_reason() = None;
        for pmc in [&self.cycles, &self.instructions] {
            if let Err(error) = pmc.ioctl(PERF_EVENT_IOC_RESET) {
                self.record_error(pmc.name, &error);
            }
            if let Err(error) = pmc.ioctl(PERF_EVENT_IOC_ENABLE) {
                self.record_error(pmc.name, &error);
            }
        }
    }

    /// Disables both counters.
    ///
    /// A failed disable is recorded because the measurement can no longer be
    /// trusted; the counters remain conservative (unavailable) afterwards.
    pub fn stop(&self) {
        for pmc in [&self.cycles, &self.instructions] {
            if let Err(error) = pmc.ioctl(PERF_EVENT_IOC_DISABLE) {
                self.record_error(pmc.name, &error);
            }
        }
    }

    /// Reads `(cycles, instructions)`.
    ///
    /// Returns `None` when either counter could not be opened, a reset/enable
    /// or read failed, or a read was short (typed reason retrievable via
    /// [`Self::unavailable_reason`]). A genuinely zero counter is still
    /// returned as `Some((0, 0))`.
    pub fn cycles_instructions(&self) -> Option<(u64, u64)> {
        if self.cycles.fd.is_none() || self.instructions.fd.is_none() {
            return None;
        }
        if self.lock_runtime_reason().is_some() {
            return None;
        }
        let cycles = match self.cycles.read() {
            Ok(value) => value,
            Err(error) => {
                self.record_error(self.cycles.name, &error);
                return None;
            }
        };
        let instructions = match self.instructions.read() {
            Ok(value) => value,
            Err(error) => {
                self.record_error(self.instructions.name, &error);
                return None;
            }
        };
        Some((cycles, instructions))
    }

    /// Human-readable reason the counters are unavailable (empty when available).
    pub fn unavailable_reason(&self) -> String {
        let mut parts = Vec::new();
        for pmc in [&self.cycles, &self.instructions] {
            if let Some(reason) = &pmc.reason {
                parts.push(format!("{}:{}", pmc.name, reason));
            }
        }
        if let Some(reason) = self.lock_runtime_reason().as_ref() {
            parts.push(format!("runtime:{reason}"));
        }
        parts.join(",")
    }

    /// Integer instructions-per-cycle, or `None` when unavailable / zero cycles.
    pub fn ipc(&self) -> Option<f64> {
        let (cycles, instructions) = self.cycles_instructions()?;
        if cycles == 0 {
            return None;
        }
        Some(instructions as f64 / cycles as f64)
    }

    /// Builds a counter pair over caller-owned raw fds (tests only).
    #[cfg(test)]
    pub(super) fn from_raw_fds(cycles_fd: i32, instructions_fd: i32) -> Self {
        Self {
            cycles: Pmc {
                name: "cycles",
                fd: Some(cycles_fd),
                reason: None,
            },
            instructions: Pmc {
                name: "instructions",
                fd: Some(instructions_fd),
                reason: None,
            },
            runtime_reason: std::sync::Mutex::new(None),
        }
    }
}
