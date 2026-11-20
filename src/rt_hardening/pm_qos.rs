// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! PM QoS (`/dev/cpu_dma_latency`) latency guard (off-RT only).
//!
//! Requests zero DMA latency from the kernel to keep deep CPU C-states from
//! adding wake-up latency to the audio path. This function must be called
//! exclusively outside the RT thread, during audio host initialization.
//! Errors are returned as `Result` — never as panic.

#![cfg(all(feature = "rt-hardening", target_os = "linux"))]

use std::fs::File;
use std::io::{self, Write};

/// RAII guard holding the `/dev/cpu_dma_latency` handle.
///
/// While alive, the kernel keeps the requested latency bound. Dropping the
/// guard (closing the fd) releases the request.
#[derive(Debug)]
pub struct PmQosGuard {
    _handle: File,
    latency_us: u32,
}

impl PmQosGuard {
    /// Returns the requested latency in microseconds.
    pub fn latency_us(&self) -> u32 {
        self.latency_us
    }
}

/// Requests `latency_us` microseconds of CPU DMA latency via the kernel PM
/// QoS interface (off-RT only).
///
/// The returned [`PmQosGuard`] must be kept alive for the protection to hold;
/// dropping it revokes the request. System-wide effect: it constrains all
/// CPU cores, not just the calling thread.
#[cold]
pub fn request_cpu_dma_latency(latency_us: u32) -> io::Result<PmQosGuard> {
    request_cpu_dma_latency_at(std::path::Path::new("/dev/cpu_dma_latency"), latency_us)
}

#[cold]
fn request_cpu_dma_latency_at(path: &std::path::Path, latency_us: u32) -> io::Result<PmQosGuard> {
    // Kernel /dev/cpu_dma_latency interprets the 4-byte payload as signed s32;
    // clamp to i32::MAX (>= 0) so the guard never writes or reports a negative bound.
    let clamped = latency_us.min(i32::MAX as u32);
    let mut file = File::options().write(true).open(path)?;
    file.write_all(&(clamped as i32).to_ne_bytes())
        .map_err(|e| {
            log::warn!("PM QoS: failed to write to {} ({e}).", path.display());
            e
        })?;
    log::info!("PM QoS lock: CPU DMA latency bound {clamped} us requested.");
    Ok(PmQosGuard {
        _handle: file,
        latency_us: clamped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pm_qos_missing_device_is_err_not_panic() {
        if std::path::Path::new("/dev/cpu_dma_latency").exists() {
            let _ = request_cpu_dma_latency(0);
        } else {
            assert!(request_cpu_dma_latency(0).is_err());
        }
    }

    #[test]
    fn test_pm_qos_guard_stores_and_clamps_latency_us() {
        let tmp_path = std::env::temp_dir().join(format!(
            "nam_pm_qos_test_{}_{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        {
            File::create(&tmp_path).expect("create temp file");
            let guard = request_cpu_dma_latency_at(&tmp_path, 42).expect("write latency");
            assert_eq!(guard.latency_us(), 42);
            drop(guard);
            let bytes = std::fs::read(&tmp_path).expect("read temp file");
            assert_eq!(bytes, 42_i32.to_ne_bytes());

            let clamped_guard =
                request_cpu_dma_latency_at(&tmp_path, u32::MAX).expect("write clamped latency");
            assert_eq!(clamped_guard.latency_us(), i32::MAX as u32);
        }
        let _ = std::fs::remove_file(&tmp_path);
    }
}
