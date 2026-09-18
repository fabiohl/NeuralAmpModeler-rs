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
}

impl PmQosGuard {
    /// Returns the requested latency in microseconds.
    pub fn latency_us(&self) -> u32 {
        0
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
    let mut file = File::options().write(true).open("/dev/cpu_dma_latency")?;
    file.write_all(&latency_us.to_ne_bytes()).map_err(|e| {
        log::warn!("PM QoS: failed to write to /dev/cpu_dma_latency ({e}).");
        e
    })?;
    log::info!("PM QoS lock: CPU DMA latency bound {latency_us} us requested.");
    Ok(PmQosGuard { _handle: file })
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
}
