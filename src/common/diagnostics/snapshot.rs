// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Fábio Henrique de Lima Silva (fhl.bsb@gmail.com) All rights reserved.

//! Runtime snapshot types for diagnostic capture.

use std::sync::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Thread-safe global storage for the currently active sample rate (populated in active standalone session).
pub static ACTIVE_SAMPLE_RATE: AtomicU32 = AtomicU32::new(0);

/// Thread-safe global storage for the currently active model path/name (populated in active standalone session).
pub static ACTIVE_MODEL_NAME: RwLock<String> = RwLock::new(String::new());

/// Thread-safe global storage for the currently active model snapshot info.
pub static ACTIVE_MODEL_INFO: RwLock<Option<ModelInfo>> = RwLock::new(None);

/// Detailed model information for runtime diagnostic snapshot.
#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    /// Neural network architecture type (e.g. "WaveNet" or "LSTM").
    pub arch_label: String,
    /// Model topology variant (e.g. "Standard", "A2-Lite", "1x12").
    pub topology: String,
    /// Number of channels inside the model.
    pub channels: usize,
    /// Receptive field size of the model.
    pub receptive_field: usize,
    /// Native sample rate of the model (Hz).
    pub model_sample_rate: u32,
    /// Weights layout format (e.g. "Original", "GateMajorLstm", "Interleaved4WaveNet").
    pub weights_layout: String,
    /// Basename of the model file.
    pub path_basename: String,
}

/// Consumer-provided audio metadata for snapshot capture.
#[derive(Debug, Clone, Default)]
pub struct AudioMetadata {
    /// Active channel count (e.g. 1 for mono, 2 for stereo).
    pub channel_count: usize,
    /// Host name provided by the consumer (e.g. "Standalone", "HostPlugin").
    pub host_name: String,
}

/// Detailed audio configuration for runtime diagnostic snapshot.
#[derive(Debug, Clone)]
pub struct AudioInfo {
    /// Active sample rate in Hz.
    pub sample_rate: u32,
    /// Buffer size in frames.
    pub buffer_size: usize,
    /// Active channel count.
    pub channel_count: usize,
    /// Audio host name provided by the consumer.
    pub host_name: String,
}

impl Default for AudioInfo {
    fn default() -> Self {
        Self {
            sample_rate: 0,
            buffer_size: 0,
            channel_count: 0,
            host_name: "N/A".to_string(),
        }
    }
}

/// Real-time thread scheduling and affinity information.
#[derive(Debug, Clone)]
pub struct RtInfo {
    /// Thread priority.
    pub thread_priority: i32,
    /// Scheduler policy ("FIFO", "DEADLINE", "OTHER", or "UNKNOWN").
    pub scheduler: String,
    /// CPU core index the thread is pinned to, if any.
    pub cpu_pinned: Option<usize>,
    /// Whether transparent huge pages / hugetlb pages are active.
    pub huge_pages_active: bool,
    /// Huge page allocation mode: "off", "thp", or "hugetlb".
    pub huge_page_mode: String,
}

impl Default for RtInfo {
    fn default() -> Self {
        Self {
            thread_priority: -1,
            scheduler: "UNKNOWN".to_string(),
            cpu_pinned: None,
            huge_pages_active: false,
            huge_page_mode: "off".to_string(),
        }
    }
}

/// Snapshot of the dynamic latency telemetry.
#[derive(Debug, Clone, Default)]
pub struct TelemetrySnapshot {
    /// Median latency in microseconds.
    pub p50_us: u64,
    /// 99th percentile latency in microseconds.
    pub p99_us: u64,
    /// 99.9th percentile latency in microseconds.
    pub p999_us: u64,
    /// Maximum latency peak in microseconds.
    pub max_us: u64,
    /// Total processed blocks.
    pub total_blocks: u64,
    /// Total virtual XRUNs/overloads.
    pub xruns: u32,
    /// Total count of GC items successfully drained.
    pub drains: u32,
    /// Total host buffer misses on the capture (input) side.
    pub input_buffer_miss: u32,
    /// Total host buffer misses on the playback (output) side.
    pub output_buffer_miss: u32,
}

/// Snapshot of the dynamic runtime state, captured on-demand.
#[derive(Debug, Clone, Default)]
pub struct RuntimeSnapshot {
    /// Active model info, if loaded.
    pub model: Option<ModelInfo>,
    /// Audio configuration.
    pub audio: AudioInfo,
    /// Real-time scheduling info.
    pub rt: RtInfo,
    /// Telemetry metrics.
    pub telemetry: TelemetrySnapshot,
    /// Accumulated OR of all RT_STATUS_* flags ever seen since startup.
    pub flags_seen: u64,
}

impl RuntimeSnapshot {
    /// Captures a runtime snapshot from the given provider.
    pub fn capture(provider: &impl HasRuntimeSnapshot, consumer: &AudioMetadata) -> Self {
        Self {
            model: provider.model_info(),
            audio: provider.audio_info(consumer),
            rt: provider.rt_info(),
            telemetry: provider.telemetry_snapshot(),
            flags_seen: provider.flags_seen(),
        }
    }
}

/// Trait for querying dynamic runtime snapshot data.
pub trait HasRuntimeSnapshot {
    /// Returns model snapshot info, if loaded.
    fn model_info(&self) -> Option<ModelInfo>;
    /// Returns current audio snapshot configuration, enriched with consumer metadata.
    fn audio_info(&self, consumer: &AudioMetadata) -> AudioInfo;
    /// Returns real-time and scheduler snapshot info.
    fn rt_info(&self) -> RtInfo;
    /// Returns telemetry snapshot data.
    fn telemetry_snapshot(&self) -> TelemetrySnapshot;
    /// Returns OR-accumulated flags seen since startup.
    fn flags_seen(&self) -> u64;
}

impl HasRuntimeSnapshot for crate::common::spsc::RtStatusFlags {
    fn model_info(&self) -> Option<ModelInfo> {
        if let Ok(info_guard) = ACTIVE_MODEL_INFO.read() {
            info_guard.clone()
        } else {
            None
        }
    }

    fn audio_info(&self, consumer: &AudioMetadata) -> AudioInfo {
        let sr = self.active_rate.load(Ordering::Relaxed);
        let sr = if sr == 0 {
            ACTIVE_SAMPLE_RATE.load(Ordering::Relaxed)
        } else {
            sr
        };
        let buffer_size = self.last_n_samples.load(Ordering::Relaxed) as usize;
        AudioInfo {
            sample_rate: sr,
            buffer_size,
            channel_count: consumer.channel_count,
            host_name: consumer.host_name.clone(),
        }
    }

    fn rt_info(&self) -> RtInfo {
        let thread_priority = self.confirmed_priority.load(Ordering::Relaxed);
        let scheduler_val = self.rt_policy.load(Ordering::Relaxed);
        let scheduler = if scheduler_val == -1 {
            "UNKNOWN".to_string()
        } else if scheduler_val == libc::SCHED_FIFO || scheduler_val == libc::SCHED_RR {
            "FIFO".to_string()
        } else if scheduler_val == 6 {
            "DEADLINE".to_string()
        } else {
            "OTHER".to_string()
        };

        let cpu = self.rt_cpu.load(Ordering::Relaxed);
        let cpu_pinned = if cpu >= 0 { Some(cpu as usize) } else { None };
        let huge_tlb = self.check_flag(crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
            || (self.flags_seen.load(Ordering::Relaxed)
                & crate::common::spsc::RT_STATUS_HUGEPAGE_OK)
                != 0;
        let thp_active = self.check_flag(crate::common::spsc::RT_STATUS_THP_ACTIVE)
            || (self.flags_seen.load(Ordering::Relaxed)
                & crate::common::spsc::RT_STATUS_THP_ACTIVE)
                != 0;
        let huge_pages_active = huge_tlb || thp_active;
        let huge_page_mode = if huge_tlb {
            "hugetlb".to_string()
        } else if thp_active {
            "thp".to_string()
        } else {
            "off".to_string()
        };

        RtInfo {
            thread_priority,
            scheduler,
            cpu_pinned,
            huge_pages_active,
            huge_page_mode,
        }
    }

    fn telemetry_snapshot(&self) -> TelemetrySnapshot {
        let p50_us = self.latency_hist.get_percentile(0.50) / 1000;
        let p99_us = self.latency_hist.get_percentile(0.99) / 1000;
        let p999_us = self.latency_hist.get_percentile(0.999) / 1000;
        let max_us = self.latency_hist.get_max() / 1000;
        let total_blocks = self.latency_hist.total_count();
        let xruns = self.xruns.load(Ordering::Relaxed);
        let drains = self.drains.load(Ordering::Relaxed);
        let input_buffer_miss = self.input_buffer_miss.load(Ordering::Relaxed);
        let output_buffer_miss = self.output_buffer_miss.load(Ordering::Relaxed);

        TelemetrySnapshot {
            p50_us,
            p99_us,
            p999_us,
            max_us,
            total_blocks,
            xruns,
            drains,
            input_buffer_miss,
            output_buffer_miss,
        }
    }

    fn flags_seen(&self) -> u64 {
        let current_bits = self.status_bits.load(Ordering::Relaxed);
        self.flags_seen.fetch_or(current_bits, Ordering::Relaxed);
        self.flags_seen.load(Ordering::Relaxed)
    }
}
